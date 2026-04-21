"""
Benchmark script for:
  (A) Standard vs torch.compile'd attention at different scales
  (B) Vanilla vs torch.compile'd full Transformer model
"""

import math
import sys
import time
import itertools

import torch

# Make cs336_basics importable
sys.path.insert(0, "cs336-basics")
from cs336_basics.model import BasicsTransformerLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def mem_allocated_mb(device):
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / (1024**2)
    return float("nan")


def free_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Attention implementation (no head dimension)
# ---------------------------------------------------------------------------


def pytorch_attention(q, k, v):
    """Standard scaled dot-product attention. q/k/v: (batch, seq, d)."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    attn = torch.bmm(q, k.transpose(-2, -1)) * scale  # (B, S, S)
    attn = torch.softmax(attn, dim=-1)
    return torch.bmm(attn, v)  # (B, S, d)


compiled_attention = torch.compile(pytorch_attention)


# ---------------------------------------------------------------------------
# Part A: attention benchmark (uncompiled vs compiled)
# ---------------------------------------------------------------------------


def _bench_attn_fn(fn, batch_size, seq_len, d_head, device, n_warmup=10, n_iters=100):
    """
    Returns (fwd_ms, mem_before_bwd_mb, bwd_ms, oom).
    mem_before_bwd_mb is measured after all forward graphs are built,
    before any backward call.
    """
    try:
        # -- warm-up ----------------------------------------------------------
        for _ in range(n_warmup):
            q = torch.randn(
                batch_size, seq_len, d_head, device=device, requires_grad=True
            )
            k = torch.randn(
                batch_size, seq_len, d_head, device=device, requires_grad=True
            )
            v = torch.randn(
                batch_size, seq_len, d_head, device=device, requires_grad=True
            )
            out = fn(q, k, v)
            out.sum().backward()
            sync(device)
            del q, k, v, out
        free_cache(device)

        # -- time forward passes ----------------------------------------------
        base = torch.randn(batch_size, seq_len, d_head, device=device)

        fwd_times, graphs = [], []
        for _ in range(n_iters):
            qi = base.detach().requires_grad_(True)
            ki = base.detach().requires_grad_(True)
            vi = base.detach().requires_grad_(True)

            sync(device)
            t0 = time.perf_counter()
            out = fn(qi, ki, vi)
            sync(device)
            fwd_times.append((time.perf_counter() - t0) * 1e3)

            graphs.append((out, qi, ki, vi))  # keep graph alive for bwd

        fwd_ms = sum(fwd_times) / len(fwd_times)

        # -- memory right before first backward -------------------------------
        sync(device)
        mem_mb = mem_allocated_mb(device)

        # -- time backward passes ---------------------------------------------
        do = torch.ones_like(graphs[0][0])
        bwd_times = []
        for out, qi, ki, vi in graphs:
            sync(device)
            t0 = time.perf_counter()
            out.backward(do)
            sync(device)
            bwd_times.append((time.perf_counter() - t0) * 1e3)

        bwd_ms = sum(bwd_times) / len(bwd_times)

        del graphs, base, do
        free_cache(device)
        return fwd_ms, mem_mb, bwd_ms, False

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            free_cache(device)
            return None, None, None, True
        raise


def run_attention_benchmark(device):
    batch_size = 8
    d_heads = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]

    col = f"{'d_head':>6} {'seq':>6}  {'fwd_ms':>8} {'mem_MB':>8} {'bwd_ms':>8}"
    sep = "-" * len(col)

    for label, fn in [
        ("UNCOMPILED", pytorch_attention),
        ("COMPILED  ", compiled_attention),
    ]:
        print(f"\n=== Attention — {label} ===")
        print(col)
        print(sep)
        for d_head, seq_len in itertools.product(d_heads, seq_lens):
            fwd, mem, bwd, oom = _bench_attn_fn(fn, batch_size, seq_len, d_head, device)
            if oom:
                print(f"{d_head:>6} {seq_len:>6}  {'OOM':>8} {'OOM':>8} {'OOM':>8}")
            else:
                print(f"{d_head:>6} {seq_len:>6}  {fwd:>8.3f} {mem:>8.1f} {bwd:>8.3f}")


# ---------------------------------------------------------------------------
# Part B: full Transformer model benchmark (vanilla vs compiled)
# ---------------------------------------------------------------------------

# A small GPT-style config; adjust to fit your GPU.
TRANSFORMER_CONFIGS = [
    dict(
        vocab_size=50257,
        context_length=256,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        rope_theta=10000.0,
        batch=4,
        seq=256,
        label="small-256",
    ),
    dict(
        vocab_size=50257,
        context_length=1024,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        rope_theta=10000.0,
        batch=4,
        seq=1024,
        label="small-1024",
    ),
]


def _bench_transformer(model, batch, seq, device, n_warmup=5, n_iters=100):
    """
    Returns (fwd_ms, fwd_bwd_opt_ms, oom).
    """
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # -- warm-up ----------------------------------------------------------
        for _ in range(n_warmup):
            ids = torch.randint(0, 50257, (batch, seq), device=device)
            logits = model(ids)
            loss = logits.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sync(device)
            del ids, logits, loss
        free_cache(device)

        # -- forward only -----------------------------------------------------
        fwd_times = []
        for _ in range(n_iters):
            ids = torch.randint(0, 50257, (batch, seq), device=device)
            sync(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model(ids)
            sync(device)
            fwd_times.append((time.perf_counter() - t0) * 1e3)
            del ids, logits
        fwd_ms = sum(fwd_times) / len(fwd_times)
        free_cache(device)

        # -- forward + backward + optimizer step ------------------------------
        full_times = []
        for _ in range(n_iters):
            ids = torch.randint(0, 50257, (batch, seq), device=device)
            optimizer.zero_grad()
            sync(device)
            t0 = time.perf_counter()
            logits = model(ids)
            loss = logits.sum()
            loss.backward()
            optimizer.step()
            sync(device)
            full_times.append((time.perf_counter() - t0) * 1e3)
            del ids, logits, loss
        full_ms = sum(full_times) / len(full_times)
        free_cache(device)

        return fwd_ms, full_ms, False

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            free_cache(device)
            return None, None, True
        raise


def run_transformer_benchmark(device):
    print("\n=== Transformer: Vanilla vs Compiled ===")
    hdr = f"{'config':<14} {'variant':<12} {'fwd_ms':>9} {'fwd+bwd+opt_ms':>16}"
    print(hdr)
    print("-" * len(hdr))

    for cfg in TRANSFORMER_CONFIGS:
        label = cfg["label"]
        batch = cfg["batch"]
        seq = cfg["seq"]
        model_kwargs = {
            k: v for k, v in cfg.items() if k not in ("batch", "seq", "label")
        }

        for variant, compile_model in [("vanilla", False), ("compiled", True)]:
            model = BasicsTransformerLM(**model_kwargs).to(device)
            if compile_model:
                model = torch.compile(model)

            fwd, full, oom = _bench_transformer(model, batch, seq, device)
            if oom:
                print(f"{label:<14} {variant:<12} {'OOM':>9} {'OOM':>16}")
            else:
                print(f"{label:<14} {variant:<12} {fwd:>9.3f} {full:>16.3f}")

            del model
            free_cache(device)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not available — running on CPU (no OOM, timings slow).")

    run_attention_benchmark(device)
    run_transformer_benchmark(device)


if __name__ == "__main__":
    main()
