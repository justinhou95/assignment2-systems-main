"""
nsys profiling script with NVTX annotations.

Usage (single run):
    uv run nsys profile -o result --pytorch python cs336_systems/nsys_profile_script.py \
        --size small --context-length 128

Usage (sweep all sizes and context lengths):
    for size in small medium large xl 2.7B; do
      for ctx in 128 256 512 1024; do
        uv run nsys profile -o ${size}_${ctx} --pytorch \
          python cs336_systems/nsys_profile_script.py --size $size --context-length $ctx
      done
    done
"""

import argparse
import math

import torch
import torch.cuda.nvtx as nvtx
from einops import einsum, rearrange

import cs336_basics.model as model_module
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, softmax

# ---------------------------------------------------------------------------
# Annotated scaled dot-product attention with NVTX sub-ranges
# ---------------------------------------------------------------------------


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]

    with nvtx.range("computing attention scores"):
        attention_scores = einsum(
            Q, K, "... query d_k, ... key d_k -> ... query key"
        ) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)

    with nvtx.range("final matmul"):
        output = einsum(
            attention_weights, V, "... query key, ... key d_v -> ... query d_v"
        )

    return output


# ---------------------------------------------------------------------------
# Model size configurations (Table 1)
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7B": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="nsys profiling script with NVTX annotations"
    )
    parser.add_argument(
        "--size",
        choices=list(MODEL_CONFIGS.keys()),
        default="small",
        help="Model size (Table 1)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=128,
        choices=[128, 256, 512, 1024],
        help="Context length",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=3,
        help="Warm-up steps (excluded from profiling via NVTX filter)",
    )
    parser.add_argument("--steps", type=int, default=3, help="Number of profiled steps")
    parser.add_argument(
        "--mode",
        choices=["forward", "forward_backward", "full_training"],
        default="full_training",
        help="forward: forward pass only; "
        "forward_backward: forward + backward; "
        "full_training: forward + backward + optimizer step",
    )
    parser.add_argument(
        "--annotate-attention",
        action="store_true",
        default=True,
        help="Swap in the NVTX-annotated scaled_dot_product_attention",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Swap in the annotated attention if requested
    if args.annotate_attention:
        model_module.scaled_dot_product_attention = (
            annotated_scaled_dot_product_attention
        )

    cfg = MODEL_CONFIGS[args.size]
    print(f"Building model: size={args.size}, context_length={args.context_length}")
    print(
        f"  d_model={cfg['d_model']}, d_ff={cfg['d_ff']}, "
        f"num_layers={cfg['num_layers']}, num_heads={cfg['num_heads']}"
    )

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=args.rope_theta,
    ).to(args.device)
    model.train()

    optimizer = None
    if args.mode == "full_training":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x0 = torch.randint(args.vocab_size, size=[args.batch_size, args.context_length])
    y0 = torch.randint(args.vocab_size, size=[args.batch_size, args.context_length])

    def run_step(step_label: str):
        x = x0.to(args.device)
        y = y0.to(args.device)

        with nvtx.range(f"{step_label}/forward"):
            logits = model(x)

        if args.mode in ("forward_backward", "full_training"):
            loss = cross_entropy(
                rearrange(logits, "b s v -> (b s) v"),
                rearrange(y, "b s -> (b s)"),
            )
            with nvtx.range(f"{step_label}/backward"):
                loss.backward()

        if args.mode == "full_training" and optimizer is not None:
            with nvtx.range(f"{step_label}/optimizer_step"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        else:
            model.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    # Warm-up steps — wrapped in a single "warmup" NVTX range so they
    # can be filtered out when viewing the profile in Nsight Systems.
    # ------------------------------------------------------------------
    print(f"Running {args.warmup_steps} warm-up steps...")
    with nvtx.range("warmup"):
        for i in range(args.warmup_steps):
            run_step(f"warmup_step_{i}")
            if "cuda" in args.device:
                torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Profiled steps — each wrapped in its own "step_N" NVTX range.
    # ------------------------------------------------------------------
    print(f"Running {args.steps} profiled steps (mode={args.mode})...")
    with nvtx.range("profiled"):
        for i in range(args.steps):
            with nvtx.range(f"step_{i}"):
                run_step(f"step_{i}")
            if "cuda" in args.device:
                torch.cuda.synchronize()

    print("Done.")


if __name__ == "__main__":
    main()
