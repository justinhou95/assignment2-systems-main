import argparse
import timeit
from contextlib import nullcontext

from einops import rearrange
import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark")
    parser.add_argument(
        "--size",
        type=str,
        default="small",
        choices=["small", "medium", "large", "xl", "2.7B"],
    )
    parser.add_argument(
        "--mode",
        choices=["forward", "forward_backward", "forward_backward_optimizer"],
        default="forward_backward",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--mixed-precision", action="store_true", help="Use BF16 mixed precision"
    )
    parser.add_argument(
        "--profile-memory", action="store_true", help="Record CUDA memory snapshot"
    )
    parser.add_argument("--context-length", type=int, default=128)
    return parser.parse_args()


MODEL_CONFIGS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7B": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


def benchmark(args):
    vocab_size = 10000
    batch_size = 4
    context_length = args.context_length
    rope_theta = 10000
    cfg = MODEL_CONFIGS[args.size]
    d_model, d_ff, num_layers, num_heads = (
        cfg["d_model"],
        cfg["d_ff"],
        cfg["num_layers"],
        cfg["num_heads"],
    )
    model = BasicsTransformerLM(
        vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta
    )
    model = model.to(args.device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Move inputs to device once, not on every step
    x0 = torch.randint(
        vocab_size, size=[batch_size, context_length], device=args.device
    )
    y0 = torch.randint(
        vocab_size, size=[batch_size, context_length], device=args.device
    )

    amp_ctx = (
        torch.autocast(device_type=args.device, dtype=torch.bfloat16)
        if args.mixed_precision
        else nullcontext()
    )
    print(f"Precision mode: {'BF16 mixed' if args.mixed_precision else 'FP32 full'}")

    def run_step():
        with amp_ctx:
            logits = model(x0)
            if args.mode in ("forward_backward", "forward_backward_optimizer"):
                loss = cross_entropy(
                    rearrange(logits, "b s v -> (b s) v"), rearrange(y0, "b s -> (b s)")
                )
        if args.mode in ("forward_backward", "forward_backward_optimizer"):
            loss.backward()
        if args.mode == "forward_backward_optimizer":
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Warm-up
    print("Running warm-up steps...")
    for _ in range(args.warmup_steps):
        run_step()
        if "cuda" in args.device:
            torch.cuda.synchronize()

    is_cuda = "cuda" in args.device

    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    if args.profile_memory and is_cuda:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    # Timed steps
    print("Running timed steps...")
    times = []
    for _ in range(args.steps):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = timeit.default_timer()
        run_step()
        if is_cuda:
            torch.cuda.synchronize()
        t1 = timeit.default_timer()
        times.append(t1 - t0)

    if args.profile_memory and is_cuda:
        snapshot_file = (
            f"memory_snapshot_{args.size}_ctx{args.context_length}_{args.mode}.pickle"
        )
        torch.cuda.memory._dump_snapshot(snapshot_file)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"Memory snapshot saved to {snapshot_file}")

    times_ms = [t * 1000 for t in times]
    mean = sum(times_ms) / len(times_ms)
    std = (sum((t - mean) ** 2 for t in times_ms) / len(times_ms)) ** 0.5
    print(f"Results ({args.steps} steps, mode={args.mode}):")
    print(
        f"  Mean: {mean:.2f} ms  Std: {std:.2f} ms  Min: {min(times_ms):.2f} ms  Max: {max(times_ms):.2f} ms"
    )

    if is_cuda:
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak memory: {peak_mb:.1f} MB ({peak_mb / 1024:.2f} GB)")


if __name__ == "__main__":
    args = parse_args()
    benchmark(args)
