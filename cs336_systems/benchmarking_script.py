import argparse
import timeit

from einops import rearrange
import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark")
    parser.add_argument("--size", type=str, default="small")
    parser.add_argument(
        "--mode", choices=["forward", "forward_backward"], default="forward_backward"
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def benchmark(args):
    vocab_size = 10000
    batch_size = 4
    context_length = 100
    rope_theta = 10000
    if args.size == "small":
        d_model = 768
        d_ff = 3072
        num_layers = 12
        num_heads = 12
    model = BasicsTransformerLM(
        vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta
    )
    model = model.to(args.device)
    model.train()

    x0 = torch.randint(vocab_size, size=[batch_size, context_length])
    y0 = torch.randint(vocab_size, size=[batch_size, context_length])

    def run_step():
        x = x0.to(args.device)
        y = y0.to(args.device)
        logits = model(x)
        if args.mode == "forward_backward":
            loss = cross_entropy(
                rearrange(logits, "b s v -> (b s) v"), rearrange(y, "b s -> (b s)")
            )
            loss.backward()
            model.zero_grad(set_to_none=True)

    # Warm-up
    print("Running warm-up steps...")
    for _ in range(args.warmup_steps):
        run_step()
        if "cuda" in args.device:
            torch.cuda.synchronize()

    # Timed steps
    print("Running timed steps...")
    times = []
    for _ in range(args.steps):
        if "cuda" in args.device:
            torch.cuda.synchronize()
        t0 = timeit.default_timer()
        run_step()
        if "cuda" in args.device:
            torch.cuda.synchronize()
        t1 = timeit.default_timer()
        times.append(t1 - t0)

    times_ms = [t * 1000 for t in times]
    mean = sum(times_ms) / len(times_ms)
    std = (sum((t - mean) ** 2 for t in times_ms) / len(times_ms)) ** 0.5
    print(f"\nResults ({args.steps} steps, mode={args.mode}):")
    print(
        f"  Mean: {mean:.2f} ms  Std: {std:.2f} ms  Min: {min(times_ms):.2f} ms  Max: {max(times_ms):.2f} ms"
    )


if __name__ == "__main__":
    args = parse_args()
    benchmark(args)
