import argparse
import statistics
import timeit

import torch

from llm_lab.systems.benchmarks.nvtx import benchmark_nvtx_range, nvtx_range


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark mixed-precision repeated accumulation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--num-elements", type=int, default=1_048_576)
    parser.add_argument("--value", type=float, default=0.01)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def synchronize(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def accumulate(args, acc_dtype: torch.dtype, inc_dtype: torch.dtype) -> torch.Tensor:
    s = torch.zeros(args.num_elements, dtype=acc_dtype, device=args.device)
    x = torch.full((args.num_elements,), args.value, dtype=inc_dtype, device=args.device)

    for _ in range(args.steps):
        s += x

    synchronize(args.device)
    return s


def make_case(args, name: str, acc_dtype: torch.dtype, inc_dtype: torch.dtype):
    def step():
        with nvtx_range(f"accumulation/{name}"):
            return accumulate(args, acc_dtype, inc_dtype)

    return step


def summarize_case(args, name: str, step):
    for _ in range(args.warmups):
        result = step()
    synchronize(args.device)

    timer = timeit.Timer(step)
    with benchmark_nvtx_range(f"benchmark/{name}"):
        repeat_times = timer.repeat(repeat=args.repeats, number=args.iters)

    per_iter = [elapsed / args.iters for elapsed in repeat_times]
    result = step()
    mean_value = result.float().mean().item()
    expected = args.steps * args.value

    print(f"case: {name}")
    print(f"  expected: {expected:.8f}")
    print(f"  mean result: {mean_value:.8f}")
    print(f"  abs error: {abs(mean_value - expected):.8f}")
    print(f"  best: {min(per_iter) * 1000:.3f} ms")
    print(f"  mean: {statistics.mean(per_iter) * 1000:.3f} ms")
    if len(per_iter) > 1:
        print(f"  stdev: {statistics.stdev(per_iter) * 1000:.3f} ms")
    else:
        print("  stdev: 0.000 ms")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    cases = [
        ("fp32_accum_fp32_increment", torch.float32, torch.float32),
        ("fp16_accum_fp16_increment", torch.float16, torch.float16),
        ("fp32_accum_fp16_increment", torch.float32, torch.float16),
    ]

    print(f"device: {args.device}")
    print(f"steps: {args.steps}")
    print(f"num_elements: {args.num_elements}")
    print(f"value: {args.value}")
    print(f"warmups: {args.warmups}")
    print(f"iters/repeat: {args.iters}")
    print(f"repeats: {args.repeats}")

    for name, acc_dtype, inc_dtype in cases:
        summarize_case(args, name, make_case(args, name, acc_dtype, inc_dtype))


if __name__ == "__main__":
    main()
