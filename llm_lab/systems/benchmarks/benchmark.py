import argparse
import contextlib
from pathlib import Path
import statistics
import timeit
from collections.abc import Iterator

import torch

from llm_lab.model.layers import TransformerLM
from llm_lab.systems.benchmarks.nvtx import benchmark_nvtx_range, install_nvtx_attention
from llm_lab.training.losses import cross_entropy
from llm_lab.training.optimizer import AdamW, gradient_clipping


MODEL_SIZES = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark TransformerLM training/inference paths")

    parser.add_argument("--mode", choices=["forward", "backward", "train-step"], default="forward")
    parser.add_argument("--model-size", choices=["custom", *MODEL_SIZES], default="custom")
    parser.add_argument("--precision", choices=["full", "fp16-mixed", "bf16-mixed"], default="full")
    parser.add_argument("--compare-precision", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compare-compile", action="store_true")
    parser.add_argument("--record-memory-history", action="store_true")
    parser.add_argument("--memory-profile-dir", type=Path, default=Path("profiles/nsight"))
    parser.add_argument("--memory-history-max-entries", type=int, default=1_000_000)

    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=682)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=None)

    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def apply_model_size(args):
    if args.model_size == "custom":
        return
    for name, value in MODEL_SIZES[args.model_size].items():
        setattr(args, name, value)


def device_type(device: str) -> str:
    return device.split(":", maxsplit=1)[0]


@contextlib.contextmanager
def precision_context(args) -> Iterator[None]:
    mixed_precision_dtypes = {
        "fp16-mixed": torch.float16,
        "bf16-mixed": torch.bfloat16,
    }
    if args.precision == "bf16-mixed":
        with torch.amp.autocast(device_type=device_type(args.device), dtype=mixed_precision_dtypes[args.precision]):
            yield
    elif args.precision == "fp16-mixed":
        with torch.amp.autocast(device_type=device_type(args.device), dtype=mixed_precision_dtypes[args.precision]):
            yield
    else:
        with contextlib.nullcontext():
            yield


def synchronize(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def make_batch(vocab_size, batch_size, context_length, device):
    inputs = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        dtype=torch.long,
        device=device,
    )
    targets = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        dtype=torch.long,
        device=device,
    )
    return inputs, targets


def make_model(args):
    install_nvtx_attention()
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    )
    if args.compile:
        model = torch.compile(model)
    return model


def make_step(args, model, optimizer, inputs, targets):
    if args.mode == "forward":
        model.eval()

        def step():
            with torch.no_grad(), precision_context(args):
                model(inputs)
            synchronize(args.device)

        return step

    if args.mode == "backward":
        model.train()

        def step():
            optimizer.zero_grad(set_to_none=True)
            with precision_context(args):
                logits = model(inputs)
                loss = cross_entropy(logits.float(), targets)
            loss.backward()
            synchronize(args.device)

        return step

    if args.mode == "train-step":
        model.train()

        def step():
            optimizer.zero_grad(set_to_none=True)
            with precision_context(args):
                logits = model(inputs)
                loss = cross_entropy(logits.float(), targets)
            loss.backward()
            if args.grad_clip is not None:
                gradient_clipping(model.parameters(), args.grad_clip)
            optimizer.step()
            synchronize(args.device)

        return step

    raise ValueError(f"Unsupported benchmark mode: {args.mode}")


@contextlib.contextmanager
def cuda_memory_history(args, label: str) -> Iterator[Path | None]:
    if not args.record_memory_history:
        yield None
        return
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("--record-memory-history requires a CUDA device")

    args.memory_profile_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = args.memory_profile_dir / f"memory_snapshot_{label}.pickle"
    torch.cuda.memory._record_memory_history(max_entries=args.memory_history_max_entries)
    try:
        yield snapshot_path
    finally:
        torch.cuda.memory._dump_snapshot(str(snapshot_path))
        torch.cuda.memory._record_memory_history(enabled=None)


def run_benchmark(args):
    apply_model_size(args)
    torch.manual_seed(args.seed)

    model = make_model(args)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    inputs, targets = make_batch(
        args.vocab_size,
        args.batch_size,
        args.context_length,
        args.device,
    )
    step = make_step(args, model, optimizer, inputs, targets)

    for _ in range(args.warmups):
        step()
    synchronize(args.device)

    timer = timeit.Timer(step)
    with benchmark_nvtx_range("benchmark"):
        compile_label = "compiled" if args.compile else "eager"
        memory_label = f"{args.model_size}_{args.mode}_{args.precision}_{compile_label}"
        with cuda_memory_history(args, memory_label) as memory_snapshot_path:
            repeat_times = timer.repeat(repeat=args.repeats, number=args.iters)
    per_iter = [elapsed / args.iters for elapsed in repeat_times]

    tokens_per_iter = args.batch_size * args.context_length
    best = min(per_iter)
    mean = statistics.mean(per_iter)
    stdev = statistics.stdev(per_iter) if len(per_iter) > 1 else 0.0

    print(f"model_size: {args.model_size}")
    print(f"mode: {args.mode}")
    print(f"precision: {args.precision}")
    print(f"compile: {args.compile}")
    print(f"device: {args.device}")
    print(f"vocab_size: {args.vocab_size}")
    print(f"batch_size: {args.batch_size}")
    print(f"context_length: {args.context_length}")
    print(f"d_model: {args.d_model}")
    print(f"d_ff: {args.d_ff}")
    print(f"num_layers: {args.num_layers}")
    print(f"num_heads: {args.num_heads}")
    print(f"tokens/iter: {tokens_per_iter}")
    print(f"warmups: {args.warmups}")
    print(f"iters/repeat: {args.iters}")
    print(f"repeats: {args.repeats}")
    print(f"best: {best * 1000:.3f} ms")
    print(f"mean: {mean * 1000:.3f} ms")
    print(f"stdev: {stdev * 1000:.3f} ms")
    print(f"best throughput: {tokens_per_iter / best:.1f} tokens/s")
    print(f"mean throughput: {tokens_per_iter / mean:.1f} tokens/s")
    if memory_snapshot_path is not None:
        print(f"memory snapshot: {memory_snapshot_path}")
    return {
        "best": best,
        "mean": mean,
        "stdev": stdev,
        "tokens_per_iter": tokens_per_iter,
    }


def main():
    args = parse_args()
    compile_modes = (False, True) if args.compare_compile else (args.compile,)
    precision_modes = ("full", "fp16-mixed", "bf16-mixed") if args.compare_precision else (args.precision,)

    if len(compile_modes) > 1 or len(precision_modes) > 1:
        results = {}
        for compile_enabled in compile_modes:
            compile_label = "compiled" if compile_enabled else "eager"
            for precision in precision_modes:
                result_key = (compile_label, precision)
                run_args = argparse.Namespace(**vars(args))
                run_args.compile = compile_enabled
                run_args.precision = precision
                print(f"--- {compile_label} {precision} ---")
                results[result_key] = run_benchmark(run_args)

        baseline_key = ("eager", "full") if args.compare_compile and args.compare_precision else next(iter(results))
        baseline_mean = results[baseline_key]["mean"]
        print("--- comparison ---")
        print(f"mean {baseline_key[0]} {baseline_key[1]}: {baseline_mean * 1000:.3f} ms")
        for (compile_label, precision), result in results.items():
            if (compile_label, precision) == baseline_key:
                continue
            mean = result["mean"]
            speedup = baseline_mean / mean
            print(f"mean {compile_label} {precision}: {mean * 1000:.3f} ms")
            print(f"{compile_label} {precision} speedup vs baseline: {speedup:.3f}x")
    else:
        run_benchmark(args)


if __name__ == "__main__":
    main()
