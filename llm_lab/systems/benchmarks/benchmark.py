import argparse
import statistics
import timeit

import torch

from llm_lab.model.layers import TransformerLM
from llm_lab.systems.benchmarks.nvtx import benchmark_nvtx_range, install_nvtx_attention
from llm_lab.training.losses import cross_entropy
from llm_lab.training.optimizer import AdamW, gradient_clipping


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark TransformerLM training/inference paths")

    parser.add_argument("--mode", choices=["forward", "backward", "train-step"], default="forward")

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
    return TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    )


def make_step(args, model, optimizer, inputs, targets):
    if args.mode == "forward":
        model.eval()

        def step():
            with torch.no_grad():
                model(inputs)
            synchronize(args.device)

        return step

    if args.mode == "backward":
        model.train()

        def step():
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            loss.backward()
            synchronize(args.device)

        return step

    if args.mode == "train-step":
        model.train()

        def step():
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            loss.backward()
            if args.grad_clip is not None:
                gradient_clipping(model.parameters(), args.grad_clip)
            optimizer.step()
            synchronize(args.device)

        return step

    raise ValueError(f"Unsupported benchmark mode: {args.mode}")


def main():
    args = parse_args()
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
        repeat_times = timer.repeat(repeat=args.repeats, number=args.iters)
    per_iter = [elapsed / args.iters for elapsed in repeat_times]

    tokens_per_iter = args.batch_size * args.context_length
    best = min(per_iter)
    mean = statistics.mean(per_iter)
    stdev = statistics.stdev(per_iter) if len(per_iter) > 1 else 0.0

    print(f"mode: {args.mode}")
    print(f"device: {args.device}")
    print(f"batch_size: {args.batch_size}")
    print(f"context_length: {args.context_length}")
    print(f"tokens/iter: {tokens_per_iter}")
    print(f"warmups: {args.warmups}")
    print(f"iters/repeat: {args.iters}")
    print(f"repeats: {args.repeats}")
    print(f"best: {best * 1000:.3f} ms")
    print(f"mean: {mean * 1000:.3f} ms")
    print(f"stdev: {stdev * 1000:.3f} ms")
    print(f"best throughput: {tokens_per_iter / best:.1f} tokens/s")
    print(f"mean throughput: {tokens_per_iter / mean:.1f} tokens/s")


if __name__ == "__main__":
    main()
