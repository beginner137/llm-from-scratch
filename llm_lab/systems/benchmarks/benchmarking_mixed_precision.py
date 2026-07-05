import argparse
import contextlib
import statistics
import timeit
from collections.abc import Iterator

import torch
import torch.nn as nn

from llm_lab.systems.benchmarks.nvtx import benchmark_nvtx_range, nvtx_range


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark autocast dtypes through a toy model")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--in-features", type=int, default=1024)
    parser.add_argument("--out-features", type=int, default=1024)
    parser.add_argument("--input-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--autocast-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--no-autocast", action="store_true")
    parser.add_argument("--include-backward", action="store_true")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def device_type(device: str) -> str:
    return device.split(":", maxsplit=1)[0]


def synchronize(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def tensor_dtypes(value) -> str:
    if isinstance(value, torch.Tensor):
        return str(value.dtype).removeprefix("torch.")
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(tensor_dtypes(item) for item in value) + ")"
    return type(value).__name__


def register_dtype_hooks(model: ToyModel, records: list[tuple[str, str, str]]):
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs, output):
            records.append((name, tensor_dtypes(inputs), tensor_dtypes(output)))

        return hook

    for name in ("fc1", "relu", "ln", "fc2"):
        handles.append(getattr(model, name).register_forward_hook(make_hook(name)))

    return handles


@contextlib.contextmanager
def maybe_autocast(args) -> Iterator[None]:
    enabled = not args.no_autocast
    with torch.amp.autocast(
        device_type=device_type(args.device),
        dtype=dtype_from_name(args.autocast_dtype),
        enabled=enabled,
    ):
        yield


def make_step(args, model: ToyModel, inputs: torch.Tensor):
    def step():
        with torch.inference_mode(), maybe_autocast(args), nvtx_range("toy_model/forward"):
            output = model(inputs)
        synchronize(args.device)
        return output

    return step


def print_backward_dtype_report(args, model: ToyModel, inputs: torch.Tensor):
    targets = torch.randn(
        inputs.shape[0],
        args.out_features,
        dtype=torch.float32,
        device=args.device,
    )
    model.zero_grad(set_to_none=True)
    with maybe_autocast(args), nvtx_range("toy_model/backward_dtype_report"):
        output = model(inputs)
        loss = torch.nn.functional.mse_loss(output.float(), targets)
    loss.backward()
    synchronize(args.device)

    print(f"loss dtype: {str(loss.dtype).removeprefix('torch.')}")
    print("gradient dtypes:")
    for name, parameter in model.named_parameters():
        grad_dtype = "None" if parameter.grad is None else str(parameter.grad.dtype).removeprefix("torch.")
        print(f"  {name}: {grad_dtype}")

    model.zero_grad(set_to_none=True)


def print_dtype_report(args, model: ToyModel, inputs: torch.Tensor):
    records = []
    handles = register_dtype_hooks(model, records)
    try:
        output = make_step(args, model, inputs)()
    finally:
        for handle in handles:
            handle.remove()

    print(f"input dtype: {str(inputs.dtype).removeprefix('torch.')}")
    print(f"output dtype: {str(output.dtype).removeprefix('torch.')}")
    print("parameter dtypes:")
    for name, parameter in model.named_parameters():
        print(f"  {name}: {str(parameter.dtype).removeprefix('torch.')}")
    print("module input/output dtypes:")
    for name, input_dtype, output_dtype in records:
        print(f"  {name}: input={input_dtype}, output={output_dtype}")
    if args.include_backward:
        print_backward_dtype_report(args, model, inputs)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model = ToyModel(args.in_features, args.out_features).to(args.device)
    inputs = torch.randn(
        args.batch_size,
        args.in_features,
        dtype=dtype_from_name(args.input_dtype),
        device=args.device,
    )
    step = make_step(args, model, inputs)

    print(f"device: {args.device}")
    print(f"batch_size: {args.batch_size}")
    print(f"in_features: {args.in_features}")
    print(f"out_features: {args.out_features}")
    print(f"autocast: {not args.no_autocast}")
    print(f"autocast_dtype: {args.autocast_dtype}")
    print(f"include_backward: {args.include_backward}")
    print_dtype_report(args, model, inputs)

    for _ in range(args.warmups):
        step()
    synchronize(args.device)

    timer = timeit.Timer(step)
    with benchmark_nvtx_range("benchmark/mixed_precision_toy_model"):
        repeat_times = timer.repeat(repeat=args.repeats, number=args.iters)

    per_iter = [elapsed / args.iters for elapsed in repeat_times]
    print(f"warmups: {args.warmups}")
    print(f"iters/repeat: {args.iters}")
    print(f"repeats: {args.repeats}")
    print(f"best: {min(per_iter) * 1000:.3f} ms")
    print(f"mean: {statistics.mean(per_iter) * 1000:.3f} ms")
    if len(per_iter) > 1:
        print(f"stdev: {statistics.stdev(per_iter) * 1000:.3f} ms")
    else:
        print("stdev: 0.000 ms")


if __name__ == "__main__":
    main()
