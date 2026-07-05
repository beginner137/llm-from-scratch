import argparse

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        device=None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms
        return self.weight * x


def parse_args():
    parser = argparse.ArgumentParser(description="Print tensors saved and loaded by autograd")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def print_tensor(prefix: str, tensor: torch.Tensor):
    shape = tensor.shape
    dtype = tensor.dtype
    grad_fn = tensor.grad_fn
    print(f"{prefix}: {shape=}, {dtype=}, {grad_fn=}")


def pack_hook(tensor: torch.Tensor):
    print_tensor("Saving residual", tensor)
    return tensor


def unpack_hook(tensor: torch.Tensor):
    print_tensor("Loading residual", tensor)
    return tensor


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    x = torch.randn(
        (args.batch_size, args.context_length, args.hidden_size),
        device=args.device,
        requires_grad=True,
    )
    ln = RMSNorm(x.shape[-1], device=args.device)

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        y = ln(x)
        y.sum().backward()


if __name__ == "__main__":
    main()
