import contextlib
import math
from collections.abc import Iterator
from contextvars import ContextVar

import torch

import llm_lab.model.layers as model_layers
from llm_lab.model.functional import softmax


_emit_nested_ranges = ContextVar("emit_nested_nvtx_ranges", default=False)


@contextlib.contextmanager
def nvtx_range(message: str, *, force: bool = False) -> Iterator[None]:
    should_emit = force or _emit_nested_ranges.get()
    if should_emit and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(message)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


@contextlib.contextmanager
def benchmark_nvtx_range(message: str) -> Iterator[None]:
    token = _emit_nested_ranges.set(True)
    try:
        with nvtx_range(message, force=True):
            yield
    finally:
        _emit_nested_ranges.reset(token)


def nvtx_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    with nvtx_range("attention/scores"):
        scores = Q @ (K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])

    if mask is not None:
        with nvtx_range("attention/mask"):
            scores = scores.masked_fill(~mask, float("-inf"))

    with nvtx_range("attention/softmax"):
        attn = softmax(scores, dim=-1)

    with nvtx_range("attention/output"):
        return attn @ V


def install_nvtx_attention() -> None:
    model_layers.scaled_dot_product_attention = nvtx_scaled_dot_product_attention
