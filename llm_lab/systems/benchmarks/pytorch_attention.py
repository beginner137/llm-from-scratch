import argparse
import math
import statistics
import time

import torch


DEFAULT_D_MODELS = (16, 32, 64, 128)
DEFAULT_SEQ_LENS = (256, 1024, 4096, 8192)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark naive PyTorch scaled dot-product attention")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-models", type=int, nargs="+",
                        default=list(DEFAULT_D_MODELS))
    parser.add_argument("--seq-lens", type=int, nargs="+",
                        default=list(DEFAULT_SEQ_LENS))
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--causal-mask", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compare-compile", action="store_true")
    parser.add_argument("--stop-on-oom", action="store_true")
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


def synchronize(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def reset_cuda_memory(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def memory_allocated_mib(device: str) -> float:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return float("nan")


def max_memory_allocated_mib(device: str) -> float:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return float("nan")


def is_oom(error: RuntimeError) -> bool:
    return "out of memory" in str(error).lower()


def make_causal_mask(seq_len: int, device: str) -> torch.Tensor:
    return torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).tril()


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None):
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return attn @ v


def clear_grads(*tensors: torch.Tensor):
    for tensor in tensors:
        tensor.grad = None


def estimate_attention_memory_mib(batch_size: int, seq_len: int, d_model: int, dtype: torch.dtype):
    element_size = torch.empty((), dtype=dtype).element_size()
    qkv_bytes = 3 * batch_size * seq_len * d_model * element_size
    scores_bytes = batch_size * seq_len * seq_len * element_size
    attn_bytes = batch_size * seq_len * seq_len * element_size
    output_bytes = batch_size * seq_len * d_model * element_size
    return {
        "qkv_mib": qkv_bytes / 1024**2,
        "scores_mib": scores_bytes / 1024**2,
        "attn_mib": attn_bytes / 1024**2,
        "output_mib": output_bytes / 1024**2,
    }


def time_forward(
    args,
    attention_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
):
    times = []
    for _ in range(args.warmups):
        clear_grads(q, k, v)
        loss = attention_fn(q, k, v, mask).sum()
        synchronize(args.device)
        del loss

    for _ in range(args.iters):
        clear_grads(q, k, v)
        start = time.perf_counter()
        loss = attention_fn(q, k, v, mask).sum()
        synchronize(args.device)
        times.append(time.perf_counter() - start)
        del loss
    return statistics.mean(times)


def time_backward(
    args,
    attention_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
):
    backward_times = []
    memory_before_backward = []
    peak_forward_memory = []

    for _ in range(args.warmups):
        clear_grads(q, k, v)
        loss = attention_fn(q, k, v, mask).sum()
        synchronize(args.device)
        loss.backward()
        synchronize(args.device)
        del loss

    for _ in range(args.iters):
        clear_grads(q, k, v)
        reset_cuda_memory(args.device)
        loss = attention_fn(q, k, v, mask).sum()
        synchronize(args.device)
        memory_before_backward.append(memory_allocated_mib(args.device))
        peak_forward_memory.append(max_memory_allocated_mib(args.device))

        start = time.perf_counter()
        loss.backward()
        synchronize(args.device)
        backward_times.append(time.perf_counter() - start)
        del loss

    clear_grads(q, k, v)
    return {
        "backward_mean_s": statistics.mean(backward_times),
        "memory_before_backward_mib": statistics.mean(memory_before_backward),
        "peak_forward_memory_mib": statistics.mean(peak_forward_memory),
    }


def run_case(args, attention_fn, seq_len: int, d_model: int, dtype: torch.dtype):
    q = k = v = mask = None
    try:
        reset_cuda_memory(args.device)
        q = torch.randn(
            args.batch_size,
            seq_len,
            d_model,
            device=args.device,
            dtype=dtype,
            requires_grad=True,
        )
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        if args.causal_mask:
            mask = make_causal_mask(seq_len, args.device)

        synchronize(args.device)
        input_memory_mib = memory_allocated_mib(args.device)
        forward_mean_s = time_forward(args, attention_fn, q, k, v, mask)
        backward_stats = time_backward(args, attention_fn, q, k, v, mask)
        peak_total_memory_mib = max_memory_allocated_mib(args.device)

        return {
            "status": "ok",
            "input_memory_mib": input_memory_mib,
            "forward_mean_s": forward_mean_s,
            "backward_mean_s": backward_stats["backward_mean_s"],
            "memory_before_backward_mib": backward_stats["memory_before_backward_mib"],
            "peak_forward_memory_mib": backward_stats["peak_forward_memory_mib"],
            "peak_total_memory_mib": peak_total_memory_mib,
            "error": "",
        }
    except RuntimeError as error:
        if not is_oom(error):
            raise
        return {
            "status": "oom",
            "input_memory_mib": float("nan"),
            "forward_mean_s": float("nan"),
            "backward_mean_s": float("nan"),
            "memory_before_backward_mib": float("nan"),
            "peak_forward_memory_mib": float("nan"),
            "peak_total_memory_mib": max_memory_allocated_mib(args.device),
            "error": str(error).splitlines()[0],
        }
    finally:
        del q, k, v, mask
        reset_cuda_memory(args.device)


def fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def print_header():
    print(
        "status,compile,seq_len,d_model,forward_ms,backward_ms,"
        "mem_before_backward_mib,peak_mem_mib,est_scores_attn_mib,error"
    )


def print_row(args, compile_mode: str, seq_len: int, d_model: int, dtype: torch.dtype, result: dict):
    estimates = estimate_attention_memory_mib(
        args.batch_size, seq_len, d_model, dtype)
    scores_attn_mib = estimates["scores_mib"] + estimates["attn_mib"]
    print(
        f"{result['status']},{compile_mode},{seq_len},{d_model},"
        f"{fmt(result['forward_mean_s'] * 1000)},"
        f"{fmt(result['backward_mean_s'] * 1000)},"
        f"{fmt(result['memory_before_backward_mib'])},"
        f"{fmt(result['peak_total_memory_mib'])},"
        f"{fmt(scores_attn_mib)},"
        f"{result['error']}"
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = dtype_from_name(args.dtype)
    modes = (False, True) if args.compare_compile else (args.compile,)

    print_header()
    for compile_enabled in modes:
        attention_fn = torch.compile(
            attention) if compile_enabled else attention
        compile_mode = "compiled" if compile_enabled else "eager"
        for d_model in args.d_models:
            for seq_len in args.seq_lens:
                result = run_case(args, attention_fn, seq_len, d_model, dtype)
                print_row(args, compile_mode, seq_len, d_model, dtype, result)
                if args.stop_on_oom and result["status"] == "oom":
                    return


if __name__ == "__main__":
    main()
