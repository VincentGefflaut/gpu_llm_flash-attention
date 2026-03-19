import argparse
import csv
import os
from typing import Callable, cast

import torch
import torch.nn.functional as F

from flash_attention.flash_attention import FlashAttentionTriton


def _events_elapsed_ms(fn: Callable[[], None], n_iters: int) -> float:
    times = []
    for _ in range(n_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sum(times) / len(times)


def _saved_activations_mib(output: torch.Tensor) -> float:
    grad_fn = output.grad_fn
    if grad_fn is None:
        return 0.0
    total_bytes = 0
    for tensor in getattr(grad_fn, "saved_tensors", ()):  # custom autograd path
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024.0 * 1024.0)


def _compile_if_available(fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]):
    if not hasattr(torch, "compile"):
        return fn
    try:
        return torch.compile(fn)
    except Exception:
        return fn


def _run_forward(
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    nb_warmup: int,
    nb_passes: int,
):
    for _ in range(nb_warmup):
        _ = fn(q, k, v)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    def _step():
        _ = fn(q, k, v)

    forward_ms = _events_elapsed_ms(_step, nb_passes)
    forward_peak_mib = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    out = fn(q, k, v)
    saved_mib = _saved_activations_mib(out)
    return forward_ms, forward_peak_mib, saved_mib


def _run_backward(
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    nb_warmup: int,
    nb_passes: int,
):
    for _ in range(nb_warmup):
        out = fn(q, k, v)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        q.grad = None
        k.grad = None
        v.grad = None
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    def _step():
        out = fn(q, k, v)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        q.grad = None
        k.grad = None
        v.grad = None

    backward_ms = _events_elapsed_ms(_step, nb_passes)
    backward_peak_mib = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    return backward_ms, backward_peak_mib


def _flash_triton_impl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, FlashAttentionTriton.apply(q, k, v, False))


def _pytorch_sdpa_impl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, is_causal=False)


def _benchmark_one(
    impl_name: str,
    impl_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    d_model: int,
    seq_len: int,
    batch_size: int,
    dtype: torch.dtype,
    nb_warmup: int,
    nb_forward_passes: int,
    nb_backward_passes: int,
    gpu_name: str,
):
    q = torch.randn(batch_size, seq_len, d_model, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seq_len, d_model, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seq_len, d_model, device="cuda", dtype=dtype, requires_grad=True)

    row = {
        "implementation": impl_name,
        "d_model": d_model,
        "seq_len": seq_len,
        "forward_ms": None,
        "forward_peak_MiB": None,
        "backward_ms": None,
        "backward_peak_MiB": None,
        "saved_activations_MiB": None,
        "status": "ok",
        "gpu": gpu_name,
    }

    try:
        f_ms, f_peak, saved_mib = _run_forward(impl_fn, q, k, v, nb_warmup, nb_forward_passes)
        row["forward_ms"] = f_ms
        row["forward_peak_MiB"] = f_peak
        row["saved_activations_MiB"] = saved_mib
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            row["status"] = "OOM"
            torch.cuda.empty_cache()
            return row
        raise

    try:
        b_ms, b_peak = _run_backward(impl_fn, q, k, v, nb_warmup, nb_backward_passes)
        row["backward_ms"] = b_ms
        row["backward_peak_MiB"] = b_peak
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            row["status"] = "OOM(backward)"
            torch.cuda.empty_cache()
            return row
        raise

    return row


def _format_float(value):
    if value is None:
        return "-"
    return f"{value:.3f}"


def _print_summary(rows):
    headers = [
        "implementation",
        "d_model",
        "seq_len",
        "forward_ms",
        "forward_peak_MiB",
        "backward_ms",
        "backward_peak_MiB",
        "saved_activations_MiB",
        "status",
        "gpu",
    ]
    print(" | ".join(headers))
    print("-" * 140)
    for row in rows:
        print(
            " | ".join(
                [
                    str(row["implementation"]),
                    str(row["d_model"]),
                    str(row["seq_len"]),
                    _format_float(row["forward_ms"]),
                    _format_float(row["forward_peak_MiB"]),
                    _format_float(row["backward_ms"]),
                    _format_float(row["backward_peak_MiB"]),
                    _format_float(row["saved_activations_MiB"]),
                    str(row["status"]),
                    str(row["gpu"]),
                ]
            )
        )


def _save_csv(rows, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "implementation",
        "d_model",
        "seq_len",
        "forward_ms",
        "forward_peak_MiB",
        "backward_ms",
        "backward_peak_MiB",
        "saved_activations_MiB",
        "status",
        "gpu",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Flash Attention Triton vs PyTorch SDPA")
    parser.add_argument(
        "--impl",
        choices=["all", "flash_triton", "pytorch_sdpa"],
        default="all",
        help="Implementation to benchmark",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark.")

    device = "cuda"
    dtype = torch.float32
    batch_size = 8
    nb_warmup = 10
    nb_forward_passes = 100
    nb_backward_passes = 100

    d_models = [64]
    context_lengths = [256, 1024, 4096, 8192, 16384]

    _ = device  # explicit to match the assignment parameter section
    gpu_name = torch.cuda.get_device_name(0)

    impls = []
    if args.impl in ("all", "pytorch_sdpa"):
        impls.append(("pytorch_sdpa", _compile_if_available(_pytorch_sdpa_impl)))
    if args.impl in ("all", "flash_triton"):
        impls.append(("flash_triton", _compile_if_available(_flash_triton_impl)))

    rows = []
    for d_model in d_models:
        for seq_len in context_lengths:
            for impl_name, impl_fn in impls:
                print(f"Running {impl_name} with d_model={d_model}, seq_len={seq_len}")
                row = _benchmark_one(
                    impl_name=impl_name,
                    impl_fn=impl_fn,
                    d_model=d_model,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    nb_warmup=nb_warmup,
                    nb_forward_passes=nb_forward_passes,
                    nb_backward_passes=nb_backward_passes,
                    gpu_name=gpu_name,
                )
                rows.append(row)

    _print_summary(rows)

    output_path = "outputs/csv/attention_benchmark.csv"
    _save_csv(rows, output_path)
    print(f"Saved benchmark results to {output_path}")


if __name__ == "__main__":
    main()
