#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is FP8 worth it on this GPU? Time BF16 against FP8 (``torch._scaled_mm``) for d3-flash's GEMM shapes.

Decides whether the FP8 linear-layer patch (launcher/fp8_linear.py) is worth it on a given GPU. For each shape
``[M tokens, K] x [K, N]`` it times a BF16 ``F.linear`` and the FP8 e4m3 ``torch._scaled_mm`` with
per-tensor and, where the device supports it, per-row/per-channel scales, and prints achieved TFLOPS and
the FP8/BF16 speed-up. Shapes are the projections of a Qwen3.5-9B layer (hidden 4096, MLP 12288) at the
token counts one forward holds (about 512-4096 on this host). Prints the measured numbers only; a
rowwise scaling that the torch build does not support is reported as such, not as a failure.

Needs a CUDA GPU; takes a few seconds and a few hundred MiB. Run it inside the d3 image, ideally while no
model server is busy (a shared GPU makes the numbers meaningless):

    docker run --rm --gpus all --entrypoint python \
        -v "$PWD/benchmarks:/bench:ro" vllm-srun-d3:<tag> /bench/fp8_gemm_check.py
"""

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

TOKEN_COUNTS = (512, 1024, 2048, 4096)
# (K, N, name): the large projections of a Qwen3.5-9B decoder layer.
SHAPES = ((4096, 12288, "mlp gate/up"), (12288, 4096, "mlp down"), (4096, 4096, "attention/projection"))
FP8_MAX = 448.0  # largest finite float8_e4m3fn value


@dataclass(frozen=True)
class Timing:
    """Milliseconds per call and achieved TFLOPS of one GEMM variant."""

    milliseconds: float
    tflops: float


def time_gpu(work: Callable[[], object], flops: float, warmup: int, iterations: int) -> Timing:
    """Mean GPU time of ``work`` (CUDA events, after ``warmup`` untimed calls)."""
    for _ in range(warmup):
        work()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        work()
    stop.record()
    torch.cuda.synchronize()
    milliseconds = start.elapsed_time(stop) / iterations
    return Timing(milliseconds, flops / (milliseconds * 1e-3) / 1e12)


def fp8_per_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``tensor`` as e4m3 plus its FP32 scalar scale (``tensor ~= fp8 * scale``)."""
    scale = (tensor.abs().amax().float() / FP8_MAX).clamp(min=1e-12)
    return (tensor.float() / scale).to(torch.float8_e4m3fn), scale


def fp8_per_row(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``tensor`` as e4m3 plus one FP32 scale per row, shape ``[rows, 1]``."""
    scale = (tensor.abs().amax(dim=1, keepdim=True).float() / FP8_MAX).clamp(min=1e-12)
    return (tensor.float() / scale).to(torch.float8_e4m3fn), scale


def benchmark_shape(tokens: int, k: int, n: int, warmup: int, iterations: int) -> dict[str, Timing | str]:
    """Timings of the BF16, per-tensor FP8 and rowwise FP8 GEMM of one shape."""
    activations = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    flops = 2.0 * tokens * k * n
    results: dict[str, Timing | str] = {
        "bf16": time_gpu(lambda: F.linear(activations, weight), flops, warmup, iterations)
    }

    weight_fp8, weight_scale = fp8_per_tensor(weight)
    activations_fp8, activations_scale = fp8_per_tensor(activations)
    try:
        results["fp8 per-tensor"] = time_gpu(
            lambda: torch._scaled_mm(
                activations_fp8, weight_fp8.t(), scale_a=activations_scale, scale_b=weight_scale,
                out_dtype=torch.bfloat16,
            ),
            flops, warmup, iterations,
        )
    except (RuntimeError, NotImplementedError) as error:
        results["fp8 per-tensor"] = f"unsupported: {str(error).splitlines()[0][:90]}"

    weight_rows, weight_row_scale = fp8_per_row(weight)
    activations_rows, activations_row_scale = fp8_per_row(activations)
    try:
        results["fp8 per-row"] = time_gpu(
            lambda: torch._scaled_mm(
                activations_rows, weight_rows.t(), scale_a=activations_row_scale,
                scale_b=weight_row_scale.t().contiguous(), out_dtype=torch.bfloat16,
            ),
            flops, warmup, iterations,
        )
    except (RuntimeError, NotImplementedError) as error:
        results["fp8 per-row"] = f"unsupported: {str(error).splitlines()[0][:90]}"
    return results


def format_results(tokens: int, k: int, n: int, name: str, results: dict[str, Timing | str]) -> list[str]:
    """Report lines of one shape, with the speed-up of each FP8 variant over BF16."""
    baseline = results["bf16"]
    assert isinstance(baseline, Timing)
    lines = [f"M={tokens:5d} K={k:5d} N={n:5d} ({name})"]
    for variant, timing in results.items():
        if isinstance(timing, str):
            lines.append(f"    {variant:15s} {timing}")
            continue
        speedup = "" if variant == "bf16" else f"   x{baseline.milliseconds / timing.milliseconds:.2f} vs bf16"
        lines.append(f"    {variant:15s} {timing.milliseconds:7.3f} ms {timing.tflops:7.1f} TFLOPS{speedup}")
    return lines


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--warmup", type=int, default=5, help="Untimed calls per variant (default: %(default)s).")
    parser.add_argument("--iterations", type=int, default=20, help="Timed calls per variant (default: %(default)s).")
    parser.add_argument("--tokens", type=int, nargs="+", default=list(TOKEN_COUNTS), help="Token counts M to time.")
    return parser


def main() -> int:
    args = create_parser().parse_args()
    if not torch.cuda.is_available():
        print("no CUDA device available", file=sys.stderr)
        return 1
    if not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "_scaled_mm"):
        print(f"torch {torch.__version__} has no float8_e4m3fn / _scaled_mm", file=sys.stderr)
        return 1
    properties = torch.cuda.get_device_properties(0)
    print(f"{properties.name} sm_{properties.major}{properties.minor}, torch {torch.__version__}, "
          f"cuda {torch.version.cuda}")
    for k, n, name in SHAPES:
        for tokens in args.tokens:
            results = benchmark_shape(tokens, k, n, args.warmup, args.iterations)
            print("\n".join(format_results(tokens, k, n, name, results)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
