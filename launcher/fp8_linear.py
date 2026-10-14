#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FP8 (e4m3) replacements for the large ``nn.Linear`` layers of a decoder backbone.

On a GB10 the BF16 GEMMs of a d3-flash forward run at ~90-100 TFLOPS and the FP8 ones at ~160-185
(``benchmarks/fp8_gemm_check.py``); once the element-wise ops are fused the GEMMs are 75-80 % of
kernel time. ``Fp8Linear`` keeps the weight as e4m3 with one FP32 scale per output channel (so a channel with
large weights does not flatten the others), quantizes the activations per token on the fly and multiplies
with ``torch._scaled_mm``. The bias stays in the module's dtype. The BF16 weight is dropped, which also
halves the weight memory of the converted layers.

Only layers where it pays are converted: BF16 weights, both feature counts multiples of 16 (a
``_scaled_mm`` requirement) and at least ``MIN_FEATURES`` wide. On a device without CUDA the module
dequantizes and calls ``F.linear``, so the numerics can be unit-tested without a GPU.

This is an approximation: the answers differ from the BF16 ones by quantization error, so validate it on
the data you care about before enabling it (``VLLM_SRUN_FP8`` in ``d3_launcher.py``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

try:  # Triton ships with the CUDA builds of torch; the CPU path needs none of it
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only where Triton is missing
    triton = None
    tl = None

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0  # largest finite e4m3fn value
SCALE_FLOOR = 1e-12
FEATURE_MULTIPLE = 16
# The two small gated-delta gate projections (32 outputs) gain nothing and are the most sensitive.
MIN_FEATURES = 1024


def quantize_rows_torch(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``tensor`` ([rows, features]) as e4m3 plus one FP32 scale per row ([rows, 1]); ``fp8 * scale ~= tensor``.

    Several element-wise passes over the tensor: fine for weights (once), slow for activations (every call).
    """
    scale = (tensor.abs().amax(dim=1, keepdim=True).float() / FP8_MAX).clamp_(min=SCALE_FLOOR)
    return (tensor / scale.to(tensor.dtype)).to(FP8_DTYPE), scale


if triton is not None:

    @triton.jit
    def _quantize_rows_kernel(  # type: ignore[no-untyped-def]
        x_ptr, q_ptr, scale_ptr, features, stride_x, stride_q,
        FP8_MAX_VALUE: tl.constexpr, FLOOR: tl.constexpr, BLOCK: tl.constexpr,
    ):  # fmt: skip
        """One program per row: absolute maximum, scale and e4m3 cast in a single read and write of the row."""
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < features
        values = tl.load(x_ptr + row * stride_x + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.maximum(tl.max(tl.abs(values), axis=0) / FP8_MAX_VALUE, FLOOR)
        tl.store(q_ptr + row * stride_q + offsets, (values / scale).to(tl.float8e4nv), mask=mask)
        tl.store(scale_ptr + row, scale)


def quantize_rows_triton(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``quantize_rows_torch`` in one fused kernel (CUDA, 2-D, contiguous rows)."""
    rows, features = tensor.shape
    tensor = tensor.contiguous()
    quantized = torch.empty((rows, features), dtype=FP8_DTYPE, device=tensor.device)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=tensor.device)
    block = triton.next_power_of_2(features)
    _quantize_rows_kernel[(rows,)](
        tensor, quantized, scale, features, tensor.stride(0), quantized.stride(0),
        FP8_MAX_VALUE=FP8_MAX, FLOOR=SCALE_FLOOR, BLOCK=block, num_warps=8 if block > 4096 else 4,
    )  # fmt: skip
    return quantized, scale


def quantize_rows(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The fused Triton quantizer on CUDA where available, else the PyTorch one."""
    if triton is not None and tensor.is_cuda and tensor.dim() == 2:
        return quantize_rows_triton(tensor)
    return quantize_rows_torch(tensor)


def is_convertible(linear: nn.Linear, min_features: int = MIN_FEATURES) -> bool:
    """Whether ``linear`` is a BF16 layer wide enough, and aligned, for ``Fp8Linear``."""
    return (
        linear.weight.dtype == torch.bfloat16
        and linear.in_features % FEATURE_MULTIPLE == 0
        and linear.out_features % FEATURE_MULTIPLE == 0
        and min(linear.in_features, linear.out_features) >= min_features
    )


class Fp8Linear(nn.Module):
    """``F.linear`` with an e4m3 weight (per-output-channel scale) and per-token dynamic activation scales."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        with torch.no_grad():
            weight, scale = quantize_rows(linear.weight.data)
        # Parameters, not buffers: the runtime checks the loaded parameter count against the package manifest.
        self.weight_fp8 = nn.Parameter(weight, requires_grad=False)  # [out, in]
        self.register_buffer("weight_scale", scale.t().contiguous())  # [1, out]
        self.bias = nn.Parameter(linear.bias.data, requires_grad=False) if linear.bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        rows = x.reshape(-1, self.in_features)
        if not rows.is_cuda:
            weight = (self.weight_fp8.float() * self.weight_scale.t()).to(x.dtype)
            bias = self.bias.to(x.dtype) if self.bias is not None else None
            return F.linear(x, weight, bias)
        activations, scale = quantize_rows(rows)
        out = torch._scaled_mm(
            activations,
            self.weight_fp8.t(),
            scale_a=scale,
            scale_b=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
        )
        return out.reshape(*shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, fp8=e4m3"


def convert_linears(module: nn.Module, min_features: int = MIN_FEATURES) -> int:
    """Replace every eligible ``nn.Linear`` below ``module`` with an ``Fp8Linear``; returns how many."""
    converted = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if is_convertible(child, min_features):
                setattr(module, name, Fp8Linear(child))
                converted += 1
        else:
            converted += convert_linears(child, min_features)
    return converted
