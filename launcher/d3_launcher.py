#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Start ``vllm-srun`` for Decision 3.0 with the opt-in speed-ups of this image.

The runtime serves a Decision 3.0 model with exact element-wise ops: RMSNorm, residual adds, SiLU gating
and the gated-delta preparation run as many small PyTorch kernels. On a GB10 (273 GB/s memory) those are
30-38 % of a forward. The accelerator has fused Triton versions, but they are "approximate" kernels (they
differ from the eager ops by BF16 rounding) and the runtime only runs them for a family that consents
(``DtypePolicy.approximate_kernels``), which the Decision 3.0 family does not. Switching them on needs
the compiled ``causal-conv1d`` package too (the Dockerfile installs it): without it ``gdn_prep`` has no
fused kernel and the runtime skips the fused layers altogether.

Measured on the GB10 (594-token rows, ``benchmarks/profile_forward.py``): 3,500 -> 4,850 tokens/s
with 32 of 32 layers fused. Answers move by rounding only, so check them on your gold set before relying
on it.

Always applied (a compatibility fix, not a switch):

* The runtime's packed forwards (rows of a merged batch laid back to back) call
  ``torch.nn.attention.varlen.varlen_attn(..., scale=..., enable_gqa=...)``. The PyTorch of the NGC image
  (2.10 alpha) has neither argument, so every merged batch of rows of unequal length answered HTTP 500
  ``TypeError: varlen_attn() got an unexpected keyword argument 'scale'`` (about 1-5 % of requests once
  several clients ran). The launcher drops that kernel when the installed signature does not match, so
  merged batches are padded instead, which is what the runtime does where no such kernel exists.

Switches (environment variables, all off unless set to 1/true/yes/on):

``VLLM_SRUN_FUSED_KERNELS``  let the d3 model run the accelerator's fused approximate kernels.
``VLLM_SRUN_FP8``            hold the backbone's large linear layers as FP8 (e4m3) with per-channel weight
                             and per-token activation scales (``fp8_linear.py``): ~1.9x faster GEMMs on a
                             GB10 and ~7 GiB less weight memory, at the cost of quantization error.

Usage (the image's entrypoint; arguments are passed to ``vllm-srun`` unchanged):

    VLLM_SRUN_FUSED_KERNELS=1 VLLM_SRUN_FP8=1 python d3_launcher.py serve vllm-sr/d3-flash --device cuda:0
"""

import dataclasses
import inspect
import logging
import os
import sys
from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any, TypeVar

LOGGER = logging.getLogger("vllm_srun.d3_launcher")

FUSED_KERNELS_VARIABLE = "VLLM_SRUN_FUSED_KERNELS"
FP8_VARIABLE = "VLLM_SRUN_FP8"
TRUTHY = {"1", "true", "yes", "on"}
# What the runtime's `_wrap_varlen` passes to PyTorch's variable-length attention.
VARLEN_KEYWORDS = frozenset({"scale", "enable_gqa"})
VARLEN_KERNEL = "varlen_attention"
VARLEN_SOURCE = "torch-varlen-attention"

SpecT = TypeVar("SpecT")


def switch_enabled(environment: Mapping[str, str], variable: str) -> bool:
    """Whether ``variable`` is set to a true value in ``environment``."""
    return environment.get(variable, "").strip().lower() in TRUTHY


def consent_to_approximate_kernels(spec: SpecT) -> SpecT:
    """``spec`` (a ``ModelSpec``) with its ``DtypePolicy`` consenting to the accelerator's approximate kernels."""
    policy: Any = getattr(spec, "dtype")  # noqa: B009 - a dataclass field, named for the type checker
    return dataclasses.replace(spec, dtype=dataclasses.replace(policy, approximate_kernels=True))  # type: ignore[type-var]


def describe_with_consent(stock_describe: Callable[..., SpecT]) -> Callable[..., SpecT]:
    """Wrap ``Family.describe`` so the spec it returns consents to approximate kernels."""

    @wraps(stock_describe)
    def describe(self: Any, package: Any) -> SpecT:
        return consent_to_approximate_kernels(stock_describe(self, package))

    return describe


def varlen_attention_is_compatible(varlen_attn: Callable[..., Any]) -> bool:
    """Whether ``varlen_attn`` takes every keyword the runtime passes (``scale``, ``enable_gqa``)."""
    try:
        parameters = inspect.signature(varlen_attn).parameters
    except (TypeError, ValueError):
        return False
    return VARLEN_KEYWORDS <= set(parameters)


def without_varlen_attention(stock_kernels: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``Accelerator.kernels`` so the returned set has no PyTorch variable-length attention kernel."""

    @wraps(stock_kernels)
    def kernels(self: Any, device: Any) -> Any:
        kernel_set = stock_kernels(self, device)
        available = kernel_set.available.get(VARLEN_KERNEL, [])
        kernel_set.available[VARLEN_KERNEL] = [kernel for kernel in available if kernel.source != VARLEN_SOURCE]
        return kernel_set

    return kernels


def place_with_fp8_linears(stock_place: Callable[..., Any], convert: Callable[[Any], int]) -> Callable[..., Any]:
    """Wrap ``NativeEngine._place`` so the placed model's backbone has its large linear layers in FP8.

    ``convert`` is ``fp8_linear.convert_linears``; it runs before the first forward, so nothing captured
    by the fast path (graphs, fused layers) sees the BF16 modules.
    """

    @wraps(stock_place)
    def place(self: Any, *args: Any, **kwargs: Any) -> Any:
        model = stock_place(self, *args, **kwargs)
        converted = convert(model.backbone)
        LOGGER.warning("d3 launcher: %d linear layers held as FP8", converted)
        return model

    return place


def apply_patches(environment: Mapping[str, str]) -> list[str]:
    """Install the needed patches into the imported ``vllm_srun`` modules; returns their names."""
    applied: list[str] = []
    try:
        from torch.nn.attention.varlen import varlen_attn
    except ImportError:
        varlen_attn = None
    if varlen_attn is not None and not varlen_attention_is_compatible(varlen_attn):
        from vllm_srun.accel.gpu import GPUAccelerator

        GPUAccelerator.kernels = without_varlen_attention(GPUAccelerator.kernels)  # type: ignore[method-assign]
        applied.append("no packed forwards (torch varlen_attn takes no scale/enable_gqa)")
    if switch_enabled(environment, FUSED_KERNELS_VARIABLE):
        from vllm_srun.families.decision3.family import Decision3Family

        Decision3Family.describe = describe_with_consent(Decision3Family.describe)  # type: ignore[method-assign]
        applied.append("fused kernels")
    if switch_enabled(environment, FP8_VARIABLE):
        from fp8_linear import convert_linears
        from vllm_srun.engines.native.engine import NativeEngine

        NativeEngine._place = place_with_fp8_linears(NativeEngine._place, convert_linears)  # type: ignore[method-assign]
        applied.append("FP8 linear layers")
    return applied


def main() -> int:
    """Apply the enabled patches, then run the ``vllm-srun`` CLI."""
    from vllm_srun.cli import default_identity
    from vllm_srun.cli import main as run_vllm_srun

    # The engine modules import torch's compile caches, which need the identity defaults the CLI
    # otherwise sets first (the container uid has no passwd entry).
    default_identity()
    applied = apply_patches(os.environ)
    if applied:
        LOGGER.warning("d3 launcher: %s", ", ".join(applied))
    sys.argv[0] = "vllm-srun"
    return int(run_vllm_srun())


if __name__ == "__main__":
    sys.exit(main())
