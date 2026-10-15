#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for launcher/d3_launcher.py (no vllm-srun runtime needed).

Covers the opt-in switch parsing and the pure spec transformation that makes a Decision 3.0 model
consent to the accelerator's approximate kernels, using stand-ins for the runtime's ``ModelSpec`` and
``DtypePolicy`` dataclasses.
"""

import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

LAUNCHER_PATH = Path(__file__).parents[1] / "launcher" / "d3_launcher.py"


@dataclasses.dataclass(frozen=True)
class FakePolicy:
    weights: str = "float32"
    gpu_weights: str | None = "bfloat16"
    approximate_kernels: bool = False


@dataclasses.dataclass(frozen=True)
class FakeSpec:
    name: str
    dtype: FakePolicy


@pytest.fixture(scope="module")
def launcher() -> ModuleType:
    """Import the launcher as a module (launcher/ is not a package)."""
    spec = importlib.util.spec_from_file_location("d3_launcher", LAUNCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), (" ON ", True), ("Yes", True), ("0", False), ("", False), ("no", False)],
)
def test_switch_enabled_parses_true_values(launcher: ModuleType, value: str, expected: bool) -> None:
    assert launcher.switch_enabled({launcher.FUSED_KERNELS_VARIABLE: value}, launcher.FUSED_KERNELS_VARIABLE) is expected


def test_switch_is_off_when_unset(launcher: ModuleType) -> None:
    assert launcher.switch_enabled({}, launcher.FUSED_KERNELS_VARIABLE) is False


def test_unified_memory_is_on_when_unset_and_off_when_set_to_zero(launcher: ModuleType) -> None:
    variable = launcher.UNIFIED_MEMORY_VARIABLE

    assert launcher.switch_enabled({}, variable, default=True) is True
    assert launcher.switch_enabled({variable: "1"}, variable, default=True) is True
    assert launcher.switch_enabled({variable: "0"}, variable, default=True) is False
    assert launcher.switch_enabled({variable: ""}, variable, default=True) is False


def test_consent_sets_only_the_approximate_kernels_flag(launcher: ModuleType) -> None:
    spec = FakeSpec("d3-flash", FakePolicy())

    consented = launcher.consent_to_approximate_kernels(spec)

    assert consented.dtype.approximate_kernels is True
    assert consented.dtype.gpu_weights == "bfloat16"
    assert consented.name == "d3-flash"
    assert spec.dtype.approximate_kernels is False


def test_describe_wrapper_returns_a_consenting_spec_and_keeps_the_arguments(launcher: ModuleType) -> None:
    seen: list[tuple[object, object]] = []

    def stock_describe(self: object, package: object) -> FakeSpec:
        seen.append((self, package))
        return FakeSpec("d3-flash", FakePolicy())

    wrapped = launcher.describe_with_consent(stock_describe)
    family, package = object(), object()

    result = wrapped(family, package)

    assert seen == [(family, package)]
    assert result.dtype.approximate_kernels is True
    assert wrapped.__name__ == "stock_describe"


def test_varlen_attention_with_scale_and_gqa_is_compatible(launcher: ModuleType) -> None:
    def varlen_attn(query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, *, scale=None, enable_gqa=False):  # type: ignore[no-untyped-def]
        return query

    assert launcher.varlen_attention_is_compatible(varlen_attn) is True


def test_varlen_attention_of_torch_2_10_is_not_compatible(launcher: ModuleType) -> None:
    def varlen_attn(query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, is_causal=False, return_aux=None):  # type: ignore[no-untyped-def]
        return query

    assert launcher.varlen_attention_is_compatible(varlen_attn) is False


def test_varlen_attention_without_a_signature_is_not_compatible(launcher: ModuleType) -> None:
    assert launcher.varlen_attention_is_compatible(print) in (True, False)  # never raises
    assert launcher.varlen_attention_is_compatible(object()) is False


def test_without_varlen_attention_drops_only_the_torch_kernel(launcher: ModuleType) -> None:
    @dataclasses.dataclass(frozen=True)
    class FakeKernel:
        source: str

    class FakeKernelSet:
        def __init__(self) -> None:
            self.available = {
                "varlen_attention": [FakeKernel("torch-varlen-attention"), FakeKernel("torch-reference")],
                "sdpa": [FakeKernel("torch-reference")],
            }

    def stock_kernels(self: object, device: object) -> FakeKernelSet:
        return FakeKernelSet()

    wrapped = launcher.without_varlen_attention(stock_kernels)

    kernel_set = wrapped(object(), object())

    assert [kernel.source for kernel in kernel_set.available["varlen_attention"]] == ["torch-reference"]
    assert [kernel.source for kernel in kernel_set.available["sdpa"]] == ["torch-reference"]
