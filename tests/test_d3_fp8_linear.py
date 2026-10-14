#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for launcher/fp8_linear.py on the CPU (the GPU path needs a Blackwell/Hopper device).

Covers the row quantizer (scale and round trip), which layers are converted, the dequantizing CPU forward
against ``F.linear``, bias handling, and ``convert_linears`` replacing nested layers while leaving the small
and non-BF16 ones alone.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
F = torch.nn.functional

MODULE_PATH = Path(__file__).parents[1] / "launcher" / "fp8_linear.py"


@pytest.fixture(scope="module")
def fp8() -> ModuleType:
    """Import the module by path (launcher/ is not a package)."""
    spec = importlib.util.spec_from_file_location("fp8_linear", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bf16_linear(in_features: int, out_features: int, bias: bool = False) -> "nn.Linear":
    torch.manual_seed(0)
    return nn.Linear(in_features, out_features, bias=bias).to(torch.bfloat16)


def test_quantize_rows_scales_each_row_to_the_e4m3_range(fp8: ModuleType) -> None:
    tensor = torch.tensor([[1.0, -2.0, 4.0], [100.0, 50.0, -25.0]], dtype=torch.bfloat16)

    quantized, scale = fp8.quantize_rows(tensor)

    assert quantized.dtype == fp8.FP8_DTYPE
    assert scale.shape == (2, 1) and scale.dtype == torch.float32
    assert scale[0].item() == pytest.approx(4.0 / fp8.FP8_MAX)
    assert quantized.float().abs().amax(dim=1).tolist() == [fp8.FP8_MAX, fp8.FP8_MAX]


def test_quantize_rows_round_trips_within_the_e4m3_precision(fp8: ModuleType) -> None:
    torch.manual_seed(1)
    tensor = torch.randn(16, 256).to(torch.bfloat16)

    quantized, scale = fp8.quantize_rows(tensor)
    restored = quantized.float() * scale

    relative = (restored - tensor.float()).abs().sum() / tensor.float().abs().sum()
    assert relative.item() < 0.05


def test_an_all_zero_row_does_not_divide_by_zero(fp8: ModuleType) -> None:
    quantized, scale = fp8.quantize_rows(torch.zeros(2, 32, dtype=torch.bfloat16))
    assert torch.isfinite(quantized.float()).all()
    assert (scale > 0).all()


@pytest.mark.parametrize(
    ("in_features", "out_features", "dtype", "expected"),
    [
        (4096, 12288, torch.bfloat16, True),
        (4096, 32, torch.bfloat16, False),  # a narrow gate projection
        (4100, 4096, torch.bfloat16, False),  # not a multiple of 16
        (4096, 4096, torch.float32, False),
    ],
)
def test_is_convertible(fp8: ModuleType, in_features: int, out_features: int, dtype: "torch.dtype", expected: bool) -> None:
    linear = nn.Linear(in_features, out_features, bias=False).to(dtype)
    assert fp8.is_convertible(linear) is expected


def test_cpu_forward_is_close_to_the_bf16_linear(fp8: ModuleType) -> None:
    linear = bf16_linear(1024, 1024)
    converted = fp8.Fp8Linear(linear)
    x = torch.randn(3, 5, 1024).to(torch.bfloat16)

    expected = F.linear(x, linear.weight).float()
    actual = converted(x).float()

    assert actual.shape == expected.shape
    relative = (actual - expected).norm() / expected.norm()
    assert relative.item() < 0.06


def test_bias_is_kept_and_added(fp8: ModuleType) -> None:
    linear = bf16_linear(1024, 1024, bias=True)
    with torch.no_grad():
        linear.bias.fill_(3.0)
    converted = fp8.Fp8Linear(linear)

    zeros = torch.zeros(2, 1024, dtype=torch.bfloat16)

    assert torch.allclose(converted(zeros).float(), torch.full((2, 1024), 3.0), atol=1e-2)


def test_converted_layer_keeps_the_parameter_count(fp8: ModuleType) -> None:
    """The runtime compares the loaded parameter count with the manifest, so FP8 weights must be parameters."""
    linear = bf16_linear(1024, 2048, bias=True)
    converted = fp8.Fp8Linear(linear)

    assert sum(p.numel() for p in converted.parameters()) == sum(p.numel() for p in linear.parameters())


def test_fp8_weight_is_one_byte_per_element(fp8: ModuleType) -> None:
    converted = fp8.Fp8Linear(bf16_linear(1024, 2048))
    assert converted.weight_fp8.element_size() == 1
    assert converted.weight_fp8.shape == (2048, 1024)
    assert converted.weight_scale.shape == (1, 2048)


def test_convert_linears_replaces_nested_eligible_layers_only(fp8: ModuleType) -> None:
    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.big = bf16_linear(1024, 2048)
            self.gate = bf16_linear(1024, 32)
            self.fp32 = nn.Linear(1024, 1024, bias=False)

    class Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([Layer(), Layer()])
            self.embed = nn.Embedding(8, 16)

    backbone = Backbone()

    converted = fp8.convert_linears(backbone)

    assert converted == 2
    for layer in backbone.layers:
        assert isinstance(layer.big, fp8.Fp8Linear)
        assert isinstance(layer.gate, nn.Linear)
        assert isinstance(layer.fp32, nn.Linear)
    assert isinstance(backbone.embed, nn.Embedding)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device with Triton")
def test_triton_quantizer_matches_the_torch_one(fp8: ModuleType) -> None:
    torch.manual_seed(2)
    for features in (1024, 4096, 12288):
        tensor = (torch.randn(37, features, device="cuda") * 3).to(torch.bfloat16)

        expected, expected_scale = fp8.quantize_rows_torch(tensor)
        actual, actual_scale = fp8.quantize_rows_triton(tensor)

        assert torch.allclose(actual_scale, expected_scale, rtol=1e-6)
        restored_expected = expected.float() * expected_scale
        restored_actual = actual.float() * actual_scale
        assert (restored_actual - restored_expected).abs().max().item() <= 0.07 * tensor.float().abs().max().item()
