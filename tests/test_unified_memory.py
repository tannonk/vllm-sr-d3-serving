#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for launcher/unified_memory.py (no torch or vllm-srun runtime needed).

Covers the ``/proc/meminfo`` reader, shared-pool detection, the ``GPUAccelerator.devices`` wrapper that reports
reclaimable memory as free on a unified-memory host, page-cache eviction and the ``NativeEngine._place`` wrapper,
using stand-ins for the runtime's ``DeviceInfo`` and ``ModelSpec``.
"""

import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

LAUNCHER_DIRECTORY = Path(__file__).parents[1] / "launcher"
GIB = 1 << 30
# The numbers of the DGX Spark this was found on: a full page cache leaves almost nothing "free".
HOST_TOTAL_KIB = 125505848
MEMINFO_FULL_CACHE = f"MemTotal: {HOST_TOTAL_KIB} kB\nMemFree: 683864 kB\nMemAvailable: 119647040 kB\n"


@dataclasses.dataclass(frozen=True)
class FakeDevice:
    index: int
    total_memory: int | None
    free_memory: int | None


@pytest.fixture(scope="module")
def unified_memory() -> ModuleType:
    """Import the module by path (launcher/ is not a package)."""
    spec = importlib.util.spec_from_file_location("unified_memory", LAUNCHER_DIRECTORY / "unified_memory.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def meminfo(tmp_path: Path) -> Path:
    path = tmp_path / "meminfo"
    path.write_text(MEMINFO_FULL_CACHE, encoding="utf-8")
    return path


def spark_device(free_gib: float) -> FakeDevice:
    return FakeDevice(index=0, total_memory=HOST_TOTAL_KIB * 1024, free_memory=int(free_gib * GIB))


def test_reads_mem_available_not_mem_free(unified_memory: ModuleType, meminfo: Path) -> None:
    assert unified_memory.read_meminfo_bytes("MemAvailable", meminfo) == 119647040 * 1024
    assert unified_memory.read_meminfo_bytes("MemFree", meminfo) == 683864 * 1024


def test_missing_field_or_file_is_unknown(unified_memory: ModuleType, meminfo: Path, tmp_path: Path) -> None:
    assert unified_memory.read_meminfo_bytes("Nonexistent", meminfo) is None
    assert unified_memory.read_meminfo_bytes("MemAvailable", tmp_path / "absent") is None


def test_malformed_field_is_unknown(unified_memory: ModuleType, tmp_path: Path) -> None:
    broken = tmp_path / "meminfo"
    broken.write_text("MemAvailable: lots kB\n", encoding="utf-8")

    assert unified_memory.read_meminfo_bytes("MemAvailable", broken) is None


@pytest.mark.parametrize(
    ("device_gib", "host_gib", "expected"),
    [(119.7, 119.7, True), (115.0, 119.7, True), (80.0, 119.7, False), (24.0, 125.0, False), (0, 119.7, False)],
)
def test_shared_pool_means_device_total_matches_host_total(
    unified_memory: ModuleType, device_gib: float, host_gib: float, expected: bool
) -> None:
    assert unified_memory.shares_host_memory(int(device_gib * GIB), int(host_gib * GIB)) is expected


def test_shared_pool_device_reports_reclaimable_memory_as_free(unified_memory: ModuleType, meminfo: Path) -> None:
    wrapped = unified_memory.devices_with_reclaimable_memory(lambda self: [spark_device(free_gib=0.6)], meminfo)

    (device,) = wrapped(object())

    assert device.free_memory == 119647040 * 1024
    assert device.total_memory == HOST_TOTAL_KIB * 1024


def test_reported_free_memory_is_capped_at_the_device_total(unified_memory: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "meminfo"
    path.write_text(f"MemTotal: {HOST_TOTAL_KIB} kB\nMemAvailable: {HOST_TOTAL_KIB + 4096} kB\n", encoding="utf-8")
    wrapped = unified_memory.devices_with_reclaimable_memory(lambda self: [spark_device(free_gib=1)], path)

    (device,) = wrapped(object())

    assert device.free_memory == device.total_memory


def test_never_lowers_what_the_device_reported(unified_memory: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "meminfo"
    path.write_text(f"MemTotal: {HOST_TOTAL_KIB} kB\nMemAvailable: {2 * 1024 * 1024} kB\n", encoding="utf-8")
    wrapped = unified_memory.devices_with_reclaimable_memory(lambda self: [spark_device(free_gib=60)], path)

    (device,) = wrapped(object())

    assert device.free_memory == 60 * GIB


def test_discrete_gpu_keeps_its_own_free_memory(unified_memory: ModuleType, meminfo: Path) -> None:
    discrete = FakeDevice(index=0, total_memory=24 * GIB, free_memory=3 * GIB)
    wrapped = unified_memory.devices_with_reclaimable_memory(lambda self: [discrete], meminfo)

    assert wrapped(object()) == [discrete]


def test_unknown_free_memory_stays_unknown(unified_memory: ModuleType, meminfo: Path) -> None:
    device = FakeDevice(index=0, total_memory=HOST_TOTAL_KIB * 1024, free_memory=None)
    wrapped = unified_memory.devices_with_reclaimable_memory(lambda self: [device], meminfo)

    assert wrapped(object()) == [device]


def test_unreadable_meminfo_returns_the_stock_devices(unified_memory: ModuleType, tmp_path: Path) -> None:
    stock = [spark_device(free_gib=0.6)]
    wrapped = unified_memory.devices_with_reclaimable_memory(lambda self: stock, tmp_path / "absent")

    assert wrapped(object()) == stock


def test_devices_wrapper_keeps_the_stock_name(unified_memory: ModuleType, meminfo: Path) -> None:
    def stock_devices(self: object) -> list[FakeDevice]:
        return []

    assert unified_memory.devices_with_reclaimable_memory(stock_devices, meminfo).__name__ == "stock_devices"


def test_evict_advises_each_readable_file_and_skips_missing_ones(unified_memory: ModuleType, tmp_path: Path) -> None:
    first, second = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    first.write_bytes(b"x" * 4096)
    second.write_bytes(b"y" * 4096)

    advised = unified_memory.evict_from_page_cache([first, tmp_path / "absent.safetensors", second])

    assert advised == 2
    assert first.read_bytes() == b"x" * 4096


def test_checkpoint_files_include_the_adapter_when_there_is_one(unified_memory: ModuleType) -> None:
    adapter = SimpleNamespace(weight_files=("adapter.safetensors",))
    plain = SimpleNamespace(backbone=SimpleNamespace(weight_files=("a.safetensors", "b.safetensors"), lora=None))
    adapted = SimpleNamespace(backbone=SimpleNamespace(weight_files=("a.safetensors",), lora=adapter))

    assert unified_memory.checkpoint_files(plain) == [Path("a.safetensors"), Path("b.safetensors")]
    assert unified_memory.checkpoint_files(adapted) == [Path("a.safetensors"), Path("adapter.safetensors")]


def test_place_wrapper_evicts_then_delegates_with_the_same_arguments(
    unified_memory: ModuleType, meminfo: Path, tmp_path: Path
) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"z" * 4096)
    spec = SimpleNamespace(backbone=SimpleNamespace(weight_files=(weights,), lora=None))
    calls: list[tuple[object, ...]] = []

    def stock_place(self: object, stacks: object, spec: object, *args: object, **kwargs: object) -> str:
        calls.append((self, stacks, spec, args, kwargs))
        return "placed"

    wrapped = unified_memory.place_after_evicting_checkpoints(stock_place, meminfo)
    engine, stacks = object(), object()

    result = wrapped(engine, stacks, spec, "accelerator", device="cuda:0")

    assert result == "placed"
    assert calls == [(engine, stacks, spec, ("accelerator",), {"device": "cuda:0"})]
