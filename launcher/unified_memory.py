#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Make ``vllm-srun`` placement see the memory a unified-memory host can really offer (DGX Spark).

On a GB10 the CPU and the GPU share one memory pool. ``torch.cuda.mem_get_info`` reports roughly the
kernel's ``MemFree`` there, which leaves out the page cache. After a few reads of large files the cache holds
nearly all of the pool (100+ GiB of ``Inactive(file)`` on a 119 GiB host) while ``MemFree`` is under 1 GiB, so
the runtime's placement check refuses a 19 GiB model with ``needs about 19.2 GiB, 0.6 GiB free`` even though the
kernel would hand the cache back the moment the GPU driver asks for it (``htop`` and ``MemAvailable`` agree).

Two corrections, both applied by ``d3_launcher.py`` unless ``VLLM_SRUN_UNIFIED_MEMORY=0``:

* ``devices_with_reclaimable_memory`` wraps ``GPUAccelerator.devices`` so a device that shares the host's pool
  reports ``MemAvailable`` as its free memory. A discrete GPU (its total memory is not the host's) is left alone,
  because there ``MemAvailable`` describes host RAM and would overstate the device.
* ``place_after_evicting_checkpoints`` wraps ``NativeEngine._place`` so the checkpoint files, which hashing the
  package left in the cache, are dropped before the weights are copied to the device. That does not depend on
  root (``posix_fadvise``), unlike ``echo 1 > /proc/sys/vm/drop_caches``.

This module imports nothing from torch or the runtime, so its logic is testable on a CPU-only machine.
"""

import dataclasses
import logging
import os
from collections.abc import Callable, Iterable
from functools import wraps
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("vllm_srun.d3_launcher.unified_memory")

MEMINFO_PATH = Path("/proc/meminfo")
GIB = 1 << 30
# A GPU whose total memory is within this share of the host's MemTotal is taken to draw on the host's pool.
SHARED_POOL_TOLERANCE = 0.10


def read_meminfo_bytes(field: str, meminfo_path: Path = MEMINFO_PATH) -> int | None:
    """The ``field`` of ``/proc/meminfo`` (e.g. ``MemAvailable``) in bytes, or None if it cannot be read."""
    try:
        lines = meminfo_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        LOGGER.warning("cannot read %s: %s", meminfo_path, error)
        return None
    prefix = f"{field}:"
    for line in lines:
        if line.startswith(prefix):
            try:
                return int(line.split()[1]) * 1024
            except (IndexError, ValueError):
                return None
    return None


def shares_host_memory(device_total_bytes: int, host_total_bytes: int) -> bool:
    """Whether a device with ``device_total_bytes`` is one that draws on the host's ``host_total_bytes`` pool."""
    if device_total_bytes <= 0 or host_total_bytes <= 0:
        return False
    return abs(device_total_bytes - host_total_bytes) <= SHARED_POOL_TOLERANCE * host_total_bytes


def with_reclaimable_free_memory(info: Any, available_bytes: int, host_total_bytes: int) -> Any:
    """``info`` (a ``DeviceInfo``) with ``free_memory`` raised to ``available_bytes`` on a shared-pool device.

    Never lowers the figure the device reported, and leaves an unknown (None) one unknown so the runtime keeps
    skipping its check.
    """
    total = info.total_memory
    if info.free_memory is None or total is None or not shares_host_memory(total, host_total_bytes):
        return info
    return dataclasses.replace(info, free_memory=max(info.free_memory, min(available_bytes, total)))


def devices_with_reclaimable_memory(
    stock_devices: Callable[..., list[Any]], meminfo_path: Path = MEMINFO_PATH
) -> Callable[..., list[Any]]:
    """Wrap ``GPUAccelerator.devices`` so shared-pool devices report the memory the kernel can reclaim."""

    @wraps(stock_devices)
    def devices(self: Any) -> list[Any]:
        infos = stock_devices(self)
        available = read_meminfo_bytes("MemAvailable", meminfo_path)
        host_total = read_meminfo_bytes("MemTotal", meminfo_path)
        if available is None or host_total is None:
            return infos
        return [with_reclaimable_free_memory(info, available, host_total) for info in infos]

    return devices


def evict_from_page_cache(files: Iterable[Path]) -> int:
    """Ask the kernel to drop the cached pages of ``files``; returns how many files were advised."""
    advised = 0
    for path in files:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError as error:
            LOGGER.warning("cannot evict %s from the page cache: %s", path, error)
            continue
        try:
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
            advised += 1
        except OSError as error:
            LOGGER.warning("cannot evict %s from the page cache: %s", path, error)
        finally:
            os.close(descriptor)
    return advised


def checkpoint_files(spec: Any) -> list[Path]:
    """The weight files of a ``ModelSpec``'s backbone, including its LoRA adapter's."""
    backbone = spec.backbone
    adapter = getattr(backbone, "lora", None)
    adapter_files = adapter.weight_files if adapter is not None else ()
    return [Path(path) for path in (*backbone.weight_files, *adapter_files)]


def place_after_evicting_checkpoints(stock_place: Callable[..., Any], meminfo_path: Path = MEMINFO_PATH) -> Callable[..., Any]:
    """Wrap ``NativeEngine._place`` so the checkpoint files leave the page cache before the device copy."""

    @wraps(stock_place)
    def place(self: Any, stacks: Any, spec: Any, *args: Any, **kwargs: Any) -> Any:
        advised = evict_from_page_cache(checkpoint_files(spec))
        available = read_meminfo_bytes("MemAvailable", meminfo_path)
        LOGGER.info(
            "host load done; dropped %d checkpoint files from the page cache, %.1f GiB available",
            advised,
            (available or 0) / GIB,
        )
        return stock_place(self, stacks, spec, *args, **kwargs)

    return place
