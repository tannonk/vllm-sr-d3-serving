# ADR 0001: Count reclaimable page cache as free memory on shared-pool GPUs

Status: accepted (2026-10-15)

## Context

On a DGX Spark (GB10) the CPU and GPU share one 119.7 GiB pool. `vllm-srun` refuses to load a model when
`DeviceInfo.free_memory` is below its estimate (`placement.py`), and fills `free_memory` from
`torch.cuda.mem_get_info`. On this hardware that returns roughly `MemFree`, which excludes the page cache.

Observed on the second Spark (cache full after normal use):

| Source | Value |
| --- | --- |
| `torch.cuda.mem_get_info` free | 0.6 GiB |
| `/proc/meminfo` `MemAvailable` | 119.6 GiB |
| d3-flash estimate (8.4 B x 2 B x 1.10 + 2 GiB) | 19.2 GiB |

The runtime raised `PlacementError`, retried with backoff and never recovered, because nothing evicted the cache.
The first Spark had a cold cache, so the same image worked there. A 21 GiB `cudaMalloc` on the failing host succeeded in
1.1 s and the kernel released ~30 GiB of cache, so the refusal was spurious.

## Decision

`launcher/unified_memory.py`, applied by `d3_launcher.py` unless `VLLM_SRUN_UNIFIED_MEMORY=0`:

1. Wrap `GPUAccelerator.devices` so a device whose total memory is within 10 % of the host's `MemTotal` reports
   `max(reported free, min(MemAvailable, total))`. Discrete GPUs and unknown values are untouched.
2. Wrap `NativeEngine._place` to `posix_fadvise(DONTNEED)` the checkpoint files before the device copy. This needs no
   root, and it also raises the free memory the runtime's CUDA-graph budget (`free // 2`) is computed from.

Default on, because it is a portability fix rather than a trade-off, and it is inert on discrete GPUs.

## Alternatives considered

- **Port the whole `unified_memory_launcher.py` of the sibling project.** Its FP32-load and size-estimate corrections
  are not needed: the pinned runtime commit reads Decision 3.0 weights straight into BF16 and estimates 2 bytes/param,
  and its `set_parameter` signature differs.
- **`--memory-budget`.** It only lowers the limit; it cannot raise the probe.
- **`drop_caches` in `serve.sh`.** Needs root, runs outside the container, and is undone as the cache refills.
- **Detect with `device.is_integrated`.** `DeviceInfo` does not carry it; comparing totals needs no extra query.

## Consequences

- Memory that really is in use (anonymous or pinned pages of other processes) is already excluded from `MemAvailable`,
  so placement still refuses a model that cannot fit. What changes is the edge case where `MemAvailable` is optimistic,
  for example memory the kernel could only free by swapping: the load then fails at allocation time, not at placement.
- Bumping `SEMANTIC_ROUTER_REF` requires re-checking `GPUAccelerator.devices` and `NativeEngine._place`.
