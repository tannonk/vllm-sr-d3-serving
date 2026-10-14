#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Where does a d3 forward spend its time? Load the model like the server and profile merged forwards.

``fp8_gemm_check.py`` shows what a GEMM can do; this shows how much of a real forward the GEMMs are.
It loads ``vllm-sr/d3-flash`` in-process with the runtime's own loader and engine options (no HTTP server,
no scheduler), builds requests shaped like the person-role workload (a state of about ``--state-tokens``
tokens, ``--rows`` role questions), runs ``run_approximate`` (what a merged serving forward runs) and
reports, per row count:

* wall time per forward, tokens/s, and the share of wall time a GPU kernel was running (low = launch or
  host bound, high = the kernels themselves are the limit);
* CUDA time grouped by kind (GEMM, gated-delta/FLA, attention, convolution, element-wise, other) and the
  ten most expensive kernels.

Needs the model in the cache (run ``serve.sh`` once) and an idle GPU. Run it inside the image:

    docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" -e HOME=/models \
        -v "$HOME/.cache/vllm-srun":/models \
        -v "$PWD/benchmarks":/bench:ro -v "$PWD/launcher":/launcher:ro --entrypoint python \
        vllm-srun-d3:<tag> /bench/profile_forward.py --rows 1 2 4
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

MODEL_ID = "vllm-sr/d3-flash"
SENTENCE = "The host welcomes the guest to this weekly show about technology, science and society. "
# A five-option role question: the workload the published numbers were measured on.
ROLE_CRITERIA = {
    "host": "The person who presents and leads this episode on their own.",
    "co-host": "One of several people who present this episode together.",
    "guest": "A person invited to take part in this episode, for example by being interviewed or joining the conversation.",
    "mentioned": "A person who is talked about or referred to but does not take part in this episode.",
    "producer": "A person credited for behind-the-scenes work on the episode, such as production, editing or sound.",
}
KERNEL_KINDS = (
    ("gemm", re.compile(r"gemm|cutlass|cublas|nvjet|xmma|matmul|splitk|sm\d+_|ampere_|hopper_", re.I)),
    ("gated-delta / FLA", re.compile(r"chunk|gated_delta|fwd_prepare|recompute|solve_tril|fla|l2norm|kkt|wy_", re.I)),
    ("attention", re.compile(r"flash|fmha|attention|sdpa|softmax|varlen", re.I)),
    ("convolution", re.compile(r"conv", re.I)),
    ("element-wise / copy", re.compile(r"elementwise|vectorized|copy|cat|index|fill|mul|add|silu|sigmoid|norm|pow|rsqrt|cast|to_", re.I)),
)


def kernel_kind(name: str) -> str:
    """The first kind of ``KERNEL_KINDS`` whose pattern matches the kernel's name, else ``other``."""
    for kind, pattern in KERNEL_KINDS:
        if pattern.search(name):
            return kind
    return "other"


def build_questions(rows: int) -> dict[str, Any]:
    """``rows`` role questions about distinct names, each with the same five options."""
    return {
        f"person_{index}": {
            "type": "choice",
            "instructions": f'What is the role of "Person Number {index}" in this podcast episode?',
            "criteria": ROLE_CRITERIA,
        }
        for index in range(rows)
    }


def load_model(cache_dir: str | None, approximate_kernels: bool = False) -> Any:
    """The d3-flash package loaded as the server loads it: resolved, verified, placed, engine options as served.

    ``approximate_kernels`` makes the model consent to the accelerator's fused (approximate) kernels, which the
    d3 family does not do by itself; the server never sets it.
    """
    from vllm_srun.cli import default_identity

    default_identity()
    from vllm_srun.accel.autotune import freeze_autotune

    freeze_autotune("/models/autotune")  # before FLA is imported, as the server does
    from vllm_srun.accel.cuda import CUDAAccelerator
    from vllm_srun.engines.native.engine import NativeEngine
    from vllm_srun.families.decision3.family import Decision3Family
    from vllm_srun.plugins.base import EngineOptions, RegistryOptions
    from vllm_srun.registry.resolve import resolve

    options = RegistryOptions(cache_dir=cache_dir, offline=True)
    reference = resolve(MODEL_ID, revision=None, cache_dir=cache_dir, offline=True)
    family = Decision3Family(options)
    package = family.verify(family.fetch(reference))
    spec = family.describe(package)
    if approximate_kernels:
        from dataclasses import replace

        spec = replace(spec, dtype=replace(spec.dtype, approximate_kernels=True))
    accelerator = CUDAAccelerator()
    device = accelerator.devices()[0]
    # The serving options of `--dtype auto`: reduced copies and consented kernels allowed.
    engine_options = EngineOptions(exact_kernels_only=False, reduced_precision=True)
    engine_model = accelerator.execute(
        device, lambda: NativeEngine().load(spec, accelerator, device, engine_options)
    )
    return accelerator.execute(device, lambda: family.load(package, spec, engine_model)), accelerator, device


def time_forwards(run: Any, iterations: int) -> float:
    """Mean wall seconds of ``run()`` over ``iterations`` calls, each ended by a device synchronize."""
    import torch

    run()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        run()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / iterations


def profile_forwards(run: Any, iterations: int) -> tuple[dict[str, float], list[tuple[str, float, int]], float]:
    """CUDA seconds per kernel kind, the top kernels (name, seconds, calls) and the CUDA total, per iteration."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        for _ in range(iterations):
            run()
        torch.cuda.synchronize()
    by_kind: dict[str, float] = defaultdict(float)
    kernels: list[tuple[str, float, int]] = []
    for event in profiler.key_averages():
        if getattr(event, "device_type", None) != torch.autograd.DeviceType.CUDA:
            continue
        seconds = getattr(event, "self_device_time_total", None)
        if seconds is None:
            seconds = getattr(event, "self_cuda_time_total", 0.0)
        seconds = seconds * 1e-6 / iterations
        by_kind[kernel_kind(event.key)] += seconds
        kernels.append((event.key, seconds, int(event.count // iterations)))
    kernels.sort(key=lambda entry: entry[1], reverse=True)
    return dict(by_kind), kernels, sum(by_kind.values())


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4], help="Questions per forward (default: %(default)s).")
    parser.add_argument("--state-tokens", type=int, default=400, help="Approximate state length (default: %(default)s).")
    parser.add_argument("--iterations", type=int, default=5, help="Timed forwards per row count (default: %(default)s).")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face hub cache (default: HF_HUB_CACHE).")
    parser.add_argument("--approximate-kernels", action="store_true",
                        help="Let the model use the accelerator's fused approximate kernels (what the server never does).")
    parser.add_argument("--fp8", action="store_true",
                        help="Convert the backbone's large linear layers to FP8 (needs launcher/ mounted at /launcher).")
    parser.add_argument("--dump", type=Path, default=None,
                        help="Write the probabilities of every row to this JSON file, to compare configurations.")
    return parser


def main() -> int:
    args = create_parser().parse_args()
    import torch

    if not torch.cuda.is_available():
        print("no CUDA device available", file=sys.stderr)
        return 1
    model, accelerator, device = load_model(args.cache_dir, args.approximate_kernels)
    if args.fp8:
        sys.path.insert(0, "/launcher")
        from fp8_linear import convert_linears

        before = torch.cuda.memory_allocated() / 2**30
        print(f"converted {convert_linears(model.engine_model.backbone)} linear layers to FP8 "
              f"(allocated {before:.1f} -> {torch.cuda.memory_allocated() / 2**30:.1f} GiB)")
    state = (SENTENCE * max(1, args.state_tokens // 16)).strip()
    print(f"{torch.cuda.get_device_name(0)}, model memory {model.engine_model.memory_bytes() / 2**30:.1f} GiB, "
          f"packs merged rows: {model.packs_merged_rows}")
    print(f"receipt: {json.dumps(model.engine_model.receipt(), default=str, indent=1)}")
    dumped: dict[str, Any] = {}
    for rows in args.rows:
        plan = model.plan(state, build_questions(rows))
        items = plan.items
        tokens = sum(len(item.ids) for item in items)

        def run() -> Any:
            return accelerator.execute(device, lambda: model.run_approximate(items))

        dumped[str(rows)] = [[float(value) for value in row] if row is not None else None for row in run()]
        wall = time_forwards(run, args.iterations)
        by_kind, kernels, cuda_total = profile_forwards(run, 3)
        print(f"\nrows={rows} tokens={tokens} ({tokens // rows}/row)  wall {wall * 1000:.0f} ms  "
              f"{tokens / wall:.0f} tokens/s  GPU kernels busy {100 * cuda_total / wall:.0f}% of wall")
        for kind, seconds in sorted(by_kind.items(), key=lambda entry: entry[1], reverse=True):
            print(f"    {kind:22s} {seconds * 1000:8.1f} ms  {100 * seconds / cuda_total:5.1f}% of kernel time")
        for name, seconds, calls in kernels[:10]:
            print(f"      {seconds * 1000:7.1f} ms x{calls:<4d} {name[:100]}")
    if args.dump is not None:
        args.dump.write_text(json.dumps(dumped), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
