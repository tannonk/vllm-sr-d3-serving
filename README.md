# d3-spark-serving

Throughput-tuned serving of [vllm-sr Decision 3.0](https://huggingface.co/vllm-sr/d3-flash) models
(`d3-flash`, 8.4B, Qwen3.5-based, text + image + video) behind the `vllm-srun` runtime, in a container that
builds and runs on **aarch64 + CUDA hosts such as the NVIDIA DGX Spark (GB10)**. It exists because the stock
runtime is not the fast path on that hardware: it ships with kernels, a PyTorch API call and defaults that cost
about half of the throughput the chip can deliver.

The server speaks the System One API: `POST /v1/systemone` with a `state` and a set of `questions`
(`choice`, `noul` yes/no, `score`), and answers a probability per option. Nothing is generated.

## Key findings

Measured on one GB10 on 2026-10-14 with a person-role classification workload (rows of ~620 tokens, ~1.9
questions per request). Full tables and raw data: [`results/2026-10-14-gb10.md`](results/2026-10-14-gb10.md).

1. **The GPU is the limit, not the server.** Kernels were busy 97-100% of wall time at ~3,500 tokens/s as shipped.
   One client process already nearly saturates it; a second one hides client-side gaps and more only queue.
2. **About a third of a forward was unfused element-wise ops** (RMSNorm, residual add, SiLU gating, gated-delta
   preparation): the GB10's 273 GB/s memory makes them expensive. The runtime has fused Triton kernels but
   skips them for Decision 3.0 (the family does not consent) and also whenever the compiled `causal-conv1d`
   package is missing. Switching them on: **+35% tokens/s, 0 of 824 predictions changed**.
3. **FP8 (e4m3) linear layers add another ~1.4x** (GEMMs run ~1.9x faster: ~185 vs ~95 TFLOPS) and cut weights
   from 15.6 to 9.2 GiB. Cost: **1.5-1.7% of predictions change**, all near-ties; gold role accuracy moved
   by +5 and -3 people of 380 (noise). Opt-in.
4. **Total: 1.9x end-to-end (2.2x excluding startup)** with fused + FP8; 1.35x with the defaults.
5. **A latent crash fixed.** The runtime's packed-forward path calls `varlen_attn(scale=, enable_gqa=)`, which
   the NGC image's torch 2.10 alpha lacks: with several clients ~4.7% of requests returned HTTP 500 (and a
   client in "map" mode may keep the original value without telling you). The launcher drops that kernel when the
   signature does not match.
6. **`--max-num-seqs 1` is the best batching setting here:** 3% faster than 128 rows per forward, less queueing,
   and an answer never depends on the requests around it.
7. **You cannot share the state across questions** (a big saving for other Decision models): d3-flash has
   `attention_mode: noncausal_full_attention`, so the state's hidden states depend on the question.
8. **The model card's ~99 requests/s is hardware and request size**, not a setting: it is an AMD MI325X (about an
   order of magnitude more BF16 compute) on short one-question text.

## Results

End-to-end wall time for 297 episodes (4 client processes, ~15 s of startup included; zero failed requests):

| Stack | few-shot 0 | few-shot 10 | Predictions changed (of 824) | Gold role accuracy (380 people) |
| --- | --- | --- | --- | --- |
| Runtime as shipped | 77 s | 116 s | - | 79.2% / 80.5% |
| + fused kernels (**default here**) | 58 s | 86 s | 0 | 79.2% / 80.5% |
| + FP8 linear layers (opt-in) | 42 s | 62 s | 13 (near-ties) | 80.5% / 79.7% |

Where a forward spends its time (idle GPU, 594-token rows):

| Stack | tokens/s | GEMM | element-wise / copy | gated-delta (FLA) |
| --- | --- | --- | --- | --- |
| shipped | ~3,500 | 54-62% | 29-36% | 5-7% |
| + causal-conv1d (exact) | ~3,700 | 57-66% | 27-34% | 5-7% |
| + fused kernels | ~4,850 | 75-81% | 11-14% | 6-9% |
| + FP8 linear layers | ~6,100-6,700 | 59-68% | 15-19% | 10-13% |

Serving configuration (fused + FP8, few-shot 10):

| `--max-num-seqs` | client processes | requests/s | tokens/s | mean queue wait |
| --- | --- | --- | --- | --- |
| 128 | 4 | 3.43 | 6,335 | 835 ms |
| 1 | 4 | 3.53 | 6,509 | 722 ms |
| 1 | 2 | 3.50 | 6,477 | 222 ms |

Tested with: GB10 (sm_121), `nvcr.io/nvidia/pytorch:26.01-py3` (torch 2.10.0a0, CUDA 13.1, Triton 3.6.0),
`vllm-srun` 0.2.0 at semantic-router `a61d0e9`, `fla-core` 0.5.2, `causal-conv1d` 1.7.0, `vllm-sr/d3-flash`
revision `521525e`.

## Quick start

Requires Docker with the NVIDIA runtime. The first run builds the image (~6 minutes, mostly compiling
`causal-conv1d`) and downloads the model (~17 GB) into `~/.cache/vllm-srun`.

```bash
bash serve.sh                         # vllm-sr/d3-flash on :8000, fused kernels, one row per forward
VLLM_SRUN_FP8=1 bash serve.sh         # + FP8 linear layers (validate on your data first)
bash serve.sh vllm-sr/d3-flash 8001   # another port

curl -s localhost:8000/health         # 503 while loading, {"status": "ready", ...} afterwards
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "model": "d3-flash",
  "state": "Welcome to the show, I am Sam. Today I am speaking with Dr. Ada Lovelace.",
  "questions": {
    "role": {
      "type": "choice",
      "instructions": "What is the role of \"Ada Lovelace\" in this podcast episode?",
      "criteria": {"host": "Runs the show", "guest": "Invited person", "mentioned": "Only talked about"}
    }
  }
}'
```

Use **2 client processes** (more only queue on one GPU), and keep requests small: cost is about
`questions x (state + question)` tokens because the state cannot be shared.

## Switches

All are environment variables read by `serve.sh` (header of the script lists them with context).

| Variable | Default | Effect |
| --- | --- | --- |
| `VLLM_SRUN_UNIFIED_MEMORY` | `1` | On hosts whose GPU shares the CPU's memory pool (DGX Spark), count reclaimable page cache as free GPU memory; see [Troubleshooting](#troubleshooting) |
| `VLLM_SRUN_FUSED_KERNELS` | `1` | Run the accelerator's fused Triton kernels (+35%; BF16 rounding differences only) |
| `VLLM_SRUN_FP8` | `0` | Hold the large linear layers as FP8 e4m3 (+~1.4x, ~7 GiB less memory, near-tie flips) |
| `VLLM_SRUN_MAX_NUM_SEQS` | `1` | Rows per forward; `1` runs every row alone |
| `VLLM_SRUN_MAX_NUM_BATCHED_TOKENS` | `16384` | Upper bound on tokens per forward; setting it skips the startup calibration |
| `VLLM_SRUN_BATCH_WINDOW_MS` | runtime (2) | How long the scheduler waits to collect requests |
| `VLLM_SRUN_MEDIA_WORKERS` | `0` | Image/video decoder processes (text-only by default) |
| `VLLM_SRUN_MEMORY_BUDGET_GIB` | off | Refuse to load a model estimated above this |
| `VLLM_SRUN_AUTOTUNE_DIR` | `autotune` | Subdirectory of the cache where Triton choices and compiled kernels persist |
| `VLLM_SRUN_CACHE_DIR` | `~/.cache/vllm-srun` | Model and kernel cache (mounted at `/models`) |
| `VLLM_SRUN_EXTRA_ARGS` | - | Extra `vllm-srun serve` arguments |

## How it works

```
Dockerfile              NGC PyTorch + vllm-srun@a61d0e9 + fla-core + compiled causal-conv1d + launcher
serve.sh                builds the image (tagged by content), runs it with the flags above
launcher/d3_launcher.py entrypoint: wraps the runtime at start-up, no source patching
launcher/fp8_linear.py  Fp8Linear (per-channel weight scales, per-token activation scales, torch._scaled_mm,
                        single-pass Triton quantiser) and convert_linears()
launcher/unified_memory.py  MemAvailable-based free-memory probe and page-cache eviction for shared CPU/GPU pools
benchmarks/             server_window.py, profile_forward.py, fp8_gemm_check.py
tests/                  CPU unit tests of the launcher, the FP8 layer and the window arithmetic
results/                the benchmark record and raw /metrics windows
```

What the launcher does at start-up:

- **Always:** if `torch.nn.attention.varlen.varlen_attn` has no `scale` / `enable_gqa` parameters, it removes the
  runtime's variable-length attention kernel so merged batches are padded instead of crashing.
- **`VLLM_SRUN_UNIFIED_MEMORY=1` (default):** wraps `GPUAccelerator.devices` so a GPU that shares the host's memory pool
  reports `MemAvailable` as its free memory, and wraps `NativeEngine._place` so the checkpoint files are dropped from the
  page cache before the device copy. Discrete GPUs are left alone.
- **`VLLM_SRUN_FUSED_KERNELS=1`:** wraps `Decision3Family.describe` so the model's `DtypePolicy` consents to the
  accelerator's approximate kernels. This needs the compiled `causal-conv1d` (the runtime registers the fused
  `gdn_prep` only with it; otherwise the receipt reads `fused_skipped: kernels missing: ['gdn_prep']`).
- **`VLLM_SRUN_FP8=1`:** wraps `NativeEngine._place` to replace the backbone's eligible `nn.Linear` layers
  (BF16, both dimensions multiples of 16 and >= 1024) with `Fp8Linear`. The weights are `nn.Parameter`s because the
  runtime checks the loaded parameter count against the package manifest.

The launcher patches internals of the pinned commit. Re-check `Decision3Family.describe`,
`GPUAccelerator.kernels`, `GPUAccelerator.devices` and `NativeEngine._place` when bumping `SEMANTIC_ROUTER_REF` in the Dockerfile.

## Troubleshooting

**`PlacementError: no device can serve d3-flash: cuda:0: d3-flash needs about 19.2 GiB, 0.6 GiB free`, retried forever,
while `htop` shows free memory.** The DGX Spark's CPU and GPU share one pool, and `torch.cuda.mem_get_info` reports
`MemFree`, which excludes the page cache. After enough file reads the cache holds almost the whole pool
(`grep -E 'MemFree|MemAvailable|^Cached' /proc/meminfo`: `MemFree` under 1 GiB, `MemAvailable` ~115 GiB). The kernel
hands the cache back as soon as the GPU allocates, so the model fits; only the runtime's pre-check refuses. The launcher
fixes this by default (`VLLM_SRUN_UNIFIED_MEMORY=1`). If you see the error with a build that predates it, or with
`VLLM_SRUN_UNIFIED_MEMORY=0`, `sync; echo 1 | sudo tee /proc/sys/vm/drop_caches` is a temporary workaround. Why it is
done this way: [`docs/adr/0001-unified-memory-free-probe.md`](docs/adr/0001-unified-memory-free-probe.md).

## Reproducing the numbers

```bash
# server throughput over a window (any client, any time): reads /metrics only
python benchmarks/server_window.py --window 20 --repeat 6

# how a forward spends its time; needs an idle GPU and the model in the cache
docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" -e HOME=/models -e HF_HUB_OFFLINE=1 \
    -v "$HOME/.cache/vllm-srun":/models -v "$PWD/benchmarks":/bench:ro -v "$PWD/launcher":/launcher:ro \
    --entrypoint python vllm-srun-d3:<tag> /bench/profile_forward.py --rows 1 2 4 [--approximate-kernels] [--fp8]

# BF16 vs FP8 GEMM speed on your GPU: decide whether FP8 is worth it before enabling it
docker run --rm --gpus all --entrypoint python -v "$PWD/benchmarks:/bench:ro" \
    vllm-srun-d3:<tag> /bench/fp8_gemm_check.py
```

`--approximate-kernels` and `--fp8` print the receipt (`fused_layers`, kernels in use) so you can confirm the
patches took effect. The end-to-end A/B in the results was done by replaying the same requests through a server
with each configuration and comparing per-person probabilities and gold role accuracy.

## Porting to another machine

- **GPU architecture:** FP8 needs a GPU with e4m3 `torch._scaled_mm` support (Ada/Hopper/Blackwell); run
  `fp8_gemm_check.py` first. `causal-conv1d` is built for `TORCH_CUDA_ARCH_LIST=12.0` (the sm_120 binary runs on
  sm_121); change it in the Dockerfile for other architectures.
- **Other base images:** the Dockerfile pins torch to whatever the base image ships so pip cannot replace it with
  a CPU wheel. If your torch has `varlen_attn(scale=, enable_gqa=)` the launcher leaves packed forwards on.
- **Other Decision 3.0 models** (`d3-mini`, `d3-nano`, ...): `serve.sh vllm-sr/<name>`; all use the same Qwen3.5
  backbone so the patches apply, but only d3-flash was measured.
- **Throughput expectations:** tokens/s scale with the chip's BF16/FP8 throughput and memory bandwidth; re-measure
  with `server_window.py` rather than extrapolating from this record.

## Tests

```bash
pip install -r requirements-dev.txt
pytest                      # 51 tests, CPU only; the one Triton test is skipped without CUDA
ruff check .
```

## Caveats

- FP8 and fused kernels are numerical approximations. Fused kernels changed no prediction on the gold subset
  and FP8 changed 1.5-1.7%; both were checked on 297 German episodes only. Validate on your data.
- Image and video input work through the runtime but were not benchmarked; video needs
  `opencv-python-headless`, which the image does not install.
- Written against a runtime commit that is a day old at the time of writing; its CLI flags and internals can move.

