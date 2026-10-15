#!/bin/bash

# Serves a vllm-sr Decision 3.0 model (default vllm-sr/d3-flash) at POST /v1/systemone (and
# /v1/decisions) in a standalone aarch64-compatible container (e.g. DGX Spark). See README.md.
#
# Usage: ./serve.sh [MODEL] [PORT]
#   ./serve.sh                          # vllm-sr/d3-flash on :8000 (~27 GiB of GPU memory)
#   ./serve.sh vllm-sr/d3-flash 8001    # another port
#
# The default flags are chosen for throughput. Concurrent requests share forwards (cross-request
# batching), but on a GB10 one client with ~600-token rows already keeps the GPU ~100% busy: 2-4
# client workers (--num-proc / --num-shards) hide client-side gaps, more only add queueing.
# Measure with: python benchmarks/server_window.py
#
# Environment variables (all optional):
#   VLLM_SRUN_MAX_NUM_BATCHED_TOKENS  tokens one forward may hold (default 16384, the runtime's value for
#                              64-160 GB GPUs). Setting it skips the runtime's startup calibration, which
#                              caps forwards at 0.3 s and can under-batch a GB10. Decision 3.0 sizes its
#                              own forwards too (about 0.3 s of work, never below 2048 tokens), so this is
#                              an upper bound. Lower = less activation memory.
#   VLLM_SRUN_MAX_NUM_SEQS     most rows (one per question) one forward holds (default 1: every row runs alone, so
#                              an answer never depends on the requests around it). On a GB10 with ~600-token rows a
#                              single row already fills the GPU: 1 was 3% faster than 128 and queued less.
#                              Unset the default with a larger value (e.g. 128) to let rows of requests share forwards.
#   VLLM_SRUN_BATCH_WINDOW_MS  how long the scheduler waits after a request for more (unset = 2 ms).
#   VLLM_SRUN_MEDIA_WORKERS    processes that decode request images and videos (default 0: text-only
#                              workloads, where idle workers only cost host memory on a shared CPU/GPU pool). Video input
#                              also needs opencv-python-headless, which the image does not install.
#   VLLM_SRUN_FUSED_KERNELS    1 (default) = run the accelerator's fused Triton kernels for RMSNorm, residual add,
#                              SiLU gating and gated-delta preparation. ~+35% tokens/s on the GB10; answers differ
#                              from the exact kernels by BF16 rounding only (0 of 824 predictions changed on the
#                              gold subset). 0 = exact kernels. See launcher/d3_launcher.py.
#   VLLM_SRUN_FP8              1 = hold the large linear layers as FP8 (e4m3, per-channel weight scales, per-token
#                              activation scales): ~1.9x faster GEMMs and ~7 GiB less memory, with quantization
#                              error. Validate on your data first. See launcher/fp8_linear.py.
#   VLLM_SRUN_UNIFIED_MEMORY   1 (default) = on a host whose GPU shares the CPU's memory pool (DGX Spark), count
#                              reclaimable page cache as free GPU memory and drop the checkpoint files from the cache
#                              before the device copy. Without it the runtime reads MemFree, which a full page cache
#                              drives under 1 GiB, and refuses the model ("needs about 19.2 GiB, 0.6 GiB free").
#                              0 = stock behaviour. Discrete GPUs are unaffected. See launcher/unified_memory.py.
#   VLLM_SRUN_MEMORY_BUDGET_GIB  refuse to load a model whose estimated size exceeds this (unset = off).
#   VLLM_SRUN_AUTOTUNE_DIR     subdirectory of the cache dir where Triton autotuning and compiled kernels are
#                              kept between runs (default autotune): a restart skips the tuning pause and
#                              runs the same kernels as the run before.
#   VLLM_SRUN_CACHE_DIR        model and kernel cache (default ~/.cache/vllm-srun).
#   VLLM_SRUN_EXTRA_ARGS       extra `vllm-srun serve` arguments, split on spaces.

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${1:-vllm-sr/d3-flash}"
PORT="${2:-8000}"
# Tagged by content, so an edited Dockerfile or launcher is rebuilt instead of served from a stale image
IMAGE="vllm-srun-d3:$(cat "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR/launcher/d3_launcher.py" "$SCRIPT_DIR/launcher/fp8_linear.py" "$SCRIPT_DIR/launcher/unified_memory.py" | sha256sum | cut -c1-12)"
CACHE_DIR="${VLLM_SRUN_CACHE_DIR:-$HOME/.cache/vllm-srun}"
MAX_NUM_BATCHED_TOKENS="${VLLM_SRUN_MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${VLLM_SRUN_MAX_NUM_SEQS:-1}"
BATCH_WINDOW_MS="${VLLM_SRUN_BATCH_WINDOW_MS:-}"
MEDIA_WORKERS="${VLLM_SRUN_MEDIA_WORKERS:-0}"
MEMORY_BUDGET_GIB="${VLLM_SRUN_MEMORY_BUDGET_GIB:-}"
AUTOTUNE_DIR="${VLLM_SRUN_AUTOTUNE_DIR:-autotune}"
FUSED_KERNELS="${VLLM_SRUN_FUSED_KERNELS:-1}"
FP8="${VLLM_SRUN_FP8:-0}"
UNIFIED_MEMORY="${VLLM_SRUN_UNIFIED_MEMORY:-1}"

SERVE_ARGS=(
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --media-workers "$MEDIA_WORKERS"
    --autotune-cache "/models/$AUTOTUNE_DIR"
)
if [ -n "$MAX_NUM_SEQS" ]; then
    SERVE_ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
fi
if [ -n "$BATCH_WINDOW_MS" ]; then
    SERVE_ARGS+=(--batch-window-ms "$BATCH_WINDOW_MS")
fi
if [ -n "$MEMORY_BUDGET_GIB" ]; then
    SERVE_ARGS+=(--memory-budget "$MEMORY_BUDGET_GIB")
fi
if [ -n "${VLLM_SRUN_EXTRA_ARGS:-}" ]; then
    # Intentional word splitting: the variable holds several arguments
    # shellcheck disable=SC2206
    SERVE_ARGS+=($VLLM_SRUN_EXTRA_ARGS)
fi

mkdir -p "$CACHE_DIR"

if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    echo "Building Docker image..."
    docker build -t "$IMAGE" "$SCRIPT_DIR" || exit 1
fi

# Run the Docker container with GPU support and port mapping (the runtime listens on 8100)
docker run --rm -it --gpus all --ipc=host -p "$PORT":8100 \
    --user "$(id -u):$(id -g)" -e HOME=/models \
    -e HF_TOKEN \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8 \
    -e VLLM_SRUN_FUSED_KERNELS="$FUSED_KERNELS" \
    -e VLLM_SRUN_FP8="$FP8" \
    -e VLLM_SRUN_UNIFIED_MEMORY="$UNIFIED_MEMORY" \
    -v "$CACHE_DIR":/models \
    "$IMAGE" \
    serve "$MODEL" \
    --device cuda:0 \
    --host 0.0.0.0 \
    "${SERVE_ARGS[@]}"
