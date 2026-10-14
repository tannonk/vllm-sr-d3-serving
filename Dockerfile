# vllm-sr Decision 3.0 runtime (`vllm-srun`, `vllm-sr/d3-*`) for aarch64 + CUDA hosts such as the DGX Spark.
#
# The Decision 3.0 family, cross-request batching by default and the vLLM-style engine arguments
# (--max-num-seqs, --max-num-batched-tokens, --batch-window-ms) only exist from the commit below on
# (2026-10-14); that runtime removed the `--profile` / `--max-batch-tokens` flags of earlier ones.
#
# The upstream `vllm-sr-cuda` image is amd64-only, so the runtime is installed on top of NVIDIA's
# PyTorch image, which ships an aarch64 build of torch for the host's CUDA version.
# `vllm-srun` is not published to PyPI; it is installed from a pinned semantic-router commit.
#
# `serve.sh` builds this image and tags it by the content of this file.
FROM nvcr.io/nvidia/pytorch:26.01-py3

ARG SEMANTIC_ROUTER_REF=a61d0e9ab109abc6cf1ab3c2201f5dbced2fe8fb
# FLA's gated-delta kernels, which the runtime uses when importable (pure Python on top of Triton).
# 24 of d3-flash's 32 layers are gated-delta layers; without FLA they run a PyTorch reference path.
# `fla-core` is the kernel package: `flash-linear-attention` only adds Transformers model code that
# the runtime never imports. 0.5.2 is the version the upstream throughput records were made with.
ARG FLA_VERSION=0.5.2

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf

# Constrain torch to the NGC build so pip cannot replace it with a CPU/amd64-style wheel
RUN python -c "import torch; print(f'torch=={torch.__version__}')" > /tmp/torch-constraint.txt \
    && pip install -c /tmp/torch-constraint.txt \
        "vllm-srun @ git+https://github.com/vllm-project/semantic-router.git@${SEMANTIC_ROUTER_REF}#subdirectory=src/model-runtime" \
        "fla-core==${FLA_VERSION}" \
    && python -c "import torch, vllm_srun; assert torch.version.cuda, 'torch lost its CUDA build'" \
    && python -c "import fla.ops.gated_delta_rule"

# The compiled causal-conv1d CUDA extension (~6 min to build). Its presence is what makes the runtime
# register the fused `gdn_prep` kernel; without it the runtime skips the fused decoder layers entirely
# (receipt: `fused_skipped: kernels missing: ['gdn_prep']`). It is also an exact kernel, ~6% faster than
# the PyTorch convolution on its own. TORCH_CUDA_ARCH_LIST=12.0: the sm_120 binary runs on the GB10's sm_121.
ARG CAUSAL_CONV1D_VERSION=1.7.0
RUN MAX_JOBS=8 CAUSAL_CONV1D_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="12.0" \
        pip install -c /tmp/torch-constraint.txt --no-build-isolation "causal-conv1d==${CAUSAL_CONV1D_VERSION}" \
    && python -c "import causal_conv1d"

# This torch build allocates host memory with mimalloc, which keeps freed pages for reuse. Without
# an immediate purge the host copy of the weights lingers after they are copied to the GPU, which on
# a shared CPU/GPU pool is memory the server never gets back.
ENV MIMALLOC_PURGE_DELAY=0

# Runs `vllm-srun` unchanged unless a switch is set (see the launcher's docstring)
COPY launcher/d3_launcher.py launcher/fp8_linear.py /opt/d3/

EXPOSE 8100
ENTRYPOINT ["python", "/opt/d3/d3_launcher.py"]
CMD ["--help"]
