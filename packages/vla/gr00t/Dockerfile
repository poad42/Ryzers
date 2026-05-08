# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

# Install deps. Drop python3-pybind11 and hipcc (not packaged on Ubuntu 24.04
# under those names; pybind11-dev is sufficient and hipcc ships with rocm-dev).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    rocm-dev \
    cmake \
    pybind11-dev \
    ffmpeg \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libavutil-dev \
    libswresample-dev \
    libswscale-dev \
    pkg-config \
 && rm -rf /var/lib/apt/lists/*

RUN pip install zmq diffusers decord

# Build pytorch3d since there's no wheel for python3.12
ENV FORCE_CUDA=1
# pytorch3d (no python3.12 wheel; build from source). Its HIP kernels reference
# bare cub::BlockReduce and #include <cub/cub.cuh>; on ROCm those symbols live
# in hipcub. Rewrite both before the build.
RUN git clone https://github.com/facebookresearch/pytorch3d.git /ryzers/pytorch3d && \
    cd /ryzers/pytorch3d && \
    grep -rl --include="*.hip" --include="*.cuh" --include="*.h" --include="*.cu" \
        'cub::' . | xargs -r sed -i 's/cub::/hipcub::/g' && \
    grep -rl --include="*.hip" --include="*.cuh" --include="*.h" --include="*.cu" \
        '#include <cub/cub.cuh>' . | xargs -r sed -i 's|#include <cub/cub.cuh>|#include <hipcub/hipcub.hpp>|g' && \
    pip install --no-build-isolation -e .

# Install flash attention
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" 
ENV BUILD_TYPE="rocm"
RUN git clone --recursive https://github.com/ROCm/flash-attention && \
    cd flash-attention && \
    git checkout main_perf && \
    python setup.py install

# Build torchcodec from source
ENV PYTORCH_ROCM_ARCH="gfx1151"
# torchcodec build is fragile on ROCm 7.2.2 (FFmpeg API gaps); soft-fail.
# Tests don't require it.
RUN git clone https://github.com/pytorch/torchcodec.git && \
    cd /ryzers/torchcodec && \
    sed -i 's/-Werror//g' src/torchcodec/_core/CMakeLists.txt && \
    (pip install -e . --no-build-isolation || \
     echo "WARN: torchcodec failed to build on ROCm 7.2.2; skipping")

RUN git clone https://github.com/NVIDIA/Isaac-GR00T && \
    cd Isaac-GR00T && \
    git checkout 4af2b62 && \
    pip install -e .

COPY test.py /ryzers/test.py
COPY test.sh /ryzers/test.sh
RUN chmod +x /ryzers/test.sh

CMD ["/bin/bash", "-c", "/ryzers/test.sh"]
