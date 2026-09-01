# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Global: must be declared before the first FROM so `FROM cuda${CUDA_MAJOR}` works.
# ./i4h passes CUDA_MAJOR=12 or 13 from the host driver.
ARG CUDA_MAJOR=12

# ============================================================================
# Stage 1: Download mesh assets (requires Python 3.11 for i4h-asset-catalog)
# ============================================================================
FROM ubuntu:22.04 AS mesh-downloader

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=off
ENV ACCEPT_EULA=Y

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /downloads

RUN python3.11 -m venv /downloads/.venv && \
    /downloads/.venv/bin/pip install git+https://github.com/isaac-for-healthcare/i4h-asset-catalog.git

RUN echo "Yes" | /downloads/.venv/bin/i4h-asset-retrieve \
    --download-dir assets --sub-path Props/ABDPhantom/Organs --version 0.2.0

# ============================================================================
# Stage 2: Runtime image
# ============================================================================
# CUDA 12.6 cannot compile Blackwell (sm_120/sm_121), so pick a matching CUDA
# devel image. BASE_IMAGE from the HoloHub CLI is the Holoscan SDK tag and is
# intentionally unused: this simulator needs nvcc, not Holoscan.
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS cuda12
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04 AS cuda13
FROM cuda${CUDA_MAJOR}

LABEL maintainer="NVIDIA Isaac for Healthcare"
LABEL description="GPU-accelerated raytracing ultrasound simulator"

ARG COMPUTE_CAPACITY=8.0
ARG CUDA_MAJOR=12

ENV PIP_NO_CACHE_DIR=off
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        python3-pip \
        build-essential \
        cmake \
        git \
        ninja-build \
        libx11-6 \
        libxcursor1 \
        libxrandr2 \
        libxinerama1 \
        libxi6 \
        libxext6 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH}

WORKDIR /workspace/i4h-sim/ultrasound-simulator

# Install ultrasound_simulator; layer is cached independently of source code changes.
COPY ultrasound-simulator/pyproject.toml ultrasound-simulator/README.md ./
COPY ultrasound-simulator/cmake ./cmake
COPY ultrasound-simulator/CMakeLists.txt ./
COPY ultrasound-simulator/csrc ./csrc
COPY ultrasound-simulator/include ./include
COPY ultrasound-simulator/ultrasound_simulator ./ultrasound_simulator
# COMPUTE_CAPACITY "12.1" -> CMAKE_CUDA_ARCHITECTURES 121
RUN CUDA_ARCH="$(printf '%s' "${COMPUTE_CAPACITY}" | tr -d '.')" \
    && CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} -DBUILD_EXAMPLES=OFF" \
       pip install ".[all]"

# Mesh assets baked into the image at /opt/ultrasound-mesh.
# i4h modes point ULTRASOUND_MESH_DIR to this path by default.
COPY --from=mesh-downloader /downloads/assets/*/Props/ABDPhantom/Organs /opt/ultrasound-mesh
ENV HOLOHUB_ALWAYS_BUILD=false
