# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# =============================================================================
# Fluoroscopy Simulator Docker Image
#
# Build:
#   docker build -t fluorosim .
#
# Run demo (with GPU):
#   docker run --rm --gpus all -v /tmp/fluorosim_output:/output fluorosim
#
# Run interactive:
#   docker run --rm --gpus all -it fluorosim bash
#
# =============================================================================

FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

LABEL maintainer="NVIDIA Isaac for Healthcare"
LABEL description="GPU-accelerated fluoroscopy (X-ray) simulator"

# Disable pip cache
ENV PIP_NO_CACHE_DIR=off
ENV PYTHONUNBUFFERED=1

# Install Python, build dependencies, and X11 libs (slangpy requires libX11)
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        python3-pip \
        build-essential \
        libx11-6 \
        libxcursor1 \
        libxrandr2 \
        libxinerama1 \
        libxi6 \
        libxext6 \
        libgl1 \
        unzip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.12 /usr/bin/python

# Create workspace
WORKDIR /app

# Copy package files
COPY pyproject.toml README.md ./
COPY fluorosim/ ./fluorosim/
COPY examples/ ./examples/

# Install kaggle
RUN pip install --break-system-packages kaggle

# Install fluorosim with all dependencies
RUN pip install --break-system-packages -e ".[all]"
