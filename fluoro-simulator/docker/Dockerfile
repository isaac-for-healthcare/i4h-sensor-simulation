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

FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

LABEL maintainer="NVIDIA Isaac for Healthcare"
LABEL description="GPU-accelerated fluoroscopy (X-ray) simulator"

ENV PIP_NO_CACHE_DIR=off
ENV PIP_BREAK_SYSTEM_PACKAGES=1

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

WORKDIR /workspace/i4h-sim/fluoro-simulator

# Install dependencies directly, layer is cached independently of source code changes.
# (except for changes in pyproject.toml and README.md)
COPY fluoro-simulator/pyproject.toml fluoro-simulator/README.md ./
RUN mkdir -p fluorosim && touch fluorosim/__init__.py \
    && pip install kaggle -e ".[all]" \
    && rm fluorosim/__init__.py

ENV HOLOHUB_ALWAYS_BUILD=false
