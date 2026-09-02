#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e  # Exit on error

# Default example to run and port
EXAMPLE="${1:-examples/server.py}"
PORT="${2:-8000}"

echo "========================================"
echo "Building Ultrasound Raytracing Container"
echo "========================================"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# This image runs the example straight from the build baked into it, so both the CUDA
# toolkit and the target architecture have to match the host GPU: CUDA 12.6 cannot
# compile Blackwell (sm_120/sm_121), and CUDA 13 needs a 580+ driver. Mirrors
# get_default_cuda_version() in tools/utilities/cli/util.py, which ./i4h uses.
CUDA_MAJOR=13
CUDA_ARCHITECTURES=80
if command -v nvidia-smi >/dev/null 2>&1; then
  compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')"
  driver_major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)"
  [ -n "$compute_cap" ] && CUDA_ARCHITECTURES="$compute_cap"
  if [ -n "$driver_major" ] && [ "$driver_major" -lt 580 ]; then CUDA_MAJOR=12; fi
  # CUDA 13 dropped Maxwell/Pascal/Volta, so anything below sm_75 needs CUDA 12.
  if [ -n "$compute_cap" ] && [ "$compute_cap" -lt 75 ]; then CUDA_MAJOR=12; fi
fi

# Build the Docker image
echo ""
echo "Building Docker image (this may take ~10 minutes on first run)..."
echo "Using CUDA ${CUDA_MAJOR} for compute capability ${CUDA_ARCHITECTURES}"
docker build \
  --build-arg USER_UID=$(id -u) \
  --build-arg USER_GID=$(id -g) \
  --build-arg CUDA_MAJOR="$CUDA_MAJOR" \
  --build-arg CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
  -f .devcontainer/Dockerfile \
  -t ultrasound_simulator:latest \
  .

echo ""
echo "========================================"
echo "Build completed successfully!"
echo "========================================"
echo ""
echo "Running example: $EXAMPLE"
if [[ "$EXAMPLE" == *"server.py"* ]]; then
  echo "Server will be available at: http://localhost:$PORT"
fi
echo ""

# Run the container with GPU support
docker run --rm -it \
  --gpus all \
  -p $PORT:8000 \
  --name ultrasound_simulator-server \
  ultrasound_simulator:latest \
  python $EXAMPLE
