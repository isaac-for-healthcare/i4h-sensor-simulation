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

# Build the Docker image
echo ""
echo "Building Docker image (this may take ~10 minutes on first run)..."
docker build \
  --build-arg USER_UID=$(id -u) \
  --build-arg USER_GID=$(id -g) \
  -f .devcontainer/Dockerfile \
  -t raysim:latest \
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
  --name raysim-server \
  raysim:latest \
  python $EXAMPLE
