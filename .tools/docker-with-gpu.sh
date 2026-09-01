#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# HoloHub CLI always passes `--runtime nvidia`. Docker CE + NVIDIA Container
# Toolkit on some aarch64 hosts (DGX Spark) only registers runc, while
# `--gpus all` works. Rewrite the runtime flag when `nvidia` is not a Docker
# runtime.

set -euo pipefail

DOCKER_BIN="${HOLOHUB_DOCKER_REAL:-docker}"

runtimes="$("${DOCKER_BIN}" info --format '{{range $k, $v := .Runtimes}}{{$k}} {{end}}' 2>/dev/null || true)"
if [[ " ${runtimes} " == *" nvidia "* ]]; then
    exec "${DOCKER_BIN}" "$@"
fi

args=()
rewrite=0
skip_next=0
for arg in "$@"; do
    if [[ "${skip_next}" -eq 1 ]]; then
        skip_next=0
        continue
    fi
    if [[ "${arg}" == "--runtime=nvidia" ]]; then
        rewrite=1
        continue
    fi
    if [[ "${arg}" == "--runtime" ]]; then
        skip_next=1
        rewrite=1
        continue
    fi
    args+=("${arg}")
done

if [[ "${rewrite}" -eq 0 ]]; then
    exec "${DOCKER_BIN}" "${args[@]}"
fi

final=()
inserted=0
for arg in "${args[@]}"; do
    final+=("${arg}")
    if [[ "${inserted}" -eq 0 && "${arg}" == "run" ]]; then
        final+=("--gpus" "all")
        inserted=1
    fi
done
exec "${DOCKER_BIN}" "${final[@]}"
