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

FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

# Disable pip cache to avoid creating a cache directory with root permissions in the user's home directory.
ENV PIP_NO_CACHE_DIR=off

ARG USERNAME=raysim
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
# (See https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user)
# The 24.04 image added the non-root user 'ubuntu' which needs to be removed first
RUN touch /var/mail/ubuntu && chown ubuntu /var/mail/ubuntu && userdel -r ubuntu \
    && groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && rm -rf /var/lib/apt/lists/* \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        cmake="3.28.3-*" \
        gdb="15.0.50.20240403-*" \
        git="1:2.43.0-*" \
        python3.12-dev="3.12.3-*" \
    && rm -rf /var/lib/apt/lists/*

# Set the default user
USER $USERNAME
