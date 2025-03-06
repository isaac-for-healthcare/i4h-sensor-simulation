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

cmake_minimum_required(VERSION 3.24.0)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(raysim
    VERSION 0.1.0
    DESCRIPTION "Ray-tracing based ultrasound simulator"
    LANGUAGES CXX)
set(CMAKE_CUDA_ARCHITECTURES 89)
include(SetupCUDA) # CUDA Language enabled there after additional setup

# Build options
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_EXAMPLES "Build C++ examples" ON)

add_subdirectory(third_party ${CMAKE_CURRENT_BINARY_DIR}/thirdparty)

# Add core C++ libraries
add_subdirectory(csrc)

# Add C++ examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples/cpp)
endif()
