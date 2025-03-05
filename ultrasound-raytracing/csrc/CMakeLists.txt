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


find_package(CUDAToolkit REQUIRED)

include(FetchContent)

###############################
# Setup third party dependencies
###############################

# spdlog
set(SPDLOG_BUILD_PIC ON CACHE INTERNAL "")
FetchContent_Declare(
    spdlog
    DOWNLOAD_EXTRACT_TIMESTAMP True
    URL https://github.com/gabime/spdlog/archive/refs/tags/v1.15.0.tar.gz
    )

# stb
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG 5c20573
    GIT_PROGRESS TRUE
    )

# assimp
set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "")
set(ASSIMP_BUILD_TESTS OFF CACHE INTERNAL "")
set(ASSIMP_INSTALL OFF CACHE INTERNAL "")
set(ASSIMP_BUILD_ZLIB ON CACHE INTERNAL "")
FetchContent_Declare(
    assimp
    DOWNLOAD_EXTRACT_TIMESTAMP True
    URL https://github.com/assimp/assimp/archive/refs/tags/v5.4.3.tar.gz
    )

# optix-toolkit
FetchContent_Declare(
    optix-toolkit
    DOWNLOAD_EXTRACT_TIMESTAMP True
    URL https://github.com/NVIDIA/optix-toolkit/archive/refs/tags/v1.0.0.tar.gz
    )

set(mathdx_VERSION 25.01)
FetchContent_Declare(
    mathdx
    DOWNLOAD_EXTRACT_TIMESTAMP True
    URL https://developer.download.nvidia.com/compute/cuFFTDx/redist/cuFFTDx/nvidia-mathdx-${mathdx_VERSION}.0.tar.gz
    )

# enable CMP0077 to allow overwriting option() statements in FetchContent sub-projects
cmake_policy(SET CMP0077 NEW)

FetchContent_MakeAvailable(spdlog stb assimp mathdx)

# we don't want to include the OptixToolkit CMake file
FetchContent_GetProperties(optix-toolkit)
if(NOT optix-toolkit_POPULATED)
  FetchContent_Populate(optix-toolkit)
endif()

find_package(mathdx REQUIRED CONFIG
    COMPONENTS cufftdx
    PATHS ${mathdx_SOURCE_DIR}/nvidia/mathdx/${mathdx_VERSION}/lib/cmake/mathdx)

if(BUILD_PYTHON_BINDINGS)
    # pybind11
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG v2.13.6
        GIT_PROGRESS TRUE
    )

    FetchContent_MakeAvailable(pybind11)
endif()

# CUDA compiler options
# - `--use_fast_math` make use of fast math library, implies --ftz=true --prec-div=false --prec-sqrt=false --fmad=true.
# - `--generate-line-info` generate line info to be able to profile kernels
# - `--expt-relaxed-constexpr` allow calling constexpr host function from device code and vice versa,
#   used to call std::numeric_limits functions from device
set(RAY_SIM_CUDA_FLAGS "--use_fast_math;--generate-line-info;--expt-relaxed-constexpr")

# Add subdirectories for each component
add_subdirectory(cuda)
add_subdirectory(core)
add_subdirectory(python)
