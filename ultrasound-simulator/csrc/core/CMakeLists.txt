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

################################################
# Build the core simulation static library
################################################

add_library(ray_sim_core STATIC
    hitable.cpp
    material.cpp
    pose.cpp
    raytracing_ultrasound_simulator.cpp
    world.cpp
    write_image.cpp
    )



set_target_properties(ray_sim_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    )

target_include_directories(ray_sim_core
    PRIVATE
        ${stb_SOURCE_DIR}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )

target_link_libraries(ray_sim_core
    PRIVATE
        assimp::assimp
        ray_sim_cuda
    PUBLIC
        CUDA::cudart_static
        spdlog::spdlog
    )
