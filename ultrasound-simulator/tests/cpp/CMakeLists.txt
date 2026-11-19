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

add_executable(mesh_test mesh_test.cpp)
add_executable(radial_geometry_test radial_geometry_test.cpp)

foreach(test_target mesh_test radial_geometry_test)
    set_target_properties(${test_target} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
endforeach()

target_link_libraries(mesh_test PRIVATE ray_sim_core ray_sim_cuda)
target_link_libraries(radial_geometry_test PRIVATE ray_sim_core)

add_test(NAME mesh_test COMMAND mesh_test)
add_test(NAME radial_geometry_test COMMAND radial_geometry_test)
