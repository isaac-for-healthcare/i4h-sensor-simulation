# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(mesh_test mesh_test.cpp)

set_target_properties(mesh_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(mesh_test PRIVATE ray_sim_core ray_sim_cuda)

add_test(NAME mesh_test COMMAND mesh_test)
