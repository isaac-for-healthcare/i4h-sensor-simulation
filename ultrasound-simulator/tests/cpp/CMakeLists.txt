# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
