# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(radial_geometry_test radial_geometry_test.cpp)

set_target_properties(radial_geometry_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(radial_geometry_test PRIVATE ray_sim_core)

add_test(NAME radial_geometry_test COMMAND radial_geometry_test)
