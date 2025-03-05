/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CPP_POSE
#define CPP_POSE

#include "raysim/cuda/cuda_helper.hpp"
#include "raysim/cuda/matrix.hpp"

namespace raysim {

// 3D pose with position and orientation
class Pose {
 public:
  // Constructor declaration
  Pose(float3 position, float3 rotation);

  float3 position_;  // [x, y, z]
  float3 rotation_;  // [rx, ry, rz] in radians

  float33 rotation_matrix_;
};

}  // namespace raysim

#endif /* CPP_POSE */
