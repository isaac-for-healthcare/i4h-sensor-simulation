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

#ifndef CPP_TRANSFORM_UTILS
#define CPP_TRANSFORM_UTILS

#include <vector_functions.h>  // For normalize() and float3 operators
#include "raysim/core/pose.hpp"
#include "raysim/cuda/matrix.hpp"

namespace raysim {

/**
 * Transform a point from local to world coordinates using a pose.
 *
 * @param pose The pose defining the transformation
 * @param point The point in local coordinates
 * @return The point in world coordinates
 */
inline float3 transform_point(const Pose& pose, const float3& point) {
  // Use built-in matrix-vector multiplication
  float3 rotated = pose.rotation_matrix_ * point;

  // Explicitly add components to avoid operator+ issues with const float3
  return make_float3(
      rotated.x + pose.position_.x, rotated.y + pose.position_.y, rotated.z + pose.position_.z);
}

/**
 * Transform a direction vector from local to world coordinates using a pose.
 * Note: Unlike points, directions only need to be rotated, not translated.
 *
 * @param pose The pose defining the transformation
 * @param direction The direction in local coordinates
 * @return The direction in world coordinates
 */
inline float3 transform_direction(const Pose& pose, const float3& direction) {
  // Use built-in matrix-vector multiplication (no translation)
  return pose.rotation_matrix_ * direction;
}

}  // namespace raysim

#endif /* CPP_TRANSFORM_UTILS */
