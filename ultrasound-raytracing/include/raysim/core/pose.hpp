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

/**
 * Convert degrees to radians
 *
 * @param degrees Angle in degrees
 * @return Angle in radians
 */
constexpr float deg2rad(float degrees) {
  return degrees * (M_PI / 180.0f);
}

/**
 * Convert radians to degrees
 *
 * @param radians Angle in radians
 * @return Angle in degrees
 */
constexpr float rad2deg(float radians) {
  return radians * (180.0f / M_PI);
}

// 3D pose with position and orientation
class Pose {
 public:
  // Default constructor
  Pose()
      : position_(make_float3(0.f, 0.f, 0.f)),
        rotation_(make_float3(0.f, 0.f, 0.f)),
        rotation_matrix_(make_identity()) {}

  // Constructor declaration
  Pose(float3 position, float3 rotation);

  float3 position_;  // [x, y, z]
  float3 rotation_;  // [rx, ry, rz] in radians

  float33 rotation_matrix_;

  /**
   * Apply rotation to a vector using this pose's rotation matrix.
   * This is used for both directions (which only need rotation)
   * and as part of point transformation (which needs both rotation and translation).
   *
   * @param vec The vector to rotate in local coordinates
   * @return The rotated vector
   */
  inline float3 apply_rotation(const float3& vec) const { return rotation_matrix_ * vec; }

  /**
   * Transform a point from local to world coordinates using this pose.
   * This applies both rotation and translation.
   *
   * @param point The point in local coordinates
   * @return The point in world coordinates
   */
  inline float3 transform_point(const float3& point) const {
    // First rotate, then translate
    float3 rotated = apply_rotation(point);

    // Add translation
    return make_float3(rotated.x + position_.x, rotated.y + position_.y, rotated.z + position_.z);
  }

  /**
   * Transform a direction vector from local to world coordinates using this pose.
   * Note: Unlike points, directions only need to be rotated, not translated.
   *
   * @param direction The direction in local coordinates
   * @return The direction in world coordinates
   */
  inline float3 transform_direction(const float3& direction) const {
    // Use built-in matrix-vector multiplication (no translation)
    return rotation_matrix_ * direction;
  }
};

}  // namespace raysim

#endif /* CPP_POSE */
