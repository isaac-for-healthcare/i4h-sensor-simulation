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

/**
 * Represents a 3D pose with position and orientation.
 * Handles coordinate transformations between local and world space.
 *
 * Note: position_, rotation_, and rotation_matrix_ are public to allow direct
 * access by GPU/OptiX code for performance reasons. In CPU code, prefer using
 * the transformation methods local_to_world_point() and local_to_world_direction().
 */
class Pose {
 public:
  /**
   * Default constructor initializes to identity pose at origin
   */
  Pose()
      : position_(make_float3(0.f, 0.f, 0.f)),
        rotation_(make_float3(0.f, 0.f, 0.f)),
        rotation_matrix_(make_identity()) {}

  /**
   * Construct a pose from position and rotation
   *
   * @param position Position vector [x, y, z] in world coordinates (mm)
   * @param rotation Rotation vector [rx, ry, rz] in radians
   */
  Pose(float3 position, float3 rotation);

  // Core data members - public for direct GPU access
  float3 position_;          // [x, y, z] in world coordinates (mm)
  float3 rotation_;          // [rx, ry, rz] in radians
  float33 rotation_matrix_;  // Cached rotation matrix for efficient transforms

  /**
   * Transform a point from local to world coordinates.
   * Applies both rotation and translation.
   *
   * For CPU code, prefer this over direct matrix operations.
   *
   * @param local_point Point in local coordinates
   * @return Point in world coordinates
   */
  float3 local_to_world_point(const float3& local_point) const {
    float3 rotated = rotation_matrix_ * local_point;
    return make_float3(rotated.x + position_.x, rotated.y + position_.y, rotated.z + position_.z);
  }

  /**
   * Transform a direction vector from local to world coordinates.
   * Only applies rotation, not translation.
   *
   * For CPU code, prefer this over direct matrix operations.
   *
   * @param local_dir Direction vector in local coordinates
   * @return Direction vector in world coordinates
   */
  float3 local_to_world_direction(const float3& local_dir) const {
    return rotation_matrix_ * local_dir;
  }
};

}  // namespace raysim

#endif /* CPP_POSE */
