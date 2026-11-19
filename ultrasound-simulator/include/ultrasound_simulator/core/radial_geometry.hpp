/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CPP_RADIAL_GEOMETRY
#define CPP_RADIAL_GEOMETRY

#include <cuda_runtime.h>

#include <math.h>
#include <cstdint>

#include "ultrasound_simulator/core/probe_types.hpp"

namespace ultrasound_simulator {

constexpr float RADIAL_PI = 3.14159265358979323846f;
constexpr float RADIAL_TWO_PI = 2.0f * RADIAL_PI;

/** Rotation sign s in {-1, +1}. */
__host__ __device__ inline float radial_rotation_sign(RadialRotationDirection direction) {
  return static_cast<float>(static_cast<int>(direction));
}

/** Scanline angle: theta_i = theta_0 + s 2 pi i / N. */
__host__ __device__ inline float radial_scanline_angle_rad(
    uint32_t scanline_index, uint32_t num_scanlines, float start_angle_degrees,
    RadialRotationDirection direction = RadialRotationDirection::POSITIVE) {
  const float start_angle_rad = start_angle_degrees * (RADIAL_PI / 180.0f);
  return start_angle_rad + radial_rotation_sign(direction) * RADIAL_TWO_PI *
                               (static_cast<float>(scanline_index) / num_scanlines);
}

/** Emitter orbit: o_i = rho (sin(theta_i), 0, cos(theta_i)). */
__host__ __device__ inline float3 radial_scanline_origin(
    uint32_t scanline_index, uint32_t num_scanlines, float start_angle_degrees,
    float transducer_offset_radius,
    RadialRotationDirection rotation_direction = RadialRotationDirection::POSITIVE) {
  const float angle = radial_scanline_angle_rad(
      scanline_index, num_scanlines, start_angle_degrees, rotation_direction);
  return make_float3(
      transducer_offset_radius * sinf(angle), 0.0f, transducer_offset_radius * cosf(angle));
}

/** Unit beam direction: d_i = (cos(beta) sin(theta_i), sin(beta), cos(beta) cos(theta_i)). */
__host__ __device__ inline float3 radial_scanline_direction(
    uint32_t scanline_index, uint32_t num_scanlines, float start_angle_degrees,
    float beam_tilt_degrees = 0.0f,
    RadialRotationDirection direction = RadialRotationDirection::POSITIVE) {
  const float angle =
      radial_scanline_angle_rad(scanline_index, num_scanlines, start_angle_degrees, direction);
  const float tilt = beam_tilt_degrees * (RADIAL_PI / 180.0f);
  const float transverse_scale = cosf(tilt);
  return make_float3(transverse_scale * sinf(angle), sinf(tilt), transverse_scale * cosf(angle));
}

/**
 * Pixel pitch of the scan-converted image: the circular field of view (diameter 2 * far) is fit
 * to the shorter image dimension, so both axes share one spacing and the image is not stretched.
 * Requires width >= 2 and height >= 2.
 */
__host__ __device__ inline float radial_pixel_spacing(float far, uint32_t width, uint32_t height) {
  const uint32_t shorter_side = (width < height) ? width : height;
  return (2.0f * far) / static_cast<float>(shorter_side - 1);
}

__host__ __device__ inline float wrap_unit_interval(float value) {
  return value - floorf(value);
}

/**
 * Map the local x-z display plane to normalised depth and periodic A-line position.
 *
 * r = sqrt(x^2 + z^2), phi = atan2(x, z),
 * u = r / far, v = wrap(s (phi - theta_0) / (2 pi)).
 * Returns false outside the visible annulus.
 */
__host__ __device__ inline bool radial_scan_coordinates(
    float x, float z, float far, float dead_zone_radius, float start_angle_degrees,
    float& depth_coordinate, float& scanline_coordinate,
    RadialRotationDirection rotation_direction = RadialRotationDirection::POSITIVE) {
  const float radius = sqrtf(x * x + z * z);
  if ((radius < dead_zone_radius) || (radius > far)) { return false; }

  depth_coordinate = radius / far;
  const float angle = atan2f(x, z);
  const float start_angle_rad = start_angle_degrees * (RADIAL_PI / 180.0f);
  scanline_coordinate = wrap_unit_interval(radial_rotation_sign(rotation_direction) *
                                           (angle - start_angle_rad) / RADIAL_TWO_PI);
  return true;
}

}  // namespace ultrasound_simulator

#endif /* CPP_RADIAL_GEOMETRY */
