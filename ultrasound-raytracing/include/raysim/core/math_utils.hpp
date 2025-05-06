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

#ifndef CPP_MATH_UTILS
#define CPP_MATH_UTILS

#include <cmath>

namespace raysim {
namespace math {
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
 * Linear interpolation between two values
 *
 * @param a First value
 * @param b Second value
 * @param t Interpolation parameter (0.0 to 1.0)
 * @return Interpolated value
 */
template <typename T>
constexpr T lerp(const T& a, const T& b, float t) {
  return a + t * (b - a);
}

/**
 * Clamp a value between a minimum and maximum
 *
 * @param val Value to clamp
 * @param min Minimum value
 * @param max Maximum value
 * @return Clamped value
 */
template <typename T>
constexpr T clamp(const T& val, const T& min, const T& max) {
  return (val < min) ? min : ((val > max) ? max : val);
}

/**
 * Check if two floating point values are approximately equal
 *
 * @param a First value
 * @param b Second value
 * @param epsilon Maximum allowed difference
 * @return True if approximately equal
 */
inline bool approximately_equal(float a, float b, float epsilon = 1e-5f) {
  return std::fabs(a - b) <= epsilon;
}
}  // namespace math
}  // namespace raysim

#endif /* CPP_MATH_UTILS */
