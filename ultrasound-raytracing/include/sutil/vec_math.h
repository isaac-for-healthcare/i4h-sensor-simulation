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

// Lightweight replacement for the heavy sutil/vec_math.h from the NVIDIA
// OptiX SDK.  Provides only the subset used by this project (dot/length/
// normalize/arithmetic on float3).  This is a minimal subset of the original
// OptiX SDK vec_math.h file.

#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <sutil/Preprocessor.h>
#include <vector_types.h>

// Basic constructors are already available via vector_types.h:
//   make_float2, make_float3, make_float4, etc.
// We only implement convenience arithmetic helpers needed in kernels.

// -----------------------------------------------------------------------------
// float3 helpers
// -----------------------------------------------------------------------------
SUTIL_INLINE SUTIL_HOSTDEVICE float3 operator+(const float3& a, const float3& b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

SUTIL_INLINE SUTIL_HOSTDEVICE float3 operator-(const float3& a, const float3& b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

SUTIL_INLINE SUTIL_HOSTDEVICE float3 operator*(const float3& a, const float s) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}

SUTIL_INLINE SUTIL_HOSTDEVICE float3 operator*(const float s, const float3& a) {
  return a * s;
}

SUTIL_INLINE SUTIL_HOSTDEVICE float3 operator/(const float3& a, const float s) {
  const float inv = 1.0f / s;
  return a * inv;
}

SUTIL_INLINE SUTIL_HOSTDEVICE float dot(const float3& a, const float3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

SUTIL_INLINE SUTIL_HOSTDEVICE float length(const float3& v) {
  return sqrtf(dot(v, v));
}

SUTIL_INLINE SUTIL_HOSTDEVICE float3 normalize(const float3& v) {
  const float invLen = rsqrtf(dot(v, v));
  return v * invLen;
}

SUTIL_INLINE SUTIL_HOSTDEVICE float3 cross(const float3& a, const float3& b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

SUTIL_INLINE SUTIL_HOSTDEVICE float clamp(float f, float a, float b) {
  return fminf(b, fmaxf(a, f));
}

SUTIL_INLINE SUTIL_HOSTDEVICE float3 clamp(const float3& v, float a, float b) {
  return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

SUTIL_INLINE SUTIL_HOSTDEVICE float3& operator+=(float3& a, const float3& b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}
SUTIL_INLINE SUTIL_HOSTDEVICE float3& operator-=(float3& a, const float3& b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  return a;
}
SUTIL_INLINE SUTIL_HOSTDEVICE float3& operator*=(float3& a, const float s) {
  a.x *= s;
  a.y *= s;
  a.z *= s;
  return a;
}
SUTIL_INLINE SUTIL_HOSTDEVICE float3& operator/=(float3& a, const float s) {
  const float inv = 1.0f / s;
  a *= inv;
  return a;
}
SUTIL_INLINE SUTIL_HOSTDEVICE float3 operator-(const float3& v) {
  return make_float3(-v.x, -v.y, -v.z);
}
// Construct float3 from float4 by dropping w
SUTIL_INLINE SUTIL_HOSTDEVICE float3 make_float3(const float4& a) {
  return make_float3(a.x, a.y, a.z);
}
