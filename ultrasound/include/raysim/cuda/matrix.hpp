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

#ifndef CPP_MATRIX
#define CPP_MATRIX

#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

namespace raysim {

/// Column major 3x3 matrix
class float33 {
 public:
  __host__ __device__ inline void set_value(float val) {
    for (int32_t i = 0; i < 3; i++) {
      for (int32_t j = 0; j < 3; j++) { element(i, j) = val; }
    }
  }

  __host__ __device__ inline float33& operator*=(const float33& rhs) {
    float33 mt(*this);
    set_value(0.0f);

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int c = 0; c < 3; c++) { element(i, j) += mt(i, c) * rhs(c, j); }
      }
    }
    return *this;
  }

  // dst = M * v
  __host__ __device__ inline friend float3 operator*(const float33& mat, const float3& v) {
    float3 result;
    result.x = v.x * mat.element(0, 0) + v.y * mat.element(0, 1) + v.z * mat.element(0, 2);
    result.y = v.x * mat.element(1, 0) + v.y * mat.element(1, 1) + v.z * mat.element(1, 2);
    result.z = v.x * mat.element(2, 0) + v.y * mat.element(2, 1) + v.z * mat.element(2, 2);

    return result;
  }

  // dst = v * M
  __host__ __device__ inline friend float3 operator*(const float3& v, const float33& mat) {
    float3 result;
    result.x = v.x * mat.element(0, 0) + v.y * mat.element(1, 0) + v.z * mat.element(2, 0);
    result.y = v.x * mat.element(0, 1) + v.y * mat.element(1, 1) + v.z * mat.element(2, 1);
    result.z = v.x * mat.element(0, 2) + v.y * mat.element(1, 2) + v.z * mat.element(2, 2);

    return result;
  }

  __host__ __device__ inline float& operator()(int row, int col) { return element(row, col); }

  __host__ __device__ inline const float& operator()(int row, int col) const {
    return element(row, col);
  }

  __host__ __device__ inline float& element(int row, int col) {
    assert(((row < 3) && col < 3));
    return this->data_[row + (col * 3)];
  }

  __host__ __device__ inline const float& element(int row, int col) const {
    assert(((row < 3) && col < 3));
    return data_[row + (col * 3)];
  }

  __host__ __device__ inline float33& operator+=(const float33& mat) {
    for (int i = 0; i < 3; ++i) {
      element(0, i) += mat.element(0, i);
      element(1, i) += mat.element(1, i);
      element(2, i) += mat.element(2, i);
      element(3, i) += mat.element(3, i);
    }
    return *this;
  }

  union {
    struct {
      float xx, yx, zx;
      float xy, yy, zy;
      float xz, yz, zz;
    };  // component access
    float data_[3 * 3];  // array access
  };
};

__host__ __device__ inline float33 make_identity() {
  float33 matrix;
  matrix.element(0, 0) = 1.0f;
  matrix.element(0, 1) = 0.0f;
  matrix.element(0, 2) = 0.0f;

  matrix.element(1, 0) = 0.0f;
  matrix.element(1, 1) = 1.0f;
  matrix.element(1, 2) = 0.0f;

  matrix.element(2, 0) = 0.0f;
  matrix.element(2, 1) = 0.0f;
  matrix.element(2, 2) = 1.0f;

  return matrix;
}

__host__ __device__ inline float33 transpose(const float33& matrix) {
  float33 mtrans;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) { mtrans(i, j) = matrix.element(j, i); }
  }
  return mtrans;
}

__host__ __device__ inline float33 make_rotationX(float angle) {
  float33 matrix = make_identity();

  float cosa = cosf(angle);
  float sina = sinf(angle);

  matrix.element(0, 0) = 1.0f;
  matrix.element(1, 1) = cosa;
  matrix.element(1, 2) = -sina;
  matrix.element(2, 1) = sina;
  matrix.element(2, 2) = cosa;

  return matrix;
}

__host__ __device__ inline float33 make_rotationY(float angle) {
  float33 matrix = make_identity();

  float cosa = cosf(angle);
  float sina = sinf(angle);

  matrix.element(0, 0) = cosa;
  matrix.element(0, 2) = sina;
  matrix.element(1, 1) = 1.0;
  matrix.element(2, 0) = -sina;
  matrix.element(2, 2) = cosa;

  return matrix;
}

__host__ __device__ inline float33 make_rotationZ(float angle) {
  float33 matrix = make_identity();

  float cosa = cosf(angle);
  float sina = sinf(angle);

  matrix.element(0, 0) = cosa;
  matrix.element(0, 1) = -sina;
  matrix.element(1, 0) = sina;
  matrix.element(1, 1) = cosa;
  matrix.element(2, 2) = 1.0;

  return matrix;
}

__host__ __device__ inline float33 make_rotation(float rx, float ry, float rz) {
  float33 matrix = make_identity();

  if (rz) { matrix = make_rotationX(rz); }
  if (ry) { matrix *= make_rotationY(ry); }
  if (rx) { matrix *= make_rotationZ(rx); }

  return matrix;
}

}  // namespace raysim

#endif /* CPP_MATRIX */
