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

#ifndef CPP_CUDA_ALGORITHMS
#define CPP_CUDA_ALGORITHMS

#include <array>

#include "cuda_helper.hpp"

namespace raysim {

class CUDAAlgorithms {
 public:
  /**
   * Construct a new CUDAAlgorithms object
   */
  CUDAAlgorithms();

  /**
   * Normalize a buffer in place.
   *
   * @param buffer [in] buffer data
   * @param size [in] buffer data size
   * @param buffer_min_max [in] min max buffer
   * @param stream [in] CUDA stream
   */
  void normalize(CudaMemory* buffer, uint2 size, CudaMemory* buffer_min_max, cudaStream_t stream);

  /**
   * Row convolution filter.
   *
   * @param source [in] source buffer data
   * @param size [in] buffer size
   * @param dst [in] destination buffer data
   * @param kernel [in] kernel buffer
   * @param stream [in] CUDA stream
   */
  void convolve_rows(CudaMemory* source, uint3 size, CudaMemory* dst, CudaMemory* kernel,
                     cudaStream_t stream);

  /**
   * Column convolution filter.
   *
   * @param source [in] source buffer data
   * @param size [in] buffer size
   * @param dst [in] destination buffer data
   * @param kernel [in] kernel buffer
   * @param stream [in] CUDA stream
   */
  void convolve_columns(CudaMemory* source, uint3 size, CudaMemory* dst, CudaMemory* kernel,
                        cudaStream_t stream);

  /**
   * Plane convolution filter.
   *
   * @param source [in] source buffer data
   * @param size [in] buffer size
   * @param dst [in] destination buffer data
   * @param kernel [in] kernel buffer
   * @param stream [in] CUDA stream
   */
  void convolve_planes(CudaMemory* source, uint3 size, CudaMemory* dst, CudaMemory* kernel,
                       cudaStream_t stream);

  /**
   * Compute the arithmetic mean along planes.
   *
   * @param source [in] source buffer data
   * @param size [in] buffer size
   * @param dst [in] destination buffer data
   * @param stream [in] CUDA stream
   */
  void mean_planes(CudaMemory* source, uint3 size, CudaMemory* dst, cudaStream_t stream);

  /**
   * Log compression in place.
   *
   * buffer[i] = log10(min(buffer[i], minimum) / max(buffer)) * mutliplicator
   *
   * @param buffer [in] input data
   * @param size [in]
   * @param multiplicator [in] multiplicator
   * @param minimum [in]
   * @param stream [in] CUDA stream
   */
  void log_compression(CudaMemory* buffer, uint2 size, float mutliplicator, float minimum,
                       cudaStream_t stream);

  /**
   * Multiply each row with values from multiplicator array.
   *
   * @param buffer [in]
   * @param size [in]
   * @param multiplicator [in]
   * @param stream [in] CUDA stream
   */
  void mul_row(CudaMemory* buffer, uint2 size, CudaMemory* multiplicator, cudaStream_t stream);

  /**
   * Apply hilbert transform to each row.
   *
   * @param buffer  [in]
   * @param size  [in]
   * @param stream [in] CUDA stream
   */
  void hilbert_row(CudaMemory* buffer, uint2 size, cudaStream_t stream);

  /**
   * Apply pepper vertical filter (median filter with clamping).
   *
   * Applies a vertical median filter and clamps the center pixel to be within
   * a range of [median-d_min, median+d_max].
   *
   * @param source [in] source buffer data
   * @param size [in] buffer size
   * @param dst [out] destination buffer data
   * @param filter_size [in] size of the vertical filter kernel (must be odd and <= 11)
   * @param d_min [in] minimum distance from median (lower bound)
   * @param d_max [in] maximum distance from median (upper bound)
   * @param stream [in] CUDA stream
   */
  void pepper_vertical_filter(CudaMemory* source, uint2 size, CudaMemory* dst, uint32_t filter_size,
                              float d_min, float d_max, cudaStream_t stream);

  /**
   * Convert curvilinear scan data to Cartesian coordinates for display
   *
   * @param scan_lines 2D array where each row is a scan line (shape: n_angles x n_depths)
   * @param size Size of scan line array
   * @param sector_angle Field of view in degrees
   * @param radius Radius of curvature of the transducer in mm
   * @param far Far depth for samples along scan lines
   * @param output_size Width and height of output image in pixels
   * @param stream [in] CUDA stream
   */
  std::unique_ptr<CudaMemory> scan_convert_curvilinear(CudaMemory* scan_lines, uint2 size,
                                                       float sector_angle, float radius, float far,
                                                       uint2 output_size, cudaStream_t stream);

  /**
   * Convert linear array scan data to display format
   * For linear arrays, this is mostly a pass-through with optional aspect ratio correction
   *
   * @param scan_lines 2D array where each row is a scan line (shape: n_elements x n_depths)
   * @param size Size of scan line array
   * @param width Width of the linear array in mm
   * @param far Far depth for samples along scan lines
   * @param output_size Width and height of output image in pixels
   * @param stream [in] CUDA stream
   */
  std::unique_ptr<CudaMemory> scan_convert_linear(CudaMemory* scan_lines, uint2 size, float width,
                                                  float far, uint2 output_size,
                                                  cudaStream_t stream);

  /**
   * Convert phased array sector scan data to Cartesian coordinates for display
   *
   * @param scan_lines 2D array where each row is a scan line (shape: n_angles x n_depths)
   * @param size Size of scan line array
   * @param sector_angle Field of view in degrees
   * @param far Far depth for samples along scan lines
   * @param output_size Width and height of output image in pixels
   * @param stream [in] CUDA stream
   */
  std::unique_ptr<CudaMemory> scan_convert_phased(CudaMemory* scan_lines, uint2 size,
                                                  float sector_angle, float far, uint2 output_size,
                                                  cudaStream_t stream);

 private:
  const CudaLauncher normalize_launcher_;
  const CudaLauncher convolve_rows_launcher_;
  const CudaLauncher convolve_columns_launcher_;
  const CudaLauncher convolve_planes_launcher_;
  const CudaLauncher mean_planes_launcher_;
  const CudaLauncher log_compression_launcher_;
  const CudaLauncher mul_rows_launcher_;
  const CudaLauncher pepper_vertical_launcher_;
  const CudaLauncher scan_convert_curvilinear_launcher_;
  const CudaLauncher scan_convert_linear_launcher_;
  const CudaLauncher scan_convert_phased_launcher_;

  static const size_t NUM_SUB_STREAMS =
      2;  //< Some algorithms run parallel operations in sub-streams

  UniqueCudaEvent sub_event_;
  std::array<UniqueCudaStream, NUM_SUB_STREAMS> sub_streams_;

  CudaMemory log_compression_sorted_;
  CudaMemory temp_log_compression_;
  std::shared_ptr<CudaArray> scan_convert_curvilinear_array_;
  std::unique_ptr<CudaTexture> scan_convert_curvilinear_texture_;
  std::shared_ptr<CudaArray> scan_convert_linear_array_;
  std::unique_ptr<CudaTexture> scan_convert_linear_texture_;
  std::shared_ptr<CudaArray> scan_convert_phased_array_;
  std::unique_ptr<CudaTexture> scan_convert_phased_texture_;
};

}  // namespace raysim

#endif /* CPP_CUDA_ALGORITHMS */
