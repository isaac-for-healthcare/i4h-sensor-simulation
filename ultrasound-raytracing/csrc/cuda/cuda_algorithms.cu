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

#include "raysim/cuda/cuda_algorithms.hpp"

#include <sutil/vec_math.h>
#include <cub/cub.cuh>
#include <cufftdx/cufftdx.hpp>

namespace raysim {

static __global__ void normalize_kernel(float* __restrict__ buffer, uint2 size,
                                        const float* __restrict__ min_max) {
  const uint2 index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if ((index.x >= size.x) || (index.y >= size.y)) { return; }

  buffer += index.y * size.x + index.x;

  *buffer = (*buffer - min_max[0]) / (min_max[1] - min_max[0]);
}

/// @todo the convolution kernels are using a naiive implementation, improve see
/// https://github.com/zchee/cuda-sample/tree/master/3_Imaging/convolutionSeparable

static __global__ void convolve_rows_kernel(const float* __restrict__ source, uint3 size,
                                            float* __restrict__ dst,
                                            const float* __restrict__ kernel,
                                            uint32_t kernel_radius) {
  const uint3 index = make_uint3(blockIdx.x * blockDim.x + threadIdx.x,
                                 blockIdx.y * blockDim.y + threadIdx.y,
                                 blockIdx.z * blockDim.z + threadIdx.z);

  if ((index.x >= size.x) || (index.y >= size.y) || (index.z >= size.z)) { return; }

  const int k_min = -min(index.x, kernel_radius);
  const int k_max = min(size.x - 1 - index.x, kernel_radius);
  const int offset = ((index.z * size.y) + index.y) * size.x + index.x;
  source += offset;

  float sum = 0.f;
  for (int k = k_min; k <= k_max; ++k) { sum += source[k] * kernel[k + kernel_radius]; }

  dst[offset] = sum;
}

static __global__ void convolve_columns_kernel(const float* __restrict__ source, uint3 size,
                                               float* __restrict__ dst,
                                               const float* __restrict__ kernel,
                                               uint32_t kernel_radius) {
  const uint3 index = make_uint3(blockIdx.x * blockDim.x + threadIdx.x,
                                 blockIdx.y * blockDim.y + threadIdx.y,
                                 blockIdx.z * blockDim.z + threadIdx.z);

  if ((index.x >= size.x) || (index.y >= size.y) || (index.z >= size.z)) { return; }

  const int k_min = -min(index.y, kernel_radius);
  const int k_max = min(size.y - 1 - index.y, kernel_radius);
  const int offset = ((index.z * size.y) + index.y) * size.x + index.x;
  source += offset + k_min * size.x;

  float sum = 0.f;
  for (int k = k_min; k <= k_max; ++k) {
    sum += *source * kernel[k + kernel_radius];
    source += size.x;
  }

  dst[offset] = sum;
}

static __global__ void convolve_planes_kernel(const float* __restrict__ source, uint3 size,
                                              float* __restrict__ dst,
                                              const float* __restrict__ kernel,
                                              uint32_t kernel_radius) {
  const uint3 index = make_uint3(blockIdx.x * blockDim.x + threadIdx.x,
                                 blockIdx.y * blockDim.y + threadIdx.y,
                                 blockIdx.z * blockDim.z + threadIdx.z);

  if ((index.x >= size.x) || (index.y >= size.y) || (index.z >= size.z)) { return; }

  const int k_min = -min(index.z, kernel_radius);
  const int k_max = min(size.z - 1 - index.z, kernel_radius);
  const int offset = ((index.z * size.y) + index.y) * size.x + index.x;
  source += offset + k_min * size.x * size.y;

  float sum = 0.f;
  for (int k = k_min; k <= k_max; ++k) {
    sum += *source * kernel[k + kernel_radius];
    source += size.x * size.y;
  }

  dst[offset] = sum;
}

static __global__ void mean_planes_kernel(const float* __restrict__ source, uint3 size,
                                          float* __restrict__ dst) {
  const uint2 index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if ((index.x >= size.x) || (index.y >= size.y)) { return; }

  const int offset = index.y * size.x + index.x;
  source += offset;

  const int plane_size = size.x * size.y;

  float sum = 0.f;
  for (int plane = 0; plane < size.z; ++plane) {
    sum += *source;
    source += plane_size;
  }

  dst[offset] = sum / size.z;
}

static __global__ void log_compression_kernel(float* __restrict__ buffer, uint2 size,
                                              const float* __restrict__ quantile,
                                              float mutliplicator, float minimum) {
  const uint2 index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if ((index.x >= size.x) || (index.y >= size.y)) { return; }

  const uint32_t offset = index.y * size.x + index.x;

  buffer[offset] = log10f(max(buffer[offset], minimum) / (*quantile)) * mutliplicator;
}

static __global__ void mul_rows_kernel(float* __restrict__ buffer, uint2 size,
                                       const float* __restrict__ multiplicator) {
  const uint2 index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if ((index.x >= size.x) || (index.y >= size.y)) { return; }

  buffer[index.y * size.x + index.x] *= multiplicator[index.x];
}

static __global__ void median_clip_kernel(const float* __restrict__ source, uint2 size,
                                          float* __restrict__ dst, uint32_t filter_size,
                                          float d_min, float d_max) {
  const uint2 index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if ((index.x >= size.x) || (index.y >= size.y)) { return; }

  const uint32_t offset = index.y * size.x + index.x;

  // Calculate median filter bounds
  const int half_size = filter_size / 2;
  const int y_min = max(0, (int)index.y - half_size);
  const int y_max = min((int)size.y - 1, (int)index.y + half_size);

  // Collect values for median calculation
  float values[11];  // Maximum filter size is 11
  int count = 0;

  for (int y = y_min; y <= y_max; ++y) { values[count++] = source[y * size.x + index.x]; }

  // Simple bubble sort for small arrays (efficient for small filter sizes)
  for (int i = 0; i < count - 1; ++i) {
    for (int j = 0; j < count - i - 1; ++j) {
      if (values[j] > values[j + 1]) {
        float temp = values[j];
        values[j] = values[j + 1];
        values[j + 1] = temp;
      }
    }
  }

  // Get median value
  float median = values[count / 2];

  // Calculate bounds
  float lower_bound = fmaxf(d_min, median);
  float upper_bound = fminf(d_max, median);

  // Clamp original value to bounds
  float original_value = source[offset];
  dst[offset] = fmaxf(lower_bound, fminf(original_value, upper_bound));
}

// cuFFTDx needs to know the SM architecture, this is only known when compiling device code. Use the
// lowest supported arch for host code.
#ifdef __CUDA_ARCH__
#define CUFFTDX_ARCH __CUDA_ARCH__
#else
#define CUFFTDX_ARCH 700
#endif

// The FFT size needs to be known at compile time
static constexpr uint32_t HILBERT_FFT_SIZE = 4096;

// Define the forward FFT type
using HilbertForwardFFT = decltype(cufftdx::Size<HILBERT_FFT_SIZE>() + cufftdx::Precision<float>() +
                                   cufftdx::Type<cufftdx::fft_type::r2c>() + cufftdx::Block() +
                                   cufftdx::ElementsPerThread<8>() + cufftdx::FFTsPerBlock<1>() +
                                   cufftdx::SM<CUFFTDX_ARCH>());

// Define the inverse FFT type
using HilbertInverseFFT =
    decltype(cufftdx::Size<HilbertForwardFFT::input_length>() + cufftdx::Precision<float>() +
             cufftdx::Type<cufftdx::fft_type::c2c>() +
             cufftdx::Direction<cufftdx::fft_direction::inverse>() + cufftdx::Block() +
             cufftdx::ElementsPerThread<HilbertForwardFFT::elements_per_thread>() +
             cufftdx::FFTsPerBlock<1>() + cufftdx::SM<CUFFTDX_ARCH>());

// Since we execute both forward and inverse in one kernel the parameters have to match
static_assert(HilbertForwardFFT::max_threads_per_block == HilbertInverseFFT::max_threads_per_block);
static_assert(HilbertForwardFFT::storage_size == HilbertInverseFFT::storage_size);
static_assert(HilbertForwardFFT::shared_memory_size == HilbertInverseFFT::shared_memory_size);

static __launch_bounds__(HilbertForwardFFT::max_threads_per_block) __global__
    void hilbert_kernel(float* __restrict__ buffer) {
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= gridDim.y) { return; }

  // Jump to current row
  buffer += row * HilbertForwardFFT::input_length;

  // If there are threads with partial work load then we need to check for bounds
  constexpr bool has_partial_load =
      ((HilbertForwardFFT::input_length % HilbertForwardFFT::elements_per_thread) != 0);

  // Local array for thread
  HilbertForwardFFT::value_type thread_data[HilbertForwardFFT::storage_size];

  // Load data from global memory to registers
  unsigned int index = threadIdx.x;
  for (unsigned int i = 0; i < HilbertForwardFFT::elements_per_thread; ++i) {
    if (!has_partial_load || (index < HilbertForwardFFT::input_length)) {
      reinterpret_cast<float*>(thread_data)[i] = buffer[index];
      index += HilbertForwardFFT::stride;
    }
  }

  // Execute forward FFT
  extern __shared__ HilbertForwardFFT::value_type shared_mem[];
  HilbertForwardFFT().execute(thread_data, shared_mem);

  // Zero out negative frequencies and double positive frequencies (keep zero frequency) and copy
  // to output
  constexpr unsigned int half_size = HilbertForwardFFT::input_length >> 1;
  index = threadIdx.x;
  for (unsigned int i = 0; i < HilbertForwardFFT::elements_per_thread; ++i) {
    if (!has_partial_load || (index < HilbertForwardFFT::input_length)) {
      thread_data[i] *= (index < half_size) ? 2.0f : (index > half_size) ? 0.f : 1.f;
    }
    index += HilbertForwardFFT::stride;
  }

  // Execute inverse FFT
  HilbertInverseFFT().execute(thread_data, shared_mem);

  // Convert to real, scale and copy to buffer
  index = threadIdx.x;
  for (unsigned int i = 0; i < HilbertInverseFFT::elements_per_thread; ++i) {
    if (!has_partial_load || (index < HilbertInverseFFT::input_length)) {
      const auto value = thread_data[i];
      buffer[index] = sqrtf(value.x * value.x + value.y * value.y) *
                      (1.f / float(HilbertInverseFFT::input_length));
    }
    index += HilbertForwardFFT::stride;
  }
}

static __global__ void scan_convert_curvilinear_kernel(cudaTextureObject_t input, uint2 input_size,
                                                       float* __restrict__ output,
                                                       uint2 output_size, float sector_angle,
                                                       float near, float far, float scale_x,
                                                       float offset_z) {
  const uint2 index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if ((index.x >= output_size.x) || (index.y >= output_size.y)) { return; }

  float2 coord = make_float2((float(index.x) / float(output_size.x)) * 2.f - 1.f,
                             float(index.y) / float(output_size.y));
  // Scale to fit
  coord.x *= scale_x;
  coord.y = (coord.y * (1.f - offset_z)) + offset_z;

  // Mask out near and far
  const float dist = sqrtf(coord.x * coord.x + coord.y * coord.y);
  if ((dist < near) || (dist > far)) {
    output[index.y * output_size.x + index.x] = std::numeric_limits<float>::lowest();
    return;
  }

  // Mask out outside of opening angle
  const float angle = atanf(coord.x / coord.y);
  if (fabsf(angle) > sector_angle / 2.f) {
    output[index.y * output_size.x + index.x] = std::numeric_limits<float>::lowest();
    return;
  }

  const float source_x = (dist - near) / (far - near);
  const float source_y = angle / sector_angle + 0.5f;

  output[index.y * output_size.x + index.x] = tex2D<float>(input, source_x, source_y);
}

static __global__ void scan_convert_linear_kernel(cudaTextureObject_t input, uint2 input_size,
                                                  float* __restrict__ output, uint2 output_size,
                                                  float width, float far) {
  const uint2 index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if ((index.x >= output_size.x) || (index.y >= output_size.y)) { return; }

  // Calculate physical aspect ratio of the linear probe's field of view
  const float physical_aspect = width / far;

  // Calculate the normalized coordinates in the output image space [0,1] x [0,1]
  const float normalized_x = float(index.x) / float(output_size.x - 1);
  const float normalized_y = float(index.y) / float(output_size.y - 1);

  // For x: Convert to centered coordinates in [-0.5, 0.5] range
  // For y: Keep y = 0 at the top of the image (probe surface)
  const float centered_x = normalized_x - 0.5f;

  // Calculate the region where the linear probe's field of view is displayed
  // For square output images, we need to adjust based on aspect ratio
  float scale_factor;
  if (physical_aspect < 1.0f) {
    // Width is smaller than depth - add black bars on sides
    scale_factor = physical_aspect;
  } else {
    // Width is larger than depth - use full width (rare for ultrasound)
    scale_factor = 1.0f;
  }

  // Scale the x coordinate to account for the aspect ratio
  const float scaled_x = centered_x / scale_factor;

  // Check if we're outside the valid image region (add black bars)
  if (fabsf(scaled_x) > 0.5f) {
    output[index.y * output_size.x + index.x] = std::numeric_limits<float>::lowest();
    return;
  }

  // Map from normalized coordinates to physical coordinates
  const float px = scaled_x * width;    // Map to [-width/2, width/2]
  const float pz = normalized_y * far;  // Map to [0, far] with 0 at the top

  // Check if point is within the rectangular field of view
  if (fabsf(px) > width / 2.0f || pz < 0.0f || pz > far) {
    output[index.y * output_size.x + index.x] = std::numeric_limits<float>::lowest();
    return;
  }

  // Map to texture coordinates [0,1] x [0,1]
  // Linear arrays have scanlines running along elements (lateral dimension)
  // Each element's scanline represents depth data from that position
  const float source_y = (px + width / 2.0f) / width;  // Map lateral position to scanline index
  const float source_x = pz / far;                     // Map depth to position along scanline

  output[index.y * output_size.x + index.x] = tex2D<float>(input, source_x, source_y);
}

static __global__ void scan_convert_phased_kernel(cudaTextureObject_t input, uint2 input_size,
                                                  float* __restrict__ output, uint2 output_size,
                                                  float sector_angle, float far) {
  const uint2 index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if ((index.x >= output_size.x) || (index.y >= output_size.y)) { return; }

  // Black out pixels by default
  output[index.y * output_size.x + index.x] = std::numeric_limits<float>::lowest();

  // Convert sector angle to radians
  const float sector_angle_rad = (sector_angle / 180.0f) * M_PI;
  const float half_angle_rad = sector_angle_rad / 2.0f;

  // PROPER PHASED ARRAY SECTOR GEOMETRY:
  // 1. Define origin at the top center of the image
  const float origin_x = output_size.x / 2.0f;
  const float origin_y = 0.0f;

  // 2. Get current pixel coordinates
  const float px = static_cast<float>(index.x);
  const float py = static_cast<float>(index.y);

  // Skip if we're at the top (y=0) and outside the width of the probe
  // This ensures a clean flat interface
  if (py == 0.0f && fabsf(px - origin_x) > (output_size.x / 2.0f * 0.1f)) { return; }

  // Calculate the ray angle for this pixel
  // For points not at the origin, use atan2 to get the angle from vertical
  float theta;
  float depth;

  if (py == 0.0f) {
    // At the top interface, use a special case to ensure it's flat
    // All top-row pixels use the angle that corresponds to their x position
    theta = (px - origin_x) / origin_x * half_angle_rad;
    depth = 0.0f;
  } else {
    // For all other pixels, calculate properly from the origin
    theta = atan2f(px - origin_x, py - origin_y);

    // Calculate depth along the ray (distance from origin)
    depth = sqrtf((px - origin_x) * (px - origin_x) + (py - origin_y) * (py - origin_y));

    // Scale depth to be in physical units [0, far]
    depth = depth * far / output_size.y;
  }

  // Skip if outside the sector angle
  if (fabsf(theta) > half_angle_rad) { return; }

  // Skip if beyond max depth
  if (depth > far) { return; }

  // Map to texture coordinates
  // For angle: convert from [-half_angle, half_angle] to [0, 1]
  const float source_y = (theta + half_angle_rad) / sector_angle_rad;

  // For depth: normalize to [0, 1]
  const float source_x = depth / far;

  // Sample the scan line data
  output[index.y * output_size.x + index.x] = tex2D<float>(input, source_x, source_y);
}

CUDAAlgorithms::CUDAAlgorithms()
    : normalize_launcher_((void*)&normalize_kernel),
      convolve_rows_launcher_((void*)&convolve_rows_kernel),
      convolve_columns_launcher_((void*)&convolve_columns_kernel),
      convolve_planes_launcher_((void*)&convolve_planes_kernel),
      mean_planes_launcher_((void*)&mean_planes_kernel),
      log_compression_launcher_((void*)&log_compression_kernel),
      mul_rows_launcher_((void*)&mul_rows_kernel),
      median_clip_launcher_((void*)&median_clip_kernel),
      scan_convert_curvilinear_launcher_((void*)&scan_convert_curvilinear_kernel),
      scan_convert_linear_launcher_((void*)&scan_convert_linear_kernel),
      scan_convert_phased_launcher_((void*)&scan_convert_phased_kernel) {
  CUDA_CHECK(cudaFuncSetAttribute(hilbert_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  HilbertForwardFFT::shared_memory_size));
  sub_event_.reset([] {
    CUevent event;
    CUDA_CHECK(cudaEventCreate(&event, CU_EVENT_BLOCKING_SYNC));
    return event;
  }());

  for (auto&& sub_stream : sub_streams_) {
    sub_stream.reset([] {
      cudaStream_t stream;
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      return stream;
    }());
  }
}

void CUDAAlgorithms::normalize(CudaMemory* buffer, uint2 size, CudaMemory* buffer_min_max,
                               cudaStream_t stream) {
  normalize_launcher_.launch(size,
                             stream,
                             reinterpret_cast<float*>(buffer->get_ptr(stream)),
                             reinterpret_cast<const float*>(buffer_min_max->get_ptr(stream)));
}

void CUDAAlgorithms::convolve_rows(CudaMemory* source, uint3 size, CudaMemory* dst,
                                   CudaMemory* kernel, cudaStream_t stream) {
  if (!((kernel->get_size() / sizeof(float)) & 1)) {
    throw std::runtime_error("Expected uneven kernel size");
  }

  convolve_rows_launcher_.launch(size,
                                 stream,
                                 reinterpret_cast<const float*>(source->get_ptr(stream)),
                                 size,
                                 reinterpret_cast<float*>(dst->get_ptr(stream)),
                                 reinterpret_cast<const float*>(kernel->get_ptr(stream)),
                                 kernel->get_size() / sizeof(float) / 2);
}

void CUDAAlgorithms::convolve_columns(CudaMemory* source, uint3 size, CudaMemory* dst,
                                      CudaMemory* kernel, cudaStream_t stream) {
  if (!((kernel->get_size() / sizeof(float)) & 1)) {
    throw std::runtime_error("Expected uneven kernel size");
  }

  convolve_columns_launcher_.launch(size,
                                    stream,
                                    reinterpret_cast<const float*>(source->get_ptr(stream)),
                                    size,
                                    reinterpret_cast<float*>(dst->get_ptr(stream)),
                                    reinterpret_cast<const float*>(kernel->get_ptr(stream)),
                                    kernel->get_size() / sizeof(float) / 2);
}

void CUDAAlgorithms::convolve_planes(CudaMemory* source, uint3 size, CudaMemory* dst,
                                     CudaMemory* kernel, cudaStream_t stream) {
  if (!((kernel->get_size() / sizeof(float)) & 1)) {
    throw std::runtime_error("Expected uneven kernel size");
  }

  convolve_planes_launcher_.launch(size,
                                   stream,
                                   reinterpret_cast<const float*>(source->get_ptr(stream)),
                                   size,
                                   reinterpret_cast<float*>(dst->get_ptr(stream)),
                                   reinterpret_cast<const float*>(kernel->get_ptr(stream)),
                                   kernel->get_size() / sizeof(float) / 2);
}

void CUDAAlgorithms::mean_planes(CudaMemory* source, uint3 size, CudaMemory* dst,
                                 cudaStream_t stream) {
  mean_planes_launcher_.launch(make_uint2(size.x, size.y),
                               stream,
                               reinterpret_cast<const float*>(source->get_ptr(stream)),
                               size,
                               reinterpret_cast<float*>(dst->get_ptr(stream)));
}

void CUDAAlgorithms::log_compression(CudaMemory* buffer, uint2 size, float mutliplicator,
                                     float minimum, cudaStream_t stream) {
  const uint32_t num_items = size.x * size.y;
  float* const d_data = reinterpret_cast<float*>(buffer->get_ptr(stream));

  // get 0.9999 quantile of buffer
  log_compression_sorted_.resize(buffer->get_size(), stream);
  float* const d_sorted = reinterpret_cast<float*>(log_compression_sorted_.get_ptr(stream));

  {
    // Determine temporary device storage requirements
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr,
                                              temp_storage_bytes,
                                              d_data,
                                              d_sorted,
                                              num_items,
                                              0,
                                              sizeof(float) * 8 /*end_bit*/,
                                              stream));

    temp_log_compression_.resize(temp_storage_bytes, stream);

    // Run max-reduction
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(temp_log_compression_.get_ptr(stream),
                                              temp_storage_bytes,
                                              d_data,
                                              d_sorted,
                                              num_items,
                                              0,
                                              sizeof(float) * 8 /*end_bit*/,
                                              stream));
  }

  const float* const d_quantile = d_sorted + uint32_t(0.99999f * (num_items - 1) + 0.5f);

  log_compression_launcher_.launch(size, stream, d_data, size, d_quantile, mutliplicator, minimum);
}

void CUDAAlgorithms::mul_row(CudaMemory* buffer, uint2 size, CudaMemory* multiplicator,
                             cudaStream_t stream) {
  if (size.x != multiplicator->get_size() / sizeof(float)) {
    throw std::runtime_error("Unexpected multiplicator buffer size");
  }

  mul_rows_launcher_.launch(size,
                            stream,
                            reinterpret_cast<float*>(buffer->get_ptr(stream)),
                            size,
                            reinterpret_cast<const float*>(multiplicator->get_ptr(stream)));
}

void CUDAAlgorithms::hilbert_row(CudaMemory* buffer, uint2 size, cudaStream_t stream) {
  if (size.x != HILBERT_FFT_SIZE) {
    std::stringstream buf;
    buf << "Hilbert: row length of " << size.x << " does not match supported row length of "
        << HILBERT_FFT_SIZE << ". The row length is set at compile time.";
    throw std::runtime_error(buf.str().c_str());
  }

  const dim3 grid{1, size.y, 1};

  hilbert_kernel<<<grid,
                   HilbertForwardFFT::block_dim,
                   HilbertForwardFFT::shared_memory_size,
                   stream>>>(reinterpret_cast<float*>(buffer->get_ptr(stream)));
}

void CUDAAlgorithms::median_clip_filter(CudaMemory* source, uint2 size, CudaMemory* dst,
                                        uint32_t filter_size, float d_min, float d_max,
                                        cudaStream_t stream) {
  if (filter_size > 11 || filter_size % 2 == 0) {
    throw std::runtime_error("Filter size must be odd and <= 11");
  }

  median_clip_launcher_.launch(size,
                               stream,
                               reinterpret_cast<const float*>(source->get_ptr(stream)),
                               size,
                               reinterpret_cast<float*>(dst->get_ptr(stream)),
                               filter_size,
                               d_min,
                               d_max);
}

std::unique_ptr<CudaMemory> CUDAAlgorithms::scan_convert_curvilinear(CudaMemory* scan_lines,
                                                                     uint2 input_size,
                                                                     float sector_angle, float near,
                                                                     float far, uint2 output_size,
                                                                     cudaStream_t stream) {
  // Create the array and the texture
  if (scan_convert_curvilinear_array_ &&
      ((scan_convert_curvilinear_array_->get_size().width != input_size.x) ||
       (scan_convert_curvilinear_array_->get_size().height != input_size.y))) {
    scan_convert_curvilinear_array_.reset();
  }
  if (!scan_convert_curvilinear_array_) {
    scan_convert_curvilinear_array_ = std::make_shared<CudaArray>(
        cudaExtent({input_size.x, input_size.y, 0}), cudaChannelFormatKindFloat, sizeof(float));
    scan_convert_curvilinear_texture_ = std::make_unique<CudaTexture>(
        scan_convert_curvilinear_array_, cudaAddressModeClamp, cudaFilterModeLinear);
  }

  // Upload scan lines
  scan_convert_curvilinear_array_->upload(scan_lines, stream);

  // Create the output memory
  auto grid_z = std::make_unique<CudaMemory>(output_size.x * output_size.y * sizeof(float), stream);

  // Calculate the image bounds
  const float sector_angle_rad = (sector_angle / 360.f) * 2.f * M_PI;
  const float max_x = std::sin(sector_angle_rad * 0.5f);                 // width / 2
  const float min_z = std::cos(sector_angle_rad * 0.5f) * (near / far);  // depth

  scan_convert_curvilinear_launcher_.launch(output_size,
                                            stream,
                                            scan_convert_curvilinear_texture_->get_texture().get(),
                                            input_size,
                                            reinterpret_cast<float*>(grid_z->get_ptr(stream)),
                                            output_size,
                                            sector_angle_rad,
                                            near / far,
                                            far / far,
                                            max_x,
                                            min_z);

  return std::move(grid_z);
}

std::unique_ptr<CudaMemory> CUDAAlgorithms::scan_convert_linear(CudaMemory* scan_lines,
                                                                uint2 input_size, float width,
                                                                float far, uint2 output_size,
                                                                cudaStream_t stream) {
  // Create the array and the texture
  if (scan_convert_linear_array_ &&
      ((scan_convert_linear_array_->get_size().width != input_size.x) ||
       (scan_convert_linear_array_->get_size().height != input_size.y))) {
    scan_convert_linear_array_.reset();
  }
  if (!scan_convert_linear_array_) {
    scan_convert_linear_array_ = std::make_shared<CudaArray>(
        cudaExtent({input_size.x, input_size.y, 0}), cudaChannelFormatKindFloat, sizeof(float));
    scan_convert_linear_texture_ = std::make_unique<CudaTexture>(
        scan_convert_linear_array_, cudaAddressModeClamp, cudaFilterModeLinear);
  }

  // Upload scan lines
  scan_convert_linear_array_->upload(scan_lines, stream);

  // Create the output memory
  auto grid_z = std::make_unique<CudaMemory>(output_size.x * output_size.y * sizeof(float), stream);

  // For linear arrays, scan conversion is mostly a direct mapping
  scan_convert_linear_launcher_.launch(output_size,
                                       stream,
                                       scan_convert_linear_texture_->get_texture().get(),
                                       input_size,
                                       reinterpret_cast<float*>(grid_z->get_ptr(stream)),
                                       output_size,
                                       width,
                                       far);

  return std::move(grid_z);
}

std::unique_ptr<CudaMemory> CUDAAlgorithms::scan_convert_phased(CudaMemory* scan_lines,
                                                                uint2 input_size,
                                                                float sector_angle, float far,
                                                                uint2 output_size,
                                                                cudaStream_t stream) {
  // Create the array and the texture
  if (scan_convert_phased_array_ &&
      ((scan_convert_phased_array_->get_size().width != input_size.x) ||
       (scan_convert_phased_array_->get_size().height != input_size.y))) {
    scan_convert_phased_array_.reset();
  }
  if (!scan_convert_phased_array_) {
    scan_convert_phased_array_ = std::make_shared<CudaArray>(
        cudaExtent({input_size.x, input_size.y, 0}), cudaChannelFormatKindFloat, sizeof(float));
    scan_convert_phased_texture_ = std::make_unique<CudaTexture>(
        scan_convert_phased_array_, cudaAddressModeClamp, cudaFilterModeLinear);
  }

  // Upload scan lines
  scan_convert_phased_array_->upload(scan_lines, stream);

  // Create the output memory
  auto grid_z = std::make_unique<CudaMemory>(output_size.x * output_size.y * sizeof(float), stream);

  // Convert from polar coordinates to Cartesian for display
  scan_convert_phased_launcher_.launch(output_size,
                                       stream,
                                       scan_convert_phased_texture_->get_texture().get(),
                                       input_size,
                                       reinterpret_cast<float*>(grid_z->get_ptr(stream)),
                                       output_size,
                                       sector_angle,
                                       far);

  return std::move(grid_z);
}

}  // namespace raysim
