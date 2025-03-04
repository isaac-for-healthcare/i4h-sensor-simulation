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

#include "raysim/core/write_image.hpp"

#include <example/cufftdx/common.hpp>
#include <example/cufftdx/conv_3d/io_strided_conv_smem.hpp>

// we use the code form the cufftdx example directory which unfortunatly is not part of the cufftdx
// namespace, so we need to define the namespace manually here
namespace cufftdx {
namespace example = ::example;
}

// cuFFTDx needs to know the SM architecture, this is only known when compiling device code. Use the
// lowest supported arch for host code.
// #ifdef __CUDA_ARCH__
// #define CUFFTDX_ARCH __CUDA_ARCH__
// #else
#define CUFFTDX_ARCH 700
// #endif

namespace raysim {

static __global__ void normalize_kernel(float* __restrict__ buffer, uint2 size,
                                        const float* __restrict__ min_max) {
  const uint2 index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if ((index.x >= size.x) || (index.y >= size.y)) { return; }

  buffer += index.y * size.x + index.x;

  *buffer = (*buffer - min_max[0]) / (min_max[1] - min_max[0]);
}

#ifdef FFT_CONF
template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void fft_kernel(typename FFT::input_type* source, typename FFT::output_type* dst) {
  // Local array for thread
  typename FFT::value_type thread_data[FFT::storage_size];

  // Load data from global memory to registers
  cufftdx::example::io<FFT>::load(source, thread_data, threadIdx.y);

  // Execute FFT
  extern __shared__ typename FFT::value_type shared_mem[];
  FFT().execute(thread_data, shared_mem);

  // Copy to dst
  cufftdx::example::io<FFT>::store(thread_data, dst, threadIdx.y);
}

template <class FFT, class IO>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void fft_kernel(int subbatches, typename FFT::input_type* source, typename FFT::output_type* dst) {
  if (threadIdx.y + blockIdx.x * FFT::ffts_per_block >= subbatches) { return; }

  // Local array for thread
  typename FFT::value_type thread_data[FFT::storage_size];
  extern __shared__ __align__(16) typename FFT::value_type shared_mem[];

  // Load data from global memory to registers
  IO io;
  io.load_gmem_to_rmem(source, shared_mem, thread_data);

  __syncthreads();

  // Execute FFT
  FFT().execute(thread_data, shared_mem);

  __syncthreads();

  // Copy to dst
  io.store_rmem_to_gmem(thread_data, shared_mem, dst);
}

static __global__ void convolve_2d_kernel(cufftdx::complex<float>* __restrict__ buffer, uint2 size,
                                          const cufftdx::complex<float>* __restrict__ x_kernel,
                                          const cufftdx::complex<float>* __restrict__ y_kernel) {
  const uint2 index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if ((index.x >= size.x) || (index.y >= size.y)) { return; }

  buffer[index.y * size.x + index.x] *= x_kernel[index.x] * y_kernel[index.y];
}

static __global__ void convolve_3d_kernel(float2* __restrict__ buffer, uint3 size,
                                          const float2* __restrict__ x_kernel,
                                          const float2* __restrict__ y_kernel,
                                          const float2* __restrict__ z_kernel) {
  const uint3 index = make_uint3(blockIdx.x * blockDim.x + threadIdx.x,
                                 blockIdx.y * blockDim.y + threadIdx.y,
                                 blockIdx.z * blockDim.z + threadIdx.z);

  if ((index.x >= size.x) || (index.y >= size.y) || (index.z >= size.z)) { return; }

  buffer[((index.z * size.y) + index.y) * size.x + index.x] *=
      x_kernel[index.x] * y_kernel[index.y] * z_kernel[index.z];
}

static __global__ void imag_to_real_kernel(const cufftdx::complex<float>* __restrict__ source,
                                           uint2 size, float* __restrict__ dst) {
  const uint2 index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if ((index.x >= size.x) || (index.y >= size.y)) { return; }

  cufftdx::complex<float> value = source[index.y * size.x + index.x];
  dst[index.y * size.x + index.x] = sqrtf(value.x * value.x + value.y * value.y);
}

#else
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
#endif

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
  const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

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
#pragma unroll
  for (unsigned int i = 0; i < HilbertForwardFFT::elements_per_thread; ++i) {
    if (!has_partial_load || (index < HilbertForwardFFT::input_length)) {
      reinterpret_cast<float*>(thread_data)[i] = buffer[index];
      index += HilbertForwardFFT::stride;
    }
  }

  // Execute forward FFT
  extern __shared__ HilbertForwardFFT::value_type shared_mem[];
  HilbertForwardFFT().execute(thread_data, shared_mem);

  // Zero out negative frequencies and double positive frequencies (keep zero frequency)
  constexpr unsigned int half_size = HilbertForwardFFT::input_length >> 1;
  index = threadIdx.x;
#pragma unroll
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
#pragma unroll
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
                                                       uint2 output_size, float opening_angle,
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
  if (fabsf(angle) > opening_angle / 2.f) {
    output[index.y * output_size.x + index.x] = std::numeric_limits<float>::lowest();
    return;
  }

  const float source_x = (dist - near) / (far - near);
  const float source_y = angle / opening_angle + 0.5f;

  output[index.y * output_size.x + index.x] = tex2D<float>(input, source_x, source_y);
}

CUDAAlgorithms::CUDAAlgorithms()
    : normalize_launcher_((void*)&normalize_kernel),
#ifdef FFT_CONF
      convolve_2d_launcher_((void*)&convolve_2d_kernel),
      convolve_3d_launcher_((void*)&convolve_3d_kernel),
      imag_to_real_launcher_((void*)&imag_to_real_kernel),
#else
      convolve_rows_launcher_((void*)&convolve_rows_kernel),
      convolve_columns_launcher_((void*)&convolve_columns_kernel),
      convolve_planes_launcher_((void*)&convolve_planes_kernel),
#endif
      mean_planes_launcher_((void*)&mean_planes_kernel),
      log_compression_launcher_((void*)&log_compression_kernel),
      mul_rows_launcher_((void*)&mul_rows_kernel),
      scan_convert_curvilinear_launcher_((void*)&scan_convert_curvilinear_kernel) {
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

#ifdef FFT_CONF
void CUDAAlgorithms::fft_r2c(CudaMemory* source, CudaMemory* dst, cudaStream_t stream) {
  const size_t source_size = source->get_size() / sizeof(float);
  const size_t dst_size = dst->get_size() / sizeof(float2);

  if (source_size > 4096) { throw std::runtime_error("Source size is too large"); }

  using BaseFFT = decltype(cufftdx::Precision<float>() + cufftdx::Type<cufftdx::fft_type::r2c>() +
                           cufftdx::RealFFTOptions<cufftdx::complex_layout::full,
                                                   cufftdx::real_mode::normal>() +
                           cufftdx::Block() + cufftdx::ElementsPerThread<8>() +
                           cufftdx::FFTsPerBlock<1>() + cufftdx::SM<CUFFTDX_ARCH>());

  if (dst_size == 4096) {
    using FFT = decltype(BaseFFT() + cufftdx::Size<4096>());
    auto kernel = fft_kernel<FFT>;

    // Pad the input with zeros and place it at 0
    CudaMemory temp_input(dst_size * sizeof(float), stream);
    CUDA_CHECK(cudaMemsetAsync(temp_input.get_ptr(stream), 0, dst_size * sizeof(float), stream));
    CUDA_CHECK(
        cudaMemcpyAsync(reinterpret_cast<float*>(temp_input.get_ptr(stream)),
                        reinterpret_cast<const float*>(source->get_ptr(stream)) + source_size / 2,
                        source_size / 2 * sizeof(float),
                        cudaMemcpyDeviceToDevice,
                        stream));
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<float*>(temp_input.get_ptr(stream)) + dst_size - source_size / 2,
        reinterpret_cast<const float*>(source->get_ptr(stream)),
        source_size / 2 * sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream));

    CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));
    kernel<<<1, FFT::block_dim, FFT::shared_memory_size, stream>>>(
        reinterpret_cast<FFT::input_type*>(temp_input.get_ptr(stream)),
        reinterpret_cast<FFT::output_type*>(dst->get_ptr(stream)));
    CUDA_CHECK(cudaPeekAtLastError());

#if 0
    CudaMemory temp_imag_to_real(dst_size * sizeof(float), stream);
    imag_to_real_launcher_.launch(dst_size,
                                  stream,
                                  reinterpret_cast<cufftdx::complex<float>*>(dst->get_ptr(stream)),
                                  make_uint2(dst_size, 1),
                                  reinterpret_cast<float*>(temp_imag_to_real.get_ptr(stream)));

    write_image(&temp_input, make_uint2(dst_size, 1), "kernel_in.png");
    write_image(&temp_imag_to_real, make_uint2(dst_size, 1), "kernel.png");
#endif
  } else if (dst_size == 16) {
    using FFT = decltype(BaseFFT() + cufftdx::Size<16>());
    auto kernel = fft_kernel<FFT>;

    // Pad the input with zeros and place it at 0
    CudaMemory temp_input(dst_size * sizeof(float), stream);
    CUDA_CHECK(cudaMemsetAsync(temp_input.get_ptr(stream), 0, dst_size * sizeof(float), stream));
    CUDA_CHECK(
        cudaMemcpyAsync(reinterpret_cast<float*>(temp_input.get_ptr(stream)),
                        reinterpret_cast<const float*>(source->get_ptr(stream)) + source_size / 2,
                        source_size / 2 * sizeof(float),
                        cudaMemcpyDeviceToDevice,
                        stream));
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<float*>(temp_input.get_ptr(stream)) + dst_size - source_size / 2,
        reinterpret_cast<const float*>(source->get_ptr(stream)),
        source_size / 2 * sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream));

    CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));
    kernel<<<1, FFT::block_dim, FFT::shared_memory_size, stream>>>(
        reinterpret_cast<FFT::input_type*>(temp_input.get_ptr(stream)),
        reinterpret_cast<FFT::output_type*>(dst->get_ptr(stream)));
    CUDA_CHECK(cudaPeekAtLastError());
  } else {
    throw std::runtime_error("Unexpected FFT size");
  }
}

void CUDAAlgorithms::convolve(CudaMemory* buffer, uint3 size, CudaMemory* x_kernel,
                              CudaMemory* y_kernel, CudaMemory* z_kernel, cudaStream_t stream) {
  constexpr unsigned int X_SIZE = 4096;
  constexpr unsigned int Y_SIZE = 4096;
  constexpr unsigned int Z_SIZE = 16;

  if (size.x != X_SIZE) {
    std::stringstream buf;
    buf << "Convolve: x size of " << size.x << " does not match supported x size of " << X_SIZE
        << ".";
    throw std::runtime_error(buf.str().c_str());
  }
  if (size.y != Y_SIZE) {
    std::stringstream buf;
    buf << "Convolve: y size of " << size.y << " does not match supported y size of " << Y_SIZE
        << ".";
    throw std::runtime_error(buf.str().c_str());
  }
  if ((size.z != 1) && (size.z != Z_SIZE)) {
    std::stringstream buf;
    buf << "Convolve: z size of " << size.z << " does not match supported z size of " << Z_SIZE
        << ".";
    throw std::runtime_error(buf.str().c_str());
  }

  if ((x_kernel->get_size() / sizeof(float) / 2) != X_SIZE) {
    throw std::runtime_error("Unexpected x kernel size");
  }
  if ((y_kernel->get_size() / sizeof(float) / 2) != Y_SIZE) {
    throw std::runtime_error("Unexpected y kernel size");
  }
  if (z_kernel && ((z_kernel->get_size() / sizeof(float) / 2) != Z_SIZE)) {
    throw std::runtime_error("Unexpected z kernel size");
  }

  using BaseFFT =
      decltype(cufftdx::Precision<float>() + cufftdx::Block() + cufftdx::SM<CUFFTDX_ARCH>());

  using BaseXFFT = decltype(BaseFFT() + cufftdx::Size<X_SIZE>() +
                            cufftdx::RealFFTOptions<cufftdx::complex_layout::full,
                                                    cufftdx::real_mode::normal>());
  using BaseXForwardFFT = decltype(BaseXFFT() + cufftdx::Type<cufftdx::fft_type::r2c>() +

                                   cufftdx::Direction<cufftdx::fft_direction::forward>());
  using XForwardFFT = decltype(BaseXForwardFFT() +
                               cufftdx::FFTsPerBlock<BaseXForwardFFT::suggested_ffts_per_block>());
  using BaseXInverseFFT = decltype(BaseXFFT() + cufftdx::Type<cufftdx::fft_type::c2r>() +
                                   cufftdx::Direction<cufftdx::fft_direction::inverse>());
  using XInverseFFT = decltype(BaseXInverseFFT() +
                               cufftdx::FFTsPerBlock<BaseXInverseFFT::suggested_ffts_per_block>());

  using BaseYFFT =
      decltype(BaseFFT() + cufftdx::Size<Y_SIZE>() + cufftdx::Type<cufftdx::fft_type::c2c>());
  using BaseYForwardFFT =
      decltype(BaseYFFT() + cufftdx::Direction<cufftdx::fft_direction::forward>());
  using YForwardFFT = decltype(BaseYForwardFFT() +
                               cufftdx::FFTsPerBlock<BaseYForwardFFT::suggested_ffts_per_block>());
  using BaseYInverseFFT =
      decltype(BaseYFFT() + cufftdx::Direction<cufftdx::fft_direction::inverse>());
  using YInverseFFT = decltype(BaseYInverseFFT() +
                               cufftdx::FFTsPerBlock<BaseYInverseFFT::suggested_ffts_per_block>());

  using BaseZFFT =
      decltype(BaseFFT() + cufftdx::Size<Z_SIZE>() + cufftdx::Type<cufftdx::fft_type::c2c>());
  using BaseZForwardFFT =
      decltype(BaseZFFT() + cufftdx::Direction<cufftdx::fft_direction::forward>());
  using ZForwardFFT = decltype(BaseZForwardFFT() +
                               cufftdx::FFTsPerBlock<BaseZForwardFFT::suggested_ffts_per_block>());
  using BaseZInverseFFT =
      decltype(BaseZFFT() + cufftdx::Direction<cufftdx::fft_direction::inverse>());
  using ZInverseFFT = decltype(BaseZInverseFFT() +
                               cufftdx::FFTsPerBlock<BaseZInverseFFT::suggested_ffts_per_block>());

  temp_convolve_.resize(size.x * size.y * size.z * sizeof(XForwardFFT::value_type), stream);

  {
    using FFT = XForwardFFT;
    using IO = cufftdx::example::io_strided_conv_smem<cufftdx::example::dimension::z,
                                                      true /*Front*/,
                                                      1 /*Batches*/,
                                                      ZForwardFFT,
                                                      ZInverseFFT,
                                                      YForwardFFT,
                                                      YInverseFFT,
                                                      XForwardFFT,
                                                      XInverseFFT>;
    auto kernel = fft_kernel<FFT, IO>;

    CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, IO::get_shared_bytes()));
    const dim3 grid{cufftdx::example::div_up(size.y * size.z, FFT::ffts_per_block), 1, 1};
    kernel<<<grid, FFT::block_dim, IO::get_shared_bytes(), stream>>>(size.y * size.z,
        reinterpret_cast<FFT::input_type*>(buffer->get_ptr(stream)),
        reinterpret_cast<FFT::output_type*>(temp_convolve_.get_ptr(stream)));
    CUDA_CHECK(cudaPeekAtLastError());

#if 0
    CudaMemory temp_imag_to_real(size.x * size.y * sizeof(float), stream);
    imag_to_real_launcher_.launch(
        size,
        stream,
        reinterpret_cast<cufftdx::complex<float>*>(temp_convolve_.get_ptr(stream)),
        size,
        reinterpret_cast<float*>(temp_imag_to_real.get_ptr(stream)));

    write_image(&temp_imag_to_real, make_uint2(size.x, size.y), "convolve_fwd_x.png");
#endif
  }

  {
    using FFT = YForwardFFT;
    using IO = cufftdx::example::io_strided_conv_smem<cufftdx::example::dimension::y,
                                                      true /*Front*/,
                                                      1 /*Batches*/,
                                                      ZForwardFFT,
                                                      ZInverseFFT,
                                                      YForwardFFT,
                                                      YInverseFFT,
                                                      XForwardFFT,
                                                      XInverseFFT>;
    auto kernel = fft_kernel<FFT, IO>;

    CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, IO::get_shared_bytes()));
    const dim3 grid{cufftdx::example::div_up(size.x * size.z, FFT::ffts_per_block), 1, 1};
    kernel<<<grid, FFT::block_dim, IO::get_shared_bytes(), stream>>>(size.x * size.z,
        reinterpret_cast<FFT::input_type*>(temp_convolve_.get_ptr(stream)),
        reinterpret_cast<FFT::output_type*>(temp_convolve_.get_ptr(stream)));
    CUDA_CHECK(cudaPeekAtLastError());

#if 0
    CudaMemory temp_imag_to_real(size.x * size.y * sizeof(float), stream);
    imag_to_real_launcher_.launch(
        size,
        stream,
        reinterpret_cast<cufftdx::complex<float>*>(temp_convolve_.get_ptr(stream)),
        size,
        reinterpret_cast<float*>(temp_imag_to_real.get_ptr(stream)));

    write_image(&temp_imag_to_real, make_uint2(size.x, size.y), "convolve_fwd_y.png");
#endif
  }

  if (size.z != 1) {
    using FFT = ZForwardFFT;
    using IO = cufftdx::example::io_strided_conv_smem<cufftdx::example::dimension::x,
                                                      true /*Front*/,
                                                      1 /*Batches*/,
                                                      ZForwardFFT,
                                                      ZInverseFFT,
                                                      YForwardFFT,
                                                      YInverseFFT,
                                                      XForwardFFT,
                                                      XInverseFFT>;
    auto kernel = fft_kernel<FFT, IO>;

    CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, IO::get_shared_bytes()));
    const dim3 grid{cufftdx::example::div_up(size.x * size.y, FFT::ffts_per_block), 1, 1};
    kernel<<<grid, FFT::block_dim, IO::get_shared_bytes(), stream>>>(size.x * size.y,
        reinterpret_cast<FFT::input_type*>(temp_convolve_.get_ptr(stream)),
        reinterpret_cast<FFT::output_type*>(temp_convolve_.get_ptr(stream)));
    CUDA_CHECK(cudaPeekAtLastError());

#if 0
    CudaMemory temp_imag_to_real(size.x * size.y * sizeof(float), stream);
    imag_to_real_launcher_.launch(
        size,
        stream,
        reinterpret_cast<cufftdx::complex<float>*>(temp_convolve_.get_ptr(stream)),
        size,
        reinterpret_cast<float*>(temp_imag_to_real.get_ptr(stream)));

    write_image(&temp_imag_to_real, make_uint2(size.x, size.y), "convolve_fwd_z.png");
#endif
  }

  if (size.z != 1) {
    convolve_3d_launcher_.launch(
        size,
        stream,
        reinterpret_cast<cufftdx::complex<float>*>(temp_convolve_.get_ptr(stream)),
        size,
        reinterpret_cast<const cufftdx::complex<float>*>(x_kernel->get_ptr(stream)),
        reinterpret_cast<const cufftdx::complex<float>*>(y_kernel->get_ptr(stream)),
        reinterpret_cast<const cufftdx::complex<float>*>(z_kernel->get_ptr(stream)));
  } else {
    convolve_2d_launcher_.launch(
        make_uint2(size.x, size.y),
        stream,
        reinterpret_cast<cufftdx::complex<float>*>(temp_convolve_.get_ptr(stream)),
        make_uint2(size.x, size.y),
        reinterpret_cast<const cufftdx::complex<float>*>(x_kernel->get_ptr(stream)),
        reinterpret_cast<const cufftdx::complex<float>*>(y_kernel->get_ptr(stream)));
  }
  CUDA_CHECK(cudaPeekAtLastError());

#if 0
  CudaMemory temp_imag_to_real(size.x * size.y * sizeof(float), stream);
  imag_to_real_launcher_.launch(
      size,
      stream,
      reinterpret_cast<cufftdx::complex<float>*>(temp_convolve_.get_ptr(stream)),
      size,
      reinterpret_cast<float*>(temp_imag_to_real.get_ptr(stream)));

  write_image(&temp_imag_to_real, make_uint2(size.x, size.y), "convolve.png");
#endif

  if (size.z != 1) {
    using FFT = ZInverseFFT;
    using IO = cufftdx::example::io_strided_conv_smem<cufftdx::example::dimension::x,
                                                      false /*Front*/,
                                                      1 /*Batches*/,
                                                      ZForwardFFT,
                                                      ZInverseFFT,
                                                      YForwardFFT,
                                                      YInverseFFT,
                                                      XForwardFFT,
                                                      XInverseFFT>;
    auto kernel = fft_kernel<FFT, IO>;

    CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, IO::get_shared_bytes()));
    const dim3 grid{cufftdx::example::div_up(size.x * size.y, FFT::ffts_per_block), 1, 1};
    kernel<<<grid, FFT::block_dim, IO::get_shared_bytes(), stream>>>(size.x * size.y,
        reinterpret_cast<FFT::input_type*>(temp_convolve_.get_ptr(stream)),
        reinterpret_cast<FFT::output_type*>(temp_convolve_.get_ptr(stream)));
    CUDA_CHECK(cudaPeekAtLastError());
  }

  {
    using FFT = YInverseFFT;
    using IO = cufftdx::example::io_strided_conv_smem<cufftdx::example::dimension::y,
                                                      false /*Front*/,
                                                      1 /*Batches*/,
                                                      ZForwardFFT,
                                                      ZInverseFFT,
                                                      YForwardFFT,
                                                      YInverseFFT,
                                                      XForwardFFT,
                                                      XInverseFFT>;
    auto kernel = fft_kernel<FFT, IO>;

    CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, IO::get_shared_bytes()));
    const dim3 grid{cufftdx::example::div_up(size.x * size.z, FFT::ffts_per_block), 1, 1};
    kernel<<<grid, FFT::block_dim, IO::get_shared_bytes(), stream>>>(size.x * size.z,
        reinterpret_cast<FFT::input_type*>(temp_convolve_.get_ptr(stream)),
        reinterpret_cast<FFT::output_type*>(temp_convolve_.get_ptr(stream)));
    CUDA_CHECK(cudaPeekAtLastError());
#if 0
    CudaMemory temp_imag_to_real(size.x * size.y * sizeof(float), stream);
    imag_to_real_launcher_.launch(
        size,
        stream,
        reinterpret_cast<cufftdx::complex<float>*>(temp_convolve_.get_ptr(stream)),
        size,
        reinterpret_cast<float*>(temp_imag_to_real.get_ptr(stream)));

    write_image(&temp_imag_to_real, make_uint2(size.x, size.y), "convolve_inverse_y.png");
#endif
  }

  {
    using FFT = XInverseFFT;
    using IO = cufftdx::example::io_strided_conv_smem<cufftdx::example::dimension::z,
                                                      false /*Front*/,
                                                      1 /*Batches*/,
                                                      ZForwardFFT,
                                                      ZInverseFFT,
                                                      YForwardFFT,
                                                      YInverseFFT,
                                                      XForwardFFT,
                                                      XInverseFFT>;
    auto kernel = fft_kernel<FFT, IO>;

    CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, IO::get_shared_bytes()));
    const dim3 grid{cufftdx::example::div_up(size.y * size.z, FFT::ffts_per_block), 1, 1};
    kernel<<<grid, FFT::block_dim, IO::get_shared_bytes(), stream>>>(size.y * size.z,
        reinterpret_cast<FFT::input_type*>(temp_convolve_.get_ptr(stream)),
        reinterpret_cast<FFT::output_type*>(buffer->get_ptr(stream)));
    CUDA_CHECK(cudaPeekAtLastError());
#if 0
    write_image(buffer, make_uint2(size.x, size.y), "convolve_inverse_x.png");
#endif
  }
}
#else
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
#endif
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
  CUDA_CHECK(cudaPeekAtLastError());
}

std::unique_ptr<CudaMemory> CUDAAlgorithms::scan_convert_curvilinear(
    CudaMemory* scan_lines, uint2 input_size, float opening_angle, float near, float far,
    uint2 output_size, cudaStream_t stream) {
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
  const float opening_angle_rad = (opening_angle / 360.f) * 2.f * M_PI;
  const float max_x = std::sin(opening_angle_rad * 0.5f);                 // width / 2
  const float min_z = std::cos(opening_angle_rad * 0.5f) * (near / far);  // depth

  scan_convert_curvilinear_launcher_.launch(output_size,
                                            stream,
                                            scan_convert_curvilinear_texture_->get_texture().get(),
                                            input_size,
                                            reinterpret_cast<float*>(grid_z->get_ptr(stream)),
                                            output_size,
                                            opening_angle_rad,
                                            near / far,
                                            far / far,
                                            max_x,
                                            min_z);

  return std::move(grid_z);
}

}  // namespace raysim
