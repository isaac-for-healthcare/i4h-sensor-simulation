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

#ifndef CPP_CUDA_HELPER
#define CPP_CUDA_HELPER

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <memory>
#include <sstream>

#include "raysim/core/nullable_pointer.hpp"

#define CUDA_CHECK(CALL)                                                            \
  {                                                                                 \
    const cudaError_t RESULT = CALL;                                                \
    if (RESULT != cudaSuccess) {                                                    \
      std::stringstream buf;                                                        \
      buf << "[" << __FILE__ << ":" << __LINE__ << "] CUDA runtime call `" << #CALL \
          << "` failed with " << RESULT << " (" << cudaGetErrorName(RESULT)         \
          << "): " << cudaGetErrorString(RESULT);                                   \
      throw std::runtime_error(buf.str().c_str());                                  \
    }                                                                               \
  }

namespace raysim {

/// Deleter for unique_ptr of CUDA objects
template <typename T, cudaError_t func(T)>
struct Deleter {
  typedef T pointer;
  void operator()(T value) const { func(value); }
};

/// unique_ptr for of CUDA objects
using UniqueCudaMemory = std::unique_ptr<void, Deleter<void*, &cudaFree>>;
using UniqueCudaArray = std::unique_ptr<cudaArray, Deleter<cudaArray_t, &cudaFreeArray>>;
using UniqueCudaTexture =
    std::unique_ptr<Nullable<cudaTextureObject_t>,
                    Nullable<cudaTextureObject_t>::Deleter<cudaError_t, &cudaDestroyTextureObject>>;
using UniqueCudaEvent = std::unique_ptr<CUevent_st, Deleter<cudaEvent_t, &cudaEventDestroy>>;
using UniqueCudaStream = std::unique_ptr<cudaStream_t, Deleter<cudaStream_t, &cudaStreamDestroy>>;

/**
 * Cuda memory.
 */
class CudaMemory {
 public:
  /**
   * Construct
   */
  CudaMemory() = default;

  /**
   * Construct with size
   *
   * @param size [in] Size in bytes
   */
  explicit CudaMemory(size_t size);

  /**
   * Construct with size using stream ordered memory allocator
   *
   * @param size [in] size in bytes
   * @param stream [in] CUDA stream
   */
  explicit CudaMemory(size_t size, cudaStream_t stream);

  /**
   * Destroy the CUDA memory object
   */
  ~CudaMemory();

  /**
   * Resize the memory.
   *
   * @param new_size [in] new size
   */
  void resize(size_t new_size);

  /**
   * Resize the memory using stream ordered memory allocator.
   *
   * @param new_size [in] new size
   * @param stream [in] CUDA stream
   */
  void resize(size_t new_size, cudaStream_t stream);

  /**
   * Free the allocated CUDA memory.
   */
  void free();

  /**
   * Upload data to CUDA memory
   *
   * @param src
   * @param stream [in] CUDA stream used to access the memory
   * @param offset
   * @param size
   */
  void upload(const void* src, cudaStream_t stream, size_t offset = 0, size_t size = 0) const;

  /**
   * Download data from CUDA memory
   *
   * @param dst
   * @param stream [in] CUDA stream used to access the memory
   * @param offset
   * @param size
   */
  void download(void* dst, cudaStream_t stream, size_t offset = 0, size_t size = 0) const;

  /**
   * @returns the size
   */
  size_t get_size() const { return size_; }

  /**
   * Get the pointer to the memory.
   *
   * @param stream [in] CUDA stream used to access the memory
   *
   * @returns the pointer
   */
  void* get_ptr(cudaStream_t stream);

  /**
   * Get the device pointer of the memory.
   *
   * @param stream [in] CUDA stream used to access the memory
   *
   * @returns the device pointer
   */
  CUdeviceptr get_device_ptr(cudaStream_t stream);

  /**
   * @returns the CUDA memory handle
   */
  const UniqueCudaMemory& get_memory() const { return memory_; }

 private:
  size_t size_ = 0;
  bool using_stream_ordered_allocator_ = false;

  /// The stream used to allocate/access the memory
  cudaStream_t stream_ = cudaStreamDefault;

  UniqueCudaMemory memory_;
};

/**
 * Cuda array.
 */
class CudaArray {
 public:
  /**
   * Construct an array
   *
   * - A 1D array is allocated if the height and depth extents are both zero.
   * - A 2D array is allocated if only the depth extent is zero.
   * - A 3D array is allocated if all three extents are non-zero.
   *
   * @param size [in] array size
   * @param format [in] channel format
   * @param channels [in] bytes per channel
   * @param channels [in] channels per array element
   */
  explicit CudaArray(cudaExtent size, cudaChannelFormatKind format, uint32_t bytes_per_channel,
                     uint32_t channels = 1);

  CudaArray() = delete;

  /**
   * @returns the array size
   */
  cudaExtent get_size() const { return size_; }

  /**
   * @returns the array format
   */
  cudaChannelFormatKind get_format() const { return format_; }

  /**
   * @returns the CUDA array handle
   */
  const UniqueCudaArray& get_array() const { return array_; }

  /**
   * Upload data to array
   *
   * @param src
   * @param offset
   * @param size
   */
  void upload(void* src, cudaStream_t stream, cudaPos offset = {0, 0, 0},
              cudaExtent size = {0, 0, 0});

  /**
   * Upload data to array
   *
   * @param src
   * @param offset
   * @param size
   */
  void upload(CudaMemory* src, cudaStream_t stream, cudaPos offset = {0, 0, 0},
              cudaExtent size = {0, 0, 0});

 private:
  const cudaExtent size_;
  const cudaChannelFormatKind format_;
  const uint32_t bytes_per_channel_;
  const uint32_t channels_;

  UniqueCudaArray array_;
};

/**
 * Cuda texture.
 */
class CudaTexture {
 public:
  /**
   * Construct from Cuda array
   *
   * @param array [in] Cuda array
   * @param address_mode [in] Texture address mode
   * @param filter_mode [in] Texture filter mode
   * @param normalize_coords [in] If true texture coordinates are in range [0, 1], else [0, size]
   */
  explicit CudaTexture(const std::shared_ptr<CudaArray>& array,
                       cudaTextureAddressMode address_mode = cudaAddressModeClamp,
                       cudaTextureFilterMode filter_mode = cudaFilterModePoint,
                       bool normalize_coords = true);
  CudaTexture() = delete;

  /**
   * @returns the CUDA texture handle
   */
  const UniqueCudaTexture& get_texture() const { return texture_; }

 private:
  const std::shared_ptr<CudaArray> array_;
  UniqueCudaTexture texture_;
};

/**
 * RAII class to time the execution of a asyn CUDA function or kernel call.
 *
 * Usage:
 * @code{.cpp}
 * {
 *     CudaTiming timing("kernel_to_time");
 *     kernel_to_time<<<>>>();
 * }
 * @endcode
 */
class CudaTiming {
 public:
  /**
   * Construct. Starts the timing
   *
   * @param enable [in]
   * @param msg [in] message to when timing is done
   * @param stream [in] CUDA stream
   */
  explicit CudaTiming(bool enable, const std::string& msg, cudaStream_t stream);
  CudaTiming() = delete;

  /**
   * Destruct. Stops the timing and prints a message.
   */
  ~CudaTiming();

 private:
  const std::string msg_;
  const cudaStream_t stream_;
  UniqueCudaEvent start_event_;
  UniqueCudaEvent end_event_;
};

///
class CudaLauncher {
 public:
  /**
   * @brief Construct a new CUDA launcher object
   *
   * @param func
   */
  explicit CudaLauncher(void* func);

  /**
   * Launch a kernel on a 3D grid.
   *
   * @param grid [in] grid size
   * @param stream [in] stream
   * @param args [in] kernel arguments (optional)
   */
  template <class... TYPES>
  void launch(dim3 grid, cudaStream_t stream, TYPES... args) const {
    void* args_array[] = {reinterpret_cast<void*>(&args)...};
    launch_internal(grid, stream, args_array);
  }

  /**
   * Launch a kernel on a 2D grid.
   *
   * @param grid [in] grid size
   * @param stream [in] stream
   * @param args [in] kernel arguments (optional)
   */
  template <class... TYPES>
  void launch(uint2 grid, cudaStream_t stream, TYPES... args) const {
    void* args_array[] = {reinterpret_cast<void*>(&args)...};
    launch_internal({grid.x, grid.y, 1}, stream, args_array);
  }

 private:
  void* const func_;
  int optimal_block_size_;

  void launch_internal(dim3 grid, cudaStream_t stream, void** args) const;
};

}  // namespace raysim

#endif /* CPP_CUDA_HELPER */
