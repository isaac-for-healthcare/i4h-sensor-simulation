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

#include "raysim/cuda/cuda_helper.hpp"

#include <chrono>

#include <spdlog/spdlog.h>

namespace raysim {

CudaMemory::CudaMemory(size_t size) {
  resize(size);
}

CudaMemory::CudaMemory(size_t size, cudaStream_t stream) {
  resize(size, stream);
}

CudaMemory::~CudaMemory() {
  try {
    free();
  } catch (std::exception& e) {
    spdlog::error("CudaMemory destructor failed with exception '{}'", e.what());
  }
}

void CudaMemory::free() {
  if (memory_) {
    if (using_stream_ordered_allocator_) {
      // first free the current allocation
      CUDA_CHECK(cudaFreeAsync(memory_.get(), stream_));
      memory_.release();
    } else {
      memory_.reset();
    }
  }
}

void CudaMemory::upload(const void* src, cudaStream_t stream, size_t offset, size_t size) const {
  if (!memory_) {
    throw std::runtime_error(
        fmt::format("[{}:{}] CUDA memory had not been allocated", __FILE__, __LINE__));
  }
  if (size == 0) { size = size_; }
  CUDA_CHECK(cudaMemcpyAsync(memory_.get(), src, size, cudaMemcpyHostToDevice, stream));
}

void CudaMemory::download(void* dst, cudaStream_t stream, size_t offset, size_t size) const {
  if (!memory_) {
    throw std::runtime_error(
        fmt::format("[{}:{}] CUDA memory had not been allocated", __FILE__, __LINE__));
  }
  if (size == 0) { size = size_; }
  CUDA_CHECK(cudaMemcpyAsync(dst, memory_.get(), size, cudaMemcpyDeviceToHost, stream));
}

void CudaMemory::resize(size_t new_size) {
  resize(new_size, cudaStreamDefault);
}

void CudaMemory::resize(size_t new_size, cudaStream_t stream) {
  if (size_ != new_size) {
    // first free the current allocation
    free();
    // Then allocate the new one
    size_ = new_size;
    stream_ = stream;
    using_stream_ordered_allocator_ = stream != cudaStreamDefault;
    if (using_stream_ordered_allocator_) {
      memory_.reset([this] {
        void* ptr;
        CUDA_CHECK(cudaMallocAsync(&ptr, size_, stream_));
        return ptr;
      }());
    } else {
      memory_.reset([this] {
        void* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, size_));
        return ptr;
      }());
    }
  }
}

void* CudaMemory::get_ptr(cudaStream_t stream) {
  if (!memory_) {
    throw std::runtime_error(
        fmt::format("[{}:{}] CUDA memory had not been allocated", __FILE__, __LINE__));
  }
  if (using_stream_ordered_allocator_) { stream_ = stream; }
  return memory_.get();
}

CUdeviceptr CudaMemory::get_device_ptr(cudaStream_t stream) {
  if (!memory_) {
    throw std::runtime_error(
        fmt::format("[{}:{}] CUDA memory had not been allocated", __FILE__, __LINE__));
  }
  if (using_stream_ordered_allocator_) { stream_ = stream; }
  return reinterpret_cast<CUdeviceptr>(memory_.get());
};

CudaArray::CudaArray(cudaExtent size, cudaChannelFormatKind format, uint32_t bytes_per_channel,
                     uint32_t channels)
    : size_(size), format_(format), bytes_per_channel_(bytes_per_channel), channels_(channels) {
  array_ = UniqueCudaArray([this] {
    cudaArray_t array;
    cudaChannelFormatDesc desc{};
    desc.f = format_;
    desc.x = bytes_per_channel_ * 8;
    if (channels_ > 1) {
      desc.y = desc.x;
      if (channels_ > 2) {
        desc.z = desc.x;
        if (channels_ > 3) { desc.w = desc.x; }
      }
    }
    CUDA_CHECK(cudaMalloc3DArray(&array, &desc, size_));
    return array;
  }());
}

void CudaArray::upload(void* src, cudaStream_t stream, cudaPos offset, cudaExtent size) {
  if (size.width == 0) { size = size_; }
  cudaMemcpy3DParms p{};
  p.kind = cudaMemcpyHostToDevice;
  p.extent = size;
  p.srcPtr.ptr = src;
  p.srcPtr.pitch = size.width * bytes_per_channel_ * channels_;
  p.srcPtr.xsize = size.width;
  p.srcPtr.ysize = size.height;
  p.dstPos = offset;
  p.dstArray = array_.get();
  CUDA_CHECK(cudaMemcpy3DAsync(&p, stream));
}

void CudaArray::upload(CudaMemory* src, cudaStream_t stream, cudaPos offset, cudaExtent size) {
  if (size.width == 0) {
    size = size_;
    // A 2D array has size_.depth set to 0, but for copying the size.depth needs to be 1
    if (size_.depth == 0) { size.depth = 1; }
  }
  if (src->get_size() < size.width * size.height * bytes_per_channel_ * channels_) {
    throw std::runtime_error(fmt::format("[{}:{}] CudaMemory size to small", __FILE__, __LINE__));
  }
  cudaMemcpy3DParms p{};
  p.kind = cudaMemcpyDeviceToDevice;
  p.extent = size;
  p.srcPtr.ptr = src->get_ptr(stream);
  p.srcPtr.pitch = size.width * bytes_per_channel_ * channels_;
  p.srcPtr.xsize = size.width;
  p.srcPtr.ysize = size.height;
  p.dstPos = offset;
  p.dstArray = array_.get();
  CUDA_CHECK(cudaMemcpy3DAsync(&p, stream));
}

CudaTexture::CudaTexture(const std::shared_ptr<CudaArray>& array,
                         cudaTextureAddressMode address_mode, cudaTextureFilterMode filter_mode,
                         bool normalize_coords)
    : array_(array) {
  texture_.reset([this, address_mode, filter_mode, normalize_coords] {
    cudaTextureObject_t texture;
    cudaResourceDesc res_desc{};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = array_->get_array().get();

    cudaTextureDesc tex_desc{};
    tex_desc.addressMode[0] = address_mode;
    tex_desc.addressMode[1] = address_mode;
    tex_desc.addressMode[2] = address_mode;
    tex_desc.filterMode = filter_mode;
    tex_desc.normalizedCoords = normalize_coords;

    CUDA_CHECK(cudaCreateTextureObject(&texture, &res_desc, &tex_desc, nullptr));
    return texture;
  }());
}

CudaTiming::CudaTiming(bool enable, const std::string& msg, cudaStream_t stream)
    : msg_(msg), stream_(stream) {
  if (enable) {
    start_event_.reset([] {
      CUevent event;
      CUDA_CHECK(cudaEventCreate(&event, CU_EVENT_BLOCKING_SYNC));
      return event;
    }());
    end_event_.reset([] {
      CUevent event;
      CUDA_CHECK(cudaEventCreate(&event, CU_EVENT_BLOCKING_SYNC));
      return event;
    }());
    CUDA_CHECK(cudaEventRecord(start_event_.get(), stream));
  }
}

CudaTiming::~CudaTiming() {
  if (start_event_) {
    try {
      CUDA_CHECK(cudaEventRecord(end_event_.get(), stream_));
      CUDA_CHECK(cudaEventSynchronize(start_event_.get()));
      CUDA_CHECK(cudaEventSynchronize(end_event_.get()));
      const std::chrono::duration<float, std::milli> duration([this] {
        float milliseconds = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event_.get(), end_event_.get()));
        return milliseconds;
      }());
      spdlog::info("Timing {} took {} ms", msg_, duration.count());
    } catch (const std::exception& e) {
      spdlog::error("CudaTiming destructor failed with {}", e.what());
    }
  }
}

CudaLauncher::CudaLauncher(void* func) : func_(func) {
  int min_grid_size = 0;
  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &optimal_block_size_, func));
}

void CudaLauncher::launch_internal(dim3 grid, cudaStream_t stream, void** args) const {
  dim3 launch_block;

  // Find optimal launch block
  launch_block.x = 1;
  launch_block.y = 1;
  launch_block.z = 1;
  // Double each step
  while (static_cast<int>(launch_block.x * launch_block.y * launch_block.z * 2) <=
         optimal_block_size_) {
    if ((grid.z > 1) && (launch_block.y > launch_block.z)) {
      launch_block.z *= 2;
    } else if ((grid.y > 1) && (launch_block.x > launch_block.y)) {
      launch_block.y *= 2;
    } else {
      launch_block.x *= 2;
    }
  }

  // Limit block to grid
  launch_block.x = std::min(launch_block.x, grid.x);
  launch_block.y = std::min(launch_block.y, grid.y);
  launch_block.z = std::min(launch_block.z, grid.z);

  // Calculate the launch grid size
  dim3 launch_grid;
  launch_grid.x = (grid.x + (launch_block.x - 1)) / launch_block.x;
  launch_grid.y = (grid.y + (launch_block.y - 1)) / launch_block.y;
  launch_grid.z = (grid.z + (launch_block.z - 1)) / launch_block.z;

  // Launch kernel
  CUDA_CHECK(cudaLaunchKernel(func_, launch_grid, launch_block, args, 0, stream));
}

}  // namespace raysim
