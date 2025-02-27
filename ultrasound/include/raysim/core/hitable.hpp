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

#ifndef CPP_HITABLE
#define CPP_HITABLE

#include <optix_types.h>
#include <vector_types.h>

#include <memory>
#include <vector>

#include "raysim/cuda/cuda_helper.hpp"

namespace Assimp {
class Importer;
}  // namespace Assimp

namespace raysim {

struct HitGroupData;

/// Base class for all hitable objects in the scene
class Hitable {
 public:
  explicit Hitable(uint32_t material_id);
  virtual ~Hitable(){};

  virtual void build(OptixBuildInput* optix_build_input, HitGroupData* hit_group_data,
                     cudaStream_t stream) = 0;

  float3 get_aabb_min() const { return aabb_min_; }
  float3 get_aabb_max() const { return aabb_max_; }

 protected:
  const uint32_t material_id_;

  /// Axis aligned bounding box
  float3 aabb_min_{0.f, 0.f, 0.f};
  float3 aabb_max_{1.f, 1.f, 1.f};

  std::vector<uint32_t> sbt_flags_;
};

/// Sphere object that can be hit by rays
class Sphere : public Hitable {
 public:
  Sphere(float3 center, float radius, uint32_t material_id);

  void build(OptixBuildInput* optix_build_input, HitGroupData* hit_group_data,
             cudaStream_t stream) override;

 private:
  const float3 center_;
  const float radius_;

  std::unique_ptr<CudaMemory> cuda_vertex_buffer_;
  std::unique_ptr<CudaMemory> cuda_radius_buffer_;

  std::vector<CUdeviceptr> vertex_buffers_;
  std::vector<CUdeviceptr> radius_buffers_;
};

/// Infinite plane object that can be hit by rays
class Mesh : public Hitable {
 public:
  Mesh(const std::string& file_name, uint32_t material_id);

  void build(OptixBuildInput* optix_build_input, HitGroupData* hit_group_data,
             cudaStream_t stream) override;

 private:
  std::shared_ptr<Assimp::Importer> importer_;

  std::unique_ptr<CudaMemory> cuda_vertex_buffer_;
  std::unique_ptr<CudaMemory> cuda_index_buffer_;
  std::unique_ptr<CudaMemory> cuda_normal_buffer_;

  std::vector<CUdeviceptr> vertex_buffers_;
  std::vector<CUdeviceptr> normal_buffers_;
};

}  // namespace raysim

#endif /* CPP_HITABLE */
