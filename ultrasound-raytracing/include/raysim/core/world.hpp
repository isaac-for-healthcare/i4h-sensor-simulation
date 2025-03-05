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

#ifndef CPP_WORLD
#define CPP_WORLD

#include <limits>
#include <list>

#include "raysim/core/hitable.hpp"
#include "raysim/cuda/optix_helper.hpp"

namespace raysim {

struct HitGroupData;

/// Container for all objects in the scene
class World {
 public:
  explicit World(const std::string& background_material = "water");
  ~World();

  /// Add an object to the world
  /// @param hitable object to add
  void add(Hitable* hitable);

  const std::string& get_background_material() const;

  cudaTextureObject_t get_scattering_texture() const;

  /**
   * Build the OptiX structures
   *
   * @param context
   * @param stream
   */
  void build(OptixDeviceContext context, cudaStream_t stream);

  OptixTraversableHandle get_gas_handle() const;

  const std::vector<OptixBuildInput>& get_build_input() const;

  const std::vector<HitGroupData>& get_hit_group_data() const;

  float3 get_aabb_min() const;
  float3 get_aabb_max() const;

 private:
  const std::string background_material_;

  std::list<std::unique_ptr<Hitable>> objects_;

  std::shared_ptr<CudaArray> scattering_array_;
  std::unique_ptr<CudaTexture> scattering_texture_;

  OptixTraversableHandle gas_handle_;
  std::unique_ptr<CudaMemory> d_gas_output_buffer_;

  std::vector<OptixBuildInput> build_input_;
  std::vector<HitGroupData> hit_group_data_;

  /// Axis aligned bounding box
  float3 aabb_min_{std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::max()};
  float3 aabb_max_{std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest()};
};

}  // namespace raysim

#endif /* CPP_WORLD */
