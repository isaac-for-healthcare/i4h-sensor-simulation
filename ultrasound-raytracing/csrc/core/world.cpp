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

#include "raysim/core/world.hpp"

#include <optix_function_table_definition.h>

#include <random>
#include <vector>

#include <spdlog/spdlog.h>

#include "raysim/cuda/optix_trace.hpp"

namespace raysim {

World::World(const std::string& background_material) : background_material_(background_material) {
  // Generate 3D noise texture for scattering
  const uint32_t size = 256;
  spdlog::info("Generating {}x{}x{} scattering texture", size, size, size);

  scattering_array_.reset(
      new CudaArray({size, size, size}, cudaChannelFormatKindFloat, sizeof(float), 2));

  std::vector<float> values(2 * size * size * size);
  std::mt19937 generator;
  // Generate uniform distribution for channel 0 for scattering density
  std::uniform_real_distribution uniform_distribution{0.f, 1.f};
  // Generate normal distribution for channel 1 for scattering intensity
  std::normal_distribution normal_distribution{0.f, 1.f};
  for (size_t i = 0; i < values.size() / 2; ++i) {
    values[i * 2 + 0] = uniform_distribution(generator);
    values[i * 2 + 1] = normal_distribution(generator);
  }
  scattering_array_->upload(values.data(), cudaStreamDefault);

  scattering_texture_.reset(
      new CudaTexture(scattering_array_, cudaAddressModeWrap, cudaFilterModeLinear));
}

World::~World() {}

void World::add(Hitable* hitable) {
  objects_.emplace_back(hitable);

  auto obj_aabb_min = hitable->get_aabb_min();
  aabb_min_.x = std::min(aabb_min_.x, obj_aabb_min.x);
  aabb_min_.y = std::min(aabb_min_.y, obj_aabb_min.y);
  aabb_min_.z = std::min(aabb_min_.z, obj_aabb_min.z);

  auto obj_aabb_max = hitable->get_aabb_max();
  aabb_max_.x = std::max(aabb_max_.x, obj_aabb_max.x);
  aabb_max_.y = std::max(aabb_max_.y, obj_aabb_max.y);
  aabb_max_.z = std::max(aabb_max_.z, obj_aabb_max.z);
}

void World::build(OptixDeviceContext context, cudaStream_t stream) {
  build_input_.resize(objects_.size());
  hit_group_data_.resize(objects_.size());

  size_t index = 0;
  for (auto&& object : objects_) {
    object->build(&build_input_[index], &hit_group_data_[index], stream);
    ++index;
  }

  optix_build_gas(context, build_input_, &gas_handle_, &d_gas_output_buffer_, stream);
}

OptixTraversableHandle World::get_gas_handle() const {
  return gas_handle_;
}

const std::string& World::get_background_material() const {
  return background_material_;
}

cudaTextureObject_t World::get_scattering_texture() const {
  return scattering_texture_->get_texture().get();
}

const std::vector<OptixBuildInput>& World::get_build_input() const {
  return build_input_;
}

const std::vector<HitGroupData>& World::get_hit_group_data() const {
  return hit_group_data_;
}

float3 World::get_aabb_min() const {
  return aabb_min_;
}
float3 World::get_aabb_max() const {
  return aabb_max_;
}

}  // namespace raysim
