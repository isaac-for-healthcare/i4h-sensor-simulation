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

#include "raysim/core/material.hpp"

#include <algorithm>

#include "raysim/cuda/cuda_helper.hpp"

namespace raysim {

Material::Material(float impedance, float attenuation, float speed_of_sound, float mu0, float mu1,
                   float sigma, float specularity)
    : impedance_(impedance),
      attenuation_(attenuation),
      speed_of_sound_(speed_of_sound),
      mu0_(mu0),
      mu1_(mu1),
      sigma_(sigma),
      specularity_(specularity) {}

float Material::density() const {
  return impedance_ * 1e6f / speed_of_sound_;
}

Materials::Materials() {
  materials_ = {{"water", Material(1.48f, 0.0022f, 1480.f, 0.f)},
                {"blood", Material(1.61f, 0.18f, 1570.f, 0.1f, 0.1f, 0.1f)},
                {"fat", Material(1.38f, 0.63f, 1450.f, 1.f, 0.f, 1.f, 0.f)},
                {"liver", Material(1.65f, 0.7f, 1550.f, 0.7f, 0.f, 0.3f, 1e-5f)},
                {"muscle", Material(1.70f, 1.09f, 1580.f, 0.5f, 0.8f, 0.4f)},
                {"bone", Material(7.80f, 5.f, 4080.f, 0.8f, 0.9f, 0.5f)}};

  // Upload materials to device
  std::vector<Material> material_data;
  material_data.reserve(materials_.size());
  for (size_t index = 0; index < materials_.size(); ++index) {
    material_data.push_back(materials_[index].second);
  }
  const size_t material_data_size = materials_.size() * sizeof(Material);
  d_material_data_ = std::make_unique<CudaMemory>(material_data_size);
  d_material_data_->upload(&material_data[0], cudaStreamPerThread);
}

const std::unique_ptr<CudaMemory>& Materials::get_material_data() const {
  return Materials::d_material_data_;
}

uint32_t Materials::get_index(const std::string& name) const {
  const uint32_t index = std::distance(
      materials_.begin(),
      std::find_if(
          materials_.begin(),
          materials_.end(),
          [&name](const std::pair<std::string, Material>& item) { return item.first == name; }));
  if (index == materials_.size()) { throw std::runtime_error("Material not found"); }
  return index;
}

}  // namespace raysim
