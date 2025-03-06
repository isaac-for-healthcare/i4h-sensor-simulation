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

#ifndef CPP_MATERIAL
#define CPP_MATERIAL

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace raysim {

class CudaMemory;

struct Material {
  /**
   * Construct a new material
   *
   * @param impedance MRayl
   * @param attenuation dB/(cm⋅MHz)
   * @param speed_of_sound m/s
   * @param mu0 Scattering density
   * @param mu1
   * @param sigma Scattering coefficient
   * @param specularity
   */
  explicit Material(float impedance, float attenuation, float speed_of_sound, float mu0 = 0.f,
                    float mu1 = 0.f, float sigma = 0.f, float specularity = 1.f);

  float impedance_;
  float attenuation_;
  float speed_of_sound_;
  float mu0_;
  float mu1_;
  float sigma_;
  float specularity_;

  /**
   * @return density in kg/m^3
   */
  float density() const;
};

class Materials {
 public:
  Materials();

  const std::unique_ptr<CudaMemory>& get_material_data() const;

  uint32_t get_index(const std::string& name) const;

 private:
  std::vector<std::pair<std::string, Material>> materials_;
  std::unique_ptr<CudaMemory> d_material_data_;
};

}  // namespace raysim

#endif /* CPP_MATERIAL */
