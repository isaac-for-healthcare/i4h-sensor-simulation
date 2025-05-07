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

#ifndef CPP_OPTIX_TRACE
#define CPP_OPTIX_TRACE

#include <optix.h>
#include <cstdint>

#include "raysim/core/material.hpp"
#include "raysim/core/probe_types.hpp"
#include "raysim/cuda/matrix.hpp"

namespace raysim {
struct Params {
  float* scanlines;
  uint32_t buffer_size;
  float t_far;
  float min_intensity;
  uint32_t max_depth;
  Material* materials;
  uint32_t background_material_id;
  cudaTextureObject_t scattering_texture;
  OptixTraversableHandle handle;
  float source_frequency;
};

struct RayGenData {
  int probe_type;            // Type of probe (curvilinear, linear, phased)
  float opening_angle;       // Field of view in degrees (sector angle for phased array)
  float elevational_height;  // Height in elevational direction in mm
  float radius;              // Radius of curvature in mm (for curvilinear)
  float width;               // Width of linear/phased array in mm
  float3 position;           // Probe position in world coordinates
  float33 rotation_matrix;   // Probe orientation in world coordinates
};

struct MissData {};

struct HitGroupData {
  uint32_t material_id;
  uint32_t* indices;
  float3* normals;
};

struct Payload {
  float intensity;
  uint32_t depth;
  float t_ancestors;
  // use 16 bit for object and material ID to safe space
  uint16_t current_obj_id;
  uint16_t outter_obj_id;
  uint16_t current_material_id;
  uint16_t outter_material_id;
};

}  // namespace raysim

#endif /* CPP_OPTIX_TRACE */
