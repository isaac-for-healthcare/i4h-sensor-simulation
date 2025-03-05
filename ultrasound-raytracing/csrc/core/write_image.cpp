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

#include "raysim/core/write_image.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <spdlog/spdlog.h>
#include <stb_image_write.h>

namespace raysim {

void write_image(const CudaMemory* buffer, uint2 size, const std::string& filename,
                 const float2* min_max, size_t offset) {
  const uint32_t elements = size.x * size.y;

  auto host_data = std::unique_ptr<float[]>(new float[elements]);
  buffer->download(host_data.get(), cudaStreamDefault, offset, elements * sizeof(float));

  auto image = std::unique_ptr<uint8_t[]>(new uint8_t[elements]);

  float min, max;
  if (min_max) {
    min = min_max->x;
    max = min_max->y;
  } else {
    min = std::numeric_limits<float>::max();
    max = std::numeric_limits<float>::lowest();
    for (int i = 0; i < elements; ++i) {
      min = std::min(host_data.get()[i], min);
      max = std::max(host_data.get()[i], max);
    }
  }

  for (int i = 0; i < elements; ++i) {
    image.get()[i] = static_cast<uint8_t>(
        std::max(std::min((host_data.get()[i] - min) / (max - min), 1.f), 0.f) * 255.f + 0.5f);
  }

  if (!stbi_write_png(filename.c_str(), size.x, size.y, 1, image.get(), 0)) {
    throw std::runtime_error("Failed to write image");
  }

  spdlog::info("Saved {} ({}, {})", filename, min, max);
}

}  // namespace raysim
