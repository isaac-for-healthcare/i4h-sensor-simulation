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

#ifndef CPP_WRITE_IMAGE
#define CPP_WRITE_IMAGE

#include <string>

#include "raysim/cuda/cuda_helper.hpp"

namespace raysim {

void write_image(const CudaMemory* buffer, uint2 size, const std::string& filename,
                 const float2* min_max = nullptr, size_t offset = 0);

}  // namespace raysim

#endif /* CPP_WRITE_IMAGE */
