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

#ifndef CPP_OPTIX_HELPER
#define CPP_OPTIX_HELPER

#include <optix.h>
#include <optix_stubs.h>

#include <cstdlib>
#include <memory>
#include <sstream>
#include <vector>

#include "cuda_helper.hpp"

#define OPTIX_CHECK(CALL)                                                                         \
  do {                                                                                            \
    const OptixResult RESULT = CALL;                                                              \
    if (RESULT != OPTIX_SUCCESS) {                                                                \
      std::stringstream buf;                                                                      \
      buf << "[" << __FILE__ << ":" << __LINE__ << "] Optix call `" << #CALL << "` failed with "  \
          << RESULT << " (" << optixGetErrorName(RESULT) << "): " << optixGetErrorString(RESULT); \
      throw std::runtime_error(buf.str().c_str());                                                \
    }                                                                                             \
  } while (false)

// This version of the log-check macro doesn't require the user do setup
// a log buffer and size variable in the surrounding context; rather the
// macro defines a log buffer and log size variable (LOG and LOG_SIZE)
// respectively that should be passed to the message being checked.
// E.g.:
//  OPTIX_CHECK_LOG2( optixProgramGroupCreate( ..., LOG, &LOG_SIZE, ... );
//
#define OPTIX_CHECK_LOG(CALL)                                                                    \
  do {                                                                                           \
    char LOG[2048];                                                                              \
    size_t LOG_SIZE = sizeof(LOG);                                                               \
    const OptixResult result = CALL;                                                             \
    if (result != OPTIX_SUCCESS) {                                                               \
      std::stringstream buf;                                                                     \
      buf << "[" << __FILE__ << ":" << __LINE__ << "] Optix call `" << #CALL << "` failed with " \
          << result << " (" << optixGetErrorName(result) << "): " << optixGetErrorString(result) \
          << "\nLog:\n"                                                                          \
          << LOG << (LOG_SIZE > sizeof(LOG) ? "<TRUNCATED>" : "") << '\n';                       \
      throw std::runtime_error(buf.str().c_str());                                               \
    }                                                                                            \
  } while (false)

namespace raysim {

std::shared_ptr<OptixDeviceContext_t> optix_init();
void optix_create_pipeline(OptixDeviceContext context, std::shared_ptr<OptixPipeline_t>& pipeline,
                           std::shared_ptr<OptixProgramGroup_t>& raygen_prog_group,
                           std::shared_ptr<OptixProgramGroup_t>& miss_prog_group,
                           std::shared_ptr<OptixProgramGroup_t>& hitgroup_prog_group_sphere,
                           std::shared_ptr<OptixProgramGroup_t>& hitgroup_prog_group_triangle);

void optix_build_gas(OptixDeviceContext context, const std::vector<OptixBuildInput>& build_input,
                     OptixTraversableHandle* gas_handle, std::unique_ptr<CudaMemory>* gas_buffer,
                     cudaStream_t stream);

}  // namespace raysim

#endif /* CPP_OPTIX_HELPER */
