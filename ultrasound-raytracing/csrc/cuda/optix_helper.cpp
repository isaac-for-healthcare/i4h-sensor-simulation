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

#include "raysim/cuda/optix_helper.hpp"

#include <iomanip>
#include <iostream>

#include <optix_stack_size.h>

#include <spdlog/spdlog.h>

#include "optix_trace_ptx.hpp"
#include "raysim/cuda/optix_trace.hpp"

static void context_log_cb(unsigned int level, const char* tag, const char* message,
                           void* /*cbdata */) {
  spdlog::level::level_enum log_level;
  switch (level) {
    case 1:  // fatal
      log_level = spdlog::level::level_enum::critical;
      break;
    case 2:  // error
      log_level = spdlog::level::level_enum::err;
      break;
    case 3:  // warning
      log_level = spdlog::level::level_enum::warn;
      break;
    case 4:  // print
      log_level = spdlog::level::level_enum::info;
      break;
  }
  spdlog::log(log_level, "[{}]: {}", tag, message);
}

namespace raysim {

std::shared_ptr<OptixDeviceContext_t> optix_init() {
  // Initialize CUDA and Optix
  CUDA_CHECK(cudaFree(0));
  OPTIX_CHECK(optixInit());

  std::shared_ptr<OptixDeviceContext_t> context;
  {
    CUcontext cu_ctx = 0;  // zero means take the current context
    OptixDeviceContextOptions options{};
#ifndef NDEBUG
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    OptixDeviceContext t_context;
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &t_context));
    context = std::shared_ptr<OptixDeviceContext_t>(
        t_context, [](OptixDeviceContext ctx) { OPTIX_CHECK(optixDeviceContextDestroy(ctx)); });
  }

  return context;
}

void optix_create_pipeline(OptixDeviceContext context, std::shared_ptr<OptixPipeline_t>& pipeline,
                           std::shared_ptr<OptixProgramGroup_t>& raygen_prog_group,
                           std::shared_ptr<OptixProgramGroup_t>& miss_prog_group,
                           std::shared_ptr<OptixProgramGroup_t>& hitgroup_prog_group_sphere,
                           std::shared_ptr<OptixProgramGroup_t>& hitgroup_prog_group_triangle) {
  //
  // Create module
  //
  OptixModule module = nullptr;
  OptixModule sphere_module = nullptr;
  OptixModule triangle_module = nullptr;
  OptixPipelineCompileOptions pipeline_compile_options{};
  {
    OptixModuleCompileOptions module_compile_options{};
#ifdef NDEBUG
    // Generate information that does not impact performance but needed for NSight Compute to show
    // source code.
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#else
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues = sizeof(Payload) / sizeof(uint32_t);
    pipeline_compile_options.numAttributeValues = 1;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    pipeline_compile_options.usesPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE | OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    OPTIX_CHECK_LOG(optixModuleCreate(context,
                                      &module_compile_options,
                                      &pipeline_compile_options,
                                      optix_trace_ptx,
                                      std::strlen(optix_trace_ptx),
                                      LOG,
                                      &LOG_SIZE,
                                      &module));

    OptixBuiltinISOptions builtin_is_options{};

    builtin_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
    OPTIX_CHECK_LOG(optixBuiltinISModuleGet(context,
                                            &module_compile_options,
                                            &pipeline_compile_options,
                                            &builtin_is_options,
                                            &sphere_module));

    builtin_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_TRIANGLE;
    OPTIX_CHECK_LOG(optixBuiltinISModuleGet(context,
                                            &module_compile_options,
                                            &pipeline_compile_options,
                                            &builtin_is_options,
                                            &triangle_module));
  }

  //
  // Create program groups
  //
  {
    OptixProgramGroupOptions program_group_options{};  // Initialize to zeros

    OptixProgramGroupDesc raygen_prog_group_desc{};  //
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    OptixProgramGroup t_raygen_prog_group = nullptr;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context,
                                            &raygen_prog_group_desc,
                                            1,  // num program groups
                                            &program_group_options,
                                            LOG,
                                            &LOG_SIZE,
                                            &t_raygen_prog_group));
    raygen_prog_group = std::shared_ptr<OptixProgramGroup_t>(
        t_raygen_prog_group, [](OptixProgramGroup program_group) {
          OPTIX_CHECK(optixProgramGroupDestroy(program_group));
        });

    OptixProgramGroupDesc miss_prog_group_desc{};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    OptixProgramGroup t_miss_prog_group = nullptr;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context,
                                            &miss_prog_group_desc,
                                            1,  // num program groups
                                            &program_group_options,
                                            LOG,
                                            &LOG_SIZE,
                                            &t_miss_prog_group));
    miss_prog_group = std::shared_ptr<OptixProgramGroup_t>(
        t_miss_prog_group, [](OptixProgramGroup program_group) {
          OPTIX_CHECK(optixProgramGroupDestroy(program_group));
        });

    OptixProgramGroupDesc hitgroup_prog_group_desc{};
    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH = module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__sphere";
    hitgroup_prog_group_desc.hitgroup.moduleIS = sphere_module;
    OptixProgramGroup t_hitgroup_prog_group = nullptr;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context,
                                            &hitgroup_prog_group_desc,
                                            1,  // num program groups
                                            &program_group_options,
                                            LOG,
                                            &LOG_SIZE,
                                            &t_hitgroup_prog_group));
    hitgroup_prog_group_sphere = std::shared_ptr<OptixProgramGroup_t>(
        t_hitgroup_prog_group, [](OptixProgramGroup program_group) {
          OPTIX_CHECK(optixProgramGroupDestroy(program_group));
        });

    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__triangle";
    hitgroup_prog_group_desc.hitgroup.moduleIS = triangle_module;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context,
                                            &hitgroup_prog_group_desc,
                                            1,  // num program groups
                                            &program_group_options,
                                            LOG,
                                            &LOG_SIZE,
                                            &t_hitgroup_prog_group));
    hitgroup_prog_group_triangle = std::shared_ptr<OptixProgramGroup_t>(
        t_hitgroup_prog_group, [](OptixProgramGroup program_group) {
          OPTIX_CHECK(optixProgramGroupDestroy(program_group));
        });
  }

  //
  // Link pipeline
  //
  {
    OptixProgramGroup program_groups[] = {raygen_prog_group.get(),
                                          miss_prog_group.get(),
                                          hitgroup_prog_group_sphere.get(),
                                          hitgroup_prog_group_triangle.get()};

    OptixPipelineLinkOptions pipeline_link_options{};
    uint32_t max_trace_depth;
    OPTIX_CHECK(optixDeviceContextGetProperty(context,
                                              OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH,
                                              &max_trace_depth,
                                              sizeof(max_trace_depth)));
    pipeline_link_options.maxTraceDepth = max_trace_depth;
    OptixPipeline t_pipeline = nullptr;
    OPTIX_CHECK_LOG(optixPipelineCreate(context,
                                        &pipeline_compile_options,
                                        &pipeline_link_options,
                                        program_groups,
                                        sizeof(program_groups) / sizeof(program_groups[0]),
                                        LOG,
                                        &LOG_SIZE,
                                        &t_pipeline));
    pipeline = std::shared_ptr<OptixPipeline_t>(
        t_pipeline, [](OptixPipeline pipeline) { OPTIX_CHECK(optixPipelineDestroy(pipeline)); });

    OptixStackSizes stack_sizes{};
    for (auto& prog_group : program_groups) {
      OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline.get()));
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes,
                                           max_trace_depth,
                                           0,  // maxCCDepth
                                           0,  // maxDCDEpth
                                           &direct_callable_stack_size_from_traversal,
                                           &direct_callable_stack_size_from_state,
                                           &continuation_stack_size));
    OPTIX_CHECK(optixPipelineSetStackSize(pipeline.get(),
                                          direct_callable_stack_size_from_traversal,
                                          direct_callable_stack_size_from_state,
                                          continuation_stack_size,
                                          1  // maxTraversableDepth
                                          ));
  }
}

template <typename T>
T roundUp(T x, T y) {
  return ((x + y - 1) / y) * y;
}

void optix_build_gas(OptixDeviceContext context, const std::vector<OptixBuildInput>& build_input,
                     OptixTraversableHandle* gas_handle, std::unique_ptr<CudaMemory>* gas_buffer,
                     cudaStream_t stream) {
  OptixAccelBuildOptions accel_options{};
  accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                             OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
                             OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
  accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes gas_buffer_sizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(
      context, &accel_options, build_input.data(), build_input.size(), &gas_buffer_sizes));

  // non-compacted output
  const size_t compacted_size_offset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
  auto d_buffer_temp_output_gas_and_compacted_size =
      std::make_unique<CudaMemory>(compacted_size_offset + 8, stream);

  OptixAccelEmitDesc emit_property{};
  emit_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emit_property.result =
      d_buffer_temp_output_gas_and_compacted_size->get_device_ptr(stream) + compacted_size_offset;

  {
    CudaMemory d_temp_buffer_gas(gas_buffer_sizes.tempSizeInBytes, stream);

    OPTIX_CHECK(optixAccelBuild(context,
                                stream,
                                &accel_options,
                                build_input.data(),
                                build_input.size(),  // num build inputs
                                d_temp_buffer_gas.get_device_ptr(stream),
                                gas_buffer_sizes.tempSizeInBytes,
                                d_buffer_temp_output_gas_and_compacted_size->get_device_ptr(stream),
                                gas_buffer_sizes.outputSizeInBytes,
                                gas_handle,
                                &emit_property,  // emitted property list
                                1                // num emitted properties
                                ));
  }

  size_t compacted_gas_size = 0;
  CUDA_CHECK(cudaMemcpyAsync(&compacted_gas_size,
                             (void*)emit_property.result,
                             sizeof(size_t),
                             cudaMemcpyDeviceToHost,
                             stream));
  if (compacted_gas_size && (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)) {
    *gas_buffer = std::make_unique<CudaMemory>(compacted_gas_size, stream);

    // use handle as input and output
    OPTIX_CHECK(optixAccelCompact(context,
                                  0,
                                  *gas_handle,
                                  (*gas_buffer)->get_device_ptr(stream),
                                  compacted_gas_size,
                                  gas_handle));
  } else {
    *gas_buffer = std::move(d_buffer_temp_output_gas_and_compacted_size);
  }
}

}  // namespace raysim
