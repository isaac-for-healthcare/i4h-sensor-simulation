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

#ifndef CPP_RAYTRACING_ULTRASOUND_SIMULATOR
#define CPP_RAYTRACING_ULTRASOUND_SIMULATOR

#include <limits>
#include <memory>
#include <limits>

#include "raysim/cuda/cuda_helper.hpp"
#include "raysim/cuda/optix_helper.hpp"

namespace raysim {

class World;
class Materials;
class UltrasoundProbe;
class CUDAAlgorithms;

class RaytracingUltrasoundSimulator {
 public:
  /**
   * Construct a new Raytracing Unltrasound Simulator object
   *
   * @param world World object containing scene geometry
   * @param materials Materials
   */
  explicit RaytracingUltrasoundSimulator(World* world, const Materials* materials);
  RaytracingUltrasoundSimulator() = delete;

  /// Simulation parameters
  struct SimParams {
    float t_far = 180.f;          // Maximum ray distance [mm]
    uint32_t buffer_size = 4096;  // Samples per line
    uint32_t max_depth = 15;      // Maximum reflection depth
    float min_intensity = 1e-3f;  // Minimum intensity threshold
    bool use_scattering = true;   // Enable scattering simulation
    bool conv_psf = false;        // Convolve the raw hits with the PSF
    uint2 b_mode_size = make_uint2(500, 500);
    cudaStream_t stream = cudaStreamPerThread;  // CUDA stream
    bool enable_cuda_timing = false;            // Print timing of CUDA operations
    bool write_debug_images = false;            // Write debug images to `debug_images` directory
    float boundary_value = std::numeric_limits<float>::lowest();  // Corrected default: Value for
                                                                  // pixels outside scan area
  };

  /// Simulation results
  struct SimResult {
    std::unique_ptr<CudaMemory> rf_data;
    std::unique_ptr<CudaMemory> b_mode;
  };

  /**
   * Generate a single B-mode ultrasound frame
   *
   * @param probe UltrasoundProbe object
   * @param sim_params Simulation parameters
   *
   * @returns Dictionary containing simulation results
   */
  SimResult simulate(const UltrasoundProbe* probe, const SimParams& sim_params);

  /**
   * Generate a binary mask representing the scan sector geometry, bypassing simulation.
   *
   * @param probe UltrasoundProbe object defining geometry
   * @param t_far Maximum imaging depth [mm] (used to define the far boundary)
   * @param output_size Width and height of the output mask image
   * @param inside_value Value for pixels inside the sector
   * @param outside_value Value for pixels outside the sector
   * @param stream CUDA stream
   * @return A CudaMemory buffer containing the mask
   */
  std::unique_ptr<CudaMemory> generate_scan_area(const UltrasoundProbe* probe, float t_far,
                                                   uint2 output_size, float inside_value,
                                                   float outside_value,
                                                   cudaStream_t stream = cudaStreamDefault);

  /**
   * Get the minimum x value of the simulated region
   *
   * @returns Minimum x value [mm]
   */
  float get_min_x() const { return min_x_; }

  /**
   * Get the maximum x value of the simulated region
   *
   * @returns Maximum x value [mm]
   */
  float get_max_x() const { return max_x_; }

  /**
   * Get the minimum z value of the simulated region
   *
   * @returns Minimum z value [mm]
   */
  float get_min_z() const { return min_z_; }

  /**
   * Get the maximum z value of the simulated region
   *
   * @returns Maximum z value [mm]
   */
  float get_max_z() const { return max_z_; }

 private:
  World* const world_;
  const Materials* const materials_;
  const float SAMPLING_FREQ = 40e6f;

  // Boundary values of the simulated region
  float min_x_ = 0.f;
  float max_x_ = 0.f;
  float min_z_ = 0.f;
  float max_z_ = 0.f;

  std::shared_ptr<OptixDeviceContext_t> context_;
  std::shared_ptr<OptixPipeline_t> pipeline_;               ///< OptiX pipeline
  std::shared_ptr<OptixProgramGroup_t> raygen_prog_group_;  ///< OptiX ray gen program group
  std::shared_ptr<OptixProgramGroup_t> miss_prog_group_;    ///< OptiX miss program group
  std::shared_ptr<OptixProgramGroup_t>
      hitgroup_prog_group_sphere_;  ///< OptiX hit group program group for spheres
  std::shared_ptr<OptixProgramGroup_t>
      hitgroup_prog_group_triangles_;               ///< OptiX hit group program group for triangles
  OptixShaderBindingTable shader_binding_table_{};  ///< OptiX shader binding table

  std::shared_ptr<CUDAAlgorithms> cuda_algorithms_;

  CudaMemory raygen_record_;
  CudaMemory miss_record_;
  CudaMemory hitgroup_record_;

  CudaMemory pipeline_params_;

  float probe_frequency_ = 0.f;
  std::unique_ptr<CudaMemory> psf_ax_;
  float probe_element_spacing_ = 0.f;
  std::unique_ptr<CudaMemory> psf_lat_;
  float probe_elevational_height_ = 0.f;
  std::unique_ptr<CudaMemory> psf_elev_;
  CudaMemory psf_tmp_;

  std::unique_ptr<CudaMemory> tgc_curve_;

  void update_psfs(const UltrasoundProbe* probe, cudaStream_t stream);
};

}  // namespace raysim

#endif /* CPP_RAYTRACING_ULTRASOUND_SIMULATOR */
