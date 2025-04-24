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

#include "raysim/core/raytracing_ultrasound_simulator.hpp"

#include <spdlog/fmt/fmt.h>
#include <filesystem>

#include "raysim/core/ultrasound_probe.hpp"
#include "raysim/core/world.hpp"
#include "raysim/core/write_image.hpp"
#include "raysim/cuda/cuda_algorithms.hpp"
#include "raysim/cuda/optix_trace.hpp"

namespace raysim {

template <typename T>
struct SbtRecord {
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

/**
 * @brief Create Gaussian PSF kernel, optionally modulated by cosine.
 *
 * @param stream CUDA stream
 * @param width Standard deviation of Gaussian representation of point spread function [mm]
 * @param k Sampling frequency [1/mm] or [MHz]
 * @param freq Optional frequency for cosine modulation [MHz] or [1/mm]
 *
 * @returns 1D PSF kernel array
 */
static std::unique_ptr<CudaMemory> create_gaussian_psf(cudaStream_t stream, float width, float k,
                                                       float freq = 0.f) {
  const float c = 1.54f;  // [mm/us]
  const float dx = c / k;
  const uint32_t size =
      uint32_t(std::ceil(width / dx)) * 10 + 1;  // convolution filter size is 10x sigma
  std::vector<float> gaussian(size);

  const float t = -1.f / (2.f * std::pow(width, 2.f));

  float amplitude_integral = 0.f;
  for (int index = 0; index < size; ++index) {
    const float x = float(index - float(size / 2)) * dx;
    const float value = std::exp((x * x) * t);
    amplitude_integral += value;
    gaussian[index] = value;
  }
  // Normalize
  if (amplitude_integral != 0.f) {
    amplitude_integral = 1.f / amplitude_integral;
    for (int index = 0; index < size; ++index) { gaussian[index] *= amplitude_integral; }
  }

  if (freq != 0.f) {
    const float f = freq / c;  // spatial frequency of pulse 1/mm 1/wl
    for (int index = 0; index < size; ++index) {
      const float x = float(index - float(size / 2)) * dx;
      // Modulate Gaussian with cosine at given frequency
      gaussian[index] *= -std::cos(2.f * M_PI * f * x);
    }
  }

  auto buffer = std::make_unique<CudaMemory>(size * sizeof(float), stream);
  buffer->upload(gaussian.data(), stream);
  return buffer;
}

struct ControlPoint {
  float depth;  // cm
  float amp;    // dB
};

/**
 * Create a piece-wise linear TGC curve from control points.
 *
 * @param depth_samples Number of samples along depth
 * @param control_points Vector of (depth_cm, gain_db) pairs
 * @param c Speed of sound in tissue (m/s) (optional)
 * @param fs Sampling frequency (Hz) (optional)
 * @return TGC curve interpolated from control points
 */
static std::unique_ptr<CudaMemory> create_piece_wise_tgc(
    cudaStream_t stream, uint32_t depth_samples, const std::vector<ControlPoint>& control_points,
    float c = 1540.f, float fs = 50e6f) {
  std::vector<float> tgc_curve(depth_samples);

  float first_value;
  for (uint32_t i = 0; i < depth_samples; ++i) {
    // Time
    const float t = i / fs;
    // Depth in cm
    const float depth = (c * t / 2.f) * 100.f;

    // Interpolate between control points
    auto it = control_points.begin();
    while ((it != control_points.end()) && (depth < it->depth)) { ++it; }
    float value;
    if (it == control_points.end()) {
      value = control_points.back().amp;
    } else if (depth <= it->depth) {
      value = it->amp;
    } else {
      float depth_min = it->depth;
      float amp_min = it->amp;
      ++it;
      float depth_max = it->depth;
      float amp_max = it->amp;
      value = amp_min + (amp_max - amp_min) * ((depth - depth_min) / (depth_max - depth_min));
    }

    // Convert from dB to linear scale
    value = std::pow(10.f, value / 20.f);

    // Normalize to start at 1
    if (i == 0) {
      first_value = value;
      value = 1.f;
    } else {
      value = value / first_value;
    }
    tgc_curve[i] = value;
  }

  auto d_tgc_curve = std::make_unique<CudaMemory>(depth_samples * sizeof(float), stream);
  d_tgc_curve->upload(tgc_curve.data(), stream);

  return std::move(d_tgc_curve);
}

RaytracingUltrasoundSimulator::RaytracingUltrasoundSimulator(World* world,
                                                             const Materials* materials)
    : world_(world), materials_(materials) {
  // Initialize OptiX
  context_ = optix_init();

  // Create the OptiX pipeline
  optix_create_pipeline(context_.get(),
                        pipeline_,
                        raygen_prog_group_,
                        miss_prog_group_,
                        hitgroup_prog_group_sphere_,
                        hitgroup_prog_group_triangles_);

  const cudaStream_t stream = cudaStreamPerThread;

  // Build the acceleration structure
  world_->build(context_.get(), stream);

  // Set up the shader binding table
  const size_t raygen_record_size = sizeof(RayGenSbtRecord);
  raygen_record_.resize(raygen_record_size);
  shader_binding_table_.raygenRecord = raygen_record_.get_device_ptr(stream);

  MissSbtRecord ms_sbt;
  OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group_.get(), &ms_sbt));
  const size_t miss_record_size = sizeof(MissSbtRecord);
  miss_record_.resize(miss_record_size);
  miss_record_.upload(&ms_sbt, stream);
  shader_binding_table_.missRecordBase = miss_record_.get_device_ptr(stream);
  shader_binding_table_.missRecordStrideInBytes = sizeof(MissSbtRecord);
  shader_binding_table_.missRecordCount = 1;

  const std::vector<OptixBuildInput>& build_input = world_->get_build_input();
  const std::vector<HitGroupData>& hit_group_data = world_->get_hit_group_data();
  std::vector<HitGroupSbtRecord> hg_sbt(hit_group_data.size());
  for (size_t index = 0; index < hit_group_data.size(); ++index) {
    if (build_input[index].type == OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_SPHERES) {
      OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group_sphere_.get(), &hg_sbt[index]));
    } else if (build_input[index].type == OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_TRIANGLES) {
      OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group_triangles_.get(), &hg_sbt[index]));
    } else {
      throw std::runtime_error("Unhandled OptiX build input type");
    }
    hg_sbt[index].data = hit_group_data[index];
  }

  const size_t hitgroup_record_size = hg_sbt.size() * sizeof(HitGroupSbtRecord);
  hitgroup_record_.resize(hitgroup_record_size);
  hitgroup_record_.upload(&hg_sbt[0], stream);

  shader_binding_table_.hitgroupRecordBase = hitgroup_record_.get_device_ptr(stream);
  shader_binding_table_.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
  shader_binding_table_.hitgroupRecordCount = hit_group_data.size();

  pipeline_params_.resize(sizeof(Params));

  cuda_algorithms_ = std::make_shared<CUDAAlgorithms>();
}

void RaytracingUltrasoundSimulator::update_psfs(const UltrasoundProbe* probe, cudaStream_t stream) {
  if (probe_frequency_ != probe->get_frequency()) {
    probe_frequency_ = probe->get_frequency();
    psf_ax_.reset();
    psf_lat_.reset();
  }

  if (probe_element_spacing_ != probe->get_element_spacing()) {
    probe_element_spacing_ = probe->get_element_spacing();
    psf_lat_.reset();
  }

  if (probe_elevational_height_ != probe->get_elevational_height()) {
    probe_elevational_height_ = probe->get_elevational_height();
    psf_elev_.reset();
  }

  if (!psf_ax_) {
    const float k = SAMPLING_FREQ * 1e-6;  // [1/us]
    psf_ax_ = create_gaussian_psf(stream, probe->get_axial_resolution(), k, probe->get_frequency());
  }

  if (!psf_lat_) {
    psf_lat_ = create_gaussian_psf(
        stream, probe->get_lateral_resolution(), 1.f / probe->get_element_spacing());
  }

  if ((probe->get_num_el_samples() > 1) && !psf_elev_) {
    psf_elev_ = create_gaussian_psf(stream, 2.f, probe->get_elevational_spatial_frequency());
  }
}

RaytracingUltrasoundSimulator::SimResult RaytracingUltrasoundSimulator::simulate(
    const UltrasoundProbe* probe, const SimParams& sim_params) {
  CudaTiming cuda_timing(sim_params.enable_cuda_timing, "Simulation", sim_params.stream);

  // Update the ray gen record
  {
    RayGenSbtRecord rg_sbt{};
    rg_sbt.data.opening_angle = probe->get_opening_angle();
    rg_sbt.data.elevational_height =
        probe->get_num_el_samples() ? probe->get_elevational_height() : 0.f;
    rg_sbt.data.radius = probe->get_radius();
    rg_sbt.data.position = probe->get_pose().position_;
    rg_sbt.data.rotation_matrix = probe->get_pose().rotation_matrix_;

    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group_.get(), &rg_sbt));
    raygen_record_.upload(&rg_sbt, sim_params.stream);
  }

  auto d_scanlines =
      std::make_unique<CudaMemory>(sim_params.buffer_size * probe->get_num_elements() *
                                       probe->get_num_el_samples() * sizeof(float),
                                   sim_params.stream);
  // Initialize scan lines to zero, the algorithm adds reflected and refracted rays in place
  CUDA_CHECK(cudaMemsetAsync(
      d_scanlines->get_ptr(sim_params.stream), 0, d_scanlines->get_size(), sim_params.stream));

  {
    CudaTiming cuda_timing(sim_params.enable_cuda_timing, "OptiX", sim_params.stream);

    //
    // launch
    //
    Params params{};
    params.scanlines = reinterpret_cast<float*>(d_scanlines->get_ptr(sim_params.stream));
    params.buffer_size = sim_params.buffer_size;
    params.t_far = sim_params.t_far;
    params.min_intensity = sim_params.min_intensity;
    params.max_depth = sim_params.max_depth;
    params.materials =
        reinterpret_cast<Material*>(materials_->get_material_data()->get_ptr(sim_params.stream));
    params.background_material_id = materials_->get_index(world_->get_background_material());
    params.scattering_texture = world_->get_scattering_texture();
    params.handle = world_->get_gas_handle();

    pipeline_params_.upload(&params, sim_params.stream);

    OPTIX_CHECK(optixLaunch(pipeline_.get(),
                            sim_params.stream,
                            pipeline_params_.get_device_ptr(sim_params.stream),
                            pipeline_params_.get_size(),
                            &shader_binding_table_,
                            probe->get_num_elements(),
                            probe->get_num_el_samples(),
                            /*depth=*/1));
    CUDA_CHECK(cudaPeekAtLastError());
  }

  const uint2 plane_size = make_uint2(sim_params.buffer_size, probe->get_num_elements());
  const uint3 size = make_uint3(plane_size.x, plane_size.y, probe->get_num_el_samples());

  if (sim_params.write_debug_images) {
    std::filesystem::create_directory("debug_images");
    for (uint32_t plane = 0; plane < size.z; ++plane) {
      write_image(d_scanlines.get(),
                  plane_size,
                  fmt::format("debug_images/0_scanlines{0:03}.png", plane),
                  nullptr /*min_max*/,
                  plane * plane_size.x * plane_size.y * sizeof(float));
    }
  }

  // Process "RF data" into B - mode image

  // 1. PSF Convolution
  if (sim_params.conv_psf) {
    {
      CudaTiming cuda_timing(sim_params.enable_cuda_timing, "PSF Convolution", sim_params.stream);

      update_psfs(probe, sim_params.stream);

      psf_tmp_.resize(d_scanlines->get_size(), sim_params.stream);
      cuda_algorithms_->convolve_rows(
          d_scanlines.get(), size, &psf_tmp_, psf_ax_.get(), sim_params.stream);
      cuda_algorithms_->convolve_columns(
          &psf_tmp_, size, d_scanlines.get(), psf_lat_.get(), sim_params.stream);

      if (probe->get_num_el_samples() > 1) {
        cuda_algorithms_->convolve_planes(
            d_scanlines.get(), size, &psf_tmp_, psf_elev_.get(), sim_params.stream);

        auto d_plane = std::make_unique<CudaMemory>(
            sim_params.buffer_size * probe->get_num_elements() * sizeof(float), sim_params.stream);
        cuda_algorithms_->mean_planes(&psf_tmp_, size, d_plane.get(), sim_params.stream);

        d_scanlines = std::move(d_plane);
      }
    }

    if (sim_params.write_debug_images) {
      write_image(d_scanlines.get(), plane_size, "debug_images/1_psf.png");
    }
  }

  // 1.5 Time-Gain-Compensation
  {
    CudaTiming cuda_timing(
        sim_params.enable_cuda_timing, "Time-Gain-Compensation", sim_params.stream);

    if (!tgc_curve_ || (tgc_curve_->get_size() / sizeof(float) != sim_params.buffer_size)) {
      std::vector<ControlPoint> control_points{{0.f, 0.f}, {40.f, 28.f}};  // (depth [cm], amp [dB])

      tgc_curve_ = create_piece_wise_tgc(
          sim_params.stream, sim_params.buffer_size, control_points, 1540.f, SAMPLING_FREQ);
    }
    cuda_algorithms_->mul_row(d_scanlines.get(), plane_size, tgc_curve_.get(), sim_params.stream);
  }
  if (sim_params.write_debug_images) {
    write_image(d_scanlines.get(), plane_size, "debug_images/2_tgc.png");
  }

  // 2. Envelope detection
  {
    CudaTiming cuda_timing(sim_params.enable_cuda_timing, "Envelope detection", sim_params.stream);

    cuda_algorithms_->hilbert_row(d_scanlines.get(), plane_size, sim_params.stream);
  }
  if (sim_params.write_debug_images) {
    write_image(d_scanlines.get(), plane_size, "debug_images/3_envelope_detection.png");
  }

  // 3. Log compression
  {
    CudaTiming cuda_timing(sim_params.enable_cuda_timing, "Log compression", sim_params.stream);

    cuda_algorithms_->log_compression(
        d_scanlines.get(), plane_size, 20.f, 1e-19f, sim_params.stream);
  }
  if (sim_params.write_debug_images) {
    write_image(d_scanlines.get(), plane_size, "debug_images/4_log_compression.png");
  }

  // 4. Scan conversion
  std::unique_ptr<CudaMemory> b_mode;
  {
    CudaTiming cuda_timing(sim_params.enable_cuda_timing, "Scan conversion", sim_params.stream);

    b_mode = cuda_algorithms_->scan_convert_curvilinear(d_scanlines.get(),
                                                        plane_size,
                                                        probe->get_opening_angle(),
                                                        probe->get_radius(),
                                                        sim_params.t_far + probe->get_radius(),
                                                        sim_params.b_mode_size,
                                                        sim_params.stream,
                                                        sim_params.boundary_value);
  }

  // extract min and max x and z values from Probe dimensions and imaging depth
  // the Image origin is the center of the face of the probe
  // The probe origin is the center of the circle of the probe curvature
  // min_x is the bottom left corner of the image (x is the width)
  // min_z is the bottom center of the image (z is the depth)
  // max_x is the bottom right corner of the image (x is the width)
  // max_z is the top left and right corners of the image that are behind the image origing

  float opening_angle_in_rad = probe->get_opening_angle() * M_PI / 180.0f;
  float min_x = (probe->get_radius() + sim_params.t_far) * std::sin(-opening_angle_in_rad / 2.0f);
  float max_x = (probe->get_radius() + sim_params.t_far) * std::sin(opening_angle_in_rad / 2.0f);
  float z_behind_image_origin = probe->get_radius() * (1 - std::cos(opening_angle_in_rad / 2.0f));
  float min_z = -sim_params.t_far;
  float max_z = z_behind_image_origin;

  // Update the class member variables
  min_x_ = min_x;
  max_x_ = max_x;
  min_z_ = min_z;
  max_z_ = max_z;

  SimResult result;
  result.rf_data = std::move(d_scanlines);
  result.b_mode = std::move(b_mode);

  return result;
}

// Added implementation for generate_sector_mask
std::unique_ptr<CudaMemory> RaytracingUltrasoundSimulator::generate_sector_mask(
    const UltrasoundProbe* probe, float t_far, uint2 output_size,
    float inside_value, float outside_value, cudaStream_t stream) {

  if (!cuda_algorithms_) {
    throw std::runtime_error("RaytracingUltrasoundSimulator::generate_sector_mask: CUDAAlgorithms not initialized.");
  }

  // The 'near' distance is the probe's radius of curvature.
  // The 'far' distance is the probe radius + imaging depth.
  const float near_dist = probe->get_radius();
  const float far_dist = probe->get_radius() + t_far;

  // Add validation checks
   if (far_dist <= 0.f) {
       throw std::runtime_error("RaytracingUltrasoundSimulator::generate_sector_mask: Calculated far distance must be positive.");
   }
   if (far_dist <= near_dist) {
        throw std::runtime_error("RaytracingUltrasoundSimulator::generate_sector_mask: Calculated far distance must be greater than near distance (probe radius).");
   }


  return cuda_algorithms_->generate_sector_mask(output_size,
                                                probe->get_opening_angle(),
                                                near_dist,
                                                far_dist,
                                                inside_value,
                                                outside_value,
                                                stream);
}

}  // namespace raysim
