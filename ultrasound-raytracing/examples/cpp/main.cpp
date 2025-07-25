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

#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>

#include <spdlog/spdlog.h>

#include "raysim/core/curvilinear_probe.hpp"
#include "raysim/core/material.hpp"
#include "raysim/core/raytracing_ultrasound_simulator.hpp"
#include "raysim/core/world.hpp"
#include "raysim/core/write_image.hpp"

int main(int argc, char* argv[]) {
  try {
    // Create example scene
    raysim::Materials materials;
    // Create world with default background material "water"
    raysim::World world;

    bool use_body = true;
    bool use_calibration = false;
    if (use_body) {
      // world.add(new raysim::Mesh("mesh/Back_muscles.obj", materials.get_index("muscle")));
      // world.add(new raysim::Mesh("mesh/Body.obj", materials.get_index("fat")));
      // world.add(new raysim::Mesh("mesh/Colon.obj", materials.get_index("liver")));
      // world.add(new raysim::Mesh("mesh/Gallbladder.obj", materials.get_index("liver")));
      // world.add(new raysim::Mesh("mesh/Heart.obj", materials.get_index("liver")));
      // world.add(new raysim::Mesh("mesh/Kidney.obj", materials.get_index("liver")));
      world.add(std::make_unique<raysim::Mesh>("mesh/Liver.obj", materials.get_index("liver")));
      //   world.add(new raysim::Mesh("mesh/Lungs.obj", materials.get_index("water")));
      //   world.add(new raysim::Mesh("mesh/Pancreas.obj", materials.get_index("liver")));
      //   world.add(new raysim::Mesh("mesh/Ribs.obj", materials.get_index("bone")));
      //   world.add(new raysim::Mesh("mesh/Small_bowel.obj", materials.get_index("liver")));
      //   world.add(new raysim::Mesh("mesh/Spine.obj", materials.get_index("bone")));
      //   world.add(new raysim::Mesh("mesh/Spleen.obj", materials.get_index("liver")));
      //   world.add(new raysim::Mesh("mesh/Stomach.obj", materials.get_index("liver")));
      //   world.add(new raysim::Mesh("mesh/Veins.obj", materials.get_index("blood")));
    } else if (use_calibration) {
      world.add(std::make_unique<raysim::Mesh>("mesh/calibration_phantom.obj",
                                               materials.get_index("liver")));
    } else {
      world.add(std::make_unique<raysim::Sphere>(
          float3{0.f, -20.f, 0.f}, 5.f, materials.get_index("fat")));
    }

    auto aabb_min = world.get_aabb_min();
    auto aabb_max = world.get_aabb_max();

    raysim::RaytracingUltrasoundSimulator simulator(&world, &materials);

    raysim::CurvilinearProbe probe;
    probe.set_num_el_samples(1);

    // Place probe above the center of the scene pointing down
    if (use_body) {
      probe.set_pose(raysim::Pose(make_float3(10.f, -145.f, -361.f),  // (x, y, z)
                                  make_float3(-M_PI_2, 0., 0.)));     // (x, y, z)
    } else if (use_calibration) {
      probe.set_pose(raysim::Pose(make_float3(10.f, 10.f, 162.f),
                                  make_float3(M_PI_2, 2 * M_PI_2, 0)));  // (x, y, z)
    }

    // Create output directory
    const std::filesystem::path output_dir("liver_sweep");
    std::filesystem::create_directory(output_dir);

    // Sweep from left to right
    uint32_t N_frames = 100;
    float3 start_rot = make_float3(-M_PI_2, M_PI, 0.f);
    float3 end_rot = start_rot;  // Fixed rotation throughout sweep

    float3 start_pos = make_float3(-30.f, -104.f, -30.f);
    float3 end_pos = make_float3(-30.f, -104.f, 110.f);

    printf("start_pos: %f, %f, %f\n", start_pos.x, start_pos.y, start_pos.z);
    printf("end_pos: %f, %f, %f\n", end_pos.x, end_pos.y, end_pos.z);
    auto start = std::chrono::steady_clock::now();

    for (uint32_t frame = 0; frame < N_frames; ++frame) {
      // Update probe
      float3 pos = start_pos;
      pos.x += (end_pos.x - start_pos.x) / N_frames * frame;
      pos.y += (end_pos.y - start_pos.y) / N_frames * frame;
      pos.z += (end_pos.z - start_pos.z) / N_frames * frame;
      probe.set_pose(raysim::Pose(pos, start_rot));  // Use fixed rotation
      spdlog::info("Current position: ({}, {}, {})", pos.x, pos.y, pos.z);

      // Generate frame
      raysim::RaytracingUltrasoundSimulator::SimParams sim_params;
      sim_params.conv_psf = true;
      // sim_params.enable_cuda_timing = true;
      // sim_params.write_debug_images = true;
      auto result = simulator.simulate(&probe, sim_params);

      if (result.b_mode) {
        // Write frame
        auto min_max = make_float2(-60.f, 0.f);
        raysim::write_image(result.b_mode.get(),
                            sim_params.b_mode_size,
                            output_dir / fmt::format("frame_{0:03}.png", frame),
                            &min_max);
      }
    }

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);

    spdlog::info("Created and saved {} frames in {} ms ({} fps)\n",
                 N_frames,
                 elapsed.count(),
                 static_cast<float>(N_frames) / (static_cast<float>(elapsed.count()) / 1000.F));
  } catch (std::exception& e) {
    spdlog::error("Failed with exception '{}'", e.what());
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
