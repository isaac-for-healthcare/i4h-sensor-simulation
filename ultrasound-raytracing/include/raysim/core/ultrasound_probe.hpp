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

#ifndef CPP_ULTRASOUND_PROBE
#define CPP_ULTRASOUND_PROBE

#include <vector>

#include "raysim/core/curvilinear_probe.hpp"
#include "raysim/cuda/cuda_helper.hpp"
#include "raysim/cuda/matrix.hpp"

namespace raysim {

/**
 * Ultrasound probe class for backward compatibility.
 * This now inherits from CurvilinearProbe to maintain the existing API.
 */
class UltrasoundProbe : public CurvilinearProbe {
 public:
  /**
   * Initialize ultrasound probe parameters.
   * This constructor maps parameters to the CurvilinearProbe constructor.
   *
   * @param pose Probe pose (position and orientation)
   * @param num_elements Number of transducer elements
   * @param opening_angle Field of view in degrees
   * @param radius Radius of curvature in mm
   * @param frequency Center frequency in MHz
   * @param elevational_height: Height of elements in elevational direction in mm
   * @param num_el_samples Number of samples in elevational direction
   * @param f_num F-number (focal length / aperture) - unitless
   * @param speed_of_sound Speed of sound in tissue in mm/μs
   * @param pulse_duration Duration of excitation pulse in cycles
   */
  explicit UltrasoundProbe(const Pose& pose = Pose(make_float3(0.f, 0.f, 0.f),
                                                   make_float3(0.f, 0.f, 0.f)),
                           uint32_t num_elements = 256,
                           float opening_angle = 73.f,      // degrees
                           float radius = 45.f,             // mm
                           float frequency = 2.5f,          // MHz
                           float elevational_height = 7.f,  // mm
                           uint32_t num_el_samples = 1,
                           float f_num = 0.7f,           // unitless
                           float speed_of_sound = 1.54,  // mm/us
                           float pulse_duration = 2.f)
      : CurvilinearProbe(pose, num_elements, opening_angle, radius, frequency, elevational_height,
                         num_el_samples, f_num, speed_of_sound, pulse_duration) {}
};

}  // namespace raysim

#endif /* CPP_ULTRASOUND_PROBE */
