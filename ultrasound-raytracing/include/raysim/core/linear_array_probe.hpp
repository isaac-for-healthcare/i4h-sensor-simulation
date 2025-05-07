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

#ifndef CPP_LINEAR_ARRAY_PROBE
#define CPP_LINEAR_ARRAY_PROBE

#include "raysim/core/math_utils.hpp"
#include "raysim/core/probe.hpp"
#include "raysim/core/transform_utils.hpp"

namespace raysim {

/**
 * Linear array ultrasound probe implementation.
 * Elements positioned in a straight line.
 */
class LinearArrayProbe : public BaseProbe {
 public:
  /**
   * Initialize linear array probe parameters
   *
   * @param pose Probe pose (position and orientation)
   * @param num_elements_x Number of transducer elements in lateral (x) direction
   * @param width Total width of the linear array in mm
   * @param frequency Center frequency in MHz
   * @param elevational_height Height of elements in elevational direction in mm
   * @param num_el_samples Number of samples in elevational direction
   * @param f_num F-number (focal length / aperture) - unitless
   * @param speed_of_sound Speed of sound in tissue in mm/μs
   * @param pulse_duration Duration of excitation pulse in cycles (number of oscillations)
   */
  explicit LinearArrayProbe(const Pose& pose = Pose(make_float3(0.f, 0.f, 0.f),
                                                    make_float3(0.f, 0.f, 0.f)),
                            uint32_t num_elements_x = 256,
                            float width = 60.f,              // mm
                            float frequency = 5.0f,          // MHz
                            float elevational_height = 5.f,  // mm
                            uint32_t num_el_samples = 1,
                            float f_num = 1.0f,           // unitless
                            float speed_of_sound = 1.54,  // mm/us
                            float pulse_duration = 2.f)   // cycles
      : BaseProbe(pose, num_elements_x, frequency, elevational_height, num_el_samples, f_num,
                  speed_of_sound, pulse_duration, width) {}

  ProbeType get_probe_type() const override { return ProbeType::PROBE_TYPE_LINEAR_ARRAY; }
};

}  // namespace raysim

#endif /* CPP_LINEAR_ARRAY_PROBE */
