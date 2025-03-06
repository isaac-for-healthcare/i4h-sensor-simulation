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

#include "raysim/core/pose.hpp"
#include "raysim/cuda/cuda_helper.hpp"
#include "raysim/cuda/matrix.hpp"

namespace raysim {

class UltrasoundProbe {
 public:
  /**
   * Initialize ultrasound probe parameters
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
  explicit UltrasoundProbe(const Pose& pose, uint32_t num_elements = 4096,
                           float opening_angle = 73.f,      // degrees
                           float radius = 45.f,             // mm
                           float frequency = 2.5f,          // MHz
                           float elevational_height = 7.f,  // mm
                           uint32_t num_el_samples = 1,
                           float f_num = 0.7f,           // unitless
                           float speed_of_sound = 1.54,  // mm/us
                           float pulse_duration = 2.f);
  UltrasoundProbe() = delete;

  /// Update probe pose and transformed geometry
  void set_pose(const Pose& new_pose);

  /// Get the current pose
  const Pose& get_pose() const;

  /// Get number of transducer elements
  uint32_t get_num_elements() const;

  /// Get field of view in degrees
  float get_opening_angle() const;

  /// Get radius of curvature in mm
  float get_radius() const;

  /// Get center frequency in MHz
  float get_frequency() const;

  float get_element_spacing() const;

  float get_elevational_height() const;

  uint32_t get_num_el_samples() const;

  float get_axial_resolution() const;

  float get_lateral_resolution() const;

  float get_wave_length() const;

  float get_elevational_spatial_frequency() const;

 private:
  Pose pose_;
  uint32_t num_elements_;
  float opening_angle_;
  float radius_;
  float frequency_;
  float elevational_height_;
  uint32_t num_el_samples_;
  float f_num_;
  float speed_of_sound_;
  float pulse_duration_;
};

}  // namespace raysim

#endif /* CPP_ULTRASOUND_PROBE */
