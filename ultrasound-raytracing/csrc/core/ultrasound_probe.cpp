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

#include "raysim/core/ultrasound_probe.hpp"

namespace raysim {

UltrasoundProbe::UltrasoundProbe(const Pose& pose, uint32_t num_elements, float opening_angle,
                                 float radius, float frequency, float elevational_height,
                                 uint32_t num_el_samples, float f_num, float speed_of_sound,
                                 float pulse_duration)
    : pose_(pose),
      num_elements_(num_elements),
      opening_angle_(opening_angle),
      radius_(radius),
      frequency_(frequency),
      elevational_height_(elevational_height),
      num_el_samples_(num_el_samples),
      f_num_(f_num),
      speed_of_sound_(speed_of_sound),
      pulse_duration_(pulse_duration) {}

void UltrasoundProbe::set_pose(const Pose& new_pose) {
  pose_ = new_pose;
}

const Pose& UltrasoundProbe::get_pose() const {
  return pose_;
}

uint32_t UltrasoundProbe::get_num_elements() const {
  return num_elements_;
}

float UltrasoundProbe::get_opening_angle() const {
  return opening_angle_;
}

float UltrasoundProbe::get_radius() const {
  return radius_;
}

float UltrasoundProbe::get_frequency() const {
  return frequency_;
}

float UltrasoundProbe::get_element_spacing() const {
  // calculate the arclength of the probe face
  float arc_len = radius_ * opening_angle_ * M_PI / 180.f;
  return arc_len / num_elements_;  // assuming kerf = 0
}

float UltrasoundProbe::get_elevational_height() const {
  return elevational_height_;
}

uint32_t UltrasoundProbe::get_num_el_samples() const {
  return num_el_samples_;
}

float UltrasoundProbe::get_axial_resolution() const {
  return pulse_duration_ * get_wave_length() / 2.f;  // [mm]
}

float UltrasoundProbe::get_lateral_resolution() const {
  return get_wave_length() * f_num_;  // wl * z / D
}

float UltrasoundProbe::get_wave_length() const {
  return speed_of_sound_ / frequency_;  // [mm]
}

float UltrasoundProbe::get_elevational_spatial_frequency() const {
  return elevational_height_ / num_el_samples_;  // 1 / [mm]
}

}  // namespace raysim
