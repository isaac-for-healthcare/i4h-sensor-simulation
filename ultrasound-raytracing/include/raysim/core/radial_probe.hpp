/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CPP_RADIAL_PROBE
#define CPP_RADIAL_PROBE

#include <cmath>
#include <stdexcept>
#include <vector>

#include "raysim/core/probe.hpp"
#include "raysim/core/radial_geometry.hpp"

namespace raysim {

/**
 * Mechanical radial probe for IVUS and radial EBUS.
 *
 * The catheter lies along local +y. Scanline i is acquired at
 * theta_i = theta_0 + s 2 pi i / N; positive angles turn from +z towards +x.
 * Pose places the catheter, while emitter offset and beam tilt describe the
 * transducer within it.
 */
class RadialProbe : public BaseProbe {
 public:
  explicit RadialProbe(
      const Pose& pose = Pose(make_float3(0.f, 0.f, 0.f), make_float3(0.f, 0.f, 0.f)),
      uint32_t num_scanlines = 256,
      float start_angle = 0.0f,          // degrees
      float dead_zone_radius = 1.0f,     // mm
      float rotation_period = 1.f / 30,  // seconds
      float frequency = 20.0f,           // MHz
      float elevational_height = 0.5f,   // mm
      uint32_t num_el_samples = 1,
      float f_num = 1.0f,                     // unitless
      float speed_of_sound = 1.54,            // mm/us
      float pulse_duration = 2.f,             // cycles
      float transducer_offset_radius = 0.0f,  // mm
      float beam_tilt = 0.0f,                 // degrees
      RadialRotationDirection rotation_direction = RadialRotationDirection::POSITIVE)
      : BaseProbe(pose, num_scanlines, frequency, elevational_height, num_el_samples, f_num,
                  speed_of_sound, pulse_duration, 0.0f),
        start_angle_(start_angle),
        dead_zone_radius_(dead_zone_radius),
        rotation_period_(rotation_period),
        transducer_offset_radius_(transducer_offset_radius),
        beam_tilt_(beam_tilt),
        rotation_direction_(rotation_direction) {
    validate_num_scanlines(num_scanlines);
    validate_start_angle(start_angle);
    validate_dead_zone_radius(dead_zone_radius);
    validate_rotation_period(rotation_period);
    validate_transducer_offset_radius(transducer_offset_radius);
    validate_beam_tilt(beam_tilt);
    validate_rotation_direction(rotation_direction);
  }

  void get_local_element_position(uint32_t element_idx, float3& position) const override {
    validate_scanline_index(element_idx);
    position = radial_scanline_origin(
        element_idx, num_elements_x_, start_angle_, transducer_offset_radius_, rotation_direction_);
  }

  void get_local_element_direction(uint32_t element_idx, float3& direction) const override {
    validate_scanline_index(element_idx);
    direction = radial_scanline_direction(
        element_idx, num_elements_x_, start_angle_, beam_tilt_, rotation_direction_);
  }

  // A single rotating element has no fixed inter-element pitch.
  float get_element_spacing() const override { return 0.0f; }

  float get_sector_angle() const override { return 360.0f; }

  float get_width() const override { return 0.0f; }

  ProbeType get_probe_type() const override { return ProbeType::PROBE_TYPE_RADIAL; }

  uint32_t get_num_scanlines() const { return num_elements_x_; }

  void set_num_scanlines(uint32_t num_scanlines) {
    validate_num_scanlines(num_scanlines);
    num_elements_x_ = num_scanlines;
  }

  float get_start_angle() const override { return start_angle_; }

  void set_start_angle(float start_angle) {
    validate_start_angle(start_angle);
    start_angle_ = start_angle;
  }

  float get_dead_zone_radius() const override { return dead_zone_radius_; }

  void set_dead_zone_radius(float dead_zone_radius) {
    validate_dead_zone_radius(dead_zone_radius);
    dead_zone_radius_ = dead_zone_radius;
  }

  float get_rotation_period() const { return rotation_period_; }

  void set_rotation_period(float rotation_period) {
    validate_rotation_period(rotation_period);
    rotation_period_ = rotation_period;
  }

  float get_transducer_offset_radius() const override { return transducer_offset_radius_; }

  void set_transducer_offset_radius(float transducer_offset_radius) {
    validate_transducer_offset_radius(transducer_offset_radius);
    transducer_offset_radius_ = transducer_offset_radius;
  }

  float get_beam_tilt() const override { return beam_tilt_; }

  void set_beam_tilt(float beam_tilt) {
    validate_beam_tilt(beam_tilt);
    beam_tilt_ = beam_tilt;
  }

  RadialRotationDirection get_rotation_direction() const override { return rotation_direction_; }

  void set_rotation_direction(RadialRotationDirection rotation_direction) {
    validate_rotation_direction(rotation_direction);
    rotation_direction_ = rotation_direction;
  }

  /** Get A-line angle in degrees, relative to the local +z-axis towards +x. */
  float get_a_line_angle(uint32_t scanline_index) const {
    validate_scanline_index(scanline_index);
    return start_angle_ + radial_rotation_sign(rotation_direction_) * 360.0f *
                              (static_cast<float>(scanline_index) / num_elements_x_);
  }

  /** Get the A-line acquisition timestamp relative to frame start, in seconds. */
  float get_a_line_timestamp(uint32_t scanline_index) const {
    validate_scanline_index(scanline_index);
    return rotation_period_ * (static_cast<float>(scanline_index) / num_elements_x_);
  }

  std::vector<float> get_a_line_timestamps() const { return get_scanline_timestamps(); }

  std::vector<float> get_scanline_timestamps() const override {
    std::vector<float> timestamps(num_elements_x_);
    for (uint32_t scanline_index = 0; scanline_index < num_elements_x_; ++scanline_index) {
      timestamps[scanline_index] = get_a_line_timestamp(scanline_index);
    }
    return timestamps;
  }

 private:
  static void validate_num_scanlines(uint32_t num_scanlines) {
    if (num_scanlines == 0) {
      throw std::invalid_argument("num_scanlines must be greater than zero");
    }
  }

  static void validate_start_angle(float start_angle) {
    if (!std::isfinite(start_angle)) { throw std::invalid_argument("start_angle must be finite"); }
  }

  static void validate_dead_zone_radius(float dead_zone_radius) {
    if (!std::isfinite(dead_zone_radius) || (dead_zone_radius < 0.0f)) {
      throw std::invalid_argument("dead_zone_radius must be finite and non-negative");
    }
  }

  static void validate_rotation_period(float rotation_period) {
    if (!std::isfinite(rotation_period) || (rotation_period <= 0.0f)) {
      throw std::invalid_argument("rotation_period must be finite and greater than zero");
    }
  }

  static void validate_transducer_offset_radius(float transducer_offset_radius) {
    if (!std::isfinite(transducer_offset_radius) || (transducer_offset_radius < 0.0f)) {
      throw std::invalid_argument("transducer_offset_radius must be finite and non-negative");
    }
  }

  static void validate_beam_tilt(float beam_tilt) {
    if (!std::isfinite(beam_tilt) || (beam_tilt <= -90.0f) || (beam_tilt >= 90.0f)) {
      throw std::invalid_argument("beam_tilt must be finite and strictly between -90 and 90");
    }
  }

  static void validate_rotation_direction(RadialRotationDirection rotation_direction) {
    if ((rotation_direction != RadialRotationDirection::POSITIVE) &&
        (rotation_direction != RadialRotationDirection::NEGATIVE)) {
      throw std::invalid_argument("rotation_direction must be POSITIVE or NEGATIVE");
    }
  }

  void validate_scanline_index(uint32_t scanline_index) const {
    if (scanline_index >= num_elements_x_) {
      throw std::out_of_range("scanline index is out of range");
    }
  }

  float start_angle_;
  float dead_zone_radius_;
  float rotation_period_;
  float transducer_offset_radius_;
  float beam_tilt_;
  RadialRotationDirection rotation_direction_;
};

}  // namespace raysim

#endif /* CPP_RADIAL_PROBE */
