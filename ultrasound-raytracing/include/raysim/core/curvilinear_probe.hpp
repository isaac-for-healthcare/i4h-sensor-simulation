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

#ifndef CPP_CURVILINEAR_PROBE
#define CPP_CURVILINEAR_PROBE

#include "raysim/core/math_utils.hpp"
#include "raysim/core/probe.hpp"
#include "raysim/core/transform_utils.hpp"

namespace raysim {

/**
 * Curvilinear ultrasound probe implementation.
 * Elements positioned along a curved surface.
 */
class CurvilinearProbe : public BaseProbe {
 public:
  /**
   * Initialize curvilinear probe parameters
   *
   * @param pose Probe pose (position and orientation)
   * @param num_elements_x Number of transducer elements in lateral (x) direction
   * @param sector_angle Field of view in degrees
   * @param radius Radius of curvature in mm
   * @param frequency Center frequency in MHz
   * @param elevational_height Height of elements in elevational direction in mm
   * @param num_el_samples Number of samples in elevational direction
   * @param f_num F-number (focal length / aperture) - unitless
   * @param speed_of_sound Speed of sound in tissue in mm/μs
   * @param pulse_duration Duration of excitation pulse in cycles (number of oscillations)
   */
  explicit CurvilinearProbe(const Pose& pose = Pose(make_float3(0.f, 0.f, 0.f),
                                                    make_float3(0.f, 0.f, 0.f)),
                            uint32_t num_elements_x = 256,
                            float sector_angle = 73.f,       // degrees
                            float radius = 45.f,             // mm
                            float frequency = 2.5f,          // MHz
                            float elevational_height = 7.f,  // mm
                            uint32_t num_el_samples = 1,
                            float f_num = 1.0f,           // unitless
                            float speed_of_sound = 1.54,  // mm/us
                            float pulse_duration = 2.f)   // cycles
      : BaseProbe(pose, num_elements_x, frequency, elevational_height, num_el_samples, f_num,
                  speed_of_sound, pulse_duration),
        sector_angle_(sector_angle),
        radius_(radius) {}

  /**
   * Get element position in local probe coordinates
   *
   * @param element_idx Index of the element
   * @param position Output parameter for element position in local coordinates (mm)
   */
  void get_local_element_position(uint32_t element_idx, float3& position) const override {
    // Use base class utility methods for normalization
    const float norm_x = normalize_element_index_x(element_idx);

    // Calculate angle for this element
    const float angle = normalized_pos_to_angle_rad(norm_x, sector_angle_);

    // Calculate position on arc
    const float x_pos = radius_ * sinf(angle);
    const float z_pos = radius_ * (1.0f - cosf(angle));

    // Position in local coordinates (curved surface)
    position = make_float3(x_pos, 0.0f, z_pos);
  }

  /**
   * Get element ray direction in local probe coordinates
   *
   * @param element_idx Index of the element
   * @param direction Output parameter for element direction in local coordinates (normalized)
   */
  void get_local_element_direction(uint32_t element_idx, float3& direction) const override {
    // Use base class utility methods for normalization
    const float norm_x = normalize_element_index_x(element_idx);

    // Calculate angle in lateral direction
    const float angle_rad = normalized_pos_to_angle_rad(norm_x, sector_angle_);

    // For a curved surface, direction is perpendicular to the surface at that point
    direction = make_float3(sinf(angle_rad), 0.0f, cosf(angle_rad));
    // Direction is already normalized since sin²+cos²=1
  }

  /// Get sector angle (field of view) in degrees
  float get_sector_angle() const { return sector_angle_; }

  /// Set sector angle (field of view) in degrees
  void set_sector_angle(float sector_angle) { sector_angle_ = sector_angle; }

  /// Get radius of curvature in mm
  float get_radius() const { return radius_; }

  /// Set radius of curvature in mm
  void set_radius(float radius) { radius_ = radius; }

  /// Get element spacing (distance between elements) in mm
  float get_element_spacing() const override {
    if (num_elements_x_ <= 1) {
      return 0.0f;  // Or some other sensible default
    }
    // Arc length = radius * angle (in radians)
    return (radius_ * math::deg2rad(sector_angle_)) / (num_elements_x_ - 1);
  }

  ProbeType get_probe_type() const override { return ProbeType::PROBE_TYPE_CURVILINEAR; }

 private:
  float sector_angle_;  ///< Sector angle (field of view) in degrees
  float radius_;        ///< Radius of curvature in mm
};

}  // namespace raysim

#endif /* CPP_CURVILINEAR_PROBE */
