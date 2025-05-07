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
   * Get element position for a specific element
   *
   * @param element_idx Index of the element
   * @param position Output parameter for element position in world coordinates (mm)
   */
  void get_element_position(uint32_t element_idx, float3& position) const override {
    // Use base class utility methods for normalization
    const float norm_x = normalize_element_index_x(element_idx);

    // Calculate angle in lateral direction
    const float angle_rad = normalized_pos_to_angle_rad(norm_x, sector_angle_);

    // Position in local coordinates (curved surface in lateral direction)
    const float x_pos = radius_ * sinf(angle_rad);
    const float z_offset = radius_ * (1.f - cosf(angle_rad));

    // Construct final position
    position = make_float3(x_pos, 0.0f, z_offset);

    // Transform to world coordinates
    position = transform_point(pose_, position);
  }

  /**
   * Get element ray direction for a specific element
   *
   * @param element_idx Index of the element
   * @param direction Output parameter for element direction in world coordinates (normalized)
   */
  void get_element_direction(uint32_t element_idx, float3& direction) const override {
    // Use base class utility methods for normalization
    const float norm_x = normalize_element_index_x(element_idx);

    // Calculate angle in lateral direction
    const float angle_rad = normalized_pos_to_angle_rad(norm_x, sector_angle_);

    // For a curved surface, direction is perpendicular to the surface at that point
    direction = make_float3(sinf(angle_rad), 0.0f, cosf(angle_rad));

    // Direction is already normalized since sin²+cos²=1

    // Transform to world coordinates
    direction = transform_direction(pose_, direction);
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
    // Arc length = radius * angle (in radians)
    return (radius_ * math::deg2rad(sector_angle_)) / (num_elements_x_ - 1);
  }

  /// Get opening angle (legacy alias for sector_angle) in degrees
  float get_opening_angle() const { return sector_angle_; }

  ProbeType get_probe_type() const override { return ProbeType::PROBE_TYPE_CURVILINEAR; }

  /**
   * For a curvilinear probe, "width" can be interpreted as the arc length of the active surface.
   * @return The arc length of the probe face in mm.
   */
  float get_width() const override {
    // Arc length = radius * angle_in_radians
    // Ensure num_elements_x_ is at least 1 to avoid division by zero or negative if it were 0.
    // If num_elements_x_ is 1, spacing is effectively the whole arc, but typically it implies
    // multiple elements.
    if (num_elements_x_ > 1) {
      return radius_ * math::deg2rad(sector_angle_);
    } else if (num_elements_x_ == 1) {
      // A single element curvilinear doesn't really have a 'width' in the array sense,
      // but if forced, its angular extent * radius might be considered.
      // For simplicity, let's assume it's small or 0 if num_elements_x is 1 for width calc.
      // Or, more consistently, the concept of element spacing isn't well defined for a single
      // element. The original calculation for sbt_width was element_spacing * num_elements.
      // element_spacing = (radius_ * math::deg2rad(sector_angle_)) / (num_elements_x_ - 1);
      // so if num_elements_x_ == 1, it's div by zero. Let's just return the arc length.
      return radius_ * math::deg2rad(sector_angle_);
    }
    return 0.0f;  // Should not happen if num_elements_x_ is always >= 1
  }

 private:
  float sector_angle_;  ///< Sector angle (field of view) in degrees
  float radius_;        ///< Radius of curvature in mm
};

}  // namespace raysim

#endif /* CPP_CURVILINEAR_PROBE */
