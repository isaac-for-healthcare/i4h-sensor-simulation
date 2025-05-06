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

#ifndef CPP_PHASED_ARRAY_PROBE
#define CPP_PHASED_ARRAY_PROBE

#include "raysim/core/math_utils.hpp"
#include "raysim/core/probe.hpp"
#include "raysim/core/transform_utils.hpp"

namespace raysim {

/**
 * Phased array ultrasound probe implementation.
 * Elements positioned in a straight line with steered beams.
 */
class PhasedArrayProbe : public BaseProbe {
 public:
  /**
   * Initialize phased array probe parameters
   *
   * @param pose Probe pose (position and orientation)
   * @param num_elements_x Number of transducer elements in lateral (x) direction
   * @param num_elements_y Number of transducer elements in elevational (y) direction (default 1)
   * @param width Total width of the phased array in mm
   * @param sector_angle Total field of view angle in degrees
   * @param frequency Center frequency in MHz
   * @param elevational_height Height of elements in elevational direction in mm
   * @param num_el_samples Number of samples in elevational direction
   * @param f_num F-number (focal length / aperture) - unitless
   * @param speed_of_sound Speed of sound in tissue in mm/μs
   * @param pulse_duration Duration of excitation pulse in cycles (number of oscillations)
   */
  explicit PhasedArrayProbe(const Pose& pose = Pose(make_float3(0.f, 0.f, 0.f),
                                                    make_float3(0.f, 0.f, 0.f)),
                            uint32_t num_elements_x = 128, uint32_t num_elements_y = 1,
                            float width = 20.f,              // mm
                            float sector_angle = 90.f,       // degrees
                            float frequency = 3.5f,          // MHz
                            float elevational_height = 5.f,  // mm
                            uint32_t num_el_samples = 1,
                            float f_num = 1.0f,           // unitless
                            float speed_of_sound = 1.54,  // mm/us
                            float pulse_duration = 2.f)   // cycles
      : BaseProbe(pose, num_elements_x, num_elements_y, frequency, elevational_height,
                  num_el_samples, f_num, speed_of_sound, pulse_duration),
        width_(width),
        sector_angle_(sector_angle) {}

  /**
   * Get element position for a specific element
   *
   * @param element_idx Index of the element (in x direction)
   * @param position Output parameter for element position in world coordinates (mm)
   */
  void get_element_position(uint32_t element_idx, float3& position) const override {
    // Map element index to position along phased array
    const float element_spacing = get_element_spacing();
    const float x_offset = (element_idx - (num_elements_x_ - 1) / 2.0f) * element_spacing;

    // Position in local coordinates (x along array, z into tissue)
    position = make_float3(x_offset,  // x position along array
                           0.0f,      // y (elevation would be added later if needed)
                           0.0f       // z at surface (face of the probe)
    );

    // Transform to world coordinates
    position = transform_point(pose_, position);
  }

  /**
   * Get element ray direction for a specific element
   *
   * @param element_idx Index of the element (in x direction)
   * @param direction Output parameter for element direction in world coordinates (normalized)
   */
  void get_element_direction(uint32_t element_idx, float3& direction) const override {
    // Calculate steering angle for this element
    const float angle_rad = element_idx_to_steering_angle_rad(element_idx);

    // Direction in local coordinates
    direction = make_float3(sinf(angle_rad),  // x component based on steering angle
                            0.0f,             // y component (no elevation angle)
                            cosf(angle_rad)   // z component based on steering angle
    );

    // Transform direction to world coordinates
    direction = transform_direction(pose_, direction);
  }

  /// Get total width of the phased array in mm
  float get_width() const { return width_; }

  /// Set total width of the phased array in mm
  void set_width(float width) { width_ = width; }

  /// Get sector angle (total field of view) in degrees
  float get_sector_angle() const { return sector_angle_; }

  /// Set sector angle (total field of view) in degrees
  void set_sector_angle(float sector_angle) { sector_angle_ = sector_angle; }

  /// Get element spacing (distance between elements) in mm
  float get_element_spacing() const override { return width_ / (num_elements_x_ - 1); }

 private:
  /**
   * Calculate steering angle for a specific element index
   *
   * @param element_idx Index of the element (in x direction)
   * @return Steering angle in radians
   */
  float element_idx_to_steering_angle_rad(uint32_t element_idx) const {
    // Use base class utility method for normalization
    const float normalized_pos = normalize_element_index(element_idx);

    // Half the sector angle (symmetrical sector around the center)
    const float half_sector_angle = sector_angle_ / 2.0f;

    // Calculate and return the steering angle in radians
    return normalized_pos_to_angle_rad(normalized_pos * 2.0f, half_sector_angle);
  }

  float width_;         ///< Total width of the phased array in mm
  float sector_angle_;  ///< Sector angle (total field of view) in degrees
};

}  // namespace raysim

#endif /* CPP_PHASED_ARRAY_PROBE */
