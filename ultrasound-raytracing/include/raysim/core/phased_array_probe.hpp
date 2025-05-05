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
   * @param num_elements Number of transducer elements
   * @param width Total width of the phased array in mm
   * @param sector_angle Total field of view angle in degrees
   * @param frequency Center frequency in MHz
   * @param elevational_height Height of elements in elevational direction in mm
   * @param num_el_samples Number of samples in elevational direction
   * @param f_num F-number (focal length / aperture) - unitless
   * @param speed_of_sound Speed of sound in tissue in mm/μs
   * @param pulse_duration Duration of excitation pulse in cycles
   */
  explicit PhasedArrayProbe(const Pose& pose = Pose(make_float3(0.f, 0.f, 0.f),
                                                    make_float3(0.f, 0.f, 0.f)),
                            uint32_t num_elements = 128,
                            float width = 20.f,              // mm
                            float sector_angle = 90.f,       // degrees
                            float frequency = 3.5f,          // MHz
                            float elevational_height = 5.f,  // mm
                            uint32_t num_el_samples = 1,
                            float f_num = 1.0f,           // unitless
                            float speed_of_sound = 1.54,  // mm/us
                            float pulse_duration = 2.f)
      : BaseProbe(pose, num_elements, frequency, speed_of_sound, pulse_duration),
        width_(width),
        sector_angle_(sector_angle),
        elevational_height_(elevational_height),
        num_el_samples_(num_el_samples),
        f_num_(f_num) {}

  /**
   * Get element position for a specific element
   *
   * @param element_idx Index of the element
   * @param position Output parameter for element position
   */
  void get_element_position(uint32_t element_idx, float3& position) const override {
    // Map element index to position along phased array
    const float element_spacing = get_element_spacing();
    const float x_offset = (element_idx - (num_elements_ - 1) / 2.0f) * element_spacing;

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
   * @param element_idx Index of the element
   * @param direction Output parameter for element direction
   */
  void get_element_direction(uint32_t element_idx, float3& direction) const override {
    // Calculate steering angle for this element
    const float angle_rad = calculate_steering_angle(element_idx);

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
  float get_element_spacing() const { return width_ / (num_elements_ - 1); }

  /// Get height of elements in elevational direction in mm
  float get_elevational_height() const { return elevational_height_; }

  /// Set height of elements in elevational direction in mm
  void set_elevational_height(float elevational_height) {
    elevational_height_ = elevational_height;
  }

  /// Get number of samples in elevational direction
  uint32_t get_num_el_samples() const { return num_el_samples_; }

  /// Set number of samples in elevational direction
  void set_num_el_samples(uint32_t num_el_samples) { num_el_samples_ = num_el_samples; }

  /// Get F-number (focal length / aperture) - unitless
  float get_f_num() const { return f_num_; }

  /// Set F-number (focal length / aperture) - unitless
  void set_f_num(float f_num) { f_num_ = f_num; }

  /// Get axial resolution in mm
  float get_axial_resolution() const {
    // Axial resolution is approximately half the wavelength
    return get_wave_length() / 2.0f;
  }

  /// Get lateral resolution in mm
  float get_lateral_resolution() const {
    // Lateral resolution is approximately wavelength * f_number
    return get_wave_length() * f_num_;
  }

  /// Get elevational spatial frequency in 1/mm
  float get_elevational_spatial_frequency() const {
    // This is a simplified approximation
    return frequency_ / speed_of_sound_;
  }

 private:
  /**
   * Calculate steering angle for a specific element
   *
   * @param element_idx Index of the element
   * @return Steering angle in radians
   */
  float calculate_steering_angle(uint32_t element_idx) const {
    // Convert sector angle from degrees to radians
    // Use half the sector angle since we're calculating from center
    const float max_angle_rad = (sector_angle_ / 2.0f) * M_PI / 180.0f;

    // Calculate steering angle based on element index
    // Mapping element_idx from [0, num_elements-1] to [-max_angle, max_angle]
    return ((float)element_idx / (num_elements_ - 1) - 0.5f) * 2.0f * max_angle_rad;
  }

  float width_;               ///< Total width of the phased array in mm
  float sector_angle_;        ///< Sector angle (total field of view) in degrees
  float elevational_height_;  ///< Height of elements in elevational direction in mm
  uint32_t num_el_samples_;   ///< Number of samples in elevational direction
  float f_num_;               ///< F-number (focal length / aperture) - unitless
};

}  // namespace raysim

#endif /* CPP_PHASED_ARRAY_PROBE */
