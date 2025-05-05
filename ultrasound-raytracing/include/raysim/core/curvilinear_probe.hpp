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
   * @param num_elements Number of transducer elements
   * @param opening_angle Field of view in degrees
   * @param radius Radius of curvature in mm
   * @param frequency Center frequency in MHz
   * @param elevational_height Height of elements in elevational direction in mm
   * @param num_el_samples Number of samples in elevational direction
   * @param f_num F-number (focal length / aperture) - unitless
   * @param speed_of_sound Speed of sound in tissue in mm/μs
   * @param pulse_duration Duration of excitation pulse in cycles
   */
  explicit CurvilinearProbe(const Pose& pose = Pose(make_float3(0.f, 0.f, 0.f),
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
      : BaseProbe(pose, num_elements, frequency, speed_of_sound, pulse_duration),
        opening_angle_(opening_angle),
        radius_(radius),
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
    // Calculate lateral angle based on element index and opening angle
    const float lateral_angle = calculate_lateral_angle(element_idx);

    // Calculate position in local coordinates
    position = make_float3(radius_ * sinf(lateral_angle),  // x = r * sin(θ)
                           0.0f,  // y (elevation would be added later if needed)
                           radius_ * (cosf(lateral_angle) - 1.f)  // z = r * (cos(θ) - 1)
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
    // Calculate lateral angle based on element index and opening angle
    const float lateral_angle = calculate_lateral_angle(element_idx);

    // For curvilinear probe, ray directions point away from center of curvature
    // In local coordinates, center of curvature is at (0, 0, -radius)
    float3 element_pos = make_float3(radius_ * sinf(lateral_angle),         // x = r * sin(θ)
                                     0.0f,                                  // y
                                     radius_ * (cosf(lateral_angle) - 1.f)  // z = r * (cos(θ) - 1)
    );

    float3 center_of_curvature = make_float3(0.0f, 0.0f, -radius_);

    // Direction is from center of curvature to element position
    // Calculate difference manually
    float dx = element_pos.x - center_of_curvature.x;
    float dy = element_pos.y - center_of_curvature.y;
    float dz = element_pos.z - center_of_curvature.z;

    // Normalize manually
    float len = sqrtf(dx * dx + dy * dy + dz * dz);
    direction = make_float3(dx / len, dy / len, dz / len);

    // Transform direction to world coordinates
    direction = transform_direction(pose_, direction);
  }

  /// Get field of view in degrees
  float get_opening_angle() const { return opening_angle_; }

  /// Set field of view in degrees
  void set_opening_angle(float opening_angle) { opening_angle_ = opening_angle; }

  /// Get radius of curvature in mm
  float get_radius() const { return radius_; }

  /// Set radius of curvature in mm
  void set_radius(float radius) { radius_ = radius; }

  /// Get element spacing (arc length between elements) in mm
  float get_element_spacing() const {
    return 2.0f * radius_ * sinf((opening_angle_ * M_PI / 180.0f) / (2.0f * (num_elements_ - 1)));
  }

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
   * Calculate lateral angle for a specific element
   *
   * @param element_idx Index of the element
   * @return Lateral angle in radians
   */
  float calculate_lateral_angle(uint32_t element_idx) const {
    // Convert opening angle from degrees to radians
    const float opening_angle_rad = opening_angle_ * M_PI / 180.0f;

    // Calculate lateral angle based on element index
    // Mapping element_idx from [0, num_elements-1] to [-opening_angle/2, opening_angle/2]
    return ((float)element_idx / (num_elements_ - 1) - 0.5f) * opening_angle_rad;
  }

  float opening_angle_;       ///< Field of view in degrees
  float radius_;              ///< Radius of curvature in mm
  float elevational_height_;  ///< Height of elements in elevational direction in mm
  uint32_t num_el_samples_;   ///< Number of samples in elevational direction
  float f_num_;               ///< F-number (focal length / aperture) - unitless
};

}  // namespace raysim

#endif /* CPP_CURVILINEAR_PROBE */
