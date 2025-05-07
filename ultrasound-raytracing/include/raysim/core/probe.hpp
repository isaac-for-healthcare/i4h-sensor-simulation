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

#ifndef CPP_PROBE_BASE
#define CPP_PROBE_BASE

#include <vector>

#include "raysim/core/math_utils.hpp"
#include "raysim/core/pose.hpp"
#include "raysim/core/probe_types.hpp"
#include "raysim/core/transform_utils.hpp"
#include "raysim/cuda/cuda_helper.hpp"
#include "raysim/cuda/matrix.hpp"

namespace raysim {

/**
 * Base class for ultrasound probe implementations.
 * Provides common functionality and defines interface for probe-specific methods.
 */
class BaseProbe {
 public:
  /**
   * Initialize base probe parameters
   *
   * @param pose Probe pose (position in mm and orientation in radians)
   * @param num_elements_x Number of transducer elements in lateral (x) direction
   * @param frequency Center frequency in MHz
   * @param elevational_height Height of elements in elevational direction in mm
   * @param num_el_samples Number of samples in elevational direction
   * @param f_num F-number (focal length / aperture) - unitless
   * @param speed_of_sound Speed of sound in tissue in mm/μs
   * @param pulse_duration Duration of excitation pulse in cycles (number of oscillations)
   * @param width Total width of the probe in mm
   */
  explicit BaseProbe(const Pose& pose = Pose(make_float3(0.f, 0.f, 0.f),
                                             make_float3(0.f, 0.f, 0.f)),
                     uint32_t num_elements_x = 128,
                     float frequency = 5.0f,          // MHz
                     float elevational_height = 5.f,  // mm
                     uint32_t num_el_samples = 1,
                     float f_num = 1.0f,           // unitless
                     float speed_of_sound = 1.54,  // mm/us
                     float pulse_duration = 2.f,   // cycles
                     float width = 40.f)           // mm
      : pose_(pose),
        num_elements_x_(num_elements_x),
        frequency_(frequency),
        elevational_height_(elevational_height),
        num_el_samples_(num_el_samples),
        f_num_(f_num),
        speed_of_sound_(speed_of_sound),
        pulse_duration_(pulse_duration),
        width_(width) {}

  virtual ~BaseProbe() = default;

  /**
   * Get element position for a specific element
   *
   * @param element_idx Index of the element
   * @param position Output parameter for element position in world coordinates (mm)
   *
   * Default implementation for flat probes (linear and phased arrays).
   * Curvilinear probes will override this.
   */
  virtual void get_element_position(uint32_t element_idx, float3& position) const {
    // Use base class utility methods for normalization
    const float norm_x = normalize_element_index_x(element_idx);

    // Calculate position in x direction
    const float x_pos = norm_x * width_;

    // Position in local coordinates (flat surface)
    position = make_float3(x_pos, 0.0f, 0.0f);

    // Transform to world coordinates
    position = transform_point(pose_, position);
  }

  /**
   * Get element ray direction for a specific element index
   *
   * @param element_idx Index of the element
   * @param direction Output parameter for element direction in world coordinates (normalized)
   *
   * Default implementation for flat arrays (linear and phased) where all elements
   * face perpendicular to the array surface. Curvilinear probes will override this.
   */
  virtual void get_element_direction(uint32_t element_idx, float3& direction) const {
    // For flat arrays (linear and phased), all elements face forward
    direction = make_float3(0.0f, 0.0f, 1.0f);
    // Transform to world coordinates
    direction = transform_direction(pose_, direction);
  }

  /// Update probe pose (orientation in radians)
  void set_pose(const Pose& new_pose) { pose_ = new_pose; }

  /// Get the current pose
  const Pose& get_pose() const { return pose_; }

  /// Get the total number of transducer elements
  uint32_t get_num_elements() const { return num_elements_x_; }

  /// Set number of transducer elements in x direction (lateral)
  void set_num_elements(uint32_t num_elements) { num_elements_x_ = num_elements; }

  /// Get number of transducer elements in x direction (lateral) - same as get_num_elements() for
  /// compatibility
  uint32_t get_num_elements_x() const { return num_elements_x_; }

  /// Set number of transducer elements in x direction (lateral)
  void set_num_elements_x(uint32_t num_elements_x) { num_elements_x_ = num_elements_x; }

  /// Get center frequency in MHz
  float get_frequency() const { return frequency_; }

  /// Set center frequency in MHz
  void set_frequency(float frequency) { frequency_ = frequency; }

  /// Get speed of sound in tissue in mm/μs
  float get_speed_of_sound() const { return speed_of_sound_; }

  /// Set speed of sound in tissue in mm/μs
  void set_speed_of_sound(float speed_of_sound) { speed_of_sound_ = speed_of_sound; }

  /// Get duration of excitation pulse in cycles (number of oscillations)
  float get_pulse_duration() const { return pulse_duration_; }

  /// Set duration of excitation pulse in cycles (number of oscillations)
  void set_pulse_duration(uint16_t pulse_duration) { pulse_duration_ = pulse_duration; }

  /// Get wavelength in mm
  float get_wave_length() const { return speed_of_sound_ / frequency_; }

  /// Get element spacing (distance between elements) in mm - virtual
  virtual float get_element_spacing() const = 0;

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

  /**
   * Get sector angle in degrees
   * Default implementation returns 0.0
   * Overridden by derived classes that have a sector angle
   * @return Sector angle in degrees
   */
  virtual float get_sector_angle() const { return 0.0f; }

  /**
   * Get radius of curvature in mm
   * Default implementation returns 0.0
   * Overridden by derived classes that have a radius of curvature
   * @return Radius of curvature in mm
   */
  virtual float get_radius() const { return 0.0f; }

  /**
   * Get width of the active aperture of the probe
   *  F
   * Overridden by derived classes that have a meaningful width.
   * @return Width in mm
   */
  virtual float get_width() const { return width_; }

  /**
   * Get the specific type of this probe.
   * @return The ProbeType enum value.
   */
  virtual ProbeType get_probe_type() const = 0;

 protected:
  /**
   * Probe coordinate system convention:
   * - x-axis: lateral direction (along the width of the probe)
   * - y-axis: elevational direction (height of the elements)
   * - z-axis: axial direction (depth into tissue)
   */
  Pose pose_;                 ///< Probe pose (position and orientation)
  uint32_t num_elements_x_;   ///< Number of transducer elements in lateral direction
  float frequency_;           ///< Center frequency in MHz
  float elevational_height_;  ///< Height of elements in elevational direction in mm
  uint32_t num_el_samples_;   ///< Number of samples in elevational direction
  float f_num_;               ///< F-number (focal length / aperture) - unitless
  float speed_of_sound_;      ///< Speed of sound in tissue in mm/μs
  uint16_t pulse_duration_;   ///< Duration of excitation pulse in cycles (number of oscillations)
  float width_;               ///< Total width of the probe in mm

  /**
   * Normalize an element index to range [-0.5, 0.5]
   * @param element_idx Element index (0 to num_elements_x - 1)
   * @return Normalized position in range [-0.5, 0.5]
   */
  float normalize_element_index(uint32_t element_idx) const {
    return normalize_element_index_x(element_idx);
  }

  /**
   * Normalize an element index in x direction to range [-0.5, 0.5]
   * @param x_idx Element index in x direction (0 to num_elements_x - 1)
   * @return Normalized position in range [-0.5, 0.5]
   */
  float normalize_element_index_x(uint32_t x_idx) const {
    return (float)x_idx / (num_elements_x_ - 1) - 0.5f;
  }

  /**
   * Convert a normalized position and sector angle to a steering angle in radians
   * @param normalized_pos Normalized position in range [-0.5, 0.5]
   * @param sector_angle_deg Sector angle in degrees
   * @return Steering angle in radians
   */
  float normalized_pos_to_angle_rad(float normalized_pos, float sector_angle_deg) const {
    // Map normalized position to angle in degrees, then convert to radians
    return math::deg2rad(normalized_pos * sector_angle_deg);
  }
};

}  // namespace raysim

#endif /* CPP_PROBE_BASE */
