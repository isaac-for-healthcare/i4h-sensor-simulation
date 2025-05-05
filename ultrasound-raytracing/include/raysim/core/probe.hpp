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

#include "raysim/core/pose.hpp"
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
   * @param num_elements Number of transducer elements
   * @param frequency Center frequency in MHz
   * @param speed_of_sound Speed of sound in tissue in mm/μs
   * @param pulse_duration Duration of excitation pulse in cycles
   */
  explicit BaseProbe(const Pose& pose = Pose(make_float3(0.f, 0.f, 0.f),
                                             make_float3(0.f, 0.f, 0.f)),
                     uint32_t num_elements = 256,
                     float frequency = 2.5f,       // MHz
                     float speed_of_sound = 1.54,  // mm/us
                     float pulse_duration = 2.f)
      : pose_(pose),
        num_elements_(num_elements),
        frequency_(frequency),
        speed_of_sound_(speed_of_sound),
        pulse_duration_(pulse_duration) {}

  virtual ~BaseProbe() = default;

  /**
   * Get element position for a specific element index
   *
   * @param element_idx Index of the element
   * @param position Output parameter for element position
   */
  virtual void get_element_position(uint32_t element_idx, float3& position) const = 0;

  /**
   * Get element ray direction for a specific element index
   *
   * @param element_idx Index of the element
   * @param direction Output parameter for element direction
   */
  virtual void get_element_direction(uint32_t element_idx, float3& direction) const = 0;

  /// Update probe pose (orientation in radians)
  void set_pose(const Pose& new_pose) { pose_ = new_pose; }

  /// Get the current pose
  const Pose& get_pose() const { return pose_; }

  /// Get number of transducer elements
  uint32_t get_num_elements() const { return num_elements_; }

  /// Set number of transducer elements
  void set_num_elements(uint32_t num_elements) { num_elements_ = num_elements; }

  /// Get center frequency in MHz
  float get_frequency() const { return frequency_; }

  /// Set center frequency in MHz
  void set_frequency(float frequency) { frequency_ = frequency; }

  /// Get speed of sound in tissue in mm/μs
  float get_speed_of_sound() const { return speed_of_sound_; }

  /// Set speed of sound in tissue in mm/μs
  void set_speed_of_sound(float speed_of_sound) { speed_of_sound_ = speed_of_sound; }

  /// Get duration of excitation pulse in cycles
  float get_pulse_duration() const { return pulse_duration_; }

  /// Set duration of excitation pulse in cycles
  void set_pulse_duration(float pulse_duration) { pulse_duration_ = pulse_duration; }

  /// Get wavelength in mm
  float get_wave_length() const { return speed_of_sound_ / frequency_; }

 protected:
  Pose pose_;              ///< Probe pose (position and orientation)
  uint32_t num_elements_;  ///< Number of transducer elements
  float frequency_;        ///< Center frequency in MHz
  float speed_of_sound_;   ///< Speed of sound in tissue in mm/μs
  float pulse_duration_;   ///< Duration of excitation pulse in cycles
};

}  // namespace raysim

#endif /* CPP_PROBE_BASE */
