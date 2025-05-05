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

#ifndef CPP_PROBE_FACTORY
#define CPP_PROBE_FACTORY

#include <memory>
#include <stdexcept>
#include <string>

#include "raysim/core/curvilinear_probe.hpp"
#include "raysim/core/linear_array_probe.hpp"
#include "raysim/core/phased_array_probe.hpp"
#include "raysim/core/probe.hpp"

namespace raysim {

/**
 * Enum for different probe types
 */
enum class ProbeType { CURVILINEAR, LINEAR_ARRAY, PHASED_ARRAY };

/**
 * Factory class for creating different types of ultrasound probes
 */
class ProbeFactory {
 public:
  /**
   * Create a probe of the specified type with default parameters
   *
   * @param type Type of probe to create
   * @param pose Probe pose (position and orientation)
   * @return Shared pointer to the created probe
   */
  static std::shared_ptr<BaseProbe> create_probe(
      ProbeType type,
      const Pose& pose = Pose(make_float3(0.f, 0.f, 0.f), make_float3(0.f, 0.f, 0.f))) {
    switch (type) {
      case ProbeType::CURVILINEAR:
        return std::make_shared<CurvilinearProbe>(pose);
      case ProbeType::LINEAR_ARRAY:
        return std::make_shared<LinearArrayProbe>(pose);
      case ProbeType::PHASED_ARRAY:
        return std::make_shared<PhasedArrayProbe>(pose);
      default:
        throw std::runtime_error("Unknown probe type");
    }
  }

  /**
   * Create a probe from a string name
   *
   * @param type_str String name of the probe type ("curvilinear", "linear", "phased")
   * @param pose Probe pose (position and orientation)
   * @return Shared pointer to the created probe
   */
  static std::shared_ptr<BaseProbe> create_probe_from_string(
      const std::string& type_str,
      const Pose& pose = Pose(make_float3(0.f, 0.f, 0.f), make_float3(0.f, 0.f, 0.f))) {
    if (type_str == "curvilinear") {
      return create_probe(ProbeType::CURVILINEAR, pose);
    } else if (type_str == "linear") {
      return create_probe(ProbeType::LINEAR_ARRAY, pose);
    } else if (type_str == "phased") {
      return create_probe(ProbeType::PHASED_ARRAY, pose);
    } else {
      throw std::runtime_error("Unknown probe type: " + type_str);
    }
  }
};

}  // namespace raysim

#endif /* CPP_PROBE_FACTORY */
