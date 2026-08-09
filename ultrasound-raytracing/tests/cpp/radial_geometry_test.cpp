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

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "raysim/core/radial_geometry.hpp"
#include "raysim/core/radial_probe.hpp"

namespace {

constexpr float TOLERANCE = 1e-5f;

bool near(float actual, float expected, float tolerance = TOLERANCE) {
  return std::fabs(actual - expected) <= tolerance;
}

void require(bool condition, const char* message) {
  if (!condition) { throw std::runtime_error(message); }
}

void require_direction(const raysim::RadialProbe& probe, uint32_t index, float3 expected) {
  float3 direction;
  probe.get_local_element_direction(index, direction);
  require(near(direction.x, expected.x), "unexpected radial direction x component");
  require(near(direction.y, expected.y), "unexpected radial direction y component");
  require(near(direction.z, expected.z), "unexpected radial direction z component");
  require(
      near(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z, 1.0f),
      "radial direction is not normalised");
}

void require_position(const raysim::RadialProbe& probe, uint32_t index, float3 expected) {
  float3 position;
  probe.get_local_element_position(index, position);
  require(near(position.x, expected.x), "unexpected radial origin x component");
  require(near(position.y, expected.y), "unexpected radial origin y component");
  require(near(position.z, expected.z), "unexpected radial origin z component");
}

}  // namespace

int main() {
  try {
    const raysim::RadialProbe probe(raysim::Pose(), 4, 0.0f, 1.0f, 0.04f, 20.0f, 0.5f, 1);

    require_direction(probe, 0, make_float3(0.0f, 0.0f, 1.0f));
    require_direction(probe, 1, make_float3(1.0f, 0.0f, 0.0f));
    require_direction(probe, 2, make_float3(0.0f, 0.0f, -1.0f));
    require_direction(probe, 3, make_float3(-1.0f, 0.0f, 0.0f));

    for (uint32_t index = 0; index < probe.get_num_scanlines(); ++index) {
      float3 origin;
      probe.get_local_element_position(index, origin);
      require(near(origin.x, 0.0f) && near(origin.y, 0.0f) && near(origin.z, 0.0f),
              "radial A-lines do not share the transducer origin");
    }

    require(near(probe.get_a_line_angle(3), 270.0f), "A-line angles are not half-open");
    require(near(probe.get_a_line_timestamp(0), 0.0f), "first timestamp is not frame-relative");
    require(near(probe.get_a_line_timestamp(3), 0.03f), "unexpected sequential timestamp");

    const raysim::RadialProbe nonideal_probe(raysim::Pose(),
                                             4,
                                             0.0f,
                                             1.0f,
                                             0.04f,
                                             20.0f,
                                             0.5f,
                                             1,
                                             1.0f,
                                             1.54f,
                                             2.0f,
                                             0.25f,
                                             30.0f,
                                             raysim::RadialRotationDirection::NEGATIVE);
    require(near(nonideal_probe.get_a_line_angle(1), -90.0f),
            "negative rotation did not reverse the angular sequence");
    require_position(nonideal_probe, 0, make_float3(0.0f, 0.0f, 0.25f));
    require_position(nonideal_probe, 1, make_float3(-0.25f, 0.0f, 0.0f));
    require_direction(nonideal_probe, 0, make_float3(0.0f, 0.5f, 0.8660254f));
    require_direction(nonideal_probe, 1, make_float3(-0.8660254f, 0.5f, 0.0f));
    require(near(nonideal_probe.get_a_line_timestamp(1), 0.01f),
            "rotation direction changed acquisition time order");

    float depth_coordinate;
    float scanline_coordinate;
    require(raysim::radial_scan_coordinates(
                0.0f, 10.0f, 20.0f, 1.0f, 0.0f, depth_coordinate, scanline_coordinate),
            "+z point was rejected by radial scan conversion");
    require(near(depth_coordinate, 0.5f) && near(scanline_coordinate, 0.0f),
            "+z point mapped to the wrong polar coordinate");

    require(raysim::radial_scan_coordinates(
                10.0f, 0.0f, 20.0f, 1.0f, 0.0f, depth_coordinate, scanline_coordinate),
            "+x point was rejected by radial scan conversion");
    require(near(scanline_coordinate, 0.25f), "+x point mapped to the wrong scanline");

    require(raysim::radial_scan_coordinates(
                -10.0f, 0.0f, 20.0f, 1.0f, 0.0f, depth_coordinate, scanline_coordinate),
            "-x point was rejected by radial scan conversion");
    require(near(scanline_coordinate, 0.75f), "-x point mapped to the wrong scanline");

    require(raysim::radial_scan_coordinates(10.0f,
                                            0.0f,
                                            20.0f,
                                            1.0f,
                                            0.0f,
                                            depth_coordinate,
                                            scanline_coordinate,
                                            raysim::RadialRotationDirection::NEGATIVE),
            "+x point was rejected for negative rotation");
    require(near(scanline_coordinate, 0.75f),
            "+x point mapped to the wrong negative-rotation scanline");
    require(raysim::radial_scan_coordinates(-10.0f,
                                            0.0f,
                                            20.0f,
                                            1.0f,
                                            0.0f,
                                            depth_coordinate,
                                            scanline_coordinate,
                                            raysim::RadialRotationDirection::NEGATIVE),
            "-x point was rejected for negative rotation");
    require(near(scanline_coordinate, 0.25f),
            "-x point mapped to the wrong negative-rotation scanline");

    require(!raysim::radial_scan_coordinates(
                0.0f, 0.5f, 20.0f, 1.0f, 0.0f, depth_coordinate, scanline_coordinate),
            "central cut-out was not masked");
    require(!raysim::radial_scan_coordinates(
                0.0f, 21.0f, 20.0f, 1.0f, 0.0f, depth_coordinate, scanline_coordinate),
            "point outside radial range was not masked");

    bool rejected_invalid_direction = false;
    try {
      const raysim::RadialProbe invalid_direction_probe(
          raysim::Pose(),
          4,
          0.0f,
          1.0f,
          0.04f,
          20.0f,
          0.5f,
          1,
          1.0f,
          1.54f,
          2.0f,
          0.0f,
          0.0f,
          static_cast<raysim::RadialRotationDirection>(0));
      (void)invalid_direction_probe;
    } catch (const std::invalid_argument&) { rejected_invalid_direction = true; }
    require(rejected_invalid_direction, "invalid radial rotation direction was accepted");
  } catch (const std::exception& exception) {
    std::cerr << "radial geometry test failed: " << exception.what() << '\n';
    return 1;
  }

  std::cout << "radial geometry test passed\n";
  return 0;
}
