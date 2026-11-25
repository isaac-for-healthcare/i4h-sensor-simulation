/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#include "ultrasound_simulator/core/curvilinear_probe.hpp"
#include "ultrasound_simulator/core/hitable.hpp"
#include "ultrasound_simulator/core/linear_array_probe.hpp"
#include "ultrasound_simulator/core/material.hpp"
#include "ultrasound_simulator/core/phased_array_probe.hpp"
#include "ultrasound_simulator/core/radial_probe.hpp"
#include "ultrasound_simulator/core/raytracing_ultrasound_simulator.hpp"
#include "ultrasound_simulator/core/world.hpp"
#include "ultrasound_simulator/core/write_image.hpp"
#include "ultrasound_simulator/cuda/cuda_helper.hpp"

namespace py = pybind11;

// Helper function to convert numpy array to float3
float3 numpy_to_float3(py::array_t<float> array) {
  auto buf = array.request();
  if (buf.ndim != 1 || buf.shape[0] != 3) {
    throw std::runtime_error("numpy array must have shape (3,)");
  }
  float* ptr = static_cast<float*>(buf.ptr);
  return make_float3(ptr[0], ptr[1], ptr[2]);
}

// Helper function to convert float3 to numpy array
py::array_t<float> float3_to_numpy(float3 vec) {
  auto result = py::array_t<float>(3);
  auto buf = result.request();
  float* ptr = static_cast<float*>(buf.ptr);
  ptr[0] = vec.x;
  ptr[1] = vec.y;
  ptr[2] = vec.z;
  return result;
}

py::array_t<float> cuda_memory_to_numpy(ultrasound_simulator::CudaMemory& memory,
                                        const std::vector<ssize_t>& shape, cudaStream_t stream) {
  size_t num_elements = 1;
  for (const ssize_t dimension : shape) {
    if (dimension < 0) { throw std::runtime_error("numpy shape dimensions must be non-negative"); }
    num_elements *= static_cast<size_t>(dimension);
  }
  if (memory.get_size() != num_elements * sizeof(float)) {
    throw std::runtime_error("CUDA buffer size does not match requested numpy shape");
  }

  auto array = py::array_t<float>(shape);
  memory.download(array.mutable_data(), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return array;
}

py::array_t<float> vector_to_numpy(const std::vector<float>& values) {
  auto array = py::array_t<float>(values.size());
  std::copy(values.begin(), values.end(), array.mutable_data());
  return array;
}

py::array make_c_contiguous(const py::array& array, const char* name) {
  auto contiguous = py::array::ensure(array, py::array::c_style);
  if (!contiguous) {
    throw py::value_error(std::string(name) + " could not be copied to a C-contiguous array");
  }
  return contiguous;
}

template <typename T>
T read_array_value(const void* data, size_t index) {
  T value;
  std::memcpy(&value, static_cast<const char*>(data) + index * sizeof(T), sizeof(T));
  return value;
}

std::vector<float3> numpy_to_float3_vector(const py::array& array, const char* name,
                                           ssize_t expected_rows = -1) {
  if (!array.dtype().is(py::dtype::of<float>())) {
    throw py::value_error(std::string(name) + " must have dtype float32");
  }
  if (array.ndim() != 2 || array.shape(1) != 3) {
    throw py::value_error(std::string(name) + " must have shape (N, 3)");
  }
  if (expected_rows >= 0 && array.shape(0) != expected_rows) {
    throw py::value_error(std::string(name) + " must have the same number of rows as vertices");
  }

  auto contiguous = make_c_contiguous(array, name);
  const auto buffer = contiguous.request();
  std::vector<float3> result;
  result.reserve(static_cast<size_t>(buffer.shape[0]));
  for (ssize_t row = 0; row < buffer.shape[0]; ++row) {
    const auto offset = static_cast<size_t>(3 * row);
    result.push_back(make_float3(read_array_value<float>(buffer.ptr, offset),
                                 read_array_value<float>(buffer.ptr, offset + 1),
                                 read_array_value<float>(buffer.ptr, offset + 2)));
  }
  return result;
}

std::vector<uint32_t> numpy_to_triangle_indices(const py::array& array) {
  if (array.ndim() != 2 || array.shape(1) != 3) {
    throw py::value_error("indices must have shape (M, 3)");
  }

  const bool is_uint32 = array.dtype().is(py::dtype::of<uint32_t>());
  const bool is_int32 = array.dtype().is(py::dtype::of<int32_t>());
  const bool is_int64 = array.dtype().is(py::dtype::of<int64_t>());
  if (!is_uint32 && !is_int32 && !is_int64) {
    throw py::value_error("indices must have dtype uint32, int32, or int64");
  }

  auto contiguous = make_c_contiguous(array, "indices");
  const auto buffer = contiguous.request();
  const auto count = static_cast<size_t>(buffer.shape[0]) * 3;
  std::vector<uint32_t> result;
  result.reserve(count);

  if (is_uint32) {
    for (size_t index = 0; index < count; ++index) {
      result.push_back(read_array_value<uint32_t>(buffer.ptr, index));
    }
    return result;
  }

  if (is_int32) {
    for (size_t index = 0; index < count; ++index) {
      const int32_t value = read_array_value<int32_t>(buffer.ptr, index);
      if (value < 0) { throw py::value_error("indices must contain only non-negative values"); }
      result.push_back(static_cast<uint32_t>(value));
    }
    return result;
  }

  for (size_t index = 0; index < count; ++index) {
    const int64_t value = read_array_value<int64_t>(buffer.ptr, index);
    if (value < 0) { throw py::value_error("indices must contain only non-negative values"); }
    if (value > std::numeric_limits<uint32_t>::max()) {
      throw py::value_error("indices values must fit in uint32");
    }
    result.push_back(static_cast<uint32_t>(value));
  }
  return result;
}

std::unique_ptr<ultrasound_simulator::Mesh> mesh_from_numpy(const py::array& vertices,
                                                            const py::array& indices,
                                                            const py::object& normals,
                                                            uint32_t material_id) {
  auto mesh_vertices = numpy_to_float3_vector(vertices, "vertices");
  auto mesh_indices = numpy_to_triangle_indices(indices);
  std::vector<float3> mesh_normals;
  if (!normals.is_none()) {
    if (!py::isinstance<py::array>(normals)) {
      throw py::value_error("normals must be a NumPy array or None");
    }
    mesh_normals = numpy_to_float3_vector(
        py::reinterpret_borrow<py::array>(normals), "normals", vertices.shape(0));
  }
  return std::make_unique<ultrasound_simulator::Mesh>(
      std::move(mesh_vertices), std::move(mesh_indices), std::move(mesh_normals), material_id);
}

PYBIND11_MODULE(ray_sim_python, m) {
  m.doc() = R"pbdoc(
        Raytracing-based ultrasound simulator

        This module provides a CUDA-accelerated ultrasound simulation framework.
        It allows for realistic simulation of ultrasound imaging by raytracing through
        3D meshes with different material properties.

        Key Classes:
            - Materials: Manages acoustic material properties
            - World: Contains the 3D scene geometry
            - Pose: Represents 3D position and orientation
            - CurvilinearProbe, LinearArrayProbe, PhasedArrayProbe, RadialProbe: Define probe parameters
            - RaytracingUltrasoundSimulator: Main simulation engine
    )pbdoc";

  py::enum_<ultrasound_simulator::RadialRotationDirection>(m, "RadialRotationDirection", R"pbdoc(
Order in which a mechanical radial probe acquires A-lines.

POSITIVE turns from local +z towards +x; NEGATIVE takes the same angles in reverse.
)pbdoc")
      .value("NEGATIVE", ultrasound_simulator::RadialRotationDirection::NEGATIVE)
      .value("POSITIVE", ultrasound_simulator::RadialRotationDirection::POSITIVE);

  // Bind Pose class
  py::class_<ultrasound_simulator::Pose>(m, "Pose", R"pbdoc(
        Represents a 3D pose with position and rotation.

        Args:
            position (array-like): 3D position vector [x, y, z]. Can be a list or numpy array.
            rotation (array-like): 3D rotation vector [roll, pitch, yaw] in radians. Can be a list or numpy array.

        Examples:
            Using numpy arrays:
            >>> pose = Pose(position=np.array([0, 0, 0]), rotation=np.array([0, np.pi, 0]))

            Using lists:
            >>> pose = Pose([0, 0, 0], [0, np.pi, 0])

            Using keyword arguments with lists:
            >>> pose = Pose(position=[0, 0, 0], rotation=[0, np.pi, 0])
    )pbdoc")
      .def(py::init([](py::array_t<float> position, py::array_t<float> rotation) {
             return std::make_unique<ultrasound_simulator::Pose>(numpy_to_float3(position),
                                                                 numpy_to_float3(rotation));
           }),
           py::arg("position"),
           py::arg("rotation"),
           R"pbdoc(
        Initialize a new Pose.

        Args:
            position (np.ndarray): 3D position vector [x, y, z]
            rotation (np.ndarray): 3D rotation vector [roll, pitch, yaw] in radians
      )pbdoc")
      .def_property(
          "position",
          [](const ultrasound_simulator::Pose& self) { return float3_to_numpy(self.position_); },
          [](ultrasound_simulator::Pose& self, py::array_t<float> position) {
            self.position_ = numpy_to_float3(position);
          },
          "3D position vector [x, y, z]")
      .def_property(
          "rotation",
          [](const ultrasound_simulator::Pose& self) { return float3_to_numpy(self.rotation_); },
          [](ultrasound_simulator::Pose& self, py::array_t<float> rotation) {
            self.rotation_ = numpy_to_float3(rotation);
          },
          "3D rotation vector [roll, pitch, yaw] in radians");

  // Bind Material class
  py::class_<ultrasound_simulator::Material>(m, "Material", R"pbdoc(
        Represents acoustic material properties for ultrasound simulation.

        Properties:
            - impedance: Acoustic impedance (MRayl)
            - attenuation: Attenuation coefficient (dB/cm/MHz)
            - speed_of_sound: Speed of sound in material (m/s)
            - mu0: Scattering coefficient
            - mu1: Scattering coefficient
            - sigma: Surface roughness
            - specularity: Specularity coefficient
    )pbdoc")
      .def(py::init<float, float, float, float, float, float, float>(),
           py::arg("impedance"),
           py::arg("attenuation"),
           py::arg("speed_of_sound"),
           py::arg("mu0") = 0.f,
           py::arg("mu1") = 0.f,
           py::arg("sigma") = 0.f,
           py::arg("specularity") = 1.f)
      .def_readwrite(
          "impedance_", &ultrasound_simulator::Material::impedance_, "Acoustic impedance (MRayl)")
      .def_readwrite("attenuation_",
                     &ultrasound_simulator::Material::attenuation_,
                     "Attenuation coefficient (dB/cm/MHz)")
      .def_readwrite("speed_of_sound_",
                     &ultrasound_simulator::Material::speed_of_sound_,
                     "Speed of sound (m/s)")
      .def_readwrite("mu0_", &ultrasound_simulator::Material::mu0_, "Scattering coefficient mu0")
      .def_readwrite("mu1_", &ultrasound_simulator::Material::mu1_, "Scattering coefficient mu1")
      .def_readwrite("sigma_", &ultrasound_simulator::Material::sigma_, "Surface roughness")
      .def_readwrite(
          "specularity_", &ultrasound_simulator::Material::specularity_, "Specularity coefficient")
      .def("density",
           &ultrasound_simulator::Material::density,
           "Calculate material density from impedance");

  // Bind Materials class
  py::class_<ultrasound_simulator::Materials>(m, "Materials", R"pbdoc(
        Manages a collection of acoustic materials for simulation.

        Provides predefined materials like:
            - water
            - fat
            - muscle
            - liver
            - bone
            - blood
    )pbdoc")
      .def(py::init<>())
      .def("get_index", &ultrasound_simulator::Materials::get_index, R"pbdoc(
        Get the index of a predefined material.

        Args:
            name (str): Material name (e.g., "water", "fat", "muscle", "liver", "bone", "blood")

        Returns:
            int: Material index for use in World objects
      )pbdoc");

  // Bind World class
  py::class_<ultrasound_simulator::World>(m, "World", R"pbdoc(
        Represents the 3D scene for ultrasound simulation.

        Contains meshes and their material properties. The world has a background
        material (typically "water") and can contain multiple objects with different
        materials.
    )pbdoc")
      .def(py::init([](const std::string& background_material) {
             spdlog::info("Creating World with background material: {}", background_material);
             try {
               auto world = std::make_unique<ultrasound_simulator::World>(background_material);
               spdlog::info("World created successfully");
               return world;
             } catch (const std::exception& e) {
               spdlog::error("Failed to create World: {}", e.what());
               throw;
             }
           }),
           py::arg("background_material") = "water",
           R"pbdoc(
        Initialize a new World.

        Args:
            background_material (str): Name of the background material (default: "water")
      )pbdoc")
      .def(
          "add",
          [](ultrasound_simulator::World& self, py::object obj) {
            spdlog::info("Adding object to world");
            try {
              auto* hitable_ptr = obj.cast<ultrasound_simulator::Hitable*>();
              auto unique_hitable = std::unique_ptr<ultrasound_simulator::Hitable>(hitable_ptr);
              self.add(std::move(unique_hitable));
              obj.release();
              spdlog::info("Object added successfully");
            } catch (const std::exception& e) {
              spdlog::error("Failed to add object to world: {}", e.what());
              throw;
            }
          },
          R"pbdoc(
        Add an object (Mesh or Sphere) to the world.
        Note: The world takes ownership of the object.
      )pbdoc")
      .def("get_background_material",
           &ultrasound_simulator::World::get_background_material,
           "Get the background material name")
      .def(
          "get_aabb_min",
          [](const ultrasound_simulator::World& self) {
            auto min = self.get_aabb_min();
            return float3_to_numpy(min);
          },
          "Get the minimum point of the world's bounding box")
      .def(
          "get_aabb_max",
          [](const ultrasound_simulator::World& self) {
            auto max = self.get_aabb_max();
            return float3_to_numpy(max);
          },
          "Get the maximum point of the world's bounding box");

  // Bind BaseProbe class and derived classes
  py::class_<ultrasound_simulator::BaseProbe>(m, "BaseProbe", R"pbdoc(
Base class for ultrasound probes.

All probe types derive from this class and inherit common functionality such as:
- Position and orientation (pose) handling
- Element position and ray direction calculation
- Basic acoustic parameters
- Support for 1D and 2D (matrix) arrays

Note: Pose orientation is specified in radians (roll, pitch, yaw).
)pbdoc")
      .def("set_pose", &ultrasound_simulator::BaseProbe::set_pose, "Set the probe's pose")
      .def("get_pose", &ultrasound_simulator::BaseProbe::get_pose, "Get the probe's current pose")
      .def("get_num_elements_x",
           &ultrasound_simulator::BaseProbe::get_num_elements_x,
           "Get number of transducer elements in lateral (x) direction")
      .def("set_num_elements_x",
           &ultrasound_simulator::BaseProbe::set_num_elements_x,
           "Set number of transducer elements in lateral (x) direction")
      .def("get_frequency",
           &ultrasound_simulator::BaseProbe::get_frequency,
           "Get probe frequency (MHz)")
      .def("set_frequency",
           &ultrasound_simulator::BaseProbe::set_frequency,
           "Set probe frequency (MHz)")
      .def("get_speed_of_sound",
           &ultrasound_simulator::BaseProbe::get_speed_of_sound,
           "Get speed of sound (mm/μs)")
      .def("set_speed_of_sound",
           &ultrasound_simulator::BaseProbe::set_speed_of_sound,
           "Set speed of sound (mm/μs)")
      .def("get_pulse_duration",
           &ultrasound_simulator::BaseProbe::get_pulse_duration,
           "Get pulse duration (cycles)")
      .def("set_pulse_duration",
           &ultrasound_simulator::BaseProbe::set_pulse_duration,
           "Set pulse duration (cycles)")
      .def("get_wave_length", &ultrasound_simulator::BaseProbe::get_wave_length, "Get wavelength")
      .def(
          "get_element_position",
          [](const ultrasound_simulator::BaseProbe& self, uint32_t element_idx) {
            float3 position;
            self.get_element_position(element_idx, position);
            return float3_to_numpy(position);
          },
          "Get position of an element by index")
      .def(
          "get_element_direction",
          [](const ultrasound_simulator::BaseProbe& self, uint32_t element_idx) {
            float3 direction;
            self.get_element_direction(element_idx, direction);
            return float3_to_numpy(direction);
          },
          "Get direction for an element by index")
      .def("get_scanline_timestamps",
           &ultrasound_simulator::BaseProbe::get_scanline_timestamps,
           "Get scanline acquisition timestamps relative to frame start [seconds]");

  py::class_<ultrasound_simulator::CurvilinearProbe, ultrasound_simulator::BaseProbe>(
      m, "CurvilinearProbe", R"pbdoc(
Curvilinear ultrasound probe with elements positioned along a curved surface.
)pbdoc")
      .def(py::init<const ultrasound_simulator::Pose&,
                    uint32_t,
                    float,
                    float,
                    float,
                    float,
                    uint32_t,
                    float,
                    float,
                    float>(),
           py::arg("pose") = ultrasound_simulator::Pose(),
           py::arg("num_elements_x") = 256,
           py::arg("sector_angle") = 73.0f,
           py::arg("radius") = 45.0f,
           py::arg("frequency") = 2.5f,
           py::arg("elevational_height") = 7.0f,
           py::arg("num_el_samples") = 1,
           py::arg("f_num") = 0.7f,
           py::arg("speed_of_sound") = 1.54f,
           py::arg("pulse_duration") = 2.0f);

  py::class_<ultrasound_simulator::LinearArrayProbe, ultrasound_simulator::BaseProbe>(
      m, "LinearArrayProbe", R"pbdoc(
Linear array ultrasound probe with elements positioned in a straight line.
Elements emit parallel beams perpendicular to the face of the probe.
)pbdoc")
      .def(py::init<const ultrasound_simulator::Pose&,
                    uint32_t,
                    float,
                    float,
                    float,
                    uint32_t,
                    float,
                    float,
                    float>(),
           py::arg("pose") = ultrasound_simulator::Pose(),
           py::arg("num_elements_x") = 256,
           py::arg("width") = 60.0f,
           py::arg("frequency") = 5.0f,
           py::arg("elevational_height") = 5.0f,
           py::arg("num_el_samples") = 1,
           py::arg("f_num") = 1.0f,
           py::arg("speed_of_sound") = 1.54f,
           py::arg("pulse_duration") = 2.0f);

  py::class_<ultrasound_simulator::PhasedArrayProbe, ultrasound_simulator::BaseProbe>(
      m, "PhasedArrayProbe", R"pbdoc(
Phased array ultrasound probe with elements positioned in a straight line.
Elements steer beams electronically to create a sector image from a small footprint.
)pbdoc")
      .def(py::init<const ultrasound_simulator::Pose&,
                    uint32_t,
                    float,
                    float,
                    float,
                    float,
                    uint32_t,
                    float,
                    float,
                    float>(),
           py::arg("pose") = ultrasound_simulator::Pose(),
           py::arg("num_elements_x") = 128,
           py::arg("width") = 20.0f,
           py::arg("sector_angle") = 90.0f,
           py::arg("frequency") = 3.5f,
           py::arg("elevational_height") = 5.0f,
           py::arg("num_el_samples") = 1,
           py::arg("f_num") = 1.0f,
           py::arg("speed_of_sound") = 1.54f,
           py::arg("pulse_duration") = 2.0f);

  py::class_<ultrasound_simulator::RadialProbe, ultrasound_simulator::BaseProbe>(
      m, "RadialProbe", R"pbdoc(
Mechanical radial probe for IVUS and radial EBUS.

The catheter lies along local +y. Scanline i is acquired at
theta_i = theta_0 + s 2 pi i / N; positive angles turn from +z towards +x.
Pose places the catheter, while emitter offset and beam tilt describe the
transducer within it.
)pbdoc")
      .def(py::init<const ultrasound_simulator::Pose&,
                    uint32_t,
                    float,
                    float,
                    float,
                    float,
                    float,
                    uint32_t,
                    float,
                    float,
                    float,
                    float,
                    float,
                    ultrasound_simulator::RadialRotationDirection>(),
           py::arg("pose") = ultrasound_simulator::Pose(),
           py::arg("num_scanlines") = 256,
           py::arg("start_angle") = 0.0f,
           py::arg("dead_zone_radius") = 1.0f,
           py::arg("rotation_period") = 1.0f / 30.0f,
           py::arg("frequency") = 20.0f,
           py::arg("elevational_height") = 0.5f,
           py::arg("num_el_samples") = 1,
           py::arg("f_num") = 1.0f,
           py::arg("speed_of_sound") = 1.54f,
           py::arg("pulse_duration") = 2.0f,
           py::arg("transducer_offset_radius") = 0.0f,
           py::arg("beam_tilt") = 0.0f,
           py::arg("rotation_direction") = ultrasound_simulator::RadialRotationDirection::POSITIVE)
      .def("get_num_scanlines",
           &ultrasound_simulator::RadialProbe::get_num_scanlines,
           "Get the number of A-lines per revolution")
      .def("set_num_scanlines",
           &ultrasound_simulator::RadialProbe::set_num_scanlines,
           "Set the number of A-lines per revolution")
      .def("get_start_angle",
           &ultrasound_simulator::RadialProbe::get_start_angle,
           "Get the direction of A-line zero [degrees]")
      .def("set_start_angle",
           &ultrasound_simulator::RadialProbe::set_start_angle,
           "Set the direction of A-line zero [degrees]")
      .def("get_dead_zone_radius",
           &ultrasound_simulator::RadialProbe::get_dead_zone_radius,
           "Get the central cut-out radius [mm]")
      .def("set_dead_zone_radius",
           &ultrasound_simulator::RadialProbe::set_dead_zone_radius,
           "Set the central cut-out radius [mm]")
      .def("get_rotation_period",
           &ultrasound_simulator::RadialProbe::get_rotation_period,
           "Get the full-revolution acquisition period [seconds]")
      .def("set_rotation_period",
           &ultrasound_simulator::RadialProbe::set_rotation_period,
           "Set the full-revolution acquisition period [seconds]")
      .def("get_transducer_offset_radius",
           &ultrasound_simulator::RadialProbe::get_transducer_offset_radius,
           "Get the radial offset between the emitter and catheter axis [mm]")
      .def("set_transducer_offset_radius",
           &ultrasound_simulator::RadialProbe::set_transducer_offset_radius,
           "Set the radial offset between the emitter and catheter axis [mm]")
      .def("get_beam_tilt",
           &ultrasound_simulator::RadialProbe::get_beam_tilt,
           "Get the signed beam tilt from the transverse plane [degrees]")
      .def("set_beam_tilt",
           &ultrasound_simulator::RadialProbe::set_beam_tilt,
           "Set the signed beam tilt from the transverse plane [degrees]")
      .def("get_rotation_direction",
           &ultrasound_simulator::RadialProbe::get_rotation_direction,
           "Get the radial acquisition direction")
      .def("set_rotation_direction",
           &ultrasound_simulator::RadialProbe::set_rotation_direction,
           "Set the radial acquisition direction")
      .def("get_a_line_angle",
           &ultrasound_simulator::RadialProbe::get_a_line_angle,
           py::arg("scanline_index"),
           "Get an A-line angle [degrees]")
      .def("get_a_line_timestamp",
           &ultrasound_simulator::RadialProbe::get_a_line_timestamp,
           py::arg("scanline_index"),
           "Get an A-line timestamp relative to frame start [seconds]")
      .def("get_a_line_timestamps",
           &ultrasound_simulator::RadialProbe::get_a_line_timestamps,
           "Get all A-line timestamps relative to frame start [seconds]");

  // Bind SimParams struct
  py::class_<ultrasound_simulator::RaytracingUltrasoundSimulator::SimParams>(
      m, "SimParams", R"pbdoc(
        Simulation parameters for ultrasound imaging.

        Parameters:
            - t_far: Maximum imaging depth
            - buffer_size: Ray buffer size
            - max_depth: Maximum ray reflection depth
            - min_intensity: Minimum ray intensity threshold
            - use_scattering: Enable scattering simulation
            - conv_psf: Enable PSF convolution
            - median_clip_filter: Enable median clip filter for speckle reduction
            - enable_cuda_timing: Enable CUDA timing measurements
            - write_debug_images: Enable debug image output
            - b_mode_size: B-mode image size as (width, height), accepts tuple, list, or numpy array
    )pbdoc")
      .def(py::init<>())
      .def_readwrite("t_far",
                     &ultrasound_simulator::RaytracingUltrasoundSimulator::SimParams::t_far,
                     "Maximum imaging depth")
      .def_readwrite("buffer_size",
                     &ultrasound_simulator::RaytracingUltrasoundSimulator::SimParams::buffer_size,
                     "Ray buffer size")
      .def_readwrite("max_depth",
                     &ultrasound_simulator::RaytracingUltrasoundSimulator::SimParams::max_depth,
                     "Maximum ray reflection depth")
      .def_readwrite("min_intensity",
                     &ultrasound_simulator::RaytracingUltrasoundSimulator::SimParams::min_intensity,
                     "Minimum ray intensity threshold")
      .def_readwrite(
          "use_scattering",
          &ultrasound_simulator::RaytracingUltrasoundSimulator::SimParams::use_scattering,
          "Enable scattering simulation")
      .def_readwrite("conv_psf",
                     &ultrasound_simulator::RaytracingUltrasoundSimulator::SimParams::conv_psf,
                     "Enable PSF convolution")
      .def_readwrite(
          "median_clip_filter",
          &ultrasound_simulator::RaytracingUltrasoundSimulator::SimParams::median_clip_filter,
          "Enable median clip filter for speckle reduction")
      .def_readwrite(
          "enable_cuda_timing",
          &ultrasound_simulator::RaytracingUltrasoundSimulator::SimParams::enable_cuda_timing,
          "Enable CUDA timing measurements")
      .def_readwrite(
          "write_debug_images",
          &ultrasound_simulator::RaytracingUltrasoundSimulator::SimParams::write_debug_images,
          "Enable debug image output")
      .def_readwrite(
          "contact_epsilon",
          &ultrasound_simulator::RaytracingUltrasoundSimulator::SimParams::contact_epsilon,
          "Maximum distance for element activation [mm]")
      .def_property(
          "b_mode_size",
          [](ultrasound_simulator::RaytracingUltrasoundSimulator::SimParams& self) {
            return py::make_tuple(self.b_mode_size.x, self.b_mode_size.y);
          },
          [](ultrasound_simulator::RaytracingUltrasoundSimulator::SimParams& self,
             py::object size) {
            // Check if input is a numpy array
            if (py::isinstance<py::array>(size)) {
              auto array = size.cast<py::array>();
              if (array.ndim() != 1 || array.shape(0) != 2) {
                throw std::runtime_error("b_mode_size numpy array must have shape (2,)");
              }

              // Accept every integer and floating dtype whatever its item size or strides
              // (numpy's default int64 / float64 included): forcecast copies the two values
              // into a contiguous double array, which holds every uint32 exactly, and each
              // value is checked before it is stored, so nothing wraps or truncates silently.
              const char kind = array.dtype().kind();
              if (kind != 'i' && kind != 'u' && kind != 'f') {
                throw std::runtime_error("Unsupported numpy array data type for b_mode_size");
              }
              constexpr int kContiguousCopy = py::array::c_style | py::array::forcecast;
              auto values = py::array_t<double, kContiguousCopy>::ensure(array);
              if (!values) {
                throw std::runtime_error("Unsupported numpy array data type for b_mode_size");
              }
              const double* data = values.data();
              uint32_t dims[2];
              for (int i = 0; i < 2; ++i) {
                const double v = data[i];
                if (!std::isfinite(v) || v < 0.0 || v != std::floor(v) ||
                    v > static_cast<double>(std::numeric_limits<uint32_t>::max())) {
                  throw std::runtime_error(
                      "b_mode_size numpy array values must be whole numbers from 0 to "
                      "4294967295");
                }
                dims[i] = static_cast<uint32_t>(v);
              }
              self.b_mode_size = make_uint2(dims[0], dims[1]);
            }
            // Check if it's a tuple or list
            else if (py::isinstance<py::tuple>(size) || py::isinstance<py::list>(size)) {
              if (py::len(size) != 2) {
                throw std::runtime_error("b_mode_size tuple/list must have exactly 2 elements");
              }

              try {
                uint32_t width = py::cast<uint32_t>(size[py::int_(0)]);
                uint32_t height = py::cast<uint32_t>(size[py::int_(1)]);
                self.b_mode_size = make_uint2(width, height);
              } catch (const py::cast_error&) {
                throw std::runtime_error(
                    "b_mode_size elements must be convertible to unsigned integers");
              }
            } else {
              throw std::runtime_error(
                  "b_mode_size must be a tuple, list, or numpy array with 2 elements");
            }
          },
          "B-mode image size as (width, height), can be a tuple, list, or numpy array");

  // Bind RaytracingUltrasoundSimulator class
  py::class_<ultrasound_simulator::RaytracingUltrasoundSimulator>(
      m, "RaytracingUltrasoundSimulator", R"pbdoc(
        Main ultrasound simulation engine using raytracing.

        This class performs the actual ultrasound simulation by:
            1. Raytracing through the scene geometry
            2. Computing acoustic interactions
            3. Generating B-mode images
    )pbdoc")
      .def(py::init([](ultrasound_simulator::World* world,
                       const ultrasound_simulator::Materials* materials) {
             try {
               spdlog::info("Creating RaytracingUltrasoundSimulator");
               auto simulator =
                   std::make_unique<ultrasound_simulator::RaytracingUltrasoundSimulator>(world,
                                                                                         materials);
               spdlog::info("RaytracingUltrasoundSimulator created successfully");
               return simulator;
             } catch (const std::exception& e) {
               spdlog::error("Failed to create RaytracingUltrasoundSimulator: {}", e.what());
               throw;
             }
           }),
           R"pbdoc(
        Initialize a new simulator.

        Args:
            world: World instance containing the scene geometry
            materials: Materials instance with material definitions
      )pbdoc")
      .def("get_min_x",
           &ultrasound_simulator::RaytracingUltrasoundSimulator::get_min_x,
           "Get the minimum x value of the simulated region [mm]")
      .def("get_max_x",
           &ultrasound_simulator::RaytracingUltrasoundSimulator::get_max_x,
           "Get the maximum x value of the simulated region [mm]")
      .def("get_min_z",
           &ultrasound_simulator::RaytracingUltrasoundSimulator::get_min_z,
           "Get the minimum z value of the simulated region [mm]")
      .def("get_max_z",
           &ultrasound_simulator::RaytracingUltrasoundSimulator::get_max_z,
           "Get the maximum z value of the simulated region [mm]")
      .def(
          "simulate",
          [](ultrasound_simulator::RaytracingUltrasoundSimulator& self,
             const ultrasound_simulator::BaseProbe* probe,
             const ultrasound_simulator::RaytracingUltrasoundSimulator::SimParams& params) {
            try {
              spdlog::info("Starting simulation");
              auto result = self.simulate(probe, params);

              if (!result.b_mode) {
                spdlog::error("Simulation returned null b_mode");
                throw std::runtime_error("Simulation returned null b_mode");
              }

              std::vector<ssize_t> shape = {static_cast<ssize_t>(params.b_mode_size.y),
                                            static_cast<ssize_t>(params.b_mode_size.x)};
              auto array = cuda_memory_to_numpy(*result.b_mode, shape, params.stream);
              spdlog::info("Simulation completed successfully");
              return array;
            } catch (const std::exception& e) {
              spdlog::error("Exception in simulate: {}", e.what());
              throw;
            }
          },
          R"pbdoc(
        Perform ultrasound simulation.

        Args:
            probe: BaseProbe instance defining imaging parameters
            params: SimParams instance with simulation settings

        Returns:
            np.ndarray: B-mode ultrasound image
      )pbdoc")
      .def(
          "simulate_with_metadata",
          [](ultrasound_simulator::RaytracingUltrasoundSimulator& self,
             const ultrasound_simulator::BaseProbe* probe,
             const ultrasound_simulator::RaytracingUltrasoundSimulator::SimParams& params) {
            try {
              spdlog::info("Starting simulation with scanline metadata");
              auto metadata_params = params;
              metadata_params.return_raw_rf = true;
              auto result = self.simulate(probe, metadata_params);
              if (!result.b_mode || !result.rf_data || !result.raw_rf_data) {
                throw std::runtime_error("Simulation returned incomplete result buffers");
              }

              py::dict output;
              output["b_mode"] = cuda_memory_to_numpy(*result.b_mode,
                                                      {static_cast<ssize_t>(params.b_mode_size.y),
                                                       static_cast<ssize_t>(params.b_mode_size.x)},
                                                      params.stream);
              output["scanlines"] =
                  cuda_memory_to_numpy(*result.rf_data,
                                       {static_cast<ssize_t>(probe->get_num_elements()),
                                        static_cast<ssize_t>(params.buffer_size)},
                                       params.stream);
              output["rf_data"] =
                  cuda_memory_to_numpy(*result.raw_rf_data,
                                       {static_cast<ssize_t>(probe->get_num_elements()),
                                        static_cast<ssize_t>(params.buffer_size)},
                                       params.stream);
              output["scanline_timestamps"] = vector_to_numpy(result.scanline_timestamps);
              spdlog::info("Simulation with scanline metadata completed successfully");
              return output;
            } catch (const std::exception& e) {
              spdlog::error("Exception in simulate_with_metadata: {}", e.what());
              throw;
            }
          },
          R"pbdoc(
        Run the simulation and return the image with its scanline data.

        Alongside the B-mode image, this returns the direct OptiX response,
        processed A-lines and the acquisition time of each line. Use simulate()
        when only the image is needed.

        Returns:
            dict: ``b_mode`` (H, W), ``rf_data`` and ``scanlines`` (both lines,
            depth samples), and ``scanline_timestamps`` (lines,) in seconds
            relative to frame start.
      )pbdoc");

  // Bind Hitable base class
  py::class_<ultrasound_simulator::Hitable>(m, "Hitable", R"pbdoc(
        Base class for objects that can be hit by rays.

        This is the parent class for Mesh and Sphere objects.
    )pbdoc")
      .def(
          "get_aabb_min",
          [](const ultrasound_simulator::Hitable& self) {
            return float3_to_numpy(self.get_aabb_min());
          },
          "Get the minimum point of the object's bounding box")
      .def(
          "get_aabb_max",
          [](const ultrasound_simulator::Hitable& self) {
            return float3_to_numpy(self.get_aabb_max());
          },
          "Get the maximum point of the object's bounding box");

  // Bind Mesh class
  py::class_<ultrasound_simulator::Mesh, ultrasound_simulator::Hitable>(m, "Mesh", R"pbdoc(
        Represents a 3D mesh object for ultrasound simulation.

        The mesh is loaded from a file or constructed from in-memory arrays and
        assigned a material.
    )pbdoc")
      .def(py::init([](const std::string& file_name, uint32_t material_id) {
             spdlog::info(
                 "Creating Mesh from file: {} with material_id: {}", file_name, material_id);
             try {
               auto mesh = std::make_unique<ultrasound_simulator::Mesh>(file_name, material_id);
               spdlog::info("Mesh created successfully");
               return mesh;
             } catch (const std::exception& e) {
               spdlog::error("Failed to create Mesh: {}", e.what());
               throw;
             }
           }),
           py::arg("file_name"),
           py::arg("material_id"),
           R"pbdoc(
        Initialize a new mesh.

        Args:
            file_name (str): Path to OBJ file
            material_id (int): Material index from Materials.get_index()
      )pbdoc")
      .def(py::init([](const py::array& vertices,
                       const py::array& indices,
                       const py::object& normals,
                       uint32_t material_id) {
             return mesh_from_numpy(vertices, indices, normals, material_id);
           }),
           py::arg("vertices"),
           py::arg("indices"),
           py::arg("normals"),
           py::arg("material_id"),
           R"pbdoc(
        Initialise a mesh from in-memory geometry.

        Args:
            vertices (np.ndarray): Vertex positions with shape (N, 3) and dtype float32.
            indices (np.ndarray): Triangle indices with shape (M, 3) and dtype uint32,
                int32, or int64.
            normals (Optional[np.ndarray]): Per-vertex normals with shape (N, 3) and
                dtype float32. Supplied normals are normalised; if omitted, area-weighted
                normals are computed.
            material_id (int): Material index from Materials.get_index().

        Example:
            >>> import numpy as np
            >>> vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
            >>> indices = np.array([[0, 1, 2]], dtype=np.uint32)
            >>> mesh = Mesh(vertices=vertices, indices=indices, material_id=0)

        Geometry extracted from a UsdGeomMesh with usd-core can be passed directly after
        triangulating its face indices host-side; ultrasound_simulator does not depend on usd-core.
      )pbdoc")
      .def(py::init([](const py::array& vertices, const py::array& indices, uint32_t material_id) {
             return mesh_from_numpy(vertices, indices, py::none(), material_id);
           }),
           py::arg("vertices"),
           py::arg("indices"),
           py::arg("material_id"),
           R"pbdoc(
        Initialise a mesh from in-memory geometry and compute area-weighted vertex normals.

        Args:
            vertices (np.ndarray): Vertex positions with shape (N, 3) and dtype float32.
            indices (np.ndarray): Triangle indices with shape (M, 3) and dtype uint32,
                int32, or int64.
            material_id (int): Material index from Materials.get_index().
      )pbdoc");

  // Bind Sphere class
  py::class_<ultrasound_simulator::Sphere, ultrasound_simulator::Hitable>(m, "Sphere", R"pbdoc(
        Represents a sphere object for ultrasound simulation.

        A simple geometric primitive useful for testing and basic simulations.
    )pbdoc")
      .def(py::init([](py::array_t<float> center, float radius, uint32_t material_id) {
             // Convert NumPy array to float3
             float3 center_float3 = numpy_to_float3(center);
             return std::make_unique<ultrasound_simulator::Sphere>(
                 center_float3, radius, material_id);
           }),
           py::arg("center"),
           py::arg("radius"),
           py::arg("material_id"),
           R"pbdoc(
        Initialize a new sphere.

        Args:
            center (np.ndarray): 3D center position [x, y, z]
            radius (float): Sphere radius
            material_id (int): Material index from Materials.get_index()
      )pbdoc");
}
