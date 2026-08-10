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

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "raysim/core/hitable.hpp"

namespace {

const std::vector<float3> kTriangleVertices = {
    make_float3(0.0f, 0.0f, 0.0f),
    make_float3(1.0f, 0.0f, 0.0f),
    make_float3(0.0f, 1.0f, 0.0f),
};

const std::vector<uint32_t> kTriangleIndices = {0, 1, 2};

class MeshWithoutNormalsFile {
 public:
  MeshWithoutNormalsFile()
      : path_(std::filesystem::temp_directory_path() / "raysim_mesh_without_normals.obj") {
    std::ofstream stream(path_);
    stream << "o TriangleWithoutNormals\n"
              "v 0.0 0.0 0.0\n"
              "v 1.0 0.0 0.0\n"
              "v 0.0 1.0 0.0\n"
              "f 1 2 3\n";
    if (!stream) { throw std::runtime_error("failed to write the normals-free mesh fixture"); }
  }

  ~MeshWithoutNormalsFile() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

void require(bool condition, const char* message) {
  if (!condition) { throw std::runtime_error(message); }
}

template <typename Callable>
void require_invalid_argument(Callable&& callable, const char* message) {
  try {
    callable();
  } catch (const std::invalid_argument&) { return; }

  throw std::runtime_error(message);
}

void test_constructs_from_arrays_and_computes_aabb() {
  const std::vector<float3> vertices = {
      make_float3(-1.0f, -2.0f, -3.0f),
      make_float3(4.0f, -2.0f, -3.0f),
      make_float3(-1.0f, 5.0f, -3.0f),
      make_float3(-1.0f, -2.0f, 6.0f),
  };
  const std::vector<uint32_t> indices = {
      0,
      2,
      1,
      0,
      1,
      3,
      0,
      3,
      2,
      1,
      2,
      3,
  };

  const raysim::Mesh mesh(vertices, indices, {}, 3);

  const float3 aabb_min = mesh.get_aabb_min();
  require(aabb_min.x == -1.0f, "unexpected AABB minimum x component");
  require(aabb_min.y == -2.0f, "unexpected AABB minimum y component");
  require(aabb_min.z == -3.0f, "unexpected AABB minimum z component");

  const float3 aabb_max = mesh.get_aabb_max();
  require(aabb_max.x == 4.0f, "unexpected AABB maximum x component");
  require(aabb_max.y == 5.0f, "unexpected AABB maximum y component");
  require(aabb_max.z == 6.0f, "unexpected AABB maximum z component");
}

void test_rejects_empty_vertices() {
  require_invalid_argument([]() { raysim::Mesh({}, kTriangleIndices, {}, 0); },
                           "empty vertices were accepted");
}

void test_rejects_empty_indices() {
  require_invalid_argument([]() { raysim::Mesh(kTriangleVertices, {}, {}, 0); },
                           "empty indices were accepted");
}

void test_rejects_an_incomplete_triangle() {
  require_invalid_argument([]() { raysim::Mesh(kTriangleVertices, {0, 1}, {}, 0); },
                           "an incomplete triangle was accepted");
}

void test_rejects_an_out_of_range_index() {
  require_invalid_argument([]() { raysim::Mesh(kTriangleVertices, {0, 1, 3}, {}, 0); },
                           "an out-of-range index was accepted");
}

void test_rejects_the_wrong_number_of_normals() {
  const std::vector<float3> normals = {
      make_float3(0.0f, 0.0f, 1.0f),
      make_float3(0.0f, 0.0f, 1.0f),
  };

  require_invalid_argument(
      [&normals]() { raysim::Mesh(kTriangleVertices, kTriangleIndices, normals, 0); },
      "the wrong number of normals was accepted");
}

void test_file_constructor_still_rejects_a_mesh_without_normals() {
  const MeshWithoutNormalsFile file;

  bool rejected = false;
  try {
    const raysim::Mesh mesh(file.path().string(), 0);
  } catch (const std::runtime_error& error) {
    require(std::string(error.what()) == "Mesh: has no normals",
            "the normals-free mesh produced an unexpected error");
    rejected = true;
  }

  require(rejected, "the normals-free mesh was accepted");
}

}  // namespace

int main() {
  try {
    test_constructs_from_arrays_and_computes_aabb();
    test_rejects_empty_vertices();
    test_rejects_empty_indices();
    test_rejects_an_incomplete_triangle();
    test_rejects_an_out_of_range_index();
    test_rejects_the_wrong_number_of_normals();
    test_file_constructor_still_rejects_a_mesh_without_normals();
  } catch (const std::exception& exception) {
    std::cerr << "mesh test failed: " << exception.what() << '\n';
    return 1;
  }

  std::cout << "mesh test passed\n";
  return 0;
}
