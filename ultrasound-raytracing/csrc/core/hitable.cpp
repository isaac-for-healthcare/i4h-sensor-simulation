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

#include "raysim/core/hitable.hpp"

#include <mutex>
#include <string>

#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <spdlog/spdlog.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/Importer.hpp>
#include <assimp/LogStream.hpp>
#include <assimp/Logger.hpp>

#include "raysim/cuda/optix_trace.hpp"

namespace raysim {

Hitable::Hitable(uint32_t material_id) : material_id_(material_id) {}

Sphere::Sphere(float3 center, float radius, uint32_t material_id)
    : Hitable(material_id), center_(center), radius_(radius) {
  aabb_min_.x = center_.x - radius_;
  aabb_min_.y = center_.y - radius_;
  aabb_min_.z = center_.z - radius_;
  aabb_max_.x = center_.x + radius_;
  aabb_max_.y = center_.y + radius_;
  aabb_max_.z = center_.z + radius_;
}

void Sphere::build(OptixBuildInput* optix_build_input, HitGroupData* hit_group_data,
                   cudaStream_t stream) {
  cuda_vertex_buffer_ = std::make_unique<CudaMemory>(sizeof(center_), stream);
  cuda_vertex_buffer_->upload(&center_, stream);

  cuda_radius_buffer_ = std::make_unique<CudaMemory>(sizeof(radius_), stream);
  cuda_radius_buffer_->upload(&radius_, stream);

  optix_build_input->type = OPTIX_BUILD_INPUT_TYPE_SPHERES;

  vertex_buffers_.push_back(cuda_vertex_buffer_->get_device_ptr(stream));
  optix_build_input->sphereArray.vertexBuffers = vertex_buffers_.data();
  optix_build_input->sphereArray.numVertices = 1;

  radius_buffers_.push_back(cuda_radius_buffer_->get_device_ptr(stream));
  optix_build_input->sphereArray.radiusBuffers = radius_buffers_.data();

  sbt_flags_.push_back(OPTIX_GEOMETRY_FLAG_NONE);
  optix_build_input->sphereArray.flags = sbt_flags_.data();
  optix_build_input->sphereArray.numSbtRecords = sbt_flags_.size();

  hit_group_data->material_id = material_id_;
}

template <spdlog::level::level_enum LEVEL>
class AssimpLogStream : public Assimp::LogStream {
 public:
  void write(const char* message) {
    // Remove new line
    std::string m(message);
    if (!m.empty() && m[m.size() - 1] == '\n') { m.erase(m.size() - 1); }
    spdlog::log(LEVEL, "[Assimp] {}", m);
  }
};

Mesh::Mesh(const std::string& file_name, uint32_t material_id) : Hitable(material_id) {
  // Initialize the logger (only done once)
  static std::once_flag flag;
  std::call_once(flag, []() {
    Assimp::DefaultLogger::create(nullptr, Assimp::Logger::LogSeverity::NORMAL, 0);
    Assimp::DefaultLogger::get()->attachStream(new AssimpLogStream<spdlog::level::level_enum::info>,
                                               Assimp::Logger::Info);
    Assimp::DefaultLogger::get()->attachStream(new AssimpLogStream<spdlog::level::level_enum::warn>,
                                               Assimp::Logger::Warn);
    Assimp::DefaultLogger::get()->attachStream(new AssimpLogStream<spdlog::level::level_enum::err>,
                                               Assimp::Logger::Err);
  });

  importer_ = std::make_shared<Assimp::Importer>();

  auto scene = importer_->ReadFile(file_name, aiProcess_GenBoundingBoxes);
  if (!scene) {
    throw std::runtime_error(fmt::format(
        "Loading of mesh `{}` failed with error `{}`", file_name, importer_->GetErrorString()));
  }

  auto node = scene->mRootNode;
  if (node->mNumChildren != 1) { throw std::runtime_error("Mesh: only one children supported"); }

  node = node->mChildren[0];
  if (node->mNumMeshes != 1) { throw std::runtime_error("Mesh: only one mesh supported"); }

  auto mesh = scene->mMeshes[node->mMeshes[0]];

  if (!mesh->HasPositions() || !mesh->HasFaces()) {
    throw std::runtime_error("Mesh: has no position or face data");
  }

  if (!mesh->HasNormals()) { throw std::runtime_error("Mesh: has no normals"); }

  // Get the axis aligned bounding box
  aabb_min_.x = mesh->mAABB.mMin.x;
  aabb_min_.y = mesh->mAABB.mMin.y;
  aabb_min_.z = mesh->mAABB.mMin.z;
  aabb_max_.x = mesh->mAABB.mMax.x;
  aabb_max_.y = mesh->mAABB.mMax.y;
  aabb_max_.z = mesh->mAABB.mMax.z;
}

void Mesh::build(OptixBuildInput* optix_build_input, HitGroupData* hit_group_data,
                 cudaStream_t stream) {
  auto scene = importer_->GetScene();
  auto mesh = scene->mMeshes[scene->mRootNode->mChildren[0]->mMeshes[0]];

  optix_build_input->type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

  cuda_vertex_buffer_ =
      std::make_unique<CudaMemory>(mesh->mNumVertices * sizeof(aiVector3D), stream);
  cuda_vertex_buffer_->upload(mesh->mVertices, stream);

  vertex_buffers_.push_back(cuda_vertex_buffer_->get_device_ptr(stream));
  optix_build_input->triangleArray.vertexBuffers = vertex_buffers_.data();
  optix_build_input->triangleArray.numVertices = mesh->mNumVertices;
  optix_build_input->triangleArray.vertexFormat = OptixVertexFormat::OPTIX_VERTEX_FORMAT_FLOAT3;

  std::vector<unsigned int> indices(mesh->mNumFaces * 3);
  for (unsigned int face_index = 0; face_index < mesh->mNumFaces; ++face_index) {
    const aiFace* const face = &mesh->mFaces[face_index];
    if (face->mNumIndices != 3) {
      spdlog::warn("Mesh: only triangles supported, found a face with {} indices",
                   face->mNumIndices);
    }
    indices[face_index * 3 + 0] = face->mIndices[0];
    indices[face_index * 3 + 1] = face->mIndices[1];
    indices[face_index * 3 + 2] = face->mIndices[2];
  }

  cuda_index_buffer_ = std::make_unique<CudaMemory>(indices.size() * sizeof(unsigned int), stream);
  cuda_index_buffer_->upload(indices.data(), stream);

  optix_build_input->triangleArray.indexBuffer = cuda_index_buffer_->get_device_ptr(stream);
  optix_build_input->triangleArray.numIndexTriplets = indices.size() / 3;
  optix_build_input->triangleArray.indexFormat =
      OptixIndicesFormat::OPTIX_INDICES_FORMAT_UNSIGNED_INT3;

  hit_group_data->indices =
      reinterpret_cast<uint32_t*>(optix_build_input->triangleArray.indexBuffer);

  cuda_normal_buffer_ =
      std::make_unique<CudaMemory>(mesh->mNumVertices * sizeof(aiVector3D), stream);
  cuda_normal_buffer_->upload(mesh->mNormals, stream);
  hit_group_data->normals = reinterpret_cast<float3*>(cuda_normal_buffer_->get_ptr(stream));

  sbt_flags_.push_back(OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT |
                       OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING);
  optix_build_input->triangleArray.flags = sbt_flags_.data();
  optix_build_input->triangleArray.numSbtRecords = sbt_flags_.size();

  hit_group_data->material_id = material_id_;
}

}  // namespace raysim
