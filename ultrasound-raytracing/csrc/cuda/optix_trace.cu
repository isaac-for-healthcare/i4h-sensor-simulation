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

#include <optix.h>

#include <cuda/helpers.h>
#include <sutil/vec_math.h>

#include "raysim/cuda/optix_trace.hpp"

#include <OptiXToolkit/ShaderUtil/OptixSelfIntersectionAvoidance.h>

namespace raysim {

extern "C" {
__constant__ Params params;
}

static __forceinline__ __device__ Payload get_payload() {
  Payload payload;
  static_assert(sizeof(Payload) / sizeof(uint32_t) == 5);
  reinterpret_cast<uint32_t*>(&payload)[0] = optixGetPayload_0();
  reinterpret_cast<uint32_t*>(&payload)[1] = optixGetPayload_1();
  reinterpret_cast<uint32_t*>(&payload)[2] = optixGetPayload_2();
  reinterpret_cast<uint32_t*>(&payload)[3] = optixGetPayload_3();
  reinterpret_cast<uint32_t*>(&payload)[4] = optixGetPayload_4();
  return payload;
}

static __device__ float get_scattering_value(float3 pos, const Material* material,
                                             float resolution_mm = 50.f) {
  // Convert point to texture coordinates
  pos /= resolution_mm;

  const float2 scatter_val = tex3D<float2>(params.scattering_texture, pos.x, pos.y, pos.z);

  // Apply material properties
  if (scatter_val.x <= material->mu0_) { return scatter_val.y * material->sigma_; }
  return 0.f;
}

/**
 * @return offset to intensity buffer at ray distance t
 */
static __device__ uint32_t get_intensity_offset(float t) {
  uint32_t offset = uint32_t((t / params.t_far) * (params.buffer_size - 1) + 0.5f);
  assert(offset < params.buffer_size);
  return offset;
}

/// Calculate intensity using Beer-Lambert Law
static __device__ float get_intensity_at_distance(float distance, float medium_attenuation) {
  // I = I₀ * 10^(-αfd/20)
  // where α is attenuation coefficient in dB/(cm⋅MHz)
  // f is frequency in MHz
  // d is distance in cm
  const float source_freq = params.source_frequency;                            // MHz
  const float distance_cm = distance * 0.1f;                                    // Convert to cm
  const float attenuation_db = medium_attenuation * source_freq * distance_cm;  // dB
  return __powf(10.f, -attenuation_db * 0.05f);
}

/**
 * Sample intensities
 *
 * @param origin ray origin in world space
 * @param dir ray direction in world space
 * @param t_ancestors
 * @param t_min
 * @param t_max
 * @param intensity
 * @param material
 * @param intensities
 */
static __device__ void sample_intensities(float3 origin, float3 dir, float t_ancestors, float t_min,
                                          float t_max, float intensity, const Material* material,
                                          float* intensities) {
  // Early out for materials with zero scattering density or coefficient
  if ((material->mu0_ <= 0.f) || (material->sigma_ == 0.f)) { return; }

  const uint32_t steps = ((t_max - t_min) / params.t_far) * params.buffer_size + 0.5f;
  const float t_step = (t_max - t_min) / steps;
  const float3 start = origin + t_min * dir;

  intensities += get_intensity_offset(t_ancestors + t_min);

  for (uint32_t step = 0; step < steps; ++step) {
    const float distance = (step * t_step);
    const float3 pos = start + distance * dir;
    intensities[step] += get_scattering_value(pos, material) * intensity *
                         get_intensity_at_distance(distance, material->attenuation_);
  }
}

template <OptixPrimitiveType PRIM_TYPE>
static __device__ float3 get_normal(float3 ray_orig, float3 ray_dir, float t_hit, uint32_t hit_id,
                                    const HitGroupData* hit_group_data) {
  const unsigned int prim_idx = optixGetPrimitiveIndex();

  float3 normal;
  if constexpr (PRIM_TYPE == OptixPrimitiveType::OPTIX_PRIMITIVE_TYPE_TRIANGLE) {
    uint32_t* const tri = &hit_group_data->indices[prim_idx * 3];
    const float3 N0 = hit_group_data->normals[tri[0]];
    const float3 N1 = hit_group_data->normals[tri[1]];
    const float3 N2 = hit_group_data->normals[tri[2]];

    const float2 barys = optixGetTriangleBarycentrics();

    normal = (1.f - barys.x - barys.y) * N0 + barys.x * N1 + barys.y * N2;
  } else {
    assert(PRIM_TYPE == OptixPrimitiveType::OPTIX_PRIMITIVE_TYPE_SPHERE);
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    float4 q;
    // sphere center (q.x, q.y, q.z), sphere radius q.w
    optixGetSphereData(gas, prim_idx, hit_id, 0.f, &q);

    const float3 raypos = ray_orig + t_hit * ray_dir;
    normal = (raypos - make_float3(q)) / q.w;
  }
  return normal;
}

/**
 * Calculate reflected direction vector
 */
static __device__ float3 calc_reflected_dir(float3 incident_dir, float3 normal) {
  // Ensure normal points against incident direction
  if (dot(incident_dir, normal) > 0.f) { normal = -normal; }
  return incident_dir - 2.f * dot(incident_dir, normal) * normal;
}

/**
 * Calculate reflection coefficient using acoustic impedance
 */
static __device__ float calculate_reflection_coefficient(float incident_angle,
                                                         const Material* material1,
                                                         const Material* material2) {
  float Z1 = material1->impedance_;
  float Z2 = material2->impedance_;
  float cos_theta = fabsf(__cosf(incident_angle));
  float R = ((Z2 * cos_theta - Z1) / (Z2 * cos_theta + Z1));
  return R * R;
}

/**
 * Calculate refracted direction vector using Snell's law
 *
 * @returns true for total internal reflection
 */
static __device__ bool calc_refracted_dir(float3 incident_dir, float3 normal, float v1, float v2,
                                          float3* refracted_dir) {
  // Ensure normal points against incident direction
  if (dot(incident_dir, normal) > 0.f) { normal = -normal; }

  float cos_i = -dot(normal, incident_dir);  // Incidence
  float sin_i = sqrtf(1.f - cos_i * cos_i);  // Incidence
  float sin_t = (v1 / v2) * sin_i;           // Transmission

  // Check for total internal reflection
  if (sin_t >= 1) { return true; }

  float cos_t = sqrtf(1 - sin_t * sin_t);
  *refracted_dir = (v1 / v2) * incident_dir + ((v1 / v2) * cos_i - cos_t) * normal;
  return false;
}

/**
 * Calculate the reflection intensity Ir for ultrasound RF image
 *
 * Eq. 5 Mattausch2016Monte-Carlo
 *
 * @param V_r reflected ray direction vector
 * @param V_i refracted ray direction vector
 * @param total_internal_reflection
 * @param D vector from intersection point to transducer origin
 * @param n surface specularity parameter
 * @return reflection intensity Ir
 */
static __device__ float calculate_specular_intensity(float3 V_r, float3 V_i,
                                                     bool total_internal_reflection, float3 D,
                                                     float n) {
  // Calculate angles using dot product
  float cos_reflected = dot(V_r, D) / (length(V_r) * length(D));
  float cos_refracted;
  if (!total_internal_reflection) {
    cos_refracted = dot(V_i, D) / (length(V_i) * length(D));
  } else {
    cos_refracted = 0.f;
  }

  // Calculate the two terms (reflection and refraction)
  float reflected_term = max(0.f, __powf(cos_reflected, n));
  float refracted_term = max(0.f, __powf(cos_refracted, n));

  // Total intensity is sum of both terms
  float Ir = reflected_term + refracted_term;

  return Ir;
}

/**
 * Self-intersection avoidance. Get the save front and back start points.
 * See https://github.com/NVIDIA/optix-toolkit/tree/master/ShaderUtil#self-intersection-avoidance.
 *
 * @param out_front_start [out] offset spawn point on the front of the surface, safe from self
 * intersection
 * @param out_back_start [out] offset spawn point on the back of the surface, safe from self
 * intersection
 * @param out_wld_norm [out] unit length spawn point normal in world space
 */
static __device__ void get_save_start_point(float3& out_front_start, float3& out_back_start,
                                            float3& out_wld_norm) {
  // Compute a surface point, normal and conservative offset in object-space.
  float3 obj_pos, obj_norm;
  float obj_offset;
  SelfIntersectionAvoidance::getSafeTriangleSpawnOffset(obj_pos, obj_norm, obj_offset);
  // Transform the object-space position, normal and offset into world-space. The output world-space
  // offset includes the input object-space offset and accounts for the transformation.
  float3 wld_pos;
  float wld_offset;
  SelfIntersectionAvoidance::transformSafeSpawnOffset(
      wld_pos, out_wld_norm, wld_offset, obj_pos, obj_norm, obj_offset);

  // The offset is used to compute safe spawn points on the front and back of the surface.
  SelfIntersectionAvoidance::offsetSpawnPoint(
      out_front_start, out_back_start, wld_pos, out_wld_norm, wld_offset);
}

extern "C" __global__ void __raygen__rg() {
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dim = optixGetLaunchDimensions();

  const RayGenData* ray_gen_data = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());

  const float d_x = (static_cast<float>(idx.x) / static_cast<float>(dim.x)) - 0.5f;

  float3 origin;
  float3 direction;

  // Different ray generation based on probe type
  switch (ray_gen_data->probe_type) {
    case PROBE_TYPE_CURVILINEAR: {
      // Convert normalized coordinates to lateral angle in radians
      const float lateral_angle = (ray_gen_data->opening_angle * d_x) * (M_PI / 180.f);

      // Calculate element position on probe surface in probe's local coordinate system
      // where (0,0,0) is at the probe face center
      origin =
          make_float3(ray_gen_data->radius * __sinf(lateral_angle),  // x = r * sin(θ)
                      0.f,                                           // y (elevation added later)
                      ray_gen_data->radius * (__cosf(lateral_angle) - 1.f)  // z = r * (cos(θ) - 1)
          );

      // Calculate ray direction away from center of curvature
      // Center of curvature is at (0,0,-radius) in probe's local coordinate system
      direction = normalize(origin - make_float3(0.f, 0.f, -ray_gen_data->radius));
      break;
    }

    case PROBE_TYPE_LINEAR_ARRAY: {
      // For linear arrays, elements are positioned along a straight line
      // Map normalized coordinate to position along the width
      const float element_width = ray_gen_data->width;
      const float element_pos = element_width * d_x;

      // Element position in local coordinates
      origin = make_float3(element_pos,  // x position along array
                           0.f,          // y (elevation added later)
                           0.f           // z at surface (probe face)
      );

      // For linear arrays, rays travel perpendicular to the array
      direction = make_float3(0.f, 0.f, 1.f);
      break;
    }

    case PROBE_TYPE_PHASED_ARRAY: {
      // Convert normalized coordinates to steering angle in radians
      // Use half sector_angle since we're calculating from center
      const float half_angle_rad = (ray_gen_data->opening_angle / 2.0f) * (M_PI / 180.f);
      const float steering_angle = d_x * 2.0f * half_angle_rad;

      // For phased arrays, all rays originate from a single virtual point (0,0,0)
      // This is the center of the transducer array face
      origin = make_float3(0.0f,  // Center of the array
                           0.f,   // y (elevation added later)
                           0.f    // z at the surface of the probe
      );

      // Direction determined by steering angle
      direction = make_float3(__sinf(steering_angle),  // x component based on steering angle
                              0.f,                     // y component (no elevation steering)
                              __cosf(steering_angle)   // z component (along central axis)
      );

      // Normalize direction to ensure unit length vector
      direction = normalize(direction);
      break;
    }

    default: {
      // Default to curvilinear behavior as fallback
      const float lateral_angle = (ray_gen_data->opening_angle * d_x) * (M_PI / 180.f);
      origin = make_float3(ray_gen_data->radius * __sinf(lateral_angle),
                           0.f,
                           ray_gen_data->radius * (__cosf(lateral_angle) - 1.f));
      direction = normalize(origin - make_float3(0.f, 0.f, -ray_gen_data->radius));
      break;
    }
  }

  // Add elevation in probe's local coordinate system (common for all probes)
  const float d_y = (static_cast<float>(idx.y) / static_cast<float>(dim.y)) - 0.5f;
  const float elevation = ray_gen_data->elevational_height * d_y;
  origin.y = elevation;

  // Transform from probe's local coordinate system to global coordinate system
  origin = ray_gen_data->rotation_matrix * origin;
  origin += ray_gen_data->position;

  direction = ray_gen_data->rotation_matrix * direction;

  Payload ray{};
  ray.intensity = 1.f;
  ray.depth = 0;
  ray.current_material_id = params.background_material_id;
  ray.outter_material_id = 0;
  ray.current_obj_id = static_cast<uint16_t>(-1);
  ray.outter_obj_id = static_cast<uint16_t>(-1);

  optixTrace(params.handle,
             origin,
             direction,
             0.f,           // tmin
             params.t_far,  // tmax
             0.f,           // rayTime
             OptixVisibilityMask(1),
             OPTIX_RAY_FLAG_NONE,
             0,  // SBT offset
             1,  // SBT stride
             0,  // missSBTIndex
             reinterpret_cast<uint32_t*>(&ray)[0],
             reinterpret_cast<uint32_t*>(&ray)[1],
             reinterpret_cast<uint32_t*>(&ray)[2],
             reinterpret_cast<uint32_t*>(&ray)[3],
             reinterpret_cast<uint32_t*>(&ray)[4]);
  static_assert(sizeof(Payload) / sizeof(uint32_t) == 5);
}

extern "C" __global__ void __miss__ms() {
  const uint3 idx = optixGetLaunchIndex();
  const Payload ray = get_payload();

  // no hits, just do scattering up to t_far
  sample_intensities(
      optixGetWorldRayOrigin(),
      optixGetWorldRayDirection(),
      ray.t_ancestors,
      optixGetRayTmin(),
      optixGetRayTmax(),
      ray.intensity,
      &params.materials[ray.current_material_id],
      &params.scanlines[(idx.y * optixGetLaunchDimensions().x + idx.x) * params.buffer_size]);
}

template <OptixPrimitiveType PRIM_TYPE>
static __device__ void closest_hit() {
  const float3 ray_orig = optixGetWorldRayOrigin();
  const float3 ray_dir = optixGetWorldRayDirection();
  const float t_min = optixGetRayTmin();
  const float t = optixGetRayTmax();

  const Payload ray = get_payload();
  const uint32_t current_material_id = ray.current_material_id;
  const Material* current_material = &params.materials[current_material_id];
  const uint3 idx = optixGetLaunchIndex();
  float* const scanline =
      &params.scanlines[(idx.y * optixGetLaunchDimensions().x + idx.x) * params.buffer_size];

  // add scattering contribution up to hit
  sample_intensities(
      ray_orig, ray_dir, ray.t_ancestors, t_min, t, ray.intensity, current_material, scanline);

  // Don't generate secondary rays is max depth is reached
  if (ray.depth + 1 >= params.max_depth) { return; }

  const uint32_t hit_id = optixGetSbtGASIndex();
  const HitGroupData* hit_group_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());

  uint32_t next_material_id, next_obj_id;
  if (ray.current_obj_id == hit_id) {
    // Exiting an object
    next_material_id = ray.outter_material_id;
    next_obj_id = ray.outter_obj_id;
  } else {
    next_material_id = hit_group_data->material_id;
    next_obj_id = hit_id;
  }

  // Calculate final intensity
  const float final_intensity =
      ray.intensity * get_intensity_at_distance(t - t_min, current_material->attenuation_);

  // Add hit reflection contribution
  const Material* next_material = &params.materials[next_material_id];
  const float3 normal = get_normal<PRIM_TYPE>(ray_orig, ray_dir, t, hit_id, hit_group_data);
  const float incident_angle = acosf(fabsf(dot(ray_dir, normal)));
  const float R = calculate_reflection_coefficient(incident_angle, current_material, next_material);

  float3 refracted_dir;
  const bool total_internal_reflection = calc_refracted_dir(ray_dir,
                                                            normal,
                                                            current_material->speed_of_sound_,
                                                            next_material->speed_of_sound_,
                                                            &refracted_dir);
  const float3 reflected_dir = calc_reflected_dir(ray_dir, normal);
  const float reflected_intensity = final_intensity * R;
  const float refracted_intensity = final_intensity * (1 - R);

  // Add specular reflection from transmitted energy
  const float ray_coherence_attenuation = __powf(0.3f, ray.depth);
  const float specular_reflection = calculate_specular_intensity(reflected_dir,
                                                                 refracted_dir,
                                                                 total_internal_reflection,
                                                                 ray_dir,
                                                                 next_material->specularity_) *
                                    ray_coherence_attenuation;
  scanline[get_intensity_offset(ray.t_ancestors + t)] = 2.f * specular_reflection;

  // Self-intersection avoidance
  float3 front_start, back_start, wld_norm;
  if (PRIM_TYPE == OptixPrimitiveType::OPTIX_PRIMITIVE_TYPE_TRIANGLE) {
    get_save_start_point(front_start, back_start, wld_norm);
  } else {
    wld_norm = normal;
    const float epsilon = 1e-5f;
    front_start = ray_orig + ray_dir * (t - epsilon);
    back_start = ray_orig + ray_dir * (t + epsilon);
  }

  bool reflection_on = false;
  // Create reflected ray
  if (reflection_on && (reflected_intensity > params.min_intensity)) {
    Payload reflected_ray{};
    reflected_ray.intensity = reflected_intensity;
    reflected_ray.depth = ray.depth + 1;
    reflected_ray.t_ancestors = ray.t_ancestors + t;
    reflected_ray.current_material_id = ray.current_material_id;
    reflected_ray.outter_material_id = ray.outter_material_id;
    reflected_ray.current_obj_id = ray.current_obj_id;
    reflected_ray.outter_obj_id = ray.outter_obj_id;

    // Secondary rays along the surface normal should use the generated front point as origin, while
    // rays pointing away from the normal should use the back point as origin.
    const float3 start = (dot(reflected_dir, wld_norm) > 0.f) ? front_start : back_start;
    optixTrace(params.handle,
               start,
               reflected_dir,
               0.f,                                       // tmin
               params.t_far - reflected_ray.t_ancestors,  // tmax
               0.f,                                       // rayTime
               OptixVisibilityMask(1),
               OPTIX_RAY_FLAG_NONE,
               0,  // SBT offset
               1,  // SBT stride
               0,  // missSBTIndex
               reinterpret_cast<uint32_t*>(&reflected_ray)[0],
               reinterpret_cast<uint32_t*>(&reflected_ray)[1],
               reinterpret_cast<uint32_t*>(&reflected_ray)[2],
               reinterpret_cast<uint32_t*>(&reflected_ray)[3],
               reinterpret_cast<uint32_t*>(&reflected_ray)[4]);
    static_assert(sizeof(Payload) / sizeof(uint32_t) == 5);
  }

  // Create refracted ray
  if ((refracted_intensity > params.min_intensity) && !total_internal_reflection) {
    Payload refracted_ray{};
    refracted_ray.intensity = refracted_intensity;
    refracted_ray.depth = ray.depth + 1;
    refracted_ray.t_ancestors = ray.t_ancestors + t;
    refracted_ray.current_material_id = next_material_id;
    refracted_ray.outter_material_id = ray.current_material_id;
    refracted_ray.current_obj_id = next_obj_id;
    refracted_ray.outter_obj_id = ray.current_obj_id;

    // Secondary rays along the surface normal should use the generated front point as origin, while
    // rays pointing away from the normal should use the back point as origin.
    const float3 start = (dot(refracted_dir, wld_norm) > 0.f) ? front_start : back_start;
    optixTrace(params.handle,
               start,
               refracted_dir,
               0.f,                                       // tmin
               params.t_far - refracted_ray.t_ancestors,  // tmax
               0.f,                                       // rayTime
               OptixVisibilityMask(1),
               OPTIX_RAY_FLAG_NONE,
               0,  // SBT offset
               1,  // SBT stride
               0,  // missSBTIndex
               reinterpret_cast<uint32_t*>(&refracted_ray)[0],
               reinterpret_cast<uint32_t*>(&refracted_ray)[1],
               reinterpret_cast<uint32_t*>(&refracted_ray)[2],
               reinterpret_cast<uint32_t*>(&refracted_ray)[3],
               reinterpret_cast<uint32_t*>(&refracted_ray)[4]);
    static_assert(sizeof(Payload) / sizeof(uint32_t) == 5);
  }
}

extern "C" __global__ void __closesthit__sphere() {
  closest_hit<OptixPrimitiveType::OPTIX_PRIMITIVE_TYPE_SPHERE>();
}

extern "C" __global__ void __closesthit__triangle() {
  closest_hit<OptixPrimitiveType::OPTIX_PRIMITIVE_TYPE_TRIANGLE>();
}

}  // namespace raysim
