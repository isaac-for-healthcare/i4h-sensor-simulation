/*

 * SPDX-FileCopyrightText: Copyright (c) 2019 - 2024  NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <cuda_runtime.h>


struct MaterialData
{
    enum Type
    {
        PBR           = 0,
        GLASS         = 1,
        PHONG         = 2,
        CHECKER_PHONG = 3
    };

    MaterialData()
    {
        type                       = MaterialData::PBR;
        pbr.base_color             = { 1.0f, 1.0f, 1.0f, 1.0f };
        pbr.metallic               = 1.0f;
        pbr.roughness              = 1.0f;
        pbr.base_color_tex         = { 0, 0 };
        pbr.metallic_roughness_tex = { 0, 0 };
    }

    enum AlphaMode
    {
        ALPHA_MODE_OPAQUE = 0,
        ALPHA_MODE_MASK   = 1,
        ALPHA_MODE_BLEND  = 2
    };

    struct Texture
    {
        __device__ __forceinline__ operator bool() const
        {
            return tex != 0;
        }

        int                  texcoord;
        cudaTextureObject_t  tex;

        float2               texcoord_offset;
        float2               texcoord_rotation; // sin,cos
        float2               texcoord_scale;
    };

    struct Pbr
    {
        float4               base_color;
        float                metallic;
        float                roughness;

        Texture              base_color_tex;
        Texture              metallic_roughness_tex;
    };

    struct CheckerPhong
    {
        float3 Kd1, Kd2;
        float3 Ka1, Ka2;
        float3 Ks1, Ks2;
        float3 Kr1, Kr2;
        float  phong_exp1, phong_exp2;
        float2 inv_checker_size;
    };

    struct Glass
    {
        float  importance_cutoff;
        float3 cutoff_color;
        float  fresnel_exponent;
        float  fresnel_minimum;
        float  fresnel_maximum;
        float  refraction_index;
        float3 refraction_color;
        float3 reflection_color;
        float3 extinction_constant;
        float3 shadow_attenuation;
        int    refraction_maxdepth;
        int    reflection_maxdepth;
    };

    struct Phong
    {
        float3 Ka;
        float3 Kd;
        float3 Ks;
        float3 Kr;
        float  phong_exp;
    };

    Type                 type            = PBR;

    Texture              normal_tex      = { 0 , 0 };

    AlphaMode            alpha_mode      = ALPHA_MODE_OPAQUE;
    float                alpha_cutoff    = 0.f;

    float3               emissive_factor = { 0.f, 0.f, 0.f };
    Texture              emissive_tex    = { 0, 0 };

    bool                 doubleSided     = false;

    union
    {
        Pbr          pbr;
        Glass        glass;
        Phong        metal;
        CheckerPhong checker;
    };
};
