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
#include <optix.h>

#include <cuda/LocalGeometry.h>
#include <cuda/LocalShading.h>
#include <cuda/helpers.h>
#include <cuda/random.h>
#include <sutil/vec_math.h>

#include "whitted_cuda.h"

//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__pinhole()
{
    const uint3  launch_idx     = optixGetLaunchIndex();
    const uint3  launch_dims    = optixGetLaunchDimensions();
    const float3 eye            = whitted::params.eye;
    const float3 U              = whitted::params.U;
    const float3 V              = whitted::params.V;
    const float3 W              = whitted::params.W;
    const int    subframe_index = whitted::params.subframe_index;

    //
    // Generate camera ray
    //
    unsigned int seed = tea<4>( launch_idx.y * launch_dims.x + launch_idx.x, subframe_index );

    // The center of each pixel is at fraction (0.5,0.5)
    const float2 subpixel_jitter =
        subframe_index == 0 ? make_float2( 0.5f, 0.5f ) : make_float2( rnd( seed ), rnd( seed ) );

    const float2 d =
        2.0f
            * make_float2( ( static_cast<float>( launch_idx.x ) + subpixel_jitter.x ) / static_cast<float>( launch_dims.x ),
                           ( static_cast<float>( launch_idx.y ) + subpixel_jitter.y ) / static_cast<float>( launch_dims.y ) )
        - 1.0f;
    const float3 ray_direction = normalize( d.x * U + d.y * V + W );
    const float3 ray_origin    = eye;

    //
    // Trace camera ray
    //
    whitted::PayloadRadiance payload;
    payload.result     = make_float3( 0.0f );
    payload.depth      = 0;

    traceRadiance( whitted::params.handle, ray_origin, ray_direction,
                   0.00f,  // tmin
                   1e16f,  // tmax
                   &payload );

    //
    // Update results
    // TODO: timview mode
    //
    const unsigned int image_index = launch_idx.y * launch_dims.x + launch_idx.x;
    float3             accum_color = payload.result;

    if( subframe_index > 0 )
    {
        const float  a                = 1.0f / static_cast<float>( subframe_index + 1 );
        const float3 accum_color_prev = make_float3( whitted::params.accum_buffer[image_index] );
        accum_color                   = lerp( accum_color_prev, accum_color, a );
    }
    whitted::params.accum_buffer[image_index] = make_float4( accum_color, 1.0f );
    whitted::params.frame_buffer[image_index] = make_color( accum_color );
}

extern "C" __global__ void __anyhit__radiance()
{
    const whitted::HitGroupData* hit_group_data = reinterpret_cast< whitted::HitGroupData* >( optixGetSbtDataPointer() );
    if( hit_group_data->material_data.pbr.base_color_tex )
    {
        const LocalGeometry geom       = getLocalGeometry( hit_group_data->geometry_data );
        const float         base_alpha = sampleTexture<float4>( hit_group_data->material_data.pbr.base_color_tex, geom ).w;
        // force mask mode, even for blend mode, as we don't do recursive traversal.
        if( base_alpha < hit_group_data->material_data.alpha_cutoff )
            optixIgnoreIntersection();
    }
}

extern "C" __global__ void __anyhit__occlusion()
{
    const whitted::HitGroupData* hit_group_data = reinterpret_cast< whitted::HitGroupData* >( optixGetSbtDataPointer() );
    if( hit_group_data->material_data.pbr.base_color_tex )
    {
        const LocalGeometry geom       = getLocalGeometry( hit_group_data->geometry_data );
        const float         base_alpha = sampleTexture<float4>( hit_group_data->material_data.pbr.base_color_tex, geom ).w;

        if( hit_group_data->material_data.alpha_mode != MaterialData::ALPHA_MODE_OPAQUE )
        {
            if( hit_group_data->material_data.alpha_mode == MaterialData::ALPHA_MODE_MASK )
            {
                if( base_alpha < hit_group_data->material_data.alpha_cutoff )
                    optixIgnoreIntersection();
            }

            float attenuation = whitted::getPayloadOcclusion() * (1.f - base_alpha);
            if( attenuation > 0.f )
            {
                whitted::setPayloadOcclusion( attenuation );
                optixIgnoreIntersection();
            }
        }
    }
}

extern "C" __global__ void __miss__constant_radiance()
{
    whitted::setPayloadResult( whitted::params.miss_color );
}

extern "C" __global__ void __miss__occlusion()
{
    whitted::setPayloadOcclusionCommit();
}

extern "C" __global__ void __closesthit__radiance()
{
    const whitted::HitGroupData* hit_group_data = reinterpret_cast<whitted::HitGroupData*>( optixGetSbtDataPointer() );
    const LocalGeometry          geom           = getLocalGeometry( hit_group_data->geometry_data );

    //
    // Retrieve material data
    //
    float4 base_color = hit_group_data->material_data.pbr.base_color * geom.color;
    if( hit_group_data->material_data.pbr.base_color_tex )
    {
        const float4 base_color_tex = sampleTexture<float4>( hit_group_data->material_data.pbr.base_color_tex, geom );

        // don't gamma correct the alpha channel.
        const float3 base_color_tex_linear = whitted::linearize( make_float3( base_color_tex ) );

        base_color *= make_float4( base_color_tex_linear.x, base_color_tex_linear.y, base_color_tex_linear.z, base_color_tex.w );
    }

    float  metallic  = hit_group_data->material_data.pbr.metallic;
    float  roughness = hit_group_data->material_data.pbr.roughness;
    float4 mr_tex    = make_float4( 1.0f );
    if( hit_group_data->material_data.pbr.metallic_roughness_tex )
        // MR tex is (occlusion, roughness, metallic )
        mr_tex = sampleTexture<float4>( hit_group_data->material_data.pbr.metallic_roughness_tex, geom );
    roughness *= mr_tex.y;
    metallic *= mr_tex.z;

    //
    // Convert to material params
    //
    const float  F0         = 0.04f;
    const float3 diff_color = make_float3( base_color ) * ( 1.0f - F0 ) * ( 1.0f - metallic );
    const float3 spec_color = lerp( make_float3( F0 ), make_float3( base_color ), metallic );
    const float  alpha      = roughness * roughness;

    float3 result = make_float3( 0.0f );

    //
    // compute emission
    //

    float3 emissive_factor = hit_group_data->material_data.emissive_factor;
    float4 emissive_tex = make_float4( 1.0f );
    if( hit_group_data->material_data.emissive_tex )
        emissive_tex = sampleTexture<float4>( hit_group_data->material_data.emissive_tex, geom );
    result += emissive_factor * make_float3( emissive_tex );

    //
    // compute direct lighting
    //

    float3 N = geom.N;
    if( hit_group_data->material_data.normal_tex )
    {
        const int texcoord_idx = hit_group_data->material_data.normal_tex.texcoord;
        const float4 NN =
            2.0f * sampleTexture<float4>( hit_group_data->material_data.normal_tex, geom ) - make_float4( 1.0f );

        // Transform normal from texture space to rotated UV space.
        const float2 rotation = hit_group_data->material_data.normal_tex.texcoord_rotation;
        const float2 NN_proj  = make_float2( NN.x, NN.y );
        const float3 NN_trns  = make_float3( 
            dot( NN_proj, make_float2( rotation.y, -rotation.x ) ), 
            dot( NN_proj, make_float2( rotation.x,  rotation.y ) ),
            NN.z );

        N = normalize( NN_trns.x * normalize( geom.texcoord[texcoord_idx].dpdu ) + NN_trns.y * normalize( geom.texcoord[texcoord_idx].dpdv ) + NN_trns.z * geom.N );
    }

    // Flip normal to the side of the incomming ray
    if( dot( N, optixGetWorldRayDirection() ) > 0.f )
        N = -N;

    unsigned int depth = whitted::getPayloadDepth() + 1;

    for( int i = 0; i < whitted::params.lights.count; ++i )
    {
        Light light = whitted::params.lights[i];
        if( light.type == Light::Type::POINT )
        {
            if( depth < whitted::MAX_TRACE_DEPTH )
            {
                // TODO: optimize
                const float  L_dist  = length( light.point.position - geom.P );
                const float3 L       = ( light.point.position - geom.P ) / L_dist;
                const float3 V       = -normalize( optixGetWorldRayDirection() );
                const float3 H       = normalize( L + V );
                const float  N_dot_L = dot( N, L );
                const float  N_dot_V = dot( N, V );
                const float  N_dot_H = dot( N, H );
                const float  V_dot_H = dot( V, H );

                if( N_dot_L > 0.0f && N_dot_V > 0.0f )
                {
                    const float tmin        = 0.001f;           // TODO
                    const float tmax        = L_dist - 0.001f;  // TODO
                    const float attenuation = whitted::traceOcclusion( whitted::params.handle, geom.P, L, tmin, tmax );
                    if( attenuation > 0.f )
                    {
                        const float3 F     = whitted::schlick( spec_color, V_dot_H );
                        const float  G_vis = whitted::vis( N_dot_L, N_dot_V, alpha );
                        const float  D     = whitted::ggxNormal( N_dot_H, alpha );

                        const float3 diff = ( 1.0f - F ) * diff_color / M_PIf;
                        const float3 spec = F * G_vis * D;

                        result += light.point.color * attenuation * light.point.intensity * N_dot_L * ( diff + spec );
                    }
                }
            }
        }
        else if( light.type == Light::Type::AMBIENT )
        {
            result += light.ambient.color * make_float3( base_color );
        }
    }

    if( hit_group_data->material_data.alpha_mode == MaterialData::ALPHA_MODE_BLEND )
    {
        result *= base_color.w;
                
        if( depth < whitted::MAX_TRACE_DEPTH )
        {
            whitted::PayloadRadiance alpha_payload;
            alpha_payload.result = make_float3( 0.0f );
            alpha_payload.depth  = depth;
            whitted::traceRadiance( 
                whitted::params.handle, 
                optixGetWorldRayOrigin(), 
                optixGetWorldRayDirection(),
                optixGetRayTmax(),  // tmin
                1e16f,              // tmax
                &alpha_payload );

            result += alpha_payload.result * make_float3( 1.f - base_color.w );
        }
    }

    whitted::setPayloadResult( result );
}
