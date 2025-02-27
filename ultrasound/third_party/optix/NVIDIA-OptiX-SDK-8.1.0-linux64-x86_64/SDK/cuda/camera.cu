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
#include <vector_types.h>

#include "helpers.h"
#include "random.h"
#include "whitted.h"

extern "C" {
__constant__ whitted::LaunchParams params;
}

extern "C" __global__ void __raygen__pinhole_camera()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const unsigned int image_index = params.width * idx.y + idx.x;
    unsigned int       seed        = tea<16>( image_index, params.subframe_index );

    // Subpixel jitter: send the ray through a different position inside the pixel each time,
    // to provide antialiasing. The center of each pixel is at fraction (0.5,0.5)
    float2 subpixel_jitter = params.subframe_index == 0 ? make_float2( 0.5f, 0.5f ) : make_float2( rnd( seed ), rnd( seed ) );

    float2 d = ( ( make_float2( idx.x, idx.y ) + subpixel_jitter ) / make_float2( params.width, params.height ) ) * 2.f - 1.f;

    float3 ray_origin    = params.eye;
    float3 ray_direction = normalize( d.x * params.U + d.y * params.V + params.W );

    whitted::PayloadRadiance prd;
    prd.importance = 1.f;
    prd.depth      = 0;

    optixTrace( params.handle, ray_origin, ray_direction, params.scene_epsilon, 1e16f, 0.0f, OptixVisibilityMask( 1 ),
                OPTIX_RAY_FLAG_NONE, whitted::RAY_TYPE_RADIANCE, whitted::RAY_TYPE_COUNT, whitted::RAY_TYPE_RADIANCE,
                float3_as_args( prd.result ), reinterpret_cast<unsigned int&>( prd.importance ),
                reinterpret_cast<unsigned int&>( prd.depth ) );

    float4 acc_val = params.accum_buffer[image_index];
    if( params.subframe_index > 0 )
    {
        acc_val = lerp( acc_val, make_float4( prd.result, 0.f ), 1.0f / static_cast<float>( params.subframe_index + 1 ) );
    }
    else
    {
        acc_val = make_float4( prd.result, 0.f );
    }
    params.frame_buffer[image_index] = make_color( acc_val );
    params.accum_buffer[image_index] = acc_val;
}
