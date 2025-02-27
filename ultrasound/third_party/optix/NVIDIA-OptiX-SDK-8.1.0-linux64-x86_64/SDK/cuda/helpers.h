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

#include <vector_types.h>
#include <sutil/vec_math.h>


__forceinline__ __device__ float3 toSRGB( const float3& c )
{
    float  invGamma = 1.0f / 2.4f;
    float3 powed    = make_float3( powf( c.x, invGamma ), powf( c.y, invGamma ), powf( c.z, invGamma ) );
    return make_float3(
        c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f,
        c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
        c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f );
}

//__forceinline__ __device__ float dequantizeUnsigned8Bits( const unsigned char i )
//{
//    enum { N = (1 << 8) - 1 };
//    return min((float)i / (float)N), 1.f)
//}
__forceinline__ __device__ unsigned char quantizeUnsigned8Bits( float x )
{
    x = clamp( x, 0.0f, 1.0f );
    enum { N = (1 << 8) - 1, Np1 = (1 << 8) };
    return (unsigned char)min((unsigned int)(x * (float)Np1), (unsigned int)N);
}

__forceinline__ __device__ uchar4 make_color( const float3& c )
{
    // first apply gamma, then convert to unsigned char
    float3 srgb = toSRGB( clamp( c, 0.0f, 1.0f ) );
    return make_uchar4( quantizeUnsigned8Bits( srgb.x ), quantizeUnsigned8Bits( srgb.y ), quantizeUnsigned8Bits( srgb.z ), 255u );
}
__forceinline__ __device__ uchar4 make_color( const float4& c )
{
    return make_color( make_float3( c.x, c.y, c.z ) );
}

__forceinline__ __device__ float luminance( const float3& rgb )
{
    const float3 ntsc_luminance = { 0.30f, 0.59f, 0.11f };
    return dot( rgb, ntsc_luminance );
}

__forceinline__ __device__ float fresnel_schlick( const float cos_theta,
                                                  const float exponent = 5.0f,
                                                  const float minimum  = 0.0f,
                                                  const float maximum  = 1.0f )
{
    /**
      Clamp the result of the arithmetic due to floating point precision:
      the result should lie strictly within [minimum, maximum]
      return clamp(minimum + (maximum - minimum) * powf(1.0f - cos_theta, exponent),
                   minimum, maximum);
    */

    /** The max doesn't seem like it should be necessary, but without it you get
        annoying broken pixels at the center of reflective spheres where cos_theta ~ 1.
    */
    return clamp( minimum + ( maximum - minimum ) * powf( fmaxf( 0.0f, 1.0f - cos_theta ), exponent ), minimum, maximum );
}

__forceinline__ __device__ float3 fresnel_schlick( const float cos_theta, const float exponent, const float3& minimum, const float3& maximum )
{
    return make_float3( fresnel_schlick( cos_theta, exponent, minimum.x, maximum.x ),
                        fresnel_schlick( cos_theta, exponent, minimum.y, maximum.y ),
                        fresnel_schlick( cos_theta, exponent, minimum.z, maximum.z ) );
}

__forceinline__ __device__ bool refract( float3& r, const float3& i, const float3& n, const float ior )
{
    float3 nn       = n;
    float  negNdotV = dot( i, nn );
    float  eta;

    if( negNdotV > 0.0f )
    {
        eta      = ior;
        nn       = -n;
        negNdotV = -negNdotV;
    }
    else
    {
        eta = 1.f / ior;
    }

    const float k = 1.f - eta * eta * ( 1.f - negNdotV * negNdotV );

    if( k < 0.0f )
    {
        // Initialize this value, so that r always leaves this function initialized.
        r = make_float3( 0.f );
        return false;
    }
    else
    {
        r = normalize( eta * i - ( eta * negNdotV + sqrtf( k ) ) * nn );
        return true;
    }
}

__forceinline__ __device__ float3 exp( const float3& x )
{
    return make_float3( exp( x.x ), exp( x.y ), exp( x.z ) );
}

#define float3_as_args( u )                                                                                            \
    reinterpret_cast<unsigned int&>( ( u ).x ), reinterpret_cast<unsigned int&>( ( u ).y ),                            \
        reinterpret_cast<unsigned int&>( ( u ).z )
