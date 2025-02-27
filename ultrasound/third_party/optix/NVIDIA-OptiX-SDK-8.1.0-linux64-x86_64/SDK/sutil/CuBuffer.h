/*
 * Copyright (c) 2020 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <cuda.h>
#include <initializer_list>
#include <vector>

#ifndef __CUDACC__
#ifndef CUDA_CHECK
#include <sutil/Exception.h>
#endif
#else
#define CUDA_CHECK(x) x
#endif


template <typename T = char>
class CuBuffer
{
public:
    typedef T ElemType;
    typedef const T* constTPtr;

    CuBuffer( size_t count = 0 )
    {
        alloc( count );
    }
    CuBuffer( size_t count, const T* data )
    {
        alloc( count );
        upload( data );
    }
    CuBuffer( std::initializer_list<T> data )
    {
        alloc( data.size() );
        upload( data.begin() );
    }
    CuBuffer( const std::vector<T>& data )
    {
        alloc( data.size() );
        upload( data.data() );
    }
    ~CuBuffer()
    {
        free();
    }


    CuBuffer(CuBuffer&& other)
    {
        swap( other );
    }

    CuBuffer& operator=(CuBuffer&& other)
    {
        swap( other );
        return *this;
    }

    void swap( CuBuffer& other )  // nothrow
    {
        // enable ADL (not necessary in our case, but good practice)
        using std::swap;

        // by swapping the members of two objects,
        // the two objects are effectively swapped
        swap(m_ptr, other.m_ptr);
        swap(m_count, other.m_count);
        swap(m_allocCount, other.m_allocCount);
    }

protected:
    CuBuffer(const CuBuffer&) = delete;
    CuBuffer& operator=(const CuBuffer&) = delete;

public:
    void alloc( size_t count )
    {
        if( m_allocCount == count )
        {
            m_count = count;
            return;
        }
        free();
        m_allocCount = m_count = count;
        if( m_allocCount )
        {
            CUDA_CHECK( cudaMalloc( (void**)&m_ptr, m_allocCount * sizeof( T ) ) );
        }
    }
    void allocIfRequired( size_t count )
    {
        if( count <= m_allocCount )
        {
            m_count = count;
            return;
        }
        alloc( count );
    }

    T* get() { return m_ptr; }
    const T* get() const { return m_ptr; }
    T* get( size_t index ) { return m_ptr + index; }
    const T* get( size_t index ) const { return m_ptr + index; }
    // allows for usage as l-value, similar to "&buffer.get()", however, this solution avoids gcc warning "-Wstrict-aliasing" and clang warning "-Wreturn-stack-address"
    const constTPtr* getAsArray() const { return &m_ptr; }

    CUdeviceptr getCU() { return reinterpret_cast<CUdeviceptr>( m_ptr ); }
    const CUdeviceptr getCU() const { return reinterpret_cast<const CUdeviceptr>( m_ptr ); }
    CUdeviceptr getCU( size_t index ) { return reinterpret_cast<CUdeviceptr>( m_ptr + index ); }
    const CUdeviceptr getCU( size_t index ) const { return reinterpret_cast<const CUdeviceptr>( m_ptr + index ); }
    // allows for usage as l-value, similar to "&buffer.get()", however, this solution avoids gcc warning "-Wstrict-aliasing" and clang warning "-Wreturn-stack-address"
    const CUdeviceptr* getCUAsArray() const { return reinterpret_cast<const CUdeviceptr*>( &m_ptr ); }

    void free()
    {
        m_count      = 0;
        m_allocCount = 0;
        if( m_ptr )
            CUDA_CHECK( cudaFree( m_ptr ) );
        m_ptr = nullptr;
    }
    CUdeviceptr release()
    {
        m_count             = 0;
        m_allocCount        = 0;
        CUdeviceptr current = reinterpret_cast<CUdeviceptr>( m_ptr );
        m_ptr               = nullptr;
        return current;
    }
    void allocAndUpload( std::initializer_list<T> data )
    {
        alloc( data.size() );
        upload( data );
    }
    void allocAndUpload( const std::vector<T>& data )
    {
        alloc( data.size() );
        upload( data.data() );
    }
    void allocAndUpload( size_t count, const T* data )
    {
        alloc( count );
        upload( data );
    }
    void upload( const T* data )
    {
        CUDA_CHECK( cudaMemcpy( m_ptr, data, m_count * sizeof( T ), cudaMemcpyHostToDevice ) );
    }
    void upload( std::initializer_list<T> data )
    {
        assert( data.size() <= m_count );
        CUDA_CHECK( cudaMemcpy( m_ptr, data.begin(), data.size() * sizeof( T ), cudaMemcpyHostToDevice ) );
    }
    void uploadSub( size_t count, size_t offset, const T* data )
    {
        assert( count + offset <= m_count );
        CUDA_CHECK( cudaMemcpy( m_ptr + offset, data, count * sizeof( T ), cudaMemcpyHostToDevice ) );
    }
    void uploadSub( size_t offset, std::initializer_list<T> data )
    {
        assert( data.size() + offset <= m_count );
        CUDA_CHECK( cudaMemcpy( m_ptr + offset, data.begin(), data.size() * sizeof( T ), cudaMemcpyHostToDevice ) );
    }
    void uploadAsync( const T& data, CUstream stream = 0 )
    {
        CUDA_CHECK( cudaMemcpyAsync( m_ptr, &data, sizeof( T ), cudaMemcpyHostToDevice, stream ) );
    }
    void uploadAsync( const T* data, CUstream stream = 0 )
    {
        CUDA_CHECK( cudaMemcpyAsync( m_ptr, data, m_count * sizeof( T ), cudaMemcpyHostToDevice, stream ) );
    }
    void uploadAsync( std::initializer_list<T> data, CUstream stream = 0 )
    {
        assert( data.size() <= m_count );
        CUDA_CHECK( cudaMemcpyAsync( m_ptr, data.begin(), data.size() * sizeof( T ), cudaMemcpyHostToDevice, stream ) );
    }
    void uploadSubAsync( size_t count, size_t offset, const T* data, CUstream stream = 0 )
    {
        assert( count + offset <= m_count );
        CUDA_CHECK( cudaMemcpyAsync( m_ptr + offset, data, count * sizeof( T ), cudaMemcpyHostToDevice, stream ) );
    }
    void uploadSubAsync( size_t offset, std::initializer_list<T> data, CUstream stream = 0 )
    {
        assert( data.size() + offset <= m_count );
        CUDA_CHECK( cudaMemcpyAsync( m_ptr + offset, data.begin(), data.size() * sizeof( T ), cudaMemcpyHostToDevice, stream ) );
    }

    void download( T* data ) const
    {
        CUDA_CHECK( cudaMemcpy( data, m_ptr, m_count * sizeof( T ), cudaMemcpyDeviceToHost ) );
    }
    std::vector<T> download() const
    {
        std::vector<T> result( m_count );
        download( result.data() );
        return result;
    }
    void downloadSub( size_t count, size_t offset, T* data ) const
    {
        assert( count + offset <= m_count );
        CUDA_CHECK( cudaMemcpy( data, m_ptr + offset, count * sizeof( T ), cudaMemcpyDeviceToHost ) );
    }
    void downloadAsync( T* data, CUstream stream = 0 ) const
    {
        CUDA_CHECK( cudaMemcpyAsync( data, m_ptr, m_count * sizeof( T ), cudaMemcpyDeviceToHost, stream ) );
    }
    void downloadSubAsync( size_t count, size_t offset, T* data, CUstream stream = 0  ) const
    {
        assert( count + offset <= m_count );
        CUDA_CHECK( cudaMemcpyAsync( data, m_ptr + offset, count * sizeof( T ), cudaMemcpyDeviceToHost, stream ) );
    }

    void memset( char data )
    {
        CUDA_CHECK( cudaMemset( m_ptr, data, m_count * sizeof( T ) ) );
    }
    template<typename U>
    void copy( const CuBuffer<U>& other )
    {
        assert( other.byteSize() == byteSize() );
        CUDA_CHECK( cudaMemcpy( m_ptr, other.get(), m_count * sizeof( T ), cudaMemcpyDeviceToDevice ) );
    }
    void copy( const T* data )
    {
        CUDA_CHECK( cudaMemcpy( m_ptr, data, m_count * sizeof( T ), cudaMemcpyDeviceToDevice ) );
    }

    size_t size() const { return m_count; }
    size_t byteSize() const { return m_count * sizeof( T ); }
    size_t capacity() const { return m_allocCount; }
    size_t capacityByteSize() const { return m_allocCount * sizeof( T ); }
    bool   empty() const { return m_count == 0; }

    static unsigned int stride() { return sizeof( T ); }

public:
    static size_t roundUp( size_t i, size_t alignment )
    {
        return ( ( i + alignment - 1 ) / alignment ) * alignment;
    }
    static size_t pool( size_t sizeA, size_t sizeB, size_t alignmentB, size_t& offsetB )
    {
        offsetB = roundUp( sizeA, alignmentB );
        return offsetB + sizeB;
    }
    static size_t pool( size_t sizeA, size_t sizeB, size_t alignmentB, size_t& offsetB, size_t sizeC, size_t alignmentC, size_t& offsetC )
    {
        offsetC = roundUp( pool(sizeA, sizeB, alignmentB, offsetB), alignmentC );
        return offsetC + sizeC;
    }

private:
    size_t m_count      = 0;
    size_t m_allocCount = 0;
    T*     m_ptr        = nullptr;
};

#ifdef __CUDACC__
#undef CUDA_CHECK
#endif
