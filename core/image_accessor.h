// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vec.h"

// ISPC forward declarations
namespace ispc
{
  struct ImageAccessor;
}

OIDN_NAMESPACE_BEGIN

  template<typename T, typename U>
  oidn_host_device_inline T cast_ptr(U ptr)
  {
#if !defined(OIDN_COMPILE_VULKAN_DEVICE)
    return static_cast<T>(ptr);
#else
    return reinterpret<T>(ptr);
#endif
  }

#if defined(OIDN_COMPILE_VULKAN_HOST)
  static_assert(sizeof(size_t) == 8, "size_t in vulkan shader code is 64-bit");
#endif

#if defined(OIDN_COMPILE_VULKAN_DEVICE)
  typealias char = uint8_t;
  typealias size_t = uint64_t;

  // cast from __BuiltinFloatingPointType to float
  extension float
  {
   __init(__BuiltinFloatingPointType v) { this = __realCast<float>(v); }
  }

  // cast from __BuiltinFloatingPointType to half
  extension half
  {
   __init(__BuiltinFloatingPointType v) { this = __realCast<half>(v); }
  }
#endif

  struct ImageAccessor
  {
    oidn_global char* ptr;
    size_t hByteStride; // row stride in number of bytes
    size_t wByteStride; // pixel stride in number of bytes
    DataType dataType;  // data type
    int C, H, W;        // channels (1-3), height, width

    oidn_host_device_inline size_t getByteOffset(int h, int w) oidn_const_func
    {
      return size_t(h) * hByteStride + size_t(w) * wByteStride;
    }

    template<typename T = float>
    oidn_host_device_inline vec3<T> get3(int h, int w) oidn_const_func
      WHERE(T, __BuiltinFloatingPointType)
    {
      oidn_global_readonly_ptr(void) pixelPtr = cast_ptr<oidn_global_readonly_ptr(void)>(ptr + getByteOffset(h, w));
      if (dataType == DataType::Float32)
      {
        oidn_global_readonly_ptr(float) pixel = cast_ptr<oidn_global_readonly_ptr(float)>(pixelPtr);
        if (C == 3)
          return vec3<T>(T(pixel[0]), T(pixel[1]), T(pixel[2]));
        else if (C == 2)
          return vec3<T>(T(pixel[0]), T(pixel[1]), T(pixel[1]));
        else // if (C == 1)
          return vec3<T>(T(pixel[0]), T(pixel[0]), T(pixel[0]));
      }
      else // if (dataType == DataType::Float16)
      {
        oidn_global_readonly_ptr(half) pixel = cast_ptr<oidn_global_readonly_ptr(half)>(pixelPtr);
        if (C == 3)
          return vec3<T>(T(pixel[0]), T(pixel[1]), T(pixel[2]));
        else if (C == 2)
          return vec3<T>(T(pixel[0]), T(pixel[1]), T(pixel[1]));
        else // if (C == 1)
          return vec3<T>(T(pixel[0]), T(pixel[0]), T(pixel[0]));
      }
    }

    template<typename T>
    oidn_host_device_inline void set3(int h, int w, vec3<T> value) oidn_const_func
      WHERE(T, __BuiltinFloatingPointType)
    {
      oidn_global void* pixelPtr = cast_ptr<oidn_global void*>(ptr + getByteOffset(h, w));
      if (dataType == DataType::Float32)
      {
        oidn_global float* pixel = cast_ptr<oidn_global float*>(pixelPtr);
        if (C == 3)
        {
          pixel[0] = float(value.x);
          pixel[1] = float(value.y);
          pixel[2] = float(value.z);
        }
        else if (C == 2)
        {
          pixel[0] = float(value.x);
          pixel[1] = float(value.y);
        }
        else // if (C == 1)
          pixel[0] = float(value.x);
      }
      else // if (dataType == DataType::Float16)
      {
        oidn_global half* pixel = cast_ptr<oidn_global half*>(pixelPtr);
        if (C == 3)
        {
          pixel[0] = half(value.x);
          pixel[1] = half(value.y);
          pixel[2] = half(value.z);
        }
        else if (C == 2)
        {
          pixel[0] = half(value.x);
          pixel[1] = half(value.y);
        }
        else // if (C == 1)
          pixel[0] = half(value.x);
      }
    }
  };

OIDN_NAMESPACE_END