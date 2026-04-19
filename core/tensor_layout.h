// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"

OIDN_NAMESPACE_BEGIN

  // Tensor memory layout
  enum class TensorLayout
  {
    Undefined,

    x,

    chw,
    Chw8c,  // blocked
    Chw16c, // blocked
    Chw32c, // blocked

    oihw,
    OIhw8i8o,       // blocked
    OIhw16i16o,     // blocked
    OIhw2o8i8o2i,   // blocked (Xe-HPG DPAS)
    OIhw8i16o2i,    // blocked (Xe-HPC DPAS)
    OIhw2o16i16o2i, // blocked (AMX-FP16)
    IOhw8i8o,       // blocked
    IOhw16i16o,     // blocked

    hwc,
    ohwi,
  };

  // -----------------------------------------------------------------------------------------------
#if !defined(OIDN_COMPILE_VULKAN_DEVICE)

  template<TensorLayout layout>
  struct TensorLayoutTraits;

  #define TENSOR_LAYOUT_TRAITS_BYTE_OFFSET_DECL_BEGIN(DIMENSION, TYPE)\
    template<>\
    struct TensorLayoutTraits<TensorLayout::TYPE>\
    {\
      template<typename T>\
      struct ByteOffset

  #define TENSOR_LAYOUT_TRAITS_BYTE_OFFSET_DECL_END()\
      ; /* end of ByteOffset struct */\
    } /* end of TensorLayoutTraits struct */

#else

  template<typename T>
  interface ITensorByteOffset1D
  {
    uint32_t operator()(int x);
  };

  template<typename T>
  interface ITensorByteOffset3D
  {
    uint32_t operator()(int c, int h, int w);
  };

  template<typename T>
  interface ITensorByteOffset4D
  {
    uint32_t operator()(int o, int i, int h, int w);
  };

  #define TENSOR_LAYOUT_TRAITS_BYTE_OFFSET_DECL_BEGIN(DIMENSION, TYPE)\
    template<typename T>\
    struct TensorByteOffset_##TYPE : ITensorByteOffset##DIMENSION<T>

  #define TENSOR_LAYOUT_TRAITS_BYTE_OFFSET_DECL_END()

#endif

  TENSOR_LAYOUT_TRAITS_BYTE_OFFSET_DECL_BEGIN(1D, x)
  {
      static constexpr oidn_constant uint32_t xByteStride = sizeof(T);

#if !defined(OIDN_COMPILE_VULKAN_DEVICE)
      ByteOffset() = default;
#endif

      oidn_host_device_inline uint32_t operator ()(int x) oidn_const_func
      {
        return uint32_t(x) * xByteStride;
      }
  }
  TENSOR_LAYOUT_TRAITS_BYTE_OFFSET_DECL_END();

  TENSOR_LAYOUT_TRAITS_BYTE_OFFSET_DECL_BEGIN(3D, chw)
  {
      static constexpr oidn_constant uint32_t wByteStride = sizeof(T);
      uint32_t hByteStride;
      uint32_t cByteStride;

#if !defined(OIDN_COMPILE_VULKAN_DEVICE)
      ByteOffset() = default;

      oidn_host_device_inline ByteOffset(int C, int H, int W)
      {
        hByteStride = uint32_t(W) * wByteStride;
        cByteStride = uint32_t(H) * hByteStride;
      }
#endif

      oidn_host_device_inline uint32_t operator ()(int c, int h, int w) oidn_const_func
      {
        return uint32_t(c) * cByteStride +
               uint32_t(h) * hByteStride +
               uint32_t(w) * wByteStride;
      }
  }
  TENSOR_LAYOUT_TRAITS_BYTE_OFFSET_DECL_END();

  TENSOR_LAYOUT_TRAITS_BYTE_OFFSET_DECL_BEGIN(3D, hwc)
  {
      static constexpr oidn_constant uint32_t cByteStride = sizeof(T);
      uint32_t wByteStride;
      uint32_t hByteStride;

#if !defined(OIDN_COMPILE_VULKAN_DEVICE)
      ByteOffset() = default;

      oidn_host_device_inline ByteOffset(int C, int H, int W)
      {
        wByteStride = uint32_t(C) * cByteStride;
        hByteStride = uint32_t(W) * wByteStride;
      }
#endif

      oidn_host_device_inline uint32_t operator ()(int c, int h, int w) oidn_const_func
      {
        return uint32_t(c) * cByteStride +
               uint32_t(h) * hByteStride +
               uint32_t(w) * wByteStride;
      }
  }
  TENSOR_LAYOUT_TRAITS_BYTE_OFFSET_DECL_END();

  template<typename T, int B>
  struct TensorByteOffsetChwBc
#if defined(OIDN_COMPILE_VULKAN_DEVICE)
    : ITensorByteOffset3D<T>
#endif
  {
    static constexpr oidn_constant int blockC = B; // block channels

    static constexpr oidn_constant uint32_t cByteStride = sizeof(T);
    static constexpr oidn_constant uint32_t wByteStride = B * cByteStride;
    uint32_t hByteStride;
    uint32_t CByteStride;

#if !defined(OIDN_COMPILE_VULKAN_DEVICE)
    TensorByteOffsetChwBc() = default;

    oidn_host_device_inline TensorByteOffsetChwBc(int C, int H, int W)
    {
      hByteStride = uint32_t(W) * wByteStride;
      CByteStride = uint32_t(H) * hByteStride;
    }
#endif

    oidn_host_device_inline uint32_t operator ()(int c, int h, int w) oidn_const_func
    {
      return uint32_t(c/B) * CByteStride +
             uint32_t(h)   * hByteStride +
             uint32_t(w)   * wByteStride +
             uint32_t(c%B) * cByteStride;
    }
  };

#if !defined(OIDN_COMPILE_VULKAN_DEVICE)

  template<>
  struct TensorLayoutTraits<TensorLayout::Chw8c>
  {
    template<typename T>
    using ByteOffset = TensorByteOffsetChwBc<T, 8>;
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::Chw16c>
  {
    template<typename T>
    using ByteOffset = TensorByteOffsetChwBc<T, 16>;
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::Chw32c>
  {
    template<typename T>
    using ByteOffset = TensorByteOffsetChwBc<T, 32>;
  };

#else

  template<typename T>
  typealias TensorByteOffset_Chw8c = TensorByteOffsetChwBc<T, 8>;

  template<typename T>
  typealias TensorByteOffset_Chw16c = TensorByteOffsetChwBc<T, 16>;

  template<typename T>
  typealias TensorByteOffset_Chw32c = TensorByteOffsetChwBc<T, 32>;

#endif

  TENSOR_LAYOUT_TRAITS_BYTE_OFFSET_DECL_BEGIN(4D, oihw)
  {
      static constexpr oidn_constant uint32_t wByteStride = sizeof(T);
      uint32_t hByteStride;
      uint32_t iByteStride;
      uint32_t oByteStride;

#if !defined(OIDN_COMPILE_VULKAN_DEVICE)
      ByteOffset() = default;

      oidn_host_device_inline ByteOffset(int O, int I, int H, int W)
      {
        hByteStride = uint32_t(W) * wByteStride;
        iByteStride = uint32_t(H) * hByteStride;
        oByteStride = uint32_t(I) * iByteStride;
      }
#endif

      oidn_host_device_inline uint32_t operator ()(int o, int i, int h, int w) oidn_const_func
      {
        return uint32_t(o) * oByteStride +
               uint32_t(i) * iByteStride +
               uint32_t(h) * hByteStride +
               uint32_t(w) * wByteStride;
      }
  }
  TENSOR_LAYOUT_TRAITS_BYTE_OFFSET_DECL_END();

  template<typename T, int B>
  struct TensorByteOffsetOIhwBiBo
#if defined(OIDN_COMPILE_VULKAN_DEVICE)
    : ITensorByteOffset4D<T>
#endif
  {
    static constexpr oidn_constant int blockC = B; // block channels

    static constexpr oidn_constant uint32_t BoByteStride = sizeof(T);
    static constexpr oidn_constant uint32_t BiByteStride = B * BoByteStride;
    static constexpr oidn_constant uint32_t wByteStride  = B * BiByteStride;
    uint32_t hByteStride;
    uint32_t IByteStride;
    uint32_t OByteStride;

#if !defined(OIDN_COMPILE_VULKAN_DEVICE)
    TensorByteOffsetOIhwBiBo() = default;

    oidn_host_device_inline TensorByteOffsetOIhwBiBo(int O, int I, int H, int W)
    {
      hByteStride = uint32_t(W)     * wByteStride;
      IByteStride = uint32_t(H)     * hByteStride;
      OByteStride = uint32_t(I / B) * IByteStride;
    }
#endif

    oidn_host_device_inline uint32_t operator ()(int o, int i, int h, int w) oidn_const_func
    {
      return uint32_t(o / B) * OByteStride  +
             uint32_t(i / B) * IByteStride  +
             uint32_t(h)     * hByteStride  +
             uint32_t(w)     * wByteStride  +
             uint32_t(i % B) * BiByteStride +
             uint32_t(o % B) * BoByteStride;
    }
  };

#if !defined(OIDN_COMPILE_VULKAN_DEVICE)

  template<>
  struct TensorLayoutTraits<TensorLayout::OIhw8i8o>
  {
    template<typename T>
    using ByteOffset = TensorByteOffsetOIhwBiBo<T, 8>;
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::OIhw16i16o>
  {
    template<typename T>
    using ByteOffset = TensorByteOffsetOIhwBiBo<T, 16>;
  };

#else

  template<typename T>
  typealias TensorByteOffset_OIhw8i8o= TensorByteOffsetOIhwBiBo<T, 8>;

  template<typename T>
  typealias TensorByteOffset_OIhw16i16o= TensorByteOffsetOIhwBiBo<T, 16>;

#endif

  template<typename T, int B>
  struct TensorByteOffsetIOhwBiBo
#if defined(OIDN_COMPILE_VULKAN_DEVICE)
    : ITensorByteOffset4D<T>
#endif
  {
    static constexpr oidn_constant int blockC = B; // block channels

    static constexpr oidn_constant uint32_t BoByteStride = sizeof(T);
    static constexpr oidn_constant uint32_t BiByteStride = B * BoByteStride;
    static constexpr oidn_constant uint32_t wByteStride  = B * BiByteStride;
    uint32_t hByteStride;
    uint32_t OByteStride;
    uint32_t IByteStride;

#if !defined(OIDN_COMPILE_VULKAN_DEVICE)
    TensorByteOffsetIOhwBiBo() = default;

    oidn_host_device_inline TensorByteOffsetIOhwBiBo(int O, int I, int H, int W)
    {
      hByteStride = uint32_t(W)     * wByteStride;
      OByteStride = uint32_t(H)     * hByteStride;
      IByteStride = uint32_t(O / B) * OByteStride;
    }
#endif

    oidn_host_device_inline uint32_t operator ()(int o, int i, int h, int w) oidn_const_func
    {
      return uint32_t(i / B) * IByteStride  +
             uint32_t(o / B) * OByteStride  +
             uint32_t(h)     * hByteStride  +
             uint32_t(w)     * wByteStride  +
             uint32_t(i % B) * BiByteStride +
             uint32_t(o % B) * BoByteStride;
    }
  };

#if !defined(OIDN_COMPILE_VULKAN_DEVICE)

  template<>
  struct TensorLayoutTraits<TensorLayout::IOhw8i8o>
  {
    template<typename T>
    using ByteOffset = TensorByteOffsetIOhwBiBo<T, 8>;
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::IOhw16i16o>
  {
    template<typename T>
    using ByteOffset = TensorByteOffsetIOhwBiBo<T, 16>;
  };

#else

  template<typename T>
  typealias TensorByteOffset_IOhw8i8o = TensorByteOffsetIOhwBiBo<T, 8>;

  template<typename T>
  typealias TensorByteOffset_IOhw16i16o = TensorByteOffsetIOhwBiBo<T, 16>;

#endif

#if !defined(OIDN_COMPILE_VULKAN_DEVICE)

  template<typename T, int P, int Q, int R, int S>
  struct TensorByteOffsetOIhwPoQiRoSi
  {
    static_assert(P * R == Q * S, "invalid tensor layout parameters");

    static constexpr oidn_constant int B = P * R;
    static constexpr oidn_constant int blockC = B; // block channels

    static constexpr oidn_constant uint32_t SiByteStride = sizeof(T);
    static constexpr oidn_constant uint32_t RoByteStride = S * SiByteStride;
    static constexpr oidn_constant uint32_t QiByteStride = R * RoByteStride;
    static constexpr oidn_constant uint32_t PoByteStride = Q * QiByteStride;
    static constexpr oidn_constant uint32_t wByteStride  = P * PoByteStride;
    uint32_t hByteStride;
    uint32_t IByteStride;
    uint32_t OByteStride;

    TensorByteOffsetOIhwPoQiRoSi() = default;

    oidn_host_device_inline TensorByteOffsetOIhwPoQiRoSi(int O, int I, int H, int W)
    {
      hByteStride = uint32_t(W)     * wByteStride;
      IByteStride = uint32_t(H)     * hByteStride;
      OByteStride = uint32_t(I / B) * IByteStride;
    }

    oidn_host_device_inline uint32_t operator ()(int o, int i, int h, int w) const
    {
      return uint32_t(o / B)     * OByteStride  +
             uint32_t(i / B)     * IByteStride  +
             uint32_t(h)         * hByteStride  +
             uint32_t(w)         * wByteStride  +
             uint32_t(o % B / R) * PoByteStride +
             uint32_t(i % B / S) * QiByteStride +
             uint32_t(o % R)     * RoByteStride +
             uint32_t(i % S)     * SiByteStride;
    }
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::OIhw2o8i8o2i>
  {
    template<typename T>
    using ByteOffset = TensorByteOffsetOIhwPoQiRoSi<T, 2, 8, 8, 2>;
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::OIhw8i16o2i>
  {
    template<typename T>
    using ByteOffset = TensorByteOffsetOIhwPoQiRoSi<T, 1, 8, 16, 2>;
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::OIhw2o16i16o2i>
  {
    template<typename T>
    using ByteOffset = TensorByteOffsetOIhwPoQiRoSi<T, 2, 16, 16, 2>;
  };

#endif

  TENSOR_LAYOUT_TRAITS_BYTE_OFFSET_DECL_BEGIN(4D, ohwi)
  {
      static constexpr oidn_constant uint32_t iByteStride = sizeof(T);
      uint32_t wByteStride;
      uint32_t hByteStride;
      uint32_t oByteStride;

#if !defined(OIDN_COMPILE_VULKAN_DEVICE)
      ByteOffset() = default;

      oidn_host_device_inline ByteOffset(int O, int I, int H, int W)
      {
        wByteStride = uint32_t(I) * iByteStride;
        hByteStride = uint32_t(W) * wByteStride;
        oByteStride = uint32_t(H) * hByteStride;
      }
#endif

      oidn_host_device_inline uint32_t operator ()(int o, int i, int h, int w) oidn_const_func
      {
        return uint32_t(o) * oByteStride +
               uint32_t(i) * iByteStride +
               uint32_t(h) * hByteStride +
               uint32_t(w) * wByteStride;
      }
  }
  TENSOR_LAYOUT_TRAITS_BYTE_OFFSET_DECL_END();

#if !defined(OIDN_COMPILE_VULKAN_DEVICE)
  template<typename T, TensorLayout layout>
  using TensorByteOffset1D = typename TensorLayoutTraits<layout>::template ByteOffset<T>;

  template<typename T, TensorLayout layout>
  using TensorByteOffset3D = typename TensorLayoutTraits<layout>::template ByteOffset<T>;

  template<typename T, TensorLayout layout>
  using TensorByteOffset4D = typename TensorLayoutTraits<layout>::template ByteOffset<T>;
#endif
  // -----------------------------------------------------------------------------------------------

#if !defined(OIDN_COMPILE_METAL_DEVICE) && !defined(OIDN_COMPILE_VULKAN_DEVICE)
  struct TensorLayoutInfo
  {
    int rank;
    int blockC;
  };

  // Returns information about the tensor layout
  oidn_inline TensorLayoutInfo getTensorLayoutInfo(TensorLayout layout)
  {
    switch (layout)
    {
    case TensorLayout::x:
      return {1, 1};
    case TensorLayout::chw:
    case TensorLayout::hwc:
      return {3, 1};
    case TensorLayout::Chw8c:
      return {3, 8};
    case TensorLayout::Chw16c:
      return {3, 16};
    case TensorLayout::Chw32c:
      return {3, 32};
    case TensorLayout::oihw:
    case TensorLayout::ohwi:
      return {4, 1};
    case TensorLayout::IOhw8i8o:
    case TensorLayout::OIhw8i8o:
      return {4, 8};
    case TensorLayout::IOhw16i16o:
    case TensorLayout::OIhw16i16o:
    case TensorLayout::OIhw2o8i8o2i:
    case TensorLayout::OIhw8i16o2i:
      return {4, 16};
    case TensorLayout::OIhw2o16i16o2i:
      return {4, 32};
    default:
      throw std::invalid_argument("invalid tensor layout");
    }
  }

  inline std::ostream& operator <<(std::ostream& sm, TensorLayout layout)
  {
    switch (layout)
    {
    case TensorLayout::x:              sm << "x";              break;
    case TensorLayout::chw:            sm << "chw";            break;
    case TensorLayout::Chw8c:          sm << "Chw8c";          break;
    case TensorLayout::Chw16c:         sm << "Chw16c";         break;
    case TensorLayout::Chw32c:         sm << "Chw32c";         break;
    case TensorLayout::oihw:           sm << "oihw";           break;
    case TensorLayout::OIhw8i8o:       sm << "OIhw8i8o";       break;
    case TensorLayout::OIhw16i16o:     sm << "OIhw16i16o";     break;
    case TensorLayout::OIhw2o8i8o2i:   sm << "OIhw2o8i8o2i";   break;
    case TensorLayout::OIhw8i16o2i:    sm << "OIhw8i16o2i";    break;
    case TensorLayout::OIhw2o16i16o2i: sm << "OIhw2o16i16o2i"; break;
    case TensorLayout::IOhw8i8o:       sm << "IOhw8i8o";       break;
    case TensorLayout::IOhw16i16o:     sm << "IOhw16i16o";     break;
    case TensorLayout::hwc:            sm << "hwc";            break;
    case TensorLayout::ohwi:           sm << "ohwi";           break;
    default:                           sm << "?";              break;
    }

    return sm;
  }
#endif

OIDN_NAMESPACE_END
