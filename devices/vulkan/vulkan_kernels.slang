// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#define OIDN_COMPILE_VULKAN_DEVICE 1

#include "core/kernel.h"
#include "core/color.h"
#include "core/image_accessor.h"
#include "core/tensor_accessor.h"
#include "devices/gpu/gpu_input_process.h"
#include "devices/gpu/gpu_output_process.h"
#include "devices/gpu/gpu_image_copy.h"
#include "devices/gpu/gpu_autoexposure.h"
#include "devices/vulkan/vulkan_conv.h"

namespace oidn
{
  template<typename T, TensorLayout layout>
  typealias TensorByteOffset1D = TensorByteOffset_x<T>;

  template<typename T, TensorLayout layout>
  typealias TensorByteOffset3D = TensorByteOffset_hwc<T>;

  template<typename T, TensorLayout layout>
  typealias TensorByteOffset4D = TensorByteOffset_ohwi<T>;
}

struct SystemValues1D
{
  uint globalID : SV_DispatchThreadID;
  uint localID : SV_GroupThreadID;
  uint groupID : SV_GroupID;
  uint localIndex : SV_GroupIndex;
};

struct SystemValues2D
{
  uint2 globalID : SV_DispatchThreadID;
  uint2 localID : SV_GroupThreadID;
  uint2 groupID : SV_GroupID;
  uint localIndex : SV_GroupIndex;
};

struct SystemValues3D
{
  uint3 globalID : SV_DispatchThreadID;
  uint3 localID : SV_GroupThreadID;
  uint3 groupID : SV_GroupID;
  uint localIndex : SV_GroupIndex;
};

[vk::constant_id(0)]
const uint localSizeX = 1;

[vk::constant_id(1)]
const uint localSizeY = 1;

[vk::constant_id(2)]
const uint localSizeZ = 1;

[shader("compute")]
[numthreads(localSizeX, localSizeY, localSizeZ)]
void inputProcess_f16_hwc_3(SystemValues2D sv,
                            uniform oidn::GPUInputProcessKernel<half, oidn::TensorLayout::hwc, 3> params)
{
  oidn::WorkGroupItem<2> it = { sv.globalID, sv.localID, sv.groupID, sv.localIndex };
  params(it);
}

[shader("compute")]
[numthreads(localSizeX, localSizeY, localSizeZ)]
void inputProcess_f16_hwc_6(SystemValues2D sv,
                            uniform oidn::GPUInputProcessKernel<half, oidn::TensorLayout::hwc, 6> params)
{
  oidn::WorkGroupItem<2> it = { sv.globalID, sv.localID, sv.groupID, sv.localIndex };
  params(it);
}

[shader("compute")]
[numthreads(localSizeX, localSizeY, localSizeZ)]
void inputProcess_f16_hwc_9(SystemValues2D sv,
                            uniform oidn::GPUInputProcessKernel<half, oidn::TensorLayout::hwc, 9> params)
{
  oidn::WorkGroupItem<2> it = { sv.globalID, sv.localID, sv.groupID, sv.localIndex };
  params(it);
}

[shader("compute")]
[numthreads(localSizeX, localSizeY, localSizeZ)]
void outputProcess(SystemValues2D sv,
                   uniform uint32_t globalSize0,
                   uniform uint32_t globalSize1,
                   uniform oidn::GPUOutputProcessKernel<half, oidn::TensorLayout::hwc> params)
{
  oidn::WorkItem<2> it = { sv.globalID };
  if (it.getGlobalID<0>() >= (int)globalSize0 || it.getGlobalID<1>() >= (int)globalSize1) return;
  params(it);
}

[shader("compute")]
[numthreads(localSizeX, localSizeY, localSizeZ)]
void imageCopy(SystemValues2D sv,
               uniform uint32_t globalSize0,
               uniform uint32_t globalSize1,
               uniform oidn::GPUImageCopyKernel params)
{
  oidn::WorkItem<2> it = { sv.globalID };
  if (it.getGlobalID<0>() >= (int)globalSize0 || it.getGlobalID<1>() >= (int)globalSize1) return;
  params(it);
}

// we do not need this declaration after
// https://github.com/shader-slang/slang/issues/10940 is fixed
static const uint32_t maxBinSize = 16; //oidn::AutoexposureParams::maxBinSize;

groupshared oidn::GPUAutoexposureDownsampleKernel<maxBinSize>::Local autoexposureDownsampleLocal;

[shader("compute")]
[numthreads(localSizeX, localSizeY, localSizeZ)]
void autoexposureDownsample(SystemValues2D sv,
                            uniform oidn::GPUAutoexposureDownsampleKernel<maxBinSize> params)
{
  oidn::WorkGroupItem<2> it = { sv.globalID, sv.localID, sv.groupID, sv.localIndex };
  params(it, __getAddress(autoexposureDownsampleLocal));
}

groupshared oidn::GPUAutoexposureReduceKernel<1024>::Local autoexposureReduceLocal;

[shader("compute")]
[numthreads(localSizeX, localSizeY, localSizeZ)]
void autoexposureReduce_1024(SystemValues1D sv,
                             uniform oidn::GPUAutoexposureReduceKernel<1024> params)
{
  oidn::WorkGroupItem<1> it = { sv.globalID, sv.localID, sv.groupID, sv.localIndex };
  params(it, __getAddress(autoexposureReduceLocal));
}

groupshared oidn::GPUAutoexposureReduceFinalKernel<1024>::Local autoexposureReduceFinalLocal;

[shader("compute")]
[numthreads(localSizeX, localSizeY, localSizeZ)]
void autoexposureReduceFinal_1024(SystemValues1D sv,
                                  uniform oidn::GPUAutoexposureReduceFinalKernel<1024> params)
{
  oidn::WorkGroupItem<1> it = { sv.globalID, sv.localID, sv.groupID, sv.localIndex };
  params(it, __getAddress(autoexposureReduceFinalLocal));
}

template <typename T>
T relu(T x) where T : IFloat
{
  return max(x, T(0));
}

[shader("compute")]
[numthreads(localSizeX, localSizeY, localSizeZ)]
void conv(SystemValues3D sv,
          uniform uint32_t globalSize0,
          uniform uint32_t globalSize1,
          uniform uint32_t globalSize2,
          uniform oidn::VulkanConvKernel params)
{
  // 0, 1, 2 = h, w, c
  oidn::WorkItem<3> it = { sv.globalID };
  if (it.getGlobalID<0>() >= (int)globalSize0 ||
      it.getGlobalID<1>() >= (int)globalSize1 ||
      it.getGlobalID<2>() >= (int)globalSize2) return;

  const int kH = 3;
  const int kW = 3;
  const int pad = 1;

  const int srcH = params.src.H;
  const int srcW = params.src.W;
  const int srcC = params.src.C;

  const int o = it.getGlobalID<2>();
  const uint32_t postOp = params.postOp;

  if (postOp == 2)
  {
    // Upsample: src-driven dispatch; globalSize = {srcH, srcW, paddedC}
    const int h = it.getGlobalID<0>();
    const int w = it.getGlobalID<1>();

    float acc = float(params.bias[o, h, w]);
    for (int i = 0; i < srcC; ++i)
      for (int kh = 0; kh < kH; ++kh)
        for (int kw = 0; kw < kW; ++kw)
        {
          const int sh = h + kh - pad;
          const int sw = w + kw - pad;
          if (sh >= 0 && sh < srcH && sw >= 0 && sw < srcW)
            acc += float(params.src[i, sh, sw]) * float(params.weight[o, i, kh, kw]);
        }

    if (params.relu)
      acc = relu(acc);

    const half val = half(acc);
    params.dst[o, h*2+0, w*2+0] = val;
    params.dst[o, h*2+0, w*2+1] = val;
    params.dst[o, h*2+1, w*2+0] = val;
    params.dst[o, h*2+1, w*2+1] = val;
  }
  else if (postOp == 1)
  {
    // Pool: dst-driven dispatch; globalSize = {dstH, dstW, paddedC}
    const int dh = it.getGlobalID<0>();
    const int dw = it.getGlobalID<1>();

    float maxVal = -FLT_MAX;
    for (int ph = 0; ph < 2; ++ph)
      for (int pw = 0; pw < 2; ++pw)
      {
        const int h = dh * 2 + ph;
        const int w = dw * 2 + pw;

        float acc = float(params.bias[o, dh, dw]);
        for (int i = 0; i < srcC; ++i)
          for (int kh = 0; kh < kH; ++kh)
            for (int kw = 0; kw < kW; ++kw)
            {
              const int sh = h + kh - pad;
              const int sw = w + kw - pad;
              if (sh >= 0 && sh < srcH && sw >= 0 && sw < srcW)
                acc += float(params.src[i, sh, sw]) * float(params.weight[o, i, kh, kw]);
            }

        if (params.relu)
          acc = relu(acc);
        if (acc > maxVal) maxVal = acc;
      }

    params.dst[o, dh, dw] = half(maxVal);
  }
  else
  {
    // None: dst-driven dispatch; globalSize = {dstH, dstW, paddedC}
    const int h = it.getGlobalID<0>();
    const int w = it.getGlobalID<1>();

    float acc = float(params.bias[o, h, w]);
    for (int i = 0; i < srcC; ++i)
      for (int kh = 0; kh < kH; ++kh)
        for (int kw = 0; kw < kW; ++kw)
        {
          const int sh = h + kh - pad;
          const int sw = w + kw - pad;
          if (sh >= 0 && sh < srcH && sw >= 0 && sw < srcW)
            acc += float(params.src[i, sh, sw]) * float(params.weight[o, i, kh, kw]);
        }

    if (params.relu)
      acc = relu(acc);
    params.dst[o, h, w] = half(acc);
  }
}
