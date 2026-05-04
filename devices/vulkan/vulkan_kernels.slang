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

static const int convTileH = 8;
static const int convTileW = 8;
static const int convTileO = 4;
static const int convTileI = 16;
static const int convKernelH = 3;
static const int convKernelW = 3;
static const int convPad = 1;
static const int convSrcTileH = convTileH + convKernelH - 1;
static const int convSrcTileW = convTileW + convKernelW - 1;
static const int convPoolSrcTileH = convTileH * 2 + convKernelH - 1;
static const int convPoolSrcTileW = convTileW * 2 + convKernelW - 1;

int getConvSrcTileIndex(int th, int tw, int i)
{
  return (th * convSrcTileW + tw) * convTileI + i;
}

int getConvPoolSrcTileIndex(int th, int tw, int i)
{
  return (th * convPoolSrcTileW + tw) * convTileI + i;
}

int getConvWeightTileIndex(int lo, int i, int kh, int kw)
{
  return ((lo * convTileI + i) * convKernelH + kh) * convKernelW + kw;
}

struct ConvNoneLocal
{
  half src[convSrcTileH * convSrcTileW * convTileI];
  half weight[convTileO * convTileI * convKernelH * convKernelW];
};

groupshared ConvNoneLocal convNoneLocal;

struct ConvPoolLocal
{
  half src[convPoolSrcTileH * convPoolSrcTileW * convTileI];
  half weight[convTileO * convTileI * convKernelH * convKernelW];
};

groupshared ConvPoolLocal convPoolLocal;

[shader("compute")]
[numthreads(localSizeX, localSizeY, localSizeZ)]
void convNone(SystemValues3D sv,
              uniform oidn::VulkanConvKernel params)
{
  oidn::WorkGroupItem<3> it = { sv.globalID, sv.localID, sv.groupID, sv.localIndex };

  const int h = it.getGlobalID<0>();
  const int w = it.getGlobalID<1>();
  const int o = it.getGlobalID<2>();

  const int dstH = params.dst.H;
  const int dstW = params.dst.W;
  const int dstC = params.dst.C;
  const int srcH = params.src.H;
  const int srcW = params.src.W;
  const int srcC = params.src.C;

  const bool validOutput = h < dstH && w < dstW && o < dstC;
  float acc = validOutput ? float(params.bias[o, h, w]) : 0.f;

  const int baseH = it.getGroupID<0>() * convTileH;
  const int baseW = it.getGroupID<1>() * convTileW;
  const int localLinearID = it.getLocalLinearID();
  const int localSize = it.getLocalSize<0>() * it.getLocalSize<1>() * it.getLocalSize<2>();
  const int srcTileSize = convSrcTileH * convSrcTileW * convTileI;
  const int weightTileSize = convTileO * convTileI * convKernelH * convKernelW;
  const int tileH = h - baseH;
  const int tileW = w - baseW;
  const int tileO = o - it.getGroupID<2>() * convTileO;

  for (int iBase = 0; iBase < srcC; iBase += convTileI)
  {
    const int iCount = min(convTileI, srcC - iBase);

    for (int idx = localLinearID; idx < srcTileSize; idx += localSize)
    {
      const int i = idx % convTileI;
      const int tileIndex = idx / convTileI;
      const int tw = tileIndex % convSrcTileW;
      const int th = tileIndex / convSrcTileW;

      half value = half(0);
      if (i < iCount)
      {
        const int sh = baseH + th - convPad;
        const int sw = baseW + tw - convPad;
        if (sh >= 0 && sh < srcH && sw >= 0 && sw < srcW)
          value = params.src[iBase + i, sh, sw];
      }

      convNoneLocal.src[idx] = value;
    }

    for (int idx = localLinearID; idx < weightTileSize; idx += localSize)
    {
      const int kw = idx % convKernelW;
      const int kernelIndex = idx / convKernelW;
      const int kh = kernelIndex % convKernelH;
      const int channelIndex = kernelIndex / convKernelH;
      const int i = channelIndex % convTileI;
      const int lo = channelIndex / convTileI;

      half value = half(0);
      if (lo < convTileO && i < iCount)
        value = params.weight[it.getGroupID<2>() * convTileO + lo, iBase + i, kh, kw];

      convNoneLocal.weight[idx] = value;
    }

    it.groupBarrier();

    if (validOutput)
    {
      for (int kh = 0; kh < convKernelH; ++kh)
        for (int kw = 0; kw < convKernelW; ++kw)
          for (int i = 0; i < iCount; ++i)
          {
            const int srcIndex = getConvSrcTileIndex(tileH + kh, tileW + kw, i);
            const int weightIndex = getConvWeightTileIndex(tileO, i, kh, kw);
            acc += float(convNoneLocal.src[srcIndex]) * float(convNoneLocal.weight[weightIndex]);
          }
    }

    it.groupBarrier();
  }

  if (validOutput)
  {
    if (params.relu)
      acc = relu(acc);
    params.dst[o, h, w] = half(acc);
  }
}

[shader("compute")]
[numthreads(localSizeX, localSizeY, localSizeZ)]
void convUpsample(SystemValues3D sv,
                  uniform oidn::VulkanConvKernel params)
{
  oidn::WorkGroupItem<3> it = { sv.globalID, sv.localID, sv.groupID, sv.localIndex };

  const int h = it.getGlobalID<0>();
  const int w = it.getGlobalID<1>();
  const int o = it.getGlobalID<2>();

  const int srcH = params.src.H;
  const int srcW = params.src.W;
  const int srcC = params.src.C;
  const int dstC = params.dst.C;

  const bool validOutput = h < srcH && w < srcW && o < dstC;
  float acc = validOutput ? float(params.bias[o, h, w]) : 0.f;

  const int baseH = it.getGroupID<0>() * convTileH;
  const int baseW = it.getGroupID<1>() * convTileW;
  const int localLinearID = it.getLocalLinearID();
  const int localSize = it.getLocalSize<0>() * it.getLocalSize<1>() * it.getLocalSize<2>();
  const int srcTileSize = convSrcTileH * convSrcTileW * convTileI;
  const int weightTileSize = convTileO * convTileI * convKernelH * convKernelW;
  const int tileH = h - baseH;
  const int tileW = w - baseW;
  const int tileO = o - it.getGroupID<2>() * convTileO;

  for (int iBase = 0; iBase < srcC; iBase += convTileI)
  {
    const int iCount = min(convTileI, srcC - iBase);

    for (int idx = localLinearID; idx < srcTileSize; idx += localSize)
    {
      const int i = idx % convTileI;
      const int tileIndex = idx / convTileI;
      const int tw = tileIndex % convSrcTileW;
      const int th = tileIndex / convSrcTileW;

      half value = half(0);
      if (i < iCount)
      {
        const int sh = baseH + th - convPad;
        const int sw = baseW + tw - convPad;
        if (sh >= 0 && sh < srcH && sw >= 0 && sw < srcW)
          value = params.src[iBase + i, sh, sw];
      }

      convNoneLocal.src[idx] = value;
    }

    for (int idx = localLinearID; idx < weightTileSize; idx += localSize)
    {
      const int kw = idx % convKernelW;
      const int kernelIndex = idx / convKernelW;
      const int kh = kernelIndex % convKernelH;
      const int channelIndex = kernelIndex / convKernelH;
      const int i = channelIndex % convTileI;
      const int lo = channelIndex / convTileI;

      half value = half(0);
      if (lo < convTileO && i < iCount)
        value = params.weight[it.getGroupID<2>() * convTileO + lo, iBase + i, kh, kw];

      convNoneLocal.weight[idx] = value;
    }

    it.groupBarrier();

    if (validOutput)
    {
      for (int kh = 0; kh < convKernelH; ++kh)
        for (int kw = 0; kw < convKernelW; ++kw)
          for (int i = 0; i < iCount; ++i)
          {
            const int srcIndex = getConvSrcTileIndex(tileH + kh, tileW + kw, i);
            const int weightIndex = getConvWeightTileIndex(tileO, i, kh, kw);
            acc += float(convNoneLocal.src[srcIndex]) * float(convNoneLocal.weight[weightIndex]);
          }
    }

    it.groupBarrier();
  }

  if (validOutput)
  {
    if (params.relu)
      acc = relu(acc);

    const half val = half(acc);
    params.dst[o, h*2+0, w*2+0] = val;
    params.dst[o, h*2+0, w*2+1] = val;
    params.dst[o, h*2+1, w*2+0] = val;
    params.dst[o, h*2+1, w*2+1] = val;
  }
}

[shader("compute")]
[numthreads(localSizeX, localSizeY, localSizeZ)]
void convPool(SystemValues3D sv,
              uniform oidn::VulkanConvKernel params)
{
  oidn::WorkGroupItem<3> it = { sv.globalID, sv.localID, sv.groupID, sv.localIndex };

  const int dh = it.getGlobalID<0>();
  const int dw = it.getGlobalID<1>();
  const int o = it.getGlobalID<2>();

  const int dstH = params.dst.H;
  const int dstW = params.dst.W;
  const int dstC = params.dst.C;
  const int srcH = params.src.H;
  const int srcW = params.src.W;
  const int srcC = params.src.C;

  const bool validOutput = dh < dstH && dw < dstW && o < dstC;
  const float biasVal = validOutput ? float(params.bias[o, dh, dw]) : 0.f;
  float acc00 = biasVal;
  float acc01 = biasVal;
  float acc10 = biasVal;
  float acc11 = biasVal;

  const int baseDh = it.getGroupID<0>() * convTileH;
  const int baseDw = it.getGroupID<1>() * convTileW;
  const int localLinearID = it.getLocalLinearID();
  const int localSize = it.getLocalSize<0>() * it.getLocalSize<1>() * it.getLocalSize<2>();
  const int srcTileSize = convPoolSrcTileH * convPoolSrcTileW * convTileI;
  const int weightTileSize = convTileO * convTileI * convKernelH * convKernelW;
  const int tileDh = dh - baseDh;
  const int tileDw = dw - baseDw;
  const int tileO = o - it.getGroupID<2>() * convTileO;

  for (int iBase = 0; iBase < srcC; iBase += convTileI)
  {
    const int iCount = min(convTileI, srcC - iBase);

    for (int idx = localLinearID; idx < srcTileSize; idx += localSize)
    {
      const int i = idx % convTileI;
      const int tileIndex = idx / convTileI;
      const int tw = tileIndex % convPoolSrcTileW;
      const int th = tileIndex / convPoolSrcTileW;

      half value = half(0);
      if (i < iCount)
      {
        const int sh = baseDh * 2 + th - convPad;
        const int sw = baseDw * 2 + tw - convPad;
        if (sh >= 0 && sh < srcH && sw >= 0 && sw < srcW)
          value = params.src[iBase + i, sh, sw];
      }

      convPoolLocal.src[idx] = value;
    }

    for (int idx = localLinearID; idx < weightTileSize; idx += localSize)
    {
      const int kw = idx % convKernelW;
      const int kernelIndex = idx / convKernelW;
      const int kh = kernelIndex % convKernelH;
      const int channelIndex = kernelIndex / convKernelH;
      const int i = channelIndex % convTileI;
      const int lo = channelIndex / convTileI;

      half value = half(0);
      if (lo < convTileO && i < iCount)
        value = params.weight[it.getGroupID<2>() * convTileO + lo, iBase + i, kh, kw];

      convPoolLocal.weight[idx] = value;
    }

    it.groupBarrier();

    if (validOutput)
    {
      const int tileH = tileDh * 2;
      const int tileW = tileDw * 2;
      for (int kh = 0; kh < convKernelH; ++kh)
        for (int kw = 0; kw < convKernelW; ++kw)
          for (int i = 0; i < iCount; ++i)
          {
            const float weight = float(convPoolLocal.weight[getConvWeightTileIndex(tileO, i, kh, kw)]);
            acc00 += float(convPoolLocal.src[getConvPoolSrcTileIndex(tileH + kh,     tileW + kw,     i)]) * weight;
            acc01 += float(convPoolLocal.src[getConvPoolSrcTileIndex(tileH + kh,     tileW + kw + 1, i)]) * weight;
            acc10 += float(convPoolLocal.src[getConvPoolSrcTileIndex(tileH + kh + 1, tileW + kw,     i)]) * weight;
            acc11 += float(convPoolLocal.src[getConvPoolSrcTileIndex(tileH + kh + 1, tileW + kw + 1, i)]) * weight;
          }
    }

    it.groupBarrier();
  }

  if (validOutput)
  {
    if (params.relu)
    {
      acc00 = relu(acc00);
      acc01 = relu(acc01);
      acc10 = relu(acc10);
      acc11 = relu(acc11);
    }

    float maxVal = max(max(acc00, acc01), max(acc10, acc11));
    params.dst[o, dh, dw] = half(maxVal);
  }
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
    for (int kh = 0; kh < kH; ++kh)
      for (int kw = 0; kw < kW; ++kw)
      {
        const int sh = h + kh - pad;
        const int sw = w + kw - pad;
        if (sh >= 0 && sh < srcH && sw >= 0 && sw < srcW)
          for (int i = 0; i < srcC; ++i)
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
        for (int kh = 0; kh < kH; ++kh)
          for (int kw = 0; kw < kW; ++kw)
          {
            const int sh = h + kh - pad;
            const int sw = w + kw - pad;
            if (sh >= 0 && sh < srcH && sw >= 0 && sw < srcW)
              for (int i = 0; i < srcC; ++i)
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
    for (int kh = 0; kh < kH; ++kh)
      for (int kw = 0; kw < kW; ++kw)
      {
        const int sh = h + kh - pad;
        const int sw = w + kw - pad;
        if (sh >= 0 && sh < srcH && sw >= 0 && sw < srcW)
          for (int i = 0; i < srcC; ++i)
            acc += float(params.src[i, sh, sw]) * float(params.weight[o, i, kh, kw]);
      }

    if (params.relu)
      acc = relu(acc);
    params.dst[o, h, w] = half(acc);
  }
}
