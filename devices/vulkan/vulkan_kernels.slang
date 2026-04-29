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

namespace oidn
{
  template<typename T, TensorLayout layout>
  typealias TensorByteOffset1D = TensorByteOffset_x<T>;

  template<typename T, TensorLayout layout>
  typealias TensorByteOffset3D = TensorByteOffset_hwc<T>;

  template<typename T, TensorLayout layout>
  typealias TensorByteOffset4D = TensorByteOffset_oihw<T>;
}

struct SystemValues
{
  uint2 globalID : SV_DispatchThreadID;
  uint2 localID : SV_GroupThreadID;
  uint2 groupID : SV_GroupID;
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
void inputProcess3(SystemValues sv,
                   uniform oidn::GPUInputProcessKernel<half, oidn::TensorLayout::hwc, 3> params)
{
  oidn::WorkGroupItem<2> it = { sv.globalID, sv.localID, sv.groupID, sv.localIndex };
  params(it);
}

[shader("compute")]
[numthreads(localSizeX, localSizeY, localSizeZ)]
void inputProcess6(SystemValues sv,
                   uniform oidn::GPUInputProcessKernel<half, oidn::TensorLayout::hwc, 6> params)
{
  oidn::WorkGroupItem<2> it = { sv.globalID, sv.localID, sv.groupID, sv.localIndex };
  params(it);
}

[shader("compute")]
[numthreads(localSizeX, localSizeY, localSizeZ)]
void inputProcess9(SystemValues sv,
                   uniform oidn::GPUInputProcessKernel<half, oidn::TensorLayout::hwc, 9> params)
{
  oidn::WorkGroupItem<2> it = { sv.globalID, sv.localID, sv.groupID, sv.localIndex };
  params(it);
}

[shader("compute")]
[numthreads(localSizeX, localSizeY, localSizeZ)]
void outputProcess(SystemValues sv,
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
void imageCopy(SystemValues sv,
               uniform uint32_t globalSize0,
               uniform uint32_t globalSize1,
               uniform oidn::GPUImageCopyKernel params)
{
  oidn::WorkItem<2> it = { sv.globalID };
  if (it.getGlobalID<0>() >= (int)globalSize0 || it.getGlobalID<1>() >= (int)globalSize1) return;
  params(it);
}
