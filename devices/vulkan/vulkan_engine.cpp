// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vulkan_engine.h"

#include "core/conv.h"
#include "devices/gpu/gpu_input_process.h"
#include "devices/gpu/gpu_autoexposure.h"
#include "devices/gpu/gpu_output_process.h"
#include "devices/gpu/gpu_image_copy.h"
#include "devices/gpu/gpu_pool.h"
#include "devices/gpu/gpu_upsample.h"

OIDN_NAMESPACE_BEGIN

  VulkanEngine::VulkanEngine(VulkanDevice* device)
    : device(device)
  {
  }

  VulkanEngine::~VulkanEngine()
  {
  }

  Ref<Conv> VulkanEngine::newConv(const ConvDesc& desc)
  {
    return nullptr;
  }

  Ref<Pool> VulkanEngine::newPool(const PoolDesc& desc)
  {
    return nullptr;
  }

  Ref<Upsample> VulkanEngine::newUpsample(const UpsampleDesc& desc)
  {
    return nullptr;
  }

  Ref<Autoexposure> VulkanEngine::newAutoexposure(const ImageDesc& srcDesc)
  {
    return nullptr;
  }

  Ref<InputProcess> VulkanEngine::newInputProcess(const InputProcessDesc& desc)
  {
    return {};
  }

  Ref<OutputProcess> VulkanEngine::newOutputProcess(const OutputProcessDesc& desc)
  {
    return nullptr;
  }

  Ref<ImageCopy> VulkanEngine::newImageCopy()
  {
    return nullptr;
  }
    // Enqueues a host function
  void VulkanEngine::submitHostFunc(std::function<void()>&& f, const Ref<CancellationToken>& ct)
  {
  }

  void VulkanEngine::wait()
  {
  }

OIDN_NAMESPACE_END
