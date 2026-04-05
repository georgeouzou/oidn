// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/engine.h"
#include "vulkan_device.h"

OIDN_NAMESPACE_BEGIN

  class VulkanEngine : public Engine
  {
  public:
    explicit VulkanEngine(VulkanDevice* device, const VulkanQueue& queue);
    ~VulkanEngine();

    Device* getDevice() const override { return device; }

    // Vulkan
    VkDevice getVkDevice() const { return *device; }

    // Ops
    Ref<Conv> newConv(const ConvDesc& desc) override;
    Ref<Pool> newPool(const PoolDesc& desc) override;
    Ref<Upsample> newUpsample(const UpsampleDesc& desc) override;
    Ref<Autoexposure> newAutoexposure(const ImageDesc& srcDesc) override;
    Ref<InputProcess> newInputProcess(const InputProcessDesc& desc) override;
    Ref<OutputProcess> newOutputProcess(const OutputProcessDesc& desc) override;
    Ref<ImageCopy> newImageCopy() override;

    // Enqueues a host function
    void submitHostFunc(std::function<void()>&& f, const Ref<CancellationToken>& ct) override;

    void wait() override;

    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer cmdBuf);

  private:

    VulkanDevice* device = nullptr;
    VkCommandPool commandPool = VK_NULL_HANDLE;
  };

OIDN_NAMESPACE_END
