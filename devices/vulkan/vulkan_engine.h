// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/engine.h"
#include "vulkan_device.h"

OIDN_NAMESPACE_BEGIN

  class VulkanComputePipeline;

  class VulkanEngine : public Engine
  {
  public:
    explicit VulkanEngine(VulkanDevice* device, const VulkanQueue& queue);
    ~VulkanEngine();

    Device* getDevice() const override { return device; }

    // Vulkan
    VkDevice getVkDevice() const { return *device; }
    VkPhysicalDevice getVkPhysicalDevice() const { return *device; }

    // Heap
    Ref<Heap> newHeap(size_t byteSize, Storage storage) override;

    // Buffer
    SizeAndAlignment getBufferByteSizeAndAlignment(size_t byteSize, Storage storage) override;
    Ref<Buffer> newBuffer(size_t byteSize, Storage storage) override;
    Ref<Buffer> newBuffer(void* ptr, size_t byteSize) override;
    Ref<Buffer> newBuffer(const Ref<Arena>& arena, size_t byteSize, size_t byteOffset) override;

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

    int getSubgroupSize() const override { return device->getSubgroupSize(); }

    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer cmdBuf);

    VkDeviceSize getMaxBufferSize() const { return maxBufferSize; }

    Ref<VulkanComputePipeline> newComputePipeline(const uint32_t* spirvData, uint32_t spirvSize);

  private:
    VulkanDevice* device = nullptr;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkDeviceSize maxBufferSize = 0;
  };

OIDN_NAMESPACE_END
