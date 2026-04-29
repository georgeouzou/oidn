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

    template<typename PushData, int N>
    Ref<VulkanComputePipeline> newComputePipeline(const std::string& kernelName,
                                                  WorkDim<N> localSize)
    {
      return newComputePipelineImpl(kernelName, localSize, uint32_t(sizeof(PushData)));
    }

    template<typename Kernel, int N>
    Ref<VulkanComputePipeline> newComputePipeline(const std::string& kernelName)
    {
      struct PushData
      {
        uint32_t globalSize[N];
        Kernel kernel;
      };
      return newComputePipelineImpl(kernelName, suggestWorkGroupSize<N>(),
                                    uint32_t(sizeof(PushData)));
    }

    // Enqueues a kernel with explicit numGroups
    template<int N, typename Kernel>
    oidn_inline void submitKernel(WorkDim<N> numGroups,
                                  const Kernel& kernel,
                                  const Ref<VulkanComputePipeline>& pipeline)
    {
      submitKernelImpl(numGroups, &kernel, sizeof(kernel), pipeline);
    }

    // Enqueues a kernel with implicit numGroups
    template<int N, typename Kernel>
    oidn_inline void submitKernelGlobal(WorkDim<N> globalSize,
                                        const Kernel& kernel,
                                        const Ref<VulkanComputePipeline>& pipeline)
    {
      struct PushData
      {
        uint32_t globalSize[N];
        Kernel kernel;
      } pushData;

      for (int i = 0; i < N; ++i)
        pushData.globalSize[i] = globalSize[i];
      pushData.kernel = kernel;

      const WorkDim<N> numGroups = ceil_div(globalSize, suggestWorkGroupSize<N>());
      submitKernelImpl(numGroups, &pushData, sizeof(pushData), pipeline);
    }

  private:
    template<int N> WorkDim<N> suggestWorkGroupSize();

    Ref<VulkanComputePipeline> newComputePipelineImpl(const std::string& kernelName,
                                                      VulkanSize3D localSize,
                                                      uint32_t pushConstantSize);

    void submitKernelImpl(VulkanSize3D numGroups,
                          const void* kernelData,
                          size_t kernelSize,
                          const Ref<VulkanComputePipeline>& pipeline);

    VulkanDevice* device = nullptr;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkDeviceSize maxBufferSize = 0;
  };

OIDN_NAMESPACE_END
