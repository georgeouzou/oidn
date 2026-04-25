// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vulkan_engine.h"
#include "vulkan_buffer.h"
#include "vulkan_heap.h"
#include "vulkan_pipeline.h"
#include "vulkan_common.h"

#include "core/conv.h"
#include "devices/gpu/gpu_input_process.h"
#include "devices/gpu/gpu_autoexposure.h"
#include "devices/gpu/gpu_output_process.h"
#include "devices/gpu/gpu_image_copy.h"
#include "devices/gpu/gpu_pool.h"
#include "devices/gpu/gpu_upsample.h"

#include <cstring>

OIDN_NAMESPACE_BEGIN

  VulkanEngine::VulkanEngine(VulkanDevice* device, const VulkanQueue& queue)
    : device(device)
  {
    VkPhysicalDeviceMaintenance4PropertiesKHR maint4Props = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES_KHR, 0 };
    VkPhysicalDeviceProperties2 props2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, 0 };
    props2.pNext = &maint4Props;
    vkGetPhysicalDeviceProperties2(*device, &props2);
    maxBufferSize = maint4Props.maxBufferSize;

    VkCommandPoolCreateInfo cpci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, 0 };
    cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    cpci.queueFamilyIndex = queue.familyIndex;

    VkResult res = vkCreateCommandPool(*device, &cpci, nullptr, &commandPool);
    checkResult(res);
  }

  VulkanEngine::~VulkanEngine()
  {
    vkDestroyCommandPool(*device, commandPool, nullptr);
  }

  Ref<Heap> VulkanEngine::newHeap(size_t byteSize, Storage storage)
  {
    return makeRef<VulkanHeap>(this, byteSize, storage);
  }

  SizeAndAlignment VulkanEngine::getBufferByteSizeAndAlignment(size_t byteSize, Storage storage)
  {
    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, 0 };
    bci.size = byteSize;
    bci.usage = getCommonVkBufferUsageFlags();
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkDeviceBufferMemoryRequirementsKHR dbmr = { VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS_KHR, 0 };
    dbmr.pCreateInfo = &bci;

    VkMemoryRequirements2 mr = { VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, 0 };
    device->getDeviceBufferMemoryRequirements(&dbmr, &mr);

    return { static_cast<size_t>(mr.memoryRequirements.size),
             static_cast<size_t>(mr.memoryRequirements.alignment) };
  }

  Ref<Buffer> VulkanEngine::newBuffer(size_t byteSize, Storage storage)
  {
    return makeRef<VulkanBuffer>(this, byteSize, storage);
  }

  Ref<Buffer> VulkanEngine::newBuffer(void* ptr, size_t byteSize)
  {
    return makeRef<VulkanBuffer>(this, ptr, byteSize);
  }

  Ref<Buffer> VulkanEngine::newBuffer(const Ref<Arena>& arena, size_t byteSize, size_t byteOffset)
  {
    return makeRef<VulkanBuffer>(arena, byteSize, byteOffset);
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

  void VulkanEngine::submitHostFunc(std::function<void()>&& f, const Ref<CancellationToken>& ct)
  {
  }

  void VulkanEngine::wait()
  {
    vkQueueWaitIdle(device->getQueue().queue);
  }

  VkCommandBuffer VulkanEngine::beginSingleTimeCommands()
  {
    VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, 0 };
    cbai.commandPool = commandPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;

    VkCommandBuffer cmdBuf = VK_NULL_HANDLE;
    VkResult res = vkAllocateCommandBuffers(getVkDevice(), &cbai, &cmdBuf);
    checkResult(res);

    VkCommandBufferBeginInfo cbbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, 0 };
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    res = vkBeginCommandBuffer(cmdBuf, &cbbi);
    checkResult(res);

    return cmdBuf;
  }

  void VulkanEngine::endSingleTimeCommands(VkCommandBuffer cmdBuf)
  {
    VkResult res = vkEndCommandBuffer(cmdBuf);
    checkResult(res);

    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO, 0 };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmdBuf;
    res = vkQueueSubmit(device->getQueue().queue, 1, &si, VK_NULL_HANDLE);
    checkResult(res);

    vkQueueWaitIdle(device->getQueue().queue);

    vkFreeCommandBuffers(getVkDevice(), commandPool, 1, &cmdBuf);
  }

  Ref<VulkanComputePipeline> VulkanEngine::newComputePipeline(const uint32_t* spirvData, uint32_t spirvSize)
  {
    VkDevice device = getVkDevice();
    return makeRef<VulkanComputePipeline>(device, spirvData, spirvSize, VulkanSize3D{}, 8);
  }

  void VulkanEngine::submitKernelImpl(VulkanSize3D numGroups,
                                      const void* kernelData,
                                      size_t kernelSize,
                                      const Ref<VulkanComputePipeline>& pipeline)
  {
    assert(kernelData != nullptr || kernelSize == 0);
    assert(kernelSize == pipeline->getPushConstantSize());
    assert(kernelSize % 4 == 0);
    assert(kernelSize <= 256);

    VkCommandBuffer cmdBuf = beginSingleTimeCommands();

    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
    vkCmdPushConstants(cmdBuf, *pipeline, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       static_cast<uint32_t>(kernelSize), kernelData);

    vkCmdDispatch(cmdBuf, numGroups.x, numGroups.y, numGroups.z);

    endSingleTimeCommands(cmdBuf);
  }

OIDN_NAMESPACE_END
