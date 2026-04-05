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

  VulkanEngine::VulkanEngine(VulkanDevice* device, const VulkanQueue& queue)
    : device(device)
  {
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

OIDN_NAMESPACE_END
