// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vulkan_engine.h"
#include "vulkan_buffer.h"
#include "vulkan_heap.h"
#include "vulkan_pipeline.h"
#include "vulkan_common.h"

#include "devices/gpu/gpu_input_process.h"
#include "devices/gpu/gpu_autoexposure.h"
#include "devices/gpu/gpu_output_process.h"
#include "devices/gpu/gpu_image_copy.h"
#include "devices/vulkan/vulkan_conv.h"
#include "devices/vulkan/autoexposureDownsample.h"
#include "devices/vulkan/autoexposureReduce_1024.h"
#include "devices/vulkan/autoexposureReduceFinal_1024.h"
#include "devices/vulkan/inputProcess_f16_hwc_3.h"
#include "devices/vulkan/inputProcess_f16_hwc_6.h"
#include "devices/vulkan/inputProcess_f16_hwc_9.h"
#include "devices/vulkan/outputProcess.h"
#include "devices/vulkan/imageCopy.h"
#include "devices/vulkan/conv.h"

#include <cstring>

namespace
{
  struct VulkanPipelineDesc
  {
    const char* name;
    const uint32_t* spirvData;
    uint32_t spirvSize;
  };

  const VulkanPipelineDesc vulkanPipelineRegistry[] =
  {
    {
      "inputProcess_f16_hwc_3",
      reinterpret_cast<const uint32_t*>(oidn::blobs::inputProcess_f16_hwc_3),
      uint32_t(sizeof(oidn::blobs::inputProcess_f16_hwc_3)),
    },
    {
      "inputProcess_f16_hwc_6",
      reinterpret_cast<const uint32_t*>(oidn::blobs::inputProcess_f16_hwc_6),
      uint32_t(sizeof(oidn::blobs::inputProcess_f16_hwc_6)),
    },
    {
      "inputProcess_f16_hwc_9",
      reinterpret_cast<const uint32_t*>(oidn::blobs::inputProcess_f16_hwc_9),
      uint32_t(sizeof(oidn::blobs::inputProcess_f16_hwc_9)),
    },
    {
      "outputProcess_f16_hwc",
      reinterpret_cast<const uint32_t*>(oidn::blobs::outputProcess),
      uint32_t(sizeof(oidn::blobs::outputProcess)),
    },
    {
      "imageCopy",
      reinterpret_cast<const uint32_t*>(oidn::blobs::imageCopy),
      uint32_t(sizeof(oidn::blobs::imageCopy)),
    },
    {
      "autoexposureDownsample",
      reinterpret_cast<const uint32_t*>(oidn::blobs::autoexposureDownsample),
      uint32_t(sizeof(oidn::blobs::autoexposureDownsample)),
    },
    {
      "autoexposureReduce_1024",
      reinterpret_cast<const uint32_t*>(oidn::blobs::autoexposureReduce_1024),
      uint32_t(sizeof(oidn::blobs::autoexposureReduce_1024)),
    },
    {
      "autoexposureReduceFinal_1024",
      reinterpret_cast<const uint32_t*>(oidn::blobs::autoexposureReduceFinal_1024),
      uint32_t(sizeof(oidn::blobs::autoexposureReduceFinal_1024)),
    },

    {
      "conv",
      reinterpret_cast<const uint32_t*>(oidn::blobs::conv),
      uint32_t(sizeof(oidn::blobs::conv)),
    },
  };
}

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

  bool VulkanEngine::isConvSupported(PostOp postOp)
  {
    return postOp == PostOp::None || postOp == PostOp::Pool || postOp == PostOp::Upsample;
  }

  Ref<Conv> VulkanEngine::newConv(const ConvDesc& desc)
  {
    return makeRef<VulkanConv>(this, desc);
  }

  Ref<Pool> VulkanEngine::newPool(const PoolDesc& desc)
  {
    throw std::logic_error("operation is not implemented");
  }

  Ref<Upsample> VulkanEngine::newUpsample(const UpsampleDesc& desc)
  {
    throw std::logic_error("operation is not implemented");
  }

  Ref<Autoexposure> VulkanEngine::newAutoexposure(const ImageDesc& srcDesc)
  {
    // TODO: choose group size from Vulkan device limits like SYCL once we support it.
    return makeRef<GPUAutoexposure<VulkanEngine, 1024>>(this, srcDesc);
  }

  Ref<InputProcess> VulkanEngine::newInputProcess(const InputProcessDesc& desc)
  {
    return makeRef<GPUInputProcess<VulkanEngine, half, TensorLayout::hwc, 1>>(this, desc);
  }

  Ref<OutputProcess> VulkanEngine::newOutputProcess(const OutputProcessDesc& desc)
  {
    return makeRef<GPUOutputProcess<VulkanEngine, half, TensorLayout::hwc>>(this, desc);
  }

  Ref<ImageCopy> VulkanEngine::newImageCopy()
  {
    return makeRef<GPUImageCopy<VulkanEngine>>(this);
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

  // keep it simple for now
  // TODO: specialize per vendor values
  template<> WorkDim<1> VulkanEngine::suggestWorkGroupSize<1>() { return 256; }
  template<> WorkDim<2> VulkanEngine::suggestWorkGroupSize<2>() { return {16, 16}; }
  template<> WorkDim<3> VulkanEngine::suggestWorkGroupSize<3>() { return {1, 16, 16}; }

  Ref<VulkanComputePipeline> VulkanEngine::newComputePipelineImpl(const std::string& kernelName,
                                                                  VulkanSize3D localSize,
                                                                  uint32_t pushConstantSize)
  {
    for (const auto& pipeline : vulkanPipelineRegistry)
    {
      if (kernelName == pipeline.name)
      {
        return makeRef<VulkanComputePipeline>(getVkDevice(),
                                              pipeline.spirvData,
                                              pipeline.spirvSize,
                                              localSize,
                                              pushConstantSize);
      }
    }
    throw Exception(Error::InvalidArgument, "could not create Vulkan pipeline");
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
