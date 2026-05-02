// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vulkan_pipeline.h"
#include "vulkan_device.h"
#include "core/exception.h"

OIDN_NAMESPACE_BEGIN

  VulkanComputePipeline::VulkanComputePipeline(VkDevice device,
                                               const uint32_t* spirvData,
                                               uint32_t spirvSize,
                                               VulkanSize3D localSize,
                                               uint32_t pushConstantSize)
    : device(device), pushConstantSize(pushConstantSize)
  {
    VkPushConstantRange pushConstantRange = {};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = pushConstantSize;

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pushConstantRange;

    VkResult res = vkCreatePipelineLayout(device, &plci, nullptr, &pipelineLayout);
    checkResult(res);

    auto fillSpecMapEntry = [](uint32_t index, uint32_t id) -> VkSpecializationMapEntry
    {
      VkSpecializationMapEntry entry = {};
      entry.offset = index * sizeof(uint32_t);
      entry.size = sizeof(uint32_t);
      entry.constantID = id;
      return entry;
    };

    const uint32_t localSizeSpecConstantID_X = 0;
    const uint32_t localSizeSpecConstantID_Y = 1;
    const uint32_t localSizeSpecConstantID_Z = 2;

    try
    {
      VkSpecializationMapEntry specMapEntries[3] =
      {
        fillSpecMapEntry(0, localSizeSpecConstantID_X),
        fillSpecMapEntry(1, localSizeSpecConstantID_Y),
        fillSpecMapEntry(2, localSizeSpecConstantID_Z)
      };

      uint32_t specData[3] = { localSize.x, localSize.y, localSize.z };
      VkSpecializationInfo specInfo = {};
      specInfo.mapEntryCount = 3;
      specInfo.pMapEntries = specMapEntries;
      specInfo.dataSize = sizeof(specData);
      specInfo.pData = specData;

      VkPipelineShaderStageCreateInfo ssci{};
      ssci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      ssci.module = createShaderModule(spirvData, spirvSize);
      ssci.pName = "main";
      ssci.pSpecializationInfo = &specInfo;

      VkComputePipelineCreateInfo pci{};
      pci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
      pci.layout = pipelineLayout;
      pci.stage = ssci;

      res = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline);
      vkDestroyShaderModule(device, ssci.module, nullptr);
      if (res != VK_SUCCESS)
      {
        throw Exception(Error::Unknown, "failed to create Vulkan compute pipeline");
      }
    }
    catch (...)
    {
      free();
      throw;
    }
  }

  VulkanComputePipeline::~VulkanComputePipeline()
  {
    free();
  }

  void VulkanComputePipeline::free()
  {
    if (pipeline)
      vkDestroyPipeline(device, pipeline, nullptr);
    if (pipelineLayout)
      vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
  }

  VkShaderModule VulkanComputePipeline::createShaderModule(const uint32_t *spirvData, size_t spirvSize) const
  {
    VkShaderModuleCreateInfo shaderModuleInfo{};
    shaderModuleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleInfo.codeSize = spirvSize;
    shaderModuleInfo.pCode = spirvData;

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkResult res = vkCreateShaderModule(device, &shaderModuleInfo, nullptr, &shaderModule);
    checkResult(res);
    return shaderModule;
  }

OIDN_NAMESPACE_END
