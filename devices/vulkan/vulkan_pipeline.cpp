// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vulkan_pipeline.h"
#include "vulkan_device.h"
#include "core/exception.h"

OIDN_NAMESPACE_BEGIN

  VulkanComputePipeline::VulkanComputePipeline(VkDevice device, const uint32_t *spirvData, uint32_t spirvSize)
    : device(device)
  {
    VkPushConstantRange pushConstantRange = {};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = pushConstantSize;

    VkPipelineLayoutCreateInfo plci = {
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, 0 };
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pushConstantRange;

    VkResult res = vkCreatePipelineLayout(device, &plci, nullptr, &pipelineLayout);
    checkResult(res);

    try
    {
      VkPipelineShaderStageCreateInfo ssci = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0
      };
      ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      ssci.module = createShaderModule(spirvData, spirvSize);
      ssci.pName = "main";

      VkComputePipelineCreateInfo pci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, 0
      };
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
    VkShaderModuleCreateInfo shaderModuleInfo = {
      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr
    };
    shaderModuleInfo.codeSize = spirvSize;
    shaderModuleInfo.pCode = spirvData;

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkResult res = vkCreateShaderModule(device, &shaderModuleInfo, nullptr, &shaderModule);
    checkResult(res);
    return shaderModule;
  }

OIDN_NAMESPACE_END
