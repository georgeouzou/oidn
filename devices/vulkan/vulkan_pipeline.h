// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/ref.h"
#include <vulkan/vulkan.h>

OIDN_NAMESPACE_BEGIN

  class VulkanDevice;

  class VulkanComputePipeline : public RefCount
  {
  public:
    static constexpr uint32_t pushConstantSize = 8; // just an address for now

    explicit VulkanComputePipeline(VkDevice device, const uint32_t *spirvData, uint32_t spirvSize);
    ~VulkanComputePipeline();

    operator VkPipeline() const { return pipeline; }
    operator VkPipelineLayout() const { return pipelineLayout; }

  private:
    void free();
    VkShaderModule createShaderModule(const uint32_t *spirvData, size_t spirvSize) const;

    VkDevice device = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
  };

OIDN_NAMESPACE_END
