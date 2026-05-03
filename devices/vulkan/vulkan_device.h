// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/device.h"
#include <vulkan/vulkan.h>
#include <memory>

OIDN_NAMESPACE_BEGIN

  struct VulkanQueue
  {
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t familyIndex = 0;
  };

  void checkResult(VkResult result);

  class VulkanInstance
  {
  public:
    VulkanInstance();
    ~VulkanInstance();

    operator VkInstance() { return instance; }
    bool hasDebugUtilsLabels() const { return debugUtilsLabels; }

  private:
    VulkanInstance(const VulkanInstance&) = delete;
    VulkanInstance& operator=(const VulkanInstance&) = delete;

    VkInstance instance = VK_NULL_HANDLE;
    bool debugUtilsLabels = false;
  };

  // only stores metadata to recognise real VkPhysicalDevice
  class VulkanPhysicalDevice : public PhysicalDevice
  {
  public:
    VulkanPhysicalDevice(VkPhysicalDevice pDev, int score);
  };

  class VulkanDevice final : public Device
  {
  public:
    static bool isSupported(VkPhysicalDevice pDev);

    VulkanDevice(const Ref<VulkanPhysicalDevice>& physicalDevice,
                 std::shared_ptr<VulkanInstance> instance,
                 VkPhysicalDevice pDev);
    ~VulkanDevice();

    DeviceType getType() const override { return DeviceType::Vulkan; }

    void wait() override;

    operator VkDevice() const { return device; }
    operator VkPhysicalDevice() const { return physDev; }
    VulkanQueue getQueue() const { return { queue, queueFamilyIndex }; }
    void getDeviceBufferMemoryRequirements(const VkDeviceBufferMemoryRequirementsKHR *pInfo, VkMemoryRequirements2 *pMemoryRequirements) const { vkGetDeviceBufferMemoryRequirements(device, pInfo, pMemoryRequirements); }
    int getSubgroupSize() const { return subgroupSize; }

    void cmdBeginDebugUtilsLabel(VkCommandBuffer cmdBuf, const char *label) const;
    void cmdEndDebugUtilsLabel(VkCommandBuffer cmdBuf) const;

  private:
    void init() override;

    std::shared_ptr<VulkanInstance> instance; // Keep instance alive for the lifetime of this device.
    VkPhysicalDevice physDev = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;
    int subgroupSize = 1;
    bool debugUtilsLabels = false;

    PFN_vkGetDeviceBufferMemoryRequirementsKHR vkGetDeviceBufferMemoryRequirements = nullptr;
    PFN_vkCmdBeginDebugUtilsLabelEXT vkCmdBeginDebugUtilsLabel = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT vkCmdEndDebugUtilsLabel = nullptr;
  };

OIDN_NAMESPACE_END
