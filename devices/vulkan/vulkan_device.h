// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/device.h"
#include <vulkan/vulkan.h>

OIDN_NAMESPACE_BEGIN

  struct VulkanQueue
  {
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t familyIndex = 0;
  };

  void checkResult(VkResult result);

  class VulkanInstance : public RefCount
  {
  public:
    VulkanInstance();
    ~VulkanInstance();

    operator VkInstance() { return instance; }

  private:
    VkInstance instance = VK_NULL_HANDLE;
  };

  class VulkanPhysicalDevice : public PhysicalDevice
  {
  public:
    VulkanPhysicalDevice(const Ref<VulkanInstance> &instance, VkPhysicalDevice physDevice, int score);

    operator VkPhysicalDevice() { return pDev; }

  private:
    Ref<VulkanInstance> instance;
    VkPhysicalDevice pDev = VK_NULL_HANDLE;
  };

  class VulkanDevice final : public Device
  {
  public:
    static std::vector<Ref<PhysicalDevice>> getPhysicalDevices();
    static bool isSupported(VkPhysicalDevice pDev);

    explicit VulkanDevice(const Ref<VulkanPhysicalDevice>& physicalDevice);
    ~VulkanDevice();

    DeviceType getType() const override { return DeviceType::Vulkan; }

    void wait() override;

    operator VkDevice() const { return device; }
    operator VkPhysicalDevice() const { return *physicalDevice; }
    VulkanQueue getQueue() const { return { queue, queueFamilyIndex }; }
    void getDeviceBufferMemoryRequirements(const VkDeviceBufferMemoryRequirementsKHR *pInfo, VkMemoryRequirements2 *pMemoryRequirements) const { vkGetDeviceBufferMemoryRequirements(device, pInfo, pMemoryRequirements); }
    int getSubgroupSize() const { return subgroupSize; }

  private:
    void init() override;

    Ref<VulkanPhysicalDevice> physicalDevice;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;
    int subgroupSize = 1;

    PFN_vkGetDeviceBufferMemoryRequirementsKHR vkGetDeviceBufferMemoryRequirements = nullptr;
  };

OIDN_NAMESPACE_END
