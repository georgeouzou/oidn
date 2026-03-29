// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/device.h"
#include <vulkan/vulkan.h>

OIDN_NAMESPACE_BEGIN

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
    bool supportsExtension(const char *extensionName) const;

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

    void wait() override { /* TODO */ }

  private:
    void init() override;

    Ref<VulkanPhysicalDevice> physicalDevice;
    VkDevice device = VK_NULL_HANDLE;
  };

OIDN_NAMESPACE_END
