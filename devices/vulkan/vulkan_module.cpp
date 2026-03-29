// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "core/context.h"
#include "vulkan_device.h"

OIDN_NAMESPACE_BEGIN

  class VulkanDeviceFactory : public VulkanDeviceFactoryBase
  {
  public:
    Ref<Device> newDevice(const Ref<PhysicalDevice> &physicalDevice) override
    {
      assert(physicalDevice->type == DeviceType::Vulkan);
      return makeRef<VulkanDevice>(staticRefCast<VulkanPhysicalDevice>(physicalDevice));
    }
  };

  OIDN_DECLARE_INIT_MODULE(device_vulkan)
  {
    Context::registerDeviceType<VulkanDeviceFactory>(DeviceType::Vulkan,
      VulkanDevice::getPhysicalDevices());
  }

OIDN_NAMESPACE_END
