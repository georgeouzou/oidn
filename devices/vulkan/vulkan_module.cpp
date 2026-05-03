// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "core/context.h"
#include "vulkan_device.h"
#include <cstring>
#include <memory>

OIDN_NAMESPACE_BEGIN

  class VulkanDeviceFactory : public VulkanDeviceFactoryBase
  {
  public:
    // Returns a shared_ptr to the instance, creating one if none is alive.
    std::shared_ptr<VulkanInstance> getOrCreateInstance()
    {
      if (auto inst = weakInstance.lock())
        return inst;

      auto inst = std::make_shared<VulkanInstance>();
      weakInstance = inst;
      return inst;
    }

    std::vector<Ref<PhysicalDevice>> getPhysicalDevices()
    {
      auto inst = getOrCreateInstance();

      std::vector<VkPhysicalDevice> handles;
      uint32_t numDevices = 0;
      VkResult res = vkEnumeratePhysicalDevices(*inst, &numDevices, nullptr);
      if (res != VK_SUCCESS) return {};
      handles.resize(numDevices);
      res = vkEnumeratePhysicalDevices(*inst, &numDevices, handles.data());
      if (res != VK_SUCCESS) return {};

      std::vector<Ref<PhysicalDevice>> devices;
      for (size_t i = 0; i < handles.size(); ++i)
      {
        VkPhysicalDevice pDev = handles[i];
        if (!VulkanDevice::isSupported(pDev))
          continue;

        // duplicate score logic from other implementations for now
        int score = (19 << 16) - 1 - static_cast<int>(i);
        devices.push_back(makeRef<VulkanPhysicalDevice>(pDev, score));
      }
      return devices;
    }

    Ref<Device> newDevice(const Ref<PhysicalDevice>& physicalDevice) override
    {
      assert(physicalDevice->type == DeviceType::Vulkan);
      auto meta = staticRefCast<VulkanPhysicalDevice>(physicalDevice);

      auto inst = getOrCreateInstance();

      // Enumerate under the runtime instance and match by UUID.
      std::vector<VkPhysicalDevice> handles;
      uint32_t numDevices = 0;
      vkEnumeratePhysicalDevices(*inst, &numDevices, nullptr);
      handles.resize(numDevices);
      vkEnumeratePhysicalDevices(*inst, &numDevices, handles.data());

      VkPhysicalDevice matched = VK_NULL_HANDLE;
      for (VkPhysicalDevice h : handles)
      {
        VkPhysicalDeviceVulkan11Properties v11{};
        v11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
        VkPhysicalDeviceProperties2 props{};
        props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props.pNext = &v11;
        vkGetPhysicalDeviceProperties2(h, &props);

        if (memcmp(v11.deviceUUID, meta->uuid.bytes, VK_UUID_SIZE) == 0)
        {
          matched = h;
          break;
        }
      }

      if (matched == VK_NULL_HANDLE)
        throw Exception(Error::InvalidArgument, "Vulkan physical device no longer available");

      return makeRef<VulkanDevice>(meta, std::move(inst), matched);
    }

  private:
    // Non-owning. Instance lives only as long as active VulkanDevices hold it.
    std::weak_ptr<VulkanInstance> weakInstance;
  };

  OIDN_DECLARE_INIT_MODULE(device_vulkan)
  {
    auto factory = std::unique_ptr<VulkanDeviceFactory>(new VulkanDeviceFactory);
    auto physicalDevices = factory->getPhysicalDevices();
    Context::registerDeviceType<VulkanDeviceFactory>(DeviceType::Vulkan,
                                                     physicalDevices,
                                                     std::move(factory));
  }

OIDN_NAMESPACE_END
