// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vulkan_device.h"
#include "core/engine.h"
#include "core/subdevice.h"
#include "core/exception.h"
#include <cstring>
#include <optional>

namespace
{
  struct VulkanQueueInfo
  {
    std::optional<uint32_t> graphicsQueueFamily;
  };

  VulkanQueueInfo queryQueueInfo(VkPhysicalDevice pDev)
  {
    uint32_t numFamilies = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pDev, &numFamilies, nullptr);
    std::vector<VkQueueFamilyProperties> props(numFamilies);
    vkGetPhysicalDeviceQueueFamilyProperties(pDev, &numFamilies, props.data());

    auto it = std::find_if(props.begin(), props.end(),
      [](const VkQueueFamilyProperties &prop)
    {
      return prop.queueCount > 0 && prop.queueFlags & VK_QUEUE_GRAPHICS_BIT;
    });
    if (it == props.end())
      return {};

    const uint32_t graphicsQueueFamilyIndex =
      static_cast<uint32_t>(std::distance(it, props.begin()));

    return VulkanQueueInfo { graphicsQueueFamilyIndex };
  }
}

OIDN_NAMESPACE_BEGIN

  void checkResult(VkResult res)
  {
    if (res == VK_SUCCESS)
      return;

    switch (res)
    {
      case VK_ERROR_OUT_OF_HOST_MEMORY:
      case VK_ERROR_OUT_OF_DEVICE_MEMORY:
      case VK_ERROR_TOO_MANY_OBJECTS:
      case VK_ERROR_MEMORY_MAP_FAILED:
        throw Exception(Error::OutOfMemory, "Vulkan memory error");
      case VK_ERROR_INITIALIZATION_FAILED:
      case VK_ERROR_INCOMPATIBLE_DRIVER:
        throw Exception(Error::UnsupportedHardware, "Vulkan initialization error");
      default:
        throw Exception(Error::Unknown, "unknown Vulkan error");
    }
  }

  VulkanInstance::VulkanInstance()
  {
    uint32_t instanceVersion = 0;
    {
      VkResult res = vkEnumerateInstanceVersion(&instanceVersion);
      checkResult(res);
      if (VK_VERSION_MAJOR(instanceVersion) == 1 && VK_VERSION_MINOR(instanceVersion) < 2)
      {
        checkResult(VK_ERROR_INCOMPATIBLE_DRIVER);
      }
    }

    VkApplicationInfo ai = { VK_STRUCTURE_TYPE_APPLICATION_INFO, 0 };
    ai.pApplicationName = "Open Image Denoise";
    ai.applicationVersion = OIDN_VERSION;
    ai.pEngineName = "Open Image Denoise";
    ai.engineVersion = OIDN_VERSION;
    ai.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0 };
    ici.pApplicationInfo = &ai;

    VkResult res = vkCreateInstance(&ici, nullptr, &this->instance);
    checkResult(res);
  };

  VulkanInstance::~VulkanInstance()
  {
    vkDestroyInstance(this->instance, nullptr);
  };

  VulkanPhysicalDevice::VulkanPhysicalDevice(const Ref<VulkanInstance> &instance, VkPhysicalDevice pDev, int score)
    : PhysicalDevice(DeviceType::Vulkan, score),
      instance(instance),
      pDev(pDev)
  {
    const bool hasPCIBusInfo = supportsExtension(VK_EXT_PCI_BUS_INFO_EXTENSION_NAME);
    VkPhysicalDeviceProperties2 props = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, 0 };
    VkPhysicalDeviceVulkan11Properties v11Props = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES, 0 };
    VkPhysicalDeviceVulkan12Properties v12Props = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES, 0 };
    VkPhysicalDevicePCIBusInfoPropertiesEXT pciProps = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT, 0 };

    if (hasPCIBusInfo)
      v12Props.pNext = &pciProps;
    v11Props.pNext = &v12Props;
    props.pNext = &v11Props;
    vkGetPhysicalDeviceProperties2(pDev, &props);

    name = props.properties.deviceName;

    memcpy(uuid.bytes, v11Props.deviceUUID, VK_UUID_SIZE);
    uuidSupported = true;

  #if defined(_WIN32)
    if (v11Props.deviceLUIDValid)
    {
      memcpy(luid.bytes, v11Props.deviceLUID, VK_LUID_SIZE);
      nodeMask = v11Props.deviceNodeMask;
      luidSupported = true;
    }
  #endif

    if (hasPCIBusInfo)
    {
      pciDomain   = pciProps.pciDomain;
      pciBus      = pciProps.pciBus;
      pciDevice   = pciProps.pciDevice;
      pciFunction = pciProps.pciFunction;
      pciAddressSupported = true;
    }
  }

  bool VulkanPhysicalDevice::supportsExtension(const char *extensionName) const
  {
    uint32_t numExtensions = 0;
    std::vector<VkExtensionProperties> extensions;
    VkResult res = vkEnumerateDeviceExtensionProperties(pDev, nullptr, &numExtensions, nullptr);
    checkResult(res);
    extensions.resize(numExtensions);
    res = vkEnumerateDeviceExtensionProperties(pDev, nullptr, &numExtensions, extensions.data());

    const size_t extensionNameLen = strnlen(extensionName, VK_MAX_EXTENSION_NAME_SIZE);
    auto it = std::find_if(extensions.begin(), extensions.end(),
      [&](const VkExtensionProperties &p)
    {
        return memcmp(p.extensionName, extensionName, extensionNameLen) == 0;
    });
    return it != extensions.end();
  }

  std::vector<Ref<PhysicalDevice>> VulkanDevice::getPhysicalDevices()
  {
    auto instance = makeRef<VulkanInstance>();
    std::vector<VkPhysicalDevice> devices;
    uint32_t numDevices = 0;
    VkResult res = vkEnumeratePhysicalDevices(*instance, &numDevices, nullptr);
    if (res != VK_SUCCESS) return {};
    devices.resize(numDevices);
    res = vkEnumeratePhysicalDevices(*instance, &numDevices, devices.data());
    if (res != VK_SUCCESS) return {};

    std::vector<Ref<PhysicalDevice>> chosenDevices;

    for (size_t i = 0; i < devices.size(); ++i)
    {
      VkPhysicalDevice pDev = devices[i];
      if (!isSupported(pDev))
        continue;

      // duplicate score logic from other implementations for now
      int score = (19 << 16) - 1 - i;
      chosenDevices.push_back(makeRef<VulkanPhysicalDevice>(instance, pDev, score));
    }

    return chosenDevices;
  }

  bool VulkanDevice::isSupported(VkPhysicalDevice pDev)
  {
    assert(pDev != VK_NULL_HANDLE);
    // filter out gpus not supporting vulkan 1.2
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pDev, &props);
    const uint32_t major = VK_VERSION_MAJOR(props.apiVersion);
    const uint32_t minor = VK_VERSION_MINOR(props.apiVersion);

    const VulkanQueueInfo queueInfo = queryQueueInfo(pDev);
    const bool hasGraphics = queueInfo.graphicsQueueFamily.has_value();

    return !(major == 1 && minor < 2) && hasGraphics;
  }

  VulkanDevice::VulkanDevice(const Ref<VulkanPhysicalDevice>& physicalDevice)
    : physicalDevice(physicalDevice)
  {
    const VulkanQueueInfo queueInfo = queryQueueInfo(*physicalDevice);
    // TODO for now physical devices can only be created from us
    assert(queueInfo.graphicsQueueFamily.has_value());

    const float queuePriorities[] = { 1.0f };

    VkDeviceQueueCreateInfo qci = {
      VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, 0 };
    qci.queueCount = 1;
    qci.pQueuePriorities = queuePriorities;
    qci.queueFamilyIndex = *queueInfo.graphicsQueueFamily;

    VkDeviceCreateInfo ci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, 0 };
    ci.queueCreateInfoCount = 1;
    ci.pQueueCreateInfos = &qci;
    VkResult res = vkCreateDevice(*physicalDevice, &ci, nullptr, &this->device);
    checkResult(res);
  }

  VulkanDevice::~VulkanDevice()
  {
    vkDestroyDevice(device, nullptr);
  }

OIDN_NAMESPACE_END
