// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vulkan_heap.h"
#include "vulkan_common.h"
#include "vulkan_device.h"
#include "core/exception.h"

OIDN_NAMESPACE_BEGIN

  VulkanHeap::VulkanHeap(VulkanEngine* engine, size_t byteSize, Storage storage)
    : engine(engine),
      byteSize(byteSize),
      storage((storage == Storage::Undefined) ? Storage::Device : storage)
  {
    init();
  }

  VulkanHeap::~VulkanHeap()
  {
    free();
  }

  void VulkanHeap::init()
  {
    if (byteSize == 0)
      return;

    try
    {
      VkBufferCreateInfo bci{};
      bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      bci.size = byteSize;
      bci.usage = getCommonVkBufferUsageFlags();

      VkDeviceBufferMemoryRequirementsKHR dbmr{};
      dbmr.sType = VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS_KHR;
      dbmr.pCreateInfo = &bci;

      const VulkanDevice *vkDevice = static_cast<const VulkanDevice*>(engine->getDevice());
      VkMemoryRequirements2 memReqs{};
      memReqs.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
      vkDevice->getDeviceBufferMemoryRequirements(&dbmr, &memReqs);

      VkMemoryPropertyFlags memFlags = toVkMemoryFlags(storage);

      auto alloc = allocateMemory(memReqs.memoryRequirements.size, memReqs.memoryRequirements.memoryTypeBits, memFlags);
      memory = alloc.first;
      memoryTypeIndex = alloc.second;

      if (storage == Storage::Host)
      {
        VkResult res = vkMapMemory(engine->getVkDevice(), memory, 0, byteSize, 0, &mappedPtr);
        checkResult(res);
      }
    }
    catch (...)
    {
      free();
      throw;
    }
  }

  void VulkanHeap::free()
  {
    if (mappedPtr)
    {
      vkUnmapMemory(engine->getVkDevice(), memory);
      mappedPtr = nullptr;
    }
    if (memory)
    {
      vkFreeMemory(engine->getVkDevice(), memory, nullptr);
      memory = VK_NULL_HANDLE;
    }
  }

  void VulkanHeap::realloc(size_t newByteSize)
  {
    if (newByteSize == byteSize)
      return;

    preRealloc();
    free();
    byteSize = newByteSize;
    init();
    postRealloc();
  }

  uint32_t VulkanHeap::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props) const
  {
    VkPhysicalDevice pDev = engine->getVkPhysicalDevice();
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(pDev, &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
    {
      if ((typeFilter & (1u << i)) &&
          (memProps.memoryTypes[i].propertyFlags & props) == props)
      {
        return i;
      }
    }
    throw Exception(Error::OutOfMemory, "failed to find suitable memory type");
  }

  std::pair<VkDeviceMemory, uint32_t> VulkanHeap::allocateMemory(VkDeviceSize byteSize, uint32_t memoryTypeBits, VkMemoryPropertyFlags props) const
  {
    VkMemoryAllocateFlagsInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    uint32_t typeIndex = findMemoryType(memoryTypeBits, props);

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.pNext = &fi;
    mai.allocationSize = byteSize;
    mai.memoryTypeIndex = typeIndex;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkResult res = vkAllocateMemory(engine->getVkDevice(), &mai, nullptr, &memory);
    checkResult(res);
    return { memory, typeIndex };
  }

OIDN_NAMESPACE_END
