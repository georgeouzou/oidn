// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vulkan_common.h"
#include "core/exception.h"

OIDN_NAMESPACE_BEGIN

  VkMemoryPropertyFlags toVkMemoryFlags(Storage storage)
  {
    switch (storage)
    {
    case Storage::Host:
      return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    case Storage::Device:
      return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    case Storage::Managed:
      throw Exception(Error::InvalidArgument, "Vulkan managed storage mode is not supported");
    default:
      throw Exception(Error::InvalidArgument, "invalid storage mode");
    }
  }

  VkBufferUsageFlags getCommonVkBufferUsageFlags()
  {
    return VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
           VK_BUFFER_USAGE_TRANSFER_DST_BIT |
           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  }

OIDN_NAMESPACE_END
