// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/common.h"
#include <vulkan/vulkan.h>

OIDN_NAMESPACE_BEGIN

  VkMemoryPropertyFlags toVkMemoryFlags(Storage storage);
  VkBufferUsageFlags getCommonVkBufferUsageFlags();

OIDN_NAMESPACE_END
