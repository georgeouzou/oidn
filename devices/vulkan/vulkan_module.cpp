// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <dlfcn.h>
#include "core/context.h"

OIDN_NAMESPACE_BEGIN

  OIDN_DECLARE_INIT_MODULE(device_vulkan)
  {
    // Check Vulkan loader availability first
  #if defined(_WIN)
    HMODULE vulkanLoader = LoadLibraryW("vulkan-1.dll");
  #else
    void* vulkanLoader = dlopen("libvulkan.so.1", RTLD_LAZY | RTLD_LOCAL);
  #endif
    if (!vulkanLoader) return; // Vulkan not available
  }

OIDN_NAMESPACE_END
