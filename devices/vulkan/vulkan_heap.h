// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/heap.h"
#include "vulkan_engine.h"

OIDN_NAMESPACE_BEGIN

  class VulkanHeap : public Heap
  {
    friend class VulkanBuffer;

  public:
    VulkanHeap(VulkanEngine* engine, size_t byteSize, Storage storage);
    ~VulkanHeap();

    Engine* getEngine() const override { return engine; }
    size_t getByteSize() const override { return byteSize; }
    Storage getStorage() const override { return storage; }

    void realloc(size_t newByteSize) override;

    operator VkDeviceMemory() const { return memory; }

  private:
    void init();
    void free();

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props) const;
    std::pair<VkDeviceMemory, uint32_t> allocateMemory(VkDeviceSize byteSize, uint32_t memoryTypeBits, VkMemoryPropertyFlags props) const;

    VulkanEngine* engine;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mappedPtr = nullptr;
    uint32_t memoryTypeIndex = 0;
    size_t byteSize;
    Storage storage;
  };

OIDN_NAMESPACE_END
