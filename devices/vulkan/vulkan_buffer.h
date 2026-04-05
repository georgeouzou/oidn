// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/buffer.h"
#include "vulkan_engine.h"

OIDN_NAMESPACE_BEGIN

  class VulkanHeap;

  class VulkanBuffer : public Buffer
  {
  public:
    VulkanBuffer(VulkanEngine* engine, size_t byteSize, Storage storage);
    VulkanBuffer(const Ref<Arena>& arena, size_t byteSize, size_t byteOffset);
    VulkanBuffer(VulkanEngine* engine, void* data, size_t byteSize);
    ~VulkanBuffer();

    Engine* getEngine() const override { return engine; }
    void* getPtr() const override;
    void* getHostPtr() const override;
    size_t getByteSize() const override { return byteSize; }
    bool isShared() const override { return shared; }
    Storage getStorage() const override { return storage; }

    void read(size_t byteOffset, size_t byteSize, void* dstHostPtr,
              SyncMode sync = SyncMode::Blocking) override;

    void write(size_t byteOffset, size_t byteSize, const void* srcHostPtr,
               SyncMode sync = SyncMode::Blocking) override;

    operator VkBuffer() const { return buffer; }

  protected:
    void preRealloc() override;
    void postRealloc() override;

  private:
    void init();
    void free();

    VulkanEngine* engine;
    VkBuffer buffer = VK_NULL_HANDLE;
    Ref<VulkanHeap> heap; // might be buffer owned or arena owned
    VkDeviceSize byteSize;
    bool shared;
    Storage storage;
  };

OIDN_NAMESPACE_END
