// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vulkan_buffer.h"
#include "vulkan_heap.h"
#include "vulkan_common.h"
#include "core/arena.h"
#include "core/exception.h"

OIDN_NAMESPACE_BEGIN

  VulkanBuffer::VulkanBuffer(VulkanEngine* engine, size_t byteSize, Storage storage)
    : engine(engine),
      byteSize(byteSize),
      shared(false),
      storage((storage == Storage::Undefined) ? Storage::Host : storage)
  {
    if (this->storage == Storage::Managed)
      throw Exception(Error::InvalidArgument, "Vulkan managed storage mode is not supported");

    init();
  }

  VulkanBuffer::VulkanBuffer(const Ref<Arena>& arena, size_t byteSize, size_t byteOffset)
    : Buffer(arena, byteOffset),
      engine(dynamic_cast<VulkanEngine*>(arena->getEngine())),
      heap(dynamic_cast<VulkanHeap*>(arena->getHeap())),
      byteSize(byteSize),
      shared(true),
      storage(arena->getHeap()->getStorage())
  {
    if (!engine || !heap)
      throw Exception(Error::InvalidArgument, "buffer is incompatible with arena");

    const SizeAndAlignment byteSizeAndAlignment = engine->getBufferByteSizeAndAlignment(byteSize, storage);
    if (byteOffset % byteSizeAndAlignment.alignment != 0)
      throw Exception(Error::InvalidArgument, "buffer offset is unaligned");
    if (byteOffset + byteSizeAndAlignment.size > arena->getByteSize())
      throw Exception(Error::InvalidArgument, "arena region is out of bounds");

    init();
  }

  VulkanBuffer::VulkanBuffer(VulkanEngine* engine, void* data, size_t byteSize)
    : engine(engine),
      byteSize(byteSize),
      shared(true),
      storage(Storage::Host)
  {
    if (data == nullptr)
      throw Exception(Error::InvalidArgument, "buffer pointer is null");
    if (byteSize == 0)
      return;

    init();

    memcpy(heap->mappedPtr, data, byteSize);
  }

  VulkanBuffer::~VulkanBuffer()
  {
    free();
  }

  void VulkanBuffer::init()
  {
    if (byteSize == 0)
      return;

    if (byteSize > engine->getMaxBufferSize())
      throw Exception(Error::OutOfMemory, "buffer size exceeds maximum");

    try
    {
      VkBufferCreateInfo bci{};
      bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      bci.size = byteSize;
      bci.usage = getCommonVkBufferUsageFlags();

      VkResult res = vkCreateBuffer(engine->getVkDevice(), &bci, nullptr, &buffer);
      checkResult(res);

      if (arena)
      {
        // the heap is created with the same buffer usage and flags
        // so we should be able to bind the buffer successfully, as per the spec
        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(engine->getVkDevice(), buffer, &memReqs);
        const bool typeOk = (memReqs.memoryTypeBits & (1u << heap->memoryTypeIndex)) != 0;
        assert(typeOk);
      }
      else
      {
        heap = makeRef<VulkanHeap>(engine, byteSize, storage);
      }

      res = vkBindBufferMemory(engine->getVkDevice(), buffer, *heap, byteOffset);
      checkResult(res);
    }
    catch (...)
    {
      free();
      throw;
    }
  }

  void VulkanBuffer::free()
  {
    if (buffer)
    {
      vkDestroyBuffer(engine->getVkDevice(), buffer, nullptr);
      buffer = VK_NULL_HANDLE;
    }

    heap.reset();
  }

  void VulkanBuffer::preRealloc()
  {
    Buffer::preRealloc();
    free();
  }

  void VulkanBuffer::postRealloc()
  {
    init();
    Buffer::postRealloc();
  }

  void* VulkanBuffer::getPtr() const
  {
    if (!buffer)
      return nullptr;

    VkBufferDeviceAddressInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buffer;

    VkDeviceAddress a = vkGetBufferDeviceAddress(engine->getVkDevice(), &info);
    return reinterpret_cast<void*>(a);
  }

  void* VulkanBuffer::getHostPtr() const
  {
    return storage != Storage::Device ? static_cast<char*>(heap->mappedPtr) + byteOffset : nullptr;
  }

  void VulkanBuffer::read(size_t byteOffset, size_t byteSize, void* dstHostPtr, SyncMode sync)
  {
    if (byteOffset + byteSize > this->byteSize)
      throw Exception(Error::InvalidArgument, "buffer region is out of bounds");
    if (dstHostPtr == nullptr && byteSize > 0)
      throw Exception(Error::InvalidArgument, "destination host pointer is null");

    if (storage == Storage::Host)
    {
      memcpy(dstHostPtr, static_cast<const char*>(getHostPtr()) + byteOffset, byteSize);
      return;
    }

    Ref<VulkanBuffer> stagingBuffer = makeRef<VulkanBuffer>(engine, byteSize, Storage::Host);

    VkCommandBuffer cmdBuf = engine->beginSingleTimeCommands();

    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = byteOffset;
    copyRegion.dstOffset = 0;
    copyRegion.size = byteSize;
    vkCmdCopyBuffer(cmdBuf, buffer, *stagingBuffer, 1, &copyRegion);

    engine->endSingleTimeCommands(cmdBuf);

    memcpy(dstHostPtr, stagingBuffer->getHostPtr(), byteSize);
  }

  void VulkanBuffer::write(size_t byteOffset, size_t byteSize, const void* srcHostPtr, SyncMode sync)
  {
    if (byteOffset + byteSize > this->byteSize)
      throw Exception(Error::InvalidArgument, "buffer region is out of bounds");
    if (srcHostPtr == nullptr && byteSize > 0)
      throw Exception(Error::InvalidArgument, "source host pointer is null");

    if (storage == Storage::Host)
    {
      memcpy(static_cast<char*>(getHostPtr()) + byteOffset, srcHostPtr, byteSize);
      return;
    }

    Ref<VulkanBuffer> stagingBuffer = makeRef<VulkanBuffer>(engine, byteSize, Storage::Host);

    memcpy(stagingBuffer->getHostPtr(), srcHostPtr, byteSize);

    VkCommandBuffer cmdBuf = engine->beginSingleTimeCommands();

    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = byteOffset;
    copyRegion.size = byteSize;
    vkCmdCopyBuffer(cmdBuf, *stagingBuffer, buffer, 1, &copyRegion);

    engine->endSingleTimeCommands(cmdBuf);
  }

OIDN_NAMESPACE_END
