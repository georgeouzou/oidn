// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/kernel.h"
#include "core/image_accessor.h"

#if !defined(OIDN_COMPILE_METAL_DEVICE) && !defined(OIDN_COMPILE_VULKAN_DEVICE)
  #include "core/image_copy.h"
#endif

OIDN_NAMESPACE_BEGIN

  struct GPUImageCopyKernel
  {
    ImageAccessor src;
    ImageAccessor dst;

  #if !defined(OIDN_COMPILE_VULKAN_DEVICE)
    oidn_device_inline void operator ()(const oidn_private WorkItem<2>& it) const
  #else
    oidn_device_inline void operator ()(WorkItem<2> it)
  #endif
    {
      const int h = it.getGlobalID<0>();
      const int w = it.getGlobalID<1>();
      const vec3f value = src.get3<float>(h, w);
      dst.set3(h, w, value);
    }
  };

#if !defined(OIDN_COMPILE_METAL_DEVICE) && !defined(OIDN_COMPILE_VULKAN_DEVICE)

  template<typename EngineT>
  class GPUImageCopy final : public ImageCopy
  {
  public:
    explicit GPUImageCopy(EngineT* engine)
      : engine(engine) {}

    Engine* getEngine() const override { return engine; }

  #if defined(OIDN_COMPILE_METAL)
    void finalize() override
    {
      pipeline = engine->newPipeline("imageCopy");
    }
  #elif defined(OIDN_COMPILE_VULKAN_HOST)
    void finalize() override
    {
      pipeline = engine->template newComputePipeline<GPUImageCopyKernel, 2>("imageCopy");
    }
  #endif

    void submitKernels(const Ref<CancellationToken>& ct) override
    {
      check();

      GPUImageCopyKernel kernel;
      kernel.src = *src;
      kernel.dst = *dst;

    #if defined(OIDN_COMPILE_METAL)
      engine->submitKernel(WorkDim<2>(dst->getH(), dst->getW()), kernel,
                           pipeline, {src->getBuffer(), dst->getBuffer()});
    #elif defined(OIDN_COMPILE_VULKAN_HOST)
      engine->submitKernelGlobal(WorkDim<2>(dst->getH(), dst->getW()), kernel, pipeline);
    #else
      engine->submitKernel(WorkDim<2>(dst->getH(), dst->getW()), kernel);
    #endif
    }

  private:
    EngineT* engine;

  #if defined(OIDN_COMPILE_METAL)
    Ref<MetalPipeline> pipeline;
  #elif defined(OIDN_COMPILE_VULKAN_HOST)
    Ref<VulkanComputePipeline> pipeline;
  #endif
  };

#endif // !defined(OIDN_COMPILE_METAL_DEVICE) && !defined(OIDN_COMPILE_VULKAN_DEVICE)

OIDN_NAMESPACE_END
