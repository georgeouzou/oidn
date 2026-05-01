// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/kernel.h"
#include "core/tensor_accessor.h"

#if defined(OIDN_COMPILE_VULKAN_HOST)
#include "core/conv.h"
#include "vulkan_engine.h"
#endif

OIDN_NAMESPACE_BEGIN

  struct VulkanConvKernel
  {
    TensorAccessor3D<half, TensorLayout::hwc> src;
    TensorAccessor4D<half, TensorLayout::ohwi> weight;
    TensorAccessor3D<half, TensorLayout::hwc> bias;
    TensorAccessor3D<half, TensorLayout::hwc> dst;
    KernelBool relu;
    uint32_t postOp;
  };

#if defined(OIDN_COMPILE_VULKAN_HOST)

  class VulkanConv final : public Conv
  {
  public:
    VulkanConv(VulkanEngine* engine, const ConvDesc& desc)
      : Conv(desc),
        engine(engine)
    {}

    Engine* getEngine() const override { return engine; }

    void finalize() override
    {
      pipeline = engine->newComputePipeline<VulkanConvKernel, 3>("conv");
    }

    void submitKernels(const Ref<CancellationToken>& ct) override
    {
      VulkanConvKernel kernel;
      kernel.src    = *src;
      kernel.weight = *weight;
      kernel.bias   = makeBiasAccessor();
      kernel.dst    = *dst;
      kernel.relu   = KernelBool(activation == Activation::ReLU ? 1 : 0);
      kernel.postOp = uint32_t(postOp == PostOp::Pool ? 1 :
                               postOp == PostOp::Upsample ? 2 : 0);

      WorkDim<3> globalSize(0, 0, 0);
      // hwc access
      if (postOp == PostOp::Upsample)
        globalSize = WorkDim<3>(src->getH(), src->getW(), dst->getPaddedC());
      else
        globalSize = WorkDim<3>(dst->getH(), dst->getW(), dst->getPaddedC());

      engine->submitKernelGlobal(globalSize, kernel, pipeline);
    }

  private:
    TensorAccessor3D<half, TensorLayout::hwc> makeBiasAccessor() const
    {
      if (bias->getRank() == 1)
      {
        TensorAccessor3D<half, TensorLayout::hwc> acc;
        acc.getByteOffset.wByteStride = 0;
        acc.getByteOffset.hByteStride = 0;
        acc.ptr = static_cast<char*>(bias->getPtr());
        acc.C = bias->getPaddedX();
        acc.H = 1;
        acc.W = 1;
        return acc;
      }
      return static_cast<TensorAccessor3D<half, TensorLayout::hwc>>(*bias);
    }

    VulkanEngine* engine;
    Ref<VulkanComputePipeline> pipeline;
  };

#endif // defined(OIDN_COMPILE_VULKAN_HOST)

OIDN_NAMESPACE_END
