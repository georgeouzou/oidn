// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

struct InputProcessParams
{
  // TODO
};

struct PushConstants
{
  InputProcessParams *params;
};

[[vk::push_constant]]
PushConstants pushConstants;

[shader("compute")]
[numthreads(16, 16, 1)]
void inputProcess1(uint3 dispatchThreadID : SV_DispatchThreadID)
{
  // TODO
}

[shader("compute")]
[numthreads(16, 16, 1)]
void inputProcess2(uint3 dispatchThreadID : SV_DispatchThreadID)
{
  // TODO
}

[shader("compute")]
[numthreads(16, 16, 1)]
void inputProcess3(uint3 dispatchThreadID : SV_DispatchThreadID)
{
  // TODO
}
