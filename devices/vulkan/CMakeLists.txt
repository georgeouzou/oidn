## Copyright 2026 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22)

# Get the library version
include(${OIDN_ROOT_SOURCE_DIR}/cmake/oidn_version.cmake)

# Project
project(OpenImageDenoise_device_vulkan
  VERSION ${OIDN_VERSION}
  LANGUAGES CXX
)

# Vulkan SDK
find_package(Vulkan 1.2 REQUIRED)

include(oidn_slang)

set(OIDN_VULKAN_SOURCES
  vulkan_module.cpp
  vulkan_device.cpp
  vulkan_engine.cpp
  vulkan_buffer.cpp
  vulkan_heap.cpp
  vulkan_common.cpp
  vulkan_pipeline.cpp
)

add_library(OpenImageDenoise_device_vulkan SHARED
  ${OIDN_VULKAN_SOURCES}
)

slang_target_add_sources(OpenImageDenoise_device_vulkan vulkan_kernels
  SOURCES
    vulkan_kernels.slang
  INCLUDE_DIRECTORIES
    ${OIDN_ROOT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
  ENTRY_POINTS
    inputProcess_f16_hwc_3 inputProcess_f16_hwc_6 inputProcess_f16_hwc_9
    outputProcess imageCopy autoexposureDownsample autoexposureReduce_1024 autoexposureReduceFinal_1024
)

set_target_properties(OpenImageDenoise_device_vulkan PROPERTIES
  OUTPUT_NAME ${OIDN_LIBRARY_NAME}_device_vulkan
  CXX_STANDARD 17
)

if(OIDN_LIBRARY_VERSIONED)
  set_target_properties(OpenImageDenoise_device_vulkan PROPERTIES VERSION ${PROJECT_VERSION})
endif()

target_compile_definitions(OpenImageDenoise_device_vulkan PRIVATE OIDN_COMPILE_VULKAN_HOST)

target_link_libraries(OpenImageDenoise_device_vulkan
  PRIVATE OpenImageDenoise_core Vulkan::Vulkan)

oidn_strip_symbols(OpenImageDenoise_device_vulkan)
oidn_install_module(OpenImageDenoise_device_vulkan)
