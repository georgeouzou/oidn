## Copyright 2026 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

# Find slangc compiler
find_program(SLANG_COMPILER slangc
  HINTS
    ${Vulkan_INCLUDE_DIR}/../bin
    $ENV{VULKAN_SDK}/bin
    $ENV{VULKAN_SDK}/Bin
  DOC "Path to the Vulkan SDK slangc compiler"
)

if(NOT SLANG_COMPILER)
  message(FATAL_ERROR "Slangc compiler not found. Please ensure it is available in the Vulkan SDK bin directory or in PATH.")
endif()

message(STATUS "Found slangc: ${SLANG_COMPILER}")

# Builds SPIRV from the given slang shader sources and
# adds C++ sources generated from the SPIRV binary blob to the specified target
function(slang_target_add_sources target module_name)
  set(options SOURCES INCLUDE_DIRECTORIES ENTRY_POINTS)
  cmake_parse_arguments(PARSE_ARGV 2 SLANG "" "" "${options}")

  set(include_dirs "")
  foreach(inc ${SLANG_INCLUDE_DIRECTORIES})
    file(TO_NATIVE_PATH "${inc}" inc_native_path)
    list(APPEND include_dirs "-I${inc_native_path}")
  endforeach()

  set(src ${SLANG_SOURCES})
  get_filename_component(src_file ${src} ABSOLUTE)
  get_filename_component(src_dir  ${src_file} DIRECTORY)
  oidn_get_build_path(out_dir ${src_dir})

  # Generate SPIR-V for each slang entry points
  set(spirv_files "")
  foreach(entry ${SLANG_ENTRY_POINTS})
    set(spirv_file ${out_dir}/CMakeFiles/${target}.dir/${entry}.spv)
    file(RELATIVE_PATH spirv_file_rel ${CMAKE_BINARY_DIR} ${spirv_file})

    add_custom_command(
      OUTPUT ${spirv_file}
      COMMAND ${SLANG_COMPILER}
        ${include_dirs}
        -target spirv
        -stage compute
        -entry ${entry}
        -o ${spirv_file}
        ${src_file}
      DEPENDS ${src_file}
      COMMENT "Building SPIR-V file ${spirv_file_rel}"
    )

    list(APPEND spirv_files ${spirv_file})
  endforeach()

  # Generate C++ code from SPIR-V blobs
  set(cpp_files "")
  foreach(spirv_file ${spirv_files})
    oidn_generate_cpp_from_blob(gen_cpp_files "${OIDN_NAMESPACE}::blobs" ${spirv_file})
    list(APPEND cpp_files ${gen_cpp_files})
  endforeach()

  # Add the generated C++ files to the target
  get_property(target_sources TARGET ${target} PROPERTY SOURCES)
  list(APPEND target_sources ${cpp_files})
  set_target_properties(${target} PROPERTIES SOURCES "${target_sources}")
endfunction()
