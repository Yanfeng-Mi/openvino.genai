# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.c")
list(APPEND SOURCE_FILES "${CMAKE_CURRENT_BINARY_DIR}/version.cpp")

# Dependencies

include(FetchContent)

set(XGRAMMAR_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/sampling/structured_output/xgrammar_backend.hpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/src/sampling/structured_output/xgrammar_backend.cpp)
list(REMOVE_ITEM SOURCE_FILES ${XGRAMMAR_SOURCES})

# generate version files
configure_file("${OpenVINOGenAI_SOURCE_DIR}/cmake/templates/version.hpp.in"
               "${CMAKE_CURRENT_BINARY_DIR}/openvino/genai/version.hpp" @ONLY)

configure_file("${OpenVINOGenAI_SOURCE_DIR}/cmake/templates/version.cpp.in"
               "${CMAKE_CURRENT_BINARY_DIR}/version.cpp" @ONLY)


function(add_test_executable target_name)
    add_executable(${target_name} ${target_name}.cpp)
    target_link_libraries(${target_name} PRIVATE openvino::runtime openvino::core::dev openvino::runtime::dev openvino::threading)
    set_target_properties(${target_name} PROPERTIES
        # Ensure out-of-box LC_RPATH on macOS with SIP
        INSTALL_RPATH_USE_LINK_PATH ON)
    install(TARGETS ${target_name}
            RUNTIME DESTINATION test_bin/
            COMPONENT samples_bin
            EXCLUDE_FROM_ALL)
endfunction()

set (TEST_LIST
    fused_mlp_test
    moe_q41_moe3gemm_test
    test_moe_layer)

foreach(testcase IN LISTS TEST_LIST)
    add_test_executable(${testcase})
endforeach()

# test_moe_layer needs access to gguf_utils headers and gguflib
# Also needs to compile building_blocks.cpp and gguf.cpp which contain required functions
target_sources(test_moe_layer PRIVATE 
    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/gguf_utils/building_blocks.cpp"
    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/gguf_utils/gguf.cpp"
    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/gguf_utils/gguf_quants.cpp")
target_include_directories(test_moe_layer PRIVATE "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src")
if(ENABLE_GGUF AND TARGET gguflib)
    target_link_libraries(test_moe_layer PRIVATE gguflib)
    target_compile_definitions(test_moe_layer PRIVATE ENABLE_GGUF)
endif()
