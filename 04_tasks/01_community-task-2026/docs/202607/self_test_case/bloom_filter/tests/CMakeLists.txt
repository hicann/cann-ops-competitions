# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

function(apply_ascend_config TARGET)
  if(ASCEND_HOME_PATH AND CMAKE_CXX_COMPILER MATCHES "ccec$")
    target_include_directories(${TARGET} PRIVATE
      ${ASCEND_HOME_PATH}/compiler/tikcpp
      ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
      ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
      ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
      ${ASCEND_HOME_PATH}/include/experiment/runtime
      ${ASCEND_HOME_PATH}/include/experiment/msprof
    )

    target_compile_definitions(${TARGET} PRIVATE
      __CCE_KT_TEST__
    )

    target_compile_options(${TARGET} PRIVATE
      -xcce
      --cce-aicore-arch=${CCE_AICORE_ARCH}
      -Wno-macro-redefined
      -fcommon
    )

    target_link_options(${TARGET} PRIVATE
      "-Wl,--allow-multiple-definition"
      "-Wl,--whole-archive"
      "-lruntime"
      "-Wl,--no-whole-archive"
      "-lstdc++"
      "-lm"
    )

    target_link_directories(${TARGET} PRIVATE
      ${ASCEND_HOME_PATH}/runtime/lib64
      ${ASCEND_HOME_PATH}/lib64
    )

    target_link_libraries(${TARGET} PRIVATE
      runtime
      ascendcl
      tiling_api
      platform
      dl
      unified_dlog
    )
  endif()
endfunction()

if(BUILD_TESTS)
  include(${PROJECT_SOURCE_DIR}/3rdparty/Catch2/extras/Catch.cmake)

  file(GLOB TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/static_map/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/static_set/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dynamic_map/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bloom_filter/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/roaring_bitmap/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utility/*.cpp
  )

  function(collection_tests_apply_common TARGET)
    target_link_libraries(${TARGET} PRIVATE
      COLLECTION
      Catch2::Catch2WithMain
      stdc++
      m
    )

    target_include_directories(${TARGET} PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}
    )

    apply_ascend_config(${TARGET})
  endfunction()

  foreach(TEST_SRC IN LISTS TEST_SOURCES)
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
    get_filename_component(TEST_DIR  ${TEST_SRC} DIRECTORY)
    get_filename_component(TEST_DIR_NAME ${TEST_DIR} NAME)

    set(EXE_NAME "collection_tests_${TEST_DIR_NAME}_${TEST_NAME}")

    add_executable(${EXE_NAME} ${TEST_SRC})
    collection_tests_apply_common(${EXE_NAME})

    add_test(NAME ${EXE_NAME} COMMAND ${EXE_NAME})
  endforeach()
endif()


if(BUILD_PERFORMANCE)
  set(PERF_MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/performance/main.cpp)

  # 收集所有性能测试用例文件
  file(GLOB PERF_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/performance/static_map/perf_*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/performance/static_set/perf_*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/performance/dynamic_map/perf_*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/performance/bloom_filter/perf_*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/performance/roaring_bitmap/perf_*.cpp
  )

  # 为每个性能测试文件创建独立的可执行文件
  foreach(PERF_TEST_SRC IN LISTS PERF_TEST_SOURCES)
    get_filename_component(PERF_TEST_NAME ${PERF_TEST_SRC} NAME_WE)
    get_filename_component(PERF_TEST_DIR ${PERF_TEST_SRC} DIRECTORY)
    get_filename_component(PERF_TEST_SUBDIR ${PERF_TEST_DIR} NAME)

    set(EXE_NAME "${PERF_TEST_SUBDIR}_${PERF_TEST_NAME}")

    add_executable(${EXE_NAME} ${PERF_MAIN_SRC} ${PERF_TEST_SRC})

    set_target_properties(${EXE_NAME} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${PERF_TEST_SUBDIR}
    )

    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/${PERF_TEST_SUBDIR})

    target_include_directories(${EXE_NAME} PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/common
      ${CMAKE_CURRENT_SOURCE_DIR}/performance
    )

    target_link_libraries(${EXE_NAME} PRIVATE
      COLLECTION
      stdc++
      m
    )

    apply_ascend_config(${EXE_NAME})

    # 添加测试用例，使用 CTest 运行所有性能测试
    add_test(NAME ${EXE_NAME} COMMAND ${EXE_NAME})
  endforeach()
endif()



