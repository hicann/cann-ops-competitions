set(OP_NAME roll)
set(UT_EXE ${OP_NAME}_op_api_ut)

if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION "Ascend910B")
endif()

message(STATUS "Building op_api UT for SOC_VERSION: ${SOC_VERSION}")

set(OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_api)

set(OP_API_SRCS
    ${OP_API_DIR}/${OP_NAME}.cpp
    ${OP_API_DIR}/aclnn_${OP_NAME}.cpp
)

set(UT_TEST_SRCS
    test_aclnn_${OP_NAME}.cpp
)

set(UT_MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_op_api_main.cpp)

add_executable(${UT_EXE}
    ${UT_MAIN_SRC}
    ${UT_TEST_SRCS}
    ${OP_API_SRCS}
)

target_include_directories(${UT_EXE} PRIVATE
    ${UT_COMMON_INCLUDE_DIRS}
    ${OP_API_DIR}
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include/aclnn
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel
)

target_compile_definitions(${UT_EXE} PRIVATE
    BUILD_SOC_VERSION=${SOC_VERSION}
    _GLIBCXX_USE_CXX11_ABI=0
    GOOGLE_TEST
)

target_compile_options(${UT_EXE} PRIVATE
    -std=c++11
    -fPIE -fPIC
    -Wall -Werror
    -fno-access-control
)

target_link_directories(${UT_EXE} PRIVATE
    ${UT_COMMON_LIB_DIRS}
    $ENV{ASCEND_HOME_PATH}/lib64
)

target_link_libraries(${UT_EXE} PRIVATE
    gtest
    gtest_main
    gmock
    gmock_main
    metadef
    unified_dlog
    opp_registry
    register
    graph
    graph_base
    platform
    c_sec
    mmpa
    nnopbase
    opapi
    nnopbase
    runtime
    ascend_dump
    exe_graph
    acl_rt
    error_manager
    dummy_tls
    ascendalog
    runtime_common
    profapi
    ascend_protobuf
    acl_rt_impl
    ascend_watchdog
    ascend_trace
    dl
    pthread
    rt
)

set_target_properties(${UT_EXE} PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
    INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../../../../build/lib:\$ENV{ASCEND_HOME_PATH}/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver"
)

message(STATUS "")
message(STATUS "=== op_api UT Build Configuration ===")
message(STATUS "  UT Executable: ${UT_EXE}")
message(STATUS "  SOC Version: ${SOC_VERSION}")
message(STATUS "")
