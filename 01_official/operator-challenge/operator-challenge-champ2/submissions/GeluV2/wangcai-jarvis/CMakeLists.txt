cmake_minimum_required(VERSION 3.16.0)

project(wangcai_jarvis_gelu_v2)

find_package(ASC REQUIRED)

set(ASCEND_COMPUTE_UNIT ascend910_93 CACHE STRING "Ascend compute unit")
set(package_name custom)

npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}
)

add_subdirectory(op_host)
add_subdirectory(op_kernel)
