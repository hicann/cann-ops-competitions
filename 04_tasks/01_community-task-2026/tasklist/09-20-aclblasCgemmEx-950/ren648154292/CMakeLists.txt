cmake_minimum_required(VERSION 3.17)
project(aclblasCgemmEx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# 本算子是 aclblas 句柄式库的一部分，不是自定义算子（custom_opp）。
# 实际编译走 ops-blas 仓的 blas/CMakeLists.txt file(GLOB_RECURSE *.cpp)。
# 本 CMakeLists.txt 作为**任务书 §「目录结构要求」交付件**，用于说明源码构成。
#
# iter4b（2026-09-18）起 aclblasCgemmEx 迭代为薄 wrapper：
#   参数校验 → 类型分发 → 委托 upstream aclblasCgemm / aclblasSgemm
#   op_kernel/ 自研 kernel 交付件已整体删除（数值不可用，见 reports/）。
#   本仓源码构成仅保留 op_host/ 一份 .cpp。

add_library(cgemm_ex STATIC
    op_host/cgemm_ex_host.cpp
)

target_include_directories(cgemm_ex PUBLIC
    op_host
)
