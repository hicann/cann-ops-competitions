cmake_minimum_required(VERSION 3.16)
project(example_maxpool2d_backward CXX)

set(CMAKE_CXX_STANDARD 17)

# ASCEND_TOOLKIT_HOME
if(NOT DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(ASCEND_TOOLKIT_HOME "/usr/local/Ascend/ascend-toolkit/latest")
else()
    set(ASCEND_TOOLKIT_HOME $ENV{ASCEND_TOOLKIT_HOME})
endif()

include_directories(
    ${ASCEND_TOOLKIT_HOME}/include
    ${ASCEND_TOOLKIT_HOME}/opp/vendors/customize/op_api/include
)

add_executable(test_aclnn_example test_aclnn_max_pool2d_with_mask_backward.cpp)
