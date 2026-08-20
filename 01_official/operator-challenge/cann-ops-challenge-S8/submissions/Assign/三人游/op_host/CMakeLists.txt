aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} ops_srcs)

npu_op_code_gen(
    SRC ${ops_srcs}
    PACKAGE ${package_name}
    COMPILE_OPTIONS -g
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
)

execute_process(
    COMMAND ${ASCEND_PYTHON_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/patch_assign_codegen.py
            --ini ${ASCEND_AUTOGEN_PATH}/aic-${ASCEND_COMPUTE_UNIT}-ops-info.ini
            --header ${ASCEND_AUTOGEN_PATH}/aclnn_assign.h
            --source ${ASCEND_AUTOGEN_PATH}/aclnn_assign.cpp
    RESULT_VARIABLE patch_result
    ERROR_VARIABLE patch_error
)
if(NOT patch_result EQUAL 0)
    message(FATAL_ERROR "Failed to patch Assign generated files: ${patch_error}")
endif()

file(GLOB autogen_aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
set_source_files_properties(${autogen_aclnn_src} PROPERTIES GENERATED TRUE)
npu_op_library(cust_opapi ACLNN
    ${autogen_aclnn_src}
)

npu_op_library(cust_optiling TILING
    ${ops_srcs}
)

npu_op_package_add(${package_name}
    LIBRARY
        cust_opapi
        cust_optiling
)
