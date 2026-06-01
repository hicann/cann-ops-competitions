# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    # Add tiling dependency files for kernel UT.
    # set(elewise_common_tiling_files
    #         ${CANN_ROOT}/ops/built-in/op_tiling/runtime/elewise_tiling.cc
    #         )
    set(roll_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/roll_tiling.cpp
        )
    # list(APPEND roll_tiling_files ${elewise_common_tiling_files})
    # param1: op name in kernel naming style.
    # param2: soc version, such as "ascend910b".
    # param3: compile options for a representative dtype combination.
    # param4: tiling source files required by this op.
    AddOpTestCase(roll "ascend910b" "-DDTYPE_X=float" "${roll_tiling_files}")
endif()
