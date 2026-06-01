/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file aclnn_roll.h
 * @brief ACLNN Roll interface.
 */

#ifndef ACLNN_ROLL_H_
#define ACLNN_ROLL_H_

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnRollGetWorkspaceSize(
    const aclTensor* x, const aclIntArray* shifts, const aclIntArray* dims, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnRoll(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}

#include <initializer_list>

inline aclnnStatus aclnnRollGetWorkspaceSize(
    const aclTensor* x, std::initializer_list<int64_t> shifts, std::initializer_list<int64_t> dims, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    const aclIntArray* shiftsArray = aclCreateIntArray(shifts.begin(), shifts.size());
    const aclIntArray* dimsArray = aclCreateIntArray(dims.begin(), dims.size());
    aclnnStatus ret = aclnnRollGetWorkspaceSize(x, shiftsArray, dimsArray, out, workspaceSize, executor);
    if (shiftsArray != nullptr) {
        (void)aclDestroyIntArray(shiftsArray);
    }
    if (dimsArray != nullptr) {
        (void)aclDestroyIntArray(dimsArray);
    }
    return ret;
}
#endif

#endif // ACLNN_ROLL_H_
