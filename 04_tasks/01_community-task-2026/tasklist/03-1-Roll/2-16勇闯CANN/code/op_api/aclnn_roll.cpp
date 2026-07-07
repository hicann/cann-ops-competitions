/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_roll.h"
#include "roll.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/platform.h"
#include "opdev/make_op_executor.h"
#include "op_api/level2_base.h"
#include "op_api/aclnn_check.h"
#include "op_api/op_api_def.h"

using namespace op;

static const std::initializer_list<DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT8,
    DataType::DT_UINT8, DataType::DT_INT32,   DataType::DT_UINT32,
    DataType::DT_BF16};

static const std::initializer_list<DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT8, DataType::DT_UINT8,
    DataType::DT_INT32, DataType::DT_UINT32,  DataType::DT_BOOL, DataType::DT_INT64};

static inline bool CheckNotNull(
    const aclTensor* x, const aclIntArray* shifts, const aclIntArray* dims, const aclTensor* out)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(shifts, return false);
    OP_CHECK_NULL(dims, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor* x, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_MATCH(x, out->GetDataType(), return false);
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    bool is910BSocVersion = npuArch == NpuArch::DAV_2201 || IsRegBase(npuArch);
    const std::initializer_list<DataType> supportList =
        is910BSocVersion ? ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST : ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
    OP_CHECK_DTYPE_NOT_SUPPORT(x, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);
    return true;
}

static inline bool CheckShape(const aclTensor* x, const aclTensor* out)
{
    OP_CHECK_SHAPE_NOT_EQUAL(x, out, return false);
    return true;
}

static inline bool CheckArraySize(const aclIntArray* shifts, const aclIntArray* dims)
{
    if (shifts->Size() != dims->Size() && dims->Size() != 0U) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "The size of shifts and dims should be the same when the size of dims is not 0.");
        return false;
    }

    if (dims->Size() == 0U && shifts->Size() != 1U) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size of shifts must be 1 if the size of dims is 0.");
        return false;
    }
    return true;
}

static bool CheckDimsRange(const aclTensor* x, const aclIntArray* dims)
{
    auto tensorDimSize = static_cast<int64_t>(x->GetViewShape().GetDimNum());
    int64_t dimSize = static_cast<int64_t>(dims->Size());
    for (int64_t i = 0; i < dimSize; i++) {
        int64_t curDim = (*dims)[i];
        auto dimMax = std::max(-1 * tensorDimSize, tensorDimSize - 1);
        auto dimMin = std::min(-1 * tensorDimSize, tensorDimSize - 1);
        if ((curDim > dimMax) || (curDim < dimMin)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The values of dims should be in range [%ld, %ld].", dimMin, dimMax);
            return false;
        }
    }
    return true;
}

static inline bool CheckTensorDimSize(const aclTensor* x)
{
    OP_CHECK_MAX_DIM(x, MAX_SUPPORT_DIMS_NUMS, return false);
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* x, const aclIntArray* shifts, const aclIntArray* dims, const aclTensor* out)
{
    CHECK_RET(CheckNotNull(x, shifts, dims, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(x, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(x, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckArraySize(shifts, dims), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorDimSize(x), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// 处理0维tensor场景：dims必须为size 0, shifts必须为size 1。
static aclnnStatus HandleDimZeroTensor(
    const aclTensor* self, const aclIntArray* shifts, const aclIntArray* dims, const aclTensor* out,
    aclOpExecutor* executor)
{
    if (dims->Size() != 0 || shifts->Size() != 1) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "When tensor x has no dimensions, shifts should be size 1, dims should be size 0.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto viewCopyRes = l0op::ViewCopy(self, out, executor);
    CHECK_RET(viewCopyRes != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRollGetWorkspaceSize(
    const aclTensor* x, const aclIntArray* shifts, const aclIntArray* dims, const aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnRoll, DFX_IN(x, shifts, dims), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    printf("[CUSTOM] aclnnRollGetWorkspaceSize called, dimsSize=%zu\n", dims->Size());
    fflush(stdout);
    auto ret = CheckParams(x, shifts, dims, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    if (x->GetViewShape().GetDimNum() == 0) {
        auto res = HandleDimZeroTensor(x, shifts, dims, out, uniqueExecutor.get());
        CHECK_RET(res == ACLNN_SUCCESS, res);

        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    CHECK_RET(CheckDimsRange(x, dims), ACLNN_ERR_PARAM_INVALID);

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    bool needCastBool = (x->GetDataType() == DataType::DT_BOOL);
    if (needCastBool) {
        xContiguous = l0op::Cast(xContiguous, DataType::DT_INT8, uniqueExecutor.get());
        CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor* outBase = l0op::Roll(xContiguous, shifts, dims, uniqueExecutor.get());
    CHECK_RET(outBase != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (needCastBool) {
        outBase = l0op::Cast(outBase, DataType::DT_BOOL, uniqueExecutor.get());
        CHECK_RET(outBase != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto viewcopyResult = l0op::ViewCopy(outBase, out, uniqueExecutor.get());
    CHECK_RET(viewcopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRoll(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRoll);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
