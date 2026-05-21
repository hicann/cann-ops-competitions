/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_adagrad_d_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_1 = 1;
static constexpr int64_t IDX_2 = 2;
static constexpr int64_t IDX_3 = 3;

static ge::graphStatus InferShapeApplyAdagradD(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeApplyAdagradD");

    const gert::Shape* varShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShape);

    const gert::Shape* accumShape = context->GetInputShape(IDX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, accumShape);

    gert::Shape* varOutShape = context->GetOutputShape(IDX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context, varOutShape);

    gert::Shape* accumOutShape = context->GetOutputShape(IDX_3);
    OP_CHECK_NULL_WITH_CONTEXT(context, accumOutShape);

    *varOutShape = *varShape;
    *accumOutShape = *accumShape;

    OP_LOGD(context->GetNodeName(), "End to do InferShapeApplyAdagradD");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyAdagradD).InferShape(InferShapeApplyAdagradD);
}  // namespace ops
