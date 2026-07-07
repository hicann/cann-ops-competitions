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
 * \file roll_infershape.cpp
 * \brief Roll infershape: output shape equals input shape
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeRoll(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeRoll");

    const gert::Shape *inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "InputShape is nullptr");
        return ge::GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    if (outputShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "OutputShape is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto dimNum = inputShape->GetDimNum();
    outputShape->SetDimNum(dimNum);
    for (size_t i = 0; i < dimNum; ++i) {
        outputShape->SetDim(i, inputShape->GetDim(i));
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeRoll");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Roll).InferShape(InferShapeRoll);

} // namespace ops
