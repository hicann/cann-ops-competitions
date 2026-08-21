/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file relu_grad_v3_infershape.cpp
 * \brief InferShape implementation for ReluGradV3 operator
*/
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
// 定义输入输出索引常量
static constexpr int64_t INPUT_GRADIENTS_IDX = 0;  // 梯度输入索引
static constexpr int64_t INPUT_MASK_IDX = 1;       // mask输入索引
static constexpr int64_t OUTPUT_BACKPROPS_IDX = 0; // 输出backprops索引

static ge::graphStatus InferShapeReluGradV3(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeReluGradV3");

    // 1. 获取输入梯度(gradients)的shape 
    const gert::Shape* gradientsShape = context->GetInputShape(INPUT_GRADIENTS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradientsShape);

    // 2. 获取输入mask的shape 
    const gert::Shape* maskShape = context->GetInputShape(INPUT_MASK_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskShape); 

    // 3. 检查gradients和mask的维度数量是否一致
    size_t gradientsDimNum = gradientsShape->GetDimNum();
    size_t maskDimNum = maskShape->GetDimNum();
    if (gradientsDimNum != maskDimNum) {
        OP_LOGE(context->GetNodeName(), 
                "Input gradients and mask shape dims not match! gradients dim num: %zu, mask dim num: %zu",
                gradientsDimNum, maskDimNum);
        return GRAPH_FAILED;
    }

    // 4. 检查每个维度的大小是否一致
    for (size_t i = 0; i < gradientsDimNum; ++i) {
        int64_t gradDim = gradientsShape->GetDim(i);
        int64_t maskDim = maskShape->GetDim(i);
        if (gradDim != maskDim) {
            OP_LOGE(context->GetNodeName(),
                    "Dim %zu not match! gradients dim: %ld, mask dim: %ld",
                    i, gradDim, maskDim);
            return GRAPH_FAILED;
        }
    }

    // 5. 获取输出backprops的shape 
    gert::Shape* backpropsShape = context->GetOutputShape(OUTPUT_BACKPROPS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, backpropsShape); 

    // 6. 设置输出shape与输入gradients一致
    *backpropsShape = *gradientsShape;

    // 7. 打印输出shape日志
    std::string shapeStr = "[";
    size_t backpropsDimNum = backpropsShape->GetDimNum();
    for (size_t i = 0; i < backpropsDimNum; ++i) {
        if (i > 0) {
            shapeStr += ", ";
        }
        shapeStr += std::to_string(backpropsShape->GetDim(i));
    }
    shapeStr += "]";
    OP_LOGD(context->GetNodeName(), "End to do InferShapeReluGradV3, output shape: %s", shapeStr.c_str());

    return GRAPH_SUCCESS;
}

// 注册InferShape实现
IMPL_OP_INFERSHAPE(ReluGradV3).InferShape(InferShapeReluGradV3);
} // namespace ops