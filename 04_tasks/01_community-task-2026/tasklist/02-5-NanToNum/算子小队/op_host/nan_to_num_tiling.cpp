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
 * \file nan_to_num_tiling.cpp
 * \brief Tiling implementation for nan_to_num operator.
 */

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/nan_to_num_tiling_data.h"
#include "../op_kernel/nan_to_num_tiling_key.h"

namespace optiling {

using namespace Ops::Math::OpTiling;

namespace {
constexpr uint32_t UB_NUM_BF16 = 9U;
constexpr uint32_t UB_NUM_OTHER = 5U;
constexpr uint32_t BLOCK_SIZE = 256U;

constexpr float FLOAT_MAX = 3.4028235e+38f;
constexpr float FLOAT_MIN = -3.4028235e+38f;
constexpr float FLOAT16_MAX = 65504.0f;
constexpr float BFLOAT16_MAX = 3.3895314e+38f;
}

struct NanToNumCompileInfo {};

struct CoreTilingParams {
    uint64_t smallCoreDataNum = 0;
    uint64_t bigCoreDataNum = 0;
    uint64_t smallTailDataNum = 0;
    uint64_t bigTailDataNum = 0;
    uint64_t finalSmallTileNum = 0;
    uint64_t finalBigTileNum = 0;
    uint64_t tailBlockNum = 0;
};

struct ShapeInfo {
    uint64_t inputNum = 0;
    uint64_t inputBytes = 0;
    uint64_t tileBlockNum = 0;
    uint64_t tileDataNum = 0;
    uint64_t inputLengthAlgin = 0;
};

static ge::graphStatus TilingParseForNanToNum([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC ascendcPlatform(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = ascendcPlatform.GetCoreNum();

    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspaceSize(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC ascendcPlatform(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeInfo(gert::TilingContext* context, uint64_t ubSize, ShapeInfo& shapeInfo)
{
    OP_CHECK_IF(
        context == nullptr || context->GetInputShape(0) == nullptr, OP_LOGE(context, "context is nullptr"),
        return ge::GRAPH_FAILED);

    shapeInfo.inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);

    OP_CHECK_IF(shapeInfo.inputNum == 0 || typeLength == 0,
        OP_LOGE(context, "inputNum or typeLength is 0"), return ge::GRAPH_FAILED);

    shapeInfo.inputBytes = typeLength;
    uint64_t inputLength = shapeInfo.inputNum * typeLength;
    uint64_t ubDataNumber = (context->GetInputDesc(0)->GetDataType() == ge::DT_BF16) ? UB_NUM_BF16 : UB_NUM_OTHER;

    OP_CHECK_IF(ubDataNumber == 0, OP_LOGE(context, "ubDataNumber is 0"), return ge::GRAPH_FAILED);

    shapeInfo.tileBlockNum = ubSize / BLOCK_SIZE / ubDataNumber;
    shapeInfo.tileDataNum = shapeInfo.tileBlockNum * BLOCK_SIZE / shapeInfo.inputBytes;
    shapeInfo.inputLengthAlgin = (inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateCoreTilingParams(
    gert::TilingContext* context, uint64_t inputLengthAlgin, int64_t coreNum,
    uint64_t tileBlockNum, uint64_t inputBytes, uint64_t tileDataNum, CoreTilingParams& params)
{
    OP_CHECK_IF(coreNum == 0 || tileBlockNum == 0 || inputBytes == 0,
        OP_LOGE(context, "coreNum or tileBlockNum or inputBytes is 0"), return ge::GRAPH_FAILED);

    uint64_t totalBlockNum = inputLengthAlgin / BLOCK_SIZE;
    uint64_t baseBlockNum = totalBlockNum / coreNum;
    params.tailBlockNum = totalBlockNum % coreNum;

    params.smallCoreDataNum = baseBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t smallTileNum = baseBlockNum / tileBlockNum;
    params.finalSmallTileNum = (baseBlockNum % tileBlockNum == 0) ? smallTileNum : smallTileNum + 1;
    params.smallTailDataNum = params.smallCoreDataNum - tileDataNum * smallTileNum;
    params.smallTailDataNum = (params.smallTailDataNum == 0) ? tileDataNum : params.smallTailDataNum;

    uint64_t bigBlockNum = baseBlockNum + 1;
    params.bigCoreDataNum = bigBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t bigTileNum = bigBlockNum / tileBlockNum;
    params.finalBigTileNum = (bigBlockNum % tileBlockNum == 0) ? bigTileNum : bigTileNum + 1;
    params.bigTailDataNum = params.bigCoreDataNum - tileDataNum * bigTileNum;
    params.bigTailDataNum = (params.bigTailDataNum == 0) ? tileDataNum : params.bigTailDataNum;

    return ge::GRAPH_SUCCESS;
}

static float GetMaxValueByDataType(ge::DataType dType)
{
    switch (dType) {
        case ge::DT_FLOAT:
            return FLOAT_MAX;
        case ge::DT_FLOAT16:
            return FLOAT16_MAX;
        default:
            return BFLOAT16_MAX;
    }
}

static void SetTilingAttrValues(NanToNumTilingData* tiling, gert::TilingContext* context)
{
    float nanValue = 0.0f;
    float posinf = 0.0f;
    float neginf = 0.0f;

    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* attrNan = attrs->GetFloat(0);
        const float* attrPosinf = attrs->GetFloat(1);
        const float* attrNeginf = attrs->GetFloat(2);
        ge::DataType dType = context->GetInputDesc(0)->GetDataType();

        nanValue = (attrNan != nullptr) ? *attrNan : 0.0f;

        float maxVal = GetMaxValueByDataType(dType);
        posinf = (attrPosinf != nullptr) ? *attrPosinf : maxVal;
        neginf = (attrNeginf != nullptr) ? *attrNeginf : -maxVal;
    }

    tiling->nanValue = nanValue;
    tiling->posinf = posinf;
    tiling->neginf = neginf;
}

static ge::graphStatus NanToNumTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo failed"), return ge::GRAPH_FAILED);

    ShapeInfo shapeInfo;
    ret = GetShapeInfo(context, ubSize, shapeInfo);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeInfo failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(SetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "SetWorkspaceSize failed"), return ge::GRAPH_FAILED);

    NanToNumTilingData* tiling = context->GetTilingData<NanToNumTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(NanToNumTilingData), 0, sizeof(NanToNumTilingData)) != EOK,
        OP_LOGE(context, "memset_s failed"), return ge::GRAPH_FAILED);

    if (shapeInfo.tileDataNum >= shapeInfo.inputNum) {
        coreNum = 1;
    } else {
        uint64_t totalBlockNum = shapeInfo.inputLengthAlgin / BLOCK_SIZE;
        coreNum = std::min(static_cast<uint64_t>(coreNum), totalBlockNum);
    }

    CoreTilingParams params;
    ret = CalculateCoreTilingParams(
        context, shapeInfo.inputLengthAlgin, coreNum, shapeInfo.tileBlockNum,
        shapeInfo.inputBytes, shapeInfo.tileDataNum, params);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "CalculateCoreTilingParams failed"), return ge::GRAPH_FAILED);

    tiling->smallCoreDataNum = params.smallCoreDataNum;
    tiling->bigCoreDataNum = params.bigCoreDataNum;
    tiling->tileDataNum = shapeInfo.tileDataNum;
    tiling->smallTailDataNum = params.smallTailDataNum;
    tiling->bigTailDataNum = params.bigTailDataNum;
    tiling->finalSmallTileNum = params.finalSmallTileNum;
    tiling->finalBigTileNum = params.finalBigTileNum;
    tiling->tailBlockNum = params.tailBlockNum;

    SetTilingAttrValues(tiling, context);

    context->SetBlockDim(coreNum);
    context->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NanToNum).Tiling(NanToNumTilingFunc).TilingParse<NanToNumCompileInfo>(TilingParseForNanToNum);

} // namespace optiling
