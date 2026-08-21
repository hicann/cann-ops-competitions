/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file relu_grad_v3_tiling.cpp
 * \brief ReluGradV3 operator tiling implementation for AscendC
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/relu_grad_v3_tiling_data.h"
#include "../op_kernel/relu_grad_v3_tiling_key.h"

namespace optiling {

using namespace Ops::NN::OpTiling;

constexpr uint32_t BLOCK_SIZE = 32U;         
constexpr uint32_t BUFFER_NUM = 2U;
constexpr uint32_t WS_SYS_SIZE = 64U * 1024U;  // 64KB (this kernel does not use workspace)
constexpr int64_t INPUT_GRADIENTS_IDX = 0; 
constexpr int64_t INPUT_MASK_IDX = 1;       

struct ReluGradV3CompileInfo {};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    OP_LOGI(context->GetNodeName(), "[Tiling] Platform: ubSize=%llu, coreNum=%lld",
            (unsigned long long)ubSize, (long long)coreNum);
    
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context, 
    int64_t& totalIdx, 
    ge::DataType& dataType)
{
    // 1. 获取双输入Shape并校验一致性
    auto gradShape = context->GetInputShape(INPUT_GRADIENTS_IDX);
    auto maskShape = context->GetInputShape(INPUT_MASK_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskShape);

    const gert::Shape& gradShapeObj = gradShape->GetShape();
    const gert::Shape& maskShapeObj = maskShape->GetShape();

    totalIdx = static_cast<int64_t>(gradShapeObj.GetShapeSize());

    // 2. 获取双输入Desc，校验数据类型
    auto gradDesc = context->GetInputDesc(INPUT_GRADIENTS_IDX);
    auto maskDesc = context->GetInputDesc(INPUT_MASK_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskDesc);

    // 校验 mask 类型必须为 DT_UINT8
    ge::DataType maskDtype = maskDesc->GetDataType();
    if (maskDtype != ge::DT_UINT1 && maskDtype != ge::DT_UINT8) {
        OP_LOGE(context, "Invalid mask dtype: expected DT_UINT1/DT_UINT8, got %d",
                static_cast<int>(maskDtype));
        return ge::GRAPH_FAILED;
    }

    // 获取 gradients 数据类型
    dataType = gradDesc->GetDataType();
    const std::set<ge::DataType> supportedDtype = {
        ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_UINT8
    };
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "Invalid gradients dtype: %d", static_cast<int>(dataType));
        return ge::GRAPH_FAILED;
    }

    OP_LOGI(context->GetNodeName(), "[Tiling] Shape: totalIdx=%lld, dataType=%d",
            (long long)totalIdx, static_cast<int>(dataType));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE + sysWorkspaceSize;

    OP_LOGI(context->GetNodeName(), "[Tiling] Workspace: sys=%u, total=%zu (%.2f MB)",
            sysWorkspaceSize, currentWorkspace[0],
            currentWorkspace[0] / (1024.0 * 1024.0));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluGradV3TilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 获取双输入Shape/类型信息
    int64_t totalIdx = 0;
    ge::DataType dataType;
    OP_CHECK_IF(GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 空输入处理
    if (totalIdx <= 0) {
        OP_LOGI(context->GetNodeName(), "[Tiling] Empty input → skip, blockDim=1");
        ReluGradV3TilingData* tiling = context->GetTilingData<ReluGradV3TilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        memset_s(tiling, sizeof(ReluGradV3TilingData), 0, sizeof(ReluGradV3TilingData));
        context->SetBlockDim(1);
        context->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0));
        return ge::GRAPH_SUCCESS;
    }

    // 计算Workspace大小
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    // 初始化TilingData
    ReluGradV3TilingData* tiling = context->GetTilingData<ReluGradV3TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(ReluGradV3TilingData), 0, sizeof(ReluGradV3TilingData)) != EOK,
                OP_LOGE(context, "Init tiling data error"), return ge::GRAPH_FAILED);

    // 单元素字节数
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(dataType, typeLength);
    OP_CHECK_IF(typeLength == 0, OP_LOGE(context, "typeLength is 0"), return ge::GRAPH_FAILED);
    uint64_t inputBytes = static_cast<uint64_t>(typeLength);
    uint64_t inputLengthBytes = static_cast<uint64_t>(totalIdx) * inputBytes;

    constexpr int64_t SAFETY_MARGIN = 4;
    int64_t bufferCoefficient = 0;
    // 精确 bufferCoefficient = 队列(2*sz+2+2*sz) + 临时(maskHalf+cmpMask+其他) + 对齐余量
    //   half/float 用 Compares+Select(无 zeroHalf), 小 tile 回退用 outCast
    //   half:   4+2+4 + 2+0.125       = 12.125 → 14
    //   float:  8+2+8 + 2+0.125 + 8KB_Select = 20.125 -> 24 (Select internal 8KB)
    //   int8:   2+2+2 + 2+0.125+4+4   = 16.125 → 18
    //   uint8:  2+2+2 + 2+0.125+4+4   = 16.125 → 18
    //   int32:  8+2+8 + 2+0.125+4+4+4 = 32.125 → 34
    //   bf16:   4+2+4 + 2+0.125+4+4+4 = 24.125 → 26
    const char* dtypeName = "unknown";
    if (dataType == ge::DT_FLOAT) {
        bufferCoefficient = 24;
        dtypeName = "float";
    } else if (dataType == ge::DT_INT32) {
        bufferCoefficient = 34;
        dtypeName = "int32";
    } else if (dataType == ge::DT_FLOAT16) {
        bufferCoefficient = 14;
        dtypeName = "float16";
    } else if (dataType == ge::DT_BF16) {
        bufferCoefficient = 26;
        dtypeName = "bf16";
    } else if (dataType == ge::DT_INT8 || dataType == ge::DT_UINT8) {
        bufferCoefficient = 18;
        dtypeName = (dataType == ge::DT_INT8) ? "int8" : "uint8";
    }
    uint64_t ubCoeff = static_cast<uint64_t>(bufferCoefficient);

    // 计算安全的 tileDataNum 
    uint64_t elementsPerBlock = BLOCK_SIZE / inputBytes; 
    uint64_t maxTileDataNum = ubSize / ubCoeff;           

    // 向下对齐为 elementsPerBlock 的整数倍 (保证 tileBlockNum 为整数)
    if (maxTileDataNum < elementsPerBlock) {
        maxTileDataNum = elementsPerBlock; // 至少 1 个 Block
    } else {
        maxTileDataNum = (maxTileDataNum / elementsPerBlock) * elementsPerBlock;
    }

    uint32_t tileDataNum = static_cast<uint32_t>(maxTileDataNum);
    uint32_t tileBlockNum = static_cast<uint32_t>(maxTileDataNum / elementsPerBlock);

    // 总Block数 + Core数调整（负载均衡）
    uint64_t blocksTotal = (inputLengthBytes + BLOCK_SIZE - 1ULL) / BLOCK_SIZE;
    uint64_t coreNum64 = static_cast<uint64_t>(coreNum);
    if (coreNum64 > blocksTotal) coreNum64 = blocksTotal; // 避免Core数多于Block数
    if (coreNum64 == 0ULL) coreNum64 = 1ULL;              // 最少1个Core
    uint32_t finalCoreNum = static_cast<uint32_t>(coreNum64);
    // tileDataNum cap for small input
    {
        uint64_t perCore = (uint64_t)totalIdx / coreNum64;
        if (perCore == 0) perCore = 1;
        uint64_t cap = perCore * 4;
        cap = ((cap + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
        if ((uint64_t)tileDataNum > cap) {
            tileDataNum = (uint32_t)cap;
            tileBlockNum = (uint32_t)(cap / elementsPerBlock);
        }
    }

    // Small/Big Core 分块计算
    uint64_t everyCoreInputBlockNum = blocksTotal / coreNum64; // 基础Block数
    uint32_t tailBlockNum = static_cast<uint32_t>(blocksTotal % coreNum64); // 尾Block数

    // Small Core（普通Core）：核心参数计算
    uint64_t smallCoreDataNum_u = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint32_t smallCoreDataNum = static_cast<uint32_t>(smallCoreDataNum_u);
    uint32_t smallTileNum = static_cast<uint32_t>(everyCoreInputBlockNum / static_cast<uint64_t>(tileBlockNum));
    uint32_t finalSmallTileNum = ((everyCoreInputBlockNum % tileBlockNum) == 0) ? smallTileNum : (smallTileNum + 1);
    uint32_t smallTailDataNum = smallCoreDataNum - (finalSmallTileNum - 1) * tileDataNum;

    // Big Core（前tailBlockNum个Core）：核心参数计算
    uint64_t bigEveryCoreBlockNum = everyCoreInputBlockNum + 1ULL;
    uint64_t bigCoreDataNum_u = bigEveryCoreBlockNum * BLOCK_SIZE / inputBytes;
    uint32_t bigCoreDataNum = static_cast<uint32_t>(bigCoreDataNum_u);
    uint32_t bigTileNum = static_cast<uint32_t>(bigEveryCoreBlockNum / static_cast<uint64_t>(tileBlockNum));
    uint32_t finalBigTileNum = ((bigEveryCoreBlockNum % tileBlockNum) == 0) ? bigTileNum : (bigTileNum + 1);
    uint32_t bigTailDataNum = bigCoreDataNum - (finalBigTileNum - 1) * tileDataNum;
    
    // TilingData 赋值
    tiling->smallCoreDataNum = static_cast<uint64_t>(smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<uint64_t>(bigCoreDataNum);
    tiling->tileDataNum = static_cast<uint64_t>(tileDataNum);
    tiling->tailBlockNum = static_cast<uint64_t>(tailBlockNum);
    tiling->smallTailDataNum = static_cast<uint64_t>(smallTailDataNum);
    tiling->bigTailDataNum = static_cast<uint64_t>(bigTailDataNum);
    tiling->finalSmallTileNum = static_cast<uint64_t>(finalSmallTileNum);
    tiling->finalBigTileNum = static_cast<uint64_t>(finalBigTileNum);

    context->SetBlockDim(finalCoreNum);
    
    // 按数据类型设置TilingKey
    // NOTE: schMode 必须与 kernel/relu_grad_v3.cpp 中 TilingKey 枚举值一致:
    //   TILING_KEY_FLOAT16 = 0, TILING_KEY_FLOAT32 = 1, TILING_KEY_INT32 = 2,
    //   TILING_KEY_INT8 = 3, TILING_KEY_UINT8 = 4, TILING_KEY_BFLOAT16 = 5
    uint64_t schMode = 0ULL;
    switch (dataType) {
        case ge::DT_FLOAT16: schMode = ELEMENTWISE_TPL_SCH_MODE_0; break;
        case ge::DT_FLOAT:   schMode = ELEMENTWISE_TPL_SCH_MODE_1; break;
        case ge::DT_INT32:   schMode = ELEMENTWISE_TPL_SCH_MODE_2; break;
        case ge::DT_INT8:    schMode = ELEMENTWISE_TPL_SCH_MODE_3; break;
        case ge::DT_UINT8:   schMode = ELEMENTWISE_TPL_SCH_MODE_4; break;
        case ge::DT_BF16:    schMode = ELEMENTWISE_TPL_SCH_MODE_5; break;
        default: 
            OP_LOGE(context, "Unsupported dtype for TilingKey: %d", static_cast<int>(dataType));
            return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(GET_TPL_TILING_KEY(schMode));

    // 详细诊断日志
    OP_LOGI(context->GetNodeName(),
        "[Tiling] ===== RESULT: dtype=%s(%d) schMode=%llu =====",
        dtypeName, static_cast<int>(dataType), (unsigned long long)schMode);
    OP_LOGI(context->GetNodeName(),
        "[Tiling] totalIdx=%lld, inputBytes=%llu, totalBytes=%llu, blocksTotal=%llu",
        (long long)totalIdx, (unsigned long long)inputBytes,
        (unsigned long long)inputLengthBytes, (unsigned long long)blocksTotal);
    OP_LOGI(context->GetNodeName(),
        "[Tiling] ubSize=%llu, bufferCoeff=%lld, ubCoeff=%llu, elementsPerBlock=%llu",
        (unsigned long long)ubSize, (long long)bufferCoefficient,
        (unsigned long long)ubCoeff, (unsigned long long)elementsPerBlock);
    OP_LOGI(context->GetNodeName(),
        "[Tiling] tileDataNum=%u, tileBlockNum=%u, finalCoreNum=%u, tailBlockNum=%u",
        tileDataNum, tileBlockNum, finalCoreNum, tailBlockNum);
    OP_LOGI(context->GetNodeName(),
        "[Tiling] smallCore: dataNum=%u, tileNum=%u, tailDataNum=%u",
        smallCoreDataNum, finalSmallTileNum, smallTailDataNum);
    OP_LOGI(context->GetNodeName(),
        "[Tiling] bigCore:   dataNum=%u, tileNum=%u, tailDataNum=%u",
        bigCoreDataNum, finalBigTileNum, bigTailDataNum);
    OP_LOGI(context->GetNodeName(),
        "[Tiling] ===== tiling SUCCESS =====");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForReluGradV3([[maybe_unused]] gert::TilingParseContext* context)
{   
    OP_CHECK_NULL_WITH_CONTEXT(context, context); 
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ReluGradV3).Tiling(ReluGradV3TilingFunc).TilingParse<ReluGradV3CompileInfo>(TilingParseForReluGradV3);
} // namespace optiling