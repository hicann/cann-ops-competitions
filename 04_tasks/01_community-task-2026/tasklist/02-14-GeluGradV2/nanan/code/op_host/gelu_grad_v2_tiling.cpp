/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gelu_grad_v2_tiling.cc
 * \brief
 */
#include "gelu_grad_v2_tiling.h"
#include "log/log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t GM_ALIGN = 512;
constexpr uint32_t RESERVED_UB_SIZE = 8192; // 有些api需要预留ub空间
constexpr uint32_t MAX_TILEDATA = 2048;  // 最大可以到4864


static constexpr uint32_t TPL_NONE = 1;
static constexpr uint32_t TPL_TANH = 2;


void GetTilingKey(const uint32_t dtypeKey, uint32_t& tilingKey)
{
    tilingKey = dtypeKey;
}

// 获取平台信息如ubSize, coreNum
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    // 获取ubsize coreNum
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}


static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    // 1、获取平台运行信息
    uint64_t ubSize;
    int64_t realCoreNum;

    GetPlatformInfo(context, ubSize, realCoreNum);
    GeluGradV2TilingData tiling;

    // get attr info
    const char* approximatePtr = context->GetAttrs()->GetStr(0);
    std::string approximateStr = "";
    approximateStr = approximatePtr;
    uint32_t dtypeKey = 0;
    if (approximateStr == "none") {
        dtypeKey = TPL_NONE;
    } else if (approximateStr == "tanh") {
        dtypeKey = TPL_TANH;
    } else {
        OP_LOGE(context->GetNodeName(), "approximate [%s] not supported, only support [none, tanh]", approximateStr.c_str());
        return ge::GRAPH_FAILED;
    }


    uint64_t inputNum, tileBlockNum, tileDataNum;
    inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();


    uint32_t typeLength = 0;
    auto dyDataType = context->GetInputDesc(0)->GetDataType();
    uint32_t sizeofDataType = 0;
    if (ge::DT_FLOAT == dyDataType) {
        // 4 means the size of float data type in bytes
        sizeofDataType = 4;
    } else if (ge::DT_FLOAT16 == dyDataType) {
        // 2 means the size of float16 data type in bytes
        sizeofDataType = 2;
    } else if (ge::DT_BF16 == dyDataType) {
        // 2 means the size of bfloat16 data type in bytes
        sizeofDataType = 2;
    }    
    typeLength = sizeofDataType;

    uint64_t elemsPerGmBlock = (GM_ALIGN / typeLength);
    uint64_t inputLengthAlgin512 = (inputNum + elemsPerGmBlock - 1) / elemsPerGmBlock * elemsPerGmBlock;

    uint64_t smallCoreDataNum, bigCoreDataNum, smallTailDataNum, bigTailDataNum;
    uint64_t finalSmallTileNum, finalBigTileNum, tailBlockNum;

    tileDataNum = MAX_TILEDATA; 

    // 核间切分
    int64_t coreNum;
    int64_t needCoreNum = (inputLengthAlgin512 + tileDataNum * BUFFER_NUM - 1) / (tileDataNum * BUFFER_NUM);
    coreNum = ((realCoreNum) < needCoreNum) ? realCoreNum : needCoreNum;

    uint64_t needCoreDataNum = ((inputLengthAlgin512 + coreNum - 1) / coreNum);
    // 如果所需要的aicore数量少于1/4的核心数，将核心数*2
    // 如果所需的aicore数量依然少于1/4核心数，继续*2
    if ((coreNum < realCoreNum / 4) && (needCoreDataNum > 2049))
    {
        coreNum = coreNum * 2;
        needCoreDataNum = ((inputLengthAlgin512 + coreNum - 1) / coreNum);
    }
  
    needCoreDataNum = ((inputLengthAlgin512 + coreNum - 1) / coreNum);
    // 如果needCoreDataNum少于MAX_TILEDATA/2，不开启double buffer，否则开启double buffer
    uint32_t bufferNum = 2;
    uint64_t usedDb = 1;
    if (needCoreDataNum < (MAX_TILEDATA /4 * 3))
    {
        bufferNum = 1;
        usedDb = 0;
    }
    uint64_t needTileDataNum = (needCoreDataNum + bufferNum - 1) / bufferNum;
    needTileDataNum = (needTileDataNum + elemsPerGmBlock - 1) / elemsPerGmBlock * elemsPerGmBlock;
    tileDataNum = (tileDataNum < needTileDataNum) ? tileDataNum : needTileDataNum;

    uint64_t everyCoreInputBlockNum = inputLengthAlgin512 / elemsPerGmBlock / coreNum;
    tailBlockNum = (inputLengthAlgin512 / elemsPerGmBlock) % coreNum;
    smallCoreDataNum = everyCoreInputBlockNum * elemsPerGmBlock;
    finalSmallTileNum = (smallCoreDataNum + tileDataNum - 1) / tileDataNum;
    smallTailDataNum = smallCoreDataNum - (finalSmallTileNum - 1) * tileDataNum;
    bigCoreDataNum = smallCoreDataNum + elemsPerGmBlock;
    finalBigTileNum = (bigCoreDataNum + tileDataNum - 1) / tileDataNum;
    bigTailDataNum = bigCoreDataNum - (finalBigTileNum - 1) * tileDataNum;   

    // 设置tiling数据
    tiling.set_smallCoreDataNum(smallCoreDataNum);
    tiling.set_bigCoreDataNum(bigCoreDataNum);
    tiling.set_finalBigTileNum(finalBigTileNum);
    tiling.set_finalSmallTileNum(finalSmallTileNum);
    tiling.set_tileDataNum(tileDataNum);
    tiling.set_smallTailDataNum(smallTailDataNum);
    tiling.set_bigTailDataNum(bigTailDataNum);
    tiling.set_tailBlockNum(tailBlockNum);    // 设置tiling数据
    tiling.set_usedDb(usedDb);

    std::cout << "inputNum: " << inputNum << std::endl;
    std::cout << "coreNum: " << coreNum << std::endl;
    std::cout << "usedDb: " << usedDb << std::endl;
    
    std::cout << "smallCoreDataNum: " << smallCoreDataNum << std::endl;
    std::cout << "bigCoreDataNum: " << bigCoreDataNum << std::endl;
    std::cout << "finalBigTileNum: " << finalBigTileNum << std::endl;
    std::cout << "finalSmallTileNum: " << finalSmallTileNum << std::endl;
    std::cout << "tileDataNum: " << tileDataNum << std::endl;
    std::cout << "smallTailDataNum: " << smallTailDataNum << std::endl;
    std::cout << "bigTailDataNum: " << bigTailDataNum << std::endl;
    std::cout << "tailBlockNum: " << tailBlockNum << std::endl;

    context->SetBlockDim(coreNum);


    uint32_t tilingKey = 0;
    GetTilingKey(dtypeKey, tilingKey);
    context->SetTilingKey(tilingKey);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;   
}

struct GeluGradV2CompileInfo {
};

static ge::graphStatus TilingParse4GeluGradV2(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        OP_LOGE(context, "TilingParse4GeluGradV2 got context is nullptr.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GeluGradV2).Tiling(TilingFunc).TilingParse<GeluGradV2CompileInfo>(TilingParse4GeluGradV2);
} // namespace optiling
