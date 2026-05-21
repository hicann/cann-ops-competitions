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
 * \file apply_adagrad_d_tiling.cpp
 * \brief
 */

 #include "log/log.h"
 #include "util/math_util.h"
 #include "op_host/tiling_util.h"
 #include "op_host/tiling_templates_registry.h"
 #include "../op_kernel/apply_adagrad_d_tiling_data.h"
 #include "../op_kernel/apply_adagrad_d_tiling_key.h"
 #include "tiling/platform/platform_ascendc.h"
 
 namespace optiling {
 
 using namespace Ops::NN::OpTiling;
 using namespace platform_ascendc;
 
 constexpr uint32_t ALIGN_SIZE = 256;
 constexpr uint32_t UB_DATA_NUMBER_FLOAT = 12;
 constexpr uint32_t UB_DATA_NUMBER_FLOAT16 = 20;
 constexpr uint32_t UB_DATA_NUMBER_BF16 = 20;
 
 const std::set<ge::DataType> supportedDtype = {
     ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16
 };
 
 struct ApplyAdagradDCompileInfo {};
 
 // 获取平台信息如coreNum, wsSysSize
 static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, int64_t& coreNum, uint32_t& wsSysSize)
 {
     fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
     OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
     auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
     coreNum = ascendcPlatform.GetCoreNumAiv();
     OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
     wsSysSize = ascendcPlatform.GetLibApiWorkSpaceSize();
     return ge::GRAPH_SUCCESS;
 }
 
 // 获取ubsize
 static ge::graphStatus GetUBSize(gert::TilingContext* context, uint64_t& ubSize)
 {
     fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
     OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
     auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
     ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
     OP_CHECK_IF(ubSize <= 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
     return ge::GRAPH_SUCCESS;
 }

 inline const gert::Shape& EnsureNotScalar(const gert::Shape& inShape)
{
    static const gert::Shape g_vec_1_shape = {1};
    if (inShape.IsScalar()) {
        return g_vec_1_shape;
    }
    return inShape;
}
 
 // 获取属性、shape信息
 static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, ge::DataType& dataType,
                                          int64_t& inputLength, int32_t& inputBytes, int32_t& dataNumber)
 {
     auto varShape = context->GetInputShape(0);
     OP_CHECK_NULL_WITH_CONTEXT(context, varShape);
     auto accumShape = context->GetInputShape(1);
     OP_CHECK_NULL_WITH_CONTEXT(context, accumShape);
     auto lrShape = context->GetInputShape(2);
     OP_CHECK_NULL_WITH_CONTEXT(context, lrShape);
     auto gradShape = context->GetInputShape(3);
     OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);
 
     auto inputShapeVar = EnsureNotScalar(varShape->GetStorageShape());
     auto inputNum = inputShapeVar.GetShapeSize();
 
     auto outVar = context->GetOutputShape(0);
     OP_CHECK_NULL_WITH_CONTEXT(context, outVar);
     auto outAccum = context->GetOutputShape(1);
     OP_CHECK_NULL_WITH_CONTEXT(context, outAccum);

     auto varDesc = context->GetInputDesc(0);
     OP_CHECK_NULL_WITH_CONTEXT(context, varDesc);
     auto accumDesc = context->GetInputDesc(1);
     OP_CHECK_NULL_WITH_CONTEXT(context, accumDesc);
     auto lrDesc = context->GetInputDesc(2);
     OP_CHECK_NULL_WITH_CONTEXT(context, lrDesc);
     auto gradDesc = context->GetInputDesc(3);
     OP_CHECK_NULL_WITH_CONTEXT(context, gradDesc);
 
     dataType = varDesc->GetDataType();
     if (supportedDtype.count(dataType) == 0) {
         OP_LOGE(context, "invalid dtype");
         return ge::GRAPH_FAILED;
     }
 
     OP_CHECK_IF(accumDesc->GetDataType() != dataType, OP_LOGE(context, "accum dtype is not same as var"), return ge::GRAPH_FAILED);
     OP_CHECK_IF(lrDesc->GetDataType() != dataType, OP_LOGE(context, "lr dtype is not same as var"), return ge::GRAPH_FAILED);
     OP_CHECK_IF(gradDesc->GetDataType() != dataType, OP_LOGE(context, "grad dtype is not same as var"), return ge::GRAPH_FAILED);
 
     inputBytes = GetSizeByDataType(dataType);
     inputLength = inputBytes * inputNum;
 
     dataNumber = 1;
     if (dataType == ge::DT_FLOAT) {
         dataNumber = UB_DATA_NUMBER_FLOAT;
     } else if (dataType == ge::DT_FLOAT16) {
         dataNumber = UB_DATA_NUMBER_FLOAT16;
     } else if (dataType == ge::DT_BF16) {
         dataNumber = UB_DATA_NUMBER_BF16;
     }
 
     return ge::GRAPH_SUCCESS;
 }
 
 static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context, uint32_t& wsSysSize)
 {
     size_t* currentWorkspace = context->GetWorkspaceSizes(1);
     OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
     currentWorkspace[0] = wsSysSize;
     return ge::GRAPH_SUCCESS;
 }
 
 // tiling 分发入口
 static ge::graphStatus ApplyAdagradDTilingFunc(gert::TilingContext* context)
 {
     // 1、获取平台运行信息
     uint64_t ubSize;
     int64_t coreNum;
     uint32_t wsSysSize;
     OP_CHECK_IF(GetPlatformInfo(context, coreNum, wsSysSize) != ge::GRAPH_SUCCESS,
                 OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
     OP_CHECK_IF(GetUBSize(context, ubSize) != ge::GRAPH_SUCCESS,
                 OP_LOGE(context, "GetUBSize error"), return ge::GRAPH_FAILED);
 
     // 2、获取shape、属性信息
     ge::DataType dataType;
     int64_t inputLength;
     int32_t inputBytes;
     int32_t dataNumber;
     OP_CHECK_IF(GetShapeAttrsInfo(context, dataType, inputLength, inputBytes, dataNumber) != ge::GRAPH_SUCCESS,
                 OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

     bool updateSlots = *context->GetAttrs()->GetBool(0);
 
     // 3、获取WorkspaceSize信息
     OP_CHECK_IF(GetWorkspaceSize(context, wsSysSize) != ge::GRAPH_SUCCESS,
                 OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);
 
     // 4、设置tiling信息
     ApplyAdagradDTilingData* tiling = context->GetTilingData<ApplyAdagradDTilingData>();
     OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
     OP_CHECK_IF(memset_s(tiling, sizeof(ApplyAdagradDTilingData), 0, sizeof(ApplyAdagradDTilingData)) != EOK,
                 OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
 
     uint32_t tileBlockNum = (ubSize / ALIGN_SIZE) / dataNumber;
     uint32_t tileDataNum = tileBlockNum * (ALIGN_SIZE / inputBytes);
 
     uint32_t inputLengthAlign = (((inputLength + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE);
     coreNum = (coreNum < inputLengthAlign / ALIGN_SIZE) ? coreNum : inputLengthAlign / ALIGN_SIZE;
     coreNum = (coreNum >= 1) ? coreNum : 1;
 
     uint32_t everyCoreInputBlockNum = inputLengthAlign / ALIGN_SIZE / coreNum;
     uint32_t tailBlockNum = inputLengthAlign / ALIGN_SIZE % coreNum;
 
     uint32_t smallCoreDataNum = everyCoreInputBlockNum * ALIGN_SIZE / inputBytes;
     uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
     uint32_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : (smallTileNum + 1);
     uint32_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
     smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;
 
     everyCoreInputBlockNum += 1;
     uint32_t bigCoreDataNum = everyCoreInputBlockNum * ALIGN_SIZE / inputBytes;
     uint32_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
     uint32_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : (bigTileNum + 1);
     uint32_t bigTailDataNum = bigCoreDataNum - (tileDataNum * bigTileNum);
     bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;
 
     tiling->smallCoreDataNum = smallCoreDataNum;
     tiling->bigCoreDataNum = bigCoreDataNum;
     tiling->finalBigTileNum = finalBigTileNum;
     tiling->finalSmallTileNum = finalSmallTileNum;
     tiling->smallTailDataNum = smallTailDataNum;
     tiling->bigTailDataNum = bigTailDataNum;
     tiling->tileDataNum = tileDataNum;
     tiling->tailBlockNum = tailBlockNum;
     tiling->updateSlots = updateSlots;
     tiling->overlapDataNum = inputLengthAlign / inputBytes - inputLength / inputBytes; 
 
     context->SetBlockDim(static_cast<uint32_t>(coreNum));
     uint64_t tilingKey = GET_TPL_TILING_KEY(0);
     context->SetTilingKey(tilingKey);
     return ge::GRAPH_SUCCESS;
 }
 
 static ge::graphStatus TilingParseForApplyAdagradD([[maybe_unused]] gert::TilingParseContext* context)
 {
     OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
     return ge::GRAPH_SUCCESS;
 }
 
 // tiling注册入口
 IMPL_OP_OPTILING(ApplyAdagradD)
     .Tiling(ApplyAdagradDTilingFunc)
     .TilingParse<ApplyAdagradDCompileInfo>(TilingParseForApplyAdagradD);
 } // namespace optiling
 