/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * 
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License");
 * Please refer to the License for details. You may not use this file except in
 * compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR
 * A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the
 * License.
 */

#include "assign_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

constexpr uint32_t BLOCK_ALIGN_BYTES = 128;
constexpr uint32_t UB_BLOCK_BYTES = 48 * 1024;  // 与 kernel 端 constexpr 必须一致

static inline uint32_t RoundUp(uint32_t v, uint32_t a) { return (a == 0) ? v : (v + a - 1) / a * a; }

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AssignTilingData tiling;
    auto shape = context->GetInputShape(0)->GetStorageShape();
    auto dtype = context->GetInputDesc(0)->GetDataType();
    uint32_t elemSize = (dtype == ge::DT_BOOL) ? 1 : ge::GetSizeByDataType(dtype);
    if (elemSize == 0) { elemSize = 1; }

    uint64_t totalElems = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        totalElems *= static_cast<uint64_t>(shape.GetDim(i));
    }
    uint64_t totalBytes = totalElems * static_cast<uint64_t>(elemSize);

    auto ascendc = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendc.GetCoreNumAiv();
    if (aivNum == 0) { aivNum = 1; }

    uint32_t bytesPerCore;
    uint32_t usedCores;
    uint32_t tailBytes;
    if (totalBytes <= BLOCK_ALIGN_BYTES) {
        usedCores = 1;
        bytesPerCore = static_cast<uint32_t>(totalBytes);
        tailBytes = static_cast<uint32_t>(totalBytes);
    } else {
        uint64_t bpc = (totalBytes + aivNum - 1) / aivNum;
        bytesPerCore = RoundUp(static_cast<uint32_t>(bpc), BLOCK_ALIGN_BYTES);
	usedCores = (bytesPerCore == 0) ? 1 : static_cast<uint32_t>((totalBytes + bytesPerCore - 1) / bytesPerCore);
        if (usedCores > aivNum) { usedCores = aivNum; }
        uint64_t consumed = static_cast<uint64_t>(bytesPerCore) * (usedCores - 1);
        tailBytes = static_cast<uint32_t>(totalBytes - consumed);
    }

    tiling.set_bytesPerCore(bytesPerCore);
    tiling.set_tailBytes(tailBytes);

    context->SetBlockDim(usedCores);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {
class Assign : public OpDef {
public:
    explicit Assign(const char* name) : OpDef(name)
    {

        this->Input("input")
            .ParamType(REQUIRED)
	    // 输入1的参数的数据类型
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL})
            // 输入参数1的不同数据格式
		    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
	

		// 输入参数2
		this->Input("other")
            .ParamType(REQUIRED)
	    // 数据类型
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL})
	    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
	
		this->Attr("use_locking").AttrType(OPTIONAL).Bool(false);
	// 算子输入输出属性结束位置
	
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Assign);
} // namespace ops
