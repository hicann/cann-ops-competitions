
#include "gelu_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"
#include <math.h>

const uint32_t BLOCK_SIZE = 32;    // 32B对齐单位
const uint32_t BUFFER_NUM = 2;     // 双缓冲块数

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    GeluV2TilingData tilingData;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t coreNum = platform.GetCoreNum();
    uint32_t inputTypeLen = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), inputTypeLen);

    //Input
    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();

    int approximate = *context->GetAttrs()->GetInt(0);
    //tile
    uint32_t align32Length = ((inputNum * inputTypeLen + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    uint32_t inputNum32 = align32Length / inputTypeLen;
    uint32_t inputBlockNum = align32Length / BLOCK_SIZE;

    if (approximate == 0) 
        context->SetTilingKey(1);
    else
        context->SetTilingKey(2);

    // printf("app: %u\n", approximate);
    uint32_t ubDataNumber;
    if (inputTypeLen == 2) {
        if (approximate) ubDataNumber = 8;
        else ubDataNumber = 14;
    }
    else {
        if (approximate) ubDataNumber = 4;
        else ubDataNumber = 8;
    }
    uint32_t tileBlockNum = ubSize / BLOCK_SIZE / ubDataNumber; 
    uint32_t tileDataNum = tileBlockNum * BLOCK_SIZE / inputTypeLen;

    uint32_t tileNum = inputBlockNum / tileBlockNum;
    uint32_t finaltileNum = (inputBlockNum % tileBlockNum == 0) ? tileNum : tileNum + 1;
    uint32_t tailDataNum = inputNum32 - (tileDataNum * tileNum);
    
    // 填充Tiling数据
    tilingData.set_inputNum(inputNum32);
    tilingData.set_tileDataNum(tileDataNum);
    tilingData.set_tileNum(finaltileNum);
    tilingData.set_tailDataNum(tailDataNum);
    tilingData.set_inputTypeLen(inputTypeLen);
    tilingData.set_sqrt2(std::sqrt(2.0));

    // printf("inputNum: %u, tileDataNum: %u, tileNum: %u, tailDataNum: %u\n", inputNum32, tileDataNum, finaltileNum, tailDataNum);
    //other

    
    context->SetBlockDim(coreNum);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class GeluV2 : public OpDef {
public:
    explicit GeluV2(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("approximate").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(GeluV2);
}
