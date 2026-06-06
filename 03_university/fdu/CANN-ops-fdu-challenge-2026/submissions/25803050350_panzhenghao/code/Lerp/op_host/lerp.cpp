#include <algorithm>
#include <cstdint>

#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/lerp_tiling.h"
#include "../op_kernel/tiling_key_lerp.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platform.GetCoreNumAiv();

    ge::DataType inputDtype = context->GetInputDesc(0)->GetDataType();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(inputDtype, typeLength);

    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t inputBytes = inputNum * typeLength;

    constexpr uint32_t BLOCK_SIZE = 32;
    uint32_t alignedBytes = ((inputBytes + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    uint32_t totalBlockNum = alignedBytes / BLOCK_SIZE;

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const float *attrWeight = attrs->GetFloat(0);
    float weight = (attrWeight == nullptr) ? 0.0f : *attrWeight;
    uint32_t LERP_MODE = (weight == 0.0f) ? 0 : ((weight == 1.0f) ? 1 : 2);

    constexpr uint32_t MIN_BLOCKS_PER_CORE = 32;
    if (totalBlockNum == 0) {
        coreNum = 1;
    } else if ((LERP_MODE == 2 && totalBlockNum > 4 && totalBlockNum <= 8) ||
               ((LERP_MODE == 0 || LERP_MODE == 1) &&
               ((inputDtype == ge::DT_FLOAT && totalBlockNum <= 6) ||
                (inputDtype != ge::DT_FLOAT && totalBlockNum <= 4) ||
                (inputDtype == ge::DT_FLOAT16 && LERP_MODE == 0 && totalBlockNum == 5 && inputNum == 74)))) {
        coreNum = std::min(coreNum, totalBlockNum);
        coreNum = std::max(coreNum, static_cast<uint32_t>(1));
    } else {
        uint32_t targetCoreNum = (totalBlockNum + MIN_BLOCKS_PER_CORE - 1) / MIN_BLOCKS_PER_CORE;
        targetCoreNum = std::max(targetCoreNum, static_cast<uint32_t>(1));
        coreNum = std::min(coreNum, targetCoreNum);
        coreNum = std::max(coreNum, static_cast<uint32_t>(1));
    }
    context->SetBlockDim(coreNum);

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    constexpr uint32_t BUFFER_NUM = 1;
    constexpr uint32_t QUEUE_TENSOR_NUM = 3;
    uint32_t ubBytesPerElement = BUFFER_NUM * QUEUE_TENSOR_NUM * typeLength;
    uint32_t elementsPerBlock = BLOCK_SIZE / typeLength;
    uint32_t maxTileDataNum = static_cast<uint32_t>(ubSize / ubBytesPerElement);
    uint32_t tileDataNum = (maxTileDataNum / elementsPerBlock) * elementsPerBlock;
    tileDataNum = std::max(tileDataNum, elementsPerBlock);
    uint32_t tileBlockNum = tileDataNum / elementsPerBlock;

    uint32_t smallCoreBlockNum = (coreNum == 0) ? 0 : (totalBlockNum / coreNum);
    uint32_t tailBlockNum = (coreNum == 0) ? 0 : (totalBlockNum % coreNum);

    uint32_t smallCoreDataNum = smallCoreBlockNum * BLOCK_SIZE / typeLength;
    uint32_t smallFullTileNum = smallCoreBlockNum / tileBlockNum;
    uint32_t finalSmallTileNum = (smallCoreBlockNum == 0) ? 0 :
        (((smallCoreBlockNum % tileBlockNum) == 0) ? smallFullTileNum : smallFullTileNum + 1);
    uint32_t smallTailDataNum = smallCoreDataNum - tileDataNum * smallFullTileNum;
    smallTailDataNum = (smallTailDataNum == 0) ? tileDataNum : smallTailDataNum;

    uint32_t bigCoreBlockNum = smallCoreBlockNum + 1;
    uint32_t bigCoreDataNum = bigCoreBlockNum * BLOCK_SIZE / typeLength;
    uint32_t bigFullTileNum = bigCoreBlockNum / tileBlockNum;
    uint32_t finalBigTileNum = (bigCoreBlockNum == 0) ? 0 :
        (((bigCoreBlockNum % tileBlockNum) == 0) ? bigFullTileNum : bigFullTileNum + 1);
    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigFullTileNum;
    bigTailDataNum = (bigTailDataNum == 0) ? tileDataNum : bigTailDataNum;

    LerpTilingData *tiling = context->GetTilingData<LerpTilingData>();
    tiling->totalLength = inputNum;
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->tailBlockNum = tailBlockNum;
    tiling->weight = weight;

    uint32_t DT_START = static_cast<uint32_t>(context->GetInputDesc(0)->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_START, LERP_MODE);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Lerp : public OpDef {
public:
    explicit Lerp(const char *name) : OpDef(name)
    {
        this->Input("start")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("end")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("weight").AttrType(REQUIRED).Float();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Lerp);
}  // namespace ops
