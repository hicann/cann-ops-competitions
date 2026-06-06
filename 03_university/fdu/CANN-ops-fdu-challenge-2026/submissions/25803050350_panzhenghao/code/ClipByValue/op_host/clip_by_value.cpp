#include <algorithm>

#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/clip_by_value_tiling.h"
#include "../op_kernel/tiling_key_clip_by_value.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platform.GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = platform.GetCoreNum();
    }

    constexpr uint32_t blockSize = 32;
    constexpr uint32_t bufferNum = 2;

    ge::DataType dtypeX = context->GetInputDesc(0)->GetDataType();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(dtypeX, typeLength);

    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t inputBytes = inputNum * typeLength;

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const float *attrMin = attrs->GetFloat(0);
    const float *attrMax = attrs->GetFloat(1);

    uint32_t DT_X = static_cast<uint32_t>(dtypeX);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    if (inputNum == 0) {
        ClipByValueTilingData *tiling = context->GetTilingData<ClipByValueTilingData>();
        tiling->smallCoreDataNum = 0;
        tiling->bigCoreDataNum = 0;
        tiling->finalBigTileNum = 0;
        tiling->finalSmallTileNum = 0;
        tiling->tileDataNum = blockSize / typeLength;
        tiling->smallTailDataNum = 0;
        tiling->bigTailDataNum = 0;
        tiling->tailBlockNum = 0;
        tiling->minValue = attrMin == nullptr ? 0.0F : *attrMin;
        tiling->maxValue = attrMax == nullptr ? 0.0F : *attrMax;

        context->SetBlockDim(1);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    uint32_t alignedBytes = (inputBytes + blockSize - 1) / blockSize * blockSize;
    uint32_t totalBlockNum = alignedBytes / blockSize;

    coreNum = std::min(coreNum, totalBlockNum);
    if (dtypeX == ge::DT_FLOAT && totalBlockNum >= 64) {
        constexpr uint32_t minBlockPerCore = 3;
        uint32_t usefulCoreNum = (totalBlockNum + minBlockPerCore - 1) / minBlockPerCore;
        coreNum = std::min(coreNum, usefulCoreNum);
    }
    coreNum = std::max(coreNum, static_cast<uint32_t>(1));
    uint32_t everyCoreBlockNum = totalBlockNum / coreNum;
    uint32_t tailBlockNum = totalBlockNum % coreNum;

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t ubBufferNum = 2 * bufferNum;
    if (dtypeX == ge::DT_INT32) {
        ubBufferNum = totalBlockNum < 64 ? (2 * bufferNum + 1) : (3 * bufferNum);
    }
    uint32_t tileBlockNum = static_cast<uint32_t>(ubSize / blockSize / ubBufferNum);
    tileBlockNum = std::max(tileBlockNum, static_cast<uint32_t>(1));
    uint32_t tileDataNum = tileBlockNum * blockSize / typeLength;

    uint32_t smallCoreDataNum = everyCoreBlockNum * blockSize / typeLength;
    uint32_t smallTileNum = everyCoreBlockNum / tileBlockNum;
    uint32_t finalSmallTileNum = everyCoreBlockNum % tileBlockNum == 0 ? smallTileNum : smallTileNum + 1;
    uint32_t smallTailDataNum = smallCoreDataNum - tileDataNum * smallTileNum;
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    uint32_t bigCoreBlockNum = everyCoreBlockNum + 1;
    uint32_t bigCoreDataNum = bigCoreBlockNum * blockSize / typeLength;
    uint32_t bigTileNum = bigCoreBlockNum / tileBlockNum;
    uint32_t finalBigTileNum = bigCoreBlockNum % tileBlockNum == 0 ? bigTileNum : bigTileNum + 1;
    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    float minValue = attrMin == nullptr ? 0.0F : *attrMin;
    float maxValue = attrMax == nullptr ? 0.0F : *attrMax;

    ClipByValueTilingData *tiling = context->GetTilingData<ClipByValueTilingData>();
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->tailBlockNum = tailBlockNum;
    tiling->minValue = minValue;
    tiling->maxValue = maxValue;

    context->SetBlockDim(coreNum);
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
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class ClipByValue : public OpDef {
public:
    explicit ClipByValue(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("min").AttrType(REQUIRED).Float();
        this->Attr("max").AttrType(REQUIRED).Float();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(ClipByValue);
}  // namespace ops
