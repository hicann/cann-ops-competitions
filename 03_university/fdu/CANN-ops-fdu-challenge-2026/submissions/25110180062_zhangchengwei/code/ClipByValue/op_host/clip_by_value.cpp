// Host tiling and operator registration.
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/clip_by_value_tiling.h"
#include "../op_kernel/tiling_key_clip_by_value.h"

namespace {
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t UB_BUFFER_PARTS = 5;
constexpr uint32_t TILE_BLOCK_ALIGN = 8;

static uint32_t GetTypeLength(ge::DataType dtype) {
    return dtype == ge::DT_FLOAT16 ? 2U : 4U;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (coreNum == 0) {
        coreNum = 1;
    }

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    uint32_t inputNum = static_cast<uint32_t>(tensorX->GetShapeSize());
    ge::DataType dtype = context->GetRequiredInputDesc(0)->GetDataType();
    uint32_t typeLength = GetTypeLength(dtype);
    uint32_t inputBytes = inputNum * typeLength;
    uint32_t alignedBytes = ((inputBytes + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    uint32_t totalBlockNum = alignedBytes / BLOCK_SIZE;

    uint32_t blockDim = 1;
    if (totalBlockNum > 0) {
        blockDim = totalBlockNum < coreNum ? totalBlockNum : coreNum;
    }
    context->SetBlockDim(blockDim);

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t tileBlockNum = static_cast<uint32_t>((ubSize / BLOCK_SIZE) / UB_BUFFER_PARTS);
    tileBlockNum = (tileBlockNum / TILE_BLOCK_ALIGN) * TILE_BLOCK_ALIGN;
    if (tileBlockNum == 0) {
        tileBlockNum = TILE_BLOCK_ALIGN;
    }

    uint32_t smallCoreBlockNum = blockDim == 0 ? 0 : totalBlockNum / blockDim;
    uint32_t tailBlockNum = blockDim == 0 ? 0 : totalBlockNum % blockDim;
    uint32_t bigCoreBlockNum = smallCoreBlockNum + 1;

    uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / typeLength;
    uint32_t smallCoreDataNum = (smallCoreBlockNum * BLOCK_SIZE) / typeLength;
    uint32_t bigCoreDataNum = (bigCoreBlockNum * BLOCK_SIZE) / typeLength;

    uint32_t smallTileNum = smallCoreBlockNum / tileBlockNum;
    uint32_t finalSmallTileNum = (smallCoreBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    uint32_t smallTailDataNum = smallCoreDataNum - tileDataNum * smallTileNum;
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    uint32_t bigTileNum = bigCoreBlockNum / tileBlockNum;
    uint32_t finalBigTileNum = (bigCoreBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const float *attrMin = attrs->GetFloat(0);
    const float *attrMax = attrs->GetFloat(1);

    ClipByValueTilingData *tiling = context->GetTilingData<ClipByValueTilingData>();
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->tailBlockNum = tailBlockNum;
    tiling->min = *attrMin;
    tiling->max = *attrMax;

    uint32_t dtypeX = static_cast<uint32_t>(dtype);
    ge::DataType outDtype = context->GetOutputDesc(0)->GetDataType();
    uint32_t dtypeY = static_cast<uint32_t>(outDtype);
    ASCENDC_TPL_SEL_PARAM(context, dtypeX, dtypeY);
    context->GetRawTilingData()->SetDataSize(sizeof(ClipByValueTilingData));
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(platform.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class ClipByValue : public OpDef {
public:
    explicit ClipByValue(const char *name) : OpDef(name) {
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
