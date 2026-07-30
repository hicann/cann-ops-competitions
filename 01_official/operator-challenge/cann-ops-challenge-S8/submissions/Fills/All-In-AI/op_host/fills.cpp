#include "fills_tiling.h"

#include "register/op_def_registry.h"
#include <algorithm>

namespace optiling {
constexpr uint32_t MAX_AIV_NUM = 40; // Ascend910B
constexpr uint32_t MID_AIV_NUM = 32;
constexpr uint32_t SMALL_AIV_NUM = 16;
constexpr uint32_t ALIGN_BYTES = 512;
constexpr uint32_t TILE_LENGTH_DEFAULT = 16384;
constexpr uint32_t TILE_LENGTH_FP16 = 24576;
constexpr uint32_t TILE_LENGTH_BF16 = 24576;
constexpr uint32_t TILE_LENGTH_FP32 = 16384;
constexpr uint32_t TILE_LENGTH_INT32 = 16384;
constexpr uint32_t TILE_LENGTH_INT16 = 32768;
constexpr uint32_t TILE_LENGTH_INT8 = 49152;

static uint32_t SelectTileLength(ge::DataType inputType)
{
    if (inputType == ge::DT_FLOAT16) {
        return TILE_LENGTH_FP16;
    }
    if (inputType == ge::DT_BF16) {
        return TILE_LENGTH_BF16;
    }
    if (inputType == ge::DT_FLOAT) {
        return TILE_LENGTH_FP32;
    }
    if (inputType == ge::DT_INT32) {
        return TILE_LENGTH_INT32;
    }
    if (inputType == ge::DT_INT16) {
        return TILE_LENGTH_INT16;
    }
    if (inputType == ge::DT_INT8 || inputType == ge::DT_UINT8) {
        return TILE_LENGTH_INT8;
    }
    return TILE_LENGTH_DEFAULT;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingData tiling;
    const uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    auto inputType = context->GetInputTensor(0)->GetDataType();
    uint32_t dataTypeBytes = 4;
    if (inputType == ge::DT_FLOAT16 || inputType == ge::DT_BF16 || inputType == ge::DT_INT16) {
        dataTypeBytes = 2;
    } else if (inputType == ge::DT_UINT8 || inputType == ge::DT_INT8) {
        dataTypeBytes = 1;
    }

    const uint32_t blockElem = std::max(1U, ALIGN_BYTES / dataTypeBytes);
    const uint32_t blockNum = (totalLength + blockElem - 1U) / blockElem;
    uint32_t maxAiv = MAX_AIV_NUM;
    if (totalLength < 32768U) {
        maxAiv = SMALL_AIV_NUM;
    } else if (totalLength < 131072U) {
        maxAiv = MID_AIV_NUM;
    }
    const uint32_t aivNum = std::min(maxAiv, std::max(1U, blockNum));
    const uint32_t smallCoreLength = (blockNum / aivNum) * blockElem;
    const uint32_t formerNum = blockNum % aivNum;
    const uint32_t tileLength = std::max(SelectTileLength(inputType), blockElem);

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const float *value = attrs->GetAttrPointer<float>(0);

    context->SetBlockDim(aivNum);
    tiling.set_totalLength(totalLength);
    tiling.set_tileLength(tileLength);
    tiling.set_smallCoreLength(smallCoreLength);
    tiling.set_incCoreLength(blockElem);
    tiling.set_formerNum(formerNum);
    tiling.set_value(*value);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

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
    const auto inputType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputType);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Fills : public OpDef {
public:
    explicit Fills(const char *name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8,
                       ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND});

        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8,
                       ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND});

        this->Attr("value").AttrType(REQUIRED).Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(Fills);
} // namespace ops
