#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/clip_by_value_tiling.h"
#include "../op_kernel/tiling_key_clip_by_value.h"

#include <cstddef>
#include <cstdint>

namespace optiling {
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t LOCAL_TENSOR_NUM = 2;
constexpr uint32_t TARGET_ELEMENTS_PER_CORE = 4096;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t coreNum = platform.GetCoreNumAiv();

    auto inputShape = context->GetInputShape(0);
    uint64_t shapeSize = inputShape->GetOriginShape().GetShapeSize();
    uint32_t length = static_cast<uint32_t>(shapeSize);

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    ge::DataType dtypeX = tensorX->GetDataType();
    uint32_t dtypeSize = static_cast<uint32_t>(ge::GetSizeByDataType(dtypeX));
    uint32_t alignNum = 32 / dtypeSize;
    uint32_t DT_X = static_cast<uint32_t>(dtypeX);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const float *attrMin = attrs->GetFloat(0);
    const float *attrMax = attrs->GetFloat(1);

    uint32_t maxCoreNum = static_cast<uint32_t>(coreNum > 0 ? coreNum : 1);
    uint32_t alignedLength = length / alignNum * alignNum;
    uint32_t alignedBlockNum = alignedLength / alignNum;
    uint32_t blockDim = 1;
    if (alignedBlockNum > 0 && length > TARGET_ELEMENTS_PER_CORE) {
        blockDim = (length + TARGET_ELEMENTS_PER_CORE - 1) / TARGET_ELEMENTS_PER_CORE;
        if (blockDim > maxCoreNum) {
            blockDim = maxCoreNum;
        }
        if (blockDim > alignedBlockNum) {
            blockDim = alignedBlockNum;
        }
    }
    context->SetBlockDim(blockDim);

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t maxTileLength = static_cast<uint32_t>(ubSize / (BUFFER_NUM * LOCAL_TENSOR_NUM * dtypeSize));
    maxTileLength = maxTileLength / alignNum * alignNum;
    if (maxTileLength == 0) {
        maxTileLength = alignNum;
    }

    uint32_t baseBlockNum = alignedBlockNum / blockDim;
    uint32_t tailBlockNum = alignedBlockNum % blockDim;
    uint32_t maxBlockLength = (baseBlockNum + (tailBlockNum > 0 ? 1 : 0)) * alignNum;
    maxBlockLength += length - alignedLength;
    uint32_t tileLength = maxBlockLength < maxTileLength ? maxBlockLength : maxTileLength;
    if (tileLength == 0) {
        tileLength = alignNum;
    }
    tileLength = (tileLength + alignNum - 1) / alignNum * alignNum;
    uint32_t tileNum = (maxBlockLength + tileLength - 1) / tileLength;
    if (tileNum == 0) {
        tileNum = 1;
    }

    ClipByValueTilingData *tiling = context->GetTilingData<ClipByValueTilingData>();
    tiling->length = length;
    tiling->tileLength = tileLength;
    tiling->tileNum = tileNum;
    tiling->minValue = attrMin == nullptr ? 0.0f : *attrMin;
    tiling->maxValue = attrMax == nullptr ? 0.0f : *attrMax;

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
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
