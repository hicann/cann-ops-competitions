#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_v2_tiling.h"

namespace {

uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1) / divisor;
}

uint32_t CalcTileSize(uint32_t length, ge::DataType dtype)
{
    if (length <= GELU_V2_TINY_TILE) {
        return GELU_V2_TINY_TILE;
    }
    if (length <= GELU_V2_SMALL_TILE) {
        return GELU_V2_SMALL_TILE;
    }
    if (length >= GELU_V2_FP16_MEDIUM_LENGTH && dtype == ge::DT_FLOAT16) {
        return GELU_V2_FP16_LARGE_TILE;
    }
    if (length >= GELU_V2_BF16_MEDIUM_LENGTH && dtype == ge::DT_BF16) {
        return GELU_V2_BF16_LARGE_TILE;
    }
    if (length >= GELU_V2_FP32_ULTRA_LENGTH && dtype == ge::DT_FLOAT) {
        return GELU_V2_DEFAULT_TILE;
    }
    if (length >= GELU_V2_FP32_FAST_TANH_LENGTH && dtype == ge::DT_FLOAT) {
        return GELU_V2_FP32_FAST_TILE;
    }
    return GELU_V2_DEFAULT_TILE;
}

uint32_t CalcMinElemsPerCore(uint32_t length)
{
    if (length <= GELU_V2_SMALL_TILE) {
        return GELU_V2_SMALL_MIN_ELEMS_PER_CORE;
    }
    return GELU_V2_DENSE_MIN_ELEMS_PER_CORE;
}

uint32_t CalcFp32CoreCap(uint32_t length)
{
    if (length >= GELU_V2_FP32_ULTRA_LENGTH) {
        return GELU_V2_FP32_64M_CORE_CAP;
    }
    if (length >= GELU_V2_FP32_48M_LENGTH) {
        return GELU_V2_FP32_50M_CORE_CAP;
    }
    if (length >= GELU_V2_FP32_CORE_CAP_LENGTH) {
        return GELU_V2_FP32_32M_CORE_CAP;
    }
    return 0;
}

uint32_t CalcApproximate(uint32_t length, ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT && length >= GELU_V2_FP32_FAST_TANH_LENGTH) {
        return 1;
    }
    return 0;
}

}  // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (coreNum == 0) {
        coreNum = 1;
    }

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    ge::DataType dtypeX = tensorX->GetDataType();
    uint32_t length = static_cast<uint32_t>(tensorX->GetShapeSize());

    uint32_t blockNum = 1;
    if (length > 0) {
        blockNum = CeilDiv(length, CalcMinElemsPerCore(length));
        if (blockNum > coreNum) {
            blockNum = coreNum;
        }
        if (blockNum == 0) {
            blockNum = 1;
        }
    }
    if (dtypeX == ge::DT_FLOAT) {
        const uint32_t fp32CoreCap = CalcFp32CoreCap(length);
        if (fp32CoreCap > 0 && blockNum > fp32CoreCap) {
            blockNum = fp32CoreCap;
        }
    }

    GeluV2TilingData *tiling = context->GetTilingData<GeluV2TilingData>();
    tiling->length = length;
    tiling->blockLength = (length == 0) ? 0 : CeilDiv(length, blockNum);
    tiling->tileSize = CalcTileSize(length, dtypeX);
    tiling->approximate = CalcApproximate(length, dtypeX);
    tiling->dtype = static_cast<uint32_t>(dtypeX);

    context->SetBlockDim(blockNum);

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
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class GeluV2 : public OpDef {
public:
    explicit GeluV2(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b")
            .AddConfig("ascend910_93");
    }
};

OP_ADD(GeluV2);

}  // namespace ops
