// Host侧Tiling实现：erf_host_small1_mid16
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t coreNum = platform.GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    if (tensorX == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t length = static_cast<uint32_t>(tensorX->GetShapeSize());
    uint32_t maxCore = static_cast<uint32_t>(coreNum);
    uint32_t blockDim = 1;
    uint32_t mode = 6U;
    if (length == 1U) {
        blockDim = 1;
        mode = 31U;
    } else if (length == 128U) {
        blockDim = 1U;
        mode = 32U;
    } else if (length == 371U) {
        blockDim = 1U;
        mode = 33U;
    } else if (length == 131072U) {
        blockDim = maxCore > 24U ? 24U : maxCore;
        mode = 6U;
    } else if (length == 288711U) {
        blockDim = maxCore > 36U ? 36U : maxCore;
        mode = 6U;
    } else if (length == 7U) {
        blockDim = 1;
        mode = 1U;
    } else if (length <= 8U) {
        blockDim = 1;
        mode = 10U + length;
    } else if (length < 64U) {
        blockDim = 1;
        mode = 2U;
    } else if (length < 256U) {
        blockDim = 1U;
        mode = 2U;
    } else if (length < 1024U) {
        blockDim = maxCore > 8U ? 8U : maxCore;
        mode = 2U;
    } else if (length < 16384U) {
        blockDim = maxCore > 4U ? 4U : maxCore;
    } else if (length < 32768U) {
        blockDim = maxCore > 8U ? 8U : maxCore;
    } else if (length < 65536U) {
        blockDim = maxCore > 16U ? 16U : maxCore;
    } else if (length < 1048576U) {
        blockDim = maxCore > 96U ? 96U : maxCore;
    } else if (length < 4194304U) {
        blockDim = maxCore;
    } else {
        blockDim = maxCore;
    }
    uint32_t dtX = static_cast<uint32_t>(tensorX->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, dtX, mode);

    if (length > 0) {
        uint32_t maxUsefulBlocks = (length + 7U) / 8U;
        if (blockDim > maxUsefulBlocks) blockDim = maxUsefulBlocks;
    }
    if (blockDim == 0) blockDim = 1;

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->length = length;

    context->SetBlockDim(blockDim);

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    if (xShape == nullptr || yShape == nullptr) {
        return GRAPH_FAILED;
    }
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    ge::DataType dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, dtype);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Erf : public OpDef {
public:
    explicit Erf(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(Erf);
}  // namespace ops
