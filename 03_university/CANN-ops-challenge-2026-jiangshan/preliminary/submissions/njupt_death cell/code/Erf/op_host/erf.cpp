#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
constexpr uint32_t ALIGN_NUM = 32 / sizeof(float);
constexpr uint32_t SINGLE_CORE_THRESHOLD = 256;
constexpr uint32_t SMALL_BLOCK_THRESHOLD = 512;
constexpr uint32_t SMALL_BLOCK_LENGTH = 512;
constexpr uint32_t SMALL_BLOCK_SHIFT = 9;
constexpr uint32_t SMALL_BLOCK_MASK = SMALL_BLOCK_LENGTH - 1;
constexpr uint32_t LARGE_BLOCK_LENGTH = 2048;
constexpr uint32_t LARGE_BLOCK_MASK = 32 / sizeof(float) - 1;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    ge::DataType dtypeX = tensorX->GetDataType();
    uint32_t lengthX = static_cast<uint32_t>(tensorX->GetShapeSize());

    uint32_t dtX = static_cast<uint32_t>(dtypeX);
    ASCENDC_TPL_SEL_PARAM(context, dtX);

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    tiling->length = lengthX;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    if (lengthX <= SINGLE_CORE_THRESHOLD) {
        tiling->perBlock = lengthX;
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (coreNum == 0) {
        coreNum = 1;
    }

    uint32_t blockDim = (lengthX <= SMALL_BLOCK_THRESHOLD)
        ? ((lengthX + SMALL_BLOCK_LENGTH - 1) >> SMALL_BLOCK_SHIFT)
        : ((lengthX + LARGE_BLOCK_LENGTH - 1) / LARGE_BLOCK_LENGTH);
    if (blockDim > coreNum) {
        blockDim = coreNum;
    }

    if (blockDim == 1) {
        tiling->perBlock = lengthX;
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    uint32_t rawPerBlock = (lengthX + blockDim - 1) / blockDim;
    tiling->perBlock = (lengthX <= SMALL_BLOCK_THRESHOLD)
        ? ((rawPerBlock + SMALL_BLOCK_MASK) & ~SMALL_BLOCK_MASK)
        : ((rawPerBlock + LARGE_BLOCK_MASK) & ~LARGE_BLOCK_MASK);
    blockDim = (lengthX + tiling->perBlock - 1) / tiling->perBlock;
    context->SetBlockDim(blockDim);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
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
