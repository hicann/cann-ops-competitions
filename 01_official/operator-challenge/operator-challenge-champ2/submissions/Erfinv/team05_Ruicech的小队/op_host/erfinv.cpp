// Host-side operator definition and tiling for erfinv.
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erfinv_tiling.h"
#include "../op_kernel/tiling_key_erfinv.h"

namespace {

constexpr uint32_t FLOAT_SIZE = sizeof(float);
constexpr uint32_t UB_TENSOR_COUNT = 8;  // input/output double buffer + 4 temp tensors.
constexpr uint64_t UB_RESERVE_BYTES = 1024;

uint32_t AlignDown(uint32_t value, uint32_t align)
{
    return value / align * align;
}

uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1) / divisor;
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

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    ge::DataType dtypeX = tensorX->GetDataType();
    uint32_t length = static_cast<uint32_t>(tensorX->GetShapeSize());

    uint32_t blockNum = (length == 0) ? 1 : ((length < coreNum) ? length : coreNum);
    uint32_t blockLength = (length == 0) ? 0 : CeilDiv(length, blockNum);

    uint32_t ubLength = ERFINV_UB_ALIGN;
    if (ubSize > UB_RESERVE_BYTES) {
        uint32_t maxByUb = static_cast<uint32_t>((ubSize - UB_RESERVE_BYTES) / (FLOAT_SIZE * UB_TENSOR_COUNT));
        ubLength = AlignDown(maxByUb, ERFINV_UB_ALIGN);
        if (ubLength == 0) {
            ubLength = ERFINV_UB_ALIGN;
        }
        if (ubLength > ERFINV_UB_LENGTH_LIMIT) {
            ubLength = ERFINV_UB_LENGTH_LIMIT;
        }
    }

    ErfinvTilingData *tiling = context->GetTilingData<ErfinvTilingData>();
    tiling->length = length;
    tiling->blockLength = blockLength;
    tiling->ubLength = ubLength;

    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dtypeX));
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
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class Erfinv : public OpDef {
public:
    explicit Erfinv(const char *name) : OpDef(name)
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

OP_ADD(Erfinv);

}  // namespace ops
