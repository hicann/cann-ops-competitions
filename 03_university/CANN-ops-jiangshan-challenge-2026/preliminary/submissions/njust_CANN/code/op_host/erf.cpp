// Host tiling implementation for erf.
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace {
constexpr uint32_t TILE_LENGTH = 8192;
constexpr uint32_t ALIGN_NUM = 8;
constexpr uint32_t SINGLE_CORE_LENGTH = 32768;
constexpr uint32_t TINY_LENGTH = 2048;
constexpr uint32_t MID_MULTI_CORE_LOWER = 12288;
constexpr uint32_t MID_MULTI_CORE_UPPER = 24576;
constexpr uint32_t TINY_CORE_CAP = 8;

uint32_t CeilDiv(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

uint32_t AlignUp(uint32_t a, uint32_t b) {
    return CeilDiv(a, b) * b;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t numCores = platform.GetCoreNumAiv();

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    if (tensorX == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType dtypeX = tensorX->GetDataType();
    uint32_t length = static_cast<uint32_t>(tensorX->GetShapeSize());

    uint32_t coreLimit = numCores > 0 ? static_cast<uint32_t>(numCores) : 1;
    uint32_t usedCoreNum = 1;
    if (length > 0) {
        if (length <= TINY_LENGTH) {
            uint32_t units = CeilDiv(length, 16);
            usedCoreNum = units < coreLimit ? units : coreLimit;
            usedCoreNum = usedCoreNum < TINY_CORE_CAP ? usedCoreNum : TINY_CORE_CAP;
        } else if (length > MID_MULTI_CORE_LOWER && length <= MID_MULTI_CORE_UPPER) {
            usedCoreNum = coreLimit < 2U ? coreLimit : 2U;
        } else {
            usedCoreNum = length <= SINGLE_CORE_LENGTH ? 1 : CeilDiv(length, TILE_LENGTH);
            usedCoreNum = usedCoreNum > coreLimit ? coreLimit : usedCoreNum;
        }
        usedCoreNum = usedCoreNum == 0 ? 1 : usedCoreNum;
    }

    uint32_t blockLength = length == 0 ? 0 : AlignUp(CeilDiv(length, usedCoreNum), ALIGN_NUM);
    uint32_t tileLength = blockLength == 0 ? ALIGN_NUM : blockLength;
    tileLength = tileLength > TILE_LENGTH ? TILE_LENGTH : tileLength;

    uint32_t baseUnits = 0;
    uint32_t remainUnits = 0;
    if (length > 0 && usedCoreNum > 0) {
        uint32_t units = CeilDiv(length, 16);
        baseUnits = units / usedCoreNum;
        remainUnits = units - baseUnits * usedCoreNum;
    }

    uint32_t dtX = static_cast<uint32_t>(dtypeX);
    uint32_t mode = 0U;
    if (length <= TINY_LENGTH) {
        mode = (length & (ALIGN_NUM - 1)) == 0 ? 2U : 1U;
    }
    ASCENDC_TPL_SEL_PARAM(context, dtX, mode);

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->length = length;
    tiling->blockLength = blockLength;
    tiling->tileLength = tileLength;
    tiling->baseUnits = baseUnits;
    tiling->remainUnits = remainUnits;

    context->SetBlockDim(usedCoreNum);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    if (xShape == nullptr || yShape == nullptr) {
        return GRAPH_FAILED;
    }
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Erf : public OpDef {
public:
    explicit Erf(const char *name) : OpDef(name) {
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
