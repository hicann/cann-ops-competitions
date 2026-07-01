#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t coreNum = platform.GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = 1;
    }

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    ge::DataType dtypeX = tensorX->GetDataType();
    int32_t dtypeSize = ge::GetSizeByDataType(dtypeX);
    uint32_t totalLength = static_cast<uint32_t>(tensorX->GetShapeSize());

    uint32_t DT_X = static_cast<uint32_t>(dtypeX);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    constexpr uint32_t BUFFER_COUNT = 3;
    constexpr uint32_t MIN_ELEMENTS_PER_CORE = 1024;
    uint32_t tileLength =
        static_cast<uint32_t>(ubSize / (BUFFER_COUNT * dtypeSize));
    tileLength = tileLength / 8 * 8;
    if (tileLength == 0) {
        tileLength = 8;
    }

    uint32_t blockDim =
        (totalLength + MIN_ELEMENTS_PER_CORE - 1) / MIN_ELEMENTS_PER_CORE;
    if (blockDim == 0) {
        blockDim = 1;
    }
    if (blockDim > static_cast<uint32_t>(coreNum)) {
        blockDim = static_cast<uint32_t>(coreNum);
    }

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    tiling->totalLength = totalLength;
    tiling->tileLength = tileLength;
    tiling->blockDim = blockDim;

    context->SetBlockDim(blockDim);
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
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
