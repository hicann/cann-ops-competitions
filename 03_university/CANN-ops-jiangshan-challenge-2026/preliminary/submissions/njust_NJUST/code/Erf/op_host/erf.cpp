#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/tiling_key_erf.h"

// Minimal tiling payload; work partitioning is computed in kernel.
struct ErfTilingDataHost {
    uint64_t length;
};

static uint32_t ChooseBlockDim(uint64_t elemCount, uint32_t maxCores) {
    if (elemCount <= 32768ULL)      return (maxCores < 8)  ? maxCores : 8;
    if (elemCount <= 131072ULL)     return (maxCores < 16) ? maxCores : 16;
    if (elemCount <= 262144ULL)     return (maxCores < 20) ? maxCores : 20;
    if (elemCount <= 524288ULL)     return (maxCores < 32) ? maxCores : 32;
    return (maxCores < 40) ? maxCores : 40;
}

namespace optiling {

ge::graphStatus TilingFunc(gert::TilingContext *ctx) {
    using namespace platform_ascendc;
    auto pf = PlatformAscendC(ctx->GetPlatformInfo());

    uint32_t aiv = (uint32_t)pf.GetCoreNumAiv();
    uint32_t aic = (uint32_t)pf.GetCoreNumAic();
    uint32_t av = (aiv > 0) ? aiv : ((aic > 0) ? aic : 1);

    auto *t = ctx->GetRequiredInputTensor(0);
    if (!t || t->GetDataType() != ge::DT_FLOAT)
        return ge::GRAPH_FAILED;

    uint64_t n = (uint64_t)t->GetShapeSize();
    uint32_t bd = ChooseBlockDim(n, av);

    uint32_t dt = (uint32_t)t->GetDataType();
    ASCENDC_TPL_SEL_PARAM(ctx, dt);

    auto *d = ctx->GetTilingData<ErfTilingDataHost>();
    if (!d) return ge::GRAPH_FAILED;
    d->length = n;

    ctx->SetBlockDim(bd);

    auto *ws = ctx->GetWorkspaceSizes(1);
    if (ws) ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}

namespace ge {
graphStatus InferShape(gert::InferShapeContext *c) {
    auto *xs = c->GetInputShape(0);
    auto *ys = c->GetOutputShape(0);
    if (!xs || !ys) return GRAPH_FAILED;
    *ys = *xs;
    return GRAPH_SUCCESS;
}
graphStatus InferDataType(gert::InferDataTypeContext *c) {
    c->SetOutputDataType(0, c->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Erf : public OpDef {
public:
    explicit Erf(const char *n) : OpDef(n) {
        this->Input("x").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(Erf);
}
