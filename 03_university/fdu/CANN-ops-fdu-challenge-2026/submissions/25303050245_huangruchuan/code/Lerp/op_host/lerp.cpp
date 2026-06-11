#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstdint>

#include "../op_kernel/lerp_tiling.h"
#include "../op_kernel/tiling_key_lerp.h"

namespace optiling {

namespace {

constexpr uint32_t kTargetElemsPerCore = 4096;

static uint32_t PickBlockDim(uint32_t length, int32_t hardwareCores)
{
    uint32_t cores = hardwareCores > 0 ? static_cast<uint32_t>(hardwareCores) : 1U;
    if (length == 0) {
        return 1;
    }
    uint32_t blocks = (length + kTargetElemsPerCore - 1) / kTargetElemsPerCore;
    if (blocks > cores) {
        blocks = cores;
    }
    if (blocks > length) {
        blocks = length;
    }
    return blocks == 0 ? 1 : blocks;
}

static ge::graphStatus ReadInputs(gert::TilingContext *context, uint32_t &length, ge::DataType &dtype)
{
    const gert::Tensor *start = context->GetRequiredInputTensor(0);
    const gert::Tensor *end = context->GetRequiredInputTensor(1);
    if (start == nullptr || end == nullptr) {
        return ge::GRAPH_FAILED;
    }
    length = static_cast<uint32_t>(start->GetShapeSize());
    if (length != static_cast<uint32_t>(end->GetShapeSize())) {
        return ge::GRAPH_FAILED;
    }
    dtype = start->GetDataType();
    return ge::GRAPH_SUCCESS;
}

}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    uint32_t length = 0;
    ge::DataType dtype = ge::DT_FLOAT;
    if (ReadInputs(context, length, dtype) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const float *weight = attrs->GetFloat(0);
    if (weight == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint32_t dtypeKey = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, dtypeKey);

    LerpTilingData *tiling = context->GetTilingData<LerpTilingData>();
    tiling->length = length;
    tiling->weight = *weight;

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    context->SetBlockDim(PickBlockDim(length, platform.GetCoreNumAiv()));

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *startShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    if (startShape == nullptr || yShape == nullptr) {
        return GRAPH_FAILED;
    }
    *yShape = *startShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Lerp : public OpDef {
public:
    explicit Lerp(const char *name) : OpDef(name) {
        this->Input("start")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("end")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("weight").AttrType(REQUIRED).Float();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Lerp);
}
