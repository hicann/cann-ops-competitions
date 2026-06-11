#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstdint>

#include "../op_kernel/clip_by_value_tiling.h"
#include "../op_kernel/tiling_key_clip_by_value.h"

namespace optiling {
namespace {

constexpr uint32_t kTargetElemsPerCore = 8192;

static uint32_t PickBlockDim(uint32_t length, int32_t maxCores)
{
    uint32_t cores = maxCores > 0 ? static_cast<uint32_t>(maxCores) : 1U;
    uint32_t blocks = (length + kTargetElemsPerCore - 1) / kTargetElemsPerCore;
    if (blocks == 0) {
        blocks = 1;
    }
    return blocks > cores ? cores : blocks;
}

}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::Tensor *x = context->GetRequiredInputTensor(0);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (x == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const float *minValue = attrs->GetFloat(0);
    const float *maxValue = attrs->GetFloat(1);
    if (minValue == nullptr || maxValue == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint32_t length = static_cast<uint32_t>(x->GetShapeSize());
    uint32_t dtypeKey = static_cast<uint32_t>(x->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, dtypeKey);

    ClipByValueTilingData *tiling = context->GetTilingData<ClipByValueTilingData>();
    tiling->length = length;
    tiling->minValue = *minValue;
    tiling->maxValue = *maxValue;

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    context->SetBlockDim(PickBlockDim(length, platform.GetCoreNumAiv()));

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    *context->GetOutputShape(0) = *context->GetInputShape(0);
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
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(ClipByValue);

}  // namespace ops
