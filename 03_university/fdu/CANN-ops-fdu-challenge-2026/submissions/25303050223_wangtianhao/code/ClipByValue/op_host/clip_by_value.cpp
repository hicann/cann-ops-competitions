#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/clip_by_value_tiling.h"
#include "../op_kernel/tiling_key_clip_by_value.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t maxCoreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());

    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_x = tensor_x->GetDataType();
    uint32_t length_x = tensor_x->GetShapeSize();

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const float *attr_min = attrs->GetFloat(0);
    const float *attr_max = attrs->GetFloat(1);

    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    constexpr uint32_t TILE_DATA_NUM_HOST = 4096;
    constexpr uint32_t SINGLE_CORE_THRESHOLD = 8192;

    uint32_t coreNum = 1;
    if (length_x <= SINGLE_CORE_THRESHOLD) {
        coreNum = 1;
    } else {
        coreNum = maxCoreNum;
    }

    if (coreNum == 0) {
        coreNum = 1;
    }

    uint32_t baseBlockLength = (length_x + coreNum - 1) / coreNum;
    uint32_t blockLength =
        ((baseBlockLength + TILE_DATA_NUM_HOST - 1) / TILE_DATA_NUM_HOST) * TILE_DATA_NUM_HOST;

    ClipByValueTilingData *tiling = context->GetTilingData<ClipByValueTilingData>();
    tiling->length = length_x;
    tiling->blockLength = blockLength;
    tiling->coreNum = coreNum;
    tiling->minValue = *attr_min;
    tiling->maxValue = *attr_max;

    context->SetBlockDim(coreNum);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *x_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class ClipByValue : public OpDef {
public:
    explicit ClipByValue(const char *name) : OpDef(name) {
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