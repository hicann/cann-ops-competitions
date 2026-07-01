#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
constexpr uint64_t BLOCK_SPLIT_LENGTH = 1024;
constexpr uint64_t TILE_LENGTH = 4096;

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t num_cores_aiv = platform.GetCoreNumAiv();
    if (num_cores_aiv <= 0) {
        num_cores_aiv = 1;
    }

    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_x = tensor_x->GetDataType();
    uint64_t length_x = tensor_x->GetShapeSize();

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    tiling->length = length_x;

    uint64_t needCores = length_x == 0 ? 1 : (length_x + BLOCK_SPLIT_LENGTH - 1) / BLOCK_SPLIT_LENGTH;
    uint32_t blockDim = static_cast<uint32_t>(
        needCores < static_cast<uint64_t>(num_cores_aiv) ? needCores : static_cast<uint64_t>(num_cores_aiv));
    context->SetBlockDim(blockDim);
    uint64_t blockLength = (length_x + blockDim - 1) / blockDim;
    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    uint32_t USE_PIPE = blockLength > TILE_LENGTH ? 1 : 0;
    uint32_t ONE_CORE = blockDim == 1 ? 1 : 0;
    ASCENDC_TPL_SEL_PARAM(context, DT_X, USE_PIPE, ONE_CORE);

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
            .AddConfig("ascend910b")
            .AddConfig("ascend910_93");
    }
};
OP_ADD(Erf);
}  // namespace ops
