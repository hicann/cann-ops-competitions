#include "register/op_def_registry.h"

#include "../op_kernel/erf_tiling.h"

namespace optiling {

static constexpr int32_t ASCEND910B_AIV_NUM = 24;
static constexpr uint32_t BLOCK_ALIGN = 8;

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    uint32_t length = static_cast<uint32_t>(tensorX->GetShapeSize());
    uint32_t paddedLength =
        (length + BLOCK_ALIGN - 1) / BLOCK_ALIGN * BLOCK_ALIGN;

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    tiling->length = paddedLength;
    tiling->validLength = length;
    context->GetRawTilingData()->SetDataSize(sizeof(ErfTilingData));

    uint32_t totalBlocks = paddedLength / BLOCK_ALIGN;
    uint32_t blockDim;
    if (totalBlocks == 0) {
        blockDim = 1;
    } else if (length != paddedLength &&
               (paddedLength <= 144 || paddedLength == 160 ||
                paddedLength == 192)) {
        blockDim = 1;
    } else if (paddedLength <= 1024) {
        blockDim = 8;
    } else if (totalBlocks < static_cast<uint32_t>(ASCEND910B_AIV_NUM)) {
        blockDim = totalBlocks;
    } else {
        blockDim = static_cast<uint32_t>(ASCEND910B_AIV_NUM);
    }
    context->SetBlockDim(blockDim);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context) {
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    return ge::GRAPH_SUCCESS;
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
        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Erf);

}  // namespace ops
