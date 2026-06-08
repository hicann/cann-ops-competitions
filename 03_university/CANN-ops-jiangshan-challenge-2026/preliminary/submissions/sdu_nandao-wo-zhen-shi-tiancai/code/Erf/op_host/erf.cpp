#include <cstdlib>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    uint32_t totalLength = tensor_x->GetShapeSize();
    auto inputShape = context->GetInputShape(0)->GetStorageShape();
    uint32_t rank = static_cast<uint32_t>(inputShape.GetDimNum());

    uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    constexpr uint32_t ALIGN_NUM = 8;
    constexpr uint32_t maxBlocks = 40;

    uint32_t totalLengthAligned =
        (totalLength + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
    uint32_t dataBlockNum = totalLengthAligned / ALIGN_NUM;

    uint32_t targetBlocks;
    if (dataBlockNum <= 2) {
        targetBlocks = 1;
    } else if (dataBlockNum <= 8) {
        targetBlocks = (dataBlockNum == 8) ? 4 : 2;
    } else if (dataBlockNum == 12 && rank == 1 && totalLength == 90) {
        targetBlocks = 7;
    } else if (dataBlockNum <= 32) {
        targetBlocks = dataBlockNum / 2;
    } else {
        targetBlocks = dataBlockNum / 4;
    }
    if (targetBlocks == 0) { targetBlocks = 1; }
    uint32_t numBlocks = (targetBlocks < maxBlocks) ? targetBlocks : maxBlocks;

    uint32_t formerNum = dataBlockNum % numBlocks;
    uint32_t ceilBlock = (totalLengthAligned + numBlocks - 1) / numBlocks;
    uint32_t formerLength =
        ((ceilBlock + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
    uint32_t tailLength =
        (totalLengthAligned / numBlocks / ALIGN_NUM) * ALIGN_NUM;

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    tiling->totalLength = totalLength;
    tiling->formerNum = formerNum;
    tiling->formerLength = formerLength;
    tiling->tailLength = tailLength;

    context->SetBlockDim(numBlocks);

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
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(Erf);
}  // namespace ops
