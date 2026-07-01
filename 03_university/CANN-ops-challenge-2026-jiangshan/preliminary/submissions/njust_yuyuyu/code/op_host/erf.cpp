#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (coreNum == 0) {
        coreNum = 1;
    }

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    ge::DataType dtypeX = tensorX->GetDataType();
    uint32_t lengthX = static_cast<uint32_t>(tensorX->GetShapeSize());

    constexpr uint32_t kTileLength = 4096;
    constexpr uint32_t kSmallTensorThreshold = 4096;
    constexpr uint32_t kTargetElemsPerCore = 8192;
    uint32_t blockNum = 1;
    if (lengthX > 0) {
        const uint32_t tileNum = (lengthX + kTileLength - 1) / kTileLength;
        if (lengthX <= kSmallTensorThreshold) {
            blockNum = 1;
        } else if (tileNum <= 4) {
            blockNum = coreNum >= 2 ? 2 : 1;
        } else {
            blockNum = (lengthX + kTargetElemsPerCore - 1) / kTargetElemsPerCore;
            if (blockNum == 0) {
                blockNum = 1;
            }
            if (blockNum > tileNum) {
                blockNum = tileNum;
            }
            if (blockNum > coreNum) {
                blockNum = coreNum;
            }
        }
        if (blockNum > tileNum) {
            blockNum = tileNum;
        }
    }

    uint32_t dtX = static_cast<uint32_t>(dtypeX);
    ASCENDC_TPL_SEL_PARAM(context, dtX);

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    tiling->length = lengthX;
    tiling->blockNum = blockNum;

    context->SetBlockDim(blockNum);
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
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
