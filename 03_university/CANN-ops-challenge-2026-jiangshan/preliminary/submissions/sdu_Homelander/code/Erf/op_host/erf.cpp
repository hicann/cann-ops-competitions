#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {

constexpr uint32_t HOST_TILE_LENGTH = 9728;

constexpr uint32_t TINY_BLOCK_DIM =8;
constexpr uint32_t MID_BLOCK_DIM = 20;
constexpr uint32_t LARGE_BLOCK_DIM = 40;

// 极小数据阈值：先按 tile 数判断。
// 这里先给 8 个 tile 作为起点，后面根据实测微调。
constexpr uint32_t TINY_TILE_THRESHOLD = 6;

// 大数据阈值：沿用你刚才 20 / 40 分档的思路。
// 这个值用你刚才已经起效的阈值替换。
constexpr uint32_t LARGE_TILE_THRESHOLD = 512;

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    int32_t numCoresAiv = platform.GetCoreNumAiv();

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    ge::DataType dtypeX = tensorX->GetDataType();

    uint32_t lengthX = static_cast<uint32_t>(tensorX->GetShapeSize());

    uint32_t DT_X = static_cast<uint32_t>(dtypeX);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    tiling->length = lengthX;

    uint32_t totalTileNum = (lengthX + HOST_TILE_LENGTH - 1) / HOST_TILE_LENGTH;

    uint32_t blockDim = MID_BLOCK_DIM;

    if (totalTileNum <= TINY_TILE_THRESHOLD) {
        blockDim = TINY_BLOCK_DIM;
    } else if (totalTileNum >= LARGE_TILE_THRESHOLD) {
        blockDim = LARGE_BLOCK_DIM;
    } else {
        blockDim = MID_BLOCK_DIM;
    }

    context->SetBlockDim(blockDim);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

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
    ge::DataType xDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, xDtype);

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

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Erf);

}  // namespace ops