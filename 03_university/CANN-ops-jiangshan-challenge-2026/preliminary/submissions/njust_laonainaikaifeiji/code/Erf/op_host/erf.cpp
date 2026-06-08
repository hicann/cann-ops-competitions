#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t SMALL_TILE_LENGTH = 1024;
constexpr uint32_t MID_TILE_LENGTH = 4096;
constexpr uint32_t LARGE_TILE_LENGTH = 8192;
constexpr uint32_t UB_BUFFER_NUM = BUFFER_NUM * 2 + 2;
constexpr uint32_t ALIGN_NUM = 8;

static uint32_t AlignUp(uint32_t value, uint32_t align) {
    return (value + align - 1) / align * align;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t num_cores_aiv = platform.GetCoreNumAiv();
    uint64_t ub_size = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_x = tensor_x->GetDataType();
    uint32_t length_x = static_cast<uint32_t>(tensor_x->GetShapeSize());

    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    uint32_t block_dim = static_cast<uint32_t>(num_cores_aiv);
    if (block_dim == 0) {
        block_dim = 1;
    }
    if (length_x < block_dim) {
        block_dim = length_x == 0 ? 1 : length_x;
    }
    if (length_x == 0) {
        ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
        tiling->length = 0;
        tiling->blockLength = 0;
        tiling->tileLength = SMALL_TILE_LENGTH;
        context->SetBlockDim(1);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
    uint32_t block_length = AlignUp((length_x + block_dim - 1) / block_dim, ALIGN_NUM);
    block_dim = (length_x + block_length - 1) / block_length;
    if (block_dim == 0) {
        block_dim = 1;
    }

    uint32_t tile_length = LARGE_TILE_LENGTH;
    if (length_x <= SMALL_TILE_LENGTH) {
        tile_length = SMALL_TILE_LENGTH;
    } else if (length_x <= MID_TILE_LENGTH) {
        tile_length = MID_TILE_LENGTH;
    }
    uint32_t max_tile_length = static_cast<uint32_t>(ub_size / sizeof(float) / UB_BUFFER_NUM);
    max_tile_length = max_tile_length / ALIGN_NUM * ALIGN_NUM;
    if (max_tile_length > 0 && max_tile_length < tile_length) {
        tile_length = max_tile_length;
    }
    if (block_length < tile_length) {
        tile_length = block_length;
    }
    if (tile_length < ALIGN_NUM) {
        tile_length = ALIGN_NUM;
    }

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    tiling->length = length_x;
    tiling->blockLength = block_length;
    tiling->tileLength = tile_length;

    context->SetBlockDim(block_dim);
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
    ge::DataType dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, dtype);
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
