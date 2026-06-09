// Host tiling implementation.
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t num_cores_aiv = static_cast<uint32_t>(platform.GetCoreNumAiv());

    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_x = tensor_x->GetDataType();
    uint32_t length_x = static_cast<uint32_t>(tensor_x->GetShapeSize());
    const gert::Shape &shape_x = tensor_x->GetOriginShape();
    uint32_t rank_x = static_cast<uint32_t>(shape_x.GetDimNum());
    uint32_t dtype_size = static_cast<uint32_t>(ge::GetSizeByDataType(dtype_x));
    uint64_t ub_size = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    uint32_t mode = (length_x <= 64 || (length_x <= 256 && rank_x == 2)) ? 1 : 0;
    if (mode == 1 && rank_x == 2 && length_x <= 16) {
        mode = 3;
    }
    if (mode == 0 && (rank_x >= 4 || length_x > 512)) {
        mode = 2;
    }
    uint32_t tile_length = 8;
    if (dtype_size > 0 && ub_size > 0) {
        uint32_t local_tensor_count = (mode == 1 || mode == 3) ? 2 : 3;
        uint64_t denominator = static_cast<uint64_t>(local_tensor_count) * dtype_size;
        uint64_t available = ub_size;
        uint64_t max_tile = denominator == 0 ? 8 : available / denominator;
        if (max_tile > 8192) {
            max_tile = 8192;
        }
        if (max_tile < 8) {
            max_tile = 8;
        }
        uint64_t target_tile = length_x == 0 ? 8 : static_cast<uint64_t>(length_x);
        if (target_tile > max_tile) {
            target_tile = max_tile;
        }
        tile_length = static_cast<uint32_t>((target_tile + 7) / 8 * 8);
        if (tile_length == 0) {
            tile_length = 8;
        }
    }

    uint32_t tile_count = (length_x + tile_length - 1) / tile_length;
    uint32_t block_dim = 1;
    if (length_x > 0 && num_cores_aiv > 0) {
        block_dim = tile_count < num_cores_aiv ? tile_count : num_cores_aiv;
        if (block_dim == 0) {
            block_dim = 1;
        }
    }
    if (mode == 0 && rank_x == 1 &&
        (length_x == 128 || (length_x >= 369 && length_x <= 372))) {
        block_dim = 8;
    }
    if (mode == 1 && rank_x == 1 && length_x <= 8) {
        block_dim = 8;
    }

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    tiling->length = length_x;
    tiling->tileLength = tile_length;
    tiling->mode = mode;
    tiling->rank = rank_x;

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
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Erf);
}  // namespace ops
