#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t UB_RESERVE_BYTES = 16 * 1024;
constexpr uint32_t SMALL_LENGTH = 1024;
constexpr uint32_t MEDIUM_LENGTH = 65536;
constexpr uint32_t SMALL_TILE_LENGTH = 2048;
constexpr uint32_t MEDIUM_TILE_LENGTH = 4096;
constexpr uint32_t MAX_TILE_LENGTH = 12288;
constexpr uint32_t SMALL_CORE_GRAIN = 2048;
constexpr uint32_t LARGE_CORE_GRAIN = 8192;

static uint64_t AlignDown(uint64_t value, uint64_t align) {
    return value / align * align;
}

static uint32_t AlignUp(uint32_t value, uint32_t align) {
    return (value + align - 1) / align * align;
}

static uint32_t CeilDiv(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t num_cores_aiv = platform.GetCoreNumAiv();
    uint64_t ub_size = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_x = tensor_x->GetDataType();
    uint32_t dtype_size_x = static_cast<uint32_t>(ge::GetSizeByDataType(dtype_x));
    uint32_t length_x = static_cast<uint32_t>(tensor_x->GetShapeSize());

    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    uint32_t max_core = num_cores_aiv > 0 ? static_cast<uint32_t>(num_cores_aiv) : 1;
    uint32_t align_num = std::max<uint32_t>(1, BLOCK_BYTES / dtype_size_x);
    uint32_t block_dim = 1;
    uint32_t tile_length = SMALL_TILE_LENGTH;
    if (length_x > 0) {
        if (length_x <= SMALL_LENGTH) {
            block_dim = 1;
            tile_length = SMALL_TILE_LENGTH;
        } else if (length_x <= MEDIUM_LENGTH) {
            block_dim = std::min<uint32_t>(max_core, CeilDiv(length_x, SMALL_CORE_GRAIN));
            tile_length = MEDIUM_TILE_LENGTH;
        } else {
            block_dim = std::min<uint32_t>(max_core, CeilDiv(length_x, LARGE_CORE_GRAIN));
            uint64_t ub_available = ub_size > UB_RESERVE_BYTES ? ub_size - UB_RESERVE_BYTES : ub_size;
            uint64_t ub_tile = AlignDown(ub_available / (5 * dtype_size_x), align_num);
            tile_length = static_cast<uint32_t>(std::max<uint64_t>(MEDIUM_TILE_LENGTH,
                std::min<uint64_t>(MAX_TILE_LENGTH, ub_tile)));
        }
    }
    uint32_t block_length = length_x == 0 ? 0 : AlignUp(CeilDiv(length_x, block_dim), align_num);

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    tiling->length = length_x;
    tiling->blockLength = block_length;
    tiling->tileLength = tile_length;

    context->SetBlockDim(block_dim);
    size_t *current_workspace = context->GetWorkspaceSizes(1);
    current_workspace[0] = 0;
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
