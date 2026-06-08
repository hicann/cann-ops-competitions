// Host 侧 Tiling 实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

constexpr uint32_t EXPECTED_AIV_CORES = 40u;

namespace optiling {

static inline uint32_t align_up(uint32_t v, uint32_t a) {
    return ((v + a - 1u) / a) * a;
}
static inline uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

// 离散分层核数选择
static inline uint32_t SelectBlockDim(uint32_t length, uint32_t max_cores) {
    if (length <= 32768u)  return std::min(8u, max_cores);   
    if (length <= 131072u) return std::min(16u, max_cores);
    if (length <= 262144u) return std::min(20u, max_cores);
    if (length <= 524288u) return std::min(32u, max_cores);
    return std::min(40u, max_cores);
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t num_cores = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (num_cores != EXPECTED_AIV_CORES) {
        return ge::GRAPH_FAILED;
    }

    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_x = tensor_x->GetDataType();
    uint32_t dtype_size = static_cast<uint32_t>(ge::GetSizeByDataType(dtype_x));
    uint32_t length = tensor_x->GetShapeSize();

    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    const uint32_t ALIGN_NUM = 32u / dtype_size;
    const uint32_t INNER_TILE = 8192u;

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    tiling->length = length;

    uint32_t per_core = ALIGN_NUM;
    uint32_t tail_core_len = 0u;
    uint32_t tile_length = ALIGN_NUM;
    uint32_t tail_core_full_tiles = 0u;
    uint32_t tail_core_aligned_tail = 0u;
    uint32_t mode = 0u;
    uint32_t block_dim = 1u;

    if (length == 0u) {
        block_dim = 1u;
        per_core = ALIGN_NUM;
        tile_length = ALIGN_NUM;
        mode = 0u;
    } 
    else if (length <= 16384u) {
        // 【你要求的修改】：小于等于16384统一使用8核
        block_dim = 8u;
        uint32_t base = ceil_div(length, block_dim);
        per_core = align_up(base, ALIGN_NUM);
        block_dim = ceil_div(length, per_core);   // 重新校正，防止对齐后核数变化
        if (block_dim > 8u) block_dim = 8u;

        uint32_t last_start = (block_dim - 1u) * per_core;
        uint32_t last_raw = length - last_start;
        tail_core_len = align_up(last_raw, ALIGN_NUM);
        tile_length = per_core;
        mode = 0u;        // 统一走单 tile 多核模式
    } 
    else if (length <= num_cores * INNER_TILE) {
        // MEDIUM
        block_dim = SelectBlockDim(length, num_cores);
        uint32_t base = ceil_div(length, block_dim);
        per_core = align_up(base, ALIGN_NUM);
        block_dim = ceil_div(length, per_core);
        if (block_dim > num_cores) block_dim = num_cores;

        uint32_t last_start = (block_dim - 1u) * per_core;
        uint32_t last_raw = length - last_start;
        tail_core_len = align_up(last_raw, ALIGN_NUM);
        tile_length = per_core;
        mode = 0u;
    } 
    else {
        // LARGE
        uint32_t base = ceil_div(length, num_cores);
        per_core = align_up(base, INNER_TILE);
        block_dim = ceil_div(length, per_core);
        if (block_dim > num_cores) block_dim = num_cores;

        uint32_t last_start = (block_dim - 1u) * per_core;
        uint32_t last_raw = length - last_start;
        tail_core_full_tiles = last_raw / INNER_TILE;
        uint32_t raw_tail = last_raw - tail_core_full_tiles * INNER_TILE;
        tail_core_aligned_tail = align_up(raw_tail, ALIGN_NUM);
        tail_core_len = last_raw;
        tile_length = INNER_TILE;
        mode = 1u;
    }

    tiling->perCore = per_core;
    tiling->tailCoreLen = tail_core_len;
    tiling->tileLength = tile_length;
    tiling->tailCoreFullTiles = tail_core_full_tiles;
    tiling->tailCoreAlignedTail = tail_core_aligned_tail;
    tiling->mode = mode;

    context->SetBlockDim(block_dim);
    size_t *ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

// InferShape 和 InferDataType 保持不变
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
}

namespace ops {
    class Erf : public OpDef {
    public:
        explicit Erf(const char *name) : OpDef(name) {
            this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
            this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
        }
    };
    OP_ADD(Erf);
}