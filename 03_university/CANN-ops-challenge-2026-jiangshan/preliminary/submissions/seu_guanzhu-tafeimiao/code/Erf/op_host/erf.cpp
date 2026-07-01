#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t num_cores = platform.GetCoreNumAiv();
    uint64_t ub_size;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_x = tensor_x->GetDataType();
    uint32_t total_length = tensor_x->GetShapeSize();
    constexpr uint32_t ALIGN_ELEM = 128;
    constexpr uint32_t DIRECT_ALIGN_ELEM = 8;
    constexpr uint32_t WORK_BUFFER_NUM = 3;      // bufAbs + bufX2 + bufAcc
    constexpr uint32_t UB_OVERHEAD_BYTES = 0;

    constexpr uint32_t TILE_BUFFERS = 4;           // for Medium/Direct: 1TQ + 3TB
    constexpr uint32_t TILE_BUFFERS_LARGE = 7;      // for Large: 2TQ x 2slot + 3TB

    uint32_t available_ub = ub_size > UB_OVERHEAD_BYTES ? (ub_size - UB_OVERHEAD_BYTES) : ub_size;
    uint32_t max_tile = available_ub / (TILE_BUFFERS * sizeof(float));
    max_tile = (max_tile / ALIGN_ELEM) * ALIGN_ELEM;
    if (max_tile < ALIGN_ELEM) max_tile = ALIGN_ELEM;

    uint32_t max_tile_large = available_ub / (TILE_BUFFERS_LARGE * sizeof(float));
    max_tile_large = (max_tile_large / ALIGN_ELEM) * ALIGN_ELEM;
    if (max_tile_large < ALIGN_ELEM) max_tile_large = ALIGN_ELEM;

    uint32_t tile_data_num;
    uint32_t block_dim;
    uint32_t IS_SPLIT;

    if (total_length <= max_tile) {
        tile_data_num = ((total_length + DIRECT_ALIGN_ELEM - 1) / DIRECT_ALIGN_ELEM) * DIRECT_ALIGN_ELEM;
        block_dim = 1;
        IS_SPLIT = (total_length <= 1024 && ((total_length & (DIRECT_ALIGN_ELEM - 1)) == 0)) ? 7 : 5;
    } else if (total_length <= max_tile * 4) {
        bool is_low_medium = total_length <= max_tile * 2;
        bool is_mid_medium = !is_low_medium && total_length <= max_tile * 3;
        tile_data_num = max_tile;
        uint32_t coef = is_low_medium ? 9 : 12;
        block_dim = std::min(static_cast<uint32_t>(num_cores),
                             (total_length * coef + max_tile - 1) / max_tile);
        if (block_dim < 1) block_dim = 1;
        uint32_t search_start = block_dim;
        if (!is_low_medium) {
            search_start = std::min(static_cast<uint32_t>(num_cores), search_start + (is_mid_medium ? 3 : 6));
        }
        uint32_t aligned_block_dim = 0;
        for (uint32_t b = search_start; b > 0; --b) {
            if (total_length % b != 0) {
                continue;
            }
            uint32_t per_core = total_length / b;
            if (per_core <= tile_data_num && ((per_core & 7u) == 0)) {
                aligned_block_dim = b;
                break;
            }
        }
        if (aligned_block_dim != 0) {
            block_dim = aligned_block_dim;
            IS_SPLIT = 4;
        } else {
            IS_SPLIT = 1;
        }
    } else {
        tile_data_num = max_tile_large;
        block_dim = std::min(static_cast<uint32_t>(num_cores),
                             (total_length * 12 + max_tile_large - 1) / max_tile_large);
        if (block_dim < 1) block_dim = 1;
        IS_SPLIT = 0;
    }

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    tiling->totalLength = total_length;
    tiling->tileDataNum = tile_data_num;

    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(DT_X, IS_SPLIT);
    context->SetTilingKey(tilingKey);

    context->SetBlockDim(block_dim);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto input_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, input_dtype);
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class Erf : public OpDef {
public:
    explicit Erf(const char *name) : OpDef(name)
    {
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
