// Host tiling implementation
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <cstdint>

#include "../op_kernel/lerp_tiling.h"
#include "../op_kernel/tiling_key_lerp.h"

namespace optiling {
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BUFFER_COUNT = 7;

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t num_cores_aiv = platform.GetCoreNumAiv();
    uint64_t ub_size;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    const gert::Tensor *tensor_start = context->GetRequiredInputTensor(0);
    ge::DataType dtype_start = tensor_start->GetDataType();
    uint32_t dtype_size_start = ge::GetSizeByDataType(dtype_start);
    uint32_t length_start = tensor_start->GetShapeSize();

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const float *attr_weight = attrs->GetFloat(0);

    uint32_t DT_START = static_cast<uint32_t>(dtype_start);
    ASCENDC_TPL_SEL_PARAM(context, DT_START);

    uint32_t data_block_length = 32 / dtype_size_start;
    uint32_t data_block_count = length_start / data_block_length;
    uint32_t block_dim = 1;
    if (data_block_count > 0) {
        block_dim = static_cast<uint32_t>(num_cores_aiv);
        if (block_dim > data_block_count) {
            block_dim = data_block_count;
        }
    }

    uint32_t data_blocks_per_core = data_block_count == 0 ? 0 : (data_block_count + block_dim - 1) / block_dim;
    uint32_t block_length = data_blocks_per_core * data_block_length;
    uint32_t max_tile_length = ub_size / (BUFFER_COUNT * dtype_size_start);
    max_tile_length = max_tile_length / data_block_length * data_block_length;
    if (max_tile_length == 0) {
        max_tile_length = data_block_length;
    }

    uint32_t tile_num = (block_length + max_tile_length * BUFFER_NUM - 1) / (max_tile_length * BUFFER_NUM);
    if (tile_num == 0) {
        tile_num = 1;
    }

    LerpTilingData *tiling = context->GetTilingData<LerpTilingData>();
    tiling->length = length_start;
    tiling->tileNum = tile_num;
    tiling->weight = *attr_weight;
    context->GetRawTilingData()->SetDataSize(sizeof(LerpTilingData));

    context->SetBlockDim(block_dim);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *startShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = *startShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Lerp : public OpDef {
public:
    explicit Lerp(const char *name) : OpDef(name) {
        this->Input("start")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("end")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("weight").AttrType(REQUIRED).Float();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(Lerp);
}  // namespace ops
