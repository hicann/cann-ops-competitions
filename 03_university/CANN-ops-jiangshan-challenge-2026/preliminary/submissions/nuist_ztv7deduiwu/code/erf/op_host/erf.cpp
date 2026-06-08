// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace {
int64_t CeilDiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

int64_t CeilAlign(int64_t value, int64_t align) {
    return ((value + align - 1) / align) * align;
}

int64_t FloorDiv(int64_t a, int64_t b) {
    return a / b;
}

int64_t FloorAlign(int64_t value, int64_t align) {
    return (value / align) * align;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t num_cores = platform.GetCoreNumAiv();
    uint64_t ub_size;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_x = tensor_x->GetDataType();
    int32_t dtype_size = static_cast<int32_t>(ge::GetSizeByDataType(dtype_x));
    int64_t total_num = static_cast<int64_t>(tensor_x->GetShapeSize());

    int64_t ub_size_elems = static_cast<int64_t>(ub_size) / dtype_size;
    int64_t ub_block_size = 8;
    if (dtype_size == 2) {
        ub_block_size = 16;
    }

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    tiling->totalNum = total_num;

    if (total_num == 0) {
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    constexpr int64_t KERNEL_BUFFER_NUM = 2;
    constexpr int64_t KERNEL_TEMP_FACTOR = 4;
    int64_t total_buffers_per_ele = 2 * KERNEL_BUFFER_NUM + KERNEL_TEMP_FACTOR;
    int64_t ub_factor = FloorAlign(FloorDiv(ub_size_elems, total_buffers_per_ele), ub_block_size);

    if (ub_factor < ub_block_size) {
        ub_factor = ub_block_size;
    }

    int64_t block_factor = CeilAlign(CeilDiv(total_num, num_cores), ub_block_size);
    int64_t used_cores = CeilDiv(total_num, block_factor);
    if (used_cores > num_cores) {
        used_cores = num_cores;
    }
    if (used_cores < 1) {
        used_cores = 1;
    }

    tiling->blockFactor = block_factor;
    tiling->ubFactor = ub_factor;

    context->SetBlockDim(used_cores);

    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

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
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(Erf);
}  // namespace ops
