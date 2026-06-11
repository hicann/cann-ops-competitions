// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace {
constexpr uint32_t FLOAT_ALIGN_NUM = 8;
constexpr uint32_t MIN_ELEMENTS_PER_CORE = 512;
constexpr uint32_t MAX_TILE_LENGTH = 32768;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_TENSOR_NUM = 2 * BUFFER_NUM + 1;

uint32_t CeilDiv(uint32_t value, uint32_t factor)
{
    return (value + factor - 1) / factor;
}

uint32_t AlignDown(uint32_t value, uint32_t align)
{
    return value / align * align;
}

uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return CeilDiv(value, align) * align;
}
}  // namespace

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t aiv_core_num = platform.GetCoreNumAiv();
        uint32_t max_core_num = aiv_core_num > 0 ? static_cast<uint32_t>(aiv_core_num) : 1;

        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t length_x = tensor_x->GetShapeSize();

        uint32_t used_core_num = 1;
        if (length_x > 0) {
            used_core_num = CeilDiv(length_x, MIN_ELEMENTS_PER_CORE);
            used_core_num = used_core_num > max_core_num ? max_core_num : used_core_num;
            used_core_num = used_core_num == 0 ? 1 : used_core_num;
        }

        uint32_t tile_length = static_cast<uint32_t>(
            ub_size / (sizeof(float) * UB_TENSOR_NUM));
        tile_length = tile_length > MAX_TILE_LENGTH ? MAX_TILE_LENGTH : tile_length;
        tile_length = AlignDown(tile_length, FLOAT_ALIGN_NUM);
        tile_length = tile_length < FLOAT_ALIGN_NUM ? FLOAT_ALIGN_NUM : tile_length;

        uint32_t block_length = length_x == 0 ? 0 : AlignUp(CeilDiv(length_x, used_core_num), FLOAT_ALIGN_NUM);
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        uint32_t small_mode_limit = tile_length * 2;
        uint32_t schMode = length_x <= small_mode_limit ? ERF_TPL_MODE_SMALL : ERF_TPL_MODE_GENERAL;
        ASCENDC_TPL_SEL_PARAM(context, DT_X, schMode);

        ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
        tiling->totalLength = length_x;
        tiling->blockLength = block_length;
        tiling->tileLength = tile_length;

        context->SetBlockDim(used_core_num);
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
        ge::DataType x_dtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, x_dtype);
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
                .UnknownShapeFormat({ge::FORMAT_ND})
                .AutoContiguous();
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
                .AutoContiguous();
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Erf);
}  // namespace ops
