// Host侧Tiling实现 —— V6
// V2策略 + TILE_LENGTH扩到12288（V6释放了2个calc buffer的UB）
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

        int32_t num_cores = platform.GetCoreNumAiv();
        if (num_cores <= 0) {
            num_cores = platform.GetCoreNumAic();
        }
        if (num_cores <= 0) {
            num_cores = 1;
        }

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t length_x = tensor_x->GetShapeSize();

        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
        tiling->length = length_x;

        constexpr uint32_t TILE_LENGTH = 8192;

        int32_t used_cores = 1;
        if (length_x >= TILE_LENGTH) {
            uint32_t tile_like_blocks = (length_x + TILE_LENGTH - 1) / TILE_LENGTH;
            used_cores = static_cast<int32_t>(tile_like_blocks);
            if (used_cores > num_cores) {
                used_cores = num_cores;
            }
            if (used_cores < 1) {
                used_cores = 1;
            }
        }

        context->SetBlockDim(used_cores);

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *x_shape = context->GetInputShape(0);
        gert::Shape *y_shape = context->GetOutputShape(0);
        if (x_shape != nullptr && y_shape != nullptr) {
            *y_shape = *x_shape;
        }
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const ge::DataType x_dtype = context->GetInputDataType(0);
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
