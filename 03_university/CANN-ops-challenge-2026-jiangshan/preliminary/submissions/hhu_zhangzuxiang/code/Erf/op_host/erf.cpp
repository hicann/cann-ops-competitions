#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t length_x = tensor_x->GetShapeSize();

        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
        tiling->length = length_x;

        uint32_t tile_length = 16384;
        if (ub_size > 0) {
            uint32_t ub_limit = static_cast<uint32_t>(ub_size / (4u * sizeof(float)));
            if (ub_limit >= 8u && ub_limit < tile_length) {
                tile_length = ub_limit & ~7u;
            }
        }
        if (tile_length < 8u) {
            tile_length = 8u;
        }
        tiling->tileLength = tile_length;

        constexpr uint32_t block_split_length = 1024;
        uint32_t block_dim = (length_x + block_split_length - 1) / block_split_length;
        if (block_dim == 0) {
            block_dim = 1;
        }
        if (num_cores_aiv > 0 && block_dim > static_cast<uint32_t>(num_cores_aiv)) {
            block_dim = static_cast<uint32_t>(num_cores_aiv);
        }
        if (length_x > 0 && block_dim > length_x) {
            block_dim = length_x;
        }
        if (block_dim == 0) {
            block_dim = 1;
        }
        context->SetBlockDim(block_dim);

        tiling->blockLength = (length_x + block_dim - 1) / block_dim;
        if (tiling->blockLength == 0) {
            tiling->blockLength = 1;
        }
        tiling->blockLength = (tiling->blockLength + 7u) & ~7u;

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
