// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
    static constexpr uint32_t kTileLength = 8192;
    static constexpr uint32_t kLargeTileLength = 12288;
    static constexpr uint32_t kMediumTensorElemsPerCore = 4096;
    static constexpr uint32_t kMediumTensorBoostMaxLength = 131072;
    static constexpr uint32_t kMediumTensorMinElemsPerCore = 4096;

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        uint32_t length_x = tensor_x->GetShapeSize();

        uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
        tiling->length = length_x;

        if (length_x <= kTileLength) {
            context->SetBlockDim(1);
            size_t *currentWorkspace = context->GetWorkspaceSizes(1);
            currentWorkspace[0] = 0;
            return ge::GRAPH_SUCCESS;
        }

        uint32_t block_dim = 1;
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();

        if (length_x <= kMediumTensorBoostMaxLength) {
            uint32_t boosted_cores = (length_x + kMediumTensorElemsPerCore - 1) / kMediumTensorElemsPerCore;
            if (boosted_cores > static_cast<uint32_t>(num_cores_aiv)) {
                boosted_cores = static_cast<uint32_t>(num_cores_aiv);
            }
            context->SetBlockDim(boosted_cores);
            size_t *currentWorkspace = context->GetWorkspaceSizes(1);
            currentWorkspace[0] = 0;
            return ge::GRAPH_SUCCESS;
        }

        block_dim = (length_x + kTileLength - 1) / kTileLength;
        if (block_dim > static_cast<uint32_t>(num_cores_aiv)) {
            block_dim = static_cast<uint32_t>(num_cores_aiv);
        }

        uint32_t min_elems_per_core = kMediumTensorMinElemsPerCore;
        if (length_x > kMediumTensorBoostMaxLength &&
            length_x > static_cast<uint32_t>(num_cores_aiv) * kTileLength) {
            min_elems_per_core = kLargeTileLength;
        }
        uint32_t min_cores_by_work = (length_x + min_elems_per_core - 1) / min_elems_per_core;
        if (length_x > kMediumTensorBoostMaxLength && block_dim > min_cores_by_work) {
            block_dim = min_cores_by_work;
        }
        if (block_dim == 0) {
            block_dim = 1;
        }

        context->SetBlockDim(block_dim);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *input_shape = context->GetRequiredInputShape(0);
        gert::Shape *output_shape = context->GetOutputShape(0);
        if (input_shape == nullptr || output_shape == nullptr) {
            return GRAPH_FAILED;
        }
        *output_shape = *input_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        ge::DataType input_dtype = context->GetRequiredInputDataType(0);
        if (input_dtype == ge::DT_UNDEFINED) {
            return GRAPH_FAILED;
        }
        return context->SetOutputDataType(0, input_dtype);
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
