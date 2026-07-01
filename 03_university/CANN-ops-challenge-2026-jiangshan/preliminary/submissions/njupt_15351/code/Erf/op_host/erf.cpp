// Host side tiling and inference for torch.erf / torch.special.erf.
#include <algorithm>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t num_cores_aiv = static_cast<uint32_t>(platform.GetCoreNumAiv());
        if (num_cores_aiv == 0) {
            num_cores_aiv = 1;
        }

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        uint32_t length_x = tensor_x->GetShapeSize();

        ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
        tiling->length = length_x;

        constexpr uint32_t kAlignElements = 32 / sizeof(float);
        constexpr uint32_t kLargeAlignElements = 512 / sizeof(float);
        constexpr uint32_t kDirectLength = 128;
        constexpr uint32_t kLargeSplitLength = 8192;
        constexpr uint32_t kSingleCoreLimit = 128;
        constexpr uint32_t kElementsPerCore = 512;
        uint32_t split_align = length_x >= kLargeSplitLength ? kLargeAlignElements : kAlignElements;
        uint32_t aligned_blocks = (length_x + split_align - 1) / split_align;
        tiling->split_align = split_align;
        tiling->use_direct = length_x <= kDirectLength ? 1U : 0U;
        context->SetTilingKey(TILING_KEY_DEFAULT);

        uint32_t block_dim = 1;
        if (length_x > kSingleCoreLimit) {
            uint32_t blocks_per_core = kElementsPerCore / split_align;
            if (blocks_per_core == 0) {
                blocks_per_core = 1;
            }
            uint32_t expect_cores = (aligned_blocks + blocks_per_core - 1) / blocks_per_core;
            block_dim = std::min(num_cores_aiv, std::max(1U, expect_cores));
        }
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
        const auto input_dtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, input_dtype);
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
