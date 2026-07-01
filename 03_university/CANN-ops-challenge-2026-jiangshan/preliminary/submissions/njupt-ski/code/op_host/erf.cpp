#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
       
        int32_t num_cores_aiv = platform.GetCoreNumAiv();

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t total_length = static_cast<uint32_t>(tensor_x->GetShapeSize());

        constexpr uint32_t ALIGN_NUM = 8;
        constexpr uint32_t POLYNOMIAL_THRESHOLD = 8192;
        
       
        constexpr uint32_t tile_size = 8192; 
        uint32_t use_polynomial = total_length <= POLYNOMIAL_THRESHOLD ? 1 : 0;

        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X, use_polynomial);
        
        uint32_t total_blocks = (total_length + ALIGN_NUM - 1) / ALIGN_NUM;
        uint32_t block_dim = static_cast<uint32_t>(num_cores_aiv);
        
        if (total_blocks < block_dim) {
            block_dim = total_blocks;
        }
        if (block_dim == 0) {
            block_dim = 1;
        }

       
        uint32_t blocks_per_core = total_blocks / block_dim;
        uint32_t remainder_blocks = total_blocks % block_dim;

        ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
        tiling->totalLength = total_length;
        tiling->usedCores = block_dim;
        tiling->blocksPerCore = blocks_per_core;
        tiling->remainderBlocks = remainder_blocks;
        tiling->tileSize = tile_size;

        context->SetBlockDim(block_dim);
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
