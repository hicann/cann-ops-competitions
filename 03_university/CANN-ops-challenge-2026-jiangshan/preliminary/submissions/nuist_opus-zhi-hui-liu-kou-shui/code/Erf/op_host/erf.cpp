// PRC 2026-05-11 23:02:39 git commit: 336d71a
#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
    namespace {
    constexpr uint32_t kBufferCount = 6;
    constexpr uint32_t kMaxTileLength = 16384;
    constexpr uint32_t kMinElementsPerCore = 1024;
    }  // namespace

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t dtype_size_x = static_cast<uint32_t>(ge::GetSizeByDataType(dtype_x));
        uint64_t length_x = static_cast<uint64_t>(tensor_x->GetShapeSize());
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        uint32_t preferred_tile = kMaxTileLength;
        uint32_t block_dim = 1;
        uint64_t block_length = length_x;
        if (length_x > 0 && num_cores_aiv > 0 && dtype_size_x > 0) {
            uint32_t align_elements = std::max(32U / dtype_size_x, 1U);
            uint64_t ub_tile_cap = kMaxTileLength;
            if (ub_size > 0) {
                uint64_t available_ub = ub_size > 4096 ? ub_size - 4096 : ub_size;
                uint64_t raw_tile_cap = available_ub / (static_cast<uint64_t>(kBufferCount) * dtype_size_x);
                if (raw_tile_cap >= align_elements) {
                    ub_tile_cap = raw_tile_cap / align_elements * align_elements;
                }
            }
            uint64_t capped_tile = std::min<uint64_t>(kMaxTileLength, ub_tile_cap);
            preferred_tile = static_cast<uint32_t>(std::max<uint64_t>(align_elements, capped_tile));
            uint64_t needed_cores = (length_x + kMinElementsPerCore - 1) / kMinElementsPerCore;
            block_dim = std::min(static_cast<uint32_t>(num_cores_aiv),
                                 static_cast<uint32_t>(std::max<uint64_t>(needed_cores, 1)));
            block_length = (length_x + block_dim - 1) / block_dim;
            block_length = (block_length + align_elements - 1) / align_elements * align_elements;
        }

        ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
        tiling->length = length_x;
        tiling->blockLength = block_length;
        tiling->tileLength = preferred_tile;

        context->SetBlockDim(block_dim);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *xShape = context->GetInputShape(0);
        gert::Shape *yShape = context->GetOutputShape(0);
        if (xShape == nullptr || yShape == nullptr) {
            return GRAPH_FAILED;
        }
        *yShape = *xShape;
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

