// Host侧Tiling实现
#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/lerp_tiling.h"
#include "../op_kernel/tiling_key_lerp.h"

namespace optiling {
    constexpr uint32_t ALIGN_BYTES = 32;
    constexpr uint32_t GM_ALIGN_BYTES = 512;
    constexpr uint32_t TINY_BYTES = 1024;
    constexpr uint32_t SMALL_BYTES = 8 * 1024;
    constexpr uint32_t MID_BYTES = 32 * 1024;
    constexpr uint32_t LARGE_TILE_BYTES = 32 * 1024;

    static uint32_t AlignUp(uint32_t value, uint32_t align) {
        return align == 0 ? value : ((value + align - 1) / align) * align;
    }

    static uint32_t CeilDiv(uint32_t value, uint32_t divisor) {
        return divisor == 0 ? value : (value + divisor - 1) / divisor;
    }

    static uint32_t SelectBlockDim(uint64_t total_bytes, uint32_t core_num) {
        uint32_t block_dim = 1;
        if (total_bytes <= TINY_BYTES) {
            block_dim = 1;
        } else if (total_bytes <= 4*1024) {
            block_dim = 2;
        }else if (total_bytes <= SMALL_BYTES) {
            block_dim = 4;
        } else if (total_bytes <= MID_BYTES) {
            block_dim = 8;
        } else if (total_bytes <= 256*1024) {
            block_dim = 16;
        }
        else return 1;
        return block_dim;
    }

    static ge::graphStatus SetTiling(gert::TilingContext *context,
                                     uint32_t length,
                                     uint32_t tile_length,
                                     uint32_t per_core_length,
                                     uint32_t block_dim,
                                     float weight) {
        LerpTilingData *tiling = context->GetTilingData<LerpTilingData>();
        tiling->length = length;
        tiling->tileLength = tile_length;
        tiling->perCoreLength = per_core_length;
        tiling->weight = weight;

        context->SetBlockDim(block_dim);
        size_t *current_workspace = context->GetWorkspaceSizes(1);
        current_workspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        const gert::Tensor *tensor_start = context->GetRequiredInputTensor(0);
        ge::DataType dtype_start = tensor_start->GetDataType();
        uint32_t dtype_size_start = static_cast<uint32_t>(ge::GetSizeByDataType(dtype_start));
        if (dtype_size_start == 0) {
            dtype_size_start = 1;
        }
        uint32_t length_start = static_cast<uint32_t>(tensor_start->GetShapeSize());

        uint32_t DT_START = static_cast<uint32_t>(dtype_start);
        ASCENDC_TPL_SEL_PARAM(context, DT_START);

        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const float *attr_weight = attrs->GetFloat(0);
        float weight = attr_weight == nullptr ? 0.0f : *attr_weight;

        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t core_num = static_cast<uint32_t>(platform.GetCoreNumAiv());

        uint32_t align_elems = std::max<uint32_t>(ALIGN_BYTES / dtype_size_start, 1U);
        uint32_t gm_align_elems = std::max<uint32_t>(GM_ALIGN_BYTES / dtype_size_start, 1U);
        uint64_t total_bytes = static_cast<uint64_t>(length_start) * dtype_size_start;

        // if (length_start == 0) {
        //     return SetTiling(context, length_start, align_elems, align_elems, 1, weight);
        // }

        uint32_t block_dim = SelectBlockDim(total_bytes, core_num);
        uint32_t per_core_length = length_start;
        if (block_dim > 1) {
            per_core_length = AlignUp(CeilDiv(length_start, block_dim), gm_align_elems);
        }
        per_core_length = AlignUp(per_core_length, align_elems);

        uint32_t tile_length = per_core_length;
        if (total_bytes > MID_BYTES) {
            tile_length = std::max<uint32_t>(LARGE_TILE_BYTES / dtype_size_start, align_elems);
            tile_length = std::min<uint32_t>(tile_length, per_core_length);
            tile_length = AlignUp(tile_length, align_elems);
        }

        return SetTiling(context, length_start, tile_length, per_core_length, block_dim, weight);
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *start_shape = context->GetInputShape(0);
        gert::Shape *y_shape = context->GetOutputShape(0);
        *y_shape = *start_shape;
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
                .AddConfig("ascend910b")
                .AddConfig("ascend910_93");
        }
    };
    OP_ADD(Lerp);
}  // namespace ops
