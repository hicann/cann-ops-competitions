// Host侧Tiling实现
#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/clip_by_value_tiling.h"
#include "../op_kernel/tiling_key_clip_by_value.h"

namespace optiling {
    constexpr uint32_t ALIGN_BYTES = 32;
    constexpr uint32_t GM_ALIGN_BYTES = 512;
    constexpr uint32_t UB_RESERVE_BYTES = 16 * 1024;
    constexpr uint32_t BUFFER_NUM = 2;
    constexpr uint32_t QUEUE_NUM = 2;
    constexpr uint32_t SMALL_TENSOR_BYTES = 64 * 1024;
    constexpr uint32_t MEDIUM_TENSOR_BYTES = 256 * 1024;
    constexpr uint32_t SMALL_BYTES_PER_CORE = 4* 1024;
    constexpr uint32_t MEDIUM_BYTES_PER_CORE = 16* 1024;
    constexpr uint32_t LARGE_BYTES_PER_CORE = 32* 1024;
    constexpr uint32_t DOUBLE_BUFFER_BYTES = 32 * 1024;
    constexpr uint64_t BYTES_64_MB = 64ULL * 1024 * 1024;
    constexpr uint32_t TILING_UINT16_MAX = 65535;

    static uint32_t AlignDown(uint32_t value, uint32_t align) {
        if (align == 0) {
            return value;
        }
        return (value / align) * align;
    }

    static uint32_t AlignUp(uint32_t value, uint32_t align) {
        if (align == 0) {
            return value;
        }
        return ((value + align - 1) / align) * align;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t core_num = static_cast<uint32_t>(platform.GetCoreNumAiv());
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t dtype_size_x = static_cast<uint32_t>(ge::GetSizeByDataType(dtype_x));
        uint32_t length_x = static_cast<uint32_t>(tensor_x->GetShapeSize());

        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const float *attr_min = attrs->GetFloat(0);
        const float *attr_max = attrs->GetFloat(1);

        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        if (core_num == 0) {
            core_num = 1;
        }
        if (dtype_size_x == 0) {
            dtype_size_x = 1;
        }

        uint64_t total_bytes = static_cast<uint64_t>(length_x) * dtype_size_x;
        uint32_t block_dim = 1;
        if (length_x > 0) {
            uint32_t bytes_per_core = LARGE_BYTES_PER_CORE;
            if (total_bytes < SMALL_TENSOR_BYTES) {
                bytes_per_core = SMALL_BYTES_PER_CORE;
            } else if (total_bytes < MEDIUM_TENSOR_BYTES) {
                bytes_per_core = MEDIUM_BYTES_PER_CORE;
            }
            block_dim = static_cast<uint32_t>((total_bytes + bytes_per_core - 1) / bytes_per_core);
            block_dim = std::min(core_num, block_dim);
            block_dim = std::max<uint32_t>(block_dim, 1);
            if (total_bytes >= BYTES_64_MB) {
                block_dim = std::min<uint32_t>(block_dim, 32);
            }
        }

        uint32_t align_elems = std::max<uint32_t>(ALIGN_BYTES / dtype_size_x, 1);
        uint32_t gm_align_elems = std::max<uint32_t>(GM_ALIGN_BYTES / dtype_size_x, 1);
        uint32_t per_core_length = length_x;
        if (block_dim > 1) {
            per_core_length = static_cast<uint32_t>((length_x + block_dim - 1) / block_dim);
            per_core_length = AlignUp(per_core_length, gm_align_elems);
            if (per_core_length == 0) {
                per_core_length = gm_align_elems;
            }
        }
        uint32_t per_core_blocks = block_dim > 1 ? per_core_length / gm_align_elems : 0;
        if (per_core_blocks > TILING_UINT16_MAX) {
            block_dim = 1;
            per_core_length = length_x;
            per_core_blocks = 0;
        }

        uint64_t usable_ub = ub_size > UB_RESERVE_BYTES ? ub_size - UB_RESERVE_BYTES : ub_size;
        uint32_t tile_length = static_cast<uint32_t>(usable_ub / (QUEUE_NUM * BUFFER_NUM * dtype_size_x));
        uint64_t per_core_bytes = static_cast<uint64_t>(per_core_length) * dtype_size_x;
        if (per_core_bytes >= DOUBLE_BUFFER_BYTES) {
            tile_length = std::min<uint32_t>(tile_length, static_cast<uint32_t>(per_core_length / BUFFER_NUM));
            tile_length = AlignDown(tile_length, gm_align_elems);
        } else {
            tile_length = std::min<uint32_t>(tile_length, per_core_length);
            tile_length = AlignUp(tile_length, align_elems);
        }
        if (total_bytes >= BYTES_64_MB) {
            uint32_t large_tile_length = std::max<uint32_t>((16 * 1024) / dtype_size_x, 1);
            tile_length = std::min<uint32_t>(tile_length, large_tile_length);
        }
        tile_length = AlignDown(tile_length, align_elems);
        if (tile_length == 0) {
            tile_length = align_elems;
        }
        if (tile_length == 0) {
            tile_length = 1;
        }

        uint32_t tile_blocks = tile_length / align_elems;
        if (tile_blocks == 0) {
            tile_blocks = 1;
        }
        if (tile_blocks > TILING_UINT16_MAX) {
            tile_blocks = TILING_UINT16_MAX;
        }

        ClipByValueTilingData *tiling = context->GetTilingData<ClipByValueTilingData>();
        tiling->length = length_x;
        tiling->packedTileCore = (static_cast<uint32_t>(static_cast<uint16_t>(per_core_blocks)) << 16) |
            static_cast<uint32_t>(static_cast<uint16_t>(tile_blocks));
        tiling->minVal = attr_min == nullptr ? 0.0f : *attr_min;
        tiling->maxVal = attr_max == nullptr ? 0.0f : *attr_max;

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
    class ClipByValue : public OpDef {
    public:
        explicit ClipByValue(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("min").AttrType(REQUIRED).Float();
            this->Attr("max").AttrType(REQUIRED).Float();
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b")
                .AddConfig("ascend910_93");
        }
    };
    OP_ADD(ClipByValue);
}  // namespace ops
