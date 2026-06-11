// Host侧Tiling实现
#include <algorithm>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/addcmul_tiling.h"
#include "../op_kernel/tiling_key_addcmul.h"

namespace optiling {
    constexpr uint32_t ALIGN_BYTES = 32;
    constexpr uint32_t GM_ALIGN_BYTES = 512;
    constexpr uint32_t UB_RESERVE_BYTES = 16 * 1024;
    constexpr uint32_t BUFFER_NUM = 2;
    constexpr uint32_t QUEUE_NUM = 4;
    constexpr uint32_t TINY_TENSOR_BYTES = 16 * 1024;
    constexpr uint32_t SMALL_TENSOR_BYTES = 64 * 1024;
    constexpr uint32_t MID_TENSOR_BYTES = 256 * 1024;
    constexpr uint32_t MEDIUM_TENSOR_BYTES = 1024 * 1024;
    constexpr uint32_t MAX_TILE_BYTES = 64 * 1024;
    constexpr uint32_t INT8_HALF_TILE_LENGTH = 8 * 1024;

    static uint32_t AlignDown(uint32_t value, uint32_t align) {
        return align == 0 ? value : (value / align) * align;
    }

    static uint32_t AlignUp(uint32_t value, uint32_t align) {
        return align == 0 ? value : ((value + align - 1) / align) * align;
    }

    static uint32_t ShapeSize(const gert::Shape &shape) {
        int64_t size = shape.GetShapeSize();
        if (size <= 0) {
            return 0;
        }
        return static_cast<uint32_t>(size);
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t core_num = static_cast<uint32_t>(platform.GetCoreNumAiv());
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_input_data = context->GetRequiredInputTensor(0);
        const gert::Shape &input_shape = tensor_input_data->GetStorageShape();
        ge::DataType dtype_input_data = tensor_input_data->GetDataType();
        uint32_t dtype_size = static_cast<uint32_t>(ge::GetSizeByDataType(dtype_input_data));
        if (core_num == 0) {
            core_num = 1;
        }
        if (dtype_size == 0) {
            dtype_size = 1;
        }

        uint32_t length = ShapeSize(input_shape);
        uint64_t total_bytes = static_cast<uint64_t>(length) * dtype_size;
        uint32_t align_elems = std::max<uint32_t>(ALIGN_BYTES / dtype_size, 1);
        uint32_t gm_align_elems = std::max<uint32_t>(GM_ALIGN_BYTES / dtype_size, 1);

        uint32_t block_dim = 1;
        if (total_bytes <= TINY_TENSOR_BYTES) {
            block_dim = 2;
        } else if (total_bytes <= SMALL_TENSOR_BYTES) {
            block_dim = 8;
        } else if (total_bytes <= MID_TENSOR_BYTES) {
            block_dim = 16;
        } else if (total_bytes <= MEDIUM_TENSOR_BYTES) {
            block_dim = 32;
        } else {
            block_dim = core_num;
        }
        block_dim = std::min(core_num, std::max<uint32_t>(block_dim, 1));

        uint32_t per_core_length = length;
        if (block_dim > 1) {
            per_core_length = static_cast<uint32_t>((length + block_dim - 1) / block_dim);
            per_core_length = AlignUp(per_core_length, gm_align_elems);
            if (per_core_length == 0) {
                per_core_length = gm_align_elems;
            }
        }

        uint64_t usable_ub = ub_size > UB_RESERVE_BYTES ? ub_size - UB_RESERVE_BYTES : ub_size;
        uint32_t tile_length = static_cast<uint32_t>(usable_ub / (QUEUE_NUM * BUFFER_NUM * dtype_size));
        tile_length = std::min<uint32_t>(tile_length, std::max<uint32_t>(MAX_TILE_BYTES / dtype_size, 1));
        if (dtype_input_data == ge::DT_INT8) {
            tile_length = std::min<uint32_t>(tile_length, INT8_HALF_TILE_LENGTH);
        }
        tile_length = std::min<uint32_t>(tile_length, per_core_length);
        if (total_bytes > SMALL_TENSOR_BYTES) {
            tile_length = AlignDown(tile_length, gm_align_elems);
        } else {
            tile_length = AlignUp(tile_length, align_elems);
        }
        tile_length = AlignDown(tile_length, align_elems);
        if (tile_length == 0) {
            tile_length = align_elems;
        }
        if (tile_length == 0) {
            tile_length = 1;
        }

        uint32_t DT_INPUT_DATA = static_cast<uint32_t>(dtype_input_data);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_DATA);

        AddcmulTilingData *tiling = context->GetTilingData<AddcmulTilingData>();
        if (tiling == nullptr) {
            return ge::GRAPH_FAILED;
        }
        tiling->length = length;
        tiling->tileLength = tile_length;
        tiling->perCoreLength = per_core_length;

        context->SetBlockDim(block_dim);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *input_shape = context->GetInputShape(0);
        gert::Shape *y_shape = context->GetOutputShape(0);
        if (input_shape == nullptr || y_shape == nullptr) {
            return GRAPH_PARAM_INVALID;
        }
        *y_shape = *input_shape;
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class Addcmul : public OpDef {
    public:
        explicit Addcmul(const char *name) : OpDef(name) {
            this->Input("input_data")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("x1")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("x2")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("value")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b")
                .AddConfig("ascend910_93");
        }
    };
    OP_ADD(Addcmul);
}  // namespace ops
