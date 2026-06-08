// Host侧Tiling实现 - best2 + large tensor tile distribution fields
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
    static constexpr uint32_t ALIGN_NUM = 8;       // float32: 8 elements = 32B
    static constexpr uint32_t VEC_ALIGN_NUM = 64;  // float32: 64 elements = 256B

    static inline uint32_t AlignUp(uint32_t x, uint32_t align) {
        return (x + align - 1) / align * align;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();

        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t length_x = tensor_x->GetShapeSize();

        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        uint32_t block_dim = 1;
        if (length_x <= 4096) {
            block_dim = 1;
        } else if (length_x <= 8192) {
            block_dim = 2;
        } else if (length_x <= 24576) {
            block_dim = 4;
        } else {
            constexpr uint32_t ELEMS_PER_CORE = 1536;
            block_dim = (length_x + ELEMS_PER_CORE - 1) / ELEMS_PER_CORE;
        }

        if (block_dim < 1) {
            block_dim = 1;
        }

        if (block_dim > static_cast<uint32_t>(num_cores_aiv)) {
            block_dim = static_cast<uint32_t>(num_cores_aiv);
        }

        // 加入 CURRENT_BEST 的大 tensor 64 元素对齐策略
        uint32_t block_align = ALIGN_NUM;
        if (length_x > 16384) {
            block_align = VEC_ALIGN_NUM;
        }

        const uint32_t block_length =
            AlignUp((length_x + block_dim - 1) / block_dim, block_align);

        // 安全恢复 8192 direct：
        // block_length <= 8192 时，让 tile_length 至少等于 block_length，
        // KernelErfSingle 只使用 3 个 UB buffer，可以安全一次处理完整 block。
        // block_length > 8192 时，仍使用 normal path 的保守 tile，
        // 避免 2 个输入队列 + 2 个输出队列 + 1 个临时 buffer 占用过多 UB。
        constexpr uint32_t DIRECT_TILE_LIMIT = 8192;

        uint32_t normal_max_tile = 4096;
        if (ub_size >= 176 * 1024) {
            normal_max_tile = 6144;
        }

        uint32_t tile_length = 0;
        if (block_length <= DIRECT_TILE_LIMIT) {
            tile_length = AlignUp(block_length, ALIGN_NUM);
        } else {
            tile_length = AlignUp(normal_max_tile, ALIGN_NUM);
        }

        if (tile_length < ALIGN_NUM) {
            tile_length = ALIGN_NUM;
        }

        ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
        tiling->length = length_x;
        tiling->blockLength = block_length;
        tiling->tileLength = tile_length;

        // 加入 CURRENT_BEST 的大 tensor tile 均衡分配需要的字段
        tiling->tileNum = (length_x + tile_length - 1) / tile_length;
        tiling->blockDim = block_dim;

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
        const auto inputDataType = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDataType);
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

            this->SetInferShape(ge::InferShape)
                .SetInferDataType(ge::InferDataType);

            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };

    OP_ADD(Erf);
}  // namespace ops