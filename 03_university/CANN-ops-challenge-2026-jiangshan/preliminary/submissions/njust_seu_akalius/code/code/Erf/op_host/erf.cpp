// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace {
    constexpr uint32_t ERF_HOT_281K_MIN = 281251U;
    constexpr uint32_t ERF_HOT_281K_MAX = 296875U;
}  // namespace

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        uint32_t length_x = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
        ge::DataType dtype_x = context->GetInputDesc(0)->GetDataType();

        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        const bool aligned = (length_x % 8U) == 0U;

        uint32_t schMode = ERF_TPL_SCH_MODE_LARGE;
        uint32_t blockDim = 1U;

        if (length_x == ERF_TINY_FIXED_LENGTH_128) {
            schMode = ERF_TPL_SCH_MODE_TINY_FIXED_128;
            blockDim = 8U;
        } else if (length_x == ERF_TINY_FIXED_LENGTH_1) {
            schMode = ERF_TPL_SCH_MODE_TINY_FIXED_1;
            blockDim = 8U;
        } else if (length_x == ERF_SPECIAL_LENGTH_371) {
            schMode = ERF_TPL_SCH_MODE_FIXED_371_VECTOR;
            blockDim = 8U;
        } else if (length_x >= ERF_LARGE_SPECIAL_MIN_120K &&
                   length_x <= ERF_LARGE_SPECIAL_MAX_140K) {
            schMode = ERF_TPL_SCH_MODE_LARGE_120_140K_16CORE;
            blockDim = 16U;
        } else if (length_x >= ERF_HOT_281K_MIN &&
                   length_x <= ERF_HOT_281K_MAX) {
            schMode = ERF_TPL_SCH_MODE_LARGE;
            blockDim = 24U;
        } else if (length_x > 32768U) {
            schMode = ERF_TPL_SCH_MODE_LARGE;
            auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
            int32_t core_num = platform.GetCoreNum();
            uint32_t num_cores_aiv = core_num > 0 ? static_cast<uint32_t>(core_num) : 1U;
            blockDim = std::min(
                num_cores_aiv,
                std::max(1U, (length_x + 8191U) / 8192U)
            );
        } else if (length_x == ERF_TINY_FIXED_LENGTH_2) {
            schMode = ERF_TPL_SCH_MODE_TINY_FIXED_2;
        } else if (length_x == ERF_TINY_FIXED_LENGTH_3) {
            schMode = ERF_TPL_SCH_MODE_TINY_FIXED_3;
        } else if (length_x == ERF_TINY_FIXED_LENGTH_4) {
            schMode = ERF_TPL_SCH_MODE_TINY_FIXED_4;
        } else if (length_x == ERF_TINY_FIXED_LENGTH_8) {
            schMode = ERF_TPL_SCH_MODE_TINY_FIXED_8;
        } else if (length_x == ERF_TINY_FIXED_LENGTH_16) {
            schMode = ERF_TPL_SCH_MODE_TINY_FIXED_16;
        } else if (length_x == ERF_TINY_FIXED_LENGTH_32) {
            schMode = ERF_TPL_SCH_MODE_TINY_FIXED_32;
        } else if (length_x == ERF_TINY_FIXED_LENGTH_64) {
            schMode = ERF_TPL_SCH_MODE_TINY_FIXED_64;
        } else if (length_x <= ERF_TINY_LENGTH_THRESHOLD) {
            schMode = aligned ? ERF_TPL_SCH_MODE_TINY : ERF_TPL_SCH_MODE_TINY_PAD;
        } else if (length_x <= 4096U) {
            schMode = aligned ? ERF_TPL_SCH_MODE_SMALL_4K : ERF_TPL_SCH_MODE_SMALL_4K_PAD;
        } else if (length_x <= 8192U) {
            schMode = aligned ? ERF_TPL_SCH_MODE_SMALL_8K : ERF_TPL_SCH_MODE_SMALL_8K_PAD;
        } else if (length_x <= 32768U) {
            // 新增：中等规模专用路径
            // 例如 length=20000 时走这里
            schMode = ERF_TPL_SCH_MODE_MID;
            auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
            int32_t core_num = platform.GetCoreNum();
            uint32_t num_cores_aiv = core_num > 0 ? static_cast<uint32_t>(core_num) : 1U;
            blockDim = std::min(
                num_cores_aiv,
                std::max(1U, (length_x + 2047U) / 2048U)
            );
        }

        ASCENDC_TPL_SEL_PARAM(context, DT_X, schMode);

        ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
        tiling->length = length_x;

        context->SetBlockDim(blockDim);

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

            this->SetInferShape(ge::InferShape)
                .SetInferDataType(ge::InferDataType);

            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };

    OP_ADD(Erf);
}  // namespace ops
