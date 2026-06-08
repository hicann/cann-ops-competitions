// Host侧Tiling实现
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
        int dtype_size = ge::GetSizeByDataType(dtype_x);
        uint32_t total_length = tensor_x->GetShapeSize();

        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // ── GM 对齐 ──
        // 512B 对齐 → 满带宽无 bank 冲突；32B 对齐 → 保留多核并行
        constexpr uint32_t LARGE_GM_THRESHOLD = 150000u;
        uint32_t align_elem = (total_length > LARGE_GM_THRESHOLD)
            ? 512u / static_cast<uint32_t>(dtype_size)   // 128 for fp32
            : 32u  / static_cast<uint32_t>(dtype_size);  // 8   for fp32

        // ── 零长快速出口 ──
        if (total_length == 0) {
            ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
            tiling->totalLength = 0;
            tiling->perCoreLength = align_elem;
            tiling->tileLength = align_elem;
            context->SetBlockDim(1);
            context->GetWorkspaceSizes(1)[0] = 0;
            return ge::GRAPH_SUCCESS;
        }

        // ── 核数策略 ──
        uint32_t desired_cores;
        if (total_length <= 16u) {
            desired_cores = (total_length + 7u) / 8u;
        } else {
            desired_cores = (total_length + 15u) / 16u;
            if (total_length <= 768u && desired_cores > 16u)
                desired_cores = 16u;
            if (desired_cores > static_cast<uint32_t>(num_cores_aiv))
                desired_cores = static_cast<uint32_t>(num_cores_aiv);
        }
        if (desired_cores < 1u) desired_cores = 1u;

        uint32_t per_core = ((total_length + desired_cores - 1u) / desired_cores
                             + align_elem - 1u) / align_elem * align_elem;
        if (per_core == 0u) per_core = align_elem;

        uint32_t used_cores = (total_length + per_core - 1u) / per_core;
        if (used_cores > static_cast<uint32_t>(num_cores_aiv))
            used_cores = static_cast<uint32_t>(num_cores_aiv);
        if (used_cores == 0u) used_cores = 1u;

        // ── 最小 per_core 保证 ──
        // per_core < 2048 且核数多时，强制降到 8 核，让 per_core 增大
        // 结合 kernel 的 GetBlockNum≤8 绕过 VECCALC，可获得 pipeline 重叠
        if (per_core < 3072u && used_cores > 8u) {
            used_cores = 8u;
            per_core = ((total_length + used_cores - 1u) / used_cores
                        + align_elem - 1u) / align_elem * align_elem;
            if (per_core == 0u) per_core = align_elem;
            used_cores = (total_length + per_core - 1u) / per_core;
            if (used_cores == 0u) used_cores = 1u;
        }

        // ── 分块策略 ──
        // VECCALC: per_core ≤ 10000 且 (核数>8 或 极小微张量≤256)
        // 其他: pipeline/medium fallback → 需要合法 tile
        constexpr uint32_t PIPE_LOWER = 10000u;
        constexpr uint32_t TILE_CAP   = 8192u;
        uint32_t tile_length;

        if (per_core <= PIPE_LOWER && (used_cores > 8u || per_core <= 288u)) {
            tile_length = align_elem;   // VECCALC 路径，tile 不使用
        } else {
            uint32_t quarter = per_core / 8u;
            quarter = (quarter / align_elem) * align_elem;
            if (quarter < align_elem) quarter = align_elem;
            tile_length = (quarter < TILE_CAP) ? quarter : TILE_CAP;
        }

        ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
        tiling->totalLength = total_length;
        tiling->perCoreLength = per_core;
        tiling->tileLength = tile_length;

        context->SetBlockDim(used_cores);
        context->GetWorkspaceSizes(1)[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *x_shape = context->GetInputShape(0);
        gert::Shape *y_shape = context->GetOutputShape(0);
        if (x_shape != nullptr && y_shape != nullptr) {
            *y_shape = *x_shape;
        }
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        auto dtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, dtype);
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
