// Host端ClipByValue Tiling: 按UB容量决定tile粒度，按数据量决定启核数
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/clip_by_value_tiling.h"
#include "../op_kernel/tiling_key_clip_by_value.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // --- 拿到tiling结构体，后面往里填参数 ---
        ClipByValueTilingData *tiling = context->GetTilingData<ClipByValueTilingData>();

        // --- 输入张量 ---
        const gert::Tensor *in = context->GetRequiredInputTensor(0);
        uint32_t n_total = in->GetShapeSize();
        ge::DataType dtype = in->GetDataType();
        int dsize = ge::GetSizeByDataType(dtype);

        // --- 算子属性 ---
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const float *v_min = attrs->GetFloat(0);
        const float *v_max = attrs->GetFloat(1);

        // 按dtype选择对应kernel模板实例（编译期多态，避免运行时按类型分支）
        uint32_t DT_X = static_cast<uint32_t>(dtype);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // --- 平台能力 ---
        auto plat = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint64_t ub_sz;
        plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_sz);
        int32_t max_cores = plat.GetCoreNumAiv();

        // 硬件约束: GM地址必须32字节对齐
        const uint32_t ALIGN = 32;
        uint32_t elts_per_blk = ALIGN / static_cast<uint32_t>(dsize);

        // --- 先算tile粒度 (取决于UB，与核数无关) ---
        // DoubleBuffer: VECIN 2块 + VECOUT 2块 = 4块常驻UB
        const uint32_t DBUF = 2;
        const uint32_t NQ = 2;
        uint32_t n_bufs = DBUF * NQ;  // = 4
        uint32_t tile_sz = static_cast<uint32_t>(ub_sz) / n_bufs;
        uint32_t max_tile = (tile_sz / ALIGN) * ALIGN / static_cast<uint32_t>(dsize);
        if (!max_tile) {
            max_tile = elts_per_blk;
        }
        // tile直接拉到UB/4上限: 大tile → 少DMA次数 → 带宽利用率高
        uint32_t tile_len = max_tile;

        // --- 再算核数 (数据太少就降核，多核同步开销不划算) ---
        uint32_t cores = static_cast<uint32_t>(max_cores);
        if (!cores) {
            cores = 1;
        }
        // 每核至少分16KB，该值是实验最优 — 太小则开销大于收益，太大则小张量带宽不够
        const uint32_t MIN_CORE_BYTES = 16384;
        uint64_t total_sz = static_cast<uint64_t>(dsize) * static_cast<uint64_t>(n_total);
        uint32_t need = static_cast<uint32_t>(
            (total_sz + MIN_CORE_BYTES - 1) / MIN_CORE_BYTES);
        if (!need) {
            need = 1;
        }
        if (cores > need) {
            cores = need;
        }

        // --- 填入tiling ---
        tiling->min = *v_min;
        tiling->max = *v_max;
        tiling->totalLength = n_total;
        tiling->tileLength = tile_len;
        // 极小张量(单核+单tile装得下)走kernel快速路径
        tiling->fastPath = (cores == 1 && n_total <= tile_len) ? 1 : 0;

        context->SetBlockDim(cores);

        // workspace: 仅系统空间
        size_t *ws = context->GetWorkspaceSizes(1);
        ws[0] = static_cast<size_t>(plat.GetLibApiWorkSpaceSize());

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *in_shape = context->GetInputShape(0);
        gert::Shape *out_shape = context->GetOutputShape(0);
        *out_shape = *in_shape;
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
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(ClipByValue);
}  // namespace ops
