// AscendC Lerp 算子 —— Host 侧 Tiling 逻辑
//
// 负责在 compile-time 确定:
//   - 多少个 AI Core 参与计算 (blockDim)
//   - 每个 Core 分多少数据   (block 级切分)
//   - UB 内一次 tile 处理多少元素 (tile 级切分)
// weight 是标量属性, 由 Host 直接写入 tiling 结构体下发, kernel 侧只读.
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/lerp_tiling.h"
#include "../op_kernel/tiling_key_lerp.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // ============================================================
        // 1. 读取输入张量元信息
        // ============================================================
        const gert::Tensor *startTensor = context->GetRequiredInputTensor(0);
        ge::DataType startDtype = startTensor->GetDataType();
        int32_t dtypeBytes  = ge::GetSizeByDataType(startDtype);
        uint32_t elemCnt    = startTensor->GetShapeSize();

        // ============================================================
        // 2. 读取 weight 标量属性
        // ============================================================
        const gert::RuntimeAttrs *runtimeAttrs = context->GetAttrs();
        const float *rawWeight = runtimeAttrs->GetFloat(0);
        float wt = (rawWeight != nullptr) ? *rawWeight : 0.0f;

        // ============================================================
        // 3. 查平台算力: AIV Core 数量 & UB 总容量
        // ============================================================
        auto plat = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t aivCoreCnt = plat.GetCoreNumAiv();
        uint64_t ubMemCap;
        plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubMemCap);

        // ============================================================
        // 4. 注册 tiling key —— 按 dtype 派发不同 kernel 模板实例
        // ============================================================
        uint32_t DT_START = static_cast<uint32_t>(startDtype);
        ASCENDC_TPL_SEL_PARAM(context, DT_START);

        // ============================================================
        // 5. 对齐粒度 & block 总数
        // AscendC GM→UB 搬运以 32B 对齐最高效; 未对齐的数据用 Pad 指令兜底
        // ============================================================
        constexpr uint32_t CACHELINE = 32;
        if (dtypeBytes <= 0) {
            dtypeBytes = 4;   // fallback: 当 dtype 未知时按 f32 估算
        }
        uint32_t chunkElems = CACHELINE / static_cast<uint32_t>(dtypeBytes);
        if (chunkElems == 0) {
            chunkElems = 1;   // 极端情况: 单个元素 > 32B, 不会发生, 纯防御
        }

        // ceil(elemCnt / chunkElems)
        uint32_t blockTotal  = (elemCnt + chunkElems - 1) / chunkElems;
        if (blockTotal == 0) { blockTotal = 1; }

        // ============================================================
        // 6. 多核负载分配 (block 粒度)
        // 数据太少时减少用核, 避免部分核空跑浪费功耗
        // ============================================================
        uint32_t activeCores = static_cast<uint32_t>(aivCoreCnt > 0 ? aivCoreCnt : 1);
        if (blockTotal < activeCores) {
            activeCores = blockTotal;
        }
        if (activeCores == 0) { activeCores = 1; }

        // ============================================================
        // 7. UB 分块: 算一个 tile 最多容纳多少元素
        //
        // UB 容量分配:
        //   ubAvailable = ubMemCap - 2KB (预留)
        //   预留的 2KB 给栈帧 / spill / 标量等
        //
        // 3 个队列 (start / end / y), 每个 BUFFER_NUM 个 slot,
        // 双缓冲保证 MTE2 与 Vector 可 overlap 执行.
        // ============================================================
        constexpr uint32_t BUF_CNT      = 2;
        constexpr uint32_t UB_HEADROOM  = 2u * 1024u;
        uint32_t perSlotBytes = 3u * BUF_CNT * static_cast<uint32_t>(dtypeBytes);

        uint64_t ubAvailable = (ubMemCap > UB_HEADROOM) ? (ubMemCap - UB_HEADROOM)
                                                        : (ubMemCap / 2);
        uint32_t usableByteCnt = static_cast<uint32_t>(ubAvailable);
        uint32_t tileElems = usableByteCnt / perSlotBytes;
        // 向下取整到 chunkElems 的整数倍, 保证每次搬运都是对齐的
        tileElems = (tileElems / chunkElems) * chunkElems;
        if (tileElems == 0) {
            tileElems = chunkElems;
        }
        if (tileElems > elemCnt && elemCnt > 0) {
            tileElems = ((elemCnt + chunkElems - 1) / chunkElems) * chunkElems;
        }

        // ============================================================
        // 8. 下刷 tiling 结构体并设置 BlockDim
        // ============================================================
        LerpTilingData *tiling = context->GetTilingData<LerpTilingData>();
        tiling->length     = elemCnt;
        tiling->weight     = wt;
        tiling->tileLength = tileElems;
        context->SetBlockDim(activeCores);

        size_t *ws = context->GetWorkspaceSizes(1);
        ws[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

// 形状推导 & 数据类型推导 —— Lerp 是逐元素算子, 输出与输入同 shape / 同 dtype
namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
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
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Lerp);
}  // namespace ops
