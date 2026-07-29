/* Erfinv OpDef + Tiling + InferShape/InferDataType。
 * y = erfinv(x), x∈[-1,1]。x/y 同 dtype（fp16/bf16/fp32）。elementwise。
 * 首版正确性 + 泛化（任意 shape 分块、非 32 对齐、三 dtype）。 */
#include "erfinv_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
// tileLen 受 UB(192KB) 限制，按 dtype + kernel 分支结构分档（VECCALC buf 恒 fp32 4L+256pad）：
//   fp32 单分支单缓冲：4 buf(X/W/WW/P1) + qIn/qOut(4L) → UB=24L+1024 → 6400（已非 overhead-bound，无需更长）。
//   fp16 单分支单缓冲：4 buf + qIn/qOut(2L) → UB=20L+1024 → 9472（fp16 依赖长 tile；3-buf 合并破坏 bank 错开反退化）。
//   bf16 两分支单缓冲：5 buf+Sel + qIn/qOut(2L) → UB=24.125L+1312 → 7680。
static constexpr uint32_t TILE_LEN_FP32 = 6400;
static constexpr uint32_t TILE_LEN_FP16 = 9472;
static constexpr uint32_t TILE_LEN_BF16 = 7680;
static constexpr uint32_t MIN_PER_CORE  = 8192u;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ErfinvTilingData tiling;
    auto inS = context->GetInputShape(0)->GetStorageShape();
    uint64_t totalLen = 1;
    for (size_t i = 0; i < inS.GetDimNum(); ++i) totalLen *= (uint64_t)inS.GetDim(i);
    if (totalLen == 0) totalLen = 1;

    auto dt = context->GetInputDesc(0)->GetDataType();
    uint32_t dtypeCode = 2;
    if (dt == ge::DT_FLOAT16) dtypeCode = 0;
    else if (dt == ge::DT_BF16) dtypeCode = 1;

    auto plat = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = plat.GetCoreNumAiv();
    if (coreNum == 0) coreNum = 8;
    uint32_t want = (uint32_t)(totalLen / MIN_PER_CORE);
    if (want < 1u) want = 1u;
    if (coreNum > want) coreNum = want;
    // ★ 核数恒为偶数且 ≥2（哪怕最小 case）：偶核利于 910B 双 AIV 调度，避免单核拖尾。
    if (coreNum & 1u) coreNum -= 1u;
    if (coreNum < 2u) coreNum = 2u;

    // ★ 核长按 ALN=16 元素(=32B/fp16，覆盖 fp32 的 8 与 fp16/bf16 的 16)对齐，使各核 tile 全 32B 对齐 →
    //   CopyIn 可用 DataCopy（仅末核零头用 DataCopyPad），CopyOut 一律 DataCopy 向上对齐。
    //   末核额外吃 tailExtra(<ALN) 零头；其向上对齐写只溢出到 totalLen 之外的分配余量（无跨核竞争）。
    const uint64_t ALN = 16;
    uint64_t nBlk = totalLen / ALN;                 // 完整 16-元素块数
    uint32_t tailExtra = (uint32_t)(totalLen % ALN);// 末尾零头 0..15
    uint64_t perBlk = nBlk / coreNum;
    uint32_t remBlk = (uint32_t)(nBlk % coreNum);   // 前 remBlk 个核各多 1 块
    uint32_t tileLen = (dtypeCode == 2) ? TILE_LEN_FP32
                     : (dtypeCode == 0) ? TILE_LEN_FP16 : TILE_LEN_BF16;
    tiling.set_totalLen((uint32_t)totalLen);
    tiling.set_tileLen(tileLen);
    tiling.set_bigCoreNum(remBlk);
    tiling.set_bigCoreLen((uint32_t)((perBlk + 1) * ALN));
    tiling.set_smallCoreLen((uint32_t)(perBlk * ALN));
    tiling.set_coreNum(coreNum);
    tiling.set_dtypeCode(dtypeCode);
    tiling.set_tailExtra(tailExtra);
    // ★ workspace 翻转仅在 shape 够大时启用：小 shape 每核 ≤1 tile，翻转是 no-op 却仍读写 workGm，纯开销。
    //   阈值 1<<20 元素 —— 此规模下每核才有多 tile，相邻调用反向遍历才能复用上次驻留 L2 的尾部。
    tiling.set_useFlip(totalLen >= (1u << 20) ? 1u : 0u);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(coreNum);
    size_t* ws = context->GetWorkspaceSizes(1);
    // 系统 workspace + 32B flip flag（核0 自增，相邻调用正/逆序交替复用 L2）
    ws[0] = static_cast<size_t>(plat.GetLibApiWorkSpaceSize()) + 32u;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x = context->GetInputShape(0);
    gert::Shape* y = context->GetOutputShape(0);
    if (x == nullptr || y == nullptr) return GRAPH_FAILED;
    *y = *x;
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Erfinv : public OpDef {
public:
    explicit Erfinv(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);

        OpAICoreConfig cfg;
        cfg.DynamicCompileStaticFlag(true).DynamicFormatFlag(false)
           .DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true)
           .NeedCheckSupportFlag(false).PrecisionReduceFlag(true)
           .ExtendCfgInfo("opFile.value", "erfinv");
        this->AICore().AddConfig("ascend910b", cfg);
    }
};
OP_ADD(Erfinv);
} // namespace ops
