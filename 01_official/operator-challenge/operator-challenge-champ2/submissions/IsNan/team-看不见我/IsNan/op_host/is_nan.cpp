/* IsNan OpDef + Tiling + InferShape/InferDataType。
 * elementwise：y[i] = isnan(x[i]) ? true : false。x: fp16/bf16/fp32 → y: bool。
 * 不考虑性能，先正确性 + 泛化（任意 shape，非 32 对齐，三 dtype）。 */
#include "is_nan_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

static constexpr uint32_t TILE_LEN     = 8192;   // 每 tile 元素数（UB 容 fp32 工作区，留余量）
static constexpr uint32_t MIN_PER_CORE = 8192u;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    IsNanTilingData tiling;
    auto inS = context->GetInputShape(0)->GetStorageShape();
    uint64_t totalLen = 1;
    for (size_t i = 0; i < inS.GetDimNum(); ++i) totalLen *= (uint64_t)inS.GetDim(i);
    if (totalLen == 0) totalLen = 1;

    // dtype → TilingKey（1=fp32, 2=fp16, 3=bf16）：kernel 按 key 只编译/加载对应分支
    auto dt = context->GetInputDesc(0)->GetDataType();
    uint64_t tilingKey = 1;  // fp32 默认
    if (dt == ge::DT_FLOAT16) tilingKey = 2;
    else if (dt == ge::DT_BF16) tilingKey = 3;

    auto plat = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t maxCore = plat.GetCoreNumAiv();
    if (maxCore == 0) maxCore = 8;
    uint32_t want = (uint32_t)(totalLen / MIN_PER_CORE);
    if (want < 2u) want = 2u;                       // ★ 最少 2 核：双核启停开销 < 单核
    uint32_t coreNum = (maxCore < want) ? maxCore : want;
    if (coreNum > 2u && (coreNum & 1u)) coreNum -= 1u;
    if ((uint64_t)coreNum > totalLen) coreNum = (uint32_t)totalLen;
    if (coreNum == 0) coreNum = 1;

    // TilingData 仅 totalLen + tileLen；分核/tailExtra 由 kernel 用 GetBlockNum() 重算（ALN=32 对齐逻辑同 kernel）。
    tiling.set_totalLen((uint32_t)totalLen);
    tiling.set_tileLen(TILE_LEN);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(coreNum);
    context->SetTilingKey(tilingKey);
    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = static_cast<size_t>(plat.GetLibApiWorkSpaceSize()) + 32u;  // 系统 ws + 32B flip flag
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
    context->SetOutputDataType(0, ge::DT_BOOL);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class IsNan : public OpDef {
public:
    explicit IsNan(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);

        OpAICoreConfig cfg;
        cfg.DynamicCompileStaticFlag(true)
           .DynamicFormatFlag(false)
           .DynamicRankSupportFlag(true)
           .DynamicShapeSupportFlag(true)
           .NeedCheckSupportFlag(false)
           .PrecisionReduceFlag(true)
           .ExtendCfgInfo("opFile.value", "is_nan");
        this->AICore().AddConfig("ascend910b", cfg);
    }
};
OP_ADD(IsNan);
} // namespace ops
