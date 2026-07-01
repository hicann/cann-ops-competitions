// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {

static constexpr uint32_t DTYPE_BYTES           = 4;     // float32
static constexpr uint32_t ALIGN_ELEM            = 8;     // 32B / 4B = 8 float32
static constexpr uint32_t TOTAL_SLOTS           = 5;     // 4 队列槽 + 1 TBuf（配合 kernel）
static constexpr uint32_t MIN_TILE_ELEM         = 128;   // 双缓冲有效最小 tile（8的倍数）
static constexpr uint32_t SINGLE_CORE_THRESHOLD = 96;  // <= 此值强制单核
static constexpr uint32_t MED_DATA_THRESHOLD    = 128; // <= 此值限制核数
static constexpr uint32_t MED_CORE_LIMIT        = 8;     // 中等数据最大核数

static inline uint32_t AlignUp8(uint32_t n) {
    return (n + ALIGN_ELEM - 1) / ALIGN_ELEM * ALIGN_ELEM;
}
static inline uint32_t AlignDown8(uint32_t n) {
    return (n / ALIGN_ELEM) * ALIGN_ELEM;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    // 1. 平台信息
    auto     platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum  = static_cast<uint32_t>(platform.GetCoreNumAiv());
    uint64_t ubSize   = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    // 2. 输入 tensor
    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    ge::DataType        dtypeX  = tensorX->GetDataType();
    uint32_t            total   = static_cast<uint32_t>(tensorX->GetShapeSize());

    // 3. tiling key（类型分发）
    uint32_t DT_X = static_cast<uint32_t>(dtypeX);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 4. 空 tensor
    if (total == 0) {
        ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
        tiling->totalLength   = 0;
        tiling->tileLength    = MIN_TILE_ELEM;
        tiling->coreNum       = 1;
        tiling->lengthPerCore = 0;
        tiling->tailLength    = 0;
        context->SetBlockDim(1);
        context->GetWorkspaceSizes(1)[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    // 5. 三档核数策略
    uint32_t actualCores;
    if (total <= SINGLE_CORE_THRESHOLD) {
        // 小数据：强制单核，彻底消除多核调度 overhead
        actualCores = 1;
    } else if (total <= MED_DATA_THRESHOLD) {
        // 中等数据：限制核数，平衡并行收益与同步开销
        uint32_t cap = (coreNum < MED_CORE_LIMIT) ? coreNum : MED_CORE_LIMIT;
        uint32_t maxUsable = total / ALIGN_ELEM;
        if (maxUsable == 0) maxUsable = 1;
        actualCores = (cap < maxUsable) ? cap : maxUsable;
    } else {
        // 大数据：用满所有核
        uint32_t maxUsable = total / ALIGN_ELEM;
        if (maxUsable == 0) maxUsable = 1;
        actualCores = (coreNum < maxUsable) ? coreNum : maxUsable;
    }
    if (actualCores == 0) actualCores = 1;

    // 6. 每核元素数（向上对齐到 8）
    uint32_t lengthPerCore = AlignUp8((total + actualCores - 1) / actualCores);

    // 防止 lengthPerCore*(actualCores-1) >= total（导致 tailLength <= 0）
    while (actualCores > 1 &&
           (uint64_t)lengthPerCore * (uint64_t)(actualCores - 1) >= (uint64_t)total) {
        actualCores--;
    }

    uint32_t tailLength = total - lengthPerCore * (actualCores - 1);

    // 7. tileLength：UB 上限决定最大值，MIN_TILE_ELEM 决定最小有效值
    //    TOTAL_SLOTS=5，留 10% UB 余量
    uint32_t ubAvailBytes = static_cast<uint32_t>((double)ubSize * 0.90);
    uint32_t maxTileElem  = AlignDown8(ubAvailBytes / (TOTAL_SLOTS * DTYPE_BYTES));
    if (maxTileElem < ALIGN_ELEM) maxTileElem = ALIGN_ELEM;

    uint32_t maxCoreLen = (tailLength > lengthPerCore) ? tailLength : lengthPerCore;

    uint32_t tileLength;
    if (maxCoreLen <= MIN_TILE_ELEM) {
        // 数据量小到不超过一个 MIN_TILE_ELEM：整个核数据当一个 tile，无循环
        tileLength = AlignUp8(maxCoreLen);
    } else {
        tileLength = (maxTileElem < maxCoreLen) ? maxTileElem : AlignDown8(maxCoreLen);
        if (tileLength < MIN_TILE_ELEM) {
            tileLength = MIN_TILE_ELEM;
        }
    }
    // 最终确保 tileLength 是 8 的整数倍且 >= ALIGN_ELEM
    tileLength = AlignDown8(tileLength);
    if (tileLength < ALIGN_ELEM) tileLength = ALIGN_ELEM;

    // 8. 填充结构体
    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    tiling->totalLength   = total;
    tiling->tileLength    = tileLength;
    tiling->coreNum       = actualCores;
    tiling->lengthPerCore = lengthPerCore;
    tiling->tailLength    = tailLength;

    context->SetBlockDim(actualCores);
    context->GetWorkspaceSizes(1)[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *xShape = context->GetInputShape(0);
        gert::Shape       *yShape = context->GetOutputShape(0);
        *yShape = *xShape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
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