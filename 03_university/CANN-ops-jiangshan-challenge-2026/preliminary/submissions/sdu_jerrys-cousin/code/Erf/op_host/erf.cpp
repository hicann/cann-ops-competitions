#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
// 向上对齐函数
static inline uint32_t AlignUp(uint32_t x, uint32_t align) {
    return (x + align - 1) & ~(align - 1);
}

static inline void SetLargeTiling(ErfTilingData *tiling, uint32_t totalLength, uint32_t usedCoreNum,
                                  uint32_t alignNum, uint32_t mode) {
    uint32_t blockLength = AlignUp((totalLength + usedCoreNum - 1) / usedCoreNum, alignNum);
    usedCoreNum = (totalLength + blockLength - 1) / blockLength;
    if (usedCoreNum == 0) usedCoreNum = 1;
    uint32_t tileLength = blockLength > ERF_MAX_TILE_LENGTH ? ERF_MAX_TILE_LENGTH : blockLength;
    if (mode == ERF_MODE_DIRECT_MULTI) {
        tileLength = blockLength;
    }
    tiling->mode = mode;
    tiling->totalLength = totalLength;
    tiling->blockLength = blockLength;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tileLength = tileLength;
    tiling->tailLength = totalLength - blockLength * (usedCoreNum - 1);
    tiling->maxCalcLength = tileLength;
}

// Tiling 函数
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    uint32_t dtX = static_cast<uint32_t>(tensorX->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, dtX);

    uint32_t totalLength = static_cast<uint32_t>(tensorX->GetShapeSize());
    constexpr uint32_t alignNum = 64; // 256 Bytes / 4 Bytes
    constexpr uint32_t copyAlignNum = 8; // 32 Bytes / 4 Bytes

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();

    if (totalLength <= 128) {
        uint32_t blockLength = (totalLength == 0) ? copyAlignNum : AlignUp(totalLength, copyAlignNum);
				tiling->mode = ERF_MODE_SMALL;
        tiling->totalLength = totalLength;
        tiling->blockLength = blockLength;
        tiling->usedCoreNum = 1;
        tiling->tileLength = blockLength;
        tiling->tailLength = totalLength;
        tiling->maxCalcLength = blockLength;
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }
    
    if (totalLength <= ERF_SMALL_THRESHOLD) {
        uint32_t blockLength = (totalLength == 0) ? alignNum : AlignUp(totalLength, alignNum);
				tiling->mode = ERF_MODE_SMALL;
        tiling->totalLength = totalLength;
        tiling->blockLength = blockLength;
        tiling->usedCoreNum = 1;
        tiling->tileLength = blockLength;
        tiling->tailLength = totalLength;
        tiling->maxCalcLength = blockLength;
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (coreNum == 0) coreNum = 1;

    if (totalLength > 65536 && totalLength <= 131072) {
        uint32_t usedCoreNum = coreNum > 20 ? 20 : coreNum;
        SetLargeTiling(tiling, totalLength, usedCoreNum, alignNum, ERF_MODE_DIRECT_MULTI);
        context->SetBlockDim(tiling->usedCoreNum);
        return ge::GRAPH_SUCCESS;
    }

    if (totalLength > 262144 && totalLength <= 327680) {
        uint32_t usedCoreNum = coreNum > 25 ? 25 : coreNum;
        SetLargeTiling(tiling, totalLength, usedCoreNum, alignNum, ERF_MODE_DIRECT_MULTI);
        context->SetBlockDim(tiling->usedCoreNum);
        return ge::GRAPH_SUCCESS;
    }

    // 计算 usedCoreNum, tileLength
    uint32_t usedCoreNum = (totalLength <= ERF_TWO_CORE_THRESHOLD) ? 
                           (coreNum > 1 ? 2 : 1) : coreNum;
    SetLargeTiling(tiling, totalLength, usedCoreNum, alignNum, ERF_MODE_LARGE);
    context->SetBlockDim(tiling->usedCoreNum);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling


// 算子原型注册
namespace ops {
class Erf : public OpDef {
public:
    explicit Erf(const char *name) : OpDef(name) {
        Input("x").DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        Output("y").DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
				SetInferShape([](gert::InferShapeContext *context) {
            *context->GetOutputShape(0) = *context->GetInputShape(0);
            return ge::GRAPH_SUCCESS;
        }).SetInferDataType([](gert::InferDataTypeContext *context) {
            context->SetOutputDataType(0, ge::DT_FLOAT);
            return ge::GRAPH_SUCCESS;
        });
				AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(Erf);
}  // namespace ops
