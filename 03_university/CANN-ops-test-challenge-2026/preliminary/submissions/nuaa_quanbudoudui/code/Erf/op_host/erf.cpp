#include <cstdint>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
namespace {
constexpr uint32_t ALIGN_NUM = 8U;
constexpr uint32_t MIN_BASE = 256U;
constexpr uint32_t DIRECT_LIMIT = 8192U;

static inline uint32_t AlignUp8(uint32_t v)
{
    return (v + ALIGN_NUM - 1U) & ~(ALIGN_NUM - 1U);
}

static inline void MakeTargetLoadSchedule(uint32_t totalLength, uint32_t &blockDim, uint32_t &baseLen)
{
    // V129 target-load scheduler: good for tiny/small and stable for large.
    constexpr uint32_t maxCore = 32U;
    uint32_t targetLoad;
    if (totalLength <= 2048U) {
        targetLoad = 1024U;
    } else if (totalLength <= 8192U) {
        targetLoad = 512U;
    } else if (totalLength <= 32768U) {
        targetLoad = 384U;
    } else {
        targetLoad = 256U;
    }

    blockDim = (totalLength + targetLoad - 1U) / targetLoad;
    if (blockDim == 0U) {
        blockDim = 1U;
    }
    if (blockDim > maxCore) {
        blockDim = maxCore;
    }

    baseLen = (totalLength + blockDim - 1U) / blockDim;
    if (baseLen < MIN_BASE) {
        baseLen = MIN_BASE;
    }
    baseLen = AlignUp8(baseLen);
    if (baseLen == 0U) {
        baseLen = ALIGN_NUM;
    }

    blockDim = (totalLength + baseLen - 1U) / baseLen;
    if (blockDim == 0U) {
        blockDim = 1U;
    }
    if (blockDim > maxCore) {
        blockDim = maxCore;
    }
}

static inline void MakeCostModelSchedule(uint32_t totalLength, uint32_t &blockDim, uint32_t &baseLen)
{
    // V130 cost-model scheduler: good medium-band throughput, especially T8/T11-like cases.
    constexpr uint32_t CORE_CAND_COUNT = 8U;
    const uint32_t coreCands[CORE_CAND_COUNT] = {1U, 2U, 4U, 8U, 16U, 24U, 32U, 40U};

    blockDim = 1U;
    baseLen = AlignUp8(totalLength);
    uint64_t bestScore = 0xffffffffffffffffULL;

    for (uint32_t i = 0U; i < CORE_CAND_COUNT; ++i) {
        const uint32_t budget = coreCands[i];
        uint32_t candBase = (totalLength + budget - 1U) / budget;
        if (candBase < MIN_BASE) {
            candBase = MIN_BASE;
        }
        candBase = AlignUp8(candBase);
        if (candBase == 0U) {
            candBase = ALIGN_NUM;
        }

        uint32_t candCores = (totalLength + candBase - 1U) / candBase;
        if (candCores == 0U) {
            candCores = 1U;
        }
        if (candCores > budget) {
            candCores = budget;
        }

        uint64_t score;
        if (candBase <= DIRECT_LIMIT) {
            score = static_cast<uint64_t>(candBase) * 1024ULL +
                    static_cast<uint64_t>(candCores) * 16384ULL;
        } else {
            score = 0x100000000ULL +
                    static_cast<uint64_t>(candBase) * 2048ULL +
                    static_cast<uint64_t>(candCores) * 4096ULL;
        }

        if (score < bestScore) {
            bestScore = score;
            baseLen = candBase;
            blockDim = candCores;
        }
    }
}
} // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const uint32_t totalLength = static_cast<uint32_t>(context->GetInputShape(0)->GetStorageShape().GetShapeSize());
    ASCENDC_TPL_SEL_PARAM(context, ge::DT_FLOAT);

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    tiling->totalLength = totalLength;

    // V131: hybrid scheduler, architecture-level fusion of two validated schedulers.
    // - V129 target-load is best for tiny/small and stable large tensors.
    // - V130 cost-model helps the medium band, especially T8/T11-like cases.
    // This is not a maxCore constant tweak: it is a workload-class scheduler.
    uint32_t blockDim = 1U;
    uint32_t baseLen = AlignUp8(totalLength);

    if (totalLength > 6144U && totalLength <= 32768U) {
        MakeCostModelSchedule(totalLength, blockDim, baseLen);
    } else {
        MakeTargetLoadSchedule(totalLength, blockDim, baseLen);
    }

    context->SetBlockDim(blockDim);
    tiling->blockBaseLength = baseLen;

    if (baseLen <= DIRECT_LIMIT) {
        tiling->mode = 1;
        tiling->tileLength = baseLen;
    } else {
        tiling->mode = 0;
        tiling->tileLength = 8192U;
    }

    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Erf : public OpDef {
public:
    explicit Erf(const char *name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(Erf);
}
