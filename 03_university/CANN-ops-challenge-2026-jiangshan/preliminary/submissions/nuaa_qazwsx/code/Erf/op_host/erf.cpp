#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace {
constexpr uint32_t kDmaAlignElems = 8U;
constexpr uint32_t kMinTileElems = 8U;
constexpr uint64_t kUbReserveBytes = 8192ULL;
constexpr uint32_t kTileBufferCount = 5U;  // in(2) + out(2) + tmp(1), float32 elements

struct PlatformSnapshot {
    uint32_t aic = 0U;
    uint32_t aiv = 0U;
    uint32_t vectorCores = 1U;
    uint64_t ubBytes = 0ULL;
};

struct ShapeSnapshot {
    int32_t rank = 0;
    int64_t dim0 = 0;
    int64_t dim1 = 0;
    int64_t dim2 = 0;
    int64_t dim3 = 0;
};

struct CoreSplitPlan {
    uint64_t alignedLength = 0ULL;
    uint32_t residualElems = 0U;
    uint64_t smallCoreElems = 0ULL;
    uint64_t bigCoreElems = 0ULL;
    uint32_t bigCoreCount = 0U;
    uint64_t longestCoreElems = 0ULL;
};

static uint32_t RoundDownTo(uint64_t value, uint32_t unit) {
    return static_cast<uint32_t>((value / unit) * unit);
}

static uint32_t RoundUpTo(uint64_t value, uint32_t unit) {
    const uint64_t remainder = value % unit;
    return static_cast<uint32_t>(remainder == 0ULL ? value : value + unit - remainder);
}

static const char *GetEnvText(const char *name) {
    const char *value = std::getenv(name);
    return value == nullptr ? "" : value;
}

static bool EnableTilingDump() {
    const char *flag = std::getenv("OP_AGENT_TILING_TRACE");
    return flag != nullptr && flag[0] != '\0' && flag[0] != '0';
}

static PlatformSnapshot QueryPlatform(gert::TilingContext *context) {
    PlatformSnapshot info;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    const auto aicRaw = platform.GetCoreNumAic();
    const auto aivRaw = platform.GetCoreNumAiv();
    info.aic = aicRaw > 0 ? static_cast<uint32_t>(aicRaw) : 0U;
    info.aiv = aivRaw > 0 ? static_cast<uint32_t>(aivRaw) : 0U;
    info.vectorCores = info.aiv != 0U ? info.aiv : info.aic;
    if (info.vectorCores == 0U) {
        info.vectorCores = 1U;
    }

    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, info.ubBytes);
    return info;
}

static uint32_t ChooseTileElems(uint64_t ubBytes, uint64_t totalElems) {
    const uint64_t usableUb = ubBytes > kUbReserveBytes ? ubBytes - kUbReserveBytes : 0ULL;
    const uint64_t rawCapacity = usableUb / (kTileBufferCount * sizeof(float));
    const uint32_t alignedCapacity = RoundDownTo(rawCapacity, kDmaAlignElems);

    uint32_t tileElems = totalElems <= alignedCapacity
        ? RoundUpTo(totalElems, kDmaAlignElems)
        : alignedCapacity;

    if (tileElems < kMinTileElems) {
        tileElems = kMinTileElems;
    }
    return tileElems;
}

static uint32_t ChooseBlockDim(uint64_t length, uint32_t maxVectorCores) {
    uint32_t target = 40U;
    if (length <= 32768ULL) {
        target = 8U;
    } else if (length <= 131072ULL) {
        target = 16U;
    } else if (length <= 262144ULL) {
        target = 20U;
    } else if (length <= 524288ULL) {
        target = 32U;
    }
    return std::min(target, maxVectorCores);
}

static uint32_t ApplyBlockDimOverride(uint32_t selected) {
    const char *raw = std::getenv("ERF_BLOCK_DIM");
    if (raw == nullptr) {
        return selected;
    }
    const int parsed = std::atoi(raw);
    return parsed > 0 ? static_cast<uint32_t>(parsed) : selected;
}

static ShapeSnapshot ReadShapeDigest(gert::TilingContext *context) {
    ShapeSnapshot out;
    const gert::StorageShape *xShape = context->GetInputShape(0);
    if (xShape == nullptr) {
        return out;
    }

    auto storageShape = xShape->GetStorageShape();
    out.rank = storageShape.GetDimNum();
    out.dim0 = out.rank > 0 ? storageShape.GetDim(0) : 0;
    out.dim1 = out.rank > 1 ? storageShape.GetDim(1) : 0;
    out.dim2 = out.rank > 2 ? storageShape.GetDim(2) : 0;
    out.dim3 = out.rank > 3 ? storageShape.GetDim(3) : 0;
    return out;
}

static CoreSplitPlan BuildCoreSplit(uint64_t length, uint32_t blockDim) {
    CoreSplitPlan plan;
    plan.alignedLength = (length / kDmaAlignElems) * kDmaAlignElems;
    plan.residualElems = static_cast<uint32_t>(length - plan.alignedLength);

    const uint64_t alignedBlocks = plan.alignedLength / kDmaAlignElems;
    if (alignedBlocks == 0ULL) {
        plan.smallCoreElems = 0ULL;
        plan.bigCoreElems = plan.residualElems;
        plan.bigCoreCount = 1U;
        plan.residualElems = 0U;
        plan.longestCoreElems = plan.bigCoreElems;
        return plan;
    }

    const uint64_t baseBlocks = alignedBlocks / blockDim;
    const uint64_t extraBlocks = alignedBlocks % blockDim;
    plan.smallCoreElems = baseBlocks * kDmaAlignElems;
    plan.bigCoreElems = (baseBlocks + 1ULL) * kDmaAlignElems;
    plan.bigCoreCount = static_cast<uint32_t>(extraBlocks);
    plan.longestCoreElems = extraBlocks != 0ULL
        ? plan.bigCoreElems
        : plan.smallCoreElems + plan.residualElems;
    return plan;
}

static void ShapeToText(const ShapeSnapshot &shape, char *buffer, size_t bufferBytes) {
    if (shape.rank <= 1) {
        std::snprintf(buffer, bufferBytes, "%lld", static_cast<long long>(shape.dim0));
        return;
    }
    if (shape.rank == 2) {
        std::snprintf(buffer, bufferBytes, "%lldx%lld",
            static_cast<long long>(shape.dim0), static_cast<long long>(shape.dim1));
        return;
    }
    if (shape.rank == 3) {
        std::snprintf(buffer, bufferBytes, "%lldx%lldx%lld",
            static_cast<long long>(shape.dim0), static_cast<long long>(shape.dim1),
            static_cast<long long>(shape.dim2));
        return;
    }
    std::snprintf(buffer, bufferBytes, "%lldx%lldx%lldx%lld",
        static_cast<long long>(shape.dim0), static_cast<long long>(shape.dim1),
        static_cast<long long>(shape.dim2), static_cast<long long>(shape.dim3));
}

static void DumpOptionalTrace(
    uint64_t length,
    const ShapeSnapshot &shape,
    const PlatformSnapshot &platform,
    const CoreSplitPlan &split,
    uint32_t tileElems,
    uint64_t tileCount,
    uint32_t blockDim,
    uint32_t singleTile) {
    if (!EnableTilingDump()) {
        return;
    }

    char shapeText[128] = {0};
    ShapeToText(shape, shapeText, sizeof(shapeText));
    const uint64_t estimatedUbBytes = static_cast<uint64_t>(tileElems) * sizeof(float) * 2ULL;

    std::fprintf(stderr,
        "[TILING_TRACE]{"
        "\"case_id\":\"%s\","
        "\"shape\":\"%s\","
        "\"length\":%llu,"
        "\"tiling_key\":\"DT_X=float32,IS_SINGLE_TILE=%u\","
        "\"path\":\"NaiveSingleCoreOfficialErfApi\","
        "\"is_single_tile\":%s,"
        "\"tile_length\":%u,"
        "\"tile_count\":%llu,"
        "\"used_core_num\":%u,"
        "\"available_core_num\":%u,"
        "\"vector_core_num\":%u,"
        "\"aic_num\":%u,"
        "\"aiv_num\":%u,"
        "\"ub_size_bytes\":%llu,"
        "\"ub_size_kb\":%llu,"
        "\"per_core_length\":%llu,"
        "\"last_core_length\":%llu,"
        "\"estimated_ub_bytes\":%llu,"
        "\"buffer_num\":1"
        "}\n",
        GetEnvText("CASE_ID"),
        shapeText,
        static_cast<unsigned long long>(length),
        singleTile,
        singleTile != 0U ? "true" : "false",
        tileElems,
        static_cast<unsigned long long>(tileCount),
        blockDim,
        platform.vectorCores,
        platform.vectorCores,
        platform.aic,
        platform.aiv,
        static_cast<unsigned long long>(platform.ubBytes),
        static_cast<unsigned long long>(platform.ubBytes / 1024ULL),
        static_cast<unsigned long long>(split.smallCoreElems),
        static_cast<unsigned long long>(split.longestCoreElems),
        static_cast<unsigned long long>(estimatedUbBytes));
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const PlatformSnapshot platform = QueryPlatform(context);

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    if (tensorX == nullptr || tensorX->GetDataType() != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    const int64_t shapeSize = tensorX->GetShapeSize();
    const uint64_t length = shapeSize > 0 ? static_cast<uint64_t>(shapeSize) : 0ULL;

    const gert::StorageShape *xShape = context->GetInputShape(0);
    if (xShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ShapeSnapshot shape = ReadShapeDigest(context);

    const uint32_t tileElems = ChooseTileElems(platform.ubBytes, length);
    const uint64_t tileCount = tileElems == 0U ? 0ULL : (length + tileElems - 1ULL) / tileElems;
    const uint32_t blockDim = ApplyBlockDimOverride(ChooseBlockDim(length, platform.vectorCores));
    const CoreSplitPlan split = BuildCoreSplit(length, blockDim);
    const uint32_t singleTile = split.longestCoreElems <= tileElems ? 1U : 0U;

    std::fprintf(stderr,
        "[HW_INFO][Erf] opType=Erf aicNum=%u aivNum=%u vectorCoreNum=%u "
        "ubSizeBytes=%llu ubSizeKB=%llu\n",
        platform.aic,
        platform.aiv,
        platform.vectorCores,
        static_cast<unsigned long long>(platform.ubBytes),
        static_cast<unsigned long long>(platform.ubBytes / 1024ULL));
    std::fprintf(stderr,
        "[TILING_INFO][Erf] totalLength=%llu blockDim=%u coreNum=%u alignedLength=%llu tailNum=%u "
        "smallCore=%llu bigCore=%llu tileLength=%u tileNum=%llu tilingKey=DT_X=float32,IS_SINGLE_TILE=%u\n",
        static_cast<unsigned long long>(length),
        blockDim,
        platform.vectorCores,
        static_cast<unsigned long long>(split.alignedLength),
        split.residualElems,
        static_cast<unsigned long long>(split.smallCoreElems),
        static_cast<unsigned long long>(split.bigCoreElems),
        tileElems,
        static_cast<unsigned long long>(tileCount),
        singleTile);

    const uint32_t DT_X = static_cast<uint32_t>(tensorX->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X, singleTile);

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->length = length;
    tiling->usedCoreNum = blockDim;
    tiling->tileLength = tileElems;
    tiling->smallCoreDataNum = split.smallCoreElems;
    tiling->bigCoreDataNum = split.bigCoreElems;
    tiling->tailBlockNum = split.bigCoreCount;
    tiling->tailNum = split.residualElems;

    DumpOptionalTrace(length, shape, platform, split, tileElems, tileCount, blockDim, singleTile);

    context->SetBlockDim(blockDim);
    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace != nullptr) {
        workspace[0] = 0;
    }
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
