#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace erf_tiling_detail {
constexpr uint32_t kAlignElems = 64;
constexpr uint32_t kMinTileElems = 8;
constexpr uint32_t kReservedUbBytes = 8192;

inline uint32_t FloorToAlign(uint64_t n, uint32_t align) {
    return static_cast<uint32_t>((n / align) * align);
}

inline uint32_t CeilToAlign(uint64_t n, uint32_t align) {
    const uint64_t r = n % align;
    return static_cast<uint32_t>(r == 0 ? n : n + align - r);
}

inline const char *GetEnvOrEmpty(const char *name) {
    const char *v = std::getenv(name);
    return v == nullptr ? "" : v;
}

inline bool NeedTrace() {
    const char *v = std::getenv("OP_AGENT_TILING_TRACE");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

uint32_t SelectTileLength(uint64_t ubBytes, uint64_t elemNum) {
    const uint64_t usableUb = ubBytes > kReservedUbBytes ? ubBytes - kReservedUbBytes : 0;
    const uint32_t ubLimitedElems = FloorToAlign(usableUb / (5U * sizeof(float)), kAlignElems);

    uint32_t result = 0;
    if (elemNum <= ubLimitedElems) {
        result = CeilToAlign(elemNum, kAlignElems);
    } else {
        result = ubLimitedElems;
    }
    return result < kMinTileElems ? kMinTileElems : result;
}

uint32_t SelectBlockDim(uint64_t elemNum, uint32_t tileElems, uint32_t maxVectorCore) {
    (void)tileElems;
    if (elemNum <= 32768ULL) {
        return std::min<uint32_t>(8, maxVectorCore);
    }
    if (elemNum <= 131072ULL) {
        return std::min<uint32_t>(16, maxVectorCore);
    }
    if (elemNum <= 262144ULL) {
        return std::min<uint32_t>(20, maxVectorCore);
    }
    if (elemNum <= 524288ULL) {
        return std::min<uint32_t>(32, maxVectorCore);
    }
    return std::min<uint32_t>(40, maxVectorCore);
}

void ShapeToText(char *buf, size_t cap, int32_t dims, int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
    if (dims <= 1) {
        std::snprintf(buf, cap, "%lld", static_cast<long long>(d0));
        return;
    }
    if (dims == 2) {
        std::snprintf(buf, cap, "%lldx%lld", static_cast<long long>(d0), static_cast<long long>(d1));
        return;
    }
    if (dims == 3) {
        std::snprintf(buf, cap, "%lldx%lldx%lld",
            static_cast<long long>(d0), static_cast<long long>(d1), static_cast<long long>(d2));
        return;
    }
    std::snprintf(buf, cap, "%lldx%lldx%lldx%lld",
        static_cast<long long>(d0), static_cast<long long>(d1),
        static_cast<long long>(d2), static_cast<long long>(d3));
}

void DumpTrace(uint64_t elemNum,
               int32_t dims,
               int64_t d0,
               int64_t d1,
               int64_t d2,
               int64_t d3,
               uint32_t tileElems,
               uint64_t tileCount,
               uint32_t blockDim,
               uint32_t vectorCoreNum,
               uint32_t aicNum,
               uint32_t aivNum,
               uint64_t ubBytes,
               uint64_t perCoreBase,
               uint64_t maxCoreLen,
               uint32_t singleTileFlag) {
    if (!NeedTrace()) {
        return;
    }

    char shape[128] = {0};
    ShapeToText(shape, sizeof(shape), dims, d0, d1, d2, d3);
    const uint64_t estimatedUb = static_cast<uint64_t>(tileElems) * sizeof(float) * 2U;

    std::fprintf(stderr,
        "[TILING_TRACE]{"
        "\"case_id\":\"%s\","
        "\"shape\":\"%s\","
        "\"length\":%llu,"
        "\"tiling_key\":\"DT_X=float32,IS_SINGLE_TILE=%u\","
        "\"path\":\"StageSplitErf\","
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
        GetEnvOrEmpty("CASE_ID"), shape, static_cast<unsigned long long>(elemNum), singleTileFlag,
        singleTileFlag ? "true" : "false", tileElems, static_cast<unsigned long long>(tileCount),
        blockDim, vectorCoreNum, vectorCoreNum, aicNum, aivNum,
        static_cast<unsigned long long>(ubBytes), static_cast<unsigned long long>(ubBytes / 1024ULL),
        static_cast<unsigned long long>(perCoreBase), static_cast<unsigned long long>(maxCoreLen),
        static_cast<unsigned long long>(estimatedUb));
}
}  // namespace erf_tiling_detail

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    using namespace erf_tiling_detail;

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const auto rawAic = platform.GetCoreNumAic();
    const auto rawAiv = platform.GetCoreNumAiv();
    const uint32_t aicNum = rawAic > 0 ? static_cast<uint32_t>(rawAic) : 0U;
    const uint32_t aivNum = rawAiv > 0 ? static_cast<uint32_t>(rawAiv) : 0U;

    uint32_t vectorCoreNum = aivNum > 0U ? aivNum : aicNum;
    if (vectorCoreNum == 0U) {
        vectorCoreNum = 1U;
    }

    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);

    const gert::Tensor *xDesc = context->GetRequiredInputTensor(0);
    if (xDesc == nullptr || xDesc->GetDataType() != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    const int64_t shapeSize = xDesc->GetShapeSize();
    const uint64_t elemNum = shapeSize > 0 ? static_cast<uint64_t>(shapeSize) : 0ULL;

    const gert::StorageShape *xShape = context->GetInputShape(0);
    if (xShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto storage = xShape->GetStorageShape();
    const int32_t dimNum = storage.GetDimNum();
    const int64_t dim0 = dimNum > 0 ? storage.GetDim(0) : 0;
    const int64_t dim1 = dimNum > 1 ? storage.GetDim(1) : 0;
    const int64_t dim2 = dimNum > 2 ? storage.GetDim(2) : 0;
    const int64_t dim3 = dimNum > 3 ? storage.GetDim(3) : 0;

    const uint32_t tileElems = SelectTileLength(ubBytes, elemNum);
    const uint64_t tileCount = tileElems == 0 ? 0ULL : (elemNum + tileElems - 1) / tileElems;
    uint32_t blockDim = SelectBlockDim(elemNum, tileElems, vectorCoreNum);

    const char *envBlockDim = std::getenv("ERF_BLOCK_DIM");
    if (envBlockDim != nullptr) {
        const int parsed = std::atoi(envBlockDim);
        if (parsed > 0) {
            blockDim = static_cast<uint32_t>(parsed);
        }
    }

    const uint64_t alignedElemNum = (elemNum / kAlignElems) * kAlignElems;
    uint32_t tailElems = static_cast<uint32_t>(elemNum - alignedElemNum);
    const uint64_t alignedBlocks = alignedElemNum / kAlignElems;

    uint64_t smallCoreElems = 0;
    uint64_t bigCoreElems = 0;
    uint32_t bigCoreCount = 0;
    uint64_t remainderBlocks = 0;

    if (alignedBlocks == 0) {
        smallCoreElems = 0;
        bigCoreElems = tailElems;
        bigCoreCount = 1;
        tailElems = 0;
    } else {
        const uint64_t blocksPerCore = alignedBlocks / blockDim;
        remainderBlocks = alignedBlocks % blockDim;
        smallCoreElems = blocksPerCore * kAlignElems;
        bigCoreElems = (blocksPerCore + 1) * kAlignElems;
        bigCoreCount = static_cast<uint32_t>(remainderBlocks);
    }

    const uint64_t longestCore = remainderBlocks > 0 ? bigCoreElems : smallCoreElems + tailElems;
    const uint32_t singleTileFlag = longestCore <= tileElems ? 1U : 0U;

    std::fprintf(stderr,
        "[HW_INFO][Erf] opType=Erf aicNum=%u aivNum=%u vectorCoreNum=%u "
        "ubSizeBytes=%llu ubSizeKB=%llu\n",
        aicNum, aivNum, vectorCoreNum,
        static_cast<unsigned long long>(ubBytes), static_cast<unsigned long long>(ubBytes / 1024ULL));
    std::fprintf(stderr,
        "[TILING_INFO][Erf] totalLength=%llu blockDim=%u coreNum=%u alignedLength=%llu tailNum=%u "
        "smallCore=%llu bigCore=%llu tileLength=%u tileNum=%llu tilingKey=DT_X=float32,IS_SINGLE_TILE=%u\n",
        static_cast<unsigned long long>(elemNum), blockDim, vectorCoreNum,
        static_cast<unsigned long long>(alignedElemNum), tailElems,
        static_cast<unsigned long long>(smallCoreElems), static_cast<unsigned long long>(bigCoreElems),
        tileElems, static_cast<unsigned long long>(tileCount), singleTileFlag);

    uint32_t DT_X = static_cast<uint32_t>(xDesc->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X, singleTileFlag);

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->length = elemNum;
    tiling->usedCoreNum = blockDim;
    tiling->tileLength = tileElems;
    tiling->smallCoreDataNum = smallCoreElems;
    tiling->bigCoreDataNum = bigCoreElems;
    tiling->tailBlockNum = bigCoreCount;
    tiling->tailNum = tailElems;

    DumpTrace(elemNum, dimNum, dim0, dim1, dim2, dim3, tileElems, tileCount,
        blockDim, vectorCoreNum, aicNum, aivNum, ubBytes, smallCoreElems, longestCore, singleTileFlag);

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
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(Erf);
}  // namespace ops
