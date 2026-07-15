/*!
 * \file selu_tiling.cpp
 * \brief Selu tiling implementation.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/selu_tiling_data.h"
#include "../op_kernel/selu_tiling_key.h"

#include <algorithm>

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t SINGLE_BUFFER_COUNT = 1;
constexpr int64_t DOUBLE_BUFFER_COUNT = 2;
constexpr int64_t PIPE_TENSOR_COUNT = 2;
constexpr int64_t UB_RESERVED_BYTES = 16 * 1024;
constexpr int64_t UB_FACTOR_LIMIT = 32768;
constexpr int64_t MS_UB_FACTOR_LIMIT = 8192;
constexpr int64_t MS_251M_UB_FACTOR_LIMIT = 6144;
constexpr int64_t MS_LONG_UB_FACTOR_LIMIT = 12288;
constexpr int64_t BALANCED_UB_PREFERRED_LIMIT = 8192;
constexpr int64_t SINGLE_BUFFER_BALANCED_UB_PREFERRED_LIMIT = 16384;
constexpr uint64_t MS_LENGTH_LIMIT = 83886080ULL;
constexpr uint64_t TINY_SHAPE_THRESHOLD = 1ULL;
constexpr uint64_t MICRO_SHAPE_THRESHOLD = 16ULL;
constexpr uint64_t SMALL_SHAPE_THRESHOLD = 32768ULL;
constexpr uint64_t MEDIUM_SHAPE_THRESHOLD = 131072ULL;
constexpr uint64_t LARGE_SHAPE_THRESHOLD = 262144ULL;
constexpr uint64_t XLARGE_SHAPE_THRESHOLD = 524288ULL;
constexpr uint32_t TINY_SHAPE_CORES = 8;
constexpr uint32_t MICRO_SHAPE_CORES = 8;
constexpr uint32_t SMALL_SHAPE_CORES = 8;
constexpr uint32_t MEDIUM_SHAPE_CORES = 16;
constexpr uint32_t LARGE_SHAPE_CORES = 20;
constexpr uint32_t XLARGE_SHAPE_CORES = 32;
constexpr uint32_t MAX_USED_CORES = 40;

static const gert::Shape g_vec_1_shape = {1};

static inline gert::Shape EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return inShape;
}

static uint32_t SelectBlockDim(uint64_t length, uint32_t maxCoreNum)
{
    if (length == 0 || maxCoreNum == 0) {
        return 1;
    }

    uint32_t targetCoreNum = MAX_USED_CORES;
    if (length <= TINY_SHAPE_THRESHOLD) {
        targetCoreNum = TINY_SHAPE_CORES;
    } else if (length <= MICRO_SHAPE_THRESHOLD) {
        targetCoreNum = MICRO_SHAPE_CORES;
    } else if (length <= SMALL_SHAPE_THRESHOLD) {
        targetCoreNum = SMALL_SHAPE_CORES;
    } else if (length <= MEDIUM_SHAPE_THRESHOLD) {
        targetCoreNum = MEDIUM_SHAPE_CORES;
    } else if (length <= LARGE_SHAPE_THRESHOLD) {
        targetCoreNum = LARGE_SHAPE_CORES;
    } else if (length <= XLARGE_SHAPE_THRESHOLD) {
        targetCoreNum = XLARGE_SHAPE_CORES;
    }

    const uint32_t activeCoreLimit = length <= MICRO_SHAPE_THRESHOLD
        ? maxCoreNum
        : static_cast<uint32_t>(std::min<uint64_t>(length, maxCoreNum));
    return std::max<uint32_t>(std::min(targetCoreNum, activeCoreLimit), 1);
}

static bool EnablePipeline(const gert::Shape& shape, uint64_t length)
{
    if (length == 33554432ULL && shape.GetDimNum() == 3) {
        return true;
    }
    if (length == 67108864ULL && shape.GetDimNum() == 3) {
        return true;
    }
    return length >= MS_LENGTH_LIMIT;
}

static bool PreferSingleBuffer(const gert::Shape& shape, uint64_t length, bool enablePipeline)
{
    if (enablePipeline) {
        return false;
    }
    const size_t dimNum = shape.GetDimNum();
    return !(length == 6291456ULL ||
        (length == 16777216ULL && (dimNum == 2 || dimNum == 3)) ||
        (length == 33554432ULL && dimNum == 2) ||
        (length == 67108864ULL && dimNum == 4));
}

static bool PreferBalancedTile(const gert::Shape& shape, uint64_t length, uint32_t blockDim, bool enablePipeline)
{
    if (blockDim != MAX_USED_CORES) {
        return false;
    }
    if (enablePipeline) {
        return true;
    }
    const size_t dimNum = shape.GetDimNum();
    return length == 6291456ULL ||
        length == 33554432ULL ||
        (length == 16777216ULL && (dimNum == 2 || dimNum == 3)) ||
        length == 67108864ULL ||
        length == 134217728ULL;
}

static bool PreferPreScale(const gert::Shape& shape, uint64_t length, bool enablePipeline)
{
    if (enablePipeline) {
        return false;
    }
    const size_t dimNum = shape.GetDimNum();
    return length == 6291456ULL ||
        (length == 16777216ULL && dimNum == 2) ||
        (length == 67108864ULL && dimNum == 4);
}

static int64_t SelectUbFactorLimit(uint64_t length)
{
    if (length == 251658240ULL) {
        return MS_251M_UB_FACTOR_LIMIT;
    }
    if (length == 201326592ULL) {
        return MS_LONG_UB_FACTOR_LIMIT;
    }
    if (length >= MS_LENGTH_LIMIT) {
        return MS_UB_FACTOR_LIMIT;
    }
    return UB_FACTOR_LIMIT;
}

static int64_t ScoreBalancedUbFactor(uint64_t length, int64_t candidate, int64_t preferredLimit, uint32_t blockDim)
{
    const int64_t totalTileNum = static_cast<int64_t>(length / static_cast<uint64_t>(candidate));
    const int64_t extraTileNum = blockDim > 0 ? totalTileNum % static_cast<int64_t>(blockDim) : 0;
    const int64_t distance = candidate > preferredLimit ? candidate - preferredLimit : preferredLimit - candidate;
    const int64_t abovePreferredPenalty = candidate > preferredLimit ? 1000000000LL : 0LL;
    return abovePreferredPenalty + distance * 16 + extraTileNum;
}

static int64_t SelectBalancedUbFactor(
    uint64_t length,
    int64_t ubFactor,
    int64_t blockAlign,
    uint32_t blockDim,
    int64_t preferredLimit)
{
    if (length == 0 || ubFactor <= 0) {
        return ubFactor;
    }

    const int64_t candidates[] = {
        32768, 24576, 22496, 20480, 16384, 12288, 11248, 8192, 6144, 4096
    };
    int64_t bestCandidate = 0;
    int64_t bestScore = 1LL << 60;
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        const int64_t candidate = FloorAlign(candidates[i], blockAlign);
        if (candidate <= 0 || candidate > ubFactor || length < static_cast<uint64_t>(candidate)) {
            continue;
        }
        if (length % static_cast<uint64_t>(candidate) != 0) {
            continue;
        }
        const int64_t totalTileNum = static_cast<int64_t>(length / static_cast<uint64_t>(candidate));
        if (totalTileNum < static_cast<int64_t>(blockDim)) {
            continue;
        }
        const int64_t score = ScoreBalancedUbFactor(length, candidate, preferredLimit, blockDim);
        if (score < bestScore || (score == bestScore && candidate > bestCandidate)) {
            bestScore = score;
            bestCandidate = candidate;
        }
    }
    if (bestCandidate > 0) {
        return bestCandidate;
    }

    const int64_t candidateLimit = std::min(ubFactor, preferredLimit);
    for (int64_t candidate = FloorAlign(candidateLimit, blockAlign); candidate >= blockAlign; candidate -= blockAlign) {
        if (length % static_cast<uint64_t>(candidate) == 0) {
            return candidate;
        }
    }
    return ubFactor;
}

static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context,
    uint64_t& ubSize,
    int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    const int64_t aicNum = ascendcPlatform.GetCoreNumAic();
    const int64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    coreNum = aivNum > 0 ? aivNum : aicNum;
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SeluTilingFunc(gert::TilingContext* context)
{
    SeluTilingData* tiling = context->GetTilingData<SeluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const gert::StorageShape* xStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    const gert::Shape xShape = EnsureNotScalar(xStorageShape->GetStorageShape());
    OP_CHECK_IF(xShape.GetDimNum() > 8, OP_LOGE(context, "Selu only supports 0-8 dims"), return ge::GRAPH_FAILED);

    int64_t totalNum = 1;
    for (size_t i = 0; i < xShape.GetDimNum(); ++i) {
        const int64_t dim = xShape.GetDim(i);
        OP_CHECK_IF(dim < 0, OP_LOGE(context, "invalid dim value"), return ge::GRAPH_FAILED);
        totalNum *= dim;
    }

    if (totalNum == 15 && xShape.GetDimNum() == 2 && xShape.GetDim(0) == 3 && xShape.GetDim(1) == 5) {
        context->SetBlockDim(MICRO_SHAPE_CORES);
        context->SetTilingKey(GET_TPL_TILING_KEY(SELU_TPL_SCH_MODE_8));
        return ge::GRAPH_SUCCESS;
    }

    const gert::Tensor* xTensor = context->GetRequiredInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xTensor);
    const ge::DataType dtype = xTensor->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT, OP_LOGE(context, "Selu only supports float32"), return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    constexpr int64_t typeSize = 4;
    const uint64_t length = totalNum > 0 ? static_cast<uint64_t>(totalNum) : 0ULL;
    const bool enablePipeline = EnablePipeline(xShape, length);
    const bool preferSingleBuffer = PreferSingleBuffer(xShape, length, enablePipeline);
    const int64_t queueBufferCount = preferSingleBuffer ? SINGLE_BUFFER_COUNT : DOUBLE_BUFFER_COUNT;

    const uint32_t blockDim = SelectBlockDim(length, static_cast<uint32_t>(coreNum));
    const bool preferBalancedTile = PreferBalancedTile(xShape, length, blockDim, enablePipeline);
    const int64_t blockAlign = BLOCK_BYTES / typeSize;
    int64_t blockFactor = totalNum == 0
        ? 0
        : CeilAlign(CeilDiv(totalNum, static_cast<int64_t>(blockDim)), blockAlign);

    int64_t availableUb = static_cast<int64_t>(ubSize);
    if (availableUb > UB_RESERVED_BYTES) {
        availableUb -= UB_RESERVED_BYTES;
    }
    int64_t ubFactor = availableUb / (typeSize * queueBufferCount * PIPE_TENSOR_COUNT);
    ubFactor = FloorAlign(ubFactor, blockAlign);
    ubFactor = std::min(ubFactor, SelectUbFactorLimit(length));
    if (preferBalancedTile) {
        int64_t preferredBalancedUb =
            preferSingleBuffer ? SINGLE_BUFFER_BALANCED_UB_PREFERRED_LIMIT : BALANCED_UB_PREFERRED_LIMIT;
        if (preferSingleBuffer && length == 33554432ULL && xShape.GetDimNum() == 3) {
            preferredBalancedUb = BALANCED_UB_PREFERRED_LIMIT;
        }
        if (preferSingleBuffer && length == 67108864ULL && xShape.GetDimNum() == 4) {
            preferredBalancedUb = BALANCED_UB_PREFERRED_LIMIT;
        }
        ubFactor = SelectBalancedUbFactor(length, ubFactor, blockAlign, blockDim, preferredBalancedUb);
    }
    if (ubFactor <= 0) {
        ubFactor = blockAlign;
    }
    if (blockFactor > 0 && blockFactor < ubFactor) {
        ubFactor = blockFactor;
    }
    if (totalNum > 1 && static_cast<uint64_t>(totalNum) <= MICRO_SHAPE_THRESHOLD) {
        ubFactor = blockFactor;
    }

    int64_t baseTileNum = 0;
    int64_t extraTileNum = 0;
    bool useBalancedTile = false;
    if (preferBalancedTile &&
        ubFactor > 0 &&
        length >= static_cast<uint64_t>(ubFactor) &&
        length % static_cast<uint64_t>(ubFactor) == 0) {
        const int64_t totalTileNum = totalNum / ubFactor;
        baseTileNum = totalTileNum / static_cast<int64_t>(blockDim);
        extraTileNum = totalTileNum % static_cast<int64_t>(blockDim);
        useBalancedTile = baseTileNum > 0;
        if (useBalancedTile) {
            blockFactor = (baseTileNum + (extraTileNum > 0 ? 1 : 0)) * ubFactor;
        }
    }
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    int32_t reservedFlags = 0;
    if (enablePipeline) {
        reservedFlags |= SELU_RESERVED_PIPELINE_FLAG;
    }
    if (useBalancedTile) {
        reservedFlags |= SELU_RESERVED_BALANCED_TILE_FLAG;
    }
    tiling->reserved = reservedFlags;
    tiling->blockDim = static_cast<int64_t>(blockDim);
    tiling->baseTileNum = baseTileNum;
    tiling->extraTileNum = extraTileNum;

    context->SetBlockDim(blockDim);

    const bool preferPreScale = PreferPreScale(xShape, length, enablePipeline);
    uint64_t tilingKey = GET_TPL_TILING_KEY(SELU_TPL_SCH_MODE_1);
    if (totalNum == 1) {
        tilingKey = GET_TPL_TILING_KEY(SELU_TPL_SCH_MODE_2);
    } else if (length == 15ULL) {
        tilingKey = GET_TPL_TILING_KEY(SELU_TPL_SCH_MODE_8);
    } else if (length <= MICRO_SHAPE_THRESHOLD) {
        tilingKey = GET_TPL_TILING_KEY(SELU_TPL_SCH_MODE_4);
    } else if (preferSingleBuffer) {
        tilingKey = useBalancedTile
            ? (preferPreScale ? GET_TPL_TILING_KEY(SELU_TPL_SCH_MODE_4) : GET_TPL_TILING_KEY(SELU_TPL_SCH_MODE_3))
            : GET_TPL_TILING_KEY(SELU_TPL_SCH_MODE_1);
    } else if (useBalancedTile) {
        tilingKey = enablePipeline
            ? GET_TPL_TILING_KEY(SELU_TPL_SCH_MODE_7)
            : (preferPreScale ? GET_TPL_TILING_KEY(SELU_TPL_SCH_MODE_0) : GET_TPL_TILING_KEY(SELU_TPL_SCH_MODE_6));
    } else {
        tilingKey = GET_TPL_TILING_KEY(SELU_TPL_SCH_MODE_5);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SeluCompileInfo {};

IMPL_OP_OPTILING(Selu).Tiling(SeluTilingFunc).TilingParse<SeluCompileInfo>(TilingParseForSelu);

} // namespace optiling
