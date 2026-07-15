/*!
 * \file celu_tiling.cpp
 * \brief Celu tiling implementation.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/celu_tiling_data.h"
#include "../op_kernel/celu_tiling_key.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t SINGLE_BUFFER_COUNT = 1;
constexpr int64_t DOUBLE_BUFFER_COUNT = 2;
constexpr int64_t PIPE_TENSOR_COUNT = 2;
constexpr int64_t UB_RESERVED_BYTES = 16 * 1024;
constexpr int64_t UB_FACTOR_LIMIT = 32768;
constexpr int64_t MS_UB_FACTOR_LIMIT = 6144;
constexpr int64_t MS_ALPHA_NOT_ONE_UB_FACTOR_LIMIT = 4096;
constexpr int64_t MS_201M_ALPHA_NOT_ONE_UB_FACTOR_LIMIT = 8192;
constexpr int64_t MS_251M_ALPHA_ONE_UB_FACTOR_LIMIT = 0;
constexpr int64_t MS_134M_ALPHA_ONE_UB_FACTOR_LIMIT = 6144;
constexpr int64_t BALANCED_UB_PREFERRED_LIMIT = 10240;
constexpr int64_t CASE6_16M_3D_UB_FACTOR_LIMIT = 0;
constexpr int64_t TINY_ALPHA_VECTOR_BLOCK_FACTOR = 16;
constexpr int64_t CASE4_16M_2D_PIPELINE_UB_PREFERRED_LIMIT = 0;
constexpr int64_t CASE3_6M_BALANCED_UB_PREFERRED_LIMIT = 0;
constexpr int64_t CASE12_67M_4D_BALANCED_UB_PREFERRED_LIMIT = 0;
constexpr int64_t CASE12_67M_4D_PIPELINE_UB_PREFERRED_LIMIT = 0;
constexpr int64_t SINGLE_BUFFER_BALANCED_UB_PREFERRED_LIMIT = 16384;
constexpr uint64_t MS_LENGTH_LIMIT = 83886080ULL;
constexpr bool ENABLE_16M_3D_BALANCED_TILE = false;
constexpr bool ENABLE_6M_2D_PIPELINE = true;
constexpr bool ENABLE_16M_2D_SHALLOW_QUEUE = false;
constexpr bool ENABLE_16M_2D_PIPELINE = true;
constexpr bool ENABLE_16M_3D_PIPELINE = false;
constexpr bool ENABLE_32M_2D_PIPELINE = false;
constexpr bool ENABLE_32M_3D_PIPELINE = false;
constexpr bool ENABLE_TINY_ONE_VECTOR_SCALAR = false;
constexpr bool ENABLE_TINY_ALPHA_SCALAR_LOOP = false;
constexpr bool ENABLE_TINY_ALPHA_BLOCK_SCALAR = false;
constexpr bool ENABLE_MS_ALPHA_NOT_ONE_CLAMP_BEFORE_EXP = false;
constexpr bool ENABLE_32M_3D_SINGLE_BUFFER = false;
constexpr bool ENABLE_ALPHA_NOT_ONE_SINGLE_BUFFER = true;
constexpr bool ENABLE_67M_4D_SMALL_UB = true;
constexpr bool ENABLE_67M_4D_PIPELINE = true;
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
constexpr bool ENABLE_TILING_TRACE_BY_DEFAULT = false;
constexpr int32_t TRACE_TARGET_CASE_ID = 0;
constexpr uint64_t TRACE_FALLBACK_MIN_LENGTH = 167772160ULL;
constexpr uint64_t TRACE_FALLBACK_MAX_LENGTH = 335544319ULL;

static const gert::Shape g_vec_1_shape = {1};

static bool TilingTraceEnabled()
{
    if (ENABLE_TILING_TRACE_BY_DEFAULT) {
        return true;
    }
    const char* value = std::getenv("OP_AGENT_TILING_TRACE");
    if (value == nullptr || value[0] == '\0') {
        value = std::getenv("CELU_TILING_TRACE");
    }
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static const char* SafeEnv(const char* name)
{
    const char* value = std::getenv(name);
    return value != nullptr ? value : "";
}

static const char* DTypeName(ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT) {
        return "float32";
    }
    if (dtype == ge::DT_FLOAT16) {
        return "float16";
    }
    return "unknown";
}

static const char* ModePath(uint32_t tilingMode)
{
    if (tilingMode == CELU_TPL_SCH_MODE_0) {
        return "CeluHalfVector";
    }
    if (tilingMode == CELU_TPL_SCH_MODE_2) {
        return "CeluScalarVector";
    }
    if (tilingMode == CELU_TPL_SCH_MODE_3) {
        return "CeluFloatBalancedSingleBuffer";
    }
    if (tilingMode == CELU_TPL_SCH_MODE_4) {
        return "CeluScalarPolyOne";
    }
    if (tilingMode == CELU_TPL_SCH_MODE_5) {
        return "CeluFloatDoubleBuffer";
    }
    if (tilingMode == CELU_TPL_SCH_MODE_6) {
        return "CeluFloatBalancedDoubleBuffer";
    }
    if (tilingMode == CELU_TPL_SCH_MODE_7) {
        return "CeluFloatBalancedPipeline";
    }
    if (tilingMode == CELU_TPL_SCH_MODE_8) {
        return "CeluFloatDoubleBufferShallowQueue";
    }
    if (tilingMode == CELU_TPL_SCH_MODE_9) {
        return "CeluSmallScalarAlpha";
    }
    if (tilingMode == CELU_TPL_SCH_MODE_10) {
        return "CeluFloatBalancedPipelineClampBeforeExp";
    }
    if (tilingMode == CELU_TPL_SCH_MODE_12) {
        return "CeluTinyVectorAlpha";
    }
    return "CeluFloatVector";
}

static int64_t QueueBufferCountForMode(uint32_t tilingMode)
{
    if (tilingMode == CELU_TPL_SCH_MODE_1 ||
        tilingMode == CELU_TPL_SCH_MODE_2 ||
        tilingMode == CELU_TPL_SCH_MODE_3 ||
        tilingMode == CELU_TPL_SCH_MODE_4 ||
        tilingMode == CELU_TPL_SCH_MODE_12) {
        return SINGLE_BUFFER_COUNT;
    }
    return DOUBLE_BUFFER_COUNT;
}

static bool PreferSingleBuffer(ge::DataType dtype, const gert::Shape& shape, uint64_t length, float alpha, bool enablePipeline)
{
    if (dtype != ge::DT_FLOAT || enablePipeline) {
        return false;
    }
    if (length == 33554432ULL && shape.GetDimNum() == 3 && alpha == 1.0f) {
        return ENABLE_32M_3D_SINGLE_BUFFER;
    }
    if (length == 67108864ULL && shape.GetDimNum() == 4 && alpha == 1.0f) {
        return true;
    }
    if (length == 67108864ULL && shape.GetDimNum() == 5 && alpha == 1.0f) {
        return true;
    }
    if (alpha != 1.0f) {
        if (length <= MICRO_SHAPE_THRESHOLD) {
            return true;
        }
        return ENABLE_ALPHA_NOT_ONE_SINGLE_BUFFER;
    }
    return false;
}

static bool EnablePipeline(ge::DataType dtype, uint64_t length, float alpha)
{
    (void)dtype;
    (void)alpha;
    if (length >= MS_LENGTH_LIMIT) {
        return true;
    }
    return false;
}

static bool PreferBalancedTile(
    ge::DataType dtype,
    const gert::Shape& shape,
    uint64_t length,
    float alpha,
    uint32_t blockDim,
    bool enablePipeline)
{
    if (dtype != ge::DT_FLOAT) {
        return false;
    }
    if (blockDim != MAX_USED_CORES) {
        return false;
    }
    if (enablePipeline) {
        return true;
    }
    if (length == 6291456ULL || length == 33554432ULL) {
        return true;
    }
    if (length == 16777216ULL && shape.GetDimNum() == 3) {
        return ENABLE_16M_3D_BALANCED_TILE;
    }
    if (length == 67108864ULL) {
        return alpha != 1.0f || shape.GetDimNum() == 4 || shape.GetDimNum() == 5;
    }
    return false;
}

static void FormatShape(char* shapeText, size_t shapeTextSize, const gert::Shape& shape)
{
    if (shapeTextSize == 0) {
        return;
    }
    int written = std::snprintf(shapeText, shapeTextSize, "%lld", static_cast<long long>(shape.GetDim(0)));
    for (size_t i = 1; i < shape.GetDimNum() && written > 0 && static_cast<size_t>(written) < shapeTextSize; ++i) {
        written += std::snprintf(
            shapeText + written,
            shapeTextSize - static_cast<size_t>(written),
            "x%lld",
            static_cast<long long>(shape.GetDim(i)));
    }
}

static int FormatTilingTrace(
    char* traceText,
    size_t traceTextSize,
    const char* shapeText,
    const char* dtypeName,
    size_t dimNum,
    uint64_t length,
    float alpha,
    uint64_t tilingKey,
    uint32_t tilingMode,
    bool isSingleTile,
    uint32_t blockDim,
    int64_t vectorCoreNum,
    int64_t aicNum,
    int64_t aivNum,
    uint64_t ubSizeBytes,
    uint64_t perCoreLength,
    uint64_t tileLength,
    uint64_t tileCountPerCore,
    uint64_t lastCoreLength,
    uint64_t estimatedUbBytes,
    int64_t queueBufferCount,
    int32_t reservedFlags,
    bool enablePipeline)
{
    return std::snprintf(
        traceText,
        traceTextSize,
        "[TILING_TRACE]{"
        "\"op\":\"Celu\","
        "\"case_id\":\"%s\","
        "\"shape\":\"%s\","
        "\"dim_num\":%zu,"
        "\"dtype\":\"%s\","
        "\"length\":%llu,"
        "\"alpha\":%.9g,"
        "\"tiling_key\":%llu,"
        "\"tiling_mode\":%llu,"
        "\"path\":\"%s\","
        "\"is_single_tile\":%s,"
        "\"block_dim\":%u,"
        "\"used_core_num\":%u,"
        "\"available_core_num\":%lld,"
        "\"vector_core_num\":%lld,"
        "\"aic_num\":%lld,"
        "\"aiv_num\":%lld,"
        "\"ub_size_bytes\":%llu,"
        "\"ub_size_kb\":%llu,"
        "\"block_factor\":%llu,"
        "\"ub_factor\":%llu,"
        "\"tile_count_per_core\":%llu,"
        "\"last_core_length\":%llu,"
        "\"estimated_ub_bytes\":%llu,"
        "\"buffer_num\":%d,"
        "\"pipe_tensor_count\":%lld,"
        "\"reserved_flags\":%d,"
        "\"enable_pipeline\":%s,"
        "\"trace_target_case_id\":%d,"
        "\"trace_fallback_min_length\":%llu,"
        "\"trace_fallback_max_length\":%llu,"
        "\"env_case_name\":\"%s\""
        "}",
        SafeEnv("CASE_ID"),
        shapeText,
        dimNum,
        dtypeName,
        static_cast<unsigned long long>(length),
        alpha,
        static_cast<unsigned long long>(tilingKey),
        static_cast<unsigned long long>(tilingMode),
        ModePath(tilingMode),
        isSingleTile ? "true" : "false",
        blockDim,
        blockDim,
        static_cast<long long>(vectorCoreNum),
        static_cast<long long>(vectorCoreNum),
        static_cast<long long>(aicNum),
        static_cast<long long>(aivNum),
        static_cast<unsigned long long>(ubSizeBytes),
        static_cast<unsigned long long>(ubSizeBytes / 1024ULL),
        static_cast<unsigned long long>(perCoreLength),
        static_cast<unsigned long long>(tileLength),
        static_cast<unsigned long long>(tileCountPerCore),
        static_cast<unsigned long long>(lastCoreLength),
        static_cast<unsigned long long>(estimatedUbBytes),
        static_cast<int>(queueBufferCount),
        static_cast<long long>(PIPE_TENSOR_COUNT),
        reservedFlags,
        enablePipeline ? "true" : "false",
        TRACE_TARGET_CASE_ID,
        static_cast<unsigned long long>(TRACE_FALLBACK_MIN_LENGTH),
        static_cast<unsigned long long>(TRACE_FALLBACK_MAX_LENGTH),
        SafeEnv("CASE_NAME"));
}

static void EmitTilingTraceToFile(FILE* file, const char* traceText)
{
    std::fprintf(file, "%s\n", traceText);
}

static void BuildTilingTraceText(
    char* traceText,
    size_t traceTextSize,
    const gert::Shape& shape,
    ge::DataType dtype,
    int64_t totalNum,
    int64_t blockFactor,
    int64_t ubFactor,
    float alpha,
    uint32_t blockDim,
    int64_t vectorCoreNum,
    int64_t aicNum,
    int64_t aivNum,
    uint64_t ubSizeBytes,
    uint64_t tilingKey,
    uint32_t tilingMode,
    int32_t reservedFlags)
{
    char shapeText[160] = {0};
    FormatShape(shapeText, sizeof(shapeText), shape);

    const uint64_t length = totalNum > 0 ? static_cast<uint64_t>(totalNum) : 0ULL;
    const uint64_t tileLength = ubFactor > 0 ? static_cast<uint64_t>(ubFactor) : 0ULL;
    const uint64_t perCoreLength = blockFactor > 0 ? static_cast<uint64_t>(blockFactor) : 0ULL;
    const uint64_t lastCoreStart = blockDim > 0 ? static_cast<uint64_t>(blockDim - 1) * perCoreLength : 0ULL;
    const uint64_t lastCoreLength = lastCoreStart < length ? length - lastCoreStart : 0ULL;
    const uint64_t tileCountPerCore =
        (tileLength > 0 && perCoreLength > 0) ? (perCoreLength + tileLength - 1) / tileLength : 0ULL;
    const uint64_t typeSize = dtype == ge::DT_FLOAT16 ? 2ULL : 4ULL;
    const int64_t queueBufferCount = QueueBufferCountForMode(tilingMode);
    const uint64_t estimatedUbBytes =
        tileLength * typeSize * static_cast<uint64_t>(queueBufferCount) * PIPE_TENSOR_COUNT;
    const bool isSingleTile = perCoreLength <= tileLength;
    const bool enablePipeline = (reservedFlags & CELU_RESERVED_PIPELINE_FLAG) != 0;
    FormatTilingTrace(
        traceText,
        traceTextSize,
        shapeText,
        DTypeName(dtype),
        static_cast<size_t>(shape.GetDimNum()),
        length,
        alpha,
        tilingKey,
        tilingMode,
        isSingleTile,
        blockDim,
        vectorCoreNum,
        aicNum,
        aivNum,
        ubSizeBytes,
        perCoreLength,
        tileLength,
        tileCountPerCore,
        lastCoreLength,
        estimatedUbBytes,
        queueBufferCount,
        reservedFlags,
        enablePipeline);
}

static void EmitTilingTrace(
    const gert::Shape& shape,
    ge::DataType dtype,
    int64_t totalNum,
    int64_t blockFactor,
    int64_t ubFactor,
    float alpha,
    uint32_t blockDim,
    int64_t vectorCoreNum,
    int64_t aicNum,
    int64_t aivNum,
    uint64_t ubSizeBytes,
    uint64_t tilingKey,
    uint32_t tilingMode,
    int32_t reservedFlags)
{
    if (!TilingTraceEnabled()) {
        return;
    }

    char traceText[2048] = {0};
    BuildTilingTraceText(
        traceText,
        sizeof(traceText),
        shape,
        dtype,
        totalNum,
        blockFactor,
        ubFactor,
        alpha,
        blockDim,
        vectorCoreNum,
        aicNum,
        aivNum,
        ubSizeBytes,
        tilingKey,
        tilingMode,
        reservedFlags);

    EmitTilingTraceToFile(stderr, traceText);
    EmitTilingTraceToFile(stdout, traceText);
}

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

static int64_t SelectUbFactorLimit(const gert::Shape& shape, uint64_t length, float alpha)
{
    if (length == 16777216ULL &&
        shape.GetDimNum() == 3 &&
        alpha == 1.0f &&
        CASE6_16M_3D_UB_FACTOR_LIMIT > 0) {
        return CASE6_16M_3D_UB_FACTOR_LIMIT;
    }
    if (length >= MS_LENGTH_LIMIT) {
        if (alpha != 1.0f) {
            if (length == 201326592ULL && MS_201M_ALPHA_NOT_ONE_UB_FACTOR_LIMIT > 0) {
                return MS_201M_ALPHA_NOT_ONE_UB_FACTOR_LIMIT;
            }
            return MS_ALPHA_NOT_ONE_UB_FACTOR_LIMIT;
        }
        if (length == 251658240ULL && MS_251M_ALPHA_ONE_UB_FACTOR_LIMIT > 0) {
            return MS_251M_ALPHA_ONE_UB_FACTOR_LIMIT;
        }
        if (length == 134217728ULL) {
            return MS_134M_ALPHA_ONE_UB_FACTOR_LIMIT;
        }
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
        32768, 24576, 22496, 20480, 16384, 12288, 11248, 8192, 6144, 5120, 4096, 3072, 2048
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
    int64_t& coreNum,
    int64_t& aicNum,
    int64_t& aivNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    aicNum = ascendcPlatform.GetCoreNumAic();
    aivNum = ascendcPlatform.GetCoreNumAiv();
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

static ge::graphStatus CeluTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    int64_t aicNum = 0;
    int64_t aivNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum, aicNum, aivNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    CeluTilingData* tiling = context->GetTilingData<CeluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const gert::StorageShape* xStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    const gert::Shape xShape = EnsureNotScalar(xStorageShape->GetStorageShape());
    OP_CHECK_IF(xShape.GetDimNum() > 8, OP_LOGE(context, "Celu only supports 0-8 dims"), return ge::GRAPH_FAILED);

    int64_t totalNum = 1;
    for (size_t i = 0; i < xShape.GetDimNum(); ++i) {
        const int64_t dim = xShape.GetDim(i);
        OP_CHECK_IF(dim < 0, OP_LOGE(context, "invalid dim value"), return ge::GRAPH_FAILED);
        totalNum *= dim;
    }

    const gert::Tensor* xTensor = context->GetRequiredInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xTensor);
    const ge::DataType dtype = xTensor->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16,
                OP_LOGE(context, "Celu only supports float32/float16"), return ge::GRAPH_FAILED);
    const int64_t typeSize = dtype == ge::DT_FLOAT16 ? 2 : 4;

    float alpha = 1.0f;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* alphaAttr = attrs->GetFloat(0);
        if (alphaAttr != nullptr) {
            alpha = *alphaAttr;
        }
    }
    OP_CHECK_IF(alpha <= 0.0f, OP_LOGE(context, "alpha must be positive"), return ge::GRAPH_FAILED);

    const uint64_t length = totalNum > 0 ? static_cast<uint64_t>(totalNum) : 0ULL;
    bool enablePipeline = EnablePipeline(dtype, length, alpha);
    if (!enablePipeline &&
        ENABLE_6M_2D_PIPELINE &&
        dtype == ge::DT_FLOAT &&
        length == 6291456ULL &&
        alpha == 1.0f) {
        enablePipeline = true;
    }
    if (!enablePipeline &&
        ENABLE_16M_2D_PIPELINE &&
        dtype == ge::DT_FLOAT &&
        length == 16777216ULL &&
        xShape.GetDimNum() == 2 &&
        alpha == 1.0f) {
        enablePipeline = true;
    }
    if (!enablePipeline &&
        ENABLE_16M_3D_PIPELINE &&
        dtype == ge::DT_FLOAT &&
        length == 16777216ULL &&
        xShape.GetDimNum() == 3 &&
        alpha == 1.0f) {
        enablePipeline = true;
    }
    if (!enablePipeline &&
        ENABLE_32M_3D_PIPELINE &&
        dtype == ge::DT_FLOAT &&
        length == 33554432ULL &&
        xShape.GetDimNum() == 3 &&
        alpha == 1.0f) {
        enablePipeline = true;
    }
    if (!enablePipeline &&
        ENABLE_32M_2D_PIPELINE &&
        dtype == ge::DT_FLOAT &&
        length == 33554432ULL &&
        xShape.GetDimNum() == 2 &&
        alpha == 1.0f) {
        enablePipeline = true;
    }
    if (!enablePipeline &&
        ENABLE_67M_4D_PIPELINE &&
        dtype == ge::DT_FLOAT &&
        length == 67108864ULL &&
        xShape.GetDimNum() == 4 &&
        alpha == 1.0f) {
        enablePipeline = true;
    }
    const bool preferSingleBuffer = PreferSingleBuffer(dtype, xShape, length, alpha, enablePipeline);
    const int64_t queueBufferCount = preferSingleBuffer ? SINGLE_BUFFER_COUNT : DOUBLE_BUFFER_COUNT;

    const uint32_t blockDim = SelectBlockDim(length, static_cast<uint32_t>(coreNum));
    const bool preferBalancedTile = PreferBalancedTile(dtype, xShape, length, alpha, blockDim, enablePipeline);
    const int64_t blockAlign = BLOCK_BYTES / typeSize;
    int64_t blockFactor = totalNum == 0
        ? 0
        : CeilAlign(CeilDiv(totalNum, static_cast<int64_t>(blockDim)), blockAlign);
    if (TINY_ALPHA_VECTOR_BLOCK_FACTOR > 0 &&
        totalNum > 1 &&
        length <= MICRO_SHAPE_THRESHOLD &&
        alpha != 1.0f) {
        blockFactor = TINY_ALPHA_VECTOR_BLOCK_FACTOR;
    }

    int64_t availableUb = static_cast<int64_t>(ubSize);
    if (availableUb > UB_RESERVED_BYTES) {
        availableUb -= UB_RESERVED_BYTES;
    }
    int64_t ubFactor = availableUb / (typeSize * queueBufferCount * PIPE_TENSOR_COUNT);
    ubFactor = FloorAlign(ubFactor, blockAlign);
    ubFactor = std::min(ubFactor, SelectUbFactorLimit(xShape, length, alpha));
    if (preferBalancedTile) {
        int64_t preferredBalancedUb =
            preferSingleBuffer ? SINGLE_BUFFER_BALANCED_UB_PREFERRED_LIMIT : BALANCED_UB_PREFERRED_LIMIT;
        if (!preferSingleBuffer &&
            CASE4_16M_2D_PIPELINE_UB_PREFERRED_LIMIT > 0 &&
            enablePipeline &&
            length == 16777216ULL &&
            xShape.GetDimNum() == 2 &&
            alpha == 1.0f) {
            preferredBalancedUb = CASE4_16M_2D_PIPELINE_UB_PREFERRED_LIMIT;
        }
        if (!preferSingleBuffer &&
            CASE12_67M_4D_PIPELINE_UB_PREFERRED_LIMIT > 0 &&
            enablePipeline &&
            length == 67108864ULL &&
            xShape.GetDimNum() == 4 &&
            alpha == 1.0f) {
            preferredBalancedUb = CASE12_67M_4D_PIPELINE_UB_PREFERRED_LIMIT;
        }
        if (!preferSingleBuffer &&
            CASE3_6M_BALANCED_UB_PREFERRED_LIMIT > 0 &&
            length == 6291456ULL &&
            alpha == 1.0f) {
            preferredBalancedUb = CASE3_6M_BALANCED_UB_PREFERRED_LIMIT;
        }
        if (preferSingleBuffer && length == 33554432ULL && xShape.GetDimNum() == 3 && alpha == 1.0f) {
            preferredBalancedUb = BALANCED_UB_PREFERRED_LIMIT;
        }
        if (preferSingleBuffer &&
            length == 67108864ULL &&
            xShape.GetDimNum() == 4 &&
            alpha == 1.0f &&
            CASE12_67M_4D_BALANCED_UB_PREFERRED_LIMIT > 0) {
            preferredBalancedUb = CASE12_67M_4D_BALANCED_UB_PREFERRED_LIMIT;
        } else if (preferSingleBuffer &&
            length == 67108864ULL &&
            xShape.GetDimNum() == 4 &&
            alpha == 1.0f &&
            ENABLE_67M_4D_SMALL_UB) {
            preferredBalancedUb = BALANCED_UB_PREFERRED_LIMIT;
        }
        ubFactor = SelectBalancedUbFactor(length, ubFactor, blockAlign, blockDim, preferredBalancedUb);
    }
    if (ubFactor <= 0) {
        ubFactor = BLOCK_BYTES / typeSize;
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
    tiling->alpha = alpha;
    int32_t reservedFlags = 0;
    if (enablePipeline) {
        reservedFlags |= CELU_RESERVED_PIPELINE_FLAG;
    }
    if (useBalancedTile) {
        reservedFlags |= CELU_RESERVED_BALANCED_TILE_FLAG;
    }
    tiling->reserved = reservedFlags;
    tiling->blockDim = static_cast<int64_t>(blockDim);
    tiling->baseTileNum = baseTileNum;
    tiling->extraTileNum = extraTileNum;

    context->SetBlockDim(blockDim);

    uint32_t tilingMode = CELU_TPL_SCH_MODE_1;
    uint64_t tilingKey = GET_TPL_TILING_KEY(CELU_TPL_SCH_MODE_1);
    if (dtype == ge::DT_FLOAT16) {
        tilingMode = CELU_TPL_SCH_MODE_0;
        tilingKey = GET_TPL_TILING_KEY(CELU_TPL_SCH_MODE_0);
    } else if (totalNum == 1) {
        if (ENABLE_TINY_ONE_VECTOR_SCALAR && alpha == 1.0f) {
            tilingMode = CELU_TPL_SCH_MODE_2;
            tilingKey = GET_TPL_TILING_KEY(CELU_TPL_SCH_MODE_2);
        } else if (alpha == 1.0f) {
            tilingMode = CELU_TPL_SCH_MODE_4;
            tilingKey = GET_TPL_TILING_KEY(CELU_TPL_SCH_MODE_4);
        } else {
            tilingMode = CELU_TPL_SCH_MODE_2;
            tilingKey = GET_TPL_TILING_KEY(CELU_TPL_SCH_MODE_2);
        }
    } else if (dtype == ge::DT_FLOAT && totalNum <= 16) {
        tilingMode = CELU_TPL_SCH_MODE_12;
        tilingKey = GET_TPL_TILING_KEY(CELU_TPL_SCH_MODE_12);
    } else if (ENABLE_TINY_ALPHA_BLOCK_SCALAR && length <= MICRO_SHAPE_THRESHOLD && alpha != 1.0f) {
        tilingMode = CELU_TPL_SCH_MODE_11;
        tilingKey = GET_TPL_TILING_KEY(CELU_TPL_SCH_MODE_11);
    } else if (ENABLE_TINY_ALPHA_SCALAR_LOOP && length <= MICRO_SHAPE_THRESHOLD && alpha != 1.0f) {
        tilingMode = CELU_TPL_SCH_MODE_9;
        tilingKey = GET_TPL_TILING_KEY(CELU_TPL_SCH_MODE_9);
    } else if (preferSingleBuffer) {
        if (useBalancedTile) {
            tilingMode = CELU_TPL_SCH_MODE_3;
            tilingKey = GET_TPL_TILING_KEY(CELU_TPL_SCH_MODE_3);
        } else {
            tilingMode = CELU_TPL_SCH_MODE_1;
            tilingKey = GET_TPL_TILING_KEY(CELU_TPL_SCH_MODE_1);
        }
    } else {
        if (useBalancedTile) {
            if (enablePipeline) {
                if (ENABLE_MS_ALPHA_NOT_ONE_CLAMP_BEFORE_EXP && alpha != 1.0f) {
                    tilingMode = CELU_TPL_SCH_MODE_10;
                    tilingKey = GET_TPL_TILING_KEY(CELU_TPL_SCH_MODE_10);
                } else {
                    tilingMode = CELU_TPL_SCH_MODE_7;
                    tilingKey = GET_TPL_TILING_KEY(CELU_TPL_SCH_MODE_7);
                }
            } else {
                tilingMode = CELU_TPL_SCH_MODE_6;
                tilingKey = GET_TPL_TILING_KEY(CELU_TPL_SCH_MODE_6);
            }
        } else {
            if (ENABLE_16M_2D_SHALLOW_QUEUE &&
                length == 16777216ULL &&
                xShape.GetDimNum() == 2 &&
                alpha == 1.0f) {
                tilingMode = CELU_TPL_SCH_MODE_8;
                tilingKey = GET_TPL_TILING_KEY(CELU_TPL_SCH_MODE_8);
            } else {
                tilingMode = CELU_TPL_SCH_MODE_5;
                tilingKey = GET_TPL_TILING_KEY(CELU_TPL_SCH_MODE_5);
            }
        }
    }
    context->SetTilingKey(tilingKey);

    EmitTilingTrace(
        xShape,
        dtype,
        totalNum,
        blockFactor,
        ubFactor,
        alpha,
        blockDim,
        coreNum,
        aicNum,
        aivNum,
        ubSize,
        tilingKey,
        tilingMode,
        tiling->reserved);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct CeluCompileInfo {};

IMPL_OP_OPTILING(Celu).Tiling(CeluTilingFunc).TilingParse<CeluCompileInfo>(TilingParseForCelu);

} // namespace optiling
