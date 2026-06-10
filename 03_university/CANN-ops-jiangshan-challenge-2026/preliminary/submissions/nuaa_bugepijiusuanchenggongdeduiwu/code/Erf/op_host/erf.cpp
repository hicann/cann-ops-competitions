// Host tiling implementation
// op_host/erf.cpp

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

#include <cstdint>

namespace {

constexpr uint32_t COPY_ALIGN_BYTES = 32;
constexpr uint32_t VECTOR_REPEAT_BYTES = 256;
constexpr uint32_t UB_RESERVED_BYTES = 4096;

constexpr uint32_t DTYPE_SIZE = sizeof(float);
constexpr uint32_t COPY_ALIGN_ELEMS = COPY_ALIGN_BYTES / DTYPE_SIZE;        // 8 float32
constexpr uint32_t FLOAT32_REPEAT_ELEMS = VECTOR_REPEAT_BYTES / DTYPE_SIZE; // 64 float32

static_assert(COPY_ALIGN_ELEMS == 8U, "float32 32B alignment should be 8 elements.");
static_assert(FLOAT32_REPEAT_ELEMS == 64U, "float32 vector repeat should be 64 elements.");
static_assert((FLOAT32_REPEAT_ELEMS % COPY_ALIGN_ELEMS) == 0U,
              "repeat elements must be multiple of copy alignment elements.");

constexpr uint32_t LOCAL_BUFFER_COUNT = 3;

// kernel 侧每个 gap 可能使用 32B 或 64B。
// 三个 buffer: x | pad | y | pad | p，所以总共两个 gap。
// host 侧按最坏情况 64B + 64B 预留。
constexpr uint32_t LOCAL_BUFFER_GAP_PAD_BYTES_MAX = 2U * COPY_ALIGN_BYTES; // 64B
constexpr uint32_t LOCAL_BUFFER_PAD_BYTES =
    2U * LOCAL_BUFFER_GAP_PAD_BYTES_MAX; // 128B

#ifndef ERF_SPLIT_MIN_REPEAT_GROUPS_PER_CORE
#define ERF_SPLIT_MIN_REPEAT_GROUPS_PER_CORE 4
#endif

// Fine-grained tiny exact no-read policy. Bit k controls whether length == k
// uses its exact no-read scalar kernel. Default: all tiny lengths 1..31 are
// written as exact tiling-key paths. length == 0 still falls back to runtime tiny.
#ifndef ERF_TINY_EXACT_LEN_MASK
#define ERF_TINY_EXACT_LEN_MASK 0x0000FFFEU
#endif

constexpr uint32_t CORE_TARGET_ALIGN_UNITS = 32U; // large: target about 256 floats/core

#ifndef ERF_ENABLE_LOCAL_PADE_RANGE
#define ERF_ENABLE_LOCAL_PADE_RANGE 0
#endif

#ifndef ERF_LOCAL_PADE_MIN_LEN
#define ERF_LOCAL_PADE_MIN_LEN 16U
#endif

#ifndef ERF_LOCAL_PADE_MAX_LEN
#define ERF_LOCAL_PADE_MAX_LEN 128U
#endif

#ifndef ERF_LOCAL_PADE_ELEMS_PER_CORE
#define ERF_LOCAL_PADE_ELEMS_PER_CORE 16U
#endif

// Bucketed medium-direct path.  This is a 3UB one-shot path for medium sizes.
// It is intentionally not a DataCopyPad/fixed-chunk path: each core only reads and
// writes its real 32B-aligned segment, with the last 1..7 elements kept scalar.
#ifndef ERF_ENABLE_MEDIUM_BUCKET_DIRECT
#define ERF_ENABLE_MEDIUM_BUCKET_DIRECT 0
#endif

#ifndef ERF_ENABLE_LARGE_SPECIAL_CORE
#define ERF_ENABLE_LARGE_SPECIAL_CORE 1
#endif

#ifndef ERF_LARGE_SPECIAL_MIN_120K
#define ERF_LARGE_SPECIAL_MIN_120K (120U * 1024U)
#endif

#ifndef ERF_LARGE_SPECIAL_MAX_140K
#define ERF_LARGE_SPECIAL_MAX_140K (140U * 1024U)
#endif

#ifndef ERF_LARGE_SPECIAL_CORE_120_140K
#define ERF_LARGE_SPECIAL_CORE_120_140K 24U
#endif

#ifndef ERF_LARGE_SPECIAL_MIN_281K
#define ERF_LARGE_SPECIAL_MIN_281K (281U * 1024U)
#endif

#ifndef ERF_LARGE_SPECIAL_MAX_296K
#define ERF_LARGE_SPECIAL_MAX_296K (296U * 1024U)
#endif

#ifndef ERF_LARGE_SPECIAL_CORE_281_296K
#define ERF_LARGE_SPECIAL_CORE_281_296K 24U
#endif

#ifndef ERF_MEDIUM_BUCKET_MIN_LENGTH
#define ERF_MEDIUM_BUCKET_MIN_LENGTH 1025U
#endif

#ifndef ERF_MEDIUM_BUCKET_MAX_LENGTH
#define ERF_MEDIUM_BUCKET_MAX_LENGTH 8192U
#endif

#ifndef ERF_MEDIUM_BUCKET_SPLIT_LEN
#define ERF_MEDIUM_BUCKET_SPLIT_LEN 4096U
#endif

#ifndef ERF_MEDIUM_BUCKET_TARGET_LOW
#define ERF_MEDIUM_BUCKET_TARGET_LOW 768U
#endif

#ifndef ERF_MEDIUM_BUCKET_TARGET_HIGH
#define ERF_MEDIUM_BUCKET_TARGET_HIGH 512U
#endif

constexpr uint32_t MEDIUM_BUCKET_ALIGN_ELEMS = 256U;
constexpr uint32_t MEDIUM_BUCKET_MAX_ELEMS = 1792U;
constexpr uint32_t DIRECT_BUFFER_COUNT = 3U;
constexpr uint32_t TILED_BUFFER_COUNT = 3U;

static uint32_t AlignDown(uint32_t value, uint32_t align)
{
    if (align == 0U) {
        return value;
    }
    return (value / align) * align;
}

static uint32_t AlignUp(uint32_t value, uint32_t align)
{
    if (align == 0U) {
        return value;
    }
    return ((value + align - 1U) / align) * align;
}

static uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    if (b == 0U) {
        return 0U;
    }
    return (a + b - 1U) / b;
}

static uint32_t MinU32(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

static uint32_t MaxU32(uint32_t a, uint32_t b)
{
    return a > b ? a : b;
}

static uint32_t NormalizeCoreNum(uint32_t coreNum, uint32_t numCoresAiv)
{
    if (numCoresAiv == 0U) {
        numCoresAiv = 1U;
    }
    if (coreNum == 0U) {
        coreNum = 1U;
    }
    if (coreNum > numCoresAiv) {
        coreNum = numCoresAiv;
    }
    return coreNum == 0U ? 1U : coreNum;
}

static uint32_t ClampCoreNumByAlignedLength(uint32_t coreNum, uint32_t alignedLength)
{
    uint32_t alignedBlocks = alignedLength / COPY_ALIGN_ELEMS;
    if (alignedBlocks == 0U) {
        return 1U;
    }
    if (coreNum == 0U) {
        coreNum = 1U;
    }
    if (coreNum > alignedBlocks) {
        coreNum = alignedBlocks;
    }
    return coreNum == 0U ? 1U : coreNum;
}

static void BuildAlignedCoreSplit(
    uint32_t alignedLength,
    uint32_t activeCores,
    uint32_t &formerNum,
    uint32_t &formerLength,
    uint32_t &tailLength)
{
    formerNum = 0U;
    formerLength = 0U;
    tailLength = 0U;

    uint32_t totalBlocks = alignedLength / COPY_ALIGN_ELEMS;
    if (totalBlocks == 0U || activeCores == 0U) {
        return;
    }

    if (activeCores > totalBlocks) {
        activeCores = totalBlocks;
    }

    uint32_t baseBlocks = totalBlocks / activeCores;
    uint32_t extraBlocks = totalBlocks - baseBlocks * activeCores;

    formerNum = extraBlocks;
tailLength = baseBlocks * COPY_ALIGN_ELEMS;
formerLength = (extraBlocks > 0U)
                   ? ((baseBlocks + 1U) * COPY_ALIGN_ELEMS)
                   : tailLength;
}

static uint32_t ChooseRepeatGroupCores(uint32_t length, uint32_t numCoresAiv)
{
    uint32_t fullGroups = length / FLOAT32_REPEAT_ELEMS;
    if (fullGroups < static_cast<uint32_t>(ERF_SPLIT_MIN_REPEAT_GROUPS_PER_CORE)) {
        return 1U;
    }

    uint32_t coreNum = fullGroups / static_cast<uint32_t>(ERF_SPLIT_MIN_REPEAT_GROUPS_PER_CORE);
    return NormalizeCoreNum(coreNum, numCoresAiv);
}

static uint32_t CalcUbLimitElems(uint64_t ubSize, uint32_t bufferCount, uint32_t extraPadBytes)
{
    if (bufferCount == 0U) {
        return 0U;
    }

    uint64_t needReserve = static_cast<uint64_t>(UB_RESERVED_BYTES) +
                           static_cast<uint64_t>(extraPadBytes);
    if (ubSize <= needReserve) {
        return 0U;
    }

    uint64_t usableUb = ubSize - needReserve;
    uint64_t rawLimit = usableUb / (static_cast<uint64_t>(bufferCount) * DTYPE_SIZE);
    if (rawLimit > 0xFFFFFFFFULL) {
        rawLimit = 0xFFFFFFFFULL;
    }

    return AlignDown(static_cast<uint32_t>(rawLimit), COPY_ALIGN_ELEMS);
}

static uint32_t RoundMediumBucket(uint32_t elems)
{
    if (elems == 0U) {
        return 0U;
    }
    uint32_t bucket = AlignUp(elems, MEDIUM_BUCKET_ALIGN_ELEMS);
    if (bucket < MEDIUM_BUCKET_ALIGN_ELEMS) {
        bucket = MEDIUM_BUCKET_ALIGN_ELEMS;
    }
    if (bucket > MEDIUM_BUCKET_MAX_ELEMS) {
        return 0U;
    }
    return bucket;
}

static uint32_t GetMediumBucketMode(uint32_t bucket)
{
    switch (bucket) {
        case 256U: return ERF_MODE_MEDIUM_BUCKET_256;
        case 512U: return ERF_MODE_MEDIUM_BUCKET_512;
        case 768U: return ERF_MODE_MEDIUM_BUCKET_768;
        case 1024U: return ERF_MODE_MEDIUM_BUCKET_1024;
        case 1280U: return ERF_MODE_MEDIUM_BUCKET_1280;
        case 1536U: return ERF_MODE_MEDIUM_BUCKET_1536;
        case 1792U: return ERF_MODE_MEDIUM_BUCKET_1792;
        default: return ERF_MODE_GENERIC;
    }
}

static bool UseTinyExactMode(uint32_t length)
{
    if (length == 0U || length >= 32U) {
        return false;
    }
    return (((static_cast<uint32_t>(ERF_TINY_EXACT_LEN_MASK) >> length) & 1U) != 0U);
}

static uint32_t GetTinyExactMode(uint32_t length)
{
    if (!UseTinyExactMode(length)) {
        return ERF_MODE_TINY_1_31;
    }

    switch (length) {
        case 1U: return ERF_MODE_TINY_LEN_1;
        case 2U: return ERF_MODE_TINY_LEN_2;
        case 3U: return ERF_MODE_TINY_LEN_3;
        case 4U: return ERF_MODE_TINY_LEN_4;
        case 5U: return ERF_MODE_TINY_LEN_5;
        case 6U: return ERF_MODE_TINY_LEN_6;
        case 7U: return ERF_MODE_TINY_LEN_7;
        case 8U: return ERF_MODE_TINY_LEN_8;
        case 9U: return ERF_MODE_TINY_LEN_9;
        case 10U: return ERF_MODE_TINY_LEN_10;
        case 11U: return ERF_MODE_TINY_LEN_11;
        case 12U: return ERF_MODE_TINY_LEN_12;
        case 13U: return ERF_MODE_TINY_LEN_13;
        case 14U: return ERF_MODE_TINY_LEN_14;
        case 15U: return ERF_MODE_TINY_LEN_15;
        case 16U: return ERF_MODE_TINY_LEN_16;
        case 17U: return ERF_MODE_TINY_LEN_17;
        case 18U: return ERF_MODE_TINY_LEN_18;
        case 19U: return ERF_MODE_TINY_LEN_19;
        case 20U: return ERF_MODE_TINY_LEN_20;
        case 21U: return ERF_MODE_TINY_LEN_21;
        case 22U: return ERF_MODE_TINY_LEN_22;
        case 23U: return ERF_MODE_TINY_LEN_23;
        case 24U: return ERF_MODE_TINY_LEN_24;
        case 25U: return ERF_MODE_TINY_LEN_25;
        case 26U: return ERF_MODE_TINY_LEN_26;
        case 27U: return ERF_MODE_TINY_LEN_27;
        case 28U: return ERF_MODE_TINY_LEN_28;
        case 29U: return ERF_MODE_TINY_LEN_29;
        case 30U: return ERF_MODE_TINY_LEN_30;
        case 31U: return ERF_MODE_TINY_LEN_31;
        default: return ERF_MODE_TINY_1_31;
    }
}

}  // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    uint32_t numCoresAiv = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (numCoresAiv == 0U) {
        numCoresAiv = 1U;
    }

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    uint32_t lengthX = static_cast<uint32_t>(tensorX->GetShapeSize());
    uint32_t alignedLength = AlignDown(lengthX, COPY_ALIGN_ELEMS);
    uint32_t paddedLength = AlignUp(lengthX, COPY_ALIGN_ELEMS);

    uint64_t ubSizeU64 = 0U;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeU64);
    uint32_t directLimit = CalcUbLimitElems(
        ubSizeU64, DIRECT_BUFFER_COUNT, LOCAL_BUFFER_PAD_BYTES);
    uint32_t tiledLimit = CalcUbLimitElems(
        ubSizeU64, TILED_BUFFER_COUNT, LOCAL_BUFFER_PAD_BYTES);
    (void)paddedLength;

    uint32_t activeCores = 1U;
    uint32_t algoKind = ERF_ALGO_TINY_SCALAR;
    uint32_t modeKind = ERF_MODE_TINY_1_31;
    uint32_t tileLength = COPY_ALIGN_ELEMS;
    uint32_t DT_X = C_DT_FLOAT;

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();

    uint32_t formerNum = 0U;
    uint32_t formerLength = 0U;
    uint32_t tailLength = 0U;
    if (lengthX <= 31U) {
        // 0..31: direct scalar. Selected lengths use exact no-read tiling keys;
        // all other tiny lengths keep the original runtime scalar path.
        activeCores = 1U;
        algoKind = ERF_ALGO_TINY_SCALAR;
        modeKind = GetTinyExactMode(lengthX);
        tileLength = COPY_ALIGN_ELEMS;
#if ERF_ENABLE_LOCAL_PADE_RANGE
    } else if (lengthX == 128U && numCoresAiv >= 8U) {
        tiling->length = lengthX;
        tiling->alignedLength = 128U;
        tiling->tileLength = 16U;
        tiling->formerNum = 8U;
        tiling->formerLength = 16U;
        tiling->tailLength = 16U;

        context->SetBlockDim(8U);
        context->SetTilingKey(GET_TPL_TILING_KEY(
            DT_X, ERF_ALGO_LOCAL_PADE, ERF_MODE_LOCAL_PADE_128_FIXED));

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    } else if (lengthX >= static_cast<uint32_t>(ERF_LOCAL_PADE_MIN_LEN) &&
               lengthX <= static_cast<uint32_t>(ERF_LOCAL_PADE_MAX_LEN)) {
        uint32_t localPadeElemsPerCore =
            static_cast<uint32_t>(ERF_LOCAL_PADE_ELEMS_PER_CORE);
        if (localPadeElemsPerCore == 0U) {
            localPadeElemsPerCore = 1U;
        }

        uint32_t requiredCores = CeilDiv(lengthX, localPadeElemsPerCore);
        if (requiredCores <= numCoresAiv) {
            activeCores = NormalizeCoreNum(requiredCores, numCoresAiv);
            activeCores = MaxU32(activeCores, 1U);

            tiling->length = lengthX;
            tiling->alignedLength = alignedLength;
            tiling->tileLength = localPadeElemsPerCore;
            tiling->formerNum = activeCores;
            tiling->formerLength = localPadeElemsPerCore;
            tiling->tailLength = localPadeElemsPerCore;

            context->SetBlockDim(activeCores);
            context->SetTilingKey(GET_TPL_TILING_KEY(
                DT_X, ERF_ALGO_LOCAL_PADE, ERF_MODE_LOCAL_PADE_16));

            size_t *currentWorkspace = context->GetWorkspaceSizes(1);
            currentWorkspace[0] = 0;
            return ge::GRAPH_SUCCESS;
        }
#endif
    } else if (lengthX == 32U) {
        activeCores = 1U;
        algoKind = ERF_ALGO_RANGE_C1;
        modeKind = ERF_MODE_RANGE_32_FIXED;
        tileLength = 32U;
    } else if (lengthX <= 64U) {
        activeCores = 1U;
        algoKind = ERF_ALGO_RANGE_C1;
        if (lengthX == 64U) {
            modeKind = ERF_MODE_RANGE_64_FIXED;
            tileLength = 64U;
        } else {
            modeKind = ERF_MODE_RANGE_32_64_VECTOR;
            tileLength = 64U;
        }
    } else if (lengthX == 128U) {
        activeCores = 1U;
        algoKind = ERF_ALGO_RANGE_C1;
        modeKind = ERF_MODE_RANGE_128_FIXED;
        tileLength = 128U;
    } else if (lengthX <= 128U) {
        activeCores = 1U;
        algoKind = ERF_ALGO_RANGE_C1;
        modeKind = ERF_MODE_RANGE_65_128;
        tileLength = 128U;
    } else if (lengthX == 256U) {
        activeCores = 1U;
        algoKind = ERF_ALGO_RANGE_C1;
        modeKind = ERF_MODE_RANGE_256_FIXED;
        tileLength = 256U;
    } else if (lengthX <= 256U) {
        activeCores = 1U;
        algoKind = ERF_ALGO_RANGE_C1;
        modeKind = ERF_MODE_RANGE_129_256;
        tileLength = 256U;
    } else if (lengthX == 512U) {
        activeCores = 1U;
        algoKind = ERF_ALGO_RANGE_C1;
        modeKind = ERF_MODE_RANGE_512_FIXED;
        tileLength = 512U;
    } else if (lengthX <= 512U) {
        activeCores = 1U;
        algoKind = ERF_ALGO_RANGE_C1;
        modeKind = ERF_MODE_RANGE_257_512_C1;
        tileLength = 512U;
#if ERF_ENABLE_MEDIUM_BUCKET_DIRECT
    } else if (lengthX >= static_cast<uint32_t>(ERF_MEDIUM_BUCKET_MIN_LENGTH) &&
               lengthX <= static_cast<uint32_t>(ERF_MEDIUM_BUCKET_MAX_LENGTH) &&
               alignedLength > 0U && directLimit >= MEDIUM_BUCKET_ALIGN_ELEMS) {
        uint32_t targetElems =
            (lengthX <= static_cast<uint32_t>(ERF_MEDIUM_BUCKET_SPLIT_LEN))
                ? static_cast<uint32_t>(ERF_MEDIUM_BUCKET_TARGET_LOW)
                : static_cast<uint32_t>(ERF_MEDIUM_BUCKET_TARGET_HIGH);
        if (targetElems < COPY_ALIGN_ELEMS) {
            targetElems = COPY_ALIGN_ELEMS;
        }
        if (targetElems > MEDIUM_BUCKET_MAX_ELEMS) {
            targetElems = MEDIUM_BUCKET_MAX_ELEMS;
        }

        activeCores = CeilDiv(alignedLength, targetElems);
        activeCores = NormalizeCoreNum(activeCores, numCoresAiv);
        activeCores = ClampCoreNumByAlignedLength(activeCores, alignedLength);

        uint32_t tmpFormerNum = 0U;
        uint32_t tmpFormerLength = 0U;
        uint32_t tmpTailLength = 0U;
        BuildAlignedCoreSplit(alignedLength, activeCores,
                              tmpFormerNum, tmpFormerLength, tmpTailLength);
        uint32_t maxCoreLength = MaxU32(tmpFormerLength, tmpTailLength);
        uint32_t bucketLength = RoundMediumBucket(maxCoreLength);

        if (bucketLength > 0U && bucketLength <= directLimit) {
            algoKind = ERF_ALGO_MEDIUM_BUCKET;
            modeKind = GetMediumBucketMode(bucketLength);
            tileLength = bucketLength;
        } else if (lengthX <= 1024U) {
            algoKind = ERF_ALGO_RANGE_SPLIT;
            modeKind = ERF_MODE_RANGE_513_1024;
            activeCores = ChooseRepeatGroupCores(lengthX, numCoresAiv);
            activeCores = ClampCoreNumByAlignedLength(activeCores, alignedLength);
            tileLength = COPY_ALIGN_ELEMS; // updated after split fields are computed
        } else {
            algoKind = ERF_ALGO_LARGE_TILED;
            modeKind = ERF_MODE_GENERIC;
        }
#endif
    } else if (lengthX <= 1024U) {
        algoKind = ERF_ALGO_RANGE_SPLIT;
        modeKind = ERF_MODE_RANGE_513_1024;
        activeCores = ChooseRepeatGroupCores(lengthX, numCoresAiv);
        activeCores = ClampCoreNumByAlignedLength(activeCores, alignedLength);
        tileLength = COPY_ALIGN_ELEMS; // updated after split fields are computed
    } else {
        algoKind = ERF_ALGO_LARGE_TILED;
        modeKind = ERF_MODE_GENERIC;

        uint32_t targetCoreElems = COPY_ALIGN_ELEMS * CORE_TARGET_ALIGN_UNITS;
        if (targetCoreElems < COPY_ALIGN_ELEMS) {
            targetCoreElems = COPY_ALIGN_ELEMS;
        }

        activeCores = CeilDiv(lengthX, targetCoreElems);
        activeCores = NormalizeCoreNum(activeCores, numCoresAiv);
        activeCores = ClampCoreNumByAlignedLength(activeCores, alignedLength);

#if ERF_ENABLE_LARGE_SPECIAL_CORE
        if (lengthX >= static_cast<uint32_t>(ERF_LARGE_SPECIAL_MIN_120K) &&
            lengthX <= static_cast<uint32_t>(ERF_LARGE_SPECIAL_MAX_140K)) {
            activeCores = NormalizeCoreNum(
                static_cast<uint32_t>(ERF_LARGE_SPECIAL_CORE_120_140K),
                numCoresAiv);
            activeCores = ClampCoreNumByAlignedLength(activeCores, alignedLength);
        } else if (lengthX >= static_cast<uint32_t>(ERF_LARGE_SPECIAL_MIN_281K) &&
                   lengthX <= static_cast<uint32_t>(ERF_LARGE_SPECIAL_MAX_296K)) {
            activeCores = NormalizeCoreNum(
                static_cast<uint32_t>(ERF_LARGE_SPECIAL_CORE_281_296K),
                numCoresAiv);
            activeCores = ClampCoreNumByAlignedLength(activeCores, alignedLength);
        }
#endif

        uint32_t upperTileLength = tiledLimit;
        uint32_t maxNeededTileLength = AlignUp(lengthX, COPY_ALIGN_ELEMS);
        if (maxNeededTileLength < COPY_ALIGN_ELEMS) {
            maxNeededTileLength = COPY_ALIGN_ELEMS;
        }

        upperTileLength = MinU32(upperTileLength, maxNeededTileLength);
        upperTileLength = AlignDown(upperTileLength, COPY_ALIGN_ELEMS);
        if (upperTileLength < COPY_ALIGN_ELEMS) {
            return ge::GRAPH_FAILED;
        }

        uint32_t perCoreTileLength = AlignUp(CeilDiv(lengthX, activeCores), COPY_ALIGN_ELEMS);
        if (perCoreTileLength < COPY_ALIGN_ELEMS) {
            perCoreTileLength = COPY_ALIGN_ELEMS;
        }

        tileLength = MinU32(upperTileLength, perCoreTileLength);
        if (tileLength < COPY_ALIGN_ELEMS) {
            tileLength = COPY_ALIGN_ELEMS;
        }
    }

    activeCores = ClampCoreNumByAlignedLength(activeCores, alignedLength);
    BuildAlignedCoreSplit(alignedLength, activeCores, formerNum, formerLength, tailLength);

    if (algoKind == ERF_ALGO_MEDIUM_BUCKET) {
        uint32_t bucketLength = RoundMediumBucket(MaxU32(formerLength, tailLength));
        if (bucketLength == 0U) {
            return ge::GRAPH_FAILED;
        }
        tileLength = bucketLength;
        modeKind = GetMediumBucketMode(bucketLength);
    }

    if (algoKind == ERF_ALGO_RANGE_SPLIT) {
        tileLength = MaxU32(formerLength, tailLength);
        tileLength = AlignUp(tileLength, COPY_ALIGN_ELEMS);
        if (tileLength < COPY_ALIGN_ELEMS) {
            tileLength = COPY_ALIGN_ELEMS;
        }
    }

    if (algoKind == ERF_ALGO_TINY_SCALAR || algoKind == ERF_ALGO_RANGE_C1) {
        if (tileLength < COPY_ALIGN_ELEMS) {
            tileLength = COPY_ALIGN_ELEMS;
        }
    }

    tiling->length = lengthX;
    tiling->alignedLength = alignedLength;
    tiling->tileLength = tileLength;
    tiling->formerNum = formerNum;
    tiling->formerLength = formerLength;
    tiling->tailLength = tailLength;

    context->SetBlockDim(activeCores);
    context->SetTilingKey(GET_TPL_TILING_KEY(DT_X, algoKind, modeKind));

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    ge::DataType xDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, xDtype);
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class Erf : public OpDef {
public:
    explicit Erf(const char *name) : OpDef(name)
    {
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
