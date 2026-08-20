#include "atanh_tiling.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

constexpr uint32_t kAlignBytes = 32;
constexpr uint32_t kMainCopyAlignBytes = 512;
constexpr uint32_t kBankConflictSkewBytes = 256;

constexpr uint32_t kFp32PreferredBlockBytes = 48 * 1024;
constexpr uint32_t kFp32Large2dPreferredBlockBytes = 96 * 1024;
constexpr uint32_t kFp16PreferredBlockBytes = 64 * 1024;
constexpr uint32_t kFp16SmallPreferredBlockBytes = 24 * 1024;
constexpr uint32_t kFp16LargePreferredBlockBytes = 48 * 1024;
constexpr uint32_t kFp16PreferredBlockSwitchTotalBytes = 840 * 1024;
constexpr uint32_t kBf16PreferredBlockBytes = 32 * 1024;
constexpr uint32_t kBf16PreferAllCoreSwitchTotalBytes = 640 * 1024;
constexpr uint32_t kOtherPreferredBlockBytes = 128 * 1024;

constexpr uint32_t kDefaultTileBytes = 64 * 1024;
constexpr uint32_t kFp16TileBytes = 44 * 1024;
constexpr uint32_t kFp32TileBytes = 48 * 1024;
constexpr uint32_t kFp32Large2dTileBytes = 48 * 1024;
constexpr uint32_t kBf16TileBytes = 32 * 1024;
constexpr uint32_t kUbSizeBytes = 192 * 1024;
constexpr uint32_t kUbSafetyBytes = 0;

constexpr uint32_t kFp32DbThresholdBytes = 32 * 1024;
constexpr uint32_t kFp16DbThresholdBytes = 48 * 1024;
constexpr uint32_t kFp16MultiCoreDbThresholdBytes = 24 * 1024;
constexpr uint32_t kBf16DbThresholdBytes = 32 * 1024;
constexpr uint32_t kFp32Large2dThresholdBytes = 2 * 1024 * 1024;
constexpr uint32_t kMaxAivCoreNum = 40;

constexpr uint32_t kTilingKeyFp16 = 1;
constexpr uint32_t kTilingKeyFp32 = 2;
constexpr uint32_t kTilingKeyBf16 = 3;
constexpr uint32_t kTilingKeyInt32 = 4;
constexpr uint32_t kTilingKeyInt16 = 5;
constexpr uint32_t kTilingKeyUint8 = 6;
constexpr uint32_t kTilingKeyInt8 = 7;
constexpr uint32_t kTilingKeyFp16Db = 10;
constexpr uint32_t kTilingKeyBf16Db = 11;
constexpr uint32_t kTilingKeyFp32Db = 20;

uint32_t GetDataTypeSize(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_INT16:
            return 2;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        case ge::DT_UINT8:
        case ge::DT_INT8:
            return 1;
        default:
            return 0;
    }
}

uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return ((value + align - 1) / align) * align;
}

uint32_t AlignDown(uint32_t value, uint32_t align)
{
    return (value / align) * align;
}

uint32_t GetEnvUint(const char* name, uint32_t defaultValue)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    char* end = nullptr;
    unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0 || parsed > UINT32_MAX) {
        return defaultValue;
    }
    return static_cast<uint32_t>(parsed);
}

bool HasEnvValue(const char* name)
{
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0';
}

uint32_t CeilDivU32(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

uint64_t CeilDivU64(uint64_t a, uint64_t b)
{
    return (a + b - 1) / b;
}

uint32_t ClampCoreNum(uint32_t candidate, uint32_t coreNum, uint32_t totalBlocks)
{
    candidate = std::max(1U, candidate);
    candidate = std::min(candidate, std::max(1U, coreNum));
    candidate = std::min(candidate, std::max(1U, totalBlocks));
    return candidate;
}

uint32_t SelectBalancedCoreNum(uint32_t totalBlocks, uint32_t targetCoreNum)
{
    return std::max(1U, std::min(totalBlocks, targetCoreNum));
}

uint32_t GetCopyAlignBytes(uint64_t totalBytes)
{
    return totalBytes >= kMainCopyAlignBytes ? kMainCopyAlignBytes : kAlignBytes;
}

bool IsFp32Large2dCase(const gert::StorageShape* inputShape, uint64_t totalBytes)
{
    if (inputShape->GetStorageShape().GetDimNum() != 2) {
        return false;
    }
    uint32_t threshold = GetEnvUint("ATANH_FP32_2D_LARGE_THRESHOLD_BYTES", kFp32Large2dThresholdBytes);
    return totalBytes >= threshold;
}

uint32_t GetBlockAlignBytes(uint64_t totalBytes, ge::DataType dt, bool isFp32Large2dCase, uint32_t copyAlignBytes)
{
    uint32_t defaultAlignBytes = copyAlignBytes;
    if (dt == ge::DT_FLOAT && isFp32Large2dCase) {
        uint32_t blockAlignBytes =
            GetEnvUint("ATANH_FP32_2D_LARGE_BLOCK_ALIGN_BYTES", defaultAlignBytes);
        if (blockAlignBytes >= kAlignBytes && (blockAlignBytes % kAlignBytes) == 0) {
            return blockAlignBytes;
        }
    }
    return defaultAlignBytes;
}

uint32_t GetFp16PreferredBlockBytes(uint64_t totalBytes)
{
    if (HasEnvValue("ATANH_FP16_PREFERRED_BLOCK_BYTES")) {
        return GetEnvUint("ATANH_FP16_PREFERRED_BLOCK_BYTES", kFp16PreferredBlockBytes);
    }
    uint32_t switchBytes =
        GetEnvUint("ATANH_FP16_PREFERRED_BLOCK_SWITCH_TOTAL_BYTES", kFp16PreferredBlockSwitchTotalBytes);
    return totalBytes < switchBytes ? kFp16SmallPreferredBlockBytes : kFp16LargePreferredBlockBytes;
}

uint32_t GetBf16PreferredBlockBytes(uint64_t totalBytes, uint32_t coreNum)
{
    if (HasEnvValue("ATANH_BF16_PREFERRED_BLOCK_BYTES")) {
        return GetEnvUint("ATANH_BF16_PREFERRED_BLOCK_BYTES", kBf16PreferredBlockBytes);
    }
    uint32_t switchBytes =
        GetEnvUint("ATANH_BF16_PREFER_ALL_CORE_SWITCH_TOTAL_BYTES", kBf16PreferAllCoreSwitchTotalBytes);
    if (totalBytes < switchBytes && coreNum > 0) {
        uint32_t perCoreBytes = static_cast<uint32_t>(CeilDivU64(totalBytes, coreNum));
        return AlignUp(std::max(perCoreBytes, kAlignBytes), kAlignBytes);
    }
    return kBf16PreferredBlockBytes;
}

uint32_t GetPreferredBlockBytes(ge::DataType dt, bool isFp32Large2dCase, uint64_t totalBytes, uint32_t coreNum)
{
    if (dt == ge::DT_FLOAT) {
        if (isFp32Large2dCase) {
            return GetEnvUint("ATANH_FP32_2D_LARGE_PREFERRED_BLOCK_BYTES", kFp32Large2dPreferredBlockBytes);
        }
        return GetEnvUint("ATANH_FP32_PREFERRED_BLOCK_BYTES", kFp32PreferredBlockBytes);
    }
    if (dt == ge::DT_FLOAT16) {
        return GetFp16PreferredBlockBytes(totalBytes);
    }
    if (dt == ge::DT_BF16) {
        return GetBf16PreferredBlockBytes(totalBytes, coreNum);
    }
    return GetEnvUint("ATANH_OTHER_PREFERRED_BLOCK_BYTES", kOtherPreferredBlockBytes);
}

uint32_t GetInitialTilingKey(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT16:
            return kTilingKeyFp16;
        case ge::DT_FLOAT:
            return kTilingKeyFp32;
        case ge::DT_BF16:
            return kTilingKeyBf16;
        case ge::DT_INT32:
            return kTilingKeyInt32;
        case ge::DT_INT16:
            return kTilingKeyInt16;
        case ge::DT_UINT8:
            return kTilingKeyUint8;
        case ge::DT_INT8:
            return kTilingKeyInt8;
        default:
            return 0;
    }
}

bool CanUseDoubleBuffer(uint32_t blockLength, uint32_t elemPerCopyAlign)
{
    return blockLength >= elemPerCopyAlign * 2U;
}

bool MeetsDoubleBufferThreshold(uint32_t blockLength, uint32_t dataWidth, uint32_t thresholdBytes,
                                uint32_t elemPerCopyAlign)
{
    if (!CanUseDoubleBuffer(blockLength, elemPerCopyAlign)) {
        return false;
    }
    return static_cast<uint64_t>(blockLength) * dataWidth >= thresholdBytes;
}

bool ShouldUseFp32Db(uint64_t totalBytes, uint32_t blockLength, uint32_t dataWidth, uint32_t elemPerCopyAlign)
{
    if (GetEnvUint("ATANH_FP32_FORCE_DOUBLE_BUFFER", 0) != 0) {
        return CanUseDoubleBuffer(blockLength, elemPerCopyAlign);
    }
    uint32_t threshold = GetEnvUint("ATANH_FP32_DB_THRESHOLD_BYTES", kFp32DbThresholdBytes);
    (void)totalBytes;
    return MeetsDoubleBufferThreshold(blockLength, dataWidth, threshold, elemPerCopyAlign);
}

bool ShouldUseFp16Db(uint32_t blockDim, uint32_t blockLength, uint32_t dataWidth, uint32_t elemPerCopyAlign)
{
    if (GetEnvUint("ATANH_FP16_FORCE_SINGLE_BUFFER", 0) != 0) {
        return false;
    }
    if (GetEnvUint("ATANH_FP16_FORCE_DOUBLE_BUFFER", 0) != 0) {
        return CanUseDoubleBuffer(blockLength, elemPerCopyAlign);
    }
    uint32_t threshold = blockDim > 1
                             ? GetEnvUint("ATANH_FP16_MULTI_CORE_DB_THRESHOLD_BYTES", kFp16MultiCoreDbThresholdBytes)
                             : GetEnvUint("ATANH_FP16_DB_THRESHOLD_BYTES", kFp16DbThresholdBytes);
    return MeetsDoubleBufferThreshold(blockLength, dataWidth, threshold, elemPerCopyAlign);
}

bool ShouldUseBf16Db(uint32_t blockLength, uint32_t dataWidth, uint32_t elemPerCopyAlign)
{
    if (GetEnvUint("ATANH_BF16_FORCE_SINGLE_BUFFER", 0) != 0) {
        return false;
    }
    if (GetEnvUint("ATANH_BF16_FORCE_DOUBLE_BUFFER", 0) != 0) {
        return CanUseDoubleBuffer(blockLength, elemPerCopyAlign);
    }
    uint32_t threshold = GetEnvUint("ATANH_BF16_DB_THRESHOLD_BYTES", kBf16DbThresholdBytes);
    return MeetsDoubleBufferThreshold(blockLength, dataWidth, threshold, elemPerCopyAlign);
}

uint32_t GetWorkingSetBytesPerElem(ge::DataType dt, uint32_t tilingKey)
{
    switch (dt) {
        case ge::DT_FLOAT16:
            return tilingKey == kTilingKeyFp16Db ? 8 : 6;
        case ge::DT_BF16:
            return tilingKey == kTilingKeyBf16Db ? 12 : 12;
        case ge::DT_FLOAT:
            return tilingKey == kTilingKeyFp32Db ? 16 : 12;
        case ge::DT_INT32:
            return 20;
        case ge::DT_INT16:
            return 18;
        case ge::DT_UINT8:
        case ge::DT_INT8:
            return 19;
        default:
            return 0;
    }
}

uint32_t GetTmpBytesPerElem(ge::DataType dt, uint32_t tilingKey)
{
    switch (dt) {
        case ge::DT_FLOAT16:
            return tilingKey == kTilingKeyFp16Db ? 8 : 6;
        case ge::DT_BF16:
            return 12;
        case ge::DT_FLOAT:
            return tilingKey == kTilingKeyFp32Db ? 16 : 12;
        case ge::DT_INT32:
            return 20;
        case ge::DT_INT16:
            return 18;
        case ge::DT_UINT8:
        case ge::DT_INT8:
            return 19;
        default:
            return 0;
    }
}

uint32_t GetReservedUbBytes(ge::DataType dt, uint32_t tilingKey)
{
    if (dt == ge::DT_FLOAT && tilingKey == kTilingKeyFp32Db) {
        return GetEnvUint("ATANH_FP32_DB_RESERVED_UB_BYTES", kUbSafetyBytes);
    }
    if (dt == ge::DT_FLOAT16 && tilingKey == kTilingKeyFp16Db) {
        return GetEnvUint("ATANH_FP16_DB_RESERVED_UB_BYTES", kUbSafetyBytes);
    }
    if (dt == ge::DT_BF16 && tilingKey == kTilingKeyBf16Db) {
        return GetEnvUint("ATANH_BF16_DB_RESERVED_UB_BYTES", kUbSafetyBytes);
    }
    return GetEnvUint("ATANH_RESERVED_UB_BYTES", kUbSafetyBytes);
}

uint32_t GetAggressiveUbExtraBytes(ge::DataType dt, uint32_t tilingKey)
{
    if (dt == ge::DT_FLOAT && tilingKey == kTilingKeyFp32Db) {
        return GetEnvUint("ATANH_FP32_DB_UB_EXTRA_BYTES", kBankConflictSkewBytes * 3U);
    }
    if (dt == ge::DT_FLOAT16 && tilingKey == kTilingKeyFp16Db) {
        return GetEnvUint("ATANH_FP16_DB_UB_EXTRA_BYTES", kBankConflictSkewBytes * 2U);
    }
    if (dt == ge::DT_BF16 && tilingKey == kTilingKeyBf16Db) {
        return GetEnvUint("ATANH_BF16_DB_UB_EXTRA_BYTES", 0);
    }
    return 0;
}

uint32_t GetDefaultTileBytes(ge::DataType dt, uint32_t tilingKey, bool isFp32Large2dCase)
{
    if (dt == ge::DT_FLOAT) {
        if (tilingKey == kTilingKeyFp32Db && isFp32Large2dCase) {
            return GetEnvUint("ATANH_FP32_2D_LARGE_TILE_BYTES", kFp32Large2dTileBytes);
        }
        return GetEnvUint("ATANH_FP32_TILE_BYTES", kFp32TileBytes);
    }
    if (dt == ge::DT_FLOAT16) {
        return GetEnvUint("ATANH_FP16_TILE_BYTES", kFp16TileBytes);
    }
    if (dt == ge::DT_BF16) {
        return GetEnvUint("ATANH_BF16_TILE_BYTES", kBf16TileBytes);
    }
    return GetEnvUint("ATANH_OTHER_TILE_BYTES", kDefaultTileBytes);
}

uint32_t SelectBalancedBlockDim(uint32_t totalLength, uint64_t totalBytes, uint32_t preferredBlockBytes,
                                uint32_t coreNum, uint32_t dataWidth, uint32_t elemPerBlockAlign)
{
    if (totalLength == 0 || totalBytes == 0 || coreNum == 0 || dataWidth == 0) {
        return 1;
    }
    uint32_t alignedBlockElems = std::max(
        elemPerBlockAlign, AlignDown(std::max(1U, preferredBlockBytes / dataWidth), elemPerBlockAlign));
    uint32_t totalUbBlocks = CeilDivU32(totalLength, alignedBlockElems);
    uint32_t targetCoreNum = static_cast<uint32_t>(
        CeilDivU64(totalBytes, static_cast<uint64_t>(std::max(1U, preferredBlockBytes))));
    targetCoreNum = ClampCoreNum(targetCoreNum, coreNum, totalUbBlocks);
    return SelectBalancedCoreNum(totalUbBlocks, targetCoreNum);
}

uint64_t EstimateTileFootprintBytes(ge::DataType dt, uint32_t tilingKey, uint32_t tileLength)
{
    uint64_t alignedDataBytes = 0;
    uint64_t skewBytes = 0;

    if (dt == ge::DT_FLOAT && tilingKey == kTilingKeyFp32Db) {
        alignedDataBytes = AlignUp(static_cast<uint32_t>(tileLength * sizeof(float)), kAlignBytes);
        skewBytes = static_cast<uint64_t>(kBankConflictSkewBytes) * 3ULL;
        return alignedDataBytes * 4ULL + skewBytes;
    }
    if (dt == ge::DT_FLOAT16 && tilingKey == kTilingKeyFp16Db) {
        alignedDataBytes = AlignUp(static_cast<uint32_t>(tileLength * sizeof(uint16_t)), kAlignBytes);
        skewBytes = static_cast<uint64_t>(kBankConflictSkewBytes) * 2ULL;
        return alignedDataBytes * 4ULL + skewBytes;
    }
    if (dt == ge::DT_BF16 && tilingKey == kTilingKeyBf16Db) {
        uint64_t rawBytes = AlignUp(static_cast<uint32_t>(tileLength * sizeof(uint16_t)), kAlignBytes);
        uint64_t floatBytes = AlignUp(static_cast<uint32_t>(tileLength * sizeof(float)), kAlignBytes);
        return rawBytes * 2ULL + floatBytes * 2ULL;
    }
    if (dt == ge::DT_BF16 && tilingKey == kTilingKeyBf16) {
        uint64_t rawBytes = AlignUp(static_cast<uint32_t>(tileLength * sizeof(uint16_t)), kAlignBytes);
        uint64_t floatBytes = AlignUp(static_cast<uint32_t>(tileLength * sizeof(float)), kAlignBytes);
        return rawBytes * 2ULL + floatBytes * 2ULL;
    }
    if (dt == ge::DT_FLOAT16 && tilingKey == kTilingKeyFp16) {
        alignedDataBytes = AlignUp(static_cast<uint32_t>(tileLength * sizeof(uint16_t)), kAlignBytes);
        return alignedDataBytes * 3ULL;
    }
    if (dt == ge::DT_FLOAT && tilingKey == kTilingKeyFp32) {
        alignedDataBytes = AlignUp(static_cast<uint32_t>(tileLength * sizeof(float)), kAlignBytes);
        return alignedDataBytes * 3ULL;
    }
    if (dt == ge::DT_INT32) {
        uint64_t inputBytes = AlignUp(static_cast<uint32_t>(tileLength * sizeof(int32_t)), kAlignBytes);
        uint64_t floatBytes = AlignUp(static_cast<uint32_t>(tileLength * sizeof(float)), kAlignBytes);
        return inputBytes + floatBytes * 3ULL;
    }
    if (dt == ge::DT_INT16) {
        uint64_t inputBytes = AlignUp(static_cast<uint32_t>(tileLength * sizeof(int16_t)), kAlignBytes);
        uint64_t floatBytes = AlignUp(static_cast<uint32_t>(tileLength * sizeof(float)), kAlignBytes);
        return inputBytes + floatBytes * 3ULL;
    }
    if (dt == ge::DT_INT8 || dt == ge::DT_UINT8) {
        uint64_t inputBytes = AlignUp(tileLength, kAlignBytes);
        uint64_t halfBytes = AlignUp(static_cast<uint32_t>(tileLength * sizeof(uint16_t)), kAlignBytes);
        uint64_t floatBytes = AlignUp(static_cast<uint32_t>(tileLength * sizeof(float)), kAlignBytes);
        return inputBytes + halfBytes + floatBytes * 3ULL;
    }

    return static_cast<uint64_t>(tileLength) * GetTmpBytesPerElem(dt, tilingKey);
}

uint32_t SelectAggressiveTileLength(ge::DataType dt, uint32_t tilingKey, uint32_t blockLength,
                                    uint32_t elemPerCopyAlign, uint64_t ubSize)
{
    uint64_t reservedUbBytes = std::min<uint64_t>(GetReservedUbBytes(dt, tilingKey), ubSize);
    uint64_t usableUbBytes = ubSize > reservedUbBytes ? (ubSize - reservedUbBytes) : ubSize;
    if (usableUbBytes == 0) {
        usableUbBytes = ubSize;
    }
    usableUbBytes += GetAggressiveUbExtraBytes(dt, tilingKey);

    uint32_t low = elemPerCopyAlign;
    uint32_t high = AlignDown(blockLength, elemPerCopyAlign);
    if (high == 0) {
        high = elemPerCopyAlign;
    }

    uint32_t best = elemPerCopyAlign;
    while (low <= high) {
        uint32_t mid = low + ((high - low) / (elemPerCopyAlign * 2U)) * elemPerCopyAlign;
        if (mid < low) {
            mid = low;
        }
        if ((mid % elemPerCopyAlign) != 0) {
            mid = AlignDown(mid, elemPerCopyAlign);
        }
        if (mid < elemPerCopyAlign) {
            mid = elemPerCopyAlign;
        }

        uint64_t footprintBytes = EstimateTileFootprintBytes(dt, tilingKey, mid);
        if (footprintBytes <= usableUbBytes) {
            best = mid;
            if (mid == high) {
                break;
            }
            low = mid + elemPerCopyAlign;
        } else {
            if (mid <= elemPerCopyAlign) {
                break;
            }
            high = mid - elemPerCopyAlign;
        }
    }

    return std::max(elemPerCopyAlign, std::min(blockLength, best));
}

uint32_t ClampDbTileLength(uint32_t tileLength, uint32_t blockLength, uint32_t elemPerCopyAlign)
{
    if (!CanUseDoubleBuffer(blockLength, elemPerCopyAlign)) {
        return tileLength;
    }
    uint32_t maxDbTile = AlignDown(blockLength / 2U, elemPerCopyAlign);
    if (maxDbTile >= elemPerCopyAlign) {
        return std::min(tileLength, maxDbTile);
    }
    return tileLength;
}

bool HasAtLeastTwoTilesPerBlock(uint32_t blockLength, uint32_t tileLength)
{
    return tileLength != 0 && blockLength >= tileLength * 2U;
}

}  // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AtanhTilingData tiling;

    auto dt = context->GetInputTensor(0)->GetDataType();
    uint32_t tilingKey = GetInitialTilingKey(dt);
    if (tilingKey == 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t dataWidth = GetDataTypeSize(dt);
    if (dataWidth == 0) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* inputShape = context->GetInputShape(0);
    uint64_t totalLength64 = 1;
    for (int32_t i = 0; i < inputShape->GetStorageShape().GetDimNum(); ++i) {
        totalLength64 *= static_cast<uint64_t>(inputShape->GetStorageShape().GetDim(i));
    }
    if (totalLength64 == 0 || totalLength64 > static_cast<uint64_t>(UINT32_MAX)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t totalLength = static_cast<uint32_t>(totalLength64);
    uint64_t totalBytes = totalLength64 * dataWidth;
    bool isFp32Large2dCase = (dt == ge::DT_FLOAT) && IsFp32Large2dCase(inputShape, totalBytes);

    uint32_t copyAlignBytes = GetCopyAlignBytes(totalBytes);
    uint32_t elemPerCopyAlign = std::max(1U, copyAlignBytes / dataWidth);
    uint32_t blockAlignBytes = GetBlockAlignBytes(totalBytes, dt, isFp32Large2dCase, copyAlignBytes);
    uint32_t elemPerBlockAlign = std::max(1U, blockAlignBytes / dataWidth);

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = std::min(kMaxAivCoreNum, std::max(1U, platform.GetCoreNumAiv()));

    uint32_t preferredBlockBytes = GetPreferredBlockBytes(dt, isFp32Large2dCase, totalBytes, coreNum);
    uint32_t useBalancedSharding = GetEnvUint("ATANH_USE_BALANCED_SHARDING", 1);
    uint32_t blockDim = useBalancedSharding != 0
                            ? SelectBalancedBlockDim(totalLength, totalBytes, preferredBlockBytes, coreNum, dataWidth,
                                                     elemPerBlockAlign)
                            : std::min(coreNum, std::max(1U, static_cast<uint32_t>(
                                                                   CeilDivU64(totalBytes, preferredBlockBytes))));

    uint32_t blockLength = 0;
    uint32_t lastBlockLength = 0;
    while (blockDim > 0) {
        blockLength = AlignUp(CeilDivU32(totalLength, blockDim), elemPerBlockAlign);
        uint64_t prefixLength = static_cast<uint64_t>(blockLength) * static_cast<uint64_t>(blockDim - 1);
        if (prefixLength < totalLength) {
            lastBlockLength = totalLength - static_cast<uint32_t>(prefixLength);
            break;
        }
        --blockDim;
    }
    if (blockDim == 0 || blockLength == 0 || lastBlockLength == 0) {
        return ge::GRAPH_FAILED;
    }

    if (dt == ge::DT_FLOAT && ShouldUseFp32Db(totalBytes, blockLength, dataWidth, elemPerCopyAlign)) {
        tilingKey = kTilingKeyFp32Db;
    } else if (dt == ge::DT_FLOAT16 && ShouldUseFp16Db(blockDim, blockLength, dataWidth, elemPerCopyAlign)) {
        tilingKey = kTilingKeyFp16Db;
    } else if (dt == ge::DT_BF16 && ShouldUseBf16Db(blockLength, dataWidth, elemPerCopyAlign)) {
        tilingKey = kTilingKeyBf16Db;
    }

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0) {
        ubSize = kUbSizeBytes;
    }

    uint32_t tmpBytesPerElem = GetTmpBytesPerElem(dt, tilingKey);
    uint32_t workingSetBytes = GetWorkingSetBytesPerElem(dt, tilingKey);
    if (ubSize == 0 || tmpBytesPerElem == 0 || workingSetBytes == 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t tileLength = SelectAggressiveTileLength(dt, tilingKey, blockLength, elemPerCopyAlign, ubSize);
    if (tilingKey == kTilingKeyFp32Db || tilingKey == kTilingKeyFp16Db || tilingKey == kTilingKeyBf16Db) {
        tileLength = ClampDbTileLength(tileLength, blockLength, elemPerCopyAlign);
        if (!HasAtLeastTwoTilesPerBlock(blockLength, tileLength)) {
            if (tilingKey == kTilingKeyFp32Db) {
                tilingKey = kTilingKeyFp32;
            } else if (tilingKey == kTilingKeyFp16Db) {
                tilingKey = kTilingKeyFp16;
            } else if (tilingKey == kTilingKeyBf16Db) {
                tilingKey = kTilingKeyBf16;
            }
            tmpBytesPerElem = GetTmpBytesPerElem(dt, tilingKey);
            workingSetBytes = GetWorkingSetBytesPerElem(dt, tilingKey);
            if (tmpBytesPerElem == 0 || workingSetBytes == 0) {
                return ge::GRAPH_FAILED;
            }
            tileLength = SelectAggressiveTileLength(dt, tilingKey, blockLength, elemPerCopyAlign, ubSize);
        }
    }

    tiling.set_totalLength(totalLength);
    tiling.set_blockLength(blockLength);
    tiling.set_lastBlockLength(lastBlockLength);
    tiling.set_tileLength(tileLength);
    tiling.set_tmpBufferSize(tileLength * tmpBytesPerElem);
    tiling.set_copyAlignBytes(copyAlignBytes);
    tiling.set_tilingKey(tilingKey);

    context->SetTilingKey(tilingKey);
    context->SetBlockDim(blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    auto inputType = context->GetInputDataType(0);
    switch (inputType) {
        case ge::DT_INT8:
        case ge::DT_UINT8:
        case ge::DT_INT16:
        case ge::DT_INT32:
            context->SetOutputDataType(0, ge::DT_FLOAT);
            break;
        default:
            context->SetOutputDataType(0, inputType);
            break;
    }
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class Atanh : public OpDef {
public:
    explicit Atanh(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8,
                       ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Atanh);

}  // namespace ops
