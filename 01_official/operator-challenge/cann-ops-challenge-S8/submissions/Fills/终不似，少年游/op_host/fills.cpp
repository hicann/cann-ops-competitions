#include "fills_tiling.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

constexpr uint32_t kAlignBytes = 32;
constexpr uint32_t kPreferredAlignBytes = 512;
constexpr uint32_t kBytePreferredAlignBytes = 512;
constexpr uint32_t kReservedUbBytes = 0;
constexpr uint32_t kByteReservedUbBytes = 8 * 1024;
constexpr uint32_t kPreferredBlockBytes = 64 * 1024;
constexpr uint32_t kDefaultTilingKey = 0;
constexpr uint32_t kMaxActiveCoreCount = 40;

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

uint32_t GetTilingKey(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        case ge::DT_INT16:
        case ge::DT_UINT8:
        case ge::DT_INT8:
            return kDefaultTilingKey;
        default: return 0xffffffffU;
    }
}

uint32_t GetBlockDimCap(ge::DataType dt, uint32_t coreNum, uint64_t totalBytes)
{
    if (totalBytes >= 4ULL * 1024ULL * 1024ULL) {
        return coreNum;
    }
    if (totalBytes >= 1ULL * 1024ULL * 1024ULL) {
        switch (dt) {
            case ge::DT_FLOAT16:
            case ge::DT_FLOAT:
            case ge::DT_UINT8:
            case ge::DT_INT8:
                return std::min(coreNum, 32U);
            case ge::DT_INT16:
            case ge::DT_BF16:
                return std::min(coreNum, 32U);
            case ge::DT_INT32:
                return coreNum;
            default:
                return coreNum;
        }
    }

    switch (dt) {
        case ge::DT_FLOAT16:
        case ge::DT_FLOAT:
        case ge::DT_UINT8:
        case ge::DT_INT8:
            return std::min(coreNum, 8U);
        case ge::DT_INT16:
            return std::min(coreNum, 16U);
        case ge::DT_BF16:
            return std::min(coreNum, 32U);
        case ge::DT_INT32:
            return coreNum;
        default:
            return coreNum;
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

uint32_t GetEnvUint(const char* name, uint32_t defaultValue);

uint64_t AbsDiff(uint64_t lhs, uint64_t rhs)
{
    return lhs > rhs ? lhs - rhs : rhs - lhs;
}

uint64_t TailPenaltyElems(
    uint32_t lastLength,
    uint32_t dataWidth,
    uint32_t elemPerBlock,
    uint32_t preferredAlignBytes)
{
    const uint64_t lastBytes = static_cast<uint64_t>(lastLength) * dataWidth;
    uint64_t penalty = 0;
    if ((lastBytes % kAlignBytes) != 0U) {
        penalty += elemPerBlock;
    }
    if (preferredAlignBytes > kAlignBytes && lastBytes > preferredAlignBytes &&
        (lastBytes % preferredAlignBytes) != 0U) {
        penalty += elemPerBlock;
    }
    return penalty;
}

uint32_t SelectBalancedRegularBlockLength(
    uint32_t totalLength,
    uint32_t blockDim,
    uint32_t alignElems,
    uint32_t elemPerBlock,
    uint32_t dataWidth,
    uint32_t preferredAlignBytes)
{
    if (blockDim <= 1U) {
        return totalLength;
    }

    const uint32_t regularCoreCount = blockDim - 1U;
    const uint32_t maxRegularBlock =
        static_cast<uint32_t>((static_cast<uint64_t>(totalLength) - 1U) / regularCoreCount);
    const uint32_t align = std::max(1U, alignElems);
    if (maxRegularBlock == 0U) {
        return 1U;
    }

    const uint32_t meanFloor = totalLength / blockDim;
    const uint32_t meanCeil =
        static_cast<uint32_t>((static_cast<uint64_t>(totalLength) + blockDim - 1U) / blockDim);
    const uint32_t centers[] = {
        AlignDown(meanFloor, align),
        AlignUp(meanFloor, align),
        AlignDown(meanCeil, align),
        AlignUp(meanCeil, align),
        AlignDown(maxRegularBlock, align),
    };

    uint32_t best = 0;
    uint64_t bestScore = UINT64_MAX;
    auto consider = [&](uint32_t candidate) {
        if (candidate == 0U || candidate > maxRegularBlock) {
            return;
        }
        const uint64_t covered = static_cast<uint64_t>(candidate) * regularCoreCount;
        if (covered >= totalLength) {
            return;
        }
        const uint32_t lastLength = static_cast<uint32_t>(totalLength - covered);
        const uint64_t adjustedLast = static_cast<uint64_t>(lastLength) +
            TailPenaltyElems(lastLength, dataWidth, elemPerBlock, preferredAlignBytes);
        const uint64_t score = AbsDiff(static_cast<uint64_t>(candidate), adjustedLast);
        if (best == 0U || score < bestScore || (score == bestScore && candidate > best)) {
            best = candidate;
            bestScore = score;
        }
    };

    for (uint32_t center : centers) {
        if (center == 0U) {
            continue;
        }
        for (int32_t step = -8; step <= 8; ++step) {
            const int64_t candidate = static_cast<int64_t>(center) + static_cast<int64_t>(step) * align;
            if (candidate > 0 && candidate <= UINT32_MAX) {
                consider(static_cast<uint32_t>(candidate));
            }
        }
    }

    if (best == 0U) {
        best = AlignUp(meanCeil, elemPerBlock);
        if (best == 0U || best > maxRegularBlock) {
            best = std::max(1U, AlignDown(maxRegularBlock, elemPerBlock));
        }
    }
    return best;
}

uint32_t DefaultPreferredAlignBytes(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_UINT8:
        case ge::DT_INT8:
            return kBytePreferredAlignBytes;
        default:
            return kPreferredAlignBytes;
    }
}

uint32_t DefaultReservedUbBytes(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_UINT8:
        case ge::DT_INT8:
            return kByteReservedUbBytes;
        default:
            return kReservedUbBytes;
    }
}

uint32_t ResolvePreferredAlignBytes(ge::DataType dt, uint32_t dataWidth)
{
    uint32_t preferredAlignBytes =
        GetEnvUint("FILLS_PREFERRED_ALIGN_BYTES", DefaultPreferredAlignBytes(dt));
    if (preferredAlignBytes < kAlignBytes) {
        preferredAlignBytes = kAlignBytes;
    }
    preferredAlignBytes = AlignUp(preferredAlignBytes, kAlignBytes);
    preferredAlignBytes = AlignUp(preferredAlignBytes, dataWidth);
    return preferredAlignBytes;
}

uint16_t FloatToBf16Bits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1U;
    bits += 0x7FFFU + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

uint32_t GetEnvUint(const char* name, uint32_t defaultValue)
{
    const char* envValue = std::getenv(name);
    if (envValue == nullptr || envValue[0] == '\0') {
        return defaultValue;
    }

    char* end = nullptr;
    unsigned long parsed = std::strtoul(envValue, &end, 10);
    if (end == envValue || (end != nullptr && *end != '\0') || parsed == 0UL || parsed > UINT32_MAX) {
        return defaultValue;
    }
    return static_cast<uint32_t>(parsed);
}

}  // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    FillsTilingData tiling;

    auto dt = context->GetInputTensor(0)->GetDataType();
    uint32_t dataWidth = GetDataTypeSize(dt);
    uint32_t tilingKey = GetTilingKey(dt);
    if (dataWidth == 0 || tilingKey == 0xffffffffU) {
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
    uint32_t elemPerBlock = std::max(1U, kAlignBytes / dataWidth);
    uint32_t preferredAlignBytes = ResolvePreferredAlignBytes(dt, dataWidth);
    uint32_t elemPerPreferred = std::max(1U, preferredAlignBytes / dataWidth);
    uint64_t totalBytes64 = static_cast<uint64_t>(totalLength) * dataWidth;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = std::max(1U, ascendcPlatform.GetCoreNumAiv());
    coreNum = std::min(coreNum, kMaxActiveCoreCount);
    uint32_t preferredBlockBytes = GetEnvUint("FILLS_PREFERRED_BLOCK_BYTES", kPreferredBlockBytes);
    uint32_t neededCoreNum = std::max(1U, static_cast<uint32_t>((totalBytes64 + preferredBlockBytes - 1) / preferredBlockBytes));
    uint32_t blockDim = std::min(coreNum, neededCoreNum);
    blockDim = std::min(blockDim, GetBlockDimCap(dt, coreNum, totalBytes64));
    uint32_t forcedBlockDim = GetEnvUint("FILLS_FORCE_BLOCKDIM", 0);
    if (forcedBlockDim > 0) {
        blockDim = std::min(coreNum, forcedBlockDim);
    }
    uint32_t blockLength = SelectBalancedRegularBlockLength(
        totalLength, blockDim, elemPerPreferred, elemPerBlock, dataWidth, preferredAlignBytes);
    uint32_t lastBlockLength = totalLength - blockLength * (blockDim - 1);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t reservedUbBytes = GetEnvUint("FILLS_RESERVED_UB_BYTES", DefaultReservedUbBytes(dt));
    uint64_t usableUbBytes = ubSize > reservedUbBytes ? (ubSize - reservedUbBytes) : ubSize;
    uint32_t maxTileByUb = static_cast<uint32_t>(std::max<uint64_t>(elemPerPreferred, usableUbBytes / dataWidth));
    maxTileByUb = (maxTileByUb / elemPerPreferred) * elemPerPreferred;
    if (maxTileByUb == 0U) {
        maxTileByUb = elemPerPreferred;
    }
    uint32_t tileLength = std::min(blockLength, maxTileByUb);
    if (tileLength >= elemPerPreferred) {
        tileLength = AlignDown(tileLength, elemPerPreferred);
    }
    if (tileLength == 0U) {
        tileLength = std::max(elemPerBlock, std::min(blockLength, maxTileByUb));
        tileLength = AlignDown(tileLength, elemPerBlock);
        if (tileLength == 0U) {
            tileLength = elemPerBlock;
        }
    }

    const float* valueAttr = context->GetAttrs()->GetFloat(0);
    if (valueAttr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tiling.set_totalLength(totalLength);
    tiling.set_blockLength(blockLength);
    tiling.set_lastBlockLength(lastBlockLength);
    tiling.set_tileLength(tileLength);
    tiling.set_tilingKey(tilingKey);
    tiling.set_preferredAlignBytes(preferredAlignBytes);
    tiling.set_fillValue(*valueAttr);
    tiling.set_fillValueBf16(FloatToBf16Bits(*valueAttr));

    context->SetBlockDim(blockDim);
    context->SetTilingKey(tilingKey);
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
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class Fills : public OpDef {
public:
    explicit Fills(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("value")
            .AttrType(OPTIONAL)
            .Float(0.0);

        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Fills);

}  // namespace ops
