#include "fills_tiling.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

constexpr uint32_t kAlignBytes = 512;
constexpr uint32_t kReservedUbBytes = 8 * 1024;
constexpr uint32_t kPreferredBlockBytes = 64 * 1024;
constexpr uint32_t kDefaultTilingKey = 0;

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
    uint64_t totalBytes64 = static_cast<uint64_t>(totalLength) * dataWidth;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = std::max(1U, ascendcPlatform.GetCoreNumAiv());
    uint32_t preferredBlockBytes = GetEnvUint("FILLS_PREFERRED_BLOCK_BYTES", kPreferredBlockBytes);
    uint32_t neededCoreNum = std::max(1U, static_cast<uint32_t>((totalBytes64 + preferredBlockBytes - 1) / preferredBlockBytes));
    uint32_t blockDim = std::min(coreNum, neededCoreNum);
    blockDim = std::min(blockDim, GetBlockDimCap(dt, coreNum, totalBytes64));
    uint32_t forcedBlockDim = GetEnvUint("FILLS_FORCE_BLOCKDIM", 0);
    if (forcedBlockDim > 0) {
        blockDim = std::min(coreNum, forcedBlockDim);
    }
    uint32_t maxBlockDimByAlignment = std::max(1U, (totalLength + elemPerBlock - 1) / elemPerBlock);
    blockDim = std::min(blockDim, maxBlockDimByAlignment);

    uint32_t blockLength = (totalLength + blockDim - 1) / blockDim;
    blockLength = AlignUp(blockLength, elemPerBlock);
    uint32_t lastBlockLength = totalLength - blockLength * (blockDim - 1);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t usableUbBytes = ubSize > kReservedUbBytes ? (ubSize - kReservedUbBytes) : ubSize;
    uint32_t maxTileByUb = static_cast<uint32_t>(std::max<uint64_t>(elemPerBlock, usableUbBytes / dataWidth));
    maxTileByUb = (maxTileByUb / elemPerBlock) * elemPerBlock;
    uint32_t tileLength = std::max(elemPerBlock, std::min(blockLength, maxTileByUb));

    const float* valueAttr = context->GetAttrs()->GetFloat(0);
    if (valueAttr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tiling.set_totalLength(totalLength);
    tiling.set_blockLength(blockLength);
    tiling.set_lastBlockLength(lastBlockLength);
    tiling.set_tileLength(tileLength);
    tiling.set_tilingKey(tilingKey);
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
