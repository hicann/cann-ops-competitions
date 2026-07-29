#include "erfinv_tiling.h"

#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

constexpr uint32_t kAlignBytes = 32;
constexpr uint32_t kMainCopyAlignBytes = 512;
constexpr uint32_t kDefaultTileBytes = 32 * 1024;
constexpr uint32_t kPreferredBlockBytes = 8 * 1024;
constexpr uint32_t kTilingKeyFp16 = 0;
constexpr uint32_t kTilingKeyBf16 = 1;
constexpr uint32_t kTilingKeyFp32 = 2;

uint32_t GetDataTypeSize(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2;
        case ge::DT_FLOAT:
            return 4;
        default:
            return 0;
    }
}

uint32_t GetWorkingSetBytesPerElem(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 29;
        case ge::DT_FLOAT:
            return 48;
        default:
            return 64;
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

uint32_t GetCopyAlignBytes(uint64_t totalBytes)
{
    return totalBytes >= kMainCopyAlignBytes ? kMainCopyAlignBytes : kAlignBytes;
}

uint32_t GetTilingKey(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT16:
            return kTilingKeyFp16;
        case ge::DT_BF16:
            return kTilingKeyBf16;
        case ge::DT_FLOAT:
            return kTilingKeyFp32;
        default:
            return UINT32_MAX;
    }
}

}  // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ErfinvTilingData tiling;

    auto dt = context->GetInputTensor(0)->GetDataType();
    uint32_t dataWidth = GetDataTypeSize(dt);
    if (dataWidth == 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t tilingKey = GetTilingKey(dt);
    if (tilingKey == UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* inputShape = context->GetInputShape(0);
    uint64_t totalLength64 = 1;
    for (int32_t i = 0; i < inputShape->GetStorageShape().GetDimNum(); ++i) {
        int64_t dim = inputShape->GetStorageShape().GetDim(i);
        if (dim <= 0) {
            return ge::GRAPH_FAILED;
        }
        totalLength64 *= static_cast<uint64_t>(dim);
    }
    if (totalLength64 == 0 || totalLength64 > static_cast<uint64_t>(UINT32_MAX)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t totalLength = static_cast<uint32_t>(totalLength64);
    uint64_t totalBytes64 = static_cast<uint64_t>(totalLength) * dataWidth;
    uint32_t copyAlignBytes = GetCopyAlignBytes(totalBytes64);
    uint32_t elemPerCopyAlign = std::max(1U, copyAlignBytes / dataWidth);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = std::max(1U, ascendcPlatform.GetCoreNumAiv());
    uint32_t neededCoreNum =
        std::max(1U, static_cast<uint32_t>((totalBytes64 + kPreferredBlockBytes - 1) / kPreferredBlockBytes));
    uint32_t blockDim = std::min(coreNum, neededCoreNum);
    uint32_t blockLength = 0;
    uint32_t lastBlockLength = 0;
    while (blockDim > 0) {
        blockLength = AlignUp((totalLength + blockDim - 1) / blockDim, elemPerCopyAlign);
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

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0) {
        ubSize = 192 * 1024;
    }
    uint32_t workingSetBytesPerElem = GetWorkingSetBytesPerElem(dt);
    uint32_t maxTileByUb =
        static_cast<uint32_t>(std::max<uint64_t>(elemPerCopyAlign, ubSize / workingSetBytesPerElem));
    maxTileByUb = AlignDown(maxTileByUb, elemPerCopyAlign);
    if (maxTileByUb == 0) {
        maxTileByUb = elemPerCopyAlign;
    }
    uint32_t defaultTile = AlignDown(std::max(elemPerCopyAlign, kDefaultTileBytes / dataWidth), elemPerCopyAlign);
    if (defaultTile == 0) {
        defaultTile = elemPerCopyAlign;
    }
    uint32_t tileLength = std::max(elemPerCopyAlign, std::min(blockLength, std::min(defaultTile, maxTileByUb)));

    tiling.set_totalLength(totalLength);
    tiling.set_blockLength(blockLength);
    tiling.set_lastBlockLength(lastBlockLength);
    tiling.set_tileLength(tileLength);
    tiling.set_copyAlignBytes(copyAlignBytes);

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

class Erfinv : public OpDef {
public:
    explicit Erfinv(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Erfinv);

}  // namespace ops
