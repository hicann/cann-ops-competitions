#include "is_nan_tiling.h"

#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

constexpr uint32_t kReservedUbBytes = 0;
constexpr uint32_t kPreferredBlockBytes = 64 * 1024;
constexpr uint32_t kAlignBytes = 32;

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

uint32_t GetBufferBytesPerElement(ge::DataType dt)
{
    uint32_t inputBytes = GetDataTypeSize(dt);
    uint32_t outputBytes = sizeof(uint8_t);
    uint32_t maskBytes = sizeof(uint8_t);
    uint32_t boolValueBytes = sizeof(uint16_t);
    return inputBytes + outputBytes + maskBytes + boolValueBytes;
}

uint32_t GetTilingKey(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT16:
            return 0;
        case ge::DT_BF16:
            return 1;
        case ge::DT_FLOAT:
            return 2;
        default:
            return 0;
    }
}

uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return ((value + align - 1) / align) * align;
}

}  // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    IsNanTilingData tiling;

    auto dt = context->GetInputTensor(0)->GetDataType();
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
    uint64_t totalBytes64 = static_cast<uint64_t>(totalLength) * dataWidth;
    uint32_t elemAlign = std::max(1U, 512U);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = std::max(1U, ascendcPlatform.GetCoreNumAiv());
    uint32_t neededCoreNum = std::max(1U, static_cast<uint32_t>((totalBytes64 + kPreferredBlockBytes - 1) / kPreferredBlockBytes));
    uint32_t blockDim = std::min(coreNum, neededCoreNum);

    uint32_t blockLength = AlignUp((totalLength + blockDim - 1) / blockDim, elemAlign);
    uint32_t lastBlockLength = totalLength - blockLength * (blockDim - 1);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t usableUbBytes = ubSize > kReservedUbBytes ? (ubSize - kReservedUbBytes) : ubSize;
    uint64_t ubBytesPerTileElem = static_cast<uint64_t>(GetBufferBytesPerElement(dt)) * 2U;
    uint32_t maxTileByUb = static_cast<uint32_t>(
        std::max<uint64_t>(elemAlign, usableUbBytes / ubBytesPerTileElem));
    maxTileByUb = (maxTileByUb / elemAlign) * elemAlign;
    uint32_t tileLength = std::max(elemAlign, std::min(blockLength, maxTileByUb));

    tiling.set_totalLength(totalLength);
    tiling.set_blockLength(blockLength);
    tiling.set_lastBlockLength(lastBlockLength);
    tiling.set_tileLength(tileLength);

    context->SetBlockDim(blockDim);
    context->SetTilingKey(GetTilingKey(dt));
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
    (void)context;
    context->SetOutputDataType(0, ge::DT_BOOL);
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class IsNan : public OpDef {
public:
    explicit IsNan(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(IsNan);

}  // namespace ops
