#include "atanh_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint32_t kChampionBlockBytes = 1024U;
constexpr uint64_t kUbReserveBytes = 8ULL * 1024ULL;
constexpr uint64_t kUbFallbackBytes = 176ULL * 1024ULL;

static inline uint32_t MinU32(uint32_t lhs, uint32_t rhs)
{
    return (lhs < rhs) ? lhs : rhs;
}

static inline uint32_t CeilDivU32(uint32_t lhs, uint32_t rhs)
{
    return rhs == 0U ? 0U : ((lhs + rhs - 1U) / rhs);
}

static inline uint32_t DtypeBytes(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2U;
        case ge::DT_FLOAT:
        default:
            return 4U;
    }
}

static inline uint32_t SelectBlockDim(uint32_t elemCount)
{
    if (elemCount >= (1U << 20)) {
        return 40U;
    }
    if (elemCount >= (1U << 18)) {
        return 16U;
    }
    if (elemCount >= (1U << 16)) {
        return 8U;
    }
    if (elemCount >= (1U << 13)) {
        return 4U;
    }
    return 1U;
}
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* xShape = context->GetInputShape(0);
    uint64_t dataSize = 1;
    for (int i = 0; i < xShape->GetStorageShape().GetDimNum(); ++i) {
        dataSize *= static_cast<uint64_t>(xShape->GetStorageShape().GetDim(i));
        if (dataSize > UINT32_MAX) {
            return ge::GRAPH_FAILED;
        }
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t maxBlockDim = platform.GetCoreNumAiv();
    if (maxBlockDim == 0U) {
        maxBlockDim = 40U;
    }

    uint64_t ubSizeBytes = 0U;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeBytes);
    uint64_t usableUbBytes = ubSizeBytes > kUbReserveBytes ? (ubSizeBytes - kUbReserveBytes) : kUbFallbackBytes;
    if (usableUbBytes == 0U || usableUbBytes > UINT32_MAX) {
        usableUbBytes = kUbFallbackBytes;
    }

    const ge::DataType inputType = context->GetInputDesc(0)->GetDataType();
    const uint32_t elemBytes = DtypeBytes(inputType);
    uint32_t splitUnit = kChampionBlockBytes / elemBytes;
    if (splitUnit == 0U) {
        splitUnit = 1U;
    }
    const uint32_t totalBlocks = CeilDivU32(static_cast<uint32_t>(dataSize), splitUnit);
    uint32_t blockDim = SelectBlockDim(static_cast<uint32_t>(dataSize));
    blockDim = MinU32(blockDim, maxBlockDim);
    blockDim = MinU32(blockDim, totalBlocks == 0U ? 1U : totalBlocks);
    if (blockDim == 0U) {
        blockDim = 1U;
    }

    const uint32_t bytesPerElem = elemBytes + 2U * sizeof(float) + elemBytes;
    const uint64_t bytesPerChampionBlock = static_cast<uint64_t>(splitUnit) * bytesPerElem;
    uint32_t tileBlocks = bytesPerChampionBlock == 0U
        ? 1U
        : static_cast<uint32_t>(usableUbBytes / bytesPerChampionBlock);
    if (tileBlocks == 0U) {
        tileBlocks = 1U;
    }
    uint32_t tileLength = tileBlocks * splitUnit;
    if (tileLength == 0U) {
        tileLength = splitUnit;
    }

    AtanhTilingData tiling;
    tiling.set_size(static_cast<uint32_t>(dataSize));
    tiling.set_tile_length(tileLength);
    tiling.set_split_unit(splitUnit);
    tiling.set_per_core_block(totalBlocks / blockDim);
    tiling.set_last_core_block(totalBlocks % blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputType = context->GetInputDataType(0);
    switch (inputType) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_FLOAT:
            context->SetOutputDataType(0, inputType);
            break;
        default:
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Atanh : public OpDef {
public:
    explicit Atanh(const char* name) : OpDef(name)
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
        this->AICore().AddConfig("ascend310b");
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Atanh);
}
