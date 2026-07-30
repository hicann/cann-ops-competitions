#include "../op_kernel/atanh_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>

namespace optiling {
namespace {
constexpr uint32_t FLOAT_DENSITY_TILE = 4096;
constexpr uint32_t FLOAT_REGULAR_TILE = 9216;
constexpr uint32_t COMPACT_DENSITY_TILE = 4096;
constexpr uint32_t COMPACT_REGULAR_TILE = 12032;
constexpr uint32_t INT32_DENSITY_TILE = 4096;
constexpr uint32_t INT32_REGULAR_TILE = 7680;
constexpr uint32_t SMALL_TILE_LENGTH = 8192;
constexpr uint32_t SMALL_SINGLE_CORE_LIMIT = 2048;
constexpr uint32_t MIN_MULTI_CORE = 2;
constexpr uint32_t MAX_SINGLE_SHOT_CORES = 16; // empirical: cap to avoid HBM contention at 32+ cores SingleShot
constexpr uint32_t GM_ALIGN_BYTES = 512;

uint32_t GetShapeSize(const gert::Shape &shape)
{
    uint64_t size = 1;
    for (int32_t i = 0; i < shape.GetDimNum(); ++i) {
        size *= static_cast<uint64_t>(shape.GetDim(i));
    }
    return static_cast<uint32_t>(size);
}

uint32_t AlignUp(uint32_t value, uint32_t align)
{
    if (align == 0) {
        return value;
    }
    return ((value + align - 1) / align) * align;
}

uint32_t GetTypeSize(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_INT16:
            return 2;
        case ge::DT_UINT8:
        case ge::DT_INT8:
            return 1;
        default:
            return 4;
    }
}

uint32_t SelectDensityTile(ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT) return FLOAT_DENSITY_TILE;
    if (dtype == ge::DT_INT32) return INT32_DENSITY_TILE;
    return COMPACT_DENSITY_TILE;
}

uint32_t SelectRegularTile(ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT) return FLOAT_REGULAR_TILE;
    if (dtype == ge::DT_INT32) return INT32_REGULAR_TILE;
    return COMPACT_REGULAR_TILE;
}
} // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    AtanhTilingData tiling;
    const gert::StorageShape *inputStorageShape = context->GetInputShape(0);
    if (inputStorageShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t totalLength = GetShapeSize(inputStorageShape->GetStorageShape());
    const ge::DataType inputDtype = context->GetInputDesc(0)->GetDataType();
    const uint32_t typeSize = GetTypeSize(inputDtype);
    const uint32_t densityTile = SelectDensityTile(inputDtype);
    const uint32_t regularTile = SelectRegularTile(inputDtype);
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platform.GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = platform.GetCoreNum();
    }
    if (coreNum == 0) {
        coreNum = 1;
    }

    if (totalLength <= SMALL_SINGLE_CORE_LIMIT) {
        tiling.set_totalLength(totalLength);
        tiling.set_blockLength(totalLength);
        tiling.set_tileLength(SMALL_TILE_LENGTH);
        context->SetBlockDim(1);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }

    uint32_t usedCoreNum = (totalLength + densityTile - 1) / densityTile;
    usedCoreNum = std::max<uint32_t>(MIN_MULTI_CORE, std::min<uint32_t>(coreNum, usedCoreNum));
    const uint32_t blockAlignElems = std::max<uint32_t>(1, GM_ALIGN_BYTES / typeSize);
    uint32_t blockLength = AlignUp((totalLength + usedCoreNum - 1) / usedCoreNum, blockAlignElems);

    if (blockLength <= SMALL_TILE_LENGTH && usedCoreNum > MAX_SINGLE_SHOT_CORES) {
        usedCoreNum = MAX_SINGLE_SHOT_CORES;
        blockLength = AlignUp((totalLength + usedCoreNum - 1) / usedCoreNum, blockAlignElems);
    }

    uint32_t tileLength;
    if (blockLength <= SMALL_TILE_LENGTH) {
        tileLength = SMALL_TILE_LENGTH;
    } else {
        tileLength = regularTile;
    }

    tiling.set_totalLength(totalLength);
    tiling.set_blockLength(blockLength);
    tiling.set_tileLength(tileLength);
    context->SetBlockDim(usedCoreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Atanh : public OpDef {
public:
    explicit Atanh(const char *name) : OpDef(name)
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
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8,
                       ge::DT_INT8})
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
} // namespace ops
