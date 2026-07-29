#include "erfinv_tiling.h"

#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

constexpr uint64_t BLOCK_ELEMENTS = 256;
constexpr uint32_t DEFAULT_TILE_ELEMENTS = 4096;

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

}  // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const int64_t shapeSize = inputShape->GetStorageShape().GetShapeSize();
    if (shapeSize < 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t totalLength = static_cast<uint64_t>(shapeSize);

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t availableCores = platform.GetCoreNumAiv();
    if (availableCores == 0) {
        availableCores = 1;
    }

    const uint64_t totalBlocks = CeilDiv(totalLength, BLOCK_ELEMENTS);
    const uint32_t usedCores = totalBlocks == 0
        ? 1
        : static_cast<uint32_t>(std::min<uint64_t>(availableCores, totalBlocks));

    ErfinvTilingData tiling;
    tiling.set_totalLength(totalLength);
    tiling.set_blocksPerCore(totalBlocks / usedCores);
    tiling.set_extraCoreBlocks(static_cast<uint32_t>(totalBlocks % usedCores));
    tiling.set_tileLength(DEFAULT_TILE_ELEMENTS);

    context->SetBlockDim(usedCores);
    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = 0;

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
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
