#include "tensor_equal_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

namespace optiling {
constexpr uint32_t BLOCK_SIZE = 512;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t DMA_ALIGNMENT = 512;
constexpr uint32_t MIN_TILE_DATA_NUM = BLOCK_SIZE;
constexpr uint32_t MAX_BROADCAST_DIMS = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TensorEqualTilingData tiling;
    uint32_t tilingKey = 1;
    uint32_t dataTypeFlag = 0;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNumAiv();
    auto socVersion = ascendcPlatform.GetSocVersion();
    auto dataType = context->GetInputDesc(0)->GetDataType();

    if (dataType == ge::DT_BF16 &&
        (socVersion != platform_ascendc::SocVersion::ASCEND910B &&
         socVersion != platform_ascendc::SocVersion::ASCEND310B)) {
        return ge::GRAPH_FAILED;
    }

    auto selfShape = context->GetInputShape(0)->GetStorageShape();
    auto otherShape = context->GetInputShape(1)->GetStorageShape();
    uint32_t selfRank = static_cast<uint32_t>(selfShape.GetDimNum());
    uint32_t otherRank = static_cast<uint32_t>(otherShape.GetDimNum());
    uint32_t outRank = std::max(selfRank, otherRank);
    if (outRank == 0) {
        outRank = 1;
    }
    if (outRank > MAX_BROADCAST_DIMS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t outDimsArr[MAX_BROADCAST_DIMS] = {0};
    uint32_t selfPaddedArr[MAX_BROADCAST_DIMS] = {0};
    uint32_t otherPaddedArr[MAX_BROADCAST_DIMS] = {0};
    uint32_t selfStridesArr[MAX_BROADCAST_DIMS] = {0};
    uint32_t otherStridesArr[MAX_BROADCAST_DIMS] = {0};
    for (uint32_t i = 0; i < MAX_BROADCAST_DIMS; ++i) {
        outDimsArr[i] = 1;
        selfPaddedArr[i] = 1;
        otherPaddedArr[i] = 1;
    }

    for (uint32_t i = 0; i < outRank; ++i) {
        int32_t selfIdx = static_cast<int32_t>(i) - static_cast<int32_t>(outRank - selfRank);
        int32_t otherIdx = static_cast<int32_t>(i) - static_cast<int32_t>(outRank - otherRank);
        uint32_t sd = (selfIdx >= 0 && selfRank > 0) ? static_cast<uint32_t>(selfShape.GetDim(selfIdx)) : 1;
        uint32_t od = (otherIdx >= 0 && otherRank > 0) ? static_cast<uint32_t>(otherShape.GetDim(otherIdx)) : 1;
        if (sd != od && sd != 1 && od != 1) {
            return ge::GRAPH_FAILED;
        }
        selfPaddedArr[i] = sd;
        otherPaddedArr[i] = od;
        outDimsArr[i] = std::max(sd, od);
    }

    uint32_t selfStrideAcc = 1;
    uint32_t otherStrideAcc = 1;
    for (int32_t i = static_cast<int32_t>(outRank) - 1; i >= 0; --i) {
        selfStridesArr[i] = (selfPaddedArr[i] == 1) ? 0u : selfStrideAcc;
        otherStridesArr[i] = (otherPaddedArr[i] == 1) ? 0u : otherStrideAcc;
        selfStrideAcc *= selfPaddedArr[i];
        otherStrideAcc *= otherPaddedArr[i];
    }

    uint32_t broadcastNeeded = 0;
    for (uint32_t i = 0; i < outRank; ++i) {
        if (selfPaddedArr[i] != outDimsArr[i] || otherPaddedArr[i] != outDimsArr[i]) {
            broadcastNeeded = 1;
            break;
        }
    }

    uint64_t outNumel = 1;
    for (uint32_t i = 0; i < outRank; ++i) {
        outNumel *= outDimsArr[i];
    }

    uint64_t inputNum = outNumel;
    if (inputNum == 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(dataType, typeLength);

    switch (dataType) {
        case ge::DT_FLOAT16:
            dataTypeFlag = 0;
            tilingKey = 0;
            ascendcPlatform.ReserveLocalMemory(platform_ascendc::ReservedSize::RESERVED_SIZE_8K);
            break;
        case ge::DT_FLOAT:
            dataTypeFlag = 1;
            tilingKey = 1;
            break;
        case ge::DT_BF16:
            dataTypeFlag = 2;
            tilingKey = 2;
            ascendcPlatform.ReserveLocalMemory(platform_ascendc::ReservedSize::RESERVED_SIZE_8K);
            break;
        case ge::DT_INT32:
            dataTypeFlag = 3;
            tilingKey = 3;
            break;
        case ge::DT_INT16:
            dataTypeFlag = 4;
            tilingKey = 4;
            ascendcPlatform.ReserveLocalMemory(platform_ascendc::ReservedSize::RESERVED_SIZE_8K);
            break;
        case ge::DT_INT8:
            dataTypeFlag = 5;
            tilingKey = 5;
            ascendcPlatform.ReserveLocalMemory(platform_ascendc::ReservedSize::RESERVED_SIZE_8K);
            break;
        case ge::DT_UINT8:
            dataTypeFlag = 6;
            tilingKey = 6;
            ascendcPlatform.ReserveLocalMemory(platform_ascendc::ReservedSize::RESERVED_SIZE_8K);
            break;
        default:
            return ge::GRAPH_FAILED;
    }

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const uint32_t computeTypeSize = sizeof(float);
    const uint32_t maskTypeSize = sizeof(uint8_t);
    const uint32_t outTypeSize = sizeof(uint8_t);
    const bool isFloatType = (dataType == ge::DT_FLOAT);
    const bool isNarrowIntType = (dataType == ge::DT_INT16 ||
                                  dataType == ge::DT_INT8 ||
                                  dataType == ge::DT_UINT8);
    const uint32_t halfTypeSize = sizeof(uint16_t);
    const uint32_t memPerElement = isFloatType
        ? (2 * BUFFER_NUM * typeLength + BUFFER_NUM * outTypeSize + 4 * computeTypeSize + maskTypeSize)
        : (2 * BUFFER_NUM * typeLength + BUFFER_NUM * outTypeSize + 6 * computeTypeSize + maskTypeSize +
           (isNarrowIntType ? halfTypeSize : 0));

    uint64_t maxElements = std::max(static_cast<uint64_t>(ubSize / memPerElement), static_cast<uint64_t>(MIN_TILE_DATA_NUM));

    uint32_t tileBlockNum = maxElements / BLOCK_SIZE;
    tileBlockNum = tileBlockNum < 1 ? 1 : tileBlockNum;
    uint32_t tileDataNum = tileBlockNum * BLOCK_SIZE;

    uint32_t tileDataNumBytes = tileDataNum * typeLength;
    tileDataNumBytes = (tileDataNumBytes + DMA_ALIGNMENT - 1) / DMA_ALIGNMENT * DMA_ALIGNMENT;
    tileDataNum = tileDataNumBytes / typeLength;
    tileDataNum = std::max(std::min(tileDataNum, static_cast<uint32_t>(maxElements)), MIN_TILE_DATA_NUM);

    uint64_t inputLengthAlgin = ((inputNum + tileDataNum - 1) / tileDataNum) * tileDataNum;

    coreNum = std::max(std::min(coreNum, static_cast<uint32_t>(inputLengthAlgin / tileDataNum)), 1U);

    if (BLOCK_SIZE == 0 || tileDataNum < MIN_TILE_DATA_NUM) {
        return ge::GRAPH_FAILED;
    }

    uint64_t totalTiles = inputLengthAlgin / tileDataNum;
    uint64_t baseTilesPerCore = totalTiles / coreNum;
    uint64_t remainderTiles = totalTiles % coreNum;

    uint64_t smallCoreDataNum = baseTilesPerCore * tileDataNum;
    uint64_t bigCoreDataNum = (baseTilesPerCore + 1) * tileDataNum;

    auto alignComputeSize = [](uint32_t dataNum) -> uint32_t {
        constexpr uint32_t COMPUTE_ALIGNMENT = 512;
        uint32_t sizeInBytes = dataNum * sizeof(float);
        return (sizeInBytes + COMPUTE_ALIGNMENT - 1) / COMPUTE_ALIGNMENT * COMPUTE_ALIGNMENT / sizeof(float);
    };

    tiling.set_dataType(dataTypeFlag);
    tiling.set_typeLength(typeLength);
    tiling.set_rank(outRank);
    tiling.set_broadcastNeeded(broadcastNeeded);
    tiling.set_outDims(outDimsArr);
    tiling.set_selfStrides(selfStridesArr);
    tiling.set_otherStrides(otherStridesArr);
    if (broadcastNeeded) {
        tilingKey += 7;
    }
    tiling.set_smallCoreDataNum(static_cast<uint32_t>(smallCoreDataNum));
    tiling.set_bigCoreDataNum(static_cast<uint32_t>(bigCoreDataNum));
    tiling.set_tileDataNum(tileDataNum);
    tiling.set_smallTailDataNum(tileDataNum);
    tiling.set_bigTailDataNum(tileDataNum);
    tiling.set_finalSmallTileNum(static_cast<uint32_t>(baseTilesPerCore));
    tiling.set_finalBigTileNum(static_cast<uint32_t>(baseTilesPerCore + 1));
    tiling.set_tailBlockNum(static_cast<uint32_t>(remainderTiles));

    const uint32_t alignedTileDataNum = alignComputeSize(tileDataNum);
    tiling.set_bigprocessDataNum_computes(alignedTileDataNum);
    tiling.set_smallprocessDataNum_computes(alignedTileDataNum);
    tiling.set_tailbigprocessDataNum_computes(alignedTileDataNum);
    tiling.set_tailsmallprocessDataNum_computes(alignedTileDataNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = tileDataNum * sizeof(float) * BUFFER_NUM;

    context->SetBlockDim(coreNum);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1 = context->GetInputShape(0);
    const gert::Shape* x2 = context->GetInputShape(1);
    gert::Shape* y = context->GetOutputShape(0);

    size_t r1 = x1->GetDimNum();
    size_t r2 = x2->GetDimNum();
    size_t outRank = std::max(r1, r2);
    if (outRank == 0) {
        y->SetDimNum(0);
        return GRAPH_SUCCESS;
    }
    y->SetDimNum(outRank);
    for (size_t i = 0; i < outRank; ++i) {
        int64_t d1 = (i + r1 >= outRank) ? x1->GetDim(static_cast<int64_t>(i) - static_cast<int64_t>(outRank - r1)) : 1;
        int64_t d2 = (i + r2 >= outRank) ? x2->GetDim(static_cast<int64_t>(i) - static_cast<int64_t>(outRank - r2)) : 1;
        if (d1 != d2 && d1 != 1 && d2 != 1) {
            return GRAPH_FAILED;
        }
        y->SetDim(i, std::max(d1, d2));
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_BOOL);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class TensorEqual : public OpDef {
public:
    explicit TensorEqual(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT,
                       ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT,
                       ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL,
                       ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b")
            .AddConfig("ascend310b");
    }
};

OP_ADD(TensorEqual);
} // namespace ops
