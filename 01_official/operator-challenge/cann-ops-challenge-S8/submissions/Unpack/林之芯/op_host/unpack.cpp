
#include "unpack_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>

namespace {
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_RESERVE = 16 * 1024;
constexpr uint32_t MAX_BLOCK_COUNT = 4095;
constexpr uint32_t MAX_REG_FIELD = 65535;
constexpr uint32_t MAX_PAD_BLOCK_LEN = 2097151;
constexpr uint64_t PARALLEL_THRESHOLD_BYTES = 64 * 1024;
constexpr uint64_t INNER_SPLIT_THRESHOLD_BYTES = 4 * 1024;

constexpr uint32_t MODE_DIRECT = 0;
constexpr uint32_t MODE_STRIDED_ALIGNED = 1;
constexpr uint32_t MODE_STRIDED_PAD = 2;
constexpr uint32_t MODE_READ_ALL_ALIGNED = 3;
constexpr uint32_t MODE_COPY_ALL = 4;
constexpr uint32_t MODE_GATHER_INNER1 = 5;
constexpr uint32_t MODE_READ_ALL_PAD = 6;

inline uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

inline uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return align == 0 ? value : CeilDiv(value, align) * align;
}

inline uint64_t AlignDown(uint64_t value, uint64_t align)
{
    return align == 0 ? value : (value / align) * align;
}

uint32_t TuneTileOuter(uint64_t outer, uint64_t parallelDim, uint64_t maxTile,
                       uint32_t sysCoreNum, bool balanceForParallel)
{
    if (outer == 0 || maxTile == 0) {
        return 1;
    }

    uint64_t tile = std::min<uint64_t>(maxTile, MAX_BLOCK_COUNT);
    if (!balanceForParallel) {
        return static_cast<uint32_t>(std::max<uint64_t>(tile, 1));
    }

    uint64_t targetTasks = static_cast<uint64_t>(std::max<uint32_t>(sysCoreNum, 1)) * 2;
    uint64_t curTasks = parallelDim * CeilDiv(outer, tile);
    if (parallelDim > 0 && curTasks < targetTasks) {
        uint64_t desiredOuterTiles = CeilDiv(targetTasks, parallelDim);
        uint64_t balancedTile = CeilDiv(outer, desiredOuterTiles);
        tile = std::min(tile, std::max<uint64_t>(balancedTile, 1));
    }
    return static_cast<uint32_t>(std::max<uint64_t>(tile, 1));
}

uint32_t RelaxTileOuterToOneWave(uint64_t outer, uint64_t parallelDim, uint64_t maxTile,
                                 uint32_t tileOuter, uint32_t sysCoreNum)
{
    if (outer == 0 || parallelDim == 0 || maxTile == 0) {
        return tileOuter;
    }
    uint64_t targetOuterTiles = CeilDiv(sysCoreNum, parallelDim);
    targetOuterTiles = std::max<uint64_t>(targetOuterTiles, 1);
    uint64_t oneWaveTile = CeilDiv(outer, targetOuterTiles);
    if (oneWaveTile > tileOuter && oneWaveTile <= maxTile) {
        return static_cast<uint32_t>(std::max<uint64_t>(oneWaveTile, 1));
    }
    return tileOuter;
}

uint32_t TuneTileInner(uint64_t outer, uint64_t num, uint64_t inner, uint64_t maxTileInner,
                       uint32_t typeBytes, uint32_t sysCoreNum, bool balanceForParallel)
{
    if (inner == 0 || maxTileInner == 0) {
        return 1;
    }

    uint64_t tile = std::min<uint64_t>(inner, maxTileInner);
    uint64_t baseTasks = outer * num;
    uint64_t targetTasks = static_cast<uint64_t>(std::max<uint32_t>(sysCoreNum, 1)) * 2;
    if (balanceForParallel && baseTasks > 0 && baseTasks * CeilDiv(inner, tile) < targetTasks) {
        uint64_t desiredInnerTiles = CeilDiv(targetTasks, baseTasks);
        uint64_t balancedTile = CeilDiv(inner, desiredInnerTiles);
        tile = std::min(tile, std::max<uint64_t>(balancedTile, 1));
    }

    uint64_t elemsPerBlock = std::max<uint64_t>(BLOCK_SIZE / typeBytes, 1);
    if (tile >= elemsPerBlock) {
        tile = std::max<uint64_t>(AlignDown(tile, elemsPerBlock), elemsPerBlock);
    }
    return static_cast<uint32_t>(std::max<uint64_t>(tile, 1));
}

uint32_t GetTilingKey(uint32_t dtypeKey, uint32_t mode)
{
    return mode * 16 + dtypeKey;
}

uint32_t GetMappedDType(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:   return 1;
        case ge::DT_FLOAT16: return 2;
        case ge::DT_BF16:    return 3;
        case ge::DT_INT32:   return 4;
        case ge::DT_INT16:   return 5;
        case ge::DT_INT8:    return 6;
        case ge::DT_UINT8:   return 7;
        case ge::DT_BOOL:    return 8;
        default:             return 0;
    }
}
}  // namespace


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    UnpackTilingData tiling;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysCoreNum = ascendcPlatform.GetCoreNum();
    sysCoreNum = std::max<uint32_t>(sysCoreNum, 1);

    const gert::StorageShape* in_shape = context->GetInputShape(0);
    int64_t axis = *context->GetAttrs()->GetInt(1);
    uint32_t dim_cnt = in_shape->GetStorageShape().GetDimNum();

    if (axis < 0) { axis += dim_cnt; }
    if (axis < 0 || axis >= static_cast<int64_t>(dim_cnt)) {
        return ge::GRAPH_FAILED;
    }

    uint64_t outer = 1;
    uint64_t inner = 1;
    for (uint32_t i = 0; i < static_cast<uint32_t>(axis); ++i) {
        outer *= static_cast<uint64_t>(std::max<int64_t>(in_shape->GetStorageShape().GetDim(i), 0));
    }
    uint64_t num = static_cast<uint64_t>(std::max<int64_t>(in_shape->GetStorageShape().GetDim(axis), 0));
    for (uint32_t i = static_cast<uint32_t>(axis) + 1; i < dim_cnt; ++i) {
        inner *= static_cast<uint64_t>(std::max<int64_t>(in_shape->GetStorageShape().GetDim(i), 0));
    }

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t usableUb = ubSize > UB_RESERVE ? ubSize - UB_RESERVE : ubSize;
    uint64_t perBufferBytes = AlignDown(usableUb / BUFFER_NUM, BLOCK_SIZE);
    perBufferBytes = std::max<uint64_t>(perBufferBytes, BLOCK_SIZE);

    uint32_t typeBytes = ge::GetSizeByDataType(context->GetInputDesc(0)->GetDataType());
    if (typeBytes == 0) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType inputDType = context->GetInputDesc(0)->GetDataType();
    bool gatherInner1Supported = inputDType == ge::DT_FLOAT ||
                                 inputDType == ge::DT_FLOAT16 ||
                                 inputDType == ge::DT_BF16 ||
                                 inputDType == ge::DT_INT32 ||
                                 inputDType == ge::DT_INT16 ||
                                 inputDType == ge::DT_INT8 ||
                                 inputDType == ge::DT_UINT8 ||
                                 inputDType == ge::DT_BOOL;

    uint64_t innerBytes = inner * typeBytes;
    uint64_t totalBytes = outer * num * innerBytes;
    uint64_t packedInnerBytes = AlignUp(innerBytes, BLOCK_SIZE);
    uint64_t innerBlocks = innerBytes / BLOCK_SIZE;
    bool innerAligned = (innerBytes % BLOCK_SIZE) == 0;
    bool regStrideOk = innerAligned && innerBlocks > 0 && innerBlocks <= MAX_REG_FIELD &&
                       (num == 0 || (num - 1) * innerBlocks <= MAX_REG_FIELD);
    bool padStrideOk = innerBytes > 0 && innerBytes <= MAX_PAD_BLOCK_LEN &&
                       packedInnerBytes <= perBufferBytes &&
                       (num == 0 || (num - 1) * innerBytes <= UINT32_MAX);

    uint32_t mode = MODE_DIRECT;
    uint32_t tileOuter = 1;
    uint32_t tileInner = 1;
    uint32_t tileOuterNum = 0;
    uint32_t tileInnerNum = 0;
    uint32_t ubBufferSize = BLOCK_SIZE;
    uint32_t ubOutBufferSize = BLOCK_SIZE;
    uint32_t coreNum = 1;

    auto chooseDirect = [&]() {
        uint64_t maxTileInner = perBufferBytes / typeBytes;
        tileInner = TuneTileInner(outer, num, inner, maxTileInner, typeBytes, sysCoreNum,
                                  true);
        tileInnerNum = static_cast<uint32_t>(CeilDiv(inner, tileInner));
        tileOuter = 1;
        tileOuterNum = 1;
        ubBufferSize = static_cast<uint32_t>(std::max<uint64_t>(AlignUp(tileInner * static_cast<uint64_t>(typeBytes), BLOCK_SIZE), BLOCK_SIZE));
        ubOutBufferSize = ubBufferSize;
        uint64_t taskNum = outer * num * std::max<uint32_t>(tileInnerNum, 1);
        coreNum = static_cast<uint32_t>(std::min<uint64_t>(sysCoreNum, std::max<uint64_t>(taskNum, 1)));
        mode = MODE_DIRECT;
    };

    auto chooseCopyAll = [&]() {
        uint64_t totalElems = outer * inner;
        uint64_t maxTileElems = perBufferBytes / typeBytes;
        tileInner = TuneTileInner(1, 1, totalElems, maxTileElems, typeBytes, sysCoreNum,
                                  true);
        uint64_t oneWaveTile = CeilDiv(totalElems, sysCoreNum);
        if (oneWaveTile > tileInner && oneWaveTile <= maxTileElems) {
            uint64_t elemsPerBlock = std::max<uint64_t>(BLOCK_SIZE / typeBytes, 1);
            if (oneWaveTile >= elemsPerBlock) {
                oneWaveTile = std::max<uint64_t>(AlignDown(oneWaveTile, elemsPerBlock),
                                                 elemsPerBlock);
            }
            tileInner = static_cast<uint32_t>(std::max<uint64_t>(oneWaveTile, 1));
        }
        tileInnerNum = static_cast<uint32_t>(CeilDiv(totalElems, tileInner));
        tileOuter = 1;
        tileOuterNum = 1;
        ubBufferSize = static_cast<uint32_t>(std::max<uint64_t>(
            AlignUp(tileInner * static_cast<uint64_t>(typeBytes), BLOCK_SIZE), BLOCK_SIZE));
        ubOutBufferSize = ubBufferSize;
        coreNum = static_cast<uint32_t>(std::min<uint64_t>(
            sysCoreNum, std::max<uint64_t>(tileInnerNum, 1)));
        mode = MODE_COPY_ALL;
    };

    auto chooseGatherInner1 = [&]() {
        uint64_t maxRows = 1;
        uint64_t inputBytesPerRow = num * static_cast<uint64_t>(typeBytes);
        for (uint64_t rows = 1; rows <= MAX_BLOCK_COUNT && rows <= outer; ++rows) {
            uint64_t inQueueBytes = AlignUp(rows * inputBytesPerRow, BLOCK_SIZE);
            uint64_t outQueueBytes = AlignUp(rows * static_cast<uint64_t>(typeBytes), BLOCK_SIZE);
            uint64_t offsetBytes = AlignUp(rows * static_cast<uint64_t>(sizeof(uint32_t)), BLOCK_SIZE);
            if (inQueueBytes + outQueueBytes + offsetBytes > usableUb) {
                break;
            }
            maxRows = rows;
        }

        tileOuter = TuneTileOuter(outer, 1, maxRows, sysCoreNum, true);
        tileOuter = RelaxTileOuterToOneWave(outer, 1, maxRows, tileOuter, sysCoreNum);
        tileOuterNum = static_cast<uint32_t>(CeilDiv(outer, tileOuter));
        tileInner = 1;
        tileInnerNum = 1;
        ubBufferSize = static_cast<uint32_t>(std::max<uint64_t>(
            AlignUp(tileOuter * inputBytesPerRow, BLOCK_SIZE), BLOCK_SIZE));
        ubOutBufferSize = static_cast<uint32_t>(std::max<uint64_t>(
            AlignUp(tileOuter * static_cast<uint64_t>(typeBytes), BLOCK_SIZE), BLOCK_SIZE));
        coreNum = static_cast<uint32_t>(std::min<uint64_t>(
            sysCoreNum, std::max<uint64_t>(tileOuterNum, 1)));
        mode = MODE_GATHER_INNER1;
    };

    if (outer == 0 || num == 0 || inner == 0) {
        chooseDirect();
        tileOuterNum = 0;
        tileInnerNum = 0;
        coreNum = 1;
    } else {
        bool needInnerSplit = innerBytes > perBufferBytes ||
                              ((outer * num) < sysCoreNum && innerBytes >= INNER_SPLIT_THRESHOLD_BYTES) ||
                              (axis == 0 && innerBytes >= INNER_SPLIT_THRESHOLD_BYTES);

        bool readAllOk = innerAligned && num > 1 && num <= 8 &&
                         (num * innerBytes) <= perBufferBytes &&
                         innerBlocks > 0 && innerBlocks <= MAX_REG_FIELD &&
                         (num - 1) * innerBlocks <= MAX_REG_FIELD;
        bool readAllPadOk = !innerAligned && num > 1 && num <= 16 &&
                            innerBytes > 0 && innerBytes <= MAX_PAD_BLOCK_LEN &&
                            (num * innerBytes) <= perBufferBytes &&
                            (num * innerBytes) >= packedInnerBytes &&
                            (num * innerBytes - packedInnerBytes) <= UINT32_MAX;

        if (num == 1) {
            chooseCopyAll();
        } else if (inner == 1 && gatherInner1Supported && totalBytes >= PARALLEL_THRESHOLD_BYTES &&
                   (AlignUp(num * static_cast<uint64_t>(typeBytes), BLOCK_SIZE) +
                    AlignUp(static_cast<uint64_t>(typeBytes), BLOCK_SIZE) +
                    AlignUp(static_cast<uint64_t>(sizeof(uint32_t)), BLOCK_SIZE)) <= usableUb &&
                   num * static_cast<uint64_t>(typeBytes) <= static_cast<uint64_t>(INT32_MAX)) {
            chooseGatherInner1();
        } else if (needInnerSplit) {
            chooseDirect();
        } else if (readAllOk && totalBytes >= PARALLEL_THRESHOLD_BYTES &&
                   outer >= std::max<uint32_t>(sysCoreNum / 2, 1)) {
            uint64_t maxTile = perBufferBytes / (num * innerBytes);
            tileOuter = TuneTileOuter(outer, 1, maxTile, sysCoreNum,
                                      totalBytes >= PARALLEL_THRESHOLD_BYTES);
            tileOuter = RelaxTileOuterToOneWave(outer, 1, maxTile, tileOuter, sysCoreNum);
            tileOuterNum = static_cast<uint32_t>(CeilDiv(outer, tileOuter));
            tileInner = 1;
            tileInnerNum = 1;
            ubBufferSize = static_cast<uint32_t>(std::max<uint64_t>(tileOuter * num * innerBytes, BLOCK_SIZE));
            ubOutBufferSize = ubBufferSize;
            coreNum = static_cast<uint32_t>(std::min<uint64_t>(sysCoreNum, std::max<uint64_t>(tileOuterNum, 1)));
            mode = MODE_READ_ALL_ALIGNED;
        } else if (readAllPadOk && totalBytes >= PARALLEL_THRESHOLD_BYTES &&
                   outer >= std::max<uint32_t>(sysCoreNum / 2, 1)) {
            uint64_t maxTile = perBufferBytes / (num * innerBytes);
            tileOuter = TuneTileOuter(outer, 1, maxTile, sysCoreNum,
                                      totalBytes >= PARALLEL_THRESHOLD_BYTES);
            tileOuter = RelaxTileOuterToOneWave(outer, 1, maxTile, tileOuter, sysCoreNum);
            tileOuterNum = static_cast<uint32_t>(CeilDiv(outer, tileOuter));
            tileInner = 1;
            tileInnerNum = 1;
            ubBufferSize = static_cast<uint32_t>(std::max<uint64_t>(
                AlignUp(tileOuter * num * innerBytes, BLOCK_SIZE), BLOCK_SIZE));
            ubOutBufferSize = ubBufferSize;
            coreNum = static_cast<uint32_t>(std::min<uint64_t>(
                sysCoreNum, std::max<uint64_t>(tileOuterNum, 1)));
            mode = MODE_READ_ALL_PAD;
        } else if (regStrideOk && innerBytes <= perBufferBytes) {
            uint64_t maxTile = perBufferBytes / innerBytes;
            tileOuter = TuneTileOuter(outer, num, maxTile, sysCoreNum,
                                      totalBytes >= PARALLEL_THRESHOLD_BYTES);
            tileOuter = RelaxTileOuterToOneWave(outer, num, maxTile, tileOuter, sysCoreNum);
            tileOuterNum = static_cast<uint32_t>(CeilDiv(outer, tileOuter));
            tileInner = 1;
            tileInnerNum = 1;
            ubBufferSize = static_cast<uint32_t>(std::max<uint64_t>(tileOuter * innerBytes, BLOCK_SIZE));
            ubOutBufferSize = ubBufferSize;
            uint64_t taskNum = num * tileOuterNum;
            coreNum = static_cast<uint32_t>(std::min<uint64_t>(sysCoreNum, std::max<uint64_t>(taskNum, 1)));
            mode = MODE_STRIDED_ALIGNED;
        } else if (padStrideOk) {
            uint64_t maxTile = perBufferBytes / packedInnerBytes;
            tileOuter = TuneTileOuter(outer, num, maxTile, sysCoreNum, true);
            tileOuter = RelaxTileOuterToOneWave(outer, num, maxTile, tileOuter, sysCoreNum);
            tileOuterNum = static_cast<uint32_t>(CeilDiv(outer, tileOuter));
            tileInner = 1;
            tileInnerNum = 1;
            ubBufferSize = static_cast<uint32_t>(std::max<uint64_t>(
                tileOuter * packedInnerBytes, BLOCK_SIZE));
            ubOutBufferSize = ubBufferSize;
            uint64_t taskNum = num * tileOuterNum;
            coreNum = static_cast<uint32_t>(std::min<uint64_t>(sysCoreNum, std::max<uint64_t>(taskNum, 1)));
            mode = MODE_STRIDED_PAD;
        } else {
            chooseDirect();
        }
    }

    tiling.set_outer(outer);
    tiling.set_num(num);
    tiling.set_inner(inner);
    tiling.set_coreNum(coreNum);
    tiling.set_tile_outer(tileOuter);
    tiling.set_tile_inner(tileInner);
    tiling.set_tile_outer_num(tileOuterNum);
    tiling.set_tile_inner_num(tileInnerNum);
    tiling.set_ub_buffer_size(ubBufferSize);
    tiling.set_ub_out_buffer_size(ubOutBufferSize);
    context->SetBlockDim(coreNum);

    uint32_t mappedDType = GetMappedDType(context->GetInputDesc(0)->GetDataType());
    if (mappedDType == 0) {
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(GetTilingKey(mappedDType, mode));

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);

    int64_t axis = 0;
    if (context->GetAttrs() != nullptr && context->GetAttrs()->GetInt(1) != nullptr) {
        axis = *context->GetAttrs()->GetInt(1);
    }
    uint32_t dim_cnt = x1_shape->GetDimNum();
    if (axis < 0) {
        axis += dim_cnt;
    }
    if (axis < 0 || axis >= static_cast<int64_t>(dim_cnt)) {
        return GRAPH_FAILED;
    }

    y_shape->SetDimNum(0);
    for (uint32_t i = 0; i < dim_cnt; ++i) {
        if (i != static_cast<uint32_t>(axis)) {
            y_shape->AppendDim(x1_shape->GetDim(i));
        }
    }
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class Unpack : public OpDef {
public:
    explicit Unpack(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("num").Int();
        this->Attr("axis").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Unpack);
}
