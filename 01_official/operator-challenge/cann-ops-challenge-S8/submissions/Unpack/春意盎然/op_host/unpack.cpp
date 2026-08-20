#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/unpack_tiling.h"

#include <algorithm>
#include <cstdint>

namespace optiling {
namespace {
// Per-buffer tile size in BYTES. Path B/C copyQueue uses BUFFER_NUM=2, so 96KB × 2 =
// 192KB UB total (full budget). BUFFER_NUM=3 with 64KB tile was tested and is slightly
// worse on 256MB cases — MTE2 is HBM-saturated, so deeper pipeline doesn't help.
constexpr uint32_t TILE_BYTES = 98304;  // 96 KB × 2 buffers = 192 KB UB
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t MAX_BLOCK_COUNT = 4095;

uint32_t GetTypeSize(const ge::DataType dtype)
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
        case ge::DT_BOOL:
        default:
            return 1;
    }
}

uint64_t ProductRange(const gert::Shape &shape, int32_t begin, int32_t end)
{
    uint64_t value = 1;
    for (int32_t i = begin; i < end; ++i) {
        value *= static_cast<uint64_t>(shape.GetDim(i));
    }
    return value;
}

uint32_t AlignUp(uint32_t value, uint32_t align)
{
    if (align == 0) return value;
    return ((value + align - 1) / align) * align;
}
} // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    UnpackTilingData tiling;
    const gert::StorageShape *inputStorageShape = context->GetInputShape(0);
    if (inputStorageShape == nullptr) return ge::GRAPH_FAILED;

    const gert::Shape &inputShape = inputStorageShape->GetStorageShape();
    const int32_t rank = inputShape.GetDimNum();
    if (rank <= 0) return ge::GRAPH_FAILED;

    const auto attrs = context->GetAttrs();
    int64_t numAttr = inputShape.GetDim(0);
    int64_t axisAttr = 0;
    if (attrs != nullptr) {
        const int64_t *numPtr = attrs->GetAttrPointer<int64_t>(0);
        const int64_t *axisPtr = attrs->GetAttrPointer<int64_t>(1);
        if (numPtr != nullptr) numAttr = *numPtr;
        if (axisPtr != nullptr) axisAttr = *axisPtr;
    }

    const int32_t axis = axisAttr < 0 ? static_cast<int32_t>(axisAttr + rank) : static_cast<int32_t>(axisAttr);
    if (axis < 0 || axis >= rank) return ge::GRAPH_FAILED;

    const uint64_t axisLength64 = static_cast<uint64_t>(inputShape.GetDim(axis));
    if (axisLength64 == 0 || static_cast<uint64_t>(numAttr) != axisLength64) return ge::GRAPH_FAILED;

    const uint64_t outerLength64 = ProductRange(inputShape, 0, axis);
    const uint64_t innerLength64 = ProductRange(inputShape, axis + 1, rank);
    const uint64_t totalLength64 = outerLength64 * axisLength64 * innerLength64;
    if (totalLength64 == 0 || totalLength64 > UINT32_MAX || outerLength64 > UINT32_MAX
        || axisLength64 > UINT32_MAX || innerLength64 > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platform.GetCoreNumAiv();
    if (coreNum == 0) coreNum = platform.GetCoreNum();
    if (coreNum == 0) coreNum = 1;

    const uint32_t typeSize = GetTypeSize(context->GetInputDesc(0)->GetDataType());
    const uint32_t axisLength = static_cast<uint32_t>(axisLength64);
    const uint32_t outerLength = static_cast<uint32_t>(outerLength64);
    const uint32_t innerLength = static_cast<uint32_t>(innerLength64);
    const uint32_t totalLength = static_cast<uint32_t>(totalLength64);

    const uint32_t tileLength = TILE_BYTES;
    const uint32_t innerBytes = innerLength * typeSize;

    uint32_t usedCoreNum = 1;
    uint32_t blockLength = 1;
    uint32_t space = 1;          // # of divisible work-units this path splits across cores
    bool pathCAlign = false;     // Path C: blockLength must be 512B-aligned
    bool applyLaunchCap = true;  // v9.33: see launch-cap below

    if (innerLength == 1) {
        const auto dtype = context->GetInputDesc(0)->GetDataType();
        const bool isBool = (dtype == ge::DT_BOOL);
        // Hybrid Path A (v9.26): row-major (split OUTER) when outer>=4096 && axis<=128
        // && !bool — per-core MTE2 reads only its row slice. Else axis-major (split AXIS).
        // The SAME rule is evaluated in the kernel — keep in sync.
        const bool useRowMajor = !isBool && (outerLength >= 4096u) && (axisLength <= 128u);
        space = useRowMajor ? outerLength : axisLength;
        usedCoreNum = std::min<uint32_t>(coreNum, space);
        // Axis-major splits the AXIS dim across cores; per-core work = (axes/core) column
        // gathers, so column parallelism matters and the launch cap (which can shrink to
        // 1 core on tiny totals) HURTS — measured 24x63: 40c=6.6us vs 1c=10us. Exclude it.
        // Row-major splits OUTER (total always >= 8192), and its per-core offset-table
        // build is redundant scalar — fewer cores wins (measured 4096x16: 40c=11.8 -> 16c=6.2).
        applyLaunchCap = useRowMajor;
    } else if (innerBytes <= tileLength) {
        // Path B: one work-unit = one (axisIdx, outerIdx) row-pair.
        const uint32_t paddedRowBytes = AlignUp(innerBytes, BLOCK_SIZE);
        uint32_t rowsPerTile = tileLength / paddedRowBytes;
        if (rowsPerTile == 0) rowsPerTile = 1;
        if (rowsPerTile > MAX_BLOCK_COUNT) rowsPerTile = MAX_BLOCK_COUNT;
        const uint32_t totalRows = outerLength * axisLength;
        const uint32_t naturalTiles = (totalRows + rowsPerTile - 1) / rowsPerTile;
        space = totalRows;
        // Small inner (padded row <= 64B): MTE3 stride-write bound, more cores help.
        // Large inner: each tile is a substantial DMA, cap at naturalTiles.
        usedCoreNum = std::min<uint32_t>(coreNum, (paddedRowBytes <= 64) ? totalRows : naturalTiles);
    } else {
        // Path C: one work-unit = one linear input/output element.
        const uint32_t elemsPerTile = tileLength / typeSize;
        space = totalLength;
        usedCoreNum = std::min<uint32_t>(coreNum, (totalLength + elemsPerTile - 1) / elemsPerTile);
        pathCAlign = true;
    }
    usedCoreNum = std::max<uint32_t>(1, usedCoreNum);

    // v9.33 LAUNCH-AWARE CORE CAP: Ascend dispatches ALL blocks before any core starts,
    // so per-blockDim dispatch latency is a fixed cost. For tiny ops (Case1/Case3 ~7-9us,
    // measured task−aiv gap ~2-3us at blockDim=40) launching 40 cores wastes more on
    // dispatch than it saves in parallelism. Require each core to own >= MIN_WORK_PER_CORE
    // elements. Big ops (Case4/5) have demand >> 40 so are unaffected.
    constexpr uint32_t MIN_WORK_PER_CORE = 4096;
    if (applyLaunchCap) {
        uint32_t workCap = (totalLength + MIN_WORK_PER_CORE - 1) / MIN_WORK_PER_CORE;
        if (workCap < 1) workCap = 1;
        if (usedCoreNum > workCap) usedCoreNum = workCap;
    }

    blockLength = (space + usedCoreNum - 1) / usedCoreNum;

    if (pathCAlign) {
        // SKILL data-copy-prof §2.2: 512B-aligned per-core GM start → up to 30% more BW.
        constexpr uint32_t ALIGN_BYTES = 512;
        const uint32_t alignElems = ALIGN_BYTES / typeSize;  // 256/128/512 for 2/4/1-byte T
        blockLength = AlignUp(blockLength, alignElems);
        usedCoreNum = std::max<uint32_t>(1, (space + blockLength - 1) / blockLength);
    }

    tiling.set_totalLength(totalLength);
    tiling.set_outerLength(outerLength);
    tiling.set_axisLength(axisLength);
    tiling.set_innerLength(innerLength);
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
    if (inputShape == nullptr || outputShape == nullptr) return GRAPH_FAILED;

    int64_t axisAttr = 0;
    const auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t *axisPtr = attrs->GetAttrPointer<int64_t>(1);
        if (axisPtr != nullptr) axisAttr = *axisPtr;
    }
    const int32_t rank = inputShape->GetDimNum();
    const int32_t axis = axisAttr < 0 ? static_cast<int32_t>(axisAttr + rank) : static_cast<int32_t>(axisAttr);
    outputShape->SetDimNum(0);
    for (int32_t i = 0; i < rank; ++i) {
        if (i != axis) {
            outputShape->AppendDim(inputShape->GetDim(i));
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
} // namespace ge

namespace ops {
class Unpack : public OpDef {
public:
    explicit Unpack(const char *name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8,
                       ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("output")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8,
                       ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("num").Int();
        this->Attr("axis").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Unpack);
} // namespace ops
