#include "scale_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <climits>
#include <cstdint>

namespace optiling {
namespace {
constexpr int kDimMaxNum = 4;
constexpr int kRawDimMaxNum = 6;
constexpr uint32_t kDefaultTileLength = 3072U;
constexpr uint32_t kChampionBlockBytes = 1024U;
constexpr uint32_t kChampionSmallBytes = 64U * 1024U;
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

static inline uint32_t AlignUpCount(uint32_t count, uint32_t alignCount)
{
    return ((count + alignCount - 1U) / alignCount) * alignCount;
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

static inline bool MulWouldOverflow(int64_t lhs, int64_t rhs)
{
    return lhs > (INT_MAX / rhs);
}

static ge::graphStatus BuildCompressedDims(const gert::Shape& xShape,
                                          const gert::Shape& scaleShape,
                                          int64_t axis,
                                          int64_t axesNum,
                                          bool scaleFromBlob,
                                          int (&xDimsOut)[kDimMaxNum],
                                          int (&scaleDimsOut)[kDimMaxNum],
                                          uint64_t& totalLength,
                                          uint32_t& lastDim,
                                          int& compressedCountOut)
{
    const int64_t rank = static_cast<int64_t>(xShape.GetDimNum());
    if (rank <= 0 || rank > kRawDimMaxNum) {
        return ge::GRAPH_FAILED;
    }

    if (axis < 0) {
        axis += rank;
    }
    if (axis < 0 || axis >= rank) {
        return ge::GRAPH_FAILED;
    }

    int64_t xDimsRaw[kRawDimMaxNum] = {};
    for (int64_t i = 0; i < rank; ++i) {
        int64_t dim = xShape.GetDim(i);
        if (dim <= 0 || dim > INT_MAX) {
            return ge::GRAPH_FAILED;
        }
        xDimsRaw[i] = dim;
    }

    const int64_t scaleRank = static_cast<int64_t>(scaleShape.GetDimNum());
    int64_t endAxis = scaleFromBlob ? ((axesNum == -1) ? rank : axis + axesNum) : (axis + scaleRank);
    if (endAxis < axis || endAxis > rank) {
        return ge::GRAPH_FAILED;
    }
    if ((endAxis - axis) != scaleRank) {
        return ge::GRAPH_FAILED;
    }

    int64_t scaleDimsRaw[kRawDimMaxNum] = {};
    for (int64_t i = 0; i < rank; ++i) {
        scaleDimsRaw[i] = 1;
    }
    for (int64_t i = 0; i < scaleRank; ++i) {
        int64_t dim = scaleShape.GetDim(i);
        if (dim <= 0 || dim > INT_MAX || dim != xDimsRaw[axis + i]) {
            return ge::GRAPH_FAILED;
        }
        scaleDimsRaw[axis + i] = dim;
    }

    int64_t xWork[kRawDimMaxNum] = {};
    int64_t scaleWork[kRawDimMaxNum] = {};
    int workCount = 0;
    for (int64_t i = 0; i < rank; ++i) {
        if (xDimsRaw[i] == 1 && scaleDimsRaw[i] == 1) {
            continue;
        }
        xWork[workCount] = xDimsRaw[i];
        scaleWork[workCount] = scaleDimsRaw[i];
        ++workCount;
    }

    if (workCount == 0) {
        xWork[0] = 1;
        scaleWork[0] = 1;
        workCount = 1;
    }

    for (int i = 0; i < kDimMaxNum; ++i) {
        xDimsOut[i] = 0;
        scaleDimsOut[i] = 0;
    }

    int compressedCount = 0;
    int lastFlag = -1;
    int64_t runX = 1;
    int64_t runScale = 1;
    for (int i = 0; i < workCount; ++i) {
        if (scaleWork[i] != 1 && scaleWork[i] != xWork[i]) {
            return ge::GRAPH_FAILED;
        }

        int flag = (xWork[i] == scaleWork[i]) ? 0 : 1;
        if (lastFlag == -1) {
            lastFlag = flag;
            runX = xWork[i];
            runScale = scaleWork[i];
            continue;
        }

        if (flag == lastFlag) {
            if (MulWouldOverflow(runX, xWork[i]) || MulWouldOverflow(runScale, scaleWork[i])) {
                return ge::GRAPH_FAILED;
            }
            runX *= xWork[i];
            runScale *= scaleWork[i];
            continue;
        }

        if (compressedCount >= kDimMaxNum) {
            return ge::GRAPH_FAILED;
        }
        xDimsOut[compressedCount] = static_cast<int>(runX);
        scaleDimsOut[compressedCount] = static_cast<int>(runScale);
        ++compressedCount;

        lastFlag = flag;
        runX = xWork[i];
        runScale = scaleWork[i];
    }

    if (compressedCount >= kDimMaxNum) {
        return ge::GRAPH_FAILED;
    }
    xDimsOut[compressedCount] = static_cast<int>(runX);
    scaleDimsOut[compressedCount] = static_cast<int>(runScale);
    ++compressedCount;

    totalLength = 1U;
    for (int i = 0; i < compressedCount; ++i) {
        totalLength *= static_cast<uint64_t>(xDimsOut[i]);
    }
    lastDim = static_cast<uint32_t>(xDimsOut[compressedCount - 1]);
    compressedCountOut = compressedCount;
    return (lastDim == 0U) ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}

static inline uint32_t SelectTileLength(uint32_t lastDim,
                                        ge::DataType dt,
                                        uint32_t hasBias,
                                        uint32_t usableUbBytes)
{
    if (lastDim == 0U) {
        return 1U;
    }

    const uint32_t defaultTile = MinU32(lastDim, kDefaultTileLength);
    if (lastDim <= defaultTile || usableUbBytes == 0U) {
        return defaultTile;
    }

    const uint32_t elemBytes = DtypeBytes(dt);
    const uint32_t alignElems = 32U / elemBytes;
    if (alignElems == 0U) {
        return defaultTile;
    }

    const uint32_t alignedLastDim = AlignUpCount(lastDim, alignElems);
    uint64_t perElemBytes = static_cast<uint64_t>(2U * (3U + hasBias)) * static_cast<uint64_t>(elemBytes);
    if (dt == ge::DT_BF16) {
        perElemBytes += static_cast<uint64_t>(2U * (3U + hasBias)) * sizeof(float);
    }

    const uint64_t requiredBytes = static_cast<uint64_t>(alignedLastDim) * perElemBytes;
    if (requiredBytes <= static_cast<uint64_t>(usableUbBytes)) {
        return lastDim;
    }

    return defaultTile;
}

static inline uint32_t SelectContiguousTinyRowBlockDim(const int (&xDims)[kDimMaxNum],
                                                       const int (&scaleDims)[kDimMaxNum],
                                                       int compressedCount,
                                                       ge::DataType dt,
                                                       uint32_t hasBias,
                                                       uint32_t usableUbBytes,
                                                       uint32_t maxBlockDim)
{
    if (compressedCount < 2 || dt == ge::DT_BF16) {
        return maxBlockDim;
    }

    if (scaleDims[compressedCount - 1] != 1) {
        return maxBlockDim;
    }
    for (int i = 0; i < compressedCount - 2; ++i) {
        if (scaleDims[i] != 1) {
            return maxBlockDim;
        }
    }

    const uint32_t scaleRows = static_cast<uint32_t>(scaleDims[compressedCount - 2]);
    if (scaleRows < 64U) {
        return maxBlockDim;
    }

    const uint32_t lastDim = static_cast<uint32_t>(xDims[compressedCount - 1]);
    const uint32_t elemBytes = DtypeBytes(dt);
    const uint32_t alignElems = 32U / elemBytes;
    if (lastDim == 0U || alignElems == 0U || lastDim > alignElems) {
        return maxBlockDim;
    }

    uint32_t totalRows = 1U;
    for (int i = 0; i < compressedCount - 1; ++i) {
        totalRows *= static_cast<uint32_t>(xDims[i]);
    }
    if (totalRows == 0U) {
        return maxBlockDim;
    }

    const uint32_t rowStride = AlignUpCount(lastDim, alignElems);
    const uint32_t rowStrideBytes = rowStride * elemBytes;
    if (rowStrideBytes > 64U) {
        return maxBlockDim;
    }

    constexpr uint32_t kDoubleBufferedXy = 2U * 2U;
    const uint64_t bytesPerRow = static_cast<uint64_t>(rowStrideBytes) *
        static_cast<uint64_t>(kDoubleBufferedXy + 1U + (hasBias != 0U ? 1U : 0U));
    if (bytesPerRow == 0U) {
        return maxBlockDim;
    }

    uint32_t rowsPerTile = static_cast<uint32_t>(usableUbBytes / bytesPerRow);
    if (rowsPerTile < 32U) {
        return maxBlockDim;
    }
    rowsPerTile = MinU32(rowsPerTile, scaleRows);
    rowsPerTile = MinU32(rowsPerTile, totalRows);
    if (rowsPerTile == 0U) {
        return maxBlockDim;
    }

    uint32_t blockDim = CeilDivU32(totalRows, rowsPerTile);
    if (blockDim == 0U) {
        blockDim = 1U;
    }
    return MinU32(blockDim, maxBlockDim);
}

static inline uint32_t ApplyChampionBlockDim(uint32_t blockDim,
                                             uint64_t totalLength,
                                             ge::DataType dt,
                                             uint32_t maxBlockDim)
{
    if (blockDim == 0U || totalLength == 0U || maxBlockDim == 0U) {
        return 1U;
    }
    const uint64_t totalBytes = totalLength * static_cast<uint64_t>(DtypeBytes(dt));
    if (totalBytes == 0U || totalBytes > kChampionSmallBytes) {
        return blockDim;
    }
    uint32_t unitBlocks =
        static_cast<uint32_t>((totalBytes + kChampionBlockBytes - 1ULL) / kChampionBlockBytes);
    if (unitBlocks == 0U) {
        unitBlocks = 1U;
    }
    if (unitBlocks < blockDim) {
        blockDim = unitBlocks;
    }
    return MinU32(blockDim, maxBlockDim);
}

}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ScaleTilingData tiling;
    const gert::StorageShape* xShapeStorage = context->GetInputShape(0);
    const gert::StorageShape* scaleShapeStorage = context->GetInputShape(1);
    if (xShapeStorage == nullptr || scaleShapeStorage == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& xShape = xShapeStorage->GetOriginShape();
    const auto& scaleShape = scaleShapeStorage->GetOriginShape();
    int64_t axis = *context->GetAttrs()->GetInt(0);
    int64_t axesNum = *context->GetAttrs()->GetInt(1);
    bool scaleFromBlob = *context->GetAttrs()->GetBool(2);

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t maxBlockDim = platform.GetCoreNumAiv();
    if (maxBlockDim == 0U) {
        maxBlockDim = 1U;
    }

    uint64_t ubSizeBytes = 0U;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeBytes);
    uint64_t usableUbBytes = ubSizeBytes > kUbReserveBytes ? (ubSizeBytes - kUbReserveBytes) : kUbFallbackBytes;
    if (usableUbBytes > UINT32_MAX || usableUbBytes == 0U) {
        usableUbBytes = kUbFallbackBytes;
    }

    int xDims[kDimMaxNum] = {};
    int scaleDims[kDimMaxNum] = {};
    uint64_t totalLength = 0U;
    uint32_t lastDim = 0U;
    int compressedCount = 0;
    ge::graphStatus status = BuildCompressedDims(
        xShape, scaleShape, axis, axesNum, scaleFromBlob,
        xDims, scaleDims, totalLength, lastDim, compressedCount);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    uint32_t hasBias = (context->GetInputShape(2) != nullptr) ? 1U : 0U;
    if (hasBias != 0U) {
        const auto& biasShape = context->GetInputShape(2)->GetOriginShape();
        if (biasShape.GetDimNum() != scaleShape.GetDimNum()) {
            return ge::GRAPH_FAILED;
        }
        for (int64_t i = 0; i < biasShape.GetDimNum(); ++i) {
            if (biasShape.GetDim(i) != scaleShape.GetDim(i)) {
                return ge::GRAPH_FAILED;
            }
        }
    }

    const ge::DataType inputType = context->GetInputDesc(0)->GetDataType();
    uint32_t tileLength = SelectTileLength(lastDim,
                                           inputType,
                                           hasBias,
                                           static_cast<uint32_t>(usableUbBytes));
    uint64_t rowCount = totalLength / static_cast<uint64_t>(lastDim);
    uint64_t chunksPerRow =
        (static_cast<uint64_t>(lastDim) + static_cast<uint64_t>(tileLength) - 1U) / static_cast<uint64_t>(tileLength);
    uint64_t taskCount = rowCount * chunksPerRow;
    if (taskCount == 0U) {
        return ge::GRAPH_FAILED;
    }

    uint32_t blockDim = (taskCount < static_cast<uint64_t>(maxBlockDim))
        ? static_cast<uint32_t>(taskCount)
        : maxBlockDim;
    blockDim = ApplyChampionBlockDim(blockDim, totalLength, inputType, maxBlockDim);
    uint32_t tinyRowBlockDim = SelectContiguousTinyRowBlockDim(
        xDims,
        scaleDims,
        compressedCount,
        inputType,
        hasBias,
        static_cast<uint32_t>(usableUbBytes),
        maxBlockDim);
    if (tinyRowBlockDim < blockDim) {
        blockDim = tinyRowBlockDim;
    }
    if (blockDim == 0U) {
        blockDim = 1U;
    }
    context->SetBlockDim(blockDim);

    tiling.set_x_dims(xDims);
    tiling.set_scale_dims(scaleDims);
    tiling.set_has_bias(hasBias);
    tiling.set_tile_length(tileLength);
    tiling.set_ub_bytes(static_cast<uint32_t>(usableUbBytes));

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

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
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Scale : public OpDef {
public:
    explicit Scale(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("axis").Int();
        this->Attr("axes_num").Int();
        this->Attr("scale_from_blob").Bool();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Scale);
}
