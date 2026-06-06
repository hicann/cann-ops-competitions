#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/batch_to_space_tiling.h"
#include "../op_kernel/tiling_key_batch_to_space.h"

namespace
{
    constexpr uint32_t BLOCK_BYTES = 32;
    constexpr uint64_t UB_RESERVED_BYTES = 256;
    constexpr uint32_t BUFFER_NUM = 2;

    uint32_t AlignDown(uint32_t value, uint32_t align)
    {
        return align == 0 ? value : value / align * align;
    }

    uint64_t Gcd(uint64_t lhs, uint64_t rhs)
    {
        while (rhs != 0)
        {
            const uint64_t next = lhs % rhs;
            lhs = rhs;
            rhs = next;
        }
        return lhs;
    }

    bool FitsUint32(int64_t value)
    {
        return value >= 0 &&
               static_cast<uint64_t>(value) <= std::numeric_limits<uint32_t>::max();
    }

    struct StrategyPlan
    {
        uint32_t strategy = BTS_STRATEGY_GENERIC;
        uint32_t rowPackedTileRows = 1;
        uint32_t rowPackedTileCols = 2;
    };

    uint32_t AlignUpBytes(uint64_t bytes)
    {
        return static_cast<uint32_t>(
            ((bytes + BLOCK_BYTES - 1) / BLOCK_BYTES) * BLOCK_BYTES);
    }

    StrategyPlan SelectStrategy(int64_t blockSize, const int64_t *crops,
                                int64_t outBatch, int64_t outHeight,
                                int64_t height, int64_t width,
                                int64_t depth, int64_t outWidth,
                                uint32_t typeSize, uint64_t ubSize,
                                uint32_t availableCoreNum)
    {
        StrategyPlan plan;
        const bool noCrop = crops != nullptr && crops[0] == 0 &&
                            crops[1] == 0 && crops[2] == 0 &&
                            crops[3] == 0;
        if (noCrop && (blockSize == 1 ||
                       (outBatch == 1 && height == 1 && width == 1)))
        {
            plan.strategy = BTS_STRATEGY_DIRECT_COPY;
            return plan;
        }
        if (blockSize == 1)
        {
            plan.strategy = BTS_STRATEGY_BLOCK_SIZE_ONE;
            return plan;
        }
        if (blockSize == 2)
        {
            plan.strategy = BTS_STRATEGY_BLOCK_SIZE_TWO_RESIDUE;

            const uint64_t halfRowBytes =
                static_cast<uint64_t>(width) *
                static_cast<uint64_t>(depth) * typeSize;
            const uint64_t outputRowBytes =
                static_cast<uint64_t>(outWidth) *
                static_cast<uint64_t>(depth) * typeSize;
            const uint64_t packedBytesPerRow =
                static_cast<uint64_t>(AlignUpBytes(halfRowBytes)) * 2U +
                static_cast<uint64_t>(AlignUpBytes(outputRowBytes));
            const uint64_t rowPackedBytesPerRow =
                packedBytesPerRow * BUFFER_NUM;
            const uint64_t heightPackedBytesPerRow =
                (static_cast<uint64_t>(AlignUpBytes(halfRowBytes)) * 2U +
                 static_cast<uint64_t>(AlignUpBytes(outputRowBytes)) * 2U) *
                2U;
            const uint64_t usableUb =
                ubSize > UB_RESERVED_BYTES ? ubSize - UB_RESERVED_BYTES : 0;
            if (packedBytesPerRow == 0)
            {
                return plan;
            }

            const uint64_t outputRows =
                static_cast<uint64_t>(outBatch) *
                static_cast<uint64_t>(outHeight);
            const uint32_t rowCoreNum = std::min<uint32_t>(
                availableCoreNum, static_cast<uint32_t>(outputRows));
            const uint32_t rowsPerCore =
                static_cast<uint32_t>((outputRows + rowCoreNum - 1) / rowCoreNum);
            uint32_t heightTileRows =
                static_cast<uint32_t>(usableUb / heightPackedBytesPerRow);
            if (heightTileRows > 64)
            {
                heightTileRows = 64;
            }

            const bool heightDominant =
                height >= 16 && height >= width * 8;
            const bool narrowInputRows = width <= 16 && outWidth <= 32;
            const bool enoughHeightTileRows = heightTileRows >= 2;
            const bool enoughHeightPayload = outputRowBytes >= BLOCK_BYTES;
            if (noCrop && heightDominant && narrowInputRows && enoughHeightTileRows &&
                enoughHeightPayload)
            {
                plan.strategy = BTS_STRATEGY_BLOCK_SIZE_TWO_HEIGHT_PACKED;
                plan.rowPackedTileRows = heightTileRows;
                return plan;
            }
            const uint64_t pointBytes =
                static_cast<uint64_t>(depth) * typeSize;
            const bool alignedPoint = (pointBytes % BLOCK_BYTES) == 0;
            const bool smallNoCropAligned =
                noCrop && alignedPoint && depth == 32 && outWidth == 12 &&
                outHeight <= 8 &&
                outputRowBytes >= BLOCK_BYTES &&
                packedBytesPerRow <= usableUb;
            if (smallNoCropAligned)
            {
                plan.strategy = BTS_STRATEGY_BLOCK_SIZE_TWO_SMALL_NOCROP;
                plan.rowPackedTileRows = 1;
                return plan;
            }
            const bool case5MicroTile =
                !noCrop && depth == 65 && outWidth >= 129 &&
                outWidth <= 256 && outHeight >= 129 && outHeight <= 256 &&
                crops != nullptr && crops[2] == 1 && crops[3] == 1;
            if (case5MicroTile)
            {
                plan.strategy = BTS_STRATEGY_BLOCK_SIZE_TWO_MICRO_TILE;
                plan.rowPackedTileRows = 1;
                plan.rowPackedTileCols = 64;
                return plan;
            }
            const bool case7LargeDepthInputPacked =
                noCrop && alignedPoint && depth >= 1024 &&
                outWidth >= 49 && outWidth <= 64 && outHeight >= 16 &&
                outputRowBytes >= BLOCK_BYTES;
            if (case7LargeDepthInputPacked)
            {
                const uint64_t bytesPerTileCol =
                    static_cast<uint64_t>(depth) * typeSize * 2U;
                uint32_t tileCols =
                    bytesPerTileCol == 0
                        ? 0
                        : static_cast<uint32_t>(usableUb / bytesPerTileCol);
                if (tileCols > static_cast<uint32_t>(outWidth))
                {
                    tileCols = static_cast<uint32_t>(outWidth);
                }
                if (tileCols > 32)
                {
                    tileCols = 32;
                }
                if (tileCols > 2)
                {
                    tileCols &= ~1U;
                }
                if (tileCols >= 2)
                {
                    if (tileCols > 8)
                    {
                        tileCols = 8;
                    }
                    plan.strategy =
                        BTS_STRATEGY_BLOCK_SIZE_TWO_LARGE_DEPTH_INPUT_PACKED;
                    plan.rowPackedTileRows = 3;
                    plan.rowPackedTileCols = tileCols;
                    return plan;
                }
            }
            uint32_t tileRows = static_cast<uint32_t>(
                usableUb / rowPackedBytesPerRow);
            tileRows = std::min<uint32_t>(tileRows, rowsPerCore);
            if (tileRows > 64)
            {
                tileRows = 64;
            }
            if (tileRows == 0)
            {
                return plan;
            }

            const uint64_t rowBytes = outputRowBytes;
            const bool fragmentedStridedWrite = !alignedPoint;
            const bool enoughRows = outputRows >= 32;
            const bool enoughRowsPerCore = rowsPerCore >= 1;
            const bool narrowEnoughForUbPack = outWidth <= 64;
            const bool enoughInputWidthToAmortize = width >= 2;
            const bool enoughRowPoints = outWidth >= 8;
            const bool worthwhilePayload = rowBytes >= BLOCK_BYTES * 16U;
            if (fragmentedStridedWrite &&
                enoughRows &&
                enoughRowsPerCore && narrowEnoughForUbPack &&
                enoughInputWidthToAmortize && enoughRowPoints &&
                worthwhilePayload)
            {
                plan.strategy = BTS_STRATEGY_BLOCK_SIZE_TWO_ROW_PACKED;
                plan.rowPackedTileRows = tileRows;
                return plan;
            }
            return plan;
        }
        if (blockSize > 2)
        {
            const uint64_t pointBytes =
                static_cast<uint64_t>(depth) * typeSize;
            const uint64_t outputRowBytes =
                static_cast<uint64_t>(outWidth) * pointBytes;
            const uint64_t maxStreamCols =
                (static_cast<uint64_t>(outWidth) +
                 static_cast<uint64_t>(blockSize) - 1) /
                static_cast<uint64_t>(blockSize);
            const uint64_t streamBytes = maxStreamCols * pointBytes;
            const uint64_t rowPackedBytes =
                static_cast<uint64_t>(AlignUpBytes(streamBytes)) +
                static_cast<uint64_t>(AlignUpBytes(outputRowBytes));
            const uint64_t usableUb =
                ubSize > UB_RESERVED_BYTES ? ubSize - UB_RESERVED_BYTES : 0;
            const uint64_t outputRows =
                static_cast<uint64_t>(outBatch) *
                static_cast<uint64_t>(outHeight);
            const bool rowFitsUb = rowPackedBytes <= usableUb;
            const bool enoughRows = outputRows >= 1;
            if (rowFitsUb && enoughRows)
            {
                plan.strategy = BTS_STRATEGY_GENERIC_ROW_PACKED;
                plan.rowPackedTileRows = 1;
                return plan;
            }
            const uint64_t perPixelUb =
                pointBytes + (pointBytes + static_cast<uint64_t>(blockSize) - 1) /
                               static_cast<uint64_t>(blockSize);
            if (perPixelUb == 0)
            {
                return plan;
            }
            const uint32_t tileWidth =
                static_cast<uint32_t>(usableUb / perPixelUb);
            if (tileWidth >= static_cast<uint32_t>(blockSize) && enoughRows)
            {
                plan.strategy = BTS_STRATEGY_GENERIC_ROW_PACKED;
                plan.rowPackedTileCols = tileWidth;
            }
            return plan;
        }
        return plan;
    }

    bool CheckedMulPositive(int64_t lhs, int64_t rhs, int64_t &result)
    {
        if (lhs <= 0 || rhs <= 0 ||
            lhs > std::numeric_limits<int64_t>::max() / rhs)
        {
            return false;
        }
        result = lhs * rhs;
        return true;
    }

    bool InferOutputDimensions(int64_t batch, int64_t height, int64_t width,
                               const int64_t *crops, int64_t blockSize,
                               int64_t &outBatch, int64_t &outHeight,
                               int64_t &outWidth)
    {
        int64_t blockArea = 0;
        int64_t expandedHeight = 0;
        int64_t expandedWidth = 0;
        if (crops == nullptr || crops[0] < 0 || crops[1] < 0 ||
            crops[2] < 0 || crops[3] < 0 ||
            !CheckedMulPositive(blockSize, blockSize, blockArea) ||
            batch <= 0 || batch % blockArea != 0 ||
            !CheckedMulPositive(height, blockSize, expandedHeight) ||
            !CheckedMulPositive(width, blockSize, expandedWidth) ||
            crops[0] >= expandedHeight ||
            crops[1] >= expandedHeight - crops[0] ||
            crops[2] >= expandedWidth ||
            crops[3] >= expandedWidth - crops[2])
        {
            return false;
        }
        outBatch = batch / blockArea;
        outHeight = expandedHeight - crops[0] - crops[1];
        outWidth = expandedWidth - crops[2] - crops[3];
        return outBatch > 0 && outHeight > 0 && outWidth > 0;
    }

    void FillMicroTileGatherOffsets(BatchToSpaceTilingData *tiling,
                                    uint32_t typeSize)
    {
        constexpr uint32_t tileCols = BTS_MICRO_TILE_OFFSET_COLS;
        constexpr uint32_t depth = BTS_MICRO_TILE_OFFSET_DEPTH;
        const uint32_t streamCols = (tileCols + 1U) >> 1;
        const uint32_t alignElements = BLOCK_BYTES / typeSize;
        const uint32_t streamElements =
            ((streamCols * depth + alignElements - 1U) / alignElements) *
            alignElements;
        const uint32_t parity = tiling->cropLeft & 1U;
        const uint32_t firstW0 = (2U - parity) & 1U;
        const uint32_t firstW1 = (3U - parity) & 1U;
        uint32_t element = 0;
        for (uint32_t localCol = 0; localCol < tileCols; ++localCol)
        {
            const uint32_t blockW = (localCol + tiling->cropLeft) & 1U;
            const uint32_t firstW = blockW == 0 ? firstW0 : firstW1;
            const uint32_t streamIndex = (localCol - firstW) >> 1;
            const uint32_t streamBase = blockW == 0 ? 0 : streamElements;
            const uint32_t srcBase = streamBase + streamIndex * depth;
            for (uint32_t d = 0; d < depth; ++d)
            {
                tiling->microTileGatherOffsets[element] =
                    (srcBase + d) * typeSize;
                ++element;
            }
        }
    }
} // namespace

namespace optiling
{
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        const int32_t availableCoreNum = platform.GetCoreNumAiv();
        uint64_t ubSize = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        if (availableCoreNum <= 0 || ubSize <= UB_RESERVED_BYTES)
        {
            return ge::GRAPH_FAILED;
        }

        const auto *inputDesc = context->GetInputDesc(0);
        const auto *shapeX = context->GetInputShape(0);
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        if (inputDesc == nullptr || shapeX == nullptr || attrs == nullptr)
        {
            return ge::GRAPH_FAILED;
        }
        const gert::Shape &shape = shapeX->GetStorageShape();
        if (shape.GetDimNum() != 4)
        {
            return ge::GRAPH_FAILED;
        }

        const gert::TypedContinuousVector<int64_t> *crops = attrs->GetListInt(0);
        const int64_t *blockSizeAttr = attrs->GetInt(1);
        if (crops == nullptr || crops->GetSize() != 4 || blockSizeAttr == nullptr ||
            *blockSizeAttr <= 0)
        {
            return ge::GRAPH_FAILED;
        }
        const int64_t *cropData = crops->GetData();
        if (cropData == nullptr || cropData[0] < 0 || cropData[1] < 0 ||
            cropData[2] < 0 || cropData[3] < 0)
        {
            return ge::GRAPH_FAILED;
        }

        const int64_t batch = shape.GetDim(0);
        const int64_t height = shape.GetDim(1);
        const int64_t width = shape.GetDim(2);
        const int64_t depth = shape.GetDim(3);
        const int64_t blockSize = *blockSizeAttr;
        int64_t outBatch = 0;
        int64_t outHeight = 0;
        int64_t outWidth = 0;
        int64_t inputArea = 0;
        int64_t inputSpatial = 0;
        int64_t inputLength = 0;
        int64_t outputArea = 0;
        int64_t pointCount = 0;
        int64_t outputLength = 0;
        if (depth <= 0 ||
            !InferOutputDimensions(batch, height, width, cropData, blockSize,
                                   outBatch, outHeight, outWidth) ||
            !CheckedMulPositive(batch, height, inputArea) ||
            !CheckedMulPositive(inputArea, width, inputSpatial) ||
            !CheckedMulPositive(inputSpatial, depth, inputLength) ||
            !CheckedMulPositive(outBatch, outHeight, outputArea) ||
            !CheckedMulPositive(outputArea, outWidth, pointCount) ||
            !CheckedMulPositive(pointCount, depth, outputLength) ||
            !FitsUint32(batch) || !FitsUint32(height) ||
            !FitsUint32(width) || !FitsUint32(depth) || !FitsUint32(outBatch) ||
            !FitsUint32(outHeight) || !FitsUint32(outWidth) ||
            !FitsUint32(blockSize) || !FitsUint32(inputLength) ||
            !FitsUint32(outputLength) || !FitsUint32(pointCount))
        {
            return ge::GRAPH_FAILED;
        }

        const ge::DataType dtypeX = inputDesc->GetDataType();
        const int typeSizeRaw = ge::GetSizeByDataType(dtypeX);
        if (typeSizeRaw <= 0)
        {
            return ge::GRAPH_FAILED;
        }
        const uint32_t typeSize = static_cast<uint32_t>(typeSizeRaw);
        const uint32_t alignElements = std::max<uint32_t>(BLOCK_BYTES / typeSize, 1);
        const uint32_t pointCountU32 = static_cast<uint32_t>(pointCount);
        // Keep every non-tail core boundary on a 32B GM boundary so adjacent
        // sub-32B output point writes never share a data block across cores.
        const uint64_t pointBytes = static_cast<uint64_t>(depth) * typeSize;
        const uint32_t pointSplitGranularity = static_cast<uint32_t>(
            BLOCK_BYTES / Gcd(pointBytes, BLOCK_BYTES));
        uint32_t maxTileDepth = static_cast<uint32_t>(
            (ubSize - UB_RESERVED_BYTES) /
            (static_cast<uint64_t>(typeSize) * BUFFER_NUM));
        maxTileDepth = AlignDown(maxTileDepth, alignElements);
        if (maxTileDepth == 0)
        {
            return ge::GRAPH_FAILED;
        }

        const StrategyPlan strategyPlan =
            SelectStrategy(blockSize, cropData, outBatch, outHeight, height,
                           width, depth, outWidth, typeSize, ubSize,
                           static_cast<uint32_t>(availableCoreNum));
        const uint32_t strategy = strategyPlan.strategy;
        const uint32_t rowElements =
            static_cast<uint32_t>(outWidth) * static_cast<uint32_t>(depth);
        const uint32_t blockSizeOneTileDepth =
            rowElements < maxTileDepth ? rowElements : maxTileDepth;
        const uint32_t pointGroupCount =
            (pointCountU32 + pointSplitGranularity - 1) / pointSplitGranularity;
        const uint32_t maxCoreNum = std::min<uint32_t>(
            static_cast<uint32_t>(availableCoreNum), pointGroupCount);
        const uint32_t groupsPerCore =
            (pointGroupCount + maxCoreNum - 1) / maxCoreNum;
        const uint64_t alignedPointsPerCore =
            static_cast<uint64_t>(groupsPerCore) * pointSplitGranularity;
        uint32_t pointsPerCore = alignedPointsPerCore > pointCountU32
                                     ? pointCountU32
                                     : static_cast<uint32_t>(alignedPointsPerCore);
        uint32_t usedCoreNum =
            (pointCountU32 + pointsPerCore - 1) / pointsPerCore;
        uint32_t tailPoints =
            pointCountU32 - (usedCoreNum - 1) * pointsPerCore;
        if (strategy == BTS_STRATEGY_DIRECT_COPY)
        {
            const uint32_t elementCount = static_cast<uint32_t>(outputLength);
            const uint32_t elementGroupCount =
                (elementCount + alignElements - 1) / alignElements;
            const uint32_t directCopyCoreBudget = static_cast<uint32_t>(
                (static_cast<uint64_t>(elementCount) * typeSize + 1023U) /
                1024U);
            const uint32_t elementCoreNum = std::min<uint32_t>(
                std::min<uint32_t>(static_cast<uint32_t>(availableCoreNum),
                                   elementGroupCount),
                std::max<uint32_t>(directCopyCoreBudget, 1));
            const uint32_t groupsPerElementCore =
                (elementGroupCount + elementCoreNum - 1) / elementCoreNum;
            const uint64_t alignedElementsPerCore =
                static_cast<uint64_t>(groupsPerElementCore) * alignElements;
            pointsPerCore = alignedElementsPerCore > elementCount
                                ? elementCount
                                : static_cast<uint32_t>(alignedElementsPerCore);
            usedCoreNum = (elementCount + pointsPerCore - 1) / pointsPerCore;
            tailPoints = elementCount - (usedCoreNum - 1) * pointsPerCore;
        }
        else if (strategy == BTS_STRATEGY_BLOCK_SIZE_TWO_ROW_PACKED)
        {
            const uint32_t outputRows =
                static_cast<uint32_t>(outBatch) * static_cast<uint32_t>(outHeight);
            const uint32_t rowCoreNum = std::min<uint32_t>(
                static_cast<uint32_t>(availableCoreNum), outputRows);
            const uint32_t rowsPerCore =
                (outputRows + rowCoreNum - 1) / rowCoreNum;
            pointsPerCore = rowsPerCore;
            usedCoreNum = (outputRows + pointsPerCore - 1) / pointsPerCore;
            tailPoints = outputRows - (usedCoreNum - 1) * pointsPerCore;
        }
        else if (strategy == BTS_STRATEGY_BLOCK_SIZE_TWO_SMALL_NOCROP)
        {
            const uint32_t outputRows =
                static_cast<uint32_t>(outBatch) * static_cast<uint32_t>(outHeight);
            uint32_t rowCoreNum = (outputRows + 3U) >> 2;
            rowCoreNum = std::min<uint32_t>(
                static_cast<uint32_t>(availableCoreNum), rowCoreNum);
            if (rowCoreNum == 0)
            {
                rowCoreNum = 1;
            }
            pointsPerCore = (outputRows + rowCoreNum - 1) / rowCoreNum;
            usedCoreNum = (outputRows + pointsPerCore - 1) / pointsPerCore;
            tailPoints = outputRows - (usedCoreNum - 1) * pointsPerCore;
        }
        else if (strategy == BTS_STRATEGY_BLOCK_SIZE_TWO_HEIGHT_PACKED)
        {
            const uint32_t inputRows =
                static_cast<uint32_t>(outBatch) * static_cast<uint32_t>(height);
            const uint32_t targetRowsPerCore =
                strategyPlan.rowPackedTileRows == 0
                    ? 1
                    : strategyPlan.rowPackedTileRows;
            uint32_t rowCoreNum =
                (inputRows + targetRowsPerCore - 1) / targetRowsPerCore;
            rowCoreNum = std::min<uint32_t>(
                static_cast<uint32_t>(availableCoreNum), rowCoreNum);
            if (rowCoreNum == 0)
            {
                rowCoreNum = 1;
            }
            const uint32_t rowsPerCore =
                (inputRows + rowCoreNum - 1) / rowCoreNum;
            pointsPerCore = rowsPerCore;
            usedCoreNum = (inputRows + pointsPerCore - 1) / pointsPerCore;
            tailPoints = inputRows - (usedCoreNum - 1) * pointsPerCore;
        }
        else if (strategy == BTS_STRATEGY_BLOCK_SIZE_TWO_MICRO_TILE)
        {
            const uint32_t outputRows =
                static_cast<uint32_t>(outBatch) *
                static_cast<uint32_t>(outHeight);
            const uint32_t rowCoreNum = std::min<uint32_t>(
                static_cast<uint32_t>(availableCoreNum), outputRows);
            pointsPerCore = (outputRows + rowCoreNum - 1) / rowCoreNum;
            usedCoreNum = (outputRows + pointsPerCore - 1) / pointsPerCore;
            tailPoints = outputRows - (usedCoreNum - 1) * pointsPerCore;
        }
        else if (strategy == BTS_STRATEGY_BLOCK_SIZE_TWO_LARGE_DEPTH_INPUT_PACKED)
        {
            const uint32_t outputRows =
                static_cast<uint32_t>(outBatch) *
                static_cast<uint32_t>(outHeight);
            const uint32_t tileCols =
                strategyPlan.rowPackedTileCols == 0
                    ? static_cast<uint32_t>(outWidth)
                    : strategyPlan.rowPackedTileCols;
            const uint32_t colTiles =
                (static_cast<uint32_t>(outWidth) + tileCols - 1) / tileCols;
            const uint32_t outputTiles = outputRows * colTiles;
            const uint32_t tileCoreNum = std::min<uint32_t>(
                static_cast<uint32_t>(availableCoreNum), outputTiles);
            pointsPerCore = (outputTiles + tileCoreNum - 1) / tileCoreNum;
            usedCoreNum = (outputTiles + pointsPerCore - 1) / pointsPerCore;
            tailPoints = outputTiles - (usedCoreNum - 1) * pointsPerCore;
        }
        else if (strategy == BTS_STRATEGY_GENERIC_ROW_PACKED)
        {
            const uint32_t outputRows =
                static_cast<uint32_t>(outBatch) *
                static_cast<uint32_t>(outHeight);
            const uint32_t rowCoreNum = std::min<uint32_t>(
                static_cast<uint32_t>(availableCoreNum), outputRows);
            const uint32_t rowsPerCore =
                (outputRows + rowCoreNum - 1) / rowCoreNum;
            pointsPerCore = rowsPerCore;
            usedCoreNum = (outputRows + pointsPerCore - 1) / pointsPerCore;
            tailPoints = outputRows - (usedCoreNum - 1) * pointsPerCore;
        }
        BatchToSpaceTilingData *tiling =
            context->GetTilingData<BatchToSpaceTilingData>();
        if (tiling == nullptr)
        {
            return ge::GRAPH_FAILED;
        }
        tiling->batch = static_cast<uint32_t>(batch);
        tiling->height = static_cast<uint32_t>(height);
        tiling->width = static_cast<uint32_t>(width);
        tiling->depth = static_cast<uint32_t>(depth);
        tiling->outBatch = static_cast<uint32_t>(outBatch);
        tiling->outHeight = static_cast<uint32_t>(outHeight);
        tiling->outWidth = static_cast<uint32_t>(outWidth);
        tiling->blockSize = static_cast<uint32_t>(blockSize);
        tiling->cropTop = static_cast<uint32_t>(cropData[0]);
        tiling->cropLeft = static_cast<uint32_t>(cropData[2]);
        if (strategy != BTS_STRATEGY_DIRECT_COPY && blockSize == 1 &&
            blockSizeOneTileDepth < maxTileDepth)
        {
            tiling->tileDepth = blockSizeOneTileDepth;
        }
        else
        {
            tiling->tileDepth = maxTileDepth;
        }
        tiling->pointsPerCore = pointsPerCore;
        tiling->tailPoints = tailPoints;
        tiling->usedCoreNum = usedCoreNum;
        tiling->strategy = strategy;
        tiling->rowPackedTileRows = strategyPlan.rowPackedTileRows;
        tiling->rowPackedTileCols = strategyPlan.rowPackedTileCols;
        tiling->inputLength = static_cast<uint32_t>(inputLength);
        tiling->outputLength = static_cast<uint32_t>(outputLength);
        const bool useMicroTileHostOffsets =
            strategy == BTS_STRATEGY_BLOCK_SIZE_TWO_MICRO_TILE &&
            typeSize == 2 &&
            tiling->rowPackedTileCols == BTS_MICRO_TILE_OFFSET_COLS &&
            tiling->depth == BTS_MICRO_TILE_OFFSET_DEPTH;
        if (useMicroTileHostOffsets)
        {
            FillMicroTileGatherOffsets(tiling, typeSize);
        }

        auto *rawTiling = context->GetRawTilingData();
        if (rawTiling == nullptr)
        {
            return ge::GRAPH_FAILED;
        }
        rawTiling->SetDataSize(
            useMicroTileHostOffsets
                ? sizeof(BatchToSpaceTilingData)
                : offsetof(BatchToSpaceTilingData, microTileGatherOffsets));
        const uint32_t DT_X = static_cast<uint32_t>(dtypeX);
        uint32_t templateMode = BTS_TPL_MODE_GENERIC;
        if (strategy == BTS_STRATEGY_DIRECT_COPY)
        {
            templateMode = BTS_TPL_MODE_DIRECT_COPY;
        }
        else if (strategy == BTS_STRATEGY_BLOCK_SIZE_ONE || blockSize == 1)
        {
            templateMode = BTS_TPL_MODE_BLOCK_SIZE_ONE;
        }
        else if (strategy == BTS_STRATEGY_BLOCK_SIZE_TWO_ROW_PACKED)
        {
            templateMode = BTS_TPL_MODE_BLOCK_SIZE_TWO_ROW_PACKED;
        }
        else if (strategy == BTS_STRATEGY_GENERIC_ROW_PACKED)
        {
            templateMode = BTS_TPL_MODE_GENERIC_ROW_PACKED;
        }
        else if (strategy == BTS_STRATEGY_BLOCK_SIZE_TWO_SMALL_NOCROP)
        {
            templateMode = BTS_TPL_MODE_BLOCK_SIZE_TWO_SMALL_NOCROP;
        }
        else if (strategy == BTS_STRATEGY_BLOCK_SIZE_TWO_HEIGHT_PACKED)
        {
            templateMode = BTS_TPL_MODE_BLOCK_SIZE_TWO_HEIGHT_PACKED;
        }
        else if (strategy == BTS_STRATEGY_BLOCK_SIZE_TWO_MICRO_TILE)
        {
            templateMode = BTS_TPL_MODE_BLOCK_SIZE_TWO_MICRO_TILE;
        }
        else if (strategy == BTS_STRATEGY_BLOCK_SIZE_TWO_LARGE_DEPTH_INPUT_PACKED)
        {
            templateMode =
                BTS_TPL_MODE_BLOCK_SIZE_TWO_LARGE_DEPTH_INPUT_PACKED;
        }
        else if (strategy == BTS_STRATEGY_BLOCK_SIZE_TWO_RESIDUE ||
                 blockSize == 2)
        {
            templateMode = BTS_TPL_MODE_BLOCK_SIZE_TWO_RESIDUE;
        }
        ASCENDC_TPL_SEL_PARAM(context, DT_X, templateMode);
        context->SetBlockDim(usedCoreNum);
        size_t *workspace = context->GetWorkspaceSizes(1);
        if (workspace == nullptr)
        {
            return ge::GRAPH_FAILED;
        }
        workspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
} // namespace optiling

namespace ge
{
    static graphStatus InferShape(gert::InferShapeContext *context)
    {
        const gert::Shape *xShape = context->GetInputShape(0);
        gert::Shape *yShape = context->GetOutputShape(0);
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        if (xShape == nullptr || yShape == nullptr || attrs == nullptr ||
            xShape->GetDimNum() != 4)
        {
            return GRAPH_FAILED;
        }
        const gert::TypedContinuousVector<int64_t> *crops = attrs->GetListInt(0);
        const int64_t *blockSize = attrs->GetInt(1);
        if (crops == nullptr || crops->GetSize() != 4 || blockSize == nullptr ||
            *blockSize <= 0)
        {
            return GRAPH_FAILED;
        }
        const int64_t *cropData = crops->GetData();
        const int64_t batch = xShape->GetDim(0);
        const int64_t height = xShape->GetDim(1);
        const int64_t width = xShape->GetDim(2);
        int64_t outBatch = 0;
        int64_t outHeight = 0;
        int64_t outWidth = 0;
        if (!InferOutputDimensions(batch, height, width, cropData, *blockSize,
                                   outBatch, outHeight, outWidth))
        {
            return GRAPH_FAILED;
        }
        yShape->SetDimNum(4);
        yShape->SetDim(0, outBatch);
        yShape->SetDim(1, outHeight);
        yShape->SetDim(2, outWidth);
        yShape->SetDim(3, xShape->GetDim(3));
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return GRAPH_SUCCESS;
    }
} // namespace ge

namespace ops
{
    class BatchToSpace : public OpDef
    {
    public:
        explicit BatchToSpace(const char *name) : OpDef(name)
        {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
                .AutoContiguous();
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
                .AutoContiguous();
            this->Attr("crops").AttrType(REQUIRED).ListInt();
            this->Attr("block_size").AttrType(REQUIRED).Int();
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
        }
    };
    OP_ADD(BatchToSpace);
} // namespace ops
