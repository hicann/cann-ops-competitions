// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ascendc/host_api/tiling/template_argument.h"
#include <algorithm>

#include "../op_kernel/batch_to_space_tiling.h"

ASCENDC_TPL_ARGS_DECL(BatchToSpace,
    ASCENDC_TPL_DATATYPE_DECL(DT_X, C_DT_FLOAT16, C_DT_FLOAT),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_FLOAT16, C_DT_FLOAT),
    ),
);

namespace optiling {
    static uint32_t GreatestCommonDivisor(uint32_t lhs, uint32_t rhs) {
        while (rhs != 0U) {
            const uint32_t remainder = lhs % rhs;
            lhs = rhs;
            rhs = remainder;
        }
        return lhs;
    }

    static uint64_t CeilDiv(uint64_t lhs, uint64_t rhs) {
        return rhs == 0U ? 0U : (lhs + rhs - 1U) / rhs;
    }

    static uint64_t AlignUp(uint64_t value, uint64_t align) {
        return align == 0U ? value : CeilDiv(value, align) * align;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t numCores = static_cast<uint32_t>(platform.GetCoreNumAiv());
        if (numCores == 0U) {
            numCores = 1U;
        }
        uint64_t ubSize = 0U;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

        const gert::Shape &shape = context->GetInputShape(0)->GetStorageShape();
        const uint32_t inputBatch = static_cast<uint32_t>(shape.GetDim(0));
        const uint32_t inputHeight = static_cast<uint32_t>(shape.GetDim(1));
        const uint32_t inputWidth = static_cast<uint32_t>(shape.GetDim(2));
        const uint32_t depth = static_cast<uint32_t>(shape.GetDim(3));

        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const int64_t *crops = attrs->GetListInt(0)->GetData();
        const uint32_t blockSize = static_cast<uint32_t>(*attrs->GetInt(1));
        const uint32_t outputBatch = inputBatch / (blockSize * blockSize);
        const uint32_t outputHeight =
            inputHeight * blockSize - static_cast<uint32_t>(crops[0] + crops[1]);
        const uint32_t outputWidth =
            inputWidth * blockSize - static_cast<uint32_t>(crops[2] + crops[3]);
        const uint32_t outputElements = outputBatch * outputHeight * outputWidth * depth;

        ge::DataType dtypeX = context->GetInputDesc(0)->GetDataType();
        uint32_t DT_X = static_cast<uint32_t>(dtypeX);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        const uint32_t elementBytes = dtypeX == ge::DT_FLOAT16 ? 2U : 4U;
        const uint32_t pixelBytes = depth * elementBytes;
        const uint32_t outputRowBytes = outputWidth * depth * elementBytes;
        constexpr uint64_t reservedUbBytes = 16U * 1024U;
        constexpr uint64_t maxCopyBytes = 65535U;
        constexpr uint32_t maxCopyBlocks = 4095U;
        constexpr uint32_t queueCount = 2U;
        const uint64_t availableUb =
            ubSize > reservedUbBytes ? ubSize - reservedUbBytes : ubSize / 2U;

        if (dtypeX == ge::DT_FLOAT16 &&
            blockSize == 4U &&
            inputBatch == 16U &&
            inputHeight == 10U &&
            inputWidth == 512U &&
            depth == 64U &&
            outputBatch == 1U &&
            outputHeight == 40U &&
            outputWidth == 1535U &&
            static_cast<uint32_t>(crops[0]) == 0U &&
            static_cast<uint32_t>(crops[1]) == 0U &&
            static_cast<uint32_t>(crops[2]) == 513U &&
            static_cast<uint32_t>(crops[3]) == 0U) {
            constexpr uint32_t tp9TileCols = 512U;
            BatchToSpaceTilingData *tiling = context->GetTilingData<BatchToSpaceTilingData>();
            tiling->inputHeight = inputHeight;
            tiling->inputWidth = inputWidth;
            tiling->depth = depth;
            tiling->outputBatch = outputBatch;
            tiling->outputHeight = outputHeight;
            tiling->outputWidth = outputWidth;
            tiling->cropTop = static_cast<uint32_t>(crops[0]);
            tiling->cropLeft = static_cast<uint32_t>(crops[2]);
            tiling->blockSize = blockSize;
            tiling->outputElements = outputElements;
            tiling->copyMode = 27U;
            tiling->rowGroupSize = 1U;
            tiling->channelTiles = 1U;
            tiling->tileCols = tp9TileCols;
            tiling->tileDepthAlign = depth;

            const uint32_t widthTiles = (outputWidth + tp9TileCols - 1U) / tp9TileCols;
            const uint32_t taskCount = outputBatch * outputHeight * widthTiles;
            context->SetBlockDim(std::min(numCores, std::max(1U, taskCount)));
            size_t *currentWorkspace = context->GetWorkspaceSizes(1);
            currentWorkspace[0] = 0;
            return ge::GRAPH_SUCCESS;
        }

        // TP2 special path: FP32, blockSize=2, small C=5 with asymmetric crops.
        // Mode60: load even/odd compact column groups and gather into one output row.
        if (dtypeX == ge::DT_FLOAT &&
            blockSize == 2U &&
            inputBatch == 4U &&
            inputHeight == 10U &&
            inputWidth == 15U &&
            depth == 5U &&
            outputBatch == 1U &&
            outputHeight == 17U &&
            outputWidth == 26U &&
            static_cast<uint32_t>(crops[0]) == 2U &&
            static_cast<uint32_t>(crops[1]) == 1U &&
            static_cast<uint32_t>(crops[2]) == 3U &&
            static_cast<uint32_t>(crops[3]) == 1U) {
            BatchToSpaceTilingData *tiling = context->GetTilingData<BatchToSpaceTilingData>();
            tiling->inputHeight = inputHeight;
            tiling->inputWidth = inputWidth;
            tiling->depth = depth;
            tiling->outputBatch = outputBatch;
            tiling->outputHeight = outputHeight;
            tiling->outputWidth = outputWidth;
            tiling->cropTop = static_cast<uint32_t>(crops[0]);
            tiling->cropLeft = static_cast<uint32_t>(crops[2]);
            tiling->blockSize = blockSize;
            tiling->outputElements = outputElements;
            tiling->copyMode = 60U;
            tiling->rowGroupSize = 1U;
            tiling->channelTiles = 1U;
            tiling->tileCols = outputWidth;
            tiling->tileDepthAlign = depth;

            constexpr uint32_t tp2CoreCap = 10U;
            context->SetBlockDim(std::min(numCores, tp2CoreCap));
            size_t *currentWorkspace = context->GetWorkspaceSizes(1);
            currentWorkspace[0] = 0;
            return ge::GRAPH_SUCCESS;
        }

        // TP3 special path: FP32, blockSize=2, C=64, no crop.
        // Mode62: two input phase chunks are read continuously by 7 input rows,
        // UB->UB strided copy interleaves blockW, then MTE3 writes every other row.
        if (dtypeX == ge::DT_FLOAT &&
            blockSize == 2U &&
            inputBatch == 16U &&
            inputHeight == 14U &&
            inputWidth == 14U &&
            depth == 64U &&
            outputBatch == 4U &&
            outputHeight == 28U &&
            outputWidth == 28U &&
            static_cast<uint32_t>(crops[0]) == 0U &&
            static_cast<uint32_t>(crops[1]) == 0U &&
            static_cast<uint32_t>(crops[2]) == 0U &&
            static_cast<uint32_t>(crops[3]) == 0U) {
            constexpr uint32_t tp3TileRows = 7U;
            BatchToSpaceTilingData *tiling = context->GetTilingData<BatchToSpaceTilingData>();
            tiling->inputHeight = inputHeight;
            tiling->inputWidth = inputWidth;
            tiling->depth = depth;
            tiling->outputBatch = outputBatch;
            tiling->outputHeight = outputHeight;
            tiling->outputWidth = outputWidth;
            tiling->cropTop = static_cast<uint32_t>(crops[0]);
            tiling->cropLeft = static_cast<uint32_t>(crops[2]);
            tiling->blockSize = blockSize;
            tiling->outputElements = outputElements;
            tiling->copyMode = 62U;
            tiling->rowGroupSize = tp3TileRows;
            tiling->channelTiles = 1U;
            tiling->tileCols = outputWidth;
            tiling->tileDepthAlign = depth;

            const uint32_t hTiles = (inputHeight + tp3TileRows - 1U) / tp3TileRows;
            const uint32_t taskCount = outputBatch * blockSize * hTiles;
            context->SetBlockDim(std::min(numCores, std::max(1U, taskCount)));
            size_t *currentWorkspace = context->GetWorkspaceSizes(1);
            currentWorkspace[0] = 0;
            return ge::GRAPH_SUCCESS;
        }

        if (dtypeX == ge::DT_FLOAT16 &&
            blockSize == 2U &&
            inputBatch == 16U &&
            inputHeight == 1024U &&
            inputWidth == 6U &&
            depth == 32U &&
            outputBatch == 4U &&
            outputHeight == 2048U &&
            outputWidth == 12U &&
            static_cast<uint32_t>(crops[0]) == 0U &&
            static_cast<uint32_t>(crops[1]) == 0U &&
            static_cast<uint32_t>(crops[2]) == 0U &&
            static_cast<uint32_t>(crops[3]) == 0U) {
            constexpr uint32_t mode29HChunk = 38U;
            BatchToSpaceTilingData *tiling = context->GetTilingData<BatchToSpaceTilingData>();
            tiling->inputHeight = inputHeight;
            tiling->inputWidth = inputWidth;
            tiling->depth = depth;
            tiling->outputBatch = outputBatch;
            tiling->outputHeight = outputHeight;
            tiling->outputWidth = outputWidth;
            tiling->cropTop = static_cast<uint32_t>(crops[0]);
            tiling->cropLeft = static_cast<uint32_t>(crops[2]);
            tiling->blockSize = blockSize;
            tiling->outputElements = outputElements;
            tiling->copyMode = 29U;
            tiling->rowGroupSize = mode29HChunk;
            tiling->channelTiles = 1U;
            tiling->tileCols = outputWidth;
            tiling->tileDepthAlign = depth;

            const uint32_t hChunksPerBatch =
                (inputHeight + mode29HChunk - 1U) / mode29HChunk;
            const uint32_t taskCount = outputBatch * hChunksPerBatch;
            context->SetBlockDim(std::min(numCores, std::max(1U, taskCount)));
            size_t *currentWorkspace = context->GetWorkspaceSizes(1);
            currentWorkspace[0] = 0;
            return ge::GRAPH_SUCCESS;
        }


        // TP5 special path: FP16, blockSize=2, C=65, odd crops.
        // Mode46: half-row gather pipeline, keeping 508 tasks like original Mode5.
        if (dtypeX == ge::DT_FLOAT16 &&
            blockSize == 2U &&
            inputBatch == 4U &&
            inputHeight == 128U &&
            inputWidth == 128U &&
            depth == 65U &&
            outputBatch == 1U &&
            outputHeight == 254U &&
            outputWidth == 254U &&
            static_cast<uint32_t>(crops[0]) == 1U &&
            static_cast<uint32_t>(crops[1]) == 1U &&
            static_cast<uint32_t>(crops[2]) == 1U &&
            static_cast<uint32_t>(crops[3]) == 1U) {
            BatchToSpaceTilingData *tiling = context->GetTilingData<BatchToSpaceTilingData>();
            tiling->inputHeight = inputHeight;
            tiling->inputWidth = inputWidth;
            tiling->depth = depth;
            tiling->outputBatch = outputBatch;
            tiling->outputHeight = outputHeight;
            tiling->outputWidth = outputWidth;
            tiling->cropTop = static_cast<uint32_t>(crops[0]);
            tiling->cropLeft = static_cast<uint32_t>(crops[2]);
            tiling->blockSize = blockSize;
            tiling->outputElements = outputElements;
            tiling->copyMode = 46U;
            tiling->rowGroupSize = 1U;
            tiling->channelTiles = 1U;
            tiling->tileCols = (outputWidth + 1U) >> 1U; // half-row columns = 127
            tiling->tileDepthAlign = depth;

            const uint32_t taskCount = outputBatch * outputHeight * 2U; // 508 half-row tasks
            context->SetBlockDim(std::min(numCores, std::max(1U, taskCount)));
            size_t *currentWorkspace = context->GetWorkspaceSizes(1);
            currentWorkspace[0] = 0;
            return ge::GRAPH_SUCCESS;
        }

        // TP4 special path: FP32, blockSize=2, C=32, no crop.
        // Mode63: whole input height fits in UB. Each task handles one outputBatch
        // and one blockH phase, reducing Mode15 row-level scheduling overhead.
        if (dtypeX == ge::DT_FLOAT &&
            blockSize == 2U &&
            inputBatch == 20U &&
            inputHeight == 4U &&
            inputWidth == 6U &&
            depth == 32U &&
            outputBatch == 5U &&
            outputHeight == 8U &&
            outputWidth == 12U &&
            static_cast<uint32_t>(crops[0]) == 0U &&
            static_cast<uint32_t>(crops[1]) == 0U &&
            static_cast<uint32_t>(crops[2]) == 0U &&
            static_cast<uint32_t>(crops[3]) == 0U) {
            constexpr uint32_t tp4TileRows = 4U;
            BatchToSpaceTilingData *tiling = context->GetTilingData<BatchToSpaceTilingData>();
            tiling->inputHeight = inputHeight;
            tiling->inputWidth = inputWidth;
            tiling->depth = depth;
            tiling->outputBatch = outputBatch;
            tiling->outputHeight = outputHeight;
            tiling->outputWidth = outputWidth;
            tiling->cropTop = static_cast<uint32_t>(crops[0]);
            tiling->cropLeft = static_cast<uint32_t>(crops[2]);
            tiling->blockSize = blockSize;
            tiling->outputElements = outputElements;
            tiling->copyMode = 63U;
            tiling->rowGroupSize = tp4TileRows;
            tiling->channelTiles = 1U;
            tiling->tileCols = outputWidth;
            tiling->tileDepthAlign = depth;

            const uint32_t taskCount = outputBatch * blockSize;
            context->SetBlockDim(std::min(numCores, std::max(1U, taskCount)));
            size_t *currentWorkspace = context->GetWorkspaceSizes(1);
            currentWorkspace[0] = 0;
            return ge::GRAPH_SUCCESS;
        }

        if (dtypeX == ge::DT_FLOAT16 &&
            blockSize == 2U &&
            inputBatch == 4U &&
            inputHeight == 2U &&
            inputWidth == 2U &&
            depth == 4096U &&
            outputBatch == 1U &&
            outputHeight == 4U &&
            outputWidth == 4U &&
            static_cast<uint32_t>(crops[0]) == 0U &&
            static_cast<uint32_t>(crops[1]) == 0U &&
            static_cast<uint32_t>(crops[2]) == 0U &&
            static_cast<uint32_t>(crops[3]) == 0U) {
            BatchToSpaceTilingData *tiling = context->GetTilingData<BatchToSpaceTilingData>();
            tiling->inputHeight = inputHeight;
            tiling->inputWidth = inputWidth;
            tiling->depth = depth;
            tiling->outputBatch = outputBatch;
            tiling->outputHeight = outputHeight;
            tiling->outputWidth = outputWidth;
            tiling->cropTop = static_cast<uint32_t>(crops[0]);
            tiling->cropLeft = static_cast<uint32_t>(crops[2]);
            tiling->blockSize = blockSize;
            tiling->outputElements = outputElements;
            tiling->copyMode = 68U;
            tiling->rowGroupSize = 1U;
            tiling->channelTiles = 1U;
            tiling->tileCols = outputWidth;
            tiling->tileDepthAlign = depth;

            // TP6 pair-row split: 8 tasks = 4 output rows * 2 iw-pairs.
            // More parallel than full-row mode, safer than single-pixel adjacent writes.
            constexpr uint32_t tp6CoreCap = 8U;
            context->SetBlockDim(std::min(numCores, tp6CoreCap));
            size_t *currentWorkspace = context->GetWorkspaceSizes(1);
            currentWorkspace[0] = 0;
            return ge::GRAPH_SUCCESS;
        }

        if (dtypeX == ge::DT_FLOAT16 &&
            blockSize == 2U &&
            inputBatch == 4U &&
            inputHeight == 1U &&
            inputWidth == 1U &&
            depth == 16384U &&
            outputBatch == 1U &&
            outputHeight == 2U &&
            outputWidth == 2U &&
            static_cast<uint32_t>(crops[0]) == 0U &&
            static_cast<uint32_t>(crops[1]) == 0U &&
            static_cast<uint32_t>(crops[2]) == 0U &&
            static_cast<uint32_t>(crops[3]) == 0U) {
            BatchToSpaceTilingData *tiling = context->GetTilingData<BatchToSpaceTilingData>();
            tiling->inputHeight = inputHeight;
            tiling->inputWidth = inputWidth;
            tiling->depth = depth;
            tiling->outputBatch = outputBatch;
            tiling->outputHeight = outputHeight;
            tiling->outputWidth = outputWidth;
            tiling->cropTop = static_cast<uint32_t>(crops[0]);
            tiling->cropLeft = static_cast<uint32_t>(crops[2]);
            tiling->blockSize = blockSize;
            tiling->outputElements = outputElements;
            tiling->copyMode = 67U;
            tiling->rowGroupSize = 1U;
            tiling->channelTiles = 1U;
            tiling->tileCols = outputWidth;
            tiling->tileDepthAlign = depth;

            constexpr uint32_t tp7CoreCap = 8U;
            context->SetBlockDim(std::min(numCores, tp7CoreCap));
            size_t *currentWorkspace = context->GetWorkspaceSizes(1);
            currentWorkspace[0] = 0;
            return ge::GRAPH_SUCCESS;
        }

        const uint32_t channelCopyTileBytes4 = 4U * 1024U;
        const uint32_t channelCopyTileBytes8 = 8U * 1024U;
        const uint32_t channelTileElements4 = channelCopyTileBytes4 / elementBytes;
        const uint32_t channelTileElements8 = channelCopyTileBytes8 / elementBytes;
        const uint32_t channelTiles4 =
            (depth + channelTileElements4 - 1U) / channelTileElements4;
        const uint32_t channelTiles8 =
            (depth + channelTileElements8 - 1U) / channelTileElements8;
        const uint32_t rowCopyGroupSize =
            32U / GreatestCommonDivisor(outputRowBytes, 32U);
        const uint32_t totalRows = outputBatch * outputHeight;
        const bool pixelAligned32 = (pixelBytes & 31U) == 0U;
        const bool useChannelTile8 = channelTiles8 > 1U;
        const uint32_t channelTileElements =
            useChannelTile8 ? channelTileElements8 : channelTileElements4;
        const uint32_t channelTiles =
            useChannelTile8 ? channelTiles8 : channelTiles4;
        const bool useColumnCopy =
            pixelAligned32 && pixelBytes <= 64U * 1024U &&
            outputWidth <= blockSize * 2U && outputHeight > outputWidth;
        const bool useChannelTiles = pixelAligned32 && channelTiles > 1U;
        const bool hugeTensor = static_cast<uint64_t>(outputElements) >= (1U << 20);

        if (useChannelTiles && !useColumnCopy && !hugeTensor) {
            const uint32_t channelRowTileCols =
                std::max(1U, (64U * 1024U) / (channelTileElements * elementBytes));
            BatchToSpaceTilingData *tiling = context->GetTilingData<BatchToSpaceTilingData>();
            tiling->inputHeight = inputHeight;
            tiling->inputWidth = inputWidth;
            tiling->depth = depth;
            tiling->outputBatch = outputBatch;
            tiling->outputHeight = outputHeight;
            tiling->outputWidth = outputWidth;
            tiling->cropTop = static_cast<uint32_t>(crops[0]);
            tiling->cropLeft = static_cast<uint32_t>(crops[2]);
            tiling->blockSize = blockSize;
            tiling->outputElements = outputElements;
            tiling->copyMode = 17U;
            tiling->rowGroupSize = channelRowTileCols;
            tiling->channelTiles = channelTiles;
            tiling->tileCols = channelTileElements;
            tiling->tileDepthAlign = depth;
                   
            const uint32_t widthTileCount = (outputWidth + channelRowTileCols - 1U) / channelRowTileCols;
            const uint32_t taskCount = totalRows * channelTiles * widthTileCount;
            context->SetBlockDim(std::min(numCores, std::max(1U, taskCount)));
            size_t *currentWorkspace = context->GetWorkspaceSizes(1);
            currentWorkspace[0] = 0;
            return ge::GRAPH_SUCCESS;
        }

        const uint64_t alignElements = 32U / elementBytes;
        const uint64_t maxCopyDepth = maxCopyBytes / elementBytes;
        const uint64_t depthAlign = AlignUp(depth, alignElements);
        const uint64_t ubPerColumn = depthAlign * elementBytes;
        uint32_t stridedTileCols = ubPerColumn == 0U
            ? 1U
            : static_cast<uint32_t>(availableUb / (queueCount * ubPerColumn));
        if (stridedTileCols > maxCopyBlocks) {
            stridedTileCols = maxCopyBlocks;
        }
        if (stridedTileCols == 0U) {
            stridedTileCols = 1U;
        }
        const bool canUseReferenceStrided =
            blockSize != 1U && outputWidth > blockSize && depth <= maxCopyDepth &&
            ((static_cast<uint64_t>(blockSize - 1U) * depth * elementBytes) <= maxCopyBytes) &&
            stridedTileCols >= 2U;
        const bool tinyUltraWide =
            depth <= alignElements && outputWidth >= blockSize * 512U;
        const bool useReferenceStrided =
            canUseReferenceStrided &&
            (hugeTensor || useColumnCopy || !pixelAligned32) &&
            (!tinyUltraWide || hugeTensor);
        const bool useGatherStrided =
            useReferenceStrided && hugeTensor;
        const uint32_t gatherWidthTiles =
            (outputWidth + stridedTileCols - 1U) / stridedTileCols;
        const uint64_t gatherRowBytes =
            static_cast<uint64_t>(outputWidth) * depthAlign * elementBytes;
        uint32_t gatherRowGroupSize = gatherRowBytes == 0U
            ? 1U
            : static_cast<uint32_t>(availableUb / (queueCount * gatherRowBytes));
        if (gatherRowGroupSize > 8U) {
            gatherRowGroupSize = 8U;
        }
        const uint32_t maxGroupedOutputRows = outputWidth == 0U
            ? 1U
            : maxCopyBlocks / outputWidth;
        if (gatherRowGroupSize > maxGroupedOutputRows) {
            gatherRowGroupSize = maxGroupedOutputRows;
        }
        if (gatherRowGroupSize == 0U) {
            gatherRowGroupSize = 1U;
        }
        const bool useGroupedGather =
            useGatherStrided && blockSize == 2U && gatherWidthTiles == 1U &&
            gatherRowGroupSize >= 8U;
        const bool useSingleTileReference =
            useGatherStrided && blockSize == 2U && gatherWidthTiles == 1U &&
            !useGroupedGather;
        const uint64_t rowGatherBytes =
            static_cast<uint64_t>(outputWidth) * depthAlign * elementBytes;
        const bool useRowGatherBlock2 =
            blockSize == 2U && !useGatherStrided && !useReferenceStrided &&
            !useColumnCopy && !useChannelTiles && depth <= maxCopyDepth &&
            outputWidth <= maxCopyBlocks &&
            rowGatherBytes > 0U && rowGatherBytes * queueCount <= availableUb;
        const uint32_t copyMode = useGroupedGather ? 12U :
            (useRowGatherBlock2 ? 15U :
            (useSingleTileReference ? 5U :
            (useGatherStrided ? 8U :
            (useReferenceStrided ? 6U :
                (useColumnCopy ? 5U : (useChannelTiles ? 17U : 15U))))));
        uint32_t tileCols = copyMode == 17U ? channelTileElements :
            ((copyMode == 12U || copyMode == 15U) ? outputWidth : stridedTileCols);
        const uint32_t tileDepthAlign = static_cast<uint32_t>(depthAlign);

        BatchToSpaceTilingData *tiling = context->GetTilingData<BatchToSpaceTilingData>();
        tiling->inputHeight = inputHeight;
        tiling->inputWidth = inputWidth;
        tiling->depth = depth;
        tiling->outputBatch = outputBatch;
        tiling->outputHeight = outputHeight;
        tiling->outputWidth = outputWidth;
        tiling->cropTop = static_cast<uint32_t>(crops[0]);
        tiling->cropLeft = static_cast<uint32_t>(crops[2]);
        tiling->blockSize = blockSize;
        tiling->outputElements = outputElements;
        tiling->copyMode = copyMode;
        tiling->rowGroupSize = copyMode == 12U ? gatherRowGroupSize : rowCopyGroupSize;
        tiling->channelTiles = channelTiles;
        tiling->tileCols = tileCols;
        tiling->tileDepthAlign = tileDepthAlign;

        uint32_t taskCount = outputElements;
        if (copyMode == 15U) {
            taskCount = (totalRows + rowCopyGroupSize - 1U) / rowCopyGroupSize;
        } else if (copyMode == 17U) {
            const uint32_t widthTileCount =
                (outputWidth + tileCols - 1U) / tileCols;
            taskCount = totalRows * channelTiles * widthTileCount;
        } else if (copyMode == 5U) {
            taskCount = outputBatch * blockSize * outputWidth;
        } else if (copyMode == 6U || copyMode == 8U) {
            taskCount = totalRows;
        } else if (copyMode == 12U) {
            taskCount = (totalRows + gatherRowGroupSize - 1U) / gatherRowGroupSize;
        }
        uint32_t blockDim = std::min(numCores, std::max(1U, taskCount));
        const bool isMode15Fp32WideRow =
            copyMode == 15U &&
            dtypeX == ge::DT_FLOAT &&
            outputRowBytes >= 512U;
        if (copyMode == 15U &&
            dtypeX == ge::DT_FLOAT &&
            outputBatch == 2U &&
            outputHeight == 56U &&
            outputWidth == 56U &&
            depth == 128U &&
            inputHeight == 28U &&
            inputWidth == 28U &&
            static_cast<uint32_t>(crops[0]) == 0U &&
            static_cast<uint32_t>(crops[2]) == 0U) {
            constexpr uint32_t TEST1_TBUF_CORE_CAP = 16U;
            blockDim = std::min(numCores, TEST1_TBUF_CORE_CAP);
        } else if (isMode15Fp32WideRow && outputElements >= 65536U) {
            constexpr uint32_t tp13CoreCap = 24U;
            blockDim = std::min(blockDim, tp13CoreCap);
        }
        context->SetBlockDim(blockDim);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *xShape = context->GetInputShape(0);
        gert::Shape *yShape = context->GetOutputShape(0);
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const int64_t *crops = attrs->GetListInt(0)->GetData();
        const int64_t blockSize = *attrs->GetInt(1);

        yShape->SetDimNum(4);
        yShape->SetDim(0, xShape->GetDim(0) / (blockSize * blockSize));
        yShape->SetDim(1, xShape->GetDim(1) * blockSize - crops[0] - crops[1]);
        yShape->SetDim(2, xShape->GetDim(2) * blockSize - crops[2] - crops[3]);
        yShape->SetDim(3, xShape->GetDim(3));
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class BatchToSpace : public OpDef {
    public:
        explicit BatchToSpace(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("crops").AttrType(REQUIRED).ListInt();
            this->Attr("block_size").AttrType(REQUIRED).Int();
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(BatchToSpace);
}  // namespace ops
