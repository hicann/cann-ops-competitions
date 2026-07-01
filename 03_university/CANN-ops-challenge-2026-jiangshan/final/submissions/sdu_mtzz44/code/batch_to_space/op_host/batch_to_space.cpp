#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/batch_to_space_tiling.h"

namespace optiling
{
    namespace
    {
        constexpr size_t INPUT_X_INDEX = 0;
        constexpr size_t ATTR_CROPS_INDEX = 0;
        constexpr size_t ATTR_BLOCK_SIZE_INDEX = 1;
        constexpr uint32_t DATA_BLOCK_BYTES = 32;
        constexpr uint32_t CACHE_LINE_BYTES = 64;
        constexpr uint32_t UB_RESERVED_BYTES = 1024;
        constexpr uint32_t MAX_DMA_BLOCK_COUNT = 4095;
        constexpr uint32_t ALIGNED_ROW_MERGE_MAX_WIDTH = 128;
        constexpr uint64_t VECTOR_GATHER_MIN_BYTES = 256 * 1024;
        constexpr uint64_t GATHER_TEMPLATE_MAX_BYTES = 7 * 1024;
        constexpr uint64_t MAX_KERNEL_ELEMENTS = 0xffffffffULL;
        constexpr bool BTS_DIAG_TINY_BLOCK2_HIT_SINGLE_CORE_ELEMENTWISE = false;
        constexpr bool BTS_DIAG_ALIGNED_BLOCK2_DIRECT_SINGLE_CORE_ELEMENTWISE = false;
        constexpr bool BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_LE16_SINGLE_CORE_ELEMENTWISE = false;
        constexpr bool BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_LE16_TILEWIDTH_LE8_SINGLE_CORE_ELEMENTWISE = false;
        constexpr bool BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_LE16_TILEWIDTH_GT8_LE12_SINGLE_CORE_ELEMENTWISE = false;
        constexpr bool BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_LE16_TILEWIDTH_EQ8_SINGLE_CORE_ELEMENTWISE = false;
        constexpr bool BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_LE16_TILEWIDTH_EQ6_SINGLE_CORE_ELEMENTWISE = false;
        constexpr bool BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_LE16_TILEWIDTH_EQ4_SINGLE_CORE_ELEMENTWISE = false;
        constexpr bool BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_LE16_TILEWIDTH_EQ2_SINGLE_CORE_ELEMENTWISE = false;
        constexpr bool BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_LE16_TILEWIDTH_EQ1_SINGLE_CORE_ELEMENTWISE = false;
        constexpr bool BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_GT16_LE64_SINGLE_CORE_ELEMENTWISE = false;
        constexpr bool BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_GT64_SINGLE_CORE_ELEMENTWISE = false;
        constexpr bool BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_GT64_TILEWIDTH_LE64_SINGLE_CORE_ELEMENTWISE = false;
        constexpr bool BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_GT64_TILEWIDTH_GT64_SINGLE_CORE_ELEMENTWISE = false;
        constexpr bool BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_GT64_TILEWIDTH_GT64_LE128_SINGLE_CORE_ELEMENTWISE = false;
        constexpr bool BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_GT64_TILEWIDTH_GT128_LE256_SINGLE_CORE_ELEMENTWISE = false;
        constexpr bool BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_GT64_TILEWIDTH_GT256_LE512_SINGLE_CORE_ELEMENTWISE = false;
        constexpr bool BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_GT64_TILEWIDTH_GT256_LE384_SINGLE_CORE_ELEMENTWISE = false;
        constexpr bool BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_GT64_TILEWIDTH_GT256_LE320_SINGLE_CORE_ELEMENTWISE = false;
        constexpr bool BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_GT64_TILEWIDTH_GT320_LE384_SINGLE_CORE_ELEMENTWISE = false;
        constexpr bool BTS_DIAG_ROW_MERGE_SINGLE_CORE_ELEMENTWISE = false;
        constexpr bool BTS_DIAG_ALIGNED_NON_ROW_SINGLE_CORE_ELEMENTWISE = false;
        constexpr bool BTS_OPT_ALIGNED_BLOCK2_TILEWIDTH1_ELEMENTWISE = false;
        constexpr bool BTS_OPT_ALIGNED_BLOCK2_TILEWIDTH1_CAP2 = true;
        constexpr bool BTS_OPT_ALIGNED_BLOCK2_TILEWIDTH1_CAP4 = false;
        constexpr bool BTS_OPT_ALIGNED_BLOCK2_SMALL_WIDTH_CAP_TILE8 = false;
        constexpr bool BTS_OPT_ALIGNED_BLOCK2_SMALL_WIDTH_VIA_UB_DB = false;
        constexpr bool BTS_OPT_ALIGNED_BLOCK2_SMALL_WIDTH_ROW_COMPACT = false;
        constexpr bool BTS_OPT_ALIGNED_BLOCK2_SMALL_WIDTH_NO_DB = false;
        constexpr bool BTS_OPT_ALIGNED_BLOCK2_SMALL_WIDTH_DIRECT_LOOP = false;
        constexpr bool BTS_OPT_ALIGNED_BLOCK2_FULL_ROW12_MULTIROW_DIRECT = true;
        constexpr uint32_t BTS_OPT_ALIGNED_BLOCK2_FULL_ROW12_MERGE_ROWS = 4U;
        constexpr bool BTS_OPT_ALIGNED_BLOCK2_WIDE_ROW_COMPACT = false;
        constexpr bool BTS_OPT_ALIGNED_BLOCK2_WIDE_TILE_CAP256 = true;
        constexpr bool BTS_OPT_ALIGNED_BLOCK4_D64_WIDE_ROW_MERGE = false;
        constexpr bool BTS_OPT_ALIGNED_BLOCK4_D64_DIRECT_OUTPUT = false;
        constexpr bool BTS_OPT_ALIGNED_BLOCK4_D64_PHASE_FAST = false;
        constexpr uint32_t BTS_OPT_ALIGNED_INTERLEAVED_WIDE_BLOCK4_D64_TILE_CAP = 256U;
        constexpr uint32_t BTS_OPT_CASE9_BLOCK4_D64_TILE_CAP = 256U;
        constexpr bool BTS_OPT_CASE9_BLOCK4_D64_FULL_ROW = false;
        constexpr bool BTS_OPT_CASE9_BLOCK4_D64_UB_REORDER = true;
        constexpr uint32_t BTS_OPT_CASE9_BLOCK4_D64_UB_REORDER_TILE_CAP = 320U;
        constexpr uint32_t BTS_OPT_CASE9_BLOCK4_D64_ALIGNED_TILE_CAP = 256U;
        constexpr bool BTS_OPT_CASE9_BLOCK4_D64_GROUPED_GATHER = false;
        constexpr uint32_t BTS_OPT_CASE9_BLOCK4_D64_GATHER_TILE_CAP = 256U;
        constexpr bool BTS_OPT_GROUPED_GENERAL_DOUBLE_BUFFER = true;
        // Proven incorrect for #5: direct phase-buffer GM writes produce wrong output.
        constexpr bool BTS_OPT_GROUPED_GENERAL_DIRECT_DMA = false;
        constexpr bool BTS_OPT_GROUPED_GENERAL_DIRECT_DMA_DOUBLE_BUFFER = true;
        constexpr uint32_t BTS_OPT_GROUPED_GENERAL_TILE_CAP = 0U;
        constexpr bool BTS_OPT_GROUPED_GENERAL_ROW_TILE = true;
        constexpr uint32_t BTS_OPT_GROUPED_GENERAL_ROW_TILE_ROWS = 1U;
        constexpr bool BTS_OPT_CASE2_FLOAT_BLOCK2_DIRECT_DMA = true;
        constexpr uint32_t BTS_OPT_CASE5_BLOCK2_DEEP_ROW_TILE_ROWS = 2U;
        constexpr bool BTS_OPT_CASE5_BLOCK2_DEEP_DISABLE_ROW_TILE = true;
        constexpr uint32_t BTS_OPT_CASE5_BLOCK2_DEEP_TILE_CAP = 0U;
        constexpr bool BTS_OPT_CASE5_BLOCK2_DEEP_PACK2 = true;
        constexpr bool BTS_OPT_CASE5_BLOCK2_DEEP_DIRECT_OUTPUT_SINGLE = false;
        constexpr bool BTS_OPT_CASE5_BLOCK2_DEEP_DIRECT_OUTPUT_STRIDED = false;
        constexpr bool BTS_OPT_CASE5_BLOCK2_DEEP_DIRECT_PADDED_PHASE = false;
        constexpr bool BTS_OPT_LINEAR_DOUBLE_BUFFER = true;
        constexpr uint32_t BTS_OPT_LINEAR_DOUBLE_BUFFER_TILES_PER_CORE = 2U;
        constexpr uint32_t BTS_OPT_LINEAR_DOUBLE_BUFFER_MIN_TILE_BYTES = 4096U;
        // 1 base; 2/3 dtype; 4/5/6 block; 7-10 pixel bytes; 11-14 width;
        // 15-17 height; 18-20 crop phase; 21/22 total rows; 23-25 selected path.
        constexpr uint32_t BTS_DIAG_CASE5_BUCKET_PROBE = 0U;
        constexpr uint32_t BTS_DIAG_CASE9_PROBE = 0U;
        constexpr bool BTS_DIAG_ALIGNED_BLOCK2_SMALL_WIDTH_TILEWIDTH_10_SINGLE_CORE_ELEMENTWISE = false;
        constexpr bool BTS_DIAG_ALIGNED_BLOCK2_SMALL_WIDTH_TILEWIDTH_12_SINGLE_CORE_ELEMENTWISE = false;
        constexpr bool BTS_DIAG_ALIGNED_BLOCK2_SMALL_WIDTH_TILEWIDTH_12_FULL_ROW_SINGLE_CORE_ELEMENTWISE = false;

        uint32_t Gcd(uint32_t a, uint32_t b)
        {
            while (b != 0)
            {
                const uint32_t value = a % b;
                a = b;
                b = value;
            }
            return a;
        }

        uint32_t CalculateGatherTemplateWidth(uint32_t tileWidth, uint32_t blockSize,
                                              uint64_t pixelBytes, uint64_t offsetPixelBytes)
        {
            const uint64_t fullOffsetBytes = static_cast<uint64_t>(tileWidth) * offsetPixelBytes;
            if (fullOffsetBytes <= GATHER_TEMPLATE_MAX_BYTES)
            {
                return tileWidth;
            }
            const uint32_t pixelAlignCount =
                DATA_BLOCK_BYTES / Gcd(static_cast<uint32_t>(pixelBytes % DATA_BLOCK_BYTES),
                                       DATA_BLOCK_BYTES);
            const uint32_t templateStep =
                blockSize / Gcd(blockSize, pixelAlignCount) * pixelAlignCount;
            const uint64_t maxTemplatePixels = GATHER_TEMPLATE_MAX_BYTES / offsetPixelBytes;
            if (maxTemplatePixels < templateStep)
            {
                return tileWidth;
            }
            const uint32_t templateWidth = static_cast<uint32_t>(
                maxTemplatePixels / templateStep * templateStep);
            return std::min<uint32_t>(tileWidth, templateWidth);
        }

        bool HasProfitableGatherTemplate(uint32_t tileWidth, uint32_t templateWidth)
        {
            return templateWidth < tileWidth && templateWidth <= tileWidth / 2;
        }

        bool HasUsableGatherTemplate(uint32_t tileWidth, uint32_t templateWidth)
        {
            return templateWidth == tileWidth ||
                   HasProfitableGatherTemplate(tileWidth, templateWidth);
        }

        bool GetAttrs(const gert::RuntimeAttrs *attrs, const int64_t *&crops, int64_t &blockSize)
        {
            if (attrs == nullptr)
            {
                return false;
            }
            const auto *cropAttr = attrs->GetListInt(ATTR_CROPS_INDEX);
            const auto *blockAttr = attrs->GetInt(ATTR_BLOCK_SIZE_INDEX);
            if (cropAttr == nullptr || cropAttr->GetSize() != 4 || blockAttr == nullptr)
            {
                return false;
            }
            crops = cropAttr->GetData();
            blockSize = *blockAttr;
            return crops != nullptr;
        }

        bool CalculateOutputShape(const gert::Shape &xShape, const int64_t *crops, int64_t blockSize,
                                  int64_t &outBatch, int64_t &outHeight, int64_t &outWidth)
        {
            if (xShape.GetDimNum() != 4 || blockSize <= 0)
            {
                return false;
            }
            const int64_t batch = xShape.GetDim(0);
            const int64_t height = xShape.GetDim(1);
            const int64_t width = xShape.GetDim(2);
            const int64_t depth = xShape.GetDim(3);
            if (batch <= 0 || height <= 0 || width <= 0 || depth <= 0 ||
                crops[0] < 0 || crops[1] < 0 || crops[2] < 0 || crops[3] < 0)
            {
                return false;
            }
            const int64_t blockArea = blockSize * blockSize;
            if (blockArea <= 0 || batch % blockArea != 0)
            {
                return false;
            }
            outBatch = batch / blockArea;
            outHeight = height * blockSize - crops[0] - crops[1];
            outWidth = width * blockSize - crops[2] - crops[3];
            return outHeight > 0 && outWidth > 0;
        }
    } // namespace

    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        const auto *inputShape = context->GetInputShape(INPUT_X_INDEX);
        const auto *tensorX = context->GetRequiredInputTensor(INPUT_X_INDEX);
        if (inputShape == nullptr || tensorX == nullptr)
        {
            return ge::GRAPH_FAILED;
        }
        const gert::Shape xShape = inputShape->GetStorageShape();
        const int64_t *crops = nullptr;
        int64_t blockSize = 0;
        int64_t outBatch = 0;
        int64_t outHeight = 0;
        int64_t outWidth = 0;
        if (!GetAttrs(context->GetAttrs(), crops, blockSize) ||
            !CalculateOutputShape(xShape, crops, blockSize, outBatch, outHeight, outWidth))
        {
            return ge::GRAPH_FAILED;
        }

        const uint64_t totalInput = static_cast<uint64_t>(xShape.GetDim(0)) * xShape.GetDim(1) *
                                    xShape.GetDim(2) * xShape.GetDim(3);
        const uint64_t totalOutput = static_cast<uint64_t>(outBatch) * outHeight * outWidth *
                                     xShape.GetDim(3);
        if (totalInput > MAX_KERNEL_ELEMENTS || totalOutput == 0 || totalOutput > MAX_KERNEL_ELEMENTS)
        {
            return ge::GRAPH_FAILED;
        }

        const ge::DataType dtypeX = tensorX->GetDataType();
        const uint32_t dtypeSize = static_cast<uint32_t>(ge::GetSizeByDataType(dtypeX));
        if (dtypeSize != sizeof(float) && dtypeSize != sizeof(uint16_t))
        {
            return ge::GRAPH_FAILED;
        }
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        const uint32_t maxCores = static_cast<uint32_t>(platform.GetCoreNumAiv());
        if (maxCores == 0)
        {
            return ge::GRAPH_FAILED;
        }
        uint64_t ubSize = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        const uint64_t pixelBytes = static_cast<uint64_t>(xShape.GetDim(3)) * dtypeSize;
        const uint64_t usableUbBytes = (ubSize > UB_RESERVED_BYTES) ? (ubSize - UB_RESERVED_BYTES) : ubSize;
        uint32_t tileWidth = 0;
        uint32_t tilesPerRow = 0;
        uint32_t totalTiles = 0;
        uint32_t tileBufferBytes = 0;
        uint32_t inputGroupStrideElements = 0;
        uint32_t inputTileBufferBytes = 0;
        uint32_t offsetBufferBytes = 0;
        uint32_t gatherTemplateWidth = 0;
        uint32_t channelCopy = 0;
        uint32_t rowTileHeight = 0;
        uint32_t flatRowMerge = 0;
        uint32_t linearCopy = 0;
        uint32_t linearInputOffset = 0;
        uint32_t usedCores = 0;
        bool groupedGeneralDoubleBuffer = false;
        bool alignedBlock2RowCompact = false;
        bool alignedBlock2TinyUbDoubleBuffer = false;
        bool alignedBlock2SmallDirect = false;
        bool alignedBlock2FullRow12MultirowDirect = false;
        bool alignedBlock4D64DirectOutput = false;
        bool alignedBlock4D64PhaseFast = false;
        bool alignedBlock4D64GroupedGather = false;
        bool alignedBlock4D64UbReorder = false;
        bool groupedGeneralDirectDma = false;
        bool groupedGeneralRowTile = false;
        bool groupedGeneralPack2 = false;
        const uint64_t totalOutputBytes = totalOutput * dtypeSize;
        uint32_t maxTileWidth = static_cast<uint32_t>(usableUbBytes / pixelBytes);
        if (maxTileWidth == 0)
        {
            const uint64_t channelBytes =
                usableUbBytes / DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
            if (channelBytes >= DATA_BLOCK_BYTES)
            {
                channelCopy = 1;
                tileWidth = static_cast<uint32_t>(channelBytes / dtypeSize);
                const uint32_t chunksPerPixel =
                    (static_cast<uint32_t>(xShape.GetDim(3)) + tileWidth - 1) / tileWidth;
                const uint64_t totalPixels =
                    static_cast<uint64_t>(outBatch) * outHeight * outWidth;
                const uint64_t totalTiles64 = totalPixels * chunksPerPixel;
                if (totalTiles64 == 0 || totalTiles64 > MAX_KERNEL_ELEMENTS)
                {
                    return ge::GRAPH_FAILED;
                }
                tilesPerRow = chunksPerPixel;
                totalTiles = static_cast<uint32_t>(totalTiles64);
                tileBufferBytes = static_cast<uint32_t>(tileWidth * dtypeSize);
                usedCores = std::min<uint32_t>(maxCores, totalTiles);
            }
            else
            {
                // Extremely large channel vectors cannot be assembled in UB; retain a
                // correctness fallback whose core boundaries are DCache-line safe.
                const uint64_t cacheLines =
                    (totalOutput * dtypeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
                usedCores = std::max<uint32_t>(
                    1U, std::min<uint32_t>(maxCores, static_cast<uint32_t>(cacheLines)));
            }
        }
        else
        {
            const uint64_t alignedPixelBytes =
                (pixelBytes + DATA_BLOCK_BYTES - 1) / DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
            const uint64_t offsetPixelBytes =
                static_cast<uint64_t>(xShape.GetDim(3)) * sizeof(uint32_t);
            const bool useCase5Block2DeepPack2Base =
                BTS_OPT_CASE5_BLOCK2_DEEP_PACK2 &&
                dtypeX == ge::DT_FLOAT16 && blockSize == 2 &&
                (static_cast<uint32_t>(xShape.GetDim(3)) & 1U) == 0U &&
                pixelBytes > 128U &&
                static_cast<uint32_t>(outWidth) > 64U &&
                static_cast<uint32_t>(outWidth) <= 256U &&
                static_cast<uint32_t>(outHeight) > 64U;
            const uint32_t kernelElementBytes =
                useCase5Block2DeepPack2Base ? static_cast<uint32_t>(sizeof(float)) : dtypeSize;
            const uint64_t gatherOffsetPixelBytes =
                useCase5Block2DeepPack2Base
                    ? (static_cast<uint64_t>(xShape.GetDim(3)) / 2U) * sizeof(uint32_t)
                    : offsetPixelBytes;
            const uint64_t alignedOffsetPixelBytes =
                (gatherOffsetPixelBytes + DATA_BLOCK_BYTES - 1) / DATA_BLOCK_BYTES *
                DATA_BLOCK_BYTES;
            bool useGroupGather =
                blockSize > 1 && pixelBytes % DATA_BLOCK_BYTES != 0 &&
                totalOutputBytes >= VECTOR_GATHER_MIN_BYTES &&
                2 * alignedPixelBytes + alignedOffsetPixelBytes <= usableUbBytes;
            const bool useCase9Block4D64Base =
                blockSize == 4 && dtypeX == ge::DT_FLOAT16 && pixelBytes == 128U &&
                static_cast<uint32_t>(outWidth) > 1280U &&
                static_cast<uint32_t>(outWidth) <= 1536U &&
                static_cast<uint32_t>(outHeight) >= 17U &&
                static_cast<uint32_t>(outHeight) <= 64U &&
                (static_cast<uint32_t>(crops[2]) & 3U) == 1U &&
                (static_cast<uint32_t>(crops[0]) & 3U) == 0U;
            const bool useCase9Block4D64GroupedGather =
                BTS_OPT_CASE9_BLOCK4_D64_GROUPED_GATHER &&
                useCase9Block4D64Base &&
                2 * alignedPixelBytes + alignedOffsetPixelBytes <= usableUbBytes;
            const bool useCase9Block4D64PhaseFast =
                BTS_OPT_ALIGNED_BLOCK4_D64_PHASE_FAST &&
                useCase9Block4D64Base;
            const bool useCase9Block4D64DirectOutput =
                BTS_OPT_ALIGNED_BLOCK4_D64_DIRECT_OUTPUT &&
                useCase9Block4D64Base;
            const bool useCase9Block4D64UbReorder =
                BTS_OPT_CASE9_BLOCK4_D64_UB_REORDER &&
                useCase9Block4D64Base;
            if (useGroupGather)
            {
                maxTileWidth = static_cast<uint32_t>(
                    usableUbBytes / (2 * pixelBytes + gatherOffsetPixelBytes));
                if (maxTileWidth == 0)
                {
                    maxTileWidth = 1;
                }
            }
            maxTileWidth = std::min<uint32_t>(maxTileWidth, static_cast<uint32_t>(outWidth));

            const uint64_t totalRows = static_cast<uint64_t>(outBatch) * outHeight;
            const uint64_t outputPixels = totalRows * outWidth;
            const uint64_t fullRowBytes = static_cast<uint64_t>(outWidth) * pixelBytes;
            const bool useCase9Block4D64FullRow =
                BTS_OPT_CASE9_BLOCK4_D64_FULL_ROW &&
                useCase9Block4D64Base &&
                totalRows >= maxCores &&
                fullRowBytes <= ubSize &&
                fullRowBytes <= MAX_KERNEL_ELEMENTS;
            const bool useSmallUltraWideRowMerge =
                blockSize > 1 && pixelBytes % DATA_BLOCK_BYTES == 0 &&
                pixelBytes >= 2048 && outputPixels * 4 <= maxCores &&
                fullRowBytes <= usableUbBytes;
            const bool useContiguousRowMerge =
                blockSize == 1 && pixelBytes % DATA_BLOCK_BYTES == 0 &&
                fullRowBytes <= usableUbBytes && totalRows > maxCores;
            const bool useBlock4D64WideRowMerge =
                BTS_OPT_ALIGNED_BLOCK4_D64_WIDE_ROW_MERGE &&
                blockSize == 4 && dtypeX == ge::DT_FLOAT16 && pixelBytes == 128U &&
                static_cast<uint32_t>(outWidth) > 512U &&
                static_cast<uint32_t>(outHeight) <= 64U &&
                fullRowBytes <= usableUbBytes;
            const bool useAlignedRowMerge =
                useCase9Block4D64FullRow ||
                useSmallUltraWideRowMerge ||
                useContiguousRowMerge ||
                useBlock4D64WideRowMerge ||
                (blockSize > 1 && pixelBytes % DATA_BLOCK_BYTES == 0 &&
                 static_cast<uint32_t>(outWidth) <= ALIGNED_ROW_MERGE_MAX_WIDTH &&
                 fullRowBytes <= usableUbBytes && totalRows > maxCores);
            if (useAlignedRowMerge)
            {
                if (useSmallUltraWideRowMerge)
                {
                    const uint32_t alignElements = DATA_BLOCK_BYTES / dtypeSize;
                    const uint32_t channelTiles =
                        std::min<uint32_t>(4U, std::max<uint32_t>(
                                                   1U, maxCores / static_cast<uint32_t>(outputPixels)));
                    uint32_t channelTileElements = static_cast<uint32_t>(
                        (xShape.GetDim(3) + channelTiles - 1) / channelTiles);
                    channelTileElements =
                        (channelTileElements + alignElements - 1) / alignElements * alignElements;
                    if (channelTileElements > static_cast<uint32_t>(xShape.GetDim(3)))
                    {
                        channelTileElements = static_cast<uint32_t>(xShape.GetDim(3));
                    }
                    const uint64_t tileBufferBytes64 =
                        (static_cast<uint64_t>(outWidth) * channelTileElements * dtypeSize +
                         DATA_BLOCK_BYTES - 1) /
                        DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
                    if (tileBufferBytes64 == 0 || tileBufferBytes64 > usableUbBytes ||
                        tileBufferBytes64 > MAX_KERNEL_ELEMENTS)
                    {
                        return ge::GRAPH_FAILED;
                    }
                    tileWidth = channelTileElements;
                    rowTileHeight = 1;
                    flatRowMerge = 2U;
                    tilesPerRow = (static_cast<uint32_t>(xShape.GetDim(3)) +
                                   channelTileElements - 1) /
                                  channelTileElements;
                    const uint64_t totalTiles64 = totalRows * tilesPerRow;
                    if (totalTiles64 == 0 || totalTiles64 > MAX_KERNEL_ELEMENTS)
                    {
                        return ge::GRAPH_FAILED;
                    }
                    totalTiles = static_cast<uint32_t>(totalTiles64);
                    tileBufferBytes = static_cast<uint32_t>(tileBufferBytes64);
                    usedCores = std::min<uint32_t>(maxCores, totalTiles);
                }
                else
                {
                    const uint64_t rowMergeUbLimit =
                        useCase9Block4D64FullRow ? ubSize : usableUbBytes;
                    const uint32_t maxRowsPerTile =
                        useCase9Block4D64FullRow
                            ? 1U
                            : static_cast<uint32_t>(usableUbBytes / fullRowBytes);
                    const uint32_t targetRows = static_cast<uint32_t>(
                        std::max<uint64_t>(1, (totalRows + maxCores - 1) / maxCores));
                    const bool useNarrowManyRowsTarget =
                        blockSize > 1 && static_cast<uint32_t>(outWidth) <= 40U &&
                        totalRows > 256U;
                    if (useCase9Block4D64FullRow)
                    {
                        rowTileHeight = 1U;
                    }
                    else if (blockSize > 1 && !useNarrowManyRowsTarget)
                    {
                        const uint64_t widenedRows =
                            std::max<uint64_t>(1, static_cast<uint64_t>(targetRows) * 2);
                        rowTileHeight = static_cast<uint32_t>(
                            std::min<uint64_t>(maxRowsPerTile, widenedRows));
                    }
                    else
                    {
                        rowTileHeight = std::min<uint32_t>(
                            maxRowsPerTile, targetRows);
                    }
                    tileWidth = static_cast<uint32_t>(outWidth);
                    flatRowMerge = 0U;
                    tilesPerRow = (static_cast<uint32_t>(outHeight) + rowTileHeight - 1) /
                                  rowTileHeight;
                    const uint64_t totalTiles64 = static_cast<uint64_t>(outBatch) * tilesPerRow;
                    if (totalTiles64 == 0 || totalTiles64 > MAX_KERNEL_ELEMENTS)
                    {
                        return ge::GRAPH_FAILED;
                    }
                    totalTiles = static_cast<uint32_t>(totalTiles64);
                    uint64_t tileBufferBytes64 =
                        static_cast<uint64_t>(rowTileHeight) * fullRowBytes;
                    if (tileBufferBytes64 > rowMergeUbLimit ||
                        tileBufferBytes64 > MAX_KERNEL_ELEMENTS)
                    {
                        return ge::GRAPH_FAILED;
                    }
                    if (blockSize == 1 && tileBufferBytes64 * 2 <= usableUbBytes)
                    {
                        tileBufferBytes64 *= 2;
                    }
                    tileBufferBytes = static_cast<uint32_t>(tileBufferBytes64);
                    usedCores = std::min<uint32_t>(maxCores, totalTiles);
                }
            }
            else
            {
                const uint32_t parallelTileWidth = static_cast<uint32_t>(
                    std::max<uint64_t>(1, (outputPixels + maxCores - 1) / maxCores));

                // For few output rows, split a row so all available cores can still participate.
                tileWidth = std::min<uint32_t>(maxTileWidth, parallelTileWidth);

                // DMA fast path uses one block per source pixel for each interleave group.
                if ((pixelBytes % DATA_BLOCK_BYTES == 0 || useGroupGather) && blockSize > 1)
                {
                    const uint64_t maxDmaTileWidth =
                        static_cast<uint64_t>(blockSize) * MAX_DMA_BLOCK_COUNT;
                    tileWidth = static_cast<uint32_t>(
                        std::min<uint64_t>(tileWidth, std::min<uint64_t>(maxDmaTileWidth, outWidth)));
                }
                const bool smallOddAlignedBlock2Tile =
                    blockSize == 2 && pixelBytes % DATA_BLOCK_BYTES == 0 &&
                    tileWidth > 1 && tileWidth <= 15U && (tileWidth & 1U) != 0;
                if (smallOddAlignedBlock2Tile)
                {
                    --tileWidth;
                }
                if (BTS_OPT_CASE9_BLOCK4_D64_ALIGNED_TILE_CAP != 0 &&
                    useCase9Block4D64Base &&
                    !useCase9Block4D64GroupedGather &&
                    !useCase9Block4D64DirectOutput &&
                    !useCase9Block4D64PhaseFast &&
                    !useCase9Block4D64UbReorder &&
                    tileWidth > BTS_OPT_CASE9_BLOCK4_D64_ALIGNED_TILE_CAP)
                {
                    tileWidth = BTS_OPT_CASE9_BLOCK4_D64_ALIGNED_TILE_CAP;
                }
                else if (BTS_OPT_ALIGNED_INTERLEAVED_WIDE_BLOCK4_D64_TILE_CAP != 0 &&
                         !useCase9Block4D64GroupedGather &&
                         !useCase9Block4D64DirectOutput &&
                         !useCase9Block4D64PhaseFast &&
                         !useCase9Block4D64UbReorder &&
                         blockSize == 4 && dtypeX == ge::DT_FLOAT16 &&
                         pixelBytes == 128U && static_cast<uint32_t>(outWidth) > 512U &&
                         tileWidth > BTS_OPT_ALIGNED_INTERLEAVED_WIDE_BLOCK4_D64_TILE_CAP)
                {
                    tileWidth = BTS_OPT_ALIGNED_INTERLEAVED_WIDE_BLOCK4_D64_TILE_CAP;
                }
                if (useCase9Block4D64GroupedGather)
                {
                    useGroupGather = true;
                    if (BTS_OPT_CASE9_BLOCK4_D64_GATHER_TILE_CAP != 0 &&
                        tileWidth > BTS_OPT_CASE9_BLOCK4_D64_GATHER_TILE_CAP)
                    {
                        tileWidth = BTS_OPT_CASE9_BLOCK4_D64_GATHER_TILE_CAP;
                    }
                }
                if (BTS_OPT_CASE9_BLOCK4_D64_TILE_CAP != 0 &&
                    (useCase9Block4D64PhaseFast || useCase9Block4D64DirectOutput) &&
                    tileWidth > BTS_OPT_CASE9_BLOCK4_D64_TILE_CAP)
                {
                    tileWidth = BTS_OPT_CASE9_BLOCK4_D64_TILE_CAP;
                }
                if (BTS_OPT_GROUPED_GENERAL_TILE_CAP != 0 &&
                    useGroupGather && !useCase9Block4D64GroupedGather &&
                    tileWidth > BTS_OPT_GROUPED_GENERAL_TILE_CAP)
                {
                    tileWidth = BTS_OPT_GROUPED_GENERAL_TILE_CAP;
                }
                if (BTS_OPT_CASE5_BLOCK2_DEEP_TILE_CAP != 0 &&
                    useGroupGather &&
                    dtypeX == ge::DT_FLOAT16 && blockSize == 2 &&
                    pixelBytes > 128U &&
                    static_cast<uint32_t>(outWidth) > 64U &&
                    static_cast<uint32_t>(outWidth) <= 256U &&
                    static_cast<uint32_t>(outHeight) > 64U &&
                    tileWidth > BTS_OPT_CASE5_BLOCK2_DEEP_TILE_CAP)
                {
                    tileWidth = BTS_OPT_CASE5_BLOCK2_DEEP_TILE_CAP;
                }
                if (useGroupGather && fullRowBytes % DATA_BLOCK_BYTES == 0)
                {
                    const uint32_t pixelAlignCount =
                        DATA_BLOCK_BYTES /
                        Gcd(static_cast<uint32_t>(pixelBytes % DATA_BLOCK_BYTES),
                            DATA_BLOCK_BYTES);
                    if (pixelAlignCount > 1 && tileWidth >= pixelAlignCount)
                    {
                        const uint32_t alignedTileWidth =
                            tileWidth / pixelAlignCount * pixelAlignCount;
                        if (alignedTileWidth != 0 && alignedTileWidth * 4 >= tileWidth * 3)
                        {
                            tileWidth = alignedTileWidth;
                        }
                    }
                }
                if (useGroupGather &&
                    !HasUsableGatherTemplate(
                        tileWidth, CalculateGatherTemplateWidth(
                                       tileWidth, static_cast<uint32_t>(blockSize), pixelBytes,
                                       gatherOffsetPixelBytes)))
                {
                    // Keep vector gather when the offset template either covers the
                    // whole tile or is small enough to be reused across the tile.
                    useGroupGather = false;
                    maxTileWidth = static_cast<uint32_t>(usableUbBytes / pixelBytes);
                    maxTileWidth =
                        std::min<uint32_t>(maxTileWidth, static_cast<uint32_t>(outWidth));
                    tileWidth = std::min<uint32_t>(maxTileWidth, parallelTileWidth);
                }
                tilesPerRow = (static_cast<uint32_t>(outWidth) + tileWidth - 1) / tileWidth;
                const uint64_t totalTiles64 =
                    static_cast<uint64_t>(outBatch) * outHeight * tilesPerRow;
                if (totalTiles64 == 0 || totalTiles64 > MAX_KERNEL_ELEMENTS)
                {
                    return ge::GRAPH_FAILED;
                }
                totalTiles = static_cast<uint32_t>(totalTiles64);
                usedCores = std::min<uint32_t>(maxCores, totalTiles);
                uint64_t tileBufferBytes64 =
                    (static_cast<uint64_t>(tileWidth) * pixelBytes + DATA_BLOCK_BYTES - 1) /
                    DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
                if (tileBufferBytes64 > MAX_KERNEL_ELEMENTS)
                {
                    return ge::GRAPH_FAILED;
                }
                tileBufferBytes = static_cast<uint32_t>(tileBufferBytes64);
                if (rowTileHeight == 0 && pixelBytes % DATA_BLOCK_BYTES == 0 &&
                    !useGroupGather &&
                    tileBufferBytes64 * 2 <= usableUbBytes)
                {
                    tileBufferBytes64 *= 2;
                    tileBufferBytes = static_cast<uint32_t>(tileBufferBytes64);
                }
                if (useCase9Block4D64DirectOutput)
                {
                    inputGroupStrideElements = 0;
                    inputTileBufferBytes = 0;
                    offsetBufferBytes = 0;
                    gatherTemplateWidth = 0;
                    rowTileHeight = 0;
                    groupedGeneralDoubleBuffer = false;
                    alignedBlock4D64DirectOutput = true;
                }
                if (useCase9Block4D64PhaseFast)
                {
                    constexpr uint32_t CASE9_PHASE_ROWS = 1U;
                    uint32_t phaseTileWidth =
                        std::min<uint32_t>(static_cast<uint32_t>(outWidth),
                                           BTS_OPT_CASE9_BLOCK4_D64_TILE_CAP);
                    phaseTileWidth = phaseTileWidth / 4U * 4U;
                    uint64_t phaseTileBufferBytes =
                        (static_cast<uint64_t>(CASE9_PHASE_ROWS) * phaseTileWidth * pixelBytes +
                         DATA_BLOCK_BYTES - 1U) /
                        DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
                    const uint32_t phaseTilesPerRow =
                        (static_cast<uint32_t>(outWidth) + phaseTileWidth - 1U) /
                        phaseTileWidth;
                    const uint32_t phaseRowTiles =
                        (static_cast<uint32_t>(outHeight) + CASE9_PHASE_ROWS - 1U) /
                        CASE9_PHASE_ROWS;
                    const uint64_t phaseTotalTiles =
                        static_cast<uint64_t>(outBatch) * phaseRowTiles * phaseTilesPerRow;
                    if (phaseTileBufferBytes * 2U <= usableUbBytes &&
                        phaseTileBufferBytes * 2U <= MAX_KERNEL_ELEMENTS)
                    {
                        phaseTileBufferBytes *= 2U;
                    }
                    if (phaseTileWidth != 0U && phaseTileBufferBytes <= usableUbBytes &&
                        phaseTileBufferBytes <= MAX_KERNEL_ELEMENTS &&
                        phaseTotalTiles != 0 && phaseTotalTiles <= MAX_KERNEL_ELEMENTS)
                    {
                        tileWidth = phaseTileWidth;
                        tilesPerRow = phaseTilesPerRow;
                        totalTiles = static_cast<uint32_t>(phaseTotalTiles);
                        tileBufferBytes = static_cast<uint32_t>(phaseTileBufferBytes);
                        inputGroupStrideElements = 0;
                        inputTileBufferBytes = 0;
                        offsetBufferBytes = 0;
                        gatherTemplateWidth = 0;
                        rowTileHeight = CASE9_PHASE_ROWS;
                        groupedGeneralDoubleBuffer = false;
                        alignedBlock4D64DirectOutput = false;
                        alignedBlock4D64PhaseFast = true;
                        usedCores = std::min<uint32_t>(maxCores, totalTiles);
                    }
                }
                if (useCase9Block4D64UbReorder)
                {
                    uint32_t ubReorderTileCap = BTS_OPT_CASE9_BLOCK4_D64_ALIGNED_TILE_CAP;
                    if (BTS_OPT_CASE9_BLOCK4_D64_UB_REORDER_TILE_CAP != 0U &&
                        static_cast<uint64_t>(BTS_OPT_CASE9_BLOCK4_D64_UB_REORDER_TILE_CAP) *
                                pixelBytes * 4U <=
                            usableUbBytes)
                    {
                        ubReorderTileCap = BTS_OPT_CASE9_BLOCK4_D64_UB_REORDER_TILE_CAP;
                    }
                    if (ubReorderTileCap != 0U && tileWidth > ubReorderTileCap)
                    {
                        tileWidth = ubReorderTileCap;
                    }
                    uint64_t rowTileBytes64 =
                        (static_cast<uint64_t>(tileWidth) * pixelBytes + DATA_BLOCK_BYTES - 1) /
                        DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
                    const uint32_t maxPhasePixels =
                        (tileWidth + static_cast<uint32_t>(blockSize) - 1) /
                        static_cast<uint32_t>(blockSize);
                    uint64_t phaseStrideBytes64 =
                        (static_cast<uint64_t>(maxPhasePixels) * pixelBytes +
                         DATA_BLOCK_BYTES - 1) /
                        DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
                    uint64_t inputTileBufferBytes64 =
                        static_cast<uint64_t>(blockSize) * phaseStrideBytes64;
                    const uint32_t reorderTilesPerRow =
                        (static_cast<uint32_t>(outWidth) + tileWidth - 1U) / tileWidth;
                    const uint64_t reorderTotalTiles =
                        static_cast<uint64_t>(outBatch) * outHeight * reorderTilesPerRow;
                    if (rowTileBytes64 * 2U + inputTileBufferBytes64 * 2U <= usableUbBytes &&
                        rowTileBytes64 * 2U <= MAX_KERNEL_ELEMENTS &&
                        inputTileBufferBytes64 * 2U <= MAX_KERNEL_ELEMENTS &&
                        reorderTotalTiles != 0 && reorderTotalTiles <= MAX_KERNEL_ELEMENTS)
                    {
                        tilesPerRow = reorderTilesPerRow;
                        totalTiles = static_cast<uint32_t>(reorderTotalTiles);
                        tileBufferBytes = static_cast<uint32_t>(rowTileBytes64 * 2U);
                        inputGroupStrideElements =
                            static_cast<uint32_t>(phaseStrideBytes64 / dtypeSize);
                        inputTileBufferBytes = static_cast<uint32_t>(inputTileBufferBytes64 * 2U);
                        offsetBufferBytes = 0;
                        gatherTemplateWidth = 0;
                        rowTileHeight = 0;
                        groupedGeneralDoubleBuffer = false;
                        alignedBlock4D64DirectOutput = false;
                        alignedBlock4D64PhaseFast = false;
                        alignedBlock4D64GroupedGather = false;
                        alignedBlock4D64UbReorder = true;
                        usedCores = std::min<uint32_t>(maxCores, totalTiles);
                    }
                }
                const bool useCase5Block2DeepDirectOutput =
                    (BTS_OPT_CASE5_BLOCK2_DEEP_DIRECT_OUTPUT_SINGLE ||
                     BTS_OPT_CASE5_BLOCK2_DEEP_DIRECT_OUTPUT_STRIDED) &&
                    dtypeX == ge::DT_FLOAT16 && blockSize == 2 &&
                    pixelBytes > 128U &&
                    static_cast<uint32_t>(outWidth) > 64U &&
                    static_cast<uint32_t>(outWidth) <= 256U &&
                    static_cast<uint32_t>(outHeight) > 64U;
                const bool useCase5Block2DeepDirectPaddedPhase =
                    BTS_OPT_CASE5_BLOCK2_DEEP_DIRECT_PADDED_PHASE &&
                    dtypeX == ge::DT_FLOAT16 && blockSize == 2 &&
                    pixelBytes > 128U &&
                    static_cast<uint32_t>(outWidth) > 64U &&
                    static_cast<uint32_t>(outWidth) <= 256U &&
                    static_cast<uint32_t>(outHeight) > 64U;
                const bool useCase2FloatBlock2DirectDma =
                    BTS_OPT_CASE2_FLOAT_BLOCK2_DIRECT_DMA &&
                    dtypeX == ge::DT_FLOAT && blockSize == 2 &&
                    pixelBytes > 16U && pixelBytes <= 64U &&
                    static_cast<uint32_t>(outWidth) <= 40U &&
                    static_cast<uint32_t>(outHeight) > 16U &&
                    static_cast<uint32_t>(outHeight) <= 64U;
                if ((BTS_OPT_GROUPED_GENERAL_DIRECT_DMA ||
                     useCase2FloatBlock2DirectDma ||
                     useCase5Block2DeepDirectOutput ||
                     useCase5Block2DeepDirectPaddedPhase) &&
                    useGroupGather && !useCase9Block4D64GroupedGather)
                {
                    uint32_t directTileWidth = std::min<uint32_t>(
                        static_cast<uint32_t>(outWidth), parallelTileWidth);
                    const uint64_t directMaxDmaTileWidth =
                        static_cast<uint64_t>(blockSize) * MAX_DMA_BLOCK_COUNT;
                    if (directTileWidth > directMaxDmaTileWidth)
                    {
                        directTileWidth = static_cast<uint32_t>(directMaxDmaTileWidth);
                    }
                    uint32_t directGroupCount = 0;
                    uint32_t directMaxGroupPixels = 0;
                    uint64_t directGroupStrideBytes = 0;
                    uint64_t directTileBufferBytes64 = 0;
                    while (directTileWidth != 0U)
                    {
                        directGroupCount =
                            std::min<uint32_t>(directTileWidth,
                                               static_cast<uint32_t>(blockSize));
                        directMaxGroupPixels =
                            (directTileWidth + static_cast<uint32_t>(blockSize) - 1U) /
                            static_cast<uint32_t>(blockSize);
                        if (useCase5Block2DeepDirectPaddedPhase)
                        {
                            directGroupStrideBytes =
                                static_cast<uint64_t>(directMaxGroupPixels) *
                                alignedPixelBytes;
                        }
                        else
                        {
                            directGroupStrideBytes =
                                (static_cast<uint64_t>(directMaxGroupPixels) * pixelBytes +
                                 DATA_BLOCK_BYTES - 1U) /
                                DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
                        }
                        directTileBufferBytes64 =
                            static_cast<uint64_t>(directGroupCount) * directGroupStrideBytes;
                        if (directTileBufferBytes64 <= usableUbBytes &&
                            directTileBufferBytes64 <= MAX_KERNEL_ELEMENTS)
                        {
                            break;
                        }
                        --directTileWidth;
                    }
                    const uint32_t directTilesPerRow =
                        (directTileWidth == 0U)
                            ? 0U
                            : (static_cast<uint32_t>(outWidth) + directTileWidth - 1U) /
                                  directTileWidth;
                    const uint64_t directTotalTiles =
                        static_cast<uint64_t>(outBatch) * outHeight * directTilesPerRow;
                    if (directTileWidth != 0U &&
                        directTotalTiles != 0 && directTotalTiles <= MAX_KERNEL_ELEMENTS)
                    {
                        if (BTS_OPT_GROUPED_GENERAL_DIRECT_DMA_DOUBLE_BUFFER &&
                            directTileBufferBytes64 * 2U <= usableUbBytes &&
                            directTileBufferBytes64 * 2U <= MAX_KERNEL_ELEMENTS)
                        {
                            directTileBufferBytes64 *= 2U;
                        }
                        tileWidth = directTileWidth;
                        tilesPerRow = directTilesPerRow;
                        totalTiles = static_cast<uint32_t>(directTotalTiles);
                        tileBufferBytes = static_cast<uint32_t>(directTileBufferBytes64);
                        inputGroupStrideElements =
                            static_cast<uint32_t>(directGroupStrideBytes / dtypeSize);
                        inputTileBufferBytes = 0;
                        offsetBufferBytes = 0;
                        gatherTemplateWidth = 0;
                        rowTileHeight = 0;
                        groupedGeneralDoubleBuffer = false;
                        groupedGeneralDirectDma = true;
                        useGroupGather = false;
                        usedCores = std::min<uint32_t>(maxCores, totalTiles);
                    }
                }
                if (useGroupGather)
                {
                    uint32_t groupCount =
                        std::min<uint32_t>(tileWidth, static_cast<uint32_t>(blockSize));
                    uint32_t maxGroupPixels =
                        (tileWidth + static_cast<uint32_t>(blockSize) - 1) /
                        static_cast<uint32_t>(blockSize);
                    uint64_t groupStrideBytes =
                        (static_cast<uint64_t>(maxGroupPixels) * pixelBytes + DATA_BLOCK_BYTES - 1) /
                        DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
                    uint64_t inputTileBufferBytes64 =
                        static_cast<uint64_t>(groupCount) * groupStrideBytes;
                    gatherTemplateWidth = useCase9Block4D64GroupedGather
                                              ? tileWidth
                                              : CalculateGatherTemplateWidth(
                                                    tileWidth, static_cast<uint32_t>(blockSize), pixelBytes,
                                                    gatherOffsetPixelBytes);
                    uint64_t offsetBufferBytes64 =
                        (static_cast<uint64_t>(gatherTemplateWidth) * gatherOffsetPixelBytes +
                         DATA_BLOCK_BYTES - 1) /
                        DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
                    while (tileBufferBytes64 + inputTileBufferBytes64 + offsetBufferBytes64 >
                               usableUbBytes &&
                           tileWidth > 1)
                    {
                        --tileWidth;
                        tileBufferBytes64 =
                            (static_cast<uint64_t>(tileWidth) * pixelBytes + DATA_BLOCK_BYTES - 1) /
                            DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
                        groupCount =
                            std::min<uint32_t>(tileWidth, static_cast<uint32_t>(blockSize));
                        maxGroupPixels =
                            (tileWidth + static_cast<uint32_t>(blockSize) - 1) /
                            static_cast<uint32_t>(blockSize);
                        groupStrideBytes =
                            (static_cast<uint64_t>(maxGroupPixels) * pixelBytes + DATA_BLOCK_BYTES - 1) /
                            DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
                        inputTileBufferBytes64 =
                            static_cast<uint64_t>(groupCount) * groupStrideBytes;
                        gatherTemplateWidth = useCase9Block4D64GroupedGather
                                                  ? tileWidth
                                                  : CalculateGatherTemplateWidth(
                                                        tileWidth, static_cast<uint32_t>(blockSize), pixelBytes,
                                                        gatherOffsetPixelBytes);
                        offsetBufferBytes64 =
                            (static_cast<uint64_t>(gatherTemplateWidth) * gatherOffsetPixelBytes +
                             DATA_BLOCK_BYTES - 1) /
                            DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
                    }
                    if (tileBufferBytes64 + inputTileBufferBytes64 + offsetBufferBytes64 >
                            usableUbBytes ||
                        inputTileBufferBytes64 > MAX_KERNEL_ELEMENTS ||
                        offsetBufferBytes64 > MAX_KERNEL_ELEMENTS ||
                        !HasUsableGatherTemplate(tileWidth, gatherTemplateWidth))
                    {
                        inputTileBufferBytes = 0;
                        offsetBufferBytes = 0;
                        gatherTemplateWidth = 0;
                        maxTileWidth = static_cast<uint32_t>(usableUbBytes / pixelBytes);
                        maxTileWidth =
                            std::min<uint32_t>(maxTileWidth, static_cast<uint32_t>(outWidth));
                        tileWidth = std::min<uint32_t>(maxTileWidth, parallelTileWidth);
                        tileBufferBytes64 =
                            (static_cast<uint64_t>(tileWidth) * pixelBytes + DATA_BLOCK_BYTES - 1) /
                            DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
                        tileBufferBytes = static_cast<uint32_t>(tileBufferBytes64);
                        tilesPerRow = (static_cast<uint32_t>(outWidth) + tileWidth - 1) / tileWidth;
                        totalTiles = static_cast<uint32_t>(
                            static_cast<uint64_t>(outBatch) * outHeight * tilesPerRow);
                        usedCores = std::min<uint32_t>(maxCores, totalTiles);
                    }
                    else
                    {
                        if (BTS_OPT_GROUPED_GENERAL_DOUBLE_BUFFER &&
                            tileBufferBytes64 * 2 + inputTileBufferBytes64 * 2 +
                                    offsetBufferBytes64 <=
                                usableUbBytes)
                        {
                            tileBufferBytes64 *= 2;
                            inputTileBufferBytes64 *= 2;
                            groupedGeneralDoubleBuffer = true;
                        }
                        tileBufferBytes = static_cast<uint32_t>(tileBufferBytes64);
                        inputGroupStrideElements =
                            static_cast<uint32_t>(groupStrideBytes / kernelElementBytes);
                        inputTileBufferBytes = static_cast<uint32_t>(inputTileBufferBytes64);
                        offsetBufferBytes = static_cast<uint32_t>(offsetBufferBytes64);
                        tilesPerRow = (static_cast<uint32_t>(outWidth) + tileWidth - 1) / tileWidth;
                        const uint64_t gatherTiles64 =
                            static_cast<uint64_t>(outBatch) * outHeight * tilesPerRow;
                        if (gatherTiles64 == 0 || gatherTiles64 > MAX_KERNEL_ELEMENTS)
                        {
                            return ge::GRAPH_FAILED;
                        }
                        totalTiles = static_cast<uint32_t>(gatherTiles64);
                        usedCores = std::min<uint32_t>(maxCores, totalTiles);
                        if (useCase9Block4D64GroupedGather)
                        {
                            alignedBlock4D64GroupedGather = true;
                        }
                        if (useCase5Block2DeepPack2Base)
                        {
                            groupedGeneralPack2 = true;
                        }
                    }
                }
                if (BTS_OPT_GROUPED_GENERAL_ROW_TILE &&
                    !groupedGeneralDirectDma && !alignedBlock4D64UbReorder &&
                    inputTileBufferBytes != 0 && blockSize > 1 &&
                    pixelBytes % DATA_BLOCK_BYTES != 0 &&
                    fullRowBytes <= usableUbBytes && fullRowBytes <= MAX_KERNEL_ELEMENTS)
                {
                    const bool useCase5Block2DeepGeneral =
                        dtypeX == ge::DT_FLOAT16 && blockSize == 2 &&
                        pixelBytes > 128U &&
                        static_cast<uint32_t>(outWidth) > 64U &&
                        static_cast<uint32_t>(outWidth) <= 256U &&
                        static_cast<uint32_t>(outHeight) > 64U;
                    if (BTS_OPT_CASE5_BLOCK2_DEEP_DISABLE_ROW_TILE &&
                        useCase5Block2DeepGeneral)
                    {
                        // Keep the tiled grouped-gather path for #5; full-row row-tile
                        // is correct but much slower for this bucket.
                    }
                    else
                    {
                    const uint32_t rowGroupCount =
                        std::min<uint32_t>(static_cast<uint32_t>(outWidth),
                                           static_cast<uint32_t>(blockSize));
                    const uint32_t rowMaxGroupPixels =
                        (static_cast<uint32_t>(outWidth) +
                         static_cast<uint32_t>(blockSize) - 1U) /
                        static_cast<uint32_t>(blockSize);
                    const uint64_t rowGroupStrideBytes =
                        (static_cast<uint64_t>(rowMaxGroupPixels) * pixelBytes +
                         DATA_BLOCK_BYTES - 1U) /
                        DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
                    const uint64_t rowInputBytes =
                        static_cast<uint64_t>(rowGroupCount) * rowGroupStrideBytes;
                    const uint64_t rowOffsetPixelBytes =
                        static_cast<uint64_t>(xShape.GetDim(3)) * sizeof(uint32_t);
                    const uint32_t rowTemplateWidth = CalculateGatherTemplateWidth(
                        static_cast<uint32_t>(outWidth), static_cast<uint32_t>(blockSize),
                        pixelBytes, rowOffsetPixelBytes);
                    const uint64_t rowOffsetBytes =
                        (static_cast<uint64_t>(rowTemplateWidth) * rowOffsetPixelBytes +
                         DATA_BLOCK_BYTES - 1U) /
                        DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
                    const uint64_t rowExtraBytes = rowInputBytes + rowOffsetBytes;
                    const uint64_t rowUsableBytes =
                        (usableUbBytes > rowExtraBytes) ? (usableUbBytes - rowExtraBytes) : 0ULL;
                    const bool useCase5Block2DeepRowTile =
                        useCase5Block2DeepGeneral &&
                        totalRows >=
                            static_cast<uint64_t>(maxCores) *
                                BTS_OPT_CASE5_BLOCK2_DEEP_ROW_TILE_ROWS;
                    const uint32_t rowTileRows =
                        useCase5Block2DeepRowTile
                            ? BTS_OPT_CASE5_BLOCK2_DEEP_ROW_TILE_ROWS
                            : BTS_OPT_GROUPED_GENERAL_ROW_TILE_ROWS;
                    const uint32_t rowMergeLimit =
                        std::min<uint32_t>(rowTileRows, static_cast<uint32_t>(outHeight));
                    const uint32_t rowMergeRows = static_cast<uint32_t>(
                        std::min<uint64_t>(rowMergeLimit, rowUsableBytes / fullRowBytes));
                    if (rowMergeRows != 0U &&
                        HasUsableGatherTemplate(static_cast<uint32_t>(outWidth),
                                                rowTemplateWidth) &&
                        rowInputBytes <= MAX_KERNEL_ELEMENTS &&
                        rowOffsetBytes <= MAX_KERNEL_ELEMENTS)
                    {
                        const uint64_t rowStrideBytes =
                            (fullRowBytes + DATA_BLOCK_BYTES - 1U) /
                            DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
                        const uint64_t rowTileBytes =
                            static_cast<uint64_t>(rowMergeRows) * rowStrideBytes;
                        const uint32_t rowTilesPerBatch =
                            (static_cast<uint32_t>(outHeight) + rowMergeRows - 1U) /
                            rowMergeRows;
                        const uint64_t rowTotalTiles =
                            static_cast<uint64_t>(outBatch) * rowTilesPerBatch;
                        if (rowTileBytes + rowExtraBytes <= usableUbBytes &&
                            rowTileBytes <= MAX_KERNEL_ELEMENTS &&
                            rowTotalTiles != 0 && rowTotalTiles <= MAX_KERNEL_ELEMENTS)
                        {
                            tileWidth = static_cast<uint32_t>(outWidth);
                            rowTileHeight = rowMergeRows;
                            tilesPerRow = rowTilesPerBatch;
                            totalTiles = static_cast<uint32_t>(rowTotalTiles);
                            tileBufferBytes = static_cast<uint32_t>(rowTileBytes);
                            inputGroupStrideElements =
                                static_cast<uint32_t>(rowGroupStrideBytes / dtypeSize);
                            inputTileBufferBytes = static_cast<uint32_t>(rowInputBytes);
                            offsetBufferBytes = static_cast<uint32_t>(rowOffsetBytes);
                            gatherTemplateWidth = rowTemplateWidth;
                            groupedGeneralDoubleBuffer = false;
                            groupedGeneralRowTile = true;
                            usedCores = std::min<uint32_t>(maxCores, totalTiles);
                        }
                    }
                    }
                }
                const bool rowCompactKnownGood =
                    smallOddAlignedBlock2Tile && outWidth > 16 && outWidth <= 64;
                if (rowCompactKnownGood &&
                    fullRowBytes <= usableUbBytes && fullRowBytes <= MAX_KERNEL_ELEMENTS)
                {
                    const uint64_t rowCompactTiles = totalRows;
                    if (rowCompactTiles != 0 && rowCompactTiles <= MAX_KERNEL_ELEMENTS)
                    {
                        alignedBlock2RowCompact = true;
                        tileWidth = static_cast<uint32_t>(outWidth);
                        tilesPerRow = 1;
                        totalTiles = static_cast<uint32_t>(rowCompactTiles);
                        tileBufferBytes = static_cast<uint32_t>(
                            (fullRowBytes + DATA_BLOCK_BYTES - 1) /
                            DATA_BLOCK_BYTES * DATA_BLOCK_BYTES);
                        inputGroupStrideElements = 0;
                        inputTileBufferBytes = 0;
                        offsetBufferBytes = 0;
                        gatherTemplateWidth = 0;
                        rowTileHeight = 1U;
                        groupedGeneralDoubleBuffer = false;
                        usedCores = std::min<uint32_t>(maxCores, totalTiles);
                    }
                }
                const bool tinyBlock2Group2Candidate =
                    !alignedBlock2RowCompact && smallOddAlignedBlock2Tile &&
                    outWidth <= 16 && tileWidth <= 4U;
                if (tinyBlock2Group2Candidate)
                {
                    if (BTS_DIAG_TINY_BLOCK2_HIT_SINGLE_CORE_ELEMENTWISE)
                    {
                        channelCopy = 0;
                        linearCopy = 0;
                        flatRowMerge = 0;
                        tileWidth = 1U;
                        tilesPerRow = 1U;
                        totalTiles = 1U;
                        tileBufferBytes = 0;
                        inputGroupStrideElements = 0;
                        inputTileBufferBytes = 0;
                        offsetBufferBytes = 0;
                        gatherTemplateWidth = 0;
                        rowTileHeight = 0;
                        groupedGeneralDoubleBuffer = false;
                        alignedBlock2RowCompact = false;
                        alignedBlock2TinyUbDoubleBuffer = false;
                        usedCores = 1U;
                    }
                    else
                    {
                        uint32_t groupedTileWidth = tileWidth << 1;
                        if (groupedTileWidth > static_cast<uint32_t>(outWidth))
                        {
                            groupedTileWidth = static_cast<uint32_t>(outWidth);
                        }
                        uint64_t groupedTileBufferBytes64 =
                            (static_cast<uint64_t>(groupedTileWidth) * pixelBytes +
                             DATA_BLOCK_BYTES - 1) /
                            DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
                        if (groupedTileBufferBytes64 * 2U <= usableUbBytes &&
                            groupedTileBufferBytes64 * 2U <= MAX_KERNEL_ELEMENTS)
                        {
                            groupedTileBufferBytes64 *= 2U;
                            tileWidth = groupedTileWidth;
                            tilesPerRow =
                                (static_cast<uint32_t>(outWidth) + tileWidth - 1) / tileWidth;
                            const uint64_t groupedTotalTiles =
                                static_cast<uint64_t>(outBatch) * outHeight * tilesPerRow;
                            if (groupedTotalTiles != 0 && groupedTotalTiles <= MAX_KERNEL_ELEMENTS)
                            {
                                totalTiles = static_cast<uint32_t>(groupedTotalTiles);
                                tileBufferBytes = static_cast<uint32_t>(groupedTileBufferBytes64);
                                inputGroupStrideElements = 0;
                                inputTileBufferBytes = 0;
                                offsetBufferBytes = 0;
                                gatherTemplateWidth = 0;
                                rowTileHeight = 0;
                                groupedGeneralDoubleBuffer = false;
                                usedCores = std::min<uint32_t>(maxCores, totalTiles);
                                alignedBlock2TinyUbDoubleBuffer = true;
                            }
                        }
                    }
                }
                if (BTS_OPT_ALIGNED_BLOCK2_SMALL_WIDTH_ROW_COMPACT &&
                    !alignedBlock2RowCompact && !alignedBlock2TinyUbDoubleBuffer &&
                    blockSize == 2 && pixelBytes % DATA_BLOCK_BYTES == 0 &&
                    outWidth <= 16 && tileWidth > 8U && tileWidth <= 12U &&
                    fullRowBytes <= usableUbBytes && fullRowBytes <= MAX_KERNEL_ELEMENTS)
                {
                    const uint64_t rowCompactTiles = totalRows;
                    if (rowCompactTiles != 0 && rowCompactTiles <= MAX_KERNEL_ELEMENTS)
                    {
                        alignedBlock2RowCompact = true;
                        tileWidth = static_cast<uint32_t>(outWidth);
                        tilesPerRow = 1U;
                        totalTiles = static_cast<uint32_t>(rowCompactTiles);
                        tileBufferBytes = static_cast<uint32_t>(
                            (fullRowBytes + DATA_BLOCK_BYTES - 1) /
                            DATA_BLOCK_BYTES * DATA_BLOCK_BYTES);
                        inputGroupStrideElements = 0;
                        inputTileBufferBytes = 0;
                        offsetBufferBytes = 0;
                        gatherTemplateWidth = 0;
                        rowTileHeight = 1U;
                        groupedGeneralDoubleBuffer = false;
                        usedCores = std::min<uint32_t>(maxCores, totalTiles);
                    }
                }
                if (BTS_OPT_ALIGNED_BLOCK2_SMALL_WIDTH_NO_DB &&
                    !alignedBlock2RowCompact && !alignedBlock2TinyUbDoubleBuffer &&
                    blockSize == 2 && pixelBytes % DATA_BLOCK_BYTES == 0 &&
                    outWidth <= 16 && tileWidth > 8U && tileWidth <= 12U)
                {
                    const uint64_t singleTileBufferBytes =
                        (static_cast<uint64_t>(tileWidth) * pixelBytes +
                         DATA_BLOCK_BYTES - 1) /
                        DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
                    if (singleTileBufferBytes != 0 && singleTileBufferBytes <= MAX_KERNEL_ELEMENTS)
                    {
                        tileBufferBytes = static_cast<uint32_t>(singleTileBufferBytes);
                        inputGroupStrideElements = 0;
                        inputTileBufferBytes = 0;
                        offsetBufferBytes = 0;
                        gatherTemplateWidth = 0;
                        rowTileHeight = 0;
                        groupedGeneralDoubleBuffer = false;
                    }
                }
                if (BTS_OPT_ALIGNED_BLOCK2_SMALL_WIDTH_DIRECT_LOOP &&
                    !alignedBlock2RowCompact && !alignedBlock2TinyUbDoubleBuffer &&
                    blockSize == 2 && pixelBytes % DATA_BLOCK_BYTES == 0 &&
                    outWidth <= 16 && tileWidth > 8U && tileWidth <= 12U)
                {
                    inputGroupStrideElements = 0;
                    inputTileBufferBytes = 0;
                    offsetBufferBytes = 0;
                    gatherTemplateWidth = 0;
                    rowTileHeight = 0;
                    groupedGeneralDoubleBuffer = false;
                    alignedBlock2SmallDirect = true;
                }
                if (BTS_OPT_ALIGNED_BLOCK2_SMALL_WIDTH_CAP_TILE8 &&
                    !alignedBlock2RowCompact && !alignedBlock2TinyUbDoubleBuffer &&
                    blockSize == 2 && pixelBytes % DATA_BLOCK_BYTES == 0 &&
                    outWidth <= 16 && tileWidth > 8U && tileWidth <= 12U)
                {
                    tileWidth = 8U;
                    tilesPerRow = (static_cast<uint32_t>(outWidth) + tileWidth - 1) / tileWidth;
                    const uint64_t cappedTotalTiles =
                        static_cast<uint64_t>(outBatch) * outHeight * tilesPerRow;
                    if (cappedTotalTiles != 0 && cappedTotalTiles <= MAX_KERNEL_ELEMENTS)
                    {
                        totalTiles = static_cast<uint32_t>(cappedTotalTiles);
                        uint64_t cappedTileBufferBytes =
                            (static_cast<uint64_t>(tileWidth) * pixelBytes +
                             DATA_BLOCK_BYTES - 1) /
                            DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
                        if (cappedTileBufferBytes * 2U <= usableUbBytes &&
                            cappedTileBufferBytes * 2U <= MAX_KERNEL_ELEMENTS)
                        {
                            cappedTileBufferBytes *= 2U;
                        }
                        if (cappedTileBufferBytes <= MAX_KERNEL_ELEMENTS)
                        {
                            tileBufferBytes = static_cast<uint32_t>(cappedTileBufferBytes);
                            inputGroupStrideElements = 0;
                            inputTileBufferBytes = 0;
                            offsetBufferBytes = 0;
                            gatherTemplateWidth = 0;
                            rowTileHeight = 0;
                            groupedGeneralDoubleBuffer = false;
                            usedCores = std::min<uint32_t>(maxCores, totalTiles);
                        }
                    }
                }
                if (BTS_OPT_ALIGNED_BLOCK2_SMALL_WIDTH_VIA_UB_DB &&
                    !alignedBlock2RowCompact && !alignedBlock2TinyUbDoubleBuffer &&
                    blockSize == 2 && pixelBytes % DATA_BLOCK_BYTES == 0 &&
                    outWidth <= 16 && tileWidth > 8U && tileWidth <= 12U)
                {
                    inputGroupStrideElements = 0;
                    inputTileBufferBytes = 0;
                    offsetBufferBytes = 0;
                    gatherTemplateWidth = 0;
                    rowTileHeight = 0;
                    groupedGeneralDoubleBuffer = false;
                    alignedBlock2TinyUbDoubleBuffer = true;
                }
                if (BTS_OPT_ALIGNED_BLOCK2_FULL_ROW12_MULTIROW_DIRECT &&
                    !alignedBlock2RowCompact && !alignedBlock2TinyUbDoubleBuffer &&
                    blockSize == 2 && pixelBytes % DATA_BLOCK_BYTES == 0 &&
                    outWidth == 12 && outHeight > 1 && tileWidth == 12U &&
                    fullRowBytes <= usableUbBytes)
                {
                    const uint32_t maxRowsByUb = static_cast<uint32_t>(
                        std::min<uint64_t>(usableUbBytes / fullRowBytes,
                                           MAX_DMA_BLOCK_COUNT / 6U));
                    const uint32_t targetMergeRows = std::min<uint32_t>(
                        BTS_OPT_ALIGNED_BLOCK2_FULL_ROW12_MERGE_ROWS,
                        static_cast<uint32_t>(outHeight));
                    const uint32_t mergedRows = std::min<uint32_t>(maxRowsByUb, targetMergeRows);
                    if (mergedRows > 1U)
                    {
                        const uint32_t mergedTilesPerRow =
                            (static_cast<uint32_t>(outHeight) + mergedRows - 1) / mergedRows;
                        const uint64_t mergedTiles =
                            static_cast<uint64_t>(outBatch) * mergedTilesPerRow;
                        const uint64_t mergedTileBytes =
                            static_cast<uint64_t>(mergedRows) * fullRowBytes;
                        if (mergedTiles != 0 && mergedTiles <= MAX_KERNEL_ELEMENTS &&
                            mergedTileBytes <= usableUbBytes &&
                            mergedTileBytes <= MAX_KERNEL_ELEMENTS)
                        {
                            uint64_t mergedTileBufferBytes =
                                (mergedTileBytes + DATA_BLOCK_BYTES - 1) /
                                DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
                            if (mergedTileBufferBytes * 2U <= usableUbBytes &&
                                mergedTileBufferBytes * 2U <= MAX_KERNEL_ELEMENTS)
                            {
                                mergedTileBufferBytes *= 2U;
                            }
                            rowTileHeight = mergedRows;
                            tilesPerRow = mergedTilesPerRow;
                            totalTiles = static_cast<uint32_t>(mergedTiles);
                            tileBufferBytes = static_cast<uint32_t>(mergedTileBufferBytes);
                            inputGroupStrideElements = 0;
                            inputTileBufferBytes = 0;
                            offsetBufferBytes = 0;
                            gatherTemplateWidth = 0;
                            groupedGeneralDoubleBuffer = false;
                            alignedBlock2FullRow12MultirowDirect = true;
                            usedCores = std::min<uint32_t>(maxCores, totalTiles);
                        }
                    }
                }
                if (BTS_OPT_ALIGNED_BLOCK2_WIDE_ROW_COMPACT &&
                    !alignedBlock2RowCompact && !alignedBlock2TinyUbDoubleBuffer &&
                    !alignedBlock2FullRow12MultirowDirect &&
                    blockSize == 2 && pixelBytes % DATA_BLOCK_BYTES == 0 &&
                    outWidth > 64 && tileWidth > 320U && tileWidth <= 384U &&
                    fullRowBytes <= usableUbBytes && fullRowBytes <= MAX_KERNEL_ELEMENTS)
                {
                    const uint64_t rowCompactTiles = totalRows;
                    if (rowCompactTiles != 0 && rowCompactTiles <= MAX_KERNEL_ELEMENTS)
                    {
                        alignedBlock2RowCompact = true;
                        tileWidth = static_cast<uint32_t>(outWidth);
                        tilesPerRow = 1U;
                        totalTiles = static_cast<uint32_t>(rowCompactTiles);
                        tileBufferBytes = static_cast<uint32_t>(
                            (fullRowBytes + DATA_BLOCK_BYTES - 1) /
                            DATA_BLOCK_BYTES * DATA_BLOCK_BYTES);
                        inputGroupStrideElements = 0;
                        inputTileBufferBytes = 0;
                        offsetBufferBytes = 0;
                        gatherTemplateWidth = 0;
                        rowTileHeight = 1U;
                        groupedGeneralDoubleBuffer = false;
                        alignedBlock2TinyUbDoubleBuffer = false;
                        alignedBlock2SmallDirect = false;
                        alignedBlock2FullRow12MultirowDirect = false;
                        usedCores = std::min<uint32_t>(maxCores, totalTiles);
                    }
                }
                if (BTS_OPT_ALIGNED_BLOCK2_WIDE_TILE_CAP256 &&
                    !alignedBlock2RowCompact && !alignedBlock2TinyUbDoubleBuffer &&
                    !alignedBlock2FullRow12MultirowDirect &&
                    blockSize == 2 && pixelBytes % DATA_BLOCK_BYTES == 0 &&
                    outWidth > 64 && tileWidth > 320U && tileWidth <= 384U)
                {
                    const uint32_t cappedTileWidth = 256U;
                    const uint32_t cappedTilesPerRow =
                        (static_cast<uint32_t>(outWidth) + cappedTileWidth - 1U) /
                        cappedTileWidth;
                    const uint64_t cappedTotalTiles =
                        static_cast<uint64_t>(outBatch) * outHeight * cappedTilesPerRow;
                    uint64_t cappedTileBufferBytes =
                        (static_cast<uint64_t>(cappedTileWidth) * pixelBytes +
                         DATA_BLOCK_BYTES - 1) /
                        DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
                    if (cappedTotalTiles != 0 && cappedTotalTiles <= MAX_KERNEL_ELEMENTS &&
                        cappedTileBufferBytes <= usableUbBytes &&
                        cappedTileBufferBytes <= MAX_KERNEL_ELEMENTS)
                    {
                        if (cappedTileBufferBytes * 2U <= usableUbBytes &&
                            cappedTileBufferBytes * 2U <= MAX_KERNEL_ELEMENTS)
                        {
                            cappedTileBufferBytes *= 2U;
                        }
                        tileWidth = cappedTileWidth;
                        tilesPerRow = cappedTilesPerRow;
                        totalTiles = static_cast<uint32_t>(cappedTotalTiles);
                        tileBufferBytes = static_cast<uint32_t>(cappedTileBufferBytes);
                        inputGroupStrideElements = 0;
                        inputTileBufferBytes = 0;
                        offsetBufferBytes = 0;
                        gatherTemplateWidth = 0;
                        rowTileHeight = 0;
                        groupedGeneralDoubleBuffer = false;
                        alignedBlock2RowCompact = false;
                        alignedBlock2TinyUbDoubleBuffer = false;
                        alignedBlock2SmallDirect = false;
                        alignedBlock2FullRow12MultirowDirect = false;
                        usedCores = std::min<uint32_t>(maxCores, totalTiles);
                    }
                }
                if ((BTS_OPT_ALIGNED_BLOCK2_TILEWIDTH1_CAP2 ||
                     BTS_OPT_ALIGNED_BLOCK2_TILEWIDTH1_CAP4) &&
                    !alignedBlock2RowCompact && !alignedBlock2TinyUbDoubleBuffer &&
                    blockSize == 2 && pixelBytes % DATA_BLOCK_BYTES == 0 &&
                    outWidth <= 16 && outWidth > 1 && tileWidth == 1U)
                {
                    const uint32_t cappedTileWidth =
                        BTS_OPT_ALIGNED_BLOCK2_TILEWIDTH1_CAP4 ? 4U : 2U;
                    const uint32_t cappedTilesPerRow =
                        (static_cast<uint32_t>(outWidth) + cappedTileWidth - 1U) / cappedTileWidth;
                    const uint64_t cappedTotalTiles =
                        static_cast<uint64_t>(outBatch) * outHeight * cappedTilesPerRow;
                    uint64_t cappedTileBufferBytes =
                        (static_cast<uint64_t>(cappedTileWidth) * pixelBytes +
                         DATA_BLOCK_BYTES - 1) /
                        DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
                    if (cappedTotalTiles != 0 && cappedTotalTiles <= MAX_KERNEL_ELEMENTS &&
                        cappedTileBufferBytes <= usableUbBytes &&
                        cappedTileBufferBytes <= MAX_KERNEL_ELEMENTS)
                    {
                        if (cappedTileBufferBytes * 2U <= usableUbBytes &&
                            cappedTileBufferBytes * 2U <= MAX_KERNEL_ELEMENTS)
                        {
                            cappedTileBufferBytes *= 2U;
                        }
                        tileWidth = cappedTileWidth;
                        tilesPerRow = cappedTilesPerRow;
                        totalTiles = static_cast<uint32_t>(cappedTotalTiles);
                        tileBufferBytes = static_cast<uint32_t>(cappedTileBufferBytes);
                        inputGroupStrideElements = 0;
                        inputTileBufferBytes = 0;
                        offsetBufferBytes = 0;
                        gatherTemplateWidth = 0;
                        rowTileHeight = 0;
                        groupedGeneralDoubleBuffer = false;
                        alignedBlock2RowCompact = false;
                        alignedBlock2TinyUbDoubleBuffer = false;
                        alignedBlock2SmallDirect = false;
                        usedCores = std::min<uint32_t>(maxCores, totalTiles);
                    }
                }
                if (BTS_OPT_ALIGNED_BLOCK2_TILEWIDTH1_ELEMENTWISE &&
                    !alignedBlock2RowCompact && !alignedBlock2TinyUbDoubleBuffer &&
                    blockSize == 2 && pixelBytes % DATA_BLOCK_BYTES == 0 &&
                    outWidth <= 16 && tileWidth == 1U)
                {
                    const uint64_t cacheLines =
                        (totalOutput * dtypeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
                    channelCopy = 0;
                    linearCopy = 0;
                    flatRowMerge = 0;
                    tileBufferBytes = 0;
                    inputGroupStrideElements = 0;
                    inputTileBufferBytes = 0;
                    offsetBufferBytes = 0;
                    gatherTemplateWidth = 0;
                    rowTileHeight = 0;
                    groupedGeneralDoubleBuffer = false;
                    alignedBlock2RowCompact = false;
                    alignedBlock2TinyUbDoubleBuffer = false;
                    alignedBlock2SmallDirect = false;
                    alignedBlock2FullRow12MultirowDirect = false;
                    usedCores = std::max<uint32_t>(
                        1U, std::min<uint32_t>(maxCores, static_cast<uint32_t>(cacheLines)));
                }
            }
        }

        if (blockSize == 1 && crops[0] == 0 && crops[1] == 0 && crops[2] == 0 && crops[3] == 0)
        {
            channelCopy = 0;
            linearCopy = 1;
            linearInputOffset = 0;
            const uint32_t maxLinearElements =
                static_cast<uint32_t>(std::max<uint64_t>(1, usableUbBytes / dtypeSize));
            const uint32_t parallelElements = static_cast<uint32_t>(
                std::max<uint64_t>(1, (totalOutput + maxCores - 1) / maxCores));
            tileWidth = std::min<uint32_t>(maxLinearElements, parallelElements);
            uint64_t tileBufferBytes64 =
                (static_cast<uint64_t>(tileWidth) * dtypeSize + DATA_BLOCK_BYTES - 1) /
                DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
            if (BTS_OPT_LINEAR_DOUBLE_BUFFER && maxCores > 1U)
            {
                const uint32_t maxLinearDbElements = static_cast<uint32_t>(
                    std::max<uint64_t>(1, usableUbBytes / (2U * dtypeSize)));
                const uint64_t targetTiles =
                    static_cast<uint64_t>(maxCores) * BTS_OPT_LINEAR_DOUBLE_BUFFER_TILES_PER_CORE;
                const uint32_t parallelDbElements = static_cast<uint32_t>(
                    std::max<uint64_t>(1, (totalOutput + targetTiles - 1) / targetTiles));
                const uint32_t dbTileWidth =
                    std::min<uint32_t>(maxLinearDbElements, parallelDbElements);
                const uint64_t dbTileBufferBytes =
                    (static_cast<uint64_t>(dbTileWidth) * dtypeSize + DATA_BLOCK_BYTES - 1) /
                    DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
                const uint32_t dbTotalTiles =
                    static_cast<uint32_t>((totalOutput + dbTileWidth - 1) / dbTileWidth);
                if (dbTotalTiles > maxCores &&
                    dbTileBufferBytes >= BTS_OPT_LINEAR_DOUBLE_BUFFER_MIN_TILE_BYTES &&
                    dbTileBufferBytes * 2U <= usableUbBytes)
                {
                    tileWidth = dbTileWidth;
                    tileBufferBytes64 = dbTileBufferBytes * 2U;
                }
            }
            tilesPerRow = 1;
            totalTiles = static_cast<uint32_t>((totalOutput + tileWidth - 1) / tileWidth);
            tileBufferBytes = static_cast<uint32_t>(tileBufferBytes64);
            usedCores = std::min<uint32_t>(maxCores, totalTiles);
            inputGroupStrideElements = 0;
            inputTileBufferBytes = 0;
            offsetBufferBytes = 0;
            gatherTemplateWidth = 0;
            flatRowMerge = 0;
        }
        uint64_t tilingKey = 0;
        const bool isGeneralPath =
            flatRowMerge == 0 && linearCopy == 0 && channelCopy == 0 &&
            tileBufferBytes != 0 && rowTileHeight == 0 &&
            pixelBytes % DATA_BLOCK_BYTES != 0;
        const bool isRowMergePath =
            flatRowMerge == 0 && linearCopy == 0 && channelCopy == 0 &&
            tileBufferBytes != 0 && rowTileHeight != 0;
        const uint64_t rowMergeRows =
            static_cast<uint64_t>(outBatch) * static_cast<uint64_t>(outHeight);
        const bool isRowMergeWide =
            isRowMergePath && blockSize > 1 && outWidth > 40;
        const bool isRowMergeNarrowManyRows =
            isRowMergePath && blockSize > 1 && outWidth <= 40 && rowMergeRows > 256;
        const bool isRowMergeOtherInterleaved =
            isRowMergePath && blockSize > 1 && !isRowMergeWide && !isRowMergeNarrowManyRows;
        bool scalarGeneralRowTile = groupedGeneralRowTile;
        if (isGeneralPath && inputTileBufferBytes == 0 && blockSize == 2 &&
            !groupedGeneralDirectDma)
        {
            const uint64_t scalarRowBytes = static_cast<uint64_t>(outWidth) * pixelBytes;
            const uint32_t scalarGroupCount =
                std::min<uint32_t>(static_cast<uint32_t>(outWidth), static_cast<uint32_t>(blockSize));
            const uint32_t scalarMaxGroupPixels =
                (static_cast<uint32_t>(outWidth) + static_cast<uint32_t>(blockSize) - 1) /
                static_cast<uint32_t>(blockSize);
            const uint64_t scalarGroupStrideBytes =
                (static_cast<uint64_t>(scalarMaxGroupPixels) * pixelBytes + DATA_BLOCK_BYTES - 1) /
                DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
            const uint64_t scalarInputBytes =
                static_cast<uint64_t>(scalarGroupCount) * scalarGroupStrideBytes;
            const uint64_t scalarOffsetPixelBytes =
                static_cast<uint64_t>(xShape.GetDim(3)) * sizeof(uint32_t);
            const uint32_t scalarTemplateWidth = CalculateGatherTemplateWidth(
                static_cast<uint32_t>(outWidth), static_cast<uint32_t>(blockSize),
                pixelBytes, scalarOffsetPixelBytes);
            const uint64_t scalarOffsetBytes =
                (static_cast<uint64_t>(scalarTemplateWidth) * scalarOffsetPixelBytes +
                 DATA_BLOCK_BYTES - 1) /
                DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
            const bool scalarUseGather =
                HasUsableGatherTemplate(static_cast<uint32_t>(outWidth), scalarTemplateWidth) &&
                scalarInputBytes + scalarOffsetBytes < usableUbBytes &&
                scalarInputBytes <= MAX_KERNEL_ELEMENTS &&
                scalarOffsetBytes <= MAX_KERNEL_ELEMENTS;
            const uint64_t scalarExtraBytes =
                scalarUseGather ? (scalarInputBytes + scalarOffsetBytes) : 0;
            const uint64_t scalarRowStrideBytes =
                (scalarRowBytes + DATA_BLOCK_BYTES - 1) /
                DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
            const uint64_t scalarUsableRowBytes =
                (usableUbBytes > scalarExtraBytes) ? (usableUbBytes - scalarExtraBytes) : usableUbBytes;
            const uint32_t maxRowsPerScalarTile =
                static_cast<uint32_t>(std::max<uint64_t>(1, scalarUsableRowBytes / scalarRowStrideBytes));
            const uint32_t targetRows = static_cast<uint32_t>(
                std::max<uint64_t>(1, (rowMergeRows + maxCores - 1) / maxCores));
            const uint32_t scalarRowTileHeight =
                std::min<uint32_t>(maxRowsPerScalarTile, targetRows);
            const uint64_t scalarTileBytes =
                static_cast<uint64_t>(scalarRowTileHeight) * scalarRowStrideBytes;
            if (scalarTileBytes <= usableUbBytes && scalarTileBytes <= MAX_KERNEL_ELEMENTS)
            {
                tileWidth = static_cast<uint32_t>(outWidth);
                rowTileHeight = scalarRowTileHeight;
                tilesPerRow = (static_cast<uint32_t>(outHeight) + rowTileHeight - 1) /
                              rowTileHeight;
                const uint64_t scalarTotalTiles =
                    static_cast<uint64_t>(outBatch) * static_cast<uint64_t>(tilesPerRow);
                if (scalarTotalTiles != 0 && scalarTotalTiles <= MAX_KERNEL_ELEMENTS)
                {
                    totalTiles = static_cast<uint32_t>(scalarTotalTiles);
                    tileBufferBytes = static_cast<uint32_t>(scalarTileBytes);
                    usedCores = std::min<uint32_t>(maxCores, totalTiles);
                    if (scalarUseGather &&
                        scalarTileBytes + scalarInputBytes + scalarOffsetBytes <= usableUbBytes)
                    {
                        inputGroupStrideElements =
                            static_cast<uint32_t>(scalarGroupStrideBytes / dtypeSize);
                        inputTileBufferBytes = static_cast<uint32_t>(scalarInputBytes);
                        offsetBufferBytes = static_cast<uint32_t>(scalarOffsetBytes);
                        gatherTemplateWidth = scalarTemplateWidth;
                    }
                    scalarGeneralRowTile = true;
                }
            }
        }
        const bool diagCase5BaseCandidate =
            flatRowMerge == 0 && linearCopy == 0 && channelCopy == 0 &&
            tileBufferBytes != 0 && blockSize > 1 &&
            pixelBytes % DATA_BLOCK_BYTES != 0 &&
            (groupedGeneralRowTile || groupedGeneralDoubleBuffer ||
             groupedGeneralDirectDma || inputTileBufferBytes != 0);
        bool diagCase5Hit = false;
        if (BTS_DIAG_CASE5_BUCKET_PROBE != 0 && diagCase5BaseCandidate)
        {
            const uint32_t cropTopPhase =
                static_cast<uint32_t>(crops[0]) % static_cast<uint32_t>(blockSize);
            const uint32_t cropLeftPhase =
                static_cast<uint32_t>(crops[2]) % static_cast<uint32_t>(blockSize);
            switch (BTS_DIAG_CASE5_BUCKET_PROBE)
            {
            case 1U:
                diagCase5Hit = true;
                break;
            case 2U:
                diagCase5Hit = dtypeX == ge::DT_FLOAT16;
                break;
            case 3U:
                diagCase5Hit = dtypeX == ge::DT_FLOAT;
                break;
            case 4U:
                diagCase5Hit = blockSize == 2;
                break;
            case 5U:
                diagCase5Hit = blockSize == 3;
                break;
            case 6U:
                diagCase5Hit = blockSize == 4;
                break;
            case 7U:
                diagCase5Hit = pixelBytes <= 16U;
                break;
            case 8U:
                diagCase5Hit = pixelBytes > 16U && pixelBytes <= 64U;
                break;
            case 9U:
                diagCase5Hit = pixelBytes > 64U && pixelBytes <= 128U;
                break;
            case 10U:
                diagCase5Hit = pixelBytes > 128U;
                break;
            case 11U:
                diagCase5Hit = outWidth <= 64;
                break;
            case 12U:
                diagCase5Hit = outWidth > 64 && outWidth <= 256;
                break;
            case 13U:
                diagCase5Hit = outWidth > 256 && outWidth <= 1024;
                break;
            case 14U:
                diagCase5Hit = outWidth > 1024;
                break;
            case 15U:
                diagCase5Hit = outHeight <= 16;
                break;
            case 16U:
                diagCase5Hit = outHeight > 16 && outHeight <= 64;
                break;
            case 17U:
                diagCase5Hit = outHeight > 64;
                break;
            case 18U:
                diagCase5Hit = cropLeftPhase == 0U;
                break;
            case 19U:
                diagCase5Hit = cropLeftPhase == 1U;
                break;
            case 20U:
                diagCase5Hit = cropTopPhase == 0U;
                break;
            case 21U:
                diagCase5Hit = rowMergeRows <= maxCores;
                break;
            case 22U:
                diagCase5Hit = rowMergeRows > maxCores;
                break;
            case 23U:
                diagCase5Hit = groupedGeneralRowTile || scalarGeneralRowTile;
                break;
            case 24U:
                diagCase5Hit = groupedGeneralDoubleBuffer && rowTileHeight == 0;
                break;
            case 25U:
                diagCase5Hit = inputTileBufferBytes != 0;
                break;
            default:
                diagCase5Hit = false;
                break;
            }
        }
        if (diagCase5Hit)
        {
            tileWidth = 1U;
            tilesPerRow = 1U;
            totalTiles = 1U;
            tileBufferBytes = 0;
            inputGroupStrideElements = 0;
            inputTileBufferBytes = 0;
            offsetBufferBytes = 0;
            gatherTemplateWidth = 0;
            rowTileHeight = 0;
            groupedGeneralDoubleBuffer = false;
            alignedBlock2RowCompact = false;
            alignedBlock2TinyUbDoubleBuffer = false;
            alignedBlock2SmallDirect = false;
            alignedBlock2FullRow12MultirowDirect = false;
            alignedBlock4D64DirectOutput = false;
            alignedBlock4D64PhaseFast = false;
            alignedBlock4D64GroupedGather = false;
            alignedBlock4D64UbReorder = false;
            groupedGeneralDirectDma = false;
            groupedGeneralRowTile = false;
            scalarGeneralRowTile = false;
            usedCores = 1U;
        }
        if (BTS_DIAG_ROW_MERGE_SINGLE_CORE_ELEMENTWISE &&
            flatRowMerge == 0 && linearCopy == 0 && channelCopy == 0 &&
            tileBufferBytes != 0 && rowTileHeight != 0 && !scalarGeneralRowTile)
        {
            tileWidth = 1U;
            tilesPerRow = 1U;
            totalTiles = 1U;
            tileBufferBytes = 0;
            inputGroupStrideElements = 0;
            inputTileBufferBytes = 0;
            offsetBufferBytes = 0;
            gatherTemplateWidth = 0;
            rowTileHeight = 0;
            groupedGeneralDoubleBuffer = false;
            alignedBlock2RowCompact = false;
            alignedBlock2TinyUbDoubleBuffer = false;
            alignedBlock2SmallDirect = false;
            alignedBlock2FullRow12MultirowDirect = false;
            usedCores = 1U;
        }
        if (BTS_DIAG_ALIGNED_NON_ROW_SINGLE_CORE_ELEMENTWISE &&
            flatRowMerge == 0 && linearCopy == 0 && channelCopy == 0 &&
            tileBufferBytes != 0 && rowTileHeight == 0 &&
            blockSize > 1 && pixelBytes % DATA_BLOCK_BYTES == 0)
        {
            tileWidth = 1U;
            tilesPerRow = 1U;
            totalTiles = 1U;
            tileBufferBytes = 0;
            inputGroupStrideElements = 0;
            inputTileBufferBytes = 0;
            offsetBufferBytes = 0;
            gatherTemplateWidth = 0;
            rowTileHeight = 0;
            groupedGeneralDoubleBuffer = false;
            alignedBlock2RowCompact = false;
            alignedBlock2TinyUbDoubleBuffer = false;
            alignedBlock2SmallDirect = false;
            alignedBlock2FullRow12MultirowDirect = false;
            usedCores = 1U;
        }
        const bool diagCase9BaseCandidate =
            flatRowMerge == 0 && linearCopy == 0 && channelCopy == 0 &&
            tileBufferBytes != 0 && rowTileHeight == 0 &&
            blockSize == 4 && dtypeX == ge::DT_FLOAT16 &&
            pixelBytes == 128U && outWidth > 512;
        bool diagCase9Hit = false;
        if (BTS_DIAG_CASE9_PROBE != 0 && diagCase9BaseCandidate)
        {
            switch (BTS_DIAG_CASE9_PROBE)
            {
            case 1U:
                diagCase9Hit = (static_cast<uint32_t>(crops[2]) & 3U) == 0U;
                break;
            case 2U:
                diagCase9Hit = (static_cast<uint32_t>(crops[2]) & 3U) == 1U;
                break;
            case 3U:
                diagCase9Hit = (static_cast<uint32_t>(crops[2]) & 3U) == 2U;
                break;
            case 4U:
                diagCase9Hit = (static_cast<uint32_t>(crops[2]) & 3U) == 3U;
                break;
            case 5U:
                diagCase9Hit = (static_cast<uint32_t>(crops[0]) & 3U) == 0U;
                break;
            case 6U:
                diagCase9Hit = (static_cast<uint32_t>(crops[0]) & 3U) == 1U;
                break;
            case 7U:
                diagCase9Hit = (static_cast<uint32_t>(crops[0]) & 3U) == 2U;
                break;
            case 8U:
                diagCase9Hit = (static_cast<uint32_t>(crops[0]) & 3U) == 3U;
                break;
            case 9U:
                diagCase9Hit = outWidth <= 768;
                break;
            case 10U:
                diagCase9Hit = outWidth <= 1024;
                break;
            case 11U:
                diagCase9Hit = outWidth <= 1536;
                break;
            case 12U:
                diagCase9Hit = outWidth <= 2048;
                break;
            case 13U:
                diagCase9Hit = outHeight <= 16;
                break;
            case 14U:
                diagCase9Hit = outHeight <= 64;
                break;
            case 15U:
                diagCase9Hit = outHeight <= 256;
                break;
            case 16U:
                diagCase9Hit = outBatch == 1;
                break;
            case 17U:
                diagCase9Hit =
                    static_cast<uint64_t>(outBatch) * outHeight <= maxCores;
                break;
            case 18U:
                diagCase9Hit = tilesPerRow == 3U;
                break;
            case 19U:
                diagCase9Hit = tilesPerRow == 4U;
                break;
            case 20U:
                diagCase9Hit = tilesPerRow >= 5U;
                break;
            case 21U:
                diagCase9Hit = crops[2] == 0;
                break;
            case 22U:
                diagCase9Hit = crops[0] == 0;
                break;
            case 23U:
                diagCase9Hit = outWidth <= 1280;
                break;
            default:
                diagCase9Hit = false;
                break;
            }
        }
        if (diagCase9Hit)
        {
            tileWidth = 1U;
            tilesPerRow = 1U;
            totalTiles = 1U;
            tileBufferBytes = 0;
            inputGroupStrideElements = 0;
            inputTileBufferBytes = 0;
            offsetBufferBytes = 0;
            gatherTemplateWidth = 0;
            rowTileHeight = 0;
            groupedGeneralDoubleBuffer = false;
            alignedBlock2RowCompact = false;
            alignedBlock2TinyUbDoubleBuffer = false;
            alignedBlock2SmallDirect = false;
            alignedBlock2FullRow12MultirowDirect = false;
            usedCores = 1U;
        }
        const bool diagAlignedBlock2DirectBaseCandidate =
            flatRowMerge == 0 && linearCopy == 0 && channelCopy == 0 &&
            tileBufferBytes != 0 && rowTileHeight == 0 &&
            !alignedBlock2RowCompact && !alignedBlock2TinyUbDoubleBuffer &&
            blockSize == 2 && pixelBytes % DATA_BLOCK_BYTES == 0;
        const bool diagAlignedBlock2DirectCandidate =
            diagAlignedBlock2DirectBaseCandidate &&
            (BTS_DIAG_ALIGNED_BLOCK2_DIRECT_SINGLE_CORE_ELEMENTWISE ||
             (BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_LE16_SINGLE_CORE_ELEMENTWISE &&
              outWidth <= 16) ||
             (BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_LE16_TILEWIDTH_LE8_SINGLE_CORE_ELEMENTWISE &&
              outWidth <= 16 && tileWidth <= 8U) ||
             (BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_LE16_TILEWIDTH_GT8_LE12_SINGLE_CORE_ELEMENTWISE &&
              outWidth <= 16 && tileWidth > 8U && tileWidth <= 12U) ||
             (BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_LE16_TILEWIDTH_EQ8_SINGLE_CORE_ELEMENTWISE &&
              outWidth <= 16 && tileWidth == 8U) ||
             (BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_LE16_TILEWIDTH_EQ6_SINGLE_CORE_ELEMENTWISE &&
              outWidth <= 16 && tileWidth == 6U) ||
             (BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_LE16_TILEWIDTH_EQ4_SINGLE_CORE_ELEMENTWISE &&
              outWidth <= 16 && tileWidth == 4U) ||
             (BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_LE16_TILEWIDTH_EQ2_SINGLE_CORE_ELEMENTWISE &&
              outWidth <= 16 && tileWidth == 2U) ||
             (BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_LE16_TILEWIDTH_EQ1_SINGLE_CORE_ELEMENTWISE &&
              outWidth <= 16 && tileWidth == 1U) ||
             (BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_GT16_LE64_SINGLE_CORE_ELEMENTWISE &&
              outWidth > 16 && outWidth <= 64) ||
             (BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_GT64_SINGLE_CORE_ELEMENTWISE &&
              outWidth > 64) ||
             (BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_GT64_TILEWIDTH_LE64_SINGLE_CORE_ELEMENTWISE &&
              outWidth > 64 && tileWidth <= 64U) ||
             (BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_GT64_TILEWIDTH_GT64_SINGLE_CORE_ELEMENTWISE &&
              outWidth > 64 && tileWidth > 64U) ||
             (BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_GT64_TILEWIDTH_GT64_LE128_SINGLE_CORE_ELEMENTWISE &&
              outWidth > 64 && tileWidth > 64U && tileWidth <= 128U) ||
             (BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_GT64_TILEWIDTH_GT128_LE256_SINGLE_CORE_ELEMENTWISE &&
              outWidth > 64 && tileWidth > 128U && tileWidth <= 256U) ||
             (BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_GT64_TILEWIDTH_GT256_LE512_SINGLE_CORE_ELEMENTWISE &&
              outWidth > 64 && tileWidth > 256U && tileWidth <= 512U) ||
             (BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_GT64_TILEWIDTH_GT256_LE384_SINGLE_CORE_ELEMENTWISE &&
              outWidth > 64 && tileWidth > 256U && tileWidth <= 384U) ||
             (BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_GT64_TILEWIDTH_GT256_LE320_SINGLE_CORE_ELEMENTWISE &&
              outWidth > 64 && tileWidth > 256U && tileWidth <= 320U) ||
             (BTS_DIAG_ALIGNED_BLOCK2_DIRECT_OUTWIDTH_GT64_TILEWIDTH_GT320_LE384_SINGLE_CORE_ELEMENTWISE &&
              outWidth > 64 && tileWidth > 320U && tileWidth <= 384U) ||
             (BTS_DIAG_ALIGNED_BLOCK2_SMALL_WIDTH_TILEWIDTH_10_SINGLE_CORE_ELEMENTWISE &&
              outWidth <= 16 && tileWidth == 10U) ||
             (BTS_DIAG_ALIGNED_BLOCK2_SMALL_WIDTH_TILEWIDTH_12_SINGLE_CORE_ELEMENTWISE &&
              outWidth <= 16 && tileWidth == 12U) ||
             (BTS_DIAG_ALIGNED_BLOCK2_SMALL_WIDTH_TILEWIDTH_12_FULL_ROW_SINGLE_CORE_ELEMENTWISE &&
              outWidth == tileWidth && tileWidth == 12U));
        if (diagAlignedBlock2DirectCandidate)
        {
            tileWidth = 1U;
            tilesPerRow = 1U;
            totalTiles = 1U;
            tileBufferBytes = 0;
            inputGroupStrideElements = 0;
            inputTileBufferBytes = 0;
            offsetBufferBytes = 0;
            gatherTemplateWidth = 0;
            rowTileHeight = 0;
            groupedGeneralDoubleBuffer = false;
            usedCores = 1U;
        }
        if (dtypeX == ge::DT_FLOAT16)
        {
            if (flatRowMerge != 0)
            {
                tilingKey = 3UL;
            }
            else if (linearCopy != 0)
            {
                tilingKey = 1UL;
            }
            else if (channelCopy != 0)
            {
                tilingKey = 5UL;
            }
            else if (tileBufferBytes == 0)
            {
                tilingKey = 7UL;
            }
            else if (alignedBlock2RowCompact)
            {
                tilingKey = 29UL;
            }
            else if (alignedBlock2TinyUbDoubleBuffer)
            {
                tilingKey = 31UL;
            }
            else if (alignedBlock2SmallDirect)
            {
                tilingKey = 33UL;
            }
            else if (alignedBlock2FullRow12MultirowDirect)
            {
                tilingKey = 35UL;
            }
            else if (alignedBlock4D64DirectOutput)
            {
                tilingKey = 37UL;
            }
            else if (alignedBlock4D64PhaseFast)
            {
                tilingKey = 39UL;
            }
            else if (alignedBlock4D64UbReorder)
            {
                tilingKey = 41UL;
            }
            else if (groupedGeneralPack2)
            {
                tilingKey = 42UL;
            }
            else if (groupedGeneralDirectDma)
            {
                tilingKey = 21UL;
            }
            else if (alignedBlock4D64GroupedGather)
            {
                tilingKey = 23UL;
            }
            else if (rowTileHeight != 0 && !scalarGeneralRowTile)
            {
                if (isRowMergeWide)
                {
                    tilingKey = 15UL;
                }
                else if (isRowMergeNarrowManyRows)
                {
                    tilingKey = 9UL;
                }
                else if (isRowMergeOtherInterleaved)
                {
                    tilingKey = 19UL;
                }
                else
                {
                    tilingKey = 9UL;
                }
            }
            else if (pixelBytes % DATA_BLOCK_BYTES == 0)
            {
                if (blockSize == 2)
                {
                    tilingKey = 27UL;
                }
                else
                {
                    tilingKey = (blockSize > 1) ? 25UL : 11UL;
                }
            }
            else
            {
                tilingKey = groupedGeneralDoubleBuffer ? 23UL : 13UL;
            }
        }
        else
        {
            if (flatRowMerge != 0)
            {
                tilingKey = 4UL;
            }
            else if (linearCopy != 0)
            {
                tilingKey = 2UL;
            }
            else if (channelCopy != 0)
            {
                tilingKey = 6UL;
            }
            else if (tileBufferBytes == 0)
            {
                tilingKey = 8UL;
            }
            else if (alignedBlock2RowCompact)
            {
                tilingKey = 30UL;
            }
            else if (alignedBlock2TinyUbDoubleBuffer)
            {
                tilingKey = 32UL;
            }
            else if (alignedBlock2SmallDirect)
            {
                tilingKey = 34UL;
            }
            else if (alignedBlock2FullRow12MultirowDirect)
            {
                tilingKey = 36UL;
            }
            else if (alignedBlock4D64DirectOutput)
            {
                tilingKey = 38UL;
            }
            else if (alignedBlock4D64PhaseFast)
            {
                tilingKey = 40UL;
            }
            else if (groupedGeneralDirectDma)
            {
                tilingKey = 22UL;
            }
            else if (alignedBlock4D64GroupedGather)
            {
                tilingKey = 24UL;
            }
            else if (rowTileHeight != 0 && !scalarGeneralRowTile)
            {
                if (isRowMergeWide)
                {
                    tilingKey = 16UL;
                }
                else if (isRowMergeNarrowManyRows)
                {
                    tilingKey = 10UL;
                }
                else if (isRowMergeOtherInterleaved)
                {
                    tilingKey = 20UL;
                }
                else
                {
                    tilingKey = 10UL;
                }
            }
            else if (pixelBytes % DATA_BLOCK_BYTES == 0)
            {
                if (blockSize == 2)
                {
                    tilingKey = 28UL;
                }
                else
                {
                    tilingKey = (blockSize > 1) ? 26UL : 12UL;
                }
            }
            else
            {
                tilingKey = groupedGeneralDoubleBuffer ? 24UL : 14UL;
            }
        }
        context->SetTilingKey(tilingKey);

        auto *tiling = context->GetTilingData<BatchToSpaceTilingData>();
        if (tiling == nullptr)
        {
            return ge::GRAPH_FAILED;
        }
        tiling->batch = static_cast<uint32_t>(xShape.GetDim(0));
        tiling->height = static_cast<uint32_t>(xShape.GetDim(1));
        tiling->width = static_cast<uint32_t>(xShape.GetDim(2));
        tiling->depth = groupedGeneralPack2
                            ? static_cast<uint32_t>(xShape.GetDim(3) / 2)
                            : static_cast<uint32_t>(xShape.GetDim(3));
        tiling->outBatch = static_cast<uint32_t>(outBatch);
        tiling->outHeight = static_cast<uint32_t>(outHeight);
        tiling->outWidth = static_cast<uint32_t>(outWidth);
        tiling->cropTop = static_cast<uint32_t>(crops[0]);
        tiling->cropLeft = static_cast<uint32_t>(crops[2]);
        tiling->blockSize = static_cast<uint32_t>(blockSize);
        tiling->totalOutputElements = groupedGeneralPack2
                                          ? static_cast<uint32_t>(totalOutput / 2)
                                          : static_cast<uint32_t>(totalOutput);
        tiling->tileWidth = tileWidth;
        tiling->tilesPerRow = tilesPerRow;
        tiling->totalTiles = totalTiles;
        tiling->tileBufferBytes = tileBufferBytes;
        tiling->inputGroupStrideElements = inputGroupStrideElements;
        tiling->inputTileBufferBytes = inputTileBufferBytes;
        tiling->offsetBufferBytes = offsetBufferBytes;
        tiling->gatherTemplateWidth = gatherTemplateWidth;
        tiling->rowTileHeight = rowTileHeight;
        tiling->flatRowMerge = flatRowMerge;
        tiling->linearInputOffset = linearInputOffset;
        tiling->usedCoreNum = usedCores;

        context->SetBlockDim(usedCores);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (currentWorkspace == nullptr)
        {
            return ge::GRAPH_FAILED;
        }
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
} // namespace optiling

namespace ge
{
    static graphStatus InferShape(gert::InferShapeContext *context)
    {
        const gert::Shape *xShape = context->GetInputShape(0);
        gert::Shape *yShape = context->GetOutputShape(0);
        if (xShape == nullptr || yShape == nullptr)
        {
            return GRAPH_FAILED;
        }
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        if (attrs == nullptr)
        {
            return GRAPH_FAILED;
        }
        const auto *cropAttr = attrs->GetListInt(0);
        const int64_t *blockAttr = attrs->GetInt(1);
        if (cropAttr == nullptr || cropAttr->GetSize() != 4 || blockAttr == nullptr)
        {
            return GRAPH_FAILED;
        }
        const int64_t *crops = cropAttr->GetData();
        const int64_t blockSize = *blockAttr;
        if (crops == nullptr || xShape->GetDimNum() != 4 || blockSize <= 0)
        {
            return GRAPH_FAILED;
        }
        const int64_t blockArea = blockSize * blockSize;
        const int64_t outHeight = xShape->GetDim(1) * blockSize - crops[0] - crops[1];
        const int64_t outWidth = xShape->GetDim(2) * blockSize - crops[2] - crops[3];
        if (xShape->GetDim(0) <= 0 || xShape->GetDim(1) <= 0 || xShape->GetDim(2) <= 0 ||
            xShape->GetDim(3) <= 0 || blockArea <= 0 || xShape->GetDim(0) % blockArea != 0 ||
            crops[0] < 0 || crops[1] < 0 || crops[2] < 0 || crops[3] < 0 ||
            outHeight <= 0 || outWidth <= 0)
        {
            return GRAPH_FAILED;
        }
        *yShape = *xShape;
        yShape->SetDim(0, xShape->GetDim(0) / blockArea);
        yShape->SetDim(1, outHeight);
        yShape->SetDim(2, outWidth);
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
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("crops").AttrType(REQUIRED).ListInt();
            this->Attr("block_size").AttrType(REQUIRED).Int();
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(BatchToSpace);
} // namespace ops
