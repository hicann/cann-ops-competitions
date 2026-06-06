#include "../op_kernel/batch_to_space_tiling.h"
#include "../op_kernel/tiling_key_batch_to_space.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

constexpr uint32_t DATA_BLOCK_BYTES = 32U;
constexpr uint32_t MAX_DMA_BLOCK_COUNT = 4095U;
constexpr uint32_t MAX_DMA_BLOCK_LEN = 2097151U;  // DataCopyExtParams::blockLen
constexpr uint64_t UB_RESERVE_BYTES = 16ULL * 1024ULL;
constexpr uint64_t INTERLEAVE_MIN_SPATIAL = 64ULL;

// V8 double-buffer gating (evaluated in the host where totalJobs and the launched
// core count are both known). Double buffering only pays off when each core owns
// enough jobs to amortize the pipeline prologue AND moves enough bytes that the
// extra MTE2/MTE3 events overlap real traffic; otherwise the event overhead is a
// net regression on tiny shapes. The kernel still verifies a second UB tile can
// actually be claimed before turning the pipe on.
constexpr uint64_t DB_MIN_JOBS_PER_CORE = 4ULL;       // >= 4 jobs/core to pipeline
constexpr uint64_t DB_MIN_BYTES_PER_CORE = 8192ULL;   // >= 8KB/core of real traffic

// V10 flag-merge (batched MTE2/MTE3 events) gating. With one tile per sync the
// fixed SetFlag/WaitFlag cost dominates throughput once tiles get small. When a
// tile is small enough that several fit in UB we load a whole GROUP of tiles,
// raise ONE MTE2->MTE3 flag for the group, store the group, then ONE MTE3->MTE2
// flag -- cutting the TPipe event traffic from 4 per tile to 4 per group. This
// is applied to the contiguous no-crop paths and is mutually exclusive with the
// double-buffer pipe (a merged group is itself a coarse-grained software stage).
constexpr uint64_t FLAG_MERGE_TILE_MAX_BYTES = 4096ULL;  // "small tile" cutoff
constexpr uint64_t FLAG_MERGE_MIN_GROUP = 4ULL;          // only merge >= 4 tiles
constexpr uint64_t FLAG_MERGE_MAX_GROUP = 16ULL;         // cap UB use + group loop

inline uint32_t CeilDivU32(const uint32_t x, const uint32_t y)
{
    return (x + y - 1U) / y;
}

inline uint64_t AlignUp32(const uint64_t bytes)
{
    return (bytes + DATA_BLOCK_BYTES - 1ULL) / DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
}

}  // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const auto &xShape = context->GetInputShape(0)->GetStorageShape();
    const int64_t inputBatch64 = xShape.GetDim(0);
    const int64_t inputHeight64 = xShape.GetDim(1);
    const int64_t inputWidth64 = xShape.GetDim(2);
    const int64_t channels64 = xShape.GetDim(3);

    const auto *attrs = context->GetAttrs();
    const int64_t *crops = attrs->GetListInt(0)->GetData();
    const int64_t blockSize64 = *attrs->GetInt(1);
    const int64_t blockArea64 = blockSize64 * blockSize64;
    if (blockArea64 == 0 || inputBatch64 % blockArea64 != 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t outputHeight64 = inputHeight64 * blockSize64 - crops[0] - crops[1];
    const int64_t outputWidth64 = inputWidth64 * blockSize64 - crops[2] - crops[3];
    if (outputHeight64 <= 0 || outputWidth64 <= 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t inputBatch = static_cast<uint32_t>(inputBatch64);
    const uint32_t inputHeight = static_cast<uint32_t>(inputHeight64);
    const uint32_t inputWidth = static_cast<uint32_t>(inputWidth64);
    const uint32_t channels = static_cast<uint32_t>(channels64);
    const uint32_t blockSize = static_cast<uint32_t>(blockSize64);
    const uint32_t outputBatch = static_cast<uint32_t>(inputBatch64 / blockArea64);
    const uint32_t outputHeight = static_cast<uint32_t>(outputHeight64);
    const uint32_t outputWidth = static_cast<uint32_t>(outputWidth64);

    uint32_t elementBytes = 0U;
    uint64_t tilingKey = 0U;
    const ge::DataType dtype = context->GetInputDesc(0)->GetDataType();
    if (dtype == ge::DT_FLOAT16) {
        elementBytes = 2U;
        tilingKey = BATCH_TO_SPACE_TILING_KEY_FP16;
    } else if (dtype == ge::DT_FLOAT) {
        elementBytes = 4U;
        tilingKey = BATCH_TO_SPACE_TILING_KEY_FP32;
    } else {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivCoreNum = platform.GetCoreNumAiv();
    if (aivCoreNum == 0U) {
        aivCoreNum = 1U;
    }

    // V9: core partitioning refined per best practices 10_0014 / 10_0015.
    //
    // The same three core buckets (8 / 16 / 40) are kept, but the cascade is
    // sharpened in two ways relative to the V5 absolute-threshold version:
    //
    //   1. Channel-bound is RELATIVE, not an absolute "C > 400". A copy is only
    //      MTE2-bandwidth bound when channels dominate the spatial footprint
    //      (channels >= spatial * CHANNEL_DOMINATE_RATIO). In that regime a
    //      handful of cores already saturate the read bus, so we cap at 8 and
    //      adding cores only buys launch/contention overhead. A large C paired
    //      with a *large* spatial extent is NOT channel-bound -- it still has
    //      plenty of independent spatial jobs, so it falls through to 40 cores
    //      (V5 would have wrongly throttled it to 8).
    //
    //   2. Small-spatial prefers WHOLE-ROW tiles (preferWholeRows). Splitting a
    //      short row into sub-tiles just multiplies tiny MTE setups + per-job
    //      scalar overhead; instead each core owns several full rows, keeping the
    //      copies contiguous and avoiding the per-tile fan-out cost.
    //
    // NOTE: targetCores is computed *before* tiling so the per-row tile split
    // (desiredTilesPerRow) fans out against the number of cores we actually
    // launch, not the physical AIV count.
    constexpr uint32_t CORES_C_SATURATED = 8U;     // channel-bound (MTE2 limited)
    constexpr uint32_t CORES_SMALL_SPATIAL = 16U;  // H*W < 1024
    constexpr uint32_t CORES_LARGE = 40U;          // large spatial: max parallel

    constexpr uint32_t CHANNEL_THRESHOLD = 400U;       // absolute floor (from V5)
    constexpr uint64_t CHANNEL_DOMINATE_RATIO = 4ULL;  // C >= 4x spatial => bound
    constexpr uint64_t SPATIAL_THRESHOLD = 1024ULL;

    const uint64_t spatialSize =
        static_cast<uint64_t>(inputHeight) * inputWidth;

    // Channel-bound only when C is both absolutely large AND large relative to
    // the spatial footprint; otherwise the spatial jobs give us parallelism to
    // exploit and capping at 8 cores would leave the array idle.
    const bool channelBound =
        channels > CHANNEL_THRESHOLD &&
        static_cast<uint64_t>(channels) >= spatialSize * CHANNEL_DOMINATE_RATIO;

    uint32_t targetCores;
    if (channelBound) {
        targetCores = CORES_C_SATURATED;
    } else if (spatialSize < SPATIAL_THRESHOLD) {
        targetCores = CORES_SMALL_SPATIAL;
    } else {
        targetCores = CORES_LARGE;
    }
    // Small-spatial: keep rows whole and give each core multiple rows instead of
    // fragmenting rows across cores.
    const bool preferWholeRows =
        !channelBound && spatialSize < SPATIAL_THRESHOLD;
    // Tile splitting should never fan out wider than the cores we will launch
    // or wider than what the hardware physically has.
    const uint32_t tileCoreBudget = std::min<uint32_t>(targetCores, aivCoreNum);

    uint64_t ubBytes = 0ULL;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    uint64_t usableUb = (ubBytes > UB_RESERVE_BYTES) ? (ubBytes - UB_RESERVE_BYTES) : ubBytes;
    usableUb = usableUb / DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
    if (usableUb < DATA_BLOCK_BYTES) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t pixelBytes = static_cast<uint64_t>(channels) * elementBytes;
    const uint64_t paddedPixelBytes = AlignUp32(pixelBytes);
    const uint64_t outputGapBytes = static_cast<uint64_t>(blockSize - 1U) * pixelBytes;
    const uint64_t spatialPixels = static_cast<uint64_t>(inputWidth) * inputHeight;
    const bool noCrop = (crops[0] == 0 && crops[1] == 0 && crops[2] == 0 && crops[3] == 0);

    uint32_t mode = BATCH_TO_SPACE_MODE_ROW_SCATTER;
    uint32_t tilePixels = 0U;
    uint32_t tilesPerRow = 0U;
    uint32_t channelTileElems = 0U;
    uint32_t bufferBytes = 0U;
    uint64_t totalJobs = 0ULL;

    const bool rowScatterSupported =
        pixelBytes >= 1ULL &&
        pixelBytes <= static_cast<uint64_t>(MAX_DMA_BLOCK_LEN) &&
        paddedPixelBytes <= usableUb &&
        outputGapBytes <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());

    // V7: enable block_interleave (continuous write) as the preferred main path.
    //
    // Following best practices 10_0017 / 10_0013, whenever the pixel is exactly
    // 32B-aligned (pixelBytes % 32 == 0) the blockSize input rows are gathered
    // into strided UB slots and the whole interleaved output row is flushed with
    // ONE contiguous MTE3 write. This removes the strided row_scatter MTE3 write
    // that wastes ~(block-1)/block of the write bandwidth (each pixel forces a
    // fresh burst with a dstGap stride).
    //
    // V12: row_scatter is now a *pure fallback* -- it is selected ONLY when the
    // continuous-write (block_interleave) condition cannot be met. Crop is no
    // longer routed to row_scatter based on a "small crop" heuristic: the crop
    // interleave kernel already emits ONE contiguous write per tile while the
    // per-phase strided MTE2 loads skip the cropped columns (block_interleave +
    // padding), so it keeps the contiguous-write bandwidth win for ANY crop size.
    // The only remaining gate is therefore the structural one shared with the
    // no-crop path: a 32B-aligned pixel and enough spatial work to amortize the
    // per-tile setup. Anything that fails that (non-32B C, tiny spatial extent)
    // still falls back to row_scatter; an oversized C that does not fit UB falls
    // back to the channel path (which moves channelTileElems per copy, below).
    const bool blockInterleaveEligible =
        rowScatterSupported &&
        pixelBytes >= static_cast<uint64_t>(DATA_BLOCK_BYTES) &&
        (pixelBytes % static_cast<uint64_t>(DATA_BLOCK_BYTES)) == 0ULL &&
        spatialPixels >= INTERLEAVE_MIN_SPATIAL;

    if (blockInterleaveEligible) {
        // UB holds one interleaved output tile of blockSize*tilePixels pixels.
        const uint64_t perPixUb = static_cast<uint64_t>(blockSize) * pixelBytes;
        uint64_t maxTile64 = usableUb / perPixUb;
        maxTile64 = std::min<uint64_t>(maxTile64, static_cast<uint64_t>(MAX_DMA_BLOCK_LEN) / perPixUb);
        maxTile64 = std::min<uint64_t>(maxTile64, MAX_DMA_BLOCK_COUNT);
        maxTile64 = std::min<uint64_t>(maxTile64, inputWidth);
        if (maxTile64 >= 1ULL) {
            // V4: split each row enough to expose >= tileCoreBudget jobs so the
            // case fans out across exactly the cores we launch (ceil(cores/rows),
            // capped by width). Using tileCoreBudget (not the physical AIV count)
            // keeps tiles from being over-fragmented when we start fewer cores.
            const uint64_t rowCount =
                static_cast<uint64_t>(outputBatch) * inputHeight * blockSize;
            // V9: small-spatial keeps whole rows (one tile per row) and lets each
            // core own several rows; otherwise split rows to expose tileCoreBudget.
            const uint32_t desiredTilesPerRow =
                (!preferWholeRows && rowCount > 0ULL && rowCount < tileCoreBudget)
                    ? static_cast<uint32_t>((tileCoreBudget + rowCount - 1ULL) / rowCount)
                    : 1U;
            const uint32_t parallelTilePixels =
                CeilDivU32(inputWidth, std::max<uint32_t>(desiredTilesPerRow, 1U));
            tilePixels = std::min<uint32_t>(static_cast<uint32_t>(maxTile64),
                                            std::max<uint32_t>(parallelTilePixels, 1U));
            tilesPerRow = CeilDivU32(inputWidth, tilePixels);
            bufferBytes = static_cast<uint32_t>(AlignUp32(
                static_cast<uint64_t>(blockSize) * tilePixels * pixelBytes +
                static_cast<uint64_t>(blockSize) * DATA_BLOCK_BYTES));
            totalJobs = rowCount * tilesPerRow;
            mode = noCrop ? BATCH_TO_SPACE_MODE_BLOCK_INTERLEAVE_NO_CROP
                          : BATCH_TO_SPACE_MODE_BLOCK_INTERLEAVE_CROP;
        }
    }

    if (mode == BATCH_TO_SPACE_MODE_ROW_SCATTER && rowScatterSupported) {
        uint64_t maxTile64 = usableUb / paddedPixelBytes;
        maxTile64 = std::min<uint64_t>(maxTile64, inputWidth);
        maxTile64 = std::min<uint64_t>(maxTile64, MAX_DMA_BLOCK_COUNT);
        const uint32_t maxTilePixels = std::max<uint32_t>(static_cast<uint32_t>(maxTile64), 1U);

        const uint64_t rowCount = static_cast<uint64_t>(inputBatch) * inputHeight;
        // V9: small-spatial keeps whole rows (one tile per row) and lets each
        // core own several rows; otherwise split rows to expose tileCoreBudget.
        const uint32_t desiredTilesPerRow =
            (!preferWholeRows && rowCount > 0ULL && rowCount < tileCoreBudget)
                ? static_cast<uint32_t>((tileCoreBudget + rowCount - 1ULL) / rowCount)
                : 1U;
        const uint32_t parallelTilePixels =
            CeilDivU32(inputWidth, std::max<uint32_t>(desiredTilesPerRow, 1U));

        tilePixels = std::min<uint32_t>(maxTilePixels, std::max<uint32_t>(parallelTilePixels, 1U));
        tilesPerRow = CeilDivU32(inputWidth, tilePixels);
        bufferBytes = static_cast<uint32_t>(static_cast<uint64_t>(tilePixels) * paddedPixelBytes);
        totalJobs = rowCount * tilesPerRow;
    } else if (!rowScatterSupported) {
        // Extremely large C or unsupported stride: one output pixel by channel chunks.
        //
        // V8 (point 2): size the channel chunk to move as much of the pixel as
        // possible in ONE DataCopyPad so the per-job scalar chunk loop runs the
        // fewest times. The chunk is the largest 32B-aligned span that fits both
        // UB and the single-transfer blockLen limit, but never larger than the
        // pixel itself (clamping to channels avoids reserving UB we cannot use and
        // keeps the loop at a single iteration whenever the whole pixel fits).
        const uint64_t pixelCap =
            std::min<uint64_t>(usableUb, static_cast<uint64_t>(MAX_DMA_BLOCK_LEN));
        uint64_t chunkBytes = std::min<uint64_t>(pixelCap, AlignUp32(pixelBytes));
        chunkBytes = (chunkBytes / DATA_BLOCK_BYTES) * DATA_BLOCK_BYTES;
        mode = BATCH_TO_SPACE_MODE_CHANNEL_FALLBACK;
        bufferBytes = static_cast<uint32_t>(chunkBytes);
        channelTileElems = bufferBytes / elementBytes;
        if (channelTileElems == 0U) {
            return ge::GRAPH_FAILED;
        }
        totalJobs = static_cast<uint64_t>(outputBatch) * outputHeight * outputWidth;
    }

    if (totalJobs == 0ULL || bufferBytes == 0U) {
        return ge::GRAPH_FAILED;
    }

    // V004 LOCK: launch exactly the bucket's core count (tileCoreBudget =
    // min(targetCores, aivCoreNum)) UNCONDITIONALLY. Unlike V10 there is no
    // `min(.., totalJobs)` clamp: whatever bucket a shape lands in (8/16/40) is
    // locked in even if it produces fewer jobs than cores; the spare cores get
    // jobCount_ == 0 and simply idle.
    uint32_t blockDim = tileCoreBudget;
    if (blockDim == 0U) {
        return ge::GRAPH_FAILED;
    }

    // V8 (point 1): decide double buffering here, now that the launched core count
    // (blockDim) is fixed. Only the contiguous-write no-crop paths are pipelined
    // in the kernel; for those, gate on the *minimum* per-core job count
    // (totalJobs / blockDim, the floor every core is guaranteed to reach) and the
    // bytes that minimum moves. If either is below threshold the kernel keeps the
    // single-buffer path and never allocates the extra MTE2/MTE3 events.
    const bool pipeModeEligible =
        (mode == BATCH_TO_SPACE_MODE_BLOCK_INTERLEAVE_NO_CROP) ||
        (mode == BATCH_TO_SPACE_MODE_ROW_SCATTER && noCrop);
    const uint64_t minJobsPerCore = totalJobs / static_cast<uint64_t>(blockDim);
    const uint64_t singleTileBytes = static_cast<uint64_t>(bufferBytes);
    const uint64_t minBytesPerCore = minJobsPerCore * singleTileBytes;

    // V10: prefer flag-merge for the small-tile case (many cheap tiles per core),
    // where collapsing per-tile events into per-group events wins more than the
    // double-buffer pipe. Merging widens the UB buffer to hold the whole group;
    // it is mutually exclusive with double buffering.
    uint32_t flagMergeGroup = 1U;
    if (pipeModeEligible && singleTileBytes > 0ULL &&
        singleTileBytes < FLAG_MERGE_TILE_MAX_BYTES) {
        uint64_t group = usableUb / singleTileBytes;          // tiles that fit UB
        group = std::min<uint64_t>(group, minJobsPerCore);    // never exceed jobs/core
        group = std::min<uint64_t>(group, FLAG_MERGE_MAX_GROUP);
        if (group >= FLAG_MERGE_MIN_GROUP) {
            flagMergeGroup = static_cast<uint32_t>(group);
            bufferBytes = static_cast<uint32_t>(singleTileBytes * group);
        }
    }

    // Double buffering only when we did NOT merge (merging is its own staging).
    const uint32_t enableDoubleBuffer =
        (flagMergeGroup == 1U && pipeModeEligible &&
         minJobsPerCore >= DB_MIN_JOBS_PER_CORE &&
         minBytesPerCore >= DB_MIN_BYTES_PER_CORE)
            ? 1U
            : 0U;

    BatchToSpaceTilingData tiling;
    tiling.set_totalJobs(totalJobs);
    tiling.set_mode(mode);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_channels(channels);
    tiling.set_outputBatch(outputBatch);
    tiling.set_outputHeight(outputHeight);
    tiling.set_outputWidth(outputWidth);
    tiling.set_blockSize(blockSize);
    tiling.set_cropTop(static_cast<int32_t>(crops[0]));
    tiling.set_cropBottom(static_cast<int32_t>(crops[1]));
    tiling.set_cropLeft(static_cast<int32_t>(crops[2]));
    tiling.set_cropRight(static_cast<int32_t>(crops[3]));
    tiling.set_tilePixels(tilePixels);
    tiling.set_tilesPerRow(tilesPerRow);
    tiling.set_channelTileElems(channelTileElems);
    tiling.set_bufferBytes(bufferBytes);
    tiling.set_enableDoubleBuffer(enableDoubleBuffer);
    tiling.set_flagMergeGroup(flagMergeGroup);

    context->SetBlockDim(blockDim);
    context->SetTilingKey(tilingKey);
    auto *raw = context->GetRawTilingData();
    tiling.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(tiling.GetDataSize());

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0U;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr ||
        context->GetOutputShape(0) == nullptr || context->GetAttrs() == nullptr) {
        return GRAPH_FAILED;
    }

    const gert::Shape *xShape = context->GetInputShape(0);
    if (xShape->GetDimNum() != 4U) {
        return GRAPH_FAILED;
    }

    const auto *cropsAttr = context->GetAttrs()->GetListInt(0);
    const auto *blockSizeAttr = context->GetAttrs()->GetInt(1);
    if (cropsAttr == nullptr || cropsAttr->GetSize() != 4U || blockSizeAttr == nullptr) {
        return GRAPH_FAILED;
    }

    const int64_t *crops = cropsAttr->GetData();
    const int64_t blockSize = *blockSizeAttr;
    if (blockSize <= 0 || crops[0] < 0 || crops[1] < 0 || crops[2] < 0 || crops[3] < 0 ||
        xShape->GetDim(0) % (blockSize * blockSize) != 0) {
        return GRAPH_FAILED;
    }

    const int64_t outputHeight = xShape->GetDim(1) * blockSize - crops[0] - crops[1];
    const int64_t outputWidth = xShape->GetDim(2) * blockSize - crops[2] - crops[3];
    if (outputHeight <= 0 || outputWidth <= 0) {
        return GRAPH_FAILED;
    }

    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    yShape->SetDim(0, xShape->GetDim(0) / (blockSize * blockSize));
    yShape->SetDim(1, outputHeight);
    yShape->SetDim(2, outputWidth);
    yShape->SetDim(3, xShape->GetDim(3));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class BatchToSpace : public OpDef {
public:
    explicit BatchToSpace(const char *name) : OpDef(name)
    {
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

        this->Attr("crops").AttrType(REQUIRED).ListInt({});
        this->Attr("block_size").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910_93")
            .AddConfig("ascend910b");
    }
};

OP_ADD(BatchToSpace);

}  // namespace ops
