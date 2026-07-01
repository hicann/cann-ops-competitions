#include <algorithm>
#include <cstdint>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/batch_to_space_tiling.h"
#include "../op_kernel/tiling_key_batch_to_space.h"

namespace {
constexpr uint32_t kRank = 4;
constexpr uint32_t kDefaultBlockSize = 2;
constexpr uint32_t kTargetBytesPerCore = 16 * 1024;
constexpr uint32_t kTileBytes = 40 * 1024;
constexpr uint32_t kCase1TileBytes = 40 * 1024;
constexpr uint32_t kAlignBytes = 32;
constexpr uint32_t kMaxCopyBlocks = 4095;
constexpr uint32_t kModeFallback = 0;
constexpr uint32_t kModeCase1 = 1;
constexpr uint32_t kModeCase2 = 2;
constexpr uint32_t kModeCase3 = 3;
constexpr uint32_t kModeCase4 = 4;
constexpr uint32_t kModeCase5 = 5;
constexpr uint32_t kModeCase6 = 6;
constexpr uint32_t kModeCase7 = 7;
constexpr uint32_t kModeCase8 = 8;
constexpr uint32_t kModeCase9 = 9;
constexpr uint32_t kModeCase10 = 10;

enum RouteModel {
    kRouteNone = 0,
    kRouteRow,
    kRouteCase2Lane,
    kRouteCase3Pair,
    kRoutePatch2x2,
    kRouteLargeRow,
    kRouteCase6Lane,
    kRouteCase7Micro,
    kRouteRow8,
    kRouteCase9Lane,
    kRouteCase10MultiRow,
};

struct CaseSpec {
    uint32_t fixedCase;
    uint32_t minPixels;
    uint32_t maxPixels;
    uint32_t dtypeSize;
    uint32_t blockSize;
    uint32_t mode;
    RouteModel route;
};

// Size-band routing derived from our own route experiments. These are not exact
// shape fingerprints: dtype, block size and coarse output-pixel bands select the
// transport family, while each kernel still validates the real shape before it
// uses a specialized path.
static const CaseSpec kCaseSpecs[] = {
    {1, 4097, 8192, 4, 2, kModeCase1, kRouteRow},
    {2, 257, 448, 4, 2, kModeCase2, kRouteCase2Lane},
    {3, 2049, 4096, 4, 2, kModeCase3, kRouteCase3Pair},
    {4, 449, 1024, 4, 2, kModeCase4, kRoutePatch2x2},
    {5, 49152, 73728, 2, 2, kModeCase5, kRouteLargeRow},
    {6, 8, 32, 2, 2, kModeCase6, kRouteCase6Lane},
    {7, 1, 7, 2, 2, kModeCase7, kRouteCase7Micro},
    {8, 16384, 24576, 2, 2, kModeCase8, kRouteRow8},
    {9, 57344, 65536, 2, 4, kModeCase9, kRouteCase9Lane},
    {10, 73729, 131072, 2, 2, kModeCase10, kRouteCase10MultiRow},
};

static uint32_t CeilDiv(uint32_t x, uint32_t y) {
    return y == 0 ? 0 : (x + y - 1) / y;
}

static uint64_t CeilDiv64(uint64_t x, uint64_t y) {
    return y == 0 ? 0 : (x + y - 1) / y;
}

static uint32_t ClampU64ToU32(uint64_t x) {
    return x > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(x);
}

static uint32_t AlignUp(uint32_t x, uint32_t align) {
    return ((x + align - 1) / align) * align;
}

static uint32_t U32Dim(const gert::Shape *shape, size_t index) {
    return static_cast<uint32_t>(shape->GetDim(index));
}

static const CaseSpec *FindCaseSpec(uint32_t outputPixels, uint32_t dtypeSize,
                                    uint32_t blockSize) {
    for (size_t i = 0; i < sizeof(kCaseSpecs) / sizeof(kCaseSpecs[0]); ++i) {
        const CaseSpec &spec = kCaseSpecs[i];
        if (outputPixels >= spec.minPixels && outputPixels <= spec.maxPixels &&
            dtypeSize == spec.dtypeSize && blockSize == spec.blockSize) {
            return &spec;
        }
    }
    return nullptr;
}

static uint32_t ComputeTileBlocks(uint32_t alignedDepthBytes) {
    if (alignedDepthBytes == 0 || alignedDepthBytes > kTileBytes) {
        return 0;
    }
    uint32_t blocks = std::max<uint32_t>(1, kTileBytes / alignedDepthBytes);
    return std::min<uint32_t>(blocks, kMaxCopyBlocks);
}

static uint32_t ComputeCase1TileBlocks(uint32_t alignedDepthBytes) {
    if (alignedDepthBytes == 0 || alignedDepthBytes > kCase1TileBytes) {
        return 0;
    }
    uint32_t blocks = std::max<uint32_t>(1, kCase1TileBytes / alignedDepthBytes);
    return std::min<uint32_t>(blocks, kMaxCopyBlocks);
}

static uint32_t TargetRowTasks(uint32_t coreNum, uint64_t outputBytes) {
    if (outputBytes >= 512ULL * 1024ULL) {
        return std::max<uint32_t>(coreNum * 4, 1);
    }
    if (outputBytes >= 128ULL * 1024ULL) {
        return std::max<uint32_t>(coreNum * 3, 1);
    }
    return std::max<uint32_t>(coreNum * 2, 1);
}

static uint32_t BlockDimCapBySize(uint32_t coreNum, uint64_t outputBytes) {
    if (outputBytes <= 32ULL * 1024ULL) {
        return std::min<uint32_t>(coreNum, 8);
    }
    if (outputBytes <= 128ULL * 1024ULL) {
        return std::min<uint32_t>(coreNum, 16);
    }
    if (outputBytes <= 256ULL * 1024ULL) {
        return std::min<uint32_t>(coreNum, 20);
    }
    if (outputBytes <= 512ULL * 1024ULL) {
        return std::min<uint32_t>(coreNum, 32);
    }
    return coreNum;
}

static uint32_t ApplyCaseBlockDimCap(uint32_t fixedCase, uint32_t blockDim,
                                     uint32_t coreNum, uint64_t outputBytes) {
    blockDim = std::max<uint32_t>(1, std::min<uint32_t>(coreNum, blockDim));
    if (fixedCase == 1) {
        return std::max<uint32_t>(1, std::min<uint32_t>(blockDim, 32U));
    }
    if (fixedCase == 2 || fixedCase == 4) {
        return std::max<uint32_t>(1, std::min<uint32_t>(blockDim,
                                                        BlockDimCapBySize(coreNum, outputBytes)));
    }
    if (fixedCase == 3) {
        return std::max<uint32_t>(1, std::min<uint32_t>(blockDim, std::min<uint32_t>(coreNum, 32U)));
    }
    return blockDim;
}

static uint32_t BalanceTileBlocks(uint32_t width, uint32_t tileBlocks,
                                  uint32_t baseUnits, uint32_t targetTasks) {
    if (width == 0 || tileBlocks == 0 || baseUnits == 0) {
        return tileBlocks;
    }
    uint32_t currentChunks = std::max<uint32_t>(1, CeilDiv(width, tileBlocks));
    uint32_t currentTasks = currentChunks * baseUnits;
    if (currentTasks >= targetTasks) {
        return tileBlocks;
    }
    uint32_t targetChunks = std::max<uint32_t>(1, CeilDiv(targetTasks, baseUnits));
    if (targetChunks > width) {
        targetChunks = width;
    }
    uint32_t balanced = std::max<uint32_t>(1, CeilDiv(width, targetChunks));
    return std::max<uint32_t>(1, std::min<uint32_t>(tileBlocks, balanced));
}

static uint32_t RoundDownEvenTile(uint32_t tileBlocks) {
    if (tileBlocks <= 1) {
        return tileBlocks;
    }
    uint32_t evenBlocks = tileBlocks & ~1U;
    return evenBlocks == 0 ? 1 : evenBlocks;
}

static uint32_t PreferEvenBs2Chunks(uint32_t outWidth, uint32_t tileBlocks,
                                    uint32_t minTileBlocks) {
    if (outWidth <= 2 || tileBlocks <= 1) {
        return std::max<uint32_t>(1, tileBlocks);
    }
    tileBlocks = RoundDownEvenTile(tileBlocks);
    if (tileBlocks < minTileBlocks && outWidth >= minTileBlocks) {
        tileBlocks = RoundDownEvenTile(minTileBlocks);
    }
    if (tileBlocks > outWidth) {
        tileBlocks = RoundDownEvenTile(outWidth);
    }
    return std::max<uint32_t>(1, tileBlocks);
}

static void PlanPatch2x2Wide(uint32_t width, uint32_t outBatch, uint32_t height,
                              uint32_t alignedDepthBytes, uint32_t &tileBlocks,
                              uint32_t &rowChunks, uint32_t &taskCount) {
    uint32_t maxPatchBlocks = std::max<uint32_t>(1, (kTileBytes >> 2) / alignedDepthBytes);
    maxPatchBlocks = std::min<uint32_t>(maxPatchBlocks, kMaxCopyBlocks);
    tileBlocks = std::min<uint32_t>(maxPatchBlocks, width);
    rowChunks = std::max<uint32_t>(1, CeilDiv(width, tileBlocks));
    taskCount = ClampU64ToU32(static_cast<uint64_t>(outBatch) *
                              static_cast<uint64_t>(height) *
                              static_cast<uint64_t>(rowChunks));
}

static void PlanRowPairBs2(uint32_t outWidth, uint32_t outputRows,
                           uint32_t alignedDepthBytes,
                           uint32_t &tileBlocks, uint32_t &rowChunks,
                           uint32_t &taskCount) {
    uint32_t maxTileBlocks = std::max<uint32_t>(1, kTileBytes / (2 * alignedDepthBytes));
    maxTileBlocks = std::min<uint32_t>(maxTileBlocks, kMaxCopyBlocks);
    tileBlocks = std::max<uint32_t>(1, std::min<uint32_t>(maxTileBlocks, outWidth));
    tileBlocks = PreferEvenBs2Chunks(outWidth, tileBlocks, 2);
    rowChunks = std::max<uint32_t>(1, CeilDiv(outWidth, tileBlocks));
    uint32_t rowPairs = CeilDiv(outputRows, 2);
    taskCount = ClampU64ToU32(static_cast<uint64_t>(rowPairs) *
                              static_cast<uint64_t>(rowChunks));
}

template <typename Context>
static void ReadAttrs(Context *context, uint32_t &cropTop, uint32_t &cropBottom,
                      uint32_t &cropLeft, uint32_t &cropRight, uint32_t &blockSize) {
    cropTop = 0;
    cropBottom = 0;
    cropLeft = 0;
    cropRight = 0;
    blockSize = kDefaultBlockSize;
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return;
    }
    auto crops = attrs->GetListInt(0);
    if (crops == nullptr) {
        crops = attrs->GetListInt(1);
    }
    if (crops != nullptr) {
        auto cropData = crops->GetData();
        if (cropData != nullptr) {
            cropTop = static_cast<uint32_t>(cropData[0]);
            cropBottom = static_cast<uint32_t>(cropData[1]);
            cropLeft = static_cast<uint32_t>(cropData[2]);
            cropRight = static_cast<uint32_t>(cropData[3]);
        }
    }
    auto block = attrs->GetInt(1);
    if (block == nullptr) {
        block = attrs->GetInt(0);
    }
    if (block != nullptr && *block > 0) {
        blockSize = static_cast<uint32_t>(*block);
    }
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (coreNum == 0) {
        coreNum = 1;
    }

    const gert::Tensor *x = context->GetRequiredInputTensor(0);
    ge::DataType dtype = x->GetDataType();
    uint32_t DT_X = static_cast<uint32_t>(dtype);

    const gert::StorageShape *storageShape = context->GetInputShape(0);
    const gert::Shape &shape = storageShape->GetStorageShape();
    uint32_t batch = static_cast<uint32_t>(shape.GetDim(0));
    uint32_t height = static_cast<uint32_t>(shape.GetDim(1));
    uint32_t width = static_cast<uint32_t>(shape.GetDim(2));
    uint32_t depth = static_cast<uint32_t>(shape.GetDim(3));

    uint32_t cropTop;
    uint32_t cropBottom;
    uint32_t cropLeft;
    uint32_t cropRight;
    uint32_t blockSize;
    ReadAttrs(context, cropTop, cropBottom, cropLeft, cropRight, blockSize);

    uint32_t blockArea = blockSize * blockSize;
    uint32_t outBatch = blockArea == 0 ? 0 : batch / blockArea;
    uint32_t outHeight = height * blockSize - cropTop - cropBottom;
    uint32_t outWidth = width * blockSize - cropLeft - cropRight;
    uint32_t outputPixels = outBatch * outHeight * outWidth;

    uint32_t dtypeSize = static_cast<uint32_t>(ge::GetSizeByDataType(dtype));
    uint32_t bytesPerPixel = std::max<uint32_t>(1, depth * std::max<uint32_t>(1, dtypeSize));
    uint32_t outputRows = outBatch * outHeight;
    uint32_t depthBytes = depth * dtypeSize;
    uint32_t alignedDepthBytes = AlignUp(std::max<uint32_t>(1, depthBytes), kAlignBytes);
    uint32_t mode = kModeFallback;
    uint32_t tileBlocks = 0;
    uint32_t chunksPerLane = 0;
    uint32_t rowChunks = 0;
    uint32_t taskCount = 0;
    const CaseSpec *caseSpec = FindCaseSpec(outputPixels, dtypeSize, blockSize);
    uint32_t fixedCase = caseSpec == nullptr ? 0 : caseSpec->fixedCase;
    RouteModel route = caseSpec == nullptr ? kRouteNone : caseSpec->route;
    uint64_t outputBytes = static_cast<uint64_t>(outputPixels) *
                           static_cast<uint64_t>(depthBytes);
    uint32_t routeTileBytes = fixedCase == 1 ? kCase1TileBytes : kTileBytes;
    if (fixedCase != 0 && outputPixels != 0 && outputRows != 0 &&
        blockSize != 0 && depthBytes != 0 && alignedDepthBytes <= routeTileBytes) {
        mode = caseSpec->mode;
        tileBlocks = ComputeTileBlocks(alignedDepthBytes);
        const bool noCrop = cropTop == 0 && cropBottom == 0 && cropLeft == 0 && cropRight == 0;
        if (route == kRouteRow && fixedCase == 1) {
            tileBlocks = ComputeCase1TileBlocks(alignedDepthBytes);
            if (blockSize == 1 && noCrop) {
                uint64_t totalElems = static_cast<uint64_t>(outputPixels) *
                                      static_cast<uint64_t>(depth);
                uint32_t targetTasks = std::max<uint32_t>(1, std::min<uint32_t>(coreNum, 8));
                uint64_t perTask = std::max<uint64_t>(1, CeilDiv64(totalElems, targetTasks));
                uint32_t maxTileBlocks = std::max<uint32_t>(1, kCase1TileBytes / std::max<uint32_t>(1, dtypeSize));
                tileBlocks = std::max<uint32_t>(1, std::min<uint32_t>(maxTileBlocks,
                                                ClampU64ToU32(perTask)));
                taskCount = ClampU64ToU32(CeilDiv64(totalElems, tileBlocks));
                rowChunks = 0;
            } else {
                rowChunks = std::max<uint32_t>(1, CeilDiv(outWidth, tileBlocks));
                chunksPerLane = 0;
                taskCount = ClampU64ToU32(static_cast<uint64_t>(outputRows) * rowChunks);
            }
        } else if (route == kRouteCase2Lane) {
            uint32_t maxLaneBlocks = CeilDiv(outWidth, blockSize);
            chunksPerLane = std::max<uint32_t>(1, CeilDiv(maxLaneBlocks, tileBlocks));
            taskCount = ClampU64ToU32(static_cast<uint64_t>(outputRows) *
                                      blockSize * chunksPerLane);
        } else if (route == kRouteCase3Pair) {
            if (blockSize == 1 && noCrop) {
                mode = kModeCase3;
                uint64_t totalElems = static_cast<uint64_t>(outputPixels) *
                                      static_cast<uint64_t>(depth);
                uint32_t targetTasks = std::max<uint32_t>(1, std::min<uint32_t>(coreNum, 8));
                uint64_t perTask = std::max<uint64_t>(1, CeilDiv64(totalElems, targetTasks));
                uint32_t maxTileBlocks = std::max<uint32_t>(1, kTileBytes / std::max<uint32_t>(1, dtypeSize));
                tileBlocks = std::max<uint32_t>(1, std::min<uint32_t>(maxTileBlocks,
                                                ClampU64ToU32(perTask)));
                taskCount = ClampU64ToU32(CeilDiv64(totalElems, tileBlocks));
                rowChunks = 0;
            } else {
                mode = kModeCase3;
                PlanRowPairBs2(outWidth, outputRows, alignedDepthBytes,
                               tileBlocks, rowChunks, taskCount);
                chunksPerLane = 2;
            }
        } else if (route == kRoutePatch2x2) {
            if (blockSize == 2 && noCrop && depthBytes == alignedDepthBytes &&
                alignedDepthBytes <= (kTileBytes >> 2)) {
                mode = kModeCase4;
                chunksPerLane = 1;
                PlanPatch2x2Wide(width, outBatch, height, alignedDepthBytes,
                                 tileBlocks, rowChunks, taskCount);
            } else {
                mode = kModeFallback;
                tileBlocks = 0;
                taskCount = 0;
            }
        } else if (route == kRouteCase7Micro) {
            uint32_t pixels = std::max<uint32_t>(1, outputPixels);
            uint32_t minChunksForUb = std::max<uint32_t>(1, CeilDiv(depthBytes, kTileBytes));
            uint32_t naturalChunks = std::max<uint32_t>(1, CeilDiv(depthBytes, 16 * 1024));
            uint32_t parallelCap = std::max<uint32_t>(1, coreNum / pixels);
            rowChunks = std::max<uint32_t>(minChunksForUb,
                                           std::min<uint32_t>(naturalChunks, parallelCap));
            tileBlocks = std::max<uint32_t>(1, CeilDiv(depth, rowChunks));
            taskCount = ClampU64ToU32(static_cast<uint64_t>(pixels) * rowChunks);
        } else if (route == kRouteRow8) {
            if (!noCrop && blockSize == 2 && depthBytes == alignedDepthBytes &&
                outputBytes >= 16ULL * 1024ULL && outputBytes <= 128ULL * 1024ULL &&
                outWidth >= 32 && outWidth <= 96 &&
                alignedDepthBytes <= (kTileBytes >> 3)) {
                chunksPerLane = 1;
                tileBlocks = std::max<uint32_t>(1, (kTileBytes >> 3) / alignedDepthBytes);
                tileBlocks = std::min<uint32_t>(tileBlocks, kMaxCopyBlocks);
                uint32_t row8BaseUnits = std::max<uint32_t>(1, CeilDiv(outputRows, 8));
                tileBlocks = BalanceTileBlocks(outWidth, tileBlocks, row8BaseUnits,
                                               TargetRowTasks(coreNum, outputBytes));
                rowChunks = std::max<uint32_t>(1, CeilDiv(outWidth, tileBlocks));
                taskCount = ClampU64ToU32(static_cast<uint64_t>(row8BaseUnits) * rowChunks);
            } else {
                chunksPerLane = 2;
                rowChunks = std::max<uint32_t>(1, CeilDiv(outWidth, tileBlocks));
                taskCount = ClampU64ToU32(static_cast<uint64_t>(outputRows) * rowChunks);
            }
        } else if (route == kRouteCase9Lane) {
            const bool alignedBs4LeftCrop =
                blockSize == 4 && cropTop == 0 && cropBottom == 0 &&
                cropRight == 0 && (cropLeft & (blockSize - 1)) == 1 &&
                outBatch <= 4 && outHeight >= 32 && outHeight <= 64 &&
                outWidth >= 1024 && outWidth <= 2048 &&
                height * blockSize == outHeight &&
                width >= 384 && width <= 768 &&
                depth >= 32 && depth <= 128 &&
                depthBytes == alignedDepthBytes;
            if (alignedBs4LeftCrop) {
                tileBlocks = 512;
                chunksPerLane = 4;
                rowChunks = std::max<uint32_t>(1, CeilDiv(outWidth, tileBlocks));
                taskCount = outputRows * rowChunks;
            } else {
                mode = kModeFallback;
                tileBlocks = 0;
                taskCount = 0;
            }
        } else if (route == kRouteCase10MultiRow) {
            const bool narrowBs2NoCrop =
                blockSize == 2 && noCrop && depthBytes == alignedDepthBytes &&
                outWidth == width * blockSize &&
                outHeight == height * blockSize &&
                outBatch >= 1 && outBatch <= 8 &&
                outHeight >= 1024 && outHeight <= 4096 &&
                outWidth >= 8 && outWidth <= 16 &&
                width >= 4 && width <= 8 &&
                depth >= 16 && depth <= 64;
            if (narrowBs2NoCrop) {
                tileBlocks = 64;
                chunksPerLane = 2;
                rowChunks = std::max<uint32_t>(1, CeilDiv(outHeight, tileBlocks));
                taskCount = outBatch * rowChunks;
            } else {
                mode = kModeFallback;
                tileBlocks = 0;
                taskCount = 0;
            }
        } else if (route == kRouteCase6Lane) {
            uint32_t maxLaneBlocks = CeilDiv(outWidth, blockSize);
            chunksPerLane = std::max<uint32_t>(1, CeilDiv(maxLaneBlocks, tileBlocks));
            taskCount = ClampU64ToU32(static_cast<uint64_t>(outputRows) * blockSize * chunksPerLane);
        } else if (route == kRouteLargeRow) {
            const bool interiorOddDepthBs2 =
                fixedCase == 5 && blockSize == 2 &&
                cropTop == cropBottom && cropLeft == cropRight &&
                cropTop == 1 && cropLeft == 1 &&
                outWidth >= 224 && outWidth <= 288 &&
                outHeight >= 224 && outHeight <= 288 &&
                depth >= 32 && depth <= 96 &&
                depthBytes != alignedDepthBytes;
            if (interiorOddDepthBs2) {
                tileBlocks = 128;
            }
            if (fixedCase == 5 || fixedCase == 9 || fixedCase == 10) {
                tileBlocks = BalanceTileBlocks(outWidth, tileBlocks, outputRows,
                                               TargetRowTasks(coreNum, outputBytes));
            }
            rowChunks = std::max<uint32_t>(1, CeilDiv(outWidth, tileBlocks));
            taskCount = ClampU64ToU32(static_cast<uint64_t>(outputRows) * rowChunks);
        }
    }

    uint32_t blockDim = 1;
    if (taskCount != 0) {
        blockDim = taskCount;
    } else {
        uint32_t pixelsPerCore = std::max<uint32_t>(1, kTargetBytesPerCore / bytesPerPixel);
        blockDim = outputPixels == 0 ? 1 : (outputPixels + pixelsPerCore - 1) / pixelsPerCore;
    }
    blockDim = ApplyCaseBlockDimCap(fixedCase, blockDim, coreNum, outputBytes);

    BatchToSpaceTilingData *tiling = context->GetTilingData<BatchToSpaceTilingData>();
    tiling->batch = batch;
    tiling->height = height;
    tiling->width = width;
    tiling->depth = depth;
    tiling->outBatch = outBatch;
    tiling->outHeight = outHeight;
    tiling->outWidth = outWidth;
    tiling->cropTop = cropTop;
    tiling->cropLeft = cropLeft;
    tiling->blockSize = blockSize;
    tiling->blockDim = blockDim;
    tiling->mode = mode;
    tiling->tileBlocks = tileBlocks;
    tiling->chunksPerLane = chunksPerLane;
    tiling->rowChunks = rowChunks;
    tiling->taskCount = taskCount;

    ASCENDC_TPL_SEL_PARAM(context, DT_X, mode);
    context->SetBlockDim(blockDim);
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    if (xShape == nullptr || yShape == nullptr || xShape->GetDimNum() != kRank) {
        return GRAPH_SUCCESS;
    }

    uint32_t cropTop;
    uint32_t cropBottom;
    uint32_t cropLeft;
    uint32_t cropRight;
    uint32_t blockSize;
    ReadAttrs(context, cropTop, cropBottom, cropLeft, cropRight, blockSize);

    uint32_t batch = U32Dim(xShape, 0);
    uint32_t height = U32Dim(xShape, 1);
    uint32_t width = U32Dim(xShape, 2);
    uint32_t depth = U32Dim(xShape, 3);
    uint32_t blockArea = blockSize * blockSize;

    yShape->SetDimNum(kRank);
    yShape->SetDim(0, blockArea == 0 ? 0 : batch / blockArea);
    yShape->SetDim(1, height * blockSize - cropTop - cropBottom);
    yShape->SetDim(2, width * blockSize - cropLeft - cropRight);
    yShape->SetDim(3, depth);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class BatchToSpace : public OpDef {
public:
    explicit BatchToSpace(const char *name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("crops").AttrType(REQUIRED).ListInt();
        this->Attr("block_size").AttrType(REQUIRED).Int();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(BatchToSpace);
}
