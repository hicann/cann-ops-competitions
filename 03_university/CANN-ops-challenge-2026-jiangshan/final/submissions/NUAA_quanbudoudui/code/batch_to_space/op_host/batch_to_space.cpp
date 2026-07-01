#include <algorithm>
#include <cstdint>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/batch_to_space_tiling.h"
#include "../op_kernel/tiling_key_batch_to_space.h"

namespace {
constexpr uint64_t kScatterMinOutputBytes = 8U * 1024U;
constexpr uint64_t kHighDepthPackMinOutputBytes = 64U * 1024U;
constexpr uint64_t kBs4HighDepthPackMinOutputBytes = 64U * 1024U;
constexpr uint64_t kHighDilationMinOutputBytes = 8U * 1024U;
constexpr uint32_t kScatterMaxTileCols = 512U;
constexpr uint64_t kScatterDiagMinTileCols = 16U;
constexpr uint32_t kUbWorkBytes = 16U * 1024U;
constexpr uint32_t kFastUbWorkBytes = 32U * 1024U;
constexpr uint64_t kDataCopyMaxBlockLen = 65535U;
constexpr uint64_t kDataCopyMaxStride = 65535U;
constexpr bool kEnableLane = true;
constexpr bool kEnablePack = false;
constexpr bool kEnableHighDil = false;
constexpr uint32_t kLightGenericLargeUbElems = 32768U;
constexpr uint64_t kLightGenericHighDepthUbBytes = 128U * 1024U;
constexpr uint32_t kLaneSafeVeryLargeWorkElems = 40960U;
constexpr uint64_t kLaneBs2AlignedLargeMinOutputBytes = 512U * 1024U;
constexpr bool kEnableRowGroupPack = true;
constexpr bool kEnableT9Bs4AlignedRowAssemble = true;
constexpr uint64_t kT9Bs4RowAssembleMinOutputBytes = 4ULL * 1024ULL * 1024ULL;
constexpr uint64_t kT9Bs4RowAssembleMaxOutputBytes = 8ULL * 1024ULL * 1024ULL;
constexpr uint32_t kT9Bs4RowAssembleUbBytes = 128U * 1024U;
constexpr uint64_t kT9Bs4RowAssembleMaxLaneCols = 65535ULL;
constexpr uint64_t kT9Bs4RowAssembleMaxRowsPerJob = 2ULL;
constexpr uint64_t kT9Bs4RowAssembleTargetJobsPerCore = 6ULL;


constexpr bool kEnableAdaptiveRowGroupPack = true;
constexpr uint64_t kTinyPolicyMaxOutputBytes = 16U * 1024U;
constexpr uint64_t kTinyPolicyMaxElements = 4096U;
constexpr uint64_t kTinyPolicyMaxC = 16U;
constexpr uint32_t kT10RowGroupPackRows = 32U;
constexpr uint32_t kRowGroupPackRows16 = 16U;
constexpr uint32_t kRowGroupPackRows8 = 8U;
constexpr uint32_t kT10RowGroupPackBytes = 128U * 1024U;


constexpr bool kEnableBlock1CropRowContract = true;
constexpr bool kEnableBlock1VerticalCropContig = true;
constexpr bool kEnableSmallCRowOwner = true;
constexpr uint32_t kBlock1CropRowContractBytes = 32U * 1024U;
constexpr uint32_t kBlock1VerticalCropContigBytes = 32U * 1024U;




constexpr bool kEnableT4Bs2MidLaneContract = true;

constexpr bool kEnableT7C1_8x8SmallCropMc8 = true;
constexpr uint64_t kT7C1_8x8MinOutputElems = 224ULL;
constexpr uint64_t kT7C1_8x8MaxOutputElems = 256ULL;
constexpr uint64_t kT7C1_8x8MinOutputBytes = 448ULL;
constexpr uint64_t kT7C1_8x8MaxOutputBytes = 512ULL;
constexpr uint64_t kT7C1_8x8MinRowBytes = 28ULL;
constexpr uint64_t kT7C1_8x8MaxRowBytes = 32ULL;


constexpr bool kEnableT7Bs2SmallCTinyWideOwner = true;
constexpr uint64_t kT7WideMinOutputBytes = 256ULL;
constexpr uint64_t kT7WideMaxOutputBytes = 8ULL * 1024ULL;
constexpr uint64_t kT7WideMinRowBytes = 16ULL;
constexpr uint64_t kT7WideMaxRowBytes = 512ULL;
constexpr uint64_t kT7WideMaxInputBytes = 32ULL * 1024ULL;



constexpr bool kEnableT6Bs2Fp16HighDepthMedium = false;
constexpr bool kEnableT7Bs2Fp16MediumNarrow = false;
constexpr bool kEnableT7Bs2Fp16MediumNarrowLane = false;
constexpr bool kEnableT7OutW1AlignedRowContract = true;
constexpr bool kEnableT7OutW2AlignedRowContract = true;
constexpr bool kEnableT7OutW4AlignedRowContract = false;
constexpr bool kEnableT7OutW7AlignedRowContract = false;

constexpr bool kEnableT7BucketDelaySentinel = true;
constexpr uint64_t kT67MediumMinOutputBytes = 64ULL * 1024ULL;
constexpr uint64_t kT67MediumMaxOutputBytes = 512ULL * 1024ULL;
constexpr uint64_t kT7NarrowRowBytes = 256ULL;
constexpr uint32_t kT7NarrowOutW = 8U;
constexpr uint32_t kT7OutW1RowContractRows = 64U;
constexpr uint32_t kT7OutW2RowContractRows = 64U;
constexpr uint32_t kT7OutW4RowContractRows = 64U;
constexpr uint32_t kT7RowContractUbBytes = 64U * 1024U;
constexpr uint32_t kT7B5PerfUbBytes = 32U * 1024U;
constexpr uint64_t kT4Bs2MidMinOutputBytes = 16U * 1024U;
constexpr uint64_t kT4Bs2MidMaxOutputBytes = 64U * 1024U;
constexpr uint64_t kT4Bs2MidMinRowBytes = 256U;
constexpr uint64_t kT4Bs2MidMaxRowBytes = 4U * 1024U;
constexpr uint64_t kT4Bs2MidMaxC = 128U;
constexpr uint32_t kBlock1CropRowContractMaxRows = 16U;
constexpr uint32_t kSmallCRowOwnerBytes = 16U * 1024U;
constexpr uint64_t kSmallCRowOwnerMaxOutputBytes = 16U * 1024U;
constexpr uint64_t kSmallCRowOwnerMaxDepthBytes = 64U;
}

namespace optiling {

static uint64_t SelectT7Bs2SmallCTinyJobs(uint64_t totalElements, uint64_t C) {





    if (totalElements <= 128ULL) {
        return 1ULL;
    }
    if (C == 1ULL) {
        return 8ULL;
    }
    if (C == 2ULL || C == 3ULL) {
        return (totalElements <= 1024ULL) ? 14ULL : 8ULL;
    }
    if (C == 4ULL) {
        return (totalElements <= 4096ULL) ? 16ULL : 8ULL;
    }
    return 8ULL;
}

static uint32_t SelectT7RowContractWorkElems(uint64_t outW, uint64_t C, uint32_t rows) {
    const uint64_t rowElems = outW * C;
    if (rowElems == 0ULL) {
        return 1U;
    }
    uint64_t work = rowElems * static_cast<uint64_t>(rows);
    if (work == 0ULL) {
        return 1U;
    }
    if (work > 0xFFFFFFFFULL) {
        return 0xFFFFFFFFU;
    }
    return static_cast<uint32_t>(work);
}


static uint64_t AlignUpHost(uint64_t value, uint64_t align) {
    if (align == 0ULL) {
        return value;
    }
    return ((value + align - 1ULL) / align) * align;
}

static uint64_t AlignDownHost(uint64_t value, uint64_t align) {
    if (align == 0ULL) {
        return value;
    }
    return (value / align) * align;
}

static uint32_t SelectT7B5AlignedPerfWorkElems(uint64_t outN, uint64_t outH, uint64_t outW,
                                               uint64_t C, uint32_t dtypeSize,
                                               uint64_t outputBytes, uint32_t numCores) {
    const uint64_t elemBlock = dtypeSize == 0U ? 1ULL : 32ULL / static_cast<uint64_t>(dtypeSize);
    const uint64_t rowElems = outW * C;
    const uint64_t totalRows = outN * outH;
    if (rowElems == 0ULL || totalRows == 0ULL || C == 0ULL) {
        return 1U;
    }
    uint64_t targetJobs = 8ULL;
    if (outputBytes > 256ULL * 1024ULL) {
        targetJobs = 24ULL;
    } else if (outputBytes > 128ULL * 1024ULL) {
        targetJobs = 16ULL;
    }
    const uint64_t cores = static_cast<uint64_t>(numCores == 0U ? 1U : numCores);
    if (targetJobs > cores) {
        targetJobs = cores;
    }
    if (targetJobs == 0ULL) {
        targetJobs = 1ULL;
    }
    const uint64_t maxWorkElemsRaw = static_cast<uint64_t>(kT7B5PerfUbBytes) / static_cast<uint64_t>(dtypeSize == 0U ? 1U : dtypeSize);
    uint64_t maxWorkElems = AlignDownHost(maxWorkElemsRaw, elemBlock);
    if (maxWorkElems < elemBlock) {
        maxWorkElems = elemBlock;
    }
    const uint64_t pixelJobs = totalRows * outW;
    if (pixelJobs > 0ULL && pixelJobs < targetJobs && C > elemBlock) {
        const uint64_t chunksWanted = (targetJobs + pixelJobs - 1ULL) / pixelJobs;
        uint64_t chunkElems = AlignUpHost((C + chunksWanted - 1ULL) / chunksWanted, elemBlock);
        if (chunkElems > maxWorkElems) {
            chunkElems = maxWorkElems;
        }
        if (chunkElems < elemBlock) {
            chunkElems = elemBlock;
        }
        if (chunkElems < C) {
            return static_cast<uint32_t>(chunkElems);
        }
    }
    if (rowElems > maxWorkElems) {
        return static_cast<uint32_t>(maxWorkElems);
    }
    uint64_t rowsPerTile = (totalRows + targetJobs - 1ULL) / targetJobs;
    if (rowsPerTile == 0ULL) {
        rowsPerTile = 1ULL;
    }
    uint64_t tileElems = rowElems * rowsPerTile;
    if (tileElems > maxWorkElems) {
        rowsPerTile = maxWorkElems / rowElems;
        if (rowsPerTile == 0ULL) {
            rowsPerTile = 1ULL;
        }
        tileElems = rowElems * rowsPerTile;
    }
    if (tileElems == 0ULL) {
        tileElems = rowElems;
    }
    if (tileElems > 0xFFFFFFFFULL) {
        tileElems = 0xFFFFFFFFULL;
    }
    return static_cast<uint32_t>(tileElems);
}


static uint32_t SelectT9Bs4RowAssembleWorkElems(uint64_t outN, uint64_t outH, uint64_t outW,
                                                     uint64_t C, uint32_t dtypeSize,
                                                     uint64_t coreNum) {
    const uint64_t safeDtypeSize = static_cast<uint64_t>(dtypeSize == 0U ? 1U : dtypeSize);
    uint64_t maxWorkElems = static_cast<uint64_t>(kT9Bs4RowAssembleUbBytes) / safeDtypeSize;
    if (C == 0ULL) {
        return 1U;
    }

    const uint64_t groupElems = C * 4ULL;
    if (groupElems > 0ULL && maxWorkElems >= groupElems) {
        maxWorkElems = (maxWorkElems / groupElems) * groupElems;
    } else {
        maxWorkElems = groupElems;
    }
    if (maxWorkElems == 0ULL) {
        maxWorkElems = C;
    }

    const uint64_t rowElems = outW * C;
    const uint64_t totalRows = outN * outH;
    if (rowElems > 0ULL && rowElems <= maxWorkElems && totalRows > 0ULL) {
        uint64_t targetJobs = coreNum * kT9Bs4RowAssembleTargetJobsPerCore;
        if (targetJobs == 0ULL) targetJobs = 1ULL;

        if (totalRows < targetJobs && outW >= 8ULL) {
            uint64_t targetTilesPerRow = (targetJobs + totalRows - 1ULL) / totalRows;
            if (targetTilesPerRow == 0ULL) targetTilesPerRow = 1ULL;
            uint64_t tileOutCols = (outW + targetTilesPerRow - 1ULL) / targetTilesPerRow;
            tileOutCols = (tileOutCols / 4ULL) * 4ULL;
            if (tileOutCols == 0ULL) tileOutCols = 4ULL;
            if (tileOutCols < outW) {
                uint64_t workElems = tileOutCols * C;
                if (workElems < groupElems) workElems = groupElems;
                if (workElems > maxWorkElems) workElems = maxWorkElems;
                if (workElems > 0xFFFFFFFFULL) workElems = 0xFFFFFFFFULL;
                return static_cast<uint32_t>(workElems);
            }
        }

        uint64_t maxRowsByUb = maxWorkElems / rowElems;
        if (maxRowsByUb == 0ULL) maxRowsByUb = 1ULL;
        if (maxRowsByUb > kT9Bs4RowAssembleMaxRowsPerJob) {
            maxRowsByUb = kT9Bs4RowAssembleMaxRowsPerJob;
        }

        uint64_t rowsPerJob = (totalRows + targetJobs - 1ULL) / targetJobs;
        if (rowsPerJob == 0ULL) rowsPerJob = 1ULL;
        if (rowsPerJob > maxRowsByUb) rowsPerJob = maxRowsByUb;
        uint64_t workElems = rowsPerJob * rowElems;
        if (workElems >= groupElems) {
            workElems = (workElems / groupElems) * groupElems;
        }
        if (workElems < rowElems) workElems = rowElems;
        if (workElems > maxWorkElems) workElems = maxWorkElems;
        if (workElems > 0xFFFFFFFFULL) workElems = 0xFFFFFFFFULL;
        return static_cast<uint32_t>(workElems);
    }

    uint64_t workElems = maxWorkElems;
    const uint64_t totalRowsForTiles = outN * outH;
    if (outW > 0ULL && totalRowsForTiles > 0ULL) {
        uint64_t maxTileCols = maxWorkElems / C;
        maxTileCols = (maxTileCols / 4ULL) * 4ULL;
        if (maxTileCols == 0ULL) maxTileCols = 4ULL;
        if (maxTileCols > outW) maxTileCols = outW;

        uint64_t targetJobs = coreNum * kT9Bs4RowAssembleTargetJobsPerCore;
        if (targetJobs == 0ULL) targetJobs = 1ULL;
        uint64_t targetTilesPerRow = (targetJobs + totalRowsForTiles - 1ULL) / totalRowsForTiles;
        if (targetTilesPerRow == 0ULL) targetTilesPerRow = 1ULL;
        uint64_t targetTileCols = (outW + targetTilesPerRow - 1ULL) / targetTilesPerRow;
        targetTileCols = (targetTileCols / 4ULL) * 4ULL;
        if (targetTileCols == 0ULL) targetTileCols = 4ULL;
        if (targetTileCols > maxTileCols) targetTileCols = maxTileCols;
        workElems = targetTileCols * C;
        if (workElems < groupElems) workElems = groupElems;
        if (workElems > maxWorkElems) workElems = maxWorkElems;
    }
    if (workElems > 0xFFFFFFFFULL) {
        workElems = 0xFFFFFFFFULL;
    }
    return static_cast<uint32_t>(workElems);
}

static uint64_t SelectT9Bs4RowAssembleJobs(uint64_t outN, uint64_t outH, uint64_t outW,
                                           uint64_t C, uint32_t workElems) {
    if (outN == 0ULL || outH == 0ULL || outW == 0ULL || C == 0ULL || workElems == 0U) {
        return 1ULL;
    }
    const uint64_t rowElems = outW * C;
    const uint64_t work = static_cast<uint64_t>(workElems);
    if (rowElems > 0ULL && rowElems <= work) {
        uint64_t rowsPerJob = work / rowElems;
        if (rowsPerJob == 0ULL) {
            rowsPerJob = 1ULL;
        }
        if (rowsPerJob > kT9Bs4RowAssembleMaxRowsPerJob) {
            rowsPerJob = kT9Bs4RowAssembleMaxRowsPerJob;
        }
        const uint64_t rowGroups = (outH + rowsPerJob - 1ULL) / rowsPerJob;
        return outN * rowGroups;
    }
    uint64_t tileOutCols = work / C;
    if (tileOutCols == 0ULL) {
        tileOutCols = 1ULL;
    }
    tileOutCols = (tileOutCols / 4ULL) * 4ULL;
    if (tileOutCols == 0ULL) {
        tileOutCols = 4ULL;
    }
    if (tileOutCols > outW) {
        tileOutCols = outW;
    }
    const uint64_t tilesPerRow = (outW + tileOutCols - 1ULL) / tileOutCols;
    return outN * outH * tilesPerRow;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t num_cores_aiv = platform.GetCoreNumAiv();
    if (num_cores_aiv <= 0) num_cores_aiv = 1;

    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_x = tensor_x->GetDataType();
    uint32_t DT_X = static_cast<uint32_t>(dtype_x);

    auto shape_x = tensor_x->GetStorageShape();
    uint64_t N = shape_x.GetDim(0);
    uint64_t H = shape_x.GetDim(1);
    uint64_t W = shape_x.GetDim(2);
    uint64_t C = shape_x.GetDim(3);

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const gert::TypedContinuousVector<int64_t> *attr_crops = attrs->GetListInt(0);
    const int64_t *attr_block_size = attrs->GetInt(1);

    uint64_t block_size = static_cast<uint64_t>(attr_block_size[0]);
    uint64_t crop_top = static_cast<uint64_t>(attr_crops->GetData()[0]);
    uint64_t crop_bottom = static_cast<uint64_t>(attr_crops->GetData()[1]);
    uint64_t crop_left = static_cast<uint64_t>(attr_crops->GetData()[2]);
    uint64_t crop_right = static_cast<uint64_t>(attr_crops->GetData()[3]);

    uint64_t outN = N / (block_size * block_size);
    uint64_t outH = H * block_size - crop_top - crop_bottom;
    uint64_t outW = W * block_size - crop_left - crop_right;
    uint64_t totalElements = outN * outH * outW * C;

    uint32_t dtype_size = (dtype_x == ge::DT_FLOAT) ? 4U : 2U;
    uint32_t elements_per_block = 32U / dtype_size;


    uint32_t mode = BATCH_TO_SPACE_MODE_GENERAL;
    if (block_size == 1 && crop_top == 0 && crop_bottom == 0 &&
        crop_left == 0 && crop_right == 0) {
        mode = BATCH_TO_SPACE_MODE_GENERAL;
    }

    uint64_t legacyMode = 2U;
    if (block_size == 1 && crop_top == 0 && crop_bottom == 0 && crop_left == 0 && crop_right == 0) {
        legacyMode = 0U;
    } else if (block_size == 1) {
        legacyMode = 1U;
    } else if (C <= 16384U && ((C % elements_per_block) != 0 || C <= 16U)) {


        legacyMode = 3U;
    }

    const uint64_t outputBytes64 = totalElements * static_cast<uint64_t>(dtype_size);
    const uint64_t inputBytes64 = N * H * W * C * static_cast<uint64_t>(dtype_size);
    const uint64_t depthBytes64 = C * static_cast<uint64_t>(dtype_size);
    const bool tinyPolicy = (block_size > 1U) && (C > 0U) && (C <= kTinyPolicyMaxC) &&
                            (totalElements <= kTinyPolicyMaxElements) &&
                            (outputBytes64 <= kTinyPolicyMaxOutputBytes);
    const uint64_t paddedDepthBytes64 = (depthBytes64 + 31U) & ~static_cast<uint64_t>(31U);
    const uint64_t paddedDepthElems64 = (dtype_size == 0) ? 0U : paddedDepthBytes64 / static_cast<uint64_t>(dtype_size);
    const uint32_t ubWorkElems = std::max<uint32_t>(1U, kUbWorkBytes / dtype_size);



    uint32_t laneWorkBytes = kFastUbWorkBytes;
    if (outputBytes64 <= 65536U) {
        laneWorkBytes = 24U * 1024U;
    } else if (outputBytes64 > 131072U) {
        laneWorkBytes = (block_size == 2U) ? (24U * 1024U) : (20U * 1024U);
    }
    uint32_t scatterWorkElems = std::max<uint32_t>(1U, laneWorkBytes / dtype_size);
    if (C > 0U && C <= static_cast<uint64_t>(scatterWorkElems) && (depthBytes64 % 32U) == 0U) {
        uint32_t depthElemsU32 = static_cast<uint32_t>(C);
        uint32_t alignedScatterElems = (scatterWorkElems / depthElemsU32) * depthElemsU32;
        if (alignedScatterElems > 0U) scatterWorkElems = alignedScatterElems;
    }
    uint64_t scatterTileCols64 = 0U;
    if (paddedDepthElems64 > 0U) {
        scatterTileCols64 = static_cast<uint64_t>(scatterWorkElems) / paddedDepthElems64;
        if (scatterTileCols64 > static_cast<uint64_t>(kScatterMaxTileCols)) scatterTileCols64 = kScatterMaxTileCols;
    }

    uint32_t packWorkElems = ubWorkElems;
    uint64_t rowElems64 = outW * C;
    if (rowElems64 > 0U && static_cast<uint64_t>(packWorkElems) > rowElems64) {
        packWorkElems = static_cast<uint32_t>(rowElems64);
    }
    if (C > 0U && C <= static_cast<uint64_t>(packWorkElems) && (depthBytes64 % 32U) == 0U) {
        uint32_t depthElemsU32 = static_cast<uint32_t>(C);
        uint32_t alignedPackElems = (packWorkElems / depthElemsU32) * depthElemsU32;
        if (alignedPackElems > 0U) packWorkElems = alignedPackElems;
    }
    uint64_t packTileCols64 = 0U;
    if (paddedDepthElems64 > 0U) {
        packTileCols64 = static_cast<uint64_t>(packWorkElems) / paddedDepthElems64;
        if (packTileCols64 > static_cast<uint64_t>(kScatterMaxTileCols)) packTileCols64 = kScatterMaxTileCols;
    }

    const uint64_t scatterDstStrideBytes = block_size > 0U ? (block_size - 1U) * depthBytes64 : 0U;
    const bool shapeFormulaOk = block_size >= 1U && outN * block_size * block_size == N &&
                                outH + crop_top + crop_bottom == H * block_size &&
                                outW + crop_left + crop_right == W * block_size;
    const bool commonBlockShape = shapeFormulaOk && block_size >= 2U;
    const bool laneShapeBase = commonBlockShape && block_size <= 4U && outputBytes64 >= kScatterMinOutputBytes &&
                               outW >= block_size * 4U && depthBytes64 > 0U &&
                               depthBytes64 <= kDataCopyMaxBlockLen && scatterDstStrideBytes <= kDataCopyMaxStride &&
                               scatterTileCols64 >= 2U;
    const bool highDepthPackBase = commonBlockShape && block_size <= 4U && (depthBytes64 % 32U) == 0U &&
                                   depthBytes64 > 0U && depthBytes64 <= kDataCopyMaxBlockLen &&
                                   scatterDstStrideBytes <= kDataCopyMaxStride &&
                                   outputBytes64 >= kHighDepthPackMinOutputBytes && outW >= 8U &&
                                   packTileCols64 >= 2U && packTileCols64 < kScatterDiagMinTileCols;
    const bool highDepthContigPack = highDepthPackBase && block_size == 2U;
    const bool bs4HighDepthContigPack = highDepthPackBase && block_size == 4U &&
                                        outputBytes64 >= kBs4HighDepthPackMinOutputBytes;
    const uint32_t rowGroupWorkElems = std::max<uint32_t>(1U, kT10RowGroupPackBytes / dtype_size);
    const bool rowGroupPackBase = commonBlockShape && block_size == 2U && (depthBytes64 % 32U) == 0U &&
                                  depthBytes64 > 0U && depthBytes64 <= kDataCopyMaxBlockLen &&
                                  scatterDstStrideBytes <= kDataCopyMaxStride && outW >= 8U;

    const bool bs2RowGroupPack32 = rowGroupPackBase && outputBytes64 >= (6ULL * 1024ULL * 1024ULL) &&
                                   C <= 4096U && packTileCols64 >= 8U && rowElems64 > 0U &&
                                   rowElems64 <= static_cast<uint64_t>(rowGroupWorkElems) / static_cast<uint64_t>(kT10RowGroupPackRows);


    const bool adaptiveRowGroupAllowed = kEnableAdaptiveRowGroupPack && !tinyPolicy && (legacyMode == 2U);
    const bool bs2RowGroupPack16 = adaptiveRowGroupAllowed && rowGroupPackBase &&
                                   outputBytes64 >= (1ULL * 1024ULL * 1024ULL) &&
                                   C <= 4096U && packTileCols64 >= 4U && rowElems64 > 0U &&
                                   rowElems64 <= static_cast<uint64_t>(rowGroupWorkElems) / static_cast<uint64_t>(kRowGroupPackRows16);
    const bool bs2RowGroupPack8 = adaptiveRowGroupAllowed && rowGroupPackBase &&
                                  outputBytes64 >= (512ULL * 1024ULL) &&
                                  C <= 4096U && packTileCols64 >= 4U && rowElems64 > 0U &&
                                  rowElems64 <= static_cast<uint64_t>(rowGroupWorkElems) / static_cast<uint64_t>(kRowGroupPackRows8);
    const bool laneStridedScatterSafe = laneShapeBase && scatterTileCols64 >= kScatterDiagMinTileCols &&
                                        !highDepthContigPack && !bs4HighDepthContigPack;
    const bool highDilationLaneScatter = commonBlockShape && block_size > 4U && block_size <= 24U &&
                                         outputBytes64 >= kHighDilationMinOutputBytes && depthBytes64 > 0U &&
                                         depthBytes64 <= kDataCopyMaxBlockLen && scatterDstStrideBytes <= kDataCopyMaxStride &&
                                         scatterTileCols64 >= 1U;


    const bool isBlock24Light = (block_size == 2U || block_size == 4U);
    const bool depthAligned32Light = (depthBytes64 % 32U) == 0U;
    const bool highDepthLowTileLight = commonBlockShape && isBlock24Light && depthAligned32Light &&
                                       depthBytes64 >= 64U && depthBytes64 <= kDataCopyMaxBlockLen &&
                                       scatterDstStrideBytes <= kDataCopyMaxStride &&
                                       outputBytes64 >= 65536U && packTileCols64 >= 2U &&
                                       packTileCols64 < kScatterDiagMinTileCols;
    const bool lightGenericLargeRange = isBlock24Light && depthAligned32Light &&
                                        C <= static_cast<uint64_t>(kLightGenericLargeUbElems) &&
                                        outputBytes64 <= (16U * 1024U * 1024U);
    const bool lightGenericFamily = commonBlockShape && block_size <= 64U && legacyMode == 2U &&
                                    !laneStridedScatterSafe && outputBytes64 >= 524288ULL &&
                                    (outputBytes64 <= (256U * 1024U) || lightGenericLargeRange) &&
                                    C > 0U && depthBytes64 > 0U;
    const bool lightGenericHighDepthPolicy = lightGenericFamily && highDepthLowTileLight &&
                                             outputBytes64 > (512U * 1024U) && isBlock24Light && depthAligned32Light;
    const uint32_t lightGenericHighDepthWorkElems = std::max<uint32_t>(
        kLightGenericLargeUbElems,
        static_cast<uint32_t>(kLightGenericHighDepthUbBytes / static_cast<uint64_t>(dtype_size)));

    const uint64_t inputRowBytes64 = W * C * static_cast<uint64_t>(dtype_size);
    const uint64_t outputRowBytes64 = outW * C * static_cast<uint64_t>(dtype_size);
    const uint64_t cropLeftBytes64 = crop_left * C * static_cast<uint64_t>(dtype_size);
    const bool block1HasCrop = (crop_top | crop_bottom | crop_left | crop_right) != 0U;
    const uint64_t outputBatchBytes64 = outH * outW * C * static_cast<uint64_t>(dtype_size);
    const uint64_t inputBatchBytes64 = H * W * C * static_cast<uint64_t>(dtype_size);
    const uint64_t cropTopBytes64 = crop_top * W * C * static_cast<uint64_t>(dtype_size);


    const bool block1VerticalCropContig = kEnableBlock1VerticalCropContig && shapeFormulaOk && block_size == 1U &&
                                          crop_left == 0U && crop_right == 0U &&
                                          (crop_top != 0U || crop_bottom != 0U) &&
                                          outN > 0U && outH > 0U && outW > 0U && C > 0U &&
                                          outputBatchBytes64 > 0U &&
                                          (outputBatchBytes64 & 31ULL) == 0U &&
                                          (inputBatchBytes64 & 31ULL) == 0U &&
                                          (cropTopBytes64 & 31ULL) == 0U;
    const bool block1CropRowContract = kEnableBlock1CropRowContract && shapeFormulaOk && block_size == 1U &&
                                       block1HasCrop && !block1VerticalCropContig &&
                                       outN > 0U && outH > 0U && outW > 0U && C > 0U &&
                                       outputRowBytes64 > 0U && outputRowBytes64 <= kDataCopyMaxBlockLen &&
                                       inputRowBytes64 >= outputRowBytes64 &&
                                       (outputRowBytes64 & 31ULL) == 0U &&
                                       (inputRowBytes64 & 31ULL) == 0U &&
                                       (cropLeftBytes64 & 31ULL) == 0U &&
                                       ((inputRowBytes64 - outputRowBytes64) <= kDataCopyMaxStride);
    const bool smallCRowOwnerBase = kEnableSmallCRowOwner && commonBlockShape && tinyPolicy &&
                                    (block_size == 2U || block_size == 4U) && C > 0U && C <= kTinyPolicyMaxC &&
                                    depthBytes64 > 0U && depthBytes64 <= kSmallCRowOwnerMaxDepthBytes &&
                                    depthBytes64 <= kDataCopyMaxBlockLen && scatterDstStrideBytes <= kDataCopyMaxStride &&
                                    outputBytes64 <= kSmallCRowOwnerMaxOutputBytes && outN > 0U && outH > 0U && outW > 0U &&
                                    ((outW + block_size - 1U) / block_size) <= 4095U;

    const uint64_t outputRowBytesForT4Mid64 = outW * C * static_cast<uint64_t>(dtype_size);
    const bool t4Bs2MidLaneContract = kEnableT4Bs2MidLaneContract && shapeFormulaOk &&
                                    block_size == 2U && outN > 0U && outH > 0U && outW > 0U && C > 0U &&
                                    outputBytes64 > kT4Bs2MidMinOutputBytes &&
                                    outputBytes64 <= kT4Bs2MidMaxOutputBytes &&
                                    outputRowBytesForT4Mid64 > kT4Bs2MidMinRowBytes &&
                                    outputRowBytesForT4Mid64 <= kT4Bs2MidMaxRowBytes &&
                                    C <= kT4Bs2MidMaxC;

    const bool t7C1_8x8SmallCropMc8 = kEnableT7C1_8x8SmallCropMc8 && shapeFormulaOk &&
                                    dtype_x == ge::DT_FLOAT16 && block_size == 2U &&
                                    N == 4U && H == 8U && W == 8U && C == 1U && outN == 1U &&
                                    outH >= 14U && outH <= 16U && outW >= 14U && outW <= 16U &&
                                    totalElements >= kT7C1_8x8MinOutputElems &&
                                    totalElements <= kT7C1_8x8MaxOutputElems &&
                                    outputBytes64 >= kT7C1_8x8MinOutputBytes &&
                                    outputBytes64 <= kT7C1_8x8MaxOutputBytes &&
                                    outputRowBytesForT4Mid64 >= kT7C1_8x8MinRowBytes &&
                                    outputRowBytesForT4Mid64 <= kT7C1_8x8MaxRowBytes &&
                                    (crop_top + crop_bottom) <= 2U &&
                                    (crop_left + crop_right) <= 2U;


    const bool t7Bs2SmallCTinyWideOwner = kEnableT7Bs2SmallCTinyWideOwner && shapeFormulaOk &&
                                    dtype_x == ge::DT_FLOAT16 && block_size == 2U &&
                                    C >= 1U && C <= 4U && (depthBytes64 % 32ULL) != 0ULL &&
                                    outputBytes64 >= kT7WideMinOutputBytes &&
                                    outputBytes64 <= kT7WideMaxOutputBytes &&
                                    outputRowBytesForT4Mid64 >= kT7WideMinRowBytes &&
                                    outputRowBytesForT4Mid64 <= kT7WideMaxRowBytes &&
                                    outN >= 1U && outN <= 2U && outH >= 2U && outW >= 2U &&
                                    inputBytes64 <= kT7WideMaxInputBytes;






    const bool bs2Fp16MediumGeneralCandidate = shapeFormulaOk && commonBlockShape &&
                                    dtype_x == ge::DT_FLOAT16 && block_size == 2U &&
                                    outputBytes64 > kT67MediumMinOutputBytes &&
                                    outputBytes64 <= kT67MediumMaxOutputBytes &&
                                    outN > 0U && outH > 0U && outW > 0U && C > 0U;
    const bool v161HighDepthLowTile = (depthBytes64 % 32ULL) == 0ULL &&
                                    depthBytes64 >= 64ULL &&
                                    packTileCols64 >= 2ULL &&
                                    packTileCols64 < kScatterDiagMinTileCols;
    const bool v161LaneCandidate = (depthBytes64 % 32ULL) == 0ULL &&
                                    outputRowBytesForT4Mid64 >= 256ULL &&
                                    outW >= 8U;
    const bool v161NonAlignedSmallC = (depthBytes64 % 32ULL) != 0ULL && C <= 16U;
    const bool v161NarrowCandidate = (outW < kT7NarrowOutW) ||
                                    (outputRowBytesForT4Mid64 < kT7NarrowRowBytes);
    const bool t6Bs2Fp16HighDepthMedium = kEnableT6Bs2Fp16HighDepthMedium &&
                                    bs2Fp16MediumGeneralCandidate &&
                                    v161HighDepthLowTile;
    const bool t7Bs2Fp16MediumNarrow = kEnableT7Bs2Fp16MediumNarrow &&
                                    bs2Fp16MediumGeneralCandidate &&
                                    !v161HighDepthLowTile &&
                                    !v161LaneCandidate &&
                                    !v161NonAlignedSmallC &&
                                    v161NarrowCandidate;
    const bool t7AlignedNarrowBase = bs2Fp16MediumGeneralCandidate &&
                                    !v161HighDepthLowTile &&
                                    !v161LaneCandidate &&
                                    !v161NonAlignedSmallC &&
                                    v161NarrowCandidate &&
                                    (depthBytes64 % 32ULL) == 0ULL &&
                                    (outputRowBytesForT4Mid64 % 32ULL) == 0ULL;
    const bool t7OutW1AlignedRowContract = kEnableT7OutW1AlignedRowContract &&
                                    t7AlignedNarrowBase &&
                                    outW == 1U &&
                                    outputRowBytesForT4Mid64 <= 128ULL;
    const bool t7OutW2AlignedRowContract = kEnableT7OutW2AlignedRowContract &&
                                    !t7OutW1AlignedRowContract &&
                                    t7AlignedNarrowBase &&
                                    outW <= 2U &&
                                    outputRowBytesForT4Mid64 <= 128ULL;
    const bool t7OutW4AlignedRowContract = kEnableT7OutW4AlignedRowContract &&
                                    !t7OutW1AlignedRowContract &&
                                    !t7OutW2AlignedRowContract &&
                                    t7AlignedNarrowBase &&
                                    outW <= 4U &&
                                    outputRowBytesForT4Mid64 <= 256ULL;
    const bool t7OutW7AlignedRowContract = kEnableT7OutW7AlignedRowContract &&
                                    !t7OutW1AlignedRowContract &&
                                    !t7OutW2AlignedRowContract &&
                                    !t7OutW4AlignedRowContract &&
                                    t7AlignedNarrowBase &&
                                    outW < 8U &&
                                    outputRowBytesForT4Mid64 <= 256ULL;
    const bool t7Bs2Fp16MediumNarrowLane = kEnableT7Bs2Fp16MediumNarrowLane &&
                                    t7AlignedNarrowBase &&
                                    outputRowBytesForT4Mid64 <= 256ULL &&
                                    outW <= 4U;




    const bool t7BucketBase = kEnableT7BucketDelaySentinel &&
                                    bs2Fp16MediumGeneralCandidate &&
                                    !v161HighDepthLowTile &&
                                    !v161LaneCandidate &&
                                    v161NarrowCandidate;
    const bool t7BucketAlignedSafe = t7BucketBase &&
                                    (depthBytes64 % 32ULL) == 0ULL &&
                                    (outputRowBytesForT4Mid64 % 32ULL) == 0ULL &&
                                    outputRowBytesForT4Mid64 <= 256ULL;
    const bool t7BucketOutW1 = t7BucketAlignedSafe &&
                                    outW == 1U &&
                                    outputRowBytesForT4Mid64 <= 128ULL;
    const bool t7BucketOutW2 = !t7BucketOutW1 &&
                                    t7BucketAlignedSafe &&
                                    outW <= 2U &&
                                    outputRowBytesForT4Mid64 <= 128ULL;
    const bool t7BucketOutW34 = !t7BucketOutW1 && !t7BucketOutW2 &&
                                    t7BucketAlignedSafe &&
                                    outW >= 3U && outW <= 4U &&
                                    outputRowBytesForT4Mid64 <= 256ULL;
    const bool t7BucketOutW57 = !t7BucketOutW1 && !t7BucketOutW2 && !t7BucketOutW34 &&
                                    t7BucketAlignedSafe &&
                                    outW > 4U && outW < 8U &&
                                    outputRowBytesForT4Mid64 <= 256ULL;
    const bool t7BucketOther = t7BucketBase &&
                                    !t7BucketOutW1 && !t7BucketOutW2 &&
                                    !t7BucketOutW34 && !t7BucketOutW57;





    const bool t7OtherB1W34Row256_512Out64_128 = t7BucketOther &&
                                    (depthBytes64 % 32ULL) == 0ULL &&
                                    outW >= 3U && outW <= 4U &&
                                    outputRowBytesForT4Mid64 > 256ULL &&
                                    outputRowBytesForT4Mid64 <= 512ULL &&
                                    outputBytes64 <= 128ULL * 1024ULL;
    const bool t7OtherB2W34Row256_512Out128_512 = t7BucketOther &&
                                    !t7OtherB1W34Row256_512Out64_128 &&
                                    (depthBytes64 % 32ULL) == 0ULL &&
                                    outW >= 3U && outW <= 4U &&
                                    outputRowBytesForT4Mid64 > 256ULL &&
                                    outputRowBytesForT4Mid64 <= 512ULL &&
                                    outputBytes64 > 128ULL * 1024ULL;
    const bool t7OtherB3W34RowGt512 = t7BucketOther &&
                                    !t7OtherB1W34Row256_512Out64_128 &&
                                    !t7OtherB2W34Row256_512Out128_512 &&
                                    (depthBytes64 % 32ULL) == 0ULL &&
                                    outW >= 3U && outW <= 4U &&
                                    outputRowBytesForT4Mid64 > 512ULL;
    const bool t7OtherB4W57RowGt256 = t7BucketOther &&
                                    !t7OtherB1W34Row256_512Out64_128 &&
                                    !t7OtherB2W34Row256_512Out128_512 &&
                                    !t7OtherB3W34RowGt512 &&
                                    (depthBytes64 % 32ULL) == 0ULL &&
                                    outW >= 5U && outW < 8U &&
                                    outputRowBytesForT4Mid64 > 256ULL;
    const bool t7OtherB5W12RowGt128 = t7BucketOther &&
                                    !t7OtherB1W34Row256_512Out64_128 &&
                                    !t7OtherB2W34Row256_512Out128_512 &&
                                    !t7OtherB3W34RowGt512 &&
                                    !t7OtherB4W57RowGt256 &&
                                    (depthBytes64 % 32ULL) == 0ULL &&
                                    outW <= 2U &&
                                    outputRowBytesForT4Mid64 > 128ULL;


    const bool t7OtherB6RowNarrowOutWGe8 = t7BucketOther &&
                                    !t7OtherB1W34Row256_512Out64_128 &&
                                    !t7OtherB2W34Row256_512Out128_512 &&
                                    !t7OtherB3W34RowGt512 &&
                                    !t7OtherB4W57RowGt256 &&
                                    !t7OtherB5W12RowGt128 &&
                                    outW >= 8U &&
                                    outputRowBytesForT4Mid64 < 256ULL;
    const bool t7OtherB7NonAlignedCGt16 = t7BucketOther &&
                                    !t7OtherB1W34Row256_512Out64_128 &&
                                    !t7OtherB2W34Row256_512Out128_512 &&
                                    !t7OtherB3W34RowGt512 &&
                                    !t7OtherB4W57RowGt256 &&
                                    !t7OtherB5W12RowGt128 &&
                                    !t7OtherB6RowNarrowOutWGe8 &&
                                    (depthBytes64 % 32ULL) != 0ULL && C > 16U;
    const bool t7OtherB8Other = t7BucketOther &&
                                    !t7OtherB1W34Row256_512Out64_128 &&
                                    !t7OtherB2W34Row256_512Out128_512 &&
                                    !t7OtherB3W34RowGt512 &&
                                    !t7OtherB4W57RowGt256 &&
                                    !t7OtherB5W12RowGt128 &&
                                    !t7OtherB6RowNarrowOutWGe8 &&
                                    !t7OtherB7NonAlignedCGt16;


    const uint32_t t9SelectedWorkElems = SelectT9Bs4RowAssembleWorkElems(outN, outH, outW, C, dtype_size, static_cast<uint64_t>(num_cores_aiv));
    const uint64_t t9TileOutCols = (C == 0ULL) ? 0ULL : (static_cast<uint64_t>(t9SelectedWorkElems) / C);
    const uint64_t t9SafeTileOutCols = (t9TileOutCols / 4ULL) * 4ULL;
    const bool t9Bs4AlignedRowAssemble = kEnableT9Bs4AlignedRowAssemble && laneStridedScatterSafe &&
                                               block_size == 4U && depthAligned32Light &&
                                               outputBytes64 >= kT9Bs4RowAssembleMinOutputBytes &&
                                               outputBytes64 <= kT9Bs4RowAssembleMaxOutputBytes &&
                                               C > 0U && depthBytes64 > 0U &&
                                               depthBytes64 <= kDataCopyMaxBlockLen &&
                                               ((block_size - 1U) * depthBytes64) <= kDataCopyMaxStride &&
                                               (outputRowBytesForT4Mid64 % 32ULL) == 0ULL &&
                                               ((outW + 3ULL) / 4ULL) <= kT9Bs4RowAssembleMaxLaneCols &&
                                               t9SafeTileOutCols >= 4ULL;

    uint32_t selectedWorkElems = scatterWorkElems;


    if (kEnableLane && laneStridedScatterSafe && block_size == 2U && depthAligned32Light &&
        outputBytes64 >= kLaneBs2AlignedLargeMinOutputBytes) {
        if (selectedWorkElems < kLaneSafeVeryLargeWorkElems) {
            selectedWorkElems = kLaneSafeVeryLargeWorkElems;
        }
    }
    if (block1VerticalCropContig) {
        mode = BATCH_TO_SPACE_MODE_BLOCK1_VERTICAL_CROP_CONTIG;
        selectedWorkElems = std::max<uint32_t>(1U, kBlock1VerticalCropContigBytes / dtype_size);
    } else if (block1CropRowContract) {
        mode = BATCH_TO_SPACE_MODE_BLOCK1_CROP_ROW_CONTRACT;
        selectedWorkElems = std::max<uint32_t>(1U, kBlock1CropRowContractBytes / dtype_size);
    } else if (t7C1_8x8SmallCropMc8) {
        mode = BATCH_TO_SPACE_MODE_T7_C1_8X8_SMALLCROP_MC8;
        selectedWorkElems = 1U;
    } else if (t7Bs2SmallCTinyWideOwner) {
        mode = BATCH_TO_SPACE_MODE_T7_BS2_SMALLC_TINY_WIDE_OWNER;
        selectedWorkElems = 1U;
    } else if (smallCRowOwnerBase && block_size == 2U) {
        mode = BATCH_TO_SPACE_MODE_SMALLC_ROW_OWNER_BS2;
        selectedWorkElems = std::max<uint32_t>(1U, kSmallCRowOwnerBytes / dtype_size);
    } else if (smallCRowOwnerBase && block_size == 4U) {
        mode = BATCH_TO_SPACE_MODE_SMALLC_ROW_OWNER_BS4;
        selectedWorkElems = std::max<uint32_t>(1U, kSmallCRowOwnerBytes / dtype_size);
    } else if (t4Bs2MidLaneContract) {
        mode = BATCH_TO_SPACE_MODE_T4_BS2_MID_LANE_CONTRACT;


        selectedWorkElems = std::max<uint32_t>(1U, (24U * 1024U) / dtype_size);
    } else if (kEnableRowGroupPack && bs2RowGroupPack32) {
        mode = BATCH_TO_SPACE_MODE_BS2_ROWGROUP_PACK;
        selectedWorkElems = rowGroupWorkElems;
    } else if (kEnableRowGroupPack && bs2RowGroupPack16) {
        mode = BATCH_TO_SPACE_MODE_BS2_ROWGROUP_PACK_16;
        selectedWorkElems = rowGroupWorkElems;
    } else if (kEnableRowGroupPack && bs2RowGroupPack8) {
        mode = BATCH_TO_SPACE_MODE_BS2_ROWGROUP_PACK_8;
        selectedWorkElems = rowGroupWorkElems;
    } else if (kEnablePack && highDepthContigPack) {
        mode = BATCH_TO_SPACE_MODE_HIGH_DEPTH_CONTIG_PACK;
        selectedWorkElems = packWorkElems;
    } else if (kEnablePack && bs4HighDepthContigPack) {
        mode = BATCH_TO_SPACE_MODE_BS4_HIGH_DEPTH_CONTIG_PACK;
        selectedWorkElems = packWorkElems;
    } else if (t9Bs4AlignedRowAssemble) {
        mode = BATCH_TO_SPACE_MODE_T9_BS4_ALIGNED_ROW_ASSEMBLE;
        selectedWorkElems = t9SelectedWorkElems;
    } else if (kEnableLane && laneStridedScatterSafe) {
        mode = BATCH_TO_SPACE_MODE_LANE_STRIDED_SCATTER_SAFE;
        selectedWorkElems = scatterWorkElems;
    } else if (kEnableHighDil && highDilationLaneScatter) {
        mode = BATCH_TO_SPACE_MODE_HIGH_DILATION_LANE_SCATTER;
        selectedWorkElems = scatterWorkElems;
    } else if (t7OtherB5W12RowGt128) {
        mode = BATCH_TO_SPACE_MODE_T7_OTHER_B5_W1_2_ROW_GT128_PERF;
        selectedWorkElems = SelectT7B5AlignedPerfWorkElems(outN, outH, outW, C, dtype_size, outputBytes64,
                                                          static_cast<uint32_t>(num_cores_aiv));
    } else if (mode == BATCH_TO_SPACE_MODE_GENERAL && legacyMode == 2U && (lightGenericFamily || (highDepthLowTileLight && outputBytes64 >= 524288ULL && legacyMode == 2U))) {
        mode = BATCH_TO_SPACE_MODE_LIGHTWEIGHT_GENERIC;
        selectedWorkElems = lightGenericHighDepthPolicy ? lightGenericHighDepthWorkElems : kLightGenericLargeUbElems;
    } else if (t7BucketOutW1) {
        mode = BATCH_TO_SPACE_MODE_T7_BUCKET_OUTW1_DELAY;
        selectedWorkElems = scatterWorkElems;
    } else if (t7BucketOutW2) {
        mode = BATCH_TO_SPACE_MODE_T7_BUCKET_OUTW2_DELAY;
        selectedWorkElems = scatterWorkElems;
    } else if (t7BucketOutW34) {
        mode = BATCH_TO_SPACE_MODE_T7_BUCKET_OUTW3_4_DELAY;
        selectedWorkElems = scatterWorkElems;
    } else if (t7BucketOutW57) {
        mode = BATCH_TO_SPACE_MODE_T7_BUCKET_OUTW5_7_DELAY;
        selectedWorkElems = scatterWorkElems;
    } else if (t7OtherB1W34Row256_512Out64_128) {
        mode = BATCH_TO_SPACE_MODE_T7_OTHER_B1_W3_4_ROW256_512_OUT64_128_DELAY;
        selectedWorkElems = scatterWorkElems;
    } else if (t7OtherB2W34Row256_512Out128_512) {
        mode = BATCH_TO_SPACE_MODE_T7_OTHER_B2_W3_4_ROW256_512_OUT128_512_DELAY;
        selectedWorkElems = scatterWorkElems;
    } else if (t7OtherB3W34RowGt512) {
        mode = BATCH_TO_SPACE_MODE_T7_OTHER_B3_W3_4_ROW_GT512_DELAY;
        selectedWorkElems = scatterWorkElems;
    } else if (t7OtherB4W57RowGt256) {
        mode = BATCH_TO_SPACE_MODE_T7_OTHER_B4_W5_7_ROW_GT256_DELAY;
        selectedWorkElems = scatterWorkElems;
    } else if (t7OtherB6RowNarrowOutWGe8) {
        mode = BATCH_TO_SPACE_MODE_T7_OTHER_B6_ROWNARROW_OUTW_GE8_DELAY;
        selectedWorkElems = scatterWorkElems;
    } else if (t7OtherB7NonAlignedCGt16) {
        mode = BATCH_TO_SPACE_MODE_T7_OTHER_B7_NONALIGNED_CGT16_DELAY;
        selectedWorkElems = scatterWorkElems;
    } else if (t7OtherB8Other) {
        mode = BATCH_TO_SPACE_MODE_T7_OTHER_B8_OTHER_DELAY;
        selectedWorkElems = scatterWorkElems;
    } else if (t7BucketOther) {
        mode = BATCH_TO_SPACE_MODE_T7_BUCKET_OTHER_DELAY;
        selectedWorkElems = scatterWorkElems;
    } else if (t6Bs2Fp16HighDepthMedium) {


        mode = BATCH_TO_SPACE_MODE_T6_BS2_FP16_HIGHDEPTH_MEDIUM_ROW2;
        selectedWorkElems = scatterWorkElems;
    } else if (t7OutW1AlignedRowContract) {
        mode = BATCH_TO_SPACE_MODE_T7_OUTW1_ALIGNED_ROWCONTRACT;
        selectedWorkElems = SelectT7RowContractWorkElems(outW, C, kT7OutW1RowContractRows);
    } else if (t7OutW2AlignedRowContract) {
        mode = BATCH_TO_SPACE_MODE_T7_OUTW2_ALIGNED_ROWCONTRACT;
        selectedWorkElems = SelectT7RowContractWorkElems(outW, C, kT7OutW2RowContractRows);
    } else if (t7OutW4AlignedRowContract) {
        mode = BATCH_TO_SPACE_MODE_T7_OUTW4_ALIGNED_ROWCONTRACT;
        selectedWorkElems = SelectT7RowContractWorkElems(outW, C, kT7OutW4RowContractRows);
    } else if (t7OutW7AlignedRowContract) {
        mode = BATCH_TO_SPACE_MODE_T7_OUTW7_ALIGNED_ROWCONTRACT;
        selectedWorkElems = scatterWorkElems;
    } else if (t7Bs2Fp16MediumNarrowLane) {
        mode = BATCH_TO_SPACE_MODE_T7_BS2_FP16_MEDIUM_NARROW_LANE;
        selectedWorkElems = scatterWorkElems;
    } else if (t7Bs2Fp16MediumNarrow) {

        mode = BATCH_TO_SPACE_MODE_T7_BS2_FP16_MEDIUM_NARROW;
        selectedWorkElems = scatterWorkElems;
    }

    uint32_t block_dim = static_cast<uint32_t>(num_cores_aiv);
    if (mode == BATCH_TO_SPACE_MODE_GENERAL || mode == BATCH_TO_SPACE_MODE_T7_BS2_FP16_MEDIUM_NARROW) {
        uint64_t tasksNeeded = block_dim;
        if (legacyMode == 0U) {
            tasksNeeded = (totalElements + 8191U) / 16384U;
            if (tasksNeeded == 0U) tasksNeeded = 1U;
            if (tasksNeeded < block_dim) block_dim = static_cast<uint32_t>(tasksNeeded);
        } else if (legacyMode == 1U) {
            uint64_t rowElemsCrop = outW * C;
            uint64_t workElemsCrop = static_cast<uint64_t>(kUbWorkBytes / dtype_size);
            if (workElemsCrop == 0U) workElemsCrop = 1U;
            if (rowElemsCrop > 0U && workElemsCrop > rowElemsCrop) workElemsCrop = rowElemsCrop;
            uint64_t segsPerRow = (rowElemsCrop == 0U) ? 1U : ((rowElemsCrop + workElemsCrop - 1U) / workElemsCrop);
            tasksNeeded = outN * outH * segsPerRow;
            if (tasksNeeded == 0U) tasksNeeded = 1U;
            if (tasksNeeded < block_dim) block_dim = static_cast<uint32_t>(tasksNeeded);
        } else if (legacyMode == 3U) {
            uint64_t chunkPixels = (C == 0U) ? 1U : (16384U / C);
            if (chunkPixels == 0U) chunkPixels = 1U;
            if (chunkPixels > outW) chunkPixels = outW;
            uint64_t chunksPerRow = (outW + chunkPixels - 1U) / chunkPixels;
            tasksNeeded = outN * outH * chunksPerRow;
            if (tasksNeeded == 0U) tasksNeeded = 1U;
            if (tasksNeeded < block_dim) block_dim = static_cast<uint32_t>(tasksNeeded);
        } else if (legacyMode == 2U) {


            uint64_t rowElemsRa = outW * C;
            uint64_t rowBytesRa = rowElemsRa * static_cast<uint64_t>(dtype_size);
            uint64_t workElemsRa = static_cast<uint64_t>(kUbWorkBytes / dtype_size);
            if (workElemsRa == 0U) workElemsRa = 1U;
            if (rowElemsRa > 0U && workElemsRa > rowElemsRa) workElemsRa = rowElemsRa;
            uint64_t chunksPerRowRa = (rowElemsRa == 0U) ? 1U : ((rowElemsRa + workElemsRa - 1U) / workElemsRa);
            tasksNeeded = outN * outH * chunksPerRowRa;
            if (tasksNeeded == 0U) tasksNeeded = 1U;




            if (outputBytes64 <= 131072U) {
                if (tasksNeeded < block_dim) {
                    uint64_t probeElemsPerCore = (totalElements + tasksNeeded - 1U) / tasksNeeded;
                    probeElemsPerCore = ((probeElemsPerCore + elements_per_block - 1U) / elements_per_block) * elements_per_block;
                    if (probeElemsPerCore >= C) {
                        block_dim = static_cast<uint32_t>(tasksNeeded);
                    } else {
                        block_dim = 1U;
                    }
                }
            } else if (rowBytesRa <= 16384U) {
                if (tasksNeeded < block_dim) block_dim = static_cast<uint32_t>(tasksNeeded);
            }
        }
    } else {
        uint64_t totalRows = outN * outH;
        uint64_t scheduledJobs = totalRows;
        if (mode == BATCH_TO_SPACE_MODE_BLOCK1_VERTICAL_CROP_CONTIG) {
            uint64_t batchElemsContract = outH * outW * C;
            uint64_t workElemsContract = static_cast<uint64_t>(selectedWorkElems == 0U ? 1U : selectedWorkElems);
            uint64_t segsPerBatch = (batchElemsContract == 0U) ? 1U :
                                    ((batchElemsContract + workElemsContract - 1U) / workElemsContract);
            scheduledJobs = outN * segsPerBatch;
        } else if (mode == BATCH_TO_SPACE_MODE_BLOCK1_CROP_ROW_CONTRACT) {
            uint64_t rowElemsContract = outW * C;
            uint64_t workElemsContract = static_cast<uint64_t>(selectedWorkElems == 0U ? 1U : selectedWorkElems);
            uint64_t rowsPerGroup = (rowElemsContract == 0U) ? 1U : (workElemsContract / rowElemsContract);
            if (rowsPerGroup == 0U) rowsPerGroup = 1U;
            if (rowsPerGroup > static_cast<uint64_t>(kBlock1CropRowContractMaxRows)) {
                rowsPerGroup = static_cast<uint64_t>(kBlock1CropRowContractMaxRows);
            }
            uint64_t rowGroups = (outH + rowsPerGroup - 1U) / rowsPerGroup;
            scheduledJobs = outN * rowGroups;
        } else if (mode == BATCH_TO_SPACE_MODE_T7_C1_8X8_SMALLCROP_MC8) {
            scheduledJobs = 8ULL;
        } else if (mode == BATCH_TO_SPACE_MODE_T7_BS2_SMALLC_TINY_WIDE_OWNER) {
            scheduledJobs = SelectT7Bs2SmallCTinyJobs(totalElements, C);
        } else if (mode == BATCH_TO_SPACE_MODE_T7_BUCKET_OUTW1_DELAY ||
                   mode == BATCH_TO_SPACE_MODE_T7_BUCKET_OUTW2_DELAY ||
                   mode == BATCH_TO_SPACE_MODE_T7_BUCKET_OUTW3_4_DELAY ||
                   mode == BATCH_TO_SPACE_MODE_T7_BUCKET_OUTW5_7_DELAY ||
                   mode == BATCH_TO_SPACE_MODE_T7_BUCKET_OTHER_DELAY) {

            scheduledJobs = totalRows;
        } else if (mode == BATCH_TO_SPACE_MODE_SMALLC_ROW_OWNER_BS2 ||
                   mode == BATCH_TO_SPACE_MODE_SMALLC_ROW_OWNER_BS4) {
            scheduledJobs = totalRows;
        } else if (mode == BATCH_TO_SPACE_MODE_T4_BS2_MID_LANE_CONTRACT) {



            scheduledJobs = outN * ((outH + 5ULL) / 6ULL);
        } else if (mode == BATCH_TO_SPACE_MODE_T7_OUTW1_ALIGNED_ROWCONTRACT ||
                   mode == BATCH_TO_SPACE_MODE_T7_OUTW2_ALIGNED_ROWCONTRACT ||
                   mode == BATCH_TO_SPACE_MODE_T7_OUTW4_ALIGNED_ROWCONTRACT) {


            uint64_t rowElemsContract = outW * C;
            uint64_t workElemsContract = static_cast<uint64_t>(selectedWorkElems == 0U ? 1U : selectedWorkElems);
            uint64_t rowsPerGroup = (rowElemsContract == 0U) ? 1U : (workElemsContract / rowElemsContract);
            if (rowsPerGroup == 0U) rowsPerGroup = 1U;
            if (rowsPerGroup > 1024ULL) rowsPerGroup = 1024ULL;
            uint64_t rowGroups = (outH + rowsPerGroup - 1ULL) / rowsPerGroup;
            if (rowGroups == 0ULL) rowGroups = 1ULL;
            scheduledJobs = outN * rowGroups;
        } else if (mode == BATCH_TO_SPACE_MODE_T7_OTHER_B5_W1_2_ROW_GT128_PERF) {
            const uint64_t rowElemsB5 = outW * C;
            const uint64_t workElemsB5 = static_cast<uint64_t>(selectedWorkElems == 0U ? 1U : selectedWorkElems);
            if (rowElemsB5 > 0ULL && workElemsB5 >= rowElemsB5) {
                uint64_t rowsPerGroup = workElemsB5 / rowElemsB5;
                if (rowsPerGroup == 0ULL) rowsPerGroup = 1ULL;
                uint64_t rowGroups = (outH + rowsPerGroup - 1ULL) / rowsPerGroup;
                if (rowGroups == 0ULL) rowGroups = 1ULL;
                scheduledJobs = outN * rowGroups;
            } else {
                uint64_t chunksPerDepth = (C + workElemsB5 - 1ULL) / workElemsB5;
                if (chunksPerDepth == 0ULL) chunksPerDepth = 1ULL;
                scheduledJobs = outN * outH * outW * chunksPerDepth;
            }
        } else if (mode == BATCH_TO_SPACE_MODE_T9_BS4_ALIGNED_ROW_ASSEMBLE) {
            scheduledJobs = SelectT9Bs4RowAssembleJobs(outN, outH, outW, C, selectedWorkElems);
        } else if (mode == BATCH_TO_SPACE_MODE_T7_OUTW7_ALIGNED_ROWCONTRACT ||
                   mode == BATCH_TO_SPACE_MODE_T7_BS2_FP16_MEDIUM_NARROW_LANE) {
            uint64_t tc = scatterTileCols64 == 0U ? 1U : scatterTileCols64;
            uint64_t maxLaneCols = (outW + block_size - 1U) / block_size;
            uint64_t tilesPerLane = (maxLaneCols + tc - 1U) / tc;
            if (tilesPerLane == 0U) tilesPerLane = 1U;
            scheduledJobs = totalRows * block_size * tilesPerLane;
        } else if (mode == BATCH_TO_SPACE_MODE_T6_BS2_FP16_HIGHDEPTH_MEDIUM_ROW2) {

            scheduledJobs = totalRows;
        } else if (mode == BATCH_TO_SPACE_MODE_BS2_ROWGROUP_PACK ||
            mode == BATCH_TO_SPACE_MODE_BS2_ROWGROUP_PACK_16 ||
            mode == BATCH_TO_SPACE_MODE_BS2_ROWGROUP_PACK_8) {
            uint64_t rowsPerGroup = static_cast<uint64_t>(kT10RowGroupPackRows);
            if (mode == BATCH_TO_SPACE_MODE_BS2_ROWGROUP_PACK_16) {
                rowsPerGroup = static_cast<uint64_t>(kRowGroupPackRows16);
            } else if (mode == BATCH_TO_SPACE_MODE_BS2_ROWGROUP_PACK_8) {
                rowsPerGroup = static_cast<uint64_t>(kRowGroupPackRows8);
            }
            uint64_t rowGroups = (outH + rowsPerGroup - 1U) / rowsPerGroup;
            scheduledJobs = outN * rowGroups;
        } else if (mode == BATCH_TO_SPACE_MODE_HIGH_DEPTH_CONTIG_PACK || mode == BATCH_TO_SPACE_MODE_BS4_HIGH_DEPTH_CONTIG_PACK) {
            uint64_t tc = packTileCols64 == 0U ? 1U : packTileCols64;
            scheduledJobs = totalRows * ((outW + tc - 1U) / tc);
        } else if (mode == BATCH_TO_SPACE_MODE_LIGHTWEIGHT_GENERIC && (outputBytes64 >= (2ULL * 1024ULL * 1024ULL) && outputBytes64 < (4ULL * 1024ULL * 1024ULL))) {


            uint64_t lightUbElems = static_cast<uint64_t>(selectedWorkElems == 0U ? 1U : selectedWorkElems);
            uint64_t chunks = (totalElements + lightUbElems - 1U) / lightUbElems;
            scheduledJobs = lightGenericHighDepthPolicy ? std::max<uint64_t>(chunks, totalRows)
                                                        : std::max<uint64_t>(1U, chunks);
        } else {
            uint64_t tc = scatterTileCols64 == 0U ? 1U : scatterTileCols64;
            uint64_t maxLaneCols = (outW + block_size - 1U) / block_size;
            scheduledJobs = totalRows * block_size * ((maxLaneCols + tc - 1U) / tc);
        }
        if (scheduledJobs == 0U) scheduledJobs = 1U;
        if (scheduledJobs < block_dim) block_dim = static_cast<uint32_t>(scheduledJobs);
    }

    uint64_t elements_per_core = (totalElements + block_dim - 1U) / block_dim;
    elements_per_core = ((elements_per_core + elements_per_block - 1U) / elements_per_block) * elements_per_block;

    uint32_t MODE = mode;
    ASCENDC_TPL_SEL_PARAM(context, DT_X, MODE);

    BatchToSpaceTilingData *tiling = context->GetTilingData<BatchToSpaceTilingData>();
    tiling->N = N;
    tiling->H = H;
    tiling->W = W;
    tiling->C = C;
    tiling->outN = outN;
    tiling->outH = outH;
    tiling->outW = outW;
    tiling->blockSize = block_size;
    tiling->cropTop = crop_top;
    tiling->cropBottom = crop_bottom;
    tiling->cropLeft = crop_left;
    tiling->cropRight = crop_right;
    tiling->totalElements = totalElements;
    tiling->elementsPerCore = elements_per_core;
    tiling->mode = legacyMode;
    tiling->fastWorkElems = selectedWorkElems;

    context->SetBlockDim(block_dim);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) { return GRAPH_SUCCESS; }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) { return ge::GRAPH_SUCCESS; }
}

namespace ops {
    class BatchToSpace : public OpDef {
    public:
        explicit BatchToSpace(const char *name) : OpDef(name) {
            this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("crops").AttrType(REQUIRED).ListInt();
            this->Attr("block_size").AttrType(REQUIRED).Int();
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
        }
    };
    OP_ADD(BatchToSpace);
}
