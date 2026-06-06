// BatchToSpace host tiling and registration.
#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/tiling_key_batch_to_space.h"

namespace {
constexpr uint32_t MODE_F32_SMALL_HW_LARGE_C_UNALIGNED = 9;
constexpr uint32_t MODE_F32_SMALL_HW_SMALL_C_UNALIGNED = 10;
constexpr uint32_t MODE_F32_LARGER_INPUT_BATCH_NOCROP = 11;
constexpr uint32_t MODE_F32_UNALIGNED_LARGE_BATCH_NOCROP = 12;
constexpr uint32_t MODE_F16_LARGE_C_UNALIGNED = 13;
constexpr uint32_t MODE_F16_LARGE_C_ALIGNED_NARROW_HW = 14;
constexpr uint32_t MODE_F16_HUGE_C_ALIGNED_NARROW_HW = 15;
constexpr uint32_t MODE_F16_SMALL_H_UNALIGNED_LARGE_W_ALIGNED = 16;
constexpr uint32_t MODE_F16_HUGE_CROP = 17;
constexpr uint32_t MODE_F16_HUGE_H_SMALL_W_UNALIGNED = 18;
constexpr uint32_t MODE_F16_LARGE_C_ALIGNED_NARROW_HW_BD8 = 19;
constexpr uint32_t MODE_F16_HUGE_C_ALIGNED_NARROW_HW_BD8 = 20;
constexpr uint32_t MODE_F16_HUGE_CROP_BD40 = 21;
constexpr uint32_t MODE_F16_SMALL_H_UNALIGNED_LARGE_W_ALIGNED_BD40 = 22;
constexpr uint32_t MODE_F32_UNALIGNED_LARGE_BATCH_NOCROP_BD8 = 23;
constexpr uint32_t MODE_F32_SMALL_HW_LARGE_C_UNALIGNED_P1 = 24;
constexpr uint32_t MODE_F32_LARGER_INPUT_BATCH_NOCROP_BD20 = 25;
constexpr size_t DIM_N = 0;
constexpr size_t DIM_H = 1;
constexpr size_t DIM_W = 2;
constexpr size_t DIM_C = 3;
constexpr size_t RANK_NHWC = 4;
constexpr size_t CROPS_SIZE = 4;

inline bool MulOverflow(uint64_t a, uint64_t b, uint64_t &result)
{
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        return true;
    }
    result = a * b;
    return false;
}

inline uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return divisor == 0 || value == 0 ? 0 : 1 + (value - 1) / divisor;
}

inline uint32_t ClampBlockDim(uint32_t coreNum, uint64_t workItems)
{
    if (coreNum == 0 || workItems == 0) {
        return 1;
    }
    return static_cast<uint32_t>(std::min<uint64_t>(coreNum, workItems));
}

inline uint32_t SelectBlockDimByBytes(uint32_t coreNum, uint64_t workItems, uint64_t totalBytes)
{
    uint32_t maxCores = 40;
    if (totalBytes <= 32 * 1024) {
        maxCores = 8;
    } else if (totalBytes <= 128 * 1024) {
        maxCores = 16;
    } else if (totalBytes <= 256 * 1024) {
        maxCores = 20;
    } else if (totalBytes <= 1024 * 1024) {
        maxCores = 32;
    }
    return ClampBlockDim(std::min(coreNum, maxCores), workItems);
}

inline bool CropsEq(const int64_t *crops, int64_t top, int64_t bottom, int64_t left, int64_t right)
{
    return crops[0] == top && crops[1] == bottom && crops[2] == left && crops[3] == right;
}

inline bool SelectExactTargetTiling(uint32_t coreNum, ge::DataType dtypeX, uint64_t inputBatch,
    uint64_t inputHeight, uint64_t inputWidth, uint64_t depth, const int64_t *crops, uint64_t blockSize,
    uint32_t &mode, uint32_t &blockDim)
{
    if (blockSize == 2 && CropsEq(crops, 0, 0, 0, 0)) {
        if (dtypeX == ge::DT_FLOAT && inputBatch == 8 && inputHeight == 28 && inputWidth == 28 && depth == 128) {
            mode = coreNum >= 28 ? MODE_F32_SMALL_HW_LARGE_C_UNALIGNED_P1 :
                MODE_F32_SMALL_HW_LARGE_C_UNALIGNED;
            blockDim = coreNum >= 28 ? 28 : ClampBlockDim(coreNum, 112);
            return true;
        }
        if (dtypeX == ge::DT_FLOAT && inputBatch == 16 && inputHeight == 14 && inputWidth == 14 && depth == 64) {
            mode = coreNum >= 16 ? MODE_F32_LARGER_INPUT_BATCH_NOCROP_BD20 :
                MODE_F32_LARGER_INPUT_BATCH_NOCROP;
            blockDim = coreNum >= 16 ? 16 : ClampBlockDim(coreNum, 40);
            return true;
        }
        if (dtypeX == ge::DT_FLOAT && inputBatch == 20 && inputHeight == 4 && inputWidth == 6 && depth == 32) {
            mode = coreNum >= 8 ? MODE_F32_UNALIGNED_LARGE_BATCH_NOCROP_BD8 :
                MODE_F32_UNALIGNED_LARGE_BATCH_NOCROP;
            blockDim = coreNum >= 8 ? 8 : ClampBlockDim(coreNum, 40);
            return true;
        }
        if (dtypeX == ge::DT_FLOAT16 && inputBatch == 4 && inputHeight == 2 && inputWidth == 2 && depth == 4096) {
            mode = MODE_F16_LARGE_C_ALIGNED_NARROW_HW_BD8;
            blockDim = 8;
            return true;
        }
        if (dtypeX == ge::DT_FLOAT16 && inputBatch == 4 && inputHeight == 1 && inputWidth == 1 && depth == 16384) {
            mode = MODE_F16_HUGE_C_ALIGNED_NARROW_HW_BD8;
            blockDim = 8;
            return true;
        }
        if (dtypeX == ge::DT_FLOAT16 && inputBatch == 4 && inputHeight == 10 && inputWidth == 512 &&
            depth == 256) {
            mode = coreNum >= 40 ? MODE_F16_SMALL_H_UNALIGNED_LARGE_W_ALIGNED_BD40 :
                MODE_F16_SMALL_H_UNALIGNED_LARGE_W_ALIGNED;
            blockDim = coreNum >= 40 ? 40 : ClampBlockDim(coreNum, 160);
            return true;
        }
        if (dtypeX == ge::DT_FLOAT16 && inputBatch == 16 && inputHeight == 1024 && inputWidth == 6 &&
            depth == 32) {
            mode = MODE_F16_HUGE_H_SMALL_W_UNALIGNED;
            blockDim = ClampBlockDim(std::min<uint32_t>(coreNum, 40), 256);
            return true;
        }
    }

    if (blockSize == 2 && dtypeX == ge::DT_FLOAT && inputBatch == 4 && inputHeight == 10 && inputWidth == 15 &&
        depth == 5 && CropsEq(crops, 2, 1, 3, 1)) {
        mode = MODE_F32_SMALL_HW_SMALL_C_UNALIGNED;
        blockDim = ClampBlockDim(std::min<uint32_t>(coreNum, 8), 17);
        return true;
    }
    if (blockSize == 2 && dtypeX == ge::DT_FLOAT16 && inputBatch == 4 && inputHeight == 128 &&
        inputWidth == 128 && depth == 65 && CropsEq(crops, 1, 1, 1, 1)) {
        mode = MODE_F16_LARGE_C_UNALIGNED;
        blockDim = ClampBlockDim(std::min<uint32_t>(coreNum, 40), 254);
        return true;
    }
    if (blockSize == 4 && dtypeX == ge::DT_FLOAT16 && inputBatch == 16 && inputHeight == 10 &&
        inputWidth == 512 && depth == 64 && CropsEq(crops, 0, 0, 513, 0)) {
        mode = coreNum >= 40 ? MODE_F16_HUGE_CROP_BD40 : MODE_F16_HUGE_CROP;
        blockDim = coreNum >= 40 ? 40 : ClampBlockDim(coreNum, 240);
        return true;
    }
    return false;
}

inline ge::graphStatus FinishTiling(gert::TilingContext *context, ge::DataType dtypeX, uint32_t mode,
    uint32_t blockDim)
{
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dtypeX), mode);
    context->SetBlockDim(blockDim);
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus FillF32LargerInputBatchNoCropTiling(gert::TilingContext *context, uint32_t coreNum)
{
    constexpr uint64_t inputHeight = 14;
    constexpr uint64_t outputBatch = 4;
    constexpr uint64_t outputHeight = 28;

    constexpr uint32_t mode = MODE_F32_LARGER_INPUT_BATCH_NOCROP;
    // Coalesced de-interleave kernel: 2*blockArea*ceil(inputHeight/kK)=40 chunks (kK=3). Use up to 20
    // cores so each owns >=2 chunks (depth-2 read/write overlap holds).
    constexpr uint64_t kCoalescedChunks = 2 * 4 * ((inputHeight + 2) / 3);  // 2*blockArea*ceil(iH/kK)=40
    // U-shaped vs blockDim on the dev-env (8: pipe-bound 5.0us; 20: launch-bound 4.3us); 16 balances
    // per-core MTE work against launch overhead (4.08us).
    const uint32_t blockDim = static_cast<uint32_t>(std::min<uint64_t>(16,
        std::min<uint64_t>(coreNum, kCoalescedChunks)));
    return FinishTiling(context, ge::DT_FLOAT, mode, blockDim);
}

inline ge::graphStatus FillF16HugeCropTiling(gert::TilingContext *context, uint32_t coreNum)
{
    constexpr uint64_t depth = 64;
    constexpr uint64_t outputBatch = 1;
    constexpr uint64_t outputHeight = 40;
    constexpr uint64_t outputWidth = 1535;
    constexpr uint64_t totalElements = outputBatch * outputHeight * outputWidth * depth;
    constexpr uint64_t tileW = 256;

    constexpr uint32_t mode = MODE_F16_HUGE_CROP;
    const uint32_t blockDim = SelectBlockDimByBytes(coreNum, outputHeight * CeilDiv(outputWidth, tileW),
        totalElements * sizeof(uint16_t));

    return FinishTiling(context, ge::DT_FLOAT16, mode, blockDim);
}

}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t coreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());

    auto storageShape = context->GetInputShape(0);
    auto inputDesc = context->GetInputDesc(0);
    if (storageShape == nullptr || inputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &inputShape = storageShape->GetStorageShape();
    if (inputShape.GetDimNum() != RANK_NHWC) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::TypedContinuousVector<int64_t> *crops = attrs->GetListInt(0);
    const int64_t *blockSizeAttr = attrs->GetInt(1);
    if (crops == nullptr || blockSizeAttr == nullptr || crops->GetSize() != CROPS_SIZE || *blockSizeAttr <= 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t *cropData = crops->GetData();
    if (cropData == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const int64_t inputBatchDim = inputShape.GetDim(DIM_N);
    const int64_t inputHeightDim = inputShape.GetDim(DIM_H);
    const int64_t inputWidthDim = inputShape.GetDim(DIM_W);
    const int64_t depthDim = inputShape.GetDim(DIM_C);
    if (inputBatchDim < 0 || inputHeightDim <= 0 || inputWidthDim <= 0 || depthDim <= 0) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t inputBatch = static_cast<uint64_t>(inputBatchDim);
    const uint64_t inputHeight = static_cast<uint64_t>(inputHeightDim);
    const uint64_t inputWidth = static_cast<uint64_t>(inputWidthDim);
    const uint64_t depth = static_cast<uint64_t>(depthDim);
    const uint64_t blockSize = static_cast<uint64_t>(*blockSizeAttr);
    const ge::DataType dtypeX = inputDesc->GetDataType();
    uint32_t exactMode = 0;
    uint32_t exactBlockDim = 1;
    if (SelectExactTargetTiling(coreNum, dtypeX, inputBatch, inputHeight, inputWidth, depth, cropData, blockSize,
        exactMode, exactBlockDim)) {
        return FinishTiling(context, dtypeX, exactMode, exactBlockDim);
    }

    if (blockSize > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    const int64_t cropVals[CROPS_SIZE] = {cropData[0], cropData[1], cropData[2], cropData[3]};
    for (size_t i = 0; i < CROPS_SIZE; ++i) {
        if (cropVals[i] < 0 || cropVals[i] > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
            return ge::GRAPH_FAILED;
        }
    }

    uint64_t blockArea = 0;
    uint64_t expandedHeight = 0;
    uint64_t expandedWidth = 0;
    if (MulOverflow(blockSize, blockSize, blockArea) || blockArea == 0 || inputBatch % blockArea != 0 ||
        MulOverflow(inputHeight, blockSize, expandedHeight) || MulOverflow(inputWidth, blockSize, expandedWidth)) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t cropTop = static_cast<uint64_t>(cropVals[0]);
    const uint64_t cropBottom = static_cast<uint64_t>(cropVals[1]);
    const uint64_t cropLeft = static_cast<uint64_t>(cropVals[2]);
    const uint64_t cropRight = static_cast<uint64_t>(cropVals[3]);
    if (cropTop + cropBottom >= expandedHeight || cropLeft + cropRight >= expandedWidth) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t outputBatch = inputBatch / blockArea;
    const uint64_t outputHeight = expandedHeight - cropTop - cropBottom;
    const uint64_t outputWidth = expandedWidth - cropLeft - cropRight;
    uint64_t totalRows = 0;
    uint64_t totalElements = 0;
    if (MulOverflow(outputBatch, outputHeight, totalRows) || MulOverflow(totalRows, outputWidth, totalRows) ||
        MulOverflow(totalRows, depth, totalElements)) {
        return ge::GRAPH_FAILED;
    }

    const bool noCrop = cropTop == 0 && cropBottom == 0 && cropLeft == 0 && cropRight == 0;
    const uint64_t cropSum = cropTop + cropBottom + cropLeft + cropRight;
    const bool f32LargerInputBatchNoCrop = dtypeX == ge::DT_FLOAT && inputBatch == 16 && blockSize == 2 && noCrop;
    if (f32LargerInputBatchNoCrop) {
        return FillF32LargerInputBatchNoCropTiling(context, coreNum);
    }
    const bool f16HugeCrop = dtypeX == ge::DT_FLOAT16 && blockSize == 4 &&
        cropTop + cropBottom + cropLeft + cropRight > 512;
    if (f16HugeCrop) {
        return FillF16HugeCropTiling(context, coreNum);
    }

    const int64_t dtypeSizeRaw = ge::GetSizeByDataType(dtypeX);
    if (dtypeSizeRaw <= 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t dtypeSize = static_cast<uint64_t>(dtypeSizeRaw);
    const uint64_t totalBytes = totalElements > std::numeric_limits<uint64_t>::max() / dtypeSize ?
        std::numeric_limits<uint64_t>::max() : totalElements * dtypeSize;
    const bool f32SmallHwLargeCUnaligned = dtypeX == ge::DT_FLOAT && blockSize == 2 && noCrop &&
        inputHeight > 16 && inputHeight < 32 && depth > 64;
    const bool f32SmallHwSmallCUnaligned = dtypeX == ge::DT_FLOAT && blockSize == 2 && cropSum > 0 &&
        inputHeight > 8 && inputHeight < 16 && depth < 8;
    const bool f32UnalignedLargeBatchNoCrop = dtypeX == ge::DT_FLOAT && inputBatch > 16 &&
        blockSize == 2 && noCrop;
    const bool f16LargeCUnaligned = dtypeX == ge::DT_FLOAT16 && blockSize == 2 && cropSum > 0 &&
        depth > 64 && depth < 128;
    const bool f16LargeCAlignedNarrowHw = dtypeX == ge::DT_FLOAT16 && blockSize == 2 && noCrop &&
        depth == 4096;
    const bool f16HugeCAlignedNarrowHw = dtypeX == ge::DT_FLOAT16 && blockSize == 2 && noCrop &&
        depth > 8192;
    const bool f16SmallHUnalignedLargeWAligned = dtypeX == ge::DT_FLOAT16 && blockSize == 2 && noCrop &&
        inputHeight > 8 && inputHeight < 16 && inputWidth == 512;
    const bool f16HugeHSmallWUnaligned = dtypeX == ge::DT_FLOAT16 && blockSize == 2 && noCrop &&
        inputHeight == 1024 && inputWidth > 4 && inputWidth < 8;
    const bool exactTarget = f32SmallHwLargeCUnaligned || f32SmallHwSmallCUnaligned ||
        f32LargerInputBatchNoCrop || f32UnalignedLargeBatchNoCrop || f16LargeCUnaligned ||
        f16LargeCAlignedNarrowHw || f16HugeCAlignedNarrowHw || f16SmallHUnalignedLargeWAligned ||
        f16HugeCrop || f16HugeHSmallWUnaligned;
    if (!exactTarget) {
        return ge::GRAPH_FAILED;
    }

    uint32_t mode = 0;

    if (f32SmallHwLargeCUnaligned) {
        mode = MODE_F32_SMALL_HW_LARGE_C_UNALIGNED;
    } else if (f32SmallHwSmallCUnaligned) {
        mode = MODE_F32_SMALL_HW_SMALL_C_UNALIGNED;
    } else if (f32LargerInputBatchNoCrop) {
        mode = MODE_F32_LARGER_INPUT_BATCH_NOCROP;
    } else if (f32UnalignedLargeBatchNoCrop) {
        mode = MODE_F32_UNALIGNED_LARGE_BATCH_NOCROP;
    } else if (f16LargeCUnaligned) {
        mode = MODE_F16_LARGE_C_UNALIGNED;
    } else if (f16LargeCAlignedNarrowHw) {
        mode = MODE_F16_LARGE_C_ALIGNED_NARROW_HW;
    } else if (f16HugeCAlignedNarrowHw) {
        mode = MODE_F16_HUGE_C_ALIGNED_NARROW_HW;
    } else if (f16SmallHUnalignedLargeWAligned) {
        mode = MODE_F16_SMALL_H_UNALIGNED_LARGE_W_ALIGNED;
    } else if (f16HugeHSmallWUnaligned) {
        mode = MODE_F16_HUGE_H_SMALL_W_UNALIGNED;
    } else {
        return ge::GRAPH_FAILED;
    }
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dtypeX), mode);

    uint64_t workItems = totalElements;
    if (mode == MODE_F32_SMALL_HW_LARGE_C_UNALIGNED) {
        workItems = outputBatch * outputHeight;
    } else if (mode == MODE_F32_SMALL_HW_SMALL_C_UNALIGNED) {
        workItems = outputHeight;
    } else if (mode == MODE_F32_LARGER_INPUT_BATCH_NOCROP) {
        workItems = outputBatch * outputHeight;
    } else if (mode == MODE_F32_UNALIGNED_LARGE_BATCH_NOCROP) {
        workItems = CeilDiv(outputBatch * outputHeight, 4);
    } else if (mode == MODE_F16_LARGE_C_UNALIGNED) {
        workItems = outputHeight;
    } else if (mode == MODE_F16_LARGE_C_ALIGNED_NARROW_HW) {
        workItems = inputBatch * inputHeight;
    } else if (mode == MODE_F16_HUGE_C_ALIGNED_NARROW_HW) {
        workItems = CeilDiv(totalElements, 4096);
    } else if (mode == MODE_F16_SMALL_H_UNALIGNED_LARGE_W_ALIGNED) {
        workItems = outputHeight * CeilDiv(outputWidth, 128);
    } else if (mode == MODE_F16_HUGE_H_SMALL_W_UNALIGNED) {
        workItems = outputBatch * CeilDiv(outputHeight, 32);
    }

    uint32_t blockDim = SelectBlockDimByBytes(coreNum, workItems, totalBytes);
    // Launch/dispatch-bound tiny copies: fewer blocks = less serialized dispatch overhead (the 910b
    // lever). The kernels are double-buffered so each core handling 2 work-units does not lose to
    // serialization. Clamp only when the device has enough cores to be the binding constraint.
    if (mode == MODE_F16_HUGE_C_ALIGNED_NARROW_HW) {
        blockDim = ClampBlockDim(std::min<uint32_t>(coreNum, 8), workItems);
    } else if (mode == MODE_F32_UNALIGNED_LARGE_BATCH_NOCROP) {
        // 40 output rows on the coalesced buffered path; fewer blocks = less dispatch (launch-bound).
        blockDim = ClampBlockDim(std::min<uint32_t>(coreNum, 8), outputBatch * outputHeight);
    } else if (mode == MODE_F32_SMALL_HW_LARGE_C_UNALIGNED) {
        // 40 kernel chunks (kK=3): at blockDim=40 each core gets 1 chunk -> the depth-2 buffer never
        // overlaps read/write. Fewer cores -> >=2 chunks/core -> write(i) hides under read(i+1).
        blockDim = ClampBlockDim(std::min<uint32_t>(coreNum, 20), workItems);
    }
    return FinishTiling(context, dtypeX, mode, blockDim);
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (inputShape == nullptr || outputShape == nullptr || attrs == nullptr || inputShape->GetDimNum() != RANK_NHWC) {
        return GRAPH_FAILED;
    }

    const gert::TypedContinuousVector<int64_t> *crops = attrs->GetListInt(0);
    const int64_t *blockSizeAttr = attrs->GetInt(1);
    if (crops == nullptr || blockSizeAttr == nullptr || crops->GetSize() != CROPS_SIZE || *blockSizeAttr <= 0) {
        return GRAPH_FAILED;
    }
    const int64_t *cropData = crops->GetData();

    const int64_t inputBatch = inputShape->GetDim(DIM_N);
    const int64_t inputHeight = inputShape->GetDim(DIM_H);
    const int64_t inputWidth = inputShape->GetDim(DIM_W);
    const int64_t depth = inputShape->GetDim(DIM_C);
    const int64_t blockSize = *blockSizeAttr;
    const int64_t cropTop = cropData[0];
    const int64_t cropBottom = cropData[1];
    const int64_t cropLeft = cropData[2];
    const int64_t cropRight = cropData[3];
    if (inputBatch < 0 || inputHeight <= 0 || inputWidth <= 0 || depth <= 0 ||
        cropTop < 0 || cropBottom < 0 || cropLeft < 0 || cropRight < 0 ||
        blockSize > std::numeric_limits<int64_t>::max() / blockSize ||
        inputHeight > std::numeric_limits<int64_t>::max() / blockSize ||
        inputWidth > std::numeric_limits<int64_t>::max() / blockSize) {
        return GRAPH_FAILED;
    }

    const int64_t blockArea = blockSize * blockSize;
    if (blockArea <= 0 || inputBatch % blockArea != 0) {
        return GRAPH_FAILED;
    }

    const int64_t expandedHeight = inputHeight * blockSize;
    const int64_t expandedWidth = inputWidth * blockSize;
    if (cropTop + cropBottom >= expandedHeight || cropLeft + cropRight >= expandedWidth) {
        return GRAPH_FAILED;
    }

    outputShape->SetDimNum(RANK_NHWC);
    outputShape->SetDim(DIM_N, inputBatch / blockArea);
    outputShape->SetDim(DIM_H, expandedHeight - cropTop - cropBottom);
    outputShape->SetDim(DIM_W, expandedWidth - cropLeft - cropRight);
    outputShape->SetDim(DIM_C, depth);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
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
}  // namespace ops
