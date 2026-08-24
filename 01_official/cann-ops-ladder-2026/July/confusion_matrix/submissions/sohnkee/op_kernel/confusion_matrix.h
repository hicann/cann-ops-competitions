/*!
 * \file confusion_matrix.h
 * \brief ConfusionMatrix kernel implementation
 */

#ifndef CONFUSIONMATRIX_H
#define CONFUSIONMATRIX_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "confusion_matrix_tiling_data.h"
#include "confusion_matrix_tiling_key.h"

namespace NsConfusionMatrix {

using namespace AscendC;

constexpr uint32_t CONFUSION_MATRIX_ZERO_TILE_ELEMS = 16384;
constexpr uint64_t CONFUSION_MATRIX_BULK_CLEAR_MIN_ELEMS = 1024;
constexpr uint64_t CONFUSION_MATRIX_TINY_FASTPATH_MAX_SAMPLES = 64;
constexpr uint64_t CONFUSION_MATRIX_TINY_FASTPATH_MAX_OUTPUT = 256;
constexpr uint64_t CONFUSION_MATRIX_SMALL_ROW_SPLIT_MIN_SAMPLES = 512;
constexpr uint64_t CONFUSION_MATRIX_SMALL_ROW_SPLIT_MAX_CLASSES = 16;
constexpr uint32_t CONFUSION_MATRIX_C3_VECTOR_TILE_ELEMS = 3072;
constexpr uint32_t CONFUSION_MATRIX_C3_VECTOR_DATA_BUFFER_BYTES =
    CONFUSION_MATRIX_C3_VECTOR_TILE_ELEMS * sizeof(int32_t) * 2;
constexpr uint32_t CONFUSION_MATRIX_C3_VECTOR_MASK_BYTES = CONFUSION_MATRIX_C3_VECTOR_TILE_ELEMS / 8;
constexpr uint32_t CONFUSION_MATRIX_C3_VECTOR_MASK_BUFFER_BYTES = CONFUSION_MATRIX_C3_VECTOR_MASK_BYTES * 3;
constexpr uint32_t CONFUSION_MATRIX_C3_VECTOR_WORK_BUFFER_BYTES =
    CONFUSION_MATRIX_C3_VECTOR_DATA_BUFFER_BYTES + CONFUSION_MATRIX_C3_VECTOR_MASK_BUFFER_BYTES;
constexpr uint32_t CONFUSION_MATRIX_SPARSE_CAPACITY = 10240;
constexpr uint32_t CONFUSION_MATRIX_SPARSE_COUNT_LOCAL_OFFSET = CONFUSION_MATRIX_SPARSE_CAPACITY;
constexpr uint32_t CONFUSION_MATRIX_SPARSE_OCCUPIED_LOCAL_OFFSET = CONFUSION_MATRIX_SPARSE_CAPACITY * 2;
constexpr uint32_t CONFUSION_MATRIX_BULK_BUFFER_BYTES =
    CONFUSION_MATRIX_SPARSE_CAPACITY * sizeof(int32_t) * 3;
constexpr uint32_t CONFUSION_MATRIX_SPARSE_FILTER_TILE_ELEMS = 5120;
constexpr uint32_t CONFUSION_MATRIX_SPARSE_FILTER_LABEL_LOCAL_OFFSET = CONFUSION_MATRIX_SPARSE_CAPACITY * 3;
constexpr uint32_t CONFUSION_MATRIX_SPARSE_FILTER_PREDICTION_LOCAL_OFFSET =
    CONFUSION_MATRIX_SPARSE_FILTER_LABEL_LOCAL_OFFSET + CONFUSION_MATRIX_SPARSE_FILTER_TILE_ELEMS;
constexpr uint32_t CONFUSION_MATRIX_SPARSE_FILTER_COMPACT_LABEL_LOCAL_OFFSET =
    CONFUSION_MATRIX_SPARSE_FILTER_PREDICTION_LOCAL_OFFSET + CONFUSION_MATRIX_SPARSE_FILTER_TILE_ELEMS;
constexpr uint32_t CONFUSION_MATRIX_SPARSE_FILTER_MASK_BYTES = CONFUSION_MATRIX_SPARSE_FILTER_TILE_ELEMS / 8;
constexpr uint32_t CONFUSION_MATRIX_SPARSE_FILTER_MASK_LOCAL_OFFSET_BYTES =
    (CONFUSION_MATRIX_SPARSE_FILTER_COMPACT_LABEL_LOCAL_OFFSET + CONFUSION_MATRIX_SPARSE_FILTER_TILE_ELEMS) *
    sizeof(int32_t);
constexpr uint32_t CONFUSION_MATRIX_SPARSE_FILTER_STORAGE_BYTES =
    CONFUSION_MATRIX_SPARSE_FILTER_MASK_LOCAL_OFFSET_BYTES + CONFUSION_MATRIX_SPARSE_FILTER_MASK_BYTES * 3;
constexpr uint32_t CONFUSION_MATRIX_SPARSE_FILTER_BUFFER_BYTES = 184U * 1024U;
constexpr uint64_t CONFUSION_MATRIX_SPARSE_NORMAL_L2_BUDGET_BYTES = 192UL * 1024UL * 1024UL;
constexpr uint64_t CONFUSION_MATRIX_UINT32_OFFSET_MAX = 0xFFFFFFFFULL;

static_assert((CONFUSION_MATRIX_SPARSE_FILTER_MASK_LOCAL_OFFSET_BYTES & 31U) == 0U,
    "sparse vector-filter mask storage must be 32-byte aligned");
static_assert(CONFUSION_MATRIX_SPARSE_FILTER_STORAGE_BYTES <= CONFUSION_MATRIX_SPARSE_FILTER_BUFFER_BYTES,
    "sparse vector-filter storage exceeds the Vector Core UB budget");

__aicore__ inline void ProcessTinyInt32NoWeight(
    GM_ADDR labels,
    GM_ADDR predictions,
    GM_ADDR y,
    const ConfusionMatrixTilingData* tilingData)
{
    GlobalTensor<int32_t> labelsGm;
    GlobalTensor<int32_t> predictionsGm;
    GlobalTensor<int32_t> outputGm;
    labelsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(labels));
    predictionsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(predictions));
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(y));

    const uint32_t totalNum = static_cast<uint32_t>(tilingData->totalNum);
    const uint32_t numClasses = static_cast<uint32_t>(tilingData->numClasses);
    const uint32_t outputNum = numClasses * numClasses;

    for (uint32_t i = 0; i < outputNum; ++i) {
        outputGm.SetValue(i, static_cast<int32_t>(0));
    }

    if (totalNum <= 6) {
#define CM_TINY_I32_NOWEIGHT_STEP(IDX) \
        if (totalNum > IDX) { \
            const uint32_t label = static_cast<uint32_t>(labelsGm.GetValue(IDX)); \
            const uint32_t prediction = static_cast<uint32_t>(predictionsGm.GetValue(IDX)); \
            const uint32_t outOffset = label * numClasses + prediction; \
            const int32_t oldValue = outputGm.GetValue(outOffset); \
            outputGm.SetValue(outOffset, oldValue + 1); \
        }
        CM_TINY_I32_NOWEIGHT_STEP(0);
        CM_TINY_I32_NOWEIGHT_STEP(1);
        CM_TINY_I32_NOWEIGHT_STEP(2);
        CM_TINY_I32_NOWEIGHT_STEP(3);
        CM_TINY_I32_NOWEIGHT_STEP(4);
        CM_TINY_I32_NOWEIGHT_STEP(5);
#undef CM_TINY_I32_NOWEIGHT_STEP
        return;
    }

    for (uint32_t i = 0; i < totalNum; ++i) {
        const uint32_t label = static_cast<uint32_t>(labelsGm.GetValue(i));
        const uint32_t prediction = static_cast<uint32_t>(predictionsGm.GetValue(i));
        const uint32_t outOffset = label * numClasses + prediction;
        const int32_t oldValue = outputGm.GetValue(outOffset);
        outputGm.SetValue(outOffset, oldValue + 1);
    }
}

__aicore__ inline void ProcessTinyInt32WeightInt32(
    GM_ADDR labels,
    GM_ADDR predictions,
    GM_ADDR weights,
    GM_ADDR y,
    const ConfusionMatrixTilingData* tilingData)
{
    GlobalTensor<int32_t> labelsGm;
    GlobalTensor<int32_t> predictionsGm;
    GlobalTensor<int32_t> weightsGm;
    GlobalTensor<int32_t> outputGm;
    labelsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(labels));
    predictionsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(predictions));
    weightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(weights));
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(y));

    const uint32_t totalNum = static_cast<uint32_t>(tilingData->totalNum);
    const uint32_t numClasses = static_cast<uint32_t>(tilingData->numClasses);
    const uint32_t outputNum = numClasses * numClasses;

    for (uint32_t i = 0; i < outputNum; ++i) {
        outputGm.SetValue(i, static_cast<int32_t>(0));
    }

    if (totalNum <= 6) {
#define CM_TINY_I32_WEIGHT_STEP(IDX) \
        if (totalNum > IDX) { \
            const uint32_t label = static_cast<uint32_t>(labelsGm.GetValue(IDX)); \
            const uint32_t prediction = static_cast<uint32_t>(predictionsGm.GetValue(IDX)); \
            const uint32_t outOffset = label * numClasses + prediction; \
            const int32_t oldValue = outputGm.GetValue(outOffset); \
            outputGm.SetValue(outOffset, oldValue + weightsGm.GetValue(IDX)); \
        }
        CM_TINY_I32_WEIGHT_STEP(0);
        CM_TINY_I32_WEIGHT_STEP(1);
        CM_TINY_I32_WEIGHT_STEP(2);
        CM_TINY_I32_WEIGHT_STEP(3);
        CM_TINY_I32_WEIGHT_STEP(4);
        CM_TINY_I32_WEIGHT_STEP(5);
#undef CM_TINY_I32_WEIGHT_STEP
        return;
    }

    for (uint32_t i = 0; i < totalNum; ++i) {
        const uint32_t label = static_cast<uint32_t>(labelsGm.GetValue(i));
        const uint32_t prediction = static_cast<uint32_t>(predictionsGm.GetValue(i));
        const uint32_t outOffset = label * numClasses + prediction;
        const int32_t oldValue = outputGm.GetValue(outOffset);
        outputGm.SetValue(outOffset, oldValue + weightsGm.GetValue(i));
    }
}

__aicore__ inline void ProcessTinyInt64WeightFloat(
    GM_ADDR labels,
    GM_ADDR predictions,
    GM_ADDR weights,
    GM_ADDR y,
    const ConfusionMatrixTilingData* tilingData)
{
    GlobalTensor<int64_t> labelsGm;
    GlobalTensor<int64_t> predictionsGm;
    GlobalTensor<float> weightsGm;
    GlobalTensor<float> outputGm;
    labelsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(labels));
    predictionsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(predictions));
    weightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));

    const uint64_t totalNum = static_cast<uint64_t>(tilingData->totalNum);
    const uint64_t numClasses = static_cast<uint64_t>(tilingData->numClasses);
    const uint64_t outputNum = numClasses * numClasses;

    for (uint64_t i = 0; i < outputNum; ++i) {
        outputGm.SetValue(i, 0.0f);
    }

    if (totalNum <= 6) {
#define CM_TINY_I64_FLOAT_STEP(IDX) \
        if (totalNum > IDX) { \
            const uint64_t label = static_cast<uint64_t>(labelsGm.GetValue(IDX)); \
            const uint64_t prediction = static_cast<uint64_t>(predictionsGm.GetValue(IDX)); \
            const uint64_t outOffset = label * numClasses + prediction; \
            const float oldValue = outputGm.GetValue(outOffset); \
            outputGm.SetValue(outOffset, oldValue + weightsGm.GetValue(IDX)); \
        }
        CM_TINY_I64_FLOAT_STEP(0);
        CM_TINY_I64_FLOAT_STEP(1);
        CM_TINY_I64_FLOAT_STEP(2);
        CM_TINY_I64_FLOAT_STEP(3);
        CM_TINY_I64_FLOAT_STEP(4);
        CM_TINY_I64_FLOAT_STEP(5);
#undef CM_TINY_I64_FLOAT_STEP
        return;
    }

    for (uint64_t i = 0; i < totalNum; ++i) {
        const uint64_t label = static_cast<uint64_t>(labelsGm.GetValue(i));
        const uint64_t prediction = static_cast<uint64_t>(predictionsGm.GetValue(i));
        const uint64_t outOffset = label * numClasses + prediction;
        const float oldValue = outputGm.GetValue(outOffset);
        outputGm.SetValue(outOffset, oldValue + weightsGm.GetValue(i));
    }
}

template <bool useMte2TailPadding>
__aicore__ inline void ProcessC3VectorHistogram(
    GM_ADDR labels,
    GM_ADDR predictions,
    GM_ADDR y,
    const ConfusionMatrixTilingData* tilingData);

class ConfusionMatrix {
public:
    __aicore__ inline ConfusionMatrix() {};

    __aicore__ inline void Init(
        GM_ADDR labels,
        GM_ADDR predictions,
        GM_ADDR weights,
        GM_ADDR y,
        const ConfusionMatrixTilingData* tilingData);
    __aicore__ inline void Process();
#if TILING_KEY_VAR == CONFUSIONMATRIX_TPL_SCH_MODE_6
    __aicore__ inline void ProcessAtomicSparse();
#endif

private:
    __aicore__ inline int64_t ReadLabel(uint64_t offset);
    __aicore__ inline int64_t ReadPrediction(uint64_t offset);
    __aicore__ inline int32_t ReadWeightAsInt32(uint64_t offset);
    __aicore__ inline int64_t ReadWeightAsInt64(uint64_t offset);
    __aicore__ inline float ReadWeightAsFloat(uint64_t offset);
    __aicore__ inline void ClearOutput();
    __aicore__ inline void ClearOutputRange(uint64_t start, uint64_t end);
    __aicore__ inline void ClearOutputRangeInt32(uint64_t start, uint64_t end);
    __aicore__ inline void ClearOutputRangeFloat(uint64_t start, uint64_t end);
    __aicore__ inline void AddToOutput(uint64_t offset, uint64_t sampleIdx);
    __aicore__ inline void ProcessSingleCore();
    __aicore__ inline void ProcessSplitOutput();
    __aicore__ inline bool CanUseSmallOutputRowSplit(uint64_t totalNum) const;
    __aicore__ inline void ProcessSmallOutputRowSplit(uint64_t totalNum);
    __aicore__ inline bool CanUseSparseSplitInt32(uint64_t totalNum) const;
    __aicore__ inline void ProcessSplitOutputSparseInt32(
        uint64_t rowStart,
        uint64_t rowEnd,
        uint64_t classCount,
        uint64_t totalNum);

private:
    GlobalTensor<int32_t> labelsI32Gm_;
    GlobalTensor<int64_t> labelsI64Gm_;
    GlobalTensor<int32_t> predictionsI32Gm_;
    GlobalTensor<int64_t> predictionsI64Gm_;
    GlobalTensor<int32_t> weightsI32Gm_;
    GlobalTensor<int64_t> weightsI64Gm_;
    GlobalTensor<float> weightsF32Gm_;
    GlobalTensor<int32_t> outputI32Gm_;
    GlobalTensor<int32_t> outputI32BypassGm_;
    GlobalTensor<int64_t> outputI64Gm_;
    GlobalTensor<float> outputF32Gm_;
    TPipe pipe_;
    TBuf<TPosition::VECOUT> zeroBuf_;

    int64_t totalNum_ = 0;
    int64_t blockFactor_ = 1;
    int64_t numClasses_ = 0;
    int64_t outputNum_ = 0;
    int32_t labelsDtype_ = CONFUSION_MATRIX_DTYPE_INT32;
    int32_t predictionsDtype_ = CONFUSION_MATRIX_DTYPE_INT32;
    int32_t weightsDtype_ = CONFUSION_MATRIX_DTYPE_INT32;
    int32_t outputDtype_ = CONFUSION_MATRIX_DTYPE_INT32;
    int32_t hasWeights_ = 0;
    int32_t reserved_ = 0;
    int32_t useBulkClear_ = 0;
    uint64_t sparseNormalStart_ = 0;
};

__aicore__ inline void ConfusionMatrix::Init(
    GM_ADDR labels,
    GM_ADDR predictions,
    GM_ADDR weights,
    GM_ADDR y,
    const ConfusionMatrixTilingData* tilingData)
{
    labelsI32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(labels));
    labelsI64Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(labels));
    predictionsI32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(predictions));
    predictionsI64Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(predictions));
    outputI32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(y));
    outputI32BypassGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(y));
    outputI64Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(y));
    outputF32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));

    totalNum_ = tilingData->totalNum;
    blockFactor_ = tilingData->blockFactor;
    numClasses_ = tilingData->numClasses;
    outputNum_ = numClasses_ * numClasses_;
    labelsDtype_ = tilingData->labelsDtype;
    predictionsDtype_ = tilingData->predictionsDtype;
    weightsDtype_ = tilingData->weightsDtype;
    outputDtype_ = tilingData->outputDtype;
    hasWeights_ = tilingData->hasWeights;
    reserved_ = tilingData->reserved;
    if ((reserved_ & CONFUSION_MATRIX_RESERVED_VECTOR_FILTER_SPARSE) != 0) {
        outputI32BypassGm_.SetL2CacheHint<CacheRwMode::WRITE>(CacheMode::CACHE_MODE_DISABLE);
    }
    if ((outputDtype_ == CONFUSION_MATRIX_DTYPE_INT32 || outputDtype_ == CONFUSION_MATRIX_DTYPE_FLOAT32) &&
        outputNum_ >= static_cast<int64_t>(CONFUSION_MATRIX_BULK_CLEAR_MIN_ELEMS)) {
        useBulkClear_ = 1;
        const uint32_t bufferBytes =
            (reserved_ & CONFUSION_MATRIX_RESERVED_VECTOR_FILTER_SPARSE) != 0 ?
            CONFUSION_MATRIX_SPARSE_FILTER_BUFFER_BYTES : CONFUSION_MATRIX_BULK_BUFFER_BYTES;
        pipe_.InitBuffer(zeroBuf_, bufferBytes);
    }

    if (hasWeights_ != 0) {
        weightsI32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(weights));
        weightsI64Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(weights));
        weightsF32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
    }
}

__aicore__ inline int64_t ConfusionMatrix::ReadLabel(uint64_t offset)
{
    if (labelsDtype_ == CONFUSION_MATRIX_DTYPE_INT64) {
        return labelsI64Gm_.GetValue(offset);
    }
    return static_cast<int64_t>(labelsI32Gm_.GetValue(offset));
}

__aicore__ inline int64_t ConfusionMatrix::ReadPrediction(uint64_t offset)
{
    if (predictionsDtype_ == CONFUSION_MATRIX_DTYPE_INT64) {
        return predictionsI64Gm_.GetValue(offset);
    }
    return static_cast<int64_t>(predictionsI32Gm_.GetValue(offset));
}

__aicore__ inline int32_t ConfusionMatrix::ReadWeightAsInt32(uint64_t offset)
{
    if (hasWeights_ == 0) {
        return 1;
    }
    if (weightsDtype_ == CONFUSION_MATRIX_DTYPE_INT64) {
        return static_cast<int32_t>(weightsI64Gm_.GetValue(offset));
    }
    if (weightsDtype_ == CONFUSION_MATRIX_DTYPE_FLOAT32) {
        return static_cast<int32_t>(weightsF32Gm_.GetValue(offset));
    }
    return weightsI32Gm_.GetValue(offset);
}

__aicore__ inline int64_t ConfusionMatrix::ReadWeightAsInt64(uint64_t offset)
{
    if (hasWeights_ == 0) {
        return 1;
    }
    if (weightsDtype_ == CONFUSION_MATRIX_DTYPE_INT64) {
        return weightsI64Gm_.GetValue(offset);
    }
    if (weightsDtype_ == CONFUSION_MATRIX_DTYPE_FLOAT32) {
        return static_cast<int64_t>(weightsF32Gm_.GetValue(offset));
    }
    return static_cast<int64_t>(weightsI32Gm_.GetValue(offset));
}

__aicore__ inline float ConfusionMatrix::ReadWeightAsFloat(uint64_t offset)
{
    if (hasWeights_ == 0) {
        return 1.0f;
    }
    if (weightsDtype_ == CONFUSION_MATRIX_DTYPE_INT64) {
        return static_cast<float>(weightsI64Gm_.GetValue(offset));
    }
    if (weightsDtype_ == CONFUSION_MATRIX_DTYPE_FLOAT32) {
        return weightsF32Gm_.GetValue(offset);
    }
    return static_cast<float>(weightsI32Gm_.GetValue(offset));
}

__aicore__ inline void ConfusionMatrix::ClearOutput()
{
    ClearOutputRange(0, static_cast<uint64_t>(outputNum_));
}

__aicore__ inline void ConfusionMatrix::ClearOutputRange(uint64_t start, uint64_t end)
{
    const uint64_t outputNum = static_cast<uint64_t>(outputNum_);
    if (start >= outputNum) {
        return;
    }
    if (end > outputNum) {
        end = outputNum;
    }
    if (outputDtype_ == CONFUSION_MATRIX_DTYPE_FLOAT32) {
        if (useBulkClear_ != 0 && end - start >= CONFUSION_MATRIX_BULK_CLEAR_MIN_ELEMS) {
            ClearOutputRangeFloat(start, end);
            return;
        }
        for (uint64_t i = start; i < end; ++i) {
            outputF32Gm_.SetValue(i, 0.0f);
        }
    } else if (outputDtype_ == CONFUSION_MATRIX_DTYPE_INT64) {
        for (uint64_t i = start; i < end; ++i) {
            outputI64Gm_.SetValue(i, static_cast<int64_t>(0));
        }
    } else {
        if (useBulkClear_ != 0 && end - start >= CONFUSION_MATRIX_BULK_CLEAR_MIN_ELEMS) {
            ClearOutputRangeInt32(start, end);
            return;
        }
        for (uint64_t i = start; i < end; ++i) {
            outputI32Gm_.SetValue(i, static_cast<int32_t>(0));
        }
    }
}

__aicore__ inline void ConfusionMatrix::ClearOutputRangeInt32(uint64_t start, uint64_t end)
{
    const bool useMixedL2 =
        (reserved_ & CONFUSION_MATRIX_RESERVED_VECTOR_FILTER_SPARSE) != 0 &&
        sparseNormalStart_ > start && sparseNormalStart_ < end;
    uint64_t offset = start;
    while (offset < end && (offset & 7UL) != 0) {
        if (useMixedL2 && offset < sparseNormalStart_) {
            outputI32BypassGm_.SetValue(offset, static_cast<int32_t>(0));
        } else {
            outputI32Gm_.SetValue(offset, static_cast<int32_t>(0));
        }
        ++offset;
    }

    const uint64_t alignedEnd = end & (~7UL);
    if (offset >= alignedEnd) {
        while (offset < end) {
            outputI32Gm_.SetValue(offset, static_cast<int32_t>(0));
            ++offset;
        }
        return;
    }

    LocalTensor<int32_t> zeroLocal = zeroBuf_.Get<int32_t>();
    const uint32_t clearTileElems =
        (reserved_ & CONFUSION_MATRIX_RESERVED_VECTOR_FILTER_SPARSE) != 0 ?
        CONFUSION_MATRIX_SPARSE_FILTER_BUFFER_BYTES / sizeof(int32_t) :
        CONFUSION_MATRIX_ZERO_TILE_ELEMS;
    Duplicate(zeroLocal, static_cast<int32_t>(0), clearTileElems);
    TEventID eventIdVToMte3 = static_cast<TEventID>(0);
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    // Keep the long case-8 clear copies on 512-byte GM boundaries.  The row
    // partition only guarantees 32-byte alignment, so issue one short prefix
    // before the 184 KiB tiles.  Other cases keep their original clear path.
    if ((reserved_ & CONFUSION_MATRIX_RESERVED_VECTOR_FILTER_SPARSE) != 0) {
        const uint32_t offsetIn512Bytes = static_cast<uint32_t>(offset & 127UL);
        if (offsetIn512Bytes != 0 && offset < alignedEnd) {
            uint64_t prefixElems = 128U - offsetIn512Bytes;
            const uint64_t remainingElems = alignedEnd - offset;
            if (prefixElems > remainingElems) {
                prefixElems = remainingElems;
            }
            if (useMixedL2 && offset < sparseNormalStart_) {
                const uint64_t bypassRemaining = sparseNormalStart_ - offset;
                if (prefixElems > bypassRemaining) {
                    prefixElems = bypassRemaining;
                }
                DataCopy(outputI32BypassGm_[offset], zeroLocal, static_cast<uint32_t>(prefixElems));
            } else {
                DataCopy(outputI32Gm_[offset], zeroLocal, static_cast<uint32_t>(prefixElems));
            }
            offset += prefixElems;
        }
    }

    while (offset < alignedEnd) {
        uint64_t segmentEnd = alignedEnd;
        if (useMixedL2 && offset < sparseNormalStart_) {
            segmentEnd = sparseNormalStart_;
        }
        uint64_t elems = segmentEnd - offset;
        if (elems > clearTileElems) {
            elems = clearTileElems;
        }
        if (useMixedL2 && offset < sparseNormalStart_) {
            DataCopy(outputI32BypassGm_[offset], zeroLocal, static_cast<uint32_t>(elems));
        } else {
            DataCopy(outputI32Gm_[offset], zeroLocal, static_cast<uint32_t>(elems));
        }
        offset += elems;
    }
    PipeBarrier<PIPE_ALL>();

    while (offset < end) {
        if (useMixedL2 && offset < sparseNormalStart_) {
            outputI32BypassGm_.SetValue(offset, static_cast<int32_t>(0));
        } else {
            outputI32Gm_.SetValue(offset, static_cast<int32_t>(0));
        }
        ++offset;
    }
}

__aicore__ inline void ConfusionMatrix::ClearOutputRangeFloat(uint64_t start, uint64_t end)
{
    uint64_t offset = start;
    while (offset < end && (offset & 7UL) != 0) {
        outputF32Gm_.SetValue(offset, 0.0f);
        ++offset;
    }

    const uint64_t alignedEnd = end & (~7UL);
    if (offset >= alignedEnd) {
        while (offset < end) {
            outputF32Gm_.SetValue(offset, 0.0f);
            ++offset;
        }
        return;
    }

    LocalTensor<float> zeroLocal = zeroBuf_.Get<float>();
    Duplicate(zeroLocal, 0.0f, CONFUSION_MATRIX_ZERO_TILE_ELEMS);
    TEventID eventIdVToMte3 = static_cast<TEventID>(0);
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    while (offset < alignedEnd) {
        uint64_t elems = alignedEnd - offset;
        if (elems > CONFUSION_MATRIX_ZERO_TILE_ELEMS) {
            elems = CONFUSION_MATRIX_ZERO_TILE_ELEMS;
        }
        DataCopy(outputF32Gm_[offset], zeroLocal, static_cast<uint32_t>(elems));
        offset += elems;
    }
    PipeBarrier<PIPE_ALL>();

    while (offset < end) {
        outputF32Gm_.SetValue(offset, 0.0f);
        ++offset;
    }
}

__aicore__ inline void ConfusionMatrix::AddToOutput(uint64_t offset, uint64_t sampleIdx)
{
    if (outputDtype_ == CONFUSION_MATRIX_DTYPE_FLOAT32) {
        const float oldValue = outputF32Gm_.GetValue(offset);
        outputF32Gm_.SetValue(offset, oldValue + ReadWeightAsFloat(sampleIdx));
    } else if (outputDtype_ == CONFUSION_MATRIX_DTYPE_INT64) {
        const int64_t oldValue = outputI64Gm_.GetValue(offset);
        outputI64Gm_.SetValue(offset, oldValue + ReadWeightAsInt64(sampleIdx));
    } else {
        const int32_t oldValue = outputI32Gm_.GetValue(offset);
        outputI32Gm_.SetValue(offset, oldValue + ReadWeightAsInt32(sampleIdx));
    }
}

__aicore__ inline void ConfusionMatrix::Process()
{
    if ((reserved_ & CONFUSION_MATRIX_RESERVED_SPLIT_OUTPUT) != 0) {
        ProcessSplitOutput();
        return;
    }
    ProcessSingleCore();
}

#if TILING_KEY_VAR == CONFUSIONMATRIX_TPL_SCH_MODE_6
__aicore__ inline void ConfusionMatrix::ProcessAtomicSparse()
{
    const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
    const uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
    const uint64_t classCount = static_cast<uint64_t>(numClasses_);
    const uint64_t rowsPerCore = static_cast<uint64_t>(blockFactor_);
    const uint64_t rowStart = blockIdx * rowsPerCore;
    uint64_t rowEnd = rowStart + rowsPerCore;
    if (rowEnd > classCount) {
        rowEnd = classCount;
    }

    if (rowStart < classCount) {
        const uint64_t clearStart = rowStart * classCount;
        const uint64_t clearEnd = rowEnd * classCount;
        sparseNormalStart_ = clearStart;
        uint64_t normalTailElems =
            CONFUSION_MATRIX_SPARSE_NORMAL_L2_BUDGET_BYTES / sizeof(int32_t) / blockNum;
        normalTailElems &= ~127UL;
        if (normalTailElems != 0 && clearEnd - clearStart > normalTailElems) {
            sparseNormalStart_ = (clearEnd - normalTailElems + 127UL) & ~127UL;
            if (sparseNormalStart_ > clearEnd) {
                sparseNormalStart_ = clearEnd;
            }
        }
        ClearOutputRange(clearStart, clearEnd);
    }

    SyncAll();

    LocalTensor<int32_t> atomicLocal = zeroBuf_.Get<int32_t>();
    for (uint32_t lane = 0; lane < 8; ++lane) {
        atomicLocal.SetValue(lane * 8 + lane, static_cast<int32_t>(1));
    }
    const TEventID atomicDataReadyEvent = static_cast<TEventID>(0);
    SetFlag<HardEvent::S_MTE3>(atomicDataReadyEvent);
    WaitFlag<HardEvent::S_MTE3>(atomicDataReadyEvent);

    const uint64_t totalNum = static_cast<uint64_t>(totalNum_);
    const uint64_t samplesPerCore = (totalNum + blockNum - 1) / blockNum;
    const uint64_t sampleStart = blockIdx * samplesPerCore;
    uint64_t sampleEnd = sampleStart + samplesPerCore;
    if (sampleEnd > totalNum) {
        sampleEnd = totalNum;
    }

    SetAtomicAdd<int32_t>();
    for (uint64_t i = sampleStart; i < sampleEnd; ++i) {
        const int32_t labelValue = labelsI32Gm_.GetValue(i);
        const int32_t predictionValue = predictionsI32Gm_.GetValue(i);
        if (labelValue < 0 || predictionValue < 0 ||
            labelValue >= numClasses_ || predictionValue >= numClasses_) {
            continue;
        }
        const uint64_t outOffset = static_cast<uint64_t>(labelValue) * classCount +
            static_cast<uint64_t>(predictionValue);
        const uint32_t lane = static_cast<uint32_t>(outOffset & 7UL);
        DataCopy(outputI32Gm_[outOffset - lane], atomicLocal[lane * 8], 8);
    }
    SetAtomicNone();
}
#endif

__aicore__ inline void ConfusionMatrix::ProcessSingleCore()
{
    const uint64_t totalNum = static_cast<uint64_t>(totalNum_);
    if (CanUseSmallOutputRowSplit(totalNum)) {
        ProcessSmallOutputRowSplit(totalNum);
        return;
    }

    if (GetBlockIdx() != 0) {
        return;
    }
    ClearOutput();

    const int64_t numClasses = numClasses_;
    if (labelsDtype_ == CONFUSION_MATRIX_DTYPE_INT32 &&
        predictionsDtype_ == CONFUSION_MATRIX_DTYPE_INT32 &&
        outputDtype_ == CONFUSION_MATRIX_DTYPE_INT32 &&
        hasWeights_ == 0) {
        if (numClasses == 3 && totalNum >= 512) {
            int32_t h0 = 0;
            int32_t h1 = 0;
            int32_t h2 = 0;
            int32_t h3 = 0;
            int32_t h4 = 0;
            int32_t h5 = 0;
            int32_t h6 = 0;
            int32_t h7 = 0;
            int32_t h8 = 0;
            uint64_t i = 0;
#define CM_C3_HIST9_STEP(IDX) \
            do { \
                const uint64_t label = static_cast<uint64_t>(labelsI32Gm_.GetValue(IDX)); \
                const uint64_t prediction = static_cast<uint64_t>(predictionsI32Gm_.GetValue(IDX)); \
                switch (label * 3 + prediction) { \
                    case 0: ++h0; break; \
                    case 1: ++h1; break; \
                    case 2: ++h2; break; \
                    case 3: ++h3; break; \
                    case 4: ++h4; break; \
                    case 5: ++h5; break; \
                    case 6: ++h6; break; \
                    case 7: ++h7; break; \
                    case 8: ++h8; break; \
                    default: break; \
                } \
            } while (false)
            for (; i + 5 < totalNum; i += 6) {
                CM_C3_HIST9_STEP(i);
                CM_C3_HIST9_STEP(i + 1);
                CM_C3_HIST9_STEP(i + 2);
                CM_C3_HIST9_STEP(i + 3);
                CM_C3_HIST9_STEP(i + 4);
                CM_C3_HIST9_STEP(i + 5);
            }
            for (; i < totalNum; ++i) {
                CM_C3_HIST9_STEP(i);
            }
#undef CM_C3_HIST9_STEP
            outputI32Gm_.SetValue(0, h0);
            outputI32Gm_.SetValue(1, h1);
            outputI32Gm_.SetValue(2, h2);
            outputI32Gm_.SetValue(3, h3);
            outputI32Gm_.SetValue(4, h4);
            outputI32Gm_.SetValue(5, h5);
            outputI32Gm_.SetValue(6, h6);
            outputI32Gm_.SetValue(7, h7);
            outputI32Gm_.SetValue(8, h8);
            return;
        }
        for (uint64_t i = 0; i < totalNum; ++i) {
            const uint64_t label = static_cast<uint64_t>(labelsI32Gm_.GetValue(i));
            const uint64_t prediction = static_cast<uint64_t>(predictionsI32Gm_.GetValue(i));
            const uint64_t outOffset = label * static_cast<uint64_t>(numClasses) + prediction;
            const int32_t oldValue = outputI32Gm_.GetValue(outOffset);
            outputI32Gm_.SetValue(outOffset, oldValue + 1);
        }
        return;
    }

    for (uint64_t i = 0; i < totalNum; ++i) {
        const int64_t label = ReadLabel(i);
        const int64_t prediction = ReadPrediction(i);
        if (label < 0 || prediction < 0 || label >= numClasses || prediction >= numClasses) {
            continue;
        }
        const uint64_t outOffset = static_cast<uint64_t>(label) * static_cast<uint64_t>(numClasses) +
            static_cast<uint64_t>(prediction);
        AddToOutput(outOffset, i);
    }
}

template <bool useMte2TailPadding>
__aicore__ inline void ProcessC3VectorHistogram(
    GM_ADDR labels,
    GM_ADDR predictions,
    GM_ADDR y,
    const ConfusionMatrixTilingData* tilingData)
{
    GlobalTensor<int32_t> labelsGm;
    GlobalTensor<int32_t> predictionsGm;
    labelsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(labels));
    predictionsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(predictions));

    TPipe pipe;
    TBuf<TPosition::VECCALC> workBuf;
    pipe.InitBuffer(workBuf, CONFUSION_MATRIX_C3_VECTOR_WORK_BUFFER_BYTES);

    const uint64_t totalNum = static_cast<uint64_t>(tilingData->totalNum);
    LocalTensor<int32_t> dataLocal = workBuf.Get<int32_t>();
    LocalTensor<int32_t> labelsLocal = dataLocal;
    LocalTensor<int32_t> predictionsLocal = dataLocal[CONFUSION_MATRIX_C3_VECTOR_TILE_ELEMS];
    LocalTensor<uint8_t> maskStorage =
        workBuf.Get<uint8_t>()[CONFUSION_MATRIX_C3_VECTOR_DATA_BUFFER_BYTES];
    LocalTensor<uint8_t> labelMask = maskStorage;
    LocalTensor<uint8_t> predictionMask0 = maskStorage[CONFUSION_MATRIX_C3_VECTOR_MASK_BYTES];
    LocalTensor<uint8_t> predictionMask1 = maskStorage[CONFUSION_MATRIX_C3_VECTOR_MASK_BYTES * 2];

    int32_t h0 = 0;
    int32_t h1 = 0;
    int32_t h2 = 0;
    const int32_t row = static_cast<int32_t>(GetBlockIdx());
    uint64_t offset = 0;
    const TEventID copyEvent = static_cast<TEventID>(0);
    const TEventID countEvent = static_cast<TEventID>(1);

    do {
        const uint64_t remaining = totalNum - offset;
        const bool usePaddedTail = remaining <= CONFUSION_MATRIX_C3_VECTOR_TILE_ELEMS && (remaining & 7UL) == 0;
        uint32_t copyCount = remaining > CONFUSION_MATRIX_C3_VECTOR_TILE_ELEMS ?
            CONFUSION_MATRIX_C3_VECTOR_TILE_ELEMS : static_cast<uint32_t>(remaining);
        uint32_t vectorCount = copyCount;
        const uint32_t paddingCount = (64U - (copyCount & 63U)) & 63U;
        if (usePaddedTail) {
            vectorCount = (copyCount + 63U) & ~63U;
            DataCopyParams inputParams{1, static_cast<uint16_t>(copyCount * sizeof(int32_t)), 0, 0};
            DataCopyPadParams inputPadParams{
                true, 0, static_cast<uint8_t>(useMte2TailPadding ? paddingCount : 0U),
                useMte2TailPadding ? static_cast<uint64_t>(static_cast<uint32_t>(-1)) : 0UL};
            DataCopyPad(labelsLocal, labelsGm[offset], inputParams, inputPadParams);
            DataCopyPad(predictionsLocal, predictionsGm[offset], inputParams, inputPadParams);
        } else {
            vectorCount &= ~63U;
            copyCount = vectorCount;
            DataCopy(labelsLocal, labelsGm[offset], vectorCount);
            DataCopy(predictionsLocal, predictionsGm[offset], vectorCount);
        }
        SetFlag<HardEvent::MTE2_V>(copyEvent);
        WaitFlag<HardEvent::MTE2_V>(copyEvent);
        if constexpr (!useMte2TailPadding) {
            if (vectorCount > copyCount) {
                Duplicate(labelsLocal[copyCount], static_cast<int32_t>(-1), vectorCount - copyCount);
                Duplicate(predictionsLocal[copyCount], static_cast<int32_t>(-1), vectorCount - copyCount);
                PipeBarrier<PIPE_V>();
            }
        }

#define CM_C3_MASK_COUNT(MASK, COUNT) \
        do { \
            uint64_t selectedCount = 0; \
            GatherMask( \
                predictionsLocal, labelsLocal, MASK.ReinterpretCast<uint32_t>(), true, vectorCount, \
                {1, 1, 8, 0}, selectedCount); \
            PipeBarrier<PIPE_V>(); \
            SetFlag<HardEvent::V_S>(countEvent); \
            WaitFlag<HardEvent::V_S>(countEvent); \
            COUNT = static_cast<int32_t>(selectedCount); \
        } while (false)
        CompareScalar(labelMask, labelsLocal, row, CMPMODE::EQ, vectorCount);
        PipeBarrier<PIPE_V>();
        uint64_t selectedRowCount = 0;
        GatherMask(
            labelsLocal, predictionsLocal, labelMask.ReinterpretCast<uint32_t>(), true, vectorCount,
            {1, 1, 8, 0}, selectedRowCount);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(countEvent);
        WaitFlag<HardEvent::V_S>(countEvent);
        const int32_t rowCount = static_cast<int32_t>(selectedRowCount);
        if (rowCount > 0) {
            const uint32_t compactVectorCount = (static_cast<uint32_t>(rowCount) + 63U) & ~63U;
            if (compactVectorCount > static_cast<uint32_t>(rowCount)) {
                Duplicate(
                    labelsLocal[static_cast<uint32_t>(rowCount)], static_cast<int32_t>(-1),
                    compactVectorCount - static_cast<uint32_t>(rowCount));
                PipeBarrier<PIPE_V>();
            }
            CompareScalar(predictionMask0, labelsLocal, static_cast<int32_t>(0), CMPMODE::EQ, compactVectorCount);
            CompareScalar(predictionMask1, labelsLocal, static_cast<int32_t>(1), CMPMODE::EQ, compactVectorCount);
            PipeBarrier<PIPE_V>();
            vectorCount = compactVectorCount;
            int32_t count0 = 0;
            int32_t count1 = 0;
            CM_C3_MASK_COUNT(predictionMask0, count0);
            CM_C3_MASK_COUNT(predictionMask1, count1);
            h0 += count0;
            h1 += count1;
            h2 += rowCount - count0 - count1;
        }
#undef CM_C3_MASK_COUNT
        offset += copyCount;
    } while (offset + 64 <= totalNum);

    for (; offset < totalNum; ++offset) {
        const int32_t label = labelsGm.GetValue(offset);
        const int32_t prediction = predictionsGm.GetValue(offset);
        if (label != row || prediction < 0 || prediction >= 3) {
            continue;
        }
        switch (prediction) {
            case 0: ++h0; break;
            case 1: ++h1; break;
            case 2: ++h2; break;
            default: break;
        }
    }

    GlobalTensor<int32_t> outputGm;
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(y));
    const uint64_t outputOffset = static_cast<uint64_t>(row) * 3;
    outputGm.SetValue(outputOffset, h0);
    outputGm.SetValue(outputOffset + 1, h1);
    outputGm.SetValue(outputOffset + 2, h2);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline bool ConfusionMatrix::CanUseSmallOutputRowSplit(uint64_t totalNum) const
{
    if (numClasses_ <= 1 || numClasses_ > static_cast<int64_t>(CONFUSION_MATRIX_SMALL_ROW_SPLIT_MAX_CLASSES)) {
        return false;
    }
    const uint64_t classCount = static_cast<uint64_t>(numClasses_);
    const uint64_t rowBytes = classCount * sizeof(int32_t);
    return labelsDtype_ == CONFUSION_MATRIX_DTYPE_INT32 &&
        predictionsDtype_ == CONFUSION_MATRIX_DTYPE_INT32 &&
        outputDtype_ == CONFUSION_MATRIX_DTYPE_INT32 &&
        hasWeights_ == 0 &&
        totalNum >= CONFUSION_MATRIX_SMALL_ROW_SPLIT_MIN_SAMPLES &&
        rowBytes >= 32 &&
        (rowBytes & 31UL) == 0 &&
        GetBlockNum() > 1;
}

__aicore__ inline void ConfusionMatrix::ProcessSmallOutputRowSplit(uint64_t totalNum)
{
    const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
    const uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
    const uint64_t classCount = static_cast<uint64_t>(numClasses_);
    const uint64_t rowsPerCore = (classCount + blockNum - 1) / blockNum;
    const uint64_t rowStart = blockIdx * rowsPerCore;
    if (rowStart >= classCount) {
        return;
    }
    uint64_t rowEnd = rowStart + rowsPerCore;
    if (rowEnd > classCount) {
        rowEnd = classCount;
    }

    ClearOutputRange(rowStart * classCount, rowEnd * classCount);

    for (uint64_t i = 0; i < totalNum; ++i) {
        const int32_t labelValue = labelsI32Gm_.GetValue(i);
        if (labelValue < 0) {
            continue;
        }
        const uint64_t label = static_cast<uint64_t>(labelValue);
        if (label < rowStart || label >= rowEnd) {
            continue;
        }
        const int32_t predictionValue = predictionsI32Gm_.GetValue(i);
        if (predictionValue < 0) {
            continue;
        }
        const uint64_t prediction = static_cast<uint64_t>(predictionValue);
        if (prediction >= classCount) {
            continue;
        }
        const uint64_t outOffset = label * classCount + prediction;
        const int32_t oldValue = outputI32Gm_.GetValue(outOffset);
        outputI32Gm_.SetValue(outOffset, oldValue + 1);
    }
}

__aicore__ inline bool ConfusionMatrix::CanUseSparseSplitInt32(uint64_t totalNum) const
{
    if (useBulkClear_ == 0 || totalNum == 0 || totalNum > CONFUSION_MATRIX_SPARSE_CAPACITY ||
        outputNum_ > static_cast<int64_t>(CONFUSION_MATRIX_UINT32_OFFSET_MAX)) {
        return false;
    }
    const uint64_t outputNum = static_cast<uint64_t>(outputNum_);
    return outputNum / totalNum >= totalNum;
}

__aicore__ inline void ConfusionMatrix::ProcessSplitOutputSparseInt32(
    uint64_t rowStart,
    uint64_t rowEnd,
    uint64_t classCount,
    uint64_t totalNum)
{
    const bool useVectorFilter =
        (reserved_ & CONFUSION_MATRIX_RESERVED_VECTOR_FILTER_SPARSE) != 0;
    LocalTensor<int32_t> offsetLowLocal = zeroBuf_.Get<int32_t>();
    LocalTensor<int32_t> countLocal = zeroBuf_.Get<int32_t>()[CONFUSION_MATRIX_SPARSE_COUNT_LOCAL_OFFSET];
    LocalTensor<int32_t> occupiedSlotLocal =
        zeroBuf_.Get<int32_t>()[CONFUSION_MATRIX_SPARSE_OCCUPIED_LOCAL_OFFSET];
    if (!useVectorFilter) {
        Duplicate(countLocal, static_cast<int32_t>(0), CONFUSION_MATRIX_SPARSE_CAPACITY);
        TEventID eventIdVToS = static_cast<TEventID>(1);
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
    }
    uint32_t occupiedCount = 0;
    const int32_t rowStartI32 = static_cast<int32_t>(rowStart);
    const int32_t rowEndI32 = static_cast<int32_t>(rowEnd);
    const int32_t classCountI32 = static_cast<int32_t>(classCount);
#define CM_ACCUMULATE_SPARSE_OFFSET(OUT_OFFSET_LOW) \
    do { \
        const uint32_t currentOffsetLow = (OUT_OFFSET_LOW); \
        const uint32_t hashValue = currentOffsetLow ^ (currentOffsetLow >> 7); \
        uint32_t slot = hashValue % CONFUSION_MATRIX_SPARSE_CAPACITY; \
        for (uint32_t probe = 0; probe < CONFUSION_MATRIX_SPARSE_CAPACITY; ++probe) { \
            const int32_t oldCount = countLocal.GetValue(slot); \
            if (oldCount == 0) { \
                offsetLowLocal.SetValue(slot, static_cast<int32_t>(currentOffsetLow)); \
                countLocal.SetValue(slot, static_cast<int32_t>(1)); \
                occupiedSlotLocal.SetValue(occupiedCount, static_cast<int32_t>(slot)); \
                ++occupiedCount; \
                break; \
            } \
            if (static_cast<uint32_t>(offsetLowLocal.GetValue(slot)) == currentOffsetLow) { \
                countLocal.SetValue(slot, oldCount + 1); \
                break; \
            } \
            ++slot; \
            if (slot == CONFUSION_MATRIX_SPARSE_CAPACITY) { \
                slot = 0; \
            } \
        } \
    } while (false)

    if (useVectorFilter) {
        LocalTensor<int32_t> labelsLocal =
            zeroBuf_.Get<int32_t>()[CONFUSION_MATRIX_SPARSE_FILTER_LABEL_LOCAL_OFFSET];
        LocalTensor<int32_t> predictionsLocal =
            zeroBuf_.Get<int32_t>()[CONFUSION_MATRIX_SPARSE_FILTER_PREDICTION_LOCAL_OFFSET];
        LocalTensor<int32_t> compactLabelsLocal =
            zeroBuf_.Get<int32_t>()[CONFUSION_MATRIX_SPARSE_FILTER_COMPACT_LABEL_LOCAL_OFFSET];
        LocalTensor<float> labelsFloatLocal = compactLabelsLocal.ReinterpretCast<float>();
        LocalTensor<uint8_t> maskStorage =
            zeroBuf_.Get<uint8_t>()[CONFUSION_MATRIX_SPARSE_FILTER_MASK_LOCAL_OFFSET_BYTES];
        LocalTensor<uint8_t> lowerMask = maskStorage;
        LocalTensor<uint8_t> upperMask = maskStorage[CONFUSION_MATRIX_SPARSE_FILTER_MASK_BYTES];
        LocalTensor<uint8_t> ownedMask = maskStorage[CONFUSION_MATRIX_SPARSE_FILTER_MASK_BYTES * 2];

        const TEventID copyEvent = static_cast<TEventID>(0);
        const TEventID selectedCountEvent = static_cast<TEventID>(2);
        uint64_t offset = 0;
        while (offset + 64 <= totalNum) {
            const uint64_t remaining = totalNum - offset;
            const bool usePaddedTail = remaining <= CONFUSION_MATRIX_SPARSE_FILTER_TILE_ELEMS &&
                (remaining & 7UL) == 0;
            uint32_t copyCount = remaining > CONFUSION_MATRIX_SPARSE_FILTER_TILE_ELEMS ?
                CONFUSION_MATRIX_SPARSE_FILTER_TILE_ELEMS : static_cast<uint32_t>(remaining);
            uint32_t vectorCount = copyCount;
            if (usePaddedTail) {
                vectorCount = (copyCount + 63U) & ~63U;
                DataCopyParams inputParams{1, static_cast<uint16_t>(copyCount * sizeof(int32_t)), 0, 0};
                DataCopyPadParams inputPadParams{true, 0, 0, 0};
                DataCopyPad(labelsLocal, labelsI32Gm_[offset], inputParams, inputPadParams);
                DataCopyPad(predictionsLocal, predictionsI32Gm_[offset], inputParams, inputPadParams);
            } else {
                vectorCount &= ~63U;
                copyCount = vectorCount;
                DataCopy(labelsLocal, labelsI32Gm_[offset], vectorCount);
                DataCopy(predictionsLocal, predictionsI32Gm_[offset], vectorCount);
            }
            SetFlag<HardEvent::MTE2_V>(copyEvent);
            WaitFlag<HardEvent::MTE2_V>(copyEvent);
            if (vectorCount > copyCount) {
                Duplicate(labelsLocal[copyCount], static_cast<int32_t>(-1), vectorCount - copyCount);
                Duplicate(predictionsLocal[copyCount], static_cast<int32_t>(-1), vectorCount - copyCount);
                PipeBarrier<PIPE_V>();
            }

            Cast(labelsFloatLocal, labelsLocal, RoundMode::CAST_NONE, vectorCount);
            PipeBarrier<PIPE_V>();
            CompareScalar(lowerMask, labelsFloatLocal, static_cast<float>(rowStartI32), CMPMODE::GE, vectorCount);
            CompareScalar(upperMask, labelsFloatLocal, static_cast<float>(rowEndI32), CMPMODE::LT, vectorCount);
            PipeBarrier<PIPE_V>();
            And(
                ownedMask.ReinterpretCast<uint16_t>(), lowerMask.ReinterpretCast<uint16_t>(),
                upperMask.ReinterpretCast<uint16_t>(), vectorCount / 16);
            PipeBarrier<PIPE_V>();

            uint64_t selectedLabelCount = 0;
            GatherMask(
                compactLabelsLocal, labelsLocal, ownedMask.ReinterpretCast<uint32_t>(), true, vectorCount,
                {1, 1, 8, 0}, selectedLabelCount);
            PipeBarrier<PIPE_V>();
            uint64_t selectedCount = 0;
            GatherMask(
                labelsLocal, predictionsLocal, ownedMask.ReinterpretCast<uint32_t>(), true, vectorCount,
                {1, 1, 8, 0}, selectedCount);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(selectedCountEvent);
            WaitFlag<HardEvent::V_S>(selectedCountEvent);
            for (uint32_t j = 0; j < static_cast<uint32_t>(selectedCount); ++j) {
                const int32_t labelValue = compactLabelsLocal.GetValue(j);
                const int32_t predictionValue = labelsLocal.GetValue(j);
                if (predictionValue < 0 || predictionValue >= classCountI32) {
                    continue;
                }
                const uint64_t outOffset = static_cast<uint64_t>(labelValue) * classCount +
                    static_cast<uint64_t>(predictionValue);
                CM_ACCUMULATE_SPARSE_OFFSET(static_cast<uint32_t>(outOffset));
            }
            offset += copyCount;
        }

        for (; offset < totalNum; ++offset) {
            const int32_t labelValue = labelsI32Gm_.GetValue(offset);
            if (labelValue < rowStartI32 || labelValue >= rowEndI32) {
                continue;
            }
            const int32_t predictionValue = predictionsI32Gm_.GetValue(offset);
            if (predictionValue < 0 || predictionValue >= classCountI32) {
                continue;
            }
            const uint64_t outOffset = static_cast<uint64_t>(labelValue) * classCount +
                static_cast<uint64_t>(predictionValue);
            CM_ACCUMULATE_SPARSE_OFFSET(static_cast<uint32_t>(outOffset));
        }
    } else {
        for (uint64_t i = 0; i < totalNum; ++i) {
            const int32_t labelValue = labelsI32Gm_.GetValue(i);
            if (labelValue < rowStartI32 || labelValue >= rowEndI32) {
                continue;
            }
            const int32_t predictionValue = predictionsI32Gm_.GetValue(i);
            if (predictionValue < 0 || predictionValue >= classCountI32) {
                continue;
            }
            const uint64_t outOffset = static_cast<uint64_t>(labelValue) * classCount +
                static_cast<uint64_t>(predictionValue);
            CM_ACCUMULATE_SPARSE_OFFSET(static_cast<uint32_t>(outOffset));
        }
    }
#undef CM_ACCUMULATE_SPARSE_OFFSET

    for (uint32_t i = 0; i < occupiedCount; ++i) {
        const uint32_t k = static_cast<uint32_t>(occupiedSlotLocal.GetValue(i));
        const int32_t count = countLocal.GetValue(k);
        const uint64_t outOffset = static_cast<uint64_t>(static_cast<uint32_t>(offsetLowLocal.GetValue(k)));
        if (useVectorFilter && outOffset < sparseNormalStart_) {
            outputI32BypassGm_.SetValue(outOffset, count);
        } else {
            outputI32Gm_.SetValue(outOffset, count);
        }
    }
}

__aicore__ inline void ConfusionMatrix::ProcessSplitOutput()
{
    const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
    const uint64_t classCount = static_cast<uint64_t>(numClasses_);
    const uint64_t rowsPerCore = static_cast<uint64_t>(blockFactor_);
    const uint64_t rowStart = blockIdx * rowsPerCore;
    if (rowStart >= classCount) {
        return;
    }
    uint64_t rowEnd = rowStart + rowsPerCore;
    if (rowEnd > classCount) {
        rowEnd = classCount;
    }

    const uint64_t start = rowStart * classCount;
    const uint64_t end = rowEnd * classCount;
    sparseNormalStart_ = start;
    if ((reserved_ & CONFUSION_MATRIX_RESERVED_VECTOR_FILTER_SPARSE) != 0) {
        const uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
        uint64_t normalTailElems =
            CONFUSION_MATRIX_SPARSE_NORMAL_L2_BUDGET_BYTES / sizeof(int32_t) / blockNum;
        normalTailElems &= ~127UL;
        if (normalTailElems != 0 && end - start > normalTailElems) {
            sparseNormalStart_ = (end - normalTailElems + 127UL) & ~127UL;
            if (sparseNormalStart_ > end) {
                sparseNormalStart_ = end;
            }
        }
    }
    ClearOutputRange(start, end);

    const uint64_t totalNum = static_cast<uint64_t>(totalNum_);
    if (labelsDtype_ == CONFUSION_MATRIX_DTYPE_INT32 &&
        predictionsDtype_ == CONFUSION_MATRIX_DTYPE_INT32 &&
        outputDtype_ == CONFUSION_MATRIX_DTYPE_INT32 &&
        hasWeights_ == 0) {
        if (CanUseSparseSplitInt32(totalNum)) {
            ProcessSplitOutputSparseInt32(rowStart, rowEnd, classCount, totalNum);
            return;
        }
        for (uint64_t i = 0; i < totalNum; ++i) {
            const int32_t labelValue = labelsI32Gm_.GetValue(i);
            if (labelValue < 0) {
                continue;
            }
            const uint64_t label = static_cast<uint64_t>(labelValue);
            if (label < rowStart || label >= rowEnd) {
                continue;
            }
            const int32_t predictionValue = predictionsI32Gm_.GetValue(i);
            if (predictionValue < 0) {
                continue;
            }
            const uint64_t prediction = static_cast<uint64_t>(predictionValue);
            if (prediction >= classCount) {
                continue;
            }
            const uint64_t outOffset = label * classCount + prediction;
            const int32_t oldValue = outputI32Gm_.GetValue(outOffset);
            outputI32Gm_.SetValue(outOffset, oldValue + 1);
        }
        return;
    }

    for (uint64_t i = 0; i < totalNum; ++i) {
        const int64_t label = ReadLabel(i);
        if (label < 0) {
            continue;
        }
        const uint64_t labelOffset = static_cast<uint64_t>(label);
        if (labelOffset < rowStart || labelOffset >= rowEnd) {
            continue;
        }
        const int64_t prediction = ReadPrediction(i);
        if (prediction < 0 || prediction >= numClasses_) {
            continue;
        }
        const uint64_t outOffset = labelOffset * classCount + static_cast<uint64_t>(prediction);
        AddToOutput(outOffset, i);
    }
}

} // namespace NsConfusionMatrix
#endif // CONFUSIONMATRIX_H
