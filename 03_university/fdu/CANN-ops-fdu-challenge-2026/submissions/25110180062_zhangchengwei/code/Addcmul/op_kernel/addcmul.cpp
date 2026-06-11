// Kernel implementation.
#include "kernel_operator.h"

#include "addcmul_tiling.h"
#include "tiling_key_addcmul.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <class T>
class KernelAddcmul {
public:
    __aicore__ inline KernelAddcmul() {}

    __aicore__ inline void Init(GM_ADDR inputData, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
        const AddcmulTilingData &tilingData) {
        tiling_ = tilingData;
        inputBase_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(inputData));
        x1Base_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x1));
        x2Base_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x2));
        value_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(value), 1);
        yBase_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));
        valueScalar_ = value_.GetValue(0);

        if (tiling_.sameShape != 0) {
            InitVectorRange(inputData, x1, x2, y);
            pipe_.InitBuffer(inputQueue_, BUFFER_NUM, tileDataNum_ * sizeof(T));
            pipe_.InitBuffer(x1Queue_, BUFFER_NUM, tileDataNum_ * sizeof(T));
            pipe_.InitBuffer(x2Queue_, BUFFER_NUM, tileDataNum_ * sizeof(T));
            pipe_.InitBuffer(outQueue_, BUFFER_NUM, tileDataNum_ * sizeof(T));
        } else {
            InitScalarRange();
        }
    }

    __aicore__ inline void Process() {
        if (tiling_.sameShape != 0) {
            ProcessVector();
            PipeBarrier<PIPE_ALL>();
            if (GetBlockIdx() == 0 && tiling_.vectorLength < tiling_.length) {
                ProcessScalarRange(tiling_.vectorLength, tiling_.length, true);
            }
        } else {
            ProcessScalar();
        }
    }

private:
    __aicore__ inline void InitVectorRange(GM_ADDR inputData, GM_ADDR x1, GM_ADDR x2, GM_ADDR y) {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t globalOffset = tiling_.bigCoreDataNum * blockIdx;
        tileDataNum_ = tiling_.tileDataNum;

        if (blockIdx < tiling_.tailBlockNum) {
            coreDataNum_ = tiling_.bigCoreDataNum;
            tileNum_ = tiling_.finalBigTileNum;
            tailDataNum_ = tiling_.bigTailDataNum;
        } else {
            coreDataNum_ = tiling_.smallCoreDataNum;
            tileNum_ = tiling_.finalSmallTileNum;
            tailDataNum_ = tiling_.smallTailDataNum;
            globalOffset -= (tiling_.bigCoreDataNum - tiling_.smallCoreDataNum) *
                (blockIdx - tiling_.tailBlockNum);
        }

        input_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(inputData) + globalOffset, coreDataNum_);
        x1_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x1) + globalOffset, coreDataNum_);
        x2_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x2) + globalOffset, coreDataNum_);
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y) + globalOffset, coreDataNum_);
    }

    __aicore__ inline void InitScalarRange() {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t base = tiling_.length / tiling_.blockDim;
        uint32_t tail = tiling_.length % tiling_.blockDim;
        scalarStart_ = blockIdx * base + (blockIdx < tail ? blockIdx : tail);
        scalarCount_ = base + (blockIdx < tail ? 1U : 0U);
    }

    __aicore__ inline void ProcessVector() {
        for (uint32_t i = 0; i < tileNum_; ++i) {
            processDataNum_ = (i == tileNum_ - 1) ? tailDataNum_ : tileDataNum_;
            CopyIn(i);
            ComputeVector();
            CopyOut(i);
        }
    }

    __aicore__ inline void CopyIn(uint32_t progress) {
        LocalTensor<T> inputLocal = inputQueue_.AllocTensor<T>();
        LocalTensor<T> x1Local = x1Queue_.AllocTensor<T>();
        LocalTensor<T> x2Local = x2Queue_.AllocTensor<T>();
        DataCopy(inputLocal, input_[progress * tileDataNum_], processDataNum_);
        DataCopy(x1Local, x1_[progress * tileDataNum_], processDataNum_);
        DataCopy(x2Local, x2_[progress * tileDataNum_], processDataNum_);
        inputQueue_.EnQue(inputLocal);
        x1Queue_.EnQue(x1Local);
        x2Queue_.EnQue(x2Local);
    }

    __aicore__ inline void ComputeVector() {
        LocalTensor<T> inputLocal = inputQueue_.DeQue<T>();
        LocalTensor<T> x1Local = x1Queue_.DeQue<T>();
        LocalTensor<T> x2Local = x2Queue_.DeQue<T>();
        LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();

        Mul(outLocal, x1Local, x2Local, processDataNum_);
        Muls(outLocal, outLocal, valueScalar_, processDataNum_);
        Add(outLocal, inputLocal, outLocal, processDataNum_);

        outQueue_.EnQue(outLocal);
        inputQueue_.FreeTensor(inputLocal);
        x1Queue_.FreeTensor(x1Local);
        x2Queue_.FreeTensor(x2Local);
    }

    __aicore__ inline void CopyOut(uint32_t progress) {
        LocalTensor<T> outLocal = outQueue_.DeQue<T>();
        DataCopy(y_[progress * tileDataNum_], outLocal, processDataNum_);
        outQueue_.FreeTensor(outLocal);
    }

    __aicore__ inline uint32_t Offset(uint32_t index, const uint32_t strides[ADDCMUL_MAX_DIMS]) const {
        uint32_t offset = 0;
        uint32_t tmp = index;
        for (int32_t dim = static_cast<int32_t>(tiling_.rank) - 1; dim >= 0; --dim) {
            uint32_t coord = tmp % tiling_.shape[dim];
            tmp /= tiling_.shape[dim];
            offset += coord * strides[dim];
        }
        return offset;
    }

    __aicore__ inline void ProcessScalar() {
        ProcessScalarRange(scalarStart_, scalarStart_ + scalarCount_, false);
    }

    __aicore__ inline void ProcessScalarRange(uint32_t begin, uint32_t end, bool vectorTail) {
        float scalar = static_cast<float>(valueScalar_);
        for (uint32_t outIndex = begin; outIndex < end; ++outIndex) {
            float inputValue = static_cast<float>(inputBase_.GetValue(Offset(outIndex, tiling_.inputStrides)));
            float x1Value = static_cast<float>(x1Base_.GetValue(Offset(outIndex, tiling_.x1Strides)));
            float x2Value = static_cast<float>(x2Base_.GetValue(Offset(outIndex, tiling_.x2Strides)));
            if (vectorTail) {
                float product = static_cast<float>(static_cast<T>(x1Value * x2Value));
                float scaled = static_cast<float>(static_cast<T>(product * scalar));
                yBase_.SetValue(outIndex, static_cast<T>(inputValue + scaled));
            } else {
                yBase_.SetValue(outIndex, static_cast<T>(inputValue + x1Value * x2Value * scalar));
            }
        }
    }

    AddcmulTilingData tiling_;
    GlobalTensor<T> inputBase_;
    GlobalTensor<T> x1Base_;
    GlobalTensor<T> x2Base_;
    GlobalTensor<T> value_;
    GlobalTensor<T> yBase_;
    GlobalTensor<T> input_;
    GlobalTensor<T> x1_;
    GlobalTensor<T> x2_;
    GlobalTensor<T> y_;
    uint32_t coreDataNum_;
    uint32_t tileNum_;
    uint32_t tileDataNum_;
    uint32_t tailDataNum_;
    uint32_t processDataNum_;
    uint32_t scalarStart_;
    uint32_t scalarCount_;
    T valueScalar_;
    TPipe pipe_;
    TQue<TPosition::VECIN, BUFFER_NUM> inputQueue_;
    TQue<TPosition::VECIN, BUFFER_NUM> x1Queue_;
    TQue<TPosition::VECIN, BUFFER_NUM> x2Queue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueue_;
};

template <>
class KernelAddcmul<int32_t> {
public:
    __aicore__ inline KernelAddcmul() {}
    __aicore__ inline void Init(GM_ADDR inputData, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
        const AddcmulTilingData &tilingData) {
        tiling_ = tilingData;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(inputData));
        x1_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(x1));
        x2_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(x2));
        value_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(value), 1);
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(y));
        valueScalar_ = value_.GetValue(0);
        if (tiling_.sameShape != 0) {
            InitVectorRange(inputData, x1, x2, y);
            pipe_.InitBuffer(inputQueue_, BUFFER_NUM, tileDataNum_ * sizeof(int32_t));
            pipe_.InitBuffer(x1Queue_, BUFFER_NUM, tileDataNum_ * sizeof(int32_t));
            pipe_.InitBuffer(x2Queue_, BUFFER_NUM, tileDataNum_ * sizeof(int32_t));
            pipe_.InitBuffer(outQueue_, BUFFER_NUM, tileDataNum_ * sizeof(int32_t));
        } else {
            InitScalarRange();
        }
    }

    __aicore__ inline void Process() {
        if (tiling_.sameShape != 0) {
            ProcessVector();
            PipeBarrier<PIPE_ALL>();
            if (GetBlockIdx() == 0 && tiling_.vectorLength < tiling_.length) {
                ProcessScalarRange(tiling_.vectorLength, tiling_.length);
            }
            return;
        }
        ProcessScalarRange(scalarStart_, scalarStart_ + scalarCount_);
    }

private:
    __aicore__ inline void ProcessScalarRange(uint32_t begin, uint32_t end) {
        for (uint32_t outIndex = begin; outIndex < end; ++outIndex) {
            int64_t result = static_cast<int64_t>(input_.GetValue(Offset(outIndex, tiling_.inputStrides))) +
                static_cast<int64_t>(x1_.GetValue(Offset(outIndex, tiling_.x1Strides))) *
                static_cast<int64_t>(x2_.GetValue(Offset(outIndex, tiling_.x2Strides))) *
                static_cast<int64_t>(valueScalar_);
            y_.SetValue(outIndex, static_cast<int32_t>(result));
        }
    }

    __aicore__ inline void InitVectorRange(GM_ADDR inputData, GM_ADDR x1, GM_ADDR x2, GM_ADDR y) {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t globalOffset = tiling_.bigCoreDataNum * blockIdx;
        tileDataNum_ = tiling_.tileDataNum;

        if (blockIdx < tiling_.tailBlockNum) {
            coreDataNum_ = tiling_.bigCoreDataNum;
            tileNum_ = tiling_.finalBigTileNum;
            tailDataNum_ = tiling_.bigTailDataNum;
        } else {
            coreDataNum_ = tiling_.smallCoreDataNum;
            tileNum_ = tiling_.finalSmallTileNum;
            tailDataNum_ = tiling_.smallTailDataNum;
            globalOffset -= (tiling_.bigCoreDataNum - tiling_.smallCoreDataNum) *
                (blockIdx - tiling_.tailBlockNum);
        }

        inputVec_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(inputData) + globalOffset, coreDataNum_);
        x1Vec_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(x1) + globalOffset, coreDataNum_);
        x2Vec_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(x2) + globalOffset, coreDataNum_);
        yVec_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(y) + globalOffset, coreDataNum_);
    }

    __aicore__ inline void ProcessVector() {
        for (uint32_t i = 0; i < tileNum_; ++i) {
            processDataNum_ = (i == tileNum_ - 1) ? tailDataNum_ : tileDataNum_;
            LocalTensor<int32_t> inputLocal = inputQueue_.AllocTensor<int32_t>();
            LocalTensor<int32_t> x1Local = x1Queue_.AllocTensor<int32_t>();
            LocalTensor<int32_t> x2Local = x2Queue_.AllocTensor<int32_t>();
            DataCopy(inputLocal, inputVec_[i * tileDataNum_], processDataNum_);
            DataCopy(x1Local, x1Vec_[i * tileDataNum_], processDataNum_);
            DataCopy(x2Local, x2Vec_[i * tileDataNum_], processDataNum_);
            inputQueue_.EnQue(inputLocal);
            x1Queue_.EnQue(x1Local);
            x2Queue_.EnQue(x2Local);

            inputLocal = inputQueue_.DeQue<int32_t>();
            x1Local = x1Queue_.DeQue<int32_t>();
            x2Local = x2Queue_.DeQue<int32_t>();
            LocalTensor<int32_t> outLocal = outQueue_.AllocTensor<int32_t>();
            Mul(outLocal, x1Local, x2Local, processDataNum_);
            Muls(outLocal, outLocal, valueScalar_, processDataNum_);
            Add(outLocal, inputLocal, outLocal, processDataNum_);
            outQueue_.EnQue(outLocal);
            inputQueue_.FreeTensor(inputLocal);
            x1Queue_.FreeTensor(x1Local);
            x2Queue_.FreeTensor(x2Local);

            outLocal = outQueue_.DeQue<int32_t>();
            DataCopy(yVec_[i * tileDataNum_], outLocal, processDataNum_);
            outQueue_.FreeTensor(outLocal);
        }
    }

    __aicore__ inline void InitScalarRange() {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t base = tiling_.length / tiling_.blockDim;
        uint32_t tail = tiling_.length % tiling_.blockDim;
        scalarStart_ = blockIdx * base + (blockIdx < tail ? blockIdx : tail);
        scalarCount_ = base + (blockIdx < tail ? 1U : 0U);
    }

    __aicore__ inline uint32_t Offset(uint32_t index, const uint32_t strides[ADDCMUL_MAX_DIMS]) const {
        uint32_t offset = 0;
        uint32_t tmp = index;
        for (int32_t dim = static_cast<int32_t>(tiling_.rank) - 1; dim >= 0; --dim) {
            uint32_t coord = tmp % tiling_.shape[dim];
            tmp /= tiling_.shape[dim];
            offset += coord * strides[dim];
        }
        return offset;
    }

    AddcmulTilingData tiling_;
    GlobalTensor<int32_t> input_;
    GlobalTensor<int32_t> x1_;
    GlobalTensor<int32_t> x2_;
    GlobalTensor<int32_t> value_;
    GlobalTensor<int32_t> y_;
    GlobalTensor<int32_t> inputVec_;
    GlobalTensor<int32_t> x1Vec_;
    GlobalTensor<int32_t> x2Vec_;
    GlobalTensor<int32_t> yVec_;
    uint32_t coreDataNum_;
    uint32_t tileNum_;
    uint32_t tileDataNum_;
    uint32_t tailDataNum_;
    uint32_t processDataNum_;
    uint32_t scalarStart_;
    uint32_t scalarCount_;
    int32_t valueScalar_;
    TPipe pipe_;
    TQue<TPosition::VECIN, BUFFER_NUM> inputQueue_;
    TQue<TPosition::VECIN, BUFFER_NUM> x1Queue_;
    TQue<TPosition::VECIN, BUFFER_NUM> x2Queue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueue_;
};

template <>
class KernelAddcmul<int8_t> {
public:
    __aicore__ inline KernelAddcmul() {}
    __aicore__ inline void Init(GM_ADDR inputData, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
        const AddcmulTilingData &tilingData) {
        tiling_ = tilingData;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(inputData));
        x1_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(x1));
        x2_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(x2));
        value_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(value), 1);
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(y));
        valueScalar_ = value_.GetValue(0);
        InitScalarRange();
    }

    __aicore__ inline void Process() {
        ProcessScalarRange(scalarStart_, scalarStart_ + scalarCount_);
    }

private:
    __aicore__ inline int8_t WrapInt8(int32_t value) const {
        int32_t wrapped = value & 255;
        if (wrapped >= 128) {
            wrapped -= 256;
        }
        return static_cast<int8_t>(wrapped);
    }

    __aicore__ inline void ProcessScalarRange(uint32_t begin, uint32_t end) {
        for (uint32_t outIndex = begin; outIndex < end; ++outIndex) {
            int32_t result = static_cast<int32_t>(input_.GetValue(Offset(outIndex, tiling_.inputStrides))) +
                static_cast<int32_t>(x1_.GetValue(Offset(outIndex, tiling_.x1Strides))) *
                static_cast<int32_t>(x2_.GetValue(Offset(outIndex, tiling_.x2Strides))) *
                static_cast<int32_t>(valueScalar_);
            y_.SetValue(outIndex, WrapInt8(result));
        }
    }

    __aicore__ inline void InitScalarRange() {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t base = tiling_.length / tiling_.blockDim;
        uint32_t tail = tiling_.length % tiling_.blockDim;
        scalarStart_ = blockIdx * base + (blockIdx < tail ? blockIdx : tail);
        scalarCount_ = base + (blockIdx < tail ? 1U : 0U);
    }

    __aicore__ inline uint32_t Offset(uint32_t index, const uint32_t strides[ADDCMUL_MAX_DIMS]) const {
        uint32_t offset = 0;
        uint32_t tmp = index;
        for (int32_t dim = static_cast<int32_t>(tiling_.rank) - 1; dim >= 0; --dim) {
            uint32_t coord = tmp % tiling_.shape[dim];
            tmp /= tiling_.shape[dim];
            offset += coord * strides[dim];
        }
        return offset;
    }

    AddcmulTilingData tiling_;
    GlobalTensor<int8_t> input_;
    GlobalTensor<int8_t> x1_;
    GlobalTensor<int8_t> x2_;
    GlobalTensor<int8_t> value_;
    GlobalTensor<int8_t> y_;
    uint32_t scalarStart_;
    uint32_t scalarCount_;
    int8_t valueScalar_;
};

template <typename T>
__aicore__ inline void RunKernel(GM_ADDR inputData, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
    GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(AddcmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddcmulTilingData, tilingData, tiling);
    KernelAddcmul<T> op;
    op.Init(inputData, x1, x2, value, y, tilingData);
    op.Process();
}

template <typename DT_INPUT_DATA>
__global__ __aicore__ void addcmul(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling) {
    RunKernel<DT_INPUT_DATA>(input_data, x1, x2, value, y, tiling);
}
