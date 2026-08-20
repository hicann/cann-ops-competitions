#include "kernel_operator.h"
using namespace AscendC;

#include <cstdint>

__aicore__ inline uint16_t dup_byte(uint8_t a) {
    return (static_cast<uint16_t>(a) << 8) | a;
}

constexpr uint32_t BufferNum_broadcast_cols = 1;
// scale张量始终是需要广播的张量,只考虑了二维列广播
template <typename T>
class KernelScale_broadcast_cols {
public:
    __aicore__ inline KernelScale_broadcast_cols() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR out, uint32_t smallSize, uint32_t incSize, uint32_t formerNum,
                                uint32_t totalSize, uint32_t *mmInputDims, uint32_t *mmOtherDims, uint32_t *mmOutputDims, uint8_t nOutputDims,
                                bool hasBias, TPipe *pipeIn) {
        this->pipe = pipeIn;
        this->hasBias = hasBias;
        this->mmInputDims = mmInputDims;
        this->mmOtherDims = mmOtherDims;
        this->mmOutputDims = mmOutputDims;
        this->nOutputDims = nOutputDims;
        this->totalSize = totalSize;

        if (nOutputDims == 2) {
            this->D = 1;
            this->H = mmOutputDims[0];
            this->W = mmOutputDims[1];
        } else if (nOutputDims == 3) {
            this->D = mmOutputDims[0];
            this->H = mmOutputDims[1];
            this->W = mmOutputDims[2];
        }

        this->beginIndex = 0;
        if (GetBlockIdx() < formerNum) {
            this->size = smallSize + incSize;
            beginIndex = this->size * GetBlockIdx();
        } else {
            this->size = smallSize;
            beginIndex = this->size * GetBlockIdx() + formerNum * incSize;
        }
        if (beginIndex + size > H) {
            this->size = H - beginIndex;
        }
        inputGm.SetGlobalBuffer((__gm__ T *)input + beginIndex * W, totalSize);
        scaleGm.SetGlobalBuffer((__gm__ T *)scale + beginIndex, totalSize);
        if (hasBias) {
            biasGm.SetGlobalBuffer((__gm__ T *)bias + beginIndex, totalSize);
        }
        outGm.SetGlobalBuffer((__gm__ T *)out + beginIndex * W, totalSize);

        if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value || std::is_same<T, bfloat16_t>::value) {
            uint32_t spaceSize = min((uint32_t)(W * sizeof(T)), (uint32_t)(27) * 1024 / BufferNum_sca);
            this->n_elements_per_iter = spaceSize / sizeof(T);
            this->smallLoopTimes = (W + n_elements_per_iter - 1) / n_elements_per_iter;
            pipe->InitBuffer(inputBuf, BufferNum_sca, spaceSize);
            pipe->InitBuffer(outBuf, BufferNum_sca, spaceSize + 32);
            pipe->InitBuffer(tmpBuf, BufferNum_sca, spaceSize * 4);
        } else if (std::is_same<T, int64_t>::value) {
            uint32_t spaceSize = min((uint32_t)(W * sizeof(T)), (uint32_t)(40) * 1024 / BufferNum_sca);
            this->n_elements_per_iter = spaceSize / sizeof(T);
            this->smallLoopTimes = (W + n_elements_per_iter - 1) / n_elements_per_iter;
            pipe->InitBuffer(inputBuf, BufferNum_sca, spaceSize);
            pipe->InitBuffer(outBuf, BufferNum_sca, spaceSize + 32);
            pipe->InitBuffer(tmpBuf, BufferNum_sca, spaceSize);
        } else {
            uint32_t spaceSize = min((uint32_t)(W * sizeof(T)), (uint32_t)(63) * 1024 / BufferNum_sca);
            this->n_elements_per_iter = spaceSize / sizeof(T);
            this->smallLoopTimes = (W + n_elements_per_iter - 1) / n_elements_per_iter;
            pipe->InitBuffer(inputBuf, BufferNum_sca, spaceSize);
            pipe->InitBuffer(outBuf, BufferNum_sca, spaceSize + 32);
        }
    }

    __aicore__ inline void Process_iter(uint32_t offset, uint32_t outputIdxOffset) {
        uint32_t iterSize = n_elements_per_iter;
        const T scaleValue = scaleGm.GetValue(offset);
        const T biasValue = hasBias ? biasGm.GetValue(offset) : static_cast<T>(0);

        uint32_t outputIdx = 0;
        for (uint32_t i = 0; i < smallLoopTimes; ++i) {
            iterSize = min(n_elements_per_iter, W - outputIdx);
            uint32_t half_iterSize = iterSize / 2 / 16 * 16;

            uint16_t blockCount = 1;
            uint32_t blockLen = iterSize * sizeof(T);
            DataCopyExtParams copyParams{blockCount, blockLen, 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

            LocalTensor<T> inputLocal = inputBuf.AllocTensor<T>();
            DataCopyPad(inputLocal, inputGm[outputIdx + outputIdxOffset], copyParams, padParams);
            inputBuf.EnQue<T>(inputLocal);
            inputLocal = inputBuf.DeQue<T>();
            LocalTensor<T> outLocal = outBuf.AllocTensor<T>();

            if constexpr (std::is_same<T, float>::value || std::is_same<T, half>::value || std::is_same<T, int32_t>::value || std::is_same<T, int16_t>::value ||
                          std::is_same<T, bool>::value) {
                ComputeScalar_iter(outLocal, inputLocal, scaleValue, biasValue, iterSize, hasBias);
            } else if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
                LocalTensor<half> inputLocal_fp16 = tmpBuf.AllocTensor<half>();
                LocalTensor<half> outLocal_fp16 = inputLocal.template ReinterpretCast<half>();
                Cast(inputLocal_fp16, inputLocal, RoundMode::CAST_NONE, iterSize);
                ComputeScalar_iter(outLocal_fp16, inputLocal_fp16, static_cast<half>(scaleValue), static_cast<half>(biasValue), half_iterSize, hasBias);
                Cast(outLocal, outLocal_fp16, RoundMode::CAST_NONE, half_iterSize);
                ComputeScalar_iter(outLocal_fp16, inputLocal_fp16[half_iterSize], static_cast<half>(scaleValue), static_cast<half>(biasValue),
                                   iterSize - half_iterSize, hasBias);
                Cast(outLocal[half_iterSize], outLocal_fp16, RoundMode::CAST_NONE, iterSize - half_iterSize);
                tmpBuf.FreeTensor<half>(inputLocal_fp16);
            } else if constexpr (std::is_same<T, bfloat16_t>::value) {
                LocalTensor<float> inputLocal_fp32 = tmpBuf.AllocTensor<float>();
                LocalTensor<float> outLocal_fp32 = inputLocal_fp32[(iterSize + 7) / 8 * 8];
                Cast(inputLocal_fp32, inputLocal, RoundMode::CAST_NONE, iterSize);
                ComputeScalar_iter(outLocal_fp32, inputLocal_fp32, ToFloat(scaleValue), hasBias ? ToFloat(biasValue) : 0.0f, half_iterSize, hasBias);
                Cast(outLocal, outLocal_fp32, RoundMode::CAST_RINT, half_iterSize);
                ComputeScalar_iter(outLocal_fp32, inputLocal_fp32[half_iterSize], ToFloat(scaleValue), hasBias ? ToFloat(biasValue) : 0.0f,
                                   iterSize - half_iterSize, hasBias);
                Cast(outLocal[half_iterSize], outLocal_fp32, RoundMode::CAST_RINT, iterSize - half_iterSize);
                tmpBuf.FreeTensor<float>(inputLocal_fp32);
            }
            inputBuf.FreeTensor<T>(inputLocal);
            outBuf.EnQue<T>(outLocal);
            outLocal = outBuf.DeQue<T>();
            DataCopyExtParams storeParams{blockCount, blockLen, 0, 0, 0};
            DataCopyPad(outGm[outputIdx + outputIdxOffset], outLocal, storeParams);
            outBuf.FreeTensor<T>(outLocal);

            outputIdx += n_elements_per_iter;
        }
    }

    template <typename U>
    __aicore__ inline void Compute_iter(const LocalTensor<U> &outLocal, const LocalTensor<U> &inputLocal, const LocalTensor<U> &scaleLocal,
                                        const LocalTensor<U> &biasLocal, uint32_t iterSize, bool hasBias) {
        if constexpr (std::is_same<U, bool>::value) {
            LocalTensor<uint32_t> inputLocal_uint16 = inputLocal.template ReinterpretCast<uint32_t>();
            LocalTensor<uint32_t> scaleLocal_uint16 = scaleLocal.template ReinterpretCast<uint32_t>();
            LocalTensor<uint32_t> outLocal_uint16 = outLocal.template ReinterpretCast<uint32_t>();
            And(outLocal_uint16, inputLocal_uint16, scaleLocal_uint16, iterSize / 2);
        } else {
            Mul(outLocal, inputLocal, scaleLocal, iterSize);
            if (hasBias) {
                Add(outLocal, outLocal, biasLocal, iterSize);
            }
        }
    }

    template <typename U, typename V>
    __aicore__ inline void ComputeScalar_iter(const LocalTensor<U> &outLocal, const LocalTensor<U> &inputLocal, V scaleValue,
                                              V biasValue, uint32_t iterSize, bool hasBias) {
        Muls(outLocal, inputLocal, scaleValue, iterSize);
        if (hasBias) {
            Adds(outLocal, outLocal, biasValue, iterSize);
        }
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < size; ++i) {
            Process_iter(i, i * W);
        }
    }

private:
    TPipe *pipe;
    bool hasBias = false;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> scaleGm;
    GlobalTensor<T> biasGm;
    GlobalTensor<T> outGm;
    TQue<QuePosition::VECIN, 1> inputBuf;
    TQue<QuePosition::VECOUT, 1> outBuf;
    TQue<QuePosition::VECCALC, 1> tmpBuf;
    uint32_t size;
    uint32_t smallLoopTimes;
    uint32_t n_elements_per_iter;

    uint32_t totalSize;
    uint32_t inputSize;
    uint32_t otherSize;

    uint32_t *mmInputDims;
    uint32_t *mmOtherDims;
    uint32_t *mmOutputDims;
    int nOutputDims;
    uint32_t beginIndex;
    uint32_t max_size_for_broadcast;
    int D;
    int H;
    int W;
};
