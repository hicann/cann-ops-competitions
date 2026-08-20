#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t BufferNum_sca = 1;

template <typename T>
class KernelScale_sca {
public:
    __aicore__ inline KernelScale_sca() {}
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
        this->beginIndex = 0;
        if (GetBlockIdx() < formerNum) {
            this->size = smallSize + incSize;
            beginIndex = this->size * GetBlockIdx();
        } else {
            this->size = smallSize;
            beginIndex = this->size * GetBlockIdx() + formerNum * incSize;
        }
        this->inputSize = 1;
        this->otherSize = 1;
        for (int i = 0; i < nOutputDims; i++) {
            this->inputSize *= mmInputDims[i];
            this->otherSize *= mmOtherDims[i];
        }
        inputGm.SetGlobalBuffer((__gm__ T *)input, totalSize);
        scaleGm.SetGlobalBuffer((__gm__ T *)scale, totalSize);
        if (hasBias) {
            biasGm.SetGlobalBuffer((__gm__ T *)bias, totalSize);
        }
        outGm.SetGlobalBuffer((__gm__ T *)out, totalSize);
        if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value || std::is_same<T, bfloat16_t>::value) {
            uint32_t spaceSize = min((uint32_t)(size * sizeof(T)), (uint32_t)(27) * 1024 / BufferNum_sca);
            this->n_elements_per_iter = spaceSize / sizeof(T);
            this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;
            pipe->InitBuffer(inputBuf, BufferNum_sca, spaceSize);
            pipe->InitBuffer(scaleBuf, BufferNum_sca, spaceSize + 32);
            if (hasBias) {
                pipe->InitBuffer(biasBuf, BufferNum_sca, spaceSize + 32);
            }
            pipe->InitBuffer(outBuf, BufferNum_sca, spaceSize + 32);
            pipe->InitBuffer(tmpBuf, BufferNum_sca, spaceSize * 4);
        } else if (std::is_same<T, int64_t>::value) {
            uint32_t spaceSize = min((uint32_t)(size * sizeof(T)), (uint32_t)(40) * 1024 / BufferNum_sca);
            this->n_elements_per_iter = spaceSize / sizeof(T);
            this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;
            pipe->InitBuffer(inputBuf, BufferNum_sca, spaceSize);
            pipe->InitBuffer(scaleBuf, BufferNum_sca, spaceSize + 32);
            if (hasBias) {
                pipe->InitBuffer(biasBuf, BufferNum_sca, spaceSize + 32);
            }
            pipe->InitBuffer(outBuf, BufferNum_sca, spaceSize + 32);
            pipe->InitBuffer(tmpBuf, BufferNum_sca, spaceSize);
        } else {
            uint32_t spaceSize = min((uint32_t)(size * sizeof(T)), (uint32_t)(63) * 1024 / BufferNum_sca);
            this->n_elements_per_iter = spaceSize / sizeof(T);
            this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;
            pipe->InitBuffer(inputBuf, BufferNum_sca, spaceSize);
            pipe->InitBuffer(scaleBuf, BufferNum_sca, spaceSize + 32);
            if (hasBias) {
                pipe->InitBuffer(biasBuf, BufferNum_sca, spaceSize + 32);
            }
            pipe->InitBuffer(outBuf, BufferNum_sca, spaceSize + 32);
        }
    }

    __aicore__ inline int mapIndex(int outputIdx, uint32_t *inputShape, uint32_t *outputShape, const int nDims) {
        int inputIdx = 0;
        int stride = 1;
        int remainingIdx = outputIdx;
        for (int dim = nDims - 1; dim >= 0; --dim) {
            int coord = remainingIdx % outputShape[dim];
            remainingIdx = remainingIdx / outputShape[dim];
            if (inputShape[dim] != 1) {
                inputIdx += coord * stride;
                stride *= inputShape[dim];
            }
        }
        return inputIdx;
    }

    __aicore__ inline void CopyInBroadcastSca(const GlobalTensor<T> &srcGm, const LocalTensor<T> &inputLocal, uint32_t offset, uint32_t iterSize,
                                              uint32_t *inputShape, uint32_t *outputShape, const int nDims) {
        for (uint32_t i = 0; i < iterSize; i++) {
            int outputIdx = i + offset;
            int inputIdx = mapIndex(outputIdx, inputShape, outputShape, nDims);
            T val = srcGm.GetValue(inputIdx);
            inputLocal(i) = val;
        }
    }

    __aicore__ inline void Process_iter(uint32_t offset, uint32_t iterSize) {
        LocalTensor<T> inputLocal = inputBuf.AllocTensor<T>();
        LocalTensor<T> scaleLocal = scaleBuf.AllocTensor<T>();
        LocalTensor<T> biasLocal;
        if (hasBias) {
            biasLocal = biasBuf.AllocTensor<T>();
        }

        if (inputSize != totalSize) {
            CopyInBroadcastSca(inputGm, inputLocal, offset, iterSize, mmInputDims, mmOutputDims, nOutputDims);
        } else {
            DataCopy(inputLocal, inputGm[offset], iterSize);
        }
        if (otherSize != totalSize) {
            CopyInBroadcastSca(scaleGm, scaleLocal, offset, iterSize, mmOtherDims, mmOutputDims, nOutputDims);
            if (hasBias) {
                CopyInBroadcastSca(biasGm, biasLocal, offset, iterSize, mmOtherDims, mmOutputDims, nOutputDims);
            }
        } else {
            DataCopy(scaleLocal, scaleGm[offset], iterSize);
            if (hasBias) {
                DataCopy(biasLocal, biasGm[offset], iterSize);
            }
        }

        inputBuf.EnQue<T>(inputLocal);
        scaleBuf.EnQue<T>(scaleLocal);
        if (hasBias) {
            biasBuf.EnQue<T>(biasLocal);
        }
        inputLocal = inputBuf.DeQue<T>();
        scaleLocal = scaleBuf.DeQue<T>();
        if (hasBias) {
            biasLocal = biasBuf.DeQue<T>();
        }

        LocalTensor<T> outLocal = outBuf.AllocTensor<T>();
        if constexpr (std::is_same<T, float>::value || std::is_same<T, half>::value || std::is_same<T, int32_t>::value || std::is_same<T, int16_t>::value ||
                      std::is_same<T, bool>::value) {
            Compute_iter(outLocal, inputLocal, scaleLocal, biasLocal, iterSize, hasBias);
        } else if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
            LocalTensor<half> inputLocal_fp16 = tmpBuf.AllocTensor<half>();
            LocalTensor<half> scaleLocal_fp16 = inputLocal_fp16[iterSize];
            LocalTensor<half> biasLocal_fp16 = scaleLocal_fp16[iterSize];
            LocalTensor<half> outLocal_fp16 = inputLocal.template ReinterpretCast<half>();
            Cast(inputLocal_fp16, inputLocal, RoundMode::CAST_NONE, iterSize);
            Cast(scaleLocal_fp16, scaleLocal, RoundMode::CAST_NONE, iterSize);
            if (hasBias) {
                Cast(biasLocal_fp16, biasLocal, RoundMode::CAST_NONE, iterSize);
            }
            Compute_iter(outLocal_fp16, inputLocal_fp16, scaleLocal_fp16, biasLocal_fp16, iterSize, hasBias);
            Cast(outLocal, outLocal_fp16, RoundMode::CAST_NONE, iterSize);
            tmpBuf.FreeTensor<half>(inputLocal_fp16);
        } else if constexpr (std::is_same<T, bfloat16_t>::value) {
            LocalTensor<float> inputLocal_fp32 = tmpBuf.AllocTensor<float>();
            LocalTensor<float> scaleLocal_fp32 = inputLocal_fp32[iterSize];
            LocalTensor<float> biasLocal_fp32 = scaleLocal_fp32[iterSize];
            LocalTensor<float> outLocal_fp32 = inputLocal.template ReinterpretCast<float>();
            Cast(inputLocal_fp32, inputLocal, RoundMode::CAST_NONE, iterSize);
            Cast(scaleLocal_fp32, scaleLocal, RoundMode::CAST_NONE, iterSize);
            if (hasBias) {
                Cast(biasLocal_fp32, biasLocal, RoundMode::CAST_NONE, iterSize);
            }
            Compute_iter(outLocal_fp32, inputLocal_fp32, scaleLocal_fp32, biasLocal_fp32, iterSize, hasBias);
            Cast(outLocal, outLocal_fp32, RoundMode::CAST_RINT, iterSize);
            tmpBuf.FreeTensor<float>(inputLocal_fp32);
        }

        inputBuf.FreeTensor<T>(inputLocal);
        scaleBuf.FreeTensor<T>(scaleLocal);
        if (hasBias) {
            biasBuf.FreeTensor<T>(biasLocal);
        }

        outBuf.EnQue<T>(outLocal);
        outLocal = outBuf.DeQue<T>();
        DataCopy(outGm[offset], outLocal, iterSize);
        outBuf.FreeTensor<T>(outLocal);
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

    __aicore__ inline void Process() {
        uint32_t blockBeginIndex = 0;
        for (uint32_t i = 0; i < smallLoopTimes - 1; ++i) {
            Process_iter(blockBeginIndex + beginIndex, n_elements_per_iter);
            blockBeginIndex += n_elements_per_iter;
        }
        uint32_t iterSize = min(n_elements_per_iter, size - blockBeginIndex);
        Process_iter(blockBeginIndex + beginIndex, iterSize);
    }

private:
    TPipe *pipe;
    bool hasBias = false;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> scaleGm;
    GlobalTensor<T> biasGm;
    GlobalTensor<T> outGm;
    TQue<QuePosition::VECIN, 1> inputBuf;
    TQue<QuePosition::VECIN, 1> scaleBuf;
    TQue<QuePosition::VECIN, 1> biasBuf;
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
};
