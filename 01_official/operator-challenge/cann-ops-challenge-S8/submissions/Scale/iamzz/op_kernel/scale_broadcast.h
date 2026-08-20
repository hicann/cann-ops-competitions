#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t BufferNum_broadcast = 1;
// scale张量始终是需要广播的张量,只考虑了二维行广播
template <typename T>
class KernelScale_broadcast {
public:
    __aicore__ inline KernelScale_broadcast() {}
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
        if (beginIndex + size > W) {
            this->size = W - beginIndex;
        }
        inputGm.SetGlobalBuffer((__gm__ T *)input + beginIndex, totalSize);
        scaleGm.SetGlobalBuffer((__gm__ T *)scale + beginIndex, totalSize);
        if (hasBias) {
            biasGm.SetGlobalBuffer((__gm__ T *)bias + beginIndex, totalSize);
        }
        outGm.SetGlobalBuffer((__gm__ T *)out + beginIndex, totalSize);

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

    __aicore__ inline void Process_iter(uint32_t offset, uint32_t iterSize) {
        uint16_t blockCount = 1;
        uint32_t blockLen = iterSize * sizeof(T);
        DataCopyExtParams copyParams{blockCount, blockLen, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        LocalTensor<T> scaleLocal = scaleBuf.AllocTensor<T>();
        DataCopyPad(scaleLocal, scaleGm[offset], copyParams, padParams);
        scaleBuf.EnQue<T>(scaleLocal);
        scaleLocal = scaleBuf.DeQue<T>();

        LocalTensor<T> biasLocal;
        if (hasBias) {
            biasLocal = biasBuf.AllocTensor<T>();
            DataCopyPad(biasLocal, biasGm[offset], copyParams, padParams);
            biasBuf.EnQue<T>(biasLocal);
            biasLocal = biasBuf.DeQue<T>();
        }

        uint32_t half_iterSize = iterSize / 2 / 16 * 16;
        for (int32_t i = 0; i < H; i++) {
            LocalTensor<T> inputLocal = inputBuf.AllocTensor<T>();
            uint32_t outputIdx = i * W + offset;
            DataCopyPad(inputLocal, inputGm[outputIdx], copyParams, padParams);

            inputBuf.EnQue<T>(inputLocal);
            inputLocal = inputBuf.DeQue<T>();
            LocalTensor<T> outLocal = outBuf.AllocTensor<T>();

            if constexpr (std::is_same<T, float>::value || std::is_same<T, half>::value || std::is_same<T, int32_t>::value || std::is_same<T, int16_t>::value ||
                          std::is_same<T, bool>::value) {
                Compute_iter(outLocal, inputLocal, scaleLocal, biasLocal, iterSize, hasBias);
            } else if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
                LocalTensor<half> inputLocal_fp16 = tmpBuf.AllocTensor<half>();
                LocalTensor<half> scaleLocal_fp16 = inputLocal_fp16[(iterSize + 15) / 16 * 16];
                LocalTensor<half> biasLocal_fp16 = scaleLocal_fp16[(iterSize + 15) / 16 * 16];
                LocalTensor<half> outLocal_fp16 = inputLocal.template ReinterpretCast<half>();
                Cast(inputLocal_fp16, inputLocal, RoundMode::CAST_NONE, iterSize);
                Cast(scaleLocal_fp16, scaleLocal, RoundMode::CAST_NONE, iterSize);
                if (hasBias) {
                    Cast(biasLocal_fp16, biasLocal, RoundMode::CAST_NONE, iterSize);
                }
                Compute_iter(outLocal_fp16, inputLocal_fp16, scaleLocal_fp16, biasLocal_fp16, half_iterSize, hasBias);
                Cast(outLocal, outLocal_fp16, RoundMode::CAST_NONE, half_iterSize);
                Compute_iter(outLocal_fp16, inputLocal_fp16[half_iterSize], scaleLocal_fp16[half_iterSize], biasLocal_fp16[half_iterSize],
                             iterSize - half_iterSize, hasBias);
                Cast(outLocal[half_iterSize], outLocal_fp16, RoundMode::CAST_NONE, iterSize - half_iterSize);
                tmpBuf.FreeTensor<half>(inputLocal_fp16);
            } else if constexpr (std::is_same<T, bfloat16_t>::value) {
                LocalTensor<float> inputLocal_fp32 = tmpBuf.AllocTensor<float>();
                LocalTensor<float> scaleLocal_fp32 = inputLocal_fp32[(iterSize + 7) / 8 * 8];
                LocalTensor<float> biasLocal_fp32 = scaleLocal_fp32[(iterSize + 7) / 8 * 8];
                LocalTensor<float> outLocal_fp32 = inputLocal.template ReinterpretCast<float>();
                Cast(inputLocal_fp32, inputLocal, RoundMode::CAST_NONE, iterSize);
                Cast(scaleLocal_fp32, scaleLocal, RoundMode::CAST_NONE, iterSize);
                if (hasBias) {
                    Cast(biasLocal_fp32, biasLocal, RoundMode::CAST_NONE, iterSize);
                }
                Compute_iter(outLocal_fp32, inputLocal_fp32, scaleLocal_fp32, biasLocal_fp32, half_iterSize, hasBias);
                Cast(outLocal, outLocal_fp32, RoundMode::CAST_RINT, half_iterSize);
                Compute_iter(outLocal_fp32, inputLocal_fp32[half_iterSize], scaleLocal_fp32[half_iterSize], biasLocal_fp32[half_iterSize],
                             iterSize - half_iterSize, hasBias);
                Cast(outLocal[half_iterSize], outLocal_fp32, RoundMode::CAST_RINT, iterSize - half_iterSize);
                tmpBuf.FreeTensor<float>(inputLocal_fp32);
            }

            inputBuf.FreeTensor<T>(inputLocal);
            outBuf.EnQue<T>(outLocal);
            outLocal = outBuf.DeQue<T>();
            DataCopyExtParams storeParams{blockCount, blockLen, 0, 0, 0};
            DataCopyPad(outGm[outputIdx], outLocal, storeParams);
            outBuf.FreeTensor<T>(outLocal);
        }

        scaleBuf.FreeTensor<T>(scaleLocal);
        if (hasBias) {
            biasBuf.FreeTensor<T>(biasLocal);
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

    __aicore__ inline void Process() {
        uint32_t blockBeginIndex = 0;
        for (uint32_t i = 0; i < smallLoopTimes - 1; ++i) {
            Process_iter(blockBeginIndex, n_elements_per_iter);
            blockBeginIndex += n_elements_per_iter;
        }
        uint32_t iterSize = min(n_elements_per_iter, size - blockBeginIndex);
        Process_iter(blockBeginIndex, iterSize);
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
    int D;
    int H;
    int W;
};
