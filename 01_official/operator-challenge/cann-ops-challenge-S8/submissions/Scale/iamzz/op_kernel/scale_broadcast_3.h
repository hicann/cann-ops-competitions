#include "kernel_operator.h"
using namespace AscendC;
#include <type_traits>

constexpr uint32_t BufferNum_broadcast_3 = 1;

__aicore__ inline uint32_t FindClosestPowerOfTwo(uint32_t n) {
    constexpr uint32_t totalShiftBits = 63;
    return totalShiftBits - ScalarCountLeadingZero(n);
}

template <typename T>
__aicore__ inline void BinaryBroadcastFirstAxis(const LocalTensor<T> &dstTensor,
                                                const LocalTensor<T> &srcTensor,
                                                uint32_t firstAxis,
                                                uint32_t lastAxis) {
    uint32_t k = FindClosestPowerOfTwo(firstAxis);
    uint32_t splitK = 1 << k;
    uint32_t remain = firstAxis - splitK;

    LocalTensor<T> currBuff = dstTensor;
    if (splitK >= 1) {
        Adds(currBuff, srcTensor, static_cast<T>(0.0), lastAxis);
    }
    uint32_t stride = 1;
    while (stride < splitK) {
        uint32_t copyRows = stride;
        Adds<T>(currBuff[stride * lastAxis], currBuff, static_cast<T>(0.0), lastAxis * copyRows);
        stride <<= 1;
    }
    if (remain > 0) {
        Adds(currBuff[splitK * lastAxis], currBuff, static_cast<T>(0.0), lastAxis * remain);
    }
}

// Xt5-style 3D broadcast kernel. This file is kept aligned with Fmin's
// broadcast_3 structure and is intended for patterns where scale/bias can be
// read as [D, 1, W] rows.
template <typename T>
class KernelScale_broadcast_3 {
public:
    __aicore__ inline KernelScale_broadcast_3() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR out,
                                uint32_t smallSize, uint32_t incSize, uint32_t formerNum,
                                uint32_t totalSize, uint32_t *mmInputDims, uint32_t *mmOtherDims,
                                uint32_t *mmOutputDims, uint8_t nOutputDims, bool hasBias,
                                TPipe *pipeIn) {
        this->pipe = pipeIn;
        this->hasBias = hasBias;
        this->mmInputDims = mmInputDims;
        this->mmOtherDims = mmOtherDims;
        this->mmOutputDims = mmOutputDims;
        this->nOutputDims = nOutputDims;
        this->totalSize = totalSize;

        this->D = mmOutputDims[0];
        this->H = mmOutputDims[1];
        this->W = mmOutputDims[2];

        this->beginIndex = 0;
        if (GetBlockIdx() < formerNum) {
            this->size = smallSize + incSize;
            beginIndex = this->size * GetBlockIdx();
        } else {
            this->size = smallSize;
            beginIndex = this->size * GetBlockIdx() + formerNum * incSize;
        }
        if (beginIndex + size > D) {
            this->size = D - beginIndex;
        }

        inputGm.SetGlobalBuffer((__gm__ T *)input + beginIndex * H * W, totalSize);
        scaleGm.SetGlobalBuffer((__gm__ T *)scale + beginIndex * W, totalSize);
        if (hasBias) {
            biasGm.SetGlobalBuffer((__gm__ T *)bias + beginIndex * W, totalSize);
        }
        outGm.SetGlobalBuffer((__gm__ T *)out + beginIndex * H * W, totalSize);

        realW = (W * sizeof(T) + 31U) / 32U * 32U / sizeof(T);
        uint32_t ubSize = 63U * 1024U;
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            ubSize = 27U * 1024U;
        }
        if (hasBias) {
            ubSize = 27U * 1024U;
        }
        uint32_t maxRows = ubSize / (realW * sizeof(T)) / BufferNum_broadcast_3;
        nLinesPerIter = (H > maxRows) ? maxRows : H;
        if (nLinesPerIter == 0U) {
            nLinesPerIter = 1U;
        }
        this->nElementsPerIter = nLinesPerIter * realW;
        uint32_t spaceSize = nElementsPerIter * sizeof(T);
        this->bigLoopTimes = size;
        this->smallLoopTimes = (H + nLinesPerIter - 1U) / nLinesPerIter;

        pipe->InitBuffer(inputBuf, BufferNum_broadcast_3, spaceSize + 256U);
        pipe->InitBuffer(scaleBuf, BufferNum_broadcast_3, spaceSize + 256U);
        if (hasBias) {
            pipe->InitBuffer(biasBuf, BufferNum_broadcast_3, spaceSize + 256U);
        }
        pipe->InitBuffer(outBuf, BufferNum_broadcast_3, spaceSize + 256U);
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            pipe->InitBuffer(tmpBuf, BufferNum_broadcast_3, spaceSize * 4U + 256U);
        }
    }

    __aicore__ inline void Process_iter(uint32_t scaleOffset) {
        DataCopyExtParams copyParams{(uint16_t)1, (uint32_t)(W * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        LocalTensor<T> scaleLocal = scaleBuf.AllocTensor<T>();
        DataCopyPad(scaleLocal, scaleGm[scaleOffset], copyParams, padParams);
        scaleBuf.EnQue<T>(scaleLocal);
        scaleLocal = scaleBuf.DeQue<T>();

        LocalTensor<T> biasLocal;
        if (hasBias) {
            biasLocal = biasBuf.AllocTensor<T>();
            DataCopyPad(biasLocal, biasGm[scaleOffset], copyParams, padParams);
            biasBuf.EnQue<T>(biasLocal);
            biasLocal = biasBuf.DeQue<T>();
        }

        if constexpr (std::is_same<T, bfloat16_t>::value) {
            BinaryBroadcastFirstAxis(scaleLocal.template ReinterpretCast<half>(), scaleLocal.template ReinterpretCast<half>(), nLinesPerIter, realW);
            if (hasBias) {
                BinaryBroadcastFirstAxis(biasLocal.template ReinterpretCast<half>(), biasLocal.template ReinterpretCast<half>(), nLinesPerIter, realW);
            }
        } else {
            BinaryBroadcastFirstAxis(scaleLocal, scaleLocal, nLinesPerIter, realW);
            if (hasBias) {
                BinaryBroadcastFirstAxis(biasLocal, biasLocal, nLinesPerIter, realW);
            }
        }

        uint32_t currLine = 0;
        for (uint32_t i = 0; i < smallLoopTimes; ++i) {
            uint32_t iterSize = min((uint32_t)nLinesPerIter, (uint32_t)(H - currLine)) * realW;
            uint32_t halfIterSize = iterSize / 2U / 16U * 16U;

            LocalTensor<T> inputLocal = inputBuf.AllocTensor<T>();
            uint32_t outputIdx = currLine * W + scaleOffset * H;
            uint16_t blockCount = iterSize / realW;
            uint32_t blockLen = W * sizeof(T);
            DataCopyExtParams inParams{blockCount, blockLen, 0, 0, 0};
            DataCopyPadExtParams<T> inPad{false, 0, 0, 0};
            DataCopyPad(inputLocal, inputGm[outputIdx], inParams, inPad);
            inputBuf.EnQue<T>(inputLocal);
            inputLocal = inputBuf.DeQue<T>();

            LocalTensor<T> outLocal = outBuf.AllocTensor<T>();
            if constexpr (std::is_same<T, float>::value || std::is_same<T, half>::value) {
                Compute_iter(outLocal, inputLocal, scaleLocal, biasLocal, iterSize, hasBias);
            } else if constexpr (std::is_same<T, bfloat16_t>::value) {
                LocalTensor<float> inputFp32 = tmpBuf.AllocTensor<float>();
                LocalTensor<float> scaleFp32 = inputFp32[(iterSize + 7U) / 8U * 8U];
                LocalTensor<float> biasFp32 = scaleFp32[(iterSize + 7U) / 8U * 8U];
                LocalTensor<float> outFp32 = inputLocal.template ReinterpretCast<float>();
                Cast(inputFp32, inputLocal, RoundMode::CAST_NONE, iterSize);
                Cast(scaleFp32, scaleLocal, RoundMode::CAST_NONE, iterSize);
                if (hasBias) {
                    Cast(biasFp32, biasLocal, RoundMode::CAST_NONE, iterSize);
                }
                Compute_iter(outFp32, inputFp32, scaleFp32, biasFp32, halfIterSize, hasBias);
                Cast(outLocal, outFp32, RoundMode::CAST_RINT, halfIterSize);
                Compute_iter(outFp32, inputFp32[halfIterSize], scaleFp32[halfIterSize], biasFp32[halfIterSize],
                             iterSize - halfIterSize, hasBias);
                Cast(outLocal[halfIterSize], outFp32, RoundMode::CAST_RINT, iterSize - halfIterSize);
                tmpBuf.FreeTensor<float>(inputFp32);
            }

            inputBuf.FreeTensor<T>(inputLocal);
            outBuf.EnQue<T>(outLocal);
            outLocal = outBuf.DeQue<T>();
            DataCopyExtParams outParams{blockCount, blockLen, 0, 0, 0};
            DataCopyPad(outGm[outputIdx], outLocal, outParams);
            outBuf.FreeTensor<T>(outLocal);
            currLine += nLinesPerIter;
        }

        scaleBuf.FreeTensor<T>(scaleLocal);
        if (hasBias) {
            biasBuf.FreeTensor<T>(biasLocal);
        }
    }

    template <typename U>
    __aicore__ inline void Compute_iter(const LocalTensor<U> &outLocal, const LocalTensor<U> &inputLocal,
                                        const LocalTensor<U> &scaleLocal, const LocalTensor<U> &biasLocal,
                                        uint32_t iterSize, bool hasBias) {
        Mul(outLocal, inputLocal, scaleLocal, iterSize);
        if (hasBias) {
            Add(outLocal, outLocal, biasLocal, iterSize);
        }
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < bigLoopTimes; ++i) {
            Process_iter(i * W);
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
    TQue<QuePosition::VECIN, 1> scaleBuf;
    TQue<QuePosition::VECIN, 1> biasBuf;
    TQue<QuePosition::VECOUT, 1> outBuf;
    TQue<QuePosition::VECCALC, 1> tmpBuf;

    uint32_t size;
    uint32_t bigLoopTimes;
    uint32_t smallLoopTimes;
    uint32_t nElementsPerIter;

    uint32_t totalSize;
    uint32_t *mmInputDims;
    uint32_t *mmOtherDims;
    uint32_t *mmOutputDims;
    int nOutputDims;
    uint32_t beginIndex;
    uint32_t D;
    uint32_t H;
    uint32_t W;
    uint32_t realW;
    uint32_t nLinesPerIter;
};
