#include "kernel_operator.h"
using namespace AscendC;
#include <type_traits>

constexpr uint32_t BufferNum_broadcast_3_midaxis = 1;

__aicore__ inline uint32_t FindClosestPowerOfTwo_midaxis(uint32_t n) {
    constexpr uint32_t totalShiftBits = 63;
    return totalShiftBits - ScalarCountLeadingZero(n);
}

template <typename T>
__aicore__ inline void BinaryBroadcastFirstAxis_midaxis(const LocalTensor<T> &dstTensor,
                                                        const LocalTensor<T> &srcTensor,
                                                        uint32_t firstAxis,
                                                        uint32_t lastAxis) {
    uint32_t k = FindClosestPowerOfTwo_midaxis(firstAxis);
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

// Scale-specific 3D broadcast:
//   input/output: [D, H, W]
//   scale/bias:   [1, H, 1]
// The host splits H across cores. A broadcasted H-tile is reused for all D
// slices owned by the same core.
template <typename T>
class KernelScale_broadcast_3_midaxis {
public:
    __aicore__ inline KernelScale_broadcast_3_midaxis() {}

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
        this->rowCount = H;

        this->beginRow = 0;
        if (GetBlockIdx() < formerNum) {
            this->size = smallSize + incSize;
            beginRow = this->size * GetBlockIdx();
        } else {
            this->size = smallSize;
            beginRow = this->size * GetBlockIdx() + formerNum * incSize;
        }
        if (beginRow >= rowCount) {
            this->size = 0;
        } else if (beginRow + size > rowCount) {
            this->size = rowCount - beginRow;
        }

        inputGm.SetGlobalBuffer((__gm__ T *)input, totalSize);
        scaleGm.SetGlobalBuffer((__gm__ T *)scale, H);
        if (hasBias) {
            biasGm.SetGlobalBuffer((__gm__ T *)bias, H);
        }
        outGm.SetGlobalBuffer((__gm__ T *)out, totalSize);

        realW = W;
        uint32_t ubPerQueue = 63U * 1024U;
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            ubPerQueue = 27U * 1024U;
        } else {
            ubPerQueue = hasBias ? (44U * 1024U) : (52U * 1024U);
        }
        uint32_t maxRows = ubPerQueue / (realW * sizeof(T)) / BufferNum_broadcast_3_midaxis;
        if (maxRows == 0U) {
            maxRows = 1U;
        }
        if ((W * sizeof(T)) % 32U != 0U && maxRows >= 16U) {
            maxRows = maxRows / 16U * 16U;
        }
        nLinesPerIter = size > maxRows ? maxRows : size;
        if (nLinesPerIter == 0U) {
            nLinesPerIter = 1U;
        }
        nElementsPerIter = nLinesPerIter * realW;
        smallLoopTimes = size == 0U ? 0U : (size + nLinesPerIter - 1U) / nLinesPerIter;

        uint32_t spaceSize = nElementsPerIter * sizeof(T);
        pipe->InitBuffer(inputBuf, BufferNum_broadcast_3_midaxis, spaceSize + 256U);
        pipe->InitBuffer(scaleBuf, BufferNum_broadcast_3_midaxis, spaceSize + 256U);
        pipe->InitBuffer(compactBuf, BufferNum_broadcast_3_midaxis, nLinesPerIter * sizeof(T) + 256U);
        if (hasBias) {
            pipe->InitBuffer(biasBuf, BufferNum_broadcast_3_midaxis, spaceSize + 256U);
        }
        pipe->InitBuffer(outBuf, BufferNum_broadcast_3_midaxis, spaceSize + 256U);
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            pipe->InitBuffer(tmpBuf, BufferNum_broadcast_3_midaxis, spaceSize * 4U + 256U);
        } else {
            constexpr uint32_t oneBlockElems = 32U / sizeof(T);
            const uint32_t alignedW = (W + oneBlockElems - 1U) / oneBlockElems * oneBlockElems;
            const uint32_t brcbTmpElems = oneBlockElems * oneBlockElems;
            const uint32_t copyTmpElems = ((W * sizeof(T)) % 32U == 0U) ? 0U : oneBlockElems * alignedW;
            const uint32_t compactBytes = (nLinesPerIter * sizeof(T) + 31U) / 32U * 32U;
            const uint32_t broadcastTmpBytes = (brcbTmpElems + copyTmpElems) * sizeof(T);
            pipe->InitBuffer(tmpBuf, BufferNum_broadcast_3_midaxis, compactBytes + broadcastTmpBytes + 256U);
        }
    }

    __aicore__ inline void Process_iter(uint32_t startH, uint32_t rows) {
        const uint32_t iterSize = rows * realW;
        const uint32_t halfIterSize = iterSize / 2U / 16U * 16U;

        if constexpr (std::is_same<T, float>::value || std::is_same<T, half>::value) {
            LocalTensor<T> scaleLocal = scaleBuf.AllocTensor<T>();
            LocalTensor<T> biasLocal;
            if (hasBias) {
                biasLocal = biasBuf.AllocTensor<T>();
            }

            LocalTensor<uint8_t> tmpLocal = tmpBuf.AllocTensor<uint8_t>();
            MaterializeParamBroadcast_midaxis(scaleLocal, scaleGm, tmpLocal, startH, rows);
            if (hasBias) {
                MaterializeParamBroadcast_midaxis(biasLocal, biasGm, tmpLocal, startH, rows);
            }
            tmpBuf.FreeTensor<uint8_t>(tmpLocal);

            for (uint32_t dIdx = 0; dIdx < D; ++dIdx) {
                const uint32_t inputOffset = (dIdx * H + startH) * W;
                LocalTensor<T> inputLocal = inputBuf.AllocTensor<T>();
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(iterSize * sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                DataCopyPad(inputLocal, inputGm[inputOffset], copyParams, padParams);
                inputBuf.EnQue<T>(inputLocal);
                inputLocal = inputBuf.DeQue<T>();

                LocalTensor<T> outLocal = outBuf.AllocTensor<T>();
                Compute_iter(outLocal, inputLocal, scaleLocal, biasLocal, iterSize, hasBias);
                inputBuf.FreeTensor<T>(inputLocal);

                outBuf.EnQue<T>(outLocal);
                outLocal = outBuf.DeQue<T>();
                DataCopyExtParams storeParams{1, static_cast<uint32_t>(iterSize * sizeof(T)), 0, 0, 0};
                DataCopyPad(outGm[inputOffset], outLocal, storeParams);
                outBuf.FreeTensor<T>(outLocal);
            }

            scaleBuf.FreeTensor<T>(scaleLocal);
            if (hasBias) {
                biasBuf.FreeTensor<T>(biasLocal);
            }
        } else if constexpr (std::is_same<T, bfloat16_t>::value) {
            LocalTensor<uint8_t> tmpLocal = tmpBuf.AllocTensor<uint8_t>();
            LocalTensor<T> scaleLocal = scaleBuf.AllocTensor<T>();
            LocalTensor<T> biasLocal;
            MaterializeParamBroadcast_midaxis(scaleLocal, scaleGm, tmpLocal, startH, rows);
            if (hasBias) {
                biasLocal = biasBuf.AllocTensor<T>();
                MaterializeParamBroadcast_midaxis(biasLocal, biasGm, tmpLocal, startH, rows);
            }
            tmpBuf.FreeTensor<uint8_t>(tmpLocal);

            LocalTensor<float> scaleFp32 = tmpBuf.AllocTensor<float>();
            LocalTensor<float> biasFp32 = scaleFp32[(iterSize + 7U) / 8U * 8U];
            Cast(scaleFp32, scaleLocal, RoundMode::CAST_NONE, iterSize);
            if (hasBias) {
                Cast(biasFp32, biasLocal, RoundMode::CAST_NONE, iterSize);
            }

            for (uint32_t dIdx = 0; dIdx < D; ++dIdx) {
                const uint32_t inputOffset = (dIdx * H + startH) * W;
                LocalTensor<T> inputLocal = inputBuf.AllocTensor<T>();
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(iterSize * sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                DataCopyPad(inputLocal, inputGm[inputOffset], copyParams, padParams);
                inputBuf.EnQue<T>(inputLocal);
                inputLocal = inputBuf.DeQue<T>();

                LocalTensor<T> outLocal = outBuf.AllocTensor<T>();
                LocalTensor<float> inputFp32 = tmpBuf.AllocTensor<float>();
                LocalTensor<float> outFp32 = inputLocal.template ReinterpretCast<float>();
                Cast(inputFp32, inputLocal, RoundMode::CAST_NONE, iterSize);
                Compute_iter(outFp32, inputFp32, scaleFp32, biasFp32, halfIterSize, hasBias);
                Cast(outLocal, outFp32, RoundMode::CAST_RINT, halfIterSize);
                Compute_iter(outFp32, inputFp32[halfIterSize], scaleFp32[halfIterSize], biasFp32[halfIterSize],
                             iterSize - halfIterSize, hasBias);
                Cast(outLocal[halfIterSize], outFp32, RoundMode::CAST_RINT, iterSize - halfIterSize);
                tmpBuf.FreeTensor<float>(inputFp32);
                inputBuf.FreeTensor<T>(inputLocal);

                outBuf.EnQue<T>(outLocal);
                outLocal = outBuf.DeQue<T>();
                DataCopyExtParams storeParams{1, static_cast<uint32_t>(iterSize * sizeof(T)), 0, 0, 0};
                DataCopyPad(outGm[inputOffset], outLocal, storeParams);
                outBuf.FreeTensor<T>(outLocal);
            }

            tmpBuf.FreeTensor<float>(scaleFp32);
            scaleBuf.FreeTensor<T>(scaleLocal);
            if (hasBias) {
                biasBuf.FreeTensor<T>(biasLocal);
            }
        }
    }

    template <typename U>
    __aicore__ inline void Compute_iter(const LocalTensor<U> &outLocal, const LocalTensor<U> &inputLocal,
                                        const LocalTensor<U> &scaleLocal, const LocalTensor<U> &biasLocal,
                                        uint32_t iterSize, bool hasBias) {
        if (hasBias) {
            Adds(outLocal, biasLocal, static_cast<U>(0), iterSize);
            MulAddDst(outLocal, inputLocal, scaleLocal, iterSize);
        } else {
            Mul(outLocal, inputLocal, scaleLocal, iterSize);
        }
    }

    __aicore__ inline void MaterializeParamBroadcast_midaxis(const LocalTensor<T> &dstLocal,
                                                             const GlobalTensor<T> &srcGm,
                                                             LocalTensor<uint8_t> &sharedTmpBuffer,
                                                             uint32_t startRow,
                                                             uint32_t rows) {
        LocalTensor<T> compactLocal = compactBuf.AllocTensor<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(rows * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(compactLocal, srcGm[startRow], copyParams, padParams);
        compactBuf.EnQue<T>(compactLocal);
        compactLocal = compactBuf.DeQue<T>();

        const uint32_t compactBytes = (rows * sizeof(T) + 31U) / 32U * 32U;
        LocalTensor<uint8_t> broadcastTmp = sharedTmpBuffer[compactBytes];
        const uint32_t srcShape[2] = {rows, 1U};
        const uint32_t dstShape[2] = {rows, realW};
        Broadcast<T, 2, 1>(dstLocal, compactLocal, dstShape, srcShape, broadcastTmp);
        compactBuf.FreeTensor<T>(compactLocal);
    }

    __aicore__ inline void Process() {
        uint32_t processedRows = 0;
        while (processedRows < size) {
            const uint32_t rows = min(nLinesPerIter, size - processedRows);
            Process_iter(beginRow + processedRows, rows);
            processedRows += rows;
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
    TQue<QuePosition::VECIN, 1> compactBuf;
    TQue<QuePosition::VECIN, 1> biasBuf;
    TQue<QuePosition::VECOUT, 1> outBuf;
    TQue<QuePosition::VECCALC, 1> tmpBuf;

    uint32_t size = 0;
    uint32_t smallLoopTimes = 0;
    uint32_t nElementsPerIter = 0;
    uint32_t nLinesPerIter = 1;
    uint32_t totalSize = 0;
    uint32_t rowCount = 0;
    uint32_t beginRow = 0;
    uint32_t realW = 0;

    uint32_t *mmInputDims;
    uint32_t *mmOtherDims;
    uint32_t *mmOutputDims;
    int nOutputDims;
    uint32_t D;
    uint32_t H;
    uint32_t W;
};
