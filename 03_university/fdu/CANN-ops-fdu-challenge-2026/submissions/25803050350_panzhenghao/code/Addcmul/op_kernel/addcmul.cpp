#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"

#include "addcmul_tiling.h"
#include "tiling_key_addcmul.h"

constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t BLOCK_SIZE = 32;

template <bool NEED_BROADCAST_STATE>
struct BroadcastState {
};

template <>
struct BroadcastState<true> {
    uint32_t dimNum;
    uint32_t noBroadcast;
    uint32_t outShape[ADDCMUL_MAX_DIMS];
    uint32_t inputDataStride[ADDCMUL_MAX_DIMS];
    uint32_t x1Stride[ADDCMUL_MAX_DIMS];
    uint32_t x2Stride[ADDCMUL_MAX_DIMS];
};

template <typename TYPE, uint32_t ADD_CMUL_MODE>
class KernelAddcmul : private BroadcastState<ADD_CMUL_MODE == 1> {
public:
    __aicore__ inline KernelAddcmul() {}

    __aicore__ inline void Init(GM_ADDR inputData, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
        const AddcmulTilingData &tiling)
    {
        this->totalLength = tiling.totalLength;
        this->tileDataNum = tiling.tileDataNum;
        if constexpr (ADD_CMUL_MODE == 1) {
            this->dimNum = tiling.dimNum;
            this->noBroadcast = tiling.noBroadcast;
            for (uint32_t i = 0; i < ADDCMUL_MAX_DIMS; ++i) {
                this->outShape[i] = tiling.outShape[i];
                this->inputDataStride[i] = tiling.inputDataStride[i];
                this->x1Stride[i] = tiling.x1Stride[i];
                this->x2Stride[i] = tiling.x2Stride[i];
            }
        }

        uint32_t blockIdx = AscendC::GetBlockIdx();
        this->globalStart = tiling.bigCoreDataNum * blockIdx;
        this->coreDataNum = tiling.bigCoreDataNum;
        this->tileNum = tiling.finalBigTileNum;
        this->tailDataNum = tiling.bigTailDataNum;

        if (blockIdx >= tiling.tailBlockNum) {
            this->coreDataNum = tiling.smallCoreDataNum;
            this->tileNum = tiling.finalSmallTileNum;
            this->tailDataNum = tiling.smallTailDataNum;
            this->globalStart -= (tiling.bigCoreDataNum - tiling.smallCoreDataNum) * (blockIdx - tiling.tailBlockNum);
        }

        if (this->globalStart >= this->totalLength) {
            this->validCoreDataNum = 0;
            this->tileNum = 0;
            this->tailDataNum = 0;
        } else {
            uint32_t remainingDataNum = this->totalLength - this->globalStart;
            this->validCoreDataNum = remainingDataNum < this->coreDataNum ? remainingDataNum : this->coreDataNum;
        }

        if (this->totalLength == 0) {
            return;
        }

        valueGm.SetGlobalBuffer((__gm__ TYPE *)value, 1);
        this->valueScalar = valueGm.GetValue(0);

        if constexpr (ADD_CMUL_MODE == 0 || ADD_CMUL_MODE == 2 || ADD_CMUL_MODE == 5 || ADD_CMUL_MODE == 6) {
            inputDataGm.SetGlobalBuffer((__gm__ TYPE *)inputData + this->globalStart, this->coreDataNum);
            if constexpr (ADD_CMUL_MODE == 6) {
                x1Gm.SetGlobalBuffer((__gm__ TYPE *)x1 + this->globalStart, this->coreDataNum);
                x2Gm.SetGlobalBuffer((__gm__ TYPE *)x2 + this->globalStart, this->coreDataNum);
                this->lastDim = 0;
            } else if constexpr (ADD_CMUL_MODE == 5) {
                x1Gm.SetGlobalBuffer((__gm__ TYPE *)x2 + this->globalStart, this->coreDataNum);
                x2Gm.SetGlobalBuffer((__gm__ TYPE *)x1, tiling.x1Length);
                this->lastDim = tiling.outShape[tiling.dimNum - 1];
            } else if constexpr (ADD_CMUL_MODE == 2) {
                x1Gm.SetGlobalBuffer((__gm__ TYPE *)x1 + this->globalStart, this->coreDataNum);
                x2Gm.SetGlobalBuffer((__gm__ TYPE *)x2, tiling.x2Length);
                this->lastDim = tiling.outShape[tiling.dimNum - 1];
            } else {
                x1Gm.SetGlobalBuffer((__gm__ TYPE *)x1 + this->globalStart, this->coreDataNum);
                x2Gm.SetGlobalBuffer((__gm__ TYPE *)x2 + this->globalStart, this->coreDataNum);
                this->lastDim = 0;
            }
            yGm.SetGlobalBuffer((__gm__ TYPE *)y + this->globalStart, this->coreDataNum);

            pipe.InitBuffer(inputDataQueue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));
            pipe.InitBuffer(x1Queue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));
            pipe.InitBuffer(x2Queue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));
            pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));
        } else if constexpr (ADD_CMUL_MODE == 3 || ADD_CMUL_MODE == 4) {
            inputDataGm.SetGlobalBuffer((__gm__ TYPE *)inputData + this->globalStart, this->coreDataNum);
            if constexpr (ADD_CMUL_MODE == 3) {
                x1Gm.SetGlobalBuffer((__gm__ TYPE *)x1 + this->globalStart, this->coreDataNum);
                x2Gm.SetGlobalBuffer((__gm__ TYPE *)x2, 1);
            } else {
                x1Gm.SetGlobalBuffer((__gm__ TYPE *)x2 + this->globalStart, this->coreDataNum);
                x2Gm.SetGlobalBuffer((__gm__ TYPE *)x1, 1);
            }
            this->scalarMulValue = x2Gm.GetValue(0);
            if constexpr (std::is_same_v<TYPE, float>) {
                this->scalarScaleValue = static_cast<TYPE>(this->scalarMulValue * this->valueScalar);
            }
            yGm.SetGlobalBuffer((__gm__ TYPE *)y + this->globalStart, this->coreDataNum);

            pipe.InitBuffer(inputDataQueue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));
            pipe.InitBuffer(x1Queue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));
            pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));
        } else if constexpr (ADD_CMUL_MODE == 7) {
            inputDataGm.SetGlobalBuffer((__gm__ TYPE *)inputData, 1);
            x1Gm.SetGlobalBuffer((__gm__ TYPE *)x1 + this->globalStart, this->coreDataNum);
            x2Gm.SetGlobalBuffer((__gm__ TYPE *)x2 + this->globalStart, this->coreDataNum);
            yGm.SetGlobalBuffer((__gm__ TYPE *)y + this->globalStart, this->coreDataNum);
            this->scalarInputValue = inputDataGm.GetValue(0);

            pipe.InitBuffer(x1Queue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));
            pipe.InitBuffer(x2Queue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));
            pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));
        } else if constexpr (ADD_CMUL_MODE == 8) {
            inputDataGm.SetGlobalBuffer((__gm__ TYPE *)inputData + this->globalStart, this->coreDataNum);
            x1Gm.SetGlobalBuffer((__gm__ TYPE *)x1, 1);
            x2Gm.SetGlobalBuffer((__gm__ TYPE *)x2, 1);
            yGm.SetGlobalBuffer((__gm__ TYPE *)y + this->globalStart, this->coreDataNum);
            if constexpr (std::is_same_v<TYPE, half>) {
                half scalarMul = static_cast<half>(static_cast<float>(x1Gm.GetValue(0)) *
                                                   static_cast<float>(x2Gm.GetValue(0)));
                this->scalarAddValue = static_cast<half>(static_cast<float>(scalarMul) *
                                                         static_cast<float>(this->valueScalar));
            } else if constexpr (std::is_same_v<TYPE, float>) {
                this->scalarAddValue = CalcValue(static_cast<TYPE>(0), x1Gm.GetValue(0), x2Gm.GetValue(0),
                                                 this->valueScalar);
            } else if constexpr (std::is_same_v<TYPE, int32_t>) {
                this->scalarAddValue = CalcValue(static_cast<TYPE>(0), x1Gm.GetValue(0), x2Gm.GetValue(0),
                                                 this->valueScalar);
            } else {
                this->scalarMulValue = x1Gm.GetValue(0);
                this->scalarInputValue = x2Gm.GetValue(0);
            }

            pipe.InitBuffer(inputDataQueue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));
            pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));
        } else if constexpr (ADD_CMUL_MODE == 9 || ADD_CMUL_MODE == 10 || ADD_CMUL_MODE == 11) {
            yGm.SetGlobalBuffer((__gm__ TYPE *)y + this->globalStart, this->coreDataNum);
            if constexpr (ADD_CMUL_MODE == 9) {
                inputDataGm.SetGlobalBuffer((__gm__ TYPE *)inputData + this->globalStart, this->coreDataNum);
                x1Gm.SetGlobalBuffer((__gm__ TYPE *)x1 + this->globalStart, this->coreDataNum);
                x2Gm.SetGlobalBuffer((__gm__ TYPE *)x2, tiling.x2Length);
                this->repeatBlock = tiling.x2Length;
            } else if constexpr (ADD_CMUL_MODE == 10) {
                inputDataGm.SetGlobalBuffer((__gm__ TYPE *)inputData + this->globalStart, this->coreDataNum);
                x1Gm.SetGlobalBuffer((__gm__ TYPE *)x1, tiling.x1Length);
                x2Gm.SetGlobalBuffer((__gm__ TYPE *)x2 + this->globalStart, this->coreDataNum);
                this->repeatBlock = tiling.x1Length;
            } else {
                inputDataGm.SetGlobalBuffer((__gm__ TYPE *)inputData, tiling.inputDataLength);
                x1Gm.SetGlobalBuffer((__gm__ TYPE *)x1 + this->globalStart, this->coreDataNum);
                x2Gm.SetGlobalBuffer((__gm__ TYPE *)x2 + this->globalStart, this->coreDataNum);
                this->repeatBlock = tiling.inputDataLength;
            }

            pipe.InitBuffer(inputDataQueue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));
            pipe.InitBuffer(x1Queue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));
            pipe.InitBuffer(x2Queue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));
            pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));
        } else {
            inputDataGm.SetGlobalBuffer((__gm__ TYPE *)inputData, tiling.inputDataLength);
            x1Gm.SetGlobalBuffer((__gm__ TYPE *)x1, tiling.x1Length);
            x2Gm.SetGlobalBuffer((__gm__ TYPE *)x2, tiling.x2Length);
            yGm.SetGlobalBuffer((__gm__ TYPE *)y, this->totalLength);
            pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));
        }
    }

    __aicore__ inline void Process()
    {
        if (this->totalLength == 0) {
            return;
        }
        if constexpr (ADD_CMUL_MODE == 0) {
            ProcessVector();
        } else if constexpr (ADD_CMUL_MODE == 2 || ADD_CMUL_MODE == 5) {
            ProcessLastDimBroadcast();
        } else if constexpr (ADD_CMUL_MODE == 3 || ADD_CMUL_MODE == 4) {
            ProcessScalarOperand();
        } else if constexpr (ADD_CMUL_MODE == 6) {
            if constexpr (std::is_same_v<TYPE, int32_t>) {
                ProcessVector();
            } else {
                ProcessLocalScalar();
            }
        } else if constexpr (ADD_CMUL_MODE == 7) {
            ProcessInputScalar();
        } else if constexpr (ADD_CMUL_MODE == 8) {
            ProcessAddScalar();
        } else if constexpr (ADD_CMUL_MODE == 9 || ADD_CMUL_MODE == 10 || ADD_CMUL_MODE == 11) {
            ProcessSuffixBroadcast();
        } else {
            ProcessScalar();
        }
    }

private:
    __aicore__ inline void ProcessVector()
    {
        for (uint32_t i = 0; i < this->tileNum; ++i) {
            uint32_t tileOffset = i * this->tileDataNum;
            if (tileOffset >= this->validCoreDataNum) {
                break;
            }
            uint32_t remain = this->validCoreDataNum - tileOffset;
            this->processDataNum = remain < this->tileDataNum ? remain : this->tileDataNum;
            CopyInVector(i);
            ComputeVector();
            CopyOutVector(i);
        }
    }

    __aicore__ inline void ProcessScalar()
    {
        for (uint32_t i = 0; i < this->tileNum; ++i) {
            uint32_t tileOffset = i * this->tileDataNum;
            if (tileOffset >= this->validCoreDataNum) {
                break;
            }
            uint32_t remain = this->validCoreDataNum - tileOffset;
            this->processDataNum = remain < this->tileDataNum ? remain : this->tileDataNum;
            ComputeScalar(i);
            CopyOutScalar(i);
        }
    }

    __aicore__ inline void ProcessLastDimBroadcast()
    {
        uint32_t done = 0;
        while (done < this->validCoreDataNum) {
            const uint32_t globalIndex = this->globalStart + done;
            const uint32_t inRowOffset = globalIndex % this->lastDim;
            const uint32_t rowRemain = this->lastDim - inRowOffset;
            uint32_t remain = this->validCoreDataNum - done;
            this->processDataNum = remain < this->tileDataNum ? remain : this->tileDataNum;
            this->processDataNum = this->processDataNum < rowRemain ? this->processDataNum : rowRemain;
            CopyInVectorOffsets(done, inRowOffset);
            ComputeVector();
            CopyOutVectorOffset(done);
            done += this->processDataNum;
        }
    }

    __aicore__ inline void ProcessScalarOperand()
    {
        for (uint32_t i = 0; i < this->tileNum; ++i) {
            uint32_t tileOffset = i * this->tileDataNum;
            if (tileOffset >= this->validCoreDataNum) {
                break;
            }
            uint32_t remain = this->validCoreDataNum - tileOffset;
            this->processDataNum = remain < this->tileDataNum ? remain : this->tileDataNum;
            CopyInScalarOperand(tileOffset);
            ComputeScalarOperand();
            CopyOutVectorOffset(tileOffset);
        }
    }

    __aicore__ inline void ProcessLocalScalar()
    {
        for (uint32_t i = 0; i < this->tileNum; ++i) {
            uint32_t tileOffset = i * this->tileDataNum;
            if (tileOffset >= this->validCoreDataNum) {
                break;
            }
            uint32_t remain = this->validCoreDataNum - tileOffset;
            this->processDataNum = remain < this->tileDataNum ? remain : this->tileDataNum;
            CopyInVector(i);
            ComputeLocalScalar();
            CopyOutVector(i);
        }
    }

    __aicore__ inline void ProcessInputScalar()
    {
        for (uint32_t i = 0; i < this->tileNum; ++i) {
            uint32_t tileOffset = i * this->tileDataNum;
            if (tileOffset >= this->validCoreDataNum) {
                break;
            }
            uint32_t remain = this->validCoreDataNum - tileOffset;
            this->processDataNum = remain < this->tileDataNum ? remain : this->tileDataNum;
            CopyInMulOperands(tileOffset);
            ComputeInputScalar();
            CopyOutVectorOffset(tileOffset);
        }
    }

    __aicore__ inline void ProcessAddScalar()
    {
        for (uint32_t i = 0; i < this->tileNum; ++i) {
            uint32_t tileOffset = i * this->tileDataNum;
            if (tileOffset >= this->validCoreDataNum) {
                break;
            }
            uint32_t remain = this->validCoreDataNum - tileOffset;
            this->processDataNum = remain < this->tileDataNum ? remain : this->tileDataNum;
            CopyInInputOnly(tileOffset);
            ComputeAddScalar();
            CopyOutVectorOffset(tileOffset);
        }
    }

    __aicore__ inline void ProcessSuffixBroadcast()
    {
        for (uint32_t i = 0; i < this->tileNum; ++i) {
            uint32_t tileOffset = i * this->tileDataNum;
            if (tileOffset >= this->validCoreDataNum) {
                break;
            }
            uint32_t remain = this->validCoreDataNum - tileOffset;
            this->processDataNum = remain < this->tileDataNum ? remain : this->tileDataNum;
            CopyInSuffixBroadcast(tileOffset);
            if constexpr (std::is_same_v<TYPE, int8_t>) {
                ComputeLocalScalar();
            } else {
                ComputeVector();
            }
            CopyOutVectorOffset(tileOffset);
        }
    }

    __aicore__ inline void CopyInVector(uint32_t progress)
    {
        CopyInVectorOffsets(progress * this->tileDataNum, progress * this->tileDataNum);
    }

    __aicore__ inline void CopyInVectorOffsets(uint32_t dataOffset, uint32_t x2Offset)
    {
        AscendC::LocalTensor<TYPE> inputDataLocal = inputDataQueue.AllocTensor<TYPE>();
        AscendC::LocalTensor<TYPE> x1Local = x1Queue.AllocTensor<TYPE>();
        AscendC::LocalTensor<TYPE> x2Local = x2Queue.AllocTensor<TYPE>();
        CopyGmToLocal(inputDataLocal, inputDataGm, dataOffset);
        CopyGmToLocal(x1Local, x1Gm, dataOffset);
        CopyGmToLocal(x2Local, x2Gm, x2Offset);
        inputDataQueue.EnQue(inputDataLocal);
        x1Queue.EnQue(x1Local);
        x2Queue.EnQue(x2Local);
    }

    __aicore__ inline void CopyInScalarOperand(uint32_t dataOffset)
    {
        AscendC::LocalTensor<TYPE> inputDataLocal = inputDataQueue.AllocTensor<TYPE>();
        AscendC::LocalTensor<TYPE> x1Local = x1Queue.AllocTensor<TYPE>();
        CopyGmToLocal(inputDataLocal, inputDataGm, dataOffset);
        CopyGmToLocal(x1Local, x1Gm, dataOffset);
        inputDataQueue.EnQue(inputDataLocal);
        x1Queue.EnQue(x1Local);
    }

    __aicore__ inline void CopyInMulOperands(uint32_t dataOffset)
    {
        AscendC::LocalTensor<TYPE> x1Local = x1Queue.AllocTensor<TYPE>();
        AscendC::LocalTensor<TYPE> x2Local = x2Queue.AllocTensor<TYPE>();
        CopyGmToLocal(x1Local, x1Gm, dataOffset);
        CopyGmToLocal(x2Local, x2Gm, dataOffset);
        x1Queue.EnQue(x1Local);
        x2Queue.EnQue(x2Local);
    }

    __aicore__ inline void CopyInInputOnly(uint32_t dataOffset)
    {
        AscendC::LocalTensor<TYPE> inputDataLocal = inputDataQueue.AllocTensor<TYPE>();
        CopyGmToLocal(inputDataLocal, inputDataGm, dataOffset);
        inputDataQueue.EnQue(inputDataLocal);
    }

    __aicore__ inline void CopyInSuffixBroadcast(uint32_t dataOffset)
    {
        AscendC::LocalTensor<TYPE> inputDataLocal = inputDataQueue.AllocTensor<TYPE>();
        AscendC::LocalTensor<TYPE> x1Local = x1Queue.AllocTensor<TYPE>();
        AscendC::LocalTensor<TYPE> x2Local = x2Queue.AllocTensor<TYPE>();

        if constexpr (ADD_CMUL_MODE == 9) {
            CopyGmToLocal(inputDataLocal, inputDataGm, dataOffset);
            CopyGmToLocal(x1Local, x1Gm, dataOffset);
            CopyRepeatBlockToLocal(x2Local, x2Gm, dataOffset);
        } else if constexpr (ADD_CMUL_MODE == 10) {
            CopyGmToLocal(inputDataLocal, inputDataGm, dataOffset);
            CopyRepeatBlockToLocal(x1Local, x1Gm, dataOffset);
            CopyGmToLocal(x2Local, x2Gm, dataOffset);
        } else {
            CopyRepeatBlockToLocal(inputDataLocal, inputDataGm, dataOffset);
            CopyGmToLocal(x1Local, x1Gm, dataOffset);
            CopyGmToLocal(x2Local, x2Gm, dataOffset);
        }

        inputDataQueue.EnQue(inputDataLocal);
        x1Queue.EnQue(x1Local);
        x2Queue.EnQue(x2Local);
    }

    __aicore__ inline void ComputeVector()
    {
        AscendC::LocalTensor<TYPE> inputDataLocal = inputDataQueue.DeQue<TYPE>();
        AscendC::LocalTensor<TYPE> x1Local = x1Queue.DeQue<TYPE>();
        AscendC::LocalTensor<TYPE> x2Local = x2Queue.DeQue<TYPE>();
        AscendC::LocalTensor<TYPE> yLocal = outQueue.AllocTensor<TYPE>();

        if constexpr (std::is_same_v<TYPE, half>) {
            AscendC::Mul(x1Local, x1Local, x2Local, this->processDataNum);
            AscendC::Muls(x1Local, x1Local, this->valueScalar, this->processDataNum);
            AscendC::Add(yLocal, inputDataLocal, x1Local, this->processDataNum);
        } else {
            AscendC::Mul(x1Local, x1Local, x2Local, this->processDataNum);
            AscendC::Muls(x1Local, x1Local, this->valueScalar, this->processDataNum);
            AscendC::Add(yLocal, inputDataLocal, x1Local, this->processDataNum);
        }

        outQueue.EnQue<TYPE>(yLocal);
        inputDataQueue.FreeTensor(inputDataLocal);
        x1Queue.FreeTensor(x1Local);
        x2Queue.FreeTensor(x2Local);
    }

    __aicore__ inline void ComputeScalarOperand()
    {
        AscendC::LocalTensor<TYPE> inputDataLocal = inputDataQueue.DeQue<TYPE>();
        AscendC::LocalTensor<TYPE> x1Local = x1Queue.DeQue<TYPE>();
        AscendC::LocalTensor<TYPE> yLocal = outQueue.AllocTensor<TYPE>();

        if constexpr (std::is_same_v<TYPE, half>) {
            AscendC::Muls(x1Local, x1Local, this->scalarMulValue, this->processDataNum);
            AscendC::Muls(x1Local, x1Local, this->valueScalar, this->processDataNum);
            AscendC::Add(yLocal, inputDataLocal, x1Local, this->processDataNum);
        } else if constexpr (std::is_same_v<TYPE, float>) {
            AscendC::Muls(x1Local, x1Local, this->scalarScaleValue, this->processDataNum);
            AscendC::Add(yLocal, inputDataLocal, x1Local, this->processDataNum);
        } else if constexpr (std::is_same_v<TYPE, int32_t>) {
            AscendC::Muls(x1Local, x1Local, this->scalarMulValue, this->processDataNum);
            AscendC::Muls(x1Local, x1Local, this->valueScalar, this->processDataNum);
            AscendC::Add(yLocal, inputDataLocal, x1Local, this->processDataNum);
        } else {
            for (uint32_t i = 0; i < this->processDataNum; ++i) {
                TYPE result = CalcValue(inputDataLocal.GetValue(i), x1Local.GetValue(i), this->scalarMulValue,
                                        this->valueScalar);
                yLocal.SetValue(i, result);
            }
        }

        outQueue.EnQue<TYPE>(yLocal);
        inputDataQueue.FreeTensor(inputDataLocal);
        x1Queue.FreeTensor(x1Local);
    }

    __aicore__ inline void ComputeLocalScalar()
    {
        AscendC::LocalTensor<TYPE> inputDataLocal = inputDataQueue.DeQue<TYPE>();
        AscendC::LocalTensor<TYPE> x1Local = x1Queue.DeQue<TYPE>();
        AscendC::LocalTensor<TYPE> x2Local = x2Queue.DeQue<TYPE>();
        AscendC::LocalTensor<TYPE> yLocal = outQueue.AllocTensor<TYPE>();

        for (uint32_t i = 0; i < this->processDataNum; ++i) {
            TYPE result = CalcValue(inputDataLocal.GetValue(i), x1Local.GetValue(i), x2Local.GetValue(i),
                                    this->valueScalar);
            yLocal.SetValue(i, result);
        }

        outQueue.EnQue<TYPE>(yLocal);
        inputDataQueue.FreeTensor(inputDataLocal);
        x1Queue.FreeTensor(x1Local);
        x2Queue.FreeTensor(x2Local);
    }

    __aicore__ inline void ComputeInputScalar()
    {
        AscendC::LocalTensor<TYPE> x1Local = x1Queue.DeQue<TYPE>();
        AscendC::LocalTensor<TYPE> x2Local = x2Queue.DeQue<TYPE>();
        AscendC::LocalTensor<TYPE> yLocal = outQueue.AllocTensor<TYPE>();

        if constexpr (std::is_same_v<TYPE, half>) {
            AscendC::Mul(x1Local, x1Local, x2Local, this->processDataNum);
            AscendC::Muls(x1Local, x1Local, this->valueScalar, this->processDataNum);
            AscendC::Adds(yLocal, x1Local, this->scalarInputValue, this->processDataNum);
        } else if constexpr (std::is_same_v<TYPE, float>) {
            AscendC::Mul(x1Local, x1Local, x2Local, this->processDataNum);
            AscendC::Muls(x1Local, x1Local, this->valueScalar, this->processDataNum);
            AscendC::Adds(yLocal, x1Local, this->scalarInputValue, this->processDataNum);
        } else if constexpr (std::is_same_v<TYPE, int32_t>) {
            AscendC::Mul(x1Local, x1Local, x2Local, this->processDataNum);
            AscendC::Muls(x1Local, x1Local, this->valueScalar, this->processDataNum);
            AscendC::Adds(yLocal, x1Local, this->scalarInputValue, this->processDataNum);
        } else {
            for (uint32_t i = 0; i < this->processDataNum; ++i) {
                TYPE result = CalcValue(this->scalarInputValue, x1Local.GetValue(i), x2Local.GetValue(i),
                                        this->valueScalar);
                yLocal.SetValue(i, result);
            }
        }

        outQueue.EnQue<TYPE>(yLocal);
        x1Queue.FreeTensor(x1Local);
        x2Queue.FreeTensor(x2Local);
    }

    __aicore__ inline void ComputeAddScalar()
    {
        AscendC::LocalTensor<TYPE> inputDataLocal = inputDataQueue.DeQue<TYPE>();
        AscendC::LocalTensor<TYPE> yLocal = outQueue.AllocTensor<TYPE>();

        if constexpr (std::is_same_v<TYPE, half>) {
            AscendC::Adds(yLocal, inputDataLocal, this->scalarAddValue, this->processDataNum);
        } else if constexpr (std::is_same_v<TYPE, float>) {
            AscendC::Adds(yLocal, inputDataLocal, this->scalarAddValue, this->processDataNum);
        } else if constexpr (std::is_same_v<TYPE, int32_t>) {
            AscendC::Adds(yLocal, inputDataLocal, this->scalarAddValue, this->processDataNum);
        } else {
            for (uint32_t i = 0; i < this->processDataNum; ++i) {
                TYPE result = CalcValue(inputDataLocal.GetValue(i), this->scalarMulValue, this->scalarInputValue,
                                        this->valueScalar);
                yLocal.SetValue(i, result);
            }
        }

        outQueue.EnQue<TYPE>(yLocal);
        inputDataQueue.FreeTensor(inputDataLocal);
    }

    __aicore__ inline void CopyOutVector(uint32_t progress)
    {
        CopyOutVectorOffset(progress * this->tileDataNum);
    }

    __aicore__ inline void CopyOutVectorOffset(uint32_t dataOffset)
    {
        AscendC::LocalTensor<TYPE> yLocal = outQueue.DeQue<TYPE>();
        CopyLocalToGm(yGm, dataOffset, yLocal);
        outQueue.FreeTensor(yLocal);
    }

    __aicore__ inline void ComputeScalar(uint32_t progress)
    {
        AscendC::LocalTensor<TYPE> yLocal = outQueue.AllocTensor<TYPE>();
        const uint32_t tileGlobalStart = this->globalStart + progress * this->tileDataNum;
        for (uint32_t i = 0; i < this->processDataNum; ++i) {
            const uint32_t globalIndex = tileGlobalStart + i;
            uint32_t inputDataOffset = globalIndex;
            uint32_t x1Offset = globalIndex;
            uint32_t x2Offset = globalIndex;
            if (this->noBroadcast == 0) {
                inputDataOffset = CalcBroadcastOffset(globalIndex, this->inputDataStride);
                x1Offset = CalcBroadcastOffset(globalIndex, this->x1Stride);
                x2Offset = CalcBroadcastOffset(globalIndex, this->x2Stride);
            }
            TYPE result = CalcValue(inputDataGm.GetValue(inputDataOffset), x1Gm.GetValue(x1Offset),
                                    x2Gm.GetValue(x2Offset), this->valueScalar);
            yLocal.SetValue(i, result);
        }
        outQueue.EnQue<TYPE>(yLocal);
    }

    __aicore__ inline void CopyOutScalar(uint32_t progress)
    {
        AscendC::LocalTensor<TYPE> yLocal = outQueue.DeQue<TYPE>();
        CopyLocalToGm(yGm, this->globalStart + progress * this->tileDataNum, yLocal);
        outQueue.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyGmToLocal(AscendC::LocalTensor<TYPE> local, AscendC::GlobalTensor<TYPE> &gm,
                                         uint32_t offset)
    {
        CopyGmToLocal(local, gm, offset, this->processDataNum);
    }

    __aicore__ inline void CopyGmToLocal(AscendC::LocalTensor<TYPE> local, AscendC::GlobalTensor<TYPE> &gm,
                                         uint32_t offset, uint32_t copyNum)
    {
        uint32_t copyBytes = copyNum * sizeof(TYPE);
        uint32_t offsetBytes = offset * sizeof(TYPE);
        if (copyBytes % BLOCK_SIZE == 0 && offsetBytes % BLOCK_SIZE == 0) {
            AscendC::DataCopy(local, gm[offset], copyNum);
        } else {
            AscendC::DataCopyExtParams copyParams = {1, copyBytes, 0, 0, 0};
            uint32_t paddingBytes = copyBytes % BLOCK_SIZE == 0 ? 0 : BLOCK_SIZE - copyBytes % BLOCK_SIZE;
            uint8_t rightPadding = static_cast<uint8_t>(paddingBytes / sizeof(TYPE));
            AscendC::DataCopyPadExtParams<TYPE> padParams = {true, 0, rightPadding, 0};
            AscendC::DataCopyPad<TYPE>(local, gm[offset], copyParams, padParams);
        }
    }

    __aicore__ inline void CopyRepeatBlockToLocal(AscendC::LocalTensor<TYPE> local,
                                                  AscendC::GlobalTensor<TYPE> &gm, uint32_t dataOffset)
    {
        uint32_t done = 0;
        uint32_t repeatOffset = (this->globalStart + dataOffset) % this->repeatBlock;
        if (repeatOffset + this->processDataNum <= this->repeatBlock) {
            CopyGmToLocal(local, gm, repeatOffset);
            return;
        }
        while (done < this->processDataNum) {
            const uint32_t blockRemain = this->repeatBlock - repeatOffset;
            const uint32_t processRemain = this->processDataNum - done;
            const uint32_t copyNum = blockRemain < processRemain ? blockRemain : processRemain;
            if ((done * sizeof(TYPE)) % BLOCK_SIZE == 0) {
                CopyGmToLocal(local[done], gm, repeatOffset, copyNum);
            } else {
                for (uint32_t i = 0; i < copyNum; ++i) {
                    local.SetValue(done + i, gm.GetValue(repeatOffset + i));
                }
            }
            done += copyNum;
            repeatOffset = 0;
        }
    }

    __aicore__ inline void CopyBroadcastToLocal(AscendC::LocalTensor<TYPE> local,
                                                AscendC::GlobalTensor<TYPE> &gm, const uint32_t *stride,
                                                uint32_t dataOffset)
    {
        const uint32_t tileGlobalStart = this->globalStart + dataOffset;
        for (uint32_t i = 0; i < this->processDataNum; ++i) {
            const uint32_t gmOffset = CalcBroadcastOffset(tileGlobalStart + i, stride);
            local.SetValue(i, gm.GetValue(gmOffset));
        }
    }

    __aicore__ inline void CopyLocalToGm(AscendC::GlobalTensor<TYPE> &gm, uint32_t offset,
                                         AscendC::LocalTensor<TYPE> local)
    {
        uint32_t copyBytes = this->processDataNum * sizeof(TYPE);
        uint32_t offsetBytes = offset * sizeof(TYPE);
        if (copyBytes % BLOCK_SIZE == 0 && offsetBytes % BLOCK_SIZE == 0) {
            AscendC::DataCopy(gm[offset], local, this->processDataNum);
        } else {
            AscendC::DataCopyExtParams copyParams = {1, copyBytes, 0, 0, 0};
            AscendC::DataCopyPad<TYPE>(gm[offset], local, copyParams);
        }
    }

    __aicore__ inline uint32_t CalcBroadcastOffset(uint32_t linearIndex, const uint32_t *stride)
    {
        uint32_t offset = 0;
        for (int32_t dim = static_cast<int32_t>(this->dimNum) - 1; dim >= 0; --dim) {
            const uint32_t coord = linearIndex % this->outShape[dim];
            linearIndex = linearIndex / this->outShape[dim];
            offset += coord * stride[dim];
        }
        return offset;
    }

    __aicore__ inline TYPE CalcValue(TYPE inputDataValue, TYPE x1Value, TYPE x2Value, TYPE value)
    {
        if constexpr (std::is_same_v<TYPE, half>) {
            half product = static_cast<half>(static_cast<float>(x1Value) * static_cast<float>(x2Value));
            half scaled = static_cast<half>(static_cast<float>(product) * static_cast<float>(value));
            return static_cast<half>(static_cast<float>(inputDataValue) + static_cast<float>(scaled));
        } else if constexpr (std::is_same_v<TYPE, float>) {
            return inputDataValue + x1Value * x2Value * value;
        } else if constexpr (std::is_same_v<TYPE, int8_t>) {
            int32_t result = static_cast<int32_t>(inputDataValue) + static_cast<int32_t>(x1Value) *
                static_cast<int32_t>(x2Value) * static_cast<int32_t>(value);
            return WrapInt8(result);
        } else {
            int64_t result = static_cast<int64_t>(inputDataValue) + static_cast<int64_t>(x1Value) *
                static_cast<int64_t>(x2Value) * static_cast<int64_t>(value);
            return WrapInt32(result);
        }
    }

    __aicore__ inline int8_t WrapInt8(int32_t value)
    {
        return static_cast<int8_t>(static_cast<uint8_t>(value));
    }

    __aicore__ inline int32_t WrapInt32(int64_t value)
    {
        return static_cast<int32_t>(static_cast<uint32_t>(value));
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inputDataQueue;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> x1Queue;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> x2Queue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::GlobalTensor<TYPE> inputDataGm;
    AscendC::GlobalTensor<TYPE> x1Gm;
    AscendC::GlobalTensor<TYPE> x2Gm;
    AscendC::GlobalTensor<TYPE> valueGm;
    AscendC::GlobalTensor<TYPE> yGm;
    uint32_t totalLength;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    uint32_t validCoreDataNum;
    uint32_t globalStart;
    uint32_t lastDim;
    uint32_t repeatBlock;
    TYPE valueScalar;
    TYPE scalarMulValue;
    TYPE scalarInputValue;
    TYPE scalarAddValue;
    TYPE scalarScaleValue;
};

template <typename DT_INPUT_DATA, uint32_t ADD_CMUL_MODE>
__global__ __aicore__ void addcmul(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
                                   GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddcmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddcmulTilingData, tilingData, tiling);
    KernelAddcmul<DT_INPUT_DATA, ADD_CMUL_MODE> op;
    op.Init(input_data, x1, x2, value, y, tilingData);
    op.Process();
}
