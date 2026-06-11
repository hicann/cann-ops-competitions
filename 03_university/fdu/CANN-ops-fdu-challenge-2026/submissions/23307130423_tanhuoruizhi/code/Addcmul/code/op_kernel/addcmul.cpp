#include "kernel_operator.h"

#include <type_traits>

#include "addcmul_tiling.h"
#include "tiling_key_addcmul.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
struct ComputeScalar {
    using Type = T;
};

template <>
struct ComputeScalar<int8_t> {
    using Type = half;
};

template <typename T>
using ScaleType = typename ComputeScalar<T>::Type;

template <typename DT>
class StreamAddcmulKernel {
public:
    __aicore__ inline StreamAddcmulKernel() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR output,
                                const AddcmulTilingData &cfg) {
        uint32_t coreIdx = GetBlockIdx();
        chunkLength = cfg.elemsPerCore;
        if (GetBlockNum() == coreIdx + 1) {
            chunkLength += cfg.tailElements;
        }
        tileLength = cfg.tileElements;
        uint32_t alignElems = cfg.alignElements;
        if (chunkLength % alignElems != 0) {
            chunkLength += alignElems - chunkLength % alignElems;
        }

        uint32_t gmBase = cfg.elemsPerCore * coreIdx;
        inputGm.SetGlobalBuffer((__gm__ DT *)input + gmBase, chunkLength);
        x1Gm.SetGlobalBuffer((__gm__ DT *)x1 + gmBase, chunkLength);
        x2Gm.SetGlobalBuffer((__gm__ DT *)x2 + gmBase, chunkLength);
        valueGm.SetGlobalBuffer((__gm__ DT *)value, 1);
        outputGm.SetGlobalBuffer((__gm__ DT *)output + gmBase, chunkLength);

        loopCount = chunkLength / tileLength + (chunkLength % tileLength > 0);
        pipe.InitBuffer(inputQueue, BUFFER_NUM, tileLength * sizeof(DT));
        pipe.InitBuffer(x1Queue, BUFFER_NUM, tileLength * sizeof(DT));
        pipe.InitBuffer(x2Queue, BUFFER_NUM, tileLength * sizeof(DT));
        pipe.InitBuffer(outputQueue, BUFFER_NUM, tileLength * sizeof(DT));
        pipe.InitBuffer(workA, tileLength * sizeof(half));
        pipe.InitBuffer(workB, tileLength * sizeof(half));
        scale = valueGm.GetValue(0);
    }

    __aicore__ inline void Process() {
        if (loopCount == 0) {
            return;
        }
        for (uint32_t loop = 0; loop + 1 < loopCount; ++loop) {
            CopyIn(loop, tileLength);
            Compute(tileLength);
            CopyOut(loop, tileLength);
        }
        uint32_t tail = chunkLength - tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, tail);
        Compute(tail);
        CopyOut(loopCount - 1, tail);
    }

private:
    __aicore__ inline void CopyIn(uint32_t tileIdx, uint32_t validLen) {
        LocalTensor<DT> inputLocal = inputQueue.AllocTensor<DT>();
        LocalTensor<DT> x1Local = x1Queue.AllocTensor<DT>();
        LocalTensor<DT> x2Local = x2Queue.AllocTensor<DT>();
        uint32_t offset = tileIdx * tileLength;
        DataCopy(inputLocal, inputGm[offset], validLen);
        DataCopy(x1Local, x1Gm[offset], validLen);
        DataCopy(x2Local, x2Gm[offset], validLen);
        inputQueue.EnQue(inputLocal);
        x1Queue.EnQue(x1Local);
        x2Queue.EnQue(x2Local);
    }

    __aicore__ inline void Compute(uint32_t validLen) {
        LocalTensor<DT> inputLocal = inputQueue.DeQue<DT>();
        LocalTensor<DT> x1Local = x1Queue.DeQue<DT>();
        LocalTensor<DT> x2Local = x2Queue.DeQue<DT>();
        LocalTensor<DT> outputLocal = outputQueue.AllocTensor<DT>();

        if constexpr (std::is_same_v<DT, int8_t> || std::is_same_v<DT, signed char>) {
            LocalTensor<half> tempA = workA.Get<half>();
            LocalTensor<half> tempB = workB.Get<half>();
            Cast(tempA, x1Local, RoundMode::CAST_NONE, validLen);
            Cast(tempB, x2Local, RoundMode::CAST_NONE, validLen);
            Mul(tempA, tempA, tempB, validLen);
            Muls(tempA, tempA, scale, validLen);
            Cast(tempB, inputLocal, RoundMode::CAST_NONE, validLen);
            Add(tempA, tempA, tempB, validLen);
            Cast(outputLocal, tempA, RoundMode::CAST_NONE, validLen);
        } else {
            Mul(x1Local, x1Local, x2Local, validLen);
            Muls(x1Local, x1Local, scale, validLen);
            Add(outputLocal, x1Local, inputLocal, validLen);
        }

        inputQueue.FreeTensor(inputLocal);
        x1Queue.FreeTensor(x1Local);
        x2Queue.FreeTensor(x2Local);
        outputQueue.EnQue(outputLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx, uint32_t validLen) {
        LocalTensor<DT> outputLocal = outputQueue.DeQue<DT>();
        DataCopy(outputGm[tileIdx * tileLength], outputLocal, validLen);
        outputQueue.FreeTensor(outputLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<TPosition::VECIN, BUFFER_NUM> x1Queue;
    TQue<TPosition::VECIN, BUFFER_NUM> x2Queue;
    TQue<TPosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<TPosition::VECCALC> workA;
    TBuf<TPosition::VECCALC> workB;
    GlobalTensor<DT> inputGm;
    GlobalTensor<DT> x1Gm;
    GlobalTensor<DT> x2Gm;
    GlobalTensor<DT> valueGm;
    GlobalTensor<DT> outputGm;
    uint32_t chunkLength;
    uint32_t tileLength;
    uint32_t loopCount;
    ScaleType<DT> scale;
};

template <typename DT>
class PeriodicAddcmulKernel {
public:
    __aicore__ inline PeriodicAddcmulKernel() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR output,
                                const AddcmulTilingData &cfg) {
        inputPeriod = cfg.inputElements;
        x1Period = cfg.x1Elements;
        x2Period = cfg.x2Elements;
        outputLength = cfg.outputElements;

        uint32_t coreIdx = GetBlockIdx();
        chunkLength = cfg.elemsPerCore;
        if (GetBlockNum() == coreIdx + 1) {
            chunkLength += cfg.tailElements;
        }
        tileLength = cfg.tileElements;
        uint32_t alignElems = cfg.alignElements;
        if (chunkLength % alignElems != 0) {
            chunkLength += alignElems - chunkLength % alignElems;
        }
        startOffset = cfg.elemsPerCore * coreIdx;

        inputGm.SetGlobalBuffer((__gm__ DT *)input, outputLength);
        x1Gm.SetGlobalBuffer((__gm__ DT *)x1, outputLength);
        x2Gm.SetGlobalBuffer((__gm__ DT *)x2, outputLength);
        valueGm.SetGlobalBuffer((__gm__ DT *)value, 1);
        outputGm.SetGlobalBuffer((__gm__ DT *)output, outputLength);

        pipe.InitBuffer(inputQueue, BUFFER_NUM, tileLength * sizeof(DT));
        pipe.InitBuffer(x1Queue, BUFFER_NUM, tileLength * sizeof(DT));
        pipe.InitBuffer(x2Queue, BUFFER_NUM, tileLength * sizeof(DT));
        pipe.InitBuffer(outputQueue, BUFFER_NUM, tileLength * sizeof(DT));
        pipe.InitBuffer(workA, tileLength * sizeof(half));
        pipe.InitBuffer(workB, tileLength * sizeof(half));
        scale = valueGm.GetValue(0);
    }

    __aicore__ inline void Process() {
        uint32_t loops = chunkLength / tileLength + (chunkLength % tileLength > 0);
        if (loops == 0) {
            return;
        }
        for (uint32_t loop = 0; loop + 1 < loops; ++loop) {
            uint32_t flatPos = startOffset + loop * tileLength;
            CopyIn(flatPos, tileLength);
            Compute(tileLength);
            CopyOut(flatPos, tileLength);
        }
        uint32_t flatPos = startOffset + (loops - 1) * tileLength;
        uint32_t tail = chunkLength - tileLength * (loops - 1);
        CopyIn(flatPos, tail);
        Compute(tail);
        CopyOut(flatPos, tail);
    }

private:
    __aicore__ inline void CopyIn(uint32_t flatPos, uint32_t validLen) {
        LocalTensor<DT> inputLocal = inputQueue.AllocTensor<DT>();
        LocalTensor<DT> x1Local = x1Queue.AllocTensor<DT>();
        LocalTensor<DT> x2Local = x2Queue.AllocTensor<DT>();
        DataCopy(inputLocal, inputGm[flatPos % inputPeriod], validLen);
        DataCopy(x1Local, x1Gm[flatPos % x1Period], validLen);
        DataCopy(x2Local, x2Gm[flatPos % x2Period], validLen);
        inputQueue.EnQue(inputLocal);
        x1Queue.EnQue(x1Local);
        x2Queue.EnQue(x2Local);
    }

    __aicore__ inline void Compute(uint32_t validLen) {
        LocalTensor<DT> inputLocal = inputQueue.DeQue<DT>();
        LocalTensor<DT> x1Local = x1Queue.DeQue<DT>();
        LocalTensor<DT> x2Local = x2Queue.DeQue<DT>();
        LocalTensor<DT> outputLocal = outputQueue.AllocTensor<DT>();

        if constexpr (std::is_same_v<DT, int8_t> || std::is_same_v<DT, signed char>) {
            LocalTensor<half> tempA = workA.Get<half>();
            LocalTensor<half> tempB = workB.Get<half>();
            Cast(tempA, x1Local, RoundMode::CAST_NONE, validLen);
            Cast(tempB, x2Local, RoundMode::CAST_NONE, validLen);
            Mul(tempA, tempA, tempB, validLen);
            Muls(tempA, tempA, scale, validLen);
            Cast(tempB, inputLocal, RoundMode::CAST_NONE, validLen);
            Add(tempB, tempA, tempB, validLen);
            Cast(tempA.ReinterpretCast<int16_t>(), tempB, RoundMode::CAST_RINT, validLen);
            ShiftLeft(tempA.ReinterpretCast<int16_t>(), tempA.ReinterpretCast<int16_t>(), int16_t(8), validLen);
            ShiftRight(tempA.ReinterpretCast<int16_t>(), tempA.ReinterpretCast<int16_t>(), int16_t(8), validLen);
            Cast(tempB, tempA.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, validLen);
            Cast(outputLocal, tempB, RoundMode::CAST_NONE, validLen);
        } else {
            Mul(x1Local, x1Local, x2Local, validLen);
            Muls(x1Local, x1Local, scale, validLen);
            Add(outputLocal, x1Local, inputLocal, validLen);
        }

        inputQueue.FreeTensor(inputLocal);
        x1Queue.FreeTensor(x1Local);
        x2Queue.FreeTensor(x2Local);
        outputQueue.EnQue(outputLocal);
    }

    __aicore__ inline void CopyOut(uint32_t flatPos, uint32_t validLen) {
        LocalTensor<DT> outputLocal = outputQueue.DeQue<DT>();
        DataCopy(outputGm[flatPos], outputLocal, validLen);
        outputQueue.FreeTensor(outputLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<TPosition::VECIN, BUFFER_NUM> x1Queue;
    TQue<TPosition::VECIN, BUFFER_NUM> x2Queue;
    TQue<TPosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<TPosition::VECCALC> workA;
    TBuf<TPosition::VECCALC> workB;
    GlobalTensor<DT> inputGm;
    GlobalTensor<DT> x1Gm;
    GlobalTensor<DT> x2Gm;
    GlobalTensor<DT> valueGm;
    GlobalTensor<DT> outputGm;
    uint32_t inputPeriod;
    uint32_t x1Period;
    uint32_t x2Period;
    uint32_t outputLength;
    uint32_t chunkLength;
    uint32_t tileLength;
    uint32_t startOffset;
    ScaleType<DT> scale;
};

template <typename DT>
__aicore__ inline void RunStreamPath(GM_ADDR input, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR output,
                                     const AddcmulTilingData &cfg) {
    StreamAddcmulKernel<DT> op;
    op.Init(input, x1, x2, value, output, cfg);
    op.Process();
}

template <typename DT>
__aicore__ inline void RunPeriodicPath(GM_ADDR input, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR output,
                                       const AddcmulTilingData &cfg) {
    PeriodicAddcmulKernel<DT> op;
    op.Init(input, x1, x2, value, output, cfg);
    op.Process();
}

template <typename DT>
__global__ __aicore__ void addcmul(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
                                   GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(AddcmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddcmulTilingData, cfg, tiling);
    bool sameLength = cfg.inputElements == cfg.outputElements &&
                      cfg.x1Elements == cfg.outputElements &&
                      cfg.x2Elements == cfg.outputElements;
    if (sameLength) {
        RunStreamPath<DT>(input_data, x1, x2, value, y, cfg);
        return;
    }
    RunPeriodicPath<DT>(input_data, x1, x2, value, y, cfg);
}
