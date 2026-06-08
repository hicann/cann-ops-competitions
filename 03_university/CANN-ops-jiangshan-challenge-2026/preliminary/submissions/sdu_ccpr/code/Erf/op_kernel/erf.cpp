// Kernel侧核函数实现 - Erf算子
#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;
constexpr uint32_t ALIGN_NUM = 8;
constexpr uint32_t REPEAT_NUM = 64;

template <class DT_X>
__aicore__ inline void ResolveCoreRange(uint64_t totalLength, uint64_t bigCoreDataNum,
                                        uint64_t smallCoreDataNum, uint32_t bigCoreCount,
                                        uint32_t usedCoreCount, uint64_t blockIdx,
                                        uint64_t &coreOffset, uint64_t &blockLength) {
    if (blockIdx >= usedCoreCount) {
        coreOffset = 0;
        blockLength = 0;
        return;
    }

    uint64_t coreCapacity = 0;
    if (blockIdx < bigCoreCount) {
        coreOffset = blockIdx * bigCoreDataNum;
        coreCapacity = bigCoreDataNum;
    } else {
        uint64_t smallIdx = blockIdx - bigCoreCount;
        coreOffset = static_cast<uint64_t>(bigCoreCount) * bigCoreDataNum + smallIdx * smallCoreDataNum;
        coreCapacity = smallCoreDataNum;
    }

    if (coreOffset >= totalLength) {
        blockLength = 0;
        return;
    }
    uint64_t remaining = totalLength - coreOffset;
    blockLength = remaining > coreCapacity ? coreCapacity : remaining;
}

template <class DT_X>
__aicore__ inline void ComputeErfTanh(LocalTensor<DT_X> t_x, LocalTensor<DT_X> t_out,
                                      LocalTensor<DT_X> t_tmp, uint32_t len) {
    constexpr DT_X ALPHA      = (DT_X)1.1283791671f;
    constexpr DT_X ALPHA_BETA = (DT_X)(1.1283791671f * 0.08943f);

    Mul(t_tmp, t_x, t_x, len);
    Muls(t_tmp, t_tmp, ALPHA_BETA, len);
    Adds(t_tmp, t_tmp, ALPHA, len);
    Mul(t_tmp, t_tmp, t_x, len);
    Tanh(t_out, t_tmp, len);
}

template <class DT_X>
__aicore__ inline void ComputeErfTanhInplace(LocalTensor<DT_X> t_x, LocalTensor<DT_X> t_tmp,
                                             uint32_t len) {
    constexpr DT_X ALPHA      = (DT_X)1.1283791671f;
    constexpr DT_X ALPHA_BETA = (DT_X)(1.1283791671f * 0.08943f);

    Mul(t_tmp, t_x, t_x, len);
    Muls(t_tmp, t_tmp, ALPHA_BETA, len);
    Adds(t_tmp, t_tmp, ALPHA, len);
    Mul(t_tmp, t_tmp, t_x, len);
    Tanh(t_x, t_tmp, len);
}

template <class DT_X>
__aicore__ inline void ComputeErfSigmoidInplace(LocalTensor<DT_X> t_x, LocalTensor<DT_X> t_tmp,
                                                 uint32_t len) {
    constexpr DT_X TWO_ALPHA      = (DT_X)(2.0f * 1.1283791671f);
    constexpr DT_X TWO_ALPHA_BETA = (DT_X)(2.0f * 1.1283791671f * 0.08943f);

    Mul(t_tmp, t_x, t_x, len);
    Muls(t_tmp, t_tmp, TWO_ALPHA_BETA, len);
    Adds(t_tmp, t_tmp, TWO_ALPHA, len);
    Mul(t_tmp, t_tmp, t_x, len);
    Sigmoid(t_x, t_tmp, len);
    Muls(t_x, t_x, (DT_X)2.0f, len);
    Adds(t_x, t_x, (DT_X)(-1.0f), len);
}

__aicore__ inline void ComputeErfMixed(LocalTensor<float> t_x, LocalTensor<float> t_out,
                                        LocalTensor<half> t_x16, LocalTensor<half> t_tmp16,
                                        uint32_t len) {
    const half ALPHA_F16      = (half)1.1289408179f;
    const half ALPHA_BETA_F16 = (half)(1.1289408179f * 0.0896250874f);

    Cast(t_x16, t_x, RoundMode::CAST_ROUND, len);
    Mul(t_tmp16, t_x16, t_x16, len);
    Muls(t_tmp16, t_tmp16, ALPHA_BETA_F16, len);
    Adds(t_tmp16, t_tmp16, ALPHA_F16, len);
    Mul(t_tmp16, t_tmp16, t_x16, len);
    Cast(t_x, t_tmp16, RoundMode::CAST_NONE, len);
    Tanh(t_out, t_x, len);
}

__aicore__ inline void ComputeErfMixedInplace(LocalTensor<float> t_x, LocalTensor<float> t_tmp,
                                               LocalTensor<half> t_x16, LocalTensor<half> t_tmp16,
                                               uint32_t len) {
    const half ALPHA_F16      = (half)1.1289408179f;
    const half ALPHA_BETA_F16 = (half)(1.1289408179f * 0.0896250874f);

    Cast(t_x16, t_x, RoundMode::CAST_ROUND, len);
    Mul(t_tmp16, t_x16, t_x16, len);
    Muls(t_tmp16, t_tmp16, ALPHA_BETA_F16, len);
    Adds(t_tmp16, t_tmp16, ALPHA_F16, len);
    Mul(t_tmp16, t_tmp16, t_x16, len);
    Cast(t_tmp, t_tmp16, RoundMode::CAST_NONE, len);
    Tanh(t_x, t_tmp, len);
}

template <class DT_X, uint32_t BUFFER_NUM>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint64_t totalLength, uint64_t bigCoreDataNum,
                                uint64_t smallCoreDataNum, uint32_t bigCoreCount,
                                uint32_t usedCoreCount, uint32_t tileLength) {
        uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
        uint64_t coreOffset = 0;
        ResolveCoreRange<DT_X>(totalLength, bigCoreDataNum, smallCoreDataNum, bigCoreCount, usedCoreCount, blockIdx,
                               coreOffset, blockLength);
        this->tileLength = tileLength;
        if (blockLength == 0) {
            return;
        }

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + coreOffset, blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + coreOffset, blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(calcBuf, tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (blockLength == 0) {
            return;
        }

        uint64_t loopCount = (blockLength + static_cast<uint64_t>(tileLength) - 1) /
                             static_cast<uint64_t>(tileLength);
        for (uint64_t i = 0; i < loopCount; ++i) {
            uint32_t currentLen = tileLength;
            uint64_t offset = i * static_cast<uint64_t>(tileLength);
            if (i == loopCount - 1) {
                currentLen = static_cast<uint32_t>(blockLength - offset);
            }
            CopyIn(offset, currentLen);
            Compute(currentLen);
            CopyOut(offset, currentLen);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t len) {
        LocalTensor<DT_X> bufBase = inQueueX.template AllocTensor<DT_X>();
        uint32_t alignedLen = ((len + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
        DataCopy(bufBase, xGm[offset], alignedLen);
        inQueueX.EnQue(bufBase);
    }

    __aicore__ inline void Compute(uint32_t len) {
        LocalTensor<DT_X> t_x = inQueueX.template DeQue<DT_X>();
        LocalTensor<DT_X> t_out = outQueueY.template AllocTensor<DT_X>();
        LocalTensor<DT_X> calcLocal = calcBuf.template Get<DT_X>();
        LocalTensor<half> t_x16 = calcLocal.template ReinterpretCast<half>();
        LocalTensor<half> t_tmp16 = t_x16[tileLength];
        uint32_t computeLen = ((len + REPEAT_NUM - 1) / REPEAT_NUM) * REPEAT_NUM;
        ComputeErfMixed(t_x, t_out, t_x16, t_tmp16, computeLen);

        inQueueX.FreeTensor(t_x);
        outQueueY.template EnQue<DT_X>(t_out);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t len) {
        LocalTensor<DT_X> t_out = outQueueY.template DeQue<DT_X>();
        uint32_t alignedLen = ((len + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
        DataCopy(yGm[offset], t_out, alignedLen);
        outQueueY.FreeTensor(t_out);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> calcBuf;

    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;

    uint32_t tileLength;
    uint64_t blockLength;
};

template <class DT_X>
class KernelErfSingle {
public:
    __aicore__ inline KernelErfSingle() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint64_t totalLength, uint64_t bigCoreDataNum,
                                uint64_t smallCoreDataNum, uint32_t bigCoreCount,
                                uint32_t usedCoreCount, uint32_t tileLength) {
        (void)bigCoreDataNum;
        (void)smallCoreDataNum;
        (void)bigCoreCount;
        this->totalLength = totalLength;
        this->usedCoreCount = usedCoreCount;
        this->tileLength = tileLength;
        if (totalLength == 0 || static_cast<uint64_t>(GetBlockIdx()) >= usedCoreCount) {
            this->totalLength = 0;
            return;
        }

        xGm.SetGlobalBuffer((__gm__ DT_X *)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, totalLength);

        pipe.InitBuffer(buf, 3 * tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (totalLength == 0) {
            return;
        }
        uint64_t tileCount = (totalLength + static_cast<uint64_t>(tileLength) - 1) /
                             static_cast<uint64_t>(tileLength);
        LocalTensor<DT_X> t_x = buf.template Get<DT_X>();
        LocalTensor<DT_X> t_tmp = t_x[tileLength];
        LocalTensor<half> t_x16 = t_x[2 * tileLength].template ReinterpretCast<half>();
        LocalTensor<half> t_tmp16 = t_x16[tileLength];
        event_t eventIdLoad = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t eventIdStore = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        for (uint64_t tileIdx = static_cast<uint64_t>(GetBlockIdx()); tileIdx < tileCount;
             tileIdx += static_cast<uint64_t>(usedCoreCount)) {
            uint64_t offset = tileIdx * static_cast<uint64_t>(tileLength);
            uint32_t currentLen = tileLength;
            uint64_t remain = totalLength - offset;
            if (remain < static_cast<uint64_t>(tileLength)) {
                currentLen = static_cast<uint32_t>(remain);
            }

            uint32_t alignedLen = ((currentLen + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
            DataCopy(t_x, xGm[offset], alignedLen);
            SetFlag<HardEvent::MTE2_V>(eventIdLoad);
            WaitFlag<HardEvent::MTE2_V>(eventIdLoad);

            uint32_t computeLen = ((currentLen + REPEAT_NUM - 1) / REPEAT_NUM) * REPEAT_NUM;
            ComputeErfMixedInplace(t_x, t_tmp, t_x16, t_tmp16, computeLen);

            SetFlag<HardEvent::V_MTE3>(eventIdStore);
            WaitFlag<HardEvent::V_MTE3>(eventIdStore);

            DataCopy(yGm[offset], t_x, alignedLen);
        }
    }

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> buf;

    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;

    uint32_t tileLength;
    uint32_t usedCoreCount;
    uint64_t totalLength;
};

template <uint32_t schMode>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfOpTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfOpTilingData, tiling_data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if constexpr (schMode == ERF_SCH_MODE_SINGLE_BUFFER) {
        KernelErfSingle<float> op;
        op.Init(x, y, tiling_data.totalLength, tiling_data.bigCoreDataNum, tiling_data.smallCoreDataNum,
                tiling_data.bigCoreCount, tiling_data.usedCoreCount, tiling_data.tileLength);
        op.Process();
    }

    if constexpr (schMode == ERF_SCH_MODE_DOUBLE_BUFFER) {
        KernelErf<float, 2> op;
        op.Init(x, y, tiling_data.totalLength, tiling_data.bigCoreDataNum, tiling_data.smallCoreDataNum,
                tiling_data.bigCoreCount, tiling_data.usedCoreCount, tiling_data.tileLength);
        op.Process();
    }

    if constexpr (schMode == ERF_SCH_MODE_DIRECT_ONE_TILE) {
        TPipe pipe;
        TBuf<QuePosition::VECCALC> buf;
        GlobalTensor<float> xGm;
        GlobalTensor<float> yGm;

        constexpr uint32_t MIXED_ALIGN = 128;
        const uint32_t currentLen = static_cast<uint32_t>(tiling_data.totalLength);
        const uint32_t bufferLen = ((currentLen + MIXED_ALIGN - 1) / MIXED_ALIGN) * MIXED_ALIGN;

        xGm.SetGlobalBuffer((__gm__ float *)x, tiling_data.totalLength);
        yGm.SetGlobalBuffer((__gm__ float *)y, tiling_data.totalLength);

        pipe.InitBuffer(buf, 3 * bufferLen * sizeof(float));

        LocalTensor<float> t_x = buf.Get<float>();
        LocalTensor<float> t_tmp = t_x[bufferLen];
        LocalTensor<half> t_x16 = t_x[2 * bufferLen].template ReinterpretCast<half>();
        LocalTensor<half> t_tmp16 = t_x16[bufferLen];

        event_t eventIdLoad = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t eventIdStore = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

        DataCopy(t_x, xGm, bufferLen);
        SetFlag<HardEvent::MTE2_V>(eventIdLoad);
        WaitFlag<HardEvent::MTE2_V>(eventIdLoad);

        ComputeErfMixedInplace(t_x, t_tmp, t_x16, t_tmp16, bufferLen);

        SetFlag<HardEvent::V_MTE3>(eventIdStore);
        WaitFlag<HardEvent::V_MTE3>(eventIdStore);

        DataCopy(yGm, t_x, bufferLen);
    }

    if constexpr (schMode == ERF_SCH_MODE_DIRECT_SMALL) {
        const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t tileLen = static_cast<uint64_t>(tiling_data.tileLength);
        const uint64_t tileCount = (tiling_data.totalLength + tileLen - 1) / tileLen;
        const uint64_t usedCoreCount = static_cast<uint64_t>(tiling_data.usedCoreCount);
        if (blockIdx >= usedCoreCount) {
            return;
        }

        TPipe pipe;
        TBuf<QuePosition::VECCALC> buf;
        GlobalTensor<float> xGm;
        GlobalTensor<float> yGm;

        const uint32_t maxBufferLen = ((tiling_data.tileLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
        xGm.SetGlobalBuffer((__gm__ float *)x, tiling_data.totalLength);
        yGm.SetGlobalBuffer((__gm__ float *)y, tiling_data.totalLength);

        pipe.InitBuffer(buf, 3 * maxBufferLen * sizeof(float));

        LocalTensor<float> t_x = buf.Get<float>();
        LocalTensor<float> t_tmp = t_x[maxBufferLen];
        LocalTensor<half> t_x16 = t_x[2 * maxBufferLen].template ReinterpretCast<half>();
        LocalTensor<half> t_tmp16 = t_x16[maxBufferLen];

        event_t eventIdLoad = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t eventIdStore = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

        for (uint64_t tileIdx = blockIdx; tileIdx < tileCount; tileIdx += usedCoreCount) {
            uint64_t offset = tileIdx * tileLen;
            bool isTailBlock = (tileIdx + 1 == tileCount);
            uint32_t currentLen = isTailBlock
                                      ? static_cast<uint32_t>(tiling_data.totalLength - offset)
                                      : tiling_data.tileLength;
            uint32_t alignedLen = ((currentLen + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;

            DataCopy(t_x, xGm[offset], alignedLen);
            SetFlag<HardEvent::MTE2_V>(eventIdLoad);
            WaitFlag<HardEvent::MTE2_V>(eventIdLoad);

            uint32_t computeLen = ((currentLen + REPEAT_NUM - 1) / REPEAT_NUM) * REPEAT_NUM;
            ComputeErfMixedInplace(t_x, t_tmp, t_x16, t_tmp16, computeLen);

            SetFlag<HardEvent::V_MTE3>(eventIdStore);
            WaitFlag<HardEvent::V_MTE3>(eventIdStore);

            DataCopy(yGm[offset], t_x, alignedLen);
        }
    }

    if constexpr (schMode == ERF_SCH_MODE_DIRECT_TINY) {
        TPipe pipe;
        TBuf<QuePosition::VECCALC> buf;
        GlobalTensor<float> xGm;
        GlobalTensor<float> yGm;

        const uint32_t currentLen = static_cast<uint32_t>(tiling_data.totalLength);
        const uint32_t bufferLen = ((currentLen + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x, tiling_data.totalLength);
        yGm.SetGlobalBuffer((__gm__ float *)y, tiling_data.totalLength);

        pipe.InitBuffer(buf, 2 * bufferLen * sizeof(float));

        LocalTensor<float> t_x = buf.Get<float>();
        LocalTensor<float> t_tmp = t_x[bufferLen];

        event_t eventIdLoad = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t eventIdStore = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

        DataCopy(t_x, xGm, bufferLen);
        SetFlag<HardEvent::MTE2_V>(eventIdLoad);
        WaitFlag<HardEvent::MTE2_V>(eventIdLoad);

        ComputeErfSigmoidInplace<float>(t_x, t_tmp, bufferLen);

        SetFlag<HardEvent::V_MTE3>(eventIdStore);
        WaitFlag<HardEvent::V_MTE3>(eventIdStore);

        DataCopy(yGm, t_x, bufferLen);
    }
}
