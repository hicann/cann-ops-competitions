// Kernel侧核函数实现
#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t STATIC_SMALL_EXACT_CORE_LENGTH = 16;
constexpr uint32_t STATIC_SMALL_RANGE_CORE_LENGTH = 32;
constexpr uint32_t STATIC_TINY_TILE_LENGTH = 128;
constexpr uint32_t STATIC_TILE_LENGTH = 512;
constexpr uint32_t STATIC_X_ADDR = 0;
constexpr int32_t STATIC_EVENT_ID = 0;

// Fast odd degree-9 polynomial path with direct saturation for |x| >= 2.34.
template <class DT_X>
__aicore__ inline void ComputeErfFastDirect(
    LocalTensor<DT_X> xLocal,
    LocalTensor<DT_X> yLocal,
    LocalTensor<DT_X> xSquared,
    uint32_t calcLength
) {
    Mul(xSquared, xLocal, xLocal, calcLength);

    Muls(yLocal, xSquared, static_cast<DT_X>(0.000718526484f), calcLength);
    Adds(yLocal, yLocal, static_cast<DT_X>(-0.012470558286f), calcLength);
    Mul(yLocal, yLocal, xSquared, calcLength);
    Adds(yLocal, yLocal, static_cast<DT_X>(0.089340314269f), calcLength);
    Mul(yLocal, yLocal, xSquared, calcLength);
    Adds(yLocal, yLocal, static_cast<DT_X>(-0.360910087824f), calcLength);
    Mul(yLocal, yLocal, xSquared, calcLength);
    Adds(yLocal, yLocal, static_cast<DT_X>(1.126596808434f), calcLength);
    Mul(yLocal, yLocal, xLocal, calcLength);
    Mins(yLocal, yLocal, static_cast<DT_X>(1.0f), calcLength);
    Maxs(yLocal, yLocal, static_cast<DT_X>(-1.0f), calcLength);

}

template <class DT_X, uint32_t STATIC_LEN, bool IS_ALIGNED>
__aicore__ inline void ProcessStaticErf(GM_ADDR x, GM_ADDR y, uint32_t gmOffset, uint32_t calcLength) {
    if (calcLength == 0) return;

    // 静态 UB 地址连续排布：x / y / xSquared。
    constexpr uint32_t staticYAddr = STATIC_LEN * sizeof(float);
    constexpr uint32_t staticTmpAddr = staticYAddr + STATIC_LEN * sizeof(float);

    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    xGm.SetGlobalBuffer((__gm__ DT_X*)x + gmOffset, calcLength);
    yGm.SetGlobalBuffer((__gm__ DT_X*)y + gmOffset, calcLength);

    LocalTensor<DT_X> xLocal(TPosition::VECCALC, STATIC_X_ADDR, STATIC_LEN);
    LocalTensor<DT_X> yLocal(TPosition::VECCALC, staticYAddr, STATIC_LEN);
    LocalTensor<DT_X> xSquared(TPosition::VECCALC, staticTmpAddr, STATIC_LEN);

    if constexpr (IS_ALIGNED) {
        DataCopy(xLocal, xGm, calcLength);
    } else {
        DataCopyParams copyParams{1, static_cast<uint16_t>(calcLength * sizeof(DT_X)), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm, copyParams, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(STATIC_EVENT_ID);
    WaitFlag<HardEvent::MTE2_V>(STATIC_EVENT_ID);

    ComputeErfFastDirect(xLocal, yLocal, xSquared, calcLength);

    SetFlag<HardEvent::V_MTE3>(STATIC_EVENT_ID);
    WaitFlag<HardEvent::V_MTE3>(STATIC_EVENT_ID);
    if constexpr (IS_ALIGNED) {
        DataCopy(yGm, yLocal, calcLength);
    } else {
        DataCopyParams copyParams{1, static_cast<uint16_t>(calcLength * sizeof(DT_X)), 0, 0};
        DataCopyPad(yGm, yLocal, copyParams);
    }
}

template <class DT_X, int USE_ALIGNED>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length, uint32_t tileLength,
        uint32_t smallCoreDataNum, uint32_t bigCoreDataNum, uint32_t tailBlockNum) {
        uint32_t blockIdx = GetBlockIdx();
        if (blockIdx < tailBlockNum) {
            this->offset = blockIdx * bigCoreDataNum;
            this->coreLength = bigCoreDataNum;
        } else {
            this->offset = tailBlockNum * bigCoreDataNum + (blockIdx - tailBlockNum) * smallCoreDataNum;
            this->coreLength = smallCoreDataNum;
        }
        this->coreLength = (this->offset < length) ? Min(this->coreLength, length - this->offset) : 0;
        this->tileLength = tileLength;

        xGm.SetGlobalBuffer((__gm__ DT_X*)x + this->offset, this->coreLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + this->offset, this->coreLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(tmpBuf, tileLength * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        if (coreLength == 0) {
            return;
        }
        ProcessQueue();
    }

private:
    // 普通队列路径：用于 length > 512 或显式回退的小张量队列实验。
    __aicore__ inline void ProcessQueue() {
        if (coreLength <= tileLength) {
            CopyIn(0, coreLength);
            ComputeTile(coreLength);
            CopyOut(0, coreLength);
            return;
        }

        uint32_t remain = coreLength;
        uint32_t gmOffset = 0;
        while (remain > 0) {
            uint32_t calcLength = remain > tileLength ? tileLength : remain;
            CopyIn(gmOffset, calcLength);
            ComputeTile(calcLength);
            CopyOut(gmOffset, calcLength);
            gmOffset += calcLength;
            remain -= calcLength;
        }
    }

    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) {
        return a < b ? a : b;
    }

    __aicore__ inline void CopyIn(uint32_t gmOffset, uint32_t calcLength) {
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        if constexpr (USE_ALIGNED == 1) {
            DataCopy(xLocal, xGm[gmOffset], calcLength);
        } else {
            if (calcLength % 8 == 0) {
                DataCopy(xLocal, xGm[gmOffset], calcLength);
            } else {
                DataCopyParams copyParams{1, static_cast<uint16_t>(calcLength * sizeof(DT_X)), 0, 0};
                DataCopyPadParams padParams{false, 0, 0, 0};
                DataCopyPad(xLocal, xGm[gmOffset], copyParams, padParams);
            }
        }
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void ComputeTile(uint32_t calcLength) {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        LocalTensor<DT_X> xSquared = tmpBuf.Get<DT_X>();

        ComputeErfFastDirect(xLocal, yLocal, xSquared, calcLength);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t gmOffset, uint32_t calcLength) {
        LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        if constexpr (USE_ALIGNED == 1) {
            DataCopy(yGm[gmOffset], yLocal, calcLength);
        } else {
            if (calcLength % 8 == 0) {
                DataCopy(yGm[gmOffset], yLocal, calcLength);
            } else {
                DataCopyParams copyParams{1, static_cast<uint16_t>(calcLength * sizeof(DT_X)), 0, 0};
                DataCopyPad(yGm[gmOffset], yLocal, copyParams);
            }
        }
        outQueueY.FreeTensor(yLocal);
    }

    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<TPosition::VECCALC> tmpBuf;

    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    uint32_t offset;
    uint32_t coreLength;
    uint32_t tileLength;

};

template <typename DT_X, int USE_SMALL, int USE_ALIGNED = 0, int USE_TINY = 0>
 __global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    if constexpr (USE_SMALL == 1) {
        if (tiling_data.blockDim > 1) {
            uint32_t gmOffset = GetBlockIdx() * tiling_data.bigCoreDataNum;
            uint32_t calcLength = 0;
            if (gmOffset < tiling_data.length) {
                uint32_t remain = tiling_data.length - gmOffset;
                calcLength = remain > tiling_data.bigCoreDataNum ? tiling_data.bigCoreDataNum : remain;
            }
            if (tiling_data.bigCoreDataNum == STATIC_SMALL_EXACT_CORE_LENGTH) {
                InitSocState();
                ProcessStaticErf<DT_X, STATIC_SMALL_EXACT_CORE_LENGTH, (USE_ALIGNED == 1)>(
                    x, y, gmOffset, calcLength);
            } else if (tiling_data.bigCoreDataNum == STATIC_SMALL_RANGE_CORE_LENGTH) {
                InitSocState();
                ProcessStaticErf<DT_X, STATIC_SMALL_RANGE_CORE_LENGTH, (USE_ALIGNED == 1)>(
                    x, y, gmOffset, calcLength);
            } else {
                KernelErf<DT_X, USE_ALIGNED> op;
                op.Init(
                    x,
                    y,
                    tiling_data.length,
                    tiling_data.tileLength,
                    tiling_data.smallCoreDataNum,
                    tiling_data.bigCoreDataNum,
                    tiling_data.tailBlockNum
                );
                op.Process();
            }
        } else if (tiling_data.bigCoreDataNum > STATIC_TILE_LENGTH) {
            KernelErf<DT_X, USE_ALIGNED> op;
            op.Init(
                x,
                y,
                tiling_data.length,
                tiling_data.tileLength,
                tiling_data.smallCoreDataNum,
                tiling_data.bigCoreDataNum,
                tiling_data.tailBlockNum
            );
            op.Process();
        } else if constexpr (USE_TINY == 1) {
            InitSocState();
            ProcessStaticErf<DT_X, STATIC_TINY_TILE_LENGTH, (USE_ALIGNED == 1)>(x, y, 0, tiling_data.length);
        } else {
            InitSocState();
            ProcessStaticErf<DT_X, STATIC_TILE_LENGTH, (USE_ALIGNED == 1)>(x, y, 0, tiling_data.length);
        }
        return;
    }

    KernelErf<DT_X, USE_ALIGNED> op;
    op.Init(
        x,
        y,
        tiling_data.length,
        tiling_data.tileLength,
        tiling_data.smallCoreDataNum,
        tiling_data.bigCoreDataNum,
        tiling_data.tailBlockNum
    );
    op.Process();
}
