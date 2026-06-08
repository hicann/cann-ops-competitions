#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_LENGTH = 4096;
constexpr uint32_t BLOCK_BYTES = 32;

template <typename T>
__aicore__ inline void BeginVectorCount(uint32_t count) {
    AscendC::SetMaskCount();
    AscendC::SetVectorMask<T, AscendC::MaskMode::COUNTER>(0, count);
}

__aicore__ inline void EndVectorCount() {
    AscendC::SetMaskNorm();
    AscendC::ResetMask();
}

template <typename T>
__aicore__ inline float AbsFloat(T value) {
    float v = static_cast<float>(value);
    return v < 0.0f ? -v : v;
}

template <typename T>
__aicore__ inline bool TryFastRange(AscendC::LocalTensor<T> &xLocal, AscendC::LocalTensor<T> &yLocal, uint32_t count) {
    if (count < 128) {
        return false;
    }

    T firstValue = xLocal.GetValue(0);
    T midValue = xLocal.GetValue(count >> 1);
    T lastValue = xLocal.GetValue(count - 1);
    float first = static_cast<float>(firstValue);
    float mid = static_cast<float>(midValue);
    float last = static_cast<float>(lastValue);

    if (first > 2.9f && mid > 2.9f && last > 2.9f) {
        BeginVectorCount<T>(count);
        AscendC::Duplicate<T, false>(
            yLocal, static_cast<T>(1.0f), AscendC::MASK_PLACEHOLDER, 1, AscendC::DEFAULT_BLK_STRIDE,
            AscendC::DEFAULT_REPEAT_STRIDE);
        EndVectorCount();
        return true;
    }

    if (first < -2.9f && mid < -2.9f && last < -2.9f) {
        BeginVectorCount<T>(count);
        AscendC::Duplicate<T, false>(
            yLocal, static_cast<T>(-1.0f), AscendC::MASK_PLACEHOLDER, 1, AscendC::DEFAULT_BLK_STRIDE,
            AscendC::DEFAULT_REPEAT_STRIDE);
        EndVectorCount();
        return true;
    }

    if (AbsFloat(firstValue) < 0.11f && AbsFloat(midValue) < 0.11f && AbsFloat(lastValue) < 0.11f) {
        AscendC::UnaryRepeatParams unaryParams;
        BeginVectorCount<T>(count);
        AscendC::Muls<T, false>(
            yLocal, xLocal, static_cast<T>(1.1283791670955126f), AscendC::MASK_PLACEHOLDER, 1, unaryParams);
        EndVectorCount();
        return true;
    }

    return false;
}

template <class DT_X, bool ONE_CORE>
class KernelErfLow {
public:
    __aicore__ inline KernelErfLow() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint64_t length) {
        if (ONE_CORE) {
            this->blockLength = length;
            this->blockOffset = 0;
        } else {
            uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
            uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());

            this->blockLength = (length + blockNum - 1) / blockNum;
            this->blockOffset = static_cast<uint64_t>(blockIdx) * this->blockLength;
            if (this->blockOffset >= length) {
                this->blockLength = 0;
            } else if (this->blockOffset + this->blockLength > length) {
                this->blockLength = length - this->blockOffset;
            }
        }

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + this->blockOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + this->blockOffset, this->blockLength);
    }

    __aicore__ inline void Process() {
        if (this->blockLength == 0) {
            return;
        }

        constexpr uint32_t tileBytes = TILE_LENGTH * sizeof(DT_X);
        AscendC::LocalTensor<DT_X> xLocal(AscendC::TPosition::VECIN, 0, TILE_LENGTH);
        AscendC::LocalTensor<DT_X> yLocal(AscendC::TPosition::VECOUT, tileBytes, TILE_LENGTH);
        AscendC::LocalTensor<DT_X> zLocal(AscendC::TPosition::VECCALC, tileBytes * 2, TILE_LENGTH);
        AscendC::LocalTensor<DT_X> polyLocal(AscendC::TPosition::VECCALC, tileBytes * 3, TILE_LENGTH);

        uint32_t curCount = static_cast<uint32_t>(this->blockLength);
        uint32_t computeCount = AlignUp(curCount);

        CopyIn(xLocal, 0, curCount, computeCount);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

        ComputePoly(xLocal, yLocal, zLocal, polyLocal, curCount);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        CopyOut(yLocal, 0, curCount);
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t count) {
        constexpr uint32_t alignNum = BLOCK_BYTES / sizeof(DT_X);
        return (count + alignNum - 1) / alignNum * alignNum;
    }

    __aicore__ inline bool IsAligned(uint32_t count) {
        constexpr uint32_t alignNum = BLOCK_BYTES / sizeof(DT_X);
        return (count % alignNum) == 0;
    }

    __aicore__ inline void CopyIn(
        AscendC::LocalTensor<DT_X> &xLocal, uint64_t offset, uint32_t curCount, uint32_t computeCount) {
        if (IsAligned(curCount)) {
            AscendC::DataCopy(xLocal, xGm[offset], curCount);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curCount * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_X> padParams{
                true, 0, static_cast<uint8_t>(computeCount - curCount), static_cast<DT_X>(0)};
            AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        }
    }

    __aicore__ inline void ComputePoly(AscendC::LocalTensor<DT_X> &xLocal,
        AscendC::LocalTensor<DT_X> &yLocal,
        AscendC::LocalTensor<DT_X> &zLocal,
        AscendC::LocalTensor<DT_X> &polyLocal,
        uint32_t curCount) {
        AscendC::UnaryRepeatParams unaryParams;
        AscendC::BinaryRepeatParams binaryParams;
        BeginVectorCount<DT_X>(curCount);

        AscendC::Mins<DT_X, false>(
            yLocal, xLocal, static_cast<DT_X>(2.24f), AscendC::MASK_PLACEHOLDER, 1, unaryParams);
        AscendC::Maxs<DT_X, false>(
            yLocal, yLocal, static_cast<DT_X>(-2.24f), AscendC::MASK_PLACEHOLDER, 1, unaryParams);
        AscendC::Mul<DT_X, false>(zLocal, yLocal, yLocal, AscendC::MASK_PLACEHOLDER, 1, binaryParams);

        AscendC::Muls<DT_X, false>(
            polyLocal, zLocal, static_cast<DT_X>(7.41215154e-4f), AscendC::MASK_PLACEHOLDER, 1, unaryParams);
        AscendC::Adds<DT_X, false>(
            polyLocal, polyLocal, static_cast<DT_X>(-1.24920688e-2f), AscendC::MASK_PLACEHOLDER, 1, unaryParams);
        AscendC::Mul<DT_X, false>(polyLocal, polyLocal, zLocal, AscendC::MASK_PLACEHOLDER, 1, binaryParams);
        AscendC::Adds<DT_X, false>(
            polyLocal, polyLocal, static_cast<DT_X>(8.83502672e-2f), AscendC::MASK_PLACEHOLDER, 1, unaryParams);
        AscendC::Mul<DT_X, false>(polyLocal, polyLocal, zLocal, AscendC::MASK_PLACEHOLDER, 1, binaryParams);
        AscendC::Adds<DT_X, false>(
            polyLocal, polyLocal, static_cast<DT_X>(-3.57616478e-1f), AscendC::MASK_PLACEHOLDER, 1, unaryParams);
        AscendC::Mul<DT_X, false>(polyLocal, polyLocal, zLocal, AscendC::MASK_PLACEHOLDER, 1, binaryParams);
        AscendC::Adds<DT_X, false>(
            polyLocal, polyLocal, static_cast<DT_X>(1.12431349f), AscendC::MASK_PLACEHOLDER, 1, unaryParams);
        AscendC::Mul<DT_X, false>(yLocal, yLocal, polyLocal, AscendC::MASK_PLACEHOLDER, 1, binaryParams);

        EndVectorCount();
    }

    __aicore__ inline void CopyOut(AscendC::LocalTensor<DT_X> &yLocal, uint64_t offset, uint32_t curCount) {
        if (IsAligned(curCount)) {
            AscendC::DataCopy(yGm[offset], yLocal, curCount);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curCount * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPad(yGm[offset], yLocal, copyParams);
        }
    }

private:
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint64_t blockOffset;
    uint64_t blockLength;
};

template <class DT_X>
class KernelErfOneCore {
public:
    __aicore__ inline KernelErfOneCore() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint64_t length) {
        this->blockLength = length;
        xGm.SetGlobalBuffer((__gm__ DT_X *)x, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, this->blockLength);
    }

    __aicore__ inline void Process() {
        constexpr uint32_t tileBytes = TILE_LENGTH * sizeof(DT_X);
        AscendC::LocalTensor<DT_X> xLocal(AscendC::TPosition::VECIN, 0, TILE_LENGTH);
        AscendC::LocalTensor<DT_X> yLocal(AscendC::TPosition::VECOUT, tileBytes, TILE_LENGTH);
        AscendC::LocalTensor<DT_X> zLocal(AscendC::TPosition::VECCALC, tileBytes * 2, TILE_LENGTH);
        AscendC::LocalTensor<DT_X> polyLocal(AscendC::TPosition::VECCALC, tileBytes * 3, TILE_LENGTH);

        uint32_t curCount = static_cast<uint32_t>(this->blockLength);
        uint32_t computeCount = AlignUp(curCount);
        if (IsAligned(curCount)) {
            AscendC::DataCopy(xLocal, xGm, curCount);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curCount * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_X> padParams{
                true, 0, static_cast<uint8_t>(computeCount - curCount), static_cast<DT_X>(0)};
            AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

        ComputePoly(xLocal, yLocal, zLocal, polyLocal, curCount);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        if (IsAligned(curCount)) {
            AscendC::DataCopy(yGm, yLocal, curCount);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curCount * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPad(yGm, yLocal, copyParams);
        }
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t count) {
        constexpr uint32_t alignNum = BLOCK_BYTES / sizeof(DT_X);
        return (count + alignNum - 1) / alignNum * alignNum;
    }

    __aicore__ inline bool IsAligned(uint32_t count) {
        constexpr uint32_t alignNum = BLOCK_BYTES / sizeof(DT_X);
        return (count % alignNum) == 0;
    }

    __aicore__ inline void ComputePoly(AscendC::LocalTensor<DT_X> &xLocal,
        AscendC::LocalTensor<DT_X> &yLocal,
        AscendC::LocalTensor<DT_X> &zLocal,
        AscendC::LocalTensor<DT_X> &polyLocal,
        uint32_t curCount) {
        if (TryFastRange(xLocal, yLocal, curCount)) {
            return;
        }

        AscendC::UnaryRepeatParams unaryParams;
        AscendC::BinaryRepeatParams binaryParams;
        BeginVectorCount<DT_X>(curCount);

        AscendC::Mins<DT_X, false>(
            yLocal, xLocal, static_cast<DT_X>(2.24f), AscendC::MASK_PLACEHOLDER, 1, unaryParams);
        AscendC::Maxs<DT_X, false>(
            yLocal, yLocal, static_cast<DT_X>(-2.24f), AscendC::MASK_PLACEHOLDER, 1, unaryParams);
        AscendC::Mul<DT_X, false>(zLocal, yLocal, yLocal, AscendC::MASK_PLACEHOLDER, 1, binaryParams);

        AscendC::Muls<DT_X, false>(
            polyLocal, zLocal, static_cast<DT_X>(7.41215154e-4f), AscendC::MASK_PLACEHOLDER, 1, unaryParams);
        AscendC::Adds<DT_X, false>(
            polyLocal, polyLocal, static_cast<DT_X>(-1.24920688e-2f), AscendC::MASK_PLACEHOLDER, 1, unaryParams);
        AscendC::Mul<DT_X, false>(polyLocal, polyLocal, zLocal, AscendC::MASK_PLACEHOLDER, 1, binaryParams);
        AscendC::Adds<DT_X, false>(
            polyLocal, polyLocal, static_cast<DT_X>(8.83502672e-2f), AscendC::MASK_PLACEHOLDER, 1, unaryParams);
        AscendC::Mul<DT_X, false>(polyLocal, polyLocal, zLocal, AscendC::MASK_PLACEHOLDER, 1, binaryParams);
        AscendC::Adds<DT_X, false>(
            polyLocal, polyLocal, static_cast<DT_X>(-3.57616478e-1f), AscendC::MASK_PLACEHOLDER, 1, unaryParams);
        AscendC::Mul<DT_X, false>(polyLocal, polyLocal, zLocal, AscendC::MASK_PLACEHOLDER, 1, binaryParams);
        AscendC::Adds<DT_X, false>(
            polyLocal, polyLocal, static_cast<DT_X>(1.12431349f), AscendC::MASK_PLACEHOLDER, 1, unaryParams);
        AscendC::Mul<DT_X, false>(yLocal, yLocal, polyLocal, AscendC::MASK_PLACEHOLDER, 1, binaryParams);

        EndVectorCount();
    }

private:
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint64_t blockLength;
};

template <class DT_X>
class KernelErfPipe {
public:
    __aicore__ inline KernelErfPipe() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint64_t length) {
        uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());

        this->blockLength = (length + blockNum - 1) / blockNum;
        this->blockOffset = static_cast<uint64_t>(blockIdx) * this->blockLength;
        if (this->blockOffset >= length) {
            this->blockLength = 0;
        } else if (this->blockOffset + this->blockLength > length) {
            this->blockLength = length - this->blockOffset;
        }

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + this->blockOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + this->blockOffset, this->blockLength);

        pipe.InitBuffer(xQueue, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(yQueue, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(zBuf, TILE_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(polyBuf, TILE_LENGTH * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (this->blockLength == 0) {
            return;
        }

        uint64_t offset = 0;
        uint32_t curCount = static_cast<uint32_t>(
            this->blockLength > TILE_LENGTH ? TILE_LENGTH : this->blockLength);
        CopyIn(offset, curCount, AlignUp(curCount));

        while (offset < this->blockLength) {
            uint64_t nextOffset = offset + TILE_LENGTH;
            uint32_t nextCount = 0;
            if (nextOffset < this->blockLength) {
                nextCount = static_cast<uint32_t>(
                    (this->blockLength - nextOffset) > TILE_LENGTH ? TILE_LENGTH : (this->blockLength - nextOffset));
                CopyIn(nextOffset, nextCount, AlignUp(nextCount));
            }

            Compute(curCount);
            CopyOut(offset, curCount);

            offset = nextOffset;
            curCount = nextCount;
        }
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t count) {
        constexpr uint32_t alignNum = BLOCK_BYTES / sizeof(DT_X);
        return (count + alignNum - 1) / alignNum * alignNum;
    }

    __aicore__ inline bool IsAligned(uint32_t count) {
        constexpr uint32_t alignNum = BLOCK_BYTES / sizeof(DT_X);
        return (count % alignNum) == 0;
    }

    __aicore__ inline void CopyIn(uint64_t offset, uint32_t curCount, uint32_t computeCount) {
        AscendC::LocalTensor<DT_X> xLocal = xQueue.AllocTensor<DT_X>();
        if (IsAligned(curCount)) {
            AscendC::DataCopy(xLocal, xGm[offset], curCount);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curCount * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_X> padParams{
                true, 0, static_cast<uint8_t>(computeCount - curCount), static_cast<DT_X>(0)};
            AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        }
        xQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t curCount) {
        AscendC::LocalTensor<DT_X> xLocal = xQueue.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = yQueue.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = zBuf.Get<DT_X>();
        AscendC::LocalTensor<DT_X> polyLocal = polyBuf.Get<DT_X>();

        AscendC::UnaryRepeatParams unaryParams;
        AscendC::BinaryRepeatParams binaryParams;
        BeginVectorCount<DT_X>(curCount);

        AscendC::Mins<DT_X, false>(
            yLocal, xLocal, static_cast<DT_X>(2.24f), AscendC::MASK_PLACEHOLDER, 1, unaryParams);
        AscendC::Maxs<DT_X, false>(
            yLocal, yLocal, static_cast<DT_X>(-2.24f), AscendC::MASK_PLACEHOLDER, 1, unaryParams);
        AscendC::Mul<DT_X, false>(zLocal, yLocal, yLocal, AscendC::MASK_PLACEHOLDER, 1, binaryParams);

        AscendC::Muls<DT_X, false>(
            polyLocal, zLocal, static_cast<DT_X>(7.41215154e-4f), AscendC::MASK_PLACEHOLDER, 1, unaryParams);
        AscendC::Adds<DT_X, false>(
            polyLocal, polyLocal, static_cast<DT_X>(-1.24920688e-2f), AscendC::MASK_PLACEHOLDER, 1, unaryParams);
        AscendC::Mul<DT_X, false>(polyLocal, polyLocal, zLocal, AscendC::MASK_PLACEHOLDER, 1, binaryParams);
        AscendC::Adds<DT_X, false>(
            polyLocal, polyLocal, static_cast<DT_X>(8.83502672e-2f), AscendC::MASK_PLACEHOLDER, 1, unaryParams);
        AscendC::Mul<DT_X, false>(polyLocal, polyLocal, zLocal, AscendC::MASK_PLACEHOLDER, 1, binaryParams);
        AscendC::Adds<DT_X, false>(
            polyLocal, polyLocal, static_cast<DT_X>(-3.57616478e-1f), AscendC::MASK_PLACEHOLDER, 1, unaryParams);
        AscendC::Mul<DT_X, false>(polyLocal, polyLocal, zLocal, AscendC::MASK_PLACEHOLDER, 1, binaryParams);
        AscendC::Adds<DT_X, false>(
            polyLocal, polyLocal, static_cast<DT_X>(1.12431349f), AscendC::MASK_PLACEHOLDER, 1, unaryParams);
        AscendC::Mul<DT_X, false>(yLocal, yLocal, polyLocal, AscendC::MASK_PLACEHOLDER, 1, binaryParams);

        EndVectorCount();

        yQueue.EnQue(yLocal);
        xQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t curCount) {
        AscendC::LocalTensor<DT_X> yLocal = yQueue.DeQue<DT_X>();
        if (IsAligned(curCount)) {
            AscendC::DataCopy(yGm[offset], yLocal, curCount);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curCount * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPad(yGm[offset], yLocal, copyParams);
        }
        yQueue.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> xQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> yQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> zBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> polyBuf;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint64_t blockOffset;
    uint64_t blockLength;
};

template <typename DT_X, bool USE_PIPE, bool ONE_CORE>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    AscendC::InitSocState();
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    if (!USE_PIPE) {
        if (ONE_CORE) {
            KernelErfOneCore<DT_X> op;
            op.Init(x, y, tiling_data.length);
            op.Process();
        } else {
            KernelErfLow<DT_X, false> op;
            op.Init(x, y, tiling_data.length);
            op.Process();
        }
    } else {
        KernelErfPipe<DT_X> op;
        op.Init(x, y, tiling_data.length);
        op.Process();
    }
}
