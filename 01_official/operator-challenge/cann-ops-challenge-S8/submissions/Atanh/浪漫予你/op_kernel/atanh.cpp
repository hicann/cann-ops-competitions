#include "kernel_operator.h"

using namespace AscendC;

template <typename T>
class KernelAtanhFloatManual {
private:
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;

    uint32_t size;
    uint32_t tileLength;
    uint32_t lastTileLength;
    uint32_t tileNum;

    static constexpr uint32_t TILE_LENGTH = 11 * 1024;

    static constexpr uint32_t xAddrPing = 0x00000;
    static constexpr uint32_t yAddrPing = 0x0B100;
    static constexpr uint32_t xAddrPong = 0x16200;
    static constexpr uint32_t yAddrPong = 0x21300;

public:
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t smallSize,
        uint32_t formerNum,
        uint32_t tilingIncSize,
        uint32_t totalLength
    ) {
        uint32_t srcBeginIndex;
        uint32_t incSize = tilingIncSize;

        if (GetBlockIdx() < formerNum) {
            size = smallSize + incSize;
            srcBeginIndex = size * GetBlockIdx();
        } else {
            size = smallSize;
            srcBeginIndex = size * GetBlockIdx() + formerNum * incSize;
        }

        if (srcBeginIndex + size > totalLength) {
            size = totalLength - srcBeginIndex;
        }

        xGm.SetGlobalBuffer((__gm__ T*)x + srcBeginIndex, size);
        yGm.SetGlobalBuffer((__gm__ T*)y + srcBeginIndex, size);

        tileLength = min(uint32_t(size), uint32_t(TILE_LENGTH));
        tileNum = (size + tileLength - 1) / tileLength;
        lastTileLength = size - (tileNum - 1) * tileLength;
    }

    __aicore__ inline void ComputeBody(
        LocalTensor<T>& xLocal,
        LocalTensor<T>& yLocal,
        uint32_t calcLength
    ) {
        Adds(yLocal, xLocal, T(1.0f), calcLength);
        Muls(xLocal, xLocal, T(-1.0f), calcLength);
        Adds(xLocal, xLocal, T(1.0f), calcLength);

        Ln(yLocal, yLocal, calcLength);
        Ln(xLocal, xLocal, calcLength);

        Sub(yLocal, yLocal, xLocal, calcLength);
        Muls(yLocal, yLocal, T(0.5f), calcLength);
    }

    __aicore__ inline void RunOneTile(
        uint32_t offset,
        uint32_t calcLength,
        int32_t eventID,
        LocalTensor<T>& xLocal,
        LocalTensor<T>& yLocal
    ) {
        WaitFlag<HardEvent::MTE3_MTE2>(eventID);

        DataCopy(xLocal, xGm[offset], calcLength);

        SetFlag<HardEvent::MTE2_V>(eventID);
        WaitFlag<HardEvent::MTE2_V>(eventID);

        ComputeBody(xLocal, yLocal, calcLength);

        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);

        DataCopy(yGm[offset], yLocal, calcLength);

        SetFlag<HardEvent::MTE3_MTE2>(eventID);
    }

    __aicore__ inline void Process() {
        LocalTensor<T> xPing(TPosition::VECCALC, xAddrPing, tileLength);
        LocalTensor<T> yPing(TPosition::VECCALC, yAddrPing, tileLength);
        LocalTensor<T> xPong(TPosition::VECCALC, xAddrPong, tileLength);
        LocalTensor<T> yPong(TPosition::VECCALC, yAddrPong, tileLength);

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

        uint32_t offset = 0;

        for (uint32_t i = 0; i + 1 < tileNum; ++i) {
            int32_t eventID = (i & 1) ? EVENT_ID1 : EVENT_ID0;

            LocalTensor<T>& xLocal = (i & 1) ? xPong : xPing;
            LocalTensor<T>& yLocal = (i & 1) ? yPong : yPing;

            RunOneTile(offset, tileLength, eventID, xLocal, yLocal);
            offset += tileLength;
        }

        {
            uint32_t i = tileNum - 1;
            int32_t eventID = (i & 1) ? EVENT_ID1 : EVENT_ID0;

            LocalTensor<T>& xLocal = (i & 1) ? xPong : xPing;
            LocalTensor<T>& yLocal = (i & 1) ? yPong : yPing;

            RunOneTile(offset, lastTileLength, eventID, xLocal, yLocal);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
};


template <typename T>
class KernelAtanhHalfManual {
private:
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;

    uint32_t size;
    uint32_t tileLength;
    uint32_t lastTileLength;
    uint32_t tileNum;

    static constexpr uint32_t TILE_LENGTH = 16 * 1024;

    static constexpr uint32_t xAddrPing = 0x00000;
    static constexpr uint32_t yAddrPing = 0x08100;
    static constexpr uint32_t xAddrPong = 0x10200;
    static constexpr uint32_t yAddrPong = 0x18300;

public:
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t smallSize,
        uint32_t formerNum,
        uint32_t tilingIncSize,
        uint32_t totalLength
    ) {
        uint32_t srcBeginIndex;
        uint32_t incSize = tilingIncSize;

        if (GetBlockIdx() < formerNum) {
            size = smallSize + incSize;
            srcBeginIndex = size * GetBlockIdx();
        } else {
            size = smallSize;
            srcBeginIndex = size * GetBlockIdx() + formerNum * incSize;
        }

        if (srcBeginIndex + size > totalLength) {
            size = totalLength - srcBeginIndex;
        }

        xGm.SetGlobalBuffer((__gm__ T*)x + srcBeginIndex, size);
        yGm.SetGlobalBuffer((__gm__ T*)y + srcBeginIndex, size);

        tileLength = min(uint32_t(size), uint32_t(TILE_LENGTH));
        tileNum = (size + tileLength - 1) / tileLength;
        lastTileLength = size - (tileNum - 1) * tileLength;
    }

    __aicore__ inline void ComputeBody(
        LocalTensor<T>& xLocal,
        LocalTensor<T>& yLocal,
        uint32_t calcLength
    ) {
        Adds(yLocal, xLocal, T(1.0f), calcLength);
        Muls(xLocal, xLocal, T(-1.0f), calcLength);
        Adds(xLocal, xLocal, T(1.0f), calcLength);

        Ln(yLocal, yLocal, calcLength);
        Ln(xLocal, xLocal, calcLength);

        Sub(yLocal, yLocal, xLocal, calcLength);
        Muls(yLocal, yLocal, T(0.5f), calcLength);
    }

    __aicore__ inline void RunOneTile(
        uint32_t offset,
        uint32_t calcLength,
        int32_t eventID,
        LocalTensor<T>& xLocal,
        LocalTensor<T>& yLocal
    ) {
        WaitFlag<HardEvent::MTE3_MTE2>(eventID);

        DataCopy(xLocal, xGm[offset], calcLength);

        SetFlag<HardEvent::MTE2_V>(eventID);
        WaitFlag<HardEvent::MTE2_V>(eventID);

        ComputeBody(xLocal, yLocal, calcLength);

        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);

        DataCopy(yGm[offset], yLocal, calcLength);

        SetFlag<HardEvent::MTE3_MTE2>(eventID);
    }

    __aicore__ inline void Process() {
        LocalTensor<T> xPing(TPosition::VECCALC, xAddrPing, tileLength);
        LocalTensor<T> yPing(TPosition::VECCALC, yAddrPing, tileLength);
        LocalTensor<T> xPong(TPosition::VECCALC, xAddrPong, tileLength);
        LocalTensor<T> yPong(TPosition::VECCALC, yAddrPong, tileLength);

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

        uint32_t offset = 0;

        for (uint32_t i = 0; i + 1 < tileNum; ++i) {
            int32_t eventID = (i & 1) ? EVENT_ID1 : EVENT_ID0;

            LocalTensor<T>& xLocal = (i & 1) ? xPong : xPing;
            LocalTensor<T>& yLocal = (i & 1) ? yPong : yPing;

            RunOneTile(offset, tileLength, eventID, xLocal, yLocal);
            offset += tileLength;
        }

        {
            uint32_t i = tileNum - 1;
            int32_t eventID = (i & 1) ? EVENT_ID1 : EVENT_ID0;

            LocalTensor<T>& xLocal = (i & 1) ? xPong : xPing;
            LocalTensor<T>& yLocal = (i & 1) ? yPong : yPing;

            RunOneTile(offset, lastTileLength, eventID, xLocal, yLocal);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
};


template <typename T>
class KernelAtanhBFloatManual {
private:
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;

    uint32_t size;
    uint32_t tileLength;
    uint32_t lastTileLength;
    uint32_t tileNum;

    static constexpr uint32_t TILE_LENGTH = 9 * 1024;

    static constexpr uint32_t xyAddrPing  = 0x00000;
    static constexpr uint32_t xFAddrPing  = 0x05000;
    static constexpr uint32_t tmpAddrPing = 0x0E100;

    static constexpr uint32_t xyAddrPong  = 0x18000;
    static constexpr uint32_t xFAddrPong  = 0x1D100;
    static constexpr uint32_t tmpAddrPong = 0x26200;

public:
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t smallSize,
        uint32_t formerNum,
        uint32_t tilingIncSize,
        uint32_t totalLength
    ) {
        uint32_t srcBeginIndex;
        uint32_t incSize = tilingIncSize;

        if (GetBlockIdx() < formerNum) {
            size = smallSize + incSize;
            srcBeginIndex = size * GetBlockIdx();
        } else {
            size = smallSize;
            srcBeginIndex = size * GetBlockIdx() + formerNum * incSize;
        }

        if (srcBeginIndex + size > totalLength) {
            size = totalLength - srcBeginIndex;
        }

        xGm.SetGlobalBuffer((__gm__ T*)x + srcBeginIndex, size);
        yGm.SetGlobalBuffer((__gm__ T*)y + srcBeginIndex, size);

        tileLength = min(uint32_t(size), uint32_t(TILE_LENGTH));
        tileNum = (size + tileLength - 1) / tileLength;
        lastTileLength = size - (tileNum - 1) * tileLength;
    }

    __aicore__ inline void ComputeBody(
        LocalTensor<T>& xyLocal,
        LocalTensor<float>& xFloat,
        LocalTensor<float>& tmp,
        uint32_t calcLength
    ) {
        Cast(xFloat, xyLocal, RoundMode::CAST_NONE, calcLength);

        Adds(tmp, xFloat, 1.0f, calcLength);
        Muls(xFloat, xFloat, -1.0f, calcLength);
        Adds(xFloat, xFloat, 1.0f, calcLength);

        Div(tmp, tmp, xFloat, calcLength);
        Ln(tmp, tmp, calcLength);
        Muls(tmp, tmp, 0.5f, calcLength);

        Cast(xyLocal, tmp, RoundMode::CAST_RINT, calcLength);
    }

    __aicore__ inline void RunOneTile(
        uint32_t offset,
        uint32_t calcLength,
        int32_t eventID,
        LocalTensor<T>& xyLocal,
        LocalTensor<float>& xFloat,
        LocalTensor<float>& tmp
    ) {
        WaitFlag<HardEvent::MTE3_MTE2>(eventID);

        DataCopy(xyLocal, xGm[offset], calcLength);

        SetFlag<HardEvent::MTE2_V>(eventID);
        WaitFlag<HardEvent::MTE2_V>(eventID);

        ComputeBody(xyLocal, xFloat, tmp, calcLength);

        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);

        DataCopy(yGm[offset], xyLocal, calcLength);

        SetFlag<HardEvent::MTE3_MTE2>(eventID);
    }

    __aicore__ inline void Process() {
        LocalTensor<T> xyPing(TPosition::VECCALC, xyAddrPing, tileLength);
        LocalTensor<float> xFloatPing(TPosition::VECCALC, xFAddrPing, tileLength);
        LocalTensor<float> tmpPing(TPosition::VECCALC, tmpAddrPing, tileLength);

        LocalTensor<T> xyPong(TPosition::VECCALC, xyAddrPong, tileLength);
        LocalTensor<float> xFloatPong(TPosition::VECCALC, xFAddrPong, tileLength);
        LocalTensor<float> tmpPong(TPosition::VECCALC, tmpAddrPong, tileLength);

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

        uint32_t offset = 0;

        for (uint32_t i = 0; i + 1 < tileNum; ++i) {
            int32_t eventID = (i & 1) ? EVENT_ID1 : EVENT_ID0;

            LocalTensor<T>& xyLocal = (i & 1) ? xyPong : xyPing;
            LocalTensor<float>& xFloat = (i & 1) ? xFloatPong : xFloatPing;
            LocalTensor<float>& tmp = (i & 1) ? tmpPong : tmpPing;

            RunOneTile(offset, tileLength, eventID, xyLocal, xFloat, tmp);
            offset += tileLength;
        }

        {
            uint32_t i = tileNum - 1;
            int32_t eventID = (i & 1) ? EVENT_ID1 : EVENT_ID0;

            LocalTensor<T>& xyLocal = (i & 1) ? xyPong : xyPing;
            LocalTensor<float>& xFloat = (i & 1) ? xFloatPong : xFloatPing;
            LocalTensor<float>& tmp = (i & 1) ? tmpPong : tmpPing;

            RunOneTile(offset, lastTileLength, eventID, xyLocal, xFloat, tmp);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
};


template <typename T>
class KernelAtanhIntManual {
private:
    GlobalTensor<T> xGm;
    GlobalTensor<float> yGm;

    uint32_t size;
    uint32_t tileLength;
    uint32_t lastTileLength;
    uint32_t tileNum;

    static constexpr uint32_t TILE_LENGTH =
        (std::is_same_v<T, int32_t>) ? 8192 :
        (std::is_same_v<T, int16_t>) ? 10 * 1024 :
        (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) ? 12288 :
                                                                    8192;

    static constexpr uint32_t xAddrPing = 0x00000;

    static constexpr uint32_t halfAddrPing =
        (std::is_same_v<T, int32_t>) ? 0x08100 :
        (std::is_same_v<T, int16_t>) ? 0x05100 :
                                       0x03100;

    static constexpr uint32_t tmpAddrPing =
        (std::is_same_v<T, int32_t>) ? 0x0C200 :
        (std::is_same_v<T, int16_t>) ? 0x0A200 :
                                       0x09200;

    static constexpr uint32_t xAddrPong =
        (std::is_same_v<T, int32_t>) ? 0x14300 :
        (std::is_same_v<T, int16_t>) ? 0x14300 :
                                       0x15300;

    static constexpr uint32_t halfAddrPong =
        (std::is_same_v<T, int32_t>) ? 0x1C400 :
        (std::is_same_v<T, int16_t>) ? 0x19400 :
                                       0x18400;

    static constexpr uint32_t tmpAddrPong =
        (std::is_same_v<T, int32_t>) ? 0x20500 :
        (std::is_same_v<T, int16_t>) ? 0x1E500 :
                                       0x1E500;

public:
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t smallSize,
        uint32_t formerNum,
        uint32_t tilingIncSize,
        uint32_t totalLength
    ) {
        uint32_t srcBeginIndex;
        uint32_t incSize = tilingIncSize;

        if (GetBlockIdx() < formerNum) {
            size = smallSize + incSize;
            srcBeginIndex = size * GetBlockIdx();
        } else {
            size = smallSize;
            srcBeginIndex = size * GetBlockIdx() + formerNum * incSize;
        }

        if (srcBeginIndex + size > totalLength) {
            size = totalLength - srcBeginIndex;
        }

        xGm.SetGlobalBuffer((__gm__ T*)x + srcBeginIndex, size);
        yGm.SetGlobalBuffer((__gm__ float*)y + srcBeginIndex, size);

        tileLength = min(uint32_t(size), uint32_t(TILE_LENGTH));
        tileNum = (size + tileLength - 1) / tileLength;
        lastTileLength = size - (tileNum - 1) * tileLength;
    }

    __aicore__ inline void ComputeBody(
        LocalTensor<T>& xLocal,
        LocalTensor<half>& xHalf,
        LocalTensor<float>& tmpFloat,
        uint32_t calcLength
    ) {
        Cast(xHalf, xLocal, RoundMode::CAST_NONE, calcLength);

        Abs(xHalf, xHalf, calcLength);
        Muls(xHalf, xHalf, half(-1.0f), calcLength);

        Cast(tmpFloat, xHalf, RoundMode::CAST_NONE, calcLength);
        Sqrt(tmpFloat, tmpFloat, calcLength);
    }

    __aicore__ inline void RunOneTile(
        uint32_t offset,
        uint32_t calcLength,
        int32_t eventID,
        LocalTensor<T>& xLocal,
        LocalTensor<half>& xHalf,
        LocalTensor<float>& tmpFloat
    ) {
        WaitFlag<HardEvent::MTE3_MTE2>(eventID);

        DataCopy(xLocal, xGm[offset], calcLength);

        SetFlag<HardEvent::MTE2_V>(eventID);
        WaitFlag<HardEvent::MTE2_V>(eventID);

        ComputeBody(xLocal, xHalf, tmpFloat, calcLength);

        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);

        DataCopy(yGm[offset], tmpFloat, calcLength);

        SetFlag<HardEvent::MTE3_MTE2>(eventID);
    }

    __aicore__ inline void Process() {
        LocalTensor<T> xPing(TPosition::VECCALC, xAddrPing, tileLength);
        LocalTensor<half> xHalfPing(TPosition::VECCALC, halfAddrPing, tileLength);
        LocalTensor<float> tmpFloatPing(TPosition::VECCALC, tmpAddrPing, tileLength);

        LocalTensor<T> xPong(TPosition::VECCALC, xAddrPong, tileLength);
        LocalTensor<half> xHalfPong(TPosition::VECCALC, halfAddrPong, tileLength);
        LocalTensor<float> tmpFloatPong(TPosition::VECCALC, tmpAddrPong, tileLength);

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

        uint32_t offset = 0;

        for (uint32_t i = 0; i + 1 < tileNum; ++i) {
            int32_t eventID = (i & 1) ? EVENT_ID1 : EVENT_ID0;

            LocalTensor<T>& xLocal = (i & 1) ? xPong : xPing;
            LocalTensor<half>& xHalf = (i & 1) ? xHalfPong : xHalfPing;
            LocalTensor<float>& tmpFloat = (i & 1) ? tmpFloatPong : tmpFloatPing;

            RunOneTile(offset, tileLength, eventID, xLocal, xHalf, tmpFloat);
            offset += tileLength;
        }

        {
            uint32_t i = tileNum - 1;
            int32_t eventID = (i & 1) ? EVENT_ID1 : EVENT_ID0;

            LocalTensor<T>& xLocal = (i & 1) ? xPong : xPing;
            LocalTensor<half>& xHalf = (i & 1) ? xHalfPong : xHalfPing;
            LocalTensor<float>& tmpFloat = (i & 1) ? tmpFloatPong : tmpFloatPing;

            RunOneTile(offset, lastTileLength, eventID, xLocal, xHalf, tmpFloat);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
};


extern "C" __global__ __aicore__ void atanh(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling
) {
    GET_TILING_DATA(tiling_data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if constexpr (std::is_same_v<DTYPE_X, float>) {
        KernelAtanhFloatManual<DTYPE_X> op;
        op.Init(
            x,
            y,
            tiling_data.smallSize,
            tiling_data.formerNum,
            tiling_data.incSize,
            tiling_data.totalLength
        );
        op.Process();
    } else if constexpr (std::is_same_v<DTYPE_X, half>) {
        KernelAtanhHalfManual<DTYPE_X> op;
        op.Init(
            x,
            y,
            tiling_data.smallSize,
            tiling_data.formerNum,
            tiling_data.incSize,
            tiling_data.totalLength
        );
        op.Process();
    } else if constexpr (std::is_same_v<DTYPE_X, bfloat16_t>) {
        KernelAtanhBFloatManual<DTYPE_X> op;
        op.Init(
            x,
            y,
            tiling_data.smallSize,
            tiling_data.formerNum,
            tiling_data.incSize,
            tiling_data.totalLength
        );
        op.Process();
    } else {
        KernelAtanhIntManual<DTYPE_X> op;
        op.Init(
            x,
            y,
            tiling_data.smallSize,
            tiling_data.formerNum,
            tiling_data.incSize,
            tiling_data.totalLength
        );
        op.Process();
    }
}