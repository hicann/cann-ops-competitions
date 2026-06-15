// Kernel侧核函数实现 -- 自实现 Erf 多项式
//
// 该版本使用 TBuf 直接管理单 tile 输入/输出/临时区，并显式补齐
// MTE2/VEC/MTE3 事件同步，避免无队列版本读写未完成的数据。

#include "kernel_operator.h"
#include "lib/math/erf.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

constexpr uint32_t ELEM_PER_BLOCK = 8;  // float32: 32B / 4B = 8
constexpr uint32_t TILE_MODE_RATIONAL = 0x80000000U;
constexpr uint32_t TILE_MODE_FAST_TBUF = 0x40000000U;
constexpr uint32_t TILE_LEN_MASK = 0x3fffffffU;

constexpr float ERF_BOUNDARY = 2.45f;
constexpr float ERF_C0 = 1.1265501402e+00f;
constexpr float ERF_C1 = -3.6674583137e-01f;
constexpr float ERF_C2 = 9.8786676579e-02f;
constexpr float ERF_C3 = -1.7358669461e-02f;
constexpr float ERF_C4 = 1.7329661201e-03f;
constexpr float ERF_C5 = -7.3396627542e-05f;

constexpr float ERF9_BOUNDARY = 2.25f;
constexpr float ERF9_C0 = 1.1241971832e+00f;
constexpr float ERF9_C1 = -3.5714610510e-01f;
constexpr float ERF9_C2 = 8.7919688171e-02f;
constexpr float ERF9_C3 = -1.2357589910e-02f;
constexpr float ERF9_C4 = 7.2783427561e-04f;

constexpr float ERF_R_BOUNDARY = 3.92f;
constexpr float ERF_P0 = 0.29639384698e5f;
constexpr float ERF_P1 = 0.50637915060e4f;
constexpr float ERF_P2 = 0.13938061484e4f;
constexpr float ERF_P3 = 0.10162808918e3f;
constexpr float ERF_P4 = 0.75517016694e1f;
constexpr float ERF_P5 = 0.053443748819f;
constexpr float ERF_Q0 = 0.26267224157e5f;
constexpr float ERF_Q1 = 0.13243365831e5f;
constexpr float ERF_Q2 = 0.30231248150e4f;
constexpr float ERF_Q3 = 0.39856963806e3f;
constexpr float ERF_Q4 = 0.31212858877e2f;

template <class DT_X>
class KernelErfLen1Scalar {
public:
    __aicore__ inline KernelErfLen1Scalar() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, 1);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, 1);
    }

    __aicore__ inline void Process() {
        float x0 = xGm_.GetValue(0);
        if (x0 > ERF_BOUNDARY) {
            x0 = ERF_BOUNDARY;
        } else if (x0 < -ERF_BOUNDARY) {
            x0 = -ERF_BOUNDARY;
        }

        float z = x0 * x0;
        float y0 = ERF_C5 * z + ERF_C4;
        y0 = y0 * z + ERF_C3;
        y0 = y0 * z + ERF_C2;
        y0 = y0 * z + ERF_C1;
        y0 = y0 * z + ERF_C0;
        yGm_.SetValue(0, y0 * x0);
    }

private:
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
};

template <class DT_X>
class KernelErfLen128Builtin {
public:
    __aicore__ inline KernelErfLen128Builtin() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, 128);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, 128);
        pipe_.InitBuffer(xBuf_, 128 * sizeof(DT_X));
        pipe_.InitBuffer(yBuf_, 128 * sizeof(DT_X));
        pipe_.InitBuffer(tmpBuf_, 128 * sizeof(DT_X) * 3);
    }

    __aicore__ inline void Process() {
        LocalTensor<DT_X> xLocal = xBuf_.Get<DT_X>(128);
        LocalTensor<DT_X> yLocal = yBuf_.Get<DT_X>(128);
        LocalTensor<uint8_t> tmpLocal = tmpBuf_.Get<uint8_t>(128 * sizeof(DT_X) * 3);

        DataCopy(xLocal, xGm_[0], 128);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

        Erf<DT_X, false>(yLocal, xLocal, tmpLocal, 128);

        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopy(yGm_[0], yLocal, 128);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECIN>   xBuf_;
    TBuf<TPosition::VECOUT>  yBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
};

template <class DT_X, bool staticMode = false, bool staticRational = false, bool staticFastTbuf = false>
class KernelErfTBuf {
public:
    __aicore__ inline KernelErfTBuf() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t coreLen, uint32_t tileLen,
                                uint32_t tileNum, uint32_t lastTileLen,
                                uint32_t coreOffset, bool tailUnaligned,
                                bool useRational, bool useFastTbuf) {
        tileLen_       = tileLen;
        tileNum_       = tileNum;
        lastTileLen_   = lastTileLen;
        tailUnaligned_ = tailUnaligned;
        useRational_   = useRational;
        useFastTbuf_   = useFastTbuf;

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x + coreOffset, coreLen);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y + coreOffset, coreLen);

        pipe_.InitBuffer(xBuf_, tileLen_ * sizeof(DT_X));
        pipe_.InitBuffer(yBuf_, tileLen_ * sizeof(DT_X));
        pipe_.InitBuffer(tmpBuf_, tileLen_ * sizeof(float) * (useRational ? 3 : 2));
    }

    __aicore__ inline void Process() {
        if (tileNum_ > 0) {
            if constexpr (!staticMode) {
                if (!useFastTbuf_) {
                    WaitVBeforeMte2();
                }
            } else if constexpr (!staticFastTbuf) {
                WaitVBeforeMte2();
            }
            RunTile(0, lastTileLen_, tailUnaligned_);
            if constexpr (!staticMode) {
                if (!useFastTbuf_) {
                    WaitMte3BeforeV();
                }
            } else if constexpr (!staticFastTbuf) {
                WaitMte3BeforeV();
            }
        }
    }

private:
    __aicore__ inline uint32_t AlignLen(uint32_t length) const {
        return (length + ELEM_PER_BLOCK - 1) / ELEM_PER_BLOCK * ELEM_PER_BLOCK;
    }

    __aicore__ inline void WaitMte2BeforeV() {
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    }

    __aicore__ inline void WaitVBeforeMte2() {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }

    __aicore__ inline void WaitVBeforeMte3() {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
    }

    __aicore__ inline void WaitMte3BeforeV() {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID3);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
    }

    __aicore__ inline void CopyIn(LocalTensor<DT_X> xLocal, uint32_t progress, uint32_t length, bool usePad) {
        if (!usePad) {
            DataCopy(xLocal, xGm_[progress * tileLen_], length);
        } else {
            DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen   = length * sizeof(DT_X);
            params.srcStride  = 0;
            params.dstStride  = 0;
            params.rsv        = 0;
            DataCopyPadExtParams<DT_X> padParams;
            padParams.isPad        = false;
            padParams.leftPadding  = 0;
            padParams.rightPadding = 0;
            padParams.paddingValue = (DT_X)0;
            DataCopyPad(xLocal, xGm_[progress * tileLen_], params, padParams);
        }
        WaitMte2BeforeV();
    }

    __aicore__ inline void ComputePoly(LocalTensor<DT_X> xLocal, LocalTensor<DT_X> yLocal, uint32_t length) {
        uint32_t alignedLen = AlignLen(length);

        LocalTensor<float> tmpAll = tmpBuf_.Get<float>();
        LocalTensor<float> tmpA = tmpAll;
        LocalTensor<float> tmpB = tmpAll[tileLen_];

        UnaryRepeatParams un;
        BinaryRepeatParams bin;

        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, alignedLen);

        Mins<float, false>(tmpA, xLocal, ERF_BOUNDARY, MASK_PLACEHOLDER, 1, un);
        Maxs<float, false>(tmpA, tmpA, -ERF_BOUNDARY, MASK_PLACEHOLDER, 1, un);

        Mul<float, false>(tmpB, tmpA, tmpA, MASK_PLACEHOLDER, 1, bin);

        Muls<float, false>(yLocal, tmpB, ERF_C5, MASK_PLACEHOLDER, 1, un);
        Adds<float, false>(yLocal, yLocal, ERF_C4, MASK_PLACEHOLDER, 1, un);
        Mul<float, false>(yLocal, yLocal, tmpB, MASK_PLACEHOLDER, 1, bin);
        Adds<float, false>(yLocal, yLocal, ERF_C3, MASK_PLACEHOLDER, 1, un);
        Mul<float, false>(yLocal, yLocal, tmpB, MASK_PLACEHOLDER, 1, bin);
        Adds<float, false>(yLocal, yLocal, ERF_C2, MASK_PLACEHOLDER, 1, un);
        Mul<float, false>(yLocal, yLocal, tmpB, MASK_PLACEHOLDER, 1, bin);
        Adds<float, false>(yLocal, yLocal, ERF_C1, MASK_PLACEHOLDER, 1, un);
        Mul<float, false>(yLocal, yLocal, tmpB, MASK_PLACEHOLDER, 1, bin);
        Adds<float, false>(yLocal, yLocal, ERF_C0, MASK_PLACEHOLDER, 1, un);
        Mul<float, false>(yLocal, yLocal, tmpA, MASK_PLACEHOLDER, 1, bin);

        SetMaskNorm();
        ResetMask();
    }

    __aicore__ inline void ComputeRational(LocalTensor<DT_X> xLocal, LocalTensor<DT_X> yLocal, uint32_t length) {
        uint32_t alignedLen = AlignLen(length);

        LocalTensor<float> tmpAll = tmpBuf_.Get<float>();
        LocalTensor<float> tmpA = tmpAll;
        LocalTensor<float> tmpB = tmpAll[tileLen_];
        LocalTensor<float> tmpC = tmpAll[tileLen_ * 2];

        UnaryRepeatParams un;
        BinaryRepeatParams bin;

        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, alignedLen);

        Mins<float, false>(tmpA, xLocal, ERF_R_BOUNDARY, MASK_PLACEHOLDER, 1, un);
        Maxs<float, false>(tmpA, tmpA, -ERF_R_BOUNDARY, MASK_PLACEHOLDER, 1, un);

        Mul<float, false>(tmpB, tmpA, tmpA, MASK_PLACEHOLDER, 1, bin);

        Muls<float, false>(tmpC, tmpB, ERF_P5, MASK_PLACEHOLDER, 1, un);
        Adds<float, false>(tmpC, tmpC, ERF_P4, MASK_PLACEHOLDER, 1, un);
        Mul<float, false>(tmpC, tmpC, tmpB, MASK_PLACEHOLDER, 1, bin);
        Adds<float, false>(tmpC, tmpC, ERF_P3, MASK_PLACEHOLDER, 1, un);
        Mul<float, false>(tmpC, tmpC, tmpB, MASK_PLACEHOLDER, 1, bin);
        Adds<float, false>(tmpC, tmpC, ERF_P2, MASK_PLACEHOLDER, 1, un);
        Mul<float, false>(tmpC, tmpC, tmpB, MASK_PLACEHOLDER, 1, bin);
        Adds<float, false>(tmpC, tmpC, ERF_P1, MASK_PLACEHOLDER, 1, un);
        Mul<float, false>(tmpC, tmpC, tmpB, MASK_PLACEHOLDER, 1, bin);
        Adds<float, false>(tmpC, tmpC, ERF_P0, MASK_PLACEHOLDER, 1, un);
        Mul<float, false>(tmpC, tmpC, tmpA, MASK_PLACEHOLDER, 1, bin);

        Adds<float, false>(tmpA, tmpB, ERF_Q4, MASK_PLACEHOLDER, 1, un);
        Mul<float, false>(tmpA, tmpA, tmpB, MASK_PLACEHOLDER, 1, bin);
        Adds<float, false>(tmpA, tmpA, ERF_Q3, MASK_PLACEHOLDER, 1, un);
        Mul<float, false>(tmpA, tmpA, tmpB, MASK_PLACEHOLDER, 1, bin);
        Adds<float, false>(tmpA, tmpA, ERF_Q2, MASK_PLACEHOLDER, 1, un);
        Mul<float, false>(tmpA, tmpA, tmpB, MASK_PLACEHOLDER, 1, bin);
        Adds<float, false>(tmpA, tmpA, ERF_Q1, MASK_PLACEHOLDER, 1, un);
        Mul<float, false>(tmpA, tmpA, tmpB, MASK_PLACEHOLDER, 1, bin);
        Adds<float, false>(tmpA, tmpA, ERF_Q0, MASK_PLACEHOLDER, 1, un);

        Div<float, false>(yLocal, tmpC, tmpA, MASK_PLACEHOLDER, 1, bin);

        SetMaskNorm();
        ResetMask();
    }

    __aicore__ inline void CopyOut(LocalTensor<DT_X> yLocal, uint32_t progress, uint32_t length, bool usePad) {
        WaitVBeforeMte3();
        if (!usePad) {
            DataCopy(yGm_[progress * tileLen_], yLocal, length);
        } else {
            DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen   = length * sizeof(DT_X);
            params.srcStride  = 0;
            params.dstStride  = 0;
            params.rsv        = 0;
            DataCopyPad(yGm_[progress * tileLen_], yLocal, params);
        }
    }

    __aicore__ inline void RunTile(uint32_t progress, uint32_t length, bool usePad) {
        uint32_t alignedLen = AlignLen(length);
        LocalTensor<DT_X> xLocal = xBuf_.Get<DT_X>(alignedLen);
        LocalTensor<DT_X> yLocal = yBuf_.Get<DT_X>(alignedLen);
        CopyIn(xLocal, progress, length, usePad);
        if constexpr (staticMode) {
            if constexpr (staticRational) {
                ComputeRational(xLocal, yLocal, length);
            } else {
                ComputePoly(xLocal, yLocal, length);
            }
        } else {
            if (useRational_) {
                ComputeRational(xLocal, yLocal, length);
            } else {
                ComputePoly(xLocal, yLocal, length);
            }
        }
        CopyOut(yLocal, progress, length, usePad);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECIN>   xBuf_;
    TBuf<TPosition::VECOUT>  yBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;

    uint32_t tileLen_;
    uint32_t tileNum_;
    uint32_t lastTileLen_;
    bool     tailUnaligned_;
    bool     useRational_;
    bool     useFastTbuf_;
};

constexpr int32_t BUFFER_NUM = 2;

template <class DT_X>
class KernelErfTQue {
public:
    __aicore__ inline KernelErfTQue() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t coreLen, uint32_t tileLen,
                                uint32_t tileNum, uint32_t lastTileLen,
                                uint32_t coreOffset, bool tailUnaligned) {
        tileLen_       = tileLen;
        tileNum_       = tileNum;
        lastTileLen_   = lastTileLen;
        tailUnaligned_ = tailUnaligned;

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x + coreOffset, coreLen);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y + coreOffset, coreLen);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLen_ * sizeof(DT_X));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tileLen_ * sizeof(DT_X));
        pipe_.InitBuffer(tmpBuf_, tileLen_ * sizeof(float) * 2);
    }

    __aicore__ inline void Process() {
        if (tileNum_ == 0) {
            return;
        }

        if (tailUnaligned_ && tileNum_ == 1) {
            CopyInPad(0, lastTileLen_);
        } else {
            CopyInAligned(0, tileLen_);
        }

        for (uint32_t i = 0; i < tileNum_; i++) {
            if (i + 1 < tileNum_) {
                uint32_t nextLen = (i + 1 == tileNum_ - 1) ? lastTileLen_ : tileLen_;
                if (tailUnaligned_ && (i + 1 == tileNum_ - 1)) {
                    CopyInPad(i + 1, nextLen);
                } else {
                    CopyInAligned(i + 1, nextLen);
                }
            }

            uint32_t curLen = (i == tileNum_ - 1) ? lastTileLen_ : tileLen_;
            Compute(curLen);
            if (tailUnaligned_ && (i == tileNum_ - 1)) {
                CopyOutPad(i, curLen);
            } else {
                CopyOutAligned(i, curLen);
            }
        }
    }

private:
    __aicore__ inline void CopyInAligned(uint32_t progress, uint32_t length) {
        LocalTensor<DT_X> xLocal = inQueueX_.AllocTensor<DT_X>();
        DataCopy(xLocal, xGm_[progress * tileLen_], length);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInPad(uint32_t progress, uint32_t length) {
        LocalTensor<DT_X> xLocal = inQueueX_.AllocTensor<DT_X>();
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen   = length * sizeof(DT_X);
        params.srcStride  = 0;
        params.dstStride  = 0;
        params.rsv        = 0;
        DataCopyPadExtParams<DT_X> padParams;
        padParams.isPad        = false;
        padParams.leftPadding  = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = (DT_X)0;
        DataCopyPad(xLocal, xGm_[progress * tileLen_], params, padParams);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t length) {
        LocalTensor<DT_X> xLocal = inQueueX_.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueueY_.AllocTensor<DT_X>();

        uint32_t alignedLen = (length + ELEM_PER_BLOCK - 1) / ELEM_PER_BLOCK * ELEM_PER_BLOCK;

        LocalTensor<float> tmpAll = tmpBuf_.Get<float>();
        LocalTensor<float> tmpA = tmpAll;
        LocalTensor<float> tmpB = tmpAll[tileLen_];

        UnaryRepeatParams un;
        BinaryRepeatParams bin;

        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, alignedLen);

        Mins<float, false>(tmpA, xLocal, ERF9_BOUNDARY,  MASK_PLACEHOLDER, 1, un);
        Maxs<float, false>(tmpA, tmpA,   -ERF9_BOUNDARY, MASK_PLACEHOLDER, 1, un);

        Mul<float, false>(tmpB, tmpA, tmpA, MASK_PLACEHOLDER, 1, bin);

        Muls<float, false>(yLocal, tmpB, ERF9_C4, MASK_PLACEHOLDER, 1, un);
        Adds<float, false>(yLocal, yLocal, ERF9_C3, MASK_PLACEHOLDER, 1, un);
        Mul<float, false> (yLocal, yLocal, tmpB, MASK_PLACEHOLDER, 1, bin);
        Adds<float, false>(yLocal, yLocal, ERF9_C2, MASK_PLACEHOLDER, 1, un);
        Mul<float, false> (yLocal, yLocal, tmpB, MASK_PLACEHOLDER, 1, bin);
        Adds<float, false>(yLocal, yLocal, ERF9_C1, MASK_PLACEHOLDER, 1, un);
        Mul<float, false> (yLocal, yLocal, tmpB, MASK_PLACEHOLDER, 1, bin);
        Adds<float, false>(yLocal, yLocal, ERF9_C0, MASK_PLACEHOLDER, 1, un);
        Mul<float, false> (yLocal, yLocal, tmpA, MASK_PLACEHOLDER, 1, bin);

        SetMaskNorm();
        ResetMask();

        outQueueY_.EnQue<DT_X>(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOutAligned(uint32_t progress, uint32_t length) {
        LocalTensor<DT_X> yLocal = outQueueY_.DeQue<DT_X>();
        DataCopy(yGm_[progress * tileLen_], yLocal, length);
        outQueueY_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutPad(uint32_t progress, uint32_t length) {
        LocalTensor<DT_X> yLocal = outQueueY_.DeQue<DT_X>();
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen   = length * sizeof(DT_X);
        params.srcStride  = 0;
        params.dstStride  = 0;
        params.rsv        = 0;
        DataCopyPad(yGm_[progress * tileLen_], yLocal, params);
        outQueueY_.FreeTensor(yLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM>  inQueueX_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    TBuf<TPosition::VECCALC>              tmpBuf_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;

    uint32_t tileLen_;
    uint32_t tileNum_;
    uint32_t lastTileLen_;
    bool     tailUnaligned_;
};

template <typename DT_X, uint64_t MODE>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);

    uint32_t coreId   = GetBlockIdx();
    uint32_t totalLen = tiling_data.totalLength;
    if constexpr (MODE == ERF_TPL_MODE_LEN1_SCALAR) {
        if (coreId == 0) {
            KernelErfLen1Scalar<DT_X> op;
            op.Init(x, y);
            op.Process();
        }
        return;
    }

    uint32_t coreLen  = tiling_data.coreLength;
    uint32_t tileInfo = tiling_data.tileLen;
    uint32_t tileLen  = tileInfo & TILE_LEN_MASK;
    bool useRational  = (tileInfo & TILE_MODE_RATIONAL) != 0;
    bool useFastTbuf  = (tileInfo & TILE_MODE_FAST_TBUF) != 0;

    uint32_t startOffset = coreId * coreLen;
    if (startOffset >= totalLen) return;

    uint32_t endOffset = startOffset + coreLen;
    if (endOffset > totalLen) endOffset = totalLen;
    uint32_t myLen = endOffset - startOffset;

    bool isLastUsedCore = (endOffset == totalLen);
    bool tailUnaligned  = isLastUsedCore && (myLen % ELEM_PER_BLOCK != 0);

    uint32_t tileNum     = (myLen + tileLen - 1) / tileLen;
    uint32_t lastTileLen = myLen - (tileNum - 1) * tileLen;

    if constexpr (MODE == ERF_TPL_MODE_CASE_FAST_POLY) {
        KernelErfTBuf<DT_X, true, false, true> op;
        op.Init(x, y, myLen, tileLen, tileNum, lastTileLen, startOffset, tailUnaligned, false, true);
        op.Process();
    } else if constexpr (MODE == ERF_TPL_MODE_LEN128_RATIONAL) {
        if (coreId != 0) {
            return;
        }
        KernelErfLen128Builtin<DT_X> op;
        op.Init(x, y);
        op.Process();
    } else if (totalLen <= 8192) {
        KernelErfTBuf<DT_X> op;
        op.Init(x, y, myLen, tileLen, tileNum, lastTileLen, startOffset, tailUnaligned, useRational, useFastTbuf);
        op.Process();
    } else {
        KernelErfTQue<DT_X> op;
        op.Init(x, y, myLen, tileLen, tileNum, lastTileLen, startOffset, tailUnaligned);
        op.Process();
    }
}
