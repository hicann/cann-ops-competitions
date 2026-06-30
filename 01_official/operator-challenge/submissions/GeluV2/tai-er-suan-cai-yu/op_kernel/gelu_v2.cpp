#include "kernel_operator.h"
#include "gelu_piece128d1_coeff.hpp"

namespace {
constexpr uint32_t BLOCK_BYTES = 32u;
constexpr uint32_t TILE_ELEMS = 4096u;
constexpr uint32_t BUFFER_NUM = 2u;
constexpr float GELU_ALPHA = 0.7978845608028654f;
constexpr float GELU_BETA = 0.044715f;
constexpr float GELU_NEG2 = -2.0f;
constexpr float GELU_NEG2_ALPHA = GELU_NEG2 * GELU_ALPHA;
constexpr float GELU_CHEB_T = 4.1f;
constexpr float GELU_CHEB_T2 = GELU_CHEB_T * GELU_CHEB_T;
constexpr float GELU_CHEB_T_NEG = -4.5f;
constexpr float GELU_CHEB_U_SCALE = 0.0987654321f;
constexpr float GELU_CHEB_C0 = 3.9884099750e-01f;
constexpr float GELU_CHEB_C1 = -2.4637479957e-01f;
constexpr float GELU_CHEB_C2 = 9.6889253620e-02f;
constexpr float GELU_CHEB_C3 = -3.6924400311e-02f;
constexpr float GELU_CHEB_C4 = 1.3007321412e-02f;
constexpr float GELU_CHEB_C5 = -4.1842477229e-03f;
constexpr float GELU_CHEB_C6 = 1.2252532267e-03f;
constexpr float GELU_CHEB_C7 = -3.2833139162e-04f;
constexpr float GELU_CHEB_C8 = 7.8715395680e-05f;
constexpr float GELU_CHEB_C9 = -1.7887350609e-05f;
constexpr float GELU_PIECE_W = 0.2626562500f;
constexpr float GELU_PIECE_INV_W = 3.8072575848f;
constexpr float GELU_PIECE_T_SCALE = 7.6145151695f;
constexpr float GELU_PIECE_MAX_INDEX = 63.0f;
constexpr float GELU_PIECE_BYTE_SCALE = 4.0f;
constexpr float GELU_PIECE_C0_0 = 7.5773244117e-01f;
constexpr float GELU_PIECE_C0_1 = 6.8777210660e-01f;
constexpr float GELU_PIECE_C0_2 = 6.2930427707e-01f;
constexpr float GELU_PIECE_C0_3 = 5.8009045152e-01f;
constexpr float GELU_PIECE_C0_4 = 5.3836601057e-01f;
constexpr float GELU_PIECE_C0_5 = 5.0273479506e-01f;
constexpr float GELU_PIECE_C0_6 = 4.7208792351e-01f;
constexpr float GELU_PIECE_C0_7 = 4.4554114274e-01f;
constexpr float GELU_PIECE_C0_8 = 4.2238637280e-01f;
constexpr float GELU_PIECE_C0_9 = 4.0205414238e-01f;
constexpr float GELU_PIECE_C0_10 = 3.8408439553e-01f;
constexpr float GELU_PIECE_C0_11 = 3.6810374561e-01f;
constexpr float GELU_PIECE_C0_12 = 3.5380770496e-01f;
constexpr float GELU_PIECE_C0_13 = 3.4094676265e-01f;
constexpr float GELU_PIECE_C0_14 = 3.2931544507e-01f;
constexpr float GELU_PIECE_C0_15 = 3.1874369391e-01f;
constexpr float GELU_PIECE_C0_16 = 3.0909004914e-01f;
constexpr float GELU_PIECE_C0_17 = 3.0023624120e-01f;
constexpr float GELU_PIECE_C0_18 = 2.9208288661e-01f;
constexpr float GELU_PIECE_C0_19 = 2.8454604966e-01f;
constexpr float GELU_PIECE_C0_20 = 2.7755448614e-01f;
constexpr float GELU_PIECE_C0_21 = 2.7104742541e-01f;
constexpr float GELU_PIECE_C0_22 = 2.6497277911e-01f;
constexpr float GELU_PIECE_C0_23 = 2.5928568859e-01f;
constexpr float GELU_PIECE_C0_24 = 2.5394734256e-01f;
constexpr float GELU_PIECE_C0_25 = 2.4892401075e-01f;
constexpr float GELU_PIECE_C0_26 = 2.4418625099e-01f;
constexpr float GELU_PIECE_C0_27 = 2.3970825580e-01f;
constexpr float GELU_PIECE_C0_28 = 2.3546731201e-01f;
constexpr float GELU_PIECE_C0_29 = 2.3144335170e-01f;
constexpr float GELU_PIECE_C0_30 = 2.2761857787e-01f;
constexpr float GELU_PIECE_C0_31 = 2.2397715080e-01f;
constexpr float GELU_PIECE_C1_0 = -3.8338678807e-02f;
constexpr float GELU_PIECE_C1_1 = -3.1928996669e-02f;
constexpr float GELU_PIECE_C1_2 = -2.6780293649e-02f;
constexpr float GELU_PIECE_C1_3 = -2.2623897501e-02f;
constexpr float GELU_PIECE_C1_4 = -1.9251177188e-02f;
constexpr float GELU_PIECE_C1_5 = -1.6499691459e-02f;
constexpr float GELU_PIECE_C1_6 = -1.4242606551e-02f;
constexpr float GELU_PIECE_C1_7 = -1.2380598110e-02f;
constexpr float GELU_PIECE_C1_8 = -1.0835643454e-02f;
constexpr float GELU_PIECE_C1_9 = -9.5462542649e-03f;
constexpr float GELU_PIECE_C1_10 = -8.4638085080e-03f;
constexpr float GELU_PIECE_C1_11 = -7.5497224866e-03f;
constexpr float GELU_PIECE_C1_12 = -6.7732660525e-03f;
constexpr float GELU_PIECE_C1_13 = -6.1098710035e-03f;
constexpr float GELU_PIECE_C1_14 = -5.5398183404e-03f;
constexpr float GELU_PIECE_C1_15 = -5.0472170955e-03f;
constexpr float GELU_PIECE_C1_16 = -4.6192079907e-03f;
constexpr float GELU_PIECE_C1_17 = -4.2453408107e-03f;
constexpr float GELU_PIECE_C1_18 = -3.9170862793e-03f;
constexpr float GELU_PIECE_C1_19 = -3.6274523084e-03f;
constexpr float GELU_PIECE_C1_20 = -3.3706814234e-03f;
constexpr float GELU_PIECE_C1_21 = -3.1420114740e-03f;
constexpr float GELU_PIECE_C1_22 = -2.9374858055e-03f;
constexpr float GELU_PIECE_C1_23 = -2.7538021834e-03f;
constexpr float GELU_PIECE_C1_24 = -2.5881921647e-03f;
constexpr float GELU_PIECE_C1_25 = -2.4383244531e-03f;
constexpr float GELU_PIECE_C1_26 = -2.3022272034e-03f;
constexpr float GELU_PIECE_C1_27 = -2.1782253377e-03f;
constexpr float GELU_PIECE_C1_28 = -2.0648897934e-03f;
constexpr float GELU_PIECE_C1_29 = -1.9609962810e-03f;
constexpr float GELU_PIECE_C1_30 = -1.8654916473e-03f;
constexpr float GELU_PIECE_C1_31 = -1.7774663394e-03f;
constexpr float GELU_PIECE_C2_0 = 1.7863510820e-03f;
constexpr float GELU_PIECE_C2_1 = 1.4316062500e-03f;
constexpr float GELU_PIECE_C2_2 = 1.1529057949e-03f;
constexpr float GELU_PIECE_C2_3 = 9.3318321234e-04f;
constexpr float GELU_PIECE_C2_4 = 7.5932168932e-04f;
constexpr float GELU_PIECE_C2_5 = 6.2121964581e-04f;
constexpr float GELU_PIECE_C2_6 = 5.1108106595e-04f;
constexpr float GELU_PIECE_C2_7 = 4.2287583774e-04f;
constexpr float GELU_PIECE_C2_8 = 3.5192880970e-04f;
constexpr float GELU_PIECE_C2_9 = 2.9460641301e-04f;
constexpr float GELU_PIECE_C2_10 = 2.4807732485e-04f;
constexpr float GELU_PIECE_C2_11 = 2.1012939057e-04f;
constexpr float GELU_PIECE_C2_12 = 1.7902934877e-04f;
constexpr float GELU_PIECE_C2_13 = 1.5341516611e-04f;
constexpr float GELU_PIECE_C2_14 = 1.3221325139e-04f;
constexpr float GELU_PIECE_C2_15 = 1.1457467929e-04f;
constexpr float GELU_PIECE_C2_16 = 9.9825961462e-05f;
constexpr float GELU_PIECE_C2_17 = 8.7430968295e-05f;
constexpr float GELU_PIECE_C2_18 = 7.6961412207e-05f;
constexpr float GELU_PIECE_C2_19 = 6.8073916085e-05f;
constexpr float GELU_PIECE_C2_20 = 6.0492156162e-05f;
constexpr float GELU_PIECE_C2_21 = 5.3992922648e-05f;
constexpr float GELU_PIECE_C2_22 = 4.8395211151e-05f;
constexpr float GELU_PIECE_C2_23 = 4.3551663617e-05f;
constexpr float GELU_PIECE_C2_24 = 3.9341834593e-05f;
constexpr float GELU_PIECE_C2_25 = 3.5666878750e-05f;
constexpr float GELU_PIECE_C2_26 = 3.2445347619e-05f;
constexpr float GELU_PIECE_C2_27 = 2.9609854100e-05f;
constexpr float GELU_PIECE_C2_28 = 2.7104417543e-05f;
constexpr float GELU_PIECE_C2_29 = 2.4882343955e-05f;
constexpr float GELU_PIECE_C2_30 = 2.2904528084e-05f;
constexpr float GELU_PIECE_C2_31 = 2.1138089010e-05f;
}

__aicore__ inline uint32_t AlignBytes(uint32_t bytes)
{
    return ((bytes + BLOCK_BYTES - 1u) / BLOCK_BYTES) * BLOCK_BYTES;
}

template <typename T>
__aicore__ inline uint32_t AlignElems32B(uint32_t elems)
{
    return AlignBytes(elems * sizeof(T)) / sizeof(T);
}

template <typename T>
__aicore__ inline void CopyGmToUb(AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src, uint32_t validElems)
{
    const uint32_t validBytes = validElems * sizeof(T);
    if (validBytes % BLOCK_BYTES == 0u) {
        AscendC::DataCopy(dst, src, validElems);
        return;
    }
    AscendC::DataCopyExtParams copyParams{1, validBytes, 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = static_cast<uint8_t>(AlignElems32B<T>(validElems) - validElems);
    padParams.paddingValue = static_cast<T>(0);
    AscendC::DataCopyPad(dst, src, copyParams, padParams);
}

template <typename T>
__aicore__ inline void CopyUbToGm(AscendC::GlobalTensor<T> dst, AscendC::LocalTensor<T> src, uint32_t validElems)
{
    event_t evOut = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evOut);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evOut);

    const uint32_t validBytes = validElems * sizeof(T);
    if (validBytes % BLOCK_BYTES == 0u) {
        AscendC::DataCopy(dst, src, validElems);
        return;
    }
    AscendC::DataCopyExtParams copyParams{1, validBytes, 0, 0, 0};
    AscendC::DataCopyPad(dst, src, copyParams);
}

__aicore__ inline void GeluExpSigmoidFloat(AscendC::LocalTensor<float> y, AscendC::LocalTensor<float> x,
                                           AscendC::LocalTensor<float> x3, AscendC::LocalTensor<float> tmp,
                                           uint32_t count)
{
    AscendC::Mul(x3, x, x, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(x3, x3, x, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(x3, x3, GELU_BETA, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(tmp, x, x3, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(tmp, tmp, GELU_NEG2_ALPHA, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(tmp, tmp, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(tmp, tmp, 1.0f, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div(y, x, tmp, count);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void ChebStep(AscendC::LocalTensor<float> t, AscendC::LocalTensor<float> b1,
                                AscendC::LocalTensor<float> b2, AscendC::LocalTensor<float> old,
                                float coeff, uint32_t count)
{
    AscendC::Muls(old, b1, 1.0f, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(b1, b1, t, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(b1, b1, 2.0f, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(b1, b1, b2, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(b1, b1, coeff, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(b2, old, 1.0f, count);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void InitPieceLut(AscendC::LocalTensor<float> c0, AscendC::LocalTensor<float> c1)
{
    for (int32_t i = 0; i < GeluPiece128D1Coeff::kSegments; ++i) {
        c0.SetValue(i, GeluPiece128D1Coeff::kC0[i]);
        c1.SetValue(i, GeluPiece128D1Coeff::kC1[i]);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void GeluChebExactFloat(AscendC::LocalTensor<float> y, AscendC::LocalTensor<float> x,
                                          AscendC::LocalTensor<float> u, AscendC::LocalTensor<float> t,
                                          AscendC::LocalTensor<float> b2, AscendC::LocalTensor<float> old,
                                          AscendC::LocalTensor<uint8_t> mask,
                                          AscendC::LocalTensor<int32_t> index,
                                          AscendC::LocalTensor<float> c0Lut,
                                          AscendC::LocalTensor<float> c1Lut,
                                          uint32_t count)
{
    AscendC::Mul(u, x, x, count);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Muls(t, u, GELU_PIECE_INV_W, count);
    AscendC::Mins(t, t, GELU_PIECE_MAX_INDEX, count);
    AscendC::Cast(index, t, AscendC::RoundMode::CAST_FLOOR, count);
    AscendC::ShiftLeft(index, index, 2, count);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Gather(y, c1Lut, index.ReinterpretCast<uint32_t>(), 0, count);
    AscendC::Gather(b2, c0Lut, index.ReinterpretCast<uint32_t>(), 0, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::FusedMulAdd(y, u, b2, count);
    AscendC::Muls(old, x, 0.5f, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::FusedMulAdd(y, u, old, count);
    AscendC::PipeBarrier<PIPE_V>();

    if ((count & 63u) == 0u) {
        AscendC::CompareScalar(mask, u, GELU_CHEB_T2, AscendC::CMPMODE::GT, count);
    } else {
        AscendC::Duplicate(old, GELU_CHEB_T2, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(mask, u, old, AscendC::CMPMODE::GT, count);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Maxs(old, x, 0.0f, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Select(y, mask, old, y, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
class KernelGeluV2 {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
    {
        (void)workspace;
        GET_TILING_DATA(tilingData, tiling);
        totalSize_ = tilingData.totalSize;
        approximate_ = tilingData.approximate;
        xGm_.SetGlobalBuffer((__gm__ T *)x, totalSize_);
        yGm_.SetGlobalBuffer((__gm__ T *)y, totalSize_);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, AlignElems32B<T>(TILE_ELEMS) * sizeof(T));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, AlignElems32B<T>(TILE_ELEMS) * sizeof(T));
        pipe_.InitBuffer(calcBuf_, 6u * AlignElems32B<float>(TILE_ELEMS) * sizeof(float));
        pipe_.InitBuffer(maskBuf_, AlignBytes(TILE_ELEMS * sizeof(uint8_t)));
        pipe_.InitBuffer(indexBuf_, AlignElems32B<int32_t>(TILE_ELEMS) * sizeof(int32_t));
        pipe_.InitBuffer(c0Buf_, AlignBytes(GeluPiece128D1Coeff::kSegments * sizeof(float)));
        pipe_.InitBuffer(c1Buf_, AlignBytes(GeluPiece128D1Coeff::kSegments * sizeof(float)));
        InitPieceLut(c0Buf_.Get<float>(), c1Buf_.Get<float>());
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint32_t coreCeil = (totalSize_ + blockNum - 1u) / blockNum;
        const uint32_t start = blockIdx * coreCeil;
        if (start >= totalSize_) {
            return;
        }
        uint32_t coreLen = totalSize_ - start;
        if (coreLen > coreCeil) {
            coreLen = coreCeil;
        }

        uint32_t done = 0;
        if (coreLen == 0u) {
            return;
        }
        uint32_t firstValid = coreLen > TILE_ELEMS ? TILE_ELEMS : coreLen;
        CopyIn(start, firstValid);
        while (done < coreLen) {
            const uint32_t validElems = (coreLen - done) > TILE_ELEMS ? TILE_ELEMS : (coreLen - done);
            const uint32_t offset = start + done;
            done += validElems;
            if (done < coreLen) {
                const uint32_t nextValid = (coreLen - done) > TILE_ELEMS ? TILE_ELEMS : (coreLen - done);
                CopyIn(start + done, nextValid);
            }
            Compute(validElems);
            CopyOut(offset, validElems);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t validElems)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();
        CopyGmToUb(xLocal, xGm_[offset], validElems);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t validElems)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY_.AllocTensor<T>();

        AscendC::LocalTensor<float> xFloat = calcBuf_.Get<float>();
        AscendC::LocalTensor<float> yFloat = calcBuf_.Get<float>()[AlignElems32B<float>(TILE_ELEMS)];
        AscendC::LocalTensor<float> x3 = calcBuf_.Get<float>()[2u * AlignElems32B<float>(TILE_ELEMS)];
        AscendC::LocalTensor<float> tmp = calcBuf_.Get<float>()[3u * AlignElems32B<float>(TILE_ELEMS)];
        AscendC::LocalTensor<float> tmp2 = calcBuf_.Get<float>()[4u * AlignElems32B<float>(TILE_ELEMS)];
        AscendC::LocalTensor<float> old = calcBuf_.Get<float>()[5u * AlignElems32B<float>(TILE_ELEMS)];
        AscendC::LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
        AscendC::LocalTensor<int32_t> index = indexBuf_.Get<int32_t>();

        AscendC::Cast(xFloat, xLocal, AscendC::RoundMode::CAST_NONE, validElems);
        AscendC::PipeBarrier<PIPE_V>();
        if (approximate_ == 0u) {
            GeluChebExactFloat(yFloat, xFloat, x3, tmp, tmp2, old, mask, index,
                               c0Buf_.Get<float>(), c1Buf_.Get<float>(), validElems);
        } else {
            GeluExpSigmoidFloat(yFloat, xFloat, x3, tmp, validElems);
        }
        AscendC::Cast(yLocal, yFloat, AscendC::RoundMode::CAST_RINT, validElems);
        AscendC::PipeBarrier<PIPE_V>();

        outQueueY_.EnQue<T>(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t validElems)
    {
        AscendC::LocalTensor<T> yLocal = outQueueY_.DeQue<T>();
        CopyUbToGm(yGm_[offset], yLocal, validElems);
        outQueueY_.FreeTensor(yLocal);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> c0Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> c1Buf_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> yGm_;
    uint32_t totalSize_ = 0u;
    uint32_t approximate_ = 0u;
};

class KernelGeluV2Float {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
    {
        (void)workspace;
        GET_TILING_DATA(tilingData, tiling);
        totalSize_ = tilingData.totalSize;
        approximate_ = tilingData.approximate;
        xGm_.SetGlobalBuffer((__gm__ float *)x, totalSize_);
        yGm_.SetGlobalBuffer((__gm__ float *)y, totalSize_);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, AlignElems32B<float>(TILE_ELEMS) * sizeof(float));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, AlignElems32B<float>(TILE_ELEMS) * sizeof(float));
        pipe_.InitBuffer(calcBuf_, 4u * AlignElems32B<float>(TILE_ELEMS) * sizeof(float));
        pipe_.InitBuffer(maskBuf_, AlignBytes(TILE_ELEMS * sizeof(uint8_t)));
        pipe_.InitBuffer(indexBuf_, AlignElems32B<int32_t>(TILE_ELEMS) * sizeof(int32_t));
        pipe_.InitBuffer(c0Buf_, AlignBytes(GeluPiece128D1Coeff::kSegments * sizeof(float)));
        pipe_.InitBuffer(c1Buf_, AlignBytes(GeluPiece128D1Coeff::kSegments * sizeof(float)));
        InitPieceLut(c0Buf_.Get<float>(), c1Buf_.Get<float>());
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint32_t coreCeil = (totalSize_ + blockNum - 1u) / blockNum;
        const uint32_t start = blockIdx * coreCeil;
        if (start >= totalSize_) {
            return;
        }
        uint32_t coreLen = totalSize_ - start;
        if (coreLen > coreCeil) {
            coreLen = coreCeil;
        }

        uint32_t done = 0;
        if (coreLen == 0u) {
            return;
        }
        uint32_t firstValid = coreLen > TILE_ELEMS ? TILE_ELEMS : coreLen;
        CopyIn(start, firstValid);
        while (done < coreLen) {
            const uint32_t validElems = (coreLen - done) > TILE_ELEMS ? TILE_ELEMS : (coreLen - done);
            const uint32_t offset = start + done;
            done += validElems;
            if (done < coreLen) {
                const uint32_t nextValid = (coreLen - done) > TILE_ELEMS ? TILE_ELEMS : (coreLen - done);
                CopyIn(start + done, nextValid);
            }
            Compute(validElems);
            CopyOut(offset, validElems);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t validElems)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX_.AllocTensor<float>();
        CopyGmToUb(xLocal, xGm_[offset], validElems);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t validElems)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX_.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY_.AllocTensor<float>();

        AscendC::LocalTensor<float> x3 = calcBuf_.Get<float>();
        AscendC::LocalTensor<float> tmp = calcBuf_.Get<float>()[AlignElems32B<float>(TILE_ELEMS)];
        AscendC::LocalTensor<float> tmp2 = calcBuf_.Get<float>()[2u * AlignElems32B<float>(TILE_ELEMS)];
        AscendC::LocalTensor<float> old = calcBuf_.Get<float>()[3u * AlignElems32B<float>(TILE_ELEMS)];
        AscendC::LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
        AscendC::LocalTensor<int32_t> index = indexBuf_.Get<int32_t>();
        if (approximate_ == 0u) {
            GeluChebExactFloat(yLocal, xLocal, x3, tmp, tmp2, old, mask, index,
                               c0Buf_.Get<float>(), c1Buf_.Get<float>(), validElems);
        } else {
            GeluExpSigmoidFloat(yLocal, xLocal, x3, tmp, validElems);
        }

        outQueueY_.EnQue<float>(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t validElems)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY_.DeQue<float>();
        CopyUbToGm(yGm_[offset], yLocal, validElems);
        outQueueY_.FreeTensor(yLocal);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> c0Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> c1Buf_;
    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> yGm_;
    uint32_t totalSize_ = 0u;
    uint32_t approximate_ = 0u;
};

extern "C" __global__ __aicore__ void gelu_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
#if (ORIG_DTYPE_X == DT_FLOAT)
    KernelGeluV2Float op;
#else
    KernelGeluV2<DTYPE_X> op;
#endif
    op.Init(x, y, workspace, tiling);
    op.Process();
}
