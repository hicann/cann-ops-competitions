#include "kernel_operator.h"

namespace {
constexpr uint32_t TILE_ELEMS = 4096u;
constexpr uint32_t INT32_TILE_ELEMS = 5120u;
#if (ORIG_DTYPE_X1 == DT_FLOAT)
constexpr uint32_t FP_TILE_ELEMS = 5120u;
#else
constexpr uint32_t FP_TILE_ELEMS = TILE_ELEMS;
#endif
constexpr uint32_t BLOCK_BYTES = 32u;
constexpr uint32_t NARROW_TO_HALF_OUTPUTS_PER_REPEAT = 128u;
}

#define KEY_SCALAR 1
#define KEY_BYTE_CONTIG 2
#define KEY_BYTE_ROW 3
#define KEY_INT16_CONTIG 4
#define KEY_INT16_ROW 5
#define KEY_INT32_ROW 6
#define KEY_FP_CONTIG 7
#define KEY_FP_ROW 8
#define KEY_INT32_CONTIG 9

__aicore__ inline uint32_t CeilDiv(uint32_t n, uint32_t d)
{
    return d == 0u ? 0u : (n + d - 1u) / d;
}

__aicore__ inline bool ShouldUseBroadcastRowVector(const TensorEqualTilingData &tilingData)
{
    uint32_t rowLen = tilingData.outShape3;
    uint32_t outRows = tilingData.outShape0 * tilingData.outShape1 * tilingData.outShape2;
    uint32_t x1InnerStride = tilingData.x1Stride3;
    uint32_t x2InnerStride = tilingData.x2Stride3;
    if (tilingData.rank == 1u) {
        rowLen = tilingData.outShape0;
        outRows = 1u;
        x1InnerStride = tilingData.x1Stride0;
        x2InnerStride = tilingData.x2Stride0;
    } else if (tilingData.rank == 2u) {
        rowLen = tilingData.outShape1;
        outRows = tilingData.outShape0;
        x1InnerStride = tilingData.x1Stride1;
        x2InnerStride = tilingData.x2Stride1;
    } else if (tilingData.rank == 3u) {
        rowLen = tilingData.outShape2;
        outRows = tilingData.outShape0 * tilingData.outShape1;
        x1InnerStride = tilingData.x1Stride2;
        x2InnerStride = tilingData.x2Stride2;
    }
    const bool hasBroadcast = tilingData.x1Stride0 != tilingData.x2Stride0 ||
                              tilingData.x1Stride1 != tilingData.x2Stride1 ||
                              tilingData.x1Stride2 != tilingData.x2Stride2 ||
                              tilingData.x1Stride3 != tilingData.x2Stride3;
    return rowLen >= 1u && (x1InnerStride == 0u || x1InnerStride == 1u) &&
           (x2InnerStride == 0u || x2InnerStride == 1u) && hasBroadcast && outRows > 0u;
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
    padParams.rightPadding = static_cast<uint8_t>((((validBytes + BLOCK_BYTES - 1u) / BLOCK_BYTES) * BLOCK_BYTES - validBytes) / sizeof(T));
    padParams.paddingValue = 0;
    AscendC::DataCopyPad(dst, src, copyParams, padParams);
}

__aicore__ inline void CopyUbToGmU8(AscendC::GlobalTensor<uint8_t> dst, AscendC::LocalTensor<uint8_t> src,
                                    uint32_t validElems)
{
    if (validElems % BLOCK_BYTES == 0u) {
        AscendC::DataCopy(dst, src, validElems);
        return;
    }
    AscendC::DataCopyExtParams copyParams{1, validElems, 0, 0, 0};
    AscendC::DataCopyPad(dst, src, copyParams);
}

template <uint32_t ELEM_BYTES>
__aicore__ inline void ProcessScalarFixed(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    const uint32_t outSize = tilingData.outSize;
    const uint32_t shape123 = tilingData.outShape1 * tilingData.outShape2 * tilingData.outShape3;
    const uint32_t shape23 = tilingData.outShape2 * tilingData.outShape3;

    AscendC::GlobalTensor<uint8_t> x1Gm;
    AscendC::GlobalTensor<uint8_t> x2Gm;
    AscendC::GlobalTensor<uint8_t> yGm;
    x1Gm.SetGlobalBuffer((__gm__ uint8_t *)x1, outSize * ELEM_BYTES);
    x2Gm.SetGlobalBuffer((__gm__ uint8_t *)x2, outSize * ELEM_BYTES);
    yGm.SetGlobalBuffer((__gm__ uint8_t *)y, outSize);

    const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
    for (uint32_t i = blockIdx; i < outSize; i += blockNum) {
        uint32_t rem = i;
        const uint32_t idx0 = rem / shape123;
        rem %= shape123;
        const uint32_t idx1 = rem / shape23;
        rem %= shape23;
        const uint32_t idx2 = rem / tilingData.outShape3;
        const uint32_t idx3 = rem % tilingData.outShape3;
        const uint32_t x1Byte =
            (idx0 * tilingData.x1Stride0 + idx1 * tilingData.x1Stride1 + idx2 * tilingData.x1Stride2 + idx3 * tilingData.x1Stride3) * ELEM_BYTES;
        const uint32_t x2Byte =
            (idx0 * tilingData.x2Stride0 + idx1 * tilingData.x2Stride1 + idx2 * tilingData.x2Stride2 + idx3 * tilingData.x2Stride3) * ELEM_BYTES;

        bool equal = x1Gm.GetValue(x1Byte) == x2Gm.GetValue(x2Byte);
        if constexpr (ELEM_BYTES >= 2u) {
            equal = equal && x1Gm.GetValue(x1Byte + 1u) == x2Gm.GetValue(x2Byte + 1u);
        }
        if constexpr (ELEM_BYTES >= 4u) {
            equal = equal && x1Gm.GetValue(x1Byte + 2u) == x2Gm.GetValue(x2Byte + 2u) &&
                    x1Gm.GetValue(x1Byte + 3u) == x2Gm.GetValue(x2Byte + 3u);
        }
        yGm.SetValue(i, equal ? 1u : 0u);
    }
}

#if (ORIG_DTYPE_X1 == DT_UINT8 || ORIG_DTYPE_X1 == DT_INT8 || ORIG_DTYPE_X1 == DT_BOOL)
class KernelTensorEqualByteVector {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingData, tiling);
        outSize_ = tilingData.outSize;
        const bool contiguous = tilingData.x1Stride0 == tilingData.x2Stride0 &&
                                tilingData.x1Stride1 == tilingData.x2Stride1 &&
                                tilingData.x1Stride2 == tilingData.x2Stride2 &&
                                tilingData.x1Stride3 == tilingData.x2Stride3;
        useVector_ = contiguous;
        x1Gm_.SetGlobalBuffer((__gm__ uint8_t *)x1, outSize_);
        x2Gm_.SetGlobalBuffer((__gm__ uint8_t *)x2, outSize_);
        yGm_.SetGlobalBuffer((__gm__ uint8_t *)y, outSize_);
        pipe_.InitBuffer(inX1_, 1, TILE_ELEMS * sizeof(uint8_t));
        pipe_.InitBuffer(inX2_, 1, TILE_ELEMS * sizeof(uint8_t));
        pipe_.InitBuffer(outY_, 1, TILE_ELEMS * sizeof(uint8_t));
        pipe_.InitBuffer(halfBuf_, 4u * TILE_ELEMS * sizeof(half));
        pipe_.InitBuffer(maskBuf_, TILE_ELEMS * sizeof(uint8_t));
    }

    __aicore__ inline bool UseVector() const
    {
        return useVector_;
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint32_t coreCeil = CeilDiv(outSize_, blockNum);
        const uint32_t start = blockIdx * coreCeil;
        if (start >= outSize_) {
            return;
        }
        uint32_t coreLen = outSize_ - start;
        if (coreLen > coreCeil) {
            coreLen = coreCeil;
        }

        uint32_t done = 0;
        while (done < coreLen) {
            const uint32_t valid = (coreLen - done) > TILE_ELEMS ? TILE_ELEMS : (coreLen - done);
            const uint32_t offset = start + done;
            AscendC::LocalTensor<uint8_t> x1Local = inX1_.AllocTensor<uint8_t>();
            AscendC::LocalTensor<uint8_t> x2Local = inX2_.AllocTensor<uint8_t>();
            AscendC::LocalTensor<uint8_t> yLocal = outY_.AllocTensor<uint8_t>();
            AscendC::LocalTensor<half> x1Half = halfBuf_.Get<half>();
            AscendC::LocalTensor<half> x2Half = halfBuf_.Get<half>()[TILE_ELEMS];
            AscendC::LocalTensor<half> oneHalf = halfBuf_.Get<half>()[2u * TILE_ELEMS];
            AscendC::LocalTensor<half> boolHalf = halfBuf_.Get<half>()[3u * TILE_ELEMS];
            AscendC::LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();

            CopyGmToUb(x1Local, x1Gm_[offset], valid);
            CopyGmToUb(x2Local, x2Gm_[offset], valid);
            event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            inX1_.EnQue(x1Local);
            inX2_.EnQue(x2Local);

            x1Local = inX1_.DeQue<uint8_t>();
            x2Local = inX2_.DeQue<uint8_t>();
            AscendC::UnaryRepeatParams castHighParams{1, 1, 8, 4};
            const uint32_t castRepeat = CeilDiv(valid, NARROW_TO_HALF_OUTPUTS_PER_REPEAT);
            AscendC::Cast(x1Half, x1Local, AscendC::RoundMode::CAST_NONE, 256, castRepeat, castHighParams);
            AscendC::Cast(x2Half, x2Local, AscendC::RoundMode::CAST_NONE, 256, castRepeat, castHighParams);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Compare(maskLocal, x1Half, x2Half, AscendC::CMPMODE::EQ, valid);
            AscendC::Duplicate(oneHalf, static_cast<half>(1.0f), valid);
            AscendC::Duplicate(boolHalf, static_cast<half>(0.0f), valid);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select(boolHalf, maskLocal, oneHalf, boolHalf, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, valid);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(yLocal, boolHalf, AscendC::RoundMode::CAST_RINT, valid);
            AscendC::PipeBarrier<PIPE_V>();

            outY_.EnQue<uint8_t>(yLocal);
            inX1_.FreeTensor(x1Local);
            inX2_.FreeTensor(x2Local);

            yLocal = outY_.DeQue<uint8_t>();
            event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
            CopyUbToGmU8(yGm_[offset], yLocal, valid);
            outY_.FreeTensor(yLocal);
            done += valid;
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inX1_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inX2_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> halfBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf_;
    AscendC::GlobalTensor<uint8_t> x1Gm_;
    AscendC::GlobalTensor<uint8_t> x2Gm_;
    AscendC::GlobalTensor<uint8_t> yGm_;
    uint32_t outSize_ = 0u;
    bool useVector_ = false;
};
#endif

#if (ORIG_DTYPE_X1 == DT_UINT8 || ORIG_DTYPE_X1 == DT_INT8 || ORIG_DTYPE_X1 == DT_BOOL)
class KernelTensorEqualByteBroadcastRowVector {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingData, tiling);
        InitShape(tilingData);
        const bool vectorizable = rowLen_ >= 1u && (x1InnerStride_ == 0u || x1InnerStride_ == 1u) &&
                                  (x2InnerStride_ == 0u || x2InnerStride_ == 1u);
        const bool hasBroadcast = tilingData.x1Stride0 != tilingData.x2Stride0 ||
                                  tilingData.x1Stride1 != tilingData.x2Stride1 ||
                                  tilingData.x1Stride2 != tilingData.x2Stride2 ||
                                  tilingData.x1Stride3 != tilingData.x2Stride3;
        useVector_ = vectorizable && hasBroadcast && outRows_ > 0u;
        if (!useVector_) {
            return;
        }
        x1Gm_.SetGlobalBuffer((__gm__ uint8_t *)x1, tilingData.outSize);
        x2Gm_.SetGlobalBuffer((__gm__ uint8_t *)x2, tilingData.outSize);
        yGm_.SetGlobalBuffer((__gm__ uint8_t *)y, tilingData.outSize);
        pipe_.InitBuffer(inX1_, 1, TILE_ELEMS * sizeof(uint8_t));
        pipe_.InitBuffer(inX2_, 1, TILE_ELEMS * sizeof(uint8_t));
        pipe_.InitBuffer(outY_, 1, TILE_ELEMS * sizeof(uint8_t));
        pipe_.InitBuffer(halfBuf_, 4u * TILE_ELEMS * sizeof(half));
        pipe_.InitBuffer(scalarBuf_, TILE_ELEMS * sizeof(int16_t));
        pipe_.InitBuffer(maskBuf_, TILE_ELEMS * sizeof(uint8_t));
    }

    __aicore__ inline bool UseVector() const
    {
        return useVector_;
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        for (uint32_t row = blockIdx; row < outRows_; row += blockNum) {
            const uint32_t x1RowBase = ComputeRowBase(row, x1Stride0_, x1Stride1_, x1Stride2_);
            const uint32_t x2RowBase = ComputeRowBase(row, x2Stride0_, x2Stride1_, x2Stride2_);
            const uint32_t yRowBase = row * rowLen_;
            uint32_t done = 0u;
            while (done < rowLen_) {
                const uint32_t valid = (rowLen_ - done) > TILE_ELEMS ? TILE_ELEMS : (rowLen_ - done);
                AscendC::LocalTensor<uint8_t> x1Local = inX1_.AllocTensor<uint8_t>();
                AscendC::LocalTensor<uint8_t> x2Local = inX2_.AllocTensor<uint8_t>();
                AscendC::LocalTensor<uint8_t> yLocal = outY_.AllocTensor<uint8_t>();
                AscendC::LocalTensor<half> x1Half = halfBuf_.Get<half>();
                AscendC::LocalTensor<half> x2Half = halfBuf_.Get<half>()[TILE_ELEMS];
                AscendC::LocalTensor<half> oneHalf = halfBuf_.Get<half>()[2u * TILE_ELEMS];
                AscendC::LocalTensor<half> boolHalf = halfBuf_.Get<half>()[3u * TILE_ELEMS];
                AscendC::LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();
                LoadTileHalf(x1Half, x1Local, x1Gm_, x1RowBase, x1InnerStride_, done, valid);
                LoadTileHalf(x2Half, x2Local, x2Gm_, x2RowBase, x2InnerStride_, done, valid);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Compare(maskLocal, x1Half, x2Half, AscendC::CMPMODE::EQ, valid);
                AscendC::Duplicate(oneHalf, static_cast<half>(1.0f), valid);
                AscendC::Duplicate(boolHalf, static_cast<half>(0.0f), valid);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Select(boolHalf, maskLocal, oneHalf, boolHalf, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, valid);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(yLocal, boolHalf, AscendC::RoundMode::CAST_RINT, valid);
                AscendC::PipeBarrier<PIPE_V>();
                outY_.EnQue<uint8_t>(yLocal);
                inX1_.FreeTensor(x1Local);
                inX2_.FreeTensor(x2Local);
                yLocal = outY_.DeQue<uint8_t>();
                event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
                CopyUbToGmU8(yGm_[yRowBase + done], yLocal, valid);
                outY_.FreeTensor(yLocal);
                done += valid;
            }
        }
    }

private:
    __aicore__ inline void LoadTileHalf(AscendC::LocalTensor<half> halfLocal,
                                        AscendC::LocalTensor<uint8_t> local,
                                        AscendC::GlobalTensor<uint8_t> gm,
                                        uint32_t rowBase, uint32_t innerStride, uint32_t done, uint32_t valid)
    {
        if (innerStride == 1u) {
            CopyGmToUb(local, gm[rowBase + done], valid);
            event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            AscendC::UnaryRepeatParams castHighParams{1, 1, 8, 4};
            const uint32_t castRepeat = CeilDiv(valid, NARROW_TO_HALF_OUTPUTS_PER_REPEAT);
            AscendC::Cast(halfLocal, local, AscendC::RoundMode::CAST_NONE, 256, castRepeat, castHighParams);
        } else {
            AscendC::LocalTensor<int16_t> scalarLocal = scalarBuf_.Get<int16_t>();
            AscendC::Duplicate(scalarLocal, static_cast<int16_t>(gm.GetValue(rowBase)), valid);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(halfLocal, scalarLocal, AscendC::RoundMode::CAST_NONE, valid);
        }
    }

    __aicore__ inline void InitShape(const TensorEqualTilingData &tilingData)
    {
        rank_ = tilingData.rank;
        if (rank_ == 1u) {
            rowLen_ = tilingData.outShape0; outRows_ = 1u; x1InnerStride_ = tilingData.x1Stride0; x2InnerStride_ = tilingData.x2Stride0;
        } else if (rank_ == 2u) {
            rowLen_ = tilingData.outShape1; outRows_ = tilingData.outShape0; x1InnerStride_ = tilingData.x1Stride1; x2InnerStride_ = tilingData.x2Stride1;
        } else if (rank_ == 3u) {
            rowLen_ = tilingData.outShape2; outRows_ = tilingData.outShape0 * tilingData.outShape1; x1InnerStride_ = tilingData.x1Stride2; x2InnerStride_ = tilingData.x2Stride2;
        } else {
            rowLen_ = tilingData.outShape3; outRows_ = tilingData.outShape0 * tilingData.outShape1 * tilingData.outShape2; x1InnerStride_ = tilingData.x1Stride3; x2InnerStride_ = tilingData.x2Stride3;
        }
        outShape1_ = tilingData.outShape1; outShape2_ = tilingData.outShape2;
        x1Stride0_ = tilingData.x1Stride0; x1Stride1_ = tilingData.x1Stride1; x1Stride2_ = tilingData.x1Stride2;
        x2Stride0_ = tilingData.x2Stride0; x2Stride1_ = tilingData.x2Stride1; x2Stride2_ = tilingData.x2Stride2;
    }

    __aicore__ inline uint32_t ComputeRowBase(uint32_t row, uint32_t stride0, uint32_t stride1, uint32_t stride2) const
    {
        if (rank_ <= 1u) { return 0u; }
        if (rank_ == 2u) { return row * stride0; }
        if (rank_ == 3u) { const uint32_t idx0 = row / outShape1_; const uint32_t idx1 = row % outShape1_; return idx0 * stride0 + idx1 * stride1; }
        uint32_t rem = row; const uint32_t idx0 = rem / (outShape1_ * outShape2_); rem %= (outShape1_ * outShape2_);
        const uint32_t idx1 = rem / outShape2_; const uint32_t idx2 = rem % outShape2_;
        return idx0 * stride0 + idx1 * stride1 + idx2 * stride2;
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inX1_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inX2_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> halfBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf_;
    AscendC::GlobalTensor<uint8_t> x1Gm_;
    AscendC::GlobalTensor<uint8_t> x2Gm_;
    AscendC::GlobalTensor<uint8_t> yGm_;
    uint32_t rank_ = 0u, rowLen_ = 0u, outRows_ = 0u, outShape1_ = 1u, outShape2_ = 1u;
    uint32_t x1Stride0_ = 0u, x1Stride1_ = 0u, x1Stride2_ = 0u, x1InnerStride_ = 0u;
    uint32_t x2Stride0_ = 0u, x2Stride1_ = 0u, x2Stride2_ = 0u, x2InnerStride_ = 0u;
    bool useVector_ = false;
};
#endif

#if (ORIG_DTYPE_X1 == DT_INT16 || ORIG_DTYPE_X1 == DT_BF16)
class KernelTensorEqualInt16Vector {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingData, tiling);
        outSize_ = tilingData.outSize;
        const bool contiguous = tilingData.x1Stride0 == tilingData.x2Stride0 &&
                                tilingData.x1Stride1 == tilingData.x2Stride1 &&
                                tilingData.x1Stride2 == tilingData.x2Stride2 &&
                                tilingData.x1Stride3 == tilingData.x2Stride3;
        useVector_ = contiguous;
        x1Gm_.SetGlobalBuffer((__gm__ int16_t *)x1, outSize_);
        x2Gm_.SetGlobalBuffer((__gm__ int16_t *)x2, outSize_);
        yGm_.SetGlobalBuffer((__gm__ uint8_t *)y, outSize_);
        pipe_.InitBuffer(inX1_, 1, TILE_ELEMS * sizeof(int16_t));
        pipe_.InitBuffer(inX2_, 1, TILE_ELEMS * sizeof(int16_t));
        pipe_.InitBuffer(outY_, 1, TILE_ELEMS * sizeof(uint8_t));
        pipe_.InitBuffer(calcBuf_, 7u * TILE_ELEMS * sizeof(int16_t));
        pipe_.InitBuffer(floatBuf_, TILE_ELEMS * sizeof(float));
        pipe_.InitBuffer(halfBuf_, TILE_ELEMS * sizeof(half));
    }

    __aicore__ inline bool UseVector() const
    {
        return useVector_;
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint32_t coreCeil = CeilDiv(outSize_, blockNum);
        const uint32_t start = blockIdx * coreCeil;
        if (start >= outSize_) {
            return;
        }
        uint32_t coreLen = outSize_ - start;
        if (coreLen > coreCeil) {
            coreLen = coreCeil;
        }

        uint32_t done = 0;
        while (done < coreLen) {
            const uint32_t valid = (coreLen - done) > TILE_ELEMS ? TILE_ELEMS : (coreLen - done);
            const uint32_t offset = start + done;
            AscendC::LocalTensor<int16_t> x1Local = inX1_.AllocTensor<int16_t>();
            AscendC::LocalTensor<int16_t> x2Local = inX2_.AllocTensor<int16_t>();
            AscendC::LocalTensor<uint8_t> yLocal = outY_.AllocTensor<uint8_t>();

            CopyGmToUb(x1Local, x1Gm_[offset], valid);
            CopyGmToUb(x2Local, x2Gm_[offset], valid);
            event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            inX1_.EnQue(x1Local);
            inX2_.EnQue(x2Local);

            x1Local = inX1_.DeQue<int16_t>();
            x2Local = inX2_.DeQue<int16_t>();
            AscendC::LocalTensor<int16_t> negative = calcBuf_.Get<int16_t>();
            AscendC::LocalTensor<int16_t> sign = calcBuf_.Get<int16_t>()[TILE_ELEMS];
            AscendC::LocalTensor<int16_t> lhsOrRhs = calcBuf_.Get<int16_t>()[2u * TILE_ELEMS];
            AscendC::LocalTensor<int16_t> lhsAndRhs = calcBuf_.Get<int16_t>()[3u * TILE_ELEMS];
            AscendC::LocalTensor<int16_t> xorLR = calcBuf_.Get<int16_t>()[4u * TILE_ELEMS];
            AscendC::LocalTensor<int16_t> zero = calcBuf_.Get<int16_t>()[5u * TILE_ELEMS];
            AscendC::LocalTensor<int16_t> nonZeroSign = calcBuf_.Get<int16_t>()[6u * TILE_ELEMS];
            AscendC::LocalTensor<float> boolFloat = floatBuf_.Get<float>();
            AscendC::LocalTensor<half> boolHalf = halfBuf_.Get<half>();

            BuildNonZeroSign(x1Local, x2Local, negative, sign, lhsOrRhs, lhsAndRhs, xorLR, zero, nonZeroSign, valid);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(boolFloat, nonZeroSign, AscendC::RoundMode::CAST_NONE, valid);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(boolFloat, boolFloat, -3.0517578125e-5f, valid);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(boolFloat, boolFloat, -1.0f, valid);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(boolFloat, boolFloat, 1.0f, valid);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(boolHalf, boolFloat, AscendC::RoundMode::CAST_RINT, valid);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(yLocal, boolHalf, AscendC::RoundMode::CAST_RINT, valid);
            AscendC::PipeBarrier<PIPE_V>();

            outY_.EnQue<uint8_t>(yLocal);
            inX1_.FreeTensor(x1Local);
            inX2_.FreeTensor(x2Local);

            yLocal = outY_.DeQue<uint8_t>();
            event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
            CopyUbToGmU8(yGm_[offset], yLocal, valid);
            outY_.FreeTensor(yLocal);
            done += valid;
        }
    }

private:
    __aicore__ inline void BuildNonZeroSign(
        AscendC::LocalTensor<int16_t> lhs,
        AscendC::LocalTensor<int16_t> rhs,
        AscendC::LocalTensor<int16_t> negative,
        AscendC::LocalTensor<int16_t> sign,
        AscendC::LocalTensor<int16_t> lhsOrRhs,
        AscendC::LocalTensor<int16_t> lhsAndRhs,
        AscendC::LocalTensor<int16_t> xorLR,
        AscendC::LocalTensor<int16_t> zero,
        AscendC::LocalTensor<int16_t> nonZeroSign,
        uint32_t validLength)
    {
        AscendC::Duplicate<int16_t>(sign, static_cast<int16_t>(-32768), validLength);
        AscendC::Duplicate<int16_t>(zero, static_cast<int16_t>(0), validLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Or(lhsOrRhs, lhs, rhs, validLength);
        AscendC::And(lhsAndRhs, lhs, rhs, validLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Not(lhsAndRhs, lhsAndRhs, validLength);
        AscendC::And(xorLR, lhsOrRhs, lhsAndRhs, validLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(negative, zero, xorLR, validLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Or(nonZeroSign, xorLR, negative, validLength);
        AscendC::And(nonZeroSign, nonZeroSign, sign, validLength);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inX1_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inX2_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> halfBuf_;
    AscendC::GlobalTensor<int16_t> x1Gm_;
    AscendC::GlobalTensor<int16_t> x2Gm_;
    AscendC::GlobalTensor<uint8_t> yGm_;
    uint32_t outSize_ = 0u;
    bool useVector_ = false;
};
#endif

#if (ORIG_DTYPE_X1 == DT_INT16 || ORIG_DTYPE_X1 == DT_BF16)
class KernelTensorEqualInt16BroadcastRowVector {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingData, tiling);
        InitShape(tilingData);
        useVector_ = ShouldUseBroadcastRowVector(tilingData);
        if (!useVector_) {
            return;
        }
        x1Gm_.SetGlobalBuffer((__gm__ int16_t *)x1, tilingData.outSize);
        x2Gm_.SetGlobalBuffer((__gm__ int16_t *)x2, tilingData.outSize);
        yGm_.SetGlobalBuffer((__gm__ uint8_t *)y, tilingData.outSize);
        pipe_.InitBuffer(inX1_, 1, TILE_ELEMS * sizeof(int16_t));
        pipe_.InitBuffer(inX2_, 1, TILE_ELEMS * sizeof(int16_t));
        pipe_.InitBuffer(outY_, 1, TILE_ELEMS * sizeof(uint8_t));
        pipe_.InitBuffer(calcBuf_, 7u * TILE_ELEMS * sizeof(int16_t));
        pipe_.InitBuffer(floatBuf_, TILE_ELEMS * sizeof(float));
        pipe_.InitBuffer(halfBuf_, TILE_ELEMS * sizeof(half));
    }

    __aicore__ inline bool UseVector() const
    {
        return useVector_;
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        for (uint32_t row = blockIdx; row < outRows_; row += blockNum) {
            const uint32_t x1RowBase = ComputeRowBase(row, x1Stride0_, x1Stride1_, x1Stride2_);
            const uint32_t x2RowBase = ComputeRowBase(row, x2Stride0_, x2Stride1_, x2Stride2_);
            const uint32_t yRowBase = row * rowLen_;
            uint32_t done = 0u;
            while (done < rowLen_) {
                const uint32_t valid = (rowLen_ - done) > TILE_ELEMS ? TILE_ELEMS : (rowLen_ - done);
                AscendC::LocalTensor<int16_t> x1Local = inX1_.AllocTensor<int16_t>();
                AscendC::LocalTensor<int16_t> x2Local = inX2_.AllocTensor<int16_t>();
                AscendC::LocalTensor<uint8_t> yLocal = outY_.AllocTensor<uint8_t>();
                LoadTile(x1Local, x1Gm_, x1RowBase, x1InnerStride_, done, valid);
                LoadTile(x2Local, x2Gm_, x2RowBase, x2InnerStride_, done, valid);

                AscendC::LocalTensor<int16_t> negative = calcBuf_.Get<int16_t>();
                AscendC::LocalTensor<int16_t> sign = calcBuf_.Get<int16_t>()[TILE_ELEMS];
                AscendC::LocalTensor<int16_t> lhsOrRhs = calcBuf_.Get<int16_t>()[2u * TILE_ELEMS];
                AscendC::LocalTensor<int16_t> lhsAndRhs = calcBuf_.Get<int16_t>()[3u * TILE_ELEMS];
                AscendC::LocalTensor<int16_t> xorLR = calcBuf_.Get<int16_t>()[4u * TILE_ELEMS];
                AscendC::LocalTensor<int16_t> zero = calcBuf_.Get<int16_t>()[5u * TILE_ELEMS];
                AscendC::LocalTensor<int16_t> nonZeroSign = calcBuf_.Get<int16_t>()[6u * TILE_ELEMS];
                AscendC::LocalTensor<float> boolFloat = floatBuf_.Get<float>();
                AscendC::LocalTensor<half> boolHalf = halfBuf_.Get<half>();

                BuildNonZeroSign(x1Local, x2Local, negative, sign, lhsOrRhs, lhsAndRhs, xorLR, zero, nonZeroSign, valid);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(boolFloat, nonZeroSign, AscendC::RoundMode::CAST_NONE, valid);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(boolFloat, boolFloat, -3.0517578125e-5f, valid);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(boolFloat, boolFloat, -1.0f, valid);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Adds(boolFloat, boolFloat, 1.0f, valid);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(boolHalf, boolFloat, AscendC::RoundMode::CAST_RINT, valid);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(yLocal, boolHalf, AscendC::RoundMode::CAST_RINT, valid);
                AscendC::PipeBarrier<PIPE_V>();

                outY_.EnQue<uint8_t>(yLocal);
                inX1_.FreeTensor(x1Local);
                inX2_.FreeTensor(x2Local);
                yLocal = outY_.DeQue<uint8_t>();
                event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
                CopyUbToGmU8(yGm_[yRowBase + done], yLocal, valid);
                outY_.FreeTensor(yLocal);
                done += valid;
            }
        }
    }

private:
    __aicore__ inline void LoadTile(AscendC::LocalTensor<int16_t> local, AscendC::GlobalTensor<int16_t> gm,
                                    uint32_t rowBase, uint32_t innerStride, uint32_t done, uint32_t valid)
    {
        if (innerStride == 1u) {
            CopyGmToUb(local, gm[rowBase + done], valid);
            event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
        } else {
            AscendC::Duplicate(local, gm.GetValue(rowBase), valid);
        }
    }

    __aicore__ inline void BuildNonZeroSign(
        AscendC::LocalTensor<int16_t> lhs,
        AscendC::LocalTensor<int16_t> rhs,
        AscendC::LocalTensor<int16_t> negative,
        AscendC::LocalTensor<int16_t> sign,
        AscendC::LocalTensor<int16_t> lhsOrRhs,
        AscendC::LocalTensor<int16_t> lhsAndRhs,
        AscendC::LocalTensor<int16_t> xorLR,
        AscendC::LocalTensor<int16_t> zero,
        AscendC::LocalTensor<int16_t> nonZeroSign,
        uint32_t validLength)
    {
        AscendC::Duplicate<int16_t>(sign, static_cast<int16_t>(-32768), validLength);
        AscendC::Duplicate<int16_t>(zero, static_cast<int16_t>(0), validLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Or(lhsOrRhs, lhs, rhs, validLength);
        AscendC::And(lhsAndRhs, lhs, rhs, validLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Not(lhsAndRhs, lhsAndRhs, validLength);
        AscendC::And(xorLR, lhsOrRhs, lhsAndRhs, validLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(negative, zero, xorLR, validLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Or(nonZeroSign, xorLR, negative, validLength);
        AscendC::And(nonZeroSign, nonZeroSign, sign, validLength);
    }

    __aicore__ inline void InitShape(const TensorEqualTilingData &tilingData)
    {
        rank_ = tilingData.rank;
        if (rank_ == 1u) {
            rowLen_ = tilingData.outShape0; outRows_ = 1u; x1InnerStride_ = tilingData.x1Stride0; x2InnerStride_ = tilingData.x2Stride0;
        } else if (rank_ == 2u) {
            rowLen_ = tilingData.outShape1; outRows_ = tilingData.outShape0; x1InnerStride_ = tilingData.x1Stride1; x2InnerStride_ = tilingData.x2Stride1;
        } else if (rank_ == 3u) {
            rowLen_ = tilingData.outShape2; outRows_ = tilingData.outShape0 * tilingData.outShape1; x1InnerStride_ = tilingData.x1Stride2; x2InnerStride_ = tilingData.x2Stride2;
        } else {
            rowLen_ = tilingData.outShape3; outRows_ = tilingData.outShape0 * tilingData.outShape1 * tilingData.outShape2; x1InnerStride_ = tilingData.x1Stride3; x2InnerStride_ = tilingData.x2Stride3;
        }
        outShape1_ = tilingData.outShape1; outShape2_ = tilingData.outShape2;
        x1Stride0_ = tilingData.x1Stride0; x1Stride1_ = tilingData.x1Stride1; x1Stride2_ = tilingData.x1Stride2;
        x2Stride0_ = tilingData.x2Stride0; x2Stride1_ = tilingData.x2Stride1; x2Stride2_ = tilingData.x2Stride2;
    }

    __aicore__ inline uint32_t ComputeRowBase(uint32_t row, uint32_t stride0, uint32_t stride1, uint32_t stride2) const
    {
        if (rank_ <= 1u) { return 0u; }
        if (rank_ == 2u) { return row * stride0; }
        if (rank_ == 3u) { const uint32_t idx0 = row / outShape1_; const uint32_t idx1 = row % outShape1_; return idx0 * stride0 + idx1 * stride1; }
        uint32_t rem = row; const uint32_t idx0 = rem / (outShape1_ * outShape2_); rem %= (outShape1_ * outShape2_);
        const uint32_t idx1 = rem / outShape2_; const uint32_t idx2 = rem % outShape2_;
        return idx0 * stride0 + idx1 * stride1 + idx2 * stride2;
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inX1_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inX2_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> halfBuf_;
    AscendC::GlobalTensor<int16_t> x1Gm_;
    AscendC::GlobalTensor<int16_t> x2Gm_;
    AscendC::GlobalTensor<uint8_t> yGm_;
    uint32_t rank_ = 0u, rowLen_ = 0u, outRows_ = 0u, outShape1_ = 1u, outShape2_ = 1u;
    uint32_t x1Stride0_ = 0u, x1Stride1_ = 0u, x1Stride2_ = 0u, x1InnerStride_ = 0u;
    uint32_t x2Stride0_ = 0u, x2Stride1_ = 0u, x2Stride2_ = 0u, x2InnerStride_ = 0u;
    bool useVector_ = false;
};
#endif

#if (ORIG_DTYPE_X1 == DT_INT32)
class KernelTensorEqualInt32Vector {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingData, tiling);
        outSize_ = tilingData.outSize;
        x1Gm_.SetGlobalBuffer((__gm__ int32_t *)x1, outSize_);
        x2Gm_.SetGlobalBuffer((__gm__ int32_t *)x2, outSize_);
        yGm_.SetGlobalBuffer((__gm__ uint8_t *)y, outSize_);
        pipe_.InitBuffer(inX1_, 2, INT32_TILE_ELEMS * sizeof(int32_t));
        pipe_.InitBuffer(inX2_, 2, INT32_TILE_ELEMS * sizeof(int32_t));
        pipe_.InitBuffer(outY_, 1, INT32_TILE_ELEMS * sizeof(uint8_t));
        pipe_.InitBuffer(calcBuf_, INT32_TILE_ELEMS * sizeof(int32_t));
        pipe_.InitBuffer(floatBuf_, 2u * INT32_TILE_ELEMS * sizeof(float));
        pipe_.InitBuffer(halfBuf_, INT32_TILE_ELEMS * sizeof(half));
        pipe_.InitBuffer(maskBuf_, INT32_TILE_ELEMS * sizeof(uint8_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint32_t coreCeil = CeilDiv(outSize_, blockNum);
        const uint32_t start = blockIdx * coreCeil;
        if (start >= outSize_) {
            return;
        }
        uint32_t coreLen = outSize_ - start;
        if (coreLen > coreCeil) {
            coreLen = coreCeil;
        }

        uint32_t done = 0u;
        uint32_t valid = coreLen > INT32_TILE_ELEMS ? INT32_TILE_ELEMS : coreLen;
        uint32_t offset = start;
        AscendC::LocalTensor<int32_t> x1Local = inX1_.AllocTensor<int32_t>();
        AscendC::LocalTensor<int32_t> x2Local = inX2_.AllocTensor<int32_t>();
        CopyGmToUb(x1Local, x1Gm_[offset], valid);
        CopyGmToUb(x2Local, x2Gm_[offset], valid);
        event_t eventIdFirstMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdFirstMte2ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdFirstMte2ToV);

        while (done < coreLen) {
            const uint32_t nextDone = done + valid;
            const bool hasNext = nextDone < coreLen;
            uint32_t nextValid = 0u;
            AscendC::LocalTensor<int32_t> nextX1Local;
            AscendC::LocalTensor<int32_t> nextX2Local;
            event_t eventIdNextMte2ToV;
            if (hasNext) {
                nextValid = (coreLen - nextDone) > INT32_TILE_ELEMS ? INT32_TILE_ELEMS : (coreLen - nextDone);
                const uint32_t nextOffset = start + nextDone;
                nextX1Local = inX1_.AllocTensor<int32_t>();
                nextX2Local = inX2_.AllocTensor<int32_t>();
                CopyGmToUb(nextX1Local, x1Gm_[nextOffset], nextValid);
                CopyGmToUb(nextX2Local, x2Gm_[nextOffset], nextValid);
                eventIdNextMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdNextMte2ToV);
            }

            AscendC::LocalTensor<uint8_t> yLocal = outY_.AllocTensor<uint8_t>();

            AscendC::LocalTensor<int32_t> diffLocal = calcBuf_.Get<int32_t>();
            AscendC::LocalTensor<float> oneFloat = floatBuf_.Get<float>();
            AscendC::LocalTensor<float> boolFloat = floatBuf_.Get<float>()[INT32_TILE_ELEMS];
            AscendC::LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();
            AscendC::LocalTensor<half> boolHalf = halfBuf_.Get<half>();

            AscendC::Sub(diffLocal, x1Local, x2Local, valid);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::CompareScalar<int32_t, uint8_t>(maskLocal, diffLocal, static_cast<int32_t>(0),
                                                     AscendC::CMPMODE::EQ, valid);
            AscendC::Duplicate(oneFloat, 1.0f, valid);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select<float, uint8_t>(boolFloat, maskLocal, oneFloat, 0.0f,
                                            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, valid);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(boolHalf, boolFloat, AscendC::RoundMode::CAST_RINT, valid);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(yLocal, boolHalf, AscendC::RoundMode::CAST_RINT, valid);
            AscendC::PipeBarrier<PIPE_V>();

            outY_.EnQue<uint8_t>(yLocal);
            inX1_.FreeTensor(x1Local);
            inX2_.FreeTensor(x2Local);
            yLocal = outY_.DeQue<uint8_t>();
            event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
            CopyUbToGmU8(yGm_[offset], yLocal, valid);
            outY_.FreeTensor(yLocal);
            if (!hasNext) {
                done = coreLen;
                continue;
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdNextMte2ToV);
            done = nextDone;
            valid = nextValid;
            offset = start + done;
            x1Local = nextX1Local;
            x2Local = nextX2Local;
        }
    }

private:
    __aicore__ inline void BuildNonZeroSign(
        AscendC::LocalTensor<int32_t> lhs,
        AscendC::LocalTensor<int32_t> rhs,
        AscendC::LocalTensor<int32_t> negative,
        AscendC::LocalTensor<int32_t> sign,
        AscendC::LocalTensor<int32_t> lhsOrRhs,
        AscendC::LocalTensor<int32_t> lhsAndRhs,
        AscendC::LocalTensor<int32_t> xorLR,
        AscendC::LocalTensor<int32_t> zero,
        AscendC::LocalTensor<int32_t> nonZeroSign,
        uint32_t validLength)
    {
        AscendC::Duplicate<int32_t>(sign, static_cast<int32_t>(-2147483647 - 1), validLength);
        AscendC::Duplicate<int32_t>(zero, static_cast<int32_t>(0), validLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Or(lhsOrRhs, lhs, rhs, validLength);
        AscendC::And(lhsAndRhs, lhs, rhs, validLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Not(lhsAndRhs, lhsAndRhs, validLength);
        AscendC::And(xorLR, lhsOrRhs, lhsAndRhs, validLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(negative, zero, xorLR, validLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Or(nonZeroSign, xorLR, negative, validLength);
        AscendC::And(nonZeroSign, nonZeroSign, sign, validLength);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inX1_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inX2_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> halfBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf_;
    AscendC::GlobalTensor<int32_t> x1Gm_;
    AscendC::GlobalTensor<int32_t> x2Gm_;
    AscendC::GlobalTensor<uint8_t> yGm_;
    uint32_t outSize_ = 0u;
};

class KernelTensorEqualInt32BroadcastRowVector {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingData, tiling);
        InitShape(tilingData);
        const bool vectorizable = rowLen_ >= 1u && (x1InnerStride_ == 0u || x1InnerStride_ == 1u) &&
                                  (x2InnerStride_ == 0u || x2InnerStride_ == 1u);
        const bool hasBroadcast = tilingData.x1Stride0 != tilingData.x2Stride0 ||
                                  tilingData.x1Stride1 != tilingData.x2Stride1 ||
                                  tilingData.x1Stride2 != tilingData.x2Stride2 ||
                                  tilingData.x1Stride3 != tilingData.x2Stride3;
        useVector_ = vectorizable && hasBroadcast && outRows_ > 0u;
        if (!useVector_) {
            return;
        }
        x1Gm_.SetGlobalBuffer((__gm__ int32_t *)x1, tilingData.outSize);
        x2Gm_.SetGlobalBuffer((__gm__ int32_t *)x2, tilingData.outSize);
        yGm_.SetGlobalBuffer((__gm__ uint8_t *)y, tilingData.outSize);
        pipe_.InitBuffer(inX1_, 1, TILE_ELEMS * sizeof(int32_t));
        pipe_.InitBuffer(inX2_, 1, TILE_ELEMS * sizeof(int32_t));
        pipe_.InitBuffer(outY_, 1, TILE_ELEMS);
        pipe_.InitBuffer(calcBuf_, TILE_ELEMS * sizeof(int32_t));
        pipe_.InitBuffer(floatBuf_, 2u * TILE_ELEMS * sizeof(float));
        pipe_.InitBuffer(halfBuf_, TILE_ELEMS * sizeof(half));
        pipe_.InitBuffer(maskBuf_, TILE_ELEMS * sizeof(uint8_t));
    }

    __aicore__ inline bool UseVector() const
    {
        return useVector_;
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        for (uint32_t row = blockIdx; row < outRows_; row += blockNum) {
            const uint32_t x1RowBase = ComputeRowBase(row, x1Stride0_, x1Stride1_, x1Stride2_);
            const uint32_t x2RowBase = ComputeRowBase(row, x2Stride0_, x2Stride1_, x2Stride2_);
            const uint32_t yRowBase = row * rowLen_;
            uint32_t done = 0u;
            while (done < rowLen_) {
                const uint32_t valid = (rowLen_ - done) > TILE_ELEMS ? TILE_ELEMS : (rowLen_ - done);
                AscendC::LocalTensor<int32_t> x1Local = inX1_.AllocTensor<int32_t>();
                AscendC::LocalTensor<int32_t> x2Local = inX2_.AllocTensor<int32_t>();
                AscendC::LocalTensor<uint8_t> yLocal = outY_.AllocTensor<uint8_t>();
                LoadTile(x1Local, x1Gm_, x1RowBase, x1InnerStride_, done, valid);
                LoadTile(x2Local, x2Gm_, x2RowBase, x2InnerStride_, done, valid);
                AscendC::LocalTensor<int32_t> diffLocal = calcBuf_.Get<int32_t>();
                AscendC::LocalTensor<float> oneFloat = floatBuf_.Get<float>();
                AscendC::LocalTensor<float> boolFloat = floatBuf_.Get<float>()[TILE_ELEMS];
                AscendC::LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();
                AscendC::LocalTensor<half> halfLocal = halfBuf_.Get<half>();

                AscendC::Sub(diffLocal, x1Local, x2Local, valid);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::CompareScalar<int32_t, uint8_t>(maskLocal, diffLocal, static_cast<int32_t>(0),
                                                         AscendC::CMPMODE::EQ, valid);
                AscendC::Duplicate(oneFloat, 1.0f, valid);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Select<float, uint8_t>(boolFloat, maskLocal, oneFloat, 0.0f,
                                                AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, valid);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(halfLocal, boolFloat, AscendC::RoundMode::CAST_RINT, valid);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(yLocal, halfLocal, AscendC::RoundMode::CAST_RINT, valid);
                AscendC::PipeBarrier<PIPE_V>();
                outY_.EnQue<uint8_t>(yLocal);
                inX1_.FreeTensor(x1Local);
                inX2_.FreeTensor(x2Local);
                yLocal = outY_.DeQue<uint8_t>();
                event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
                CopyUbToGmU8(yGm_[yRowBase + done], yLocal, valid);
                outY_.FreeTensor(yLocal);
                done += valid;
            }
        }
    }

private:
    __aicore__ inline void BuildNonZeroSign(
        AscendC::LocalTensor<int32_t> lhs,
        AscendC::LocalTensor<int32_t> rhs,
        AscendC::LocalTensor<int32_t> negative,
        AscendC::LocalTensor<int32_t> sign,
        AscendC::LocalTensor<int32_t> lhsOrRhs,
        AscendC::LocalTensor<int32_t> lhsAndRhs,
        AscendC::LocalTensor<int32_t> xorLR,
        AscendC::LocalTensor<int32_t> zero,
        AscendC::LocalTensor<int32_t> nonZeroSign,
        uint32_t validLength)
    {
        AscendC::Duplicate<int32_t>(sign, static_cast<int32_t>(-2147483647 - 1), validLength);
        AscendC::Duplicate<int32_t>(zero, static_cast<int32_t>(0), validLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Or(lhsOrRhs, lhs, rhs, validLength);
        AscendC::And(lhsAndRhs, lhs, rhs, validLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Not(lhsAndRhs, lhsAndRhs, validLength);
        AscendC::And(xorLR, lhsOrRhs, lhsAndRhs, validLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(negative, zero, xorLR, validLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Or(nonZeroSign, xorLR, negative, validLength);
        AscendC::And(nonZeroSign, nonZeroSign, sign, validLength);
    }

    __aicore__ inline void LoadTile(AscendC::LocalTensor<int32_t> local, AscendC::GlobalTensor<int32_t> gm,
                                    uint32_t rowBase, uint32_t innerStride, uint32_t done, uint32_t valid)
    {
        if (innerStride == 1u) {
            CopyGmToUb(local, gm[rowBase + done], valid);
            event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
        } else {
            AscendC::Duplicate(local, gm.GetValue(rowBase), valid);
        }
    }

    __aicore__ inline void InitShape(const TensorEqualTilingData &tilingData)
    {
        rank_ = tilingData.rank;
        if (rank_ == 1u) {
            rowLen_ = tilingData.outShape0; outRows_ = 1u; x1InnerStride_ = tilingData.x1Stride0; x2InnerStride_ = tilingData.x2Stride0;
        } else if (rank_ == 2u) {
            rowLen_ = tilingData.outShape1; outRows_ = tilingData.outShape0; x1InnerStride_ = tilingData.x1Stride1; x2InnerStride_ = tilingData.x2Stride1;
        } else if (rank_ == 3u) {
            rowLen_ = tilingData.outShape2; outRows_ = tilingData.outShape0 * tilingData.outShape1; x1InnerStride_ = tilingData.x1Stride2; x2InnerStride_ = tilingData.x2Stride2;
        } else {
            rowLen_ = tilingData.outShape3; outRows_ = tilingData.outShape0 * tilingData.outShape1 * tilingData.outShape2; x1InnerStride_ = tilingData.x1Stride3; x2InnerStride_ = tilingData.x2Stride3;
        }
        outShape1_ = tilingData.outShape1; outShape2_ = tilingData.outShape2;
        x1Stride0_ = tilingData.x1Stride0; x1Stride1_ = tilingData.x1Stride1; x1Stride2_ = tilingData.x1Stride2;
        x2Stride0_ = tilingData.x2Stride0; x2Stride1_ = tilingData.x2Stride1; x2Stride2_ = tilingData.x2Stride2;
    }

    __aicore__ inline uint32_t ComputeRowBase(uint32_t row, uint32_t stride0, uint32_t stride1, uint32_t stride2) const
    {
        if (rank_ <= 1u) { return 0u; }
        if (rank_ == 2u) { return row * stride0; }
        if (rank_ == 3u) { const uint32_t idx0 = row / outShape1_; const uint32_t idx1 = row % outShape1_; return idx0 * stride0 + idx1 * stride1; }
        uint32_t rem = row; const uint32_t idx0 = rem / (outShape1_ * outShape2_); rem %= (outShape1_ * outShape2_);
        const uint32_t idx1 = rem / outShape2_; const uint32_t idx2 = rem % outShape2_;
        return idx0 * stride0 + idx1 * stride1 + idx2 * stride2;
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inX1_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inX2_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> halfBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf_;
    AscendC::GlobalTensor<int32_t> x1Gm_;
    AscendC::GlobalTensor<int32_t> x2Gm_;
    AscendC::GlobalTensor<uint8_t> yGm_;
    uint32_t rank_ = 0u, rowLen_ = 0u, outRows_ = 0u, outShape1_ = 1u, outShape2_ = 1u;
    uint32_t x1Stride0_ = 0u, x1Stride1_ = 0u, x1Stride2_ = 0u, x1InnerStride_ = 0u;
    uint32_t x2Stride0_ = 0u, x2Stride1_ = 0u, x2Stride2_ = 0u, x2InnerStride_ = 0u;
    bool useVector_ = false;
};
#endif

#if (ORIG_DTYPE_X1 == DT_FLOAT || ORIG_DTYPE_X1 == DT_FLOAT16)
class KernelTensorEqualBroadcastRowVector {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingData, tiling);
        outSize_ = tilingData.outSize;
        rank_ = tilingData.rank;
        if (rank_ == 1u) {
            rowLen_ = tilingData.outShape0;
            outRows_ = 1u;
            x1InnerStride_ = tilingData.x1Stride0;
            x2InnerStride_ = tilingData.x2Stride0;
        } else if (rank_ == 2u) {
            rowLen_ = tilingData.outShape1;
            outRows_ = tilingData.outShape0;
            x1InnerStride_ = tilingData.x1Stride1;
            x2InnerStride_ = tilingData.x2Stride1;
        } else if (rank_ == 3u) {
            rowLen_ = tilingData.outShape2;
            outRows_ = tilingData.outShape0 * tilingData.outShape1;
            x1InnerStride_ = tilingData.x1Stride2;
            x2InnerStride_ = tilingData.x2Stride2;
        } else {
            rowLen_ = tilingData.outShape3;
            outRows_ = tilingData.outShape0 * tilingData.outShape1 * tilingData.outShape2;
            x1InnerStride_ = tilingData.x1Stride3;
            x2InnerStride_ = tilingData.x2Stride3;
        }
        outShape0_ = tilingData.outShape0;
        outShape1_ = tilingData.outShape1;
        outShape2_ = tilingData.outShape2;
        x1Stride0_ = tilingData.x1Stride0;
        x1Stride1_ = tilingData.x1Stride1;
        x1Stride2_ = tilingData.x1Stride2;
        x1Stride3_ = tilingData.x1Stride3;
        x2Stride0_ = tilingData.x2Stride0;
        x2Stride1_ = tilingData.x2Stride1;
        x2Stride2_ = tilingData.x2Stride2;
        x2Stride3_ = tilingData.x2Stride3;

        const bool lastDimVectorizable = rowLen_ >= 1u &&
                                         (x1InnerStride_ == 0u || x1InnerStride_ == 1u) &&
                                         (x2InnerStride_ == 0u || x2InnerStride_ == 1u);
        const bool hasBroadcast = x1Stride0_ != x2Stride0_ || x1Stride1_ != x2Stride1_ ||
                                  x1Stride2_ != x2Stride2_ || x1Stride3_ != x2Stride3_;
        useVector_ = lastDimVectorizable && hasBroadcast && outRows_ > 0u;
        if (!useVector_) {
            return;
        }

        x1Gm_.SetGlobalBuffer((__gm__ DTYPE_X1 *)x1, outSize_);
        x2Gm_.SetGlobalBuffer((__gm__ DTYPE_X1 *)x2, outSize_);
        yGm_.SetGlobalBuffer((__gm__ uint8_t *)y, outSize_);
        pipe_.InitBuffer(inX1_, 1, FP_TILE_ELEMS * sizeof(DTYPE_X1));
        pipe_.InitBuffer(inX2_, 1, FP_TILE_ELEMS * sizeof(DTYPE_X1));
        pipe_.InitBuffer(outY_, 1, FP_TILE_ELEMS);
        pipe_.InitBuffer(calcBuf_, 2u * FP_TILE_ELEMS * sizeof(DTYPE_X1));
        pipe_.InitBuffer(maskBuf_, FP_TILE_ELEMS);
#if (ORIG_DTYPE_X1 == DT_FLOAT)
        pipe_.InitBuffer(halfBuf_, FP_TILE_ELEMS * sizeof(half));
#endif
    }

    __aicore__ inline bool UseVector() const
    {
        return useVector_;
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        for (uint32_t row = blockIdx; row < outRows_; row += blockNum) {
            const uint32_t x1RowBase = ComputeRowBase(row, x1Stride0_, x1Stride1_, x1Stride2_);
            const uint32_t x2RowBase = ComputeRowBase(row, x2Stride0_, x2Stride1_, x2Stride2_);
            const uint32_t yRowBase = row * rowLen_;

            uint32_t done = 0u;
            while (done < rowLen_) {
                const uint32_t valid = (rowLen_ - done) > FP_TILE_ELEMS ? FP_TILE_ELEMS : (rowLen_ - done);
                AscendC::LocalTensor<DTYPE_X1> x1Local = inX1_.AllocTensor<DTYPE_X1>();
                AscendC::LocalTensor<DTYPE_X1> x2Local = inX2_.AllocTensor<DTYPE_X1>();
                AscendC::LocalTensor<uint8_t> yLocal = outY_.AllocTensor<uint8_t>();
                AscendC::LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();
                AscendC::LocalTensor<DTYPE_X1> oneLocal = calcBuf_.Get<DTYPE_X1>();
                AscendC::LocalTensor<DTYPE_X1> boolLocal = calcBuf_.Get<DTYPE_X1>()[FP_TILE_ELEMS];

                const bool copyX1 = x1InnerStride_ == 1u;
                const bool copyX2 = x2InnerStride_ == 1u;
                if (copyX1) {
                    CopyGmToUb(x1Local, x1Gm_[x1RowBase + done], valid);
                }
                if (copyX2) {
                    CopyGmToUb(x2Local, x2Gm_[x2RowBase + done], valid);
                }
                if (copyX1 || copyX2) {
                    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                }
                if (!copyX1) {
                    AscendC::Duplicate(x1Local, x1Gm_.GetValue(x1RowBase), valid);
                }
                if (!copyX2) {
                    AscendC::Duplicate(x2Local, x2Gm_.GetValue(x2RowBase), valid);
                }
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::Compare(maskLocal, x1Local, x2Local, AscendC::CMPMODE::EQ, valid);
                AscendC::Duplicate(oneLocal, static_cast<DTYPE_X1>(1.0f), valid);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Select<DTYPE_X1, uint8_t>(boolLocal, maskLocal, oneLocal, static_cast<DTYPE_X1>(0.0f),
                                                   AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, valid);
                AscendC::PipeBarrier<PIPE_V>();
#if (ORIG_DTYPE_X1 == DT_FLOAT)
                AscendC::LocalTensor<half> halfLocal = halfBuf_.Get<half>();
                AscendC::Cast(halfLocal, boolLocal, AscendC::RoundMode::CAST_RINT, valid);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(yLocal, halfLocal, AscendC::RoundMode::CAST_RINT, valid);
#else
                AscendC::Cast(yLocal, boolLocal, AscendC::RoundMode::CAST_RINT, valid);
#endif
                AscendC::PipeBarrier<PIPE_V>();

                outY_.EnQue<uint8_t>(yLocal);
                inX1_.FreeTensor(x1Local);
                inX2_.FreeTensor(x2Local);

                yLocal = outY_.DeQue<uint8_t>();
                event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
                CopyUbToGmU8(yGm_[yRowBase + done], yLocal, valid);
                outY_.FreeTensor(yLocal);
                done += valid;
            }
        }
    }

private:
    __aicore__ inline uint32_t ComputeRowBase(uint32_t row, uint32_t stride0, uint32_t stride1, uint32_t stride2) const
    {
        if (rank_ <= 1u) {
            return 0u;
        }
        if (rank_ == 2u) {
            return row * stride0;
        }
        if (rank_ == 3u) {
            const uint32_t idx0 = row / outShape1_;
            const uint32_t idx1 = row % outShape1_;
            return idx0 * stride0 + idx1 * stride1;
        }
        uint32_t rem = row;
        const uint32_t idx0 = rem / (outShape1_ * outShape2_);
        rem %= (outShape1_ * outShape2_);
        const uint32_t idx1 = rem / outShape2_;
        const uint32_t idx2 = rem % outShape2_;
        return idx0 * stride0 + idx1 * stride1 + idx2 * stride2;
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inX1_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inX2_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf_;
#if (ORIG_DTYPE_X1 == DT_FLOAT)
    AscendC::TBuf<AscendC::TPosition::VECCALC> halfBuf_;
#endif
    AscendC::GlobalTensor<DTYPE_X1> x1Gm_;
    AscendC::GlobalTensor<DTYPE_X1> x2Gm_;
    AscendC::GlobalTensor<uint8_t> yGm_;
    uint32_t outSize_ = 0u;
    uint32_t rank_ = 0u;
    uint32_t rowLen_ = 0u;
    uint32_t outRows_ = 0u;
    uint32_t outShape0_ = 1u;
    uint32_t outShape1_ = 1u;
    uint32_t outShape2_ = 1u;
    uint32_t x1Stride0_ = 0u;
    uint32_t x1Stride1_ = 0u;
    uint32_t x1Stride2_ = 0u;
    uint32_t x1Stride3_ = 0u;
    uint32_t x2Stride0_ = 0u;
    uint32_t x2Stride1_ = 0u;
    uint32_t x2Stride2_ = 0u;
    uint32_t x2Stride3_ = 0u;
    uint32_t x1InnerStride_ = 0u;
    uint32_t x2InnerStride_ = 0u;
    bool useVector_ = false;
};

class KernelTensorEqualPureVector {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingData, tiling);
        outSize_ = tilingData.outSize;
        const bool contiguous = tilingData.x1Stride0 == tilingData.x2Stride0 &&
                                tilingData.x1Stride1 == tilingData.x2Stride1 &&
                                tilingData.x1Stride2 == tilingData.x2Stride2 &&
                                tilingData.x1Stride3 == tilingData.x2Stride3;
        useVector_ = contiguous;
        x1Gm_.SetGlobalBuffer((__gm__ DTYPE_X1 *)x1, outSize_);
        x2Gm_.SetGlobalBuffer((__gm__ DTYPE_X1 *)x2, outSize_);
        yGm_.SetGlobalBuffer((__gm__ uint8_t *)y, outSize_);
        pipe_.InitBuffer(inX1_, 1, FP_TILE_ELEMS * sizeof(DTYPE_X1));
        pipe_.InitBuffer(inX2_, 1, FP_TILE_ELEMS * sizeof(DTYPE_X1));
        pipe_.InitBuffer(outY_, 1, FP_TILE_ELEMS);
        pipe_.InitBuffer(calcBuf_, 2u * FP_TILE_ELEMS * sizeof(DTYPE_X1));
        pipe_.InitBuffer(maskBuf_, FP_TILE_ELEMS);
#if (ORIG_DTYPE_X1 == DT_FLOAT)
        pipe_.InitBuffer(halfBuf_, FP_TILE_ELEMS * sizeof(half));
#endif
    }

    __aicore__ inline bool UseVector() const
    {
        return useVector_;
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint32_t coreCeil = (outSize_ + blockNum - 1u) / blockNum;
        const uint32_t start = blockIdx * coreCeil;
        if (start >= outSize_) {
            return;
        }
        uint32_t coreLen = outSize_ - start;
        if (coreLen > coreCeil) {
            coreLen = coreCeil;
        }

        uint32_t done = 0;
        while (done < coreLen) {
            const uint32_t valid = (coreLen - done) > FP_TILE_ELEMS ? FP_TILE_ELEMS : (coreLen - done);
            const uint32_t offset = start + done;
            AscendC::LocalTensor<DTYPE_X1> x1Local = inX1_.AllocTensor<DTYPE_X1>();
            AscendC::LocalTensor<DTYPE_X1> x2Local = inX2_.AllocTensor<DTYPE_X1>();
            AscendC::LocalTensor<uint8_t> yLocal = outY_.AllocTensor<uint8_t>();
            AscendC::LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();
            AscendC::LocalTensor<DTYPE_X1> oneLocal = calcBuf_.Get<DTYPE_X1>();
            AscendC::LocalTensor<DTYPE_X1> boolLocal = calcBuf_.Get<DTYPE_X1>()[FP_TILE_ELEMS];

            CopyGmToUb(x1Local, x1Gm_[offset], valid);
            CopyGmToUb(x2Local, x2Gm_[offset], valid);
            event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            inX1_.EnQue(x1Local);
            inX2_.EnQue(x2Local);

            x1Local = inX1_.DeQue<DTYPE_X1>();
            x2Local = inX2_.DeQue<DTYPE_X1>();
            AscendC::Compare(maskLocal, x1Local, x2Local, AscendC::CMPMODE::EQ, valid);
            AscendC::Duplicate(oneLocal, static_cast<DTYPE_X1>(1.0f), valid);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select<DTYPE_X1, uint8_t>(boolLocal, maskLocal, oneLocal, static_cast<DTYPE_X1>(0.0f),
                                               AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, valid);
            AscendC::PipeBarrier<PIPE_V>();
#if (ORIG_DTYPE_X1 == DT_FLOAT)
            AscendC::LocalTensor<half> halfLocal = halfBuf_.Get<half>();
            AscendC::Cast(halfLocal, boolLocal, AscendC::RoundMode::CAST_RINT, valid);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(yLocal, halfLocal, AscendC::RoundMode::CAST_RINT, valid);
#else
            AscendC::Cast(yLocal, boolLocal, AscendC::RoundMode::CAST_RINT, valid);
#endif
            AscendC::PipeBarrier<PIPE_V>();
            outY_.EnQue<uint8_t>(yLocal);
            inX1_.FreeTensor(x1Local);
            inX2_.FreeTensor(x2Local);

            yLocal = outY_.DeQue<uint8_t>();
            event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
            CopyUbToGmU8(yGm_[offset], yLocal, valid);
            outY_.FreeTensor(yLocal);
            done += valid;
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inX1_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inX2_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf_;
#if (ORIG_DTYPE_X1 == DT_FLOAT)
    AscendC::TBuf<AscendC::TPosition::VECCALC> halfBuf_;
#endif
    AscendC::GlobalTensor<DTYPE_X1> x1Gm_;
    AscendC::GlobalTensor<DTYPE_X1> x2Gm_;
    AscendC::GlobalTensor<uint8_t> yGm_;
    uint32_t outSize_ = 0u;
    bool useVector_ = false;
};
#endif

extern "C" __global__ __aicore__ void tensor_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
#if (ORIG_DTYPE_X1 == DT_UINT8 || ORIG_DTYPE_X1 == DT_INT8 || ORIG_DTYPE_X1 == DT_BOOL)
    if (TILING_KEY_IS(KEY_BYTE_CONTIG)) {
        KernelTensorEqualByteVector op;
        op.Init(x1, x2, y, tiling);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(KEY_BYTE_ROW)) {
        KernelTensorEqualByteBroadcastRowVector op;
        op.Init(x1, x2, y, tiling);
        op.Process();
        return;
    }
#elif (ORIG_DTYPE_X1 == DT_INT16 || ORIG_DTYPE_X1 == DT_BF16)
    if (TILING_KEY_IS(KEY_INT16_CONTIG)) {
        KernelTensorEqualInt16Vector op;
        op.Init(x1, x2, y, tiling);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(KEY_INT16_ROW)) {
        KernelTensorEqualInt16BroadcastRowVector op;
        op.Init(x1, x2, y, tiling);
        op.Process();
        return;
    }
#elif (ORIG_DTYPE_X1 == DT_INT32)
    if (TILING_KEY_IS(KEY_INT32_CONTIG)) {
        KernelTensorEqualInt32Vector op;
        op.Init(x1, x2, y, tiling);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(KEY_INT32_ROW)) {
        KernelTensorEqualInt32BroadcastRowVector op;
        op.Init(x1, x2, y, tiling);
        op.Process();
        return;
    }
#elif (ORIG_DTYPE_X1 == DT_FLOAT || ORIG_DTYPE_X1 == DT_FLOAT16)
    if (TILING_KEY_IS(KEY_FP_CONTIG)) {
        KernelTensorEqualPureVector op;
        op.Init(x1, x2, y, tiling);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(KEY_FP_ROW)) {
        KernelTensorEqualBroadcastRowVector op;
        op.Init(x1, x2, y, tiling);
        op.Process();
        return;
    }
#endif
    if (TILING_KEY_IS(KEY_SCALAR)) {
#if (ORIG_DTYPE_X1 == DT_FLOAT || ORIG_DTYPE_X1 == DT_INT32)
        ProcessScalarFixed<4u>(x1, x2, y, workspace, tiling);
#elif (ORIG_DTYPE_X1 == DT_FLOAT16 || ORIG_DTYPE_X1 == DT_BF16 || ORIG_DTYPE_X1 == DT_INT16)
        ProcessScalarFixed<2u>(x1, x2, y, workspace, tiling);
#else
        ProcessScalarFixed<1u>(x1, x2, y, workspace, tiling);
#endif
        return;
    }
#if 0
#if (ORIG_DTYPE_X1 == DT_UINT8 || ORIG_DTYPE_X1 == DT_INT8 || ORIG_DTYPE_X1 == DT_BOOL)
    KernelTensorEqualByteVector op;
    op.Init(x1, x2, y, tiling);
    if (op.UseVector()) {
        op.Process();
        return;
    }
    GET_TILING_DATA(tilingDataForByteRow, tiling);
    if (ShouldUseBroadcastRowVector(tilingDataForByteRow)) {
        KernelTensorEqualByteBroadcastRowVector byteRowOp;
        byteRowOp.Init(x1, x2, y, tiling);
        if (byteRowOp.UseVector()) {
            byteRowOp.Process();
            return;
        }
    }
#elif (ORIG_DTYPE_X1 == DT_INT16)
    KernelTensorEqualInt16Vector op;
    op.Init(x1, x2, y, tiling);
    if (op.UseVector()) {
        op.Process();
        return;
    }
#endif
#if (ORIG_DTYPE_X1 == DT_INT32)
    GET_TILING_DATA(tilingDataForInt32Row, tiling);
    if (ShouldUseBroadcastRowVector(tilingDataForInt32Row)) {
        KernelTensorEqualInt32BroadcastRowVector int32RowOp;
        int32RowOp.Init(x1, x2, y, tiling);
        if (int32RowOp.UseVector()) {
            int32RowOp.Process();
            return;
        }
    }
#endif
#if (ORIG_DTYPE_X1 == DT_FLOAT || ORIG_DTYPE_X1 == DT_FLOAT16)
    KernelTensorEqualPureVector op;
    op.Init(x1, x2, y, tiling);
    if (op.UseVector()) {
        op.Process();
        return;
    }
    GET_TILING_DATA(tilingDataForFpRow, tiling);
    if (ShouldUseBroadcastRowVector(tilingDataForFpRow)) {
        KernelTensorEqualBroadcastRowVector rowOp;
        rowOp.Init(x1, x2, y, tiling);
        if (rowOp.UseVector()) {
            rowOp.Process();
            return;
        }
    }
#endif
#endif
#if (ORIG_DTYPE_X1 == DT_FLOAT || ORIG_DTYPE_X1 == DT_INT32)
    ProcessScalarFixed<4u>(x1, x2, y, workspace, tiling);
#elif (ORIG_DTYPE_X1 == DT_FLOAT16 || ORIG_DTYPE_X1 == DT_BF16 || ORIG_DTYPE_X1 == DT_INT16)
    ProcessScalarFixed<2u>(x1, x2, y, workspace, tiling);
#else
    ProcessScalarFixed<1u>(x1, x2, y, workspace, tiling);
#endif
}
