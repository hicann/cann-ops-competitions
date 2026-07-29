#include "kernel_operator.h"
#include <type_traits>

using namespace AscendC;

constexpr int32_t TE_MAX_SHAPE = 4;
constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t LINEAR_ALIGN_ELEMS = 512;

template <class T> struct ComputeType { using type = T; };
template <> struct ComputeType<int8_t> { using type = half; };
template <> struct ComputeType<uint8_t> { using type = half; };
template <> struct ComputeType<int16_t> { using type = float; };
template <> struct ComputeType<int32_t> { using type = float; };
template <> struct ComputeType<bfloat16_t> { using type = float; };
template <class T> using compute_t = typename ComputeType<T>::type;

template <class T> inline constexpr bool kIsInt32 = std::is_same_v<T, int32_t>;
template <class T> inline constexpr bool kNeedCast = !std::is_same_v<T, compute_t<T>>;
template <class T> inline constexpr bool kIsBit8 =
    std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>;

__aicore__ inline uint32_t AlignUp32(uint32_t bytes)
{
    return (bytes + 31U) & ~31U;
}

template <class T>
__aicore__ inline void DuplicateInput(LocalTensor<T>& dst, T value, uint32_t count)
{
    if constexpr (kIsBit8<T>) {
        uint16_t raw = static_cast<uint16_t>(static_cast<uint8_t>(value));
        int16_t packed = static_cast<int16_t>(raw | static_cast<uint16_t>(raw << 8));
        auto dst16 = dst.template ReinterpretCast<int16_t>();
        Duplicate(dst16, packed, (count + 1U) / 2U);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        auto dstU16 = dst.template ReinterpretCast<uint16_t>();
        uint16_t raw = dstU16.GetValue(0);
        Duplicate(dstU16, raw, count);
    } else {
        Duplicate(dst, value, count);
    }
}

template <class T>
__aicore__ inline void ComputeEqualTile(
    LocalTensor<T>& x1, LocalTensor<T>& x2,
    LocalTensor<uint8_t>& y,
    LocalTensor<compute_t<T>>& x1C,
    LocalTensor<compute_t<T>>& x2C,
    LocalTensor<int32_t>& diffI32,
    LocalTensor<half>& outHalf,
    LocalTensor<uint8_t>& maskBuf,
    LocalTensor<half>& onesHalf,
    LocalTensor<half>& zerosHalf,
    uint32_t count)
{
    if constexpr (kIsInt32<T>) {
        Sub(diffI32, x1, x2, count);
        CompareScalar(maskBuf, diffI32, static_cast<int32_t>(0), CMPMODE::EQ, count);
    } else if constexpr (!kNeedCast<T>) {
        Compare(maskBuf, x1, x2, CMPMODE::EQ, count);
    } else {
        Cast(x1C, x1, RoundMode::CAST_NONE, count);
        Cast(x2C, x2, RoundMode::CAST_NONE, count);
        Compare(maskBuf, x1C, x2C, CMPMODE::EQ, count);
    }

    Select(outHalf, maskBuf, onesHalf, zerosHalf, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);

    auto yInt8 = y.template ReinterpretCast<int8_t>();
    Cast(yInt8, outHalf, RoundMode::CAST_TRUNC, count);
}

template <typename T>
class KernelTensorEqual {
public:
    __aicore__ inline void InitLinear(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                      uint32_t totalLen, uint32_t tileLen,
                                      TPipe* pipe)
    {
        total = totalLen;
        tile_len = tileLen;
        pipe_ = pipe;
        x1Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1), totalLen);
        x2Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2), totalLen);
        yGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y), totalLen);
    }

    __aicore__ inline void InitBroadcast(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                         uint32_t totalLen,
                                         uint32_t tileLen,
                                         const int32_t* x1Shape,
                                         const int32_t* x2Shape,
                                         const int32_t* outShape,
                                         TPipe* pipe)
    {
        total = totalLen;
        tile_len = tileLen;
        pipe_ = pipe;

        for (int32_t i = 0; i < TE_MAX_SHAPE; ++i) {
            x1_shape[i] = x1Shape[i];
            x2_shape[i] = x2Shape[i];
            out_shape[i] = outShape[i];
        }

        uint32_t x1Stride = 1;
        uint32_t x2Stride = 1;
        for (int32_t i = TE_MAX_SHAPE - 1; i >= 0; --i) {
            x1_stride[i] = x1Stride;
            x2_stride[i] = x2Stride;
            x1Stride *= static_cast<uint32_t>(x1_shape[i]);
            x2Stride *= static_cast<uint32_t>(x2_shape[i]);
        }
        x1Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1), x1Stride);
        x2Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2), x2Stride);
        yGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y), totalLen);

        first_active_dim = 0;
        while (first_active_dim < TE_MAX_SHAPE && out_shape[first_active_dim] == 1) {
            ++first_active_dim;
        }
        if (first_active_dim == TE_MAX_SHAPE) {
            first_active_dim = TE_MAX_SHAPE - 1;
        }
    }

    __aicore__ inline void ProcessSameShape()
    {
        InitVectorBuffers();
        uint32_t begin;
        uint32_t end;
        GetLinearCoreRange(total, begin, end);
        ProcessLinearRange(begin, end, 1, 1);
    }

    __aicore__ inline void ProcessX1Scalar()
    {
        InitVectorBuffers();
        uint32_t begin;
        uint32_t end;
        GetLinearCoreRange(total, begin, end);
        ProcessLinearRange(begin, end, 0, 1);
    }

    __aicore__ inline void ProcessX2Scalar()
    {
        InitVectorBuffers();
        uint32_t begin;
        uint32_t end;
        GetLinearCoreRange(total, begin, end);
        ProcessLinearRange(begin, end, 1, 0);
    }

    __aicore__ inline void ProcessBroadcast()
    {
        InitVectorBuffers();
        uint32_t inner = static_cast<uint32_t>(out_shape[TE_MAX_SHAPE - 1]);
        uint32_t rowCount = (inner == 0) ? 0 : total / inner;
        uint32_t rowBegin;
        uint32_t rowEnd;
        GetCoreRange(rowCount, rowBegin, rowEnd);

        for (uint32_t row = rowBegin; row < rowEnd; ++row) {
            uint32_t x1Base;
            uint32_t x2Base;
            DecodeOuter(row, x1Base, x2Base);
            uint32_t yBase = row * inner;
            for (uint32_t innerOff = 0; innerOff < inner; innerOff += tile_len) {
                uint32_t curLen = static_cast<uint32_t>(
                    (innerOff + tile_len > inner) ? (inner - innerOff) : tile_len);
                uint32_t x1Off = x1Base + ((x1_shape[TE_MAX_SHAPE - 1] == 1) ? 0 : innerOff);
                uint32_t x2Off = x2Base + ((x2_shape[TE_MAX_SHAPE - 1] == 1) ? 0 : innerOff);
                int32_t x1LastStride = (x1_shape[TE_MAX_SHAPE - 1] == 1) ? 0 : 1;
                int32_t x2LastStride = (x2_shape[TE_MAX_SHAPE - 1] == 1) ? 0 : 1;
                ProcessTile(x1Off, x1LastStride, x2Off, x2LastStride, yBase + innerOff, curLen);
            }
        }
    }

private:
    __aicore__ inline void InitVectorBuffers()
    {
        pipe_->InitBuffer(queX1, BUFFER_NUM, tile_len * sizeof(T));
        pipe_->InitBuffer(queX2, BUFFER_NUM, tile_len * sizeof(T));
        pipe_->InitBuffer(queY, BUFFER_NUM, AlignUp32(tile_len));
        if constexpr (kNeedCast<T> && !kIsInt32<T>) {
            pipe_->InitBuffer(bufX1C, tile_len * sizeof(compute_t<T>));
            pipe_->InitBuffer(bufX2C, tile_len * sizeof(compute_t<T>));
        }
        if constexpr (kIsInt32<T>) {
            pipe_->InitBuffer(bufDiff, tile_len * sizeof(int32_t));
        }
        pipe_->InitBuffer(bufOutHalf, tile_len * sizeof(half));
        pipe_->InitBuffer(bufMask, AlignUp32(tile_len));
        pipe_->InitBuffer(bufOnes, tile_len * sizeof(half));
        pipe_->InitBuffer(bufZeros, tile_len * sizeof(half));

        onesHalf = bufOnes.Get<half>();
        zerosHalf = bufZeros.Get<half>();
        Duplicate(onesHalf, static_cast<half>(1.0f), tile_len);
        Duplicate(zerosHalf, static_cast<half>(0.0f), tile_len);
    }

    __aicore__ inline void GetCoreRange(uint32_t work, uint32_t& begin, uint32_t& end)
    {
        uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
        uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
        uint32_t base = work / blockNum;
        uint32_t tail = work - base * blockNum;
        uint32_t extra = (blockIdx < tail) ? 1 : 0;
        begin = blockIdx * base + ((blockIdx < tail) ? blockIdx : tail);
        end = begin + base + extra;
    }

    __aicore__ inline void GetLinearCoreRange(uint32_t work, uint32_t& begin, uint32_t& end)
    {
        uint32_t units = (work + LINEAR_ALIGN_ELEMS - 1) / LINEAR_ALIGN_ELEMS;
        uint32_t unitBegin;
        uint32_t unitEnd;
        GetCoreRange(units, unitBegin, unitEnd);
        begin = unitBegin * LINEAR_ALIGN_ELEMS;
        end = unitEnd * LINEAR_ALIGN_ELEMS;
        if (begin > work) {
            begin = work;
        }
        if (end > work) {
            end = work;
        }
    }

    __aicore__ inline void ProcessLinearRange(uint32_t begin, uint32_t end,
                                              int32_t x1Stride, int32_t x2Stride)
    {
        for (uint32_t offset = begin; offset < end; offset += tile_len) {
            uint32_t curLen = static_cast<uint32_t>(
                (offset + tile_len > end) ? (end - offset) : tile_len);
            uint32_t x1Off = (x1Stride == 0) ? 0 : offset;
            uint32_t x2Off = (x2Stride == 0) ? 0 : offset;
            ProcessTile(x1Off, x1Stride, x2Off, x2Stride, offset, curLen);
        }
    }

    __aicore__ inline void DecodeOuter(uint32_t row, uint32_t& x1Base, uint32_t& x2Base)
    {
        x1Base = 0;
        x2Base = 0;
        uint32_t remain = row;
        for (int32_t dim = TE_MAX_SHAPE - 2; dim >= first_active_dim; --dim) {
            uint32_t outDim = static_cast<uint32_t>(out_shape[dim]);
            uint32_t coord = remain % outDim;
            remain /= outDim;
            if (x1_shape[dim] != 1) {
                x1Base += coord * x1_stride[dim];
            }
            if (x2_shape[dim] != 1) {
                x2Base += coord * x2_stride[dim];
            }
        }
    }

    __aicore__ inline void ProcessTile(uint32_t x1Off, int32_t x1LastStride,
                                       uint32_t x2Off, int32_t x2LastStride,
                                       uint32_t yOff, uint32_t curLen)
    {
        auto x1Local = queX1.AllocTensor<T>();
        auto x2Local = queX2.AllocTensor<T>();
        LoadInputTile(x1Local, x1Global, x1Off, curLen, x1LastStride);
        LoadInputTile(x2Local, x2Global, x2Off, curLen, x2LastStride);
        queX1.EnQue(x1Local);
        queX2.EnQue(x2Local);

        auto x1In = queX1.DeQue<T>();
        auto x2In = queX2.DeQue<T>();
        auto yLocal = queY.AllocTensor<uint8_t>();

        LocalTensor<compute_t<T>> x1C;
        LocalTensor<compute_t<T>> x2C;
        if constexpr (kNeedCast<T> && !kIsInt32<T>) {
            x1C = bufX1C.Get<compute_t<T>>();
            x2C = bufX2C.Get<compute_t<T>>();
        }
        LocalTensor<int32_t> diffI32;
        if constexpr (kIsInt32<T>) {
            diffI32 = bufDiff.Get<int32_t>();
        }
        auto outHalf = bufOutHalf.Get<half>();
        auto maskBuf = bufMask.Get<uint8_t>();

        ComputeEqualTile(x1In, x2In, yLocal, x1C, x2C, diffI32, outHalf, maskBuf,
                         onesHalf, zerosHalf, curLen);

        queX1.FreeTensor(x1In);
        queX2.FreeTensor(x2In);
        queY.EnQue(yLocal);

        auto yOut = queY.DeQue<uint8_t>();
        CopyOut(yGlobal, yOff, yOut, curLen);
        queY.FreeTensor(yOut);
    }

    __aicore__ inline void LoadInputTile(LocalTensor<T>& dst, GlobalTensor<T>& gm,
                                         uint32_t gmOff, uint32_t curLen, int32_t lastStride)
    {
        if (lastStride == 1) {
            uint32_t bytes = curLen * sizeof(T);
            if ((bytes & 31U) == 0) {
                DataCopy(dst, gm[gmOff], curLen);
            }
            // TODO 
            else {
                DataCopyExtParams params{1, bytes, 0, 0, 0};
                DataCopyPadExtParams<T> pad{false, 0, 0, 0};
                DataCopyPad(dst, gm[gmOff], params, pad);
            }
            return;
        }

        if constexpr (std::is_same_v<T, bfloat16_t>) {
            DataCopyExtParams params{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> pad{false, 0, 0, 0};
            DataCopyPad(dst, gm[gmOff], params, pad);
            event_t evt = static_cast<event_t>(0);
            SetFlag<HardEvent::MTE2_S>(evt);
            WaitFlag<HardEvent::MTE2_S>(evt);
            auto dstU16 = dst.template ReinterpretCast<uint16_t>();
            uint16_t raw = dstU16.GetValue(0);
            SetFlag<HardEvent::S_V>(evt);
            WaitFlag<HardEvent::S_V>(evt);
            Duplicate(dstU16, raw, curLen);
        } else if constexpr (kIsBit8<T>) {
            T value = gm.GetValue(gmOff);
            uint16_t raw = static_cast<uint16_t>(static_cast<uint8_t>(value));
            int16_t packed = static_cast<int16_t>(raw | static_cast<uint16_t>(raw << 8));
            auto dst16 = dst.template ReinterpretCast<int16_t>();
            Duplicate(dst16, packed, (curLen + 1U) / 2U);
        } else {
            T value = gm.GetValue(gmOff);
            Duplicate(dst, value, curLen);
        }
    }

    __aicore__ inline void CopyOut(GlobalTensor<uint8_t>& dst, uint32_t off,
                                   LocalTensor<uint8_t>& src, uint32_t count)
    {
        if ((count & 31U) == 0) {
            DataCopy(dst[off], src, count);
        } else {
            DataCopyExtParams params{1, count, 0, 0, 0};
            DataCopyPad(dst[off], src, params);
        }
    }

    uint32_t total;
    uint32_t tile_len;
    int32_t x1_shape[TE_MAX_SHAPE];
    int32_t x2_shape[TE_MAX_SHAPE];
    int32_t out_shape[TE_MAX_SHAPE];
    uint32_t x1_stride[TE_MAX_SHAPE];
    uint32_t x2_stride[TE_MAX_SHAPE];
    int32_t first_active_dim;

    TPipe* pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> queX1;
    TQue<QuePosition::VECIN, BUFFER_NUM> queX2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> queY;
    TBuf<TPosition::VECCALC> bufX1C;
    TBuf<TPosition::VECCALC> bufX2C;
    TBuf<TPosition::VECCALC> bufDiff;
    TBuf<TPosition::VECCALC> bufOutHalf;
    TBuf<TPosition::VECCALC> bufMask;
    TBuf<TPosition::VECCALC> bufOnes;
    TBuf<TPosition::VECCALC> bufZeros;
    LocalTensor<half> onesHalf;
    LocalTensor<half> zerosHalf;
    GlobalTensor<T> x1Global;
    GlobalTensor<T> x2Global;
    GlobalTensor<uint8_t> yGlobal;
};

__aicore__ inline void LoadLinearTiling(GM_ADDR tiling, uint32_t& totalLen, uint32_t& tileLen)
{
    uint64_t packed = *((const __gm__ uint64_t*)tiling);
    totalLen = static_cast<uint32_t>(packed);
    tileLen = static_cast<uint32_t>(packed >> 32);
}

extern "C" __global__ __aicore__ void tensor_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                                   GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    KernelTensorEqual<DTYPE_X1> op;

    if (TILING_KEY_IS(0)) {
        uint32_t totalLen;
        uint32_t tileLen;
        LoadLinearTiling(tiling, totalLen, tileLen);
        op.InitLinear(x1, x2, y, totalLen, tileLen, &pipe);
        op.ProcessSameShape();
    } else if (TILING_KEY_IS(1)) {
        uint32_t totalLen;
        uint32_t tileLen;
        LoadLinearTiling(tiling, totalLen, tileLen);
        op.InitLinear(x1, x2, y, totalLen, tileLen, &pipe);
        op.ProcessX1Scalar();
    } else if (TILING_KEY_IS(2)) {
        uint32_t totalLen;
        uint32_t tileLen;
        LoadLinearTiling(tiling, totalLen, tileLen);
        op.InitLinear(x1, x2, y, totalLen, tileLen, &pipe);
        op.ProcessX2Scalar();
    } else if (TILING_KEY_IS(3)) {
        GET_TILING_DATA(tiling_data, tiling);
        op.InitBroadcast(x1, x2, y, tiling_data.total_len, tiling_data.tile_len,
                         tiling_data.x1_shape, tiling_data.x2_shape,
                         tiling_data.out_shape, &pipe);
        op.ProcessBroadcast();
    }
}
