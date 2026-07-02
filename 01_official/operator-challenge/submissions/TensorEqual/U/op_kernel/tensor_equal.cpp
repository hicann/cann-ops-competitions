#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include <cstdint>
#include <type_traits>

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template<typename TYPE_X1, typename TYPE_X2, typename TYPE_Y>
class KernelTensorEqual {
    using T = TYPE_X1;
    // CANN 8.5 generated kernels may use raw signed/unsigned char for int8/uint8.
    // Detect by signedness + sizeof to avoid wrong constexpr dispatch.
    static constexpr bool IS_INT8 = (sizeof(T) == 1) && std::is_signed<T>::value && !std::is_same_v<T, bool>;
    static constexpr bool IS_UINT8 = (sizeof(T) == 1) && std::is_unsigned<T>::value && !std::is_same_v<T, bool>;
    static constexpr bool IS_INT8_LIKE = IS_INT8 || IS_UINT8;
    static constexpr bool IS_HALF = std::is_same_v<T, half>;
    static constexpr bool IS_BF16 = std::is_same_v<T, bfloat16_t>;
    static constexpr bool IS_LOW_PRECISION = IS_HALF || IS_BF16;
    // V11: keep all non-broadcast dtypes on UB/vector path.
    // BF16 uses raw 16-bit storage equality: raw bf16 bits -> int16 diff -> half compare.
    static constexpr bool USE_SCALAR_PATH = false;
    static constexpr bool USE_BF16_STORAGE_VECTOR_PATH = IS_BF16;

public:
    __aicore__ inline KernelTensorEqual() {}

    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR y,
        uint8_t ALIGN_NUM,
        uint32_t block_size,
        uint32_t core_size,
        uint32_t core_remain)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->validLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->blockLength = this->validLength;
        this->tileLength = block_size;
        this->blockLength = this->blockLength +
            (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferLength = this->blockLength;

        Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1 + startPointer, bufferLength);
        Gm_x2.SetGlobalBuffer((__gm__ TYPE_X2*)x2 + startPointer, bufferLength);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, bufferLength);
        if constexpr (IS_LOW_PRECISION) {
            Gm_x1_u16.SetGlobalBuffer((__gm__ uint16_t*)x1 + startPointer, bufferLength);
            Gm_x2_u16.SetGlobalBuffer((__gm__ uint16_t*)x2 + startPointer, bufferLength);
        }

        if constexpr (USE_SCALAR_PATH) {
            return;
        }

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        pipe.InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(TYPE_X1));
        pipe.InitBuffer(Q_x2, BUFFER_NUM, this->tileLength * sizeof(TYPE_X2));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
        pipe.InitBuffer(B_bits, this->tileLength * sizeof(uint8_t));
        pipe.InitBuffer(B_result, this->tileLength * sizeof(half));
        pipe.InitBuffer(B_zero, this->tileLength * sizeof(half));

        this->zero = B_zero.Get<half>();
        Duplicate(this->zero, half(0), this->tileLength);

        if constexpr (IS_INT8_LIKE) {
            pipe.InitBuffer(B_x1, this->tileLength * sizeof(half));
            pipe.InitBuffer(B_x2, this->tileLength * sizeof(half));
        } else if constexpr (std::is_same_v<T, int16_t>) {
            pipe.InitBuffer(B_x1, this->tileLength * sizeof(half));
        } else if constexpr (USE_BF16_STORAGE_VECTOR_PATH) {
            pipe.InitBuffer(B_x1, this->tileLength * sizeof(half));
        } else if constexpr (std::is_same_v<T, int32_t>) {
            pipe.InitBuffer(B_x1, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_x2, this->tileLength * sizeof(float));
            auto floatZero = B_x2.Get<float>();
            Duplicate(floatZero, float(0), this->tileLength);
        } else if constexpr (std::is_same_v<T, float>) {
            pipe.InitBuffer(B_x2, this->tileLength * sizeof(float));
            auto floatZero = B_x2.Get<float>();
            Duplicate(floatZero, float(0), this->tileLength);
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (USE_SCALAR_PATH) {
            ProcessScalar();
            return;
        }

        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount - 1; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }

        uint32_t length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length);
        CopyOut(loopCount - 1, (length + 31) / 32 * 32);
    }

private:
    __aicore__ inline void ProcessScalar()
    {
        for (uint32_t i = 0; i < this->validLength; ++i) {
            if constexpr (IS_LOW_PRECISION) {
                // Do not evaluate half/bfloat16 scalar operators on AICore.
                // For equality, compare their 16-bit storage representation directly.
                Gm_y(i) = static_cast<TYPE_Y>(Gm_x1_u16(i) == Gm_x2_u16(i));
            } else {
                Gm_y(i) = static_cast<TYPE_Y>(Gm_x1(i) == Gm_x2(i));
            }
        }
    }

    __aicore__ inline void CopyIn(int32_t progress, uint32_t length)
    {
        LocalTensor<TYPE_X1> x1 = Q_x1.AllocTensor<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.AllocTensor<TYPE_X2>();
        DataCopy(x1, Gm_x1[progress * this->tileLength], length);
        DataCopy(x2, Gm_x2[progress * this->tileLength], length);
        Q_x1.EnQue(x1);
        Q_x2.EnQue(x2);
    }

    __aicore__ inline void Compute(int32_t progress, uint32_t length)
    {
        (void)progress;
        LocalTensor<TYPE_X1> x1 = Q_x1.DeQue<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.DeQue<TYPE_X2>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();
        auto bits = B_bits.Get<uint8_t>();
        auto result = B_result.Get<half>();
        auto inty = y.template ReinterpretCast<uint8_t>();

        if constexpr (IS_INT8_LIKE) {
            auto halfX1 = B_x1.Get<half>();
            auto halfX2 = B_x2.Get<half>();
            Cast(halfX1, x1, RoundMode::CAST_NONE, length);
            Cast(halfX2, x2, RoundMode::CAST_NONE, length);
            Sub(halfX1, halfX1, halfX2, length);
            Compare(bits, halfX1, zero, CMPMODE::NE, length);
        } else if constexpr (USE_BF16_STORAGE_VECTOR_PATH) {
            auto rawX1 = x1.template ReinterpretCast<int16_t>();
            auto rawX2 = x2.template ReinterpretCast<int16_t>();
            auto halfDiff = B_x1.Get<half>();
            Sub(rawX1, rawX1, rawX2, length);
            Cast(halfDiff, rawX1, RoundMode::CAST_NONE, length);
            Compare(bits, halfDiff, zero, CMPMODE::NE, length);
        } else {
            Sub(x1, x1, x2, length);
            if constexpr (std::is_same_v<T, int32_t>) {
                auto val = B_x1.Get<float>();
                auto floatZero = B_x2.Get<float>();
                Cast(val, x1, RoundMode::CAST_NONE, length);
                Compare(bits, val, floatZero, CMPMODE::NE, length);
            } else if constexpr (std::is_same_v<T, float>) {
                auto floatZero = B_x2.Get<float>();
                Compare(bits, x1, floatZero, CMPMODE::NE, length);
            } else if constexpr (std::is_same_v<T, int16_t>) {
                auto halfDiff = B_x1.Get<half>();
                Cast(halfDiff, x1, RoundMode::CAST_NONE, length);
                Compare(bits, halfDiff, zero, CMPMODE::NE, length);
            } else {
                Compare(bits, x1, zero, CMPMODE::NE, length);
            }
        }

        Select(result, bits, zero, half(1), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Cast(inty, result, RoundMode::CAST_ROUND, length);

        Q_x1.FreeTensor(x1);
        Q_x2.FreeTensor(x2);
        Q_y.EnQue<TYPE_Y>(y);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t length)
    {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        DataCopy(Gm_y[progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x1, Q_x2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<QuePosition::VECCALC> B_result, B_zero, B_bits;
    TBuf<QuePosition::VECCALC> B_x1, B_x2;
    LocalTensor<half> zero;
    GlobalTensor<TYPE_X1> Gm_x1;
    GlobalTensor<TYPE_X2> Gm_x2;
    GlobalTensor<TYPE_Y> Gm_y;
    GlobalTensor<uint8_t> Gm_y_u8;
    GlobalTensor<uint16_t> Gm_x1_u16;
    GlobalTensor<uint16_t> Gm_x2_u16;
    uint32_t validLength;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

template<typename TYPE_X1, typename TYPE_X2, typename TYPE_Y>
class KernelTensorEqual_Broadcast {
    using T = TYPE_X1;
    static constexpr uint32_t SHAPE_STRIDE = 5;
    static constexpr uint32_t MAX_DIM = 4;
    static constexpr bool IS_FLOAT = std::is_same_v<T, float>;
    static constexpr bool IS_HALF = std::is_same_v<T, half>;
    static constexpr bool IS_BF16 = std::is_same_v<T, bfloat16_t>;
    static constexpr bool IS_INT16 = std::is_same_v<T, int16_t>;
    static constexpr bool IS_INT32 = std::is_same_v<T, int32_t>;
    // CANN 8.5 generated kernels may use raw signed/unsigned char for int8/uint8.
    // Detect by signedness + sizeof to avoid wrong constexpr dispatch.
    static constexpr bool IS_INT8 = (sizeof(T) == 1) && std::is_signed<T>::value && !std::is_same_v<T, bool>;
    static constexpr bool IS_UINT8 = (sizeof(T) == 1) && std::is_unsigned<T>::value && !std::is_same_v<T, bool>;
    static constexpr bool IS_INT8_LIKE = IS_INT8 || IS_UINT8;
    static constexpr bool IS_LOW_PRECISION = IS_HALF || IS_BF16;
    // Keep the old safe baseline for general broadcast: int8/uint8 generic vector
    // broadcast is avoided because small fragmented chunks can regress badly.
    static constexpr bool USE_GENERAL_BROADCAST_VECTOR = IS_FLOAT || IS_HALF || IS_BF16 || IS_INT16 || IS_INT32;
    // Targeted scalar-vector path handles last-axis int8/uint8 safely via int16->half.
    static constexpr bool USE_LAST_AXIS_SCALAR_VECTOR = IS_FLOAT || IS_HALF || IS_BF16 || IS_INT16 || IS_INT32 || IS_INT8_LIKE;
    static constexpr bool NEED_BROADCAST_UB = USE_GENERAL_BROADCAST_VECTOR || USE_LAST_AXIS_SCALAR_VECTOR;
    static constexpr bool USE_BF16_STORAGE_VECTOR_PATH = IS_BF16;
    static constexpr uint32_t LAST_AXIS_VECTOR_MIN_LEN = 32;
    // Guarded int8/uint8 generic tail-contiguous vectorization.
    // Avoid tiny fragmented vector chunks, but still vectorize large contiguous inner blocks.
    static constexpr uint32_t INT8_GENERAL_VECTOR_MIN_INNER = 128;

public:
    __aicore__ inline KernelTensorEqual_Broadcast() {}

    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR y,
        uint8_t ALIGN_NUM,
        uint32_t block_size,
        uint32_t core_size,
        uint32_t core_remain)
    {
        (void)ALIGN_NUM;
        this->tileLength = block_size;
        this->startOffset = core_size * GetBlockIdx();
        this->validLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->totalLength = core_size * GetBlockNum() + core_remain;

        Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1, this->totalLength);
        Gm_x2.SetGlobalBuffer((__gm__ TYPE_X2*)x2, this->totalLength);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y, this->totalLength);
        if constexpr (IS_LOW_PRECISION) {
            Gm_x1_u16.SetGlobalBuffer((__gm__ uint16_t*)x1, this->totalLength);
            Gm_x2_u16.SetGlobalBuffer((__gm__ uint16_t*)x2, this->totalLength);
        }
        if constexpr (NEED_BROADCAST_UB) {
            Gm_y_u8.SetGlobalBuffer((__gm__ uint8_t*)y, this->totalLength);
            pipe.InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(TYPE_X1));
            pipe.InitBuffer(Q_x2, BUFFER_NUM, this->tileLength * sizeof(TYPE_X2));
            // bool output is one byte. Use uint8_t local/global view in vector path.
            pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(uint8_t));
            pipe.InitBuffer(B_bits, this->tileLength * sizeof(uint8_t));
            pipe.InitBuffer(B_result, this->tileLength * sizeof(half));
            pipe.InitBuffer(B_zero, this->tileLength * sizeof(half));

            this->zero = B_zero.Get<half>();
            Duplicate(this->zero, half(0), this->tileLength);

            if constexpr (IS_INT8_LIKE) {
                pipe.InitBuffer(B_x1, this->tileLength * sizeof(half));
                pipe.InitBuffer(B_x2, this->tileLength * sizeof(half));
            } else if constexpr (IS_BF16 || IS_INT32) {
                pipe.InitBuffer(B_x1, this->tileLength * sizeof(float));
                pipe.InitBuffer(B_x2, this->tileLength * sizeof(float));
                auto floatZero = B_x2.Get<float>();
                Duplicate(floatZero, float(0), this->tileLength);
            } else if constexpr (IS_INT16) {
                pipe.InitBuffer(B_x1, this->tileLength * sizeof(half));
            } else if constexpr (IS_FLOAT) {
                pipe.InitBuffer(B_x1, this->tileLength * sizeof(float));
                auto floatZero = B_x1.Get<float>();
                Duplicate(floatZero, float(0), this->tileLength);
            }
        }
    }

    __aicore__ inline bool IsEqualByIndex(uint32_t indexX1, uint32_t indexX2)
    {
        if constexpr (IS_LOW_PRECISION) {
            // Avoid half/bfloat16 scalar comparison on AICore.
            return Gm_x1_u16(indexX1) == Gm_x2_u16(indexX2);
        } else {
            return Gm_x1(indexX1) == Gm_x2(indexX2);
        }
    }

    __aicore__ inline bool FindTailContiguousSplit(
        uint32_t shapeInf[2 * SHAPE_STRIDE],
        uint32_t rank,
        uint32_t& splitAxis)
    {
        // General fast-path detector:
        // Find the earliest axis k such that axes [k, rank) have identical x1/x2 dims.
        // Prefix axes [0, k) may be equal or broadcast-compatible.
        //
        // Examples covered:
        //   [1, C, H, W] vs [N, C, H, W] -> k = 1, inner = C*H*W
        //   [N, 1, H, W] vs [N, C, H, W] -> k = 2, inner = H*W
        //   [N, C, 1, W] vs [N, C, H, W] -> k = 3, inner = W
        //   [1, 1, H, W] vs [N, C, H, W] -> k = 2, inner = H*W
        if (rank == 0 || rank > MAX_DIM) {
            return false;
        }

        for (uint32_t k = 0; k < rank; ++k) {
            bool suffixEqual = true;
            for (uint32_t axis = k; axis < rank; ++axis) {
                if (shapeInf[axis + 1] != shapeInf[SHAPE_STRIDE + axis + 1]) {
                    suffixEqual = false;
                    break;
                }
            }

            if (!suffixEqual) {
                continue;
            }

            bool prefixBroadcastCompatible = true;
            for (uint32_t axis = 0; axis < k; ++axis) {
                uint32_t d1 = shapeInf[axis + 1];
                uint32_t d2 = shapeInf[SHAPE_STRIDE + axis + 1];
                if (!(d1 == d2 || d1 == 1 || d2 == 1)) {
                    prefixBroadcastCompatible = false;
                    break;
                }
            }

            if (prefixBroadcastCompatible) {
                splitAxis = k;
                return true;
            }
        }

        return false;
    }

    __aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b)
    {
        return a < b ? a : b;
    }

    __aicore__ inline uint32_t GetInnerSize(uint32_t shapeInf[2 * SHAPE_STRIDE], uint32_t rank, uint32_t splitAxis)
    {
        uint32_t innerSize = 1;
        for (uint32_t axis = splitAxis; axis < rank; ++axis) {
            innerSize *= shapeInf[axis + 1];
        }
        return innerSize;
    }

    __aicore__ inline void ComputePrefixBaseOffsets(
        uint32_t shapeInf[2 * SHAPE_STRIDE],
        uint32_t rank,
        uint32_t splitAxis,
        uint32_t prefixLinear,
        uint32_t innerSize,
        uint32_t& baseX1,
        uint32_t& baseX2)
    {
        uint32_t coord[MAX_DIM] = {0, 0, 0, 0};
        uint32_t remain = prefixLinear;

        for (int32_t axis = static_cast<int32_t>(splitAxis) - 1; axis >= 0; --axis) {
            uint32_t d1 = shapeInf[axis + 1];
            uint32_t d2 = shapeInf[SHAPE_STRIDE + axis + 1];
            uint32_t outDim = d1 > d2 ? d1 : d2;
            coord[axis] = remain % outDim;
            remain = remain / outDim;
        }

        uint32_t off1 = 0;
        uint32_t off2 = 0;
        for (uint32_t axis = 0; axis < splitAxis; ++axis) {
            uint32_t d1 = shapeInf[axis + 1];
            uint32_t d2 = shapeInf[SHAPE_STRIDE + axis + 1];
            uint32_t idx1 = (d1 <= 1) ? 0 : coord[axis];
            uint32_t idx2 = (d2 <= 1) ? 0 : coord[axis];
            off1 = off1 * d1 + idx1;
            off2 = off2 * d2 + idx2;
        }

        baseX1 = off1 * innerSize;
        baseX2 = off2 * innerSize;
    }

    __aicore__ inline void CopyInVector(uint32_t offsetX1, uint32_t offsetX2, uint32_t length)
    {
        LocalTensor<TYPE_X1> x1 = Q_x1.AllocTensor<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.AllocTensor<TYPE_X2>();

        DataCopyExtParams cp1{1, static_cast<uint32_t>(length * sizeof(TYPE_X1)), 0, 0, 0};
        DataCopyPadExtParams<TYPE_X1> pad1{false, 0, 0, 0};
        DataCopyPad(x1, Gm_x1[offsetX1], cp1, pad1);

        DataCopyExtParams cp2{1, static_cast<uint32_t>(length * sizeof(TYPE_X2)), 0, 0, 0};
        DataCopyPadExtParams<TYPE_X2> pad2{false, 0, 0, 0};
        DataCopyPad(x2, Gm_x2[offsetX2], cp2, pad2);

        Q_x1.EnQue(x1);
        Q_x2.EnQue(x2);
    }

    __aicore__ inline void ComputeVector(uint32_t length)
    {
        LocalTensor<TYPE_X1> x1 = Q_x1.DeQue<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.DeQue<TYPE_X2>();
        LocalTensor<uint8_t> y = Q_y.AllocTensor<uint8_t>();

        auto bits = B_bits.Get<uint8_t>();
        auto result = B_result.Get<half>();

        if constexpr (IS_INT8_LIKE) {
            auto halfX1 = B_x1.Get<half>();
            auto halfX2 = B_x2.Get<half>();
            Cast(halfX1, x1, RoundMode::CAST_NONE, length);
            Cast(halfX2, x2, RoundMode::CAST_NONE, length);
            Sub(halfX1, halfX1, halfX2, length);
            Compare(bits, halfX1, zero, CMPMODE::NE, length);
        } else if constexpr (USE_BF16_STORAGE_VECTOR_PATH) {
            auto rawX1 = x1.template ReinterpretCast<int16_t>();
            auto rawX2 = x2.template ReinterpretCast<int16_t>();
            auto halfDiff = B_x1.Get<half>();
            Sub(rawX1, rawX1, rawX2, length);
            Cast(halfDiff, rawX1, RoundMode::CAST_NONE, length);
            Compare(bits, halfDiff, zero, CMPMODE::NE, length);
        } else {
            Sub(x1, x1, x2, length);
            if constexpr (IS_INT32) {
                auto val = B_x1.Get<float>();
                auto floatZero = B_x2.Get<float>();
                Cast(val, x1, RoundMode::CAST_NONE, length);
                Compare(bits, val, floatZero, CMPMODE::NE, length);
            } else if constexpr (IS_INT16) {
                auto halfDiff = B_x1.Get<half>();
                Cast(halfDiff, x1, RoundMode::CAST_NONE, length);
                Compare(bits, halfDiff, zero, CMPMODE::NE, length);
            } else if constexpr (IS_FLOAT) {
                auto floatZero = B_x1.Get<float>();
                Compare(bits, x1, floatZero, CMPMODE::NE, length);
            } else {
                // half path
                Compare(bits, x1, zero, CMPMODE::NE, length);
            }
        }

        // bits == 1 means x1 != x2. Select returns 0 for not-equal, 1 for equal.
        Select(result, bits, zero, half(1), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Cast(y, result, RoundMode::CAST_ROUND, length);

        Q_x1.FreeTensor(x1);
        Q_x2.FreeTensor(x2);
        Q_y.EnQue<uint8_t>(y);
    }

    __aicore__ inline void CopyOutVector(uint32_t outIndex, uint32_t length)
    {
        LocalTensor<uint8_t> y = Q_y.DeQue<uint8_t>();
        DataCopyExtParams cp{1, static_cast<uint32_t>(length * sizeof(uint8_t)), 0, 0, 0};
        DataCopyPad(Gm_y_u8[outIndex], y, cp);
        Q_y.FreeTensor(y);
    }

    __aicore__ inline void ProcessScalarRange(
        uint32_t outIndex,
        uint32_t offsetX1,
        uint32_t offsetX2,
        uint32_t length)
    {
        for (uint32_t i = 0; i < length; ++i) {
            Gm_y(outIndex + i) = static_cast<TYPE_Y>(IsEqualByIndex(offsetX1 + i, offsetX2 + i));
        }
    }

    __aicore__ inline void ProcessVectorChunk(uint32_t outIndex, uint32_t offsetX1, uint32_t offsetX2, uint32_t length)
    {
        CopyInVector(offsetX1, offsetX2, length);
        ComputeVector(length);
        CopyOutVector(outIndex, length);
    }

    __aicore__ inline bool HasLastAxisScalarVector(uint32_t shapeInf[2 * SHAPE_STRIDE], uint32_t rank, bool& scalarX1)
    {
        if (rank == 0) {
            return false;
        }
        uint32_t last = rank - 1;
        uint32_t d1 = shapeInf[last + 1];
        uint32_t d2 = shapeInf[SHAPE_STRIDE + last + 1];
        uint32_t rowLen = d1 > d2 ? d1 : d2;
        if (rowLen < LAST_AXIS_VECTOR_MIN_LEN) {
            return false;
        }
        if (d1 == 1 && d2 > 1) {
            scalarX1 = true;
            return true;
        }
        if (d2 == 1 && d1 > 1) {
            scalarX1 = false;
            return true;
        }
        return false;
    }

    __aicore__ inline uint32_t GetLastAxisRowLen(uint32_t shapeInf[2 * SHAPE_STRIDE], uint32_t rank)
    {
        uint32_t d1 = shapeInf[rank];
        uint32_t d2 = shapeInf[SHAPE_STRIDE + rank];
        return d1 > d2 ? d1 : d2;
    }

    __aicore__ inline void ComputeLastAxisBaseOffsets(
        uint32_t shapeInf[2 * SHAPE_STRIDE],
        uint32_t rank,
        uint32_t rowLinear,
        uint32_t& baseX1,
        uint32_t& baseX2)
    {
        uint32_t coord[MAX_DIM] = {0, 0, 0, 0};
        uint32_t remain = rowLinear;
        uint32_t prefixRank = rank - 1;

        for (int32_t axis = static_cast<int32_t>(prefixRank) - 1; axis >= 0; --axis) {
            uint32_t d1 = shapeInf[axis + 1];
            uint32_t d2 = shapeInf[SHAPE_STRIDE + axis + 1];
            uint32_t outDim = d1 > d2 ? d1 : d2;
            coord[axis] = remain % outDim;
            remain = remain / outDim;
        }

        uint32_t off1 = 0;
        uint32_t off2 = 0;
        for (uint32_t axis = 0; axis < prefixRank; ++axis) {
            uint32_t d1 = shapeInf[axis + 1];
            uint32_t d2 = shapeInf[SHAPE_STRIDE + axis + 1];
            uint32_t idx1 = (d1 <= 1) ? 0 : coord[axis];
            uint32_t idx2 = (d2 <= 1) ? 0 : coord[axis];
            off1 = off1 * d1 + idx1;
            off2 = off2 * d2 + idx2;
        }

        uint32_t lastDimX1 = shapeInf[rank];
        uint32_t lastDimX2 = shapeInf[SHAPE_STRIDE + rank];
        baseX1 = off1 * lastDimX1;
        baseX2 = off2 * lastDimX2;
    }

    __aicore__ inline void ComputeBf16RawDiffToBool(
        LocalTensor<uint8_t>& y,
        LocalTensor<int16_t>& rawX1,
        LocalTensor<int16_t>& rawX2,
        uint32_t length)
    {
        auto bits = B_bits.Get<uint8_t>();
        auto result = B_result.Get<half>();
        auto halfDiff = B_x1.Get<half>();
        Sub(rawX1, rawX1, rawX2, length);
        Cast(halfDiff, rawX1, RoundMode::CAST_NONE, length);
        Compare(bits, halfDiff, zero, CMPMODE::NE, length);
        Select(result, bits, zero, half(1), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Cast(y, result, RoundMode::CAST_ROUND, length);
    }

    __aicore__ inline void ProcessInt8ScalarVectorChunk(
        uint32_t outIndex,
        uint32_t offsetVector,
        uint32_t offsetScalar,
        bool scalarX1,
        uint32_t length)
    {
        LocalTensor<uint8_t> y = Q_y.AllocTensor<uint8_t>();
        auto bits = B_bits.Get<uint8_t>();
        auto halfX1 = B_x1.Get<half>();
        auto halfX2 = B_x2.Get<half>();
        auto result = B_result.Get<half>();
        auto scalarI16 = B_result.Get<int16_t>();

        if (scalarX1) {
            LocalTensor<TYPE_X2> x2 = Q_x2.AllocTensor<TYPE_X2>();
            DataCopyExtParams cp{1, static_cast<uint32_t>(length * sizeof(TYPE_X2)), 0, 0, 0};
            DataCopyPadExtParams<TYPE_X2> pad{false, 0, 0, 0};
            DataCopyPad(x2, Gm_x2[offsetVector], cp, pad);
            Q_x2.EnQue(x2);
            x2 = Q_x2.DeQue<TYPE_X2>();
            int16_t scalar = static_cast<int16_t>(Gm_x1(offsetScalar));
            Duplicate(scalarI16, scalar, length);
            Cast(halfX1, scalarI16, RoundMode::CAST_NONE, length);
            Cast(halfX2, x2, RoundMode::CAST_NONE, length);
            Q_x2.FreeTensor(x2);
        } else {
            LocalTensor<TYPE_X1> x1 = Q_x1.AllocTensor<TYPE_X1>();
            DataCopyExtParams cp{1, static_cast<uint32_t>(length * sizeof(TYPE_X1)), 0, 0, 0};
            DataCopyPadExtParams<TYPE_X1> pad{false, 0, 0, 0};
            DataCopyPad(x1, Gm_x1[offsetVector], cp, pad);
            Q_x1.EnQue(x1);
            x1 = Q_x1.DeQue<TYPE_X1>();
            int16_t scalar = static_cast<int16_t>(Gm_x2(offsetScalar));
            Cast(halfX1, x1, RoundMode::CAST_NONE, length);
            Duplicate(scalarI16, scalar, length);
            Cast(halfX2, scalarI16, RoundMode::CAST_NONE, length);
            Q_x1.FreeTensor(x1);
        }

        Sub(halfX1, halfX1, halfX2, length);
        Compare(bits, halfX1, zero, CMPMODE::NE, length);
        Select(result, bits, zero, half(1), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Cast(y, result, RoundMode::CAST_ROUND, length);
        Q_y.EnQue<uint8_t>(y);
        CopyOutVector(outIndex, length);
    }

    __aicore__ inline void ProcessBf16ScalarVectorChunk(
        uint32_t outIndex,
        uint32_t offsetVector,
        uint32_t offsetScalar,
        bool scalarX1,
        uint32_t length)
    {
        LocalTensor<uint8_t> y = Q_y.AllocTensor<uint8_t>();
        if (scalarX1) {
            LocalTensor<TYPE_X2> x2 = Q_x2.AllocTensor<TYPE_X2>();
            auto rawX1 = B_x2.Get<int16_t>();
            uint16_t scalar = Gm_x1_u16(offsetScalar);
            Duplicate(rawX1.template ReinterpretCast<uint16_t>(), scalar, length);
            DataCopyExtParams cp{1, static_cast<uint32_t>(length * sizeof(TYPE_X2)), 0, 0, 0};
            DataCopyPadExtParams<TYPE_X2> pad{false, 0, 0, 0};
            DataCopyPad(x2, Gm_x2[offsetVector], cp, pad);
            Q_x2.EnQue(x2);
            x2 = Q_x2.DeQue<TYPE_X2>();
            auto rawX2 = x2.template ReinterpretCast<int16_t>();
            ComputeBf16RawDiffToBool(y, rawX1, rawX2, length);
            Q_x2.FreeTensor(x2);
        } else {
            LocalTensor<TYPE_X1> x1 = Q_x1.AllocTensor<TYPE_X1>();
            auto rawX2 = B_x2.Get<int16_t>();
            uint16_t scalar = Gm_x2_u16(offsetScalar);
            DataCopyExtParams cp{1, static_cast<uint32_t>(length * sizeof(TYPE_X1)), 0, 0, 0};
            DataCopyPadExtParams<TYPE_X1> pad{false, 0, 0, 0};
            DataCopyPad(x1, Gm_x1[offsetVector], cp, pad);
            Duplicate(rawX2.template ReinterpretCast<uint16_t>(), scalar, length);
            Q_x1.EnQue(x1);
            x1 = Q_x1.DeQue<TYPE_X1>();
            auto rawX1 = x1.template ReinterpretCast<int16_t>();
            ComputeBf16RawDiffToBool(y, rawX1, rawX2, length);
            Q_x1.FreeTensor(x1);
        }
        Q_y.EnQue<uint8_t>(y);
        CopyOutVector(outIndex, length);
    }

    __aicore__ inline void CopyInScalarVectorGeneric(
        uint32_t offsetVector,
        uint32_t offsetScalar,
        bool scalarX1,
        uint32_t length)
    {
        LocalTensor<TYPE_X1> x1 = Q_x1.AllocTensor<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.AllocTensor<TYPE_X2>();
        if (scalarX1) {
            Duplicate(x1, static_cast<TYPE_X1>(Gm_x1(offsetScalar)), length);
            DataCopyExtParams cp{1, static_cast<uint32_t>(length * sizeof(TYPE_X2)), 0, 0, 0};
            DataCopyPadExtParams<TYPE_X2> pad{false, 0, 0, 0};
            DataCopyPad(x2, Gm_x2[offsetVector], cp, pad);
        } else {
            DataCopyExtParams cp{1, static_cast<uint32_t>(length * sizeof(TYPE_X1)), 0, 0, 0};
            DataCopyPadExtParams<TYPE_X1> pad{false, 0, 0, 0};
            DataCopyPad(x1, Gm_x1[offsetVector], cp, pad);
            Duplicate(x2, static_cast<TYPE_X2>(Gm_x2(offsetScalar)), length);
        }
        Q_x1.EnQue(x1);
        Q_x2.EnQue(x2);
    }

    __aicore__ inline void ProcessScalarVectorChunk(
        uint32_t outIndex,
        uint32_t offsetVector,
        uint32_t offsetScalar,
        bool scalarX1,
        uint32_t length)
    {
        if constexpr (IS_INT8_LIKE) {
            ProcessInt8ScalarVectorChunk(outIndex, offsetVector, offsetScalar, scalarX1, length);
        } else if constexpr (USE_BF16_STORAGE_VECTOR_PATH) {
            ProcessBf16ScalarVectorChunk(outIndex, offsetVector, offsetScalar, scalarX1, length);
        } else {
            CopyInScalarVectorGeneric(offsetVector, offsetScalar, scalarX1, length);
            ComputeVector(length);
            CopyOutVector(outIndex, length);
        }
    }

    __aicore__ inline void ProcessLastAxisScalarVectorBroadcast(
        uint32_t shapeInf[2 * SHAPE_STRIDE],
        uint32_t rank,
        bool scalarX1)
    {
        uint32_t rowLen = GetLastAxisRowLen(shapeInf, rank);
        uint32_t outIndex = this->startOffset;
        uint32_t endIndex = this->startOffset + this->validLength;

        while (outIndex < endIndex) {
            uint32_t rowLinear = outIndex / rowLen;
            uint32_t innerOffset = outIndex - rowLinear * rowLen;
            uint32_t baseX1 = 0;
            uint32_t baseX2 = 0;
            ComputeLastAxisBaseOffsets(shapeInf, rank, rowLinear, baseX1, baseX2);

            uint32_t segmentLen = MinU32(rowLen - innerOffset, endIndex - outIndex);
            while (segmentLen > 0) {
                uint32_t chunkLen = MinU32(segmentLen, this->tileLength);
                uint32_t offsetVector = scalarX1 ? baseX2 + innerOffset : baseX1 + innerOffset;
                uint32_t offsetScalar = scalarX1 ? baseX1 : baseX2;
                ProcessScalarVectorChunk(outIndex, offsetVector, offsetScalar, scalarX1, chunkLen);
                outIndex += chunkLen;
                innerOffset += chunkLen;
                segmentLen -= chunkLen;
            }
        }
    }

    __aicore__ inline void ProcessTailContiguousBroadcastVector(
        uint32_t shapeInf[2 * SHAPE_STRIDE],
        uint32_t rank,
        uint32_t splitAxis)
    {
        uint32_t innerSize = GetInnerSize(shapeInf, rank, splitAxis);
        uint32_t outIndex = this->startOffset;
        uint32_t endIndex = this->startOffset + this->validLength;

        while (outIndex < endIndex) {
            uint32_t prefixLinear = outIndex / innerSize;
            uint32_t innerOffset = outIndex - prefixLinear * innerSize;

            uint32_t baseX1 = 0;
            uint32_t baseX2 = 0;
            ComputePrefixBaseOffsets(shapeInf, rank, splitAxis, prefixLinear, innerSize, baseX1, baseX2);

            uint32_t segmentLen = MinU32(innerSize - innerOffset, endIndex - outIndex);
            uint32_t offsetX1 = baseX1 + innerOffset;
            uint32_t offsetX2 = baseX2 + innerOffset;

            while (segmentLen > 0) {
                uint32_t chunkLen = MinU32(segmentLen, this->tileLength);
                ProcessVectorChunk(outIndex, offsetX1, offsetX2, chunkLen);
                outIndex += chunkLen;
                offsetX1 += chunkLen;
                offsetX2 += chunkLen;
                segmentLen -= chunkLen;
            }
        }
    }

    __aicore__ inline void ProcessTailContiguousBroadcastScalar(
        uint32_t shapeInf[2 * SHAPE_STRIDE],
        uint32_t rank,
        uint32_t splitAxis)
    {
        uint32_t innerSize = GetInnerSize(shapeInf, rank, splitAxis);
        uint32_t outIndex = this->startOffset;
        uint32_t endIndex = this->startOffset + this->validLength;

        while (outIndex < endIndex) {
            uint32_t prefixLinear = outIndex / innerSize;
            uint32_t innerOffset = outIndex - prefixLinear * innerSize;

            uint32_t baseX1 = 0;
            uint32_t baseX2 = 0;
            ComputePrefixBaseOffsets(shapeInf, rank, splitAxis, prefixLinear, innerSize, baseX1, baseX2);

            uint32_t segmentLen = MinU32(innerSize - innerOffset, endIndex - outIndex);
            ProcessScalarRange(outIndex, baseX1 + innerOffset, baseX2 + innerOffset, segmentLen);
            outIndex += segmentLen;
        }
    }

    __aicore__ inline void ProcessTailContiguousBroadcast(
        uint32_t shapeInf[2 * SHAPE_STRIDE],
        uint32_t rank,
        uint32_t splitAxis)
    {
        uint32_t innerSize = GetInnerSize(shapeInf, rank, splitAxis);

        if constexpr (IS_INT8_LIKE) {
            // V12: int8/uint8 generic broadcast vectorization is enabled only
            // when the contiguous suffix is large enough. This keeps correctness
            // and avoids the V9-style regression where length-1/small chunks
            // spend more time in vector setup than in useful SIMD work.
            if (innerSize >= INT8_GENERAL_VECTOR_MIN_INNER) {
                ProcessTailContiguousBroadcastVector(shapeInf, rank, splitAxis);
            } else {
                ProcessTailContiguousBroadcastScalar(shapeInf, rank, splitAxis);
            }
        } else if constexpr (USE_GENERAL_BROADCAST_VECTOR) {
            ProcessTailContiguousBroadcastVector(shapeInf, rank, splitAxis);
        } else {
            ProcessTailContiguousBroadcastScalar(shapeInf, rank, splitAxis);
        }
    }

    __aicore__ inline void Process(uint32_t shapeInf[2 * SHAPE_STRIDE])
    {
        uint32_t rank = shapeInf[0];
        if (shapeInf[SHAPE_STRIDE] > rank) {
            rank = shapeInf[SHAPE_STRIDE];
        }
        if (rank > MAX_DIM) {
            return;
        }

        bool lastAxisScalarX1 = false;
        if constexpr (USE_LAST_AXIS_SCALAR_VECTOR) {
            if (HasLastAxisScalarVector(shapeInf, rank, lastAxisScalarX1)) {
                ProcessLastAxisScalarVectorBroadcast(shapeInf, rank, lastAxisScalarX1);
                return;
            }
        }

        uint32_t splitAxis = 0;
        if (FindTailContiguousSplit(shapeInf, rank, splitAxis)) {
            ProcessTailContiguousBroadcast(shapeInf, rank, splitAxis);
            return;
        }

        uint32_t outDim[MAX_DIM] = {1, 1, 1, 1};
        for (uint32_t axis = 0; axis < rank; ++axis) {
            uint32_t dimX1 = shapeInf[axis + 1];
            uint32_t dimX2 = shapeInf[SHAPE_STRIDE + axis + 1];
            outDim[axis] = dimX1 > dimX2 ? dimX1 : dimX2;
        }

        for (uint32_t localIdx = 0; localIdx < this->validLength; ++localIdx) {
            uint32_t outIndex = this->startOffset + localIdx;
            uint32_t remain = outIndex;
            uint32_t coord[MAX_DIM] = {0, 0, 0, 0};

            for (int32_t axis = static_cast<int32_t>(rank) - 1; axis >= 0; --axis) {
                uint32_t dim = outDim[axis];
                coord[axis] = remain % dim;
                remain = remain / dim;
            }

            uint32_t offsetX1 = 0;
            uint32_t offsetX2 = 0;
            for (uint32_t axis = 0; axis < rank; ++axis) {
                uint32_t dimX1 = shapeInf[axis + 1];
                uint32_t dimX2 = shapeInf[SHAPE_STRIDE + axis + 1];
                uint32_t indexX1 = (dimX1 <= 1) ? 0 : coord[axis];
                uint32_t indexX2 = (dimX2 <= 1) ? 0 : coord[axis];
                offsetX1 = offsetX1 * dimX1 + indexX1;
                offsetX2 = offsetX2 * dimX2 + indexX2;
            }

            Gm_y(outIndex) = static_cast<TYPE_Y>(IsEqualByIndex(offsetX1, offsetX2));
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x1, Q_x2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<QuePosition::VECCALC> B_result, B_zero, B_bits;
    TBuf<QuePosition::VECCALC> B_x1, B_x2;
    LocalTensor<half> zero;
    GlobalTensor<TYPE_X1> Gm_x1;
    GlobalTensor<TYPE_X2> Gm_x2;
    GlobalTensor<TYPE_Y> Gm_y;
    GlobalTensor<uint8_t> Gm_y_u8;
    GlobalTensor<uint16_t> Gm_x1_u16;
    GlobalTensor<uint16_t> Gm_x2_u16;
    uint32_t startOffset;
    uint32_t validLength;
    uint32_t totalLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void tensor_equal(
    GM_ADDR x1,
    GM_ADDR x2,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    if (tiling_data.boardCast) {
        KernelTensorEqual_Broadcast<DTYPE_X1, DTYPE_X2, DTYPE_Y> op;
        op.Init(
            x1,
            x2,
            y,
            tiling_data.ALIGN_NUM,
            tiling_data.block_size,
            tiling_data.core_size,
            tiling_data.core_remain);
        op.Process(tiling_data.shapeInf);
    } else {
        KernelTensorEqual<DTYPE_X1, DTYPE_X2, DTYPE_Y> op;
        op.Init(
            x1,
            x2,
            y,
            tiling_data.ALIGN_NUM,
            tiling_data.block_size,
            tiling_data.core_size,
            tiling_data.core_remain);
        op.Process();
    }
}
