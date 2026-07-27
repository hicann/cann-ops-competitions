#include "kernel_operator.h"
using namespace AscendC;
constexpr int block_len = 4096;
constexpr int stride = block_len * 40;
extern "C" __global__ __aicore__ void is_nan(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr(std::is_same_v<DTYPE_X, bfloat16_t>)
    {
        GlobalTensor<DTYPE_X> xGm;
        GlobalTensor<uint8_t> yGm;
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x);
        yGm.SetGlobalBuffer((__gm__ uint8_t*)y);

        LocalTensor<DTYPE_X> xLocal0, xLocal1;
        LocalTensor<float> txLocal0, txLocal1;
        LocalTensor<uint8_t> res0, res1;
        LocalTensor<half> ZERO, ONE, res_half0, res_half1;
        LocalTensor<int16_t> xLocal0_int16, xLocal1_int16;
        
        AscendC::LocalMemAllocator allocator;
        xLocal0 = allocator.Alloc<AscendC::TPosition::VECCALC, DTYPE_X>(block_len);
        xLocal1 = allocator.Alloc<AscendC::TPosition::VECCALC, DTYPE_X>(block_len);
        txLocal0 = allocator.Alloc<AscendC::TPosition::VECCALC, float>(block_len);
        txLocal1 = allocator.Alloc<AscendC::TPosition::VECCALC, float>(block_len);
        res0 = allocator.Alloc<AscendC::TPosition::VECCALC, uint8_t>(block_len);
        res1 = allocator.Alloc<AscendC::TPosition::VECCALC, uint8_t>(block_len);
        ZERO = allocator.Alloc<AscendC::TPosition::VECCALC, half>(block_len);
        ONE = allocator.Alloc<AscendC::TPosition::VECCALC, half>(block_len);
        res_half0 = allocator.Alloc<AscendC::TPosition::VECCALC, half>(block_len);
        res_half1 = allocator.Alloc<AscendC::TPosition::VECCALC, half>(block_len);
        xLocal0_int16 = allocator.Alloc<AscendC::TPosition::VECCALC, int16_t>(block_len);
        xLocal1_int16 = allocator.Alloc<AscendC::TPosition::VECCALC, int16_t>(block_len);


        Duplicate(ZERO, (half)0, block_len);
        Duplicate(ONE, (half)1, block_len);
        // TODO: user kernel impl
        int size = tiling_data.size, count = tiling_data.count;
        int start = GetBlockIdx() * block_len;
        if (start + (count - 1) * stride >= size) count --;
            SetFlag<AscendC::HardEvent::V_MTE2>(0);
            SetFlag<AscendC::HardEvent::V_MTE2>(1);
            SetFlag<AscendC::HardEvent::MTE3_V>(0);
            SetFlag<AscendC::HardEvent::MTE3_V>(1);
        for (int i = 0; i < count; i += 2)
        {
            WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            DataCopy(xLocal0, xGm[start], {1, block_len * sizeof(DTYPE_X) / 32, 0, 0});
            WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            SetFlag<AscendC::HardEvent::MTE2_V>(0);
            WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            Cast(txLocal0, xLocal0, AscendC::RoundMode::CAST_NONE, block_len);
            Abs(txLocal0, txLocal0, block_len);
            Adds(txLocal0, txLocal0, (float)100, block_len);
            Cast(xLocal0_int16, txLocal0, AscendC::RoundMode::CAST_RINT, block_len);
            Cast(txLocal0, xLocal0_int16, AscendC::RoundMode::CAST_NONE, block_len);
            CompareScalar(res0, txLocal0, (float)0, AscendC::CMPMODE::NE, block_len);
            Select(res_half0, res0, ZERO, ONE, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, block_len);
            Cast(res0, res_half0, AscendC::RoundMode::CAST_NONE, block_len);
            SetFlag<AscendC::HardEvent::V_MTE3>(0);
            WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            SetFlag<AscendC::HardEvent::V_MTE2>(0);
            DataCopy(yGm[start], res0, {1, block_len * sizeof(uint8_t) / 32, 0, 0});
            SetFlag<AscendC::HardEvent::MTE3_V>(0);
            start += stride;

            if (i + 1 == count) break;

            WaitFlag<AscendC::HardEvent::V_MTE2>(1);
            DataCopy(xLocal1, xGm[start], {1, block_len * sizeof(DTYPE_X) / 32, 0, 0});
            WaitFlag<AscendC::HardEvent::MTE3_V>(1);
            SetFlag<AscendC::HardEvent::MTE2_V>(1);
            WaitFlag<AscendC::HardEvent::MTE2_V>(1);
            Cast(txLocal1, xLocal1, AscendC::RoundMode::CAST_NONE, block_len);
            Abs(txLocal1, txLocal1, block_len);
            Adds(txLocal1, txLocal1, (float)100, block_len);
            Cast(xLocal1_int16, txLocal1, AscendC::RoundMode::CAST_RINT, block_len);
            Cast(txLocal1, xLocal1_int16, AscendC::RoundMode::CAST_NONE, block_len);
            CompareScalar(res1, txLocal1, (float)0, AscendC::CMPMODE::NE, block_len);
            Select(res_half1, res1, ZERO, ONE, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, block_len);
            Cast(res1, res_half1, AscendC::RoundMode::CAST_NONE, block_len);
            SetFlag<AscendC::HardEvent::V_MTE3>(1);
            WaitFlag<AscendC::HardEvent::V_MTE3>(1);
            SetFlag<AscendC::HardEvent::V_MTE2>(1);
            DataCopy(yGm[start], res1, {1, block_len * sizeof(uint8_t) / 32, 0, 0});
            SetFlag<AscendC::HardEvent::MTE3_V>(1);
            start += stride;
        }
            WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            WaitFlag<AscendC::HardEvent::V_MTE2>(1);
            WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            WaitFlag<AscendC::HardEvent::MTE3_V>(1);
    }
    if constexpr(std::is_same_v<DTYPE_X, float> || std::is_same_v<DTYPE_X, half>)
    {
        GlobalTensor<DTYPE_X> xGm;
        GlobalTensor<uint8_t> yGm;
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x);
        yGm.SetGlobalBuffer((__gm__ uint8_t*)y);

        // LocalTensor<DTYPE_X> xLocal0(TPosition::VECCALC, 0, block_len * sizeof(DTYPE_X)),
        //                      xLocal1(TPosition::VECCALC, 32768, block_len * sizeof(DTYPE_X));
        // LocalTensor<DTYPE_X> tmpLocal0(TPosition::VECCALC, 65536 + 256, block_len * sizeof(DTYPE_X)),
        //                      tmpLocal1(TPosition::VECCALC, 65536 + 32768 + 256, block_len * sizeof(DTYPE_X));
        // LocalTensor<uint8_t> res0(TPosition::VECCALC, 0 + block_len * sizeof(DTYPE_X) + 256, block_len * sizeof(uint8_t)),
        //                      res1(TPosition::VECCALC, 32768 + block_len * sizeof(DTYPE_X) + 256, block_len * sizeof(uint8_t));
        // LocalTensor<half> ZERO(TPosition::VECCALC, 131072 + 32768 - block_len * sizeof(half) - 256, block_len * sizeof(half)),
        //                   ONE(TPosition::VECCALC, 196352 - block_len * sizeof(half), block_len * sizeof(half));
        // LocalTensor<half> res_half0(TPosition::VECCALC, 65536 + 512 + block_len * sizeof(DTYPE_X), block_len * sizeof(half)),
        //                   res_half1(TPosition::VECCALC, 65536 + 32768 + 512 + block_len * sizeof(DTYPE_X), block_len * sizeof(half));
        // LocalTensor<int16_t> tmp0_int16(TPosition::VECCALC, 131072, block_len * sizeof(int16_t)),
        //                      tmp1_int16(TPosition::VECCALC, 131072 + 32768, block_len * sizeof(int16_t));
        LocalTensor<DTYPE_X> xLocal0,
                             xLocal1,
                             xLocal2,
                             xLocal3;
        LocalTensor<DTYPE_X> tmpLocal0,
                             tmpLocal1;
        LocalTensor<uint8_t> res0,
                             res1;
        LocalTensor<half> ZERO,
                          ONE;
        LocalTensor<half> res_half0,
                          res_half1;
        LocalTensor<int16_t> tmp0_int16,
                             tmp1_int16;
        AscendC::LocalMemAllocator allocator;
        xLocal0 = allocator.Alloc<AscendC::TPosition::VECCALC, DTYPE_X>(block_len);
        xLocal1 = allocator.Alloc<AscendC::TPosition::VECCALC, DTYPE_X>(block_len);
        xLocal2 = allocator.Alloc<AscendC::TPosition::VECCALC, DTYPE_X>(block_len);
        xLocal3 = allocator.Alloc<AscendC::TPosition::VECCALC, DTYPE_X>(block_len);
        tmpLocal0 = allocator.Alloc<AscendC::TPosition::VECCALC, DTYPE_X>(block_len);
        tmpLocal1 = allocator.Alloc<AscendC::TPosition::VECCALC, DTYPE_X>(block_len);
        res0 = allocator.Alloc<AscendC::TPosition::VECCALC, uint8_t>(block_len);
        res1 = allocator.Alloc<AscendC::TPosition::VECCALC, uint8_t>(block_len);
        ZERO = allocator.Alloc<AscendC::TPosition::VECCALC, half>(block_len);
        ONE = allocator.Alloc<AscendC::TPosition::VECCALC, half>(block_len);
        res_half0 = allocator.Alloc<AscendC::TPosition::VECCALC, half>(block_len);
        res_half1 = allocator.Alloc<AscendC::TPosition::VECCALC, half>(block_len);
        tmp0_int16 = allocator.Alloc<AscendC::TPosition::VECCALC, int16_t>(block_len);
        tmp1_int16 = allocator.Alloc<AscendC::TPosition::VECCALC, int16_t>(block_len);

        LocalTensor<half> tmp0_half = tmp0_int16.template ReinterpretCast<half>(),
                          tmp1_half = tmp1_int16.template ReinterpretCast<half>();

        Duplicate(ZERO, (half)0, block_len);
        Duplicate(ONE, (half)1, block_len);
        // TODO: user kernel impl
        int size = tiling_data.size, count = tiling_data.count;
        int start = GetBlockIdx() * block_len;
        if (start + (count - 1) * stride >= size) count --;
            SetFlag<AscendC::HardEvent::V_MTE2>(0);
            SetFlag<AscendC::HardEvent::V_MTE2>(1);
            SetFlag<AscendC::HardEvent::V_MTE2>(2);
            SetFlag<AscendC::HardEvent::V_MTE2>(3);
            SetFlag<AscendC::HardEvent::MTE3_V>(0);
            SetFlag<AscendC::HardEvent::MTE3_V>(1);
        for (int i = 0; i < count; i += 4)
        {
            WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            DataCopy(xLocal0, xGm[start], {1, block_len * sizeof(DTYPE_X) / 32, 0, 0});
            SetFlag<AscendC::HardEvent::MTE2_V>(0);
            WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            Maxs(tmpLocal0, xLocal0, (DTYPE_X)100, block_len);
            SetFlag<AscendC::HardEvent::V_MTE2>(0);
            Cast(tmp0_int16, tmpLocal0, AscendC::RoundMode::CAST_RINT, block_len);
            CompareScalar(res0, tmp0_half, (half)0, AscendC::CMPMODE::NE, block_len);
            Select(res_half0, res0, ZERO, ONE, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, block_len);
            WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            Cast(res0, res_half0, AscendC::RoundMode::CAST_NONE, block_len);
            SetFlag<AscendC::HardEvent::V_MTE3>(0);
            WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            DataCopy(yGm[start], res0, {1, block_len * sizeof(uint8_t) / 32, 0, 0});
            SetFlag<AscendC::HardEvent::MTE3_V>(0);
            start += stride;

            if (i + 1 == count) break;

            WaitFlag<AscendC::HardEvent::V_MTE2>(1);
            DataCopy(xLocal1, xGm[start], {1, block_len * sizeof(DTYPE_X) / 32, 0, 0});
            SetFlag<AscendC::HardEvent::MTE2_V>(1);
            WaitFlag<AscendC::HardEvent::MTE2_V>(1);
            Maxs(tmpLocal1, xLocal1, (DTYPE_X)100, block_len);
            SetFlag<AscendC::HardEvent::V_MTE2>(1);
            Cast(tmp1_int16, tmpLocal1, AscendC::RoundMode::CAST_RINT, block_len);
            CompareScalar(res1, tmp1_half, (half)0, AscendC::CMPMODE::NE, block_len);
            Select(res_half1, res1, ZERO, ONE, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, block_len);
            WaitFlag<AscendC::HardEvent::MTE3_V>(1);
            Cast(res1, res_half1, AscendC::RoundMode::CAST_NONE, block_len);
            SetFlag<AscendC::HardEvent::V_MTE3>(1);
            WaitFlag<AscendC::HardEvent::V_MTE3>(1);
            DataCopy(yGm[start], res1, {1, block_len * sizeof(uint8_t) / 32, 0, 0});
            SetFlag<AscendC::HardEvent::MTE3_V>(1);
            start += stride;
            
            if (i + 2 == count) break;

            WaitFlag<AscendC::HardEvent::V_MTE2>(2);
            DataCopy(xLocal2, xGm[start], {1, block_len * sizeof(DTYPE_X) / 32, 0, 0});
            SetFlag<AscendC::HardEvent::MTE2_V>(0);
            WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            Maxs(tmpLocal0, xLocal2, (DTYPE_X)100, block_len);
            SetFlag<AscendC::HardEvent::V_MTE2>(2);
            Cast(tmp0_int16, tmpLocal0, AscendC::RoundMode::CAST_RINT, block_len);
            CompareScalar(res0, tmp0_half, (half)0, AscendC::CMPMODE::NE, block_len);
            Select(res_half0, res0, ZERO, ONE, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, block_len);
            WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            Cast(res0, res_half0, AscendC::RoundMode::CAST_NONE, block_len);
            SetFlag<AscendC::HardEvent::V_MTE3>(0);
            WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            DataCopy(yGm[start], res0, {1, block_len * sizeof(uint8_t) / 32, 0, 0});
            SetFlag<AscendC::HardEvent::MTE3_V>(0);
            start += stride;

            if (i + 3 == count) break;

            WaitFlag<AscendC::HardEvent::V_MTE2>(3);
            DataCopy(xLocal3, xGm[start], {1, block_len * sizeof(DTYPE_X) / 32, 0, 0});
            SetFlag<AscendC::HardEvent::MTE2_V>(1);
            WaitFlag<AscendC::HardEvent::MTE2_V>(1);
            Maxs(tmpLocal1, xLocal3, (DTYPE_X)100, block_len);
            SetFlag<AscendC::HardEvent::V_MTE2>(3);
            Cast(tmp1_int16, tmpLocal1, AscendC::RoundMode::CAST_RINT, block_len);
            CompareScalar(res1, tmp1_half, (half)0, AscendC::CMPMODE::NE, block_len);
            Select(res_half1, res1, ZERO, ONE, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, block_len);
            WaitFlag<AscendC::HardEvent::MTE3_V>(1);
            Cast(res1, res_half1, AscendC::RoundMode::CAST_NONE, block_len);
            SetFlag<AscendC::HardEvent::V_MTE3>(1);
            WaitFlag<AscendC::HardEvent::V_MTE3>(1);
            DataCopy(yGm[start], res1, {1, block_len * sizeof(uint8_t) / 32, 0, 0});
            SetFlag<AscendC::HardEvent::MTE3_V>(1);
            start += stride;
        }
            WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            WaitFlag<AscendC::HardEvent::V_MTE2>(1);
            WaitFlag<AscendC::HardEvent::V_MTE2>(2);
            WaitFlag<AscendC::HardEvent::V_MTE2>(3);
            WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            WaitFlag<AscendC::HardEvent::MTE3_V>(1);
    }
}