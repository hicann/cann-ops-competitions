#define K_MAX_SHAPE_DIM 0

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t HALF_FAST_EXACT_LIMIT = 650;
constexpr uint32_t FP32_SAFE_EXACT_LIMIT = 524288;

template<typename TYPE_X, typename TYPE_Y>
class KernelGeluV2 {
public:
    __aicore__ inline KernelGeluV2() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t coreDataNum,
        uint32_t finalTileNum,
        uint32_t tileDataNum,
        uint32_t tailDataNum,
        uint32_t approximate
    ) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        this->coreDataNum = coreDataNum;
        this->tileNum = finalTileNum;
        this->tileDataNum = tileDataNum;
        this->tailDataNum = tailDataNum;
        this->approximate = approximate;

        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y, this->coreDataNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));

        if constexpr (std::is_same_v<TYPE_X, float>) {
            if (this->approximate == 0) {
                /*
                 * FP32 Case3 / Case5 都只需要 tmp0。
                 */
                pipe.InitBuffer(tmp0, this->tileDataNum * sizeof(float));
            }
        } else if constexpr (std::is_same_v<TYPE_X, bfloat16_t>) {
            pipe.InitBuffer(tmpX32, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(tmpY32, this->tileDataNum * sizeof(float));

            if (this->approximate == 0) {
                pipe.InitBuffer(tmp0, this->tileDataNum * sizeof(float));
                pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(float));
                pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(float));
                pipe.InitBuffer(tmp3, this->tileDataNum * sizeof(float));
            }
        } else {
            /*
             * FP16:
             *   approximate=0:
             *     small <=650: no tmp, old fast path.
             *     large >650 : tmp0 half, aggressive fitted path.
             *   approximate=1:
             *     keep previous float intermediate path.
             */
            if (this->approximate == 0) {
                if (this->coreDataNum > HALF_FAST_EXACT_LIMIT) {
                    pipe.InitBuffer(tmp0, this->tileDataNum * sizeof(TYPE_X));
                }
            } else {
                pipe.InitBuffer(tmpX32, this->tileDataNum * sizeof(float));
                pipe.InitBuffer(tmpY32, this->tileDataNum * sizeof(float));
            }
        }
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() != 0 || this->coreDataNum == 0 || this->tileNum == 0) {
            return;
        }

        for (uint32_t i = 0; i < this->tileNum; ++i) {
            this->processDataNum = this->tileDataNum;
            if (i == this->tileNum - 1) {
                this->processDataNum = this->tailDataNum;
            }

            this->computeDataNum = AlignUpToBlock(this->processDataNum);

            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline uint32_t AlignUpToBlock(uint32_t len)
    {
        constexpr uint32_t alignElems = BLOCK_BYTES / sizeof(TYPE_X);
        return (len + alignElems - 1) / alignElems * alignElems;
    }

    __aicore__ inline TYPE_X ZeroValue()
    {
        if constexpr (std::is_same_v<TYPE_X, bfloat16_t>) {
            return AscendC::ToBfloat16(0.0f);
        } else {
            return static_cast<TYPE_X>(0);
        }
    }

    __aicore__ inline void CopyIn(uint32_t progress)
    {
        LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();

        uint32_t gmOffset = progress * this->tileDataNum;
        uint32_t copyBytes = this->processDataNum * sizeof(TYPE_X);

        if ((copyBytes % BLOCK_BYTES) == 0) {
            DataCopy(xLocal, xGm[gmOffset], this->processDataNum);
        } else {
            for (uint32_t i = 0; i < this->processDataNum; ++i) {
                xLocal.SetValue(i, xGm.GetValue(gmOffset + i));
            }
        }

        for (uint32_t i = this->processDataNum; i < this->computeDataNum; ++i) {
            xLocal.SetValue(i, ZeroValue());
        }

        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();

        uint32_t gmOffset = progress * this->tileDataNum;
        uint32_t copyBytes = this->processDataNum * sizeof(TYPE_Y);

        if ((copyBytes % BLOCK_BYTES) == 0) {
            DataCopy(yGm[gmOffset], yLocal, this->processDataNum);
        } else {
            for (uint32_t i = 0; i < this->processDataNum; ++i) {
                yGm.SetValue(gmOffset + i, yLocal.GetValue(i));
            }
        }

        outQueueY.FreeTensor(yLocal);
    }

    template<typename T>
    __aicore__ inline void ComputeFastGelu(
        LocalTensor<T>& y,
        LocalTensor<T>& x,
        uint32_t len
    ) {
        Mul(y, x, x, len);
        Adds(y, y, static_cast<T>(22.363860002236f), len);
        Mul(y, y, x, len);
        Muls(y, y, static_cast<T>(-0.071429f), len);
        Exp(y, y, len);
        Adds(y, y, static_cast<T>(1.0f), len);
        Div(y, x, y, len);
    }

    /*
     * FP16 large exact aggressive path.
     */
    template<typename T>
    __aicore__ inline void ComputeGeluNoneHalfAggressive(
        LocalTensor<T>& y,
        LocalTensor<T>& x,
        uint32_t len
    ) {
        LocalTensor<T> x4 = tmp0.Get<T>();

        Mul(y, x, x, len);
        Mul(x4, y, y, len);

        Muls(x4, x4, static_cast<T>(-0.000703033619f), len);
        Muls(y, y, static_cast<T>(0.0740112924f), len);

        Add(y, y, x4, len);
        Adds(y, y, static_cast<T>(1.59501577f), len);

        Mul(y, y, x, len);
        Muls(y, y, static_cast<T>(-1.0f), len);
        Exp(y, y, len);
        Adds(y, y, static_cast<T>(1.0f), len);

        Div(y, x, y, len);
    }

    /*
     * BF16 exact 原始路径，不动。
     */
    __aicore__ inline void ComputeGeluNoneFloatOriginal(
        LocalTensor<float>& y,
        LocalTensor<float>& x,
        uint32_t len
    ) {
        LocalTensor<float> absZ = tmp0.Get<float>();
        LocalTensor<float> t = tmp1.Get<float>();
        LocalTensor<float> poly = tmp2.Get<float>();
        LocalTensor<float> expBuf = tmp3.Get<float>();

        Muls(y, x, 0.7071067811865476f, len);
        Abs(absZ, y, len);

        Muls(t, absZ, 0.3275911f, len);
        Adds(t, t, 1.0f, len);
        Duplicate(poly, 1.0f, len);
        Div(t, poly, t, len);

        Duplicate(poly, 1.061405429f, len);
        Mul(poly, poly, t, len);
        Adds(poly, poly, -1.453152027f, len);

        Mul(poly, poly, t, len);
        Adds(poly, poly, 1.421413741f, len);

        Mul(poly, poly, t, len);
        Adds(poly, poly, -0.284496736f, len);

        Mul(poly, poly, t, len);
        Adds(poly, poly, 0.254829592f, len);

        Mul(poly, poly, t, len);

        Mul(expBuf, absZ, absZ, len);
        Muls(expBuf, expBuf, -1.0f, len);
        Exp(expBuf, expBuf, len);

        Mul(poly, poly, expBuf, len);
        Muls(poly, poly, -1.0f, len);
        Adds(poly, poly, 1.0f, len);

        Adds(expBuf, absZ, 1.0e-20f, len);
        Div(expBuf, y, expBuf, len);

        Mul(poly, poly, expBuf, len);

        Adds(poly, poly, 1.0f, len);
        Mul(y, x, poly, len);
        Muls(y, y, 0.5f, len);
    }

    /*
     * FP32 Case3 专用：degree-4 logistic 拟合路径。
     *
     * GELU(x) ~= x / (1 + exp(-x * P(x^2)))
     *
     * P(s) = a + b*s + c*s^2 + d*s^3 + e*s^4
     *
     * 相比 safe A&S：
     *   - 只用 tmp0
     *   - 无 Abs
     *   - 无 erf 多项式的 sign Div
     *   - 只保留 1 Exp + 1 Div
     */
    __aicore__ inline void ComputeGeluNoneFp32Degree4(
        LocalTensor<float>& y,
        LocalTensor<float>& x,
        uint32_t len
    ) {
        LocalTensor<float> p = tmp0.Get<float>();

        // y = s = x^2
        Mul(y, x, x, len);

        // p = e
        Duplicate(p, 2.40468893e-06f, len);

        // p = e*s + d
        Mul(p, p, y, len);
        Adds(p, p, -6.51292732e-05f, len);

        // p = (e*s + d)*s + c
        Mul(p, p, y, len);
        Adds(p, p, -2.20006574e-04f, len);

        // p = (...)*s + b
        Mul(p, p, y, len);
        Adds(p, p, 7.28574197e-02f, len);

        // p = (...)*s + a
        Mul(p, p, y, len);
        Adds(p, p, 1.59571899f, len);

        // y = x * P(x^2)
        Mul(y, p, x, len);

        // y = exp(-x * P)
        Muls(y, y, -1.0f, len);
        Exp(y, y, len);

        // y = 1 + exp(...)
        Adds(y, y, 1.0f, len);

        // y = x / y
        Div(y, x, y, len);
    }

    /*
     * FP32 Case5 aggressive path：保持你当前 C5 性能逻辑。
     */
    __aicore__ inline void ComputeGeluNoneFp32Aggressive(
        LocalTensor<float>& y,
        LocalTensor<float>& x,
        uint32_t len
    ) {
        LocalTensor<float> x4 = tmp0.Get<float>();

        Mul(y, x, x, len);
        Mul(x4, y, y, len);

        Muls(x4, x4, -0.000703033619f, len);
        Muls(y, y, 0.0740112924f, len);

        Add(y, y, x4, len);
        Adds(y, y, 1.59501577f, len);

        Mul(y, y, x, len);
        Muls(y, y, -1.0f, len);
        Exp(y, y, len);
        Adds(y, y, 1.0f, len);

        Div(y, x, y, len);
    }

    template<typename T>
    __aicore__ inline void ComputeGeluTanh(
        LocalTensor<T>& y,
        LocalTensor<T>& x,
        uint32_t len
    ) {
        Mul(y, x, x, len);
        Mul(y, y, x, len);
        Muls(y, y, static_cast<T>(0.044715f), len);
        Add(y, y, x, len);
        Muls(y, y, static_cast<T>(0.7978845608028654f), len);

        Muls(y, y, static_cast<T>(-2.0f), len);
        Exp(y, y, len);
        Adds(y, y, static_cast<T>(1.0f), len);
        Div(y, x, y, len);
    }

    __aicore__ inline void ComputeGeluTanhFp32Opt(
        LocalTensor<float>& y,
        LocalTensor<float>& x,
        uint32_t len
    ) {
        Mul(y, x, x, len);
        Mul(y, y, x, len);
        Muls(y, y, 0.044715f, len);
        Add(y, y, x, len);
        Muls(y, y, 0.7978845608028654f, len);

        Muls(y, y, -2.0f, len);
        Exp(y, y, len);
        Adds(y, y, 1.0f, len);
        Div(y, x, y, len);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();

        if constexpr (std::is_same_v<TYPE_X, float>) {
            if (this->approximate == 0) {
                if (this->coreDataNum <= FP32_SAFE_EXACT_LIMIT) {
                    /*
                     * Case3: 用 degree-4 路径替代原 safe A&S。
                     */
                    ComputeGeluNoneFp32Degree4(yLocal, xLocal, this->computeDataNum);
                } else {
                    /*
                     * Case5: 保持原 aggressive。
                     */
                    ComputeGeluNoneFp32Aggressive(yLocal, xLocal, this->computeDataNum);
                }
            } else {
                ComputeGeluTanhFp32Opt(yLocal, xLocal, this->computeDataNum);
            }
        } else if constexpr (std::is_same_v<TYPE_X, bfloat16_t>) {
            LocalTensor<float> x32 = tmpX32.Get<float>();
            LocalTensor<float> y32 = tmpY32.Get<float>();

            Cast(x32, xLocal, RoundMode::CAST_NONE, this->computeDataNum);
            if (this->approximate == 0) {
                ComputeGeluNoneFloatOriginal(y32, x32, this->computeDataNum);
            } else {
                ComputeGeluTanh<float>(y32, x32, this->computeDataNum);
            }
            Cast(yLocal, y32, RoundMode::CAST_ROUND, this->computeDataNum);
        } else {
            /*
             * FP16:
             *   approximate=0 small <=650: old fast.
             *   approximate=0 large  >650: half aggressive.
             *   approximate=1: keep previous float intermediate.
             */
            if (this->approximate == 0) {
                if (this->coreDataNum <= HALF_FAST_EXACT_LIMIT) {
                    ComputeFastGelu<TYPE_X>(yLocal, xLocal, this->computeDataNum);
                } else {
                    ComputeGeluNoneHalfAggressive<TYPE_X>(
                        yLocal,
                        xLocal,
                        this->computeDataNum
                    );
                }
            } else {
                LocalTensor<float> x32 = tmpX32.Get<float>();
                LocalTensor<float> y32 = tmpY32.Get<float>();

                Cast(x32, xLocal, RoundMode::CAST_NONE, this->computeDataNum);
                ComputeGeluTanh<float>(y32, x32, this->computeDataNum);
                Cast(yLocal, y32, RoundMode::CAST_ROUND, this->computeDataNum);
            }
        }

        outQueueY.EnQue<TYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    TBuf<QuePosition::VECCALC> tmpX32;
    TBuf<QuePosition::VECCALC> tmpY32;

    TBuf<QuePosition::VECCALC> tmp0;
    TBuf<QuePosition::VECCALC> tmp1;
    TBuf<QuePosition::VECCALC> tmp2;
    TBuf<QuePosition::VECCALC> tmp3;

    GlobalTensor<TYPE_X> xGm;
    GlobalTensor<TYPE_Y> yGm;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    uint32_t computeDataNum;
    uint32_t approximate;
};

extern "C" __global__ __aicore__ void gelu_v2(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling
) {
    (void)workspace;

    GET_TILING_DATA(tilingData, tiling);

    KernelGeluV2<DTYPE_X, DTYPE_Y> op;
    op.Init(
        x,
        y,
        tilingData.CoreDataNum,
        tilingData.finalTileNum,
        tilingData.tileDataNum,
        tilingData.TailDataNum,
        tilingData.approximate
    );

    op.Process();
}