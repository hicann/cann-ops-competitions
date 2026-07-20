#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;      // 昇腾双buffer技术
constexpr int32_t APPROX_NONE = 0;
constexpr int32_t APPROX_TANH = 1;

// ---------- 数学常量 ----------
// "none" (erf) 路径
constexpr float CONST_REV_SQRT2      = 0.70710678118f;   // 1/sqrt(2)
constexpr float CONST_HALF           = 0.5f;
constexpr float CONST_ONE            = 1.0f;

// "tanh" HIGH_PRECISION 路径
constexpr float CONST_NEG_2SQRT_2_PI = -1.5957691f;      // -2 * sqrt(2/pi)
constexpr float CONST_044715         = 0.044715f;

template<typename T>
class KernelGeluV2 {
public:
    __aicore__ inline KernelGeluV2() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                int32_t CoreDataNum, int32_t finalTileNum,
                                int32_t tileDataNum, int32_t TailDataNum,
                                int32_t Approximate, TPipe* pipeIn) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        this->tileDataNum = tileDataNum;
        this->coreDataNum = CoreDataNum;
        this->tileNum     = finalTileNum;
        this->tailDataNum = TailDataNum;
        this->approximate = Approximate;

        xGm.SetGlobalBuffer((__gm__ T*)x, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ T*)y, this->coreDataNum);

        pipe = pipeIn;
        pipe->InitBuffer(inQueueX,  BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(T));

        // float16 / bfloat16 -> 需要 fp32 工作缓冲
        // float32           -> 直接在原 tensor 上计算，不需要 bufXf
        if constexpr (!std::is_same_v<T, float>) {
            pipe->InitBuffer(bufXf, this->tileDataNum * sizeof(float));
            pipe->InitBuffer(bufYf, this->tileDataNum * sizeof(float));
        }
        pipe->InitBuffer(bufTmp1, this->tileDataNum * sizeof(float));
        pipe->InitBuffer(bufTmp2, this->tileDataNum * sizeof(float));
        pipe->InitBuffer(bufTmp3, this->tileDataNum * sizeof(float));
    }

    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;     // 手动考虑了双buff
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++) {
            if (i == this->tileNum - 1) {
                this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

// def gelu_compute_erf(input_x):
//     """
//     supported dtype is float32
//     res = x/(1+exp(((((((a1*x^2+a2)*x^2+a3)*x^2+a4)*x^2+a5)*x^2+a6)*x^2+a7)*x))
//     """
//     if not after_v200():
//         # v200/v100 donot suported infnan, must do vmaxs to clip 
//         input_x = tbe.vmaxs(input_x, tvm.const(-13.25, "float32"))
//     x1 = tbe.vmins(input_x, tvm.const(5.75, "float32"))
//     x_pow = tbe.vmul(x1, x1)
//     y = tbe.vmuls(x_pow, tvm.const(-0.3512339572e-8, "float32"))
//     y = tbe.vadds(y, tvm.const(0.2645266170e-6, "float32"))
//     y = tbe.vmul(y, x_pow)
//     y = tbe.vadds(y, tvm.const(-0.7929488134e-5, "float32"))
//     y = tbe.vmul(y, x_pow)
//     y = tbe.vadds(y, tvm.const(0.1106123840e-3, "float32"))
//     y = tbe.vmul(y, x_pow)
//     y = tbe.vadds(y, tvm.const(0.6518995814e-4, "float32"))
//     y = tbe.vmul(y, x_pow)
//     y = tbe.vadds(y, tvm.const(-0.7266616915e-1, "float32"))
//     y = tbe.vmul(y, x_pow)
//     y = tbe.vadds(y, tvm.const(-0.1595769883e1, "float32"))
//     y = tbe.vmul(y, x1)
//     y = tbe.vexp(y)
//     y = tbe.vadds(y, tvm.const(1.0, "float32"))
//     res = tbe.vdiv(input_x, y)

//     return res  

    __aicore__ inline void ComputeNone(LocalTensor<float>& xf, LocalTensor<float>& yf) {
        auto y = bufTmp1.Get<float>();   // x1
        auto tmp2 = bufTmp2.Get<float>();   // x1^2
        //auto y =    bufTmp3.Get<float>();  
           // exp(...) 的结果复用 xpow 的 buffer
        const int32_t n = this->processDataNum;

        //gelu_compute_erf
        // x1 = clip(x, -13.25, 5.75)
        Maxs(tmp2, xf,   -13.25f, n);
        Mins(yf, tmp2, 5.75f,   n);       //tmp1 = x1
        Mul (tmp2, yf, yf,    n);       // tmp2 = x1^2

        // 6阶 Horner on tmp2: tmp1 作累加器 (覆盖 x1)
        Muls(y, tmp2, -0.3512339572e-8f, n);
        Adds(y, y, 0.2645266170e-6f,  n);
        Mul (y, y, tmp2,              n);
        Adds(y, y, -0.7929488134e-5f, n);
        Mul (y, y, tmp2,              n);
        Adds(y, y, 0.1106123840e-3f,  n);
        Mul (y, y, tmp2,              n);
        Adds(y, y, 0.6518995814e-4f,  n);
        Mul (y, y, tmp2,              n);
        Adds(y, y, -0.7266616915e-1f, n);
        Mul (y, y, tmp2,              n);
        Adds(y, y, -0.1595769883e1f,  n);
        // 末尾 * x: x1 已被 Horner 覆盖, 用 xf 替代
        // x>5.75 区间: poly·xf << 0, exp -> 0, 1+exp -> 1, res = xf / 1 = xf (等价于 Python)
        // x<-13.25 区间: poly·xf >> 0, exp -> inf, 1+exp -> inf, res = xf / inf = 0 (等价于 Python)
        Mul (y, y, yf,                n);
        Exp (y, y,                    n);
        Adds(y, y, CONST_ONE,         n);  // 1 + exp(...)
        Div (yf, xf, y, n);                   // res = xf / (1 + exp(...))
    
        // Adds(tmp2, tmp2, CONST_ONE,  n);        // + 1
        // Muls(tmp2, tmp2, CONST_HALF, n);        // * 0.5  -> Phi(x)
        // Mul (yf,   xf,   tmp2,       n);
    
    }

    // ============================================================
    // approximate = "tanh"  (HIGH_PRECISION 化简式)
    //   gelu(x) = x / (1 + exp(-1.5957691 * (x + 0.044715 * x^3)))
    // ============================================================
    __aicore__ inline void ComputeTanhV2(LocalTensor<float>& xf, LocalTensor<float>& yf) {
        auto tmp1 = bufTmp1.Get<float>();
        auto tmp2 = bufTmp2.Get<float>();

        Mul (tmp1, xf,   xf,                   this->processDataNum);   // x^2
        Mul (tmp1, tmp1, xf,                   this->processDataNum);   // x^3
        Muls(tmp1, tmp1, CONST_044715,         this->processDataNum);   // 0.044715 * x^3
        Add (tmp1, tmp1, xf,                   this->processDataNum);   // x + 0.044715 * x^3
        Muls(tmp1, tmp1, CONST_NEG_2SQRT_2_PI, this->processDataNum);   // * (-1.5957691)
        Exp (tmp2, tmp1,                       this->processDataNum);
        Adds(tmp2, tmp2, CONST_ONE,            this->processDataNum);   // 1 + exp(...)
        Div (yf,   xf,   tmp2,                 this->processDataNum);   // x / (1 + exp(...))
    }

    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

        if constexpr (std::is_same_v<T, float>) {
            // float 分支：直接在 fp32 上算，零 Cast
            if (this->approximate == APPROX_NONE) {
                ComputeNone(xLocal, yLocal);
            } else {
                ComputeTanhV2(xLocal, yLocal);
            }
        } else {
            // half / bfloat16_t 分支：升 fp32 -> 计算 -> 回写
            auto xf = bufXf.Get<float>();
            auto yf = bufYf.Get<float>();
            Cast(xf, xLocal, RoundMode::CAST_NONE, this->processDataNum);

            if (this->approximate == APPROX_NONE) {
                ComputeNone(xf, yf);   // 原地复用 xf 作为输出
            } else {
                ComputeTanhV2(xf, yf);
            }

            Cast(yLocal, yf, RoundMode::CAST_ROUND, this->processDataNum);   // half 与 bf16 通用
        }

        outQueueY.EnQue<T>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> bufXf,bufYf;     // 仅 half/bf16 使用
    TBuf<QuePosition::VECCALC> bufTmp1;
    TBuf<QuePosition::VECCALC> bufTmp2;
    TBuf<QuePosition::VECCALC> bufTmp3;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    int32_t coreDataNum;
    int32_t tileNum;
    int32_t tileDataNum;
    int32_t tailDataNum;
    int32_t processDataNum;
    int32_t approximate;
};

extern "C" __global__ __aicore__ void gelu_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    KernelGeluV2<DTYPE_X> op;
    op.Init(x, y,
            tiling_data.CoreDataNum,
            tiling_data.finalTileNum,
            tiling_data.tileDataNum,
            tiling_data.TailDataNum,
            tiling_data.Approximate,
            &pipe);
    op.Process();
}
