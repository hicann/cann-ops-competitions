// GeluV2 Kernel - Optimized v7 (Single-Core)
// 性能优化:
//   1. 单核专用: 无多核分配开销
//   2. 输入输出双缓冲(depth=2)，DMA/Compute完全overlap
//   3. 尾部tile精确处理: 最后一个tile仅处理实际剩余数据
//   4. tanh: GELU(x)=x/(1+exp(-2z))，用Div替代Reciprocal+NR+Mul(14→9ops)
//   5. exact: 取反多项式系数 + MulAddDst融合
//   6. float路径: 仅1个TBuf，复用队列buffer作为工作区
//   7. if constexpr编译时分支消除

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;    // 输入输出双缓冲
constexpr float GELU_ONE = 1.0f;
constexpr float GELU_HALF = 0.5f;
constexpr float ALPHA = 0.044715f;
constexpr float NEG_2_SQRT_2_OVER_PI = -1.5957691216f;  // -2*sqrt(2/pi)

// 取反erf多项式系数 (省去后续Muls(-1))
constexpr float ERF_P_OVER_SQRT2 = 0.23164190f;
constexpr float NEG_ERF_A1 = -0.254829592f;
constexpr float NEG_ERF_A2 = 0.284496736f;
constexpr float NEG_ERF_A3 = -1.421413741f;
constexpr float NEG_ERF_A4 = 1.453152027f;
constexpr float NEG_ERF_A5 = -1.061405429f;

template <typename T, uint32_t APPROXIMATE>
class KernelGeluV2 {
public:
    __aicore__ inline KernelGeluV2() {}

    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
        GET_TILING_DATA(tilingData, tiling);
        this->pipe = pipeIn;
        this->tileDataNum = tilingData.tileDataNum;
        this->tileNum = tilingData.tileNum;
        this->lastTileDataNum = tilingData.lastTileDataNum;

        // 单核: 全部数据归本核处理，无偏移
        uint32_t totalDataNum = this->tileDataNum * this->tileNum;  // GM缓冲区上界
        auto inputGm = reinterpret_cast<__gm__ T*>(input);
        auto outputGm = reinterpret_cast<__gm__ T*>(output);
        this->inputGm.SetGlobalBuffer(inputGm, totalDataNum);
        this->outputGm.SetGlobalBuffer(outputGm, totalDataNum);

        // Buffer初始化
        uint32_t bufferSizeCompute = tilingData.processDataNum_computes * sizeof(float);
        uint32_t bufferSizeCopy = this->tileDataNum * sizeof(T);

        if constexpr (std::is_same_v<T, float>) {
            pipe->InitBuffer(tempBuf1, bufferSizeCompute);
        } else {
            pipe->InitBuffer(castBuf, bufferSizeCompute);
            pipe->InitBuffer(tempBuf1, bufferSizeCompute);
            pipe->InitBuffer(tempBuf2, bufferSizeCompute);
        }

        pipe->InitBuffer(inQueueX, BUFFER_NUM, bufferSizeCopy);
        pipe->InitBuffer(outQueueY, BUFFER_NUM, bufferSizeCopy);
    }

    __aicore__ inline void Process() {
        if (this->tileNum == 0) return;
        int32_t loopCount = static_cast<int32_t>(this->tileNum);

        // 预取前2个tile到输入队列
        CopyIn(0);
        if (loopCount > 1) CopyIn(1);

        // 主循环: 最后一个tile使用lastTileDataNum
        for (int32_t i = 0; i < loopCount; ++i) {
            uint32_t count = (i < loopCount - 1) ? this->tileDataNum : this->lastTileDataNum;

            if constexpr (std::is_same_v<T, float>) {
                Compute_float(count);
            } else {
                Compute_half(count);
            }
            CopyOut(i, count);

            if (i + 2 < loopCount) CopyIn(i + 2);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        uint32_t offset = progress * this->tileDataNum;
        uint32_t count = (progress < static_cast<int32_t>(this->tileNum) - 1) ? this->tileDataNum : this->lastTileDataNum;
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopy(xLocal, this->inputGm[offset], count);
        inQueueX.EnQue(xLocal);
    }

    // Float路径
    __aicore__ inline void Compute_float(uint32_t count) {
        LocalTensor<float> temp = tempBuf1.Get<float>();
        LocalTensor<T> inputLocal = inQueueX.DeQue<T>();
        LocalTensor<T> outputLocal = outQueueY.AllocTensor<T>();

        LocalTensor<float> inputFloat = inputLocal;
        LocalTensor<float> outputFloat = outputLocal;

        if constexpr (APPROXIMATE == 0) {
            ComputeGeluExact_Float(inputFloat, outputFloat, temp, count);
        } else {
            ComputeGeluTanh_Float(inputFloat, outputFloat, temp, count);
        }

        inQueueX.FreeTensor(inputLocal);
        outQueueY.EnQue(outputLocal);
    }

    // Half/BF16路径: Cast+Compute+Cast
    __aicore__ inline void Compute_half(uint32_t count) {
        LocalTensor<float> inputData = castBuf.Get<float>();
        LocalTensor<float> temp1 = tempBuf1.Get<float>();
        LocalTensor<float> temp2 = tempBuf2.Get<float>();

        LocalTensor<T> inputLocal = inQueueX.DeQue<T>();
        LocalTensor<T> outputLocal = outQueueY.AllocTensor<T>();

        Cast(inputData, inputLocal, RoundMode::CAST_NONE, count);

        if constexpr (APPROXIMATE == 0) {
            ComputeGeluExact(inputData, temp1, temp2, count);
        } else {
            ComputeGeluTanh(inputData, temp1, temp2, count);
        }

        Cast(outputLocal, inputData, RoundMode::CAST_RINT, count);
        inQueueX.FreeTensor(inputLocal);
        outQueueY.EnQue(outputLocal);
    }

    // =====================================================================
    // Float: input + output(scratch) + temp, 结果直接在output
    // =====================================================================

    __aicore__ inline void ComputeGeluExact_Float(LocalTensor<float>& input,
                                                   LocalTensor<float>& output,
                                                   LocalTensor<float>& temp,
                                                   uint32_t count) {
        // step1: t via Reciprocal + NR (8 ops)
        Abs(temp, input, count);
        Muls(temp, temp, ERF_P_OVER_SQRT2, count);
        Adds(temp, temp, GELU_ONE, count);
        Reciprocal(output, temp, count);
        Mul(temp, temp, output, count);
        Muls(temp, temp, -1.0f, count);
        Adds(temp, temp, 2.0f, count);
        Mul(temp, output, temp, count);

        // step2: 取反Horner -poly → output (9 ops)
        Muls(output, temp, NEG_ERF_A5, count);
        Adds(output, output, NEG_ERF_A4, count);
        Mul(output, output, temp, count);
        Adds(output, output, NEG_ERF_A3, count);
        Mul(output, output, temp, count);
        Adds(output, output, NEG_ERF_A2, count);
        Mul(output, output, temp, count);
        Adds(output, output, NEG_ERF_A1, count);
        Mul(output, output, temp, count);

        // step3: exp(-x²/2) → temp (3 ops)
        Mul(temp, input, input, count);
        Muls(temp, temp, -0.5f, count);
        Exp(temp, temp, count);

        // step4: erf_abs = 1 + (-poly)*exp (2 ops)
        Mul(output, output, temp, count);
        Adds(output, output, GELU_ONE, count);

        // step5: GELU = 0.5*(x + |x|*erf_abs) → output (3 ops, MulAddDst融合)
        Abs(temp, input, count);
        MulAddDst(input, temp, output, count);  // input = x + |x|*erf_abs
        Muls(output, input, GELU_HALF, count);
    }

    __aicore__ inline void ComputeGeluTanh_Float(LocalTensor<float>& input,
                                                  LocalTensor<float>& output,
                                                  LocalTensor<float>& temp,
                                                  uint32_t count) {
        // GELU(x) = x * sigmoid(2z) = x / (1+exp(-2z))
        // 用Div直接计算x/denom，省去Reciprocal+NR+Mul: 14 ops → 9 ops

        // step1: -2z → temp (5 ops)
        Mul(temp, input, input, count);                  // x²
        Muls(temp, temp, ALPHA, count);                  // α*x²
        Mul(temp, temp, input, count);                   // α*x³
        Add(temp, input, temp, count);                   // x + α*x³
        Muls(temp, temp, NEG_2_SQRT_2_OVER_PI, count);  // -2z

        // step2: denom = 1+exp(-2z) → temp (3 ops)
        Mins(temp, temp, 88.0f, count);                  // 溢出保护
        Exp(temp, temp, count);
        Adds(temp, temp, GELU_ONE, count);

        // step3: GELU = x / denom (1 op)
        Div(output, input, temp, count);
    }

    // =====================================================================
    // Half/BF16: in-place结果在input (castBuf)
    // =====================================================================

    __aicore__ inline void ComputeGeluExact(LocalTensor<float>& input,
                                             LocalTensor<float>& temp1,
                                             LocalTensor<float>& temp2,
                                             uint32_t count) {
        // step1: NR精化 (8 ops)
        Abs(temp1, input, count);
        Muls(temp1, temp1, ERF_P_OVER_SQRT2, count);
        Adds(temp1, temp1, GELU_ONE, count);
        Reciprocal(temp2, temp1, count);
        Mul(temp1, temp1, temp2, count);
        Muls(temp1, temp1, -1.0f, count);
        Adds(temp1, temp1, 2.0f, count);
        Mul(temp1, temp2, temp1, count);

        // step2: 取反Horner → temp2 (9 ops)
        Muls(temp2, temp1, NEG_ERF_A5, count);
        Adds(temp2, temp2, NEG_ERF_A4, count);
        Mul(temp2, temp2, temp1, count);
        Adds(temp2, temp2, NEG_ERF_A3, count);
        Mul(temp2, temp2, temp1, count);
        Adds(temp2, temp2, NEG_ERF_A2, count);
        Mul(temp2, temp2, temp1, count);
        Adds(temp2, temp2, NEG_ERF_A1, count);
        Mul(temp2, temp2, temp1, count);

        // step3: exp(-x²/2) → temp1 (3 ops)
        Mul(temp1, input, input, count);
        Muls(temp1, temp1, -0.5f, count);
        Exp(temp1, temp1, count);

        // step4: erf_abs (2 ops)
        Mul(temp2, temp2, temp1, count);
        Adds(temp2, temp2, GELU_ONE, count);

        // step5: GELU → input (3 ops, MulAddDst融合)
        Abs(temp1, input, count);
        MulAddDst(input, temp1, temp2, count);  // input = x + |x|*erf_abs
        Muls(input, input, GELU_HALF, count);
    }

    __aicore__ inline void ComputeGeluTanh(LocalTensor<float>& input,
                                            LocalTensor<float>& temp1,
                                            LocalTensor<float>& temp2,
                                            uint32_t count) {
        // GELU(x) = x / (1+exp(-2z))，用Div直接计算
        // 14 ops → 9 ops

        // step1: -2z → temp1 (5 ops)
        Mul(temp1, input, input, count);
        Muls(temp1, temp1, ALPHA, count);
        Mul(temp1, temp1, input, count);
        Add(temp1, input, temp1, count);
        Muls(temp1, temp1, NEG_2_SQRT_2_OVER_PI, count);

        // step2: denom → temp1 (3 ops)
        Mins(temp1, temp1, 88.0f, count);
        Exp(temp1, temp1, count);
        Adds(temp1, temp1, GELU_ONE, count);

        // step3: GELU = x / denom → input (1 op)
        Div(input, input, temp1, count);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t count) {
        uint32_t offset = progress * this->tileDataNum;
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopy(this->outputGm[offset], yLocal, count);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    TBuf<QuePosition::VECCALC> castBuf;
    TBuf<QuePosition::VECCALC> tempBuf1;
    TBuf<QuePosition::VECCALC> tempBuf2;

    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;

    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t lastTileDataNum;
};


extern "C" __global__ __aicore__ void gelu_v2(GM_ADDR input, GM_ADDR output,
                                              GM_ADDR workspace, GM_ADDR tiling) {
    TPipe pipe;
    // tilingKey = typeKey + approximate * 3
    // exact: 0=half, 1=float, 2=bf16 | tanh: 3=half, 4=float, 5=bf16
    if (TILING_KEY_IS(0)) {
        KernelGeluV2<half, 0> op;
        op.Init(&pipe, input, output, workspace, tiling);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        KernelGeluV2<float, 0> op;
        op.Init(&pipe, input, output, workspace, tiling);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelGeluV2<bfloat16_t, 0> op;
        op.Init(&pipe, input, output, workspace, tiling);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        KernelGeluV2<half, 1> op;
        op.Init(&pipe, input, output, workspace, tiling);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        KernelGeluV2<float, 1> op;
        op.Init(&pipe, input, output, workspace, tiling);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        KernelGeluV2<bfloat16_t, 1> op;
        op.Init(&pipe, input, output, workspace, tiling);
        op.Process();
    }
}
