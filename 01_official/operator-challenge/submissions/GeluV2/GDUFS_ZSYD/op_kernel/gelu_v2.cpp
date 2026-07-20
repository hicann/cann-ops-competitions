#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue

template<typename TYPE_X, typename TYPE_Y> class KernelGeluV2 {
    using T = TYPE_X;
public:
    __aicore__ inline KernelGeluV2() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t CoreDataNum, uint32_t finalTileNum, uint32_t tileDataNum,
                                uint32_t TailDataNum, uint32_t approximate) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        this->coreDataNum = CoreDataNum;
        this->tileNum = finalTileNum;
        this->tileDataNum = tileDataNum;
        this->tailDataNum = TailDataNum;
        this->approximate = approximate;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, this->coreDataNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_Y));
        pipe.InitBuffer(workBuf, this->tileDataNum * sizeof(float));
        if constexpr (!std::is_same_v<TYPE_X, float>) {
            pipe.InitBuffer(xFloatBuf, this->tileDataNum * sizeof(float));
        }
    }
    __aicore__ inline void Process() {

        int32_t loopCount = this->tileNum;
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
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        if constexpr (std::is_same_v<TYPE_X, float>) {
            LocalTensor<float> workLocal = workBuf.Get<float>();
            ComputeGeluFloat(xLocal, workLocal, yLocal);
        } else {
            LocalTensor<float> xFloatLocal = xFloatBuf.Get<float>();
            LocalTensor<float> workLocal = workBuf.Get<float>();

            Cast(xFloatLocal, xLocal, RoundMode::CAST_NONE, this->processDataNum);
            ComputeGeluFloat(xFloatLocal, workLocal, workLocal);
            CastBack(yLocal, workLocal);
        }

        outQueueY.EnQue<TYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void ComputeGeluFloat(LocalTensor<float>& xLocal,
                                            LocalTensor<float>& workLocal,
                                            LocalTensor<float>& yLocal)
    {
        Mul(workLocal, xLocal, xLocal, this->processDataNum);

        if (this->approximate == 1) {
            ComputeTanhApprox(xLocal, workLocal, yLocal);
        } else {
            ComputeNoneApprox(xLocal, workLocal, yLocal);
        }

        // y = x / (1 + exp(-p(x))). For approximate=1, p(x) is 2*tanh_input.
        Muls(yLocal, yLocal, static_cast<float>(-1.0), this->processDataNum);
        Exp(yLocal, yLocal, this->processDataNum);
        Adds(yLocal, yLocal, static_cast<float>(1.0), this->processDataNum);
        Div(yLocal, xLocal, yLocal, this->processDataNum);
    }
    __aicore__ inline void ComputeNoneApprox(LocalTensor<float>& xLocal,
                                             LocalTensor<float>& x2Local,
                                             LocalTensor<float>& yLocal)
    {
         if (sizeof(DTYPE_X) == sizeof(float)) {
        // Logistic polynomial fitted to GELU none, max error below 1e-4 on [-10, 10].
            Muls(yLocal, x2Local, static_cast<float>(4.01528977e-11), this->processDataNum);
            Adds(yLocal, yLocal, static_cast<float>(-7.86359758e-09), this->processDataNum);
            Mul(yLocal, yLocal, x2Local, this->processDataNum);
            Adds(yLocal, yLocal, static_cast<float>(5.11335547e-07), this->processDataNum);
            Mul(yLocal, yLocal, x2Local, this->processDataNum);
            Adds(yLocal, yLocal, static_cast<float>(-7.16527169e-06), this->processDataNum);
            Mul(yLocal, yLocal, x2Local, this->processDataNum);
            Adds(yLocal, yLocal, static_cast<float>(-7.30012313e-04), this->processDataNum);
            Mul(yLocal, yLocal, x2Local, this->processDataNum);
            Adds(yLocal, yLocal, static_cast<float>(7.41330248e-02), this->processDataNum);
            Mul(yLocal, yLocal, x2Local, this->processDataNum);
            Adds(yLocal, yLocal, static_cast<float>(1.59536915), this->processDataNum);
            Mul(yLocal, yLocal, xLocal, this->processDataNum);
         }
         else {

            Muls(yLocal, x2Local, static_cast<float>(0.071429), this->processDataNum);
            Adds(yLocal, yLocal, static_cast<float>(1.597418571586), this->processDataNum);
            Mul(yLocal, yLocal, xLocal, this->processDataNum);
         }
         //if (constexpr (std::is_same_v<TYPE_X, half>)) 
        //  else{

        //     Muls(yLocal, x2Local, static_cast<float>(4.01528977e-11), this->processDataNum);
        //     Adds(yLocal, yLocal, static_cast<float>(-7.86359758e-09), this->processDataNum);
        //     Mul(yLocal, yLocal, x2Local, this->processDataNum);
        //     Adds(yLocal, yLocal, static_cast<float>(5.11335547e-07), this->processDataNum);
        //     Mul(yLocal, yLocal, x2Local, this->processDataNum);
        //     Adds(yLocal, yLocal, static_cast<float>(-7.16527169e-06), this->processDataNum);
        //     Mul(yLocal, yLocal, x2Local, this->processDataNum);
        //     Adds(yLocal, yLocal, static_cast<float>(-7.30012313e-04), this->processDataNum);
        //     Mul(yLocal, yLocal, x2Local, this->processDataNum);
        //     Adds(yLocal, yLocal, static_cast<float>(7.41330248e-02), this->processDataNum);
        //     Mul(yLocal, yLocal, x2Local, this->processDataNum);
        //     Adds(yLocal, yLocal, static_cast<float>(1.59536915), this->processDataNum);
        //     Mul(yLocal, yLocal, xLocal, this->processDataNum);


        //  }
    }
    __aicore__ inline void ComputeTanhApprox(LocalTensor<float>& xLocal,
                                             LocalTensor<float>& x2Local,
                                             LocalTensor<float>& yLocal)
    {
        // Equivalent to 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))).
        Muls(yLocal, x2Local, static_cast<float>(0.071354816092804), this->processDataNum);
        Adds(yLocal, yLocal, static_cast<float>(1.595769121605731), this->processDataNum);
        Mul(yLocal, yLocal, xLocal, this->processDataNum);
    }
    __aicore__ inline void CastBack(LocalTensor<DTYPE_Y>& yLocal, LocalTensor<float>& yFloatLocal)
    {
        if constexpr (std::is_same_v<TYPE_Y, half>) {
            Cast(yLocal, yFloatLocal, RoundMode::CAST_NONE, this->processDataNum);
        } else {
            Cast(yLocal, yFloatLocal, RoundMode::CAST_RINT, this->processDataNum);
        }
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> xFloatBuf;
    TBuf<QuePosition::VECCALC> workBuf;

    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    uint32_t approximate;
};
extern "C" __global__ __aicore__ void gelu_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelGeluV2<DTYPE_X, DTYPE_Y> op;
    op.Init(x, y, 
            tiling_data.CoreDataNum, tiling_data.finalTileNum, tiling_data.tileDataNum,
            tiling_data.TailDataNum, tiling_data.approximate);
    op.Process();
}
