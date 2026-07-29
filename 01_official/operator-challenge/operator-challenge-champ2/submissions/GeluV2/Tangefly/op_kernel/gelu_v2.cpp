#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template <class T1>
class KernelGeluV2 {
    public:
        __aicore__ inline KernelGeluV2() {}
        __aicore__ inline void Init(TPipe *pipeIn, GM_ADDR x, GM_ADDR y, int32_t xTotalLength,  
                                    int32_t tileNum, int32_t tileLength, 
                                    int32_t lastTileLength, int32_t invalidLength, 
                                    int32_t resTileNum, int32_t invalidLenPerResTile) 
        {   
            // 复制 tilingData 到类里面
            this->pipe = pipeIn;
            this->xTotalLength = xTotalLength;
            this->tileNum = tileNum;
            this->tileLength = tileLength;
            this->lastTileLength = lastTileLength;
            this->invalidLength = invalidLength;
            this->resTileNum = resTileNum;
            this->invalidLenPerResTile = invalidLenPerResTile;

            // 设置 Global Memory
            this->xGm.SetGlobalBuffer((__gm__ T1 *)x, this->xTotalLength);
            this->yGm.SetGlobalBuffer((__gm__ T1 *)y, this->xTotalLength);

            // 初始化队列 Buffer
            pipe->InitBuffer(this->inQueueX, BUFFER_NUM, this->tileLength * sizeof(T1));
            pipe->InitBuffer(this->outQueueY, BUFFER_NUM, this->tileLength * sizeof(T1));

            pipe->InitBuffer(tmpBuffer1, this->tileLength * sizeof(T1));
            pipe->InitBuffer(tmpBuffer2, this->tileLength * sizeof(T1));
        }

        __aicore__ inline void Process() {
            int32_t baseOffset = 0, tileOffset = 0;

            for(int32_t i = 0; i < this->tileNum; ++i){
                CopyIn(baseOffset, tileOffset, this->tileLength);
                Compute(baseOffset, tileOffset, this->tileLength);
                CopyOut(baseOffset, tileOffset, this->tileLength);
                tileOffset += this->tileLength;
            }
            for(int32_t i = 0; i < BUFFER_NUM; ++i){
                CopyIn(baseOffset, tileOffset, this->lastTileLength);
                Compute(baseOffset, tileOffset, this->lastTileLength);
                CopyOut(baseOffset, tileOffset, this->lastTileLength);
                tileOffset += this->lastTileLength;
            }
        }
    private:
        __aicore__ inline void CopyIn(int32_t baseOffset, int32_t tileOffset, int32_t length) {
            this->xLocal = this->inQueueX.template AllocTensor<T1>();

            // DataCopy(this->x1Local[0], this->x1Gm[baseOffset + tileOffset], length);
            // DataCopy(this->x2Local[0], this->x2Gm[baseOffset + tileOffset], length);

            DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T1)), 0, 0, 0};
            DataCopyPad(this->xLocal[0], this->xGm[baseOffset + tileOffset], copyParams, padParams);

            this->inQueueX.EnQue(this->xLocal);
        }

        __aicore__ inline void Compute(int32_t baseOffset, int32_t tileOffset, int32_t length) {
            LocalTensor<float> xLocal = inQueueX.DeQue<float>();
            LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

            // Adds(yLocal, xLocal, 0.0f, length);

            LocalTensor<float> tmp1 = tmpBuffer1.Get<float>();
            LocalTensor<float> tmp2 = tmpBuffer2.Get<float>();
            // 1. 数据类型转换
            float a0 = 0.254829592f, a1 = -0.284496736f, a2 = 1.421413741f;
            float a3 = -1.453152027f, a4 = 1.061405429f;
            float kAlpha = 0.70710678118654752440;
            float temp;

            // 1. b = x * kAlpha   yLocal -> b
            Muls(yLocal, xLocal, kAlpha, length);
            // 2. Erf
            // 2.1 t = 1.0 / (1.0 + 0.3275911 * abs(b)) tmp1 -> t
            Abs(tmp1, yLocal, length);
            Muls(tmp1, tmp1, 0.3275911f, length);
            Adds(tmp1, tmp1, 1.0f, length);
            Duplicate(tmp2, 1.0f, length);
            Div(tmp1, tmp2, tmp1, length);
            // 2.2 tau = t * (a[0] + t * (a[1] + t * (a[2] + t * (a[3] + t * a[4])))) tmp1 -> tau
            Muls(tmp2, tmp1, a4, length);
            Adds(tmp2, tmp2, a3, length);
            Mul(tmp2, tmp2, tmp1, length);
            Adds(tmp2, tmp2, a2, length);
            Mul(tmp2, tmp2, tmp1, length);
            Adds(tmp2, tmp2, a1, length);
            Mul(tmp2, tmp2, tmp1, length);
            Adds(tmp2, tmp2, a0, length);
            Mul(tmp1, tmp2, tmp1, length);
            // 2.3 erf_value = 1.0 - tau * exp(-b * b)  tmp2 -> erf_value
            Mul(tmp2, yLocal, yLocal, length);
            Muls(tmp2, tmp2, -1.0f, length);
            Exp(tmp2, tmp2, length);
            Muls(tmp1, tmp1, -1.0f, length);
            Mul(tmp2, tmp1, tmp2, length);
            Adds(tmp2, tmp2, 1.0f, length);
            // 2.4 erf_value if b >= 0 else -erf_value
            Maxs(yLocal, yLocal, 0.0f, length);
            Mins(yLocal, yLocal, 1.0f, length);
            Cast(yLocal, yLocal, RoundMode::CAST_CEIL, length);
            Adds(tmp1, yLocal, -1.0f, length);
            Add(tmp1, tmp1, yLocal, length);
            Mul(tmp2, tmp2, tmp1, length);
            // 3. res = x * 0.5 * (1 + erf_value)
            Adds(tmp2, tmp2, 1.0f, length);
            Muls(tmp2, tmp2, 0.5f, length);
            Mul(yLocal, xLocal, tmp2, length);
            

            outQueueY.EnQue<float>(yLocal);
            inQueueX.FreeTensor(xLocal);
        }
    
        __aicore__ inline void CopyOut(int32_t baseOffset, int32_t tileOffset, int32_t length) {
            this->yLocal = outQueueY.DeQue<T1>();

            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T1)), 0, 0, 0};
            DataCopyPad(yGm[baseOffset + tileOffset], this->yLocal[0], copyParams);
    
            outQueueY.FreeTensor(this->yLocal);
        }

    private:
        TPipe *pipe;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
        TBuf<QuePosition::VECCALC> tmpBuffer1, tmpBuffer2;

        LocalTensor<T1> xLocal, yLocal;
        GlobalTensor<T1> xGm, yGm;

        int32_t xTotalLength;
        int32_t tileNum, tileLength, lastTileLength, invalidLength, resTileNum, invalidLenPerResTile;
};

template <class T1>
class KernelGeluV2_V2 {
    public:
        __aicore__ inline KernelGeluV2_V2() {}
        __aicore__ inline void Init(TPipe *pipeIn, GM_ADDR x, GM_ADDR y, int32_t xTotalLength,  
                                    int32_t tileNum, int32_t tileLength, 
                                    int32_t lastTileLength, int32_t invalidLength, 
                                    int32_t resTileNum, int32_t invalidLenPerResTile) 
        {   
            // 复制 tilingData 到类里面
            this->pipe = pipeIn;
            this->xTotalLength = xTotalLength;
            this->tileNum = tileNum;
            this->tileLength = tileLength;
            this->lastTileLength = lastTileLength;
            this->invalidLength = invalidLength;
            this->resTileNum = resTileNum;
            this->invalidLenPerResTile = invalidLenPerResTile;

            // 设置 Global Memory
            this->xGm.SetGlobalBuffer((__gm__ T1 *)x, this->xTotalLength);
            this->yGm.SetGlobalBuffer((__gm__ T1 *)y, this->xTotalLength);

            // 初始化队列 Buffer
            pipe->InitBuffer(this->inQueueX, BUFFER_NUM, this->tileLength * sizeof(T1));
            pipe->InitBuffer(this->outQueueY, BUFFER_NUM, this->tileLength * sizeof(T1));

            pipe->InitBuffer(tmpBuffer1, this->tileLength * sizeof(T1));
            pipe->InitBuffer(tmpBuffer2, this->tileLength * sizeof(T1));
        }

        __aicore__ inline void Process() {
            int32_t baseOffset = 0, tileOffset = 0;

            for(int32_t i = 0; i < this->tileNum; ++i){
                CopyIn(baseOffset, tileOffset, this->tileLength);
                Compute(baseOffset, tileOffset, this->tileLength);
                CopyOut(baseOffset, tileOffset, this->tileLength);
                tileOffset += this->tileLength;
            }
            for(int32_t i = 0; i < BUFFER_NUM; ++i){
                CopyIn(baseOffset, tileOffset, this->lastTileLength);
                Compute(baseOffset, tileOffset, this->lastTileLength);
                CopyOut(baseOffset, tileOffset, this->lastTileLength);
                tileOffset += this->lastTileLength;
            }
        }
    private:
        __aicore__ inline void CopyIn(int32_t baseOffset, int32_t tileOffset, int32_t length) {
            this->xLocal = this->inQueueX.template AllocTensor<T1>();

            // DataCopy(this->x1Local[0], this->x1Gm[baseOffset + tileOffset], length);
            // DataCopy(this->x2Local[0], this->x2Gm[baseOffset + tileOffset], length);

            DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T1)), 0, 0, 0};
            DataCopyPad(this->xLocal[0], this->xGm[baseOffset + tileOffset], copyParams, padParams);

            this->inQueueX.EnQue(this->xLocal);
        }

        __aicore__ inline void Compute(int32_t baseOffset, int32_t tileOffset, int32_t length) {
            LocalTensor<float> xLocal = inQueueX.DeQue<float>();
            LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

            LocalTensor<float> tmp1 = tmpBuffer1.Get<float>();
            LocalTensor<float> tmp2 = tmpBuffer2.Get<float>();
            
            // x2 = input_x * input_x
            // inner = (1.59501207 + x2 * (0.07401461 + x2 * -0.00070356)) * input_x
            // output = input_x / (1.0 + torch.exp(-inner))
            
            // x2 = input_x * input_x
            Mul(tmp1, xLocal, xLocal, length);
            Mins(tmp1, tmp1, 100.0f, length);
            // x2 * -0.00070356
            Muls(tmp2, tmp1, -0.00070356f, length);
            // (0.07401461 + x2 * -0.00070356)
            Adds(tmp2, tmp2, 0.07401461f, length);
            // x2 * (0.07401461 + x2 * -0.00070356)
            Mul(tmp2, tmp1, tmp2, length);
            // (1.59501207 + x2 * (0.07401461 + x2 * -0.00070356))
            Adds(tmp2, tmp2, 1.59501207f, length);
            // inner = (1.59501207 + x2 * (0.07401461 + x2 * -0.00070356)) * input_x
            Mul(yLocal, tmp2, xLocal, length);
            // torch.exp(-inner)
            Muls(yLocal, yLocal, -1.0f, length);
            Exp(yLocal, yLocal, length);
            // (1.0 + torch.exp(-inner))
            Adds(yLocal, yLocal, 1.0f, length);
            Div(yLocal, xLocal, yLocal, length);

            outQueueY.EnQue<float>(yLocal);
            inQueueX.FreeTensor(xLocal);
        }
    
        __aicore__ inline void CopyOut(int32_t baseOffset, int32_t tileOffset, int32_t length) {
            this->yLocal = outQueueY.DeQue<T1>();

            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T1)), 0, 0, 0};
            DataCopyPad(yGm[baseOffset + tileOffset], this->yLocal[0], copyParams);
    
            outQueueY.FreeTensor(this->yLocal);
        }

    private:
        TPipe *pipe;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
        TBuf<QuePosition::VECCALC> tmpBuffer1, tmpBuffer2;

        LocalTensor<T1> xLocal, yLocal;
        GlobalTensor<T1> xGm, yGm;

        int32_t xTotalLength;
        int32_t tileNum, tileLength, lastTileLength, invalidLength, resTileNum, invalidLenPerResTile;
};

template <class T1>
class KernelGeluV2ForHalf {
    public:
        __aicore__ inline KernelGeluV2ForHalf() {}
        __aicore__ inline void Init(TPipe *pipeIn, GM_ADDR x, GM_ADDR y, int32_t xTotalLength,  
                                    int32_t tileNum, int32_t tileLength, 
                                    int32_t lastTileLength, int32_t invalidLength, 
                                    int32_t resTileNum, int32_t invalidLenPerResTile) 
        {   
            // 复制 tilingData 到类里面
            this->pipe = pipeIn;
            this->xTotalLength = xTotalLength;
            this->tileNum = tileNum;
            this->tileLength = tileLength;
            this->lastTileLength = lastTileLength;
            this->invalidLength = invalidLength;
            this->resTileNum = resTileNum;
            this->invalidLenPerResTile = invalidLenPerResTile;

            // 设置 Global Memory
            this->xGm.SetGlobalBuffer((__gm__ T1 *)x, this->xTotalLength);
            this->yGm.SetGlobalBuffer((__gm__ T1 *)y, this->xTotalLength);

            // 初始化队列 Buffer
            pipe->InitBuffer(this->inQueueX, BUFFER_NUM, this->tileLength * sizeof(T1));
            pipe->InitBuffer(this->outQueueY, BUFFER_NUM, this->tileLength * sizeof(T1));

            pipe->InitBuffer(tmpBuffer1, this->tileLength * sizeof(float));
            pipe->InitBuffer(tmpBuffer2, this->tileLength * sizeof(float));
            pipe->InitBuffer(tmpBuffer3, this->tileLength * sizeof(float));
        }

        __aicore__ inline void Process() {
            int32_t baseOffset = 0, tileOffset = 0;

            for(int32_t i = 0; i < this->tileNum; ++i){
                CopyIn(baseOffset, tileOffset, this->tileLength);
                Compute(baseOffset, tileOffset, this->tileLength);
                CopyOut(baseOffset, tileOffset, this->tileLength);
                tileOffset += this->tileLength;
            }
            for(int32_t i = 0; i < BUFFER_NUM; ++i){
                CopyIn(baseOffset, tileOffset, this->lastTileLength);
                Compute(baseOffset, tileOffset, this->lastTileLength);
                CopyOut(baseOffset, tileOffset, this->lastTileLength);
                tileOffset += this->lastTileLength;
            }
        }
    private:
        __aicore__ inline void CopyIn(int32_t baseOffset, int32_t tileOffset, int32_t length) {
            this->xLocal = this->inQueueX.template AllocTensor<T1>();

            // DataCopy(this->x1Local[0], this->x1Gm[baseOffset + tileOffset], length);
            // DataCopy(this->x2Local[0], this->x2Gm[baseOffset + tileOffset], length);

            DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T1)), 0, 0, 0};
            DataCopyPad(this->xLocal[0], this->xGm[baseOffset + tileOffset], copyParams, padParams);

            this->inQueueX.EnQue(this->xLocal);
        }

        __aicore__ inline void Compute(int32_t baseOffset, int32_t tileOffset, int32_t length) {
            LocalTensor<T1> xLocal = inQueueX.DeQue<T1>();
            LocalTensor<T1> yLocal = outQueueY.AllocTensor<T1>();

            LocalTensor<float> tmp1 = tmpBuffer1.Get<float>();
            LocalTensor<float> tmp2 = tmpBuffer2.Get<float>();
            LocalTensor<float> tmp3 = tmpBuffer3.Get<float>();
            // 数据类型转换 tmp1 -> a
            Cast(tmp1, xLocal, RoundMode::CAST_NONE, length);
            float a0 = 0.254829592f, a1 = -0.284496736f, a2 = 1.421413741f;
            float a3 = -1.453152027f, a4 = 1.061405429f;
            float kAlpha = 0.70710678118654752440;
            float temp;
            // 1. x = a * kAlpha   tmp1 -> x
            Muls(tmp1, tmp1, kAlpha, length);
            // 2. Erf
            // 2.1 t = 1.0 / (1.0 + 0.3275911 * abs(x)) tmp2 -> t
            Abs(tmp2, tmp1, length);
            Muls(tmp2, tmp2, 0.3275911f, length);
            Adds(tmp2, tmp2, 1.0f, length);
            Duplicate(tmp3, 1.0f, length);
            Div(tmp2, tmp3, tmp2, length);
            // 2.2 tau = t * (a[0] + t * (a[1] + t * (a[2] + t * (a[3] + t * a[4])))) tmp3 -> tau
            Muls(tmp3, tmp2, a4, length);
            Adds(tmp3, tmp3, a3, length);
            Mul(tmp3, tmp2, tmp3, length);
            Adds(tmp3, tmp3, a2, length);
            Mul(tmp3, tmp2, tmp3, length);
            Adds(tmp3, tmp3, a1, length);
            Mul(tmp3, tmp2, tmp3, length);
            Adds(tmp3, tmp3, a0, length);
            Mul(tmp3, tmp2, tmp3, length);
            // 2.3 erf_value = 1.0 - tau * exp(-x * x)  tmp2 -> erf_value
            Mul(tmp2, tmp1, tmp1, length);
            Muls(tmp2, tmp2, -1.0f, length);
            Exp(tmp2, tmp2, length);
            Muls(tmp3, tmp3, -1.0f, length);
            Mul(tmp2, tmp2, tmp3, length);
            Adds(tmp2, tmp2, 1.0f, length);
            // 2.4 erf_value if x >= 0 else -erf_value
            Maxs(tmp1, tmp1, 0.0f, length);
            Mins(tmp1, tmp1, 1.0f, length);
            Cast(tmp3, tmp1, RoundMode::CAST_CEIL, length);
            Adds(tmp1, tmp3, -1.0f, length);
            Add(tmp1, tmp1, tmp3, length);
            Mul(tmp2, tmp2, tmp1, length);

            // 3. d = a * 0.5 * (1 + erf_value)
            Adds(tmp2, tmp2, 1.0f, length);
            Muls(tmp2, tmp2, 0.5f, length);
            Cast(tmp1, xLocal, RoundMode::CAST_NONE, length);
            Mul(tmp1, tmp1, tmp2, length);

            Cast(yLocal, tmp1, RoundMode::CAST_NONE, length);

            outQueueY.EnQue<T1>(yLocal);
            inQueueX.FreeTensor(xLocal);
        }
    
        __aicore__ inline void CopyOut(int32_t baseOffset, int32_t tileOffset, int32_t length) {
            this->yLocal = outQueueY.DeQue<T1>();

            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T1)), 0, 0, 0};
            DataCopyPad(yGm[baseOffset + tileOffset], this->yLocal[0], copyParams);
    
            outQueueY.FreeTensor(this->yLocal);
        }

    private:
        TPipe *pipe;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
        TBuf<QuePosition::VECCALC> tmpBuffer1, tmpBuffer2, tmpBuffer3;

        LocalTensor<T1> xLocal, yLocal;
        GlobalTensor<T1> xGm, yGm;

        int32_t xTotalLength;
        int32_t tileNum, tileLength, lastTileLength, invalidLength, resTileNum, invalidLenPerResTile;
};

template <class T1>
class KernelGeluV2ForHalf_V2 {
    public:
        __aicore__ inline KernelGeluV2ForHalf_V2() {}
        __aicore__ inline void Init(TPipe *pipeIn, GM_ADDR x, GM_ADDR y, int32_t xTotalLength,  
                                    int32_t tileNum, int32_t tileLength, 
                                    int32_t lastTileLength, int32_t invalidLength, 
                                    int32_t resTileNum, int32_t invalidLenPerResTile) 
        {   
            // 复制 tilingData 到类里面
            this->pipe = pipeIn;
            this->xTotalLength = xTotalLength;
            this->tileNum = tileNum;
            this->tileLength = tileLength;
            this->lastTileLength = lastTileLength;
            this->invalidLength = invalidLength;
            this->resTileNum = resTileNum;
            this->invalidLenPerResTile = invalidLenPerResTile;

            // 设置 Global Memory
            this->xGm.SetGlobalBuffer((__gm__ T1 *)x, this->xTotalLength);
            this->yGm.SetGlobalBuffer((__gm__ T1 *)y, this->xTotalLength);

            // 初始化队列 Buffer
            pipe->InitBuffer(this->inQueueX, BUFFER_NUM, this->tileLength * sizeof(T1));
            pipe->InitBuffer(this->outQueueY, BUFFER_NUM, this->tileLength * sizeof(T1));
        }

        __aicore__ inline void Process() {
            int32_t baseOffset = 0, tileOffset = 0;

            for(int32_t i = 0; i < this->tileNum; ++i){
                CopyIn(baseOffset, tileOffset, this->tileLength);
                Compute(baseOffset, tileOffset, this->tileLength);
                CopyOut(baseOffset, tileOffset, this->tileLength);
                tileOffset += this->tileLength;
            }
            for(int32_t i = 0; i < BUFFER_NUM; ++i){
                CopyIn(baseOffset, tileOffset, this->lastTileLength);
                Compute(baseOffset, tileOffset, this->lastTileLength);
                CopyOut(baseOffset, tileOffset, this->lastTileLength);
                tileOffset += this->lastTileLength;
            }
        }
    private:
        __aicore__ inline void CopyIn(int32_t baseOffset, int32_t tileOffset, int32_t length) {
            this->xLocal = this->inQueueX.template AllocTensor<T1>();

            // DataCopy(this->x1Local[0], this->x1Gm[baseOffset + tileOffset], length);
            // DataCopy(this->x2Local[0], this->x2Gm[baseOffset + tileOffset], length);

            DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T1)), 0, 0, 0};
            DataCopyPad(this->xLocal[0], this->xGm[baseOffset + tileOffset], copyParams, padParams);

            this->inQueueX.EnQue(this->xLocal);
        }

        __aicore__ inline void Compute(int32_t baseOffset, int32_t tileOffset, int32_t length) {
            LocalTensor<T1> xLocal = inQueueX.DeQue<T1>();
            LocalTensor<T1> yLocal = outQueueY.AllocTensor<T1>();

            half COEFF0 = static_cast<half>(-0.071429f);
            half COEFF1 = static_cast<half>(22.363860002236);

            Mul(yLocal, xLocal, xLocal, length);
            Adds(yLocal, yLocal, COEFF1, length);
            Mul(yLocal, yLocal, xLocal, length);
            Muls(yLocal, yLocal, COEFF0, length);
            Exp(yLocal, yLocal, length);
            Adds(yLocal, yLocal, static_cast<half>(1.0f), length);
            Div(yLocal, xLocal, yLocal, length);



            outQueueY.EnQue<T1>(yLocal);
            inQueueX.FreeTensor(xLocal);
        }
    
        __aicore__ inline void CopyOut(int32_t baseOffset, int32_t tileOffset, int32_t length) {
            this->yLocal = outQueueY.DeQue<T1>();

            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T1)), 0, 0, 0};
            DataCopyPad(yGm[baseOffset + tileOffset], this->yLocal[0], copyParams);
    
            outQueueY.FreeTensor(this->yLocal);
        }

    private:
        TPipe *pipe;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

        LocalTensor<T1> xLocal, yLocal;
        GlobalTensor<T1> xGm, yGm;

        int32_t xTotalLength;
        int32_t tileNum, tileLength, lastTileLength, invalidLength, resTileNum, invalidLenPerResTile;
};

template <class T1>
class KernelGeluV2ForBF16 {
    public:
        __aicore__ inline KernelGeluV2ForBF16() {}
        __aicore__ inline void Init(TPipe *pipeIn, GM_ADDR x, GM_ADDR y, int32_t xTotalLength,  
                                    int32_t tileNum, int32_t tileLength, 
                                    int32_t lastTileLength, int32_t invalidLength, 
                                    int32_t resTileNum, int32_t invalidLenPerResTile) 
        {   
            // 复制 tilingData 到类里面
            this->pipe = pipeIn;
            this->xTotalLength = xTotalLength;
            this->tileNum = tileNum;
            this->tileLength = tileLength;
            this->lastTileLength = lastTileLength;
            this->invalidLength = invalidLength;
            this->resTileNum = resTileNum;
            this->invalidLenPerResTile = invalidLenPerResTile;

            // 设置 Global Memory
            this->xGm.SetGlobalBuffer((__gm__ T1 *)x, this->xTotalLength);
            this->yGm.SetGlobalBuffer((__gm__ T1 *)y, this->xTotalLength);

            // 初始化队列 Buffer
            pipe->InitBuffer(this->inQueueX, BUFFER_NUM, this->tileLength * sizeof(T1));
            pipe->InitBuffer(this->outQueueY, BUFFER_NUM, this->tileLength * sizeof(T1));

            pipe->InitBuffer(tmpBuffer1, this->tileLength * sizeof(float));
            pipe->InitBuffer(tmpBuffer2, this->tileLength * sizeof(float));
            pipe->InitBuffer(tmpBuffer3, this->tileLength * sizeof(float));
        }

        __aicore__ inline void Process() {
            int32_t baseOffset = 0, tileOffset = 0;

            for(int32_t i = 0; i < this->tileNum; ++i){
                CopyIn(baseOffset, tileOffset, this->tileLength);
                Compute(baseOffset, tileOffset, this->tileLength);
                CopyOut(baseOffset, tileOffset, this->tileLength);
                tileOffset += this->tileLength;
            }
            for(int32_t i = 0; i < BUFFER_NUM; ++i){
                CopyIn(baseOffset, tileOffset, this->lastTileLength);
                Compute(baseOffset, tileOffset, this->lastTileLength);
                CopyOut(baseOffset, tileOffset, this->lastTileLength);
                tileOffset += this->lastTileLength;
            }
        }
    private:
        __aicore__ inline void CopyIn(int32_t baseOffset, int32_t tileOffset, int32_t length) {
            this->xLocal = this->inQueueX.template AllocTensor<T1>();

            // DataCopy(this->x1Local[0], this->x1Gm[baseOffset + tileOffset], length);
            // DataCopy(this->x2Local[0], this->x2Gm[baseOffset + tileOffset], length);

            DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T1)), 0, 0, 0};
            DataCopyPad(this->xLocal[0], this->xGm[baseOffset + tileOffset], copyParams, padParams);

            this->inQueueX.EnQue(this->xLocal);
        }

        __aicore__ inline void Compute(int32_t baseOffset, int32_t tileOffset, int32_t length) {
            LocalTensor<T1> xLocal = inQueueX.DeQue<T1>();
            LocalTensor<T1> yLocal = outQueueY.AllocTensor<T1>();

            LocalTensor<float> tmp1 = tmpBuffer1.Get<float>();
            LocalTensor<float> tmp2 = tmpBuffer2.Get<float>();
            LocalTensor<float> tmp3 = tmpBuffer3.Get<float>();
            // 数据类型转换 tmp1 -> a
            Cast(tmp1, xLocal, RoundMode::CAST_NONE, length);
            float a0 = 0.254829592f, a1 = -0.284496736f, a2 = 1.421413741f;
            float a3 = -1.453152027f, a4 = 1.061405429f;
            float kAlpha = 0.70710678118654752440;
            float temp;
            // 1. x = a * kAlpha   tmp1 -> x
            Muls(tmp1, tmp1, kAlpha, length);
            // 2. Erf
            // 2.1 t = 1.0 / (1.0 + 0.3275911 * abs(x)) tmp2 -> t
            Abs(tmp2, tmp1, length);
            Muls(tmp2, tmp2, 0.3275911f, length);
            Adds(tmp2, tmp2, 1.0f, length);
            Duplicate(tmp3, 1.0f, length);
            Div(tmp2, tmp3, tmp2, length);
            // 2.2 tau = t * (a[0] + t * (a[1] + t * (a[2] + t * (a[3] + t * a[4])))) tmp3 -> tau
            Muls(tmp3, tmp2, a4, length);
            Adds(tmp3, tmp3, a3, length);
            Mul(tmp3, tmp2, tmp3, length);
            Adds(tmp3, tmp3, a2, length);
            Mul(tmp3, tmp2, tmp3, length);
            Adds(tmp3, tmp3, a1, length);
            Mul(tmp3, tmp2, tmp3, length);
            Adds(tmp3, tmp3, a0, length);
            Mul(tmp3, tmp2, tmp3, length);
            // 2.3 erf_value = 1.0 - tau * exp(-x * x)  tmp2 -> erf_value
            Mul(tmp2, tmp1, tmp1, length);
            Muls(tmp2, tmp2, -1.0f, length);
            Exp(tmp2, tmp2, length);
            Muls(tmp3, tmp3, -1.0f, length);
            Mul(tmp2, tmp2, tmp3, length);
            Adds(tmp2, tmp2, 1.0f, length);
            // 2.4 erf_value if x >= 0 else -erf_value
            Maxs(tmp1, tmp1, 0.0f, length);
            Mins(tmp1, tmp1, 1.0f, length);
            Cast(tmp3, tmp1, RoundMode::CAST_CEIL, length);
            Adds(tmp1, tmp3, -1.0f, length);
            Add(tmp1, tmp1, tmp3, length);
            Mul(tmp2, tmp2, tmp1, length);

            // 3. d = a * 0.5 * (1 + erf_value)
            Adds(tmp2, tmp2, 1.0f, length);
            Muls(tmp2, tmp2, 0.5f, length);
            Cast(tmp1, xLocal, RoundMode::CAST_NONE, length);
            Mul(tmp1, tmp1, tmp2, length);

            Cast(yLocal, tmp1, RoundMode::CAST_RINT, length);

            outQueueY.EnQue<T1>(yLocal);
            inQueueX.FreeTensor(xLocal);
        }
    
        __aicore__ inline void CopyOut(int32_t baseOffset, int32_t tileOffset, int32_t length) {
            this->yLocal = outQueueY.DeQue<T1>();

            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T1)), 0, 0, 0};
            DataCopyPad(yGm[baseOffset + tileOffset], this->yLocal[0], copyParams);
    
            outQueueY.FreeTensor(this->yLocal);
        }

    private:
        TPipe *pipe;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
        TBuf<QuePosition::VECCALC> tmpBuffer1, tmpBuffer2, tmpBuffer3;

        LocalTensor<T1> xLocal, yLocal;
        GlobalTensor<T1> xGm, yGm;

        int32_t xTotalLength;
        int32_t tileNum, tileLength, lastTileLength, invalidLength, resTileNum, invalidLenPerResTile;
};

template <class T1>
class KernelGeluV2Atanh {
    public:
        __aicore__ inline KernelGeluV2Atanh() {}
        __aicore__ inline void Init(TPipe *pipeIn, GM_ADDR x, GM_ADDR y, int32_t xTotalLength,  
                                    int32_t tileNum, int32_t tileLength, 
                                    int32_t lastTileLength, int32_t invalidLength, 
                                    int32_t resTileNum, int32_t invalidLenPerResTile) 
        {   
            // 复制 tilingData 到类里面
            this->pipe = pipeIn;
            this->xTotalLength = xTotalLength;
            this->tileNum = tileNum;
            this->tileLength = tileLength;
            this->lastTileLength = lastTileLength;
            this->invalidLength = invalidLength;
            this->resTileNum = resTileNum;
            this->invalidLenPerResTile = invalidLenPerResTile;

            // 设置 Global Memory
            this->xGm.SetGlobalBuffer((__gm__ T1 *)x, this->xTotalLength);
            this->yGm.SetGlobalBuffer((__gm__ T1 *)y, this->xTotalLength);

            // 初始化队列 Buffer
            pipe->InitBuffer(this->inQueueX, BUFFER_NUM, this->tileLength * sizeof(T1));
            pipe->InitBuffer(this->outQueueY, BUFFER_NUM, this->tileLength * sizeof(T1));

            pipe->InitBuffer(tmpBuffer1, this->tileLength * sizeof(T1));
        }

        __aicore__ inline void Process() {
            int32_t baseOffset = 0, tileOffset = 0;

            for(int32_t i = 0; i < this->tileNum; ++i){
                CopyIn(baseOffset, tileOffset, this->tileLength);
                Compute(baseOffset, tileOffset, this->tileLength);
                CopyOut(baseOffset, tileOffset, this->tileLength);
                tileOffset += this->tileLength;
            }
            for(int32_t i = 0; i < BUFFER_NUM; ++i){
                CopyIn(baseOffset, tileOffset, this->lastTileLength);
                Compute(baseOffset, tileOffset, this->lastTileLength);
                CopyOut(baseOffset, tileOffset, this->lastTileLength);
                tileOffset += this->lastTileLength;
            }
        }
    private:
        __aicore__ inline void CopyIn(int32_t baseOffset, int32_t tileOffset, int32_t length) {
            this->xLocal = this->inQueueX.template AllocTensor<T1>();

            // DataCopy(this->x1Local[0], this->x1Gm[baseOffset + tileOffset], length);
            // DataCopy(this->x2Local[0], this->x2Gm[baseOffset + tileOffset], length);

            DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T1)), 0, 0, 0};
            DataCopyPad(this->xLocal[0], this->xGm[baseOffset + tileOffset], copyParams, padParams);

            this->inQueueX.EnQue(this->xLocal);
        }

        __aicore__ inline void Compute(int32_t baseOffset, int32_t tileOffset, int32_t length) {
            LocalTensor<float> xLocal = inQueueX.DeQue<float>();
            LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

            LocalTensor<float> tmp1 = tmpBuffer1.Get<float>();

            float a0 = 0.044715f, a1 = 0.7978845608028654f, a2 = -2.0f;

            // tanh inner
            Mul(tmp1, xLocal, xLocal, length);
            Mul(tmp1, tmp1, xLocal, length);
            Muls(tmp1, tmp1, a0, length);
            Add(tmp1, tmp1, xLocal, length);
            Muls(tmp1, tmp1, a1, length);

            // tanh(x) = 2 / (1 + exp(-2x)) - 1
            Muls(tmp1, tmp1, a2, length);
            Exp(tmp1, tmp1, length);
            Adds(tmp1, tmp1, 1.0f, length);
            Div(yLocal, xLocal, tmp1, length);

            outQueueY.EnQue<float>(yLocal);
            inQueueX.FreeTensor(xLocal);
        }
    
        __aicore__ inline void CopyOut(int32_t baseOffset, int32_t tileOffset, int32_t length) {
            this->yLocal = outQueueY.DeQue<T1>();

            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T1)), 0, 0, 0};
            DataCopyPad(yGm[baseOffset + tileOffset], this->yLocal[0], copyParams);
    
            outQueueY.FreeTensor(this->yLocal);
        }

    private:
        TPipe *pipe;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
        TBuf<QuePosition::VECCALC> tmpBuffer1;

        LocalTensor<T1> xLocal, yLocal;
        GlobalTensor<T1> xGm, yGm;

        int32_t xTotalLength;
        int32_t tileNum, tileLength, lastTileLength, invalidLength, resTileNum, invalidLenPerResTile;
};

template <class T1>
class KernelGeluV2AtanhForHalf {
    public:
        __aicore__ inline KernelGeluV2AtanhForHalf() {}
        __aicore__ inline void Init(TPipe *pipeIn, GM_ADDR x, GM_ADDR y, int32_t xTotalLength,  
                                    int32_t tileNum, int32_t tileLength, 
                                    int32_t lastTileLength, int32_t invalidLength, 
                                    int32_t resTileNum, int32_t invalidLenPerResTile) 
        {   
            // 复制 tilingData 到类里面
            this->pipe = pipeIn;
            this->xTotalLength = xTotalLength;
            this->tileNum = tileNum;
            this->tileLength = tileLength;
            this->lastTileLength = lastTileLength;
            this->invalidLength = invalidLength;
            this->resTileNum = resTileNum;
            this->invalidLenPerResTile = invalidLenPerResTile;

            // 设置 Global Memory
            this->xGm.SetGlobalBuffer((__gm__ T1 *)x, this->xTotalLength);
            this->yGm.SetGlobalBuffer((__gm__ T1 *)y, this->xTotalLength);

            // 初始化队列 Buffer
            pipe->InitBuffer(this->inQueueX, BUFFER_NUM, this->tileLength * sizeof(T1));
            pipe->InitBuffer(this->outQueueY, BUFFER_NUM, this->tileLength * sizeof(T1));

            pipe->InitBuffer(tmpBuffer1, this->tileLength * sizeof(float));
            pipe->InitBuffer(tmpBuffer2, this->tileLength * sizeof(float));
            pipe->InitBuffer(tmpBuffer3, this->tileLength * sizeof(float));
        }

        __aicore__ inline void Process() {
            int32_t baseOffset = 0, tileOffset = 0;

            for(int32_t i = 0; i < this->tileNum; ++i){
                CopyIn(baseOffset, tileOffset, this->tileLength);
                Compute(baseOffset, tileOffset, this->tileLength);
                CopyOut(baseOffset, tileOffset, this->tileLength);
                tileOffset += this->tileLength;
            }
            for(int32_t i = 0; i < BUFFER_NUM; ++i){
                CopyIn(baseOffset, tileOffset, this->lastTileLength);
                Compute(baseOffset, tileOffset, this->lastTileLength);
                CopyOut(baseOffset, tileOffset, this->lastTileLength);
                tileOffset += this->lastTileLength;
            }
        }
    private:
        __aicore__ inline void CopyIn(int32_t baseOffset, int32_t tileOffset, int32_t length) {
            this->xLocal = this->inQueueX.template AllocTensor<T1>();

            // DataCopy(this->x1Local[0], this->x1Gm[baseOffset + tileOffset], length);
            // DataCopy(this->x2Local[0], this->x2Gm[baseOffset + tileOffset], length);

            DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T1)), 0, 0, 0};
            DataCopyPad(this->xLocal[0], this->xGm[baseOffset + tileOffset], copyParams, padParams);

            this->inQueueX.EnQue(this->xLocal);
        }

        __aicore__ inline void Compute(int32_t baseOffset, int32_t tileOffset, int32_t length) {
            LocalTensor<T1> xLocal = inQueueX.DeQue<T1>();
            LocalTensor<T1> yLocal = outQueueY.AllocTensor<T1>();

            LocalTensor<float> tmp1 = tmpBuffer1.Get<float>();
            LocalTensor<float> tmp2 = tmpBuffer2.Get<float>();
            LocalTensor<float> tmp3 = tmpBuffer3.Get<float>();
            
            float a0 = 0.044715f, a1 = 0.7978845608028654f, a2 = -2.0f;

            Cast(tmp3, xLocal, RoundMode::CAST_NONE, length);

            // tanh inner
            Mul(tmp1, tmp3, tmp3, length);
            Mul(tmp1, tmp1, tmp3, length);
            Muls(tmp1, tmp1, a0, length);
            Add(tmp1, tmp1, tmp3, length);
            Muls(tmp1, tmp1, a1, length);

            // tanh(x) = 2 / (1 + exp(-2x)) - 1
            Muls(tmp1, tmp1, a2, length);
            Exp(tmp1, tmp1, length);
            Adds(tmp1, tmp1, 1.0f, length);
            Div(tmp2, tmp3, tmp1, length);

            Cast(yLocal, tmp2, RoundMode::CAST_NONE, length);

            outQueueY.EnQue<T1>(yLocal);
            inQueueX.FreeTensor(xLocal);
        }
    
        __aicore__ inline void CopyOut(int32_t baseOffset, int32_t tileOffset, int32_t length) {
            this->yLocal = outQueueY.DeQue<T1>();

            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T1)), 0, 0, 0};
            DataCopyPad(yGm[baseOffset + tileOffset], this->yLocal[0], copyParams);
    
            outQueueY.FreeTensor(this->yLocal);
        }

    private:
        TPipe *pipe;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
        TBuf<QuePosition::VECCALC> tmpBuffer1, tmpBuffer2, tmpBuffer3;

        LocalTensor<T1> xLocal, yLocal;
        GlobalTensor<T1> xGm, yGm;

        int32_t xTotalLength;
        int32_t tileNum, tileLength, lastTileLength, invalidLength, resTileNum, invalidLenPerResTile;
};

template <class T1>
class KernelGeluV2AtanhForBF16 {
    public:
        __aicore__ inline KernelGeluV2AtanhForBF16() {}
        __aicore__ inline void Init(TPipe *pipeIn, GM_ADDR x, GM_ADDR y, int32_t xTotalLength,  
                                    int32_t tileNum, int32_t tileLength, 
                                    int32_t lastTileLength, int32_t invalidLength, 
                                    int32_t resTileNum, int32_t invalidLenPerResTile) 
        {   
            // 复制 tilingData 到类里面
            this->pipe = pipeIn;
            this->xTotalLength = xTotalLength;
            this->tileNum = tileNum;
            this->tileLength = tileLength;
            this->lastTileLength = lastTileLength;
            this->invalidLength = invalidLength;
            this->resTileNum = resTileNum;
            this->invalidLenPerResTile = invalidLenPerResTile;

            // 设置 Global Memory
            this->xGm.SetGlobalBuffer((__gm__ T1 *)x, this->xTotalLength);
            this->yGm.SetGlobalBuffer((__gm__ T1 *)y, this->xTotalLength);

            // 初始化队列 Buffer
            pipe->InitBuffer(this->inQueueX, BUFFER_NUM, this->tileLength * sizeof(T1));
            pipe->InitBuffer(this->outQueueY, BUFFER_NUM, this->tileLength * sizeof(T1));

            pipe->InitBuffer(tmpBuffer1, this->tileLength * sizeof(float));
            pipe->InitBuffer(tmpBuffer2, this->tileLength * sizeof(float));
            pipe->InitBuffer(tmpBuffer3, this->tileLength * sizeof(float));
        }

        __aicore__ inline void Process() {
            int32_t baseOffset = 0, tileOffset = 0;

            for(int32_t i = 0; i < this->tileNum; ++i){
                CopyIn(baseOffset, tileOffset, this->tileLength);
                Compute(baseOffset, tileOffset, this->tileLength);
                CopyOut(baseOffset, tileOffset, this->tileLength);
                tileOffset += this->tileLength;
            }
            for(int32_t i = 0; i < BUFFER_NUM; ++i){
                CopyIn(baseOffset, tileOffset, this->lastTileLength);
                Compute(baseOffset, tileOffset, this->lastTileLength);
                CopyOut(baseOffset, tileOffset, this->lastTileLength);
                tileOffset += this->lastTileLength;
            }
        }
    private:
        __aicore__ inline void CopyIn(int32_t baseOffset, int32_t tileOffset, int32_t length) {
            this->xLocal = this->inQueueX.template AllocTensor<T1>();

            // DataCopy(this->x1Local[0], this->x1Gm[baseOffset + tileOffset], length);
            // DataCopy(this->x2Local[0], this->x2Gm[baseOffset + tileOffset], length);

            DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T1)), 0, 0, 0};
            DataCopyPad(this->xLocal[0], this->xGm[baseOffset + tileOffset], copyParams, padParams);

            this->inQueueX.EnQue(this->xLocal);
        }

        __aicore__ inline void Compute(int32_t baseOffset, int32_t tileOffset, int32_t length) {
            LocalTensor<T1> xLocal = inQueueX.DeQue<T1>();
            LocalTensor<T1> yLocal = outQueueY.AllocTensor<T1>();

            LocalTensor<float> tmp1 = tmpBuffer1.Get<float>();
            LocalTensor<float> tmp2 = tmpBuffer2.Get<float>();
            LocalTensor<float> tmp3 = tmpBuffer3.Get<float>();
            
            float a0 = 0.044715f, a1 = 0.7978845608028654f, a2 = -2.0f;

            Cast(tmp3, xLocal, RoundMode::CAST_NONE, length);

            // tanh inner
            Mul(tmp1, tmp3, tmp3, length);
            Mul(tmp1, tmp1, tmp3, length);
            Muls(tmp1, tmp1, a0, length);
            Add(tmp1, tmp1, tmp3, length);
            Muls(tmp1, tmp1, a1, length);

            // tanh(x) = 2 / (1 + exp(-2x)) - 1
            Muls(tmp1, tmp1, a2, length);
            Exp(tmp1, tmp1, length);
            Adds(tmp1, tmp1, 1.0f, length);
            Div(tmp2, tmp3, tmp1, length);

            Cast(yLocal, tmp2, RoundMode::CAST_RINT, length);

            outQueueY.EnQue<T1>(yLocal);
            inQueueX.FreeTensor(xLocal);
        }
    
        __aicore__ inline void CopyOut(int32_t baseOffset, int32_t tileOffset, int32_t length) {
            this->yLocal = outQueueY.DeQue<T1>();

            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T1)), 0, 0, 0};
            DataCopyPad(yGm[baseOffset + tileOffset], this->yLocal[0], copyParams);
    
            outQueueY.FreeTensor(this->yLocal);
        }

    private:
        TPipe *pipe;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
        TBuf<QuePosition::VECCALC> tmpBuffer1, tmpBuffer2, tmpBuffer3;

        LocalTensor<T1> xLocal, yLocal;
        GlobalTensor<T1> xGm, yGm;

        int32_t xTotalLength;
        int32_t tileNum, tileLength, lastTileLength, invalidLength, resTileNum, invalidLenPerResTile;
};

extern "C" __global__ __aicore__ void gelu_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    if(TILING_KEY_IS(0)){
        GET_TILING_DATA_WITH_STRUCT(GeluV2TilingData1, tiling_data, tiling);
        TPipe pipe;
        // KernelGeluV2<float> op;
        KernelGeluV2_V2<float> op;
        op.Init(&pipe, x, y, tiling_data.xTotalLength, tiling_data.tileNum, tiling_data.tileLength, tiling_data.lastTileLength, tiling_data.invalidLength, tiling_data.resTileNum, tiling_data.invalidLenPerResTile);
        op.Process();
    }else if(TILING_KEY_IS(1)){
        GET_TILING_DATA_WITH_STRUCT(GeluV2TilingData1, tiling_data, tiling);
        TPipe pipe;
        KernelGeluV2ForBF16<bfloat16_t> op;
        op.Init(&pipe, x, y, tiling_data.xTotalLength, tiling_data.tileNum, tiling_data.tileLength, tiling_data.lastTileLength, tiling_data.invalidLength, tiling_data.resTileNum, tiling_data.invalidLenPerResTile);
        op.Process();
    }else if(TILING_KEY_IS(2)){
        GET_TILING_DATA_WITH_STRUCT(GeluV2TilingData1, tiling_data, tiling);
        TPipe pipe;
        // KernelGeluV2ForHalf<half> op;
        KernelGeluV2ForHalf_V2<half> op;
        op.Init(&pipe, x, y, tiling_data.xTotalLength, tiling_data.tileNum, tiling_data.tileLength, tiling_data.lastTileLength, tiling_data.invalidLength, tiling_data.resTileNum, tiling_data.invalidLenPerResTile);
        op.Process();
    }else if(TILING_KEY_IS(3)){
        GET_TILING_DATA_WITH_STRUCT(GeluV2TilingData1, tiling_data, tiling);
        TPipe pipe;
        KernelGeluV2Atanh<float> op;
        op.Init(&pipe, x, y, tiling_data.xTotalLength, tiling_data.tileNum, tiling_data.tileLength, tiling_data.lastTileLength, tiling_data.invalidLength, tiling_data.resTileNum, tiling_data.invalidLenPerResTile);
        op.Process();
    }else if(TILING_KEY_IS(4)){
        GET_TILING_DATA_WITH_STRUCT(GeluV2TilingData1, tiling_data, tiling);
        TPipe pipe;
        KernelGeluV2AtanhForBF16<bfloat16_t> op;
        op.Init(&pipe, x, y, tiling_data.xTotalLength, tiling_data.tileNum, tiling_data.tileLength, tiling_data.lastTileLength, tiling_data.invalidLength, tiling_data.resTileNum, tiling_data.invalidLenPerResTile);
        op.Process();
    }else if(TILING_KEY_IS(5)){
        GET_TILING_DATA_WITH_STRUCT(GeluV2TilingData1, tiling_data, tiling);
        TPipe pipe;
        KernelGeluV2AtanhForHalf<half> op;
        op.Init(&pipe, x, y, tiling_data.xTotalLength, tiling_data.tileNum, tiling_data.tileLength, tiling_data.lastTileLength, tiling_data.invalidLength, tiling_data.resTileNum, tiling_data.invalidLenPerResTile);
        op.Process();
    }
}
