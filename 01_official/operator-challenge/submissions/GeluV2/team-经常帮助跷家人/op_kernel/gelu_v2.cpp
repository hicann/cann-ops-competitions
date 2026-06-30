#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;

template <typename T>
class KernelGeluV2None16
{
public:
    __aicore__ inline KernelGeluV2None16() {}
    __aicore__ inline void Init(GM_ADDR src_gm, GM_ADDR dst_gm, const GeluV2TilingData& tiling, TPipe *pipe)
    {
        this->tileLength = tiling.tileDataNum;
        this->tailDataNum = tiling.tailDataNum;
        this->tileNum = tiling.tileNum;
        this->inputNum = tiling.inputNum;
        this->pipe = pipe;
        src_global.SetGlobalBuffer((__gm__ T *)src_gm, inputNum);
        dst_global.SetGlobalBuffer((__gm__ T *)dst_gm, inputNum);

        pipe->InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe->InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe->InitBuffer(tmpxBuf, this->tileLength * sizeof(float));
        pipe->InitBuffer(tmpyBuf, this->tileLength * sizeof(float));
        // pipe.InitBuffer(tmpcalBuf, tileLength * sizeof(float));
        pipe->InitBuffer(tmpSquBuf, tileLength * sizeof(float));
        pipe->InitBuffer(signBuf, tileLength * sizeof(float));
        pipe->InitBuffer(maskBuf, tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount - 1; i++)
        {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        auto length = this->tailDataNum;
        if (!length) return ;
        CopyIn(loopCount-1, length);
        Compute(loopCount-1, length);
        CopyOut(loopCount-1, length);
    }

private:
    __aicore__ inline void CopyIn(uint32_t process, uint32_t length)
    {
        LocalTensor<T> srcLocal = inQueueX.AllocTensor<T>();
        DataCopy(srcLocal, src_global[process * this->tileLength], length);
        inQueueX.EnQue(srcLocal);

    }
    __aicore__ inline void Compute(uint32_t process, uint32_t length)
    {
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        LocalTensor<T> srcLocal = inQueueX.DeQue<T>();
        LocalTensor<float> tmpxLocal = tmpxBuf.Get<float>();
        LocalTensor<float> tmpyLocal = tmpyBuf.Get<float>();
        // LocalTensor<float> tmpcalLocal = tmpcalBuf.Get<float>();
        LocalTensor<float> tmpSquLocal = tmpSquBuf.Get<float>();
        LocalTensor<float> signLocal = signBuf.Get<float>();
        LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
        Cast(tmpxLocal, srcLocal, RoundMode::CAST_NONE, length);
        CompareScalar(maskLocal, tmpxLocal, static_cast<float>(0), CMPMODE::GT, length);
        Duplicate(tmpyLocal, static_cast<float>(1), length);
        Select(signLocal, maskLocal, tmpyLocal, static_cast<float>(-1), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        //calc t and exp_term
        Abs(tmpxLocal, tmpxLocal, length);
        Mul(tmpSquLocal, tmpxLocal, tmpxLocal, length);
        Muls(tmpSquLocal, tmpSquLocal, static_cast<float>(-0.5), length);
        Exp(tmpSquLocal, tmpSquLocal, length);
        Muls(tmpxLocal, tmpxLocal, static_cast<float>(this->p), length);
        Adds(tmpxLocal, tmpxLocal, static_cast<float>(1.0), length);
        Div(tmpxLocal, tmpyLocal, tmpxLocal, length);
        //calc poly
        // Muls(tmpyLocal, tmpxLocal, (float)this->a, length);
        // Mul(tmpcalLocal, tmpxLocal, tmpxLocal, length);
        // Muls(tmpcalLocal, tmpcalLocal, (float)this->b, length);
        // Add(tmpyLocal, tmpyLocal, tmpcalLocal, length);

        // Mul(tmpcalLocal, tmpxLocal, tmpxLocal, length);
        // Mul(tmpcalLocal, tmpcalLocal, tmpxLocal, length);
        // Muls(tmpcalLocal, tmpcalLocal, (float)this->c, length);
        // Add(tmpyLocal, tmpyLocal, tmpcalLocal, length);
        Muls(tmpyLocal, tmpxLocal, static_cast<float>(this->c), length);        // c * t
        Adds(tmpyLocal, tmpyLocal, static_cast<float>(this->b), length);         // b + c * t
        Mul(tmpyLocal, tmpyLocal, tmpxLocal, length);               // t * (b + c * t)
        Adds(tmpyLocal, tmpyLocal, static_cast<float>(this->a), length);         // a + t * (b + c * t)
        Mul(tmpyLocal, tmpyLocal, tmpxLocal, length);               // t * (a + t * (b + c * t))

        //calc erf
        Mul(tmpyLocal, tmpyLocal, tmpSquLocal, length);
        Muls(tmpyLocal, tmpyLocal, static_cast<float>(-1.0), length);
        Adds(tmpyLocal, tmpyLocal, static_cast<float>(1.0), length);
        Mul(tmpyLocal, tmpyLocal, signLocal, length);
        //calc ans
        Adds(tmpyLocal, tmpyLocal, static_cast<float>(1.0), length);
        Cast(tmpxLocal, srcLocal, RoundMode::CAST_NONE, length);
        Mul(tmpyLocal, tmpyLocal, tmpxLocal, length);
        Muls(tmpyLocal, tmpyLocal, static_cast<float>(0.5), length);
        Cast(dstLocal, tmpyLocal, RoundMode::CAST_ROUND, length);

        outQueue.EnQue<T>(dstLocal);
        inQueueX.FreeTensor(srcLocal);
    }

    __aicore__ inline void CopyOut(uint32_t process, uint32_t length)
    {
        LocalTensor<T> dstLocal = outQueue.DeQue<T>();
        DataCopy(dst_global[process * this->tileLength], dstLocal, length);
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<T> src_global;
    GlobalTensor<T> dst_global;

    TPipe *pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<QuePosition::VECCALC> tmpxBuf;
    TBuf<QuePosition::VECCALC> tmpyBuf;
    
    TBuf<QuePosition::VECCALC> tmpSquBuf;
    TBuf<QuePosition::VECCALC> signBuf;
    TBuf<QuePosition::VECCALC> maskBuf;
    // TBuf<QuePosition::VECCALC> tmpcalBuf;

    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t inputNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;

    float p = 0.332671;
    float a = 0.3480242;
    float b = -0.0958798;
    float c = 0.7478556;
};

class KernelGeluV2None32
{
public:
    __aicore__ inline KernelGeluV2None32() {}
    __aicore__ inline void Init(GM_ADDR src_gm, GM_ADDR dst_gm, const GeluV2TilingData& tiling, TPipe *pipe)
    {
        this->tileLength = tiling.tileDataNum;
        this->tailDataNum = tiling.tailDataNum;
        this->tileNum = tiling.tileNum;
        this->inputNum = tiling.inputNum;
        this->pipe = pipe;
        src_global.SetGlobalBuffer((__gm__ float *)src_gm, inputNum);
        dst_global.SetGlobalBuffer((__gm__ float *)dst_gm, inputNum);

        pipe->InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe->InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(float));
        // pipe.InitBuffer(tmpcalBuf, tileLength * sizeof(float));
        pipe->InitBuffer(tmpSquBuf, tileLength * sizeof(float));
        pipe->InitBuffer(signBuf, tileLength * sizeof(float));
        pipe->InitBuffer(maskBuf, tileLength * sizeof(uint8_t));
        pipe->InitBuffer(tmpBuf, tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount - 1; i++)
        {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        auto length = this->tailDataNum;
        if (!length) return ;
        CopyIn(loopCount-1, length);
        Compute(loopCount-1, length);
        CopyOut(loopCount-1, length);
    }

private:
    __aicore__ inline void CopyIn(uint32_t process, uint32_t length)
    {
        LocalTensor<float> srcLocal = inQueueX.AllocTensor<float>();
        DataCopy(srcLocal, src_global[process * this->tileLength], length);
        inQueueX.EnQue(srcLocal);

    }
    __aicore__ inline void Compute(uint32_t process, uint32_t length)
    {
        LocalTensor<float> dstLocal = outQueue.AllocTensor<float>();
        LocalTensor<float> srcLocal = inQueueX.DeQue<float>();
        // LocalTensor<float> tmpcalLocal = tmpcalBuf.Get<float>();
        LocalTensor<float> tmpSquLocal = tmpSquBuf.Get<float>();
        LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
        LocalTensor<float> signLocal = signBuf.Get<float>();
        LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
        Muls(tmpLocal, srcLocal, static_cast<float>(0.5), length);
        CompareScalar(maskLocal, srcLocal, static_cast<float>(0), CMPMODE::GT, length);
        Duplicate(dstLocal, static_cast<float>(1), length);
        Select(signLocal, maskLocal, dstLocal, static_cast<float>(-1), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        //calc float and exp_term
        Abs(srcLocal, srcLocal, length);
        Mul(tmpSquLocal, srcLocal, srcLocal, length);
        Muls(tmpSquLocal, tmpSquLocal, static_cast<float>(-0.5), length);
        Exp(tmpSquLocal, tmpSquLocal, length);
        Muls(srcLocal, srcLocal, static_cast<float>(this->p), length);
        Adds(srcLocal, srcLocal, static_cast<float>(1.0), length);
        Div(srcLocal, dstLocal, srcLocal, length);
        //calc poly
        // Muls(dstLocal, srcLocal, (float)this->a, length);
        // Mul(tmpcalLocal, srcLocal, srcLocal, length);
        // Muls(tmpcalLocal, tmpcalLocal, (float)this->b, length);
        // Add(dstLocal, dstLocal, tmpcalLocal, length);

        // Mul(tmpcalLocal, srcLocal, srcLocal, length);
        // Mul(tmpcalLocal, tmpcalLocal, srcLocal, length);
        // Muls(tmpcalLocal, tmpcalLocal, (float)this->c, length);
        // Add(dstLocal, dstLocal, tmpcalLocal, length);
        Muls(dstLocal, srcLocal, static_cast<float>(this->c), length);        // c * t
        Adds(dstLocal, dstLocal, static_cast<float>(this->b), length);         // b + c * t
        Mul(dstLocal, dstLocal, srcLocal, length);               // t * (b + c * t)
        Adds(dstLocal, dstLocal, static_cast<float>(this->a), length);         // a + t * (b + c * t)
        Mul(dstLocal, dstLocal, srcLocal, length);               // t * (a + t * (b + c * t))

        //calc erf
        Mul(dstLocal, dstLocal, tmpSquLocal, length);
        Muls(dstLocal, dstLocal, static_cast<float>(-1.0), length);
        Adds(dstLocal, dstLocal, static_cast<float>(1.0), length);
        Mul(dstLocal, dstLocal, signLocal, length);
        //calc ans
        Adds(dstLocal, dstLocal, static_cast<float>(1.0), length);
        Mul(dstLocal, dstLocal, tmpLocal, length);

        outQueue.EnQue<float>(dstLocal);
        inQueueX.FreeTensor(srcLocal);
    }

    __aicore__ inline void CopyOut(uint32_t process, uint32_t length)
    {
        LocalTensor<float> dstLocal = outQueue.DeQue<float>();
        DataCopy(dst_global[process * this->tileLength], dstLocal, length);
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<float> src_global;
    GlobalTensor<float> dst_global;

    TPipe *pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<QuePosition::VECCALC> tmpxQueue;
    TBuf<QuePosition::VECCALC> tmpyQueue;
    TBuf<QuePosition::VECCALC> tmpSquBuf;
    TBuf<QuePosition::VECCALC> signBuf;
    TBuf<QuePosition::VECCALC> maskBuf;
    // TBuf<QuePosition::VECCALC> tmpcalBuf;
    TBuf<QuePosition::VECCALC> tmpBuf;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t inputNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;

    float p = 0.332671;
    float a = 0.3480242;
    float b = -0.0958798;
    float c = 0.7478556;
};

template <typename T>
class KernelGeluV2Atanh16
{
public:
    __aicore__ inline KernelGeluV2Atanh16() {}
    __aicore__ inline void Init(GM_ADDR src_gm, GM_ADDR dst_gm, const GeluV2TilingData& tiling, TPipe *pipe)
    {
        this->tileLength = tiling.tileDataNum;
        this->tileNum = tiling.tileNum;
        this->inputNum = tiling.inputNum;
        this->tailDataNum = tiling.tailDataNum;
        this->pipe = pipe;
        src_global.SetGlobalBuffer((__gm__ T *)src_gm, inputNum);
        dst_global.SetGlobalBuffer((__gm__ T *)dst_gm, inputNum);

        pipe->InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe->InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe->InitBuffer(tmpxBuf, this->tileLength * sizeof(float));
        pipe->InitBuffer(tmpyBuf, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount - 1; i++)
        {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        auto length = this->tailDataNum;
        if (!length) return ;
        CopyIn(loopCount-1, length);
        Compute(loopCount-1, length);
        CopyOut(loopCount-1, length);
    }

private:
    __aicore__ inline void CopyIn(uint32_t process, uint32_t length)
    {
        LocalTensor<T> srcLocal = inQueueX.AllocTensor<T>();
        DataCopy(srcLocal, src_global[process * this->tileLength], length);
        inQueueX.EnQue(srcLocal);

    }
    __aicore__ inline void Compute(uint32_t process, uint32_t length)
    {
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        LocalTensor<float> tmpxLocal = tmpxBuf.Get<float>();
        LocalTensor<float> tmpyLocal = tmpyBuf.Get<float>();
        LocalTensor<T> srcLocal = inQueueX.DeQue<T>();
        
        Cast(tmpxLocal, srcLocal, RoundMode::CAST_NONE, length);

        Mul(tmpyLocal, tmpxLocal, tmpxLocal, length);
        Mul(tmpyLocal, tmpyLocal, tmpxLocal, length);
        Muls(tmpyLocal, tmpyLocal, static_cast<float>(this->param1), length);
        Add(tmpyLocal, tmpyLocal, tmpxLocal, length);
        Muls(tmpyLocal, tmpyLocal, static_cast<float>(this->param2), length);
        Exp(tmpyLocal,tmpyLocal,length);
        Adds(tmpyLocal, tmpyLocal, static_cast<float>(1), length);
        Div(tmpyLocal,tmpxLocal,tmpyLocal,length);

        Cast(dstLocal, tmpyLocal, RoundMode::CAST_ROUND, length);
        outQueue.EnQue<T>(dstLocal);
        inQueueX.FreeTensor(srcLocal);
    }

    __aicore__ inline void CopyOut(uint32_t process, uint32_t length)
    {
        LocalTensor<T> dstLocal = outQueue.DeQue<T>();
        DataCopy(dst_global[process * this->tileLength], dstLocal, length);
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<T> src_global;
    GlobalTensor<T> dst_global;

    TPipe *pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TBuf<QuePosition::VECCALC> tmpxBuf;
    TBuf<QuePosition::VECCALC> tmpyBuf;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t inputNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    float param1 = 0.044715;
    float param2 = -1.595769122;
};

class KernelGeluV2Atanh32
{
public:
    __aicore__ inline KernelGeluV2Atanh32() {}
    __aicore__ inline void Init(GM_ADDR src_gm, GM_ADDR dst_gm, const GeluV2TilingData& tiling, TPipe *pipe)
    {
        this->tileLength = tiling.tileDataNum;
        this->tileNum = tiling.tileNum;
        this->inputNum = tiling.inputNum;
        this->tailDataNum = tiling.tailDataNum;
        this->pipe = pipe;
        src_global.SetGlobalBuffer((__gm__ float *)src_gm, inputNum);
        dst_global.SetGlobalBuffer((__gm__ float *)dst_gm, inputNum);

        pipe->InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe->InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(float));
        
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount - 1; i++)
        {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        auto length = this->tailDataNum;
        CopyIn(loopCount-1, length);
        Compute(loopCount-1, length);
        CopyOut(loopCount-1, length);
    }

private:
    __aicore__ inline void CopyIn(uint32_t process, uint32_t length)
    {
        LocalTensor<float> srcLocal = inQueueX.AllocTensor<float>();
        DataCopy(srcLocal, src_global[process * this->tileLength], length);
        inQueueX.EnQue<float>(srcLocal);

    }
    __aicore__ inline void Compute(uint32_t process, uint32_t length)
    {
        LocalTensor<float> dstLocal = outQueue.AllocTensor<float>();
        LocalTensor<float> srcLocal = inQueueX.DeQue<float>();
        Mul(dstLocal, srcLocal, srcLocal, length);
        Mul(dstLocal, dstLocal, srcLocal, length);
        Muls(dstLocal, dstLocal, static_cast<float>(this->param1), length);
        Add(dstLocal, dstLocal, srcLocal, length);
        Muls(dstLocal, dstLocal, static_cast<float>(this->param2), length);
        Exp(dstLocal,dstLocal,length);
        Adds(dstLocal, dstLocal, static_cast<float>(1), length);
        Div(dstLocal,srcLocal,dstLocal,length);

        outQueue.EnQue<float>(dstLocal);
        inQueueX.FreeTensor(srcLocal);
    }

    __aicore__ inline void CopyOut(uint32_t process, uint32_t length)
    {
        LocalTensor<float> dstLocal = outQueue.DeQue<float>();
        DataCopy(dst_global[process * this->tileLength], dstLocal, length);
        outQueue.FreeTensor(dstLocal);
    }

private:
    TPipe *pipe;
    GlobalTensor<float> src_global;
    GlobalTensor<float> dst_global;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t inputNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;

    float param1 = 0.044715;
    float param2 = -1.595769122;
};
// x/(1+e(-2*(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3)))))
extern "C" __global__ __aicore__ void gelu_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    uint32_t inputTypeLen = tiling_data.inputTypeLen;
    if (TILING_KEY_IS(1)) 
    {
        if (inputTypeLen == 2) {
            KernelGeluV2None16<DTYPE_X> op;
            op.Init(x, y, tiling_data, &pipe);
            op.Process();
        }
        else {
            KernelGeluV2None32 op;
            op.Init(x, y, tiling_data, &pipe);
            op.Process();
        }
    }
    else if (TILING_KEY_IS(2)) 
    {
        if (inputTypeLen == 2) {
            KernelGeluV2Atanh16<DTYPE_X> op;
            op.Init(x, y, tiling_data, &pipe);
            op.Process();
        }
        else {
            KernelGeluV2Atanh32 op;
            op.Init(x, y, tiling_data, &pipe);
            op.Process();
        }
    }
}
