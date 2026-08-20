#include "kernel_operator.h"

using namespace AscendC;

__aicore__ inline int align_to(int a, int b) {
    return (a + b - 1) / b * b;
}

class AtanhFloat {
public:
    __aicore__ inline AtanhFloat() {}
    
    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR input, GM_ADDR out)
    {
        pipe = pipeIn;
        this->inputGm.SetGlobalBuffer((__gm__ float *)input);
        this->outGm.SetGlobalBuffer((__gm__ float *)out);
        
        pipe->InitBuffer(inQueue, 2, 49152);
        pipe->InitBuffer(outQueue, 2, 49152);
    }
    
    __aicore__ inline void Process(int start, int len, int flag)
    {
        int loop = (len + chunksize - 1) / chunksize;
        int tile = len % chunksize;
        tile = tile == 0 ? chunksize : tile;
        if(flag) {
            for(int i = 0; i != loop - 1; i++) {
                CopyIn(start + i * chunksize, chunksize);
                Compute(chunksize);
                CopyOut(start + i * chunksize, chunksize);
            }
            CopyIn(start + (loop - 1) * chunksize, align_to(tile, 32));
            Compute(align_to(tile, 32));
            if(tile % 32) {
                CopyOutPad(start + (loop - 1) * chunksize, tile);
            } else {
                CopyOut(start + (loop - 1) * chunksize, tile);
            }
        } else {
            CopyIn(start + (loop - 1) * chunksize, align_to(tile, 32));
            Compute(align_to(tile, 32));
            if(tile % 32) {
                CopyOutPad(start + (loop - 1) * chunksize, tile);
            } else {
                CopyOut(start + (loop - 1) * chunksize, tile);
            }
            for(int i = loop - 2; i != - 1; i--) {
                CopyIn(start + i * chunksize, chunksize);
                Compute(chunksize);
                CopyOut(start + i * chunksize, chunksize);
            }
        }
    }
    
    
private:
    __aicore__ inline void CopyIn(int offset, int len)
    {
        LocalTensor<float> aLocal = inQueue.AllocTensor<float>();
        DataCopy(aLocal, inputGm[offset], len);
        inQueue.EnQue(aLocal);
    }

    __aicore__ inline void Compute(int len)
    {
        LocalTensor<float> aLocal = inQueue.DeQue<float>();
        LocalTensor<float> cLocal = outQueue.AllocTensor<float>();

        Adds(cLocal, aLocal, 1.0f, len);
        Adds(aLocal, aLocal, -1.0f, len);
        Muls(aLocal, aLocal, -1.0f, len);
        Div(cLocal, cLocal, aLocal, len);
        Ln(cLocal, cLocal, len);
        Muls(cLocal, cLocal, 0.5f, len);

        outQueue.EnQue(cLocal);
        inQueue.FreeTensor(aLocal);
    }
    
    __aicore__ inline void CopyOut(int offset, int len)
    {
        LocalTensor<float> cLocal = outQueue.DeQue<float>();
        DataCopy(outGm[offset], cLocal, len);
        outQueue.FreeTensor(cLocal);
    }
    
    __aicore__ inline void CopyOutPad(int offset, int len)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * 4), 0, 0, 0};
        LocalTensor<float> cLocal = outQueue.DeQue<float>();
        DataCopyPad(outGm[offset], cLocal, copyParams);
        outQueue.FreeTensor(cLocal);
    }

private:
    static constexpr int BUFFER_NUM = 2;
    static constexpr int chunksize = 12288;
    
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    
    GlobalTensor<float> inputGm;
    GlobalTensor<float> outGm;
};

class AtanhHalf {
public:
    __aicore__ inline AtanhHalf() {}
    
    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR input, GM_ADDR out)
    {
        pipe = pipeIn;
        this->inputGm.SetGlobalBuffer((__gm__ half *)input);
        this->outGm.SetGlobalBuffer((__gm__ half *)out);
        
        pipe->InitBuffer(inQueue, 2, 32768);
        pipe->InitBuffer(outQueue, 2, 32768);
        pipe->InitBuffer(workBuf, 65536);
    }
    
    __aicore__ inline void Process(int start, int len, int flag)
    {
        tempTensor = workBuf.Get<float>();

        int loop = (len + chunksize - 1) / chunksize;
        int tile = len % chunksize;
        tile = tile == 0 ? chunksize : tile;
        if(flag) {
            for(int i = 0; i != loop - 1; i++) {
                CopyIn(start + i * chunksize, chunksize);
                Compute(chunksize);
                CopyOut(start + i * chunksize, chunksize);
            }
            CopyIn(start + (loop - 1) * chunksize, align_to(tile, 32));
            Compute(align_to(tile, 32));
            if(tile % 32) {
                CopyOutPad(start + (loop - 1) * chunksize, tile);
            } else {
                CopyOut(start + (loop - 1) * chunksize, tile);
            }
        } else {
            CopyIn(start + (loop - 1) * chunksize, align_to(tile, 32));
            Compute(align_to(tile, 32));
            if(tile % 32) {
                CopyOutPad(start + (loop - 1) * chunksize, tile);
            } else {
                CopyOut(start + (loop - 1) * chunksize, tile);
            }
            for(int i = loop - 2; i != - 1; i--) {
                CopyIn(start + i * chunksize, chunksize);
                Compute(chunksize);
                CopyOut(start + i * chunksize, chunksize);
            }
        }
    }
    
    
private:
    __aicore__ inline void CopyIn(int offset, int len)
    {
        LocalTensor<half> aLocal = inQueue.AllocTensor<half>();
        DataCopy(aLocal, inputGm[offset], len);
        inQueue.EnQue(aLocal);
    }

    __aicore__ inline void Compute(int len)
    {
        LocalTensor<half> aLocal = inQueue.DeQue<half>();
        LocalTensor<half> cLocal = outQueue.AllocTensor<half>();

        Adds(cLocal, aLocal, half(1.0f), len);
        Adds(aLocal, aLocal, half(-1.0f), len);
        Muls(aLocal, aLocal, half(-1.0f), len);
        Div(cLocal, cLocal, aLocal, len);
        Cast(tempTensor, cLocal, RoundMode::CAST_NONE, len);
        Ln(tempTensor, tempTensor, len);
        Muls(tempTensor, tempTensor, 0.5f, len);
        Cast(cLocal, tempTensor, RoundMode::CAST_RINT, len);

        outQueue.EnQue(cLocal);
        inQueue.FreeTensor(aLocal);
    }
    
    __aicore__ inline void CopyOut(int offset, int len)
    {
        LocalTensor<half> cLocal = outQueue.DeQue<half>();
        DataCopy(outGm[offset], cLocal, len);
        outQueue.FreeTensor(cLocal);
    }
    
    __aicore__ inline void CopyOutPad(int offset, int len)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * 2), 0, 0, 0};
        LocalTensor<half> cLocal = outQueue.DeQue<half>();
        DataCopyPad(outGm[offset], cLocal, copyParams);
        outQueue.FreeTensor(cLocal);
    }

private:
    static constexpr int BUFFER_NUM = 2;
    static constexpr int chunksize = 16384;
    
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> workBuf;
    
    GlobalTensor<half> inputGm;
    GlobalTensor<half> outGm;
    LocalTensor<float> tempTensor;
};

class AtanhBf16 {
public:
    __aicore__ inline AtanhBf16() {}
    
    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR input, GM_ADDR out)
    {
        pipe = pipeIn;
        this->inputGm.SetGlobalBuffer((__gm__ bfloat16_t *)input);
        this->outGm.SetGlobalBuffer((__gm__ bfloat16_t *)out);
        
        pipe->InitBuffer(inQueue, 2, 24576);
        pipe->InitBuffer(outQueue, 2, 24576);
        pipe->InitBuffer(workBuf, 98304);
    }
    
    __aicore__ inline void Process(int start, int len, int flag)
    {
        tempTensor = workBuf.Get<float>();

        int loop = (len + chunksize - 1) / chunksize;
        int tile = len % chunksize;
        tile = tile == 0 ? chunksize : tile;
        if(flag) {
            for(int i = 0; i != loop - 1; i++) {
                CopyIn(start + i * chunksize, chunksize);
                Compute(chunksize);
                CopyOut(start + i * chunksize, chunksize);
            }
            CopyIn(start + (loop - 1) * chunksize, align_to(tile, 32));
            Compute(align_to(tile, 32));
            if(tile % 32) {
                CopyOutPad(start + (loop - 1) * chunksize, tile);
            } else {
                CopyOut(start + (loop - 1) * chunksize, tile);
            }
        } else {
            CopyIn(start + (loop - 1) * chunksize, align_to(tile, 32));
            Compute(align_to(tile, 32));
            if(tile % 32) {
                CopyOutPad(start + (loop - 1) * chunksize, tile);
            } else {
                CopyOut(start + (loop - 1) * chunksize, tile);
            }
            for(int i = loop - 2; i != - 1; i--) {
                CopyIn(start + i * chunksize, chunksize);
                Compute(chunksize);
                CopyOut(start + i * chunksize, chunksize);
            }
        }
    }
    
    
private:
    __aicore__ inline void CopyIn(int offset, int len)
    {
        LocalTensor<bfloat16_t> aLocal = inQueue.AllocTensor<bfloat16_t>();
        DataCopy(aLocal, inputGm[offset], len);
        inQueue.EnQue(aLocal);
    }

    __aicore__ inline void Compute(int len)
    {
        LocalTensor<bfloat16_t> aLocal = inQueue.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> cLocal = outQueue.AllocTensor<bfloat16_t>();

        Cast(tempTensor, aLocal, RoundMode::CAST_NONE, len);
        inQueue.FreeTensor(aLocal);
        Adds(tempTensor[chunksize], tempTensor, 1.0f, len);
        Adds(tempTensor, tempTensor, -1.0f, len);
        Muls(tempTensor, tempTensor, -1.0f, len);
        Div(tempTensor[chunksize], tempTensor[chunksize], tempTensor, len);
        Ln(tempTensor[chunksize], tempTensor[chunksize], len);
        Muls(tempTensor[chunksize], tempTensor[chunksize], 0.5f, len);
        Cast(cLocal, tempTensor[chunksize], RoundMode::CAST_RINT, len);

        outQueue.EnQue(cLocal);
    }
    
    __aicore__ inline void CopyOut(int offset, int len)
    {
        LocalTensor<bfloat16_t> cLocal = outQueue.DeQue<bfloat16_t>();
        DataCopy(outGm[offset], cLocal, len);
        outQueue.FreeTensor(cLocal);
    }
    
    __aicore__ inline void CopyOutPad(int offset, int len)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * 2), 0, 0, 0};
        LocalTensor<bfloat16_t> cLocal = outQueue.DeQue<bfloat16_t>();
        DataCopyPad(outGm[offset], cLocal, copyParams);
        outQueue.FreeTensor(cLocal);
    }

private:
    static constexpr int BUFFER_NUM = 2;
    static constexpr int chunksize = 12288;
    
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> workBuf;
    
    GlobalTensor<bfloat16_t> inputGm;
    GlobalTensor<bfloat16_t> outGm;
    LocalTensor<float> tempTensor;
};

extern "C" __global__ __aicore__ void atanh(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    const uint32_t size = tiling_data.size;
    const uint32_t chunk = tiling_data.chunk;
    int blockIdx = GetBlockIdx();
    int blockDim = GetBlockNum();
    int start = blockIdx * chunk;
    int len = min(int(size - start), int(chunk));
    if(len <= 0) return;
    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        AtanhFloat op;
        op.Init(&pipe, input, output);

        GlobalTensor<int32_t> workGm;
        workGm.SetGlobalBuffer((__gm__ int32_t *)workspace);
        op.Process(start, len, workGm(0)%2);
        if(blockIdx == 0) workGm(0) += 1;
    }
    else if (TILING_KEY_IS(2)) {
        AtanhHalf op;
        op.Init(&pipe, input, output);

        GlobalTensor<int32_t> workGm;
        workGm.SetGlobalBuffer((__gm__ int32_t *)workspace);
        op.Process(start, len, workGm(0)%2);
        if(blockIdx == 0) workGm(0) += 1;
    }
    else if (TILING_KEY_IS(3)) {
        AtanhBf16 op;
        op.Init(&pipe, input, output);
        
        GlobalTensor<int32_t> workGm;
        workGm.SetGlobalBuffer((__gm__ int32_t *)workspace);
        op.Process(start, len, workGm(0)%2);
        if(blockIdx == 0) workGm(0) += 1;
    }
}