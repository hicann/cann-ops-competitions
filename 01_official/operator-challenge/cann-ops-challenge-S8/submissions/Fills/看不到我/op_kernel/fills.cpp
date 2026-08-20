#include "kernel_operator.h"
using namespace AscendC;

class Fills {
public:
    __aicore__ inline Fills() {}

    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR out, uint32_t value_bits) {
        pipe = pipeIn;
        outGm.SetGlobalBuffer((__gm__ float *)out);

        // 把 32-bit pattern 当作 float 看；Duplicate / DataCopy 都是按 bit 拷贝，
        // 对 fp16 / bf16 / int* 而言这就是把打包后的 4 字节常量原样灌满 buffer。
        union { uint32_t u; float f; } conv;
        conv.u = value_bits;
        value = conv.f;

        pipe->InitBuffer(outQueue, 1, 196608);
    }

    __aicore__ inline void Process(int start, int len) {
        aLocal = outQueue.AllocTensor<float>();
        Duplicate(aLocal, value, min(len, chunksize));
        outQueue.EnQue(aLocal);
        outQueue.DeQue<float>();
    
        int loop = (len + chunksize - 1) / chunksize;
        int tile = len - (loop - 1) * chunksize;        // 直接算，省掉 %  和三目
        for (int i = 0; i != loop - 1; i++) {
            DataCopy(outGm[start + i * chunksize], aLocal, chunksize);
        }
        DataCopy(outGm[start + (loop - 1) * chunksize], aLocal, tile);
    
        outQueue.FreeTensor(aLocal);
    }

private:
    int chunksize = 49152;   // 16384 个 fp32 = 64KB
    float value;

    TPipe* pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> outQueue;

    GlobalTensor<float> outGm;
    LocalTensor<float> aLocal;
};

extern "C" __global__ __aicore__ void fills(GM_ADDR input, GM_ADDR output,
                                              GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    const uint32_t size       = tiling_data.size;        // 32-bit 元素个数
    const uint32_t chunk      = tiling_data.chunk;       // 同上
    const uint32_t value_bits = tiling_data.value_bits;

    int blockIdx = GetBlockIdx();
    int start = blockIdx * int(chunk);
    int len   = min(int(size) - start, int(chunk));
    if (len <= 0) return;

    TPipe pipe;
    Fills op;
    op.Init(&pipe, output, value_bits);
    op.Process(start, len);
}