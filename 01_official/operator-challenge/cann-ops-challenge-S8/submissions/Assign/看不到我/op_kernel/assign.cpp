#include "kernel_operator.h"

using namespace AscendC;

__aicore__ inline int align_to(int a, int b) {
    return (a + b - 1) / b * b;
}

template<typename T>
class Assign_NB {
public:
    __aicore__ inline Assign_NB() {}
    
    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR input, GM_ADDR other)
    {
        pipe = pipeIn;
        this->inputGm.SetGlobalBuffer((__gm__ T *)input);
        this->otherGm.SetGlobalBuffer((__gm__ T *)other);
        
        pipe->InitBuffer(queBindA, 2, 32768);
        this->chunksize = 32768 / sizeof(T);
    }
    
    __aicore__ inline void Process(int offset, int size, int flag = 0)
    {
        int loop = (size + chunksize - 1) / chunksize;
        int tile = size % chunksize;
        tile = tile == 0 ? chunksize : tile;
        if(flag) {
            for(int i = 0; i != loop - 1; i++) {
                CopyIn(offset + i * chunksize, chunksize);
                CopyOut(offset + i * chunksize, chunksize);
            }
            CopyIn(offset + (loop - 1) * chunksize, align_to(tile, 32));
            if(tile % 32) {
                CopyOutPad(offset + (loop - 1) * chunksize, tile);
            } else {
                CopyOut(offset + (loop - 1) * chunksize, tile);
            }
        } else {
            CopyIn(offset + (loop - 1) * chunksize, align_to(tile, 32));
            if(tile % 32) {
                CopyOutPad(offset + (loop - 1) * chunksize, tile);
            } else {
                CopyOut(offset + (loop - 1) * chunksize, tile);
            }
            for(int i = loop - 2; i != - 1; i--) {
                CopyIn(offset + i * chunksize, chunksize);
                CopyOut(offset + i * chunksize, chunksize);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(int offset, int len)
    {
        LocalTensor<T> aLocal = queBindA.AllocTensor<T>();
        DataCopy(aLocal, otherGm[offset], len);
        queBindA.EnQue(aLocal);
    }

    __aicore__ inline void CopyOut(int offset, int len)
    {
        LocalTensor<T> cLocal = queBindA.DeQue<T>();
        DataCopy(inputGm[offset], cLocal, len);
        queBindA.FreeTensor(cLocal);
    }
    
    __aicore__ inline void CopyOutPad(int offset, int len)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
        LocalTensor<T> cLocal = queBindA.DeQue<T>();
        DataCopyPad(inputGm[offset], cLocal, copyParams);
        queBindA.FreeTensor(cLocal);
    }
    
private:
    int chunksize;
    
    TPipe* pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBindA;
    
    GlobalTensor<T> inputGm;
    GlobalTensor<T> otherGm;
};

// ===================== 广播：按 block 复制/填充（仅保证正确性）=====================
// merge 后 shape：最后一维 other==1 → 该 block 用单值填充，否则连续拷贝一段 other。
template<typename T>
class Assign_B {
public:
    __aicore__ inline Assign_B() {}

    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR input, GM_ADDR other,
                                uint32_t size, uint32_t dims,
                                const uint32_t* ishape, const uint32_t* oshape)
    {
        pipe = pipeIn;
        inputGm.SetGlobalBuffer((__gm__ T *)input);
        otherGm.SetGlobalBuffer((__gm__ T *)other);
        D = dims;
        for (uint32_t i = 0; i < D; i++) { is_[i] = ishape[i]; os_[i] = oshape[i]; }
        ostride[D - 1] = 1;
        for (int d = (int)D - 2; d >= 0; d--) ostride[d] = ostride[d + 1] * os_[d + 1];
        inner = is_[D - 1];
        lastB = (os_[D - 1] == 1 && is_[D - 1] > 1);
        chunksize = 32768 / sizeof(T);
        pipe->InitBuffer(queBindA, 2, 32768);
        pipe->InitBuffer(queOut, 2, 32768);
    }

    __aicore__ inline void Process(uint32_t b0, uint32_t bN)
    {
        for (uint32_t b = b0; b < b0 + bN; b++) {
            // 由 block 索引算 other 偏移（广播维贡献 0）
            uint64_t ob = 0, rem = b;
            for (int d = (int)D - 2; d >= 0; d--) {
                uint32_t id = rem % is_[d];
                rem /= is_[d];
                if (os_[d] != 1) ob += (uint64_t)id * ostride[d];
            }
            uint64_t outoff = (uint64_t)b * inner;
            if (lastB) FillRun(outoff, ob, inner);
            else       CopyRun(ob, outoff, inner);
        }
    }

private:
    // 连续拷贝 len 个 other[soff..] -> input[doff..]
    __aicore__ inline void CopyRun(uint64_t soff, uint64_t doff, uint32_t len) {
        uint32_t off = 0;
        while (off < len) {
            uint32_t cur = (len - off) < (uint32_t)chunksize ? (len - off) : (uint32_t)chunksize;
            LocalTensor<T> buf = queBindA.AllocTensor<T>();
            DataCopy(buf, otherGm[soff + off], align_to((int)cur, 32));
            queBindA.EnQue(buf);
            buf = queBindA.DeQue<T>();
            if (cur % 32) {
                DataCopyExtParams p{1, (uint32_t)(cur * sizeof(T)), 0, 0, 0};
                DataCopyPad(inputGm[doff + off], buf, p);
            } else {
                DataCopy(inputGm[doff + off], buf, cur);
            }
            queBindA.FreeTensor(buf);
            off += cur;
        }
    }

    // 用单个标量 other[soff] 填充 input[doff..doff+len)
    __aicore__ inline void FillRun(uint64_t doff, uint64_t soff, uint32_t len) {
        T val = otherGm.GetValue(soff);
        uint32_t off = 0;
        while (off < len) {
            uint32_t cur = (len - off) < (uint32_t)chunksize ? (len - off) : (uint32_t)chunksize;
            LocalTensor<T> buf = queOut.AllocTensor<T>();   // VECOUT：EnQue 正确插入 vector->MTE3 同步
            if constexpr (sizeof(T) >= 2) {
                Duplicate(buf, val, align_to((int)cur, 32));
            } else {
                // 1 字节：拼 32-bit 模式走 int32 Duplicate
                uint8_t u8 = *((uint8_t*)&val);
                int32_t pat = (int32_t)((uint32_t)u8 * 0x01010101u);
                LocalTensor<int32_t> b32 = buf.template ReinterpretCast<int32_t>();
                uint32_t cnt32 = (cur + 3) / 4;
                Duplicate(b32, pat, align_to((int)cnt32, 8));
            }
            queOut.EnQue(buf);
            buf = queOut.DeQue<T>();
            if (cur % 32) {
                DataCopyExtParams p{1, (uint32_t)(cur * sizeof(T)), 0, 0, 0};
                DataCopyPad(inputGm[doff + off], buf, p);
            } else {
                DataCopy(inputGm[doff + off], buf, cur);
            }
            queOut.FreeTensor(buf);
            off += cur;
        }
    }

    int chunksize;
    TPipe* pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBindA;
    TQue<QuePosition::VECOUT, 1> queOut;
    GlobalTensor<T> inputGm, otherGm;
    uint32_t D, is_[8], os_[8], ostride[8], inner;
    bool lastB;
};

template<typename T>
__aicore__ inline void run_b(TPipe* pipe, GM_ADDR input, GM_ADDR other,
                             uint32_t size, uint32_t dims, const uint32_t* ish, const uint32_t* osh,
                             int blockIdx, int blockDim)
{
    uint32_t inner = ish[dims - 1];
    uint32_t nblk = size / inner;
    uint32_t base = nblk / blockDim, extra = nblk % blockDim;
    uint32_t b0 = base * blockIdx + (blockIdx < (int)extra ? (uint32_t)blockIdx : extra);
    uint32_t bN = base + (blockIdx < (int)extra ? 1 : 0);
    if (bN == 0) return;
    Assign_B<T> op;
    op.Init(pipe, input, other, size, dims, ish, osh);
    op.Process(b0, bN);
}

extern "C" __global__ __aicore__ void assign(GM_ADDR input, GM_ADDR other, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    const uint32_t size = tiling_data.size;
    const uint32_t dims = tiling_data.dims;
    const uint32_t* input_shape = tiling_data.input_shape;
    const uint32_t* other_shape = tiling_data.other_shape;
    int blockIdx = GetBlockIdx();
    int blockDim = GetBlockNum();
    int cacheline = 128;
    int chunks = (size + cacheline - 1) / cacheline;
    int base_chunks = chunks / blockDim;
    int tile_chunks = chunks % blockDim;
    int start = (base_chunks * blockIdx + min(blockIdx, tile_chunks)) * cacheline;
    int len = min(int(size) - start, (base_chunks + (blockIdx < tile_chunks)) * cacheline);
    TPipe pipe;

    if (TILING_KEY_IS(1)){
        if(len <= 0) return;
        Assign_NB<float> op;
        op.Init(&pipe, input, other);

        GlobalTensor<int32_t> workGm;
        workGm.SetGlobalBuffer((__gm__ int32_t *)workspace);
        op.Process(start, len, (workGm(0)+1)%2);
        if(blockIdx == 0) workGm(0) += 1;
    }
    else if (TILING_KEY_IS(2)){
        if(len <= 0) return;
        Assign_NB<half> op;
        op.Init(&pipe, input, other);

        GlobalTensor<int32_t> workGm;
        workGm.SetGlobalBuffer((__gm__ int32_t *)workspace);
        // op.Process(start, len, 1);
        op.Process(start, len, (workGm(0)+1)%2);
        if(blockIdx == 0) workGm(0) += 1;
    }
    else if (TILING_KEY_IS(3)){
        if(len <= 0) return;
        Assign_NB<uint8_t> op;
        op.Init(&pipe, input, other);

        GlobalTensor<int32_t> workGm;
        workGm.SetGlobalBuffer((__gm__ int32_t *)workspace);
        op.Process(start, len, (workGm(0)+1)%2);
        if(blockIdx == 0) workGm(0) += 1;
    }
    else if (TILING_KEY_IS(4)){
        if(len <= 0) return;
        Assign_NB<bool> op;
        op.Init(&pipe, input, other);

        GlobalTensor<int32_t> workGm;
        workGm.SetGlobalBuffer((__gm__ int32_t *)workspace);
        op.Process(start, len, (workGm(0)+1)%2);
        if(blockIdx == 0) workGm(0) += 1;
    }
    else if (TILING_KEY_IS(5)){
        run_b<int32_t>(&pipe, input, other, size, dims, input_shape, other_shape, blockIdx, blockDim);
    }
    else if (TILING_KEY_IS(6)){
        run_b<half>(&pipe, input, other, size, dims, input_shape, other_shape, blockIdx, blockDim);
    }
    else if (TILING_KEY_IS(7)){
        run_b<uint8_t>(&pipe, input, other, size, dims, input_shape, other_shape, blockIdx, blockDim);
    }
}