// Kernel: erf(x) ≈ x * P(x²) / Q(x²)
// Pipeline: TPipe+TQue double-buffer | Direct: single-buffer
// 两个独立class, tiling key 编译期选择
#include "kernel_operator.h"
#include "erf_tiling.h"
#include "tiling_key_erf.h"

constexpr int32_t BUFFER_NUM = 2;

// 系数: P deg-3, Q deg-3, clamp [-3,3]
constexpr float P_COEFFS[] = {
    1.1283874407514682e+00f, 1.5302312290334222e-01f,
    4.3474481382003299e-02f, 7.6600388996244891e-04f
};
constexpr float Q_COEFFS[] = {
    1.0f, 4.6902098586093699e-01f,
    9.4653144350888119e-02f, 9.4220839128164278e-03f
};


class KernelErfPipeline {
public:
    __aicore__ inline KernelErfPipeline(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                        const ErfTilingData &tiling, AscendC::TPipe* pipeIn) {
        pipe = pipeIn;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        bool isBig = (blockIdx < tiling.bigCoreNum);
        uint32_t extra = isBig ? 128 : 0;       // 现在一个block就是512B

        this->tailDataNum = tiling.tailDataNum + extra;

        uint32_t offset = tiling.smallCoreDataNum * blockIdx;
        offset += isBig ? 128 * blockIdx : 128 * tiling.bigCoreNum;

        this->processDataNum = tiling.processDataNum;   // 这里尾核情况下不用改，只要tileNum=0，for循环就不会进行

        bool isTail = (blockIdx == tiling.TailCoreIdx);

        this->tileNum = isTail ? 0 : tiling.tileNum;

        this->tailDataNum = isTail ? tiling.intputTailNum : this->tailDataNum;
        uint32_t coreDataNum = isTail ? tiling.intputTailNum : tiling.smallCoreDataNum + extra;

        xGm.SetGlobalBuffer((__gm__ float*)x + offset, coreDataNum);
        yGm.SetGlobalBuffer((__gm__ float*)y + offset, coreDataNum);

        uint32_t bufSz = tiling.processDataNum * 4;
        this->pipe->InitBuffer(inQueueX, BUFFER_NUM, bufSz);
        this->pipe->InitBuffer(outQueueY, BUFFER_NUM, bufSz);
        this->pipe->InitBuffer(t0Buf_, bufSz);
        this->pipe->InitBuffer(t1Buf_, bufSz);
        this->pipe->InitBuffer(t2Buf_, bufSz);
        T0 = t0Buf_.Get<float>();
        T1 = t1Buf_.Get<float>();
        T2 = t2Buf_.Get<float>();
    }

    __aicore__ inline void Process() {

        uint32_t tileElems = this->processDataNum;
        int32_t  TileNum = this->tileNum;  
        for (int32_t i = 0; i < this->tileNum; i++) {
            CopyIn(i * tileElems);
            ComputePipelined();
            CopyOut(i * tileElems);
        }

        this->processDataNum = this->tailDataNum;
        uint32_t tailOffset = TileNum * tileElems;
        CopyIn(tailOffset);
        ComputePipelined();
        CopyOut(tailOffset);
    }

private:
    __aicore__ inline void CopyIn(uint32_t gmOffset) {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[gmOffset], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void ComputePipelined() {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        ComputeCore(xLocal, yLocal, this->processDataNum, T0, T1, T2);
        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t gmOffset) {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[gmOffset], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void ComputeCore(AscendC::LocalTensor<float> &xLocal,
                                        AscendC::LocalTensor<float> &yLocal, uint32_t len,
                                        AscendC::LocalTensor<float> &T0,
                                        AscendC::LocalTensor<float> &T1,
                                        AscendC::LocalTensor<float> &T2) {
        AscendC::Maxs(T0, xLocal, (float)-3.0f, len);
        AscendC::Mins(xLocal, T0, (float)3.0f, len);

        AscendC::Mul(yLocal, xLocal, xLocal, len);

        AscendC::Muls(T1, yLocal, (float)P_COEFFS[1], len);
        AscendC::Muls(T2, yLocal, (float)P_COEFFS[3], len);
        AscendC::Adds(T1, T1, (float)P_COEFFS[0], len);
        AscendC::Adds(T2, T2, (float)P_COEFFS[2], len);
        AscendC::Mul(T0, yLocal, yLocal, len);
        AscendC::MulAddDst(T1, T0, T2, len);

        AscendC::Muls(T2, yLocal, (float)Q_COEFFS[1], len);
        AscendC::Mul(xLocal, T1, xLocal, len);
        AscendC::Adds(T2, T2, (float)Q_COEFFS[0], len);
        AscendC::Muls(T1, yLocal, (float)Q_COEFFS[3], len);
        AscendC::Adds(T1, T1, (float)Q_COEFFS[2], len);
        AscendC::MulAddDst(T2, T0, T1, len);

        AscendC::Div(yLocal, xLocal, T2, len);
    }

    AscendC::TPipe* pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> t0Buf_, t1Buf_, t2Buf_;
    AscendC::LocalTensor<float> T0, T1, T2;
    AscendC::GlobalTensor<float> xGm, yGm;
    uint32_t tileNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};


class KernelErfDirect {
public:
    __aicore__ inline KernelErfDirect(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                      const ErfTilingData &tiling, 
                                      AscendC::TPipe* pipeIn) {
        this->pipe = pipeIn;
        uint16_t blockIdx = AscendC::GetBlockIdx();
        bool isBig = (blockIdx < tiling.bigCoreNum);
        bool isTail = (blockIdx == tiling.TailCoreIdx);

        uint32_t offset = tiling.smallCoreDataNum * blockIdx;
        uint32_t extra = isBig ? 128 : 0;
        offset += isBig ? 128 * blockIdx : 128 * tiling.bigCoreNum;

        this->coreDataNum = isTail ? tiling.intputTailNum : tiling.smallCoreDataNum + extra;

        xGm.SetGlobalBuffer((__gm__ float*)x + offset, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ float*)y + offset, this->coreDataNum);

        uint32_t bufSz = tiling.processDataNum * 28 + 448; // (datanum * sizeof(float) + 32)*4 / sizeof(float)
        this->shiftY = tiling.processDataNum + 64;
        this->shiftT0 = tiling.processDataNum * 2 + 128;
        this->shiftT1 = tiling.processDataNum * 3 + 192;
        this->shiftT2 = tiling.processDataNum * 4 + 256;
        this->shiftT3 = tiling.processDataNum * 5 + 320;
        this->shiftT4 = tiling.processDataNum * 6 + 384;

        this->pipe->InitBuffer(Buf_, bufSz);
        this->T_ = Buf_.Get<float>();
    }

    __aicore__ inline void Process() {
        AscendC::DataCopy(this->T_, this->xGm, this->coreDataNum);

        // MTE2 -> V 同步
        AscendC::TQueSync<PIPE_MTE2, PIPE_V> syncMte2ToV;
        syncMte2ToV.SetFlag(0);
        syncMte2ToV.WaitFlag(0);

        ComputeCore(this->T_, this->coreDataNum);

        // V -> MTE3 同步
        AscendC::TQueSync<PIPE_V, PIPE_MTE3> syncVToMte3;
        syncVToMte3.SetFlag(0);
        syncVToMte3.WaitFlag(0);

        AscendC::DataCopy(this->yGm, T_[shiftY], this->coreDataNum);
    }

private:
     __aicore__ inline void ComputeCore(AscendC::LocalTensor<float> &T_, 
                                        uint32_t len) {
        AscendC::Maxs(T_[shiftY], T_, (float)-3.0f, len);
        AscendC::Mins(T_, T_[shiftY], (float)3.0f, len);


        AscendC::Mul(T_[shiftY], T_, T_, len);

        AscendC::Muls(T_[shiftT1], T_[shiftY], (float)P_COEFFS[1], len);
        AscendC::Muls(T_[shiftT2], T_[shiftY], (float)P_COEFFS[3], len);
        
        AscendC::Muls(T_[shiftT3], T_[shiftY], (float)Q_COEFFS[1], len);
        AscendC::Muls(T_[shiftT4], T_[shiftY], (float)Q_COEFFS[3], len);

        AscendC::Mul(T_[shiftT0], T_[shiftY], T_[shiftY], len);

        AscendC::Adds(T_[shiftT1], T_[shiftT1], (float)P_COEFFS[0], len);
        AscendC::Adds(T_[shiftT2], T_[shiftT2], (float)P_COEFFS[2], len);
        
        AscendC::Adds(T_[shiftT3], T_[shiftT3], (float)Q_COEFFS[0], len);
        AscendC::Adds(T_[shiftT4], T_[shiftT4], (float)Q_COEFFS[2], len);

        // AscendC::FusedMulAdd(T_[shiftT2], T_[shiftT0], T_[shiftT1], len);
        // AscendC::FusedMulAdd(T_[shiftT4], T_[shiftT0], T_[shiftT3], len);
         
        AscendC::Mul(T_[shiftT2], T_[shiftT2], T_[shiftT0], len);
        AscendC::Mul(T_[shiftT4], T_[shiftT4], T_[shiftT0], len);

        AscendC::Add(T_[shiftT0], T_[shiftT1], T_[shiftT2], len); // P
        AscendC::Add(T_[shiftT2], T_[shiftT3], T_[shiftT4], len); // Q

        AscendC::Mul(T_[shiftT1], T_[shiftT0], T_, len);
         // AscendC::Mul(T_[shiftT1], T_[shiftT2], T_, len);

         AscendC::Div(T_[shiftY], T_[shiftT1], T_[shiftT2], len);
         
    }

    AscendC::TPipe *pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> Buf_;
    AscendC::LocalTensor<float> T_;
    AscendC::GlobalTensor<float> xGm, yGm;
    uint32_t coreDataNum;
    uint32_t shiftY;
    uint32_t shiftT0;
    uint32_t shiftT1;
    uint32_t shiftT2;
    uint32_t shiftT3;
    uint32_t shiftT4;
};

// 编译期选择 Pipeline/Direct
template <uint32_t usePipeline>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    AscendC::TPipe pipe;
    if constexpr (usePipeline == 1) {
        KernelErfPipeline op(x, y, workspace, tiling_data, &pipe);
        op.Process();
    } else {
        KernelErfDirect op(x, y, workspace, tiling_data, &pipe);
        op.Process();
    }
}

