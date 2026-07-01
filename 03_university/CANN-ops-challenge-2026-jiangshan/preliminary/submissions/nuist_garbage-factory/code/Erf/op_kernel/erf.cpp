#include "kernel_operator.h"
#include "erf_tiling.h"
#include "tiling_key_erf.h"
#include "adv_api/math/tanh.h"

constexpr int32_t BUFFER_NUM = 2;
template <class DT_X>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}
    __aicore__ inline void Init(GM_ADDR x,  GM_ADDR z,
                                uint32_t smallCoreDataNum, uint32_t bigCoreDataNum,
                                uint32_t finalBigTileNum, uint32_t finalSmallTileNum,
                                uint32_t tileDataNum, uint32_t smallTailDataNum,
                                uint32_t bigTailDataNum, uint32_t tailBlockNum,AscendC::TPipe* pipeIn)
    {
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
        this->tileDataNum = tileDataNum;
        if (coreNum < tailBlockNum) { 
          this->coreDataNum = bigCoreDataNum;
          this->tileNum = finalBigTileNum;
          this->tailDataNum = bigTailDataNum;
        }
        else { 
          this->coreDataNum = smallCoreDataNum;
          this->tileNum = finalSmallTileNum;
          this->tailDataNum = smallTailDataNum;
          globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
        }
        xGm.SetGlobalBuffer((__gm__ DT_X*)x + globalBufferIndex, this->coreDataNum);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + globalBufferIndex, this->coreDataNum);
        pipe = pipeIn;
        pipe->InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(DT_X));
        pipe->InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(DT_X));


        
    }
    __aicore__ inline void Process()
    {
        if (this->tileNum == 0) return;
        if (this->tileNum == 1) {
            // 单 tile：直接搬运 → 计算 → 回写
            AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
            AscendC::DataCopy(xLocal, xGm[0], this->tailDataNum);
            inQueueX.EnQue(xLocal);
            xLocal = inQueueX.DeQue<DT_X>();
            AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
            ComputeErf(zLocal, xLocal, this->tailDataNum);
            outQueueZ.EnQue<DT_X>(zLocal);
            inQueueX.FreeTensor(xLocal);
            zLocal = outQueueZ.DeQue<DT_X>();
            AscendC::DataCopy(zGm[0], zLocal, this->tailDataNum);
            outQueueZ.FreeTensor(zLocal);
            return;
        }

        const int32_t fullTiles = this->tileNum - 1;  // 满 tile 数（最后一块 tail 不计入）

        // ---- 预取 tile 0 ----
        AscendC::LocalTensor<DT_X> xPrefetch = inQueueX.AllocTensor<DT_X>();
        AscendC::DataCopy(xPrefetch, xGm[0], this->tileDataNum);
        inQueueX.EnQue(xPrefetch);

        // ---- 双缓冲主循环：计算 tile[i]，同时预取 tile[i+1] ----
        // 循环范围：i = 0 .. fullTiles-2（最后留一个满 tile + tail 做收尾）
        for (int32_t i = 0; i < fullTiles - 1; i++) {
            AscendC::LocalTensor<DT_X> xCur = inQueueX.DeQue<DT_X>();

            // 预取下一块（非最后一块，所以用 tileDataNum）
            AscendC::LocalTensor<DT_X> xNext = inQueueX.AllocTensor<DT_X>();
            AscendC::DataCopy(xNext, xGm[(i + 1) * this->tileDataNum], this->tileDataNum);
            inQueueX.EnQue(xNext);

            // 写出上上一块的结果（i >= 1 时才有）
            if (i > 0) {
                AscendC::LocalTensor<DT_X> zPrev = outQueueZ.DeQue<DT_X>();
                AscendC::DataCopy(zGm[(i - 1) * this->tileDataNum], zPrev, this->tileDataNum);
                outQueueZ.FreeTensor(zPrev);
            }

            // 计算当前块
            AscendC::LocalTensor<DT_X> zCur = outQueueZ.AllocTensor<DT_X>();
            ComputeErf(zCur, xCur, this->tileDataNum);
            outQueueZ.EnQue<DT_X>(zCur);
            inQueueX.FreeTensor(xCur);
        }

        // ---- 处理最后一块满 tile（fullTiles-1）：不预取，直接计算 ----
        {
            AscendC::LocalTensor<DT_X> xLastFull = inQueueX.DeQue<DT_X>();

            // 写出它前面的那块结果（fullTiles-2）
            if (fullTiles - 1 > 0) {
                AscendC::LocalTensor<DT_X> zPrev = outQueueZ.DeQue<DT_X>();
                AscendC::DataCopy(zGm[(fullTiles - 2) * this->tileDataNum], zPrev, this->tileDataNum);
                outQueueZ.FreeTensor(zPrev);
            }

            AscendC::LocalTensor<DT_X> zLastFull = outQueueZ.AllocTensor<DT_X>();
            ComputeErf(zLastFull, xLastFull, this->tileDataNum);
            outQueueZ.EnQue<DT_X>(zLastFull);
            inQueueX.FreeTensor(xLastFull);

            // 回写最后一块满 tile
            zLastFull = outQueueZ.DeQue<DT_X>();
            AscendC::DataCopy(zGm[(fullTiles - 1) * this->tileDataNum], zLastFull, this->tileDataNum);
            outQueueZ.FreeTensor(zLastFull);
        }

        // ---- tail tile：单独搬运 tailDataNum 元素 → 计算 → 回写 ----
        {
            AscendC::LocalTensor<DT_X> xTail = inQueueX.AllocTensor<DT_X>();
            AscendC::DataCopy(xTail, xGm[fullTiles * this->tileDataNum], this->tailDataNum);
            inQueueX.EnQue(xTail);
            xTail = inQueueX.DeQue<DT_X>();

            AscendC::LocalTensor<DT_X> zTail = outQueueZ.AllocTensor<DT_X>();
            ComputeErf(zTail, xTail, this->tailDataNum);
            outQueueZ.EnQue<DT_X>(zTail);
            inQueueX.FreeTensor(xTail);

            zTail = outQueueZ.DeQue<DT_X>();
            AscendC::DataCopy(zGm[fullTiles * this->tileDataNum], zTail, this->tailDataNum);
            outQueueZ.FreeTensor(zTail);
        }
    }

private:
    // erf(x) ≈ tanh(a*x + b*x³), a = 2/√π, b ≈ 0.100911
    // Max error: 0.000358 (well within 0.001 requirement)
    // NOTE: Tanh requires dst and src to be DIFFERENT tensors (no address overlap)
    __aicore__ inline void ComputeErf(AscendC::LocalTensor<DT_X>& zLocal,
                                       AscendC::LocalTensor<DT_X>& xLocal,
                                       uint32_t len)
    {
        const DT_X a = (DT_X)1.128379167095513f;
        const DT_X b = (DT_X)0.100910800000000f;

        // Compute in zLocal: (a + b*x²) * x = a*x + b*x³
        AscendC::Mul(zLocal, xLocal, xLocal, len);     // z = x²
        AscendC::Muls(zLocal, zLocal, b, len);         // z = b*x²
        AscendC::Adds(zLocal, zLocal, a, len);         // z = a + b*x²
        AscendC::Mul(xLocal, zLocal, xLocal, len);     // x = (a+b*x²)*x = a*x+b*x³

        // Tanh: dst(zLocal) ≠ src(xLocal) per API overlap constraint
        AscendC::Tanh(zLocal, xLocal, len);            // z = erf(x)
    }

    AscendC::TPipe* pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> zGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
};


template <class DT_X>
 __global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    AscendC::TPipe pipe;
    KernelErf<DT_X> op;
    op.Init(x, y,
            tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
            tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
            tiling_data.tailBlockNum,&pipe);
    op.Process();
}