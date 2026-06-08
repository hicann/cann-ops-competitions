#include "kernel_operator.h"
#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

// 使用双缓冲
constexpr int32_t BUFFER_NUM = 2; 

template <typename DT_X>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, ErfTilingData* tiling) {
        uint32_t coreId = GetBlockIdx();
        
        // 计算当前核的起始偏移和处理数量
        // 这里的单位是 64-element block (256字节对齐)
        uint32_t startBlock = coreId * tiling->blocksPerCore + (coreId < tiling->tailBlocks ? coreId : tiling->tailBlocks);
        uint32_t myBlocks = tiling->blocksPerCore + (coreId < tiling->tailBlocks ? 1 : 0);
        
        uint32_t myOffset = startBlock * 64; 
        uint32_t myLength = myBlocks * 64;

        // 边界处理
        if (myOffset >= tiling->totalLength) {
            this->coreDataSize = 0;
            return;
        }
        if (myOffset + myLength > tiling->totalLength) {
            myLength = tiling->totalLength - myOffset;
        }

        this->coreDataSize = myLength;
        this->tileLength = tiling->tileLength;
        
        xGm.SetGlobalBuffer((__gm__ DT_X*)x + myOffset, this->coreDataSize);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + myOffset, this->coreDataSize);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (this->coreDataSize == 0) return;

        uint32_t loopCount = this->coreDataSize / this->tileLength;
        uint32_t tailLength = this->coreDataSize % this->tileLength;

        // 主循环：处理整 Tile
        for (uint32_t i = 0; i < loopCount; i++) {
            CopyIn(i * this->tileLength, this->tileLength);
            Compute(this->tileLength);
            CopyOut(i * this->tileLength, this->tileLength);
        }

        // 尾部：处理不满 Tile 的部分
        if (tailLength > 0) {
            CopyIn(loopCount * this->tileLength, tailLength);
            Compute(tailLength);
            CopyOut(loopCount * this->tileLength, tailLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length) {
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        // 强制使用 32 字节对齐的向上取整长度进行搬运，获得最高 DMA 效率
        DataCopy(xLocal, xGm[offset], (length + 7) & (~7U));
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t length) {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        
        // 调用硬件 Erf 指令，长度需对齐到 8 (32字节)
        Erf(yLocal, xLocal, (length + 7) & (~7U));
        
        outQueueY.EnQue<DT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length) {
        LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        DataCopy(yGm[offset], yLocal, (length + 7) & (~7U));
        outQueueY.FreeTensor(yLocal);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;

    uint32_t coreDataSize;
    uint32_t tileLength;
};

// 保持泛型入口
template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    
    KernelErf<DT_X> op;
    op.Init(x, y, &tiling_data);
    op.Process();
}