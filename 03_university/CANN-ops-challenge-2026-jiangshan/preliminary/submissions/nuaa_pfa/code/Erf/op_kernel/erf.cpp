// Kernel侧核函数实现 —— V6
// 大改：抛弃手写多项式，改用 AscendC 内置 Erf 单指令
// 思路：之前40分是因为tiling烂，现在配V2好tiling，vendor的硬件级实现可能更快
#include "kernel_operator.h"
#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

constexpr int32_t  BUFFER_NUM  = 2;
constexpr uint32_t TILE_LENGTH = 8192;

template <class T>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length) {
        this->totalLength = length;
        uint32_t coreNum  = GetBlockNum();
        uint32_t alignNum = 32 / sizeof(T);

        uint32_t lengthPerCore = (this->totalLength + coreNum - 1) / coreNum;
        lengthPerCore = (lengthPerCore + alignNum - 1) / alignNum * alignNum;

        uint32_t coreIdx = GetBlockIdx();
        this->coreOffset = coreIdx * lengthPerCore;

        if (this->coreOffset >= this->totalLength) {
            this->blockLength = 0;
        } else {
            this->blockLength = this->totalLength - this->coreOffset;
            if (this->blockLength > lengthPerCore) {
                this->blockLength = lengthPerCore;
            }
        }

        this->tileLength = TILE_LENGTH;
        this->tileNum    = (this->blockLength + this->tileLength - 1) / this->tileLength;

        uint32_t blockLenAligned = (this->blockLength + alignNum - 1) / alignNum * alignNum;
        xGm.SetGlobalBuffer((__gm__ T*)x + this->coreOffset, blockLenAligned);
        yGm.SetGlobalBuffer((__gm__ T*)y + this->coreOffset, blockLenAligned);

        pipe.InitBuffer(inQueueX,  BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(T));
    }

    __aicore__ inline void Process() {
        if (this->blockLength == 0) {
            return;
        }

        CopyIn(0);
        for (int32_t i = 0; i < this->tileNum - 1; ++i) {
            Compute(i);
            CopyIn(i + 1);
            CopyOut(i);
        }
        Compute(this->tileNum - 1);
        CopyOut(this->tileNum - 1);
    }

private:
    __aicore__ inline uint32_t AlignUp32(uint32_t len) {
        uint32_t alignNum = 32 / sizeof(T);
        return (len + alignNum - 1) / alignNum * alignNum;
    }

    __aicore__ inline uint32_t CalcTileLength(int32_t progress) {
        if (progress == this->tileNum - 1) {
            return this->blockLength - (uint32_t)progress * this->tileLength;
        }
        return this->tileLength;
    }

    __aicore__ inline void CopyIn(int32_t progress) {
        uint32_t copyLen = AlignUp32(CalcTileLength(progress));
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopy(xLocal, xGm[progress * this->tileLength], copyLen);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        uint32_t calLen = CalcTileLength(progress);
        uint32_t vecLen = AlignUp32(calLen);

        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

        // 直接用 vendor 实现，单指令完成 erf
        Erf(yLocal, xLocal, vecLen);

        outQueueY.EnQue<T>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        uint32_t copyLen = AlignUp32(CalcTileLength(progress));
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopy(yGm[progress * this->tileLength], yLocal, copyLen);
        outQueueY.FreeTensor(yLocal);
    }

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    TPipe pipe;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t coreOffset;
    uint32_t tileLength;
    uint32_t tileNum;
};

template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    KernelErf<DT_X> op;
    op.Init(x, y, tiling_data.length);
    op.Process();
}
