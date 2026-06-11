// Kernel侧核函数实现
#include "kernel_operator.h"

#include "lerp_tiling.h"
#include "tiling_key_lerp.h"

template <class DT_START>
class KernelLerp {
public:
    __aicore__ inline KernelLerp() {}
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR y, LerpTilingData* tiling)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = tiling->blockNum;

        if (blockIdx < blockNum - 1) {
            totalLength = tiling->numPerCore;
        } else {
            totalLength = tiling->tailNumLastCore;
        }

        uint32_t offset = blockIdx * tiling->numPerCore;

        startGm.SetGlobalBuffer((__gm__ DT_START*)start + offset, totalLength);
        endGm.SetGlobalBuffer((__gm__ DT_START*)end + offset, totalLength);
        yGm.SetGlobalBuffer((__gm__ DT_START*)y + offset, totalLength);

        weightVal = static_cast<DT_START>(tiling->weight);

        tileNum = (totalLength + TILE_LENGTH - 1) / TILE_LENGTH;

        pipe.InitBuffer(inQueueStart, DOUBLE_BUFFER, TILE_LENGTH * sizeof(DT_START));
        pipe.InitBuffer(inQueueEnd, DOUBLE_BUFFER, TILE_LENGTH * sizeof(DT_START));
        pipe.InitBuffer(outQueueY, DOUBLE_BUFFER, TILE_LENGTH * sizeof(DT_START));
        pipe.InitBuffer(tmpBuf, TILE_LENGTH * sizeof(DT_START));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < tileNum; i++) {
            uint32_t count = (i == tileNum - 1) ?
                (totalLength - i * TILE_LENGTH) : TILE_LENGTH;
            CopyIn(count);
            Compute(count);
            CopyOut(count);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t count)
    {
        auto startLocal = inQueueStart.AllocTensor<DT_START>();
        auto endLocal = inQueueEnd.AllocTensor<DT_START>();

        AscendC::DataCopyPad(startLocal, startGm,
            {1, static_cast<uint16_t>(count * sizeof(DT_START)), 0, 0},
            {false, 0, 0, 0});
        AscendC::DataCopyPad(endLocal, endGm,
            {1, static_cast<uint16_t>(count * sizeof(DT_START)), 0, 0},
            {false, 0, 0, 0});

        inQueueStart.EnQue(startLocal);
        inQueueEnd.EnQue(endLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        auto startLocal = inQueueStart.DeQue<DT_START>();
        auto endLocal = inQueueEnd.DeQue<DT_START>();
        auto yLocal = outQueueY.AllocTensor<DT_START>();
        auto tempLocal = tmpBuf.Get<DT_START>();

        // y = start + weight * (end - start)
        // temp = end - start
        AscendC::Sub(tempLocal, endLocal, startLocal, count);
        // temp = weight * temp
        AscendC::Muls(tempLocal, tempLocal, weightVal, count);
        // y = start + temp
        AscendC::Add(yLocal, startLocal, tempLocal, count);

        outQueueY.EnQue<DT_START>(yLocal);
        inQueueStart.FreeTensor(startLocal);
        inQueueEnd.FreeTensor(endLocal);
    }

    __aicore__ inline void CopyOut(uint32_t count)
    {
        auto yLocal = outQueueY.DeQue<DT_START>();
        AscendC::DataCopyPad(yGm, yLocal,
            {1, static_cast<uint16_t>(count * sizeof(DT_START)), 0, 0});
        outQueueY.FreeTensor(yLocal);
    }

    AscendC::GlobalTensor<DT_START> startGm;
    AscendC::GlobalTensor<DT_START> endGm;
    AscendC::GlobalTensor<DT_START> yGm;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueStart;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueEnd;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;

    AscendC::TPipe pipe;

    uint32_t totalLength;
    uint32_t tileNum;
    DT_START weightVal;
};

template <typename DT_START>
 __global__ __aicore__ void lerp(GM_ADDR start, GM_ADDR end, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LerpTilingData);
    GET_TILING_DATA_WITH_STRUCT(LerpTilingData, tiling_data, tiling);
    KernelLerp<DT_START> op;
    op.Init(start, end, y, &tiling_data);
    op.Process();
}

