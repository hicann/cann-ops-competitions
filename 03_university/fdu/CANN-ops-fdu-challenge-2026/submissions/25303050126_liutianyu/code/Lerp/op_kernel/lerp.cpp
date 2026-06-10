// Kernel侧核函数实现
#include "kernel_operator.h"

#include "lerp_tiling.h"
#include "tiling_key_lerp.h"

using namespace AscendC;

template <class DT_X>
class KernelLerp {
public:
    __aicore__ inline KernelLerp() {}

    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR y,
                                const LerpTilingData &tiling)
    {
        startGm.SetGlobalBuffer((__gm__ DT_X *)start);
        endGm.SetGlobalBuffer((__gm__ DT_X *)end);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y);
        length = tiling.length;
        perCoreElements = tiling.perCoreElements;
        tileLength = tiling.tileLength;
        weightScalar = static_cast<DT_X>(tiling.weight);

        pipe.InitBuffer(inQueueStart, LERP_QUEUE_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueEnd, LERP_QUEUE_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, LERP_QUEUE_NUM, tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (length == 0 || perCoreElements == 0 || tileLength == 0) {
            return;
        }

        const uint64_t startOff = static_cast<uint64_t>(GetBlockIdx()) * perCoreElements;
        if (startOff >= length) {
            return;
        }

        uint64_t endOff = startOff + perCoreElements;
        if (endOff > length) {
            endOff = length;
        }

        ProcessVector(startOff, endOff);
    }

private:
    __aicore__ inline void ProcessVector(uint64_t start, uint64_t end)
    {
        const uint32_t blockElems = LERP_DATA_BLOCK_BYTES / sizeof(DT_X);
        const uint64_t total = end - start;
        const uint64_t vectorLen = (total / blockElems) * blockElems;

        uint64_t done = 0;
        while (done < vectorLen) {
            uint32_t curLen = tileLength;
            if (static_cast<uint64_t>(curLen) > vectorLen - done) {
                curLen = static_cast<uint32_t>(vectorLen - done);
            }

            CopyIn(start + done, curLen);
            Compute(curLen);
            CopyOut(start + done, curLen);
            done += curLen;
        }

        if (start + vectorLen < end) {
            ProcessScalarTail(start + vectorLen, end);
        }
    }

    __aicore__ inline void CopyIn(uint64_t offset, uint32_t count)
    {
        LocalTensor<DT_X> sLocal = inQueueStart.AllocTensor<DT_X>();
        LocalTensor<DT_X> eLocal = inQueueEnd.AllocTensor<DT_X>();
        DataCopy(sLocal, startGm[offset], count);
        DataCopy(eLocal, endGm[offset], count);
        inQueueStart.EnQue(sLocal);
        inQueueEnd.EnQue(eLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<DT_X> sLocal = inQueueStart.DeQue<DT_X>();
        LocalTensor<DT_X> eLocal = inQueueEnd.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();

        // y = end - start
        Sub(yLocal, eLocal, sLocal, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();

        // y = weight * (end - start)
        Muls(yLocal, yLocal, weightScalar, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();

        // y = start + weight * (end - start)
        Add(yLocal, sLocal, yLocal, static_cast<int32_t>(count));

        outQueueY.EnQue<DT_X>(yLocal);
        inQueueStart.FreeTensor(sLocal);
        inQueueEnd.FreeTensor(eLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t count)
    {
        LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        DataCopy(yGm[offset], yLocal, count);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void ProcessScalarTail(uint64_t start, uint64_t end)
    {
        const float w = static_cast<float>(weightScalar);
        for (uint64_t idx = start; idx < end; ++idx) {
            float s = static_cast<float>(startGm.GetValue(idx));
            float e = static_cast<float>(endGm.GetValue(idx));
            float result = s + w * (e - s);
            yGm.SetValue(idx, static_cast<DT_X>(result));
        }

        DataCacheCleanAndInvalid<DT_X, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(yGm);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, LERP_QUEUE_NUM> inQueueStart;
    TQue<QuePosition::VECIN, LERP_QUEUE_NUM> inQueueEnd;
    TQue<QuePosition::VECOUT, LERP_QUEUE_NUM> outQueueY;
    GlobalTensor<DT_X> startGm;
    GlobalTensor<DT_X> endGm;
    GlobalTensor<DT_X> yGm;
    uint64_t length = 0;
    uint64_t perCoreElements = 0;
    uint32_t tileLength = 0;
    DT_X weightScalar;
};

template <typename DT_X>
__global__ __aicore__ void lerp(GM_ADDR start, GM_ADDR end, GM_ADDR y,
                                GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LerpTilingData);
    GET_TILING_DATA_WITH_STRUCT(LerpTilingData, tilingData, tiling);
    KernelLerp<DT_X> op;
    op.Init(start, end, y, tilingData);
    op.Process();
}
