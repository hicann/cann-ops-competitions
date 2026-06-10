// Kernel侧核函数实现
#include "kernel_operator.h"

#include "clip_by_value_tiling.h"
#include "tiling_key_clip_by_value.h"

using namespace AscendC;

template <class DT_X>
class KernelClipByValue {
public:
    __aicore__ inline KernelClipByValue() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ClipByValueTilingData &tiling)
    {
        xGm.SetGlobalBuffer((__gm__ DT_X *)x);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y);
        length = tiling.length;
        perCoreElements = tiling.perCoreElements;
        tileLength = tiling.tileLength;
        minScalar = static_cast<DT_X>(tiling.minValue);
        maxScalar = static_cast<DT_X>(tiling.maxValue);

        pipe.InitBuffer(inQueue, CLIP_BY_VALUE_QUEUE_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueue, CLIP_BY_VALUE_QUEUE_NUM, tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (length == 0 || perCoreElements == 0 || tileLength == 0) {
            return;
        }

        const uint64_t start = static_cast<uint64_t>(GetBlockIdx()) * perCoreElements;
        if (start >= length) {
            return;
        }

        uint64_t end = start + perCoreElements;
        if (end > length) {
            end = length;
        }

        ProcessVector(start, end);
    }

private:
    __aicore__ inline void ProcessVector(uint64_t start, uint64_t end)
    {
        const uint32_t blockElems = CLIP_BY_VALUE_DATA_BLOCK_BYTES / sizeof(DT_X);
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
        LocalTensor<DT_X> xLocal = inQueue.AllocTensor<DT_X>();
        DataCopy(xLocal, xGm[offset], count);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<DT_X> xLocal = inQueue.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueue.AllocTensor<DT_X>();

        // clamp(x, min, max) = min(max(x, min), max)
        Maxs(yLocal, xLocal, minScalar, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        Mins(yLocal, yLocal, maxScalar, static_cast<int32_t>(count));

        outQueue.EnQue<DT_X>(yLocal);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t count)
    {
        LocalTensor<DT_X> yLocal = outQueue.DeQue<DT_X>();
        DataCopy(yGm[offset], yLocal, count);
        outQueue.FreeTensor(yLocal);
    }

    __aicore__ inline void ProcessScalarTail(uint64_t start, uint64_t end)
    {
        float minF = static_cast<float>(minScalar);
        float maxF = static_cast<float>(maxScalar);
        for (uint64_t idx = start; idx < end; ++idx) {
            const DT_X xValue = xGm.GetValue(idx);
            float valF = static_cast<float>(xValue);
            DT_X clamped = xValue;
            if (valF < minF) {
                clamped = minScalar;
            } else if (valF > maxF) {
                clamped = maxScalar;
            }
            yGm.SetValue(idx, clamped);
        }

        DataCacheCleanAndInvalid<DT_X, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(yGm);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, CLIP_BY_VALUE_QUEUE_NUM> inQueue;
    TQue<QuePosition::VECOUT, CLIP_BY_VALUE_QUEUE_NUM> outQueue;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    uint64_t length = 0;
    uint64_t perCoreElements = 0;
    uint32_t tileLength = 0;
    DT_X minScalar;
    DT_X maxScalar;
};

template <typename DT_X>
__global__ __aicore__ void clip_by_value(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ClipByValueTilingData);
    GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tilingData, tiling);
    KernelClipByValue<DT_X> op;
    op.Init(x, y, tilingData);
    op.Process();
}
