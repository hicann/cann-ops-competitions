// Kernel侧核函数实现
#include "kernel_operator.h"

#include "clip_by_value_tiling.h"
#include "tiling_key_clip_by_value.h"

using namespace AscendC;

__aicore__ inline float ClipByValueScalarEval(float value, float minValue, float maxValue)
{
    if (value < minValue) {
        return minValue;
    }
    if (value > maxValue) {
        return maxValue;
    }
    return value;
}

__aicore__ inline half ClipByValueScalarEval(half value, half minValue, half maxValue)
{
    // min/max先在Init中转换为half，这里用转换后的边界比较，保证尾块标量路径与向量Maxs/Mins路径一致。
    const float valueF = static_cast<float>(value);
    const float minF = static_cast<float>(minValue);
    const float maxF = static_cast<float>(maxValue);
    if (valueF < minF) {
        return minValue;
    }
    if (valueF > maxF) {
        return maxValue;
    }
    return value;
}

__aicore__ inline int32_t ClipByValueScalarEval(int32_t value, int32_t minValue, int32_t maxValue)
{
    if (value < minValue) {
        return minValue;
    }
    if (value > maxValue) {
        return maxValue;
    }
    return value;
}

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

        // DataCopy按32B整块处理；最后不足32B的极小尾块用标量路径补齐。
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
        for (uint64_t idx = start; idx < end; ++idx) {
            const DT_X xValue = xGm.GetValue(idx);
            yGm.SetValue(idx, ClipByValueScalarEval(xValue, minScalar, maxScalar));
        }

        // Scalar单元写GM会先写入Data Cache，需要显式刷新到Global Memory。
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
