// Kernel侧核函数实现
#include "kernel_operator.h"

#include "lerp_tiling.h"
#include "tiling_key_lerp.h"

using namespace AscendC;

__aicore__ inline float LerpScalarEval(float startValue, float endValue, float weight)
{
    return startValue + weight * (endValue - startValue);
}

__aicore__ inline half LerpScalarEval(half startValue, half endValue, float weight)
{
    const float startF = static_cast<float>(startValue);
    const float endF = static_cast<float>(endValue);
    return static_cast<half>(startF + weight * (endF - startF));
}

template <class T>
class KernelLerp {
public:
    __aicore__ inline KernelLerp() {}

    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR y, const LerpTilingData &tiling)
    {
        startGm.SetGlobalBuffer((__gm__ T *)start);
        endGm.SetGlobalBuffer((__gm__ T *)end);
        yGm.SetGlobalBuffer((__gm__ T *)y);

        length = tiling.length;
        perCoreElements = tiling.perCoreElements;
        tileLength = tiling.tileLength;
        mode = tiling.mode;
        weight = tiling.weight;
        weightScalar = static_cast<T>(tiling.weight);

        if (tileLength > 0) {
            pipe.InitBuffer(startQueue, LERP_BUFFER_NUM, tileLength * sizeof(T));
            pipe.InitBuffer(endQueue, LERP_BUFFER_NUM, tileLength * sizeof(T));
            pipe.InitBuffer(outQueue, LERP_BUFFER_NUM, tileLength * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        if (length == 0 || perCoreElements == 0 || tileLength == 0) {
            return;
        }

        const uint64_t begin = static_cast<uint64_t>(GetBlockIdx()) * perCoreElements;
        if (begin >= length) {
            return;
        }

        uint64_t end = begin + perCoreElements;
        if (end > length) {
            end = length;
        }

        if (mode == LERP_MODE_COPY_START) {
            ProcessCopy(begin, end, false);
        } else if (mode == LERP_MODE_COPY_END) {
            ProcessCopy(begin, end, true);
        } else {
            ProcessNormal(begin, end);
        }
    }

private:
    __aicore__ inline uint64_t MainLength(uint64_t count) const
    {
        const uint32_t elemsPerBlock = LERP_DATA_BLOCK_BYTES / sizeof(T);
        return (count / elemsPerBlock) * elemsPerBlock;
    }

    __aicore__ inline uint32_t CurTile(uint64_t remain) const
    {
        return static_cast<uint32_t>((remain < static_cast<uint64_t>(tileLength)) ? remain : tileLength);
    }

    __aicore__ inline void ProcessNormal(uint64_t begin, uint64_t end)
    {
        const uint64_t total = end - begin;
        const uint64_t mainLen = MainLength(total);

        uint64_t done = 0;
        while (done < mainLen) {
            const uint32_t curLen = CurTile(mainLen - done);
            CopyIn(begin + done, curLen);
            Compute(curLen);
            CopyOut(begin + done, curLen);
            done += curLen;
        }

        if (mainLen < total) {
            ProcessScalar(begin + mainLen, end);
            DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(yGm);
        }
    }

    __aicore__ inline void ProcessCopy(uint64_t begin, uint64_t end, bool copyEnd)
    {
        const uint64_t total = end - begin;
        const uint64_t mainLen = MainLength(total);

        uint64_t done = 0;
        while (done < mainLen) {
            const uint32_t curLen = CurTile(mainLen - done);
            CopyInputOnly(begin + done, curLen, copyEnd);
            CopyByVector(begin + done, curLen);
            done += curLen;
        }

        if (mainLen < total) {
            for (uint64_t i = begin + mainLen; i < end; ++i) {
                yGm.SetValue(i, copyEnd ? endGm.GetValue(i) : startGm.GetValue(i));
            }
            DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(yGm);
        }
    }

    __aicore__ inline void CopyIn(uint64_t offset, uint32_t count)
    {
        LocalTensor<T> startLocal = startQueue.AllocTensor<T>();
        LocalTensor<T> endLocal = endQueue.AllocTensor<T>();

        DataCopy(startLocal, startGm[offset], count);
        DataCopy(endLocal, endGm[offset], count);

        startQueue.EnQue(startLocal);
        endQueue.EnQue(endLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<T> startLocal = startQueue.DeQue<T>();
        LocalTensor<T> endLocal = endQueue.DeQue<T>();
        LocalTensor<T> outLocal = outQueue.AllocTensor<T>();

        // out = start + weight * (end - start)
        // 只使用已通过版本验证过的Sub/Muls/Add接口，避免Axpy兼容风险。
        Sub(outLocal, endLocal, startLocal, count);
        PipeBarrier<PIPE_V>();
        Muls(outLocal, outLocal, weightScalar, count);
        PipeBarrier<PIPE_V>();
        Add(outLocal, startLocal, outLocal, count);

        outQueue.EnQue<T>(outLocal);
        startQueue.FreeTensor(startLocal);
        endQueue.FreeTensor(endLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t count)
    {
        LocalTensor<T> outLocal = outQueue.DeQue<T>();
        DataCopy(yGm[offset], outLocal, count);
        outQueue.FreeTensor(outLocal);
    }

    __aicore__ inline void CopyInputOnly(uint64_t offset, uint32_t count, bool copyEnd)
    {
        LocalTensor<T> inLocal = startQueue.AllocTensor<T>();
        if (copyEnd) {
            DataCopy(inLocal, endGm[offset], count);
        } else {
            DataCopy(inLocal, startGm[offset], count);
        }
        startQueue.EnQue(inLocal);
    }

    __aicore__ inline void CopyByVector(uint64_t offset, uint32_t count)
    {
        LocalTensor<T> inLocal = startQueue.DeQue<T>();
        LocalTensor<T> outLocal = outQueue.AllocTensor<T>();

        // 保持VECOUT写回结构，兼容性优先。
        Muls(outLocal, inLocal, static_cast<T>(1), count);

        outQueue.EnQue<T>(outLocal);
        startQueue.FreeTensor(inLocal);
        CopyOut(offset, count);
    }

    __aicore__ inline void ProcessScalar(uint64_t begin, uint64_t end)
    {
        for (uint64_t i = begin; i < end; ++i) {
            yGm.SetValue(i, LerpScalarEval(startGm.GetValue(i), endGm.GetValue(i), weight));
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, LERP_BUFFER_NUM> startQueue;
    TQue<QuePosition::VECIN, LERP_BUFFER_NUM> endQueue;
    TQue<QuePosition::VECOUT, LERP_BUFFER_NUM> outQueue;

    GlobalTensor<T> startGm;
    GlobalTensor<T> endGm;
    GlobalTensor<T> yGm;

    uint64_t length = 0;
    uint64_t perCoreElements = 0;
    uint32_t tileLength = 0;
    uint32_t mode = LERP_MODE_NORMAL;
    float weight = 0.0f;
    T weightScalar;
};

template <typename DT_START>
__global__ __aicore__ void lerp(GM_ADDR start, GM_ADDR end, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LerpTilingData);
    GET_TILING_DATA_WITH_STRUCT(LerpTilingData, tilingData, tiling);

    KernelLerp<DT_START> op;
    op.Init(start, end, y, tilingData);
    op.Process();
}
