#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t kAssignUbBytes = 192 * 1024;
constexpr uint32_t kAssignBufferNum = 3;
constexpr uint32_t kAssignAlignElements = 32;

template <typename T>
__aicore__ inline uint64_t MinU64(uint64_t a, uint64_t b)
{
    return a < b ? a : b;
}

__aicore__ inline void SetMte2Mte3(uint32_t bufferIndex)
{
    if (bufferIndex == 0) {
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    } else {
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID1);
    }
}

__aicore__ inline void WaitMte2Mte3(uint32_t bufferIndex)
{
    if (bufferIndex == 0) {
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    } else {
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID1);
    }
}

__aicore__ inline void SetMte3Mte2(uint32_t bufferIndex)
{
    if (bufferIndex == 0) {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    } else {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
}

__aicore__ inline void WaitMte3Mte2(uint32_t bufferIndex)
{
    if (bufferIndex == 0) {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    } else {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
}

template <typename T>
__aicore__ inline void CopyKernel(GM_ADDR ref, GM_ADDR value, const AssignTilingData& tilingData)
{
    GlobalTensor<T> refGm;
    GlobalTensor<T> valueGm;
    refGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(ref), tilingData.totalSize);
    valueGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(value), tilingData.totalSize);

    const uint64_t start = GetBlockIdx() * tilingData.perCoreSize;
    if (start >= tilingData.totalSize) {
        return;
    }
    const uint64_t count = MinU64<uint64_t>(tilingData.perCoreSize, tilingData.totalSize - start);

    TPipe pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, kAssignBufferNum> dataQueue;
    pipe.InitBuffer(dataQueue, kAssignBufferNum, kAssignUbBytes / kAssignBufferNum);

    const uint32_t tileSize = (kAssignUbBytes / kAssignBufferNum) / sizeof(T);
    uint64_t offset = 0;
    uint64_t queuedOffsets[kAssignBufferNum] = {0, 0};
    uint32_t queuedTiles[kAssignBufferNum] = {0, 0};
    uint32_t enqueueCount = 0;
    uint32_t dequeueCount = 0;
    for (; offset + kAssignAlignElements <= count;) {
        uint32_t tile = static_cast<uint32_t>(MinU64<uint64_t>(tileSize, count - offset));
        tile = tile / kAssignAlignElements * kAssignAlignElements;
        if (tile == 0) {
            break;
        }

        LocalTensor<T> local = dataQueue.AllocTensor<T>();
        DataCopy(local, valueGm[start + offset], tile);
        dataQueue.EnQue<QuePosition::VECIN, QuePosition::VECOUT, T>(local);
        queuedOffsets[enqueueCount % kAssignBufferNum] = offset;
        queuedTiles[enqueueCount % kAssignBufferNum] = tile;
        ++enqueueCount;

        if (enqueueCount - dequeueCount >= kAssignBufferNum) {
            const uint32_t outputIndex = dequeueCount % kAssignBufferNum;
            local = dataQueue.DeQue<QuePosition::VECIN, QuePosition::VECOUT, T>();
            DataCopy(refGm[start + queuedOffsets[outputIndex]], local, queuedTiles[outputIndex]);
            dataQueue.FreeTensor(local);
            ++dequeueCount;
        }
        offset += tile;
    }

    while (dequeueCount < enqueueCount) {
        const uint32_t outputIndex = dequeueCount % kAssignBufferNum;
        LocalTensor<T> local = dataQueue.DeQue<QuePosition::VECIN, QuePosition::VECOUT, T>();
        DataCopy(refGm[start + queuedOffsets[outputIndex]], local, queuedTiles[outputIndex]);
        dataQueue.FreeTensor(local);
        ++dequeueCount;
    }

    if (offset < count) {
        const uint32_t tile = static_cast<uint32_t>(count - offset);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tile * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        LocalTensor<T> local = dataQueue.AllocTensor<T>();
        DataCopyPad(local, valueGm[start + offset], copyParams, padParams);
        dataQueue.EnQue<QuePosition::VECIN, QuePosition::VECOUT, T>(local);
        local = dataQueue.DeQue<QuePosition::VECIN, QuePosition::VECOUT, T>();
        DataCopyPad(refGm[start + offset], local, copyParams);
        dataQueue.FreeTensor(local);
    }
}

extern "C" __global__ __aicore__ void assign(GM_ADDR ref, GM_ADDR value, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
#if ORIG_DTYPE_REF == DT_FLOAT16
    CopyKernel<half>(ref, value, tiling_data);
#elif ORIG_DTYPE_REF == DT_BF16
    CopyKernel<bfloat16_t>(ref, value, tiling_data);
#elif ORIG_DTYPE_REF == DT_FLOAT
    CopyKernel<float>(ref, value, tiling_data);
#elif ORIG_DTYPE_REF == DT_INT32
    CopyKernel<int32_t>(ref, value, tiling_data);
#elif ORIG_DTYPE_REF == DT_INT16
    CopyKernel<int16_t>(ref, value, tiling_data);
#elif ORIG_DTYPE_REF == DT_UINT8
    CopyKernel<uint8_t>(ref, value, tiling_data);
#elif ORIG_DTYPE_REF == DT_INT8
    CopyKernel<int8_t>(ref, value, tiling_data);
#elif ORIG_DTYPE_REF == DT_BOOL
    CopyKernel<bool>(ref, value, tiling_data);
#endif
}
