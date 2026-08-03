#include <algorithm>
#include "custom.h"
#include "log.h"
#include "exec_op.h"
#include "common.h"

namespace ops_hccl {
constexpr uint64_t HCCL_MIN_SLICE_ALIGN = 128;
constexpr uint64_t UB_MAX_DATA_SIZE = 256 * 1024 * 1024;

constexpr uint32_t DT_SIZE_TABLE[] = {
    sizeof(int8_t),   // HCCL_DATA_TYPE_INT8
    sizeof(int16_t),  // HCCL_DATA_TYPE_INT16
    sizeof(int32_t),  // HCCL_DATA_TYPE_INT32
    2,                // HCCL_DATA_TYPE_FP16
    sizeof(float),    // HCCL_DATA_TYPE_FP32
    sizeof(int64_t),  // HCCL_DATA_TYPE_INT64
    sizeof(uint64_t), // HCCL_DATA_TYPE_UINT64
    sizeof(uint8_t),  // HCCL_DATA_TYPE_UINT8
    sizeof(uint16_t), // HCCL_DATA_TYPE_UINT16
    sizeof(uint32_t), // HCCL_DATA_TYPE_UINT32
    sizeof(double)    // HCCL_DATA_TYPE_FP64
};

static uint32_t GetDataTypeSize(HcclDataType dataType)
{
    return DT_SIZE_TABLE[static_cast<uint32_t>(dataType)];
}

static HcclResult ThreadSyncBefore(const std::vector<ThreadHandle> &threads)
{
    for (uint32_t i = 1; i < threads.size(); i++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[0], threads[i], 0)));
    }
    for (uint32_t i = 1; i < threads.size(); i++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[i], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

static HcclResult ThreadSyncAfter(const std::vector<ThreadHandle> &threads)
{
    for (uint32_t i = 1; i < threads.size(); i++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[0], i - 1, CUSTOM_TIMEOUT)));
    }
    for (uint32_t i = 1; i < threads.size(); i++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[i], threads[0], i - 1)));
    }
    return HCCL_SUCCESS;
}

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    uint32_t dataTypeSize = GetDataTypeSize(param.dataType);
    uint64_t dataSize = param.count * dataTypeSize;
    uint64_t count = param.count;

    if (param.rankSize == 1) {
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, dataSize)));
        return HCCL_SUCCESS;
    }

    const uint32_t cclBuffMultiplier = param.rankSize;
    const uint64_t cclBufferSize = resCtx.localBuffer.size;
    const uint64_t cclBuffBound = cclBufferSize / cclBuffMultiplier / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
    const uint64_t maxDataSizePerLoop = std::min(UB_MAX_DATA_SIZE, cclBuffBound);
    const uint64_t maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize;
    const uint64_t loopCount = count / maxDataCountPerLoop + static_cast<uint64_t>(count % maxDataCountPerLoop != 0);
    void *const cclBuffAddr = resCtx.localBuffer.addr;
    uint64_t processedDataCount = 0;

    for (uint64_t loop = 0; loop < loopCount; loop++) {
        const uint64_t sliceCount = std::min(maxDataCountPerLoop, count - loop * maxDataCountPerLoop);
        const uint64_t sliceSize = sliceCount * dataTypeSize;
        const uint64_t inputOffset = processedDataCount * dataTypeSize;

        void *const curInputAddr = static_cast<void *>(static_cast<uint8_t *>(param.inputPtr) + inputOffset);
        const uint64_t myOutputOffset = inputOffset + param.myRank * dataSize;
        void *const myOutputAddr = static_cast<void *>(static_cast<uint8_t *>(param.outputPtr) + myOutputOffset);

        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.aicpuThread, myOutputAddr, curInputAddr, sliceSize)));

        CHK_RET(ThreadSyncBefore(resCtx.threads));

        // Phase 1: Issue all ACK notifications across all channels
        const uint32_t threadCount = static_cast<uint32_t>(resCtx.threads.size());
        for (uint32_t i = 0; i < threadCount; i++) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(resCtx.threads[i], resCtx.channels[i].handle, NOTIFY_IDX_ACK)));
        }

        // Phase 2: Wait for ACK, write data, and signal completion for each channel
        const uint64_t myRemoteOffset = sliceSize * param.myRank;
        for (uint32_t i = 0; i < threadCount; i++) {
            const ChannelHandle remoteChannelHandle = resCtx.channels[i].handle;
            void *const remoteCclBuffAddr = static_cast<void *>(static_cast<uint8_t *>(resCtx.channels[i].remoteCclMem.addr) + myRemoteOffset);

            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(resCtx.threads[i], remoteChannelHandle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(resCtx.threads[i], remoteChannelHandle, remoteCclBuffAddr, curInputAddr, sliceSize)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(resCtx.threads[i], remoteChannelHandle, NOTIFY_IDX_DATA_SIGNAL)));
        }

        // Phase 3: Wait for data signal and copy remote data to output
        for (uint32_t i = 0; i < threadCount; i++) {
            const ChannelHandle remoteChannelHandle = resCtx.channels[i].handle;
            const uint32_t remoteRank = resCtx.channels[i].remoteRank;

            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(resCtx.threads[i], remoteChannelHandle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));

            const uint64_t remoteCclBuffOffset = sliceSize * remoteRank;
            const uint64_t outputOffset = inputOffset + remoteRank * dataSize;
            void *const remoteDataAddr = static_cast<void *>(static_cast<uint8_t *>(cclBuffAddr) + remoteCclBuffOffset);
            void *const curOutputAddr = static_cast<void *>(static_cast<uint8_t *>(param.outputPtr) + outputOffset);
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.threads[i], curOutputAddr, remoteDataAddr, sliceSize)));
        }

        CHK_RET(ThreadSyncAfter(resCtx.threads));

        processedDataCount += sliceCount;
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl