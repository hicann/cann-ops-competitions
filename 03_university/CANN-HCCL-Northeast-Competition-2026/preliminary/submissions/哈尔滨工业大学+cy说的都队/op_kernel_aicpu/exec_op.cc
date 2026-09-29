#include "custom.h"
#include "log.h"
#include "exec_op.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace {

constexpr uint64_t FP32_BYTES = sizeof(float);

// ---- 大包（推送）路径的通知布局 ----
constexpr uint32_t PIPELINE_SLOT_NUM = 2;
constexpr uint32_t DATA_NOTIFY_BASE = 0;
constexpr uint32_t ACK_NOTIFY_BASE = PIPELINE_SLOT_NUM;
constexpr uint32_t WORKER_START_NOTIFY_IDX = 0;

} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    HCCL_INFO("Executing AICPU Kernel on Ascend NPU");

    if (param.root >= param.rankSize || param.myRank >= param.rankSize ||
        param.dataType != HCCL_DATA_TYPE_FP32 || param.count > UINT64_MAX / sizeof(float) / param.rankSize) {
        return HCCL_E_PARA;
    }
    const uint64_t bytes = param.count * sizeof(float);
    if (bytes == 0) {
        return HCCL_SUCCESS;
    }
    auto *input = static_cast<char *>(param.inputPtr);
    auto *output = static_cast<char *>(param.outputPtr);
    const ThreadHandle mainThread = resCtx.aicpuThread;
    // recvBuf 与 sendBuf 重合且 root!=0 时，写 output 会覆盖 rank0 的源数据，必须等所有写发完再拷。
    const bool copyRootLast = output == input && param.root != 0;
    if (param.rankSize == 1) {
        if (input != output) {
            CHK_RET(HcommLocalCopyOnThread(mainThread, output, input, bytes));
        }
        return HCCL_SUCCESS;
    }

    // 根节点每个对端独占两个slot；各rank从完整通道表求相同的最小容量。
    uint64_t bufferBytes = resCtx.localBuffer.size;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer != param.myRank) {
            bufferBytes = std::min(bufferBytes, resCtx.channels[peer].remoteCclMem.size);
        }
    }
    auto *scratch = static_cast<char *>(resCtx.localBuffer.addr);

    // ==================== 小包：两层四叉分发树 ====================
    // 每rank<=1MiB的小消息一次性准备全部切片，用两层四叉分发树，
    // 把root的15组通知/等待缩减为4组。
    // ready保护源数据，子树完成ack保护buffer复用，不需要发送worker。
    if (bytes <= (1ULL << 20) && bytes * param.rankSize <= bufferBytes) {
        const uint32_t virtualRank = param.myRank >= param.root ? param.myRank - param.root :
            param.myRank + (param.rankSize - param.root);
        const uint32_t groupSpan = (param.rankSize + 2) / 4;
        const uint32_t leader = virtualRank == 0 ? 0 : 1 + (virtualRank - 1) / groupSpan * groupSpan;
        const uint32_t parent = virtualRank == leader ? param.root : (leader + param.root) % param.rankSize;
        const uint32_t childBegin = virtualRank + 1;
        const uint32_t childEnd = virtualRank == 0 ? param.rankSize :
            (virtualRank == leader ? std::min(param.rankSize, virtualRank + groupSpan) : childBegin);
        const uint32_t childStep = virtualRank == 0 ? groupSpan : 1;
        if (virtualRank == 0) {
            // 按相对root的逻辑rank排列，任意root最多两次连续copy。
            const uint64_t tail = (param.rankSize - param.root) * bytes;
            CHK_RET(HcommLocalCopyOnThread(mainThread, scratch, input + param.root * bytes, tail));
            if (param.root != 0) {
                CHK_RET(HcommLocalCopyOnThread(mainThread, scratch + tail, input, param.root * bytes));
            }
        } else {
            const ChannelInfo &channel = resCtx.channels[parent];
            auto *remote = static_cast<char *>(channel.remoteCclMem.addr);
            const uint64_t remoteOffset = (virtualRank == leader ? virtualRank : virtualRank - leader) * bytes;
            const uint64_t receiveBytes = (virtualRank == leader ?
                std::min(groupSpan, param.rankSize - virtualRank) : 1) * bytes;
            CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, channel.handle, 2, CUSTOM_TIMEOUT));
            CHK_RET(HcommReadOnThread(mainThread, channel.handle, scratch, remote + remoteOffset, receiveBytes));
        }
        for (uint32_t child = childBegin; child < childEnd; child += childStep) {
            const uint32_t peer = (child + param.root) % param.rankSize;
            CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, resCtx.channels[peer].handle, 2));
        }
        if (virtualRank != 0 || output != input + param.root * bytes) {
            CHK_RET(HcommLocalCopyOnThread(mainThread, output, scratch, bytes));
        }
        for (uint32_t child = childBegin; child < childEnd; child += childStep) {
            const uint32_t peer = (child + param.root) % param.rankSize;
            CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, resCtx.channels[peer].handle, 0, CUSTOM_TIMEOUT));
            CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, resCtx.channels[peer].handle, 4));
        }
        if (virtualRank != 0) {
            CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, resCtx.channels[parent].handle, 0));
            CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, resCtx.channels[parent].handle, 4, CUSTOM_TIMEOUT));
        }
        return HCCL_SUCCESS;
    }

    if (resCtx.threads.size() < 2) {
        HCCL_ERROR("[%s] invalid AICPU thread resources, threadNum[%zu]", __func__, resCtx.threads.size());
        return HCCL_E_INTERNAL;
    }
    auto threadIndex = [&param](uint32_t peer) -> uint32_t {
        return peer < param.myRank ? peer + 1 : peer;
    };

    const uint32_t pipelineSlotNum = bytes > bufferBytes ? PIPELINE_SLOT_NUM : 1;
    uint64_t chunkBytes = std::min<uint64_t>(bytes, bufferBytes / pipelineSlotNum);
    chunkBytes &= ~(FP32_BYTES - 1);
    if (chunkBytes < FP32_BYTES) {
        HCCL_ERROR("[%s] HCCL buffer is too small for FP32 data, chunkBytes[%lu]", __func__, chunkBytes);
        return HCCL_E_MEMORY;
    }
    const uint64_t chunkCount = (bytes - 1) / chunkBytes + 1;

    if (param.myRank == param.root) {
        for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
            if (peer == param.root) {
                continue;
            }
            const ChannelInfo &channel = resCtx.channels[peer];
            if (channel.remoteCclMem.addr == nullptr ||
                channel.remoteCclMem.size < chunkBytes * pipelineSlotNum) {
                HCCL_ERROR("[%s] invalid remote HCCL buffer for peer[%u], addr[%p], size[%lu]",
                    __func__, peer, channel.remoteCclMem.addr, channel.remoteCclMem.size);
                return HCCL_E_MEMORY;
            }
        }

        // 先让所有 worker 等待控制线程的启动通知，保证 worker 在 Host 通知到达后才开始通信。
        for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
            if (peer == param.root) {
                continue;
            }
            const ThreadHandle worker = resCtx.threads[threadIndex(peer)];
            CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, worker, WORKER_START_NOTIFY_IDX));
            CHK_RET(HcommThreadNotifyWaitOnThread(worker, WORKER_START_NOTIFY_IDX, CUSTOM_TIMEOUT));
        }

        const uint64_t localSourceOffset = static_cast<uint64_t>(param.root) * bytes;
        // 双缓冲：对端消费 chunk N 的同时 root 写 chunk N+1，复用同一 slot 时才等 ACK。
        for (uint64_t chunk = 0; chunk < chunkCount; ++chunk) {
            const uint64_t chunkOffset = chunk * chunkBytes;
            const uint64_t chunkLength = std::min(chunkBytes, bytes - chunkOffset);
            const uint32_t slot = static_cast<uint32_t>(chunk % pipelineSlotNum);
            const uint64_t slotOffset = static_cast<uint64_t>(slot) * chunkBytes;

            for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
                if (peer == param.root) {
                    continue;
                }
                const ChannelInfo &channel = resCtx.channels[peer];
                const ThreadHandle worker = resCtx.threads[threadIndex(peer)];
                auto *remoteBuffer = static_cast<uint8_t *>(channel.remoteCclMem.addr);
                if (chunk >= pipelineSlotNum) {
                    CHK_RET(HcommChannelNotifyWaitOnThread(worker, channel.handle,
                        ACK_NOTIFY_BASE + slot, CUSTOM_TIMEOUT));
                }
                CHK_RET(HcommWriteWithNotifyOnThread(worker, channel.handle, remoteBuffer + slotOffset,
                    input + static_cast<uint64_t>(peer) * bytes + chunkOffset, chunkLength,
                    DATA_NOTIFY_BASE + slot));
            }

            // root 自身那份：与各对端区段的写无数据依赖，放在写投递之后即可与在途写重叠。
            // 原地覆盖 input 首部的情形（copyRootLast）推到所有写完成之后，见函数末尾。
            if (!copyRootLast && output != input + localSourceOffset) {
                CHK_RET(HcommLocalCopyOnThread(mainThread, output + chunkOffset,
                    input + localSourceOffset + chunkOffset, chunkLength));
            }
        }

        // 排空每个 channel 的所有 slot，避免 Host 完成通知提前触发。
        const uint32_t pendingSlotNum = static_cast<uint32_t>(std::min<uint64_t>(chunkCount, pipelineSlotNum));
        for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
            if (peer == param.root) {
                continue;
            }
            const ChannelInfo &channel = resCtx.channels[peer];
            const ThreadHandle worker = resCtx.threads[threadIndex(peer)];
            for (uint32_t slot = 0; slot < pendingSlotNum; ++slot) {
                CHK_RET(HcommChannelNotifyWaitOnThread(worker, channel.handle,
                    ACK_NOTIFY_BASE + slot, CUSTOM_TIMEOUT));
            }
        }

        // 所有 worker 汇合回控制线程。
        for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
            if (peer == param.root) {
                continue;
            }
            const uint32_t index = threadIndex(peer);
            CHK_RET(HcommThreadNotifyRecordOnThread(resCtx.threads[index], mainThread, index));
        }
        for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
            if (peer == param.root) {
                continue;
            }
            CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, threadIndex(peer), CUSTOM_TIMEOUT));
        }

        // recvBuf 与 sendBuf 重合时，先完成所有切片发送，再覆盖输入首部。
        if (copyRootLast) {
            CHK_RET(HcommLocalCopyOnThread(mainThread, output, input + localSourceOffset, bytes));
        }
        return HCCL_SUCCESS;
    }

    // 非 root 侧：等待 root 写入本端 Buffer 后，本地拷贝到 recv 输出。
    const ChannelInfo &rootChannel = resCtx.channels[param.root];
    const auto *localBuffer = static_cast<const uint8_t *>(resCtx.localBuffer.addr);
    for (uint64_t chunk = 0; chunk < chunkCount; ++chunk) {
        const uint64_t chunkOffset = chunk * chunkBytes;
        const uint64_t chunkLength = std::min(chunkBytes, bytes - chunkOffset);
        const uint32_t slot = static_cast<uint32_t>(chunk % pipelineSlotNum);
        const uint64_t slotOffset = static_cast<uint64_t>(slot) * chunkBytes;
        CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, rootChannel.handle,
            DATA_NOTIFY_BASE + slot, CUSTOM_TIMEOUT));
        CHK_RET(HcommLocalCopyOnThread(mainThread, output + chunkOffset,
            localBuffer + slotOffset, chunkLength));
        CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, rootChannel.handle,
            ACK_NOTIFY_BASE + slot));
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
