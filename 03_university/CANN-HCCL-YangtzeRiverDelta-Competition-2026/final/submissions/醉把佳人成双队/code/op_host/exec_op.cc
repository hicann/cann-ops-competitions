#include <algorithm>
#include <array>
#include <limits>
#include <vector>
#include <ccu/ccu_res.h>
#include <hccl/hcomm_primitives.h>
#include "ccu_launch.h"
#include "ccu_res.h"
#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace {

constexpr uint32_t WORKER_START_NOTIFY_ID = 0;
constexpr uint32_t SAG_READY_WORKER_NOTIFY_ID = 1;
constexpr uint32_t SAG_READY_MAIN_NOTIFY_ID = 2;
constexpr uint32_t DIRECT_WORKER_START_NOTIFY_ID = 0;

// Test 23/24极致优化：将 Chunk 大小缩短为 8MB 以极低延迟起步流水线
constexpr uint64_t PIPELINE_CHUNK_BYTES = 8ULL * 1024ULL * 1024ULL;
constexpr uint64_t DIRECT_ALG_MAX_BYTES = 512ULL * 1024ULL;

constexpr uint32_t FAST_CHAIN_MAX_CHUNKS = 128; 
constexpr uint32_t FAST_DIE_ENTRY_MAX = 4;
constexpr uint32_t FAST_SAG_MAX_SCATTER_ARGS = 4U + 2U * MAX_RANK_SIZE;

struct FixedTaskArgs {
    std::array<uint64_t, FAST_SAG_MAX_SCATTER_ARGS> values{};
    uint32_t count = 0;
};

// 极速全展开：测试点 19/22 (2*8 和 4*1 的 512KB) 突破核心！
// 纯平滑展平 CPU 层循环分支，主线程优先 Launch。
HcclResult LaunchDirectFast(const AlgResourceCtx &resCtx, const uint64_t *taskArgs, uint32_t taskArgCount)
{
    const uint32_t numThreads = resCtx.threads.count;
    const ThreadHandle mainThread = resCtx.threads.values[0];

    if (numThreads == 1) {
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread, resCtx.ccuKernelEntries.values[0].directHandle, taskArgs, taskArgCount));
        return HCCL_SUCCESS;
    }
    
    // 强制消除 For 循环，针对多 Die (2个Die) 情况硬编码平铺
    if (numThreads == 2) {
        const ThreadHandle t1 = resCtx.threads.values[1];
        CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, t1, DIRECT_WORKER_START_NOTIFY_ID));
        CHK_RET(HcommThreadNotifyWaitOnThread(t1, DIRECT_WORKER_START_NOTIFY_ID, 0));
        
        // 极限优化：先 Launch 主线程（优先响应 Root），抢跑 NPU 流水线
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread, resCtx.ccuKernelEntries.values[0].directHandle, taskArgs, taskArgCount));
        CHK_RET_CCU(HcommCcuKernelLaunch(t1, resCtx.ccuKernelEntries.values[1].directHandle, taskArgs, taskArgCount));
        
        CHK_RET(HcommThreadNotifyRecordOnThread(t1, mainThread, 1));
        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, 1, 0));
        return HCCL_SUCCESS;
    }

    // 后退通用情况
    for (uint32_t idx = 1; idx < numThreads; ++idx) {
        CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, resCtx.threads.values[idx], DIRECT_WORKER_START_NOTIFY_ID));
        CHK_RET(HcommThreadNotifyWaitOnThread(resCtx.threads.values[idx], DIRECT_WORKER_START_NOTIFY_ID, 0));
    }
    
    // 确保 mainThread 第一时间抢跑
    CHK_RET_CCU(HcommCcuKernelLaunch(mainThread, resCtx.ccuKernelEntries.values[0].directHandle, taskArgs, taskArgCount));
    
    for (uint32_t idx = 1; idx < numThreads; ++idx) {
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads.values[idx], resCtx.ccuKernelEntries.values[idx].directHandle, taskArgs, taskArgCount));
        CHK_RET(HcommThreadNotifyRecordOnThread(resCtx.threads.values[idx], mainThread, idx));
    }
    for (uint32_t idx = 1; idx < numThreads; ++idx) {
        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, idx, 0));
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchChainFast(const AlgResourceCtx &resCtx, uint64_t baseAddr, uint64_t memToken, uint64_t totalBytes, uint32_t chunkCount)
{
    std::array<uint64_t, 2U + 2U * FAST_CHAIN_MAX_CHUNKS> taskArgs{};
    taskArgs[0] = baseAddr; taskArgs[1] = memToken;
    uint64_t offset = 0;
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        const uint64_t bytes = std::min<uint64_t>(PIPELINE_CHUNK_BYTES, totalBytes - offset);
        taskArgs[2U + chunk] = offset; taskArgs[2U + chunkCount + chunk] = bytes;
        offset += bytes;
    }
    CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads.values[0], resCtx.ccuKernelEntries.values[0].chainHandle, taskArgs.data(), 2U + 2U * chunkCount));
    return HCCL_SUCCESS;
}

// 【修复核心】：引入 agArgCount 参数，防止 8+4 大数据场景中，底层只要求2个参数而上层硬塞4个导致的 Crash！
HcclResult LaunchSagOverlappedFast(const AlgResourceCtx &resCtx,
    uint32_t myRank,
    uint32_t root,
    const std::array<FixedTaskArgs, FAST_DIE_ENTRY_MAX> &scatterTaskArgs,
    const std::array<uint64_t, 4> &allGatherTaskArgs,
    uint32_t agArgCount)
{
    const ThreadHandle mainThread = resCtx.threads.values[0];
    const uint32_t numThreads = resCtx.threads.count;
    const uint32_t numEntries = resCtx.ccuKernelEntries.count;

    CHK_PRT_RET(numThreads < 2 || numEntries != numThreads,
        HCCL_ERROR("Invalid SAG resources, threads[%u], entries[%u]", numThreads, numEntries),
        HCCL_E_INTERNAL);

    for (uint32_t idx = 1; idx < numThreads; ++idx) {
        CHK_RET(HcommThreadNotifyRecordOnThread(
            mainThread, resCtx.threads.values[idx], WORKER_START_NOTIFY_ID));
        CHK_RET(HcommThreadNotifyWaitOnThread(
            resCtx.threads.values[idx], WORKER_START_NOTIFY_ID, 0));
    }

    if (myRank == root) {
        // Measured-stable ordering: worker die(s) start before the stream-bound
        // main die. The group builder puts the heavier channel group on worker.
        for (uint32_t idx = 1; idx < numEntries; ++idx) {
            const CcuKernelLaunchEntry &entry = resCtx.ccuKernelEntries.values[idx];
            CHK_PRT_RET(entry.scatterHandle == 0 || entry.allGatherHandle == 0 ||
                    entry.threadIndex != idx || scatterTaskArgs[idx].count == 0,
                HCCL_ERROR("Invalid root worker SAG entry[%u]", idx),
                HCCL_E_INTERNAL);
            CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads.values[idx],
                entry.scatterHandle,
                scatterTaskArgs[idx].values.data(),
                scatterTaskArgs[idx].count));
            CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads.values[idx],
                entry.allGatherHandle,
                allGatherTaskArgs.data(),
                agArgCount));
            CHK_RET(HcommThreadNotifyRecordOnThread(
                resCtx.threads.values[idx], mainThread, idx));
        }

        const CcuKernelLaunchEntry &mainEntry = resCtx.ccuKernelEntries.values[0];
        CHK_PRT_RET(mainEntry.scatterHandle == 0 || mainEntry.allGatherHandle == 0 ||
                mainEntry.threadIndex != 0 || scatterTaskArgs[0].count == 0,
            HCCL_ERROR("Invalid root main SAG entry"),
            HCCL_E_INTERNAL);
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            mainEntry.scatterHandle,
            scatterTaskArgs[0].values.data(),
            scatterTaskArgs[0].count));
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            mainEntry.allGatherHandle,
            allGatherTaskArgs.data(),
            agArgCount));

        for (uint32_t idx = 1; idx < numThreads; ++idx) {
            CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, idx, 0));
        }
        return HCCL_SUCCESS;
    }

    uint32_t rootEntryIndex = INVALID_VALUE_RANKID;
    for (uint32_t idx = 0; idx < numEntries; ++idx) {
        if (resCtx.ccuKernelEntries.values[idx].containsRootChannel != 0) {
            CHK_PRT_RET(rootEntryIndex != INVALID_VALUE_RANKID,
                HCCL_ERROR("Multiple die groups contain root channel"),
                HCCL_E_INTERNAL);
            rootEntryIndex = idx;
        }
    }
    CHK_PRT_RET(rootEntryIndex == INVALID_VALUE_RANKID,
        HCCL_ERROR("No die group contains root channel for rank[%u] root[%u]", myRank, root),
        HCCL_E_INTERNAL);

    for (uint32_t idx = 1; idx < numEntries; ++idx) {
        if (idx == rootEntryIndex) {
            continue;
        }
        const CcuKernelLaunchEntry &entry = resCtx.ccuKernelEntries.values[idx];
        CHK_PRT_RET(entry.allGatherHandle == 0 || entry.threadIndex != idx,
            HCCL_ERROR("Invalid nonroot worker SAG entry[%u]", idx),
            HCCL_E_INTERNAL);
        CHK_RET(HcommThreadNotifyWaitOnThread(
            resCtx.threads.values[idx], SAG_READY_WORKER_NOTIFY_ID, 0));
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads.values[idx],
            entry.allGatherHandle,
            allGatherTaskArgs.data(),
            agArgCount));
        CHK_RET(HcommThreadNotifyRecordOnThread(
            resCtx.threads.values[idx], mainThread, idx));
    }

    const CcuKernelLaunchEntry &rootEntry =
        resCtx.ccuKernelEntries.values[rootEntryIndex];
    CHK_PRT_RET(rootEntry.scatterHandle == 0 || rootEntry.allGatherHandle == 0 ||
            rootEntry.threadIndex != rootEntryIndex ||
            scatterTaskArgs[rootEntryIndex].count == 0,
        HCCL_ERROR("Invalid root-link SAG entry[%u]", rootEntryIndex),
        HCCL_E_INTERNAL);

    if (rootEntryIndex == 0) {
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            rootEntry.scatterHandle,
            scatterTaskArgs[0].values.data(),
            scatterTaskArgs[0].count));
        for (uint32_t idx = 1; idx < numThreads; ++idx) {
            CHK_RET(HcommThreadNotifyRecordOnThread(
                mainThread, resCtx.threads.values[idx], SAG_READY_WORKER_NOTIFY_ID));
        }
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            rootEntry.allGatherHandle,
            allGatherTaskArgs.data(),
            agArgCount));
    } else {
        const ThreadHandle rootThread = resCtx.threads.values[rootEntryIndex];
        CHK_RET_CCU(HcommCcuKernelLaunch(rootThread,
            rootEntry.scatterHandle,
            scatterTaskArgs[rootEntryIndex].values.data(),
            scatterTaskArgs[rootEntryIndex].count));
        CHK_RET(HcommThreadNotifyRecordOnThread(
            rootThread, mainThread, SAG_READY_MAIN_NOTIFY_ID));
        for (uint32_t idx = 1; idx < numThreads; ++idx) {
            if (idx != rootEntryIndex) {
                CHK_RET(HcommThreadNotifyRecordOnThread(
                    rootThread, resCtx.threads.values[idx], SAG_READY_WORKER_NOTIFY_ID));
            }
        }
        CHK_RET_CCU(HcommCcuKernelLaunch(rootThread,
            rootEntry.allGatherHandle,
            allGatherTaskArgs.data(),
            agArgCount));
        CHK_RET(HcommThreadNotifyRecordOnThread(
            rootThread, mainThread, rootEntryIndex));

        const CcuKernelLaunchEntry &mainEntry = resCtx.ccuKernelEntries.values[0];
        CHK_PRT_RET(mainEntry.allGatherHandle == 0 || mainEntry.threadIndex != 0,
            HCCL_ERROR("Invalid nonroot main SAG entry"),
            HCCL_E_INTERNAL);
        CHK_RET(HcommThreadNotifyWaitOnThread(
            mainThread, SAG_READY_MAIN_NOTIFY_ID, 0));
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            mainEntry.allGatherHandle,
            allGatherTaskArgs.data(),
            agArgCount));
    }

    for (uint32_t idx = 1; idx < numThreads; ++idx) {
        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, idx, 0));
    }
    return HCCL_SUCCESS;
}

} // namespace

namespace ops_hccl {

HcclResult ExecOp(const OpParam &param)
{
    CHK_PTR_NULL(param.resCtx);

    AlgResourceCtx resCtx;
    resCtx.DeSerialize(param.resCtx, param.ctxSize);

    const uint64_t typeSize = FastGetTypeSize(param.dataType);
    const uint64_t totalBytes = param.count * typeSize;
    if (param.rankSize <= 1 || totalBytes == 0) return HCCL_SUCCESS;

    uint64_t memToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.outputPtr), totalBytes, &memToken));
    const uint64_t baseAddr = reinterpret_cast<uint64_t>(param.outputPtr);

    if (resCtx.algorithm == BCAST_ALG_DIRECT) {
        const std::array<uint64_t, 3> taskArgs = {baseAddr, memToken, totalBytes};
        CHK_RET(LaunchDirectFast(resCtx, taskArgs.data(), param.myRank == param.root ? 3U : 2U));
        return HCCL_SUCCESS;
    }

    if (resCtx.algorithm == BCAST_ALG_CHAIN_PIPELINE) {
        const uint32_t chunkCount = static_cast<uint32_t>((totalBytes + PIPELINE_CHUNK_BYTES - 1ULL) / PIPELINE_CHUNK_BYTES);
        CHK_RET(LaunchChainFast(resCtx, baseAddr, memToken, totalBytes, chunkCount));
        return HCCL_SUCCESS;
    }

    if (resCtx.algorithm == BCAST_ALG_SAG_SPLIT || resCtx.algorithm == BCAST_ALG_8P4_PREFETCH_SAG) {
        const bool is8p4 = (resCtx.algorithm == BCAST_ALG_8P4_PREFETCH_SAG);

        std::array<uint64_t, MAX_RANK_SIZE> fastSliceOffsets{};
        std::array<uint64_t, MAX_RANK_SIZE> fastSliceSizes{};
        const uint64_t baseElements = param.count / param.rankSize;
        const uint64_t residualElements = param.count % param.rankSize;
        uint64_t currentOffset = 0;

        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            const uint64_t bytes = (baseElements + (rank < residualElements ? 1ULL : 0ULL)) * typeSize;
            fastSliceOffsets[rank] = currentOffset;
            fastSliceSizes[rank] = bytes;
            currentOffset += bytes;
        }

        std::array<FixedTaskArgs, FAST_DIE_ENTRY_MAX> scatterArgs{};
        for (uint32_t entryIndex = 0; entryIndex < resCtx.ccuKernelEntries.count; ++entryIndex) {
            const CcuKernelLaunchEntry &entry = resCtx.ccuKernelEntries.values[entryIndex];
            FixedTaskArgs &args = scatterArgs[entryIndex];
            args.values[0] = baseAddr;
            args.values[1] = memToken;
            
            if (param.myRank != param.root) {
                args.count = 2; // 无论是 Split 还是 8P4，只要不是 root，散发内核固定只取 2 个参数
                continue;
            }
            
            uint32_t argIdx = 2;
            if (is8p4) {
                args.values[argIdx++] = fastSliceOffsets[param.root];
                args.values[argIdx++] = fastSliceSizes[param.root];
                for (uint32_t idx = 0; idx < entry.channelCount; ++idx) {
                    args.values[argIdx + idx] = fastSliceOffsets[entry.remoteRanks[idx]];
                    args.values[argIdx + entry.channelCount + idx] = fastSliceSizes[entry.remoteRanks[idx]];
                }
                args.count = 4U + 2U * entry.channelCount;
            } else {
                for (uint32_t idx = 0; idx < entry.channelCount; ++idx) {
                    args.values[argIdx + idx] = fastSliceOffsets[entry.remoteRanks[idx]];
                    args.values[argIdx + entry.channelCount + idx] = fastSliceSizes[entry.remoteRanks[idx]];
                }
                args.count = 2U + 2U * entry.channelCount;
            }
        }

        std::array<uint64_t, 4> allGatherArgs{};
        allGatherArgs[0] = baseAddr;
        allGatherArgs[1] = memToken;
        uint32_t allGatherArgCount = 2;

        if (is8p4) {
            if (param.myRank != param.root) {
                allGatherArgs[2] = fastSliceOffsets[param.myRank];
                allGatherArgs[3] = fastSliceSizes[param.myRank];
                allGatherArgCount = 4;
            }
        } else {
            allGatherArgs[2] = fastSliceOffsets[param.myRank];
            allGatherArgs[3] = fastSliceSizes[param.myRank];
            allGatherArgCount = 4;
        }

        if (totalBytes > DIRECT_ALG_MAX_BYTES && resCtx.threads.count > 1 && resCtx.threads.count <= FAST_DIE_ENTRY_MAX) {
            // 将精确统计的 agArgCount 送入，彻底屏蔽底层参数校验错误
            CHK_RET(LaunchSagOverlappedFast(resCtx, param.myRank, param.root, scatterArgs, allGatherArgs, allGatherArgCount));
        } else {
            for (uint32_t i = 0; i < resCtx.ccuKernelEntries.count; ++i) {
                CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads.values[i], resCtx.ccuKernelEntries.values[i].scatterHandle, scatterArgs[i].values.data(), scatterArgs[i].count));
            }
            for (uint32_t i = 0; i < resCtx.ccuKernelEntries.count; ++i) {
                CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads.values[i], resCtx.ccuKernelEntries.values[i].allGatherHandle, allGatherArgs.data(), allGatherArgCount));
            }
        }
        return HCCL_SUCCESS;
    }

    return HCCL_SUCCESS;
}

} // namespace ops_hccl