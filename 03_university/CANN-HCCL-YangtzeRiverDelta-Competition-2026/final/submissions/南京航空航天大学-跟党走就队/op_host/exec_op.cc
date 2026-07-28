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
constexpr uint32_t SAG_READY_MAIN_NOTIFY_ID = 0;
constexpr uint32_t PHASE_DIRECT = 0;
constexpr uint32_t PHASE_SCATTER = 1;
constexpr uint32_t PHASE_ALLGATHER = 2;
constexpr uint32_t PHASE_FUSED = 3;
constexpr uint64_t DIRECT_ALG_MAX_BYTES = 512ULL * 1024ULL;

constexpr uint32_t FAST_DIE_ENTRY_MAX = 4;
constexpr uint32_t FAST_8P4_MAX_SCATTER_ARGS = 4U + 2U * MAX_RANK_SIZE;

struct FixedTaskArgs {
    std::array<uint64_t, FAST_8P4_MAX_SCATTER_ARGS> values{};
    uint32_t count = 0;
};

CcuKernelHandle GetPhaseHandle(const CcuKernelLaunchEntry &entry, uint32_t phase)
{
    switch (phase) {
        case PHASE_DIRECT:
            return entry.directHandle;
        case PHASE_SCATTER:
            return entry.scatterHandle;
        case PHASE_ALLGATHER:
            return entry.allGatherHandle;
        case PHASE_FUSED:
            return entry.fusedHandle;
        default:
            return 0;
    }
}

std::vector<uint64_t> BuildScatterTaskArgs(const CcuKernelLaunchEntry &entry,
    uint32_t myRank,
    uint32_t root,
    uint64_t baseAddr,
    uint64_t memToken,
    const std::vector<uint64_t> &sliceOffsets,
    const std::vector<uint64_t> &sliceSizes)
{
    std::vector<uint64_t> args;
    args.reserve(2U + (myRank == root ? 2U * entry.channelCount : 0U));
    args.push_back(baseAddr);
    args.push_back(memToken);

    // 非root Scatter Kernel只执行地址交换/等待，静态指令中只有2次LoadArg。
    if (myRank != root) {
        return args;
    }

    // root的每个Die Kernel仅加载本组实际对端的切片参数，顺序必须与
    // 注册Kernel时group.remoteRanks的顺序一致。
    for (uint32_t idx = 0; idx < entry.channelCount; ++idx) {
        args.push_back(sliceOffsets[entry.remoteRanks[idx]]);
    }
    for (uint32_t idx = 0; idx < entry.channelCount; ++idx) {
        args.push_back(sliceSizes[entry.remoteRanks[idx]]);
    }
    return args;
}

std::vector<uint64_t> Build8p4ScatterTaskArgs(const CcuKernelLaunchEntry &entry,
    uint32_t myRank,
    uint32_t root,
    uint64_t baseAddr,
    uint64_t memToken,
    const std::vector<uint64_t> &sliceOffsets,
    const std::vector<uint64_t> &sliceSizes)
{
    std::vector<uint64_t> args;
    args.reserve(4U + (myRank == root ? 2U * entry.channelCount : 0U));
    args.push_back(baseAddr);
    args.push_back(memToken);

    // 非root Scatter只向root提供接收地址并等待数据，静态路径仅有2次LoadArg。
    if (myRank != root) {
        return args;
    }

    // 8+4预取Scatter Kernel固定加载：
    // baseAddr、memToken、rootOffset、rootSize，
    // 再加载当前Die组每个peer的offset/size。
    //
    // 参数数量必须严格为：
    // 4 + 2 * entry.channelCount。
    args.push_back(sliceOffsets[root]);
    args.push_back(sliceSizes[root]);

    for (uint32_t idx = 0; idx < entry.channelCount; ++idx) {
        args.push_back(sliceOffsets[entry.remoteRanks[idx]]);
    }
    for (uint32_t idx = 0; idx < entry.channelCount; ++idx) {
        args.push_back(sliceSizes[entry.remoteRanks[idx]]);
    }

    return args;
}

std::vector<uint64_t> Build8p4AllGatherTaskArgs(uint64_t baseAddr,
    uint64_t memToken,
    uint32_t myRank,
    uint32_t root,
    const std::vector<uint64_t> &sliceOffsets,
    const std::vector<uint64_t> &sliceSizes)
{
    // root在Scatter阶段已经把自己的slice发给所有peer，AllGather仅参与最终同步。
    if (myRank == root) {
        return {baseAddr, memToken};
    }
    return {
        baseAddr,
        memToken,
        sliceOffsets[myRank],
        sliceSizes[myRank],
    };
}

std::vector<uint64_t> BuildAllGatherTaskArgs(uint64_t baseAddr,
    uint64_t memToken,
    uint32_t myRank,
    const std::vector<uint64_t> &sliceOffsets,
    const std::vector<uint64_t> &sliceSizes)
{
    return {
        baseAddr,
        memToken,
        sliceOffsets[myRank],
        sliceSizes[myRank],
    };
}

HcclResult LaunchKernelPhase(const AlgResourceCtx &resCtx,
    uint32_t phase,
    const std::vector<uint64_t> &taskArgs)
{
    CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("No CCU thread resource"), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.ccuKernelEntries.size() != resCtx.threads.size(),
        HCCL_ERROR("Kernel/thread size mismatch, kernels[%zu], threads[%zu]",
            resCtx.ccuKernelEntries.size(),
            resCtx.threads.size()),
        HCCL_E_INTERNAL);

    const ThreadHandle mainThread = resCtx.threads[0];

    // 先放行从Thread。相同notifyId位于不同从Thread上，不会互相覆盖。
    for (uint32_t idx = 1; idx < resCtx.threads.size(); ++idx) {
        CHK_RET(HcommThreadNotifyRecordOnThread(
            mainThread, resCtx.threads[idx], WORKER_START_NOTIFY_ID));
        CHK_RET(HcommThreadNotifyWaitOnThread(
            resCtx.threads[idx], WORKER_START_NOTIFY_ID, 0));
    }

    // 从Die先下发，并在该Kernel结束后向主Thread回报。
    for (uint32_t idx = 1; idx < resCtx.ccuKernelEntries.size(); ++idx) {
        const CcuKernelLaunchEntry &entry = resCtx.ccuKernelEntries[idx];
        const CcuKernelHandle handle = GetPhaseHandle(entry, phase);
        CHK_PRT_RET(handle == 0 || entry.threadIndex != idx,
            HCCL_ERROR("Invalid worker entry[%u], phase[%u], threadIndex[%u]",
                idx,
                phase,
                entry.threadIndex),
            HCCL_E_INTERNAL);

        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[idx],
            handle,
            taskArgs.data(),
            static_cast<uint32_t>(taskArgs.size())));
        CHK_RET(HcommThreadNotifyRecordOnThread(resCtx.threads[idx], mainThread, idx));
    }

    const CcuKernelLaunchEntry &mainEntry = resCtx.ccuKernelEntries[0];
    const CcuKernelHandle mainHandle = GetPhaseHandle(mainEntry, phase);
    CHK_PRT_RET(mainHandle == 0 || mainEntry.threadIndex != 0,
        HCCL_ERROR("Invalid main entry, phase[%u], threadIndex[%u]",
            phase,
            mainEntry.threadIndex),
        HCCL_E_INTERNAL);
    CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
        mainHandle,
        taskArgs.data(),
        static_cast<uint32_t>(taskArgs.size())));

    for (uint32_t idx = 1; idx < resCtx.threads.size(); ++idx) {
        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, idx, 0));
    }
    return HCCL_SUCCESS;
}

/*
 * Direct专用微秒级快路径。
 *
 * 与LaunchKernelPhase保持相同的Thread/Notify顺序，只去掉：
 * 1. std::vector<uint64_t>的堆分配；
 * 2. GetPhaseHandle中的phase分支；
 * 3. 通用phase路径中的额外判断。
 */
HcclResult LaunchDirectFast(
    const AlgResourceCtx &resCtx,
    const uint64_t *taskArgs,
    uint32_t taskArgCount)
{
    CHK_PRT_RET(resCtx.threads.empty(),
        HCCL_ERROR("No CCU thread resource"),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.ccuKernelEntries.size() != resCtx.threads.size(),
        HCCL_ERROR("Direct kernel/thread size mismatch, kernels[%zu], threads[%zu]",
            resCtx.ccuKernelEntries.size(),
            resCtx.threads.size()),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(taskArgs == nullptr || taskArgCount != 2U,
        HCCL_ERROR("Invalid direct taskArgCount[%u]", taskArgCount),
        HCCL_E_INTERNAL);

    if (resCtx.ccuKernelEntries.size() == 1) {
        const CcuKernelLaunchEntry &entry = resCtx.ccuKernelEntries[0];
        CHK_PRT_RET(entry.directHandle == 0 || entry.threadIndex != 0,
            HCCL_ERROR("Invalid single-die direct entry, threadIndex[%u]",
                entry.threadIndex),
            HCCL_E_INTERNAL);
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0],
            entry.directHandle,
            taskArgs,
            taskArgCount));
        return HCCL_SUCCESS;
    }

    const ThreadHandle mainThread = resCtx.threads[0];

    /*
     * CheckerV3要求每条slave stream的首个任务必须是local WAIT。
     * 因此在所有worker CCU_GRAPH之前恢复main->worker启动握手。
     * 不能直接以CCU_GRAPH作为slave stream首任务。
     */
    for (uint32_t idx = 1; idx < resCtx.threads.size(); ++idx) {
        CHK_RET(HcommThreadNotifyRecordOnThread(
            mainThread,
            resCtx.threads[idx],
            WORKER_START_NOTIFY_ID));
        CHK_RET(HcommThreadNotifyWaitOnThread(
            resCtx.threads[idx],
            WORKER_START_NOTIFY_ID,
            0));
    }

    for (uint32_t idx = 1; idx < resCtx.ccuKernelEntries.size(); ++idx) {
        const CcuKernelLaunchEntry &entry = resCtx.ccuKernelEntries[idx];
        CHK_PRT_RET(entry.directHandle == 0 || entry.threadIndex != idx,
            HCCL_ERROR("Invalid direct worker[%u], threadIndex[%u]",
                idx,
                entry.threadIndex),
            HCCL_E_INTERNAL);
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[idx],
            entry.directHandle,
            taskArgs,
            taskArgCount));
        CHK_RET(HcommThreadNotifyRecordOnThread(
            resCtx.threads[idx], mainThread, idx));
    }

    const CcuKernelLaunchEntry &mainEntry = resCtx.ccuKernelEntries[0];
    CHK_PRT_RET(mainEntry.directHandle == 0 || mainEntry.threadIndex != 0,
        HCCL_ERROR("Invalid direct main entry, threadIndex[%u]",
            mainEntry.threadIndex),
        HCCL_E_INTERNAL);
    CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
        mainEntry.directHandle,
        taskArgs,
        taskArgCount));

    for (uint32_t idx = 1; idx < resCtx.threads.size(); ++idx) {
        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, idx, 0));
    }
    return HCCL_SUCCESS;
}


HcclResult LaunchChainFast(const AlgResourceCtx &resCtx,
    uint64_t baseAddr,
    uint64_t memToken)
{
    CHK_PRT_RET(resCtx.threads.size() != 1 ||
            resCtx.ccuKernelEntries.size() != 1,
        HCCL_ERROR("Chain fast path requires one thread and one kernel"),
        HCCL_E_INTERNAL);
    const CcuKernelLaunchEntry &entry =
        resCtx.ccuKernelEntries[0];
    CHK_PRT_RET(entry.chainHandle == 0 ||
            entry.threadIndex != 0,
        HCCL_ERROR("Invalid chain fast entry, threadIndex[%u]",
            entry.threadIndex),
        HCCL_E_INTERNAL);

    const std::array<uint64_t, 2> taskArgs = {
        baseAddr,
        memToken,
    };

    CHK_RET_CCU(HcommCcuKernelLaunch(
        resCtx.threads[0],
        entry.chainHandle,
        taskArgs.data(),
        static_cast<uint32_t>(taskArgs.size())));

    return HCCL_SUCCESS;
}

// 8+4保持独立调度，避免其他拓扑的泛化分支改变其两Die并行任务图。
HcclResult Launch8p4SagOverlappedFast(
    const AlgResourceCtx &resCtx,
    uint32_t myRank,
    uint32_t root,
    const std::array<FixedTaskArgs, FAST_DIE_ENTRY_MAX> &scatterTaskArgs,
    const std::array<uint64_t, 4> &allGatherTaskArgs,
    uint32_t allGatherArgCount)
{
    CHK_PRT_RET(resCtx.threads.size() < 2 ||
            resCtx.threads.size() > FAST_DIE_ENTRY_MAX,
        HCCL_ERROR("8p4 fast path invalid thread count[%zu]",
            resCtx.threads.size()),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.ccuKernelEntries.size() != resCtx.threads.size(),
        HCCL_ERROR("8p4 fast path kernel/thread mismatch, kernels[%zu], threads[%zu]",
            resCtx.ccuKernelEntries.size(),
            resCtx.threads.size()),
        HCCL_E_INTERNAL);

    const uint32_t expectedAllGatherArgs = myRank == root ? 2U : 4U;
    CHK_PRT_RET(allGatherArgCount != expectedAllGatherArgs,
        HCCL_ERROR("8p4 fast AG args mismatch, expected[%u], got[%u]",
            expectedAllGatherArgs,
            allGatherArgCount),
        HCCL_E_INTERNAL);

    const ThreadHandle mainThread = resCtx.threads[0];
    for (uint32_t idx = 1; idx < resCtx.threads.size(); ++idx) {
        CHK_RET(HcommThreadNotifyRecordOnThread(
            mainThread,
            resCtx.threads[idx],
            WORKER_START_NOTIFY_ID));
        CHK_RET(HcommThreadNotifyWaitOnThread(
            resCtx.threads[idx],
            WORKER_START_NOTIFY_ID,
            0));
    }

    if (myRank == root) {
        for (uint32_t idx = 1; idx < resCtx.ccuKernelEntries.size(); ++idx) {
            const CcuKernelLaunchEntry &entry = resCtx.ccuKernelEntries[idx];
            const FixedTaskArgs &scatterArgs = scatterTaskArgs[idx];
            CHK_PRT_RET(entry.scatterHandle == 0 ||
                    entry.threadIndex != idx ||
                    scatterArgs.count == 0,
                HCCL_ERROR("Invalid 8p4 root worker entry[%u]", idx),
                HCCL_E_INTERNAL);
            CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[idx],
                entry.scatterHandle,
                scatterArgs.values.data(),
                scatterArgs.count));
            CHK_RET(HcommThreadNotifyRecordOnThread(
                resCtx.threads[idx], mainThread, idx));
        }

        const CcuKernelLaunchEntry &mainEntry = resCtx.ccuKernelEntries[0];
        const FixedTaskArgs &mainScatterArgs = scatterTaskArgs[0];
        CHK_PRT_RET(mainEntry.scatterHandle == 0 ||
                mainEntry.threadIndex != 0 ||
                mainScatterArgs.count == 0,
            HCCL_ERROR("Invalid 8p4 root main entry"),
            HCCL_E_INTERNAL);
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            mainEntry.scatterHandle,
            mainScatterArgs.values.data(),
            mainScatterArgs.count));

        for (uint32_t idx = 1; idx < resCtx.threads.size(); ++idx) {
            CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, idx, 0));
        }
        return HCCL_SUCCESS;
    }

    uint32_t rootEntryIndex = INVALID_VALUE_RANKID;
    for (uint32_t idx = 0; idx < resCtx.ccuKernelEntries.size(); ++idx) {
        if (resCtx.ccuKernelEntries[idx].containsRootChannel != 0) {
            CHK_PRT_RET(rootEntryIndex != INVALID_VALUE_RANKID,
                HCCL_ERROR("Multiple die groups contain root channel"),
                HCCL_E_INTERNAL);
            rootEntryIndex = idx;
        }
    }
    CHK_PRT_RET(rootEntryIndex == INVALID_VALUE_RANKID,
        HCCL_ERROR("No die group contains root channel for rank[%u] root[%u]",
            myRank,
            root),
        HCCL_E_INTERNAL);

    for (uint32_t idx = 1; idx < resCtx.ccuKernelEntries.size(); ++idx) {
        if (idx == rootEntryIndex) {
            continue;
        }
        const CcuKernelLaunchEntry &entry = resCtx.ccuKernelEntries[idx];
        CHK_PRT_RET(entry.allGatherHandle == 0 || entry.threadIndex != idx,
            HCCL_ERROR("Invalid 8p4 nonroot worker[%u]", idx),
            HCCL_E_INTERNAL);
        CHK_RET(HcommThreadNotifyWaitOnThread(
            resCtx.threads[idx], SAG_READY_WORKER_NOTIFY_ID, 0));
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[idx],
            entry.allGatherHandle,
            allGatherTaskArgs.data(),
            allGatherArgCount));
        CHK_RET(HcommThreadNotifyRecordOnThread(
            resCtx.threads[idx], mainThread, idx));
    }

    const CcuKernelLaunchEntry &rootEntry = resCtx.ccuKernelEntries[rootEntryIndex];
    const FixedTaskArgs &rootScatterArgs = scatterTaskArgs[rootEntryIndex];
    CHK_PRT_RET(rootEntry.scatterHandle == 0 ||
            rootEntry.allGatherHandle == 0 ||
            rootEntry.threadIndex != rootEntryIndex ||
            rootScatterArgs.count == 0,
        HCCL_ERROR("Invalid 8p4 root-link entry[%u]", rootEntryIndex),
        HCCL_E_INTERNAL);

    if (rootEntryIndex == 0) {
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            rootEntry.scatterHandle,
            rootScatterArgs.values.data(),
            rootScatterArgs.count));
        for (uint32_t idx = 1; idx < resCtx.threads.size(); ++idx) {
            CHK_RET(HcommThreadNotifyRecordOnThread(
                mainThread, resCtx.threads[idx], SAG_READY_WORKER_NOTIFY_ID));
        }
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            rootEntry.allGatherHandle,
            allGatherTaskArgs.data(),
            allGatherArgCount));
    } else {
        const ThreadHandle rootThread = resCtx.threads[rootEntryIndex];
        CHK_RET_CCU(HcommCcuKernelLaunch(rootThread,
            rootEntry.scatterHandle,
            rootScatterArgs.values.data(),
            rootScatterArgs.count));
        CHK_RET(HcommThreadNotifyRecordOnThread(
            rootThread, mainThread, SAG_READY_MAIN_NOTIFY_ID));
        for (uint32_t idx = 1; idx < resCtx.threads.size(); ++idx) {
            if (idx != rootEntryIndex) {
                CHK_RET(HcommThreadNotifyRecordOnThread(
                    rootThread, resCtx.threads[idx], SAG_READY_WORKER_NOTIFY_ID));
            }
        }
        CHK_RET_CCU(HcommCcuKernelLaunch(rootThread,
            rootEntry.allGatherHandle,
            allGatherTaskArgs.data(),
            allGatherArgCount));
        CHK_RET(HcommThreadNotifyRecordOnThread(
            rootThread, mainThread, rootEntryIndex));

        const CcuKernelLaunchEntry &mainEntry = resCtx.ccuKernelEntries[0];
        CHK_PRT_RET(mainEntry.allGatherHandle == 0 || mainEntry.threadIndex != 0,
            HCCL_ERROR("Invalid 8p4 nonroot main entry"),
            HCCL_E_INTERNAL);
        CHK_RET(HcommThreadNotifyWaitOnThread(
            mainThread, SAG_READY_MAIN_NOTIFY_ID, 0));
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            mainEntry.allGatherHandle,
            allGatherTaskArgs.data(),
            allGatherArgCount));
    }

    for (uint32_t idx = 1; idx < resCtx.threads.size(); ++idx) {
        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, idx, 0));
    }
    return HCCL_SUCCESS;
}

/*
 * 2x8固定数组快路径。设备TaskGraph保持不变，只消除每次调用的堆分配。
 */
HcclResult LaunchSagOverlappedFast(
    const AlgResourceCtx &resCtx,
    uint32_t myRank,
    uint32_t root,
    const std::array<FixedTaskArgs, FAST_DIE_ENTRY_MAX> &scatterTaskArgs,
    const std::array<uint64_t, 4> &allGatherTaskArgs,
    uint32_t allGatherArgCount)
{
    CHK_PRT_RET(resCtx.threads.size() < 2 ||
            resCtx.threads.size() > FAST_DIE_ENTRY_MAX,
        HCCL_ERROR("SAG fast path invalid thread count[%zu]",
            resCtx.threads.size()),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.ccuKernelEntries.size() != resCtx.threads.size(),
        HCCL_ERROR("SAG fast path kernel/thread mismatch, kernels[%zu], threads[%zu]",
            resCtx.ccuKernelEntries.size(),
            resCtx.threads.size()),
        HCCL_E_INTERNAL);

    CHK_PRT_RET(allGatherArgCount != 4U,
        HCCL_ERROR("SAG fast AG args mismatch, expected[4], got[%u]",
            allGatherArgCount),
        HCCL_E_INTERNAL);

    const ThreadHandle mainThread = resCtx.threads[0];

    /*
     * CheckerV3硬约束：每条slave stream的首任务必须是local WAIT。
     * 因此所有SAG worker在任何CCU_GRAPH或阶段Wait之前，统一先执行
     * main->worker启动握手。该握手只发生在Host任务图入口，不改变
     * 设备侧Scatter/AllGather TaskGraph与Channel并发顺序。
     */
    for (uint32_t idx = 1; idx < resCtx.threads.size(); ++idx) {
        CHK_RET(HcommThreadNotifyRecordOnThread(
            mainThread,
            resCtx.threads[idx],
            WORKER_START_NOTIFY_ID));
        CHK_RET(HcommThreadNotifyWaitOnThread(
            resCtx.threads[idx],
            WORKER_START_NOTIFY_ID,
            0));
    }

    if (myRank == root) {
        for (uint32_t idx = 1; idx < resCtx.ccuKernelEntries.size(); ++idx) {
            const CcuKernelLaunchEntry &entry = resCtx.ccuKernelEntries[idx];
            const FixedTaskArgs &scatterArgs = scatterTaskArgs[idx];
            CHK_PRT_RET(entry.scatterHandle == 0 ||
                    entry.allGatherHandle == 0 ||
                    entry.threadIndex != idx ||
                    scatterArgs.count == 0,
                HCCL_ERROR("Invalid SAG root worker entry[%u]", idx),
                HCCL_E_INTERNAL);
            CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[idx],
                entry.scatterHandle,
                scatterArgs.values.data(),
                scatterArgs.count));
            CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[idx],
                entry.allGatherHandle,
                allGatherTaskArgs.data(),
                allGatherArgCount));
            CHK_RET(HcommThreadNotifyRecordOnThread(
                resCtx.threads[idx], mainThread, idx));
        }

        const CcuKernelLaunchEntry &mainEntry = resCtx.ccuKernelEntries[0];
        const FixedTaskArgs &mainScatterArgs = scatterTaskArgs[0];
        CHK_PRT_RET(mainEntry.scatterHandle == 0 ||
                mainEntry.allGatherHandle == 0 ||
                mainEntry.threadIndex != 0 ||
                mainScatterArgs.count == 0,
            HCCL_ERROR("Invalid SAG root main entry"),
            HCCL_E_INTERNAL);
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            mainEntry.scatterHandle,
            mainScatterArgs.values.data(),
            mainScatterArgs.count));
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            mainEntry.allGatherHandle,
            allGatherTaskArgs.data(),
            allGatherArgCount));

        for (uint32_t idx = 1; idx < resCtx.threads.size(); ++idx) {
            CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, idx, 0));
        }
        return HCCL_SUCCESS;
    }

    uint32_t rootEntryIndex = INVALID_VALUE_RANKID;
    for (uint32_t idx = 0; idx < resCtx.ccuKernelEntries.size(); ++idx) {
        if (resCtx.ccuKernelEntries[idx].containsRootChannel != 0) {
            CHK_PRT_RET(rootEntryIndex != INVALID_VALUE_RANKID,
                HCCL_ERROR("Multiple die groups contain root channel"),
                HCCL_E_INTERNAL);
            rootEntryIndex = idx;
        }
    }
    CHK_PRT_RET(rootEntryIndex == INVALID_VALUE_RANKID,
        HCCL_ERROR("No die group contains root channel for rank[%u] root[%u]",
            myRank,
            root),
        HCCL_E_INTERNAL);

    // 非root的其他Die先排队等待own-slice就绪，再执行AllGather。
    for (uint32_t idx = 1; idx < resCtx.ccuKernelEntries.size(); ++idx) {
        if (idx == rootEntryIndex) {
            continue;
        }
        const CcuKernelLaunchEntry &entry = resCtx.ccuKernelEntries[idx];
        CHK_PRT_RET(entry.allGatherHandle == 0 || entry.threadIndex != idx,
            HCCL_ERROR("Invalid SAG nonroot worker[%u]", idx),
            HCCL_E_INTERNAL);
        CHK_RET(HcommThreadNotifyWaitOnThread(
            resCtx.threads[idx], SAG_READY_WORKER_NOTIFY_ID, 0));
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[idx],
            entry.allGatherHandle,
            allGatherTaskArgs.data(),
            allGatherArgCount));
        CHK_RET(HcommThreadNotifyRecordOnThread(
            resCtx.threads[idx], mainThread, idx));
    }

    const CcuKernelLaunchEntry &rootEntry = resCtx.ccuKernelEntries[rootEntryIndex];
    const FixedTaskArgs &rootScatterArgs = scatterTaskArgs[rootEntryIndex];
    CHK_PRT_RET(rootEntry.scatterHandle == 0 ||
        rootEntry.allGatherHandle == 0 ||
            rootEntry.threadIndex != rootEntryIndex ||
            rootScatterArgs.count == 0,
        HCCL_ERROR("Invalid SAG root-link entry[%u]", rootEntryIndex),
        HCCL_E_INTERNAL);

    if (rootEntryIndex == 0) {
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            rootEntry.scatterHandle,
            rootScatterArgs.values.data(),
            rootScatterArgs.count));
        for (uint32_t idx = 1; idx < resCtx.threads.size(); ++idx) {
            CHK_RET(HcommThreadNotifyRecordOnThread(
                mainThread, resCtx.threads[idx], SAG_READY_WORKER_NOTIFY_ID));
        }
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            rootEntry.allGatherHandle,
            allGatherTaskArgs.data(),
            allGatherArgCount));
    } else {
        const ThreadHandle rootThread = resCtx.threads[rootEntryIndex];
        CHK_RET_CCU(HcommCcuKernelLaunch(rootThread,
            rootEntry.scatterHandle,
            rootScatterArgs.values.data(),
            rootScatterArgs.count));
        CHK_RET(HcommThreadNotifyRecordOnThread(
            rootThread, mainThread, SAG_READY_MAIN_NOTIFY_ID));
        for (uint32_t idx = 1; idx < resCtx.threads.size(); ++idx) {
            if (idx != rootEntryIndex) {
                CHK_RET(HcommThreadNotifyRecordOnThread(
                    rootThread, resCtx.threads[idx], SAG_READY_WORKER_NOTIFY_ID));
            }
        }
        CHK_RET_CCU(HcommCcuKernelLaunch(rootThread,
            rootEntry.allGatherHandle,
            allGatherTaskArgs.data(),
            allGatherArgCount));
        CHK_RET(HcommThreadNotifyRecordOnThread(
            rootThread, mainThread, rootEntryIndex));

        const CcuKernelLaunchEntry &mainEntry = resCtx.ccuKernelEntries[0];
        CHK_PRT_RET(mainEntry.allGatherHandle == 0 || mainEntry.threadIndex != 0,
            HCCL_ERROR("Invalid SAG nonroot main entry"),
            HCCL_E_INTERNAL);
        CHK_RET(HcommThreadNotifyWaitOnThread(
            mainThread, SAG_READY_MAIN_NOTIFY_ID, 0));
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            mainEntry.allGatherHandle,
            allGatherTaskArgs.data(),
            allGatherArgCount));
    }

    for (uint32_t idx = 1; idx < resCtx.threads.size(); ++idx) {
        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, idx, 0));
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchKernelPhasePerEntry(const AlgResourceCtx &resCtx,
    uint32_t phase,
    const std::vector<std::vector<uint64_t>> &taskArgsPerEntry)
{
    CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("No CCU thread resource"), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.ccuKernelEntries.size() != resCtx.threads.size() ||
            taskArgsPerEntry.size() != resCtx.ccuKernelEntries.size(),
        HCCL_ERROR("Per-entry launch size mismatch, kernels[%zu], threads[%zu], args[%zu]",
            resCtx.ccuKernelEntries.size(),
            resCtx.threads.size(),
            taskArgsPerEntry.size()),
        HCCL_E_INTERNAL);

    const ThreadHandle mainThread = resCtx.threads[0];
    for (uint32_t idx = 1; idx < resCtx.threads.size(); ++idx) {
        CHK_RET(HcommThreadNotifyRecordOnThread(
            mainThread, resCtx.threads[idx], WORKER_START_NOTIFY_ID));
        CHK_RET(HcommThreadNotifyWaitOnThread(
            resCtx.threads[idx], WORKER_START_NOTIFY_ID, 0));
    }

    for (uint32_t idx = 1; idx < resCtx.ccuKernelEntries.size(); ++idx) {
        const CcuKernelLaunchEntry &entry = resCtx.ccuKernelEntries[idx];
        const CcuKernelHandle handle = GetPhaseHandle(entry, phase);
        const std::vector<uint64_t> &args = taskArgsPerEntry[idx];
        CHK_PRT_RET(handle == 0 || entry.threadIndex != idx || args.empty(),
            HCCL_ERROR("Invalid per-entry worker[%u], phase[%u], threadIndex[%u], args[%zu]",
                idx, phase, entry.threadIndex, args.size()),
            HCCL_E_INTERNAL);
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[idx],
            handle,
            args.data(),
            static_cast<uint32_t>(args.size())));
        CHK_RET(HcommThreadNotifyRecordOnThread(resCtx.threads[idx], mainThread, idx));
    }

    const CcuKernelLaunchEntry &mainEntry = resCtx.ccuKernelEntries[0];
    const CcuKernelHandle mainHandle = GetPhaseHandle(mainEntry, phase);
    const std::vector<uint64_t> &mainArgs = taskArgsPerEntry[0];
    CHK_PRT_RET(mainHandle == 0 || mainEntry.threadIndex != 0 || mainArgs.empty(),
        HCCL_ERROR("Invalid per-entry main, phase[%u], threadIndex[%u], args[%zu]",
            phase, mainEntry.threadIndex, mainArgs.size()),
        HCCL_E_INTERNAL);
    CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
        mainHandle,
        mainArgs.data(),
        static_cast<uint32_t>(mainArgs.size())));

    for (uint32_t idx = 1; idx < resCtx.threads.size(); ++idx) {
        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, idx, 0));
    }
    return HCCL_SUCCESS;
}

// 512MiB多Die攻击路径：取消全局Scatter->AllGather Host阶段栅栏。
// root在每个Die Thread上按“Scatter后立即AllGather”顺序执行，使不同Die之间发生重叠；
// 非root只在包含root Channel的Die上执行Scatter，收到本rank切片后再放行其他Die的AllGather。
HcclResult LaunchSagOverlapped(const AlgResourceCtx &resCtx,
    uint32_t myRank,
    uint32_t root,
    const std::vector<std::vector<uint64_t>> &scatterTaskArgs,
    const std::vector<uint64_t> &allGatherTaskArgs)
{
    CHK_PRT_RET(resCtx.threads.size() < 2,
        HCCL_ERROR("SAG overlap requires at least two die threads, got[%zu]", resCtx.threads.size()),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.ccuKernelEntries.size() != resCtx.threads.size() ||
            scatterTaskArgs.size() != resCtx.ccuKernelEntries.size(),
        HCCL_ERROR("Overlap launch size mismatch, kernels[%zu], threads[%zu], scatterArgs[%zu]",
            resCtx.ccuKernelEntries.size(),
            resCtx.threads.size(),
            scatterTaskArgs.size()),
        HCCL_E_INTERNAL);
    const size_t expectedAllGatherArgs =
        (resCtx.algorithm == BCAST_ALG_8P4_PREFETCH_SAG && myRank == root) ? 2U : 4U;
    CHK_PRT_RET(allGatherTaskArgs.size() != expectedAllGatherArgs,
        HCCL_ERROR("AllGather compact args mismatch, expected[%zu], got[%zu]",
            expectedAllGatherArgs,
            allGatherTaskArgs.size()),
        HCCL_E_INTERNAL);

    const ThreadHandle mainThread = resCtx.threads[0];

    for (uint32_t idx = 1; idx < resCtx.threads.size(); ++idx) {
        CHK_RET(HcommThreadNotifyRecordOnThread(
            mainThread, resCtx.threads[idx], WORKER_START_NOTIFY_ID));
        CHK_RET(HcommThreadNotifyWaitOnThread(
            resCtx.threads[idx], WORKER_START_NOTIFY_ID, 0));
    }

    if (myRank == root) {
        for (uint32_t idx = 1; idx < resCtx.ccuKernelEntries.size(); ++idx) {
            const CcuKernelLaunchEntry &entry = resCtx.ccuKernelEntries[idx];
            const std::vector<uint64_t> &scatterArgs = scatterTaskArgs[idx];
            CHK_PRT_RET(entry.scatterHandle == 0 || entry.allGatherHandle == 0 ||
                    entry.threadIndex != idx || scatterArgs.empty(),
                HCCL_ERROR("Invalid overlap worker entry[%u]", idx),
                HCCL_E_INTERNAL);
            CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[idx],
                entry.scatterHandle,
                scatterArgs.data(),
                static_cast<uint32_t>(scatterArgs.size())));
            CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[idx],
                entry.allGatherHandle,
                allGatherTaskArgs.data(),
                static_cast<uint32_t>(allGatherTaskArgs.size())));
            CHK_RET(HcommThreadNotifyRecordOnThread(resCtx.threads[idx], mainThread, idx));
        }

        const CcuKernelLaunchEntry &mainEntry = resCtx.ccuKernelEntries[0];
        const std::vector<uint64_t> &mainScatterArgs = scatterTaskArgs[0];
        CHK_PRT_RET(mainEntry.scatterHandle == 0 || mainEntry.allGatherHandle == 0 ||
                mainEntry.threadIndex != 0 || mainScatterArgs.empty(),
            HCCL_ERROR("Invalid overlap main entry"),
            HCCL_E_INTERNAL);
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            mainEntry.scatterHandle,
            mainScatterArgs.data(),
            static_cast<uint32_t>(mainScatterArgs.size())));
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            mainEntry.allGatherHandle,
            allGatherTaskArgs.data(),
            static_cast<uint32_t>(allGatherTaskArgs.size())));

        for (uint32_t idx = 1; idx < resCtx.threads.size(); ++idx) {
            CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, idx, 0));
        }
        return HCCL_SUCCESS;
    }

    uint32_t rootEntryIndex = INVALID_VALUE_RANKID;
    for (uint32_t idx = 0; idx < resCtx.ccuKernelEntries.size(); ++idx) {
        if (resCtx.ccuKernelEntries[idx].containsRootChannel != 0) {
            CHK_PRT_RET(rootEntryIndex != INVALID_VALUE_RANKID,
                HCCL_ERROR("Multiple die groups contain root channel"),
                HCCL_E_INTERNAL);
            rootEntryIndex = idx;
        }
    }
    CHK_PRT_RET(rootEntryIndex == INVALID_VALUE_RANKID,
        HCCL_ERROR("No die group contains root channel for rank[%u] root[%u]", myRank, root),
        HCCL_E_INTERNAL);

    for (uint32_t idx = 1; idx < resCtx.ccuKernelEntries.size(); ++idx) {
        if (idx == rootEntryIndex) {
            continue;
        }
        const CcuKernelLaunchEntry &entry = resCtx.ccuKernelEntries[idx];
        CHK_PRT_RET(entry.allGatherHandle == 0 || entry.threadIndex != idx,
            HCCL_ERROR("Invalid nonroot overlap worker entry[%u]", idx),
            HCCL_E_INTERNAL);
        CHK_RET(HcommThreadNotifyWaitOnThread(
            resCtx.threads[idx], SAG_READY_WORKER_NOTIFY_ID, 0));
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[idx],
            entry.allGatherHandle,
            allGatherTaskArgs.data(),
            static_cast<uint32_t>(allGatherTaskArgs.size())));
        CHK_RET(HcommThreadNotifyRecordOnThread(resCtx.threads[idx], mainThread, idx));
    }

    const CcuKernelLaunchEntry &rootEntry = resCtx.ccuKernelEntries[rootEntryIndex];
    const std::vector<uint64_t> &rootScatterArgs = scatterTaskArgs[rootEntryIndex];
    CHK_PRT_RET(rootEntry.scatterHandle == 0 || rootEntry.allGatherHandle == 0 ||
            rootEntry.threadIndex != rootEntryIndex || rootScatterArgs.empty(),
        HCCL_ERROR("Invalid root-link die entry[%u]", rootEntryIndex),
        HCCL_E_INTERNAL);

    if (rootEntryIndex == 0) {
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            rootEntry.scatterHandle,
            rootScatterArgs.data(),
            static_cast<uint32_t>(rootScatterArgs.size())));
        for (uint32_t idx = 1; idx < resCtx.threads.size(); ++idx) {
            CHK_RET(HcommThreadNotifyRecordOnThread(
                mainThread, resCtx.threads[idx], SAG_READY_WORKER_NOTIFY_ID));
        }
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            rootEntry.allGatherHandle,
            allGatherTaskArgs.data(),
            static_cast<uint32_t>(allGatherTaskArgs.size())));
    } else {
        const ThreadHandle rootThread = resCtx.threads[rootEntryIndex];
        CHK_RET_CCU(HcommCcuKernelLaunch(rootThread,
            rootEntry.scatterHandle,
            rootScatterArgs.data(),
            static_cast<uint32_t>(rootScatterArgs.size())));

        CHK_RET(HcommThreadNotifyRecordOnThread(
            rootThread, mainThread, SAG_READY_MAIN_NOTIFY_ID));
        for (uint32_t idx = 1; idx < resCtx.threads.size(); ++idx) {
            if (idx != rootEntryIndex) {
                CHK_RET(HcommThreadNotifyRecordOnThread(
                    rootThread, resCtx.threads[idx], SAG_READY_WORKER_NOTIFY_ID));
            }
        }
        CHK_RET_CCU(HcommCcuKernelLaunch(rootThread,
            rootEntry.allGatherHandle,
            allGatherTaskArgs.data(),
            static_cast<uint32_t>(allGatherTaskArgs.size())));
        CHK_RET(HcommThreadNotifyRecordOnThread(rootThread, mainThread, rootEntryIndex));

        const CcuKernelLaunchEntry &mainEntry = resCtx.ccuKernelEntries[0];
        CHK_PRT_RET(mainEntry.allGatherHandle == 0 || mainEntry.threadIndex != 0,
            HCCL_ERROR("Invalid nonroot overlap main entry"),
            HCCL_E_INTERNAL);
        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, SAG_READY_MAIN_NOTIFY_ID, 0));
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            mainEntry.allGatherHandle,
            allGatherTaskArgs.data(),
            static_cast<uint32_t>(allGatherTaskArgs.size())));
    }

    for (uint32_t idx = 1; idx < resCtx.threads.size(); ++idx) {
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
    CHK_PRT_RET(!resCtx.DeSerialize(param.resCtx, param.ctxSize),
        HCCL_ERROR("Invalid AlgResourceCtx, size[%llu]",
            static_cast<unsigned long long>(param.ctxSize)),
        HCCL_E_INTERNAL);

    CHK_PRT_RET(param.root >= param.rankSize,
        HCCL_ERROR("Invalid root[%u], rankSize[%u]", param.root, param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("Unsupported rankSize[%u]", param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("No CCU thread resource"), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.ccuKernelEntries.empty(),
        HCCL_ERROR("No CCU kernel resource"),
        HCCL_E_INTERNAL);

    const auto typeSizeIter = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(typeSizeIter == SIZE_TABLE.end(),
        HCCL_ERROR("Unsupported dataType[%d]", static_cast<int32_t>(param.dataType)),
        HCCL_E_PARA);

    const uint64_t typeSize = typeSizeIter->second;
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / typeSize,
        HCCL_ERROR("Data size overflow, count[%llu]", static_cast<unsigned long long>(param.count)),
        HCCL_E_PARA);

    const uint64_t totalBytes = param.count * typeSize;
    if (param.rankSize <= 1 || totalBytes == 0) {
        return HCCL_SUCCESS;
    }

    uint64_t memToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(
        reinterpret_cast<uint64_t>(param.outputPtr), totalBytes, &memToken));
    const uint64_t baseAddr = reinterpret_cast<uint64_t>(param.outputPtr);

    if (resCtx.algorithm == BCAST_ALG_DIRECT) {
        const std::array<uint64_t, 2> taskArgs = {
            baseAddr,
            memToken,
        };
        CHK_RET(LaunchDirectFast(
            resCtx,
            taskArgs.data(),
            static_cast<uint32_t>(taskArgs.size())));
        return HCCL_SUCCESS;
    }

    if (resCtx.algorithm == BCAST_ALG_CHAIN_PIPELINE) {
        CHK_PRT_RET(resCtx.ccuKernelEntries.size() != 1,
            HCCL_ERROR("Chain pipeline requires one die kernel, got[%zu]",
                resCtx.ccuKernelEntries.size()),
            HCCL_E_INTERNAL);

        CHK_RET(LaunchChainFast(
            resCtx,
            baseAddr,
            memToken));
        return HCCL_SUCCESS;
    }

    // 8+4使用原版独立固定数组入口，避免与2x8共享条件和TaskArg布局。
    if (resCtx.algorithm == BCAST_ALG_8P4_PREFETCH_SAG &&
        totalBytes > DIRECT_ALG_MAX_BYTES &&
        resCtx.threads.size() > 1 &&
        resCtx.threads.size() <= FAST_DIE_ENTRY_MAX) {
        std::array<uint64_t, MAX_RANK_SIZE> fastSliceOffsets{};
        std::array<uint64_t, MAX_RANK_SIZE> fastSliceSizes{};

        const uint64_t baseElements = param.count / param.rankSize;
        const uint64_t residualElements = param.count % param.rankSize;
        uint64_t currentOffset = 0;
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            const uint64_t elements =
                baseElements + (rank < residualElements ? 1ULL : 0ULL);
            const uint64_t bytes = elements * typeSize;
            CHK_PRT_RET(bytes == 0 || bytes > MAX_DATA_SIZE,
                HCCL_ERROR("Invalid fast 8p4 slice size[%llu] for rank[%u]",
                    static_cast<unsigned long long>(bytes),
                    rank),
                HCCL_E_NOT_SUPPORT);

            fastSliceOffsets[rank] = currentOffset;
            fastSliceSizes[rank] = bytes;
            currentOffset += bytes;
        }
        CHK_PRT_RET(currentOffset != totalBytes,
            HCCL_ERROR("Fast 8p4 partition mismatch, partitioned[%llu], total[%llu]",
                static_cast<unsigned long long>(currentOffset),
                static_cast<unsigned long long>(totalBytes)),
            HCCL_E_INTERNAL);

        std::array<FixedTaskArgs, FAST_DIE_ENTRY_MAX> scatterArgs{};
        for (uint32_t entryIndex = 0;
             entryIndex < resCtx.ccuKernelEntries.size();
             ++entryIndex) {
            const CcuKernelLaunchEntry &entry =
                resCtx.ccuKernelEntries[entryIndex];
            FixedTaskArgs &args = scatterArgs[entryIndex];
            args.values[0] = baseAddr;
            args.values[1] = memToken;

            if (param.myRank != param.root) {
                args.count = 2;
                continue;
            }

            CHK_PRT_RET(entry.channelCount > MAX_RANK_SIZE,
                HCCL_ERROR("Invalid fast 8p4 channelCount[%u]",
                    entry.channelCount),
                HCCL_E_INTERNAL);
            args.values[2] = fastSliceOffsets[param.root];
            args.values[3] = fastSliceSizes[param.root];
            for (uint32_t idx = 0; idx < entry.channelCount; ++idx) {
                const uint32_t remoteRank = entry.remoteRanks[idx];
                args.values[4U + idx] = fastSliceOffsets[remoteRank];
                args.values[4U + entry.channelCount + idx] =
                    fastSliceSizes[remoteRank];
            }
            args.count = 4U + 2U * entry.channelCount;
        }

        std::array<uint64_t, 4> allGatherArgs{};
        allGatherArgs[0] = baseAddr;
        allGatherArgs[1] = memToken;
        uint32_t allGatherArgCount = 2;
        if (param.myRank != param.root) {
            allGatherArgs[2] = fastSliceOffsets[param.myRank];
            allGatherArgs[3] = fastSliceSizes[param.myRank];
            allGatherArgCount = 4;
        }

        CHK_RET(Launch8p4SagOverlappedFast(resCtx,
            param.myRank,
            param.root,
            scatterArgs,
            allGatherArgs,
            allGatherArgCount));
        return HCCL_SUCCESS;
    }

    /*
     * 2x8 large-message hot path. Keep its device task graph unchanged while
     * eliminating per-call nested vector allocation.
     */
    const bool useFixedSag =
        resCtx.algorithm == BCAST_ALG_SAG_SPLIT &&
        (param.rankSize == 12 || param.rankSize == 16);
    if (useFixedSag &&
        totalBytes > DIRECT_ALG_MAX_BYTES &&
        resCtx.threads.size() > 1 &&
        resCtx.threads.size() <= FAST_DIE_ENTRY_MAX) {
        std::array<uint64_t, MAX_RANK_SIZE> fastSliceOffsets{};
        std::array<uint64_t, MAX_RANK_SIZE> fastSliceSizes{};

        const uint64_t baseElements =
            param.count / param.rankSize;
        uint64_t currentOffset = 0;
        for (uint32_t rank = 0;
             rank < param.rankSize;
             ++rank) {
            const uint64_t elements =
                rank + 1U == param.rankSize
                    ? param.count - baseElements * (param.rankSize - 1U)
                    : baseElements;
            const uint64_t bytes =
                elements * typeSize;

            CHK_PRT_RET(bytes == 0 ||
                    bytes > MAX_DATA_SIZE,
                HCCL_ERROR(
                    "Invalid fast 8p4 slice size[%llu] for rank[%u]",
                    static_cast<unsigned long long>(bytes),
                    rank),
                HCCL_E_NOT_SUPPORT);

            fastSliceOffsets[rank] =
                currentOffset;
            fastSliceSizes[rank] =
                bytes;
            currentOffset += bytes;
        }

        CHK_PRT_RET(currentOffset != totalBytes,
            HCCL_ERROR(
                "Fast 8p4 partition mismatch, partitioned[%llu], total[%llu]",
                static_cast<unsigned long long>(currentOffset),
                static_cast<unsigned long long>(totalBytes)),
            HCCL_E_INTERNAL);

        std::array<FixedTaskArgs,
            FAST_DIE_ENTRY_MAX> scatterArgs{};

        for (uint32_t entryIndex = 0;
             entryIndex < resCtx.ccuKernelEntries.size();
             ++entryIndex) {
            const CcuKernelLaunchEntry &entry =
                resCtx.ccuKernelEntries[entryIndex];
            FixedTaskArgs &args =
                scatterArgs[entryIndex];

            args.values[0] = baseAddr;
            args.values[1] = memToken;

            if (param.myRank != param.root) {
                args.count = 2;
                continue;
            }

            CHK_PRT_RET(entry.channelCount > MAX_RANK_SIZE,
                HCCL_ERROR(
                    "Invalid fast SAG channelCount[%u]",
                    entry.channelCount),
                HCCL_E_INTERNAL);

            constexpr uint32_t peerOffsetBase = 2U;

            for (uint32_t idx = 0;
                 idx < entry.channelCount;
                 ++idx) {
                const uint32_t remoteRank =
                    entry.remoteRanks[idx];
                args.values[peerOffsetBase + idx] =
                    fastSliceOffsets[remoteRank];
                args.values[
                    peerOffsetBase + entry.channelCount + idx] =
                    fastSliceSizes[remoteRank];
            }

            args.count =
                peerOffsetBase + 2U * entry.channelCount;
        }

        std::array<uint64_t, 4> allGatherArgs{};
        allGatherArgs[0] = baseAddr;
        allGatherArgs[1] = memToken;
        allGatherArgs[2] = fastSliceOffsets[param.myRank];
        allGatherArgs[3] = fastSliceSizes[param.myRank];
        constexpr uint32_t allGatherArgCount = 4U;

        CHK_RET(LaunchSagOverlappedFast(
            resCtx,
            param.myRank,
            param.root,
            scatterArgs,
            allGatherArgs,
            allGatherArgCount));

        return HCCL_SUCCESS;
    }

    // 按数据类型元素切片，保证每个slice的地址和长度均保持类型对齐。
    //
    // 8+4预取Kernel仍会加载并预发root slice，所以必须保留root对应的
    // offset/size，不能改成仅在rankSize-1个非root之间切分。
    std::vector<uint64_t> sliceOffsets(param.rankSize, 0);
    std::vector<uint64_t> sliceSizes(param.rankSize, 0);
    const uint64_t baseElements =
        param.count / param.rankSize;
    uint64_t currentOffset = 0;

    for (uint32_t rank = 0;
         rank < param.rankSize;
         ++rank) {
        const uint64_t elements =
            rank + 1U == param.rankSize
                ? param.count - baseElements * (param.rankSize - 1U)
                : baseElements;

        const uint64_t bytes =
            elements * typeSize;

        CHK_PRT_RET(bytes == 0 ||
                bytes > MAX_DATA_SIZE,
            HCCL_ERROR(
                "Invalid slice size[%llu] for rank[%u]",
                static_cast<unsigned long long>(bytes),
                rank),
            HCCL_E_NOT_SUPPORT);

        sliceOffsets[rank] =
            currentOffset;
        sliceSizes[rank] =
            bytes;
        currentOffset += bytes;
    }
    CHK_PRT_RET(currentOffset != totalBytes,
        HCCL_ERROR("Slice partition mismatch, partitioned[%llu], total[%llu]",
            static_cast<unsigned long long>(currentOffset),
            static_cast<unsigned long long>(totalBytes)),
        HCCL_E_INTERNAL);

    // 融合SAG Kernel仍使用完整参数表，因为其静态指令会加载所有rank的
    // offset/size。分离SAG则为每个Die Kernel构造紧凑参数。
    if (resCtx.algorithm == BCAST_ALG_SAG_FUSED) {
        std::vector<uint64_t> fusedTaskArgs;
        fusedTaskArgs.reserve(2 + 2 * param.rankSize);
        fusedTaskArgs.push_back(baseAddr);
        fusedTaskArgs.push_back(memToken);
        fusedTaskArgs.insert(fusedTaskArgs.end(), sliceOffsets.begin(), sliceOffsets.end());
        fusedTaskArgs.insert(fusedTaskArgs.end(), sliceSizes.begin(), sliceSizes.end());

        CHK_PRT_RET(resCtx.ccuKernelEntries.size() != 1,
            HCCL_ERROR("Fused SAG requires one die kernel, got[%zu]",
                resCtx.ccuKernelEntries.size()),
            HCCL_E_INTERNAL);
        CHK_RET(LaunchKernelPhase(resCtx, PHASE_FUSED, fusedTaskArgs));
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(resCtx.algorithm != BCAST_ALG_SAG_SPLIT &&
            resCtx.algorithm != BCAST_ALG_8P4_PREFETCH_SAG,
        HCCL_ERROR("Unknown broadcast algorithm[%u]", resCtx.algorithm),
        HCCL_E_INTERNAL);

    std::vector<std::vector<uint64_t>> scatterTaskArgs;
    scatterTaskArgs.reserve(resCtx.ccuKernelEntries.size());
    for (const CcuKernelLaunchEntry &entry : resCtx.ccuKernelEntries) {
        if (resCtx.algorithm == BCAST_ALG_8P4_PREFETCH_SAG) {
            scatterTaskArgs.push_back(Build8p4ScatterTaskArgs(entry,
                param.myRank,
                param.root,
                baseAddr,
                memToken,
                sliceOffsets,
                sliceSizes));
        } else {
            scatterTaskArgs.push_back(BuildScatterTaskArgs(entry,
                param.myRank,
                param.root,
                baseAddr,
                memToken,
                sliceOffsets,
                sliceSizes));
        }
    }
    const std::vector<uint64_t> allGatherTaskArgs =
        resCtx.algorithm == BCAST_ALG_8P4_PREFETCH_SAG
            ? Build8p4AllGatherTaskArgs(baseAddr,
                  memToken,
                  param.myRank,
                  param.root,
                  sliceOffsets,
                  sliceSizes)
            : BuildAllGatherTaskArgs(
                  baseAddr, memToken, param.myRank, sliceOffsets, sliceSizes);

    if ((param.rankSize == 16 || resCtx.algorithm == BCAST_ALG_8P4_PREFETCH_SAG) &&
        totalBytes > DIRECT_ALG_MAX_BYTES && resCtx.threads.size() > 1) {
        CHK_RET(LaunchSagOverlapped(resCtx,
            param.myRank,
            param.root,
            scatterTaskArgs,
            allGatherTaskArgs));
        return HCCL_SUCCESS;
    }

    CHK_RET(LaunchKernelPhasePerEntry(resCtx, PHASE_SCATTER, scatterTaskArgs));
    std::vector<std::vector<uint64_t>> allGatherTaskArgsPerEntry(
        resCtx.ccuKernelEntries.size(), allGatherTaskArgs);
    CHK_RET(LaunchKernelPhasePerEntry(
        resCtx, PHASE_ALLGATHER, allGatherTaskArgsPerEntry));
    return HCCL_SUCCESS;
}

} // namespace ops_hccl

