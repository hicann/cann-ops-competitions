#include <ccu/ccu_res.h>
#include <ccu/ccu_launch.h>
#include <hccl/hcomm_primitives.h>

#include <algorithm>
#include <array>
#include <limits>

#include "custom.h"
#include "exec_op.h"
#include "log.h"
#include "../op_kernel_ccu/ccu_kernel.h"

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param)
{
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);

    auto typeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(typeIt == SIZE_TABLE.end(),
        HCCL_ERROR("[ExecOp] unsupported data type[%d]", param.dataType), HCCL_E_NOT_SUPPORT);
    const uint64_t dataTypeSize = typeIt->second;
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / dataTypeSize,
        HCCL_ERROR("[ExecOp] byte count overflow"), HCCL_E_PARA);
    const uint64_t dataSize = param.count * dataTypeSize;
    if (dataSize == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("[ExecOp] CCU thread is empty"), HCCL_E_INTERNAL);
    if (param.rankSize == 1) {
        return static_cast<HcclResult>(
            HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, dataSize));
    }
    constexpr size_t stageCount = 2;
    const size_t dieCount = resCtx.threads.size();
    const bool useP4FusedKernel = param.rankSize == 4 && dieCount == 1 && resCtx.ccuKernels.size() == 1;
    const bool useStagedKernels = resCtx.ccuKernels.size() == stageCount * dieCount;
    CHK_PRT_RET(dieCount == 0 || dieCount > 2 || (!useP4FusedKernel && !useStagedKernels),
        HCCL_ERROR("[ExecOp] invalid kernel/thread resources, threadNum[%zu], kernelNum[%zu]",
            dieCount, resCtx.ccuKernels.size()), HCCL_E_INTERNAL);

    // Generic staged paths require two disjoint scratch halves. P16 owns an
    // explicit eighteen-slot layout over the full local buffer, so derive its
    // tile cap from the largest aligned slot that the layout can hold.
    constexpr uint64_t p16ScratchSlotCount = 18;
    constexpr uint64_t p16ScratchAlignment = 4096;
    uint64_t tileCapacity = std::min<uint64_t>(MAX_DATA_SIZE, resCtx.localBuffer.size / 2);
    if (param.rankSize == 16) {
        uint64_t p16SlotCapacity = resCtx.localBuffer.size / p16ScratchSlotCount;
        p16SlotCapacity -= p16SlotCapacity % p16ScratchAlignment;
        CHK_PRT_RET(p16SlotCapacity == 0 ||
            p16SlotCapacity > std::numeric_limits<uint64_t>::max() / param.rankSize,
            HCCL_ERROR("[ExecOp] invalid P16 slot capacity[%llu]",
                static_cast<unsigned long long>(p16SlotCapacity)), HCCL_E_INTERNAL);
        tileCapacity = p16SlotCapacity * param.rankSize;
    }
    tileCapacity -= tileCapacity % dataTypeSize;
    CHK_PRT_RET(tileCapacity == 0,
        HCCL_ERROR("[ExecOp] HCCL buffer is too small[%llu]",
            static_cast<unsigned long long>(resCtx.localBuffer.size)), HCCL_E_INTERNAL);

    const uint64_t inputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    const uint64_t scratchBase = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
    const uint64_t scratchSecondHalf = scratchBase + tileCapacity;
    uint64_t token = 0;
    CcuResult ccuRet = HcommCcuGetMemToken(inputAddr, dataSize, &token);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("[ExecOp] get input token failed[%d]", ccuRet), ConvertCcuToHccl(ccuRet));

    const uint64_t launchCount = (dataSize + tileCapacity - 1) / tileCapacity;
    for (uint64_t launch = 0; launch < launchCount; ++launch) {
        const uint64_t tileOffset = launch * tileCapacity;
        const uint64_t tileSize = std::min(tileCapacity, dataSize - tileOffset);
        const uint64_t tileElements = tileSize / dataTypeSize;
        const uint64_t segmentSize = (tileElements / param.rankSize) * dataTypeSize;
        const uint64_t lastSegmentSize =
            segmentSize + (tileElements % param.rankSize) * dataTypeSize;
        const uint64_t scratchStride = param.rankSize == 16 ?
            ((lastSegmentSize + p16ScratchAlignment - 1) / p16ScratchAlignment) *
                p16ScratchAlignment :
            lastSegmentSize;
        const uint64_t ownerSize = param.myRank + 1 == param.rankSize ? lastSegmentSize : segmentSize;
        if (useP4FusedKernel) {
            CHK_PRT_RET(lastSegmentSize > resCtx.localBuffer.size / param.rankSize,
                HCCL_ERROR("[ExecOp] P4 scratch slots exceed local buffer, stride[%llu], size[%llu]",
                    static_cast<unsigned long long>(lastSegmentSize),
                    static_cast<unsigned long long>(resCtx.localBuffer.size)), HCCL_E_INTERNAL);
        }
        if (param.rankSize == 16) {
            CHK_PRT_RET(scratchStride > resCtx.localBuffer.size / p16ScratchSlotCount,
                HCCL_ERROR("[ExecOp] P16 scratch slots exceed local buffer, stride[%llu], size[%llu]",
                    static_cast<unsigned long long>(scratchStride),
                    static_cast<unsigned long long>(resCtx.localBuffer.size)), HCCL_E_INTERNAL);
        }
        constexpr uint64_t p4TailAlignment = 4096;
        const uint64_t ownerMainSize = param.rankSize == 4 ?
            ownerSize - ownerSize % p4TailAlignment : ownerSize;
        const uint64_t ownOffset = tileOffset + param.myRank * segmentSize;
        const bool inputOutputEqual = param.inputPtr == param.outputPtr;
        std::array<uint64_t, CCU_TASK_ARG_NUM> taskArgs{};
        taskArgs[CCU_ARG_INPUT] = inputAddr;
        taskArgs[CCU_ARG_OUTPUT] = outputAddr;
        taskArgs[CCU_ARG_TOKEN] = token;
        taskArgs[CCU_ARG_SCRATCH_SAVE] =
            (useP4FusedKernel || param.rankSize == 16) ? scratchBase :
            (inputOutputEqual ? scratchBase : outputAddr + ownOffset);
        taskArgs[CCU_ARG_SCRATCH_PARTIAL] = param.rankSize == 16 ?
            scratchBase + scratchStride : scratchSecondHalf;
        taskArgs[CCU_ARG_TILE_OFFSET] = tileOffset;
        taskArgs[CCU_ARG_SEGMENT_SIZE] = segmentSize;
        taskArgs[CCU_ARG_LAST_SEGMENT_SIZE] = lastSegmentSize;
        taskArgs[CCU_ARG_SCRATCH_STRIDE] = scratchStride;
        taskArgs[CCU_ARG_OWNER_MAIN_SIZE] = ownerMainSize;
        taskArgs[CCU_ARG_OWNER_TAIL_SIZE] = ownerSize - ownerMainSize;
        taskArgs[CCU_ARG_INPUT_OUTPUT_EQUAL] = static_cast<uint64_t>(inputOutputEqual);
        if (useP4FusedKernel) {
            ccuRet = HcommCcuKernelLaunch(
                resCtx.threads[0], resCtx.ccuKernels[0], taskArgs.data(), taskArgs.size());
            CHK_PRT_RET(ccuRet != CCU_SUCCESS,
                HCCL_ERROR("[ExecOp] tile[%llu/%llu] P4 fused launch failed[%d]",
                    static_cast<unsigned long long>(launch),
                    static_cast<unsigned long long>(launchCount), ccuRet),
                ConvertCcuToHccl(ccuRet));
            continue;
        }
        if (param.rankSize == 16 && dieCount == 2) {
            // The secondary stream must begin with WAIT.  This validity edge
            // precedes both partial graphs and does not serialize Stage 0.
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyRecordOnThread(resCtx.threads[0], resCtx.threads[1], 0)));
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyWaitOnThread(resCtx.threads[1], 0, CUSTOM_TIMEOUT)));
            for (size_t die = 0; die < dieCount; ++die) {
                ccuRet = HcommCcuKernelLaunch(resCtx.threads[die], resCtx.ccuKernels[die],
                    taskArgs.data(), taskArgs.size());
                CHK_PRT_RET(ccuRet != CCU_SUCCESS,
                    HCCL_ERROR("[ExecOp] tile[%llu/%llu] P16 partial die[%zu] launch failed[%d]",
                        static_cast<unsigned long long>(launch),
                        static_cast<unsigned long long>(launchCount), die, ccuRet),
                    ConvertCcuToHccl(ccuRet));
            }
            // Both finalizers consume slots 0/1 as immutable inputs.  A full
            // two-way join guarantees that neither partial kernel can still be
            // writing them (or the now-dead source slots) when finalize starts.
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyRecordOnThread(resCtx.threads[0], resCtx.threads[1], 0)));
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyWaitOnThread(resCtx.threads[1], 0, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyRecordOnThread(resCtx.threads[1], resCtx.threads[0], 0)));
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyWaitOnThread(resCtx.threads[0], 0, CUSTOM_TIMEOUT)));

            ccuRet = HcommCcuKernelLaunch(resCtx.threads[0],
                resCtx.ccuKernels[dieCount], taskArgs.data(), taskArgs.size());
            CHK_PRT_RET(ccuRet != CCU_SUCCESS,
                HCCL_ERROR("[ExecOp] tile[%llu/%llu] P16 primary finalize failed[%d]",
                    static_cast<unsigned long long>(launch),
                    static_cast<unsigned long long>(launchCount), ccuRet),
                ConvertCcuToHccl(ccuRet));
            ccuRet = HcommCcuKernelLaunch(resCtx.threads[1],
                resCtx.ccuKernels[dieCount + 1], taskArgs.data(), taskArgs.size());
            CHK_PRT_RET(ccuRet != CCU_SUCCESS,
                HCCL_ERROR("[ExecOp] tile[%llu/%llu] P16 secondary finalize failed[%d]",
                    static_cast<unsigned long long>(launch),
                    static_cast<unsigned long long>(launchCount), ccuRet),
                ConvertCcuToHccl(ccuRet));
            // The primary stream is the operation completion stream.  Waiting
            // for the secondary record protects slot 2 and both finalize graphs
            // before the next tile can reuse scratch.
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyRecordOnThread(resCtx.threads[1], resCtx.threads[0], 0)));
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyWaitOnThread(resCtx.threads[0], 0, CUSTOM_TIMEOUT)));
            continue;
        }
        for (size_t stage = 0; stage < stageCount; ++stage) {
            if (dieCount == 2 && stage == 1) {
                // Primary finalize merges the complete owner shard first.  Its
                // stream record releases the secondary die only after that
                // kernel (including its channel-group distribution) completes.
                ccuRet = HcommCcuKernelLaunch(
                    resCtx.threads[0], resCtx.ccuKernels[dieCount], taskArgs.data(), taskArgs.size());
                CHK_PRT_RET(ccuRet != CCU_SUCCESS,
                    HCCL_ERROR("[ExecOp] tile[%llu/%llu] primary finalize launch failed[%d]",
                        static_cast<unsigned long long>(launch),
                        static_cast<unsigned long long>(launchCount), ccuRet),
                    ConvertCcuToHccl(ccuRet));
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyRecordOnThread(resCtx.threads[0], resCtx.threads[1], 0)));
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyWaitOnThread(resCtx.threads[1], 0, CUSTOM_TIMEOUT)));
                ccuRet = HcommCcuKernelLaunch(resCtx.threads[1],
                    resCtx.ccuKernels[dieCount + 1], taskArgs.data(), taskArgs.size());
                CHK_PRT_RET(ccuRet != CCU_SUCCESS,
                    HCCL_ERROR("[ExecOp] tile[%llu/%llu] secondary finalize launch failed[%d]",
                        static_cast<unsigned long long>(launch),
                        static_cast<unsigned long long>(launchCount), ccuRet),
                    ConvertCcuToHccl(ccuRet));
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyRecordOnThread(resCtx.threads[1], resCtx.threads[0], 0)));
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyWaitOnThread(resCtx.threads[0], 0, CUSTOM_TIMEOUT)));
                continue;
            }
            // The slave stream begins with WAIT.  The completion pair is a
            // local join: finalize cannot start until both partial graphs finish.
            if (dieCount == 2) {
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyRecordOnThread(resCtx.threads[0], resCtx.threads[1], 0)));
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyWaitOnThread(resCtx.threads[1], 0, CUSTOM_TIMEOUT)));
            }
            for (size_t die = 0; die < dieCount; ++die) {
                const size_t kernel = stage * dieCount + die;
                ccuRet = HcommCcuKernelLaunch(
                    resCtx.threads[die], resCtx.ccuKernels[kernel], taskArgs.data(), taskArgs.size());
                CHK_PRT_RET(ccuRet != CCU_SUCCESS,
                    HCCL_ERROR("[ExecOp] tile[%llu/%llu] stage[%zu] die[%zu] launch failed[%d]",
                        static_cast<unsigned long long>(launch),
                        static_cast<unsigned long long>(launchCount), stage, die, ccuRet),
                    ConvertCcuToHccl(ccuRet));
            }
            if (dieCount == 2) {
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyRecordOnThread(resCtx.threads[1], resCtx.threads[0], 0)));
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyWaitOnThread(resCtx.threads[0], 0, CUSTOM_TIMEOUT)));
            }
        }
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
