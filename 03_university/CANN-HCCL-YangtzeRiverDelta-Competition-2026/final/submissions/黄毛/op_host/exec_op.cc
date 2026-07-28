#include <algorithm>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "custom.h"
#include "exec_op.h"

namespace ops_hccl {
constexpr uint64_t TREE_THRESHOLD_BYTES = 1024 * 1024;

HcclResult LaunchKernel(ThreadHandle thread, CcuKernelHandle kernel, const uint64_t *args, uint32_t argCount)
{
    CcuResult ret = HcommCcuKernelLaunch(thread, kernel, args, argCount);
    return ret == CCU_SUCCESS ? HCCL_SUCCESS : ConvertCcuResult(ret);
}

HcclResult LaunchDiePair(const AlgResourceCtx &resource, uint32_t &kernelIndex, uint32_t end,
    const uint64_t *args, uint32_t argCount)
{
    if (resource.threadCount < 2 || kernelIndex + 1 >= end ||
        resource.kernelDies[kernelIndex] == resource.kernelDies[kernelIndex + 1]) {
        HcclResult ret = LaunchKernel(resource.threads[0], resource.ccuKernels[kernelIndex], args, argCount);
        ++kernelIndex;
        return ret;
    }

    const uint32_t first = kernelIndex;
    const uint32_t second = kernelIndex + 1;
    const uint32_t mainIndex = resource.kernelDies[first] == 0 ? first : second;
    const uint32_t workerIndex = mainIndex == first ? second : first;
    int32_t ret = HcommThreadNotifyRecordOnThread(resource.threads[0], resource.threads[1], 0);
    if (ret != HCCL_SUCCESS) {
        return static_cast<HcclResult>(ret);
    }
    ret = HcommThreadNotifyWaitOnThreadWithDefaultTimeout(resource.threads[1], 0);
    if (ret != HCCL_SUCCESS) {
        return static_cast<HcclResult>(ret);
    }
    HcclResult launchRet = LaunchKernel(
        resource.threads[1], resource.ccuKernels[workerIndex], args, argCount);
    if (launchRet != HCCL_SUCCESS) {
        return launchRet;
    }
    ret = HcommThreadNotifyRecordOnThread(resource.threads[1], resource.threads[0], 0);
    if (ret != HCCL_SUCCESS) {
        return static_cast<HcclResult>(ret);
    }
    launchRet = LaunchKernel(resource.threads[0], resource.ccuKernels[mainIndex], args, argCount);
    if (launchRet != HCCL_SUCCESS) {
        return launchRet;
    }
    ret = HcommThreadNotifyWaitOnThreadWithDefaultTimeout(resource.threads[0], 0);
    if (ret != HCCL_SUCCESS) {
        return static_cast<HcclResult>(ret);
    }
    kernelIndex += 2;
    return HCCL_SUCCESS;
}

HcclResult ExecOp(const OpParam &param)
{
    AlgResourceCtx resource;
    CHK_RET(resource.DeSerialize(param.resCtx, param.ctxSize));
    if (resource.threadCount != 0) {
        resource.threads[0] = param.cpuThread;
    }

    auto sizeIter = SIZE_TABLE.find(param.dataType);
    if (sizeIter == SIZE_TABLE.end() || resource.kernelCount == 0 || resource.threadCount == 0) {
        return HCCL_E_INTERNAL;
    }
    const uint64_t totalBytes = param.count * sizeIter->second;
    const uint64_t baseAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t token = 0;
    CcuResult ccuRet = HcommCcuGetMemToken(baseAddr, totalBytes, &token);
    if (ccuRet != CCU_SUCCESS) {
        return ConvertCcuResult(ccuRet);
    }

    const uint64_t maxSliceBytes = totalBytes < TREE_THRESHOLD_BYTES ? MAX_DATA_SIZE :
        static_cast<uint64_t>(MAX_DATA_SIZE) * param.rankSize;
    for (uint64_t offset = 0; offset < totalBytes; offset += maxSliceBytes) {
        const uint64_t sliceBytes = std::min<uint64_t>(maxSliceBytes, totalBytes - offset);
        const uint64_t chunkBytes = (sliceBytes + param.rankSize - 1) / param.rankSize;
        const uint64_t lastChunkBytes = sliceBytes - chunkBytes * (param.rankSize - 1);
        const uint64_t directTaskArgs[] = {baseAddr + offset, token, sliceBytes, param.root};
        const uint64_t largeTaskArgs[] = {
            baseAddr + offset, token, sliceBytes, param.root, chunkBytes, lastChunkBytes};
        uint32_t begin = totalBytes < TREE_THRESHOLD_BYTES ? 0 : resource.directKernelCount;
        uint32_t end = totalBytes < TREE_THRESHOLD_BYTES ? resource.directKernelCount : resource.kernelCount;
        uint32_t kernelIndex = begin;
        while (kernelIndex < end) {
            const uint64_t *taskArgs = totalBytes < TREE_THRESHOLD_BYTES ? directTaskArgs : largeTaskArgs;
            const uint32_t taskArgCount = totalBytes < TREE_THRESHOLD_BYTES ? 4 : 6;
            HcclResult launchRet = LaunchDiePair(resource, kernelIndex, end, taskArgs, taskArgCount);
            if (launchRet != HCCL_SUCCESS) {
                return launchRet;
            }
        }
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
