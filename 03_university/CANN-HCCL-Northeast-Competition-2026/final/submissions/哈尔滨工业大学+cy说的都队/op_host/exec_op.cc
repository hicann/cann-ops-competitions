#include <algorithm>
#include <vector>

#include <hcomm/hcomm_primitives.h>
#include <ccu/ccu_res.h>

#include "custom.h"
#include "exec_op.h" // 同时带入 HcommCcuKernelLaunch 的声明（含头文件路径探测）
#include "log.h"

namespace ops_hccl {
namespace {
// 单次 CCU 任务可承载的最大数据量，超过则切分后多次下发
constexpr uint64_t CCU_MAX_DATA_SIZE = MAX_DATA_SIZE;
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    // 反序列化
    CHK_PRT_RET(param.resCtx == nullptr || param.ctxSize == 0,
        HCCL_ERROR("ccu engine ctx is invalid, ctx=%p, size=%llu", param.resCtx,
            static_cast<unsigned long long>(param.ctxSize)),
        HCCL_E_PTR);
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);

    auto sizeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIt == SIZE_TABLE.end(),
        HCCL_ERROR("unsupported dataType %d", static_cast<int32_t>(param.dataType)), HCCL_E_NOT_SUPPORT);
    const uint64_t typeSize = sizeIt->second;

    CHK_PRT_RET(typeSize != 0 && param.count > UINT64_MAX / typeSize,
        HCCL_ERROR("scatter slice size overflow, count=%llu", static_cast<unsigned long long>(param.count)),
        HCCL_E_PARA);
    const uint64_t sliceBytes = param.count * typeSize; // 每个 rank 的切片字节数
    if (sliceBytes == 0) {                              // 数据量为 0，无需下发
        return HCCL_SUCCESS;
    }

    // 单卡：直接本地拷贝（recvBuf <- sendBuf）
    if (param.rankSize == 1) {
        CHK_PRT_RET(param.inputPtr == nullptr, HCCL_ERROR("rankSize == 1 but sendBuf is nullptr"), HCCL_E_PTR);
        CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("ccu thread is empty"), HCCL_E_INTERNAL);
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, sliceBytes)));
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(resCtx.threads.empty() || resCtx.ccuKernels.empty(),
        HCCL_ERROR("ccu resource is empty, threads=%zu, kernels=%zu", resCtx.threads.size(), resCtx.ccuKernels.size()),
        HCCL_E_INTERNAL);

    // 计算本端 recvBuf 的 CCU 访问 token
    const uint64_t outputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t outputToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(outputAddr, sliceBytes, &outputToken));

    // root 额外计算 sendBuf 的 token；非 root 不使用 input，复用 output 以保证 token 合法
    uint64_t inputAddr = outputAddr;
    uint64_t inputToken = outputToken;
    if (param.myRank == param.root) {
        inputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
        CHK_PRT_RET(inputAddr == 0, HCCL_ERROR("root sendBuf is nullptr"), HCCL_E_PTR);
        CHK_RET_CCU(HcommCcuGetMemToken(inputAddr, sliceBytes * param.rankSize, &inputToken));
    }

    // 中继用的中转区：本 rank 的 HCCL buffer。只有开启中继时才需要地址和 token，
    const bool relayOn = (resCtx.relayEnabled != 0);
    uint64_t stagingAddr = 0;
    uint64_t stagingToken = 0;
    if (relayOn) {
        const uint64_t needBytes = std::min<uint64_t>(CCU_MAX_DATA_SIZE, sliceBytes);
        CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size < needBytes,
            HCCL_ERROR("relay needs hccl buffer >= %llu bytes, got %llu", static_cast<unsigned long long>(needBytes),
                static_cast<unsigned long long>(resCtx.localBuffer.size)),
            HCCL_E_INTERNAL);
        stagingAddr = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
        CHK_RET_CCU(HcommCcuGetMemToken(stagingAddr, resCtx.localBuffer.size, &stagingToken));
    }

    const size_t kernelNum = resCtx.ccuKernels.size();

    const bool relaySerial = relayOn && (param.myRank == resCtx.relayHelperRank);
    // kernel 按 die 分组，最多 2 颗（scatter.cc 里 dieId >= 2 会走降级）；用栈数组存下发顺序，
    // 免得每次下发都在堆上申请一份顺序表。
    constexpr size_t MAX_LAUNCH_KERNELS = 4;
    CHK_PRT_RET(kernelNum > MAX_LAUNCH_KERNELS,
        HCCL_ERROR("too many ccu kernels, count=%zu", kernelNum), HCCL_E_INTERNAL);
    size_t launchOrder[MAX_LAUNCH_KERNELS];
    size_t launchCount = 0;
    if (relaySerial && resCtx.relayMeshKernelIdx < kernelNum) {
        launchOrder[launchCount++] = resCtx.relayMeshKernelIdx; // 助手：mesh kernel 必须先跑
    }
    for (size_t k = 0; k < kernelNum; ++k) {
        if (relaySerial && k == resCtx.relayMeshKernelIdx) {
            continue;
        }
        launchOrder[launchCount++] = k;
    }

    const bool pullCall = (!relayOn) && (sliceBytes > 0) && (sliceBytes < MIN_RELAY_SLICE_BYTES);
    const bool parallelDispatch =
        (!relaySerial) && (!pullCall) && (resCtx.threads.size() >= kernelNum) && (kernelNum > 1);

    if (parallelDispatch) {
        // 前同步：主线程(threads[0])向每个从线程(threads[1..])发 notify，从线程等待后再启动
        for (size_t k = 1; k < kernelNum; ++k) {
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyRecordOnThread(resCtx.threads[0], resCtx.threads[k], 0)));
        }
        for (size_t k = 1; k < kernelNum; ++k) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resCtx.threads[k], 0, CUSTOM_TIMEOUT)));
        }
    }

    // 按单次下发上限切分，逐片下发 CCU Kernel（channel 已按 die 拆成多个 kernel，逐一 launch）
    for (uint64_t offset = 0; offset < sliceBytes; offset += CCU_MAX_DATA_SIZE) {
        const uint64_t chunkBytes = std::min(CCU_MAX_DATA_SIZE, sliceBytes - offset);
        constexpr uint32_t MAX_TASK_ARGS = 16;
        uint64_t taskArgs[MAX_TASK_ARGS];
        uint32_t argCount = 0;
        taskArgs[argCount++] = inputAddr + offset;  // root sendBuf 基址（含本次分片偏移）
        taskArgs[argCount++] = outputAddr + offset; // 本端 recvBuf 基址（含本次分片偏移）
        taskArgs[argCount++] = inputToken;          // sendBuf token
        taskArgs[argCount++] = outputToken;         // recvBuf token
        taskArgs[argCount++] = chunkBytes;          // 本分片中每个 rank 的字节数
        taskArgs[argCount++] = sliceBytes;          // 每个 rank 完整切片的字节数（跨目的 rank 步进）
        if (relayOn) {
            taskArgs[argCount++] = stagingAddr;  // 本 rank 的中转区基址
            taskArgs[argCount++] = stagingToken; // 中转区 token
            uint64_t prefixLen = chunkBytes;
            if (resCtx.relayCount > 0 && resCtx.relayFracDen > 0) {
                prefixLen = chunkBytes * (resCtx.relayFracDen - resCtx.relayFracNum) / resCtx.relayFracDen;
                if (prefixLen == 0) {
                    prefixLen = 1; // 兜底：两段都必须非空（r == 1 时才会走到，赛题拓扑不会）
                }
                if (prefixLen >= chunkBytes) {
                    prefixLen = chunkBytes - 1;
                }
            }
            taskArgs[argCount++] = prefixLen;              // 直发段长度
            taskArgs[argCount++] = chunkBytes - prefixLen; // 中继段长度
        }
        for (size_t i = 0; i < launchCount; ++i) {
            const size_t k = launchOrder[i];
            // 并行模式下 kernel[k] 下发到 threads[k]；串行模式下全部下发到 threads[0]，按 launchOrder 顺序执行
            const ThreadHandle &thread = parallelDispatch ? resCtx.threads[k] : resCtx.threads[0];
            CHK_RET_CCU(HcommCcuKernelLaunch(thread, resCtx.ccuKernels[k], taskArgs, argCount));
        }
    }

    if (parallelDispatch) {
        // 后同步：主线程等待每个从线程的 notify（从线程 kernel 跑完后发），确保全部完成再返回
        for (size_t k = 1; k < kernelNum; ++k) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resCtx.threads[0], 0, CUSTOM_TIMEOUT)));
        }
        for (size_t k = 1; k < kernelNum; ++k) {
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyRecordOnThread(resCtx.threads[k], resCtx.threads[0], 0)));
        }
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl

