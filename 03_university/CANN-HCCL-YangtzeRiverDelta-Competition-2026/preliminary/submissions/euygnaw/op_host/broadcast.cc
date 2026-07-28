#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param;
    sprintf(param.tag, "%s", "hccl_custom_broadcast");
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    // 注册算子信息
    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

    // ==============================================
    // STEP 2.1: 申请用于 Host/Device 同步的通信资源
    // ==============================================
    // 将用户传入的 stream 转换为 thread，并申请 Notify；同时导出为 AICPU 上可用的 thread
    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size) == HCCL_SUCCESS) {
        // AICPU 资源已经存在，复用资源
        HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;

        // Host 资源已经存在，复用资源
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost;

        // 从通信域获取 HCCL Buffer（Device上的内存，默认总大小400MB）
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        // 1条控制Thread + 7条通信Worker + 1条本地拷贝Thread。
        // 每条Thread申请8个Notify：slot[0..3]用于启动，slot[4..7]用于copy完成同步。
        constexpr uint32_t threadNum = 9;
        constexpr uint32_t notifyNumPerThread = 8;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        // 为当前rank到其余所有rank申请Channel。资源与root无关，因此root=0和root=7可复用同一份上下文。
        if (param.rankSize > 1) {
            const uint32_t channelNum = param.rankSize - 1;
            std::vector<uint32_t> remoteRanks;
            remoteRanks.reserve(channelNum);
            for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
                if (rank != param.myRank) {
                    remoteRanks.push_back(rank);
                }
            }

            std::vector<HcclChannelDesc> channelDescs(channelNum);
            CHK_RET(HcclChannelDescInit(channelDescs.data(), channelNum));

            uint32_t *netLayers = nullptr;
            uint32_t netLayerNum = 0;
            CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));

            // 四个流水slot各使用3个Notify：ACK、Scatter/跨Server数据、Server内AllGather数据。
            // slot0使用[0..2]，slot1使用[3..5]，slot2使用[6..8]，slot3使用[9..11]。
            constexpr uint32_t channelNotifyNum = 12;
            for (uint32_t idx = 0; idx < channelNum; ++idx) {
                const uint32_t remoteRank = remoteRanks[idx];
                bool linkFound = false;
                for (uint32_t layerIdx = 0; layerIdx < netLayerNum; ++layerIdx) {
                    CommLink *links = nullptr;
                    uint32_t linkNum = 0;
                    CHK_RET(HcclRankGraphGetLinks(
                        comm, netLayers[layerIdx], param.myRank, remoteRank, &links, &linkNum));
                    if (linkNum == 0) {
                        continue;
                    }

                    channelDescs[idx].remoteRank = remoteRank;
                    channelDescs[idx].channelProtocol = links[0].linkAttr.linkProtocol;
                    channelDescs[idx].localEndpoint = links[0].srcEndpointDesc;
                    channelDescs[idx].remoteEndpoint = links[0].dstEndpointDesc;
                    channelDescs[idx].notifyNum = channelNotifyNum;
                    linkFound = true;
                    break;
                }

                CHK_PRT_RET(!linkFound,
                    HCCL_ERROR("No communication link from rank[%u] to rank[%u]", param.myRank, remoteRank),
                    HCCL_E_INTERNAL);
            }

            std::vector<ChannelHandle> channelHandles(channelNum);
            CHK_RET(HcclChannelAcquire(
                comm, aicpuTsEngine, channelDescs.data(), channelNum, channelHandles.data()));

            resCtxHost.channels.resize(channelNum);
            for (uint32_t idx = 0; idx < channelNum; ++idx) {
                void *remoteBufferAddr = nullptr;
                uint64_t remoteBufferSize = 0;
                CHK_RET(HcclChannelGetHcclBuffer(
                    comm, channelHandles[idx], &remoteBufferAddr, &remoteBufferSize));
                resCtxHost.channels[idx].remoteRank = remoteRanks[idx];
                resCtxHost.channels[idx].notifyNum = channelNotifyNum;
                resCtxHost.channels[idx].handle = channelHandles[idx];
                resCtxHost.channels[idx].remoteCclMem = CommBuffer{remoteBufferAddr, remoteBufferSize};
            }
        }

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
        // 申请 AICPU 通信引擎上下文，存放 AlgResourceCtx 信息
        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, seq.data(), seqSize, 0));
        // 申请 CPU 通信引擎上下文，存放 aicpuThread 句柄
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = sizeof(ThreadHandle);
        const void *aicpuThreadPtr = static_cast<const void *>(&resCtxHost.aicpuThread);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
        CHK_RET(HcclEngineCtxCopy(comm, cpuTsEngine, param.tag, aicpuThreadPtr, hostCtxSize, 0));
    }

    // ==============================================
    // STEP 3: 下发 AICPU Kernel
    // ==============================================
    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}