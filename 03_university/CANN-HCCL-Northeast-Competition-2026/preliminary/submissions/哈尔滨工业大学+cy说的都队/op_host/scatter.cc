#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param;
    sprintf(param.tag, "%s", "hccl_custom_scatter");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.root = root;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;

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
    if (root >= param.rankSize || dataType != HCCL_DATA_TYPE_FP32 ||
        recvCount > UINT64_MAX / sizeof(float) / param.rankSize) {
        return HCCL_E_PARA;
    }
    CHK_PTR_NULL(recvBuf);

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

        // TODO: 根据通信算法申请 Thread 资源
        // 创建 AICPU_TS 通信引擎上的 thread 资源
        uint32_t threadNum = param.rankSize; // 主线程 + 每个对端一个独立通信线程
        uint32_t notifyNumPerThread = param.rankSize; // 0用于启动，每个worker独立完成通知

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread,
            cpuTsEngine, &param.aicpuThreadOnCpu));

        // TODO: 根据通信算法申请 Channel 资源
        // 调用 HcclRankGraphGetLinks()、HcclChannelDescInit()、HcclChannelAcquire() 等接口按需申请 Channel 资源
        uint32_t *layerPtr = nullptr;
        uint32_t layerNum = 0;
        CHK_RET(HcclRankGraphGetLayers(comm, &layerPtr, &layerNum));
        std::vector<uint32_t> layers(layerPtr, layerPtr + layerNum);
        std::sort(layers.begin(), layers.end());
        std::vector<HcclChannelDesc> descs;
        std::vector<uint32_t> peers;
        std::vector<bool> connected(param.rankSize, false);
        connected[param.myRank] = true;
        for (uint32_t layer : layers) {
            uint32_t *rankPtr = nullptr;
            uint32_t rankNum = 0;
            CHK_RET(HcclRankGraphGetRanksByLayer(comm, layer, &rankPtr, &rankNum));
            std::vector<uint32_t> ranks(rankPtr, rankPtr + rankNum);
            CommTopo topo;
            CHK_RET(HcclRankGraphGetTopoTypeByLayer(comm, layer, &topo));
            if (topo == COMM_TOPO_1DMESH && resCtxHost.meshRanks.empty()) {
                resCtxHost.meshRanks = ranks;
                std::sort(resCtxHost.meshRanks.begin(), resCtxHost.meshRanks.end());
            }
            for (uint32_t peer : ranks) {
                if (peer >= param.rankSize || connected[peer]) {
                    continue;
                }
                CommLink *links = nullptr;
                uint32_t linkNum = 0;
                CHK_RET(HcclRankGraphGetLinks(comm, layer, param.myRank, peer, &links, &linkNum));
                if (linkNum == 0) {
                    continue;
                }
                HcclChannelDesc desc;
                CHK_RET(HcclChannelDescInit(&desc, 1));
                desc.remoteRank = peer;
                desc.localEndpoint = links[0].srcEndpointDesc;
                desc.remoteEndpoint = links[0].dstEndpointDesc;
                desc.channelProtocol = links[0].linkAttr.linkProtocol;
                desc.notifyNum = 5; // 两个buffer的credit/data-ready，以及跨调用释放通知
                descs.push_back(desc);
                peers.push_back(peer);
                connected[peer] = true;
            }
        }
        if (peers.size() + 1 != param.rankSize) {
            return HCCL_E_NOT_SUPPORT;
        }
        std::vector<ChannelHandle> handles(peers.size());
        if (!peers.empty()) {
            CHK_RET(HcclChannelAcquire(comm, aicpuTsEngine, descs.data(), descs.size(), handles.data()));
        }
        // 缓存完整的一对一通道表，后续改变root或数据量不需要重新建链。
        resCtxHost.channels.resize(param.rankSize);
        for (size_t i = 0; i < peers.size(); ++i) {
            ChannelInfo &channel = resCtxHost.channels[peers[i]];
            channel.remoteRank = peers[i];
            channel.handle = handles[i];
            channel.notifyNum = 5;
            CHK_RET(HcclChannelGetHcclBuffer(comm, handles[i], &channel.remoteCclMem.addr,
                &channel.remoteCclMem.size));
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
