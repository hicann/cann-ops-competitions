#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <vector>

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>
#include <hcomm/hcomm_primitives.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
// exec_op.h 里按“存在即包含”的方式引入了 HcclCommQueryCcuIns、
// HcommCcuKernelRegisterStart/Register/RegisterEnd 等 CCU 接口声明
#include "exec_op.h"
#include "ccu_kernel.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;

// 通信域的一层网络：层号 + 该层拓扑类型（1DMESH = server 内直连，CLOS = server 间）
struct NetLayerInfo {
    uint32_t layer;
    CommTopo topo;
};


bool PickLinkOnLayer(HcclComm comm, uint32_t layer, uint32_t myRank, uint32_t remoteRank, CommLink *outLink)
{
    CommLink *linkList = nullptr;
    uint32_t listSize = 0;
    if (HcclRankGraphGetLinks(comm, layer, myRank, remoteRank, &linkList, &listSize) != HCCL_SUCCESS) {
        return false; // 该层不覆盖这一对 rank
    }
    if (linkList == nullptr || listSize == 0) {
        return false;
    }

    // 同一层内可能有多条链路：优先 UB 直连协议，否则取第一条
    uint32_t pick = 0;
    for (uint32_t i = 0; i < listSize; ++i) {
        const CommProtocol protocol = linkList[i].linkAttr.linkProtocol;
        if (protocol == CommProtocol::COMM_PROTOCOL_UBC_CTP || protocol == CommProtocol::COMM_PROTOCOL_UBC_TP) {
            pick = i;
            break;
        }
    }
    *outLink = linkList[pick]; // 立即拷贝，规避库内内存被后续调用失效
    return true;
}

void FillChannelDesc(HcclChannelDesc &desc, uint32_t remoteRank, const CommLink &link)
{
    desc.remoteRank = remoteRank;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = link.srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
}

// 一个 die 上的 channel 集合：描述符 + 对端 rank（用于构造 kernel 注册参数）
struct ChannelGroup {
    std::vector<HcclChannelDesc> descs;
    std::vector<uint32_t> peerRank;
};

bool SelectRelayPlan(uint64_t cross, uint64_t local, uint32_t &relayCount, uint32_t &fracNum, uint32_t &fracDen)
{
    relayCount = 0;
    fracNum = 0;
    fracDen = 1;
    if (local == 0 || cross <= 4) {
        return false; // 没有 mesh，或 Clos 本来就够用（C 组：cross == 4 → r == 0）
    }
    const uint64_t m = std::min<uint64_t>(std::min(cross, local), RELAY_DEST_MAX); // 用满助手
    if (m == 0) {
        return false;
    }
    const uint64_t den = local + 4;
    relayCount = static_cast<uint32_t>(m);
    fracDen = static_cast<uint32_t>(den);
    // r ≤ 1 需要 cross − 4 ≤ local + 4；超出时截断到 1（该助手扛整片），此时不是最优
    // 但不会出错。四种赛题拓扑都不触发（A: 4/11、D: 5/6）。
    fracNum = static_cast<uint32_t>(std::min<uint64_t>(cross - 4, den));
    return true;
}

// 降级路径：挑一个能覆盖（尽量多）对端的网层，供「单 kernel 整层统一」时使用
int32_t PickUniformLayer(HcclComm comm, const std::vector<NetLayerInfo> &layers, uint32_t myRank, uint32_t rankSize)
{
    const uint32_t needCover = rankSize - 1;
    uint32_t bestCover = 0;
    int32_t bestLayer = -1;
    for (const NetLayerInfo &info : layers) {
        uint32_t cover = 0;
        for (uint32_t rank = 0; rank < rankSize; ++rank) {
            if (rank == myRank) {
                continue;
            }
            CommLink probe{};
            if (PickLinkOnLayer(comm, info.layer, myRank, rank, &probe)) {
                ++cover;
            }
        }
        if (cover > bestCover) {
            bestCover = cover;
            bestLayer = static_cast<int32_t>(info.layer);
        }
        if (cover == needCover) {
            return static_cast<int32_t>(info.layer);
        }
    }
    return bestLayer;
}
} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    OpParam param;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || root >= param.rankSize,
        HCCL_ERROR("invalid rankSize/root: rankSize=%u, root=%u", param.rankSize, root), HCCL_E_PARA);
    if (param.myRank == root) { // 仅 root 需要 sendBuf
        CHK_PTR_NULL(sendBuf);
    }

    // 构造算子参数：kernel 的编排依赖 root，故 tag 需区分 root
    int ret = std::snprintf(param.tag, sizeof(param.tag), "hccl_custom_scatter_root_%u", root);
    CHK_PRT_RET(ret <= 0, HCCL_ERROR("failed to fill param.tag"), HCCL_E_INTERNAL);
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.root = root;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;


    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    HcclResult dfxRet = HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo));
    if (dfxRet != HCCL_SUCCESS) {
        HCCL_WARNING("HcclDfxRegOpInfoByCommId failed, ret -> %d, skip dfx and continue",
            static_cast<int32_t>(dfxRet));
    }

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    const CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size) == HCCL_SUCCESS) {
        // CCU 资源已经存在，复用资源（内含 thread / channel / kernel 句柄）
        HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost;

        // ==============================================
        // STEP 2.1: 申请用于 Host/Device 同步的通信资源
        CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, 1, &param.cpuThread));
        resCtxHost.ccuThread = param.cpuThread;

        // 从通信域获取 HCCL Buffer（Device上的内存，默认总大小400MB）
        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        resCtxHost.threads.resize(1);
        resCtxHost.threads[0] = param.cpuThread;

        // 单卡场景无需 channel 与 CCU Kernel，ExecOp 中直接本地拷贝
        if (param.rankSize > 1) {
            // ==============================================
            // STEP 2.2: 申请资源：Channel、CCU Kernel
            std::vector<NetLayerInfo> layers;
            uint32_t *layerList = nullptr;
            uint32_t layerNum = 0;
            if (HcclRankGraphGetLayers(comm, &layerList, &layerNum) == HCCL_SUCCESS && layerList != nullptr &&
                layerNum > 0) {
                // layerList 内存由库内管理，须立即拷贝（重复调用可能使前次结果失效）
                layers.reserve(layerNum);
                for (uint32_t i = 0; i < layerNum; ++i) {
                    CommTopo topo = COMM_TOPO_RESERVED;
                    (void)HcclRankGraphGetTopoTypeByLayer(comm, layerList[i], &topo);
                    layers.push_back(NetLayerInfo{layerList[i], topo});
                }
            } else {
                // 取不到层次信息时退化为按 0/1 层探测，避免因该接口不可用而整体失败
                HCCL_WARNING("HcclRankGraphGetLayers failed, fallback to layer 0/1, myRank=%u", param.myRank);
                layers.push_back(NetLayerInfo{0, COMM_TOPO_RESERVED});
                layers.push_back(NetLayerInfo{1, COMM_TOPO_RESERVED});
            }


            std::map<uint32_t, ChannelGroup> groups;
            bool splitOk = true;
            for (uint32_t remoteRank = 0; remoteRank < param.rankSize && splitOk; ++remoteRank) {
                if (remoteRank == param.myRank) {
                    continue;
                }
                CommLink link{};
                bool found = false;
                for (const NetLayerInfo &info : layers) {
                    if (PickLinkOnLayer(comm, info.layer, param.myRank, remoteRank, &link)) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    splitOk = false;
                    break;
                }

                HcclChannelDesc desc;
                CHK_RET(HcclChannelDescInit(&desc, 1));
                FillChannelDesc(desc, remoteRank, link);

                uint32_t dieId = 0;
                if (HcclRankGraphGetEndpointInfo(comm, param.myRank, &desc.localEndpoint, ENDPOINT_ATTR_DIE_ID,
                        sizeof(uint32_t), &dieId) != HCCL_SUCCESS ||
                    dieId >= 2) {
                    splitOk = false;
                    break;
                }
                groups[dieId].descs.push_back(desc);
                groups[dieId].peerRank.push_back(remoteRank);
            }

            if (!splitOk) {
                // 降级：无法可靠按 die 分组时，退回「整层统一选层」单 kernel 路径
                HCCL_WARNING("per-peer die grouping failed, fallback to uniform-layer single kernel, myRank=%u",
                    param.myRank);
                groups.clear();
                const int32_t chosenLayer = PickUniformLayer(comm, layers, param.myRank, param.rankSize);
                CHK_PRT_RET(chosenLayer < 0,
                    HCCL_ERROR("no net layer can reach any peer, myRank=%u", param.myRank), HCCL_E_NOT_FOUND);
                for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
                    if (remoteRank == param.myRank) {
                        continue;
                    }
                    HcclChannelDesc desc;
                    CHK_RET(HcclChannelDescInit(&desc, 1));
                    CommLink link{};
                    CHK_PRT_RET(!PickLinkOnLayer(comm, static_cast<uint32_t>(chosenLayer), param.myRank, remoteRank,
                                    &link),
                        HCCL_ERROR("no link on layer=%d between local=%u and remote=%u", chosenLayer, param.myRank,
                            remoteRank),
                        HCCL_E_NOT_FOUND);
                    FillChannelDesc(desc, remoteRank, link);
                    groups[0].descs.push_back(desc);
                    groups[0].peerRank.push_back(remoteRank);
                }
            }
            uint32_t relayEnabled = 0;
            uint32_t relayCount = 0;
            std::vector<uint32_t> relayDest;   // 第 i 片的目的端
            std::vector<uint32_t> relayHelper; // 第 i 片的助手
            uint32_t relayHelperRank = INVALID_VALUE_RANKID;
            uint32_t relayIsHelper = 0;
            uint32_t relayMeshKernelIdx = 0;
            // 每个被中继的对端都走同一套「直发段 + 中继段」，没有整片/分数片之分
            uint32_t relayFracNum = 0;
            uint32_t relayFracDen = 1;
            std::vector<uint32_t> relayMeshPeers; // 本服（mesh）对端集合，用来认出哪个 kernel 是 mesh 那颗
            {
                auto sizeIt = SIZE_TABLE.find(param.dataType);
                const uint64_t sliceBytes = (sizeIt == SIZE_TABLE.end()) ? 0 : param.count * sizeIt->second;
                if (sliceBytes >= MIN_RELAY_SLICE_BYTES && groups.size() == 2) {
                    // 按【最内层】判定本服/跨服，不依赖 die 编号：layer 0 是 server 内的 netInstance，
                    // 跨 server 的对端在它里面查不到链路。
                    uint32_t innerLayer = layers[0].layer;
                    for (const NetLayerInfo &info : layers) {
                        if (info.layer < innerLayer) {
                            innerLayer = info.layer;
                        }
                    }
                    std::vector<uint32_t> serverRanks;
                    std::vector<uint32_t> crossRanks;
                    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
                        if (remoteRank == param.myRank) {
                            continue;
                        }
                        CommLink probe{};
                        if (PickLinkOnLayer(comm, innerLayer, param.myRank, remoteRank, &probe)) {
                            serverRanks.push_back(remoteRank);
                        } else {
                            crossRanks.push_back(remoteRank);
                        }
                    }
                    // 中转区放在 HCCL buffer 上，装不下就别开（一次最多发 CCU_MAX_DATA_SIZE）
                    if (!serverRanks.empty() && !crossRanks.empty() &&
                        cclBufferSize >= std::min<uint64_t>(MAX_DATA_SIZE, sliceBytes)) {
                        relayEnabled = 1;
                        relayMeshPeers = serverRanks;
                        const bool inRootServer =
                            (param.myRank == root) ||
                            (std::find(serverRanks.begin(), serverRanks.end(), root) != serverRanks.end());
                        if (inRootServer) {
                            std::vector<uint32_t> members = serverRanks;
                            members.push_back(param.myRank);
                            std::sort(members.begin(), members.end());
                            // 助手候选：本 server 内除 root 外、按 rank 升序
                            std::vector<uint32_t> helperPool;
                            for (uint32_t rank : members) {
                                if (rank != root) {
                                    helperPool.push_back(rank);
                                }
                            }
                            uint32_t k = 0;
                            if (SelectRelayPlan(crossRanks.size(), helperPool.size(), k, relayFracNum, relayFracDen)) {
                                for (uint32_t i = 0; i < k; ++i) {
                                    relayDest.push_back(crossRanks[i]);
                                    relayHelper.push_back(helperPool[i]);
                                    if (helperPool[i] == param.myRank) {
                                        relayHelperRank = param.myRank;
                                        relayIsHelper = 1;
                                    }
                                }
                                relayCount = k;
                            }
                        }
                        HCCL_INFO("[HcclScatter] relay plan, myRank=%u: enabled=%u count=%u r=%u/%u "
                                  "helper(me)=%u isHelper=%u",
                            param.myRank, relayEnabled, relayCount, relayFracNum, relayFracDen,
                            relayHelperRank, relayIsHelper);
                        for (uint32_t i = 0; i < relayCount; ++i) {
                            HCCL_INFO("[HcclScatter]   relay[%u]: dest=%u helper=%u (中继段 %u/%u)", i,
                                relayDest[i], relayHelper[i], relayFracNum, relayFracDen);
                        }
                    }
                }
            }
            resCtxHost.relayEnabled = relayEnabled;
            resCtxHost.relayCount = relayCount;
            resCtxHost.relayHelperRank = relayHelperRank;
            resCtxHost.relayFracNum = relayFracNum;
            resCtxHost.relayFracDen = relayFracDen;
            for (uint32_t i = 0; i < relayCount && i < RELAY_DEST_MAX; ++i) {
                resCtxHost.relayDest[i] = relayDest[i];
                resCtxHost.relayHelper[i] = relayHelper[i];
            }

            CcuInsHandle insHandle{0};
            uint32_t insNum = 0;
            CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
            CHK_PRT_RET(insNum == 0, HCCL_ERROR("no CCU instance found"), HCCL_E_INTERNAL);

            // 一个 Start/End 块内注册多个 kernel，每个 kernel 落在自己 die 上；自拷贝固定放在最小 die
            const uint32_t smallestDie = groups.begin()->first;
            CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
            uint32_t kernelIdx = 0;
            for (auto &entry : groups) {
                const uint32_t dieId = entry.first;
                ChannelGroup &group = entry.second;
                const uint32_t peerNum = static_cast<uint32_t>(group.descs.size());
                std::vector<ChannelHandle> handles(peerNum);
                if (peerNum > 0) {
                    CHK_RET(HcclChannelAcquire(comm, ccuEngine, group.descs.data(), peerNum, handles.data()));
                }

                auto kernelArg = std::make_shared<ops_hccl::CcuKernelArgScatter>();
                kernelArg->rankSize = param.rankSize;
                kernelArg->rankId = param.myRank;
                kernelArg->root = root;
                kernelArg->channelCount = peerNum;
                for (uint32_t i = 0; i < peerNum; ++i) {
                    kernelArg->channels[i] = handles[i];
                    kernelArg->peerRank[i] = group.peerRank[i];
                }
                kernelArg->doSelfCopy = (dieId == smallestDie) ? 1 : 0;
                // 中继计划是注册期常量：kernel 内靠 peerRank 里有没有这些 rank 自动选对分支，
                // 不需要再传 axisId。
                kernelArg->relayEnabled = relayEnabled;
                kernelArg->relayCount = (relayCount > RELAY_DEST_MAX) ? RELAY_DEST_MAX : relayCount;
                for (uint32_t i = 0; i < kernelArg->relayCount; ++i) {
                    kernelArg->relayDest[i] = relayDest[i];
                    kernelArg->relayHelper[i] = relayHelper[i];
                }
                kernelArg->relayIsHelper = relayIsHelper;
                kernelArg->relayFracNum = relayFracNum;
                kernelArg->relayFracDen = relayFracDen;

                {
                    auto pullIt = SIZE_TABLE.find(param.dataType);
                    const uint64_t pullSliceBytes =
                        (pullIt == SIZE_TABLE.end()) ? 0 : param.count * pullIt->second;
                    kernelArg->pullMode = (relayEnabled == 0 && pullSliceBytes > 0 &&
                                           pullSliceBytes < MIN_RELAY_SLICE_BYTES)
                                              ? 1
                                              : 0;
                }
                // 认出哪一颗是 mesh（本服）kernel：对端里有本服成员的这一组就是。
                // 助手要按 mesh → Clos 的顺序下发，靠程序序保证中转区可见。
                for (uint32_t i = 0; i < peerNum; ++i) {
                    if (std::find(relayMeshPeers.begin(), relayMeshPeers.end(), group.peerRank[i]) !=
                        relayMeshPeers.end()) {
                        relayMeshKernelIdx = kernelIdx;
                        break;
                    }
                }

                CcuKernelInfo kernelInfo{};
                const char kernelName[] = "CcuKernel";
                std::memcpy(kernelInfo.kernelFuncName, kernelName, sizeof(kernelName));
                kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuKernel);
                kernelInfo.setKernelArg(kernelArg);

                const void *kernelArgs[] = {kernelInfo.kernelArg};
                CcuKernelHandle kernelHandle{0};
                CHK_RET_CCU(HcommCcuKernelRegister(insHandle, dieId, kernelInfo.kernelFuncName, kernelInfo.kernelFunc,
                    kernelArgs, 1, &kernelHandle));
                resCtxHost.ccuKernels.push_back(kernelHandle);
                HCCL_INFO("register ccu kernel dieId=%u, peers=%u, myRank=%u", dieId, peerNum, param.myRank);
                ++kernelIdx;
            }
            CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
            resCtxHost.relayMeshKernelIdx = relayMeshKernelIdx;

            // 双线程并行：当 channel 落在两个 die（groups.size()==2）时，为除主线程外的每个 die
            // 再申请一个从线程，使各 die 的 kernel 能在各自线程上并行下发（仿生产
            // CcuTempReduceScatterMesh2Die 的 slaveThreadNum=1 双线程模式）。
            const uint32_t kernelNum = static_cast<uint32_t>(resCtxHost.ccuKernels.size());
            if (kernelNum > 1) {
                std::vector<ThreadHandle> slaves(kernelNum - 1);
                if (HcclThreadAcquire(comm, ccuEngine, kernelNum - 1, 1, slaves.data()) == HCCL_SUCCESS) {
                    for (uint32_t i = 0; i < kernelNum - 1; ++i) {
                        resCtxHost.threads.push_back(slaves[i]);
                    }
                } else {
                    // 从线程申请失败：降级为单线程串行下发（保留全部 kernel，仅少并行）
                    HCCL_WARNING("acquire slave thread failed, fallback to single-thread dispatch, myRank=%u",
                        param.myRank);
                }
            }
        }

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
        // 申请 CCU 通信引擎上下文，存放 AlgResourceCtx 信息
        std::vector<char> seq = resCtxHost.Serialize();
        param.ctxSize = seq.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, seq.data(), seq.size(), 0));
    }

    // ==============================================
    // STEP 3: 下发 CCU Kernel
    // ==============================================
    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
