#include <algorithm>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <vector>

#include <ccu/ccu_launch.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "ccu_kernel.h"
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;
constexpr uint32_t THREAD_NOTIFY_NUM = 4;
constexpr uint32_t RELAY_NOTIFY_NUM = 32;
constexpr uint64_t RELAY_TILE_BYTES = 32ULL * 1024 * 1024;
constexpr uint64_t RELAY_SCRATCH_BYTES = 2 * RELAY_TILE_BYTES;

struct RelayPlan {
    bool enabled = false;
    std::vector<uint32_t> helpers;
    std::vector<uint32_t> helperForRank;
    std::vector<HcclChannelDesc> descs;
    std::vector<uint32_t> dies;
    uint64_t bytes = 0;
};

struct DieChannelGroup {
    uint32_t dieId = 0;
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> remoteRanks;
};

HcclResult GetChannelDesc(HcclComm comm, uint32_t myRank, uint32_t remoteRank,
    const std::vector<uint32_t> &layers, HcclChannelDesc &desc, uint32_t &localDieId)
{
    for (uint32_t layer : layers) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, layer, myRank, remoteRank, &links, &linkNum));
        for (uint32_t index = 0; index < linkNum; ++index) {
            const CommLink &link = links[index];
            if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                continue;
            }
            CHK_RET(HcclChannelDescInit(&desc, 1));
            desc.remoteRank = remoteRank;
            desc.notifyNum = CHANNEL_NOTIFY_NUM;
            desc.channelProtocol = link.linkAttr.linkProtocol;
            desc.localEndpoint = link.srcEndpointDesc;
            desc.remoteEndpoint = link.dstEndpointDesc;

            EndpointAttrDieId dieId = 0;
            CHK_RET(HcclRankGraphGetEndpointInfo(
                comm, myRank, &desc.localEndpoint, ENDPOINT_ATTR_DIE_ID, sizeof(dieId), &dieId));
            localDieId = static_cast<uint32_t>(dieId);
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("No UBC_CTP link from rank[%u] to rank[%u]", myRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult RegisterKernel(CcuInsHandle insHandle, const OpParam &param, const DieChannelGroup &group,
    uint32_t threadIndex, std::vector<CcuKernelInfo> &kernelInfos, CcuKernelLaunchEntry &entry)
{
    auto kernelArg = std::make_shared<ScatterCcuKernelArg>();
    kernelArg->myRank = param.myRank;
    kernelArg->rankSize = param.rankSize;
    kernelArg->root = param.root;
    kernelArg->recvBytes = param.count * SIZE_TABLE.at(param.dataType);
    kernelArg->channelCount = static_cast<uint32_t>(group.channels.size());
    for (uint32_t index = 0; index < group.channels.size(); ++index) {
        kernelArg->channels[index] = group.channels[index];
        kernelArg->remoteRanks[index] = group.remoteRanks[index];
    }

    kernelInfos.emplace_back();
    CcuKernelInfo &info = kernelInfos.back();
    std::snprintf(info.kernelFuncName, sizeof(info.kernelFuncName), "scatter_v21_base_r%u_m%u_d%u",
        param.root, param.myRank, group.dieId);
    kernelArg->copyLocal = param.myRank == param.root && threadIndex == 0;
    // Reuse the existing pull protocol for small messages too. Every root Die
    // loads four args; peers load two. No new channels or topology queries.
    kernelArg->mode = SCATTER_PEER_PULL;
    info.kernelFunc = param.myRank == param.root
        ? reinterpret_cast<void *>(ops_hccl::CcuScatterPullRootKernel)
        : reinterpret_cast<void *>(ops_hccl::CcuScatterPullPeerKernel);
    if (kernelArg->recvBytes < 1024ULL * 1024) {
        info.kernelFunc = param.myRank == param.root
            ? reinterpret_cast<void *>(ops_hccl::CcuScatterSmallRootKernel)
            : reinterpret_cast<void *>(ops_hccl::CcuScatterSmallPeerKernel);
    }
    info.SetKernelArg(kernelArg);

    const void *kernelArgs[] = {info.kernelArg};
    entry.threadIndex = threadIndex;
    entry.channelCount = kernelArg->channelCount;
    CHK_RET_CCU(HcommCcuKernelRegister(insHandle, group.dieId, info.kernelFuncName, info.kernelFunc,
        kernelArgs, 1, &entry.handle));
    return HCCL_SUCCESS;
}

// Membership comes from the public local-instance API. Remote ranks do not
// reconstruct root's server: each channel negotiates its writer in CCU code.
HcclResult MakeRelayPlan(HcclComm comm, const OpParam &param,
    const std::vector<uint32_t> &layers, RelayPlan &plan)
{
    const uint64_t bytes = param.count * SIZE_TABLE.at(param.dataType);
    if (bytes < 8ULL * 1024 * 1024 || bytes > 2ULL * MAX_DATA_SIZE ||
        std::find(layers.begin(), layers.end(), 0) == layers.end() ||
        std::find(layers.begin(), layers.end(), 1) == layers.end()) {
        return HCCL_SUCCESS;
    }
    uint32_t *data = nullptr, size = 0;
    CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, 0, &data, &size));
    if (data == nullptr || size == 0) { return HCCL_SUCCESS; }
    std::vector<uint32_t> shape(data, data + size);
    std::sort(shape.begin(), shape.end());
    if (shape != std::vector<uint32_t>{8, 8} &&
        shape != std::vector<uint32_t>{4, 8} &&
        shape != std::vector<uint32_t>{3, 3, 3, 3}) { return HCCL_SUCCESS; }

    CHK_RET(HcclRankGraphGetRanksByLayer(comm, 0, &data, &size));
    CHK_PRT_RET(data == nullptr || size == 0, HCCL_ERROR("Missing local membership"), HCCL_E_PARA);
    std::vector<uint32_t> local(data, data + size);
    std::sort(local.begin(), local.end());
    const bool rootLocal = std::binary_search(local.begin(), local.end(), param.root);
    // Equal-size servers, or exactly two servers: root's server SIZE is
    // derivable everywhere without assuming anything about rank numbering.
    const uint32_t rootSize = rootLocal ? local.size() :
        (shape.size() == 2 ? param.rankSize - local.size() : shape[0]);
    const uint64_t count = ScatterRelayElements(param.count, rootSize - 1, param.rankSize - rootSize);
    if (count == 0) { return HCCL_SUCCESS; }
    plan.bytes = count * SIZE_TABLE.at(param.dataType);
    plan.helperForRank.assign(param.rankSize, INVALID_VALUE_RANKID);
    if (rootLocal) {
        std::vector<uint32_t> remote;
        for (uint32_t p = 0; p < param.rankSize; ++p) {
            if (!std::binary_search(local.begin(), local.end(), p)) { remote.push_back(p); }
            else if (p != param.root) { plan.helpers.push_back(p); }
        }
        for (uint32_t i = 0; i < plan.helpers.size(); ++i) {
            plan.helperForRank[remote[i]] = plan.helpers[i];
        }
    }
    // Symmetric one-channel-per-peer acquisition. Extra channels carry only
    // rendezvous metadata; the payload graph remains sparse.
    plan.descs.resize(param.rankSize);
    plan.dies.resize(param.rankSize);
    for (uint32_t p = 0; p < param.rankSize; ++p) {
        if (p != param.myRank) {
            CHK_RET(GetChannelDesc(comm, param.myRank, p, layers, plan.descs[p], plan.dies[p]));
        }
    }
    plan.enabled = true;
    return HCCL_SUCCESS;
}

HcclResult CreateRelayResources(HcclComm comm, const OpParam &param, const RelayPlan &plan,
    AlgResourceCtx &resource)
{
    const bool root = param.myRank == param.root;
    const bool helper = std::find(plan.helpers.begin(), plan.helpers.end(), param.myRank) != plan.helpers.end();
    resource.relayRole = root ? 1 : (helper ? 2 : 3);
    resource.relayBytes = plan.bytes;
    // A whole prefix uses one bank; larger transfers use two 32 MiB banks.
    // Both policies fit the same 64 MiB scratch budget on every rank.
    resource.tileBytes = plan.bytes <= RELAY_SCRATCH_BYTES ? plan.bytes : RELAY_TILE_BYTES;
    std::vector<uint32_t> peers;
    if (root) {
        for (uint32_t p = 0; p < param.rankSize; ++p) {
            if (p != param.root) {
                peers.push_back(p);
            }
        }
    } else {
        peers.push_back(param.root);
        if (helper) {
            for (uint32_t p = 0; p < param.rankSize; ++p) {
                if (plan.helperForRank[p] == param.myRank) {
                    peers.push_back(p);
                    ++resource.targetCount;
                }
            }
            CHK_RET(HcclGetHcclBuffer(comm, &resource.scratch.addr, &resource.scratch.size));
            const uint64_t banks = std::min<uint64_t>(2,
                (resource.relayBytes + resource.tileBytes - 1) / resource.tileBytes);
            const uint64_t required = banks * resource.targetCount * resource.tileBytes;
            CHK_PRT_RET(resource.scratch.addr == nullptr || resource.scratch.size < required,
                HCCL_ERROR("Relay requires %llu bytes of HCCL buffer", static_cast<unsigned long long>(required)),
                HCCL_E_MEMORY);
        } else if (plan.helperForRank[param.myRank] != INVALID_VALUE_RANKID) {
            peers.push_back(plan.helperForRank[param.myRank]);
        }
    }
    peers.clear();
    for (uint32_t p = 0; p < param.rankSize; ++p) {
        if (p != param.myRank) { peers.push_back(p); }
    }
    std::vector<HcclChannelDesc> descs;
    for (uint32_t p : peers) {
        descs.push_back(plan.descs[p]);
    }
    std::vector<ChannelHandle> handles(peers.size());
    CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_CCU, descs.data(), descs.size(), handles.data()));
    std::map<uint32_t, DieChannelGroup> groups;
    std::map<uint32_t, DieChannelGroup> controls;
    DieChannelGroup ingress;
    for (uint32_t i = 0; i < peers.size(); ++i) {
        const uint32_t p = peers[i];
        const bool spare = helper && p != param.root && plan.helperForRank[p] != param.myRank;
        DieChannelGroup &group = helper && p == param.root ? ingress :
            (spare ? controls[plan.dies[p]] : groups[plan.dies[p]]);
        group.dieId = plan.dies[p];
        group.channels.push_back(handles[i]);
        group.remoteRanks.push_back(p);
    }
    // Only fuse channels owned by the same physical CCU Die. No cross-Die
    // event visibility or shared notification state is assumed here.
    const bool resident = helper && groups.size() == 1 &&
        groups.begin()->first == ingress.dieId;
    resource.activeThreads = resident ? 1 : groups.size() + (helper ? 1 : 0);
    const uint32_t threadCount = resource.activeThreads + controls.size();
    CHK_PRT_RET(threadCount == 0 || threadCount > 15, HCCL_ERROR("Invalid relay thread count"), HCCL_E_INTERNAL);
    resource.threads.resize(threadCount);
    resource.threads[0] = param.cpuThread;
    if (threadCount > 1) {
        CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_CCU, threadCount - 1, RELAY_NOTIFY_NUM,
            &resource.threads[1]));
    }
    CcuInsHandle ins = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &ins, &insNum));
    CHK_PRT_RET(insNum != 1 || ins == 0, HCCL_ERROR("Invalid CCU instance"), HCCL_E_INTERNAL);
    CHK_RET_CCU(HcommCcuKernelRegisterStart(ins));
    std::vector<std::shared_ptr<ScatterCcuKernelArg>> args;
    auto addKernel = [&](const DieChannelGroup &group, uint32_t thread, uint32_t kind) -> HcclResult {
        auto arg = std::make_shared<ScatterCcuKernelArg>();
        arg->myRank = param.myRank;
        arg->rankSize = param.rankSize;
        arg->root = param.root;
        arg->recvBytes = param.count * SIZE_TABLE.at(param.dataType);
        arg->relayBytes = plan.bytes;
        arg->tileBytes = resource.tileBytes;
        arg->kind = kind;
        arg->copyLocal = root && thread == 0;
        arg->channelCount = group.channels.size();
        std::copy(plan.helperForRank.begin(), plan.helperForRank.end(), arg->helperForRank);
        std::copy(group.channels.begin(), group.channels.end(), arg->channels);
        std::copy(group.remoteRanks.begin(), group.remoteRanks.end(), arg->remoteRanks);
        args.push_back(arg);
        char name[64];
        std::snprintf(name, sizeof(name), "scatter_v21_r%u_m%u_d%u_k%u", param.root, param.myRank, group.dieId, kind);
        const void *kernelArgs[] = {arg.get()};
        CcuKernelLaunchEntry entry;
        entry.threadIndex = thread;
        entry.channelCount = arg->channelCount;
        entry.kind = kind;
        CHK_RET_CCU(HcommCcuKernelRegister(ins, group.dieId, name,
            reinterpret_cast<void *>(ops_hccl::CcuScatterRelayKernel), kernelArgs, 1, &entry.handle));
        resource.kernels.push_back(entry);
        return HCCL_SUCCESS;
    };
    uint32_t thread = 0;
    if (resident) {
        DieChannelGroup merged = ingress;
        const auto &output = groups.begin()->second;
        merged.channels.insert(merged.channels.end(), output.channels.begin(), output.channels.end());
        merged.remoteRanks.insert(merged.remoteRanks.end(), output.remoteRanks.begin(), output.remoteRanks.end());
        CHK_RET(addKernel(merged, 0, RELAY_RESIDENT_HELPER));
    } else if (helper) {
        for (uint32_t kind : {RELAY_INIT_INPUT}) {
            CHK_RET(addKernel(ingress, 0, kind));
        }
        thread = 1;
    }
    for (const auto &item : groups) {
        if (resident) {
            break;
        }
        if (helper) {
            for (uint32_t kind : {RELAY_INIT_OUTPUT}) {
                CHK_RET(addKernel(item.second, thread, kind));
            }
        } else {
            CHK_RET(addKernel(item.second, thread, root ? RELAY_ROOT : RELAY_RECEIVER));
        }
        ++thread;
    }
    thread = resource.activeThreads;
    for (const auto &item : controls) {
        CHK_RET(addKernel(item.second, thread++, RELAY_RECEIVER));
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(ins));
    // Cold-path diagnostic: makes it possible to distinguish a real relay run
    // from a topology fallback without adding work to warmed iterations.
    std::fprintf(stderr, "[scatter-v21] resident=%u relay role=%u rank=%u root=%u bytes=%llu helpers=%zu tile=%llu\n", static_cast<unsigned>(resident), resource.relayRole,
        param.myRank, param.root, static_cast<unsigned long long>(plan.bytes), plan.helpers.size(),
        static_cast<unsigned long long>(resource.tileBytes));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;

    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("Unsupported rank size[%u]", param.rankSize), HCCL_E_PARA);
    if (param.myRank == root) { CHK_PTR_NULL(sendBuf); }
    CHK_PRT_RET(root >= param.rankSize, HCCL_ERROR("Invalid root[%u]", root), HCCL_E_PARA);
    const auto typeIter = SIZE_TABLE.find(dataType);
    CHK_PRT_RET(typeIter == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type"), HCCL_E_PARA);
    CHK_PRT_RET(recvCount > std::numeric_limits<uint64_t>::max() / typeIter->second,
        HCCL_ERROR("Receive byte size overflow"), HCCL_E_PARA);
    const uint64_t recvBytes = recvCount * typeIter->second;
    if (recvBytes == 0) {
        return HCCL_SUCCESS;
    }

    const CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquireWithStream(comm, engine, stream,
        recvBytes >= 1024ULL * 1024 ? RELAY_NOTIFY_NUM : THREAD_NOTIFY_NUM, &param.cpuThread));
    if (param.rankSize == 1) {
        return static_cast<HcclResult>(HcommLocalCopyOnThread(param.cpuThread, recvBuf, sendBuf, recvBytes));
    }

    std::snprintf(param.tag, sizeof(param.tag), "hccl_scatter_ccu_v21_r%u_n%u_b%llu_t%u", root,
        param.rankSize, static_cast<unsigned long long>(recvBytes), static_cast<unsigned>(dataType));

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, engine, &ctx, &ctxSize) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = ctxSize;
    } else {
        uint32_t *layerData = nullptr;
        uint32_t layerNum = 0;
        CHK_RET(HcclRankGraphGetLayers(comm, &layerData, &layerNum));
        CHK_PRT_RET(layerData == nullptr || layerNum == 0, HCCL_ERROR("No network layer"), HCCL_E_NOT_FOUND);
        const std::vector<uint32_t> layers(layerData, layerData + layerNum);

        RelayPlan plan;
        CHK_RET(MakeRelayPlan(comm, param, layers, plan));
        if (plan.enabled) {
            AlgResourceCtx resource;
            CHK_RET(CreateRelayResources(comm, param, plan, resource));
            std::vector<char> sequence = resource.Serialize();
            param.ctxSize = sequence.size();
            CHK_RET(HcclEngineCtxCreate(comm, param.tag, engine, param.ctxSize, &param.resCtx));
            CHK_RET(HcclEngineCtxCopy(comm, engine, param.tag, sequence.data(), sequence.size(), 0));
            return ops_hccl::ExecOp(param);
        }
        if (recvBytes >= 1024ULL * 1024 && param.myRank == root) {
            std::fprintf(stderr, "[scatter-v21] windowed-pull fallback ranks=%u recvBytes=%llu\n",
                param.rankSize, static_cast<unsigned long long>(recvBytes));
        }

        std::vector<uint32_t> remoteRanks;
        if (param.myRank == root) {
            for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
                if (rank != root) {
                    remoteRanks.push_back(rank);
                }
            }
        } else {
            remoteRanks.push_back(root);
        }

        std::vector<HcclChannelDesc> descs(remoteRanks.size());
        std::vector<uint32_t> dieIds(remoteRanks.size());
        for (uint32_t index = 0; index < remoteRanks.size(); ++index) {
            CHK_RET(GetChannelDesc(comm, param.myRank, remoteRanks[index], layers, descs[index], dieIds[index]));
        }
        std::vector<ChannelHandle> channels(remoteRanks.size());
        CHK_RET(HcclChannelAcquire(
            comm, engine, descs.data(), static_cast<uint32_t>(descs.size()), channels.data()));

        std::map<uint32_t, DieChannelGroup> groupMap;
        for (uint32_t index = 0; index < channels.size(); ++index) {
            DieChannelGroup &group = groupMap[dieIds[index]];
            group.dieId = dieIds[index];
            group.channels.push_back(channels[index]);
            group.remoteRanks.push_back(remoteRanks[index]);
        }
        std::vector<DieChannelGroup> groups;
        for (auto &item : groupMap) {
            groups.push_back(std::move(item.second));
        }
        std::stable_sort(groups.begin(), groups.end(), [](const DieChannelGroup &left, const DieChannelGroup &right) {
            return left.channels.size() < right.channels.size();
        });

        AlgResourceCtx resource;
        resource.mode = SCATTER_PEER_PULL;
        resource.threads.resize(groups.size());
        resource.threads[0] = param.cpuThread;
        if (groups.size() > 1) {
            CHK_RET(HcclThreadAcquire(comm, engine, static_cast<uint32_t>(groups.size() - 1),
                THREAD_NOTIFY_NUM, &resource.threads[1]));
        }

        CcuInsHandle insHandle = 0;
        uint32_t insNum = 0;
        CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
        CHK_PRT_RET(insNum != 1 || insHandle == 0, HCCL_ERROR("Invalid CCU instance"), HCCL_E_INTERNAL);
        CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
        std::vector<CcuKernelInfo> kernelInfos;
        kernelInfos.reserve(groups.size());
        resource.kernels.resize(groups.size());
        for (uint32_t index = 0; index < groups.size(); ++index) {
            CHK_RET(RegisterKernel(insHandle, param, groups[index], index, kernelInfos, resource.kernels[index]));
        }
        CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));

        std::vector<char> sequence = resource.Serialize();
        param.ctxSize = sequence.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, engine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, engine, param.tag, sequence.data(), sequence.size(), 0));
    }

    return ops_hccl::ExecOp(param);
}
