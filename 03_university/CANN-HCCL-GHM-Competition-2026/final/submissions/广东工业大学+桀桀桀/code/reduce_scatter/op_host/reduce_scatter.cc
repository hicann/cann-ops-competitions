/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 */

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

#include <hccl/hccl_diag.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>
#include <hcomm/ccu/ccu_launch.h>

#include "ccu_kernel.h"
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {

constexpr uint32_t CHANNEL_NOTIFY_COUNT = 1;
constexpr uint32_t LAYER_MESH = 0;
constexpr uint32_t LAYER_CLOS = 1;
// The compact 2x8 candidate is functionally verified but regressed online:
// it trades Clos traffic for a second full local-Mesh partial reduction.
constexpr bool ENABLE_HIERARCHICAL_2X8 = false;
constexpr bool ENABLE_CCU_BUFFER_GROUP_REDUCE = false;
// Keep the direct all-link gather, but reduce the Mesh and Clos source sets
// on their respective CCUs before the final two-input reduction.
constexpr bool ENABLE_LAYER_PARTIAL_REDUCE = true;
// Online: the four NHR rounds raised 2x8 512KB latency from 22us to 38us.
// Keep its Checker-validated implementation disabled until a lower-overhead
// launch/synchronization form is available.
constexpr bool ENABLE_SMALL_NHR_2X8 = false;
// The CCL-buffer fan-in candidate aliases source and destination spans in a
// single LocalBatchReduce task.  CheckerV3 rejects that graph, so it remains
// disabled pending a non-overlapping buffer layout.
constexpr bool ENABLE_SMALL_GROUP_FANIN_2X8 = false;
// A pure 4x1 Clos domain can reduce directly into three disjoint output
// stripes.  Each phase uses every Clos channel once, while each stripe keeps
// a fixed peer order.  It avoids the baseline CCL gather and final HBM tree.
constexpr bool ENABLE_STRIPED_CLOS_DIRECT_4X1 = true;
constexpr bool ENABLE_STRIPED_SELF_COPY_OVERLAP_4X1 = true;
// The direct 8+4 graph is semantically valid but produces hundreds of
// ReadReduce tasks for its 7/4 and 3/8 channel groups.  Keep it isolated
// until a lower-task-count schedule can demonstrate a real online benefit.
constexpr bool ENABLE_STRIPED_TWO_LAYER_DIRECT_8P4 = false;

struct ChannelGroup {
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> peers;
};

HcclResult GetChannelDesc(HcclComm comm, uint32_t layer, uint32_t myRank, uint32_t peerRank,
    HcclChannelDesc &desc, bool &found)
{
    found = false;
    CommLink *links = nullptr;
    uint32_t linkCount = 0;
    CHK_RET(HcclRankGraphGetLinks(comm, layer, myRank, peerRank, &links, &linkCount));
    for (uint32_t index = 0; index < linkCount; ++index) {
        if (links[index].linkAttr.linkProtocol != COMM_PROTOCOL_UBC_CTP) {
            continue;
        }
        CHK_RET(HcclChannelDescInit(&desc, 1));
        desc.remoteRank = peerRank;
        desc.notifyNum = CHANNEL_NOTIFY_COUNT;
        desc.channelProtocol = links[index].linkAttr.linkProtocol;
        desc.localEndpoint = links[index].srcEndpointDesc;
        desc.remoteEndpoint = links[index].dstEndpointDesc;
        found = true;
        return HCCL_SUCCESS;
    }
    return HCCL_SUCCESS;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, std::array<ChannelGroup, 2> &groups)
{
    for (uint32_t peerRank = 0; peerRank < param.rankSize; ++peerRank) {
        if (peerRank == param.myRank) {
            continue;
        }

        HcclChannelDesc desc;
        bool found = false;
        uint32_t groupIndex = LAYER_MESH;
        CHK_RET(GetChannelDesc(comm, LAYER_MESH, param.myRank, peerRank, desc, found));
        if (!found) {
            groupIndex = LAYER_CLOS;
            CHK_RET(GetChannelDesc(comm, LAYER_CLOS, param.myRank, peerRank, desc, found));
        }
        CHK_PRT_RET(!found, HCCL_ERROR("no CCU-capable link from rank %u to rank %u", param.myRank, peerRank),
            HCCL_E_NOT_FOUND);

        ChannelHandle channel;
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &channel));
        groups[groupIndex].channels.push_back(channel);
        groups[groupIndex].peers.push_back(peerRank);
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterKernel(CcuInsHandle insHandle, const char *name, void *function,
    std::shared_ptr<ReduceScatterCcuKernelArg> kernelArg, CcuKernelHandle &kernelHandle)
{
    const void *kernelArgs[] = {kernelArg.get()};
    constexpr uint32_t kernelArgCount = 1;
    constexpr uint32_t dieId = 0;
    const CcuResult result = HcommCcuKernelRegister(insHandle, dieId, name, function, kernelArgs,
        kernelArgCount, &kernelHandle);
    CHK_PRT_RET(result != CCU_SUCCESS, HCCL_ERROR("failed to register %s: %d", name, result),
        ConvertCcuToHccl(result));
    return HCCL_SUCCESS;
}

std::shared_ptr<ReduceScatterCcuKernelArg> MakeKernelArg(const OpParam &param, KernelKind kind,
    const ChannelGroup *group, uint64_t slotStride, bool stageSelfInput = false)
{
    auto arg = std::make_shared<ReduceScatterCcuKernelArg>();
    arg->rankSize = param.rankSize;
    arg->rankId = param.myRank;
    arg->dataType = param.dataType;
    arg->reduceOp = param.reduceType;
    arg->kind = kind;
    arg->stageSelfInput = stageSelfInput;
    arg->useSmallTree = param.count * SIZE_TABLE.at(param.dataType) <= 128 * 1024;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        arg->slotOffsets[rank] = static_cast<uint64_t>(rank) * slotStride;
    }
    if (group != nullptr) {
        arg->commChannelCount = static_cast<uint32_t>(group->channels.size());
        for (uint32_t index = 0; index < arg->commChannelCount; ++index) {
            arg->commChannels[index] = group->channels[index];
            arg->peerRanks[index] = group->peers[index];
        }
    }
    return arg;
}

HcclResult CreateResources(HcclComm comm, OpParam &param)
{
    AlgResourceCtx resource;
    void *buffer = nullptr;
    uint64_t bufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &buffer, &bufferSize));
    resource.localBuffer = CommBuffer{buffer, bufferSize};

    resource.threads.resize(2);
    resource.threads[0] = param.cpuThread;
    CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU, 1, 1, &resource.threads[1]));
    resource.ccuThread = resource.threads[0];

    std::array<ChannelGroup, 2> groups;
    CHK_RET(AcquireChannels(comm, param, groups));
    CHK_PRT_RET(groups[LAYER_MESH].channels.size() > CUSTOM_MAX_CHANNELS ||
            groups[LAYER_CLOS].channels.size() > CUSTOM_MAX_CHANNELS,
        HCCL_ERROR("too many channels for a CCU kernel"), HCCL_E_INTERNAL);

    const uint64_t slotStride = resource.localBuffer.size / param.rankSize;
    CHK_PRT_RET(slotStride == 0, HCCL_ERROR("CCL buffer is too small"), HCCL_E_INTERNAL);
    CcuInsHandle insHandle{0};
    uint32_t insCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insCount));
    CHK_PRT_RET(insCount != 1, HCCL_ERROR("unexpected CCU instruction count: %u", insCount), HCCL_E_INTERNAL);
    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));

    std::vector<std::shared_ptr<ReduceScatterCcuKernelArg>> kernelArgs;
    const uint64_t outputBytes = param.count * SIZE_TABLE.at(param.dataType);
    const uint64_t totalInputBytes = outputBytes * param.rankSize;
    // In a balanced 2x8 domain, each output rank has exactly seven mesh
    // peers and eight Clos peers.  Reduce the local server first, then fetch
    // only the matching remote server partial sum.  This avoids sending the
    // same 8-way server contribution across all eight Clos links.
    const bool useHierarchicalTwoByEight = ENABLE_HIERARCHICAL_2X8 && outputBytes > 128 * 1024 && param.rankSize == 16 &&
        groups[LAYER_MESH].channels.size() == 7 && groups[LAYER_CLOS].channels.size() == 8;
    resource.useHierarchicalTwoByEight = useHierarchicalTwoByEight;
    const bool useCcuBufferGroupReduce = ENABLE_CCU_BUFFER_GROUP_REDUCE && totalInputBytes == 512ULL * 1024ULL &&
        param.rankSize == 16 && groups[LAYER_MESH].channels.size() == 7 && groups[LAYER_CLOS].channels.size() == 8;
    resource.useCcuBufferGroupReduce = useCcuBufferGroupReduce;
    const bool isTwoByEight = param.rankSize == 16 && groups[LAYER_MESH].channels.size() == 7 &&
        groups[LAYER_CLOS].channels.size() == 8;
    const bool useStripedClosDirect = ENABLE_STRIPED_CLOS_DIRECT_4X1 && outputBytes > 128 * 1024 &&
        param.rankSize == 4 && groups[LAYER_MESH].channels.empty() && groups[LAYER_CLOS].channels.size() == 3;
    resource.useStripedClosDirect = useStripedClosDirect;
    const uint32_t meshChannelCount = static_cast<uint32_t>(groups[LAYER_MESH].channels.size());
    const uint32_t closChannelCount = static_cast<uint32_t>(groups[LAYER_CLOS].channels.size());
    const bool useStripedTwoLayerDirect = ENABLE_STRIPED_TWO_LAYER_DIRECT_8P4 && outputBytes > 128 * 1024 &&
        param.rankSize == 12 && ((meshChannelCount == 7 && closChannelCount == 4) ||
            (meshChannelCount == 3 && closChannelCount == 8));
    resource.useStripedTwoLayerDirect = useStripedTwoLayerDirect;
    // Online performance confirms the partial form only for the 2x8,
    // two-chunk 512MiB case.  The rank4/rank12 forms add a final local
    // reduction without reducing their critical traffic, so retain their
    // original gather/final tree.
    const bool useLayerPartialReduce = ENABLE_LAYER_PARTIAL_REDUCE && !useHierarchicalTwoByEight &&
        !useCcuBufferGroupReduce && outputBytes > 128 * 1024 &&
        isTwoByEight && totalInputBytes == 512ULL * 1024ULL * 1024ULL;
    resource.useLayerPartialReduce = useLayerPartialReduce;
    const bool useSmallNhr = ENABLE_SMALL_NHR_2X8 && totalInputBytes == 512ULL * 1024ULL &&
        param.rankSize == 16 && groups[LAYER_MESH].channels.size() == 7 && groups[LAYER_CLOS].channels.size() == 8;
    resource.useSmallNhr = useSmallNhr;
    const bool useSmallGroupFanIn = ENABLE_SMALL_GROUP_FANIN_2X8 && !useSmallNhr &&
        totalInputBytes == 512ULL * 1024ULL && param.rankSize == 16 &&
        groups[LAYER_MESH].channels.size() == 7 && groups[LAYER_CLOS].channels.size() == 8;
    resource.useSmallGroupFanIn = useSmallGroupFanIn;
    // Online data: staging the local contribution inside gather only helps
    // the 12/16-rank 512MiB double-chunk cases.  It delays the one-chunk
    // 400MiB path and the 4x1 topology, so retain the original final-copy
    // path everywhere else.
    const bool useStagedSelfInput = (param.rankSize == 12 || param.rankSize == 16) &&
        totalInputBytes >= 512ULL * 1024ULL * 1024ULL;
    bool selfInputStaged = false;
    uint32_t meshPartialSlot = INVALID_VALUE_RANKID;
    uint32_t closPartialSlot = INVALID_VALUE_RANKID;
    for (uint32_t layer = 0; layer < groups.size(); ++layer) {
        if (useHierarchicalTwoByEight && layer == LAYER_CLOS) {
            continue;
        }
        if (groups[layer].channels.empty()) {
            continue;
        }
        if (useStripedTwoLayerDirect) {
            auto arg = MakeKernelArg(param, KernelKind::STRIPED_LAYER_DIRECT, &groups[layer], slotStride);
            arg->directInitializeOutput = layer == LAYER_MESH;
            resource.directStripeCounts[layer] = arg->commChannelCount;
            CcuKernelHandle handle;
            CHK_RET(RegisterKernel(insHandle, "CcuStripedLayerDirectKernel",
                reinterpret_cast<void *>(ops_hccl::CcuStripedLayerDirectKernel), arg, handle));
            resource.gatherKernelIndex[layer] = static_cast<uint32_t>(resource.ccuKernels.size());
            resource.ccuKernels.push_back(handle);
            kernelArgs.push_back(arg);
            continue;
        }
        if (useStripedClosDirect) {
            CHK_PRT_RET(layer != LAYER_CLOS, HCCL_ERROR("striped direct requires a pure Clos domain"), HCCL_E_INTERNAL);
            auto arg = MakeKernelArg(param, KernelKind::STRIPED_CLOS_DIRECT, &groups[layer], slotStride);
            // The split self-copy overlaps the two-chunk-equivalent 512MiB
            // case, but adds enough task overhead to regress the 400MiB+4B
            // tail case.  Keep the online-proven selector exact.
            arg->stageSelfInput = ENABLE_STRIPED_SELF_COPY_OVERLAP_4X1 &&
                totalInputBytes == 512ULL * 1024ULL * 1024ULL;
            CcuKernelHandle handle;
            CHK_RET(RegisterKernel(insHandle, "CcuStripedClosDirectKernel",
                reinterpret_cast<void *>(ops_hccl::CcuStripedClosDirectKernel), arg, handle));
            resource.gatherKernelIndex[layer] = static_cast<uint32_t>(resource.ccuKernels.size());
            resource.ccuKernels.push_back(handle);
            kernelArgs.push_back(arg);
            continue;
        }
        if (useSmallNhr) {
            auto arg = MakeKernelArg(param, KernelKind::NHR, &groups[layer], slotStride);
            if (layer == LAYER_CLOS) {
                arg->nhrDistances[0] = 8;
                arg->nhrStepCount = 1;
            } else {
                arg->nhrDistances[0] = 4;
                arg->nhrDistances[1] = 2;
                arg->nhrDistances[2] = 1;
                arg->nhrStepCount = 3;
                arg->nhrWriteOutput = true;
            }
            CcuKernelHandle handle;
            CHK_RET(RegisterKernel(insHandle, "CcuNhrKernel", reinterpret_cast<void *>(ops_hccl::CcuNhrKernel),
                arg, handle));
            resource.gatherKernelIndex[layer] = static_cast<uint32_t>(resource.ccuKernels.size());
            resource.ccuKernels.push_back(handle);
            kernelArgs.push_back(arg);
            continue;
        }
        if (useSmallGroupFanIn) {
            const bool includeSelf = layer == LAYER_MESH;
            auto arg = MakeKernelArg(param, KernelKind::GROUP_REDUCE, &groups[layer], slotStride);
            arg->includeSelf = includeSelf;
            arg->partialResultSlot = includeSelf ? param.myRank : groups[layer].peers.front();
            if (layer == LAYER_MESH) {
                meshPartialSlot = arg->partialResultSlot;
            } else {
                closPartialSlot = arg->partialResultSlot;
            }
            CcuKernelHandle handle;
            CHK_RET(RegisterKernel(insHandle, "CcuGroupFanInKernel",
                reinterpret_cast<void *>(ops_hccl::CcuGatherKernel), arg, handle));
            resource.gatherKernelIndex[layer] = static_cast<uint32_t>(resource.ccuKernels.size());
            resource.ccuKernels.push_back(handle);
            kernelArgs.push_back(arg);
            continue;
        }
        const bool includeSelf = useLayerPartialReduce &&
            (layer == LAYER_MESH || groups[LAYER_MESH].channels.empty());
        const bool stageSelfInput = includeSelf || (useStagedSelfInput && !selfInputStaged);
        const KernelKind kind = useHierarchicalTwoByEight ? KernelKind::HIERARCHICAL_MESH_PARTIAL :
            (useCcuBufferGroupReduce ? KernelKind::GROUP_REDUCE :
                (useLayerPartialReduce ? KernelKind::LOCAL_SERVER_REDUCE : KernelKind::GATHER));
        auto arg = MakeKernelArg(param, kind, &groups[layer], slotStride, stageSelfInput);
        arg->includeSelf = useCcuBufferGroupReduce && layer == LAYER_MESH;
        if (useLayerPartialReduce) {
            std::vector<uint32_t> localRanks = groups[layer].peers;
            if (includeSelf) {
                localRanks.push_back(param.myRank);
            }
            std::sort(localRanks.begin(), localRanks.end());
            arg->localRankCount = static_cast<uint32_t>(localRanks.size());
            for (uint32_t index = 0; index < arg->localRankCount; ++index) {
                arg->localRanks[index] = localRanks[index];
            }
            // The self-containing layer can safely use the local rank slot.
            // The other layer chooses its smallest source slot, which is
            // disjoint because the rank graph partitions the peer sets.
            arg->partialResultSlot = includeSelf ? param.myRank : localRanks.front();
            if (layer == LAYER_MESH) {
                meshPartialSlot = arg->partialResultSlot;
            } else {
                closPartialSlot = arg->partialResultSlot;
            }
        }
        if (useHierarchicalTwoByEight) {
            arg->closPeerRank = param.myRank < 8 ? param.myRank + 8 : param.myRank - 8;
            std::vector<uint32_t> localRanks = groups[LAYER_MESH].peers;
            localRanks.push_back(param.myRank);
            std::sort(localRanks.begin(), localRanks.end());
            arg->localRankCount = static_cast<uint32_t>(localRanks.size());
            for (uint32_t index = 0; index < arg->localRankCount; ++index) {
                arg->localRanks[index] = localRanks[index];
            }
        }
        CcuKernelHandle handle;
        const char *kernelName = useHierarchicalTwoByEight ? "CcuHierarchicalMeshPartialKernel" : "CcuGatherKernel";
        void *kernelFunc = useHierarchicalTwoByEight ?
            reinterpret_cast<void *>(ops_hccl::CcuHierarchicalMeshPartialKernel) :
            reinterpret_cast<void *>(ops_hccl::CcuGatherKernel);
        CHK_RET(RegisterKernel(insHandle, kernelName, kernelFunc, arg, handle));
        resource.gatherKernelIndex[layer] = static_cast<uint32_t>(resource.ccuKernels.size());
        resource.ccuKernels.push_back(handle);
        kernelArgs.push_back(arg);
        selfInputStaged = selfInputStaged || stageSelfInput;
    }

    if (useHierarchicalTwoByEight) {
        const uint32_t closPeer = param.myRank < 8 ? param.myRank + 8 : param.myRank - 8;
        ChannelGroup pairGroup;
        for (uint32_t index = 0; index < groups[LAYER_CLOS].peers.size(); ++index) {
            if (groups[LAYER_CLOS].peers[index] == closPeer) {
                pairGroup.peers.push_back(closPeer);
                pairGroup.channels.push_back(groups[LAYER_CLOS].channels[index]);
                break;
            }
        }
        CHK_PRT_RET(pairGroup.channels.size() != 1, HCCL_ERROR("2x8 matching Clos peer is unavailable"),
            HCCL_E_NOT_FOUND);
        auto closPairArg = MakeKernelArg(param, KernelKind::HIERARCHICAL_CLOS_PAIR, &pairGroup, slotStride);
        closPairArg->closPeerRank = closPeer;
        CcuKernelHandle closPairHandle;
        CHK_RET(RegisterKernel(insHandle, "CcuHierarchicalClosPairKernel",
            reinterpret_cast<void *>(ops_hccl::CcuHierarchicalClosPairKernel), closPairArg, closPairHandle));
        resource.closPairGatherKernelIndex = static_cast<uint32_t>(resource.ccuKernels.size());
        resource.ccuKernels.push_back(closPairHandle);
        kernelArgs.push_back(closPairArg);

        auto finalPairArg = MakeKernelArg(param, KernelKind::HIERARCHICAL_FINAL, nullptr, slotStride);
        finalPairArg->closPeerRank = closPeer;
        CcuKernelHandle finalPairHandle;
        CHK_RET(RegisterKernel(insHandle, "CcuHierarchicalFinalKernel",
            reinterpret_cast<void *>(ops_hccl::CcuHierarchicalFinalKernel), finalPairArg, finalPairHandle));
        resource.finalPairReduceKernelIndex = static_cast<uint32_t>(resource.ccuKernels.size());
        resource.ccuKernels.push_back(finalPairHandle);
        kernelArgs.push_back(finalPairArg);
    }

    if (useLayerPartialReduce || useSmallGroupFanIn) {
        const uint32_t firstPartialSlot = meshPartialSlot != INVALID_VALUE_RANKID ? meshPartialSlot : closPartialSlot;
        const uint32_t secondPartialSlot = closPartialSlot != INVALID_VALUE_RANKID ? closPartialSlot : meshPartialSlot;
        CHK_PRT_RET(firstPartialSlot == INVALID_VALUE_RANKID || secondPartialSlot == INVALID_VALUE_RANKID,
            HCCL_ERROR("layer partial slots are incomplete"), HCCL_E_INTERNAL);
        auto finalArg = MakeKernelArg(param, KernelKind::FINAL_PAIR_REDUCE, nullptr, slotStride);
        // CcuFinalPairReduce treats equal slots as a copy-only single-layer
        // finalization, which covers pure Clos domains such as 4x1.
        finalArg->rankId = firstPartialSlot;
        finalArg->closPeerRank = secondPartialSlot;
        CcuKernelHandle finalHandle;
        CHK_RET(RegisterKernel(insHandle, "CcuLayerPartialFinalKernel",
            reinterpret_cast<void *>(ops_hccl::CcuFinalPairReduceKernel), finalArg, finalHandle));
        resource.finalPairReduceKernelIndex = static_cast<uint32_t>(resource.ccuKernels.size());
        resource.ccuKernels.push_back(finalHandle);
        kernelArgs.push_back(finalArg);
    } else if (!useHierarchicalTwoByEight && !useCcuBufferGroupReduce && !useSmallNhr && !useStripedClosDirect &&
        !useStripedTwoLayerDirect) {
        auto finalArg = MakeKernelArg(param, KernelKind::FINAL_REDUCE, nullptr, slotStride);
        finalArg->selfInputStaged = selfInputStaged;
        CcuKernelHandle finalHandle;
        CHK_RET(RegisterKernel(insHandle, "CcuFinalReduceKernel",
            reinterpret_cast<void *>(ops_hccl::CcuFinalReduceKernel), finalArg, finalHandle));
        resource.finalReduceKernelIndex = static_cast<uint32_t>(resource.ccuKernels.size());
        resource.ccuKernels.push_back(finalHandle);
        // A pure local kernel has no externally visible Die affinity.  Keep the
        // stable thread-0 path until a resource-isolated overlap is proven.
        resource.finalReduceThreadIndex = 0;
        kernelArgs.push_back(finalArg);
    }

    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    std::vector<char> serialized = resource.Serialize();
    param.ctxSize = serialized.size();
    CHK_RET(HcclEngineCtxCreate(comm, param.tag, CommEngine::COMM_ENGINE_CCU, param.ctxSize, &param.resCtx));
    CHK_RET(HcclEngineCtxCopy(comm, CommEngine::COMM_ENGINE_CCU, param.tag, serialized.data(), serialized.size(), 0));
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32 || op != HCCL_REDUCE_SUM,
        HCCL_ERROR("only fp32 sum is supported"), HCCL_E_NOT_SUPPORT);

    OpParam param;
    (void)snprintf(param.tag, sizeof(param.tag), "%s", "hccl_custom_reducescatter_ccu");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;

    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > CUSTOM_MAX_RANK_SIZE,
        HCCL_ERROR("unsupported rank size: %u", param.rankSize), HCCL_E_NOT_SUPPORT);

    const CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquireWithStream(comm, engine, stream, 1, &param.cpuThread));
    void *context = nullptr;
    uint64_t contextSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, engine, &context, &contextSize) == HCCL_SUCCESS) {
        param.resCtx = context;
        param.ctxSize = contextSize;
    } else {
        CHK_RET(CreateResources(comm, param));
    }
    return ops_hccl::ExecOp(param);
}
