/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

// 北京科技大学南风的队伍：决赛 Host 侧资源申请与算法编排实现。

#include <algorithm>
#include <cstdio>
#include <memory>
#include <vector>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>
#include <ccu/ccu_launch.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "../op_kernel_ccu/ccu_kernel.h"

namespace {
// 拓扑层编号：层 0 表示同组链路，层 1 表示跨组链路。
constexpr uint32_t LAYER_INTRA = 0;
constexpr uint32_t LAYER_INTER = 1;
constexpr uint32_t NOTIFY_NUM_SDMA = 2;
constexpr uint32_t NOTIFY_NUM_RDMA = 3;
constexpr uint64_t LARGE_THRESHOLD = 1024 * 1024;

// 根据消息大小和 Rank 数选择小消息、扁平或 Gateway 算法。
enum class Mode { SMALL_PULL, FLAT, G2_GATEWAY };

// 描述当前 Rank 所在的本地组及其对应的远端组。
struct Groups {
    bool valid = false;
    uint32_t localSlot = 0;
    std::vector<uint32_t> local;
    std::vector<uint32_t> remote;
};

struct Channel {
    ChannelHandle handle = 0;
    uint32_t remoteRank = 0;
    uint32_t layer = 0;
    uint32_t dieId = 0;
};

// 从 RankGraph 拷贝指定层的 Rank 列表，避免后续依赖框架临时内存。
bool CopyRanks(HcclComm comm, uint32_t layer, std::vector<uint32_t> &result)
{
    uint32_t *ranks = nullptr;
    uint32_t count = 0;
    if (HcclRankGraphGetRanksByLayer(comm, layer, &ranks, &count) != HCCL_SUCCESS ||
        (count != 0 && ranks == nullptr)) {
        return false;
    }
    if (count != 0) {
        result.assign(ranks, ranks + count);
    }
    return true;
}

// 从 RankGraph 拷贝每个通信实例的规模列表。
bool CopyInstanceSizes(HcclComm comm, uint32_t layer, std::vector<uint32_t> &result)
{
    uint32_t *sizes = nullptr;
    uint32_t count = 0;
    if (HcclRankGraphGetInstSizeListByLayer(comm, layer, &sizes, &count) != HCCL_SUCCESS ||
        (count != 0 && sizes == nullptr)) {
        return false;
    }
    if (count != 0) {
        result.assign(sizes, sizes + count);
    }
    return true;
}

// 检查 Rank 列表是否由合法、唯一且连续的 Rank 组成。
bool IsUniqueContiguous(std::vector<uint32_t> ranks, uint32_t rankSize)
{
    if (ranks.empty() ||
        std::any_of(ranks.begin(), ranks.end(), [rankSize](uint32_t rank) { return rank >= rankSize; })) {
        return false;
    }
    std::sort(ranks.begin(), ranks.end());
    if (std::adjacent_find(ranks.begin(), ranks.end()) != ranks.end()) {
        return false;
    }
    for (uint32_t i = 1; i < ranks.size(); ++i) {
        if (ranks[i] != ranks[i - 1] + 1) {
            return false;
        }
    }
    return true;
}

// 读取并校验评测拓扑，计算本 Rank 在本地组中的槽位。
Groups BuildGroups(HcclComm comm, uint32_t myRank, uint32_t rankSize)
{
    Groups groups;
    std::vector<uint32_t> local;
    std::vector<uint32_t> global;
    std::vector<uint32_t> sizes;
    CommTopo intraTopo = COMM_TOPO_RESERVED;
    CommTopo interTopo = COMM_TOPO_RESERVED;
    uint32_t localSize = 0;
    uint32_t globalSize = 0;

    if (!CopyRanks(comm, LAYER_INTRA, local) ||
        !CopyRanks(comm, LAYER_INTER, global) ||
        !CopyInstanceSizes(comm, LAYER_INTRA, sizes) ||
        HcclRankGraphGetTopoTypeByLayer(comm, LAYER_INTRA, &intraTopo) != HCCL_SUCCESS ||
        HcclRankGraphGetTopoTypeByLayer(comm, LAYER_INTER, &interTopo) != HCCL_SUCCESS ||
        HcclRankGraphGetRankSizeByLayer(comm, LAYER_INTRA, &localSize) != HCCL_SUCCESS ||
        HcclRankGraphGetRankSizeByLayer(comm, LAYER_INTER, &globalSize) != HCCL_SUCCESS) {
        return groups;
    }

    std::sort(sizes.begin(), sizes.end());
    const bool shape16 = rankSize == 16 && sizes == std::vector<uint32_t>{8, 8};
    const bool shape12 = rankSize == 12 && sizes == std::vector<uint32_t>{4, 8};
    // 评测虚拟机对 2x8 和 8+4 拓扑的物理层 0 Mesh 都可能报告为 CUSTOM；
    // 因此必须结合 RankGraph 的分组关系和实例规模，而不能只依赖枚举值。
    const bool intraTypeOk = intraTopo == COMM_TOPO_1DMESH ||
        intraTopo == COMM_TOPO_CUSTOM;
    if ((!shape16 && !shape12) || !intraTypeOk || interTopo != COMM_TOPO_CLOS ||
        localSize != local.size() || globalSize != rankSize || global.size() != rankSize ||
        !IsUniqueContiguous(local, rankSize) || !IsUniqueContiguous(global, rankSize)) {
        HCCL_ERROR("Topology shape mismatch: rank=%u rankSize=%u intraType=%d interType=%d "
            "localSize=%u localRanks=%zu globalSize=%u globalRanks=%zu instCount=%zu "
            "shape16=%u shape12=%u",
            myRank, rankSize, static_cast<int>(intraTopo), static_cast<int>(interTopo),
            localSize, local.size(), globalSize, global.size(), sizes.size(),
            shape16 ? 1U : 0U, shape12 ? 1U : 0U);
        return groups;
    }

    std::sort(local.begin(), local.end());
    std::sort(global.begin(), global.end());
    const auto localIt = std::find(local.begin(), local.end(), myRank);
    if (localIt == local.end()) {
        return groups;
    }
    std::vector<uint32_t> remote;
    std::set_difference(global.begin(), global.end(), local.begin(), local.end(),
        std::back_inserter(remote));
    if (!IsUniqueContiguous(remote, rankSize) ||
        (shape16 && (local.size() != 8 || remote.size() != 8)) ||
        (shape12 && !((local.size() == 8 && remote.size() == 4) ||
                      (local.size() == 4 && remote.size() == 8)))) {
        HCCL_ERROR("Topology group mismatch: rank=%u rankSize=%u local=%zu remote=%zu",
            myRank, rankSize, local.size(), remote.size());
        return groups;
    }

    groups.valid = true;
    groups.localSlot = static_cast<uint32_t>(std::distance(local.begin(), localIt));
    groups.local = std::move(local);
    groups.remote = std::move(remote);
    return groups;
}

// 获取指定源 Rank 到目标 Rank 的第一条可用通信链路。
bool GetLink(HcclComm comm, uint32_t layer, uint32_t myRank, uint32_t remoteRank, CommLink &link)
{
    CommLink *links = nullptr;
    uint32_t count = 0;
    return HcclRankGraphGetLinks(comm, layer, myRank, remoteRank, &links, &count) == HCCL_SUCCESS &&
        links != nullptr && count != 0 && ((link = links[0]), true);
}

// 根据链路协议和拓扑层选择所需的通知资源数量。
uint32_t NotifyCount(CommProtocol protocol, uint32_t layer)
{
    return protocol == CommProtocol::COMM_PROTOCOL_HCCS || layer == LAYER_INTRA ?
        NOTIFY_NUM_SDMA : NOTIFY_NUM_RDMA;
}

// 初始化并申请一条 CCU 通道，同时记录远端 Rank、层和 DIE 信息。
HcclResult AcquireChannel(HcclComm comm, CommEngine engine, uint32_t myRank,
    uint32_t remoteRank, uint32_t layer, Channel &result)
{
    CommLink link {};
    CHK_PRT_RET(!GetLink(comm, layer, myRank, remoteRank, link),
        HCCL_ERROR("Missing link: layer=%u, src=%u, dst=%u", layer, myRank, remoteRank),
        HCCL_E_NOT_FOUND);

    EndpointAttrDieId dieId = layer;
    if (HcclRankGraphGetEndpointInfo(comm, myRank, &link.srcEndpointDesc,
        ENDPOINT_ATTR_DIE_ID, sizeof(dieId), &dieId) != HCCL_SUCCESS) {
        dieId = layer;
    }
    CommProtocol protocol = link.linkAttr.linkProtocol;
    if (protocol == CommProtocol::COMM_PROTOCOL_RESERVED) {
        protocol = layer == LAYER_INTRA ? CommProtocol::COMM_PROTOCOL_HCCS :
            CommProtocol::COMM_PROTOCOL_ROCE;
    }
    HcclChannelDesc desc;
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = remoteRank;
    desc.channelProtocol = protocol;
    desc.notifyNum = NotifyCount(protocol, layer);
    desc.localEndpoint = link.srcEndpointDesc;
    desc.remoteEndpoint = link.dstEndpointDesc;

    result.remoteRank = remoteRank;
    result.layer = layer;
    result.dieId = dieId;
    CHK_RET(HcclChannelAcquire(comm, engine, &desc, 1, &result.handle));
    return HCCL_SUCCESS;
}

// 为扁平路径选择一个能够连接所有远端 Rank 的完整拓扑层。
uint32_t SelectFlatLayer(HcclComm comm, uint32_t myRank, uint32_t rankSize)
{
    for (uint32_t layer : {LAYER_INTRA, LAYER_INTER}) {
        bool complete = true;
        for (uint32_t rank = 0; rank < rankSize && complete; ++rank) {
            if (rank != myRank) {
                CommLink link {};
                complete = GetLink(comm, layer, myRank, rank, link);
            }
        }
        if (complete) {
            return layer;
        }
    }
    return INVALID_VALUE_RANKID;
}

// 判断远端 Rank 是否需要通过 Gateway 的跨组链路访问。
bool IsGatewayCrossPeer(const Groups &groups, uint32_t remoteRank)
{
    const auto it = std::find(groups.remote.begin(), groups.remote.end(), remoteRank);
    if (it == groups.remote.end()) {
        return false;
    }
    const uint32_t remoteSlot = static_cast<uint32_t>(std::distance(groups.remote.begin(), it));
    if (groups.local.size() == 8 && groups.remote.size() == 8) {
        return (groups.localSlot ^ remoteSlot) <= 4;
    }
    return groups.local.size() == 8 ?
        remoteSlot != groups.localSlot % 4 :
        groups.localSlot != remoteSlot % 4;
}

// 在已申请的通道列表中查找指定远端 Rank 对应的通道。
const Channel *FindChannel(const std::vector<Channel> &channels, uint32_t remoteRank)
{
    const auto it = std::find_if(channels.begin(), channels.end(),
        [remoteRank](const Channel &channel) { return channel.remoteRank == remoteRank; });
    return it == channels.end() ? nullptr : &*it;
}

// 填充 Host 侧注册 CCU Kernel 所需的静态参数模板。
void FillAllGatherArg(const OpParam &param, uint32_t dataTypeSize,
    const std::vector<ChannelHandle> &channels, const std::vector<uint32_t> &ranks,
    uint32_t copyLocal, uint32_t kernelMode, AllGatherKernelArg &arg)
{
    arg.sendBuf = param.inputPtr;
    arg.recvBuf = param.outputPtr;
    arg.sendCount = param.count;
    arg.sliceOffset = 0;
    arg.sliceCount = param.count;
    arg.inputToken = 0;
    arg.outputToken = 0;
    arg.dataTypeSize = dataTypeSize;
    arg.myRank = param.myRank;
    arg.rankSize = param.rankSize;
    arg.copyLocalOutput = copyLocal;
    arg.kernelMode = kernelMode;
    arg.channelCount = static_cast<uint32_t>(channels.size());
    for (uint32_t i = 0; i < channels.size(); ++i) {
        arg.channels[i] = channels[i];
        arg.channelIndexToRank[i] = ranks[i];
    }
}

// 按流水分片数量计算当前分片的元素偏移和长度。
void FillPipelineSlice(uint64_t count, uint32_t chunk,
    uint64_t &offset, uint64_t &sliceCount)
{
    const uint64_t base = count / G2_PIPELINE_CHUNK_COUNT;
    const uint64_t remainder = count % G2_PIPELINE_CHUNK_COUNT;
    sliceCount = base + (chunk < remainder ? 1U : 0U);
    offset = chunk * base + std::min<uint64_t>(chunk, remainder);
}

// 将 Gateway 的逻辑转发段映射为实际元素偏移和传输长度。
void FillForwardTransfers(uint64_t sliceOffset, uint64_t sliceCount,
    const GatewayPlan &plan,
    uint64_t (&offsets)[G2_MAX_FORWARD_TRANSFERS],
    uint64_t (&counts)[G2_MAX_FORWARD_TRANSFERS])
{
    const uint64_t base = sliceCount / plan.forwardSegmentCount;
    const uint64_t remainder = sliceCount % plan.forwardSegmentCount;
    for (uint32_t transfer = 0;
        transfer < plan.forwardChannelIndices.size(); ++transfer) {
        const uint32_t begin = plan.forwardSegmentBegins[transfer];
        const uint32_t span = plan.forwardSegmentSpans[transfer];
        offsets[transfer] = sliceOffset + begin * base +
            std::min<uint64_t>(begin, remainder);
        const uint64_t remainderInSpan = remainder > begin ?
            std::min<uint64_t>(remainder - begin, span) : 0;
        counts[transfer] = span * base + remainderInSpan;
    }
}

// 为扁平路径注册一个 AllGather CCU Kernel。
HcclResult RegisterFlat(HcclComm comm, const OpParam &param, uint32_t dataTypeSize,
    AlgResourceCtx &context)
{
    AllGatherKernelArg arg {};
    FillAllGatherArg(param, dataTypeSize, context.flat.channels,
        context.flat.remoteRanks, 1, context.flat.kernelMode, arg);
    CcuInsHandle ins = 0;
    uint32_t count = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &ins, &count));
    CHK_PRT_RET(count != 1, HCCL_ERROR("Unexpected CCU instance count: %u", count), HCCL_E_INTERNAL);
    CHK_RET_CCU(HcommCcuKernelRegisterStart(ins));
    const void *args[] = {&arg};
    CHK_RET_CCU(HcommCcuKernelRegister(ins, context.flat.dieId, "CcuKernel",
        reinterpret_cast<const void *>(ops_hccl::CcuKernel), args, 1, &context.flat.kernel));
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(ins));
    return HCCL_SUCCESS;
}

// 为 Gateway 路径注册跨组、本地和转发三个 CCU Kernel。
HcclResult RegisterGateway(HcclComm comm, const OpParam &param,
    uint32_t dataTypeSize,
    AlgResourceCtx &context)
{
    GatewayPlan &plan = context.gateway;
    uint64_t templateOffset = 0;
    uint64_t templateCount = 0;
    FillPipelineSlice(param.count, 0, templateOffset, templateCount);
    AllGatherKernelArg cross {};
    FillAllGatherArg(param, dataTypeSize, plan.crossChannels,
        plan.crossRanks, 0, ALLGATHER_KERNEL_DIRECT_WRITE, cross);
    cross.sliceOffset = templateOffset;
    cross.sliceCount = templateCount;

    AllGatherKernelArg local {};
    FillAllGatherArg(param, dataTypeSize, plan.localChannels,
        plan.localRanks, 1, ALLGATHER_KERNEL_DIRECT_WRITE, local);
    local.sliceOffset = templateOffset;
    local.sliceCount = templateCount;

    G2ForwardKernelArg forward {};
    forward.recvBuf = param.outputPtr;
    forward.sendCount = param.count;
    forward.dataTypeSize = dataTypeSize;
    forward.forwardTransferCount =
        static_cast<uint32_t>(plan.forwardChannelIndices.size());
    forward.channelCount = static_cast<uint32_t>(plan.localChannels.size());
    std::copy(plan.localChannels.begin(), plan.localChannels.end(),
        forward.channels);
    FillForwardTransfers(templateOffset, templateCount, plan,
        forward.forwardSliceOffsets, forward.forwardSliceCounts);
    for (uint32_t transfer = 0;
        transfer < forward.forwardTransferCount; ++transfer) {
        forward.forwardChannelIndices[transfer] =
            plan.forwardChannelIndices[transfer];
        forward.forwardSourceRanks[transfer] =
            plan.forwardSourceRanks[transfer];
    }

    CcuInsHandle ins = 0;
    uint32_t count = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &ins, &count));
    CHK_PRT_RET(count != 1, HCCL_ERROR("Unexpected CCU instance count: %u", count), HCCL_E_INTERNAL);
    CHK_RET_CCU(HcommCcuKernelRegisterStart(ins));
    const void *crossArgs[] = {&cross};
    CHK_RET_CCU(HcommCcuKernelRegister(ins, plan.crossDieId, "CcuKernel",
        reinterpret_cast<const void *>(ops_hccl::CcuKernel),
        crossArgs, 1, &plan.crossKernel));
    const void *localArgs[] = {&local};
    CHK_RET_CCU(HcommCcuKernelRegister(ins, plan.localDieId,
        "CcuKernel",
        reinterpret_cast<const void *>(ops_hccl::CcuKernel),
        localArgs, 1, &plan.localKernel));
    const void *forwardArgs[] = {&forward};
    CHK_RET_CCU(HcommCcuKernelRegister(ins, plan.localDieId,
        "G2ForwardKernel",
        reinterpret_cast<const void *>(ops_hccl::G2ForwardKernel),
        forwardArgs, 1, &plan.forwardKernel));
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(ins));
    return HCCL_SUCCESS;
}

// 根据 12/16 Rank 的分组形状生成 Gateway 通道和转发表。
HcclResult CompileGateway(const Groups &groups,
    const std::vector<Channel> &channels,
    AlgResourceCtx &context)
{
    GatewayPlan &plan = context.gateway;
    const bool rank16 = groups.local.size() == 8 &&
        groups.remote.size() == 8;
    const bool rank12Large = groups.local.size() == 8 &&
        groups.remote.size() == 4;
    plan.forwardSegmentCount = rank16 ? 7U : 3U;
    for (const Channel &channel : channels) {
        std::vector<ChannelHandle> &target =
            channel.layer == LAYER_INTRA ? plan.localChannels : plan.crossChannels;
        std::vector<uint32_t> &ranks =
            channel.layer == LAYER_INTRA ? plan.localRanks : plan.crossRanks;
        target.push_back(channel.handle);
        ranks.push_back(channel.remoteRank);
        uint32_t &die = channel.layer == LAYER_INTRA ? plan.localDieId : plan.crossDieId;
        if (target.size() == 1) {
            die = channel.dieId;
        }
        CHK_PRT_RET(channel.dieId != die,
            HCCL_ERROR("G2 kernel channels span IO dies"), HCCL_E_INTERNAL);
    }
    const uint32_t expectedLocal =
        groups.local.size() == 8 ? 7U : 3U;
    const uint32_t expectedCross = rank16 ? 5U :
        rank12Large ? 3U : 6U;
    CHK_PRT_RET(plan.localChannels.size() != expectedLocal ||
            plan.crossChannels.size() != expectedCross,
        HCCL_ERROR("G2 channel shape is incomplete"), HCCL_E_INTERNAL);

    const auto addForward = [&](uint32_t destinationSlot,
        uint32_t sourceRank, uint32_t begin, uint32_t span) -> HcclResult {
        const Channel *channel = FindChannel(channels, groups.local[destinationSlot]);
        CHK_PRT_RET(channel == nullptr || channel->layer != LAYER_INTRA,
            HCCL_ERROR("G2 forwarding channel is missing"), HCCL_E_INTERNAL);
        const auto localIt = std::find(
            plan.localRanks.begin(), plan.localRanks.end(),
            channel->remoteRank);
        CHK_PRT_RET(localIt == plan.localRanks.end(),
            HCCL_ERROR("G2 forwarding channel index is missing"),
            HCCL_E_INTERNAL);
        plan.forwardChannelIndices.push_back(static_cast<uint32_t>(
            std::distance(plan.localRanks.begin(), localIt)));
        plan.forwardSourceRanks.push_back(sourceRank);
        plan.forwardSegmentBegins.push_back(begin);
        plan.forwardSegmentSpans.push_back(span);
        return HCCL_SUCCESS;
    };

    if (rank16) {
        // Rank-16 的跨组图为 5 度连接：三个缺失远端源被均衡分配到 7 条
        // 本地 Mesh 链路，其中 xor-1..3 各承担一个 3/7 分片，
        // xor-4..7 各承担四个 1/7 分片。
        for (uint32_t xorDistance = 1;
            xorDistance <= 3; ++xorDistance) {
            CHK_RET(addForward(groups.localSlot ^ xorDistance,
                groups.remote[groups.localSlot ^ 4], 0, 3));
        }
        for (uint32_t xorDistance = 4;
            xorDistance <= 7; ++xorDistance) {
            const uint32_t destinationSlot =
                groups.localSlot ^ xorDistance;
            for (uint32_t missingDistance = 5;
                missingDistance <= 7; ++missingDistance) {
                CHK_RET(addForward(destinationSlot,
                    groups.remote[groups.localSlot ^
                        (xorDistance ^ missingDistance)],
                    xorDistance - 1, 1));
            }
        }
    } else {
        for (uint32_t lane = 0; lane < 3; ++lane) {
            const uint32_t destinationSlot =
                groups.localSlot ^ (lane + 1);
            if (rank12Large) {
                CHK_RET(addForward(destinationSlot,
                    groups.remote[destinationSlot % 4], lane, 1));
            } else {
                CHK_RET(addForward(destinationSlot,
                    groups.remote[destinationSlot], lane, 1));
                CHK_RET(addForward(destinationSlot,
                    groups.remote[destinationSlot + 4], lane, 1));
            }
        }
    }
    const uint32_t expectedTransfers = rank16 ? 15U :
        rank12Large ? 3U : 6U;
    CHK_PRT_RET(plan.forwardChannelIndices.size() != expectedTransfers ||
            plan.forwardSourceRanks.size() != expectedTransfers ||
            plan.forwardSegmentBegins.size() != expectedTransfers ||
            plan.forwardSegmentSpans.size() != expectedTransfers ||
            expectedTransfers > G2_MAX_FORWARD_TRANSFERS,
        HCCL_ERROR("G2 forwarding shape is incomplete"), HCCL_E_INTERNAL);
    for (uint32_t transfer = 0; transfer < expectedTransfers; ++transfer) {
        CHK_PRT_RET(
            plan.forwardChannelIndices[transfer] >= plan.localChannels.size() ||
                plan.forwardSourceRanks[transfer] >= context.rankSize ||
                plan.forwardSegmentSpans[transfer] == 0 ||
                plan.forwardSegmentBegins[transfer] +
                    plan.forwardSegmentSpans[transfer] >
                    plan.forwardSegmentCount,
            HCCL_ERROR("G2 forwarding transfer is invalid"),
            HCCL_E_INTERNAL);
    }
    plan.valid = 1;
    return HCCL_SUCCESS;
}
} // 匿名命名空间

// 完成 AllGather 的参数校验、拓扑分析、资源申请和 Kernel 注册。
HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount,
    HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param {};
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    const auto typeIt = SIZE_TABLE.find(dataType);
    CHK_PRT_RET(typeIt == SIZE_TABLE.end(),
        HCCL_ERROR("Unsupported data type: %d", static_cast<int>(dataType)), HCCL_E_NOT_SUPPORT);
    const uint32_t dataTypeSize = typeIt->second;
    const uint64_t inputBytes = sendCount * dataTypeSize;
    // 小消息采用拉取，大消息在专用 Rank 形状上采用 Gateway，其余采用扁平路径。
    const Mode mode = inputBytes <= LARGE_THRESHOLD ? Mode::SMALL_PULL :
        ((param.rankSize == 16 || param.rankSize == 12) ?
            Mode::G2_GATEWAY : Mode::FLAT);

    // 将算法模式和消息规模编码到 tag，保证不同缓存上下文互不冲突。
    if (mode == Mode::G2_GATEWAY) {
        std::snprintf(param.tag, sizeof(param.tag), "ag_g2d5_%u_%llu",
            param.rankSize,
            static_cast<unsigned long long>(inputBytes));
    } else if (mode == Mode::SMALL_PULL) {
        std::snprintf(param.tag, sizeof(param.tag), "ag_pull_%u_%llu",
            param.rankSize, static_cast<unsigned long long>(inputBytes));
    } else {
        std::snprintf(param.tag, sizeof(param.tag), "ag_flat_%u_%llu",
            param.rankSize, static_cast<unsigned long long>(inputBytes));
    }

    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(
        commName, reinterpret_cast<void *>(&dfxInfo)));
    // 空输入无需申请资源；单 Rank 场景直接完成设备到设备拷贝。
    if (sendCount == 0) {
        return HCCL_SUCCESS;
    }
    if (param.rankSize == 1) {
        ACLCHECK(aclrtMemcpyAsync(recvBuf, inputBytes, sendBuf, inputBytes,
            ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
        return HCCL_SUCCESS;
    }

    // 所有路径均使用 CCU 引擎；Gateway 额外申请一个工作线程。
    const CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    const bool twoThread = mode == Mode::G2_GATEWAY;
    const uint32_t mainNotifyCount = twoThread ? 1U : 0U;
    CHK_RET(HcclThreadAcquireWithStream(comm, engine, stream,
        mainNotifyCount, &param.cpuThread));
    void *cached = nullptr;
    uint64_t cachedSize = 0;
    // 命中缓存时直接恢复已注册资源，跳过重复的拓扑分析和 Kernel 注册。
    if (HcclEngineCtxGet(comm, param.tag, engine, &cached, &cachedSize) == HCCL_SUCCESS) {
        param.resCtx = cached;
        param.ctxSize = cachedSize;
        return ops_hccl::ExecOp(param);
    }

    AlgResourceCtx context {};
    context.myRank = param.myRank;
    context.rankSize = param.rankSize;
    context.dataTypeSize = dataTypeSize;
    context.threads.push_back(param.cpuThread);
    const uint32_t workerCount = twoThread ? 1U : 0U;
    if (workerCount != 0) {
        std::vector<ThreadHandle> workers(workerCount, 0);
        CHK_RET(HcclThreadAcquire(comm, engine, workerCount,
            1, workers.data()));
        for (const ThreadHandle worker : workers) {
            CHK_PRT_RET(std::find(context.threads.begin(),
                    context.threads.end(), worker) != context.threads.end(),
                HCCL_ERROR("Independent CCU worker thread unavailable"),
                HCCL_E_NOT_SUPPORT);
            context.threads.push_back(worker);
        }
    }

    // 首次执行需要根据所选模式构建扁平层或 Gateway 分组。
    Groups groups;
    uint32_t flatLayer = INVALID_VALUE_RANKID;
    if (mode != Mode::G2_GATEWAY) {
        flatLayer = SelectFlatLayer(comm, param.myRank, param.rankSize);
        CHK_PRT_RET(flatLayer == INVALID_VALUE_RANKID,
            HCCL_ERROR("Flat path has no complete network layer"), HCCL_E_NOT_FOUND);
    } else {
        groups = BuildGroups(comm, param.myRank, param.rankSize);
        CHK_PRT_RET(!groups.valid,
            HCCL_ERROR("Specialized topology does not match"), HCCL_E_NOT_SUPPORT);
    }

    // 为算法实际需要的远端 Rank 建立通信通道。
    std::vector<Channel> channels;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        uint32_t layer = flatLayer;
        bool required = mode != Mode::G2_GATEWAY;
        if (mode == Mode::G2_GATEWAY) {
            const bool local = std::find(groups.local.begin(), groups.local.end(), remoteRank) != groups.local.end();
            required = local || IsGatewayCrossPeer(groups, remoteRank);
            layer = local ? LAYER_INTRA : LAYER_INTER;
        }
        if (!required) {
            continue;
        }
        Channel channel;
        CHK_RET(AcquireChannel(comm, engine, param.myRank, remoteRank, layer, channel));
        channels.push_back(channel);
    }

    // 按模式整理通道并注册对应 Kernel。
    if (mode != Mode::G2_GATEWAY) {
        CHK_PRT_RET(channels.empty(), HCCL_ERROR("Flat channel set is empty"), HCCL_E_INTERNAL);
        context.flat.dieId = channels.front().dieId;
        for (const Channel &channel : channels) {
            CHK_PRT_RET(channel.dieId != context.flat.dieId,
                HCCL_ERROR("Flat channels span IO dies"), HCCL_E_INTERNAL);
            context.flat.channels.push_back(channel.handle);
            context.flat.remoteRanks.push_back(channel.remoteRank);
        }
        context.flat.kernelMode = mode == Mode::SMALL_PULL ?
            ALLGATHER_KERNEL_SMALL_PULL : ALLGATHER_KERNEL_DIRECT_WRITE;
        CHK_RET(RegisterFlat(comm, param, dataTypeSize, context));
    } else {
        CHK_RET(CompileGateway(groups, channels, context));
        CHK_RET(RegisterGateway(
            comm, param, dataTypeSize, context));
    }

    // 缓存编译结果，使后续相同 tag 的调用直接进入 ExecOp。
    std::vector<char> serialized = context.Serialize();
    param.ctxSize = serialized.size();
    CHK_RET(HcclEngineCtxCreate(comm, param.tag, engine, param.ctxSize, &param.resCtx));
    CHK_RET(HcclEngineCtxCopy(comm, engine, param.tag,
        serialized.data(), serialized.size(), 0));
    return ops_hccl::ExecOp(param);
}
