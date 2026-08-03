/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <tuple>
#include <utility>
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

constexpr uint32_t NET_LAYER_ZERO = 0;
constexpr uint32_t NET_LAYER_ONE = 1;
constexpr uint64_t FNV_OFFSET_BASIS = 1469598103934665603ULL;
constexpr uint64_t FNV_PRIME = 1099511628211ULL;
constexpr char CONTEXT_TAG[] = "hccl_custom_allgather";

struct EndpointKey {
    uint32_t protocol;
    uint32_t addressType;
    std::array<uint8_t, 36> address;
    uint32_t locationType;
    std::array<uint8_t, 60> location;
    std::array<uint8_t, 52> extension;
};

struct LinkKey {
    EndpointKey firstEndpoint;
    EndpointKey secondEndpoint;
    uint32_t hop;
};

enum class TopologyMetadataSource : uint32_t {
    NONE = 0,
    ENDPOINT_SERVER_INDEX = 1,
    RANK_GRAPH_LAYER_ZERO = 2,
};

struct TopologyLayout {
    TopologyKind kind{TopologyKind::GENERIC};
    TopologyMetadataSource metadataSource{TopologyMetadataSource::NONE};
    bool hasLayerZero{false};
    bool hasLayerOne{false};
    bool endpointTopologyValid{true};
    bool selectedEndpointSeen{false};
    bool selectedEndpointsAllDevice{true};
    bool selectedServerIndicesAllMissing{true};
    bool missingServerIndexSeen{false};
    bool endpointServerIndexConflict{false};
    std::vector<uint32_t> layers;
    std::vector<uint32_t> localRanks;
    std::vector<uint32_t> layerOneRanks;
    std::vector<uint32_t> remoteRanks;
    std::vector<uint32_t> firstServerRanks;
    std::vector<uint32_t> secondServerRanks;
    std::vector<uint32_t> largeServerRanks;
    std::vector<uint32_t> smallServerRanks;
    std::vector<uint32_t> instanceSizes;
    std::vector<uint32_t> rankServerIndices;
    uint64_t signature{FNV_OFFSET_BASIS};
};

struct ServerRankGroup {
    uint32_t serverIndex;
    std::vector<uint32_t> ranks;
};

struct PeerChannelInfo {
    uint32_t remoteRank;
    uint32_t layer;
    uint32_t localIoDie;
    CommProtocol protocol;
    ChannelHandle channel;
};

struct ChannelPeerGroup {
    uint32_t layer;
    uint32_t localIoDie;
    std::vector<uint32_t> peerRanks;
};

struct PendingVariant {
    KernelVariantDesc desc;
};

/**
 * @brief 检查两个无符号整数相乘是否溢出并返回乘积
 * @param left 乘法左操作数
 * @param right 乘法右操作数
 * @param result 接收乘法结果的变量
 * @return 未溢出返回 true，发生溢出返回 false
 */
bool CheckedMultiply(uint64_t left, uint64_t right, uint64_t &result)
{
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

/**
 * @brief 校验一段内存地址范围的末地址计算不会发生无符号溢出
 * @param address 内存区域起始地址
 * @param size 内存区域字节长度
 * @return 范围有效返回 HCCL_SUCCESS，范围非法返回对应错误码
 */
HcclResult CheckAddressRange(const void *address, uint64_t size)
{
    if (size == 0) {
        return HCCL_SUCCESS;
    }
    uint64_t base = reinterpret_cast<uint64_t>(address);
    if (base > std::numeric_limits<uint64_t>::max() - (size - 1)) {
        HCCL_ERROR("[HcclAllGather] address range overflow, base[%p], size[%llu]", address,
            static_cast<unsigned long long>(size));
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 判断给定 rank 是否存在于已排序或未排序的 rank 列表中
 * @param ranks 待查询的 rank 列表
 * @param rank 待查找的 rank 编号
 * @return 找到返回 true，未找到返回 false
 */
bool ContainsRank(const std::vector<uint32_t> &ranks, uint32_t rank)
{
    return std::find(ranks.begin(), ranks.end(), rank) != ranks.end();
}

/**
 * @brief 查找 rank 在稳定排序列表中的位置
 * @param ranks 待查询的稳定排序 rank 列表
 * @param rank 待查找的 rank 编号
 * @param index 接收 rank 在列表中的位置
 * @return 找到返回 true，未找到返回 false
 */
bool FindRankIndex(const std::vector<uint32_t> &ranks, uint32_t rank, uint32_t &index)
{
    auto iter = std::find(ranks.begin(), ranks.end(), rank);
    if (iter == ranks.end()) {
        return false;
    }
    index = static_cast<uint32_t>(std::distance(ranks.begin(), iter));
    return true;
}

/**
 * @brief 从全集中构造不属于本地 layer-0 instance 的 rank 补集
 * @param rankSize 通信域 rank 总数
 * @param localRanks 当前 rank 所在 layer-0 instance 的 rank 列表
 * @return 按全局 rank 升序排列的补集
 */
std::vector<uint32_t> BuildRankComplement(uint32_t rankSize, const std::vector<uint32_t> &localRanks)
{
    std::vector<uint32_t> complement;
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        if (!ContainsRank(localRanks, rank)) {
            complement.push_back(rank);
        }
    }
    return complement;
}

/**
 * @brief 判断已排序 rank 列表是否完整覆盖从零开始的通信域
 * @param ranks 待校验的已排序 rank 列表
 * @param rankSize 通信域 rank 总数
 * @return 列表无缺失且无额外 rank 返回 true，否则返回 false
 */
bool CoversFullRankDomain(const std::vector<uint32_t> &ranks, uint32_t rankSize)
{
    if (ranks.size() != rankSize) {
        return false;
    }
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        if (ranks[rank] != rank) {
            return false;
        }
    }
    return true;
}

/**
 * @brief 在全部选中 Endpoint 缺失 Server 索引时从 RankGraph layer-0 实例构造规范化 2×8 分组
 * @param rankSize 通信域 rank 总数
 * @param layout 已解析的 Endpoint 状态与 RankGraph 布局
 * @param firstGroup 接收 2×8 字典序较小组
 * @param secondGroup 接收 2×8 字典序较大组
 * @return 满足严格回退条件并成功构造分组返回 true，否则返回 false
 */
bool BuildLayerZeroFallbackGroups(uint32_t rankSize, const TopologyLayout &layout,
    std::vector<uint32_t> &firstGroup, std::vector<uint32_t> &secondGroup)
{
    firstGroup.clear();
    secondGroup.clear();
    if (!layout.selectedEndpointSeen || !layout.selectedEndpointsAllDevice
        || !layout.selectedServerIndicesAllMissing || !layout.missingServerIndexSeen
        || layout.endpointServerIndexConflict || !layout.hasLayerZero || !layout.hasLayerOne
        || !CoversFullRankDomain(layout.layerOneRanks, rankSize)) {
        return false;
    }

    const std::vector<uint32_t> twoFullInstances{8, 8};
    if (rankSize == 16 && layout.instanceSizes == twoFullInstances && layout.localRanks.size() == 8
        && layout.remoteRanks.size() == 8) {
        firstGroup = layout.localRanks;
        secondGroup = layout.remoteRanks;
        if (std::lexicographical_compare(secondGroup.begin(), secondGroup.end(), firstGroup.begin(),
                firstGroup.end())) {
            std::swap(firstGroup, secondGroup);
        }
        return true;
    }

    return false;
}

/**
 * @brief 将一个整数按字节写入 FNV-1a 拓扑签名
 * @param hash 当前签名值
 * @param value 待写入签名的整数
 * @return 无返回值
 */
void HashValue(uint64_t &hash, uint64_t value)
{
    for (uint32_t byteIndex = 0; byteIndex < sizeof(value); ++byteIndex) {
        hash ^= static_cast<uint8_t>((value >> (byteIndex * 8)) & 0xFFU);
        hash *= FNV_PRIME;
    }
}

/**
 * @brief 将稳定排序的 rank 列表写入 FNV-1a 拓扑签名
 * @param hash 当前签名值
 * @param ranks 待写入签名的 rank 列表
 * @return 无返回值
 */
void HashRankList(uint64_t &hash, const std::vector<uint32_t> &ranks)
{
    HashValue(hash, ranks.size());
    for (uint32_t rank : ranks) {
        HashValue(hash, rank);
    }
}

/**
 * @brief 把 EndpointDesc 中有定义的字段归一化为无方向比较键
 * @param endpoint 待归一化的 Endpoint 描述
 * @param key 接收可稳定比较的 Endpoint 键
 * @return 成功返回 HCCL_SUCCESS，安全内存复制失败返回对应错误码
 */
HcclResult BuildEndpointKey(const EndpointDesc &endpoint, EndpointKey &key)
{
    key = EndpointKey{};
    key.protocol = static_cast<uint32_t>(endpoint.protocol);
    key.addressType = static_cast<uint32_t>(endpoint.commAddr.type);
    errno_t addressRet
        = memcpy_s(key.address.data(), key.address.size(), endpoint.commAddr.raws, sizeof(endpoint.commAddr.raws));
    if (addressRet != EOK) {
        HCCL_ERROR("[HcclAllGather] copy Endpoint address key failed, ret[%d]", addressRet);
        return HCCL_E_INTERNAL;
    }
    key.locationType = static_cast<uint32_t>(endpoint.loc.locType);
    errno_t locationRet
        = memcpy_s(key.location.data(), key.location.size(), endpoint.loc.raws, sizeof(endpoint.loc.raws));
    if (locationRet != EOK) {
        HCCL_ERROR("[HcclAllGather] copy Endpoint location key failed, ret[%d]", locationRet);
        return HCCL_E_INTERNAL;
    }
    errno_t extensionRet
        = memcpy_s(key.extension.data(), key.extension.size(), endpoint.raws, sizeof(endpoint.raws));
    if (extensionRet != EOK) {
        HCCL_ERROR("[HcclAllGather] copy Endpoint extension key failed, ret[%d]", extensionRet);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 按显式 Endpoint 字段比较两个归一化键
 * @param left 左侧 Endpoint 键
 * @param right 右侧 Endpoint 键
 * @return 左侧严格小于右侧返回 true，否则返回 false
 */
bool EndpointKeyLess(const EndpointKey &left, const EndpointKey &right)
{
    return std::tie(left.protocol, left.addressType, left.address, left.locationType, left.location, left.extension)
        < std::tie(right.protocol, right.addressType, right.address, right.locationType, right.location,
            right.extension);
}

/**
 * @brief 为一条 CommLink 构造两端方向无关的稳定比较键
 * @param link 待归一化的通信链路
 * @param key 接收包含有序 Endpoint 对和 hop 的链路键
 * @return 成功返回 HCCL_SUCCESS，Endpoint 键构造失败返回对应错误码
 */
HcclResult BuildLinkKey(const CommLink &link, LinkKey &key)
{
    EndpointKey source{};
    EndpointKey destination{};
    CHK_RET(BuildEndpointKey(link.srcEndpointDesc, source));
    CHK_RET(BuildEndpointKey(link.dstEndpointDesc, destination));
    if (EndpointKeyLess(destination, source)) {
        std::swap(source, destination);
    }
    key = LinkKey{source, destination, link.linkAttr.hop};
    return HCCL_SUCCESS;
}

/**
 * @brief 比较两个方向无关的链路键
 * @param left 左侧链路键
 * @param right 右侧链路键
 * @return 左侧严格小于右侧返回 true，否则返回 false
 */
bool LinkKeyLess(const LinkKey &left, const LinkKey &right)
{
    if (EndpointKeyLess(left.firstEndpoint, right.firstEndpoint)) {
        return true;
    }
    if (EndpointKeyLess(right.firstEndpoint, left.firstEndpoint)) {
        return false;
    }
    if (EndpointKeyLess(left.secondEndpoint, right.secondEndpoint)) {
        return true;
    }
    if (EndpointKeyLess(right.secondEndpoint, left.secondEndpoint)) {
        return false;
    }
    return left.hop < right.hop;
}

/**
 * @brief 按协议优先级和方向无关 Endpoint 键选择唯一稳定链路
 * @param comm 当前通信域
 * @param myRank 当前全局 rank 编号
 * @param remoteRank 对端全局 rank 编号
 * @param layer 目标网络层
 * @param selectedLink 接收选中链路的副本
 * @param localIoDie 接收选中链路本地 Endpoint 所属 IO Die
 * @return 成功返回 HCCL_SUCCESS，链路缺失或属性查询失败返回对应错误码
 */
HcclResult SelectDeterministicLink(HcclComm comm, uint32_t myRank, uint32_t remoteRank, uint32_t layer,
    CommLink &selectedLink, uint32_t &localIoDie)
{
    CommLink *links = nullptr;
    uint32_t linkCount = 0;
    CHK_RET(HcclRankGraphGetLinks(comm, layer, myRank, remoteRank, &links, &linkCount));
    if (links == nullptr || linkCount == 0) {
        HCCL_ERROR("[HcclAllGather] no link found, myRank[%u], remoteRank[%u], layer[%u]", myRank, remoteRank,
            layer);
        return HCCL_E_NOT_FOUND;
    }

    const CommProtocol protocolPriority[] = {COMM_PROTOCOL_UBC_CTP, COMM_PROTOCOL_UBC_TP};
    bool found = false;
    LinkKey bestKey{};
    for (CommProtocol protocol : protocolPriority) {
        for (uint32_t linkIndex = 0; linkIndex < linkCount; ++linkIndex) {
            const CommLink &candidate = links[linkIndex];
            if (candidate.linkAttr.linkProtocol != protocol) {
                continue;
            }
            LinkKey candidateKey{};
            CHK_RET(BuildLinkKey(candidate, candidateKey));
            if (!found || LinkKeyLess(candidateKey, bestKey)) {
                selectedLink = candidate;
                bestKey = candidateKey;
                found = true;
            }
        }
        if (found) {
            break;
        }
    }
    if (!found) {
        HCCL_ERROR("[HcclAllGather] supported protocol not found, myRank[%u], remoteRank[%u], layer[%u]", myRank,
            remoteRank, layer);
        return HCCL_E_NOT_FOUND;
    }

    EndpointAttrDieId dieId = 0;
    CHK_RET(HcclRankGraphGetEndpointInfo(comm, myRank, &selectedLink.srcEndpointDesc, ENDPOINT_ATTR_DIE_ID,
        sizeof(dieId), &dieId));
    if (dieId >= MAX_IO_DIE_COUNT) {
        HCCL_ERROR("[HcclAllGather] invalid local IO Die, myRank[%u], remoteRank[%u], dieId[%u]", myRank,
            remoteRank, dieId);
        return HCCL_E_NOT_SUPPORT;
    }
    localIoDie = dieId;
    return HCCL_SUCCESS;
}

/**
 * @brief 读取 RankGraph 层次、当前 layer-0 instance 和全量 instance 大小
 * @param comm 当前通信域
 * @param myRank 当前全局 rank 编号
 * @param rankSize 当前通信域 rank 数量
 * @param layout 接收 RankGraph 原始拓扑层次、当前实例、layer-1 全域和全量实例大小
 * @return 成功返回 HCCL_SUCCESS，RankGraph 内容非法时返回对应错误码
 */
HcclResult BuildTopologyLayout(HcclComm comm, uint32_t myRank, uint32_t rankSize, TopologyLayout &layout)
{
    layout.rankServerIndices.assign(rankSize, UINT32_MAX);
    uint32_t *layerData = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerData, &layerCount));
    if (layerCount == 0 || layerData == nullptr) {
        HCCL_ERROR("[HcclAllGather] RankGraph returned empty layer list");
        return HCCL_E_INTERNAL;
    }
    layout.layers.assign(layerData, layerData + layerCount);
    std::sort(layout.layers.begin(), layout.layers.end());
    if (std::adjacent_find(layout.layers.begin(), layout.layers.end()) != layout.layers.end()) {
        HCCL_ERROR("[HcclAllGather] RankGraph returned duplicate network layer");
        return HCCL_E_INTERNAL;
    }
    for (uint32_t layer : layout.layers) {
        if (layer == NET_LAYER_ZERO) {
            layout.hasLayerZero = true;
        } else if (layer == NET_LAYER_ONE) {
            layout.hasLayerOne = true;
        } else {
            HCCL_ERROR("[HcclAllGather] unsupported network layer[%u]", layer);
            return HCCL_E_NOT_SUPPORT;
        }
    }

    if (layout.hasLayerZero) {
        uint32_t *rankData = nullptr;
        uint32_t localRankCount = 0;
        CHK_RET(HcclRankGraphGetRanksByLayer(comm, NET_LAYER_ZERO, &rankData, &localRankCount));
        if (localRankCount == 0 || rankData == nullptr) {
            HCCL_ERROR("[HcclAllGather] layer-0 instance is empty");
            return HCCL_E_INTERNAL;
        }
        layout.localRanks.assign(rankData, rankData + localRankCount);
        std::sort(layout.localRanks.begin(), layout.localRanks.end());
        if (std::adjacent_find(layout.localRanks.begin(), layout.localRanks.end()) != layout.localRanks.end()
            || std::any_of(layout.localRanks.begin(), layout.localRanks.end(),
                [rankSize](uint32_t rank) { return rank >= rankSize; })
            || !ContainsRank(layout.localRanks, myRank)) {
            HCCL_ERROR("[HcclAllGather] invalid layer-0 rank instance, myRank[%u]", myRank);
            return HCCL_E_INTERNAL;
        }

        uint32_t *instanceSizeData = nullptr;
        uint32_t instanceCount = 0;
        CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, NET_LAYER_ZERO, &instanceSizeData, &instanceCount));
        if (instanceCount == 0 || instanceSizeData == nullptr) {
            HCCL_ERROR("[HcclAllGather] layer-0 instance size list is empty");
            return HCCL_E_INTERNAL;
        }
        layout.instanceSizes.assign(instanceSizeData, instanceSizeData + instanceCount);
        std::sort(layout.instanceSizes.begin(), layout.instanceSizes.end());
    } else {
        layout.localRanks.push_back(myRank);
    }

    if (layout.hasLayerOne) {
        uint32_t *rankData = nullptr;
        uint32_t layerOneRankCount = 0;
        CHK_RET(HcclRankGraphGetRanksByLayer(comm, NET_LAYER_ONE, &rankData, &layerOneRankCount));
        if (layerOneRankCount == 0 || rankData == nullptr) {
            HCCL_ERROR("[HcclAllGather] layer-1 instance is empty");
            return HCCL_E_INTERNAL;
        }
        layout.layerOneRanks.assign(rankData, rankData + layerOneRankCount);
        std::sort(layout.layerOneRanks.begin(), layout.layerOneRanks.end());
        if (std::adjacent_find(layout.layerOneRanks.begin(), layout.layerOneRanks.end())
                != layout.layerOneRanks.end()
            || std::any_of(layout.layerOneRanks.begin(), layout.layerOneRanks.end(),
                [rankSize](uint32_t rank) { return rank >= rankSize; })
            || !ContainsRank(layout.layerOneRanks, myRank)) {
            HCCL_ERROR("[HcclAllGather] invalid layer-1 rank instance, myRank[%u]", myRank);
            return HCCL_E_INTERNAL;
        }
    }
    layout.remoteRanks = BuildRankComplement(rankSize, layout.localRanks);
    return HCCL_SUCCESS;
}

/**
 * @brief 校验 Endpoint 的设备位置并记录对应 rank 的物理 Server 索引
 * @param rank Endpoint 所属的全局 rank 编号
 * @param endpoint 待校验和读取的 Endpoint 描述
 * @param layout 接收 rank 到 Server 索引的唯一映射
 * @return Endpoint 有效且映射无冲突返回 true，否则返回 false
 */
bool RecordEndpointServerIndex(uint32_t rank, const EndpointDesc &endpoint, TopologyLayout &layout)
{
    layout.selectedEndpointSeen = true;
    if (rank >= layout.rankServerIndices.size()) {
        layout.selectedEndpointsAllDevice = false;
        layout.selectedServerIndicesAllMissing = false;
        HCCL_WARNING("[HcclAllGather] endpoint rank exceeds topology map, rank[%u], mapSize[%zu]", rank,
            layout.rankServerIndices.size());
        return false;
    }
    if (endpoint.loc.locType != ENDPOINT_LOC_TYPE_DEVICE) {
        layout.selectedEndpointsAllDevice = false;
        layout.selectedServerIndicesAllMissing = false;
        HCCL_WARNING("[HcclAllGather] non-device endpoint disables formal topology, rank[%u], locType[%d]", rank,
            endpoint.loc.locType);
        return false;
    }
    uint32_t serverIndex = endpoint.loc.device.serverIdx;
    if (serverIndex == UINT32_MAX) {
        layout.missingServerIndexSeen = true;
        HCCL_WARNING("[HcclAllGather] invalid endpoint Server index disables formal topology, rank[%u]", rank);
        return false;
    }
    layout.selectedServerIndicesAllMissing = false;
    uint32_t &recordedServerIndex = layout.rankServerIndices[rank];
    if (recordedServerIndex == UINT32_MAX) {
        recordedServerIndex = serverIndex;
        return true;
    }
    if (recordedServerIndex != serverIndex) {
        layout.endpointServerIndexConflict = true;
        HCCL_WARNING("[HcclAllGather] conflicting endpoint Server index disables formal topology, rank[%u], "
                     "recorded[%u], current[%u]",
            rank, recordedServerIndex, serverIndex);
        return false;
    }
    return true;
}

/**
 * @brief 为每个远端 rank 仅申请一条确定性 CCU Channel
 * @param comm 当前通信域
 * @param myRank 当前全局 rank 编号
 * @param rankSize 当前通信域 rank 数量
 * @param layout 已解析的 RankGraph 布局
 * @param peerChannels 接收按 remoteRank 升序排列的 Channel 元数据
 * @return 成功返回 HCCL_SUCCESS，选链或 Channel 申请失败返回对应错误码
 */
HcclResult BuildPeerChannels(HcclComm comm, uint32_t myRank, uint32_t rankSize, TopologyLayout &layout,
    std::vector<PeerChannelInfo> &peerChannels)
{
    peerChannels.reserve(rankSize - 1);
    for (uint32_t remoteRank = 0; remoteRank < rankSize; ++remoteRank) {
        if (remoteRank == myRank) {
            continue;
        }
        uint32_t layer = layout.hasLayerZero && ContainsRank(layout.localRanks, remoteRank) ? NET_LAYER_ZERO
                                                                                           : NET_LAYER_ONE;
        if ((layer == NET_LAYER_ZERO && !layout.hasLayerZero) || (layer == NET_LAYER_ONE && !layout.hasLayerOne)) {
            HCCL_ERROR("[HcclAllGather] required network layer is absent, remoteRank[%u], layer[%u]", remoteRank,
                layer);
            return HCCL_E_NOT_FOUND;
        }

        CommLink selectedLink{};
        uint32_t localIoDie = 0;
        CHK_RET(SelectDeterministicLink(comm, myRank, remoteRank, layer, selectedLink, localIoDie));
        bool localEndpointValid = RecordEndpointServerIndex(myRank, selectedLink.srcEndpointDesc, layout);
        bool remoteEndpointValid = RecordEndpointServerIndex(remoteRank, selectedLink.dstEndpointDesc, layout);
        if (!localEndpointValid || !remoteEndpointValid) {
            layout.endpointTopologyValid = false;
        }


        HcclChannelDesc channelDesc;
        CHK_RET(HcclChannelDescInit(&channelDesc, 1));
        channelDesc.remoteRank = remoteRank;
        channelDesc.notifyNum = CHANNEL_NOTIFY_NUM;
        channelDesc.channelProtocol = selectedLink.linkAttr.linkProtocol;
        channelDesc.localEndpoint = selectedLink.srcEndpointDesc;
        channelDesc.remoteEndpoint = selectedLink.dstEndpointDesc;

        ChannelHandle channel = 0;
        CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_CCU, &channelDesc, 1, &channel));
        if (channel == 0) {
            HCCL_ERROR("[HcclAllGather] acquired empty channel, remoteRank[%u]", remoteRank);
            return HCCL_E_INTERNAL;
        }
        for (const PeerChannelInfo &existing : peerChannels) {
            if (existing.remoteRank == remoteRank || existing.channel == channel) {
                HCCL_ERROR("[HcclAllGather] duplicate peer or channel, remoteRank[%u]", remoteRank);
                return HCCL_E_INTERNAL;
            }
        }
        peerChannels.push_back(PeerChannelInfo{remoteRank, layer, localIoDie, selectedLink.linkAttr.linkProtocol,
            channel});
    }
    if (peerChannels.size() != rankSize - 1) {
        HCCL_ERROR("[HcclAllGather] channel count mismatch, actual[%zu], expected[%u]", peerChannels.size(),
            rankSize - 1);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 在 Channel 建立后按 RankGraph 层级与实例结构识别正式拓扑并生成稳定签名
 * @param myRank 当前全局 rank 编号
 * @param rankSize 当前通信域 rank 数量
 * @param peerChannels 已按 remoteRank 升序建立的全量 Channel 元数据
 * @param layout 接收正式拓扑分组、拓扑类型和最终静态签名
 * @return 拓扑满足或不满足正式签名均返回 HCCL_SUCCESS
 */
HcclResult FinalizeTopologyLayout(uint32_t myRank, uint32_t rankSize,
    const std::vector<PeerChannelInfo> &peerChannels, TopologyLayout &layout)
{
    layout.kind = TopologyKind::GENERIC;
    layout.metadataSource = TopologyMetadataSource::NONE;
    layout.firstServerRanks.clear();
    layout.secondServerRanks.clear();
    layout.largeServerRanks.clear();
    layout.smallServerRanks.clear();

    std::vector<std::pair<uint32_t, uint32_t>> serverRankPairs;
    serverRankPairs.reserve(rankSize);
    if (layout.rankServerIndices.size() != rankSize) {
        layout.endpointTopologyValid = false;
    }
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        uint32_t serverIndex
            = rank < layout.rankServerIndices.size() ? layout.rankServerIndices[rank] : UINT32_MAX;
        if (serverIndex == UINT32_MAX) {
            layout.endpointTopologyValid = false;
        }
        serverRankPairs.emplace_back(serverIndex, rank);
    }
    std::sort(serverRankPairs.begin(), serverRankPairs.end());

    std::vector<ServerRankGroup> serverGroups;
    for (const auto &serverRankPair : serverRankPairs) {
        if (serverGroups.empty() || serverGroups.back().serverIndex != serverRankPair.first) {
            serverGroups.push_back(ServerRankGroup{serverRankPair.first, {serverRankPair.second}});
        } else {
            serverGroups.back().ranks.push_back(serverRankPair.second);
        }
    }

    std::vector<uint32_t> physicalInstanceSizes;
    physicalInstanceSizes.reserve(serverGroups.size());
    for (const ServerRankGroup &serverGroup : serverGroups) {
        physicalInstanceSizes.push_back(static_cast<uint32_t>(serverGroup.ranks.size()));
    }
    std::sort(physicalInstanceSizes.begin(), physicalInstanceSizes.end());

    bool layerZeroMetadataValid = true;
    if (layout.hasLayerZero) {
        if (physicalInstanceSizes != layout.instanceSizes || myRank >= layout.rankServerIndices.size()) {
            layerZeroMetadataValid = false;
        } else {
            uint32_t localServerIndex = layout.rankServerIndices[myRank];
            auto localServerIter = std::find_if(serverGroups.begin(), serverGroups.end(),
                [localServerIndex](const ServerRankGroup &serverGroup) {
                    return serverGroup.serverIndex == localServerIndex;
                });
            if (localServerIter == serverGroups.end() || localServerIter->ranks != layout.localRanks) {
                layerZeroMetadataValid = false;
            }
        }
    } else if (!layout.instanceSizes.empty() || layout.localRanks != std::vector<uint32_t>{myRank}) {
        layerZeroMetadataValid = false;
    }

    const std::vector<uint32_t> fourSingletonServers{1, 1, 1, 1};
    const std::vector<uint32_t> twoFullServers{8, 8};
    const std::vector<uint32_t> eightPlusFourServers{4, 8};
    bool formalMetadataValid = layout.endpointTopologyValid && layerZeroMetadataValid && layout.hasLayerOne;
    if (formalMetadataValid && rankSize == 4 && physicalInstanceSizes == fourSingletonServers) {
        layout.kind = TopologyKind::FOUR_BY_ONE;
        layout.metadataSource = TopologyMetadataSource::ENDPOINT_SERVER_INDEX;
    } else if (formalMetadataValid && rankSize == 16 && layout.hasLayerZero && serverGroups.size() == 2
        && physicalInstanceSizes == twoFullServers) {
        layout.kind = TopologyKind::TWO_BY_EIGHT;
        layout.metadataSource = TopologyMetadataSource::ENDPOINT_SERVER_INDEX;
        layout.firstServerRanks = serverGroups[0].ranks;
        layout.secondServerRanks = serverGroups[1].ranks;
    } else if (formalMetadataValid && rankSize == 12 && layout.hasLayerZero && serverGroups.size() == 2
        && physicalInstanceSizes == eightPlusFourServers) {
        layout.kind = TopologyKind::EIGHT_PLUS_FOUR;
        layout.metadataSource = TopologyMetadataSource::ENDPOINT_SERVER_INDEX;
        if (serverGroups[0].ranks.size() == 8) {
            layout.largeServerRanks = serverGroups[0].ranks;
            layout.smallServerRanks = serverGroups[1].ranks;
        } else {
            layout.largeServerRanks = serverGroups[1].ranks;
            layout.smallServerRanks = serverGroups[0].ranks;
        }
    }

    std::vector<uint32_t> fallbackFirstGroup;
    std::vector<uint32_t> fallbackSecondGroup;
    bool layerZeroFallbackValid
        = BuildLayerZeroFallbackGroups(rankSize, layout, fallbackFirstGroup, fallbackSecondGroup);
    if (layout.kind == TopologyKind::GENERIC && layerZeroFallbackValid && rankSize == 16) {
        layout.kind = TopologyKind::TWO_BY_EIGHT;
        layout.metadataSource = TopologyMetadataSource::RANK_GRAPH_LAYER_ZERO;
        layout.firstServerRanks = fallbackFirstGroup;
        layout.secondServerRanks = fallbackSecondGroup;
    }

    std::vector<std::vector<uint32_t>> normalizedRankGroups;
    if (layout.kind == TopologyKind::FOUR_BY_ONE) {
        for (const ServerRankGroup &serverGroup : serverGroups) {
            normalizedRankGroups.push_back(serverGroup.ranks);
        }
        std::sort(normalizedRankGroups.begin(), normalizedRankGroups.end());
    } else if (layout.kind == TopologyKind::TWO_BY_EIGHT) {
        normalizedRankGroups = {layout.firstServerRanks, layout.secondServerRanks};
        std::sort(normalizedRankGroups.begin(), normalizedRankGroups.end());
    } else if (layout.kind == TopologyKind::EIGHT_PLUS_FOUR) {
        normalizedRankGroups = {layout.largeServerRanks, layout.smallServerRanks};
    }

    layout.signature = FNV_OFFSET_BASIS;
    HashValue(layout.signature, rankSize);
    HashValue(layout.signature, static_cast<uint32_t>(layout.kind));
    HashValue(layout.signature, static_cast<uint32_t>(layout.metadataSource));
    HashValue(layout.signature, layout.endpointTopologyValid ? 1U : 0U);
    HashValue(layout.signature, layerZeroMetadataValid ? 1U : 0U);
    HashValue(layout.signature, normalizedRankGroups.size());
    for (const std::vector<uint32_t> &rankGroup : normalizedRankGroups) {
        HashRankList(layout.signature, rankGroup);
    }
    HashValue(layout.signature, serverGroups.size());
    for (const ServerRankGroup &serverGroup : serverGroups) {
        HashValue(layout.signature, serverGroup.serverIndex);
        HashRankList(layout.signature, serverGroup.ranks);
    }
    HashRankList(layout.signature, layout.layers);
    HashRankList(layout.signature, layout.instanceSizes);
    HashRankList(layout.signature, layout.localRanks);

    std::vector<PeerChannelInfo> sortedPeerChannels = peerChannels;
    std::sort(sortedPeerChannels.begin(), sortedPeerChannels.end(),
        [](const PeerChannelInfo &left, const PeerChannelInfo &right) { return left.remoteRank < right.remoteRank; });
    HashValue(layout.signature, sortedPeerChannels.size());
    for (const PeerChannelInfo &peerChannel : sortedPeerChannels) {
        HashValue(layout.signature, peerChannel.remoteRank);
        HashValue(layout.signature, peerChannel.layer);
        HashValue(layout.signature, peerChannel.localIoDie);
        HashValue(layout.signature, static_cast<uint32_t>(peerChannel.protocol));
    }

    if (layout.kind == TopologyKind::GENERIC) {
        HCCL_WARNING("[HcclAllGather] formal topology signature is disabled, endpointValid[%u], "
                     "layerZeroMetadataValid[%u], layerZeroFallbackValid[%u], serverGroupCount[%zu]",
            layout.endpointTopologyValid ? 1U : 0U, layerZeroMetadataValid ? 1U : 0U,
            layerZeroFallbackValid ? 1U : 0U, serverGroups.size());
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 按 remoteRank 查找唯一 Channel 元数据
 * @param peerChannels 全量 peer Channel 列表
 * @param remoteRank 需要查找的对端 rank
 * @return 找到时返回元数据地址，未找到时返回空指针
 */
const PeerChannelInfo *FindPeerChannel(const std::vector<PeerChannelInfo> &peerChannels, uint32_t remoteRank)
{
    auto iter = std::find_if(peerChannels.begin(), peerChannels.end(), [remoteRank](const PeerChannelInfo &info) {
        return info.remoteRank == remoteRank;
    });
    return iter == peerChannels.end() ? nullptr : &(*iter);
}

/**
 * @brief 将逻辑 peer 集按实际网络层和本地 IO Die 拆成 Kernel variant
 * @param peerChannels 全量 peer Channel 列表
 * @param peerRanks 当前 phase 的逻辑对端列表
 * @param segmentRanks 当前 phase 传输的全局 block owner 列表
 * @param expectedLayer 当前 phase 预期使用的网络层
 * @param resGroupId 当前 variant 所属资源组
 * @param phaseId 当前 variant 的主 phase 标识
 * @param phaseMask 当前 variant 参与的唯一 phase 位
 * @param sourceKind 当前 variant 的源内存类别
 * @param variantType 当前 variant 的静态类型
 * @param handleSelfFirstGroup 是否由稳定排序后的首个物理组执行本地 copy
 * @param pendingVariants 接收生成的固定布局 variant
 * @return 成功返回 HCCL_SUCCESS，映射非法或 Event 容量不足返回对应错误码
 */
HcclResult AddGroupedVariants(const std::vector<PeerChannelInfo> &peerChannels,
    const std::vector<uint32_t> &peerRanks, const std::vector<uint32_t> &segmentRanks, uint32_t expectedLayer,
    uint32_t resGroupId, KernelPhaseId phaseId, uint32_t phaseMask, SourceKind sourceKind,
    KernelVariantType variantType, bool handleSelfFirstGroup, std::vector<PendingVariant> &pendingVariants)
{
    if (peerRanks.empty()) {
        return HCCL_SUCCESS;
    }
    std::vector<uint32_t> sortedPeerRanks = peerRanks;
    std::sort(sortedPeerRanks.begin(), sortedPeerRanks.end());
    if (std::adjacent_find(sortedPeerRanks.begin(), sortedPeerRanks.end()) != sortedPeerRanks.end()) {
        HCCL_ERROR("[HcclAllGather] duplicate peer found while building variant");
        return HCCL_E_INTERNAL;
    }

    std::vector<ChannelPeerGroup> groups;
    for (uint32_t peerRank : sortedPeerRanks) {
        const PeerChannelInfo *channelInfo = FindPeerChannel(peerChannels, peerRank);
        if (channelInfo == nullptr || channelInfo->layer != expectedLayer) {
            HCCL_ERROR("[HcclAllGather] peer channel layer mismatch, peerRank[%u], expectedLayer[%u]", peerRank,
                expectedLayer);
            return HCCL_E_INTERNAL;
        }
        auto groupIter = std::find_if(groups.begin(), groups.end(), [channelInfo](const ChannelPeerGroup &group) {
            return group.layer == channelInfo->layer && group.localIoDie == channelInfo->localIoDie;
        });
        if (groupIter == groups.end()) {
            groups.push_back(ChannelPeerGroup{channelInfo->layer, channelInfo->localIoDie, {peerRank}});
        } else {
            groupIter->peerRanks.push_back(peerRank);
        }
    }
    std::sort(groups.begin(), groups.end(), [](const ChannelPeerGroup &left, const ChannelPeerGroup &right) {
        return std::tie(left.layer, left.localIoDie) < std::tie(right.layer, right.localIoDie);
    });
    if (groups.size() > MAX_IO_DIE_COUNT) {
        HCCL_ERROR("[HcclAllGather] too many physical groups in one phase, groupCount[%zu]", groups.size());
        return HCCL_E_NOT_SUPPORT;
    }

    for (uint32_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        if (pendingVariants.size() >= MAX_KERNEL_VARIANT_COUNT) {
            HCCL_ERROR("[HcclAllGather] kernel variant count exceeds limit[%u]", MAX_KERNEL_VARIANT_COUNT);
            return HCCL_E_UNAVAIL;
        }
        const ChannelPeerGroup &group = groups[groupIndex];
        uint32_t operationCount = static_cast<uint32_t>(group.peerRanks.size() * segmentRanks.size());
        bool handleSelf = handleSelfFirstGroup && groupIndex == 0;
        uint32_t selfOperationCount = handleSelf ? 1U : 0U;
        if (operationCount + selfOperationCount > MAX_OPERATION_COUNT) {
            HCCL_ERROR("[HcclAllGather] Event capacity exceeded, operations[%u], selfOperations[%u]", operationCount,
                selfOperationCount);
            return HCCL_E_UNAVAIL;
        }

        PendingVariant pending{};
        errno_t clearRet = memset_s(&pending.desc, sizeof(pending.desc), 0, sizeof(pending.desc));
        if (clearRet != EOK) {
            HCCL_ERROR("[HcclAllGather] clear variant failed, ret[%d]", clearRet);
            return HCCL_E_INTERNAL;
        }
        KernelVariantDesc &variant = pending.desc;
        variant.resGroupId = resGroupId;
        variant.groupId = group.layer * MAX_IO_DIE_COUNT + group.localIoDie;
        variant.variantId = static_cast<uint32_t>(pendingVariants.size());
        variant.variantType = static_cast<uint32_t>(segmentRanks.empty() ? KernelVariantType::RECEIVE_ONLY
                                                                         : variantType);
        variant.phaseId = static_cast<uint32_t>(phaseId);
        variant.phaseMask = phaseMask;
        variant.layer = group.layer;
        variant.localIoDie = group.localIoDie;
        variant.threadSlot = 0;
        variant.notifySlot = 0;
        variant.sourceKind = static_cast<uint32_t>(sourceKind);
        variant.handleSelf = handleSelf ? 1U : 0U;
        variant.channelCount = static_cast<uint32_t>(group.peerRanks.size());
        variant.segmentCount = static_cast<uint32_t>(segmentRanks.size());
        variant.operationCount = operationCount;
        variant.selfEventBit = INVALID_EVENT_BIT;

        for (uint32_t peerIndex = 0; peerIndex < variant.channelCount; ++peerIndex) {
            uint32_t peerRank = group.peerRanks[peerIndex];
            const PeerChannelInfo *channelInfo = FindPeerChannel(peerChannels, peerRank);
            if (channelInfo == nullptr) {
                HCCL_ERROR("[HcclAllGather] peer channel disappeared, peerRank[%u]", peerRank);
                return HCCL_E_INTERNAL;
            }
            variant.channels[peerIndex] = channelInfo->channel;
            variant.peerRanks[peerIndex] = peerRank;
        }
        for (uint32_t segmentIndex = 0; segmentIndex < variant.segmentCount; ++segmentIndex) {
            variant.segmentRanks[segmentIndex] = segmentRanks[segmentIndex];
        }
        for (uint32_t operationIndex = 0; operationIndex < operationCount; ++operationIndex) {
            variant.eventBits[operationIndex] = static_cast<uint8_t>(operationIndex);
            variant.completionMask |= static_cast<uint16_t>(1U << operationIndex);
        }
        if (handleSelf) {
            variant.selfEventBit = operationCount;
            variant.completionMask |= static_cast<uint16_t>(1U << variant.selfEventBit);
        }
        pendingVariants.push_back(pending);
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 按每个 one-hot phase 使用的物理 IO Die 统一分配主从 Thread 槽位
 * @param pendingVariants 已生成并接收 threadSlot 的全部静态 variant
 * @return 分配成功返回 HCCL_SUCCESS，phase 或 IO Die 元数据非法时返回对应错误码
 */
HcclResult AssignPhaseThreadSlots(std::vector<PendingVariant> &pendingVariants)
{
    const uint32_t phaseBits[] = {PHASE_DIRECT_L0, PHASE_DIRECT_L1, PHASE_FAST_LOCAL_GATHER,
        PHASE_FAST_INTER_SERVER, PHASE_FAST_LOCAL_FORWARD};
    for (uint32_t phaseBit : phaseBits) {
        std::vector<uint32_t> phaseIoDies;
        for (const PendingVariant &pending : pendingVariants) {
            if ((pending.desc.phaseMask & phaseBit) == 0) {
                continue;
            }
            if (pending.desc.phaseMask != phaseBit || pending.desc.localIoDie >= MAX_IO_DIE_COUNT) {
                HCCL_ERROR("[HcclAllGather] invalid phase placement metadata, variantId[%u], phaseMask[0x%x], "
                           "localIoDie[%u]",
                    pending.desc.variantId, pending.desc.phaseMask, pending.desc.localIoDie);
                return HCCL_E_INTERNAL;
            }
            if (std::find(phaseIoDies.begin(), phaseIoDies.end(), pending.desc.localIoDie) == phaseIoDies.end()) {
                phaseIoDies.push_back(pending.desc.localIoDie);
            }
        }
        std::sort(phaseIoDies.begin(), phaseIoDies.end());
        if (phaseIoDies.size() > MAX_IO_DIE_COUNT) {
            HCCL_ERROR("[HcclAllGather] phase spans too many IO Dies, phaseMask[0x%x], dieCount[%zu]", phaseBit,
                phaseIoDies.size());
            return HCCL_E_NOT_SUPPORT;
        }
        for (PendingVariant &pending : pendingVariants) {
            if ((pending.desc.phaseMask & phaseBit) == 0) {
                continue;
            }
            pending.desc.threadSlot = phaseIoDies.size() == 2 && pending.desc.localIoDie == phaseIoDies[1] ? 1U : 0U;
        }
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 为 direct-push 和正式拓扑快路径生成完整静态 variant 集
 * @param myRank 当前全局 rank 编号
 * @param layout 已识别的拓扑布局
 * @param peerChannels 全量 peer Channel 列表
 * @param pendingVariants 接收生成的全部 variant
 * @param fastPathEnabled 接收当前拓扑快路径是否完整启用
 * @return 成功返回 HCCL_SUCCESS，静态调度生成失败返回对应错误码
 */
HcclResult BuildKernelVariants(uint32_t myRank, const TopologyLayout &layout,
    const std::vector<PeerChannelInfo> &peerChannels, std::vector<PendingVariant> &pendingVariants,
    bool &fastPathEnabled)
{
    constexpr uint32_t DIRECT_RES_GROUP = 0;
    constexpr uint32_t FAST_LOCAL_GATHER_RES_GROUP = 1;
    constexpr uint32_t FOUR_BY_ONE_RANK_SIZE = 4;
    constexpr uint32_t FOUR_BY_ONE_FORWARD_RES_GROUP = 2;
    constexpr uint32_t TWO_BY_EIGHT_FORWARD_RES_GROUP = 2;
    constexpr uint32_t EIGHT_PLUS_FOUR_LANE_RES_GROUP = FAST_LOCAL_GATHER_RES_GROUP;
    constexpr uint32_t EIGHT_PLUS_FOUR_FORWARD_RES_GROUP = 3;
    constexpr uint32_t EIGHT_PLUS_FOUR_LARGE_RANK_SIZE = 8;
    constexpr uint32_t EIGHT_PLUS_FOUR_SMALL_RANK_SIZE = 4;
    constexpr uint32_t EIGHT_PLUS_FOUR_OPPOSITE_MASK = 7;
    std::vector<uint32_t> localPeers = layout.localRanks;
    localPeers.erase(std::remove(localPeers.begin(), localPeers.end(), myRank), localPeers.end());
    std::vector<uint32_t> directSegments{myRank};
    bool enableHierarchicalFast = layout.kind == TopologyKind::TWO_BY_EIGHT;
    fastPathEnabled = false;

    if (!localPeers.empty()) {
        CHK_RET(AddGroupedVariants(peerChannels, localPeers, directSegments, NET_LAYER_ZERO, DIRECT_RES_GROUP,
            KernelPhaseId::DIRECT_L0, PHASE_DIRECT_L0, SourceKind::INPUT, KernelVariantType::DIRECT, true,
            pendingVariants));
    }

    CHK_RET(AddGroupedVariants(peerChannels, layout.remoteRanks, directSegments, NET_LAYER_ONE,
        DIRECT_RES_GROUP, KernelPhaseId::DIRECT_L0, PHASE_DIRECT_L0, SourceKind::INPUT, KernelVariantType::DIRECT,
        localPeers.empty(), pendingVariants));

    // 仅 2×8 继续生成分层快路径，其余拓扑固定使用 direct variants
    if (!enableHierarchicalFast) {
        CHK_RET(AssignPhaseThreadSlots(pendingVariants));
        return HCCL_SUCCESS;
    }

    if (enableHierarchicalFast) {
        CHK_RET(AddGroupedVariants(peerChannels, localPeers, directSegments, NET_LAYER_ZERO,
            FAST_LOCAL_GATHER_RES_GROUP, KernelPhaseId::FAST_LOCAL_GATHER, PHASE_FAST_LOCAL_GATHER,
            SourceKind::INPUT, KernelVariantType::DIRECT, true, pendingVariants));
    }

    if (layout.kind == TopologyKind::FOUR_BY_ONE) {
        if (myRank >= FOUR_BY_ONE_RANK_SIZE || layout.localRanks != std::vector<uint32_t>{myRank}
            || layout.remoteRanks.size() != FOUR_BY_ONE_RANK_SIZE - 1) {
            HCCL_ERROR("[HcclAllGather] invalid 4x1 rank layout, myRank[%u], localRanks[%zu], remoteRanks[%zu]",
                myRank, layout.localRanks.size(), layout.remoteRanks.size());
            return HCCL_E_INTERNAL;
        }

        // 4×1 按全局 rank 异或关系执行两轮 recursive-doubling
        const uint32_t firstRoundPeer = myRank ^ 1U;
        const uint32_t secondRoundPeer = myRank ^ 2U;
        const uint32_t pairBaseRank = myRank & ~1U;
        const std::vector<uint32_t> secondRoundSegments{pairBaseRank, pairBaseRank + 1U};
        CHK_RET(AddGroupedVariants(peerChannels, std::vector<uint32_t>{firstRoundPeer}, directSegments,
            NET_LAYER_ONE, FAST_LOCAL_GATHER_RES_GROUP, KernelPhaseId::FAST_LOCAL_GATHER,
            PHASE_FAST_LOCAL_GATHER, SourceKind::INPUT, KernelVariantType::FAST_LANE, true, pendingVariants));
        CHK_RET(AddGroupedVariants(peerChannels, std::vector<uint32_t>{secondRoundPeer}, secondRoundSegments,
            NET_LAYER_ONE, FOUR_BY_ONE_FORWARD_RES_GROUP, KernelPhaseId::FAST_LOCAL_FORWARD,
            PHASE_FAST_LOCAL_FORWARD, SourceKind::OUTPUT, KernelVariantType::FAST_FORWARD, false, pendingVariants));
        fastPathEnabled = true;
    } else if (layout.kind == TopologyKind::TWO_BY_EIGHT) {
        const std::vector<uint32_t> *ownServer = nullptr;
        const std::vector<uint32_t> *peerServer = nullptr;
        if (ContainsRank(layout.firstServerRanks, myRank)) {
            ownServer = &layout.firstServerRanks;
            peerServer = &layout.secondServerRanks;
        } else {
            ownServer = &layout.secondServerRanks;
            peerServer = &layout.firstServerRanks;
        }
        uint32_t laneIndex = 0;
        if (!FindRankIndex(*ownServer, myRank, laneIndex) || laneIndex >= peerServer->size()) {
            HCCL_ERROR("[HcclAllGather] failed to build 2x8 lane, myRank[%u]", myRank);
            return HCCL_E_INTERNAL;
        }
        uint32_t lanePeer = (*peerServer)[laneIndex];
        CHK_RET(AddGroupedVariants(peerChannels, std::vector<uint32_t>{lanePeer}, directSegments, NET_LAYER_ONE,
            FAST_LOCAL_GATHER_RES_GROUP, KernelPhaseId::FAST_LOCAL_GATHER, PHASE_FAST_LOCAL_GATHER,
            SourceKind::INPUT, KernelVariantType::FAST_LANE, false, pendingVariants));
        CHK_RET(AddGroupedVariants(peerChannels, localPeers, std::vector<uint32_t>{lanePeer}, NET_LAYER_ZERO,
            TWO_BY_EIGHT_FORWARD_RES_GROUP, KernelPhaseId::FAST_LOCAL_FORWARD, PHASE_FAST_LOCAL_FORWARD,
            SourceKind::OUTPUT, KernelVariantType::FAST_FORWARD, false, pendingVariants));
        fastPathEnabled = true;
    } else if (layout.kind == TopologyKind::EIGHT_PLUS_FOUR) {
        if (layout.largeServerRanks.size() != EIGHT_PLUS_FOUR_LARGE_RANK_SIZE
            || layout.smallServerRanks.size() != EIGHT_PLUS_FOUR_SMALL_RANK_SIZE) {
            HCCL_ERROR("[HcclAllGather] invalid 8+4 rank groups, large[%zu], small[%zu]",
                layout.largeServerRanks.size(), layout.smallServerRanks.size());
            return HCCL_E_INTERNAL;
        }

        // 8+4 并发执行组内 gather 与对顶双播种，再沿 3 维超立方体三邻居完成大侧分发
        uint32_t sideIndex = 0;
        if (FindRankIndex(layout.largeServerRanks, myRank, sideIndex)) {
            const uint32_t oppositeIndex = sideIndex ^ EIGHT_PLUS_FOUR_OPPOSITE_MASK;
            const uint32_t laneIndex = std::min(sideIndex, oppositeIndex);
            const uint32_t lanePeer = layout.smallServerRanks[laneIndex];
            CHK_RET(AddGroupedVariants(peerChannels, std::vector<uint32_t>{lanePeer}, directSegments,
                NET_LAYER_ONE, EIGHT_PLUS_FOUR_LANE_RES_GROUP, KernelPhaseId::FAST_LOCAL_GATHER,
                PHASE_FAST_LOCAL_GATHER, SourceKind::INPUT, KernelVariantType::FAST_LANE, false,
                pendingVariants));

            std::vector<uint32_t> forwardPeers{layout.largeServerRanks[sideIndex ^ 1U],
                layout.largeServerRanks[sideIndex ^ 2U], layout.largeServerRanks[sideIndex ^ 4U]};
            CHK_RET(AddGroupedVariants(peerChannels, forwardPeers,
                std::vector<uint32_t>{layout.smallServerRanks[laneIndex]}, NET_LAYER_ZERO,
                EIGHT_PLUS_FOUR_FORWARD_RES_GROUP, KernelPhaseId::FAST_LOCAL_FORWARD,
                PHASE_FAST_LOCAL_FORWARD, SourceKind::OUTPUT, KernelVariantType::FAST_FORWARD, false,
                pendingVariants));
        } else if (FindRankIndex(layout.smallServerRanks, myRank, sideIndex)) {
            const uint32_t oppositeIndex = sideIndex ^ EIGHT_PLUS_FOUR_OPPOSITE_MASK;
            std::vector<uint32_t> lanePeers{
                layout.largeServerRanks[sideIndex], layout.largeServerRanks[oppositeIndex]};
            CHK_RET(AddGroupedVariants(peerChannels, lanePeers, directSegments, NET_LAYER_ONE,
                EIGHT_PLUS_FOUR_LANE_RES_GROUP,
                KernelPhaseId::FAST_LOCAL_GATHER, PHASE_FAST_LOCAL_GATHER, SourceKind::INPUT,
                KernelVariantType::FAST_LANE, false, pendingVariants));
            std::vector<uint32_t> forwardSegments{
                layout.largeServerRanks[sideIndex], layout.largeServerRanks[oppositeIndex]};
            std::sort(forwardSegments.begin(), forwardSegments.end());
            CHK_RET(AddGroupedVariants(peerChannels, localPeers, forwardSegments, NET_LAYER_ZERO,
                EIGHT_PLUS_FOUR_FORWARD_RES_GROUP, KernelPhaseId::FAST_LOCAL_FORWARD,
                PHASE_FAST_LOCAL_FORWARD, SourceKind::OUTPUT, KernelVariantType::FAST_FORWARD, false,
                pendingVariants));
        } else {
            HCCL_ERROR("[HcclAllGather] rank is absent from 8+4 server groups, myRank[%u]", myRank);
            return HCCL_E_INTERNAL;
        }
        fastPathEnabled = true;
    }
    CHK_RET(AssignPhaseThreadSlots(pendingVariants));
    return HCCL_SUCCESS;
}

/**
 * @brief 将固定 variant 描述转换为注册期 CCU Kernel 参数
 * @param contextVariant 已完成静态审计的 Context variant
 * @param myRank 当前全局 rank 编号
 * @param rankSize 当前通信域 rank 数量
 * @param kernelArg 接收注册期固定 Kernel 参数
 * @return 成功返回 HCCL_SUCCESS，固定数组拷贝失败返回对应错误码
 */
HcclResult BuildKernelArg(const KernelVariantDesc &contextVariant, uint32_t myRank, uint32_t rankSize,
    ops_hccl::CcuKernelArgAllGather &kernelArg)
{
    errno_t clearRet = memset_s(&kernelArg, sizeof(kernelArg), 0, sizeof(kernelArg));
    if (clearRet != EOK) {
        HCCL_ERROR("[HcclAllGather] clear kernel argument failed, ret[%d]", clearRet);
        return HCCL_E_INTERNAL;
    }
    kernelArg.myRank = myRank;
    kernelArg.rankSize = rankSize;
    kernelArg.variantType = contextVariant.variantType;
    kernelArg.phaseId = contextVariant.phaseId;
    kernelArg.sourceKind = contextVariant.sourceKind;
    kernelArg.handleSelf = contextVariant.handleSelf;
    kernelArg.channelCount = contextVariant.channelCount;
    kernelArg.segmentCount = contextVariant.segmentCount;
    kernelArg.operationCount = contextVariant.operationCount;
    kernelArg.selfEventBit = contextVariant.selfEventBit;
    kernelArg.completionMask = contextVariant.completionMask;
    for (uint32_t peerIndex = 0; peerIndex < contextVariant.channelCount; ++peerIndex) {
        kernelArg.channels[peerIndex] = contextVariant.channels[peerIndex];
        kernelArg.peerRanks[peerIndex] = contextVariant.peerRanks[peerIndex];
    }
    for (uint32_t segmentIndex = 0; segmentIndex < contextVariant.segmentCount; ++segmentIndex) {
        kernelArg.segmentRanks[segmentIndex] = contextVariant.segmentRanks[segmentIndex];
    }
    for (uint32_t operationIndex = 0; operationIndex < contextVariant.operationCount; ++operationIndex) {
        kernelArg.eventBits[operationIndex] = contextVariant.eventBits[operationIndex];
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 按 resGroup 分轮注册全部 CCU Kernel 并回填 handle
 * @param comm 当前通信域
 * @param myRank 当前全局 rank 编号
 * @param rankSize 当前通信域 rank 数量
 * @param pendingVariants 待注册并接收 Kernel handle 的 variant 列表
 * @return 成功返回 HCCL_SUCCESS，CCU instruction 或注册失败返回对应错误码
 */
HcclResult RegisterKernelVariants(HcclComm comm, uint32_t myRank, uint32_t rankSize,
    std::vector<PendingVariant> &pendingVariants)
{
    if (HcommCcuKernelRegisterStart == nullptr || HcommCcuKernelRegister == nullptr
        || HcommCcuKernelRegisterEnd == nullptr) {
        HCCL_ERROR("[HcclAllGather] CCU kernel registration API is unavailable");
        return HCCL_E_NOT_SUPPORT;
    }
    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    if (insNum != 1 || insHandle == 0) {
        HCCL_ERROR("[HcclAllGather] unsupported CCU instruction count[%u]", insNum);
        return HCCL_E_NOT_SUPPORT;
    }

    std::vector<uint32_t> resourceGroups;
    resourceGroups.reserve(pendingVariants.size());
    for (const PendingVariant &pending : pendingVariants) {
        if (std::find(resourceGroups.begin(), resourceGroups.end(), pending.desc.resGroupId) == resourceGroups.end()) {
            resourceGroups.push_back(pending.desc.resGroupId);
        }
    }
    std::sort(resourceGroups.begin(), resourceGroups.end());
    std::vector<std::shared_ptr<ops_hccl::CcuKernelArgAllGather>> kernelArgs;
    kernelArgs.reserve(pendingVariants.size());
    for (uint32_t resGroup : resourceGroups) {
        CcuResult startRet = HcommCcuKernelRegisterStart(insHandle);
        if (startRet != CCU_SUCCESS) {
            HCCL_ERROR("[HcclAllGather] CCU register start failed, resGroup[%u], ret[%d]", resGroup, startRet);
            return ConvertCcuToHccl(startRet);
        }
        for (PendingVariant &pending : pendingVariants) {
            if (pending.desc.resGroupId != resGroup) {
                continue;
            }
            auto kernelArg = std::make_shared<ops_hccl::CcuKernelArgAllGather>();
            CHK_RET(BuildKernelArg(pending.desc, myRank, rankSize, *kernelArg));
            const void *staticArgs[] = {kernelArg.get()};
            CcuKernelHandle kernelHandle = 0;
            CcuResult registerRet = HcommCcuKernelRegister(insHandle, 0, "CcuKernel",
                reinterpret_cast<const void *>(ops_hccl::CcuKernel), staticArgs, 1, &kernelHandle);
            if (registerRet != CCU_SUCCESS || kernelHandle == 0) {
                HCCL_ERROR("[HcclAllGather] CCU kernel register failed, variantId[%u], ret[%d]",
                    pending.desc.variantId, registerRet);
                return registerRet == CCU_SUCCESS ? HCCL_E_INTERNAL : ConvertCcuToHccl(registerRet);
            }
            pending.desc.kernelHandle = kernelHandle;
            kernelArgs.push_back(std::move(kernelArg));
        }
        CcuResult endRet = HcommCcuKernelRegisterEnd(insHandle);
        if (endRet != CCU_SUCCESS) {
            HCCL_ERROR("[HcclAllGather] CCU register end failed, resGroup[%u], ret[%d]", resGroup, endRet);
            return ConvertCcuToHccl(endRet);
        }
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 一次性申请 Channel、从 Thread、Kernel 并构造可复用固定 Context
 * @param comm 当前通信域
 * @param param 当前算子参数
 * @param resourceCtx 接收完整且已校验的可复用资源上下文
 * @return 成功返回 HCCL_SUCCESS，资源申请或静态调度失败返回对应错误码
 */
HcclResult BuildResourceContext(HcclComm comm, const OpParam &param, AlgResourceCtx &resourceCtx)
{
    CHK_RET(resourceCtx.Initialize(param.rankSize, param.myRank));
    TopologyLayout layout;
    CHK_RET(BuildTopologyLayout(comm, param.myRank, param.rankSize, layout));
    std::vector<PeerChannelInfo> peerChannels;
    CHK_RET(BuildPeerChannels(comm, param.myRank, param.rankSize, layout, peerChannels));
    CHK_RET(FinalizeTopologyLayout(param.myRank, param.rankSize, peerChannels, layout));

    std::vector<PendingVariant> pendingVariants;
    bool fastPathEnabled = false;
    CHK_RET(BuildKernelVariants(param.myRank, layout, peerChannels, pendingVariants, fastPathEnabled));
    if (pendingVariants.empty() || pendingVariants.size() > MAX_KERNEL_VARIANT_COUNT) {
        HCCL_ERROR("[HcclAllGather] invalid generated variant count[%zu]", pendingVariants.size());
        return HCCL_E_INTERNAL;
    }

    bool needsSlaveThread = std::any_of(pendingVariants.begin(), pendingVariants.end(), [](const PendingVariant &item) {
        return item.desc.threadSlot == 1;
    });
    if (needsSlaveThread) {
        ThreadHandle slaveThread = 0;
        CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_CCU, 1, 1, &slaveThread));
        if (slaveThread == 0) {
            HCCL_ERROR("[HcclAllGather] acquired empty slave thread");
            return HCCL_E_INTERNAL;
        }
        resourceCtx.hasSlaveThread = 1;
        resourceCtx.slaveThread = slaveThread;
        resourceCtx.mainToSlaveNotifyIdx = 0;
        resourceCtx.slaveToMainNotifyIdx = 0;
    }

    CHK_RET(RegisterKernelVariants(comm, param.myRank, param.rankSize, pendingVariants));
    resourceCtx.topologyKind = static_cast<uint32_t>(layout.kind);
    resourceCtx.topologySignature = layout.signature;
    resourceCtx.fastPathEnabled = fastPathEnabled ? 1U : 0U;
    resourceCtx.variantCount = static_cast<uint32_t>(pendingVariants.size());
    for (uint32_t variantIndex = 0; variantIndex < resourceCtx.variantCount; ++variantIndex) {
        resourceCtx.variants[variantIndex] = pendingVariants[variantIndex].desc;
    }
    CHK_RET(resourceCtx.Validate());
    HCCL_INFO("[HcclAllGather] resource context ready, topologyKind[%u], signature[0x%llx], variants[%u]",
        resourceCtx.topologyKind, static_cast<unsigned long long>(resourceCtx.topologySignature),
        resourceCtx.variantCount);
    return HCCL_SUCCESS;
}

/**
 * @brief 获取已缓存 Context 或在资源完整就绪后创建新 Context
 * @param comm 当前通信域
 * @param param 当前算子参数
 * @param resourceCtx 接收反序列化或新建的资源上下文
 * @return 成功返回 HCCL_SUCCESS，缓存非法或创建失败返回对应错误码
 */
HcclResult LoadOrCreateResourceContext(HcclComm comm, const OpParam &param, AlgResourceCtx &resourceCtx)
{
    void *cachedContext = nullptr;
    uint64_t cachedSize = 0;
    HcclResult getRet = HcclEngineCtxGet(comm, param.tag, COMM_ENGINE_CCU, &cachedContext, &cachedSize);
    if (getRet == HCCL_SUCCESS) {
        CHK_RET(resourceCtx.DeSerialize(cachedContext, cachedSize));
        if (resourceCtx.myRank != param.myRank || resourceCtx.rankSize != param.rankSize) {
            HCCL_ERROR("[HcclAllGather] cached context rank metadata mismatch");
            return HCCL_E_INTERNAL;
        }
        return HCCL_SUCCESS;
    }
    if (getRet != HCCL_E_NOT_FOUND) {
        HCCL_ERROR("[HcclAllGather] get engine context failed, ret[%d]", getRet);
        return getRet;
    }

    CHK_RET(BuildResourceContext(comm, param, resourceCtx));
    std::vector<char> serializedContext;
    CHK_RET(resourceCtx.Serialize(serializedContext));
    void *newContext = nullptr;
    CHK_RET(HcclEngineCtxCreate(comm, param.tag, COMM_ENGINE_CCU, serializedContext.size(), &newContext));
    HcclResult copyRet = HcclEngineCtxCopy(comm, COMM_ENGINE_CCU, param.tag, serializedContext.data(),
        serializedContext.size(), 0);
    if (copyRet != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclAllGather] copy engine context failed, ret[%d]", copyRet);
        (void)HcclEngineCtxDestroy(comm, param.tag, COMM_ENGINE_CCU);
        return copyRet;
    }
    return HCCL_SUCCESS;
}

}

HcclResult HcclAllGather(
    void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    if (dataType != HCCL_DATA_TYPE_FP32) {
        HCCL_ERROR("[HcclAllGather] only FP32 is supported, dataType[%d]", dataType);
        return HCCL_E_PARA;
    }

    OpParam param;
    int tagLength = sprintf_s(param.tag, sizeof(param.tag), "%s", CONTEXT_TAG);
    if (tagLength < 0 || static_cast<size_t>(tagLength) >= sizeof(param.tag)) {
        HCCL_ERROR("[HcclAllGather] failed to build context tag");
        return HCCL_E_INTERNAL;
    }
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    if (param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize) {
        HCCL_ERROR("[HcclAllGather] invalid rank metadata, myRank[%u], rankSize[%u]", param.myRank,
            param.rankSize);
        return HCCL_E_PARA;
    }

    uint64_t rankDataBytes = 0;
    uint64_t totalRecvBytes = 0;
    if (!CheckedMultiply(sendCount, sizeof(float), rankDataBytes)
        || !CheckedMultiply(rankDataBytes, param.rankSize, totalRecvBytes)) {
        HCCL_ERROR("[HcclAllGather] data size multiplication overflow, sendCount[%llu], rankSize[%u]",
            static_cast<unsigned long long>(sendCount), param.rankSize);
        return HCCL_E_PARA;
    }
    CHK_RET(CheckAddressRange(sendBuf, rankDataBytes));
    CHK_RET(CheckAddressRange(recvBuf, totalRecvBytes));
    if (sendCount == 0) {
        return HCCL_SUCCESS;
    }

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {0};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CCU, stream, 1, &param.cpuThread));
    if (param.cpuThread == 0) {
        HCCL_ERROR("[HcclAllGather] acquired empty main thread");
        return HCCL_E_INTERNAL;
    }
    if (param.rankSize == 1) {
        return static_cast<HcclResult>(HcommLocalCopyOnThread(param.cpuThread, recvBuf, sendBuf, rankDataBytes));
    }

    AlgResourceCtx resourceCtx;
    CHK_RET(LoadOrCreateResourceContext(comm, param, resourceCtx));
    CHK_RET(ops_hccl::ExecOp(param, resourceCtx));
    return HCCL_SUCCESS;
}
