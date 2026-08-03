/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

// ===========================================================================
// custom.h —— 算子自定义资源结构与公共扩展（Host/Device 共用）
//
// 按官方目录结构约定：
//   - common.h  仅提供 OpParam、常量与 SIZE_TABLE（官方固定，勿改）；
//   - 本文件    承载通信资源结构（CommBuffer / ChannelInfo / AlgResourceCtx）
//               以及算法所需的扩展数据结构与公共函数。
//
// 内容：
//   【资源结构】AlgResourceCtx —— 随通信引擎上下文（HcclEngineCtx）序列化下发到
//       Device 的资源包：AICPU thread、本端 CCL Buffer、Channel 列表，
//       以及本算子扩展的分层拓扑信息（serverRanks / myLocalIdx / interPeerRank）。
//       注意：序列化/反序列化字段顺序必须严格一致。
//   【公共函数】（ops_hccl 命名空间）
//       ToHdt / ToHop   —— HCCL 类型 -> Hcomm 类型转换（枚举值一一对应，直接强转）
//       ChMap           —— remoteRank -> ChannelInfo 映射，按对端 rank 索引 Channel
//       ServerRanksOf   —— 本 Server 的 rank 列表（Host 未填拓扑时退化为全通信域）
//       WriteReduceSync —— GO/DONE 两拍握手 + 远端原子归约 的平台标准通信步封装
// ===========================================================================

#include <unordered_map>
#include <vector>

#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem;
};

struct AlgResourceCtx {
    ThreadHandle aicpuThread;          ///< AICPU_TS通信引擎上的thread资源
    CommBuffer localBuffer;            ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads; ///< AICPU_TS通信引擎上的thread资源
    std::vector<ChannelInfo> channels; ///< AICPU_TS通信引擎上的channel资源
    // ---- 扩展：分层拓扑信息（由 Host 侧 BuildTopology 填充，ReduceScatter 分层回退算法使用） ----
    std::vector<uint32_t> serverRanks;             ///< 本 rank 所在 Server 的 rank 列表（layer-0 直连域）
    uint32_t myLocalIdx = 0;                       ///< 本 rank 在 serverRanks 中的下标
    uint32_t interPeerRank = INVALID_VALUE_RANKID; ///< 跨 Server(layer-1 Clos) 配对 rank；单 Server 时为 INVALID

    // 序列化（字段顺序必须与 DeSerialize 严格一致）
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << aicpuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << channels;
        binaryStream << serverRanks;
        binaryStream << myLocalIdx;
        binaryStream << interPeerRank;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // 反序列化
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> aicpuThread;
        binaryStream >> localBuffer;
        binaryStream >> threads;
        binaryStream >> channels;
        binaryStream >> serverRanks;
        binaryStream >> myLocalIdx;
        binaryStream >> interPeerRank;
    }
};

namespace ops_hccl {

// HCCL 数据类型 -> Hcomm 数据类型（枚举布局一致）
inline HcommDataType ToHdt(HcclDataType t)
{
    return static_cast<HcommDataType>(t);
}

// HCCL 归约类型 -> Hcomm 归约类型（枚举布局一致）
inline HcommReduceOp ToHop(HcclReduceOp o)
{
    return static_cast<HcommReduceOp>(o);
}

// 获取数据类型字节数：FP32 走快速路径（benchmark 最常用），其余查表
inline uint32_t GetElemSize(HcclDataType dt)
{
    if (dt == HCCL_DATA_TYPE_FP32) {
        return sizeof(float);
    }
    const auto it = SIZE_TABLE.find(dt);
    return (it != SIZE_TABLE.end()) ? it->second : 0;
}

// 建立 remoteRank -> ChannelInfo 的映射，便于按对端 rank 直接索引 Channel
inline std::unordered_map<uint32_t, ChannelInfo> ChMap(const AlgResourceCtx &resCtx)
{
    std::unordered_map<uint32_t, ChannelInfo> m;
    for (const auto &ch : resCtx.channels) {
        m.emplace(ch.remoteRank, ch);
    }
    return m;
}

// ---------------------------------------------------------------------------
// BuildChannelArray —— 零堆分配、零 Hash 的 Channel 直索引数组
//
// 替代 ChMap() 的 std::unordered_map，适用于 rankSize ≤ 16 的场景。
// 用法：const ChannelInfo* chArray[16] = {nullptr}; BuildChannelArray(resCtx, chArray);
// 之后用 chArray[remoteRank] 做 O(1) 查找，替代 cm.find(remoteRank)。
// ---------------------------------------------------------------------------
inline void BuildChannelArray(const AlgResourceCtx &resCtx, const ChannelInfo* chArray[16])
{
    for (uint32_t i = 0; i < 16; ++i) {
        chArray[i] = nullptr;
    }
    for (const auto &ch : resCtx.channels) {
        if (ch.remoteRank < 16) {
            chArray[ch.remoteRank] = &ch;
        }
    }
}

// 本 Server 的 rank 列表（layer-0 直连域）；Host 未填（单 Server 场景）时退化为全通信域
inline std::vector<uint32_t> ServerRanksOf(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (!resCtx.serverRanks.empty()) {
        return resCtx.serverRanks;
    }
    std::vector<uint32_t> all(param.rankSize);
    for (uint32_t i = 0; i < param.rankSize; i++) {
        all[i] = i;
    }
    return all;
}

// ---------------------------------------------------------------------------
// WriteReduceSync：一次"GO/DONE 两拍握手 + 远端原子归约"标准通信步
//
//   第一拍(GO)  ：与对端互相同步，确认对端 CCL 目标区间可被覆盖；
//   WriteReduce ：把本地 localAddr 的 count 个元素原子归约进对端 remoteAddr；
//   第二拍(DONE)：确认归约完成，双方对齐后再进入下一通信步。
//
// 平台约束（实测踩坑，封装在此防止误用）：
//   1) checker 302 内存冲突检测不豁免远端原子归约，写同一区间的任务必须如此串行化；
//   2) 禁用 HcommWriteReduceWithNotifyOnThread（隐式 notify 与 checker 跨 rank
//      Record/Wait 匹配不兼容，ErrorCode 102），只能显式 Record/Wait。
// ---------------------------------------------------------------------------
inline HcclResult WriteReduceSync(ThreadHandle th, const ChannelInfo &ch, void *remoteAddr,
    const void *localAddr, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp)
{
    if (HcommChannelNotifyRecordOnThread(th, ch.handle, NOTIFY_IDX_DATA_SIGNAL) != HCCL_SUCCESS) {
        return HCCL_E_INTERNAL;
    }
    if (HcommChannelNotifyWaitOnThread(th, ch.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT) != HCCL_SUCCESS) {
        return HCCL_E_INTERNAL;
    }
    if (HcommWriteReduceOnThread(th, ch.handle, remoteAddr, localAddr, count, dataType, reduceOp) != HCCL_SUCCESS) {
        return HCCL_E_INTERNAL;
    }
    // 第二拍 (DONE)：使用 NOTIFY_IDX_ACK 而非复用 NOTIFY_IDX_DATA_SIGNAL，
    // 避免硬件 SQE 在同一 Notify ID 上产生事件状态清除等待气泡（参考 WriteSync 的做法）。
    if (HcommChannelNotifyRecordOnThread(th, ch.handle, NOTIFY_IDX_ACK) != HCCL_SUCCESS) {
        return HCCL_E_INTERNAL;
    }
    if (HcommChannelNotifyWaitOnThread(th, ch.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT) != HCCL_SUCCESS) {
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

// ---------------------------------------------------------------------------
// WriteReduceSyncTail —— 尾步归约（无 GO 握手，仅 DONE 确认）
//
// 用于 RH Steps 1-3：Step 0 的 DONE(ACK) 已完成双方同步，后续步的 GO(DATA_SIGNAL)
// 握手冗余。省去 Record+Wait 共 2 次 Hcomm API 调用/步，3 步合计省 ~3μs。
//
// 风险：checker 如报 302（缺少 DATA_SIGNAL 依赖边），需逐步回退为 WriteReduceSync。
// ---------------------------------------------------------------------------
inline HcclResult WriteReduceSyncTail(ThreadHandle th, const ChannelInfo &ch, void *remoteAddr,
    const void *localAddr, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp)
{
    if (HcommWriteReduceOnThread(th, ch.handle, remoteAddr, localAddr, count, dataType, reduceOp) != HCCL_SUCCESS) {
        return HCCL_E_INTERNAL;
    }
    // 仅 DONE 拍（ACK），无 GO 拍——前一步的 DONE 已确保双方同步
    if (HcommChannelNotifyRecordOnThread(th, ch.handle, NOTIFY_IDX_ACK) != HCCL_SUCCESS) {
        return HCCL_E_INTERNAL;
    }
    if (HcommChannelNotifyWaitOnThread(th, ch.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT) != HCCL_SUCCESS) {
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

// ===========================================================================
// DataSlice —— 缓冲区切片描述符（对标官方 template_utils.h::DataSlice）
//
// 封装 base-addr + byte-offset + size + count，消除 exec_op 中手动地址计算。
// count 字段用于 reduce 操作的元素数计数；纯读写操作可忽略。
// ===========================================================================
struct DataSlice {
    void *addr;
    uint64_t offset; // byte offset from addr
    uint64_t size;   // size in bytes
    uint64_t count;  // element count (for reduce ops)

    DataSlice(void *a, uint64_t off, uint64_t sz, uint64_t cnt)
        : addr(a), offset(off), size(sz), count(cnt)
    {
    }
    DataSlice(void *a, uint64_t off, uint64_t sz)
        : addr(a), offset(off), size(sz), count(0)
    {
    }
};

// ===========================================================================
// SendRecvInfo（精简版）—— Mesh 通信步描述符
// ===========================================================================
struct MeshSendInfo {
    ChannelInfo channel;
    void *remoteAddr; // 远端 CCL 目标地址（对端 slot）
    const void *localAddr; // 本地输入源地址
    uint64_t count; // 元素数
};

// ---------------------------------------------------------------------------
// WriteSync：一次 GO/DONE 两拍握手 + 远端 Write（非归约写入，按字节长度）
//
// 与 WriteReduceSync 结构相同，但使用 HcommWriteOnThread（接收字节长度）
// 替代 HcommWriteReduceOnThread（接收元素数+数据类型）。用于 Mesh 算法中向
// 对端独立 slot 写入数据（每个 slot 仅一个写入者，无需原子归约）。
// ---------------------------------------------------------------------------
inline HcclResult WriteSync(ThreadHandle th, const ChannelInfo &ch, void *remoteAddr,
    const void *localAddr, uint64_t len)
{
    if (HcommChannelNotifyRecordOnThread(th, ch.handle, NOTIFY_IDX_DATA_SIGNAL) != HCCL_SUCCESS) {
        return HCCL_E_INTERNAL;
    }
    if (HcommChannelNotifyWaitOnThread(th, ch.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT) != HCCL_SUCCESS) {
        return HCCL_E_INTERNAL;
    }
    if (HcommWriteOnThread(th, ch.handle, remoteAddr, const_cast<void *>(localAddr), len) != HCCL_SUCCESS) {
        return HCCL_E_INTERNAL;
    }
    if (HcommChannelNotifyRecordOnThread(th, ch.handle, NOTIFY_IDX_ACK) != HCCL_SUCCESS) {
        return HCCL_E_INTERNAL;
    }
    if (HcommChannelNotifyWaitOnThread(th, ch.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT) != HCCL_SUCCESS) {
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

// ---------------------------------------------------------------------------
// PreSyncInterThreads：主线程 → 所有从线程 启动同步
//
// 对标官方 PreSyncInterThreads。主线程向每个从线程 Record 一个 notify，
// 从线程随后 Wait 该 notify 以获取启动信号。
// ---------------------------------------------------------------------------
inline HcclResult PreSyncInterThreads(const ThreadHandle &mainThread,
    const std::vector<ThreadHandle> &subThreads, const std::vector<uint32_t> &notifyIdxMainToSub)
{
    for (size_t i = 0; i < subThreads.size(); i++) {
        if (HcommThreadNotifyRecordOnThread(mainThread, subThreads[i], notifyIdxMainToSub[i]) != HCCL_SUCCESS) {
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

// ---------------------------------------------------------------------------
// PreSyncInterThreads 栈数组重载 —— 消除 Mesh 路径中的 std::vector 堆分配
// ---------------------------------------------------------------------------
inline HcclResult PreSyncInterThreads(const ThreadHandle &mainThread,
    const ThreadHandle *subThreads, const uint32_t *notifyIdxMainToSub, uint32_t count)
{
    for (uint32_t i = 0; i < count; i++) {
        if (HcommThreadNotifyRecordOnThread(mainThread, subThreads[i], notifyIdxMainToSub[i]) != HCCL_SUCCESS) {
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

// ---------------------------------------------------------------------------
// PostSyncInterThreads：主线程等待所有从线程完成
//
// 对标官方 PostSyncInterThreads。每个从线程向主线程 Record 一个 notify，
// 主线程 Wait 所有 notify。调用前需确保从线程的 Record 任务已提交。
// ---------------------------------------------------------------------------
inline HcclResult PostSyncInterThreads(const ThreadHandle &mainThread,
    const std::vector<ThreadHandle> &subThreads, const std::vector<uint32_t> &notifyIdxSubToMain)
{
    for (size_t i = 0; i < subThreads.size(); i++) {
        if (HcommThreadNotifyWaitOnThread(mainThread, notifyIdxSubToMain[i], CUSTOM_TIMEOUT) != HCCL_SUCCESS) {
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl

#endif // OPS_HCCL_CUSTOM_H
