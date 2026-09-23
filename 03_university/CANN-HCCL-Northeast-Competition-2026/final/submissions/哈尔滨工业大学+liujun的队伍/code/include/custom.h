/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SCATTER_CUSTOM_H_486
#define OPS_HCCL_SCATTER_CUSTOM_H_486

#include <memory>
#include <cstdlib>
#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount;
};

// ccu kernel register所需信息
struct CcuKernelInfo {
    // kernel名称
    char kernelFuncName[64];
    // kernel函数
    void *kernelFunc;
    // KernelArg实例指针
    void *kernelArg;

private:
    std::shared_ptr<CcuKernelArgBase> kernelArgSmartPtr;

public:
    template <typename T> void setKernelArg(std::shared_ptr<T> arg)
    {
        kernelArgSmartPtr = std::static_pointer_cast<CcuKernelArgBase>(arg);
        kernelArg = static_cast<void *>(arg.get());
    }
};

// 通信引擎上下文（**布局必须与基线逐字节一致**：判分环境的 engine-ctx 只可靠回读基线尺寸以内的内容，
// 2026-09-17 线上实测。v6 的所有算法规划都固化进 kernelArg，运行期不再读 ctx 里的任何新增字段。）
struct AlgResourceCtx {
    ThreadHandle ccuThread;            ///< CCU通信引擎上的thread资源
    CommBuffer localBuffer;            ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads; ///< CCU通信引擎上的thread资源
    std::vector<CcuKernelHandle> ccuKernels;
    /// rank -> 该对端所在通道组下标（组 = 同一网络层/同一IO Die 的 channel 集合）；
    /// 0xFF/INVALID 表示本 rank 未与该校建立 channel。下发期用于判定"root 在本 rank 的哪一组"，
    /// 以及推导本次调用里自己的角色（是否中继卡、要不要双 kernel）。
    uint32_t groupOfRank[MAX_RANK_SIZE];
    uint32_t groupCount;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << ccuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << ccuKernels;
        for (uint32_t i = 0; i < MAX_RANK_SIZE; i++) {
            binaryStream << groupOfRank[i];
        }
        binaryStream << groupCount;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // 反序列化
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> ccuThread;
        binaryStream >> localBuffer;
        binaryStream >> threads;
        binaryStream >> ccuKernels;
        for (uint32_t i = 0; i < MAX_RANK_SIZE; i++) {
            binaryStream >> groupOfRank[i];
        }
        binaryStream >> groupCount;
    }
};

// 通道分组：同一网络层（同一 IO Die）上的对端 channel 集合，对应一个 CCU Kernel。
struct ChannelGroup {
    uint32_t layer = 0;                      ///< 命中网络层号（升序，最低层 = Server 内 Mesh 直连）
    uint32_t dieId = 0;                      ///< 组内本地 endpoint 所属 IO Die
    std::vector<uint32_t> ranks;             ///< 组内对端 rank（升序，不含自己）
    std::vector<ChannelHandle> channels;     ///< 与 ranks 一一对应
};

// 中继规划（Host 注册期算好；**不进 AlgResourceCtx**，只用于注册期固化的取值与 kernel 角色选择）。
// 约定：mesh = Server 内 Mesh 直连层（组序已由 GroupChannelsByLayer 规范化到 groups[0]），
//       clos = 跨 Server 层。k = min(对端在同 Server 的卡数 M, 跨 Server 的卡数 C)。
//   root 把前 k 个跨 Server 对端（按 rank 升序）的那一份切成 α + β：α 自己直推、β 交给同 Server 的
//   第 j 个兄弟卡（rank 升序第 j 个）转发，使 Server 内直连链路与 root 的 Clos 上行同时到终点：
//       β = S·(C−4)/(k+4)   （4 = Clos 上行相对单条直连链路的带宽倍数）
//
// ⚠️ 下面这份推导**注册期（scatter.cc）与运行期（exec_op.cc）共用同一个函数**。
// 原因：v4 线上事故的根因就是"注册期用 layer==0 判 mesh、运行期用组序号判 mesh"，两侧判据不同源，
// 一边以为要中继、另一边按直推跑，于是功能全过、中继零收益。共用函数后这类分歧从结构上不可能发生。
// 输入是 groupOfRank/groupCount —— **基线字段**（codev3 起就在用、已被线上证明能可靠回读）。
struct ScatterTopoView {
    uint32_t meshCount = 0;              ///< M：本 rank 的同 Server 对端数
    uint32_t crossCount = 0;             ///< C：本 rank 的跨 Server 对端数
    uint32_t firstMeshPeerRank = 0xFFFFFFFFu;
    uint32_t firstCrossRank = 0xFFFFFFFFu;
    uint32_t relayCount = 0;             ///< k = min(M, C)
    uint32_t relayTopo = 0;              ///< 本 rank 视角下这次拓扑是否具备中继条件（M>0 且 C>4）
    uint32_t amRoot = 0;
    uint32_t amRelay = 0;                ///< 本 rank 是否是中继卡（root 同 Server 兄弟卡里 rank 升序前 k 个）
    uint32_t relayTargetRank = 0xFFFFFFFFu; ///< 中继卡要转发的目标 rank（跨 Server 列表里 rank 升序第 j 个）
    uint32_t groupOfRoot = 0xFFFFFFFFu;  ///< root 在本 rank 的哪一组（下发时机用）
    uint32_t groupCount = 0;
};

inline ScatterTopoView DeriveTopo(const uint32_t *groupOfRank, uint32_t groupCount, uint32_t rankSize, uint32_t myRank,
    uint32_t root)
{
    ScatterTopoView v;
    v.groupCount = groupCount;
    v.amRoot = (myRank == root) ? 1u : 0u;
    const bool twoGroups = (groupCount > 1);
    for (uint32_t r = 0; r < rankSize; r++) {
        if (r == myRank) {
            continue;
        }
        if (twoGroups && groupOfRank[r] == 0) {
            v.meshCount++;
            if (v.firstMeshPeerRank == 0xFFFFFFFFu) {
                v.firstMeshPeerRank = r; // 对端按 rank 升序建组 ⇒ 第一个命中的就是最小 rank
            }
        } else {
            v.crossCount++;
            if (v.firstCrossRank == 0xFFFFFFFFu) {
                v.firstCrossRank = r;
            }
        }
        if (r == root) {
            v.groupOfRoot = groupOfRank[r];
        }
    }
    if (v.amRoot != 0) {
        v.groupOfRoot = 0; // root 自己不在分组表里，约定用第 0 组
    }
    v.relayCount = (v.meshCount < v.crossCount) ? v.meshCount : v.crossCount;
    // ⚠️ 这个开关只判"本 rank 是否**参与中继协议的信令**"（收发 PIECE_DONE 的那套扇入扇出），
    // 判据必须是**拓扑同质**的（各 Server 之间不能各说各话）：M>0（有同 Server 对端）且 C>0（有跨
    // Server 对端）。反例：8+4 root 在 8 卡侧时，8 卡侧看到 M=7/C=4，4 卡侧看到 M=3/C=8——若把
    // "C>4"（真正的转发判据）也塞进这个开关，两侧就会一个发片通知、一个不发 ⇒ 目标侧死等。
    // 是否**真的转发**（β>0）另算：它与 (M,C,k) 同源，而在 root 与其同 Server 兄弟卡之间必然相同
    // （同 Server ⇒ 同 M/C/k），所以运行期按本 rank 自己的 crossCount 判 β 是安全的。
    v.relayTopo = (v.meshCount > 0 && v.crossCount > 0 && v.relayCount > 0) ? 1u : 0u;
    if (v.relayTopo != 0 && v.amRoot == 0 && twoGroups && groupOfRank[root] == 0) {
        // 本 rank 是 root 的同 Server 兄弟卡：序号 j = 同 Server 列表（rank 升序、去掉 root）里的位置
        uint32_t j = 0;
        for (uint32_t r = 0; r < myRank; r++) {
            if (r != root && groupOfRank[r] == 0) {
                j++;
            }
        }
        if (j < v.relayCount) {
            v.amRelay = 1;
            // 目标是"跨 Server 列表（rank 升序）"里的第 j 个
            uint32_t idx = 0;
            for (uint32_t r = 0; r < rankSize; r++) {
                if (r == myRank) {
                    continue;
                }
                if (!(twoGroups && groupOfRank[r] == 0)) {
                    if (idx == j) {
                        v.relayTargetRank = r;
                        break;
                    }
                    idx++;
                }
            }
        }
    }
    return v;
}

// Semantic argument IDs. ScatterEntryArgs packs each entry's used IDs into 0..N-1.
// Host argument count and registered LoadArg count must be identical.
enum ScatterArgId : uint32_t {
    A_RECV_BUF = 0,     // 本端输出缓冲
    A_RECV_TOKEN = 1,
    A_ROOT = 2,         // 本次 root rank
    A_SELF_OFF = 3,     // myRank × chunk + sliceOff
    A_SLICE_BYTES = 4,  // 本次实际搬运长度（单次 >256MB 时由 host 切片）
    A_SLICE_OFF = 5,    // 本次搬运在"份"内的偏移
    A_SEND_BUF = 6,     // root 的输入缓冲
    A_SEND_TOKEN = 7,
    A_CHUNK = 8,        // 整份大小（算各 rank 源偏移用）
    A_ALPHA = 9,        // 直推长度（前 k 个跨 Server 目标）
    A_BETA = 10,        // 中继片长度
    A_CCL_BASE = 11,    // 中继卡暂存区（CCL buffer）地址
    A_CCL_TOKEN = 12,
    A_TARGET_IDX = 13,  // 中继卡要转发到的目标 rank
    A_RELAY_SRC_OFF = 14, // 中转片在 root sendBuf 里的偏移
    A_FIRST_PEER_OFF = 15, // 本组最小对端 rank × chunk（运行和推进源偏移的起点）
    A_LARGE_MODE = 16,     // 1 = 大份档（≥阈值）。只有"读了自留份"的中继卡入口需要它：
                           // 小份档它是唯一内核（要把 root 发布的 A、B 两组位都收干净），
                           // 大份档它是第二段（A 组已被"读中转片"那段消费掉，只等 B 组）
    A_COUNT = 17,
};

// 入口（= 注册期选的内核函数，也是运行期选槽位的依据）
// v7：中继卡的 Server 内相位**拆成两个 kernel**（先读中转片、读完立刻放行跨 Server 转发，再去读自留份）——
// 否则转发的开始时间被"自留份也读完"拖后，目标侧要等到 中转片+自留份 全部到达才凑齐（实测正是这 7%~11% 的差）。
enum ScatterEntry : uint32_t {
    ENTRY_ROOT_PUB = 0,        // root：发布 + 自留份 + 等读方回执
    ENTRY_PEER_PULL = 1,       // 非 root：等发布 → 拉自己那份 → 回执
    ENTRY_ROOT_PUSH = 2,       // root：等对端发布 → 推送（前 k 个只推 α）
    ENTRY_RECV_LARGE = 3,      // 非 root 大份收方：发布 + 等直推/片通知
    ENTRY_RELAY_READ = 4,      // 中继卡 Server 内相位 1/2：只读中转片进暂存 → 放行转发（等发布位 A 组）
    ENTRY_RELAY_OWN = 5,       // 中继卡"读自留份"（大份档的第二段 / 小份档的唯一内核；等发布位 B 组）
    ENTRY_RELAY_FWD = 6,       // 中继卡跨 Server 相位：转发中转片
    ENTRY_COUNT = 7,
};

// 每个通道组的槽位数。⚠️ **每个 die 上能注册的内核数是有限的**：3 个就报 CCU_E_UNAVAIL（资源不足，
// 见 CcuKernelMgr::AllocRes），2 个是 v6 线上验证过的上限内。所以中继卡的"两段 Server 内相位"复用
// slot0/slot1 两个槽位（slot0 = 读自留份，两种模式都用；slot1 = 读中转片，只大份档用）。
constexpr uint32_t KERNEL_SLOTS_PER_GROUP = 2;

// Host packing and kernel loading share these semantic IDs. Wire IDs are 0..N-1.
inline const std::vector<uint32_t> &ScatterEntryArgs(ScatterEntry entry)
{
    static const std::vector<uint32_t> ids[ENTRY_COUNT] = {
        {A_RECV_BUF, A_RECV_TOKEN, A_SELF_OFF, A_SLICE_BYTES, A_SEND_BUF, A_SEND_TOKEN},
        {A_RECV_BUF, A_RECV_TOKEN, A_SELF_OFF, A_SLICE_BYTES},
        {A_RECV_BUF, A_RECV_TOKEN, A_SELF_OFF, A_SLICE_BYTES, A_SEND_BUF, A_SEND_TOKEN,
            A_CHUNK, A_ALPHA, A_BETA, A_SLICE_OFF, A_FIRST_PEER_OFF},
        {A_RECV_BUF, A_RECV_TOKEN},
        {A_CCL_BASE, A_CCL_TOKEN, A_RELAY_SRC_OFF, A_BETA},
        {A_RECV_BUF, A_RECV_TOKEN, A_SELF_OFF, A_SLICE_BYTES, A_LARGE_MODE},
        {A_CCL_BASE, A_CCL_TOKEN, A_ALPHA, A_BETA},
    };
    return ids[entry];
}

// 由"角色 + 组性质 + 槽位"选入口 —— **注册期（scatter.cc）与运行期（exec_op.cc）共用这一个函数**，
// 避免两侧各写一张表（v4 的事故就是两侧判据不同源）。slot0 = 小份、slot1 = 大份。
inline ScatterEntry PickScatterEntry(bool amRoot, bool amRelay, bool isMeshGroup, bool relayTopo, uint32_t slot)
{
    if (slot == 0) {
        if (amRoot) {
            return ENTRY_ROOT_PUB;
        }
        if (amRelay) {
            // 中继卡的小份档就是"读自留份"：与它在大份档的第二段是同一个内核（等发布位 B 组）
            return isMeshGroup ? ENTRY_RELAY_OWN : ENTRY_RELAY_FWD;
        }
        return ENTRY_PEER_PULL;
    }
    if (amRoot) {
        // 中继拓扑下 root 的 Server 内相位只发布（各中继卡自己来读），跨 Server 相位才推送
        return (isMeshGroup && relayTopo) ? ENTRY_ROOT_PUB : ENTRY_ROOT_PUSH;
    }
    if (amRelay) {
        return isMeshGroup ? ENTRY_RELAY_READ : ENTRY_RELAY_FWD;
    }
    if (isMeshGroup && relayTopo) {
        // 中继拓扑下的同 Server 兄弟卡：从 root 拉自己那一份
        return ENTRY_PEER_PULL;
    }
    return ENTRY_RECV_LARGE;
}

// Scatter kernel 注册期固化参数（rankSize/rankId 在通信域生命周期内不变；地址/大小走 taskArgs）
// 注：CCU 返回码转换 ConvertCcuToHccl 与检查宏 CHK_RET_CCU 由 log.h 提供，勿在此重复定义
struct CcuKernelArgScatter : public CcuKernelArgBase {
    uint32_t rootId; // Fixed registration role, as in v7. Use a fresh communicator for another root.
    uint32_t targetRank;
    uint32_t relaySourceRanks[MAX_RANK_SIZE];
    uint32_t rankSize;                    ///< 通信域rank数
    uint32_t rankId;                      ///< 本rank id
    uint32_t groupIndex;                  ///< 本 kernel 承载的通道组下标（0 = 最低层组）
    uint32_t hasLocalSlice;               ///< 本 kernel 是否负责 root 自留份的 LocalCopy（固定放最低层组）
    uint32_t peerRanks[MAX_RANK_SIZE];    ///< 与 channels[] 一一对应的对端 rank（升序）
    uint32_t isMeshGroup;                 ///< 本 kernel 的通道组是否为 Server 内直连（最低层）
    uint32_t relayTopo;                   ///< 本 rank 视角下这次拓扑是否具备中继条件
    uint32_t relayCount;                  ///< k：被中继的跨 Server 对端个数（按 rank 升序取前 k 个）
    uint32_t firstPeerRank;               ///< 本组最小对端 rank（kernel 里用运行和推进各对端的源偏移）
};

#endif // OPS_HCCL_SCATTER_CUSTOM_H_486
