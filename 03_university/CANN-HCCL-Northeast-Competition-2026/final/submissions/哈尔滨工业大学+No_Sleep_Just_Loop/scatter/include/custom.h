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

#include <memory>
#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

// 单个 CCU kernel 最多绑定的 channel 数(rankSize 最大 16, 去掉自身)
constexpr uint32_t CUSTOM_MAX_CHANNELS = MAX_RANK_SIZE - 1;

// 执行路径。选路只在 Host 侧做一次, 结果写入 AlgResourceCtx 并随 ctx 缓存;
// ExecOp 直接按它分发, 不再用 sendBytes 二次推导 —— 二次推导需要重新探测拓扑,
// 两处判据一旦不同步就会把 taskArgs 下发给布局不匹配的 kernel。
enum class ScatterPath : uint32_t {
    ROOT_STAR = 0,      ///< 小包(Push): root 直发 + 单向 DONE, 非 root 只连 root
    WAVE = 1,           ///< 大包直发(保留 two-wave 分段能力; 4x1 等非 mixed)
    PROXY = 2,          ///< 跨服块经本 Server peer 的空闲 Clos 口分担
    ROOT_STAR_PULL = 3, ///< unused (Pull 已删); 保留数值以免 WAVE_STAR 位移
    WAVE_STAR = 4,      ///< 大包 Root-Star 控制面: Root 连所有 Peer; Peer 只连 Root + 单向 DONE
};

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount;
};

// ============================================================================
// 各执行路径使用独立的 kernelArg 结构, 互不共享字段、互不影响:
//   路径 1 RootStar : 512KB    -> CcuKernelArgRootStar(Push)
//   路径 2 Wave     : 大包非 mixed (4x1) -> CcuKernelArgWave
//   路径 3 Proxy    : 具备有效分流的 Clos-dominant (2x8 / 4x3) -> CcuKernelArgProxy
//   路径 4 WaveStar : 大包 Mesh-dominant mixed (8+4) -> CcuKernelArgWaveStar
// 真正的选路由 Host SelectPath 按拓扑能力写入 AlgResourceCtx.path, 不按绝对字节数硬编码。
// ============================================================================

// ---- 路径 1: RootStar ------------------------------------------------------
// Root 向每个 Peer 写块并单向发 DONE; Peer 只连 Root 一条 channel。
// channel 分配: Root 连所有对端; 非 Root 只连 Root。
struct CcuKernelArgRootStar : public CcuKernelArgBase {
    uint32_t peerRanks[CUSTOM_MAX_CHANNELS] = {};

    // RootStar 的 512KB 固定布局在 kernel 注册期计算，运行期直接复用。
    uint64_t peerOffsets[CUSTOM_MAX_CHANNELS] = {};
    uint64_t sliceBytes = 0;
    uint64_t selfOffset = 0;

    uint32_t rankSize = 0;
    uint32_t rankId = 0;
    uint32_t root = 0;
    // 自留块只能由一个 kernel 拷贝, 否则两个 kernel 并发写同一段 recvBuf,
    // CheckerV3 报 [ErrorCode: 302] Memory conflict。
    bool doSelfCopy = false;
    // E4 SMALL 专用: Host 已在 Launch 前拷完自留块, Small kernel 跳过 LocalCopy。
    // LARGE 保持默认 false, LARGE kernel 不读该字段。
    bool skipDeviceSelfCopy = false;
};

// ---- 路径 2: Wave ---------------------------------------------------------
// 把一个 Slice 拆成两段连续 wave(覆盖 CCU 单次传输上限), 两段都先提交再等待,
// 源与目的区间互不重叠, 链路在 wave0 排空时仍可继续跑。
// host 与 kernel 必须使用同一套 launch 参数布局, 见 ScatterKernelArgId。
enum ScatterKernelArgId : uint32_t {
    SCATTER_ARG_INPUT = 0,
    SCATTER_ARG_INPUT_TOKEN,
    SCATTER_ARG_OUTPUT,
    SCATTER_ARG_OUTPUT_TOKEN,
    SCATTER_ARG_SLICE_BYTES,
    SCATTER_ARG_CHUNK0_BYTES,
    SCATTER_ARG_CHUNK1_BYTES,
    SCATTER_KERNEL_ARG_COUNT,
};

struct CcuKernelArgWave : public CcuKernelArgBase {
    uint32_t peerRanks[CUSTOM_MAX_CHANNELS] = {};
    uint32_t rankSize = 0;
    uint32_t rankId = 0;
    uint32_t root = 0;
    bool doSelfCopy = false;
};

// ---- 路径 4: Wave-Star ----------------------------------------------------
// Scatter 数据面仍是 Root 直发, 但控制面从组内 All-to-All 收成 Root-Star:
//   Root : 按真实 Mesh/Clos 分组连所有 Peer, Ready->Write 流水, 单向 DONE
//   Peer : 只连 Root, 只 publish recvBuf, 等 DONE, 不回 ACK
// Round D: peerOffsets / selfOffset / sliceBytes 在注册期固化, Kernel 不再循环累加。
// Single-Wave: sliceBytes <= MAX_DATA_SIZE 时走专用 fast path; 更大消息仍走 two-wave。
// Host taskArgs 布局不变(仍传 slice/chunk0/chunk1), 外层 Wave/Proxy context tag 也不改。
enum ScatterWaveStarArgId : uint32_t {
    WAVESTAR_ARG_INPUT = 0,
    WAVESTAR_ARG_INPUT_TOKEN,
    WAVESTAR_ARG_OUTPUT,
    WAVESTAR_ARG_OUTPUT_TOKEN,
    WAVESTAR_ARG_SLICE_BYTES,
    WAVESTAR_ARG_CHUNK0_BYTES,
    WAVESTAR_ARG_CHUNK1_BYTES,
    WAVESTAR_KERNEL_ARG_COUNT,
};

struct CcuKernelArgWaveStar : public CcuKernelArgBase {
    uint32_t peerRanks[CUSTOM_MAX_CHANNELS] = {};

    uint64_t peerOffsets[CUSTOM_MAX_CHANNELS] = {};
    uint64_t sliceBytes = 0;
    uint64_t selfOffset = 0;

    uint32_t rankSize = 0;
    uint32_t rankId = 0;
    uint32_t root = 0;

    bool doSelfCopy = false;
    bool singleWave = false;
};

// ---- 路径 3: Proxy --------------------------------------------------------
// 让本 Server 的 peer 用各自的 Clos 口分担 root 的跨服负载。
// 动机: root 那一个 Clos 物理口是瓶颈(多条 channel 共享其带宽), 而 Server 内
//       其他 peer 的 Clos 口是空闲的。把跨服块的尾段经 Mesh 送到本 Server peer,
//       再由它用自己的 Clos 口转发出去。
// channel 分配: root 连所有对端; 代转者连 root + 它的转发目标; 其他非 root 只连 root。
struct CcuKernelArgProxy : public CcuKernelArgBase {
    uint32_t peerRanks[CUSTOM_MAX_CHANNELS] = {};
    uint32_t rankSize = 0;
    uint32_t rankId = 0;
    uint32_t root = 0;
    bool doSelfCopy = false;
    // 本 kernel 属于 Clos 组(host 侧显式给出, 不用 peer 类型反推)
    bool isClosGroup = false;
    // 代理是否启用。注册期固定且所有 rank 一致, 决定 PreSync 是否交换中转 buffer 地址。
    bool proxyEnabled = false;
    // 语义随角色而变:
    //   root     -> proxyTargets[t] 是第 t 个被代转的跨服 rank, proxyPeers[t] 是它的代转者
    //   代转者   -> proxyTargets[] 列出本 rank 要转发到的跨服 rank
    //   其他     -> 均为空
    uint32_t proxyTargetCount = 0;
    uint32_t proxyTargets[CUSTOM_MAX_CHANNELS] = {};
    uint32_t proxyPeerCount = 0;
    uint32_t proxyPeers[CUSTOM_MAX_CHANNELS] = {};

    // 4x3 专用: 代转者的"收 -> 转"改走"分块 + host 分阶段"流水。
    //
    // 动机: 原来 host 侧串行下发, Clos 内核必须等 Mesh 内核"整体"完成, 于是
    //       代转者耗时 = 收 + 转。分块后 host 按阶段下发: 阶段 k 里 Mesh 收第 k 块、
    //       Clos 同时转第 k-1 块, 把"转"藏进"收"里。阶段顺序由 host 保证,
    //       因此**不需要任何跨 Die 会合**(跨 Die 本地通知在本 VM 上不可用)。
    //
    // 为 false 时(2x8 及其余拓扑)本结构的所有路径与改动前逐字节一致。
    bool multiStageRelay = false;

    // 400M 档去串行化: 把"代理尾段投递"从数据面里摘出来, 让它先上 Mesh 链路。
    //
    // 为什么必须拆成两个 kernel 而不是在一个 kernel 里调换语句顺序:
    //   2026-09-16 平台实测确认, 在同一个 kernel 内把尾段的 Write 挪到直发块之前
    //   对时序毫无影响(798 -> 798), 硬件/翻译层并不按源码顺序决定链路占用次序。
    //   只有 kernel 边界(前一个 kernel 的 EventWait 必须完成才结束)才能给出硬保证。
    //
    // CCU 要求一个 kernel 的 channel 同 Die, 所以跨 Die 会合仍然只能由 host 的
    // thread-notify 承担 —— 这里拆的是**同 Die 内部**的先后, 不违反该约束。
    //
    //   MESH_PHASE_FULL  -> 不拆, 与改动前逐字节一致(root 的 512MB / 普通 rank)
    //   MESH_PHASE_SEED  -> root: 只做前同步 + 投递代理尾段, 结束得早
    //   MESH_PHASE_LOCAL -> root: 前同步 + 直发块 + 自留块 + 后同步
    //   MESH_PHASE_TAIL  -> 代转者: 前同步 + 等尾段就绪, 结束得早, 用来提前放行 Clos kernel
    //   MESH_PHASE_SYNC  -> 代转者: 只做后同步栅栏, 保证本 rank 自己那块数据已落位
    enum MeshPhase : uint32_t {
        MESH_PHASE_FULL = 0,
        MESH_PHASE_SEED = 1,
        MESH_PHASE_LOCAL = 2,
        MESH_PHASE_TAIL = 3,
        MESH_PHASE_SYNC = 4
    };
    uint32_t meshPhase = MESH_PHASE_FULL;

    // P2 (Ready->Write, 仅 root 的 Mesh SEED 内核): 不再"整组等齐"后再发第一个尾段,
    // 而是把前同步的等待折进中转尾段投递循环 —— 等一条就写一条。与 root Clos 的 A0 同模式。
    // 只对 4x3 中档(sendBytes < 500MB)开启; 4x3 512MB / 2x8 / 8+4 / 4x1 默认关闭。
    // 本字段只存在于 Proxy kernelArg, 不改 WAVE_STAR ABI / LoadArg 数量。
    bool seedReadyWrite = false;

    // 多目标覆盖(仅 4x3 中档): 每个代转者代转的目标个数。k==1 时路径与合入前逐字节一致。
    uint32_t targetsPerForwarder = 1;
};


// 路径 3 分块流水的阶段号。仅当 multiStageRelay 为真且本 rank 是代转者时被读取。
enum ScatterProxyStage : uint32_t {
    SCATTER_PROXY_STAGE_CHUNK0 = 0, ///< Mesh 收第 0 块; Clos 只做前同步
    SCATTER_PROXY_STAGE_CHUNK1 = 1, ///< Mesh 收第 1 块; Clos 转第 0 块
    SCATTER_PROXY_STAGE_TAIL = 2,   ///< Clos 转第 1 块并做后同步
    SCATTER_PROXY_STAGE_NONE = 3    ///< 非代转者: 阶段号无意义
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

// 三条路径共用的资源上下文(只承载资源句柄, 不含任何路径逻辑)
struct AlgResourceCtx {
    ThreadHandle ccuThread;            ///< CCU通信引擎上的thread资源
    CommBuffer localBuffer;            ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads; ///< CCU通信引擎上的thread资源
    std::vector<CcuKernelHandle> ccuKernels;
    uint32_t meshKernelIndex = INVALID_VALUE_RANKID; ///< Mesh 组 kernel 在 ccuKernels 中的下标
    uint32_t closKernelIndex = INVALID_VALUE_RANKID; ///< Clos 组 kernel 在 ccuKernels 中的下标
    /// 去串行化: Mesh 组被拆成两个 kernel 时, 后半(或 root 的 LOCAL 半)的下标。
    /// INVALID 表示没拆(此时 meshKernelIndex 那个就是完整的 Mesh kernel)。
    uint32_t meshKernelBIndex = INVALID_VALUE_RANKID;
    uint32_t path = static_cast<uint32_t>(ScatterPath::ROOT_STAR); ///< 本次选定的执行路径

    // 路径 3 专用: 代理比例与代转者标记
    bool proxyEnabled = false;
    uint32_t proxyNumerator = 0;
    uint32_t proxyDenominator = 1;
    bool isForwarder = false;
    // 路径 3 分块流水(仅 4x3 为真): host 侧据此走"分阶段下发"而不是串行下发。
    bool multiStageRelay = false;
    // 路径 3 多目标覆盖(仅 4x3 中档): 每个代转者代转几个目标。ExecProxy 据此把每目标尾段再除以 k。
    uint32_t proxyTargetsPerForwarder = 1;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << ccuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << ccuKernels;
        binaryStream << meshKernelIndex;
        binaryStream << closKernelIndex;
        binaryStream << meshKernelBIndex;
        binaryStream << path;
        binaryStream << proxyEnabled;
        binaryStream << proxyNumerator;
        binaryStream << proxyDenominator;
        binaryStream << isForwarder;
        binaryStream << multiStageRelay;
        binaryStream << proxyTargetsPerForwarder;
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
        binaryStream >> meshKernelIndex;
        binaryStream >> closKernelIndex;
        binaryStream >> meshKernelBIndex;
        binaryStream >> path;
        binaryStream >> proxyEnabled;
        binaryStream >> proxyNumerator;
        binaryStream >> proxyDenominator;
        binaryStream >> isForwarder;
        binaryStream >> multiStageRelay;
        binaryStream >> proxyTargetsPerForwarder;
    }
};

#endif // OPS_HCCL_CUSTOM_H
