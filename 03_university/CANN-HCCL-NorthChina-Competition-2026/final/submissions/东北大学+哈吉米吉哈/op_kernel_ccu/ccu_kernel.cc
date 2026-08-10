/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ccu_kernel.h"

namespace ops_hccl {
namespace {
// channel 上的变量槽位编号与 notify bit 编号（与 host 侧 desc.notifyNum = 3 匹配）
constexpr int OUTPUT_XN_ID = 1; // 对端 output 地址变量槽位
constexpr int TOKEN_XN_ID = 2;  // 对端 token 变量槽位
constexpr int CKE_IDX_0 = 0;    // notify 索引
constexpr int POST_SYNC_ID = 3; // 后同步 notify bit
constexpr uint16_t LOCAL_BOUNDARY_MASK =
    static_cast<uint16_t>(uint32_t{1} << 15);

constexpr uint64_t MakeBitMask(uint16_t highBit)
{
    return (uint64_t{1} << (highBit + 1)) - 1;
}

constexpr uint64_t PackLoopParam(uint64_t loopCtxId, uint64_t addrOffset, uint64_t loopIterNum)
{
    constexpr uint16_t ctxIdHighBit = 8;
    constexpr uint16_t ctxIdShift = 45;
    constexpr uint16_t addrHighBit = 32;
    constexpr uint16_t addrShift = 13;
    constexpr uint16_t loopNumHighBit = 13;
    return ((loopCtxId & MakeBitMask(ctxIdHighBit)) << ctxIdShift) |
           ((addrOffset & MakeBitMask(addrHighBit)) << addrShift) |
           (loopIterNum & MakeBitMask(loopNumHighBit));
}

constexpr uint64_t PackParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
{
    constexpr uint16_t repeatHighBit = 7;
    constexpr uint16_t repeatNumShift = 55;
    constexpr uint16_t repeatLoopShift = 48;
    constexpr uint16_t totalLoopShift = 41;
    return ((repeatNum & MakeBitMask(repeatHighBit)) << repeatNumShift) |
           ((repeatLoopIndex & MakeBitMask(repeatHighBit)) << repeatLoopShift) |
           ((totalLoopNum & MakeBitMask(repeatHighBit)) << totalLoopShift);
}

constexpr uint64_t PackOffsetParam(uint64_t addrOffset, uint64_t msOffset, uint64_t eventOffset)
{
    constexpr uint16_t addrHighBit = 32;
    constexpr uint16_t addrShift = 21;
    constexpr uint16_t msHighBit = 11;
    constexpr uint16_t msShift = 10;
    constexpr uint16_t eventHighBit = 10;
    return ((addrOffset & MakeBitMask(addrHighBit)) << addrShift) |
           ((msOffset & MakeBitMask(msHighBit)) << msShift) |
           (eventOffset & MakeBitMask(eventHighBit));
}

using AllGatherKernelArg = CcuKernelArgAllGatherMesh1DMem2Mem;

void BindChannelVariables(
    const AllGatherKernelArg &kernelArg,
    std::vector<ccu::Variable> &output,
    std::vector<ccu::Variable> &token)
{
    for (uint32_t channelIndex = 0;
         channelIndex < kernelArg.channelCount; channelIndex++) {
        const uint32_t peerRank = kernelArg.rankIds[channelIndex];
        output[peerRank] = ccu::GetResByChannel<ccu::Variable>(
            kernelArg.channels[channelIndex], OUTPUT_XN_ID);
        token[peerRank] = ccu::GetResByChannel<ccu::Variable>(
            kernelArg.channels[channelIndex], TOKEN_XN_ID);
    }
}

CcuResult ExchangeChannelVariables(
    const AllGatherKernelArg &kernelArg,
    const ccu::Variable &localOutput,
    const ccu::Variable &localToken)
{
    for (uint32_t channelIndex = 0;
         channelIndex < kernelArg.channelCount; channelIndex++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            kernelArg.channels[channelIndex], localOutput, OUTPUT_XN_ID,
            CKE_IDX_0, 1 << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            kernelArg.channels[channelIndex], localToken, TOKEN_XN_ID,
            CKE_IDX_0, 1 << TOKEN_XN_ID));
    }
    const uint16_t preSyncBits = static_cast<uint16_t>(
        (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID));
    for (uint32_t channelIndex = 0;
         channelIndex < kernelArg.channelCount; channelIndex++) {
        CCU_CHK_RET(ccu::NotifyWait(
            kernelArg.channels[channelIndex], CKE_IDX_0, preSyncBits));
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult SynchronizeAllChannels(const AllGatherKernelArg &kernelArg)
{
    const uint16_t postSyncBit = static_cast<uint16_t>(1U << POST_SYNC_ID);
    for (uint32_t channelIndex = 0;
         channelIndex < kernelArg.channelCount; channelIndex++) {
        CCU_CHK_RET(ccu::NotifyRecord(
            kernelArg.channels[channelIndex], CKE_IDX_0, postSyncBit));
    }
    for (uint32_t channelIndex = 0;
         channelIndex < kernelArg.channelCount; channelIndex++) {
        CCU_CHK_RET(ccu::NotifyWait(
            kernelArg.channels[channelIndex], CKE_IDX_0, postSyncBit));
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult SynchronizeOneChannel(
    const AllGatherKernelArg &kernelArg, uint32_t channelIndex)
{
    const uint16_t postSyncBit = static_cast<uint16_t>(1U << POST_SYNC_ID);
    CCU_CHK_RET(ccu::NotifyRecord(
        kernelArg.channels[channelIndex], CKE_IDX_0, postSyncBit));
    CCU_CHK_RET(ccu::NotifyWait(
        kernelArg.channels[channelIndex], CKE_IDX_0, postSyncBit));
    return CcuResult::CCU_SUCCESS;
}

void SetupBroadcastTargets(
    const AllGatherKernelArg &kernelArg,
    const std::vector<ccu::Variable> &output,
    const std::vector<ccu::Variable> &token,
    uint32_t rankId,
    const ccu::Variable &baseOffset,
    const ccu::Variable &extraOffset,
    bool addExtraOffset,
    ccu::LocalAddr &localDst,
    std::vector<ccu::RemoteAddr> &remoteDst)
{
    localDst.addr = output[rankId];
    localDst.addr += baseOffset;
    if (addExtraOffset) {
        localDst.addr += extraOffset;
    }
    localDst.token = token[rankId];
    for (uint32_t channelIndex = 0;
         channelIndex < kernelArg.channelCount; channelIndex++) {
        const uint32_t peerRank = kernelArg.rankIds[channelIndex];
        remoteDst[channelIndex].addr = output[peerRank];
        remoteDst[channelIndex].addr += baseOffset;
        if (addExtraOffset) {
            remoteDst[channelIndex].addr += extraOffset;
        }
        remoteDst[channelIndex].token = token[peerRank];
    }
}

CcuResult WriteRelaySegment(
    const AllGatherKernelArg &kernelArg,
    const std::vector<ccu::Variable> &output,
    const std::vector<ccu::Variable> &token,
    uint32_t rankId,
    uint32_t relayChannelIndex,
    const ccu::Variable &input,
    const ccu::Variable &slotOffset,
    const ccu::Variable &size,
    ccu::Event &event,
    uint16_t eventBit)
{
    const uint32_t relayPeerRank =
        kernelArg.rankIds[relayChannelIndex];
    ccu::LocalAddr src;
    src.addr = input;
    src.token = token[rankId];
    ccu::RemoteAddr dst;
    dst.addr = output[relayPeerRank];
    dst.addr += slotOffset;
    dst.token = token[relayPeerRank];
    return ccu::Write(
        kernelArg.channels[relayChannelIndex], dst, src,
        size, event, eventBit);
}

// 同一个 Kernel 内的多次广播复用同一组 Event 和 MS。Mesh 额外录制一套
// 不写回本 rank 的 Loop body，两个版本仍复用同一组 16 个 LoopEngine，
// 避免中继数据已经位于正确槽位时再次写回 HBM。
class GroupBroadcastEngine {
public:
    GroupBroadcastEngine(
        const CcuKernelArgAllGatherMesh1DMem2Mem &kernelArg,
        bool handleSelfRank,
        uint32_t channelIssueOffset = 0)
        : kernelArg_(kernelArg),
          handleSelfRank_(handleSelfRank),
          channelIssueOffset_(
              kernelArg.channelCount == 0
                  ? 0
                  : channelIssueOffset % kernelArg.channelCount),
          completedEvents_(CUSTOM_GROUP_BROADCAST_LOOP_COUNT),
          buffers_(CUSTOM_GROUP_BROADCAST_LOOP_COUNT *
                   CUSTOM_GROUP_BROADCAST_MS_INTERLEAVE)
    {
        InitMasksAndDestinations();
        for (uint32_t loopIndex = 0; loopIndex < 2; loopIndex++) {
            InitBroadcastLoop(loopIndex);
            if (handleSelfRank_) {
                InitRemoteOnlyLoop(loopIndex);
            }
        }
    }

    CcuResult Run(
        ccu::LocalAddr localDst,
        std::vector<ccu::RemoteAddr> remoteDst,
        ccu::LocalAddr src,
        ccu::Variable addrOffset,
        ccu::Variable loopIterNum,
        ccu::Variable parallelParam,
        ccu::Variable residual)
    {
        return RunInternal(
            localDst, remoteDst, src, addrOffset, loopIterNum,
            parallelParam, residual, false);
    }

    CcuResult RunWithoutSelf(
        const ccu::LocalAddr &localDst,
        const std::vector<ccu::RemoteAddr> &remoteDst,
        const ccu::LocalAddr &src,
        const ccu::Variable &addrOffset,
        const ccu::Variable &loopIterNum,
        const ccu::Variable &parallelParam,
        const ccu::Variable &residual)
    {
        // Clos 的普通 body 本身就不处理本 rank；只有 Mesh 需要切换到
        // 专用 remote-only body。
        return RunInternal(
            localDst, remoteDst, src, addrOffset, loopIterNum,
            parallelParam, residual, handleSelfRank_);
    }

    // 为 Checker 的 LoopGroup 并行拆分提供同队列本地任务边界。
    // 复用广播 Event 的空闲高位，不引入数据搬运或跨 rank 同步。
    CcuResult RunLocalBoundaryAnchor()
    {
        const ccu::Event boundaryEvent = completedEvents_[0];
        CCU_CHK_RET(
            ccu::EventRecord(boundaryEvent, LOCAL_BOUNDARY_MASK));
        CCU_CHK_RET(
            ccu::EventWait(boundaryEvent, LOCAL_BOUNDARY_MASK));
        return CcuResult::CCU_SUCCESS;
    }

private:
    CcuResult RunInternal(
        ccu::LocalAddr localDst,
        std::vector<ccu::RemoteAddr> remoteDst,
        ccu::LocalAddr src,
        ccu::Variable addrOffset,
        ccu::Variable loopIterNum,
        ccu::Variable parallelParam,
        ccu::Variable residual,
        bool useRemoteOnlyLoops)
    {
        const bool setupLocalDst =
            handleSelfRank_ && !useRemoteOnlyLoops;
        CCU_IF(addrOffset != 0)
        {
            CCU_CHK_RET(RunFullLoops(
                localDst, remoteDst, src, loopIterNum,
                setupLocalDst, useRemoteOnlyLoops));
        }
        CCU_IF(parallelParam != 0)
        {
            CCU_CHK_RET(RunTailLoops(
                localDst, remoteDst, src, addrOffset, parallelParam,
                residual, setupLocalDst, useRemoteOnlyLoops));
        }
        return CcuResult::CCU_SUCCESS;
    }

    void InitMasksAndDestinations()
    {
        for (uint32_t loopIndex = 0; loopIndex < 2; loopIndex++) {
            loopRemoteDst_[loopIndex].resize(kernelArg_.channelCount);
        }
        remoteCompletedMask_ = static_cast<uint16_t>(
            (uint32_t{1} << kernelArg_.channelCount) - 1);
        completedMask_ = remoteCompletedMask_;
        if (handleSelfRank_) {
            completedMask_ = static_cast<uint16_t>(
                completedMask_ |
                (uint32_t{1} << kernelArg_.channelCount));
        }
    }

    void PrepareBuffer(
        uint32_t loopIndex, uint32_t bufferIndex,
        const ccu::Event &completedEvent)
    {
        ccu::LocalCopy(
            buffers_[bufferIndex], loopSrc_[loopIndex],
            loopLen_[loopIndex], completedEvent, 1);
        ccu::EventWait(completedEvent, 1);
    }

    void IssueRemoteWrites(
        uint32_t loopIndex, uint32_t bufferIndex,
        const ccu::Event &completedEvent)
    {
        for (uint32_t issueIndex = 0;
             issueIndex < kernelArg_.channelCount; issueIndex++) {
            uint32_t channelIndex = issueIndex + channelIssueOffset_;
            if (channelIndex >= kernelArg_.channelCount) {
                channelIndex -= kernelArg_.channelCount;
            }
            ccu::Write(
                kernelArg_.channels[channelIndex],
                loopRemoteDst_[loopIndex][channelIndex],
                buffers_[bufferIndex], loopLen_[loopIndex], completedEvent,
                static_cast<uint16_t>(uint32_t{1} << channelIndex));
        }
    }

    void InitBroadcastLoop(uint32_t loopIndex)
    {
        const uint32_t bufferIndex =
            loopIndex * CUSTOM_GROUP_BROADCAST_MS_INTERLEAVE;
        const ccu::Event completedEvent = completedEvents_[loopIndex];
        broadcastBodies_[loopIndex].reset(new ccu::Func(
            [this, loopIndex, bufferIndex, completedEvent]() {
                PrepareBuffer(loopIndex, bufferIndex, completedEvent);
                IssueRemoteWrites(loopIndex, bufferIndex, completedEvent);
                if (handleSelfRank_) {
                    ccu::LocalCopy(
                        loopLocalDst_[loopIndex], buffers_[bufferIndex],
                        loopLen_[loopIndex], completedEvent,
                        static_cast<uint16_t>(
                            uint32_t{1} << kernelArg_.channelCount));
                }
                ccu::EventWait(completedEvent, completedMask_);
            }));
        broadcastLoops_[loopIndex].reset(
            new ccu::Loop(loopCfg_[loopIndex], *broadcastBodies_[loopIndex]));
    }

    void InitRemoteOnlyLoop(uint32_t loopIndex)
    {
        const uint32_t bufferIndex =
            loopIndex * CUSTOM_GROUP_BROADCAST_MS_INTERLEAVE;
        const ccu::Event completedEvent = completedEvents_[loopIndex];
        remoteOnlyBodies_[loopIndex].reset(new ccu::Func(
            [this, loopIndex, bufferIndex, completedEvent]() {
                PrepareBuffer(loopIndex, bufferIndex, completedEvent);
                IssueRemoteWrites(loopIndex, bufferIndex, completedEvent);
                ccu::EventWait(completedEvent, remoteCompletedMask_);
            }));
        remoteOnlyLoops_[loopIndex].reset(
            new ccu::Loop(loopCfg_[loopIndex], *remoteOnlyBodies_[loopIndex]));
    }

    CcuResult RunFullLoops(
        const ccu::LocalAddr &localDst,
        const std::vector<ccu::RemoteAddr> &remoteDst,
        const ccu::LocalAddr &src,
        const ccu::Variable &loopIterNum,
        bool setupLocalDst,
        bool useRemoteOnlyLoops)
    {
        ccu::Variable packedLoopCfg;
        packedLoopCfg = PackLoopParam(
            0, CUSTOM_GROUP_BROADCAST_BLOCK_SIZE *
                   CUSTOM_GROUP_BROADCAST_LOOP_COUNT, 0);
        packedLoopCfg += loopIterNum;
        ccu::Variable sliceSize;
        sliceSize = CUSTOM_GROUP_BROADCAST_BLOCK_SIZE;
        SetupLoop(0, src, localDst, remoteDst, sliceSize, setupLocalDst);
        loopCfg_[0] = packedLoopCfg;
        ccu::Variable packedParallelCfg;
        packedParallelCfg = PackParallelParam(
            CUSTOM_GROUP_BROADCAST_LOOP_COUNT - 1, 0, 1);
        ccu::Variable packedOffsetCfg;
        packedOffsetCfg = PackOffsetParam(
            CUSTOM_GROUP_BROADCAST_BLOCK_SIZE,
            CUSTOM_GROUP_BROADCAST_MS_INTERLEAVE, 1);
        std::vector<ccu::Loop> groupLoops{
            SelectLoop(0, useRemoteOnlyLoops)};
        ccu::LoopGroup group(
            packedParallelCfg, packedOffsetCfg,
            CUSTOM_GROUP_BROADCAST_LOOP_COUNT, groupLoops);
        return CcuResult::CCU_SUCCESS;
    }

    void AdvanceAddresses(
        ccu::LocalAddr &localDst,
        std::vector<ccu::RemoteAddr> &remoteDst,
        ccu::LocalAddr &src,
        const ccu::Variable &offset,
        bool setupLocalDst)
    {
        src.addr += offset;
        if (setupLocalDst) {
            localDst.addr += offset;
        }
        for (uint32_t channelIndex = 0;
             channelIndex < kernelArg_.channelCount; channelIndex++) {
            remoteDst[channelIndex].addr += offset;
        }
    }

    CcuResult RunTailLoops(
        ccu::LocalAddr localDst,
        std::vector<ccu::RemoteAddr> remoteDst,
        ccu::LocalAddr src,
        const ccu::Variable &addrOffset,
        ccu::Variable &parallelParam,
        const ccu::Variable &residual,
        bool setupLocalDst,
        bool useRemoteOnlyLoops)
    {
        AdvanceAddresses(
            localDst, remoteDst, src, addrOffset, setupLocalDst);
        SetupLoop(0, src, localDst, remoteDst, residual, setupLocalDst);
        AdvanceAddresses(
            localDst, remoteDst, src, residual, setupLocalDst);
        ccu::Variable sliceSize;
        sliceSize = CUSTOM_GROUP_BROADCAST_BLOCK_SIZE;
        SetupLoop(1, src, localDst, remoteDst, sliceSize, setupLocalDst);
        loopCfg_[0] = PackLoopParam(0, 0, 1);
        loopCfg_[1] = PackLoopParam(0, 0, 1);
        ccu::Variable packedOffsetCfg;
        packedOffsetCfg = PackOffsetParam(
            CUSTOM_GROUP_BROADCAST_BLOCK_SIZE,
            CUSTOM_GROUP_BROADCAST_MS_INTERLEAVE, 1);
        std::vector<ccu::Loop> groupLoops{
            SelectLoop(0, useRemoteOnlyLoops),
            SelectLoop(1, useRemoteOnlyLoops)};
        ccu::LoopGroup group(
            parallelParam, packedOffsetCfg,
            CUSTOM_GROUP_BROADCAST_LOOP_COUNT, groupLoops);
        return CcuResult::CCU_SUCCESS;
    }

    ccu::Loop &SelectLoop(uint32_t loopIndex, bool useRemoteOnlyLoops)
    {
        if (useRemoteOnlyLoops) {
            return *remoteOnlyLoops_[loopIndex];
        }
        return *broadcastLoops_[loopIndex];
    }

    void SetupLoop(
        uint32_t loopIndex,
        const ccu::LocalAddr &src,
        const ccu::LocalAddr &localDst,
        const std::vector<ccu::RemoteAddr> &remoteDst,
        const ccu::Variable &size,
        bool setupLocalDst)
    {
        loopSrc_[loopIndex].addr = src.addr;
        loopSrc_[loopIndex].token = src.token;
        if (setupLocalDst) {
            loopLocalDst_[loopIndex].addr = localDst.addr;
            loopLocalDst_[loopIndex].token = localDst.token;
        }
        for (uint32_t i = 0; i < remoteDst.size(); i++) {
            loopRemoteDst_[loopIndex][i].addr = remoteDst[i].addr;
            loopRemoteDst_[loopIndex][i].token = remoteDst[i].token;
        }
        loopLen_[loopIndex] = size;
    }

    const CcuKernelArgAllGatherMesh1DMem2Mem &kernelArg_;
    bool handleSelfRank_;
    uint32_t channelIssueOffset_;
    uint16_t remoteCompletedMask_;
    uint16_t completedMask_;
    ccu::Array<ccu::Event> completedEvents_;
    ccu::Array<ccu::CcuBuffer> buffers_;
    ccu::LocalAddr loopSrc_[2];
    ccu::LocalAddr loopLocalDst_[2];
    std::vector<ccu::RemoteAddr> loopRemoteDst_[2];
    ccu::Variable loopLen_[2];
    ccu::Variable loopCfg_[2];
    std::unique_ptr<ccu::Func> broadcastBodies_[2];
    std::unique_ptr<ccu::Loop> broadcastLoops_[2];
    std::unique_ptr<ccu::Func> remoteOnlyBodies_[2];
    std::unique_ptr<ccu::Loop> remoteOnlyLoops_[2];
};

class SmallReadExecutor {
public:
    explicit SmallReadExecutor(const AllGatherKernelArg &kernelArg)
        : kernelArg_(kernelArg),
          rankId_(kernelArg.rankId),
          input_(static_cast<uint32_t>(kernelArg.rankSize)),
          token_(static_cast<uint32_t>(kernelArg.rankSize))
    {
    }

    CcuResult Run()
    {
        BindChannelVariables(kernelArg_, input_, token_);
        CCU_CHK_RET(LoadArguments());
        CCU_CHK_RET(ExchangeChannelVariables(
            kernelArg_, input_[rankId_], token_[rankId_]));
        ccu::Event completedEvent;
        uint16_t completedMask = 0;
        CCU_CHK_RET(ReadPeers(completedEvent, completedMask));
        if (kernelArg_.ifHandleSelfRank == 1) {
            CCU_CHK_RET(CopySelf(completedEvent, completedMask));
        }
        CCU_CHK_RET(ccu::EventWait(completedEvent, completedMask));
        return CcuResult::CCU_SUCCESS;
    }

private:
    CcuResult LoadArguments()
    {
        uint32_t argId = 0;
        CCU_CHK_RET(ccu::LoadArg(input_[rankId_], argId++));
        CCU_CHK_RET(ccu::LoadArg(output_, argId++));
        CCU_CHK_RET(ccu::LoadArg(token_[rankId_], argId++));
        CCU_CHK_RET(ccu::LoadArg(sliceSize_, argId++));
        CCU_CHK_RET(ccu::LoadArg(outputStride_, argId++));
        return CcuResult::CCU_SUCCESS;
    }

    CcuResult ReadPeers(
        ccu::Event &completedEvent, uint16_t &completedMask)
    {
        ccu::Variable peerOutputOffset;
        peerOutputOffset = 0;
        uint32_t previousPeerRank = 0;
        for (uint32_t channelIndex = 0;
             channelIndex < kernelArg_.channelCount; channelIndex++) {
            const uint32_t peerRank = kernelArg_.rankIds[channelIndex];
            for (uint32_t offsetRank = previousPeerRank;
                 offsetRank < peerRank; offsetRank++) {
                peerOutputOffset += outputStride_;
            }
            previousPeerRank = peerRank;
            ccu::LocalAddr localDst;
            localDst.addr = output_;
            localDst.addr += peerOutputOffset;
            localDst.token = token_[rankId_];
            ccu::RemoteAddr remoteSrc;
            remoteSrc.addr = input_[peerRank];
            remoteSrc.token = token_[peerRank];
            const uint16_t eventBit = static_cast<uint16_t>(
                uint32_t{1} << channelIndex);
            CCU_CHK_RET(ccu::Read(
                kernelArg_.channels[channelIndex], localDst, remoteSrc,
                sliceSize_, completedEvent, eventBit));
            completedMask = static_cast<uint16_t>(completedMask | eventBit);
        }
        return CcuResult::CCU_SUCCESS;
    }

    CcuResult CopySelf(
        ccu::Event &completedEvent, uint16_t &completedMask)
    {
        ccu::Variable selfOutputOffset;
        selfOutputOffset = 0;
        for (uint32_t offsetRank = 0;
             offsetRank < rankId_; offsetRank++) {
            selfOutputOffset += outputStride_;
        }
        ccu::LocalAddr localSrc;
        localSrc.addr = input_[rankId_];
        localSrc.token = token_[rankId_];
        ccu::LocalAddr localDst;
        localDst.addr = output_;
        localDst.addr += selfOutputOffset;
        localDst.token = token_[rankId_];
        const uint16_t eventBit = static_cast<uint16_t>(
            uint32_t{1} << kernelArg_.channelCount);
        CCU_CHK_RET(ccu::LocalCopy(
            localDst, localSrc, sliceSize_, completedEvent, eventBit));
        completedMask = static_cast<uint16_t>(completedMask | eventBit);
        return CcuResult::CCU_SUCCESS;
    }

    const AllGatherKernelArg &kernelArg_;
    uint32_t rankId_;
    std::vector<ccu::Variable> input_;
    std::vector<ccu::Variable> token_;
    ccu::Variable output_;
    ccu::Variable sliceSize_;
    ccu::Variable outputStride_;
};

class DirectExecutor {
public:
    explicit DirectExecutor(const AllGatherKernelArg &kernelArg)
        : kernelArg_(kernelArg),
          rankId_(kernelArg.rankId),
          output_(static_cast<uint32_t>(kernelArg.rankSize)),
          token_(static_cast<uint32_t>(kernelArg.rankSize))
    {
    }

    CcuResult Run()
    {
        BindChannelVariables(kernelArg_, output_, token_);
        CCU_CHK_RET(LoadArguments());
        CCU_CHK_RET(ExchangeChannelVariables(
            kernelArg_, output_[rankId_], token_[rankId_]));
        CCU_CHK_RET(Broadcast());
        CCU_CHK_RET(SynchronizeAllChannels(kernelArg_));
        return CcuResult::CCU_SUCCESS;
    }

private:
    CcuResult LoadArguments()
    {
        uint32_t argId = 0;
        CCU_CHK_RET(ccu::LoadArg(input_, argId++));
        CCU_CHK_RET(ccu::LoadArg(output_[rankId_], argId++));
        CCU_CHK_RET(ccu::LoadArg(token_[rankId_], argId++));
        CCU_CHK_RET(ccu::LoadArg(inputOffset_, argId++));
        CCU_CHK_RET(ccu::LoadArg(outputOffset_, argId++));
        CCU_CHK_RET(ccu::LoadArg(sliceSize_, argId++));
        CCU_CHK_RET(ccu::LoadArg(goAddrOffset_, argId++));
        CCU_CHK_RET(ccu::LoadArg(goLoopParam_, argId++));
        CCU_CHK_RET(ccu::LoadArg(goParallelParam_, argId++));
        CCU_CHK_RET(ccu::LoadArg(goResidual_, argId++));
        return CcuResult::CCU_SUCCESS;
    }

    CcuResult Broadcast()
    {
        ccu::LocalAddr src;
        src.addr = input_;
        src.addr += inputOffset_;
        src.token = token_[rankId_];
        ccu::LocalAddr localDst;
        std::vector<ccu::RemoteAddr> remoteDst(kernelArg_.channelCount);
        SetupBroadcastTargets(
            kernelArg_, output_, token_, rankId_, outputOffset_,
            outputOffset_, false, localDst, remoteDst);
        GroupBroadcastEngine broadcastEngine(
            kernelArg_, kernelArg_.ifHandleSelfRank == 1);
        return broadcastEngine.Run(
            localDst, remoteDst, src, goAddrOffset_, goLoopParam_,
            goParallelParam_, goResidual_);
    }

    const AllGatherKernelArg &kernelArg_;
    uint32_t rankId_;
    std::vector<ccu::Variable> output_;
    std::vector<ccu::Variable> token_;
    ccu::Variable input_;
    ccu::Variable inputOffset_;
    ccu::Variable outputOffset_;
    ccu::Variable sliceSize_;
    ccu::Variable goAddrOffset_;
    ccu::Variable goLoopParam_;
    ccu::Variable goParallelParam_;
    ccu::Variable goResidual_;
};

CcuResult LoadAxisBaseArguments(
    ccu::Variable &input,
    ccu::Variable &output,
    ccu::Variable &token,
    ccu::Variable &stage,
    uint32_t &argId)
{
    CCU_CHK_RET(ccu::LoadArg(input, argId++));
    CCU_CHK_RET(ccu::LoadArg(output, argId++));
    CCU_CHK_RET(ccu::LoadArg(token, argId++));
    CCU_CHK_RET(ccu::LoadArg(stage, argId++));
    return CcuResult::CCU_SUCCESS;
}

CcuResult LoadBroadcastGoArguments(
    ccu::Variable *addrOffset,
    ccu::Variable *loopParam,
    ccu::Variable *parallelParam,
    ccu::Variable *residual,
    uint32_t goSizeCount,
    uint32_t &argId)
{
    for (uint32_t goIndex = 0; goIndex < goSizeCount; goIndex++) {
        CCU_CHK_RET(ccu::LoadArg(addrOffset[goIndex], argId++));
        CCU_CHK_RET(ccu::LoadArg(loopParam[goIndex], argId++));
        CCU_CHK_RET(ccu::LoadArg(parallelParam[goIndex], argId++));
        CCU_CHK_RET(ccu::LoadArg(residual[goIndex], argId++));
    }
    return CcuResult::CCU_SUCCESS;
}

class TwoByEightExecutor {
public:
    explicit TwoByEightExecutor(const AllGatherKernelArg &kernelArg)
        : kernelArg_(kernelArg),
          rankId_(kernelArg.rankId),
          output_(static_cast<uint32_t>(kernelArg.rankSize)),
          token_(static_cast<uint32_t>(kernelArg.rankSize))
    {
        for (uint32_t localRank = 0;
             localRank < RANKS_PER_SERVER; localRank++) {
            closChannelByLocalRank_[localRank] = MAX_RANK_SIZE;
        }
    }

    CcuResult Run()
    {
        CCU_CHK_RET(ValidateChannels());
        BindChannelVariables(kernelArg_, output_, token_);
        CCU_CHK_RET(BuildClosChannelMap());
        CCU_CHK_RET(LoadArguments());
        CCU_IF(stage_ != CUSTOM_AXIS_STAGE_2)
        {
            CCU_CHK_RET(ExchangeChannelVariables(
                kernelArg_, output_[rankId_], token_[rankId_]));
        }
        GroupBroadcastEngine engine(
            kernelArg_, kernelArg_.axisId == CUSTOM_AXIS_MESH);
        CCU_IF(stage_ == CUSTOM_AXIS_STAGE_DIRECT)
        {
            CCU_CHK_RET(RunDirect());
        }
        CCU_IF(stage_ == CUSTOM_AXIS_STAGE_1)
        {
            CCU_CHK_RET(RunStageOne(engine));
        }
        CCU_IF(stage_ == CUSTOM_AXIS_STAGE_2)
        {
            CCU_CHK_RET(engine.RunLocalBoundaryAnchor());
            CCU_CHK_RET(RunStageTwo(engine));
        }
        CCU_CHK_RET(FinishStage());
        return CcuResult::CCU_SUCCESS;
    }

private:
    enum : uint32_t {
        RANKS_PER_SERVER = 8,
        GO_PREFIX = 0,
        GO_SUFFIX = 1,
        GO_DIRECT = 2,
        GO_BULK = 3,
        GO_SIZE_COUNT = 4,
    };

    CcuResult ValidateChannels() const
    {
        const uint32_t expectedChannelCount =
            kernelArg_.axisId == CUSTOM_AXIS_MESH ? 7U : 8U;
        if (kernelArg_.channelCount != expectedChannelCount) {
            HCCL_ERROR(
                "[CcuKernel2x8Unified] axis[%u] channelCount[%u] is invalid",
                kernelArg_.axisId, kernelArg_.channelCount);
            return CcuResult::CCU_E_INTERNAL;
        }
        return CcuResult::CCU_SUCCESS;
    }

    CcuResult BuildClosChannelMap()
    {
        if (kernelArg_.axisId != CUSTOM_AXIS_CLOS) {
            return CcuResult::CCU_SUCCESS;
        }
        for (uint32_t channelIndex = 0;
             channelIndex < kernelArg_.channelCount; channelIndex++) {
            const uint32_t localRank =
                kernelArg_.rankIds[channelIndex] % RANKS_PER_SERVER;
            closChannelByLocalRank_[localRank] = channelIndex;
        }
        for (uint32_t localRank = 0;
             localRank < RANKS_PER_SERVER; localRank++) {
            if (closChannelByLocalRank_[localRank] == MAX_RANK_SIZE) {
                HCCL_ERROR(
                    "[CcuKernel2x8Unified] no Clos channel for localRank[%u]",
                    localRank);
                return CcuResult::CCU_E_INTERNAL;
            }
        }
        return CcuResult::CCU_SUCCESS;
    }

    CcuResult LoadArguments()
    {
        uint32_t argId = 0;
        CCU_CHK_RET(LoadAxisBaseArguments(
            input_, output_[rankId_], token_[rankId_], stage_, argId));
        CCU_CHK_RET(ccu::LoadArg(sliceSize_, argId++));
        CCU_CHK_RET(ccu::LoadArg(directSize_, argId++));
        CCU_CHK_RET(ccu::LoadArg(prefixSize_, argId++));
        CCU_CHK_RET(ccu::LoadArg(suffixSize_, argId++));
        CCU_CHK_RET(ccu::LoadArg(bulkSize_, argId++));
        CCU_CHK_RET(ccu::LoadArg(selfSlotOffset_, argId++));
        CCU_CHK_RET(ccu::LoadArg(relaySlotOffset_, argId++));
        return LoadBroadcastGoArguments(
            goAddrOffset_, goLoopParam_, goParallelParam_, goResidual_,
            GO_SIZE_COUNT, argId);
    }

    CcuResult RunDirect()
    {
        ccu::Event completedEvent;
        ccu::LocalAddr src;
        src.addr = input_;
        src.token = token_[rankId_];
        uint16_t completedMask = 0;
        for (uint32_t channelIndex = 0;
             channelIndex < kernelArg_.channelCount; channelIndex++) {
            const uint32_t peerRank = kernelArg_.rankIds[channelIndex];
            ccu::RemoteAddr dst;
            dst.addr = output_[peerRank];
            dst.addr += selfSlotOffset_;
            dst.token = token_[peerRank];
            const uint16_t eventBit = static_cast<uint16_t>(
                uint32_t{1} << channelIndex);
            CCU_CHK_RET(ccu::Write(
                kernelArg_.channels[channelIndex], dst, src, sliceSize_,
                completedEvent, eventBit));
            completedMask = static_cast<uint16_t>(completedMask | eventBit);
        }
        CCU_CHK_RET(CopyDirectSelf(src, completedEvent, completedMask));
        CCU_CHK_RET(ccu::EventWait(completedEvent, completedMask));
        return CcuResult::CCU_SUCCESS;
    }

    CcuResult CopyDirectSelf(
        const ccu::LocalAddr &src,
        ccu::Event &completedEvent,
        uint16_t &completedMask)
    {
        if (kernelArg_.ifHandleSelfRank != 1) {
            return CcuResult::CCU_SUCCESS;
        }
        ccu::LocalAddr localDst;
        localDst.addr = output_[rankId_];
        localDst.addr += selfSlotOffset_;
        localDst.token = token_[rankId_];
        const uint16_t eventBit = static_cast<uint16_t>(
            uint32_t{1} << kernelArg_.channelCount);
        CCU_CHK_RET(ccu::LocalCopy(
            localDst, src, sliceSize_, completedEvent, eventBit));
        completedMask = static_cast<uint16_t>(completedMask | eventBit);
        return CcuResult::CCU_SUCCESS;
    }

    CcuResult RunStageOne(GroupBroadcastEngine &engine)
    {
        if (kernelArg_.axisId == CUSTOM_AXIS_MESH) {
            return RunStageOneMesh(engine);
        }
        return RunStageOneClos(engine);
    }

    CcuResult RunStageOneMesh(GroupBroadcastEngine &engine)
    {
        ccu::LocalAddr src;
        src.addr = input_;
        src.token = token_[rankId_];
        ccu::LocalAddr localDst;
        std::vector<ccu::RemoteAddr> remoteDst(kernelArg_.channelCount);
        SetupBroadcastTargets(
            kernelArg_, output_, token_, rankId_, selfSlotOffset_,
            selfSlotOffset_, false, localDst, remoteDst);
        CCU_CHK_RET(engine.Run(
            localDst, remoteDst, src, goAddrOffset_[GO_PREFIX],
            goLoopParam_[GO_PREFIX], goParallelParam_[GO_PREFIX],
            goResidual_[GO_PREFIX]));
        return engine.RunLocalBoundaryAnchor();
    }

    CcuResult RunStageOneClos(GroupBroadcastEngine &engine)
    {
        ccu::Event directEvent;
        const uint32_t localRank = rankId_ % RANKS_PER_SERVER;
        const uint32_t relayChannelIndex =
            closChannelByLocalRank_[localRank];
        constexpr uint16_t relayEventBit = 1;
        CCU_CHK_RET(WriteRelaySegment(
            kernelArg_, output_, token_, rankId_, relayChannelIndex,
            input_, selfSlotOffset_, prefixSize_, directEvent,
            relayEventBit));
        CCU_CHK_RET(BroadcastStageOneDirect(engine));
        return ccu::EventWait(directEvent, relayEventBit);
    }

    CcuResult BroadcastStageOneDirect(GroupBroadcastEngine &engine)
    {
        return BroadcastOwnSuffix(engine, GO_DIRECT);
    }

    CcuResult BroadcastOwnSuffix(
        GroupBroadcastEngine &engine, uint32_t goIndex)
    {
        ccu::LocalAddr src;
        src.addr = input_;
        src.addr += prefixSize_;
        src.token = token_[rankId_];
        ccu::LocalAddr localDst;
        std::vector<ccu::RemoteAddr> remoteDst(kernelArg_.channelCount);
        SetupBroadcastTargets(
            kernelArg_, output_, token_, rankId_, selfSlotOffset_,
            prefixSize_, true, localDst, remoteDst);
        return engine.Run(
            localDst, remoteDst, src, goAddrOffset_[goIndex],
            goLoopParam_[goIndex], goParallelParam_[goIndex],
            goResidual_[goIndex]);
    }

    CcuResult RunStageTwo(GroupBroadcastEngine &engine)
    {
        if (kernelArg_.axisId == CUSTOM_AXIS_MESH) {
            return RunStageTwoMesh(engine);
        }
        return RunStageTwoClos(engine);
    }

    CcuResult RunStageTwoMesh(GroupBroadcastEngine &engine)
    {
        CCU_CHK_RET(BroadcastOwnSuffix(engine, GO_SUFFIX));
        ccu::LocalAddr localDst;
        std::vector<ccu::RemoteAddr> remoteDst(kernelArg_.channelCount);
        ccu::LocalAddr src;
        return BroadcastRelayPrefix(engine, localDst, remoteDst, src);
    }

    CcuResult BroadcastRelayPrefix(
        GroupBroadcastEngine &engine,
        ccu::LocalAddr &localDst,
        std::vector<ccu::RemoteAddr> &remoteDst,
        ccu::LocalAddr &src)
    {
        src.addr = output_[rankId_];
        src.addr += relaySlotOffset_;
        src.token = token_[rankId_];
        SetupBroadcastTargets(
            kernelArg_, output_, token_, rankId_, relaySlotOffset_,
            relaySlotOffset_, false, localDst, remoteDst);
        return engine.RunWithoutSelf(
            localDst, remoteDst, src, goAddrOffset_[GO_PREFIX],
            goLoopParam_[GO_PREFIX], goParallelParam_[GO_PREFIX],
            goResidual_[GO_PREFIX]);
    }

    CcuResult RunStageTwoClos(GroupBroadcastEngine &engine)
    {
        ccu::Variable bulkOffset;
        bulkOffset = prefixSize_;
        bulkOffset += directSize_;
        ccu::LocalAddr src;
        src.addr = input_;
        src.addr += bulkOffset;
        src.token = token_[rankId_];
        ccu::LocalAddr localDst;
        std::vector<ccu::RemoteAddr> remoteDst(kernelArg_.channelCount);
        SetupBroadcastTargets(
            kernelArg_, output_, token_, rankId_, selfSlotOffset_,
            bulkOffset, true, localDst, remoteDst);
        return engine.Run(
            localDst, remoteDst, src, goAddrOffset_[GO_BULK],
            goLoopParam_[GO_BULK], goParallelParam_[GO_BULK],
            goResidual_[GO_BULK]);
    }

    CcuResult FinishStage()
    {
        CCU_IF(stage_ != CUSTOM_AXIS_STAGE_1)
        {
            CCU_CHK_RET(SynchronizeAllChannels(kernelArg_));
        }
        if (kernelArg_.axisId == CUSTOM_AXIS_CLOS) {
            const uint32_t relayChannelIndex =
                closChannelByLocalRank_[rankId_ % RANKS_PER_SERVER];
            CCU_IF(stage_ == CUSTOM_AXIS_STAGE_1)
            {
                CCU_CHK_RET(SynchronizeOneChannel(
                    kernelArg_, relayChannelIndex));
            }
        }
        return CcuResult::CCU_SUCCESS;
    }

    const AllGatherKernelArg &kernelArg_;
    uint32_t rankId_;
    uint32_t closChannelByLocalRank_[RANKS_PER_SERVER];
    std::vector<ccu::Variable> output_;
    std::vector<ccu::Variable> token_;
    ccu::Variable input_;
    ccu::Variable stage_;
    ccu::Variable sliceSize_;
    ccu::Variable directSize_;
    ccu::Variable prefixSize_;
    ccu::Variable suffixSize_;
    ccu::Variable bulkSize_;
    ccu::Variable selfSlotOffset_;
    ccu::Variable relaySlotOffset_;
    ccu::Variable goAddrOffset_[GO_SIZE_COUNT];
    ccu::Variable goLoopParam_[GO_SIZE_COUNT];
    ccu::Variable goParallelParam_[GO_SIZE_COUNT];
    ccu::Variable goResidual_[GO_SIZE_COUNT];
};

class EightByFourExecutor {
public:
    explicit EightByFourExecutor(const AllGatherKernelArg &kernelArg)
        : kernelArg_(kernelArg),
          rankId_(kernelArg.rankId),
          inLargeServer_(kernelArg.rankId < LARGE_SERVER_SIZE),
          output_(static_cast<uint32_t>(kernelArg.rankSize)),
          token_(static_cast<uint32_t>(kernelArg.rankSize)),
          relayChannelIndex_(MAX_RANK_SIZE)
    {
    }

    CcuResult Run()
    {
        CCU_CHK_RET(ValidateChannels());
        BindChannelVariables(kernelArg_, output_, token_);
        CCU_CHK_RET(FindRelayChannel());
        CCU_CHK_RET(LoadArguments());
        CCU_IF(stage_ != CUSTOM_AXIS_STAGE_2)
        {
            CCU_CHK_RET(ExchangeChannelVariables(
                kernelArg_, output_[rankId_], token_[rankId_]));
        }
        const uint32_t localRankId =
            inLargeServer_ ? rankId_ : rankId_ - LARGE_SERVER_SIZE;
        const uint32_t channelIssueOffset =
            kernelArg_.axisId == CUSTOM_AXIS_CLOS ? localRankId : 0;
        GroupBroadcastEngine engine(
            kernelArg_, kernelArg_.axisId == CUSTOM_AXIS_MESH,
            channelIssueOffset);
        CCU_IF(stage_ == CUSTOM_AXIS_STAGE_DIRECT)
        {
            CCU_CHK_RET(BroadcastInputSegment(
                engine, selfSlotOffset_, selfSlotOffset_, false,
                selfSlotOffset_, false, GO_DIRECT_OR_LATE));
        }
        CCU_IF(stage_ == CUSTOM_AXIS_STAGE_1)
        {
            CCU_CHK_RET(RunStageOne(engine));
        }
        CCU_IF(stage_ == CUSTOM_AXIS_STAGE_2)
        {
            if (kernelArg_.axisId == CUSTOM_AXIS_MESH) {
                CCU_CHK_RET(RunStageTwoMesh(engine));
            } else {
                CCU_CHK_RET(RunStageTwoClos(engine));
            }
        }
        CCU_CHK_RET(FinishStage());
        return CcuResult::CCU_SUCCESS;
    }

private:
    enum : uint32_t {
        LARGE_SERVER_SIZE = 8,
        GO_DIRECT_OR_LATE = 0,
        GO_PIECE0 = 1,
        GO_PIECE1 = 2,
        GO_EARLY_CLOS = 3,
        GO_SIZE_COUNT = 4,
    };

    CcuResult ValidateChannels() const
    {
        const uint32_t expectedChannelCount =
            kernelArg_.axisId == CUSTOM_AXIS_MESH
                ? (inLargeServer_ ? 7U : 3U)
                : (inLargeServer_ ? 4U : 8U);
        if (kernelArg_.channelCount != expectedChannelCount) {
            HCCL_ERROR(
                "[CcuKernel8x4Unified] rank[%u] axis[%u] channelCount[%u] is invalid",
                rankId_, kernelArg_.axisId, kernelArg_.channelCount);
            return CcuResult::CCU_E_INTERNAL;
        }
        return CcuResult::CCU_SUCCESS;
    }

    bool HandlesRelay() const
    {
        return rankId_ < 4 || rankId_ >= LARGE_SERVER_SIZE;
    }

    CcuResult FindRelayChannel()
    {
        if (kernelArg_.axisId != CUSTOM_AXIS_CLOS || !HandlesRelay()) {
            return CcuResult::CCU_SUCCESS;
        }
        const uint32_t relayRank =
            rankId_ < 4 ? rankId_ + LARGE_SERVER_SIZE
                        : rankId_ - LARGE_SERVER_SIZE;
        for (uint32_t channelIndex = 0;
             channelIndex < kernelArg_.channelCount; channelIndex++) {
            if (kernelArg_.rankIds[channelIndex] == relayRank) {
                relayChannelIndex_ = channelIndex;
                break;
            }
        }
        if (relayChannelIndex_ == MAX_RANK_SIZE) {
            HCCL_ERROR(
                "[CcuKernel8x4Unified] rank[%u] has no relay channel", rankId_);
            return CcuResult::CCU_E_INTERNAL;
        }
        return CcuResult::CCU_SUCCESS;
    }

    CcuResult LoadArguments()
    {
        uint32_t argId = 0;
        CCU_CHK_RET(LoadAxisBaseArguments(
            input_, output_[rankId_], token_[rankId_], stage_, argId));
        CCU_CHK_RET(ccu::LoadArg(sliceSize_, argId++));
        CCU_CHK_RET(ccu::LoadArg(piece0Size_, argId++));
        CCU_CHK_RET(ccu::LoadArg(earlyClosSize_, argId++));
        CCU_CHK_RET(ccu::LoadArg(selfSlotOffset_, argId++));
        CCU_CHK_RET(ccu::LoadArg(relaySlotOffset_, argId++));
        return LoadBroadcastGoArguments(
            goAddrOffset_, goLoopParam_, goParallelParam_, goResidual_,
            GO_SIZE_COUNT, argId);
    }

    CcuResult BroadcastInputSegment(
        GroupBroadcastEngine &engine,
        const ccu::Variable &sourceOffset,
        const ccu::Variable &targetBaseOffset,
        bool addSourceOffset,
        const ccu::Variable &targetExtraOffset,
        bool addTargetExtraOffset,
        uint32_t goIndex)
    {
        ccu::LocalAddr src;
        src.addr = input_;
        if (addSourceOffset) {
            src.addr += sourceOffset;
        }
        src.token = token_[rankId_];
        ccu::LocalAddr localDst;
        std::vector<ccu::RemoteAddr> remoteDst(kernelArg_.channelCount);
        SetupBroadcastTargets(
            kernelArg_, output_, token_, rankId_, targetBaseOffset,
            targetExtraOffset, addTargetExtraOffset, localDst, remoteDst);
        return engine.Run(
            localDst, remoteDst, src, goAddrOffset_[goIndex],
            goLoopParam_[goIndex], goParallelParam_[goIndex],
            goResidual_[goIndex]);
    }

    CcuResult RunStageOne(GroupBroadcastEngine &engine)
    {
        if (kernelArg_.axisId == CUSTOM_AXIS_MESH) {
            CCU_CHK_RET(BroadcastInputSegment(
                engine, selfSlotOffset_, selfSlotOffset_, false,
                selfSlotOffset_, false, GO_PIECE0));
            return engine.RunLocalBoundaryAnchor();
        }
        if (HandlesRelay()) {
            return RunStageOneRelay(engine);
        }
        if (inLargeServer_) {
            CCU_CHK_RET(BroadcastInputSegment(
                engine, selfSlotOffset_, selfSlotOffset_, false,
                selfSlotOffset_, false, GO_EARLY_CLOS));
            return engine.RunLocalBoundaryAnchor();
        }
        return CcuResult::CCU_SUCCESS;
    }

    CcuResult RunStageOneRelay(GroupBroadcastEngine &engine)
    {
        ccu::Event relayEvent;
        constexpr uint16_t relayEventBit = 1;
        CCU_CHK_RET(WriteRelaySegment(
            kernelArg_, output_, token_, rankId_, relayChannelIndex_,
            input_, selfSlotOffset_, piece0Size_, relayEvent,
            relayEventBit));
        if (rankId_ < 4) {
            CCU_CHK_RET(BroadcastInputSegment(
                engine, piece0Size_, selfSlotOffset_, true,
                piece0Size_, true, GO_PIECE1));
        }
        return ccu::EventWait(relayEvent, relayEventBit);
    }

    CcuResult RunStageTwoMesh(GroupBroadcastEngine &engine)
    {
        CCU_CHK_RET(engine.RunLocalBoundaryAnchor());
        CCU_CHK_RET(BroadcastInputSegment(
            engine, piece0Size_, selfSlotOffset_, true,
            piece0Size_, true, GO_PIECE1));
        if (!HandlesRelay()) {
            return CcuResult::CCU_SUCCESS;
        }
        ccu::LocalAddr src;
        src.addr = output_[rankId_];
        src.addr += relaySlotOffset_;
        src.token = token_[rankId_];
        ccu::LocalAddr localDst;
        std::vector<ccu::RemoteAddr> remoteDst(kernelArg_.channelCount);
        SetupBroadcastTargets(
            kernelArg_, output_, token_, rankId_, relaySlotOffset_,
            relaySlotOffset_, false, localDst, remoteDst);
        return engine.RunWithoutSelf(
            localDst, remoteDst, src, goAddrOffset_[GO_PIECE0],
            goLoopParam_[GO_PIECE0], goParallelParam_[GO_PIECE0],
            goResidual_[GO_PIECE0]);
    }

    CcuResult RunStageTwoClos(GroupBroadcastEngine &engine)
    {
        if (rankId_ >= 4) {
            CCU_CHK_RET(engine.RunLocalBoundaryAnchor());
        }
        if (inLargeServer_ && rankId_ >= 4) {
            CCU_CHK_RET(BroadcastInputSegment(
                engine, earlyClosSize_, selfSlotOffset_, true,
                earlyClosSize_, true, GO_DIRECT_OR_LATE));
        }
        if (!inLargeServer_) {
            CCU_CHK_RET(BroadcastInputSegment(
                engine, piece0Size_, selfSlotOffset_, true,
                piece0Size_, true, GO_PIECE1));
        }
        return CcuResult::CCU_SUCCESS;
    }

    CcuResult FinishStage()
    {
        CCU_IF(stage_ != CUSTOM_AXIS_STAGE_1)
        {
            CCU_CHK_RET(SynchronizeAllChannels(kernelArg_));
        }
        if (kernelArg_.axisId == CUSTOM_AXIS_CLOS &&
            relayChannelIndex_ != MAX_RANK_SIZE) {
            CCU_IF(stage_ == CUSTOM_AXIS_STAGE_1)
            {
                CCU_CHK_RET(SynchronizeOneChannel(
                    kernelArg_, relayChannelIndex_));
            }
        }
        return CcuResult::CCU_SUCCESS;
    }

    const AllGatherKernelArg &kernelArg_;
    uint32_t rankId_;
    bool inLargeServer_;
    std::vector<ccu::Variable> output_;
    std::vector<ccu::Variable> token_;
    uint32_t relayChannelIndex_;
    ccu::Variable input_;
    ccu::Variable stage_;
    ccu::Variable sliceSize_;
    ccu::Variable piece0Size_;
    ccu::Variable earlyClosSize_;
    ccu::Variable selfSlotOffset_;
    ccu::Variable relaySlotOffset_;
    ccu::Variable goAddrOffset_[GO_SIZE_COUNT];
    ccu::Variable goLoopParam_[GO_SIZE_COUNT];
    ccu::Variable goParallelParam_[GO_SIZE_COUNT];
    ccu::Variable goResidual_[GO_SIZE_COUNT];
};
} // namespace

CcuResult CcuKernelSmallRead(CcuKernelArg arg)
{
    auto *kernelArg =
        static_cast<CcuKernelArgAllGatherMesh1DMem2Mem *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0 ||
        kernelArg->channelCount >= 16) {
        HCCL_ERROR(
            "[CcuKernelSmallRead] kernelArg is null or channelCount[%u] is invalid",
            kernelArg == nullptr ? 0 : kernelArg->channelCount);
        return CcuResult::CCU_E_INTERNAL;
    }
    return SmallReadExecutor(*kernelArg).Run();
}

CcuResult CcuKernelDirect(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllGatherMesh1DMem2Mem *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelDirect] kernelArg is null or channelCount is 0");
        return CcuResult::CCU_E_INTERNAL;
    }
    return DirectExecutor(*kernelArg).Run();
}

CcuResult CcuKernel2x8Unified(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllGatherMesh1DMem2Mem *>(arg);
    if (kernelArg == nullptr || kernelArg->rankSize != 16 ||
        (kernelArg->axisId != CUSTOM_AXIS_MESH &&
         kernelArg->axisId != CUSTOM_AXIS_CLOS)) {
        HCCL_ERROR("[CcuKernel2x8Unified] invalid kernelArg");
        return CcuResult::CCU_E_INTERNAL;
    }
    return TwoByEightExecutor(*kernelArg).Run();
}


CcuResult CcuKernel8x4Unified(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllGatherMesh1DMem2Mem *>(arg);
    if (kernelArg == nullptr || kernelArg->rankSize != 12 ||
        (kernelArg->axisId != CUSTOM_AXIS_MESH &&
         kernelArg->axisId != CUSTOM_AXIS_CLOS)) {
        HCCL_ERROR("[CcuKernel8x4Unified] invalid kernelArg");
        return CcuResult::CCU_E_INTERNAL;
    }
    return EightByFourExecutor(*kernelArg).Run();
}


CcuResult CcuKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllGatherMesh1DMem2Mem *>(arg);
    if (kernelArg == nullptr) {
        HCCL_ERROR("[CcuKernel] kernelArg is null");
        return CcuResult::CCU_E_INTERNAL;
    }

    // axisId 与 rankSize 是注册期固化值；专用路径仍保证每个 Die 只录制一个
    // 统一 Kernel，从而复用同一套 LoopGroup 资源。
    if (kernelArg->axisId == CUSTOM_AXIS_NONE) {
        return CcuKernelDirect(arg);
    }
    if (kernelArg->rankSize == 16) {
        return CcuKernel2x8Unified(arg);
    }
    if (kernelArg->rankSize == 12) {
        return CcuKernel8x4Unified(arg);
    }
    HCCL_ERROR("[CcuKernel] unsupported specialized rankSize[%llu]",
               static_cast<unsigned long long>(kernelArg->rankSize));
    return CcuResult::CCU_E_INTERNAL;
}
} // namespace ops_hccl
