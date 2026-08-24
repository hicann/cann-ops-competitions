/*!
 * \file chunk_scaled_dot_kkt.h
 * \brief ChunkScaledDotKkt kernel implementation.
 */

#ifndef CHUNKSCALEDDOTKKT_H
#define CHUNKSCALEDDOTKKT_H

#include "chunk_scaled_dot_kkt_tiling_data.h"
#include "chunk_scaled_dot_kkt_tiling_key.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

namespace NsChunkScaledDotKkt {

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t OUT_BUFFER_NUM = 2;
constexpr uint32_t GM_BLOCK_BYTES = 32;
constexpr uint32_t FP32_BLOCK_ELEMS = GM_BLOCK_BYTES / sizeof(float);
constexpr uint32_t FP32_COMPARE_ELEMS = 64;
constexpr uint32_t BETA_BATCH_MIN_ROWS = 4;
constexpr uint32_t WORKSPACE_SLOT_COUNT = 4;
constexpr uint32_t MANUAL_MIX_CORE_COUNT = 20;
constexpr uint32_t GROUP_ALIGNED_CORE_COUNT = 16;
constexpr uint32_t DEEP_PIPELINE_TASKS_PER_CORE = 20;
constexpr uint32_t DEEP_PIPELINE_TASK_THRESHOLD =
    MANUAL_MIX_CORE_COUNT * DEEP_PIPELINE_TASKS_PER_CORE;
constexpr uint32_t LONG_TASK_MIN = 2048;
constexpr uint32_t VERY_LONG_TASK_MIN = 4096;
constexpr uint32_t MANUAL_FULL_CHUNK_SIZE = 64;
// Factored scaling amortizes its setup from this many valid tail rows.
constexpr uint32_t FACTORED_TAIL_CROSSOVER_ROWS = 39;
constexpr float SAFE_EXP_NEG_INF = -64.0f;

__aicore__ inline float ScalarToFloat(const bfloat16_t value)
{
    return AscendC::ToFloat(value);
}

__aicore__ inline float ScalarToFloat(const float value)
{
    return value;
}

template <typename T, bool GH2_K128_PAIR_PIPELINE = false, bool HPG4_FULL_PIPELINE = false,
    bool TINY_FULL_PIPELINE = false, bool GH8_FULL_NO_COMPACT = false,
    bool K128_FULL_ALIGNED_DOT = false, bool MANUAL_K128_CUBE = false,
    bool PAIR_OUTPUT_MULTIBLOCK = false, bool ALIAS_BETA_FLOAT = false,
    bool ROLLING_OFFSETS = false, bool MANUAL_HPG4 = false,
    bool FUSED_FACTORED_BETA = false, bool MANUAL_HPG3 = false,
    bool FACTORED_SINGLE_HEAD = false, bool EARLY_RELEASE_TAIL = false,
    bool OVERLAP_TAIL_DOT = false, int32_t FACTORED_PAIR_POLICY = -1,
    uint32_t FACTORED_TAIL_MIN = 0, bool DIRECT_FACTORED_TAIL = false,
    bool LONG_SHARED_L1 = false, uint32_t FIXED_TASK_CORE_COUNT = 0,
    bool ONE_WORKER_PER_GROUP = false, bool HPG4_DOUBLE_OUT = false>
class ChunkScaledDotKkt {
public:
    using AMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, false>;
    using BMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using CMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using HighLevelMatmulType = matmul::MatmulImpl<AMatmulType, BMatmulType, CMatmulType>;
    using MatmulType = typename AscendC::Conditional<
        MANUAL_K128_CUBE, uint8_t, HighLevelMatmulType>::type;

    MatmulType mm;

    __aicore__ inline ChunkScaledDotKkt() {}

    template <typename TilingDataType>
    __aicore__ inline void Init(
        GM_ADDR k,
        GM_ADDR beta,
        GM_ADDR g_cumsum,
        GM_ADDR chunk_offsets,
        GM_ADDR A,
        GM_ADDR userWorkspace,
        TPipe* pipe,
        const TilingDataType* tilingData)
    {
        blockIdx_ = static_cast<uint32_t>(GetBlockIdx());
        subBlockIdx_ = static_cast<uint32_t>(GetSubBlockIdx());
        pipe_ = pipe;
        if (tilingData == nullptr || pipe == nullptr || userWorkspace == nullptr) {
            return;
        }
        if ASCEND_IS_AIV {
            blockIdx_ /= 2U;
        }

        seqLen_ = static_cast<uint32_t>(tilingData->seqLen);
        groupHeads_ = static_cast<uint32_t>(tilingData->groupHeads);
        keyDim_ = static_cast<uint32_t>(tilingData->keyDim);
        headPerGroup_ = static_cast<uint32_t>(tilingData->headPerGroup);
        taskNum_ = static_cast<uint32_t>(tilingData->taskNum);
        if constexpr (MANUAL_K128_CUBE) {
            heads_ = groupHeads_ * headPerGroup_;
            chunkNum_ = groupHeads_ == 0U ? 0U : taskNum_ / groupHeads_;
            chunkSize_ = MANUAL_FULL_CHUNK_SIZE;
            if constexpr (K128_FULL_ALIGNED_DOT) {
                taskCoreNum_ = static_cast<uint32_t>(tilingData->taskCoreNum);
            } else if constexpr (FIXED_TASK_CORE_COUNT > 0U) {
                taskCoreNum_ = FIXED_TASK_CORE_COUNT;
            } else {
                taskCoreNum_ = MANUAL_MIX_CORE_COUNT;
            }
            blockFactor_ = (taskNum_ + taskCoreNum_ - 1U) / taskCoreNum_;
            const uint32_t compactBytes = chunkSize_ * keyDim_ * static_cast<uint32_t>(sizeof(T));
            const uint32_t dotBytes = chunkSize_ * chunkSize_ * static_cast<uint32_t>(sizeof(float));
            workspacePerCore_ = (compactBytes + dotBytes) * WORKSPACE_SLOT_COUNT;
        } else {
            heads_ = static_cast<uint32_t>(tilingData->heads);
            chunkNum_ = static_cast<uint32_t>(tilingData->chunkNum);
            chunkSize_ = static_cast<uint32_t>(tilingData->chunkSize);
            taskCoreNum_ = static_cast<uint32_t>(tilingData->taskCoreNum);
            blockFactor_ = static_cast<uint32_t>(tilingData->blockFactor);
            workspacePerCore_ = static_cast<uint32_t>(tilingData->workspacePerCore);
        }

        if (seqLen_ == 0 || heads_ == 0 || groupHeads_ == 0 || keyDim_ == 0 || chunkNum_ == 0 ||
            chunkSize_ == 0 || headPerGroup_ == 0 || taskNum_ == 0 || taskCoreNum_ == 0 || blockFactor_ == 0 ||
            workspacePerCore_ == 0) {
            return;
        }
        const uint64_t kTotal = static_cast<uint64_t>(seqLen_) * groupHeads_ * keyDim_;
        const uint64_t bhTotal = static_cast<uint64_t>(seqLen_) * heads_;
        const uint64_t outTotal = bhTotal * chunkSize_;
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(k), kTotal);
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(beta), bhTotal);
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(g_cumsum), bhTotal);
        offsetsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(chunk_offsets), static_cast<uint64_t>(chunkNum_) + 1);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A), outTotal);
        compactGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(userWorkspace),
            static_cast<uint64_t>(workspacePerCore_) * taskCoreNum_ / sizeof(T));
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(userWorkspace),
            static_cast<uint64_t>(workspacePerCore_) * taskCoreNum_ / sizeof(float));

        compactWorkspaceBytes_ =
            AlignUp(chunkSize_ * keyDim_ * static_cast<uint32_t>(sizeof(T)), GM_BLOCK_BYTES);
        dotWorkspaceBytes_ = AlignUp(chunkSize_ * chunkSize_ * static_cast<uint32_t>(sizeof(float)), GM_BLOCK_BYTES);
        workspaceSlotBytes_ = compactWorkspaceBytes_ + dotWorkspaceBytes_;
        if (workspacePerCore_ < workspaceSlotBytes_ * WORKSPACE_SLOT_COUNT) {
            return;
        }
        if ASCEND_IS_AIC {
            if constexpr (MANUAL_K128_CUBE) {
                pipe_->InitBuffer(kL1Queue_, 1, chunkSize_ * keyDim_ * sizeof(T));
                if constexpr (!LONG_SHARED_L1) {
                    if (!(keyDim_ == 256U && headPerGroup_ == 4U)) {
                        pipe_->InitBuffer(kB1Queue_, 1, chunkSize_ * keyDim_ * sizeof(T));
                    }
                }
                pipe_->InitBuffer(kL0AQueue_, 1, chunkSize_ * keyDim_ * sizeof(T));
                pipe_->InitBuffer(kL0BQueue_, 1, chunkSize_ * keyDim_ * sizeof(T));
                pipe_->InitBuffer(dotL0CQueue_, 1, chunkSize_ * chunkSize_ * sizeof(float));
            } else {
                mm.Init(&tilingData->matmulTiling, pipe_);
            }
        }
        if ASCEND_IS_AIV {
            pipe_->InitBuffer(dotMatrixQueue_, BUFFER_NUM, chunkSize_ * chunkSize_ * sizeof(float));
            const uint32_t vectorElements = AlignUp(chunkSize_, FP32_COMPARE_ELEMS);
            const uint32_t headBatch = (GH2_K128_PAIR_PIPELINE || MANUAL_K128_CUBE) ? 2U : 1U;
            const uint32_t inputBufferCount = GH2_K128_PAIR_PIPELINE ? BUFFER_NUM : 2U;
            pipe_->InitBuffer(gQueue_, inputBufferCount, headBatch * vectorElements * sizeof(float));
            pipe_->InitBuffer(betaQueue_, inputBufferCount, headBatch * chunkSize_ * sizeof(T));
            const uint32_t compactElements = chunkSize_ * keyDim_;
            if constexpr (!TINY_FULL_PIPELINE && !GH8_FULL_NO_COMPACT &&
                !K128_FULL_ALIGNED_DOT && !MANUAL_K128_CUBE) {
                pipe_->InitBuffer(compactInQueue_, BUFFER_NUM, compactElements * sizeof(T));
                pipe_->InitBuffer(compactOutQueue_, BUFFER_NUM, compactElements * sizeof(T));
            }
            const uint32_t outBufferCount =
                (HPG4_FULL_PIPELINE && groupHeads_ == 2U && !HPG4_DOUBLE_OUT) ||
                    (GH2_K128_PAIR_PIPELINE && headPerGroup_ == 4U) ?
                3U : OUT_BUFFER_NUM;
            pipe_->InitBuffer(outQueue_, outBufferCount,
                headBatch * chunkSize_ * chunkSize_ * sizeof(float));
            if constexpr (!TINY_FULL_PIPELINE && !MANUAL_K128_CUBE &&
                !FACTORED_SINGLE_HEAD) {
                pipe_->InitBuffer(diffBuf_, vectorElements * sizeof(float));
                pipe_->InitBuffer(expBuf_, vectorElements * sizeof(float));
            }
            if constexpr (!MANUAL_K128_CUBE) {
                pipe_->InitBuffer(negGBuf_, vectorElements * sizeof(float));
            }
            pipe_->InitBuffer(gBrcbBuf_,
                headBatch * chunkSize_ * FP32_BLOCK_ELEMS * sizeof(float));
            if constexpr (FUSED_FACTORED_BETA) {
                pipe_->InitBuffer(gFactorBuf_, headBatch * vectorElements * sizeof(float));
            }
            if constexpr (!ALIAS_BETA_FLOAT) {
                pipe_->InitBuffer(betaFloatBuf_, headBatch * vectorElements * sizeof(float));
            }
            pipe_->InitBuffer(maskBuf_, headBatch * chunkSize_ * chunkSize_ / 8U);
            if constexpr (MANUAL_K128_CUBE) {
                pipe_->InitBuffer(offsetsQueue_, BUFFER_NUM, 256U * sizeof(int32_t));
            } else if (!K128_FULL_ALIGNED_DOT &&
                taskNum_ >= LONG_TASK_MIN &&
                (taskNum_ < VERY_LONG_TASK_MIN || keyDim_ == 128U)) {
                pipe_->InitBuffer(offsetsQueue_, BUFFER_NUM, 256U * sizeof(int32_t));
            }
        }
        valid_ = true;
    }

    __aicore__ inline void Process()
    {
        if (!valid_ || blockIdx_ >= taskCoreNum_) {
            return;
        }
        if constexpr (TINY_FULL_PIPELINE) {
            ProcessTinyFull();
            return;
        }
        if constexpr (GH2_K128_PAIR_PIPELINE) {
            if (groupHeads_ == 2U && keyDim_ == 128U) {
                ProcessGh2K128PairPipeline();
                return;
            }
        }
        if ASCEND_IS_AIC {
            if constexpr (!MANUAL_K128_CUBE) {
                mm.SetHF32(false, 0);
                mm.SetSingleShape(chunkSize_, chunkSize_, keyDim_);
                currentKRowStride_ = groupHeads_ * keyDim_;
                currentTailSize_ = chunkSize_;
            }
        }
        uint32_t taskStart = 0;
        uint32_t taskEnd = 0;
        if constexpr (ONE_WORKER_PER_GROUP) {
            taskStart = blockIdx_ * chunkNum_;
            taskEnd = taskStart + chunkNum_;
        } else if ((FIXED_TASK_CORE_COUNT > 0U ||
            (groupHeads_ == 2U && taskNum_ > 32U && taskNum_ <= 512U) ||
            (groupHeads_ == 8U && headPerGroup_ == 3U &&
                taskNum_ > 32U && taskNum_ <= 256U)) &&
            taskCoreNum_ % groupHeads_ == 0U) {
            const uint32_t coresPerGroup = taskCoreNum_ / groupHeads_;
            const uint32_t groupIdx = blockIdx_ / coresPerGroup;
            const uint32_t coreInGroup = blockIdx_ - groupIdx * coresPerGroup;
            const uint32_t chunksPerCore = chunkNum_ / coresPerGroup;
            const uint32_t extraChunks = chunkNum_ - chunksPerCore * coresPerGroup;
            const uint32_t chunkStart =
                coreInGroup * chunksPerCore + Min(coreInGroup, extraChunks);
            const uint32_t chunkEnd =
                chunkStart + chunksPerCore + (coreInGroup < extraChunks ? 1U : 0U);
            taskStart = groupIdx * chunkNum_ + chunkStart;
            taskEnd = groupIdx * chunkNum_ + chunkEnd;
        } else {
            uint32_t tasksPerCore = 0;
            uint32_t extraTasks = 0;
            if constexpr (HPG4_FULL_PIPELINE) {
                if (groupHeads_ == 8U) {
                    tasksPerCore = blockFactor_ - 1U;
                    extraTasks = taskNum_ - tasksPerCore * taskCoreNum_;
                } else {
                    tasksPerCore = taskNum_ / taskCoreNum_;
                    extraTasks = taskNum_ - tasksPerCore * taskCoreNum_;
                }
            } else {
                tasksPerCore = taskNum_ / taskCoreNum_;
                extraTasks = taskNum_ - tasksPerCore * taskCoreNum_;
            }
            taskStart = blockIdx_ * tasksPerCore + Min(blockIdx_, extraTasks);
            taskEnd = taskStart + tasksPerCore + (blockIdx_ < extraTasks ? 1U : 0U);
        }
        const uint32_t taskCount = taskEnd - taskStart;
        const uint32_t activeSlotCount =
            taskNum_ >= VERY_LONG_TASK_MIN ? 3U : WORKSPACE_SLOT_COUNT;
        const bool hasUniformFullChunks =
            static_cast<uint64_t>(seqLen_) ==
                static_cast<uint64_t>(chunkNum_) * chunkSize_;
        const bool useFullAlignedOffsets =
            LONG_SHARED_L1 || K128_FULL_ALIGNED_DOT ||
            (((MANUAL_K128_CUBE && keyDim_ == 128U) ||
                taskNum_ >= LONG_TASK_MIN) && hasUniformFullChunks);
        bool useLocalOffsets = false;
        LocalTensor<int32_t> offsetsLocal;
        if ASCEND_IS_AIV {
            if (!useFullAlignedOffsets) {
                const uint32_t firstChunk = taskStart % chunkNum_;
                if constexpr (MANUAL_K128_CUBE) {
                    useLocalOffsets = taskCount < 256U &&
                        firstChunk + taskCount <= chunkNum_;
                } else {
                    useLocalOffsets = taskNum_ >= LONG_TASK_MIN &&
                        (taskNum_ < VERY_LONG_TASK_MIN || keyDim_ == 128U) &&
                        taskCount < 256U && firstChunk + taskCount <= chunkNum_;
                }
                if (useLocalOffsets) {
                    offsetsLocal = offsetsQueue_.AllocTensor<int32_t>();
                    DataCopyExtParams offsetsCopyParams{
                        1, static_cast<uint32_t>((taskCount + 1U) * sizeof(int32_t)), 0, 0, 0};
                    DataCopyPadExtParams<int32_t> offsetsCopyPad{false, 0, 0, 0};
                    DataCopyPad(offsetsLocal, offsetsGm_[firstChunk], offsetsCopyParams, offsetsCopyPad);
                    offsetsQueue_.EnQue<int32_t>(offsetsLocal);
                    offsetsLocal = offsetsQueue_.DeQue<int32_t>();
                }
            }
        }
        int32_t previousEndValue = 0;
        uint32_t groupIdx = ONE_WORKER_PER_GROUP ? blockIdx_ : taskStart / chunkNum_;
        uint32_t chunkIdx = ONE_WORKER_PER_GROUP ? 0U : taskStart - groupIdx * chunkNum_;
        for (uint32_t localTaskIdx = 0; localTaskIdx < taskCount; ++localTaskIdx) {
            const uint32_t taskIdx = taskStart + localTaskIdx;
            const uint32_t slotIdx = localTaskIdx % activeSlotCount;
            if ASCEND_IS_AIC {
                if (localTaskIdx >= activeSlotCount) {
                    WaitPostDone(slotIdx);
                }
            }
            int32_t startValue = 0;
            int32_t endValue = 0;
            if (useFullAlignedOffsets) {
                startValue = static_cast<int32_t>(chunkIdx * chunkSize_);
                endValue = startValue + static_cast<int32_t>(chunkSize_);
            } else {
                if ASCEND_IS_AIV {
                    if (useLocalOffsets) {
                        startValue = localTaskIdx == 0U ? offsetsLocal.GetValue(0U) : previousEndValue;
                        endValue = offsetsLocal.GetValue(localTaskIdx + 1U);
                        previousEndValue = endValue;
                    } else {
                        if constexpr (ROLLING_OFFSETS) {
                            startValue = localTaskIdx == 0U || chunkIdx == 0U ?
                                offsetsGm_.GetValue(chunkIdx) : previousEndValue;
                        } else {
                            startValue = offsetsGm_.GetValue(chunkIdx);
                        }
                        endValue = offsetsGm_.GetValue(chunkIdx + 1U);
                        if constexpr (ROLLING_OFFSETS) {
                            previousEndValue = endValue;
                        }
                    }
                } else {
                    if constexpr (ROLLING_OFFSETS) {
                        startValue = localTaskIdx == 0U || chunkIdx == 0U ?
                            offsetsGm_.GetValue(chunkIdx) : previousEndValue;
                    } else {
                        startValue = taskNum_ >= LONG_TASK_MIN &&
                                taskNum_ < VERY_LONG_TASK_MIN && keyDim_ == 128U &&
                                localTaskIdx != 0U && chunkIdx != 0U ?
                            previousEndValue : offsetsGm_.GetValue(chunkIdx);
                    }
                    endValue = offsetsGm_.GetValue(chunkIdx + 1U);
                    previousEndValue = endValue;
                }
            }
            ProcessTask(taskIdx, groupIdx, chunkIdx, slotIdx, startValue, endValue);
            ++chunkIdx;
            if (chunkIdx == chunkNum_) {
                chunkIdx = 0U;
                ++groupIdx;
            }
        }
        if ASCEND_IS_AIV {
            if (useLocalOffsets) {
                offsetsQueue_.FreeTensor<int32_t>(offsetsLocal);
            }
        }
        if ASCEND_IS_AIC {
            const uint32_t pendingSlots = Min(taskCount, activeSlotCount);
            for (uint32_t i = 0; i < pendingSlots; ++i) {
                const uint32_t slotIdx = (taskCount - pendingSlots + i) % activeSlotCount;
                WaitPostDone(slotIdx);
            }
        }
    }

private:
    __aicore__ inline uint32_t GetHeadPerGroup() const
    {
        if constexpr (MANUAL_HPG4) {
            return 4U;
        }
        if constexpr (MANUAL_HPG3) {
            return 3U;
        }
        return headPerGroup_;
    }

    __aicore__ inline uint32_t GetFirstHeadCount(uint32_t taskIdx) const
    {
        if constexpr (MANUAL_HPG4) {
            return 2U;
        }
        if constexpr (MANUAL_HPG3) {
            return 2U - (taskIdx & 1U);
        }
        return (headPerGroup_ + 1U - (headPerGroup_ & taskIdx & 1U)) >> 1U;
    }

    __aicore__ inline void ProcessTinyFull()
    {
        constexpr uint32_t slotCount = WORKSPACE_SLOT_COUNT;
        if ASCEND_IS_AIC {
            mm.SetHF32(false, 0);
            mm.SetSingleShape(chunkSize_, chunkSize_, keyDim_);
            currentKRowStride_ = groupHeads_ * keyDim_;
        }

        const uint32_t groupIdx = blockIdx_;
        const uint32_t taskCount = chunkNum_;
        for (uint32_t chunkIdx = 0; chunkIdx < taskCount; ++chunkIdx) {
            const uint32_t taskIdx = groupIdx * chunkNum_ + chunkIdx;
            const uint32_t slotIdx = chunkIdx;
            const int32_t startValue = offsetsGm_.GetValue(chunkIdx);
            const int32_t endValue = offsetsGm_.GetValue(chunkIdx + 1U);
            ProcessTask(taskIdx, groupIdx, chunkIdx, slotIdx, startValue, endValue);
        }
        if ASCEND_IS_AIC {
            for (uint32_t slotIdx = 0; slotIdx < taskCount; ++slotIdx) {
                WaitPostDone(slotIdx);
            }
        }
        (void)slotCount;
    }

    __aicore__ inline uint32_t Min(uint32_t lhs, uint32_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const
    {
        return (value + align - 1U) / align * align;
    }

    __aicore__ inline bool UseHpg3EarlyRelease(uint32_t chunkLen) const
    {
        const bool useEarlyTail = EARLY_RELEASE_TAIL && chunkLen < 64U;
        const bool useK256Group2 = groupHeads_ == 2U && keyDim_ == 256U &&
            taskNum_ > 256U && taskNum_ <= 512U;
        const bool useGroup8 = groupHeads_ == 8U &&
            taskNum_ > 32U && taskNum_ <= 256U;
        const bool useManualPair = MANUAL_K128_CUBE &&
            (headPerGroup_ == 3U || headPerGroup_ == 4U);
        const bool useLargeFull =
            taskNum_ >= VERY_LONG_TASK_MIN && headPerGroup_ >= 2U && keyDim_ == 128U;
        return useEarlyTail ||
            (chunkLen == 64U &&
                ((headPerGroup_ == 3U && (useK256Group2 || useGroup8)) ||
                useManualPair || useLargeFull));
    }

    __aicore__ inline void ProcessGh2K128PairPipeline()
    {
        constexpr uint32_t groupHeads = 2U;
        constexpr uint32_t keyDim = 128U;
        constexpr uint32_t chunkSize = 64U;
        constexpr uint32_t slotCount = WORKSPACE_SLOT_COUNT;
        if ASCEND_IS_AIC {
            mm.SetHF32(false, 0);
            mm.SetSingleShape(chunkSize, chunkSize, keyDim);
            currentKRowStride_ = groupHeads * keyDim;
        }

        const uint32_t coresPerGroup = taskCoreNum_ / groupHeads;
        const uint32_t groupIdx = blockIdx_ / coresPerGroup;
        const uint32_t coreInGroup = blockIdx_ - groupIdx * coresPerGroup;
        const uint32_t chunksPerCore = blockFactor_ - 1U;
        const uint32_t extraChunks = chunkNum_ - chunksPerCore * coresPerGroup;
        const uint32_t chunkStart =
            coreInGroup * chunksPerCore + Min(coreInGroup, extraChunks);
        const uint32_t taskCount = chunksPerCore + (coreInGroup < extraChunks ? 1U : 0U);

        for (uint32_t localTaskIdx = 0; localTaskIdx < taskCount; ++localTaskIdx) {
            const uint32_t chunkIdx = chunkStart + localTaskIdx;
            const uint32_t taskIdx = groupIdx * chunkNum_ + chunkIdx;
            const uint32_t slotIdx = localTaskIdx % slotCount;
            if ASCEND_IS_AIC {
                if (localTaskIdx >= slotCount) {
                    WaitPostDone(slotIdx);
                }
            }
            const int32_t startValue = offsetsGm_.GetValue(chunkIdx);
            const int32_t endValue = offsetsGm_.GetValue(chunkIdx + 1U);
            ProcessTask(taskIdx, groupIdx, chunkIdx, slotIdx, startValue, endValue);
        }
        if ASCEND_IS_AIC {
            const uint32_t pendingSlots = Min(taskCount, slotCount);
            for (uint32_t i = 0; i < pendingSlots; ++i) {
                const uint32_t slotIdx = (taskCount - pendingSlots + i) % slotCount;
                WaitPostDone(slotIdx);
            }
        }
    }

    __aicore__ inline void ProcessGh2K128PairFull(
        uint32_t taskIdx,
        uint32_t groupIdx,
        uint32_t slotIdx,
        uint32_t start)
    {
        const uint64_t workspaceBaseBytes = static_cast<uint64_t>(blockIdx_) * workspacePerCore_ +
            static_cast<uint64_t>(slotIdx) * workspaceSlotBytes_;
        const uint64_t dotOffset =
            (workspaceBaseBytes + compactWorkspaceBytes_) / sizeof(float);
        if ASCEND_IS_AIC {
            ComputeGh2K128PairDot(dotOffset, start, groupIdx);
            SignalDotReady(slotIdx);
        }
        if ASCEND_IS_AIV {
            WaitDotReady(slotIdx);
            PostProcess(dotOffset, start, 64U, groupIdx, taskIdx, slotIdx, 0U);
        }
    }

    __aicore__ inline void ComputeGh2K128PairDot(
        uint64_t dotOffset,
        uint32_t start,
        uint32_t groupIdx)
    {
        constexpr uint32_t groupHeads = 2U;
        constexpr uint32_t keyDim = 128U;
        const uint64_t kOffset =
            (static_cast<uint64_t>(start) * groupHeads + groupIdx) * keyDim;
        mm.SetTensorA(kGm_[kOffset], false);
        mm.SetTensorB(kGm_[kOffset], true);
        mm.IterateAll(workspaceGm_[dotOffset], false);
        mm.End();
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void PostProcessGh2K128PairFull(
        uint64_t workspaceOffset,
        uint32_t start,
        uint32_t groupIdx,
        uint32_t taskIdx)
    {
        constexpr uint32_t chunkSize = 64U;
        constexpr uint32_t headPerGroup = 4U;
        constexpr uint32_t headsPerVector = headPerGroup / 2U;
        (void)taskIdx;
        LocalTensor<float> dotMatrixLocal = dotMatrixQueue_.AllocTensor<float>();
        DataCopy(dotMatrixLocal, workspaceGm_[workspaceOffset], chunkSize * chunkSize);
        dotMatrixQueue_.EnQue<float>(dotMatrixLocal);

        const uint32_t headStart = groupIdx * headPerGroup;
        const uint32_t localHeadBegin = subBlockIdx_ * headsPerVector;
        const bool overlapBetaWithDot = groupHeads_ == 8U &&
            chunkNum_ == seqLen_ / chunkSize;
        for (uint32_t i = 0; i < headsPerVector; ++i) {
            const uint32_t headIdx = headStart + localHeadBegin + i;
            const uint64_t headTokenStart = HeadTokenOffset(headIdx, start);
            LocalTensor<float> gLocal = gQueue_.template DeQue<float>();
            LocalTensor<float> outMatrixLocal = outQueue_.AllocTensor<float>();
            BuildGh2K128PairGamma(gLocal, outMatrixLocal);
            const bool dotCopyPending = i == 0U && overlapBetaWithDot;
            if (i == 0U && !dotCopyPending) {
                dotMatrixLocal = dotMatrixQueue_.DeQue<float>();
            }
            ScaleGh2K128PairFull(outMatrixLocal, dotMatrixLocal, dotCopyPending);
            gQueue_.template FreeTensor<float>(gLocal);
            outQueue_.EnQue<float>(outMatrixLocal);
            outMatrixLocal = outQueue_.DeQue<float>();
            DataCopy(outGm_[headTokenStart * chunkSize], outMatrixLocal, chunkSize * chunkSize);
            outQueue_.FreeTensor<float>(outMatrixLocal);
        }
        dotMatrixQueue_.FreeTensor<float>(dotMatrixLocal);
    }

    __aicore__ inline void PrefetchGh2K128PairG(uint64_t headTokenStart)
    {
        constexpr uint32_t chunkSize = 64U;
        LocalTensor<float> gLocal = gQueue_.template AllocTensor<float>();
        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(chunkSize * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> copyPad{false, 0, 0, 0.0f};
        DataCopyPad(gLocal, gGm_[headTokenStart], copyParams, copyPad);
        gQueue_.template EnQue<float>(gLocal);
    }

    __aicore__ inline void PrefetchGh2K128PairBeta(uint64_t headTokenStart)
    {
        constexpr uint32_t chunkSize = 64U;
        LocalTensor<T> betaLocal = betaQueue_.template AllocTensor<T>();
        DataCopy(betaLocal, betaGm_[headTokenStart], chunkSize);
        betaQueue_.template EnQue<T>(betaLocal);
    }

    __aicore__ inline void BuildGh2K128PairGamma(
        const LocalTensor<float>& gLocal,
        const LocalTensor<float>& gammaMatrixLocal)
    {
        constexpr uint32_t chunkSize = 64U;
        constexpr uint32_t matrixElements = chunkSize * chunkSize;
        LocalTensor<float> gBrcbLocal = gBrcbBuf_.Get<float>();
        Brcb(gBrcbLocal, gLocal,
            static_cast<uint8_t>(chunkSize / FP32_BLOCK_ELEMS), {1, FP32_BLOCK_ELEMS});
        BinaryRepeatParams broadcastParams{1, 0, 1, 8, 1, 0};
        Sub(gammaMatrixLocal, gBrcbLocal, gLocal,
            static_cast<uint64_t>(chunkSize), static_cast<uint8_t>(chunkSize), broadcastParams);
        LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();
        CompareScalar(maskLocal, gammaMatrixLocal, 0.0f, CMPMODE::LT, matrixElements);
        Select(gammaMatrixLocal, maskLocal, gammaMatrixLocal, SAFE_EXP_NEG_INF,
            SELMODE::VSEL_TENSOR_SCALAR_MODE, matrixElements);
        Exp<float>(gammaMatrixLocal, gammaMatrixLocal, matrixElements);
    }

    __aicore__ inline void ScaleGh2K128PairFull(
        const LocalTensor<float>& outMatrixLocal,
        LocalTensor<float>& dotMatrixLocal,
        bool dotCopyPending)
    {
        constexpr uint32_t chunkSize = 64U;
        constexpr uint32_t matrixElements = chunkSize * chunkSize;
        LocalTensor<T> betaLocal = betaQueue_.template DeQue<T>();
        LocalTensor<float> betaFloatLocal = betaFloatBuf_.Get<float>();
        LocalTensor<float> betaBrcbLocal = gBrcbBuf_.Get<float>();
        Cast(betaFloatLocal, betaLocal, RoundMode::CAST_NONE, chunkSize);
        Brcb(betaBrcbLocal, betaFloatLocal,
            static_cast<uint8_t>(chunkSize / FP32_BLOCK_ELEMS), {1, FP32_BLOCK_ELEMS});
        BinaryRepeatParams rowScaleParams{1, 1, 0, 8, 8, 1};
        Mul(outMatrixLocal, outMatrixLocal, betaBrcbLocal,
            static_cast<uint64_t>(chunkSize), static_cast<uint8_t>(chunkSize), rowScaleParams);
        if (dotCopyPending) {
            dotMatrixLocal = dotMatrixQueue_.DeQue<float>();
        }
        Mul(outMatrixLocal, outMatrixLocal, dotMatrixLocal, matrixElements);
        betaQueue_.template FreeTensor<T>(betaLocal);
    }

    __aicore__ inline void SyncVectorToMte3() const
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void SyncMte2ToMte3() const
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eventId);
        WaitFlag<HardEvent::MTE2_MTE3>(eventId);
    }

    __aicore__ inline void SyncMte3ToMte2() const
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
    }

    __aicore__ inline float LoadBetaFloat(uint64_t offset)
    {
        return ScalarToFloat(betaGm_.GetValue(offset));
    }

    __aicore__ inline void WaitPostDone(uint32_t slotIdx) const
    {
        if (slotIdx == 0) {
            CrossCoreWaitFlag(0x8);
        } else if (slotIdx == 1) {
            CrossCoreWaitFlag(0x9);
        } else if (slotIdx == 2) {
            CrossCoreWaitFlag(0xC);
        } else {
            CrossCoreWaitFlag(0xF);
        }
    }

    __aicore__ inline void SignalPostDone(uint32_t slotIdx) const
    {
        if (slotIdx == 0) {
            CrossCoreSetFlag<0x2, PIPE_MTE3>(0x8);
        } else if (slotIdx == 1) {
            CrossCoreSetFlag<0x2, PIPE_MTE3>(0x9);
        } else if (slotIdx == 2) {
            CrossCoreSetFlag<0x2, PIPE_MTE3>(0xC);
        } else {
            CrossCoreSetFlag<0x2, PIPE_MTE3>(0xF);
        }
    }

    __aicore__ inline void SignalWorkspaceConsumed(uint32_t slotIdx) const
    {
        if (slotIdx == 0) {
            CrossCoreSetFlag<0x2, PIPE_MTE2>(0x8);
        } else if (slotIdx == 1) {
            CrossCoreSetFlag<0x2, PIPE_MTE2>(0x9);
        } else if (slotIdx == 2) {
            CrossCoreSetFlag<0x2, PIPE_MTE2>(0xC);
        } else {
            CrossCoreSetFlag<0x2, PIPE_MTE2>(0xF);
        }
    }

    __aicore__ inline void WaitTailReady(uint32_t slotIdx) const
    {
        if (slotIdx == 0) {
            CrossCoreWaitFlag(0x4);
        } else if (slotIdx == 1) {
            CrossCoreWaitFlag(0x5);
        } else if (slotIdx == 2) {
            CrossCoreWaitFlag(0xA);
        } else {
            CrossCoreWaitFlag(0xD);
        }
    }

    __aicore__ inline void SignalTailReady(uint32_t slotIdx) const
    {
        if (slotIdx == 0) {
            CrossCoreSetFlag<0x2, PIPE_MTE3>(0x4);
        } else if (slotIdx == 1) {
            CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5);
        } else if (slotIdx == 2) {
            CrossCoreSetFlag<0x2, PIPE_MTE3>(0xA);
        } else {
            CrossCoreSetFlag<0x2, PIPE_MTE3>(0xD);
        }
    }

    __aicore__ inline void WaitDotReady(uint32_t slotIdx) const
    {
        if (slotIdx == 0) {
            CrossCoreWaitFlag(0x6);
        } else if (slotIdx == 1) {
            CrossCoreWaitFlag(0x7);
        } else if (slotIdx == 2) {
            CrossCoreWaitFlag(0xB);
        } else {
            CrossCoreWaitFlag(0xE);
        }
    }

    __aicore__ inline void SignalDotReady(uint32_t slotIdx) const
    {
        if (slotIdx == 0) {
            CrossCoreSetFlag<0x2, PIPE_FIX>(0x6);
        } else if (slotIdx == 1) {
            CrossCoreSetFlag<0x2, PIPE_FIX>(0x7);
        } else if (slotIdx == 2) {
            CrossCoreSetFlag<0x2, PIPE_FIX>(0xB);
        } else {
            CrossCoreSetFlag<0x2, PIPE_FIX>(0xE);
        }
    }

    __aicore__ inline void ProcessTask(
        uint32_t taskIdx,
        uint32_t groupIdx,
        uint32_t chunkIdx,
        uint32_t slotIdx,
        int32_t startValue,
        int32_t endValue)
    {
        if constexpr (LONG_SHARED_L1) {
            ProcessTaskBody(taskIdx, groupIdx, slotIdx,
                chunkIdx * MANUAL_FULL_CHUNK_SIZE, MANUAL_FULL_CHUNK_SIZE);
            return;
        }
        if (chunkIdx >= chunkNum_ || groupIdx >= groupHeads_) {
            return;
        }

        if (startValue < 0 || endValue <= startValue || static_cast<uint32_t>(startValue) >= seqLen_) {
            return;
        }
        const uint32_t start = static_cast<uint32_t>(startValue);
        const uint32_t end = Min(static_cast<uint32_t>(endValue), seqLen_);
        const uint32_t chunkLen = Min(end - start, chunkSize_);
        if (chunkLen == 0) {
            return;
        }
        ProcessTaskBody(taskIdx, groupIdx, slotIdx, start, chunkLen);
    }

    __aicore__ inline void ProcessTaskBody(
        uint32_t taskIdx,
        uint32_t groupIdx,
        uint32_t slotIdx,
        uint32_t start,
        uint32_t chunkLen)
    {
        const uint64_t workspaceBaseBytes = static_cast<uint64_t>(blockIdx_) * workspacePerCore_ +
            static_cast<uint64_t>(slotIdx) * workspaceSlotBytes_;
        const uint64_t compactOffset = workspaceBaseBytes / sizeof(T);
        const uint64_t dotOffset = (workspaceBaseBytes + compactWorkspaceBytes_) / sizeof(float);
        if ASCEND_IS_AIC {
            if constexpr (!MANUAL_K128_CUBE) {
                if (chunkLen < chunkSize_) {
                    WaitTailReady(slotIdx);
                }
            }
            ComputeDot(dotOffset, compactOffset, start, chunkLen, groupIdx);
            SignalDotReady(slotIdx);
        }
        if ASCEND_IS_AIV {
            if constexpr (!MANUAL_K128_CUBE) {
                if (chunkLen < chunkSize_) {
                    if (subBlockIdx_ == 0) {
                        CopyKeyToCompact(compactOffset, start, chunkLen, groupIdx);
                    }
                    SignalTailReady(slotIdx);
                }
            }
            if constexpr (GH2_K128_PAIR_PIPELINE) {
                if (chunkLen == 64U && headPerGroup_ == 4U) {
                    constexpr uint32_t headsPerVector = 2U;
                    const uint32_t localHeadBegin = subBlockIdx_ * headsPerVector;
                    PrefetchHeadPairInputs(
                        start, groupIdx * headPerGroup_ + localHeadBegin);
                }
            }
            if constexpr (HPG4_FULL_PIPELINE) {
                if (chunkLen == 64U && headPerGroup_ == 4U) {
                    constexpr uint32_t headsPerVector = 2U;
                    const uint32_t localHeadBegin = subBlockIdx_ * headsPerVector;
                    for (uint32_t i = 0; i < headsPerVector; ++i) {
                        const uint64_t headTokenStart = HeadTokenOffset(
                            groupIdx * headPerGroup_ + localHeadBegin + i, start);
                        PrefetchGh2K128PairG(headTokenStart);
                        PrefetchGh2K128PairBeta(headTokenStart);
                    }
                }
            }
            uint32_t prefetchedHeadCount = 0U;
            if constexpr (!GH2_K128_PAIR_PIPELINE && !HPG4_FULL_PIPELINE) {
                const bool useTinySingleHeadSplit =
                    TINY_FULL_PIPELINE && chunkLen == chunkSize_ && headPerGroup_ == 1U;
                if (useTinySingleHeadSplit) {
                    PrefetchTinySingleHeadInputs(start, groupIdx);
                } else {
                    const uint32_t firstHeadCount = GetFirstHeadCount(taskIdx);
                    const uint32_t headPerGroup = GetHeadPerGroup();
                    const uint32_t localHeadBegin = subBlockIdx_ * firstHeadCount;
                    const uint32_t localHeadEnd =
                        firstHeadCount + subBlockIdx_ * (headPerGroup - firstHeadCount);
                    const uint32_t headStart = groupIdx * headPerGroup;
                    bool pairPrefetched = false;
                    if constexpr (MANUAL_K128_CUBE) {
                        const uint32_t localHeadCount = localHeadEnd - localHeadBegin;
                        if (chunkLen == 64U && localHeadCount == 2U) {
                            PrefetchHeadPairInputs(start, headStart + localHeadBegin);
                            prefetchedHeadCount = 2U;
                            pairPrefetched = true;
                        }
                    }
                    if (!pairPrefetched) {
                        for (uint32_t i = 0U;
                            i < 2U && localHeadBegin + i < localHeadEnd &&
                            headStart + localHeadBegin + i < heads_;
                            ++i) {
                            const uint64_t headTokenStart =
                                HeadTokenOffset(headStart + localHeadBegin + i, start);
                            PrefetchG(headTokenStart, chunkLen);
                            if (chunkLen >= BETA_BATCH_MIN_ROWS) {
                                PrefetchBeta(headTokenStart, chunkLen);
                            }
                            ++prefetchedHeadCount;
                        }
                    }
                }
            }
            WaitDotReady(slotIdx);
            if constexpr (HPG4_FULL_PIPELINE) {
                if (chunkLen == 64U && headPerGroup_ == 4U) {
                    PostProcessGh2K128PairFull(dotOffset, start, groupIdx, taskIdx);
                } else {
                    PostProcess(dotOffset, start, chunkLen, groupIdx, taskIdx, slotIdx,
                        prefetchedHeadCount);
                }
            } else {
                PostProcess(dotOffset, start, chunkLen, groupIdx, taskIdx, slotIdx,
                    prefetchedHeadCount);
            }
            if constexpr (!GH2_K128_PAIR_PIPELINE) {
                if (!UseHpg3EarlyRelease(chunkLen)) {
                    SignalPostDone(slotIdx);
                }
            }
        }
    }

    __aicore__ inline void CopyKeyToCompact(
        uint64_t compactOffset,
        uint32_t start,
        uint32_t chunkLen,
        uint32_t groupIdx)
    {
        const uint32_t kStride = groupHeads_ * keyDim_;
        const uint64_t kBase = (static_cast<uint64_t>(start) * groupHeads_ + groupIdx) * keyDim_;
        const uint32_t compactElements = chunkSize_ * keyDim_;
        const uint32_t validElements = chunkLen * keyDim_;
        DataCopyExtParams copyInParams{
            static_cast<uint16_t>(chunkLen),
            static_cast<uint32_t>(keyDim_ * sizeof(T)),
            static_cast<uint32_t>((kStride - keyDim_) * sizeof(T)),
            0,
            0};
        DataCopyPadExtParams<T> copyInPad{false, 0, 0, 0};

        LocalTensor<T> keyInLocal = compactInQueue_.AllocTensor<T>();
        DataCopyPad(keyInLocal, kGm_[kBase], copyInParams, copyInPad);
        compactInQueue_.EnQue<T>(keyInLocal);

        keyInLocal = compactInQueue_.DeQue<T>();

        LocalTensor<T> keyOutLocal = compactOutQueue_.AllocTensor<T>();
        Duplicate(keyOutLocal, static_cast<T>(0.0f), compactElements);
        PipeBarrier<PIPE_V>();
        DataCopy(keyOutLocal, keyInLocal, validElements);
        compactInQueue_.FreeTensor<T>(keyInLocal);
        SyncVectorToMte3();
        compactOutQueue_.EnQue<T>(keyOutLocal);
        keyOutLocal = compactOutQueue_.DeQue<T>();
        DataCopy(compactGm_[compactOffset], keyOutLocal, compactElements);
        compactOutQueue_.FreeTensor<T>(keyOutLocal);
    }

    __aicore__ inline void ComputeDot(
        uint64_t dotOffset,
        uint64_t compactOffset,
        uint32_t start,
        uint32_t chunkLen,
        uint32_t groupIdx)
    {
        if constexpr (MANUAL_K128_CUBE) {
            const uint64_t kOffset =
                (static_cast<uint64_t>(start) * groupHeads_ + groupIdx) * keyDim_;
            ComputeDotManual(
                workspaceGm_[dotOffset], kGm_[kOffset], chunkLen, groupHeads_ * keyDim_);
        } else {
            const bool useDirectK = chunkLen == chunkSize_;
            const uint32_t kRowStride = useDirectK ? groupHeads_ * keyDim_ : keyDim_;
            if (kRowStride != currentKRowStride_) {
                mm.SetOrgShape(chunkSize_, chunkSize_, kRowStride, kRowStride);
                currentKRowStride_ = kRowStride;
            }
            if (chunkLen != currentTailSize_) {
                mm.SetTail(chunkLen, chunkLen, keyDim_);
                currentTailSize_ = chunkLen;
            }
            if (useDirectK) {
                const uint64_t kOffset =
                    (static_cast<uint64_t>(start) * groupHeads_ + groupIdx) * keyDim_;
                mm.SetTensorA(kGm_[kOffset], false);
                mm.SetTensorB(kGm_[kOffset], true);
            } else {
                mm.SetTensorA(compactGm_[compactOffset], false);
                mm.SetTensorB(compactGm_[compactOffset], true);
            }
            if (taskNum_ >= LONG_TASK_MIN &&
                (taskNum_ >= VERY_LONG_TASK_MIN || keyDim_ == 256U)) {
                mm.template Iterate<false>();
                mm.GetTensorC(workspaceGm_[dotOffset], 0, true);
            } else {
                mm.IterateAll(workspaceGm_[dotOffset], false);
            }
            mm.End();
        }
    }

    __aicore__ inline void ComputeDotManual(
        const GlobalTensor<float>& dst,
        const GlobalTensor<T>& src,
        uint32_t matrixSize,
        uint32_t srcRowStride)
    {
        constexpr uint32_t c0Size = 16U;
        const uint32_t kSize = keyDim_;
        const uint32_t alignedSize = AlignUp(matrixSize, c0Size);

        LocalTensor<T> kL1 = kL1Queue_.AllocTensor<T>();
        LocalTensor<T> kB1;
        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue = matrixSize;
        nd2nzParams.dValue = kSize;
        nd2nzParams.srcDValue = srcRowStride;
        nd2nzParams.dstNzC0Stride = alignedSize;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.dstNzMatrixStride = 0;

        const bool useAFullBlockLayout = matrixSize == alignedSize && srcRowStride <= 4095U;
        const bool reuseRuntimeL1 =
            (taskNum_ >= LONG_TASK_MIN && matrixSize == alignedSize) ||
            (keyDim_ == 256U && headPerGroup_ == 4U);
        if constexpr (LONG_SHARED_L1) {
            DataCopy(kL1, src, nd2nzParams);
        } else if (reuseRuntimeL1) {
            DataCopy(kL1, src, nd2nzParams);
        } else if (useAFullBlockLayout) {
            Nd2NzParams nd2nzAParams;
            nd2nzAParams.ndNum = matrixSize / c0Size;
            nd2nzAParams.nValue = c0Size;
            nd2nzAParams.dValue = kSize;
            nd2nzAParams.srcDValue = srcRowStride;
            nd2nzAParams.dstNzC0Stride = c0Size;
            nd2nzAParams.dstNzNStride = 1;
            nd2nzAParams.srcNdMatrixStride = c0Size * srcRowStride;
            nd2nzAParams.dstNzMatrixStride = c0Size * kSize;
            DataCopy(kL1, src, nd2nzAParams);
        } else {
            DataCopy(kL1, src, nd2nzParams);
        }
        if constexpr (!LONG_SHARED_L1) {
            if (!reuseRuntimeL1) {
                kB1 = kB1Queue_.AllocTensor<T>();
                DataCopy(kB1, src, nd2nzParams);
            }
        }
        kL1Queue_.EnQue<T>(kL1);
        if constexpr (!LONG_SHARED_L1) {
            if (!reuseRuntimeL1) {
                kB1Queue_.EnQue<T>(kB1);
            }
        }

        kL1 = kL1Queue_.DeQue<T>();
        if constexpr (!LONG_SHARED_L1) {
            if (!reuseRuntimeL1) {
                kB1 = kB1Queue_.DeQue<T>();
            }
        }
        LocalTensor<T> kL0A = kL0AQueue_.AllocTensor<T>();
        LocalTensor<T> kL0B = kL0BQueue_.AllocTensor<T>();
        LoadData2DParams loadAParams;
        loadAParams.dstGap = 0;
        loadAParams.ifTranspose = false;
        if constexpr (LONG_SHARED_L1) {
            loadAParams.repeatTimes = kSize / c0Size;
            loadAParams.srcStride = alignedSize / c0Size;
            for (uint32_t i = 0; i < alignedSize / c0Size; ++i) {
                loadAParams.startIndex = i;
                LoadData(kL0A[i * kSize * c0Size], kL1, loadAParams);
            }
        } else if (useAFullBlockLayout && !reuseRuntimeL1) {
            loadAParams.startIndex = 0;
            loadAParams.repeatTimes = (alignedSize / c0Size) * (kSize / c0Size);
            loadAParams.srcStride = 1;
            LoadData(kL0A, kL1, loadAParams);
        } else {
            loadAParams.repeatTimes = kSize / c0Size;
            loadAParams.srcStride = alignedSize / c0Size;
            for (uint32_t i = 0; i < alignedSize / c0Size; ++i) {
                loadAParams.startIndex = i;
                LoadData(kL0A[i * kSize * c0Size], kL1, loadAParams);
            }
        }

        LoadData2DParams loadBParams;
        loadBParams.startIndex = 0;
        loadBParams.repeatTimes = (alignedSize / c0Size) * (kSize / c0Size);
        loadBParams.srcStride = 1;
        loadBParams.dstGap = 0;
        loadBParams.ifTranspose = false;
        if constexpr (LONG_SHARED_L1) {
            LoadData(kL0B, kL1, loadBParams);
        } else if (reuseRuntimeL1) {
            LoadData(kL0B, kL1, loadBParams);
        } else {
            LoadData(kL0B, kB1, loadBParams);
        }
        kL0AQueue_.EnQue<T>(kL0A);
        kL0BQueue_.EnQue<T>(kL0B);
        kL1Queue_.FreeTensor(kL1);
        if constexpr (!LONG_SHARED_L1) {
            if (!reuseRuntimeL1) {
                kB1Queue_.FreeTensor(kB1);
            }
        }

        kL0A = kL0AQueue_.DeQue<T>();
        kL0B = kL0BQueue_.DeQue<T>();
        LocalTensor<float> dotL0C = dotL0CQueue_.AllocTensor<float>();
        MmadParams mmParams;
        mmParams.m = alignedSize;
        mmParams.n = matrixSize;
        mmParams.k = kSize;
        mmParams.cmatrixInitVal = true;
        mmParams.cmatrixSource = false;
        Mmad(dotL0C, kL0A, kL0B, mmParams);
        dotL0CQueue_.EnQue<float>(dotL0C);
        kL0AQueue_.FreeTensor(kL0A);
        kL0BQueue_.FreeTensor(kL0B);

        dotL0C = dotL0CQueue_.DeQue<float>();
        FixpipeParamsV220 fixParams;
        fixParams.nSize = matrixSize;
        fixParams.mSize = matrixSize;
        fixParams.srcStride = alignedSize;
        fixParams.dstStride = chunkSize_;
        fixParams.ndNum = 1;
        fixParams.quantPre = QuantMode_t::NoQuant;
        Fixpipe<float, float, CFG_ROW_MAJOR>(dst, dotL0C, fixParams);
        dotL0CQueue_.FreeTensor(dotL0C);
    }

    __aicore__ inline void PostProcess(
        uint64_t workspaceOffset,
        uint32_t start,
        uint32_t chunkLen,
        uint32_t groupIdx,
        uint32_t taskIdx,
        uint32_t slotIdx,
        uint32_t prefetchedHeadCount)
    {
        const uint32_t firstHeadCount = GetFirstHeadCount(taskIdx);
        const uint32_t headPerGroup = GetHeadPerGroup();
        const uint32_t localHeadBegin = subBlockIdx_ * firstHeadCount;
        const uint32_t localHeadEnd =
            firstHeadCount + subBlockIdx_ * (headPerGroup - firstHeadCount);
        const bool useTinySingleHeadSplit =
            TINY_FULL_PIPELINE && chunkLen == chunkSize_ && headPerGroup == 1U;
        if (localHeadBegin >= localHeadEnd && !useTinySingleHeadSplit) {
            return;
        }
        if (useTinySingleHeadSplit) {
            const uint32_t rowCount = chunkSize_ / 2U;
            const uint32_t rowBegin = subBlockIdx_ * rowCount;
            LocalTensor<float> dotMatrixLocal = dotMatrixQueue_.AllocTensor<float>();
            DataCopy(dotMatrixLocal,
                workspaceGm_[workspaceOffset + rowBegin * chunkSize_],
                rowCount * chunkSize_);
            dotMatrixQueue_.EnQue<float>(dotMatrixLocal);
            dotMatrixLocal = dotMatrixQueue_.DeQue<float>();
            ProcessTinySingleHeadRange(start, groupIdx, dotMatrixLocal);
            dotMatrixQueue_.FreeTensor<float>(dotMatrixLocal);
            return;
        }
        LocalTensor<float> dotMatrixLocal = dotMatrixQueue_.AllocTensor<float>();
        if constexpr (K128_FULL_ALIGNED_DOT || MANUAL_K128_CUBE) {
            DataCopy(dotMatrixLocal, workspaceGm_[workspaceOffset], 64U * 64U);
        } else {
            DataCopyExtParams dotCopyParams{
                1, static_cast<uint32_t>(chunkSize_ * chunkSize_ * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> dotCopyPad{false, 0, 0, 0.0f};
            DataCopyPad(dotMatrixLocal, workspaceGm_[workspaceOffset], dotCopyParams, dotCopyPad);
        }
        dotMatrixQueue_.EnQue<float>(dotMatrixLocal);
        bool overlapPairDotCopy = false;
        if constexpr (GH2_K128_PAIR_PIPELINE || MANUAL_K128_CUBE) {
            overlapPairDotCopy =
                chunkLen == 64U && headPerGroup_ == 4U &&
                localHeadEnd - localHeadBegin == 2U;
            if constexpr (MANUAL_K128_CUBE) {
                const uint32_t localHeadCount = localHeadEnd - localHeadBegin;
                overlapPairDotCopy =
                    chunkLen == 64U && localHeadCount == 2U;
            }
        }
        bool pendingMode0FullDotCopy = false;
        if constexpr (!GH2_K128_PAIR_PIPELINE && !HPG4_FULL_PIPELINE) {
            pendingMode0FullDotCopy =
                chunkLen == chunkSize_ &&
                (taskNum_ <= 32U || taskNum_ >= LONG_TASK_MIN);
            if constexpr (MANUAL_K128_CUBE) {
                pendingMode0FullDotCopy =
                    chunkLen == chunkSize_ &&
                    localHeadEnd - localHeadBegin == 1U;
            }
        }
        bool pendingBatchedTailDotCopy = false;
        if constexpr (OVERLAP_TAIL_DOT) {
            pendingBatchedTailDotCopy =
                chunkLen >= BETA_BATCH_MIN_ROWS && chunkLen < chunkSize_ &&
                !(keyDim_ == 128U && groupHeads_ == 16U && headPerGroup_ == 4U);
        } else if constexpr (GH2_K128_PAIR_PIPELINE) {
            pendingBatchedTailDotCopy =
                chunkLen >= BETA_BATCH_MIN_ROWS && chunkLen < chunkSize_;
        }
        if (!overlapPairDotCopy && !pendingMode0FullDotCopy && !pendingBatchedTailDotCopy) {
            dotMatrixLocal = dotMatrixQueue_.DeQue<float>();
        }
        if constexpr (!GH2_K128_PAIR_PIPELINE) {
            if (UseHpg3EarlyRelease(chunkLen) && !pendingMode0FullDotCopy &&
                !overlapPairDotCopy && !pendingBatchedTailDotCopy) {
                SignalWorkspaceConsumed(slotIdx);
            }
        }
        if constexpr (GH2_K128_PAIR_PIPELINE || MANUAL_K128_CUBE) {
            if (!overlapPairDotCopy && !pendingBatchedTailDotCopy) {
                if constexpr (GH2_K128_PAIR_PIPELINE) {
                    SignalWorkspaceConsumed(slotIdx);
                }
            }
            const uint32_t localHeadCount = localHeadEnd - localHeadBegin;
            if (chunkLen == 64U && localHeadCount == 2U) {
                const uint32_t firstHeadIdx = groupIdx * headPerGroup + localHeadBegin;
                PostProcessHeadPair(
                    start, firstHeadIdx, dotMatrixLocal,
                    GH2_K128_PAIR_PIPELINE ? headPerGroup_ == 4U : prefetchedHeadCount == 2U,
                    overlapPairDotCopy, slotIdx);
                dotMatrixQueue_.FreeTensor<float>(dotMatrixLocal);
                return;
            }
        }

        const uint32_t headStart = groupIdx * headPerGroup;
        if (prefetchedHeadCount == 0U) {
            PrefetchG(HeadTokenOffset(headStart + localHeadBegin, start), chunkLen);
        }
        for (uint32_t localHead = localHeadBegin; localHead < localHeadEnd; ++localHead) {
            const uint32_t headIdx = headStart + localHead;
            if (headIdx >= heads_) {
                break;
            }
            const uint64_t headTokenStart = HeadTokenOffset(headIdx, start);
            LocalTensor<float> gLocal = gQueue_.template DeQue<float>();
            const bool thisHeadPrefetched =
                localHead - localHeadBegin < prefetchedHeadCount;
            if (chunkLen >= BETA_BATCH_MIN_ROWS && !thisHeadPrefetched) {
                PrefetchBeta(headTokenStart, chunkLen);
            }
            LocalTensor<float> negGLocal;
            if constexpr (MANUAL_K128_CUBE) {
                negGLocal = gBrcbBuf_.Get<float>();
            } else {
                negGLocal = negGBuf_.Get<float>();
            }
            const bool useFactoredTail =
                FACTORED_TAIL_MIN > 0U &&
                chunkLen >= FACTORED_TAIL_MIN && chunkLen < chunkSize_;
            if (chunkLen < chunkSize_ && !useFactoredTail) {
                Muls(negGLocal, gLocal, -1.0f, AlignUp(chunkLen, FP32_COMPARE_ELEMS));
                gQueue_.template FreeTensor<float>(gLocal);
                if (localHead + 1U < localHeadEnd && headIdx + 1U < heads_ &&
                    localHead + 1U - localHeadBegin >= prefetchedHeadCount) {
                    PrefetchG(HeadTokenOffset(headIdx + 1U, start), chunkLen);
                }
            }
            LocalTensor<float> outMatrixLocal = outQueue_.AllocTensor<float>();
            if (chunkLen < chunkSize_ && !useFactoredTail) {
                Duplicate(outMatrixLocal, 0.0f, chunkLen * chunkSize_);
            }
            if (chunkLen == chunkSize_) {
                if constexpr (FACTORED_SINGLE_HEAD) {
                    BuildFactoredFullGamma(gLocal, outMatrixLocal);
                } else {
                    BuildFullGamma(gLocal, outMatrixLocal);
                }
                if (pendingMode0FullDotCopy) {
                    dotMatrixLocal = dotMatrixQueue_.DeQue<float>();
                    pendingMode0FullDotCopy = false;
                    if (UseHpg3EarlyRelease(chunkLen)) {
                        SignalWorkspaceConsumed(slotIdx);
                    }
                }
                gQueue_.template FreeTensor<float>(gLocal);
                if (localHead + 1U < localHeadEnd && headIdx + 1U < heads_ &&
                    localHead + 1U - localHeadBegin >= prefetchedHeadCount) {
                    PrefetchG(HeadTokenOffset(headIdx + 1U, start), chunkLen);
                }
                if constexpr (FACTORED_SINGLE_HEAD) {
                    Mul(outMatrixLocal, outMatrixLocal, dotMatrixLocal,
                        chunkSize_ * chunkSize_);
                } else {
                    ScaleFullChunk(outMatrixLocal, dotMatrixLocal);
                }
            } else if (chunkLen >= BETA_BATCH_MIN_ROWS) {
                if (useFactoredTail) {
                    if constexpr (DIRECT_FACTORED_TAIL) {
                        BuildDirectFactoredTailGamma(gLocal, outMatrixLocal, chunkLen);
                    } else {
                        LocalTensor<float> gBackup = gBrcbBuf_.Get<float>();
                        Adds(gBackup, gLocal, 0.0f, chunkLen);
                        Duplicate(gLocal, SAFE_EXP_NEG_INF, chunkSize_);
                        Adds(gLocal, gBackup, 0.0f, chunkLen);
                        BuildFactoredFullGamma(gLocal, outMatrixLocal);
                    }
                    gQueue_.template FreeTensor<float>(gLocal);
                    if (localHead + 1U < localHeadEnd && headIdx + 1U < heads_ &&
                        localHead + 1U - localHeadBegin >= prefetchedHeadCount) {
                        PrefetchG(HeadTokenOffset(headIdx + 1U, start), chunkLen);
                    }
                } else {
                    for (uint32_t row = 0; row < chunkLen; ++row) {
                        FillGamma(chunkLen, start + row, headIdx, negGLocal,
                            outMatrixLocal[row * chunkSize_]);
                    }
                }
                if (pendingBatchedTailDotCopy) {
                    dotMatrixLocal = dotMatrixQueue_.DeQue<float>();
                    pendingBatchedTailDotCopy = false;
                    if constexpr (GH2_K128_PAIR_PIPELINE || EARLY_RELEASE_TAIL) {
                        SignalWorkspaceConsumed(slotIdx);
                    }
                }
                if (useFactoredTail) {
                    Mul(outMatrixLocal, outMatrixLocal, dotMatrixLocal,
                        chunkSize_ * chunkSize_);
                } else {
                    ScaleTailChunk(chunkLen, outMatrixLocal, dotMatrixLocal);
                }
            } else {
                for (uint32_t row = 0; row < chunkLen; ++row) {
                    ProcessRow(start, chunkLen, row, headIdx, negGLocal, dotMatrixLocal, outMatrixLocal);
                }
            }
            outQueue_.EnQue<float>(outMatrixLocal);
            outMatrixLocal = outQueue_.DeQue<float>();
            const uint64_t outOffset = headTokenStart * chunkSize_;
            DataCopy(outGm_[outOffset], outMatrixLocal, chunkLen * chunkSize_);
            outQueue_.FreeTensor<float>(outMatrixLocal);
        }
        if (pendingBatchedTailDotCopy) {
            dotMatrixLocal = dotMatrixQueue_.DeQue<float>();
            if constexpr (GH2_K128_PAIR_PIPELINE || EARLY_RELEASE_TAIL) {
                SignalWorkspaceConsumed(slotIdx);
            }
        }
        if (pendingMode0FullDotCopy) {
            dotMatrixLocal = dotMatrixQueue_.DeQue<float>();
            if (UseHpg3EarlyRelease(chunkLen)) {
                SignalWorkspaceConsumed(slotIdx);
            }
        }
        dotMatrixQueue_.FreeTensor<float>(dotMatrixLocal);
    }

    __aicore__ inline void PrefetchTinySingleHeadInputs(uint32_t start, uint32_t groupIdx)
    {
        const uint32_t rowCount = chunkSize_ / 2U;
        const uint32_t rowBegin = subBlockIdx_ * rowCount;
        const uint64_t headTokenStart = HeadTokenOffset(groupIdx, start);

        LocalTensor<float> gLocal = gQueue_.template AllocTensor<float>();
        DataCopy(gLocal, gGm_[headTokenStart], chunkSize_);
        gQueue_.template EnQue<float>(gLocal);

        LocalTensor<T> betaLocal = betaQueue_.template AllocTensor<T>();
        DataCopy(betaLocal, betaGm_[headTokenStart + rowBegin], rowCount);
        betaQueue_.template EnQue<T>(betaLocal);
    }

    __aicore__ inline void ProcessTinySingleHeadRange(
        uint32_t start,
        uint32_t groupIdx,
        const LocalTensor<float>& dotMatrixLocal)
    {
        const uint32_t rowCount = chunkSize_ / 2U;
        const uint32_t rowBegin = subBlockIdx_ * rowCount;
        const uint64_t headTokenStart = HeadTokenOffset(groupIdx, start);
        LocalTensor<float> gLocal = gQueue_.template DeQue<float>();
        LocalTensor<float> outMatrixLocal = outQueue_.AllocTensor<float>();

        BuildFullGammaRange(gLocal, rowBegin, rowCount, outMatrixLocal);
        ScaleFullChunkRange(rowCount, outMatrixLocal, dotMatrixLocal);
        gQueue_.template FreeTensor<float>(gLocal);

        outQueue_.EnQue<float>(outMatrixLocal);
        outMatrixLocal = outQueue_.DeQue<float>();
        const uint64_t outOffset = (headTokenStart + rowBegin) * chunkSize_;
        DataCopy(outGm_[outOffset], outMatrixLocal, rowCount * chunkSize_);
        outQueue_.FreeTensor<float>(outMatrixLocal);
    }

    __aicore__ inline void PrefetchHeadPairInputs(
        uint32_t start,
        uint32_t headStart)
    {
        constexpr uint32_t chunkSize = 64U;
        constexpr uint32_t headsPerVector = 2U;
        const uint64_t firstHeadTokenStart = HeadTokenOffset(headStart, start);
        LocalTensor<float> gPairLocal = gQueue_.template AllocTensor<float>();
        DataCopyExtParams gCopyParams{
            headsPerVector,
            static_cast<uint32_t>(chunkSize * sizeof(float)),
            static_cast<uint32_t>((seqLen_ - chunkSize) * sizeof(float)),
            0,
            0};
        DataCopyPadExtParams<float> gCopyPad{false, 0, 0, 0.0f};
        DataCopyPad(gPairLocal, gGm_[firstHeadTokenStart], gCopyParams, gCopyPad);
        gQueue_.template EnQue<float>(gPairLocal);

        LocalTensor<T> betaPairLocal = betaQueue_.template AllocTensor<T>();
        DataCopyExtParams betaCopyParams{
            headsPerVector,
            static_cast<uint32_t>(chunkSize * sizeof(T)),
            static_cast<uint32_t>((seqLen_ - chunkSize) * sizeof(T)),
            0,
            0};
        DataCopyPadExtParams<T> betaCopyPad{false, 0, 0, static_cast<T>(0.0f)};
        DataCopyPad(betaPairLocal, betaGm_[firstHeadTokenStart], betaCopyParams, betaCopyPad);
        betaQueue_.template EnQue<T>(betaPairLocal);
    }

    __aicore__ inline void PostProcessHeadPair(
        uint32_t start,
        uint32_t headStart,
        LocalTensor<float>& dotMatrixLocal,
        bool inputsPrefetched,
        bool dotCopyPending,
        uint32_t slotIdx)
    {
        constexpr uint32_t chunkSize = 64U;
        constexpr uint32_t headsPerVector = 2U;
        constexpr uint32_t matrixElements = chunkSize * chunkSize;
        constexpr uint32_t pairMatrixElements = headsPerVector * matrixElements;
        if (!inputsPrefetched) {
            PrefetchHeadPairInputs(start, headStart);
        }

        LocalTensor<float> gPairLocal = gQueue_.template DeQue<float>();
        LocalTensor<float> outPairLocal = outQueue_.AllocTensor<float>();
        LocalTensor<float> gBrcbLocal = gBrcbBuf_.Get<float>();
        BinaryRepeatParams broadcastParams{1, 0, 1, 8, 1, 0};
        BinaryRepeatParams compareParams{1, 0, 1, 1, 1, 0};
        Brcb(gBrcbLocal, gPairLocal,
            static_cast<uint8_t>(headsPerVector * chunkSize / FP32_BLOCK_ELEMS),
            {1, FP32_BLOCK_ELEMS});
        LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();
        const bool useFusedFactoredBeta = UseFusedFactoredBeta();
        if (useFusedFactoredBeta) {
            for (uint32_t i = 0; i < headsPerVector; ++i) {
                const uint32_t gOffset = i * chunkSize;
                const uint32_t matrixOffset = i * matrixElements;
                const uint32_t broadcastOffset = i * chunkSize * FP32_BLOCK_ELEMS;
                Compare(maskLocal[matrixOffset / 8U], gBrcbLocal[broadcastOffset], gPairLocal[gOffset],
                    CMPMODE::LT, static_cast<uint64_t>(chunkSize), static_cast<uint8_t>(chunkSize),
                    compareParams);
            }
            LocalTensor<float> factorPairLocal = gFactorBuf_.Get<float>();
            Muls(factorPairLocal, gPairLocal, -1.0f, headsPerVector * chunkSize);
            Exp<float>(factorPairLocal, factorPairLocal, headsPerVector * chunkSize);
            Exp<float>(gPairLocal, gPairLocal, headsPerVector * chunkSize);

            LocalTensor<T> betaPairLocal = betaQueue_.template DeQue<T>();
            Cast(gBrcbLocal, betaPairLocal, RoundMode::CAST_NONE, headsPerVector * chunkSize);
            Mul(gPairLocal, gPairLocal, gBrcbLocal, headsPerVector * chunkSize);
            betaQueue_.template FreeTensor<T>(betaPairLocal);

            Brcb(gBrcbLocal, gPairLocal,
                static_cast<uint8_t>(headsPerVector * chunkSize / FP32_BLOCK_ELEMS),
                {1, FP32_BLOCK_ELEMS});
            for (uint32_t i = 0; i < headsPerVector; ++i) {
                const uint32_t gOffset = i * chunkSize;
                const uint32_t matrixOffset = i * matrixElements;
                const uint32_t broadcastOffset = i * chunkSize * FP32_BLOCK_ELEMS;
                Mul(outPairLocal[matrixOffset], gBrcbLocal[broadcastOffset], factorPairLocal[gOffset],
                    static_cast<uint64_t>(chunkSize), static_cast<uint8_t>(chunkSize), broadcastParams);
            }
            Select(outPairLocal, maskLocal, outPairLocal, 0.0f,
                SELMODE::VSEL_TENSOR_SCALAR_MODE, pairMatrixElements);
        } else {
            for (uint32_t i = 0; i < headsPerVector; ++i) {
                const uint32_t gOffset = i * chunkSize;
                const uint32_t matrixOffset = i * matrixElements;
                const uint32_t broadcastOffset = i * chunkSize * FP32_BLOCK_ELEMS;
                Sub(outPairLocal[matrixOffset], gBrcbLocal[broadcastOffset], gPairLocal[gOffset],
                    static_cast<uint64_t>(chunkSize), static_cast<uint8_t>(chunkSize), broadcastParams);
            }
            CompareScalar(maskLocal, outPairLocal, 0.0f, CMPMODE::LT, pairMatrixElements);
            Select(outPairLocal, maskLocal, outPairLocal, SAFE_EXP_NEG_INF,
                SELMODE::VSEL_TENSOR_SCALAR_MODE, pairMatrixElements);
            Exp<float>(outPairLocal, outPairLocal, pairMatrixElements);

            LocalTensor<T> betaPairLocal = betaQueue_.template DeQue<T>();
            LocalTensor<float> betaFloatLocal;
            if constexpr (ALIAS_BETA_FLOAT) {
                betaFloatLocal = maskBuf_.Get<float>();
            } else {
                betaFloatLocal = betaFloatBuf_.Get<float>();
            }
            Cast(betaFloatLocal, betaPairLocal, RoundMode::CAST_NONE, headsPerVector * chunkSize);
            Brcb(gBrcbLocal, betaFloatLocal,
                static_cast<uint8_t>(headsPerVector * chunkSize / FP32_BLOCK_ELEMS),
                {1, FP32_BLOCK_ELEMS});
            BinaryRepeatParams rowScaleParams{1, 1, 0, 8, 8, 1};
            Mul(outPairLocal, outPairLocal, gBrcbLocal,
                static_cast<uint64_t>(chunkSize),
                static_cast<uint8_t>(headsPerVector * chunkSize), rowScaleParams);
            betaQueue_.template FreeTensor<T>(betaPairLocal);
        }
        gQueue_.template FreeTensor<float>(gPairLocal);
        if (dotCopyPending) {
            dotMatrixLocal = dotMatrixQueue_.DeQue<float>();
            SignalWorkspaceConsumed(slotIdx);
        }
        for (uint32_t i = 0; i < headsPerVector; ++i) {
            const uint32_t matrixOffset = i * matrixElements;
            Mul(outPairLocal[matrixOffset], outPairLocal[matrixOffset],
                dotMatrixLocal, matrixElements);
        }

        outQueue_.EnQue<float>(outPairLocal);
        outPairLocal = outQueue_.DeQue<float>();
        if constexpr (PAIR_OUTPUT_MULTIBLOCK) {
            const uint64_t outOffset =
                HeadTokenOffset(headStart, start) * static_cast<uint64_t>(chunkSize);
            const uint32_t headOutputStrideBytes = static_cast<uint32_t>(
                static_cast<uint64_t>(seqLen_ - chunkSize) * chunkSize * sizeof(float));
            DataCopyExtParams pairOutputParams{
                headsPerVector, matrixElements * sizeof(float), 0, headOutputStrideBytes, 0};
            DataCopyPad(outGm_[outOffset], outPairLocal, pairOutputParams);
        } else {
            for (uint32_t i = 0; i < headsPerVector; ++i) {
                const uint64_t outOffset =
                    HeadTokenOffset(headStart + i, start) * static_cast<uint64_t>(chunkSize);
                DataCopy(outGm_[outOffset], outPairLocal[i * matrixElements], matrixElements);
            }
        }
        outQueue_.FreeTensor<float>(outPairLocal);
    }

    __aicore__ inline void ProcessRow(
        uint32_t start,
        uint32_t chunkLen,
        uint32_t row,
        uint32_t headIdx,
        const LocalTensor<float>& negGLocal,
        const LocalTensor<float>& dotMatrixLocal,
        const LocalTensor<float>& outMatrixLocal)
    {
        LocalTensor<float> gammaLocal;
        if constexpr (MANUAL_K128_CUBE) {
            gammaLocal = gBrcbBuf_.Get<float>()[2U * FP32_COMPARE_ELEMS];
        } else {
            gammaLocal = expBuf_.Get<float>();
        }
        FillGamma(chunkLen, start + row, headIdx, negGLocal, gammaLocal);
        const float betaValue = LoadBetaFloat(HeadTokenOffset(headIdx, start + row));
        Muls(gammaLocal, gammaLocal, betaValue, chunkLen);
        const uint32_t rowOffset = row * chunkSize_;
        Mul(outMatrixLocal[rowOffset], gammaLocal, dotMatrixLocal[rowOffset], chunkLen);
    }

    __aicore__ inline void FillGamma(
        uint32_t chunkLen,
        uint32_t globalRow,
        uint32_t headIdx,
        const LocalTensor<float>& negGLocal,
        const LocalTensor<float>& gammaLocal)
    {
        LocalTensor<float> diffLocal;
        if constexpr (MANUAL_K128_CUBE) {
            diffLocal = gBrcbBuf_.Get<float>()[FP32_COMPARE_ELEMS];
        } else {
            diffLocal = diffBuf_.Get<float>();
        }
        LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();
        const uint32_t compareCount = AlignUp(chunkLen, FP32_COMPARE_ELEMS);
        const float rowG = gGm_.GetValue(HeadTokenOffset(headIdx, globalRow));
        Adds(diffLocal, negGLocal, rowG, compareCount);
        CompareScalar(maskLocal, diffLocal, 0.0f, CMPMODE::LT, compareCount);
        Select(diffLocal, maskLocal, diffLocal, SAFE_EXP_NEG_INF,
            SELMODE::VSEL_TENSOR_SCALAR_MODE, compareCount);
        Exp<float, 0, true>(gammaLocal, diffLocal, chunkLen);
    }

    __aicore__ inline void ScaleFullChunk(
        const LocalTensor<float>& outMatrixLocal,
        const LocalTensor<float>& dotMatrixLocal)
    {
        LocalTensor<T> betaLocal = betaQueue_.template DeQue<T>();

        LocalTensor<float> betaFloatLocal;
        if constexpr (ALIAS_BETA_FLOAT) {
            betaFloatLocal = maskBuf_.Get<float>();
        } else {
            betaFloatLocal = betaFloatBuf_.Get<float>();
        }
        LocalTensor<float> betaBrcbLocal = gBrcbBuf_.Get<float>();
        Cast(betaFloatLocal, betaLocal, RoundMode::CAST_NONE, chunkSize_);
        Brcb(betaBrcbLocal, betaFloatLocal,
            static_cast<uint8_t>(chunkSize_ / FP32_BLOCK_ELEMS), {1, FP32_BLOCK_ELEMS});
        const uint32_t matrixElements = chunkSize_ * chunkSize_;
        const uint64_t rowMask = static_cast<uint64_t>(chunkSize_);
        const uint8_t rowRepeats = static_cast<uint8_t>(chunkSize_);
        BinaryRepeatParams rowScaleParams{1, 1, 0, 8, 8, 1};
        Mul(outMatrixLocal, outMatrixLocal, betaBrcbLocal,
            rowMask, rowRepeats, rowScaleParams);
        Mul(outMatrixLocal, outMatrixLocal, dotMatrixLocal, matrixElements);
        betaQueue_.template FreeTensor<T>(betaLocal);
    }

    __aicore__ inline bool UseFusedFactoredBeta() const
    {
        if constexpr (FACTORED_PAIR_POLICY >= 0) {
            return FACTORED_PAIR_POLICY != 0;
        }
        return FUSED_FACTORED_BETA &&
            ((keyDim_ == 128U &&
                (headPerGroup_ == 4U || taskNum_ > DEEP_PIPELINE_TASK_THRESHOLD ||
                    groupHeads_ == 2U)) ||
                (keyDim_ == 256U &&
                    (headPerGroup_ == 4U && taskNum_ <= DEEP_PIPELINE_TASK_THRESHOLD)));
    }

    __aicore__ inline void BuildFullGamma(
        const LocalTensor<float>& gLocal,
        const LocalTensor<float>& gammaMatrixLocal)
    {
        LocalTensor<float> gBrcbLocal = gBrcbBuf_.Get<float>();
        Brcb(gBrcbLocal, gLocal,
            static_cast<uint8_t>(chunkSize_ / FP32_BLOCK_ELEMS), {1, FP32_BLOCK_ELEMS});

        const uint64_t rowMask = static_cast<uint64_t>(chunkSize_);
        const uint8_t rowRepeats = static_cast<uint8_t>(chunkSize_);
        BinaryRepeatParams broadcastParams{1, 0, 1, 8, 1, 0};
        Sub(gammaMatrixLocal, gBrcbLocal, gLocal,
            rowMask, rowRepeats, broadcastParams);

        const uint32_t matrixElements = chunkSize_ * chunkSize_;
        LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();
        CompareScalar(maskLocal, gammaMatrixLocal, 0.0f, CMPMODE::LT, matrixElements);
        Select(gammaMatrixLocal, maskLocal, gammaMatrixLocal, SAFE_EXP_NEG_INF,
            SELMODE::VSEL_TENSOR_SCALAR_MODE, matrixElements);
        Exp<float>(gammaMatrixLocal, gammaMatrixLocal, matrixElements);
    }

    __aicore__ inline void BuildFullGammaRange(
        const LocalTensor<float>& gLocal,
        uint32_t rowBegin,
        uint32_t rowCount,
        const LocalTensor<float>& gammaMatrixLocal)
    {
        LocalTensor<float> gBrcbLocal = gBrcbBuf_.Get<float>();
        Brcb(gBrcbLocal, gLocal[rowBegin],
            static_cast<uint8_t>(rowCount / FP32_BLOCK_ELEMS), {1, FP32_BLOCK_ELEMS});

        const uint64_t rowMask = static_cast<uint64_t>(chunkSize_);
        const uint8_t rowRepeats = static_cast<uint8_t>(rowCount);
        BinaryRepeatParams broadcastParams{1, 0, 1, 8, 1, 0};
        Sub(gammaMatrixLocal, gBrcbLocal, gLocal,
            rowMask, rowRepeats, broadcastParams);

        LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();
        const uint32_t matrixElements = rowCount * chunkSize_;
        CompareScalar(maskLocal, gammaMatrixLocal, 0.0f, CMPMODE::LT, matrixElements);
        Select(gammaMatrixLocal, maskLocal, gammaMatrixLocal, SAFE_EXP_NEG_INF,
            SELMODE::VSEL_TENSOR_SCALAR_MODE, matrixElements);
        Exp<float>(gammaMatrixLocal, gammaMatrixLocal, matrixElements);
    }

    __aicore__ inline void ScaleFullChunkRange(
        uint32_t rowCount,
        const LocalTensor<float>& outMatrixLocal,
        const LocalTensor<float>& dotMatrixLocal)
    {
        LocalTensor<T> betaLocal = betaQueue_.template DeQue<T>();
        LocalTensor<float> betaFloatLocal = betaFloatBuf_.Get<float>();
        LocalTensor<float> betaBrcbLocal = gBrcbBuf_.Get<float>();
        Cast(betaFloatLocal, betaLocal, RoundMode::CAST_NONE, rowCount);
        Brcb(betaBrcbLocal, betaFloatLocal,
            static_cast<uint8_t>(rowCount / FP32_BLOCK_ELEMS), {1, FP32_BLOCK_ELEMS});

        const uint32_t matrixElements = rowCount * chunkSize_;
        const uint64_t rowMask = static_cast<uint64_t>(chunkSize_);
        const uint8_t rowRepeats = static_cast<uint8_t>(rowCount);
        BinaryRepeatParams rowScaleParams{1, 1, 0, 8, 8, 1};
        Mul(outMatrixLocal, outMatrixLocal, betaBrcbLocal,
            rowMask, rowRepeats, rowScaleParams);
        Mul(outMatrixLocal, outMatrixLocal, dotMatrixLocal, matrixElements);
        betaQueue_.template FreeTensor<T>(betaLocal);
    }

    __aicore__ inline void BuildFactoredFullGamma(
        const LocalTensor<float>& gLocal,
        const LocalTensor<float>& gammaMatrixLocal)
    {
        LocalTensor<float> gBrcbLocal = gBrcbBuf_.Get<float>();
        Brcb(gBrcbLocal, gLocal,
            static_cast<uint8_t>(chunkSize_ / FP32_BLOCK_ELEMS), {1, FP32_BLOCK_ELEMS});

        const uint64_t rowMask = static_cast<uint64_t>(chunkSize_);
        const uint8_t rowRepeats = static_cast<uint8_t>(chunkSize_);
        BinaryRepeatParams compareParams{1, 0, 1, 1, 1, 0};
        LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();
        Compare(maskLocal, gBrcbLocal, gLocal, CMPMODE::LT,
            rowMask, rowRepeats, compareParams);

        LocalTensor<float> factorLocal = gFactorBuf_.Get<float>();
        Muls(factorLocal, gLocal, -1.0f, chunkSize_);
        Exp<float>(factorLocal, factorLocal, chunkSize_);
        Exp<float>(gLocal, gLocal, chunkSize_);

        LocalTensor<T> betaLocal = betaQueue_.template DeQue<T>();
        Cast(gBrcbLocal, betaLocal, RoundMode::CAST_NONE, chunkSize_);
        Mul(gLocal, gLocal, gBrcbLocal, chunkSize_);
        betaQueue_.template FreeTensor<T>(betaLocal);

        Brcb(gBrcbLocal, gLocal,
            static_cast<uint8_t>(chunkSize_ / FP32_BLOCK_ELEMS), {1, FP32_BLOCK_ELEMS});
        BinaryRepeatParams broadcastParams{1, 0, 1, 8, 1, 0};
        Mul(gammaMatrixLocal, gBrcbLocal, factorLocal,
            rowMask, rowRepeats, broadcastParams);
        Select(gammaMatrixLocal, maskLocal, gammaMatrixLocal, 0.0f,
            SELMODE::VSEL_TENSOR_SCALAR_MODE, chunkSize_ * chunkSize_);
    }

    __aicore__ inline void BuildDirectFactoredTailGamma(
        const LocalTensor<float>& gLocal,
        const LocalTensor<float>& gammaMatrixLocal,
        uint32_t chunkLen)
    {
        LocalTensor<float> gBrcbLocal = gBrcbBuf_.Get<float>();
        Brcb(gBrcbLocal, gLocal,
            static_cast<uint8_t>(chunkSize_ / FP32_BLOCK_ELEMS), {1, FP32_BLOCK_ELEMS});

        const uint64_t rowMask = static_cast<uint64_t>(chunkSize_);
        const uint8_t rowRepeats = static_cast<uint8_t>(chunkSize_);
        BinaryRepeatParams compareParams{1, 0, 1, 1, 1, 0};
        LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();
        Compare(maskLocal, gBrcbLocal, gLocal, CMPMODE::LT,
            rowMask, rowRepeats, compareParams);

        LocalTensor<float> factorLocal = gFactorBuf_.Get<float>();
        Duplicate(factorLocal, 0.0f, chunkSize_);
        Muls(factorLocal, gLocal, -1.0f, chunkLen);
        Exp<float>(factorLocal, factorLocal, chunkLen);
        Exp<float>(gLocal, gLocal, chunkLen);

        LocalTensor<T> betaLocal = betaQueue_.template DeQue<T>();
        Cast(gBrcbLocal, betaLocal, RoundMode::CAST_NONE, chunkLen);
        Mul(gLocal, gLocal, gBrcbLocal, chunkLen);
        betaQueue_.template FreeTensor<T>(betaLocal);

        Brcb(gBrcbLocal, gLocal,
            static_cast<uint8_t>(chunkSize_ / FP32_BLOCK_ELEMS), {1, FP32_BLOCK_ELEMS});
        BinaryRepeatParams broadcastParams{1, 0, 1, 8, 1, 0};
        Mul(gammaMatrixLocal, gBrcbLocal, factorLocal,
            rowMask, rowRepeats, broadcastParams);
        Select(gammaMatrixLocal, maskLocal, gammaMatrixLocal, 0.0f,
            SELMODE::VSEL_TENSOR_SCALAR_MODE, chunkSize_ * chunkSize_);
    }

    __aicore__ inline void PrefetchBeta(uint64_t headTokenStart, uint32_t chunkLen)
    {
        LocalTensor<T> betaLocal = betaQueue_.template AllocTensor<T>();
        if (chunkLen == chunkSize_) {
            DataCopy(betaLocal, betaGm_[headTokenStart], chunkSize_);
        } else {
            DataCopyExtParams betaCopyParams{
                1, static_cast<uint32_t>(chunkLen * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> betaCopyPad{false, 0, 0, static_cast<T>(0.0f)};
            DataCopyPad(betaLocal, betaGm_[headTokenStart], betaCopyParams, betaCopyPad);
        }
        betaQueue_.template EnQue<T>(betaLocal);
    }

    __aicore__ inline void PrefetchG(uint64_t headTokenStart, uint32_t chunkLen)
    {
        LocalTensor<float> gLocal = gQueue_.template AllocTensor<float>();
        if (chunkLen == chunkSize_) {
            DataCopy(gLocal, gGm_[headTokenStart], chunkSize_);
        } else {
            DataCopyExtParams gCopyParams{
                1, static_cast<uint32_t>(chunkLen * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> gCopyPad{false, 0, 0, 0.0f};
            DataCopyPad(gLocal, gGm_[headTokenStart], gCopyParams, gCopyPad);
        }
        gQueue_.template EnQue<float>(gLocal);
    }

    __aicore__ inline void ScaleTailChunk(
        uint32_t chunkLen,
        const LocalTensor<float>& outMatrixLocal,
        const LocalTensor<float>& dotMatrixLocal)
    {
        LocalTensor<T> betaLocal = betaQueue_.template DeQue<T>();

        LocalTensor<float> betaFloatLocal;
        if constexpr (ALIAS_BETA_FLOAT) {
            betaFloatLocal = maskBuf_.Get<float>();
        } else {
            betaFloatLocal = betaFloatBuf_.Get<float>();
        }
        LocalTensor<float> betaBrcbLocal = gBrcbBuf_.Get<float>();
        Cast(betaFloatLocal, betaLocal, RoundMode::CAST_NONE, chunkLen);
        const uint8_t firstBrcbRepeats = static_cast<uint8_t>(
            (chunkLen + FP32_BLOCK_ELEMS - 1U) / FP32_BLOCK_ELEMS);
        Brcb(betaBrcbLocal, betaFloatLocal, firstBrcbRepeats, {1, FP32_BLOCK_ELEMS});
        const uint64_t rowMask = static_cast<uint64_t>(chunkLen);
        const uint8_t rowRepeats = static_cast<uint8_t>(chunkLen);
        BinaryRepeatParams rowScaleParams{1, 1, 0, 8, 8, 1};
        Mul(outMatrixLocal, outMatrixLocal, betaBrcbLocal, rowMask, rowRepeats, rowScaleParams);
        BinaryRepeatParams rowParams{1, 1, 1, 8, 8, 8};
        Mul(outMatrixLocal, outMatrixLocal, dotMatrixLocal, rowMask, rowRepeats, rowParams);
        betaQueue_.template FreeTensor<T>(betaLocal);
    }

    __aicore__ inline uint64_t HeadTokenOffset(uint32_t headIdx, uint32_t tokenIdx) const
    {
        return static_cast<uint64_t>(headIdx) * seqLen_ + tokenIdx;
    }

private:
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> dotMatrixQueue_;
    TQue<QuePosition::VECIN, (GH2_K128_PAIR_PIPELINE ? BUFFER_NUM : 2U)> gQueue_;
    TQue<QuePosition::VECIN, (GH2_K128_PAIR_PIPELINE ? BUFFER_NUM : 2U)> betaQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> offsetsQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> compactInQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> compactOutQueue_;
    TQue<QuePosition::VECOUT, OUT_BUFFER_NUM> outQueue_;
    TBuf<TPosition::VECCALC> diffBuf_;
    TBuf<TPosition::VECCALC> expBuf_;
    TBuf<TPosition::VECCALC> negGBuf_;
    TBuf<TPosition::VECCALC> gBrcbBuf_;
    TBuf<TPosition::VECCALC> gFactorBuf_;
    TBuf<TPosition::VECCALC> betaFloatBuf_;
    TBuf<TPosition::VECCALC> maskBuf_;
    TQue<TPosition::A1, 1> kL1Queue_;
    TQue<TPosition::B1, 1> kB1Queue_;
    TQue<TPosition::A2, 1> kL0AQueue_;
    TQue<TPosition::B2, 1> kL0BQueue_;
    TQue<TPosition::CO1, 1> dotL0CQueue_;

    GlobalTensor<T> kGm_;
    GlobalTensor<T> betaGm_;
    GlobalTensor<float> gGm_;
    GlobalTensor<int32_t> offsetsGm_;
    GlobalTensor<float> outGm_;
    GlobalTensor<T> compactGm_;
    GlobalTensor<float> workspaceGm_;

    uint32_t blockIdx_ = 0;
    uint32_t subBlockIdx_ = 0;
    uint32_t taskCoreNum_ = 0;
    uint32_t blockFactor_ = 0;
    uint32_t seqLen_ = 0;
    uint32_t heads_ = 0;
    uint32_t groupHeads_ = 0;
    uint32_t keyDim_ = 0;
    uint32_t chunkNum_ = 0;
    uint32_t chunkSize_ = 0;
    uint32_t headPerGroup_ = 0;
    uint32_t taskNum_ = 0;
    uint32_t workspacePerCore_ = 0;
    uint32_t compactWorkspaceBytes_ = 0;
    uint32_t dotWorkspaceBytes_ = 0;
    uint32_t workspaceSlotBytes_ = 0;
    uint32_t currentKRowStride_ = 0;
    uint32_t currentTailSize_ = 0;
    bool valid_ = false;
};

} // namespace NsChunkScaledDotKkt
#endif // CHUNKSCALEDDOTKKT_H
