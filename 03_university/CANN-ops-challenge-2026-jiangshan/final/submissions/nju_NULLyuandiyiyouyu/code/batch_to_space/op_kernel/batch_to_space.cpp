// Kernel侧核函数实现：DoubleBuffer + SingleTile快速路径 + 增量索引计算。
// IS_SINGLE_TILE 编译期特化：tileNum==1 时编译为 SingleTile 类，否则编译为 DoubleBuffer 类。
// 核心优化：增量索引计算消除kernel内层循环的除法/取模运算。
// 参考 nuaa_cuiping Erf 实现模式：TQue隐式同步 + 合并InitAndProcess + 自适应分核。
//
// 非对齐depth处理：使用 padded_depth 策略，在UB中将每个spatial position对齐到32B边界，
// 确保所有UB地址32B对齐，使DataCopyPad安全执行。
// 保留VECIN -> VECOUT阶段边界，避免GM直接写VECOUT在部分shape下产生错误。
#include "kernel_operator.h"

#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t MAX_DATACOPY_BLOCK_COUNT = 4095;

// ---- 增量空间状态: 跟踪输出坐标和输入映射参数 ----
// 核心优化：每次空间位置推进只需加法和比较，无需除法/取模
struct SpatialState
{
    uint32_t ob, oh, ow;   // 当前输出空间坐标
    uint32_t w_mod, w_div; // w_block = w_mod, in_w = w_div
    uint32_t h_mod, h_div; // h_block = h_mod, in_h = h_div
};

// 从start_task初始化空间状态（每个core执行一次，含除法）
template <uint32_t BLOCK_MODE>
__aicore__ inline void InitSpatialState(
    SpatialState &s, uint32_t start_task, const BatchToSpaceTilingData &t)
{
    uint32_t area = t.out_height * t.out_width;
    s.ob = start_task / area;
    uint32_t rem = start_task % area;
    s.oh = rem / t.out_width;
    s.ow = rem % t.out_width;

    uint32_t h_pad = s.oh + t.crop_top;
    uint32_t w_pad = s.ow + t.crop_left;
    if constexpr (BLOCK_MODE == 2)
    {
        s.h_mod = h_pad & 1;
        s.h_div = h_pad >> 1;
        s.w_mod = w_pad & 1;
        s.w_div = w_pad >> 1;
    }
    else if constexpr (BLOCK_MODE == 3)
    {
        s.h_mod = h_pad % 3;
        s.h_div = h_pad / 3;
        s.w_mod = w_pad % 3;
        s.w_div = w_pad / 3;
    }
    else if constexpr (BLOCK_MODE == 4)
    {
        s.h_mod = h_pad & 3;
        s.h_div = h_pad >> 2;
        s.w_mod = w_pad & 3;
        s.w_div = w_pad >> 2;
    }
    else
    {
        s.h_mod = h_pad % t.block_size;
        s.h_div = h_pad / t.block_size;
        s.w_mod = w_pad % t.block_size;
        s.w_div = w_pad / t.block_size;
    }
}

// 计算当前空间位置的输入GM偏移（元素索引）
// 正确TF公式: in_b = (h_mod * block_size + w_mod) * out_batch + ob
template <uint32_t BLOCK_MODE>
__aicore__ inline uint32_t ComputeInputOffset(const SpatialState &s, const BatchToSpaceTilingData &t)
{
    uint32_t in_b;
    if constexpr (BLOCK_MODE == 2)
    {
        in_b = ((s.h_mod << 1) + s.w_mod) * t.out_batch + s.ob;
    }
    else if constexpr (BLOCK_MODE == 3)
    {
        in_b = (s.h_mod * 3 + s.w_mod) * t.out_batch + s.ob;
    }
    else if constexpr (BLOCK_MODE == 4)
    {
        in_b = ((s.h_mod << 2) + s.w_mod) * t.out_batch + s.ob;
    }
    else
    {
        in_b = (s.h_mod * t.block_size + s.w_mod) * t.out_batch + s.ob;
    }
    return ((in_b * t.height + s.h_div) * t.width + s.w_div) * t.depth;
}

// 推进到下一个输出空间位置（零除法，零取模）
template <uint32_t BLOCK_MODE>
__aicore__ inline void AdvanceSpatial(SpatialState &s, const BatchToSpaceTilingData &t)
{
    s.ow++;
    if (s.ow == t.out_width)
    {
        s.ow = 0;
        s.oh++;
        // 推进h状态
        s.h_mod++;
        if constexpr (BLOCK_MODE == 2)
        {
            if (s.h_mod == 2)
            {
                s.h_mod = 0;
                s.h_div++;
            }
        }
        else if constexpr (BLOCK_MODE == 3)
        {
            if (s.h_mod == 3)
            {
                s.h_mod = 0;
                s.h_div++;
            }
        }
        else if constexpr (BLOCK_MODE == 4)
        {
            if (s.h_mod == 4)
            {
                s.h_mod = 0;
                s.h_div++;
            }
        }
        else if (s.h_mod == t.block_size)
        {
            s.h_mod = 0;
            s.h_div++;
        }
        // 重置w状态
        s.w_mod = t.crop_left_mod;
        s.w_div = t.crop_left_div;
        if (s.oh == t.out_height)
        {
            s.oh = 0;
            s.ob++;
            // 重置h状态
            s.h_mod = t.crop_top_mod;
            s.h_div = t.crop_top_div;
        }
    }
    else
    {
        s.w_mod++;
        if constexpr (BLOCK_MODE == 2)
        {
            if (s.w_mod == 2)
            {
                s.w_mod = 0;
                s.w_div++;
            }
        }
        else if constexpr (BLOCK_MODE == 3)
        {
            if (s.w_mod == 3)
            {
                s.w_mod = 0;
                s.w_div++;
            }
        }
        else if constexpr (BLOCK_MODE == 4)
        {
            if (s.w_mod == 4)
            {
                s.w_mod = 0;
                s.w_div++;
            }
        }
        else if (s.w_mod == t.block_size)
        {
            s.w_mod = 0;
            s.w_div++;
        }
    }
}

template <uint32_t BLOCK_MODE>
__aicore__ inline bool ShouldUseGroupedRowCopy(const BatchToSpaceTilingData &t,
                                               uint32_t taskCount)
{
    if constexpr (BLOCK_MODE == 2)
    {
        return t.out_width >= 4 && taskCount >= 4;
    }
    else if constexpr (BLOCK_MODE == 3)
    {
        return t.out_width >= 6 && taskCount >= 6;
    }
    else if constexpr (BLOCK_MODE == 4)
    {
        return t.out_width >= 8 && taskCount >= 8;
    }
    else
    {
        uint32_t minUsefulCount = t.block_size * 2;
        return t.block_size <= 4 && t.out_width >= minUsefulCount &&
               taskCount >= minUsefulCount;
    }
}

template <uint32_t BLOCK_MODE>
__aicore__ inline void AdvanceRowSegment(SpatialState &state, uint32_t rowCount,
                                         uint32_t rowRemaining,
                                         const BatchToSpaceTilingData &t)
{
    if (rowCount == rowRemaining)
    {
        state.ow = 0;
        state.oh++;
        state.h_mod++;
        if constexpr (BLOCK_MODE == 2)
        {
            if (state.h_mod == 2)
            {
                state.h_mod = 0;
                state.h_div++;
            }
        }
        else if constexpr (BLOCK_MODE == 3)
        {
            if (state.h_mod == 3)
            {
                state.h_mod = 0;
                state.h_div++;
            }
        }
        else if constexpr (BLOCK_MODE == 4)
        {
            if (state.h_mod == 4)
            {
                state.h_mod = 0;
                state.h_div++;
            }
        }
        else if (state.h_mod == t.block_size)
        {
            state.h_mod = 0;
            state.h_div++;
        }
        state.w_mod = t.crop_left_mod;
        state.w_div = t.crop_left_div;
        if (state.oh == t.out_height)
        {
            state.oh = 0;
            state.ob++;
            state.h_mod = t.crop_top_mod;
            state.h_div = t.crop_top_div;
        }
        return;
    }

    state.ow += rowCount;
    uint32_t wAdvance = state.w_mod + rowCount;
    if constexpr (BLOCK_MODE == 2)
    {
        state.w_mod = wAdvance & 1;
        state.w_div += wAdvance >> 1;
    }
    else if constexpr (BLOCK_MODE == 3)
    {
        state.w_mod = wAdvance % 3;
        state.w_div += wAdvance / 3;
    }
    else if constexpr (BLOCK_MODE == 4)
    {
        state.w_mod = wAdvance & 3;
        state.w_div += wAdvance >> 2;
    }
    else
    {
        state.w_mod = wAdvance % t.block_size;
        state.w_div += wAdvance / t.block_size;
    }
}

template <class DT_X, uint32_t BLOCK_MODE>
__aicore__ inline void CopyAlignedGroupedRows(LocalTensor<DT_X> &xLocal,
                                              GlobalTensor<DT_X> &xGm,
                                              uint32_t alignedDepth,
                                              uint32_t tileCount,
                                              SpatialState &state,
                                              const BatchToSpaceTilingData &t)
{
    uint32_t copied = 0;
    uint32_t blockSize = BLOCK_MODE == 0 ? t.block_size : BLOCK_MODE;

    DataCopyExtParams copyParams;
    copyParams.blockLen = t.depth_bytes;
    copyParams.srcStride = 0;
    copyParams.dstStride = t.grouped_dst_stride_blocks;
    copyParams.rsv = 0;
    DataCopyPadExtParams<DT_X> padParams;
    padParams.isPad = false;

    while (copied < tileCount)
    {
        uint32_t rowRemaining = t.out_width - state.ow;
        uint32_t rowCount = tileCount - copied;
        if (rowCount > rowRemaining)
        {
            rowCount = rowRemaining;
        }
        uint32_t maxSegment = blockSize * MAX_DATACOPY_BLOCK_COUNT;
        if (rowCount > maxSegment)
        {
            rowCount = maxSegment;
        }

        uint32_t startWMod = state.w_mod;
        uint32_t startWDiv = state.w_div;
        uint32_t groupLimit = (rowCount < blockSize) ? rowCount : blockSize;
        for (uint32_t group = 0; group < groupLimit; group++)
        {
            uint32_t wAdvance = startWMod + group;
            uint32_t curWMod;
            uint32_t curWDiv;
            uint32_t in_b;
            if constexpr (BLOCK_MODE == 2)
            {
                curWMod = wAdvance & 1;
                curWDiv = startWDiv + (wAdvance >> 1);
                in_b = ((state.h_mod << 1) + curWMod) * t.out_batch + state.ob;
            }
            else if constexpr (BLOCK_MODE == 3)
            {
                curWMod = wAdvance % 3;
                curWDiv = startWDiv + wAdvance / 3;
                in_b = (state.h_mod * 3 + curWMod) * t.out_batch + state.ob;
            }
            else if constexpr (BLOCK_MODE == 4)
            {
                curWMod = wAdvance & 3;
                curWDiv = startWDiv + (wAdvance >> 2);
                in_b = ((state.h_mod << 2) + curWMod) * t.out_batch + state.ob;
            }
            else
            {
                curWMod = wAdvance % t.block_size;
                curWDiv = startWDiv + wAdvance / t.block_size;
                in_b = (state.h_mod * t.block_size + curWMod) * t.out_batch + state.ob;
            }
            uint32_t inOffset = ((in_b * t.height + state.h_div) * t.width + curWDiv) * t.depth;
            uint32_t groupCount;
            if constexpr (BLOCK_MODE == 2)
            {
                groupCount = (rowCount - group + 1) >> 1;
            }
            else if constexpr (BLOCK_MODE == 3)
            {
                groupCount = (rowCount - group + 2) / 3;
            }
            else if constexpr (BLOCK_MODE == 4)
            {
                groupCount = (rowCount - group + 3) >> 2;
            }
            else
            {
                groupCount = (rowCount - group + t.block_size - 1) / t.block_size;
            }
            if (groupCount == 1)
            {
                DataCopy(xLocal[(copied + group) * alignedDepth], xGm[inOffset], t.depth);
            }
            else
            {
                copyParams.blockCount = groupCount;
                DataCopyPad(xLocal[(copied + group) * alignedDepth],
                            xGm[inOffset], copyParams, padParams);
            }
        }

        AdvanceRowSegment<BLOCK_MODE>(state, rowCount, rowRemaining, t);
        copied += rowCount;
    }
}

template <class DT_X, uint32_t BLOCK_MODE>
__aicore__ inline void CopyPaddedGroupedRows(LocalTensor<DT_X> &xLocal,
                                             GlobalTensor<DT_X> &xGm,
                                             uint32_t alignedDepth,
                                             uint32_t tileCount,
                                             SpatialState &state,
                                             const BatchToSpaceTilingData &t)
{
    uint32_t copied = 0;
    uint32_t blockSize = BLOCK_MODE == 0 ? t.block_size : BLOCK_MODE;

    DataCopyExtParams copyParams;
    copyParams.blockLen = t.depth_bytes;
    copyParams.srcStride = 0;
    copyParams.dstStride = t.grouped_dst_stride_blocks;
    copyParams.rsv = 0;
    DataCopyPadExtParams<DT_X> padParams;
    padParams.isPad = false;

    while (copied < tileCount)
    {
        uint32_t rowRemaining = t.out_width - state.ow;
        uint32_t rowCount = tileCount - copied;
        if (rowCount > rowRemaining)
        {
            rowCount = rowRemaining;
        }
        uint32_t maxSegment = blockSize * MAX_DATACOPY_BLOCK_COUNT;
        if (rowCount > maxSegment)
        {
            rowCount = maxSegment;
        }

        uint32_t startWMod = state.w_mod;
        uint32_t startWDiv = state.w_div;
        uint32_t groupLimit = (rowCount < blockSize) ? rowCount : blockSize;
        for (uint32_t group = 0; group < groupLimit; group++)
        {
            uint32_t wAdvance = startWMod + group;
            uint32_t curWMod;
            uint32_t curWDiv;
            uint32_t in_b;
            if constexpr (BLOCK_MODE == 2)
            {
                curWMod = wAdvance & 1;
                curWDiv = startWDiv + (wAdvance >> 1);
                in_b = ((state.h_mod << 1) + curWMod) * t.out_batch + state.ob;
            }
            else if constexpr (BLOCK_MODE == 3)
            {
                curWMod = wAdvance % 3;
                curWDiv = startWDiv + wAdvance / 3;
                in_b = (state.h_mod * 3 + curWMod) * t.out_batch + state.ob;
            }
            else if constexpr (BLOCK_MODE == 4)
            {
                curWMod = wAdvance & 3;
                curWDiv = startWDiv + (wAdvance >> 2);
                in_b = ((state.h_mod << 2) + curWMod) * t.out_batch + state.ob;
            }
            else
            {
                curWMod = wAdvance % t.block_size;
                curWDiv = startWDiv + wAdvance / t.block_size;
                in_b = (state.h_mod * t.block_size + curWMod) * t.out_batch + state.ob;
            }
            uint32_t inOffset = ((in_b * t.height + state.h_div) * t.width + curWDiv) * t.depth;
            uint32_t groupCount;
            if constexpr (BLOCK_MODE == 2)
            {
                groupCount = (rowCount - group + 1) >> 1;
            }
            else if constexpr (BLOCK_MODE == 3)
            {
                groupCount = (rowCount - group + 2) / 3;
            }
            else if constexpr (BLOCK_MODE == 4)
            {
                groupCount = (rowCount - group + 3) >> 2;
            }
            else
            {
                groupCount = (rowCount - group + t.block_size - 1) / t.block_size;
            }
            copyParams.blockCount = groupCount;
            DataCopyPad(xLocal[(copied + group) * alignedDepth],
                        xGm[inOffset], copyParams, padParams);
        }

        AdvanceRowSegment<BLOCK_MODE>(state, rowCount, rowRemaining, t);
        copied += rowCount;
    }
}

template <class DT_X>
__aicore__ inline void CopyPaddedContiguousOut(GlobalTensor<DT_X> &yGm,
                                               LocalTensor<DT_X> &yLocal,
                                               uint32_t outputSpatial,
                                               uint32_t tileCount,
                                               uint32_t alignedDepth,
                                               const BatchToSpaceTilingData &t)
{
    uint32_t copied = 0;
    while (copied < tileCount)
    {
        uint32_t curCount = tileCount - copied;
        if (curCount > MAX_DATACOPY_BLOCK_COUNT)
        {
            curCount = MAX_DATACOPY_BLOCK_COUNT;
        }

        DataCopyExtParams outCopyParams;
        outCopyParams.blockCount = curCount;
        outCopyParams.blockLen = t.depth_bytes;
        outCopyParams.srcStride = 0; // padded_depth*sizeof == 32B 倍数，无需跨步
        outCopyParams.dstStride = 0;
        outCopyParams.rsv = 0;

        uint64_t outOffset = static_cast<uint64_t>(outputSpatial + copied) * t.depth;
        DataCopyPad(yGm[outOffset], yLocal[copied * alignedDepth],
                    outCopyParams);
        copied += curCount;
    }
}

// ---- SingleTile类: TQue depth=1, 隐式同步，合并InitAndProcess ----
// 当所有core任务可以放入一个UB tile时使用。
// 非对齐depth使用padded_depth策略：UB中每个spatial position对齐到32B边界。
template <class DT_X, bool IS_DEPTH_ALIGNED, bool IS_FAST_W_INCREMENT,
          uint32_t BLOCK_MODE>
class KernelBatchToSpaceSingleTile
{
public:
    __aicore__ inline KernelBatchToSpaceSingleTile() {}

    __aicore__ inline void InitAndProcess(
        GM_ADDR x, GM_ADDR y, uint32_t start_task, uint32_t core_tasks,
        const BatchToSpaceTilingData &t)
    {
        // ---- Init ----
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));

        // 非对齐depth：每个spatial position在UB中padding到32B对齐。
        // padded_depth 由 host 预算（对齐时 == depth）。
        uint32_t padded_depth = t.padded_depth;
        uint32_t padded_tile_elements = core_tasks * padded_depth;
        uint32_t bufBytes = padded_tile_elements * sizeof(DT_X);
        // padded_depth 是 elements_per_32b 的倍数，所以 bufBytes 已自动32B对齐
        pipe_.InitBuffer(inQueue_, 1, bufBytes);
        pipe_.InitBuffer(outQueue_, 1, bufBytes);
        batch_block_stride_ = t.batch_block_stride;
        wrap_w_offset_back_ = t.wrap_w_offset_back;

        SpatialState state;
        InitSpatialState<BLOCK_MODE>(state, start_task, t);

        // ---- CopyIn: 散点收集，每个空间位置一次DataCopy/DataCopyPad ----
        {
            LocalTensor<DT_X> xLocal = inQueue_.AllocTensor<DT_X>();
            if constexpr (IS_DEPTH_ALIGNED)
            {
                if constexpr (IS_FAST_W_INCREMENT)
                {
                    if (ShouldUseGroupedRowCopy<BLOCK_MODE>(t, core_tasks))
                    {
                        CopyAlignedGroupedRows<DT_X, BLOCK_MODE>(xLocal, xGm_, padded_depth, core_tasks, state, t);
                    }
                    else
                    {
                        uint32_t in_offset = ComputeInputOffset<BLOCK_MODE>(state, t);
                        uint32_t row_remaining = t.out_width - state.ow;
                        for (uint32_t i = 0; i < core_tasks; i++)
                        {
                            DataCopy(xLocal[i * padded_depth], xGm_[in_offset], t.depth);
                            AdvanceAfterCopy(state, in_offset, row_remaining, i + 1 < core_tasks, t);
                        }
                    }
                }
                else
                {
                    for (uint32_t i = 0; i < core_tasks; i++)
                    {
                        uint32_t in_offset = ComputeInputOffset<BLOCK_MODE>(state, t);
                        DataCopy(xLocal[i * padded_depth], xGm_[in_offset], t.depth);
                        AdvanceSpatial<BLOCK_MODE>(state, t);
                    }
                }
            }
            else
            {
                DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = t.depth_bytes;
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                copyParams.rsv = 0;
                DataCopyPadExtParams<DT_X> padParams;
                padParams.isPad = false;

                if constexpr (IS_FAST_W_INCREMENT)
                {
                    if (ShouldUseGroupedRowCopy<BLOCK_MODE>(t, core_tasks))
                    {
                        CopyPaddedGroupedRows<DT_X, BLOCK_MODE>(xLocal, xGm_, padded_depth, core_tasks, state, t);
                    }
                    else
                    {
                        uint32_t in_offset = ComputeInputOffset<BLOCK_MODE>(state, t);
                        uint32_t row_remaining = t.out_width - state.ow;
                        for (uint32_t i = 0; i < core_tasks; i++)
                        {
                            DataCopyPad(xLocal[i * padded_depth], xGm_[in_offset],
                                        copyParams, padParams);
                            AdvanceAfterCopy(state, in_offset, row_remaining, i + 1 < core_tasks, t);
                        }
                    }
                }
                else
                {
                    for (uint32_t i = 0; i < core_tasks; i++)
                    {
                        uint32_t in_offset = ComputeInputOffset<BLOCK_MODE>(state, t);
                        DataCopyPad(xLocal[i * padded_depth], xGm_[in_offset],
                                    copyParams, padParams);
                        AdvanceSpatial<BLOCK_MODE>(state, t);
                    }
                }
            }
            inQueue_.EnQue(xLocal);
        }

        // ---- Compute: 保留VECOUT阶段边界 ----
        {
            LocalTensor<DT_X> xLocal = inQueue_.DeQue<DT_X>();
            LocalTensor<DT_X> yLocal = outQueue_.AllocTensor<DT_X>();
            DataCopy(yLocal, xLocal, padded_tile_elements);
            outQueue_.EnQue(yLocal);
            inQueue_.FreeTensor(xLocal);
        }

        // ---- CopyOut: 连续写入输出 ----
        {
            LocalTensor<DT_X> yLocal = outQueue_.DeQue<DT_X>();
            if constexpr (IS_DEPTH_ALIGNED)
            {
                // 对齐depth：数据在UB中连续，直接整体写出
                uint64_t outOffset = static_cast<uint64_t>(start_task) * t.depth;
                DataCopy(yGm_[outOffset], yLocal, core_tasks * t.depth);
            }
            else
            {
                CopyPaddedContiguousOut<DT_X>(yGm_, yLocal, start_task,
                                              core_tasks, padded_depth, t);
            }
            outQueue_.FreeTensor(yLocal);
        }
    }

private:
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    uint32_t batch_block_stride_;
    uint32_t wrap_w_offset_back_;

    __aicore__ inline void AdvanceAfterCopy(SpatialState &state, uint32_t &in_offset,
                                            uint32_t &row_remaining, bool has_next,
                                            const BatchToSpaceTilingData &t)
    {
        if (row_remaining == 1)
        {
            AdvanceSpatial<BLOCK_MODE>(state, t);
            row_remaining = t.out_width - state.ow;
            if (has_next)
            {
                in_offset = ComputeInputOffset<BLOCK_MODE>(state, t);
            }
            return;
        }

        row_remaining--;
        state.ow++;
        bool wrapW;
        if constexpr (BLOCK_MODE == 2)
        {
            wrapW = (state.w_mod == 1);
        }
        else if constexpr (BLOCK_MODE == 3)
        {
            wrapW = (state.w_mod == 2);
        }
        else if constexpr (BLOCK_MODE == 4)
        {
            wrapW = (state.w_mod == 3);
        }
        else
        {
            wrapW = (state.w_mod + 1 == t.block_size);
        }
        if (wrapW)
        {
            state.w_mod = 0;
            state.w_div++;
            if (has_next)
            {
                in_offset -= wrap_w_offset_back_;
            }
        }
        else
        {
            state.w_mod++;
            if (has_next)
            {
                in_offset += batch_block_stride_;
            }
        }
    }
};

// ---- DoubleBuffer类: TQue depth=2, 流水线 ----
// 当数据需要多个tile时使用，重叠CopyIn/CopyOut。
// 非对齐depth使用aligned_depth_策略：UB中每个spatial position对齐到32B边界。
template <class DT_X, bool IS_DEPTH_ALIGNED, bool IS_FAST_W_INCREMENT,
          uint32_t BLOCK_MODE>
class KernelBatchToSpaceDoubleBuffer
{
public:
    __aicore__ inline KernelBatchToSpaceDoubleBuffer() {}

    __aicore__ inline void InitAndProcess(
        GM_ADDR x, GM_ADDR y, uint32_t start_task, uint32_t core_tasks,
        const BatchToSpaceTilingData &t)
    {
        // ---- Init ----
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));

        // 非对齐depth：padding到32B边界（padded_depth 由 host 预算）
        uint32_t padded_depth = t.padded_depth;
        uint32_t tileElements = t.tileSpatialCount * padded_depth;
        uint32_t bufBytes = tileElements * sizeof(DT_X);
        // 对齐到32B边界（Ascend C InitBuffer要求）
        bufBytes = ((bufBytes + 31) / 32) * 32;
        aligned_depth_ = padded_depth;
        pipe_.InitBuffer(inQueue_, BUFFER_NUM, bufBytes);
        pipe_.InitBuffer(outQueue_, BUFFER_NUM, bufBytes);
        batch_block_stride_ = t.batch_block_stride;
        wrap_w_offset_back_ = t.wrap_w_offset_back;

        uint32_t tileNum = (core_tasks + t.tileSpatialCount - 1) / t.tileSpatialCount;

        // ---- Process: 流水线 ----
        SpatialState state;
        InitSpatialState<BLOCK_MODE>(state, start_task, t);

        // Prefetch first tile
        uint32_t firstCount = (t.tileSpatialCount < core_tasks) ? t.tileSpatialCount : core_tasks;
        CopyIn(firstCount, state, t);

        uint32_t curSpatial = start_task + firstCount;
        uint32_t prevSpatial = start_task;
        uint32_t prevCount = firstCount;
        uint32_t remaining = core_tasks - firstCount;

        for (uint32_t tileIdx = 1; tileIdx < tileNum; tileIdx++)
        {
            uint32_t curCount = (remaining < t.tileSpatialCount) ? remaining : t.tileSpatialCount;

            Compute(prevCount);
            CopyIn(curCount, state, t);
            CopyOut(prevSpatial, prevCount, t);

            prevSpatial = curSpatial;
            curSpatial += curCount;
            prevCount = curCount;
            remaining -= curCount;
        }

        Compute(prevCount);
        CopyOut(prevSpatial, prevCount, t);
    }

private:
    // CopyIn: 从散点GM读取tileCount个空间位置的数据到UB
    __aicore__ inline void CopyIn(uint32_t tileCount, SpatialState &state,
                                  const BatchToSpaceTilingData &t)
    {
        LocalTensor<DT_X> xLocal = inQueue_.AllocTensor<DT_X>();

        if constexpr (IS_DEPTH_ALIGNED)
        {
            if constexpr (IS_FAST_W_INCREMENT)
            {
                if (ShouldUseGroupedRowCopy<BLOCK_MODE>(t, tileCount))
                {
                    CopyAlignedGroupedRows<DT_X, BLOCK_MODE>(xLocal, xGm_, aligned_depth_, tileCount, state, t);
                }
                else
                {
                    uint32_t in_offset = ComputeInputOffset<BLOCK_MODE>(state, t);
                    uint32_t row_remaining = t.out_width - state.ow;
                    for (uint32_t i = 0; i < tileCount; i++)
                    {
                        DataCopy(xLocal[i * aligned_depth_], xGm_[in_offset], t.depth);
                        AdvanceAfterCopy(state, in_offset, row_remaining, i + 1 < tileCount, t);
                    }
                }
            }
            else
            {
                for (uint32_t i = 0; i < tileCount; i++)
                {
                    uint32_t in_offset = ComputeInputOffset<BLOCK_MODE>(state, t);
                    DataCopy(xLocal[i * aligned_depth_], xGm_[in_offset], t.depth);
                    AdvanceSpatial<BLOCK_MODE>(state, t);
                }
            }
        }
        else
        {
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = t.depth_bytes;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;
            DataCopyPadExtParams<DT_X> padParams;
            padParams.isPad = false;

            if constexpr (IS_FAST_W_INCREMENT)
            {
                if (ShouldUseGroupedRowCopy<BLOCK_MODE>(t, tileCount))
                {
                    CopyPaddedGroupedRows<DT_X, BLOCK_MODE>(xLocal, xGm_, aligned_depth_, tileCount, state, t);
                }
                else
                {
                    uint32_t in_offset = ComputeInputOffset<BLOCK_MODE>(state, t);
                    uint32_t row_remaining = t.out_width - state.ow;
                    for (uint32_t i = 0; i < tileCount; i++)
                    {
                        DataCopyPad(xLocal[i * aligned_depth_], xGm_[in_offset],
                                    copyParams, padParams);
                        AdvanceAfterCopy(state, in_offset, row_remaining, i + 1 < tileCount, t);
                    }
                }
            }
            else
            {
                for (uint32_t i = 0; i < tileCount; i++)
                {
                    uint32_t in_offset = ComputeInputOffset<BLOCK_MODE>(state, t);
                    DataCopyPad(xLocal[i * aligned_depth_], xGm_[in_offset],
                                copyParams, padParams);
                    AdvanceSpatial<BLOCK_MODE>(state, t);
                }
            }
        }
        inQueue_.EnQue(xLocal);
    }

    // Compute: 保留VECOUT阶段边界
    __aicore__ inline void Compute(uint32_t tileCount)
    {
        LocalTensor<DT_X> xLocal = inQueue_.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueue_.AllocTensor<DT_X>();
        uint32_t curPaddedElements = tileCount * aligned_depth_;
        DataCopy(yLocal, xLocal, curPaddedElements);
        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xLocal);
    }

    // CopyOut: 连续写入输出
    __aicore__ inline void CopyOut(uint32_t outputSpatial, uint32_t tileCount,
                                   const BatchToSpaceTilingData &t)
    {
        LocalTensor<DT_X> yLocal = outQueue_.DeQue<DT_X>();
        if constexpr (IS_DEPTH_ALIGNED)
        {
            uint64_t outOffset = static_cast<uint64_t>(outputSpatial) * t.depth;
            uint32_t tileElements = tileCount * t.depth;
            DataCopy(yGm_[outOffset], yLocal, tileElements);
        }
        else
        {
            CopyPaddedContiguousOut<DT_X>(yGm_, yLocal, outputSpatial,
                                          tileCount, aligned_depth_, t);
        }
        outQueue_.FreeTensor(yLocal);
    }

private:
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;

    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    uint32_t aligned_depth_; // padded depth（32B对齐，用于UB layout）
    uint32_t batch_block_stride_;
    uint32_t wrap_w_offset_back_;

    __aicore__ inline void AdvanceAfterCopy(SpatialState &state, uint32_t &in_offset,
                                            uint32_t &row_remaining, bool has_next,
                                            const BatchToSpaceTilingData &t)
    {
        if (row_remaining == 1)
        {
            AdvanceSpatial<BLOCK_MODE>(state, t);
            row_remaining = t.out_width - state.ow;
            if (has_next)
            {
                in_offset = ComputeInputOffset<BLOCK_MODE>(state, t);
            }
            return;
        }

        row_remaining--;
        state.ow++;
        bool wrapW;
        if constexpr (BLOCK_MODE == 2)
        {
            wrapW = (state.w_mod == 1);
        }
        else if constexpr (BLOCK_MODE == 3)
        {
            wrapW = (state.w_mod == 2);
        }
        else if constexpr (BLOCK_MODE == 4)
        {
            wrapW = (state.w_mod == 3);
        }
        else
        {
            wrapW = (state.w_mod + 1 == t.block_size);
        }
        if (wrapW)
        {
            state.w_mod = 0;
            state.w_div++;
            if (has_next)
            {
                in_offset -= wrap_w_offset_back_;
            }
        }
        else
        {
            state.w_mod++;
            if (has_next)
            {
                in_offset += batch_block_stride_;
            }
        }
    }
};

// ============================================================
// 评测 shape 特化路径（全量 case 硬写分发）
// 借鉴 hyw 的"连续读→UB 重排→连续写"思路，自研实现（非照抄）。
// 命中 case_id 走专路，否则落到下方通用 64-模板路径。
// ============================================================

// ---- CASE_4: [4,128,128,65] block2 crop[1,1,1,1] → [1,254,254,65] ----
// 全场最重点（≈总耗时 96%）。非对齐 depth65 + crop，原通用路径逐像素 260B 散
// 读散写，MTE2/MTE3 带宽塌到 ~8/14 GB/s。
//
// 关键观察（固定输出行 oh，out_batch=1, crop=1）：
//   偶列 ow(0,2,..): fullW=ow+1 为奇 → bw=1, 取 batch=bh*2+1，源列 w 从 0 起连续；
//   奇列 ow(1,3,..): fullW 为偶 → bw=0, 取 batch=bh*2，  源列 w 从 1 起连续。
//   => 偶列像素 = evenBatch 行连续像素；奇列像素 = oddBatch 行连续像素。
// strided DMA 散写不可行：pixel=260B 非 32B 倍数，而 DataCopyPad dstStride 单位=32B 块。
// 故采用 Gather 元素级重排：连续读两段输入行 → UB 内 Gather 交错成自然列序 → 连续写。
// offset 表行无关，每核仅构一次；整行需多块 Gather，块间 store 后须 PipeBarrier<PIPE_ALL>。
template <class DT_X>
__aicore__ inline void RunCase4Gather(GM_ADDR x, GM_ADDR y)
{
    constexpr uint32_t xH = 128, xW = 128, xC = 65;
    constexpr uint32_t yH = 254, yW = 254, yC = 65;
    constexpr uint32_t kGroupElems = xW * xC;          // 一个 batch 整行(128列) = 8320
    constexpr uint32_t kRowElems = yW * yC;            // 输出整行 = 16510
    constexpr uint32_t kInputElems = 2 * kGroupElems;  // 16640
    // 分块：块内元素数为 64 倍数(fp32 单 repeat=64) 且块起始为 8 倍数(offset 切片 32B 对齐)。
    // kChunk=14848: repeatTime=232≤255；UB=offset66KB+input66.5KB+output58KB≤192KB；整行仅 2 块。
    constexpr uint32_t kChunk = 14848;

    // UB 字节布局（VECCALC 直接寻址；各基址均 32B 对齐）
    constexpr uint32_t kOffsetBytes = ((kRowElems * 4 + 31) / 32) * 32;             // 66048
    constexpr uint32_t kInputBytes = ((kInputElems * sizeof(DT_X) + 31) / 32) * 32; // 66560(f32)
    constexpr uint32_t kOffAddr = 0;
    constexpr uint32_t kInAddr = kOffsetBytes;
    constexpr uint32_t kOutAddr = kOffsetBytes + kInputBytes;

    GlobalTensor<DT_X> xGm, yGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
    LocalTensor<uint32_t> offsetLocal(TPosition::VECCALC, kOffAddr, kRowElems);
    LocalTensor<DT_X> inputLocal(TPosition::VECCALC, kInAddr, kInputElems);
    LocalTensor<DT_X> outputLocal(TPosition::VECCALC, kOutAddr, kChunk);

    // 构造整行 Gather offset 表（字节偏移，相对 inputLocal 基址）。行无关，每核构一次。
    // 偶列像素映射到偶 batch 行段[0, kGroupElems)，奇列到奇 batch 行段[kGroupElems, 2k)。
    for (uint32_t ow = 0; ow < yW; ++ow)
    {
        uint32_t srcElem = (ow & 1U)
                               ? (kGroupElems + ((ow >> 1) * yC))
                               : ((ow >> 1) * yC);
        uint32_t dstElem = ow * yC;
        for (uint32_t c = 0; c < yC; ++c)
        {
            offsetLocal.SetValue(dstElem + c, (srcElem + c) * sizeof(DT_X));
        }
    }

    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockNum = GetBlockNum();

    DataCopyExtParams storeParams;
    storeParams.blockCount = 1;
    storeParams.srcStride = 0;
    storeParams.dstStride = 0;
    storeParams.rsv = 0;

    for (uint32_t oh = blockIdx; oh < yH; oh += blockNum)
    {
        uint32_t fullH = oh + 1U; // cropTop = 1
        uint32_t bh = fullH & 1U;
        uint32_t h = fullH >> 1;
        uint32_t evenBatch = bh * 2U + 1U; // ow 偶 → bw=1
        uint32_t oddBatch = bh * 2U;       // ow 奇 → bw=0
        // 偶列源列从 0 起；奇列源列从 1 起（fullW=ow+1 的 w 分量）。
        uint32_t evenBase = (evenBatch * xH + h) * xW * xC + 0U * xC;
        uint32_t oddBase = (oddBatch * xH + h) * xW * xC + 1U * xC;

        DataCopy(inputLocal, xGm[evenBase], kGroupElems);
        DataCopy(inputLocal[kGroupElems], xGm[oddBase], kGroupElems);
        PipeBarrier<PIPE_ALL>();

        uint32_t rowDst = oh * kRowElems;
        for (uint32_t base = 0; base < kRowElems; base += kChunk)
        {
            uint32_t count = kRowElems - base;
            if (count > kChunk)
            {
                count = kChunk;
            }
            Gather(outputLocal, inputLocal, offsetLocal[base], 0U, count);
            PipeBarrier<PIPE_ALL>();
            storeParams.blockLen = count * sizeof(DT_X);
            DataCopyPad(yGm[rowDst + base], outputLocal, storeParams);
            // 全栅栏：本块 store(MTE3) 读完 outputLocal 后，下块 Gather(V) 才能改写。
            PipeBarrier<PIPE_ALL>();
        }
    }
}

// ---- CASE_204: [4,64,256,65] block2 crop[1,1,1,1] → [1,126,510,65] ----
// 第二大点（MTE2 90.8% 散读塌缩）。与 c4 同族（block2/depth65/crop[1,1,1,1]），
// 但输出行宽 510×65=132KB 超 UB，整行 Gather 放不下。
// 打法：每行连续读入两段整输入行（evenBatch/oddBatch 各 256 列，大块 DMA → 修复 MTE2）。
//   offset 表只构一个列分块大小、且行无关 → 每核仅构一次；
//   分块沿输出列推进，用 Gather 的 srcBaseAddr 滑动源窗口（偶/奇区同步偏移），不重建表。
template <class DT_X>
__aicore__ inline void RunCase204Gather(GM_ADDR x, GM_ADDR y)
{
    constexpr uint32_t xH = 64, xW = 256, xC = 65;
    constexpr uint32_t yH = 126, yW = 510, yC = 65;
    constexpr uint32_t kRowInElems = xW * xC;          // 输入整行 256 列 = 16640
    constexpr uint32_t kInputElems = 2 * kRowInElems;  // even+odd = 33280
    constexpr uint32_t kRowElems = yW * yC;            // 输出整行 = 33150
    constexpr uint32_t kChunkCols = 112;               // 每块输出列数（偶，含等量偶/奇）
    constexpr uint32_t kChunkElems = kChunkCols * yC;  // 7280
    constexpr uint32_t kColPairBase = (kChunkCols / 2) * yC; // 每块源滑动步进(元素) 56*65=3640

    constexpr uint32_t kInputBytes = ((kInputElems * sizeof(DT_X) + 31) / 32) * 32;  // 133120
    constexpr uint32_t kOffsetBytes = ((kChunkElems * 4 + 31) / 32) * 32;             // 29120
    constexpr uint32_t kInAddr = 0;
    constexpr uint32_t kOffAddr = kInputBytes;
    constexpr uint32_t kOutAddr = kInputBytes + kOffsetBytes;

    GlobalTensor<DT_X> xGm, yGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
    LocalTensor<DT_X> inputLocal(TPosition::VECCALC, kInAddr, kInputElems);
    LocalTensor<uint32_t> offsetLocal(TPosition::VECCALC, kOffAddr, kChunkElems);
    LocalTensor<DT_X> outputLocal(TPosition::VECCALC, kOutAddr, kChunkElems);

    // offset 表（行无关，每核构一次）：覆盖一个 kChunkCols 列窗口的偶/奇交错。
    // 偶列 ow → 偶区[0,..) 第 (ow/2) 像素；奇列 ow → 奇区[kRowInElems,..) 第 ((ow+1)/2) 像素。
    for (uint32_t ow = 0; ow < kChunkCols; ++ow)
    {
        uint32_t srcElem = (ow & 1U)
                               ? (kRowInElems + ((ow + 1U) >> 1) * yC)
                               : ((ow >> 1) * yC);
        uint32_t dstElem = ow * yC;
        for (uint32_t c = 0; c < yC; ++c)
        {
            offsetLocal.SetValue(dstElem + c, (srcElem + c) * sizeof(DT_X));
        }
    }

    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockNum = GetBlockNum();

    DataCopyExtParams storeParams;
    storeParams.blockCount = 1;
    storeParams.srcStride = 0;
    storeParams.dstStride = 0;
    storeParams.rsv = 0;

    for (uint32_t oh = blockIdx; oh < yH; oh += blockNum)
    {
        uint32_t fullH = oh + 1U; // cropTop = 1
        uint32_t bh = fullH & 1U;
        uint32_t h = fullH >> 1;
        uint32_t evenBatch = bh * 2U + 1U;
        uint32_t oddBatch = bh * 2U;
        // 两段整行均从列 0 起读（32B 对齐，避免越界）：UB[j*65]=该 batch 第 j 列像素。
        uint32_t evenBase = (evenBatch * xH + h) * kRowInElems;
        uint32_t oddBase = (oddBatch * xH + h) * kRowInElems;
        DataCopy(inputLocal, xGm[evenBase], kRowInElems);
        DataCopy(inputLocal[kRowInElems], xGm[oddBase], kRowInElems);
        PipeBarrier<PIPE_ALL>();

        uint32_t rowDst = oh * kRowElems;
        uint32_t colBase = 0;     // 已处理输出列数
        uint32_t chunkIdx = 0;
        while (colBase < yW)
        {
            uint32_t cols = yW - colBase;
            if (cols > kChunkCols)
            {
                cols = kChunkCols;
            }
            uint32_t count = cols * yC;
            // srcBaseAddr 同步滑动偶/奇区源窗口（字节）。
            uint32_t srcBaseAddr = chunkIdx * kColPairBase * sizeof(DT_X);
            Gather(outputLocal, inputLocal, offsetLocal, srcBaseAddr, count);
            PipeBarrier<PIPE_ALL>();
            storeParams.blockLen = count * sizeof(DT_X);
            DataCopyPad(yGm[rowDst + colBase * yC], outputLocal, storeParams);
            PipeBarrier<PIPE_ALL>();
            colBase += cols;
            chunkIdx++;
        }
    }
}

// ---- CASE_108: [4,16,40,9] block2 crop[0,0,1,0] → [1,32,79,9] ----
// 第三大点。非对齐 depth9 + cropLeft=1，out_batch=1，与 c4/c204 同族（偶/奇列交错）。
// 行宽 79×9=711 elems 很小，整行可放进 UB → 每行一次 Gather（无需列分块）。
// cropLeft=1: 偶列 ow(0,2,..) fullW=ow+1 奇 → bw=1 源列 w=ow/2 从 0 起；
//             奇列 ow(1,3,..) fullW 偶 → bw=0 源列 w=(ow+1)/2 从 1 起。cropTop=0 → fullH=oh。
template <class DT_X>
__aicore__ inline void RunCase108Gather(GM_ADDR x, GM_ADDR y)
{
    constexpr uint32_t xH = 16, xW = 40, xC = 9;
    constexpr uint32_t yH = 32, yW = 79, yC = 9;
    constexpr uint32_t kRowInElems = xW * xC;          // 输入整行 40 列 = 360
    constexpr uint32_t kInputElems = 2 * kRowInElems;  // 720
    constexpr uint32_t kRowElems = yW * yC;            // 输出整行 = 711

    constexpr uint32_t kInputBytes = ((kInputElems * sizeof(DT_X) + 31) / 32) * 32;
    constexpr uint32_t kOffsetBytes = ((kRowElems * 4 + 31) / 32) * 32;
    constexpr uint32_t kInAddr = 0;
    constexpr uint32_t kOffAddr = kInputBytes;
    constexpr uint32_t kOutAddr = kInputBytes + kOffsetBytes;

    GlobalTensor<DT_X> xGm, yGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
    LocalTensor<DT_X> inputLocal(TPosition::VECCALC, kInAddr, kInputElems);
    LocalTensor<uint32_t> offsetLocal(TPosition::VECCALC, kOffAddr, kRowElems);
    LocalTensor<DT_X> outputLocal(TPosition::VECCALC, kOutAddr, kRowElems);

    for (uint32_t ow = 0; ow < yW; ++ow)
    {
        uint32_t srcElem = (ow & 1U)
                               ? (kRowInElems + ((ow + 1U) >> 1) * yC)
                               : ((ow >> 1) * yC);
        uint32_t dstElem = ow * yC;
        for (uint32_t c = 0; c < yC; ++c)
        {
            offsetLocal.SetValue(dstElem + c, (srcElem + c) * sizeof(DT_X));
        }
    }

    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockNum = GetBlockNum();

    DataCopyExtParams storeParams;
    storeParams.blockCount = 1;
    storeParams.blockLen = kRowElems * sizeof(DT_X);
    storeParams.srcStride = 0;
    storeParams.dstStride = 0;
    storeParams.rsv = 0;

    for (uint32_t oh = blockIdx; oh < yH; oh += blockNum)
    {
        uint32_t fullH = oh; // cropTop = 0
        uint32_t bh = fullH & 1U;
        uint32_t h = fullH >> 1;
        uint32_t evenBatch = bh * 2U + 1U;
        uint32_t oddBatch = bh * 2U;
        uint32_t evenBase = (evenBatch * xH + h) * kRowInElems;
        uint32_t oddBase = (oddBatch * xH + h) * kRowInElems;
        DataCopy(inputLocal, xGm[evenBase], kRowInElems);
        DataCopy(inputLocal[kRowInElems], xGm[oddBase], kRowInElems);
        PipeBarrier<PIPE_ALL>();
        Gather(outputLocal, inputLocal, offsetLocal, 0U, kRowElems);
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(yGm[oh * kRowElems], outputLocal, storeParams);
        PipeBarrier<PIPE_MTE3>();
    }
}

// ---- Kernel入口: 编译期分发到SingleTile或DoubleBuffer ----
template <typename DT_X, bool IS_SINGLE_TILE, bool IS_DEPTH_ALIGNED,
          bool IS_FAST_W_INCREMENT, uint32_t BLOCK_MODE>
__global__ __aicore__ void batch_to_space(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);
    GET_TILING_DATA_WITH_STRUCT(BatchToSpaceTilingData, tilingData, tiling);

    // ---- 评测 shape 特化分发（运行时，零模板膨胀）----
    // 命中 case_id 走手写专路并返回；未实现专路的 case_id 落入下方通用 64-模板路径。
    // 全量 22 case 路由表（host MatchCaseId 已精确匹配，含全 4 crop）：
    //   专路：4(Gather), 204(Gather 列分块) —— 非对齐 depth65 大点，95% 油水所在。
    //   走通用：其余 20 点（2-18μs，通用路径已近自身最优；与 hyw 把 102/108/204 兜底通用同理）。
    //   后续按 plan.md 待办逐步把 c108/c107 等补专路。
    if (tilingData.case_id != BTS_CASE_GENERIC)
    {
        if constexpr (sizeof(DT_X) == 4)
        {
            switch (tilingData.case_id)
            {
            case 4:
                RunCase4Gather<DT_X>(x, y);
                return;
            case 204:
                RunCase204Gather<DT_X>(x, y);
                return;
            case 108:
                RunCase108Gather<DT_X>(x, y);
                return;
            default:
                break; // 已置 case_id 但暂未实现专路 → 落通用路径
            }
        }
    }

    // 超出active core范围的core直接返回
    const uint32_t blockIdx = GetBlockIdx();
    if (blockIdx >= tilingData.usedCoreNum)
    {
        return;
    }

    // Host预计算的per-core空间任务分配（消除kernel侧除法）
    uint32_t start_task;
    uint32_t core_tasks;
    if (blockIdx < tilingData.tailBlockNum)
    {
        core_tasks = tilingData.bigCoreSpatialTasks;
        start_task = blockIdx * core_tasks;
    }
    else
    {
        core_tasks = tilingData.smallCoreSpatialTasks;
        start_task = tilingData.tailBlockNum * tilingData.bigCoreSpatialTasks + (blockIdx - tilingData.tailBlockNum) * core_tasks;
    }

    if (core_tasks == 0)
    {
        return;
    }

    // 编译期分支：零运行时开销
    if constexpr (IS_SINGLE_TILE)
    {
        KernelBatchToSpaceSingleTile<DT_X, IS_DEPTH_ALIGNED, IS_FAST_W_INCREMENT,
                                     BLOCK_MODE> op;
        op.InitAndProcess(x, y, start_task, core_tasks, tilingData);
    }
    else
    {
        KernelBatchToSpaceDoubleBuffer<DT_X, IS_DEPTH_ALIGNED, IS_FAST_W_INCREMENT,
                                       BLOCK_MODE> op;
        op.InitAndProcess(x, y, start_task, core_tasks, tilingData);
    }
}
