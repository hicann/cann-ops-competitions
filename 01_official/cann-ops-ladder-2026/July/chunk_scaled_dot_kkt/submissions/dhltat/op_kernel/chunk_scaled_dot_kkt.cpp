/*!
 * \file chunk_scaled_dot_kkt.cpp
 * \brief ChunkScaledDotKkt kernel entry
 */

#include "chunk_scaled_dot_kkt.h"

enum class ChunkScaledDotKktTilingKey : uint32_t
{
    TILING_KEY_CHUNKSCALEDDOTKKT_MODE_0 = 0,
    TILING_KEY_CHUNKSCALEDDOTKKT_MODE_1 = 1,
    TILING_KEY_CHUNKSCALEDDOTKKT_MODE_2 = 2,
    TILING_KEY_CHUNKSCALEDDOTKKT_MODE_3 = 3,
    TILING_KEY_CHUNKSCALEDDOTKKT_MODE_4 = 4,
    TILING_KEY_CHUNKSCALEDDOTKKT_MODE_5 = 5,
    TILING_KEY_CHUNKSCALEDDOTKKT_MODE_6 = 6,
    TILING_KEY_CHUNKSCALEDDOTKKT_MODE_7 = 7,
    TILING_KEY_CHUNKSCALEDDOTKKT_MODE_8 = 8,
    TILING_KEY_CHUNKSCALEDDOTKKT_MODE_9 = 9,
    TILING_KEY_CHUNKSCALEDDOTKKT_MODE_10 = 10,
    TILING_KEY_CHUNKSCALEDDOTKKT_MODE_11 = 11,
    TILING_KEY_CHUNKSCALEDDOTKKT_MODE_12 = 12,
    TILING_KEY_CHUNKSCALEDDOTKKT_MODE_13 = 13,
    TILING_KEY_CHUNKSCALEDDOTKKT_MODE_14 = 14,
    TILING_KEY_CHUNKSCALEDDOTKKT_MODE_15 = 15,
    TILING_KEY_CHUNKSCALEDDOTKKT_MODE_16 = 16,
    TILING_KEY_CHUNKSCALEDDOTKKT_MODE_17 = 17,
    TILING_KEY_CHUNKSCALEDDOTKKT_MODE_18 = 18,
    TILING_KEY_CHUNKSCALEDDOTKKT_MODE_19 = 19,
    TILING_KEY_CHUNKSCALEDDOTKKT_MODE_20 = 20,
    TILING_KEY_CHUNKSCALEDDOTKKT_MODE_22 = 22,
    TILING_KEY_CHUNKSCALEDDOTKKT_MODE_24 = 24,
    TILING_KEY_CHUNKSCALEDDOTKKT_MODE_28 = 28,
};

template <uint32_t schMode>
__global__ __aicore__ void chunk_scaled_dot_kkt(
    GM_ADDR k,
    GM_ADDR beta,
    GM_ADDR g_cumsum,
    GM_ADDR chunk_offsets,
    GM_ADDR A,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(ChunkScaledDotKktTilingData);
    AscendC::TPipe tPipe;
    if constexpr (
        (schMode >= static_cast<uint32_t>(
            ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_6) &&
            schMode <= static_cast<uint32_t>(
                ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_16)) ||
        schMode == static_cast<uint32_t>(
            ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_18) ||
        schMode == static_cast<uint32_t>(
            ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_19) ||
        schMode == static_cast<uint32_t>(
            ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_20) ||
        schMode == static_cast<uint32_t>(
            ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_22) ||
        schMode == static_cast<uint32_t>(
            ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_24)) {
        GET_TILING_DATA_WITH_STRUCT(ChunkScaledDotKktManualTilingData, tilingData, tiling);
        if constexpr (
            schMode == static_cast<uint32_t>(ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_6)) {
            NsChunkScaledDotKkt::ChunkScaledDotKkt<
                bfloat16_t, false, false, false, false, false, true, false, true,
                true, false, true, true, true, true, true, 0> op;
            op.Init(k, beta, g_cumsum, chunk_offsets, A, userWorkspace, &tPipe, &tilingData);
            op.Process();
        } else if constexpr (
            schMode == static_cast<uint32_t>(ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_7)) {
            NsChunkScaledDotKkt::ChunkScaledDotKkt<
                bfloat16_t, false, false, false, false, false, true, true, true,
                false, false, true, false, false, true, true> op;
            op.Init(k, beta, g_cumsum, chunk_offsets, A, userWorkspace, &tPipe, &tilingData);
            op.Process();
        } else if constexpr (
            schMode == static_cast<uint32_t>(ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_8)) {
            NsChunkScaledDotKkt::ChunkScaledDotKkt<
                bfloat16_t, false, false, false, false, false, true, false, true,
                true, false, true, false, false, false, false, 1> op;
            op.Init(k, beta, g_cumsum, chunk_offsets, A, userWorkspace, &tPipe, &tilingData);
            op.Process();
        } else if constexpr (
            schMode == static_cast<uint32_t>(ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_9)) {
            NsChunkScaledDotKkt::ChunkScaledDotKkt<
                bfloat16_t, false, false, false, false, false, true, true, true,
                false, true, true, false, false, true, true, 1> op;
            op.Init(k, beta, g_cumsum, chunk_offsets, A, userWorkspace, &tPipe, &tilingData);
            op.Process();
        } else if constexpr (
            schMode == static_cast<uint32_t>(ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_10)) {
            NsChunkScaledDotKkt::ChunkScaledDotKkt<
                bfloat16_t, false, false, false, false, false, true, true, true,
                false, true, true, false, false, true, true, 0,
                NsChunkScaledDotKkt::FACTORED_TAIL_CROSSOVER_ROWS> op;
            op.Init(k, beta, g_cumsum, chunk_offsets, A, userWorkspace, &tPipe, &tilingData);
            op.Process();
        } else if constexpr (
            schMode == static_cast<uint32_t>(ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_11)) {
            NsChunkScaledDotKkt::ChunkScaledDotKkt<
                bfloat16_t, false, false, false, false, false, true, true, true,
                false, false, false, true, false, true, true, 0> op;
            op.Init(k, beta, g_cumsum, chunk_offsets, A, userWorkspace, &tPipe, &tilingData);
            op.Process();
        } else if constexpr (
            schMode == static_cast<uint32_t>(ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_12)) {
            NsChunkScaledDotKkt::ChunkScaledDotKkt<
                bfloat16_t, false, false, false, false, false, true, false, true,
                true, true, true, true, false, true, true> op;
            op.Init(k, beta, g_cumsum, chunk_offsets, A, userWorkspace, &tPipe, &tilingData);
            op.Process();
        } else if constexpr (
            schMode == static_cast<uint32_t>(ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_13)) {
            NsChunkScaledDotKkt::ChunkScaledDotKkt<
                bfloat16_t, false, false, false, false, false, true, false, true,
                true, false, true, true, true, false, false, 0> op;
            op.Init(k, beta, g_cumsum, chunk_offsets, A, userWorkspace, &tPipe, &tilingData);
            op.Process();
        } else if constexpr (
            schMode == static_cast<uint32_t>(ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_14)) {
            NsChunkScaledDotKkt::ChunkScaledDotKkt<
                bfloat16_t, false, false, false, false, false, true, false, true,
                true, false, true, true, true, true, true, 1,
                NsChunkScaledDotKkt::FACTORED_TAIL_CROSSOVER_ROWS> op;
            op.Init(k, beta, g_cumsum, chunk_offsets, A, userWorkspace, &tPipe, &tilingData);
            op.Process();
        } else if constexpr (
            schMode == static_cast<uint32_t>(ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_15)) {
            NsChunkScaledDotKkt::ChunkScaledDotKkt<
                bfloat16_t, false, false, false, false, false, true, false, true,
                true, false, true, false, false, true, true> op;
            op.Init(k, beta, g_cumsum, chunk_offsets, A, userWorkspace, &tPipe, &tilingData);
            op.Process();
        } else if constexpr (
            schMode == static_cast<uint32_t>(ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_16)) {
            NsChunkScaledDotKkt::ChunkScaledDotKkt<
                bfloat16_t, false, false, false, false, false, true, false, true,
                true, false, true, false, false, false, false, 1> op;
            op.Init(k, beta, g_cumsum, chunk_offsets, A, userWorkspace, &tPipe, &tilingData);
            op.Process();
        } else if constexpr (
            schMode == static_cast<uint32_t>(ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_18)) {
            NsChunkScaledDotKkt::ChunkScaledDotKkt<
                bfloat16_t, false, false, false, false, false, true, true, true,
                false, false, true, false, false, true, true, -1, 0, false, true> op;
            op.Init(k, beta, g_cumsum, chunk_offsets, A, userWorkspace, &tPipe, &tilingData);
            op.Process();
        } else if constexpr (
            schMode == static_cast<uint32_t>(ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_19)) {
            NsChunkScaledDotKkt::ChunkScaledDotKkt<
                bfloat16_t, false, false, false, false, false, true, false, true,
                true, false, true, false, false, true, true, -1, 0, false, false,
                NsChunkScaledDotKkt::GROUP_ALIGNED_CORE_COUNT, true> op;
            op.Init(k, beta, g_cumsum, chunk_offsets, A, userWorkspace, &tPipe, &tilingData);
            op.Process();
        } else if constexpr (
            schMode == static_cast<uint32_t>(ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_20)) {
            NsChunkScaledDotKkt::ChunkScaledDotKkt<
                bfloat16_t, false, false, false, false, false, true, true, true,
                true, false, true, true, true, true, true, 0, 0, false, false,
                NsChunkScaledDotKkt::GROUP_ALIGNED_CORE_COUNT, true> op;
            op.Init(k, beta, g_cumsum, chunk_offsets, A, userWorkspace, &tPipe, &tilingData);
            op.Process();
        } else if constexpr (
            schMode == static_cast<uint32_t>(ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_22)) {
            NsChunkScaledDotKkt::ChunkScaledDotKkt<
                bfloat16_t, false, false, false, false, false, true, true, true,
                true, false, true, false, false, false, false, 1, 0, false, false,
                NsChunkScaledDotKkt::GROUP_ALIGNED_CORE_COUNT> op;
            op.Init(k, beta, g_cumsum, chunk_offsets, A, userWorkspace, &tPipe, &tilingData);
            op.Process();
        } else if constexpr (
            schMode == static_cast<uint32_t>(ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_24)) {
            NsChunkScaledDotKkt::ChunkScaledDotKkt<
                bfloat16_t, false, false, false, false, false, true, true, true,
                false, true, true, false, false, true, true, 0,
                NsChunkScaledDotKkt::FACTORED_TAIL_CROSSOVER_ROWS, false, false,
                NsChunkScaledDotKkt::GROUP_ALIGNED_CORE_COUNT> op;
            op.Init(k, beta, g_cumsum, chunk_offsets, A, userWorkspace, &tPipe, &tilingData);
            op.Process();
        }
    } else {
        GET_TILING_DATA_WITH_STRUCT(ChunkScaledDotKktTilingData, tilingData, tiling);
        if constexpr (
            schMode == static_cast<uint32_t>(ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_0)) {
            NsChunkScaledDotKkt::ChunkScaledDotKkt<bfloat16_t> op;
            op.Init(k, beta, g_cumsum, chunk_offsets, A, userWorkspace, &tPipe, &tilingData);
            op.Process();
        } else if constexpr (
            schMode == static_cast<uint32_t>(ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_1)) {
            NsChunkScaledDotKkt::ChunkScaledDotKkt<bfloat16_t, true> op;
            op.Init(k, beta, g_cumsum, chunk_offsets, A, userWorkspace, &tPipe, &tilingData);
            op.Process();
        } else if constexpr (
            schMode == static_cast<uint32_t>(ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_2)) {
            NsChunkScaledDotKkt::ChunkScaledDotKkt<bfloat16_t, false, true> op;
            op.Init(k, beta, g_cumsum, chunk_offsets, A, userWorkspace, &tPipe, &tilingData);
            op.Process();
        } else if constexpr (
            schMode == static_cast<uint32_t>(ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_3)) {
            NsChunkScaledDotKkt::ChunkScaledDotKkt<bfloat16_t, false, false, true> op;
            op.Init(k, beta, g_cumsum, chunk_offsets, A, userWorkspace, &tPipe, &tilingData);
            op.Process();
        } else if constexpr (
            schMode == static_cast<uint32_t>(ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_4)) {
            NsChunkScaledDotKkt::ChunkScaledDotKkt<bfloat16_t, false, true, false, false> op;
            op.Init(k, beta, g_cumsum, chunk_offsets, A, userWorkspace, &tPipe, &tilingData);
            op.Process();
        } else if constexpr (
            schMode == static_cast<uint32_t>(ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_5)) {
            NsChunkScaledDotKkt::ChunkScaledDotKkt<
                bfloat16_t, false, false, false, false, true, false, false,
                true, false, false, true, false, true> op;
            op.Init(k, beta, g_cumsum, chunk_offsets, A, userWorkspace, &tPipe, &tilingData);
            op.Process();
        } else if constexpr (
            schMode == static_cast<uint32_t>(ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_17)) {
            NsChunkScaledDotKkt::ChunkScaledDotKkt<
                bfloat16_t, false, false, false, false, true, true, true,
                true, false, false, true, false, true> op;
            op.Init(k, beta, g_cumsum, chunk_offsets, A, userWorkspace, &tPipe, &tilingData);
            op.Process();
        } else if constexpr (
            schMode == static_cast<uint32_t>(ChunkScaledDotKktTilingKey::TILING_KEY_CHUNKSCALEDDOTKKT_MODE_28)) {
            NsChunkScaledDotKkt::ChunkScaledDotKkt<
                bfloat16_t, false, true, false, false, false, false, false, false,
                false, false, false, false, false, false, false, -1, 0, false,
                false, 0, false, true> op;
            op.Init(k, beta, g_cumsum, chunk_offsets, A, userWorkspace, &tPipe, &tilingData);
            op.Process();
        }
    }
    tPipe.Destroy();
}
