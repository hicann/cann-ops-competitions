// Kernel侧核函数实现
#include "kernel_operator.h"

#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"

using namespace AscendC;

namespace {
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2;
constexpr uint32_t TILE_ELEMS = 8192;
constexpr uint32_t QUEUE_ELEMS = TILE_ELEMS;

constexpr uint32_t BLOCK2_DEPTH32_BURST_DEPTH = 32;
constexpr uint64_t BLOCK2_DEPTH32_OUTPUT_PIXELS = 98304;
constexpr uint64_t BLOCK2_FP32_DEPTH64_ROW_PIXELS = 3136;
constexpr uint32_t BLOCK2_CASE3_DEPTH = 64;
constexpr uint32_t BLOCK2_CASE3_OUTPUT_HEIGHT = 28;
constexpr uint32_t BLOCK2_CASE3_OUTPUT_WIDTH = 28;
constexpr uint32_t BLOCK2_CASE3_OUTPUT_BATCH = 4;
constexpr uint32_t BLOCK2_CASE3_INPUT_HEIGHT = 14;
constexpr uint32_t BLOCK2_CASE3_INPUT_WIDTH = 14;
constexpr uint32_t BLOCK2_CASE3_ROW_ELEMS = BLOCK2_CASE3_OUTPUT_WIDTH * BLOCK2_CASE3_DEPTH;
constexpr uint32_t BLOCK2_CASE3_INPUT_ROW_ELEMS = BLOCK2_CASE3_INPUT_WIDTH * BLOCK2_CASE3_DEPTH;
constexpr uint32_t BLOCK2_CASE3_INPUT_BATCH_ELEMS =
    BLOCK2_CASE3_INPUT_HEIGHT * BLOCK2_CASE3_INPUT_ROW_ELEMS;
constexpr uint32_t BLOCK2_CASE3_BLOCK_BATCH_STRIDE_ELEMS =
    BLOCK2_CASE3_OUTPUT_BATCH * BLOCK2_CASE3_INPUT_BATCH_ELEMS;
constexpr uint32_t BLOCK2_CASE3_CORE_ROWS = 5;
constexpr uint32_t BLOCK2_CASE3_CORE_ELEMS = BLOCK2_CASE3_CORE_ROWS * BLOCK2_CASE3_ROW_ELEMS;
constexpr uint32_t BLOCK2_CASE3_AIC_CORE_ROWS = 6;
constexpr uint32_t BLOCK2_CASE3_AIC_CORE_ELEMS = BLOCK2_CASE3_AIC_CORE_ROWS * BLOCK2_CASE3_ROW_ELEMS;
constexpr uint32_t BLOCK2_CASE3_TILE_ROWS = 7;
constexpr uint32_t BLOCK2_CASE3_AIV_COMPACT_ELEMS =
    BLOCK2_CASE3_TILE_ROWS * BLOCK2_CASE3_INPUT_ROW_ELEMS;
constexpr uint32_t BLOCK2_CASE3_AIV_INTERLEAVED_ELEMS =
    BLOCK2_CASE3_TILE_ROWS * BLOCK2_CASE3_ROW_ELEMS;
constexpr uint32_t BLOCK2_CASE3_AIV_HEIGHT_TILES =
    (BLOCK2_CASE3_INPUT_HEIGHT + BLOCK2_CASE3_TILE_ROWS - 1U) / BLOCK2_CASE3_TILE_ROWS;
constexpr uint32_t BLOCK2_CASE3_AIV_TASKS =
    BLOCK2_CASE3_OUTPUT_BATCH * 2U * BLOCK2_CASE3_AIV_HEIGHT_TILES;
constexpr uint32_t BLOCK2_CASE1_DEPTH = 128;
constexpr uint32_t BLOCK2_CASE1_OUTPUT_HEIGHT = 56;
constexpr uint32_t BLOCK2_CASE1_OUTPUT_WIDTH = 56;
constexpr uint32_t BLOCK2_CASE1_OUTPUT_BATCH = 2;
constexpr uint32_t BLOCK2_CASE1_INPUT_HEIGHT = 28;
constexpr uint32_t BLOCK2_CASE1_INPUT_WIDTH = 28;
constexpr uint32_t BLOCK2_CASE1_LANE_COLS = 28;
constexpr uint32_t BLOCK2_CASE1_LANE_ELEMS = BLOCK2_CASE1_LANE_COLS * BLOCK2_CASE1_DEPTH;
constexpr uint32_t BLOCK2_CASE1_ROW_ELEMS = BLOCK2_CASE1_OUTPUT_WIDTH * BLOCK2_CASE1_DEPTH;
constexpr uint32_t BLOCK2_CASE1_INPUT_ROW_ELEMS = BLOCK2_CASE1_INPUT_WIDTH * BLOCK2_CASE1_DEPTH;
constexpr uint32_t BLOCK2_CASE1_INPUT_BATCH_ELEMS =
    BLOCK2_CASE1_INPUT_HEIGHT * BLOCK2_CASE1_INPUT_ROW_ELEMS;
constexpr uint32_t BLOCK2_CASE1_CORE_ROWS = 4;
constexpr uint32_t BLOCK2_CASE1_CORE_ELEMS = BLOCK2_CASE1_CORE_ROWS * (BLOCK2_CASE1_LANE_ELEMS << 1);
constexpr uint32_t BLOCK2_CASE2_DEPTH = 5;
constexpr uint32_t BLOCK2_CASE2_ALIGNED_DEPTH = 8;
constexpr uint32_t BLOCK2_CASE2_OUTPUT_HEIGHT = 17;
constexpr uint32_t BLOCK2_CASE2_OUTPUT_WIDTH = 26;
constexpr uint32_t BLOCK2_CASE2_LANE_COLS = 13;
constexpr uint32_t BLOCK2_CASE2_LANE_OFFSET = BLOCK2_CASE2_LANE_COLS * BLOCK2_CASE2_ALIGNED_DEPTH;
constexpr uint32_t BLOCK2_CASE2_INPUT_ROW_ELEMS = 15 * BLOCK2_CASE2_DEPTH;
constexpr uint32_t BLOCK2_CASE2_INPUT_BATCH_ELEMS = 10 * BLOCK2_CASE2_INPUT_ROW_ELEMS;
constexpr uint32_t BLOCK2_CASE2_ROW_ELEMS = BLOCK2_CASE2_OUTPUT_WIDTH * BLOCK2_CASE2_DEPTH;
constexpr uint32_t BLOCK2_CASE2_BUFFER_ELEMS = BLOCK2_CASE2_LANE_OFFSET << 1;
constexpr uint32_t BLOCK2_CASE2_COMPACT_LANE_ELEMS = BLOCK2_CASE2_LANE_COLS * BLOCK2_CASE2_DEPTH;
constexpr uint32_t BLOCK2_CASE2_COMPACT_LANE_ALIGNED_ELEMS = 72;
constexpr uint32_t BLOCK2_CASE2_GATHER_OUTPUT_OFFSET = BLOCK2_CASE2_COMPACT_LANE_ALIGNED_ELEMS << 1;
constexpr uint32_t BLOCK2_CASE2_GATHER_BUFFER_ELEMS =
    BLOCK2_CASE2_GATHER_OUTPUT_OFFSET + BLOCK2_CASE2_ROW_ELEMS;
constexpr uint32_t BLOCK2_CASE4_DEPTH = 32;
constexpr uint32_t BLOCK2_CASE4_OUTPUT_HEIGHT = 8;
constexpr uint32_t BLOCK2_CASE4_OUTPUT_WIDTH = 12;
constexpr uint32_t BLOCK2_CASE4_OUTPUT_BATCH = 5;
constexpr uint32_t BLOCK2_CASE4_INPUT_HEIGHT = 4;
constexpr uint32_t BLOCK2_CASE4_INPUT_WIDTH = 6;
constexpr uint32_t BLOCK2_CASE4_ROW_ELEMS = BLOCK2_CASE4_OUTPUT_WIDTH * BLOCK2_CASE4_DEPTH;
constexpr uint32_t BLOCK2_CASE4_OUTPUT_ROWS = BLOCK2_CASE4_OUTPUT_BATCH * BLOCK2_CASE4_OUTPUT_HEIGHT;
constexpr uint32_t BLOCK2_CASE4_INPUT_ROW_ELEMS = BLOCK2_CASE4_INPUT_WIDTH * BLOCK2_CASE4_DEPTH;
constexpr uint32_t BLOCK2_CASE4_INPUT_BATCH_ELEMS =
    BLOCK2_CASE4_INPUT_HEIGHT * BLOCK2_CASE4_INPUT_ROW_ELEMS;
constexpr uint32_t BLOCK2_CASE4_CORE_ROWS = 8;
constexpr uint32_t BLOCK2_CASE4_CORE_ELEMS = BLOCK2_CASE4_CORE_ROWS * BLOCK2_CASE4_ROW_ELEMS;
constexpr uint32_t BLOCK2_CASE4_AIV_CORE_ROWS = 4;
constexpr uint32_t BLOCK2_CASE4_AIV_CORE_ELEMS =
    BLOCK2_CASE4_AIV_CORE_ROWS * BLOCK2_CASE4_ROW_ELEMS;
constexpr uint32_t BLOCK2_CASE4_TWO_ROW_ELEMS = 2 * BLOCK2_CASE4_ROW_ELEMS;
constexpr uint32_t BLOCK2_CASE4_TASK_INPUT_ELEMS = BLOCK2_CASE4_INPUT_BATCH_ELEMS << 1;
constexpr uint32_t BLOCK2_CASE4_TASK_OUTPUT_ELEMS =
    BLOCK2_CASE4_INPUT_HEIGHT * BLOCK2_CASE4_ROW_ELEMS;
constexpr uint32_t BLOCK2_CASE4_TASK_BUFFER_ELEMS =
    BLOCK2_CASE4_TASK_INPUT_ELEMS + BLOCK2_CASE4_TASK_OUTPUT_ELEMS;
constexpr uint32_t BLOCK2_CASE6_DEPTH = 4096;
constexpr uint32_t BLOCK2_CASE6_PAIR_ELEMS = BLOCK2_CASE6_DEPTH << 1;
constexpr uint32_t BLOCK2_CASE6_CORE_ELEMS = BLOCK2_CASE6_PAIR_ELEMS << 1;
constexpr uint32_t BLOCK2_CASE7_DEPTH = 16384;
constexpr uint32_t BLOCK2_CASE7_REPAIR_ELEMS = 256;
constexpr uint32_t BLOCK2_CASE8_DEPTH = 256;
constexpr uint32_t BLOCK2_CASE8_OUTPUT_HEIGHT = 20;
constexpr uint32_t BLOCK2_CASE8_OUTPUT_WIDTH = 1024;
constexpr uint32_t BLOCK2_CASE8_INPUT_HEIGHT = 10;
constexpr uint32_t BLOCK2_CASE8_INPUT_WIDTH = 512;
constexpr uint32_t BLOCK2_CASE8_COLS_PER_TILE = 64;
constexpr uint32_t BLOCK2_CASE8_LANE_TILES = 8;
constexpr uint32_t BLOCK2_CASE8_WORK_PER_ROW = BLOCK2_CASE8_LANE_TILES << 1;
constexpr uint32_t BLOCK2_CASE8_TILE_ELEMS = BLOCK2_CASE8_COLS_PER_TILE * BLOCK2_CASE8_DEPTH;
constexpr uint32_t BLOCK2_CASE10_ROW_TILE_ROWS = 20;
constexpr uint32_t BLOCK2_CASE10_ROW_ELEMS = 12 * BLOCK2_DEPTH32_BURST_DEPTH;
constexpr uint32_t BLOCK2_CASE10_ROW_TILE_ELEMS = BLOCK2_CASE10_ROW_TILE_ROWS * BLOCK2_CASE10_ROW_ELEMS;
constexpr uint32_t BLOCK2_CASE10_AIC_GROUP_ROWS = 96;
constexpr uint32_t BLOCK2_CASE10_AIC_GROUP_ELEMS =
    BLOCK2_CASE10_AIC_GROUP_ROWS * BLOCK2_CASE10_ROW_ELEMS;
constexpr uint32_t BLOCK2_CASE10_INPUT_ROW_ELEMS = 6 * BLOCK2_DEPTH32_BURST_DEPTH;
constexpr uint32_t BLOCK2_CASE10_BULK_INPUT_ROWS = 32;
constexpr uint32_t BLOCK2_CASE10_BULK_OUTPUT_ROWS = BLOCK2_CASE10_BULK_INPUT_ROWS << 1;
constexpr uint32_t BLOCK2_CASE10_BULK_LANE_ELEMS =
    BLOCK2_CASE10_BULK_INPUT_ROWS * BLOCK2_CASE10_INPUT_ROW_ELEMS;
constexpr uint32_t BLOCK2_CASE10_BULK_INPUT_ELEMS = 4 * BLOCK2_CASE10_BULK_LANE_ELEMS;
constexpr uint32_t BLOCK2_CASE10_BULK_OUTPUT_ELEMS =
    BLOCK2_CASE10_BULK_OUTPUT_ROWS * BLOCK2_CASE10_ROW_ELEMS;
constexpr uint32_t BLOCK2_CASE10_BULK_BUFFER_ELEMS =
    BLOCK2_CASE10_BULK_INPUT_ELEMS + BLOCK2_CASE10_BULK_OUTPUT_ELEMS;

constexpr uint32_t BLOCK2_PAIR_BURST_DEPTH = 65;
constexpr uint64_t BLOCK2_CASE5_OUTPUT_PIXELS = 64516;
constexpr uint32_t BLOCK2_CASE5_OUTPUT_WIDTH = 254;
constexpr uint32_t BLOCK2_CASE5_WIDTH254_LANE_COLS = 127;
constexpr uint32_t BLOCK2_CASE5_INPUT_SIZE = 128;
constexpr uint32_t BLOCK2_CASE5_HALF_A_COLS = (BLOCK2_CASE5_WIDTH254_LANE_COLS + 1U) >> 1U;
constexpr uint32_t BLOCK2_CASE5_HALF_B_COLS = BLOCK2_CASE5_WIDTH254_LANE_COLS >> 1U;
constexpr uint32_t BLOCK2_CASE5_HALF_A_ELEMS = BLOCK2_CASE5_HALF_A_COLS * BLOCK2_PAIR_BURST_DEPTH;
constexpr uint32_t BLOCK2_CASE5_TIGHT_ELEMS = BLOCK2_CASE5_WIDTH254_LANE_COLS * BLOCK2_PAIR_BURST_DEPTH;
constexpr uint32_t BLOCK2_CASE5_TIGHT_ALIGNED_ELEMS = 8256;
constexpr uint32_t BLOCK2_CASE5_COMPACT_OFFSET_ELEMS = BLOCK2_CASE5_TIGHT_ALIGNED_ELEMS;

constexpr uint32_t BLOCK4_QUAD_BURST_BUFFER_ELEMS = 24576;
constexpr uint32_t BLOCK4_ROW_SEGMENT_DEPTH = 64;
constexpr uint32_t BLOCK4_ROW_SEGMENT_PIXELS_PER_TILE =
    BLOCK4_QUAD_BURST_BUFFER_ELEMS / BLOCK4_ROW_SEGMENT_DEPTH;
constexpr uint32_t BLOCK4_ROW_SEGMENT_LANE_PIXELS = (BLOCK4_ROW_SEGMENT_PIXELS_PER_TILE + 3) >> 2;
constexpr uint32_t BLOCK4_ROW_SEGMENT_LANE_ELEMS =
    BLOCK4_ROW_SEGMENT_LANE_PIXELS * BLOCK4_ROW_SEGMENT_DEPTH;
constexpr uint64_t BLOCK4_ROW_SEGMENT_OUTPUT_PIXELS = 61400;
constexpr uint32_t BLOCK4_ROW_SEGMENT_CROP_LEFT = 513;
constexpr uint32_t BLOCK4_CASE9_OUTPUT_HEIGHT = 40;
constexpr uint32_t BLOCK4_CASE9_OUTPUT_WIDTH = 1535;
constexpr uint32_t BLOCK4_CASE9_INPUT_HEIGHT = 10;
constexpr uint32_t BLOCK4_CASE9_INPUT_WIDTH = 512;
constexpr uint32_t BLOCK4_CASE9_INPUT_BATCH_ELEMS =
    BLOCK4_CASE9_INPUT_HEIGHT * BLOCK4_CASE9_INPUT_WIDTH * BLOCK4_ROW_SEGMENT_DEPTH;
constexpr uint32_t BLOCK4_CASE9_ROW_TILES = 4;
constexpr uint32_t BLOCK4_CASE9_TOTAL_WORK = BLOCK4_CASE9_OUTPUT_HEIGHT * BLOCK4_CASE9_ROW_TILES;

__aicore__ inline uint32_t Min(uint32_t lhs, uint32_t rhs) {
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint64_t Min(uint64_t lhs, uint64_t rhs) {
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint64_t CeilDiv(uint64_t lhs, uint64_t rhs) {
    return (lhs + rhs - 1) / rhs;
}
}  // namespace

template <class DT_X>
class KernelBatchToSpaceCase6Aic {
public:
    __aicore__ inline KernelBatchToSpaceCase6Aic() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
    }

    __aicore__ inline void Process() {
        constexpr uint32_t DEPTH4096 = BLOCK2_CASE6_DEPTH;
        constexpr uint32_t PAIR_ELEMS = BLOCK2_CASE6_PAIR_ELEMS;
        constexpr uint32_t ROW_ELEMS = BLOCK2_CASE6_CORE_ELEMS;

        uint32_t blockIdx = GetBlockIdx();
        uint64_t inBase =
            (static_cast<uint64_t>(blockIdx & 1U) << 15) +
            (static_cast<uint64_t>(blockIdx >> 1U) << 13);
        uint64_t outBase = static_cast<uint64_t>(blockIdx) << 14;
        constexpr uint16_t DEPTH_BLOCKS = static_cast<uint16_t>((DEPTH4096 * sizeof(DT_X)) >> 5);
        constexpr uint16_t INPUT_PAIR_SRC_GAP_BLOCKS =
            static_cast<uint16_t>((((DEPTH4096 << 2) - DEPTH4096) * sizeof(DT_X)) >> 5);
        const DataCopyParams inParams{2, DEPTH_BLOCKS, INPUT_PAIR_SRC_GAP_BLOCKS, 0};

        LocalTensor<DT_X> rowLocal(TPosition::A1, 0, BLOCK2_CASE6_CORE_ELEMS);
        DataCopy(rowLocal, xGm[inBase], inParams);
        DataCopy(rowLocal[PAIR_ELEMS], xGm[inBase + DEPTH4096], inParams);
        SetFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
        WaitFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
        DataCopy(yGm[outBase], rowLocal, ROW_ELEMS);
    }

private:
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
};

template <class DT_X>
class KernelBatchToSpaceCase7Aic {
public:
    __aicore__ inline KernelBatchToSpaceCase7Aic() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
        pipe.InitBuffer(a1Buffer, BLOCK2_CASE7_DEPTH * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        constexpr uint32_t DEPTH16384 = BLOCK2_CASE7_DEPTH;
        uint32_t pixel = GetBlockIdx();
        uint64_t offset = static_cast<uint64_t>(pixel) << 14;

        auto data = a1Buffer.Get<DT_X>();
        DataCopy(data, xGm[offset], DEPTH16384);
        SetFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
        DataCopy(yGm[offset], data, DEPTH16384);
        WaitFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
        DataCopy(yGm[offset], data, BLOCK2_CASE7_REPAIR_ELEMS);
    }

private:
    TPipe pipe;
    TBuf<TPosition::A1> a1Buffer;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
};

template <class DT_X>
class KernelBatchToSpaceCase9Aic {
public:
    __aicore__ inline KernelBatchToSpaceCase9Aic() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
        pipe.InitBuffer(a1Buffer, BLOCK4_QUAD_BURST_BUFFER_ELEMS * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        auto data = a1Buffer.Get<DT_X>();
        for (uint32_t work = GetBlockIdx(); work < BLOCK4_CASE9_TOTAL_WORK; work += GetBlockNum()) {
            uint32_t row = work >> 2;
            uint32_t tile = work & 3;
            uint32_t startOutW = tile * BLOCK4_ROW_SEGMENT_PIXELS_PER_TILE;
            uint32_t currentPixels =
                tile == (BLOCK4_CASE9_ROW_TILES - 1) ?
                (BLOCK4_CASE9_OUTPUT_WIDTH - startOutW) : BLOCK4_ROW_SEGMENT_PIXELS_PER_TILE;

            LoadTile(data, row, startOutW, currentPixels);
            SetFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
            WaitFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
            DataCopy(yGm[(static_cast<uint64_t>(row) * BLOCK4_CASE9_OUTPUT_WIDTH + startOutW) *
                BLOCK4_ROW_SEGMENT_DEPTH], data, currentPixels * BLOCK4_ROW_SEGMENT_DEPTH);
            SetFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
            WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
        }
    }

private:
    __aicore__ inline void LoadTile(
        LocalTensor<DT_X> &data, uint32_t row, uint32_t startOutW, uint32_t currentPixels) {
        constexpr uint32_t DEPTH64 = BLOCK4_ROW_SEGMENT_DEPTH;
        constexpr uint16_t DEPTH_BLOCKS = static_cast<uint16_t>((DEPTH64 * sizeof(DT_X)) >> 5);
        constexpr uint16_t SRC_GAP_BLOCKS = static_cast<uint16_t>(
            ((BLOCK4_CASE9_INPUT_BATCH_ELEMS - DEPTH64) * sizeof(DT_X)) >> 5);
        const DataCopyParams quadCopyParams{4, DEPTH_BLOCKS, SRC_GAP_BLOCKS, 0};

        uint32_t inH = row >> 2;
        uint32_t blockH = row & 3;
        uint32_t paddedW = startOutW + BLOCK4_ROW_SEGMENT_CROP_LEFT;
        uint32_t inW = paddedW >> 2;
        uint32_t blockW = paddedW & 3;
        uint64_t baseBatchOffset =
            static_cast<uint64_t>(blockH << 2) * BLOCK4_CASE9_INPUT_BATCH_ELEMS;
        uint64_t rowBaseOffset = static_cast<uint64_t>(inH) * BLOCK4_CASE9_INPUT_WIDTH * DEPTH64;

        uint32_t idx = 0;
        uint32_t prefixPixels = Min(currentPixels, (4 - blockW) & 3);
        for (; idx < prefixPixels; ++idx) {
            uint64_t spatialOffset = rowBaseOffset + static_cast<uint64_t>(inW) * DEPTH64;
            uint64_t inBase =
                baseBatchOffset + static_cast<uint64_t>(blockW) * BLOCK4_CASE9_INPUT_BATCH_ELEMS + spatialOffset;
            DataCopy(data[idx * DEPTH64], xGm[inBase], DEPTH64);
            ++blockW;
            if (blockW == 4) {
                blockW = 0;
                ++inW;
            }
        }

        uint32_t remaining = currentPixels - idx;
        while (remaining >= 4) {
            uint64_t spatialOffset = rowBaseOffset + static_cast<uint64_t>(inW) * DEPTH64;
            DataCopy(data[idx * DEPTH64], xGm[baseBatchOffset + spatialOffset], quadCopyParams);
            idx += 4;
            remaining -= 4;
            ++inW;
        }

        for (; remaining > 0; --remaining, ++idx) {
            uint64_t spatialOffset = rowBaseOffset + static_cast<uint64_t>(inW) * DEPTH64;
            uint64_t inBase =
                baseBatchOffset + static_cast<uint64_t>(blockW) * BLOCK4_CASE9_INPUT_BATCH_ELEMS + spatialOffset;
            DataCopy(data[idx * DEPTH64], xGm[inBase], DEPTH64);
            ++blockW;
            if (blockW == 4) {
                blockW = 0;
                ++inW;
            }
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::A1> a1Buffer;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
};

template <class DT_X>
class KernelBatchToSpaceCase9Aiv {
public:
    __aicore__ inline KernelBatchToSpaceCase9Aiv() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
        pipe.InitBuffer(tightQueue, BUFFER_NUM, BLOCK4_QUAD_BURST_BUFFER_ELEMS * sizeof(DT_X));
        pipe.InitBuffer(packQueue, BUFFER_NUM, BLOCK4_QUAD_BURST_BUFFER_ELEMS * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        for (uint32_t work = GetBlockIdx(); work < BLOCK4_CASE9_TOTAL_WORK; work += GetBlockNum()) {
            uint32_t row = work >> 2;
            uint32_t tile = work & 3;
            uint32_t startOutW = tile * BLOCK4_ROW_SEGMENT_PIXELS_PER_TILE;
            uint32_t currentPixels =
                tile == (BLOCK4_CASE9_ROW_TILES - 1) ?
                (BLOCK4_CASE9_OUTPUT_WIDTH - startOutW) : BLOCK4_ROW_SEGMENT_PIXELS_PER_TILE;

            auto tight = tightQueue.AllocTensor<DT_X>();
            LoadTileTight(tight, row, startOutW, currentPixels);
            tightQueue.EnQue<DT_X>(tight);

            tight = tightQueue.DeQue<DT_X>();
            auto packed = packQueue.AllocTensor<DT_X>();
            RepackTile(packed, tight, startOutW, currentPixels);
            tightQueue.FreeTensor<DT_X>(tight);
            packQueue.EnQue<DT_X>(packed);

            packed = packQueue.DeQue<DT_X>();
            DataCopy(yGm[(static_cast<uint64_t>(row) * BLOCK4_CASE9_OUTPUT_WIDTH + startOutW) *
                BLOCK4_ROW_SEGMENT_DEPTH],
                packed,
                currentPixels * BLOCK4_ROW_SEGMENT_DEPTH);
            packQueue.FreeTensor<DT_X>(packed);
        }
    }

private:
    __aicore__ inline void CalcLaneCopy(
        uint32_t startOutW, uint32_t currentPixels, uint32_t lane,
        uint32_t &firstOffset, uint32_t &lanePixels) {
        uint32_t startBlockW = (startOutW + BLOCK4_ROW_SEGMENT_CROP_LEFT) & 3;
        firstOffset = (lane + 4 - startBlockW) & 3;
        if (firstOffset >= currentPixels) {
            lanePixels = 0;
            return;
        }
        lanePixels = ((currentPixels - 1 - firstOffset) >> 2) + 1;
    }

    __aicore__ inline void LoadTileTight(
        LocalTensor<DT_X> &tight, uint32_t row, uint32_t startOutW, uint32_t currentPixels) {
        constexpr uint32_t DEPTH64 = BLOCK4_ROW_SEGMENT_DEPTH;
        constexpr uint32_t LANE_ELEMS = BLOCK4_ROW_SEGMENT_LANE_ELEMS;
        uint32_t inH = row >> 2;
        uint32_t blockH = row & 3;
        uint64_t baseBatchOffset =
            static_cast<uint64_t>(blockH << 2) * BLOCK4_CASE9_INPUT_BATCH_ELEMS;
        uint64_t rowBaseOffset = static_cast<uint64_t>(inH) * BLOCK4_CASE9_INPUT_WIDTH * DEPTH64;

        for (uint32_t lane = 0; lane < 4; ++lane) {
            uint32_t firstOffset;
            uint32_t lanePixels;
            CalcLaneCopy(startOutW, currentPixels, lane, firstOffset, lanePixels);
            if (lanePixels == 0) {
                continue;
            }
            uint32_t inW = (startOutW + firstOffset + BLOCK4_ROW_SEGMENT_CROP_LEFT) >> 2;
            uint64_t inBase =
                baseBatchOffset + static_cast<uint64_t>(lane) * BLOCK4_CASE9_INPUT_BATCH_ELEMS +
                rowBaseOffset + static_cast<uint64_t>(inW) * DEPTH64;
            DataCopy(tight[lane * LANE_ELEMS], xGm[inBase], lanePixels * DEPTH64);
        }
    }

    __aicore__ inline void RepackTile(
        LocalTensor<DT_X> &packed, LocalTensor<DT_X> &tight,
        uint32_t startOutW, uint32_t currentPixels) {
        constexpr uint32_t DEPTH64 = BLOCK4_ROW_SEGMENT_DEPTH;
        constexpr uint32_t LANE_ELEMS = BLOCK4_ROW_SEGMENT_LANE_ELEMS;
        constexpr uint16_t DEPTH_BLOCKS = static_cast<uint16_t>((DEPTH64 * sizeof(DT_X)) >> 5);
        constexpr uint16_t OUT_GAP_BLOCKS = static_cast<uint16_t>(((DEPTH64 * 3) * sizeof(DT_X)) >> 5);
        for (uint32_t lane = 0; lane < 4; ++lane) {
            uint32_t firstOffset;
            uint32_t lanePixels;
            CalcLaneCopy(startOutW, currentPixels, lane, firstOffset, lanePixels);
            if (lanePixels == 0) {
                continue;
            }
            DataCopyParams repackParams{static_cast<uint16_t>(lanePixels), DEPTH_BLOCKS, 0, OUT_GAP_BLOCKS};
            DataCopy(
                packed[firstOffset * DEPTH64],
                tight[lane * LANE_ELEMS],
                repackParams);
        }
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> tightQueue;
    TQue<TPosition::VECOUT, BUFFER_NUM> packQueue;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
};

template <class DT_X>
class KernelBatchToSpaceCase10Aic {
public:
    __aicore__ inline KernelBatchToSpaceCase10Aic() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
        pipe.InitBuffer(a1Queue, DOUBLE_BUFFER_NUM, BLOCK2_CASE10_AIC_GROUP_ELEMS * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        constexpr uint32_t DEPTH32 = BLOCK2_DEPTH32_BURST_DEPTH;
        constexpr uint32_t DEPTH_BYTES = DEPTH32 * sizeof(DT_X);
        constexpr uint32_t INPUT_W = 6;
        constexpr uint32_t OUT_H = 2048;
        constexpr uint32_t OUTPUT_BATCH4 = 4;
        constexpr uint32_t TOTAL_ROWS = OUTPUT_BATCH4 * OUT_H;
        constexpr uint32_t GROUP_ROWS = BLOCK2_CASE10_AIC_GROUP_ROWS;
        constexpr uint16_t DEPTH_BLOCKS = static_cast<uint16_t>(DEPTH_BYTES >> 5);

        DataCopyParams laneParams{static_cast<uint16_t>(INPUT_W), DEPTH_BLOCKS, 0, DEPTH_BLOCKS};
        TEventID eventIdMte2ToMte3 = static_cast<TEventID>(0);
        TEventID eventIdMte3ToMte2Buf0 = static_cast<TEventID>(0);
        TEventID eventIdMte3ToMte2Buf1 = static_cast<TEventID>(1);

        uint32_t blockIdx = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();
        uint32_t rowsPerCore = static_cast<uint32_t>(
            CeilDiv(static_cast<uint64_t>(TOTAL_ROWS), static_cast<uint64_t>(blockNum)));
        uint32_t startRow = blockIdx * rowsPerCore;
        if (startRow >= TOTAL_ROWS) {
            return;
        }
        uint32_t endRow = Min(startRow + rowsPerCore, TOTAL_ROWS);

        auto data0 = a1Queue.AllocTensor<DT_X>();
        auto data1 = a1Queue.AllocTensor<DT_X>();
        bool outputPending0 = false;
        bool outputPending1 = false;
        uint32_t groupIdx = 0;
        for (uint32_t row = startRow; row < endRow; row += GROUP_ROWS, ++groupIdx) {
            uint32_t currentRows = Min(GROUP_ROWS, endRow - row);
            bool useBuf1 = (groupIdx & 1U) != 0;
            auto data = useBuf1 ? data1 : data0;
            TEventID eventIdMte3ToMte2 = useBuf1 ? eventIdMte3ToMte2Buf1 : eventIdMte3ToMte2Buf0;

            if (useBuf1) {
                if (outputPending1) {
                    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2Buf1);
                    outputPending1 = false;
                }
            } else if (outputPending0) {
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2Buf0);
                outputPending0 = false;
            }

            LoadGroupRows(data, row, currentRows, laneParams);
            SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
            WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
            DataCopyParams outParams{
                1,
                static_cast<uint16_t>((currentRows * BLOCK2_CASE10_ROW_ELEMS * sizeof(DT_X)) >> 5),
                0,
                0};
            DataCopy(yGm[static_cast<uint64_t>(row) * BLOCK2_CASE10_ROW_ELEMS], data, outParams);
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            if (useBuf1) {
                outputPending1 = true;
            } else {
                outputPending0 = true;
            }
        }
        if (outputPending0) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2Buf0);
        }
        if (outputPending1) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2Buf1);
        }
        a1Queue.FreeTensor<DT_X>(data1);
        a1Queue.FreeTensor<DT_X>(data0);
    }

private:
    __aicore__ inline void LoadGroupRows(
        LocalTensor<DT_X> &data, uint32_t row, uint32_t currentRows,
        const DataCopyParams &laneParams) {
        constexpr uint32_t DEPTH32 = BLOCK2_DEPTH32_BURST_DEPTH;
        constexpr uint32_t INPUT_H = 1024;
        constexpr uint32_t INPUT_W = 6;
        constexpr uint32_t OUTPUT_BATCH4 = 4;
        constexpr uint32_t OUT_H = 2048;
        constexpr uint32_t INPUT_BATCH_ELEMS = INPUT_H * INPUT_W * DEPTH32;
        constexpr uint32_t BLOCK_BATCH_STRIDE_ELEMS = OUTPUT_BATCH4 * INPUT_BATCH_ELEMS;

        for (uint32_t groupRow = 0; groupRow < currentRows; ++groupRow) {
            uint32_t currentRow = row + groupRow;
            uint32_t outB = currentRow / OUT_H;
            uint32_t outH = currentRow - outB * OUT_H;
            uint32_t inH = outH >> 1;
            uint32_t blockH = outH & 1U;
            uint64_t baseBatchOffset =
                static_cast<uint64_t>((blockH << 1) * OUTPUT_BATCH4 + outB) * INPUT_BATCH_ELEMS;
            uint64_t rowBaseOffset = static_cast<uint64_t>(inH) * INPUT_W * DEPTH32;
            uint32_t localRowOffset = groupRow * BLOCK2_CASE10_ROW_ELEMS;

            DataCopy(data[localRowOffset], xGm[baseBatchOffset + rowBaseOffset], laneParams);
            DataCopy(data[localRowOffset + DEPTH32],
                xGm[baseBatchOffset + BLOCK_BATCH_STRIDE_ELEMS + rowBaseOffset],
                laneParams);
        }
    }

private:
    TPipe pipe;
    TQue<TPosition::A1, DOUBLE_BUFFER_NUM> a1Queue;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
};

template <class DT_X>
class KernelBatchToSpaceCase10Aiv {
public:
    __aicore__ inline KernelBatchToSpaceCase10Aiv() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
        pipe.InitBuffer(buffer, BLOCK2_CASE10_BULK_BUFFER_ELEMS * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        constexpr uint32_t DEPTH32 = BLOCK2_DEPTH32_BURST_DEPTH;
        constexpr uint32_t DEPTH_BYTES = DEPTH32 * sizeof(DT_X);
        constexpr uint32_t OUTPUT_BATCH4 = 4;
        constexpr uint32_t INPUT_H = 1024;
        constexpr uint32_t INPUT_W = 6;
        constexpr uint32_t ROW_ELEMS = BLOCK2_CASE10_ROW_ELEMS;
        constexpr uint32_t INPUT_ROW_ELEMS = BLOCK2_CASE10_INPUT_ROW_ELEMS;
        constexpr uint32_t INPUT_BATCH_ELEMS = INPUT_H * INPUT_ROW_ELEMS;
        constexpr uint32_t GROUP_INPUT_ROWS = BLOCK2_CASE10_BULK_INPUT_ROWS;
        constexpr uint32_t GROUPS_PER_BATCH = INPUT_H / GROUP_INPUT_ROWS;
        constexpr uint32_t TOTAL_WORK = OUTPUT_BATCH4 * GROUPS_PER_BATCH;
        constexpr uint16_t DEPTH_BLOCKS = static_cast<uint16_t>(DEPTH_BYTES >> 5);
        constexpr uint16_t SRC_ROW_GAP_BLOCKS =
            static_cast<uint16_t>(((INPUT_ROW_ELEMS - DEPTH32) * sizeof(DT_X)) >> 5);
        constexpr uint16_t DST_PAIR_ROW_GAP_BLOCKS =
            static_cast<uint16_t>((((ROW_ELEMS << 1) - DEPTH32) * sizeof(DT_X)) >> 5);
        DataCopyParams columnRepackParams{
            static_cast<uint16_t>(GROUP_INPUT_ROWS),
            DEPTH_BLOCKS,
            SRC_ROW_GAP_BLOCKS,
            DST_PAIR_ROW_GAP_BLOCKS};

        auto data = buffer.Get<DT_X>();
        LocalTensor<DT_X> input = data;
        LocalTensor<DT_X> output = data[BLOCK2_CASE10_BULK_INPUT_ELEMS];
        for (uint32_t work = GetBlockIdx(); work < TOTAL_WORK; work += GetBlockNum()) {
            uint32_t outB = work / GROUPS_PER_BATCH;
            uint32_t group = work - outB * GROUPS_PER_BATCH;
            uint32_t inputRowStart = group * GROUP_INPUT_ROWS;
            uint32_t outputRowStart = (outB * (INPUT_H << 1)) + (inputRowStart << 1);
            LoadBulkInput(input, outB, inputRowStart, INPUT_BATCH_ELEMS);
            PipeBarrier<PIPE_ALL>();
            RepackBulkOutput(output, input, columnRepackParams);
            PipeBarrier<PIPE_ALL>();
            DataCopy(yGm[static_cast<uint64_t>(outputRowStart) * ROW_ELEMS],
                output, BLOCK2_CASE10_BULK_OUTPUT_ELEMS);
            PipeBarrier<PIPE_MTE3>();
        }
    }

private:
    __aicore__ inline void LoadBulkInput(
        LocalTensor<DT_X> &input, uint32_t outB, uint32_t inputRowStart,
        uint32_t inputBatchElems) {
        constexpr uint32_t DEPTH32 = BLOCK2_DEPTH32_BURST_DEPTH;
        constexpr uint32_t OUTPUT_BATCH4 = 4;
        constexpr uint32_t INPUT_W = 6;
        constexpr uint32_t INPUT_ROW_ELEMS = INPUT_W * DEPTH32;
        uint64_t rowOffset = static_cast<uint64_t>(inputRowStart) * INPUT_ROW_ELEMS;
        uint64_t batchBase = static_cast<uint64_t>(outB) * inputBatchElems + rowOffset;

        DataCopy(input, xGm[batchBase], BLOCK2_CASE10_BULK_LANE_ELEMS);
        DataCopy(input[BLOCK2_CASE10_BULK_LANE_ELEMS],
            xGm[batchBase + static_cast<uint64_t>(OUTPUT_BATCH4) * inputBatchElems],
            BLOCK2_CASE10_BULK_LANE_ELEMS);
        DataCopy(input[BLOCK2_CASE10_BULK_LANE_ELEMS * 2],
            xGm[batchBase + static_cast<uint64_t>(OUTPUT_BATCH4 << 1) * inputBatchElems],
            BLOCK2_CASE10_BULK_LANE_ELEMS);
        DataCopy(input[BLOCK2_CASE10_BULK_LANE_ELEMS * 3],
            xGm[batchBase + static_cast<uint64_t>(OUTPUT_BATCH4 * 3) * inputBatchElems],
            BLOCK2_CASE10_BULK_LANE_ELEMS);
    }

    __aicore__ inline void RepackBulkOutput(
        LocalTensor<DT_X> &output, LocalTensor<DT_X> &input,
        const DataCopyParams &columnRepackParams) {
        constexpr uint32_t DEPTH32 = BLOCK2_DEPTH32_BURST_DEPTH;
        constexpr uint32_t INPUT_W = 6;
        constexpr uint32_t ROW_ELEMS = BLOCK2_CASE10_ROW_ELEMS;
        constexpr uint32_t LANE_ELEMS = BLOCK2_CASE10_BULK_LANE_ELEMS;
        for (uint32_t col = 0; col < INPUT_W; ++col) {
            uint32_t srcColOffset = col * DEPTH32;
            uint32_t dstColOffset = (col << 1) * DEPTH32;
            DataCopy(output[dstColOffset],
                input[srcColOffset],
                columnRepackParams);
            DataCopy(output[dstColOffset + DEPTH32],
                input[LANE_ELEMS + srcColOffset],
                columnRepackParams);
            DataCopy(output[ROW_ELEMS + dstColOffset],
                input[(LANE_ELEMS << 1) + srcColOffset],
                columnRepackParams);
            DataCopy(output[ROW_ELEMS + dstColOffset + DEPTH32],
                input[LANE_ELEMS * 3 + srcColOffset],
                columnRepackParams);
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
};

template <class DT_X>
class KernelBatchToSpaceCase3Aic {
public:
    __aicore__ inline KernelBatchToSpaceCase3Aic() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
        pipe.InitBuffer(a1Buffer, BLOCK2_CASE3_AIC_CORE_ELEMS * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        constexpr uint16_t DEPTH_BLOCKS =
            static_cast<uint16_t>((BLOCK2_CASE3_DEPTH * sizeof(DT_X)) >> 5);
        constexpr uint16_t PAIR_SRC_GAP_BLOCKS = static_cast<uint16_t>(
            ((BLOCK2_CASE3_BLOCK_BATCH_STRIDE_ELEMS - BLOCK2_CASE3_DEPTH) * sizeof(DT_X)) >> 5);
        const DataCopyParams pairInParams{2, DEPTH_BLOCKS, PAIR_SRC_GAP_BLOCKS, 0};

        uint32_t blockIdx = GetBlockIdx();
        if (blockIdx >= 20) {
            return;
        }

        bool isSixRows = blockIdx < 12;
        uint32_t rowStart = isSixRows ? blockIdx * 6 : 72 + (blockIdx - 12) * 5;
        uint32_t outB = rowStart / BLOCK2_CASE3_OUTPUT_HEIGHT;
        uint32_t outH = rowStart - outB * BLOCK2_CASE3_OUTPUT_HEIGHT;

        auto data = a1Buffer.Get<DT_X>();
        CopyRowKnown(data, 0, outB, outH, pairInParams);
        NextCase3Row(outB, outH);
        CopyRowKnown(data, 1, outB, outH, pairInParams);
        NextCase3Row(outB, outH);
        CopyRowKnown(data, 2, outB, outH, pairInParams);
        NextCase3Row(outB, outH);
        CopyRowKnown(data, 3, outB, outH, pairInParams);
        NextCase3Row(outB, outH);
        CopyRowKnown(data, 4, outB, outH, pairInParams);
        if (isSixRows) {
            NextCase3Row(outB, outH);
            CopyRowKnown(data, 5, outB, outH, pairInParams);
        }

        SetFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
        WaitFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
        uint32_t rowCount = isSixRows ? 6 : 5;
        DataCopy(yGm[static_cast<uint64_t>(rowStart) * BLOCK2_CASE3_ROW_ELEMS],
            data, rowCount * BLOCK2_CASE3_ROW_ELEMS);
    }

private:
    __aicore__ inline void NextCase3Row(uint32_t &outB, uint32_t &outH) {
        ++outH;
        if (outH == BLOCK2_CASE3_OUTPUT_HEIGHT) {
            outH = 0;
            ++outB;
        }
    }

    __aicore__ inline void CopyRowKnown(
        LocalTensor<DT_X> &data, uint32_t localRow, uint32_t outB, uint32_t outH,
        const DataCopyParams &pairInParams) {
        constexpr uint32_t DEPTH64 = BLOCK2_CASE3_DEPTH;
        constexpr uint32_t PAIR_OUT_STRIDE = DEPTH64 << 1;

        uint32_t baseBatch = ((outH & 1) << 1) * BLOCK2_CASE3_OUTPUT_BATCH + outB;
        uint64_t rowBase =
            static_cast<uint64_t>(baseBatch) * BLOCK2_CASE3_INPUT_BATCH_ELEMS +
            static_cast<uint64_t>(outH >> 1) * BLOCK2_CASE3_INPUT_ROW_ELEMS;
        uint64_t localBase = static_cast<uint64_t>(localRow) * BLOCK2_CASE3_ROW_ELEMS;

        DataCopy(data[localBase], xGm[rowBase], pairInParams);
        DataCopy(data[localBase + PAIR_OUT_STRIDE], xGm[rowBase + DEPTH64], pairInParams);
        DataCopy(data[localBase + PAIR_OUT_STRIDE * 2], xGm[rowBase + DEPTH64 * 2], pairInParams);
        DataCopy(data[localBase + PAIR_OUT_STRIDE * 3], xGm[rowBase + DEPTH64 * 3], pairInParams);
        DataCopy(data[localBase + PAIR_OUT_STRIDE * 4], xGm[rowBase + DEPTH64 * 4], pairInParams);
        DataCopy(data[localBase + PAIR_OUT_STRIDE * 5], xGm[rowBase + DEPTH64 * 5], pairInParams);
        DataCopy(data[localBase + PAIR_OUT_STRIDE * 6], xGm[rowBase + DEPTH64 * 6], pairInParams);
        DataCopy(data[localBase + PAIR_OUT_STRIDE * 7], xGm[rowBase + DEPTH64 * 7], pairInParams);
        DataCopy(data[localBase + PAIR_OUT_STRIDE * 8], xGm[rowBase + DEPTH64 * 8], pairInParams);
        DataCopy(data[localBase + PAIR_OUT_STRIDE * 9], xGm[rowBase + DEPTH64 * 9], pairInParams);
        DataCopy(data[localBase + PAIR_OUT_STRIDE * 10], xGm[rowBase + DEPTH64 * 10], pairInParams);
        DataCopy(data[localBase + PAIR_OUT_STRIDE * 11], xGm[rowBase + DEPTH64 * 11], pairInParams);
        DataCopy(data[localBase + PAIR_OUT_STRIDE * 12], xGm[rowBase + DEPTH64 * 12], pairInParams);
        DataCopy(data[localBase + PAIR_OUT_STRIDE * 13], xGm[rowBase + DEPTH64 * 13], pairInParams);
    }

private:
    TPipe pipe;
    TBuf<TPosition::A1> a1Buffer;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
};

template <class DT_X>
class KernelBatchToSpaceCase3Aiv {
public:
    __aicore__ inline KernelBatchToSpaceCase3Aiv() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
    }

    __aicore__ inline void Process() {
        constexpr uint16_t DEPTH_BLOCKS =
            static_cast<uint16_t>((BLOCK2_CASE3_DEPTH * sizeof(DT_X)) >> 5);
        constexpr uint32_t COMPACT_BYTES = BLOCK2_CASE3_AIV_COMPACT_ELEMS * sizeof(DT_X);
        LocalTensor<DT_X> leftRows(TPosition::VECCALC, 0, BLOCK2_CASE3_AIV_COMPACT_ELEMS);
        LocalTensor<DT_X> rightRows(TPosition::VECCALC, COMPACT_BYTES, BLOCK2_CASE3_AIV_COMPACT_ELEMS);
        LocalTensor<DT_X> mergedRows(
            TPosition::VECCALC, COMPACT_BYTES << 1U, BLOCK2_CASE3_AIV_INTERLEAVED_ELEMS);

        DataCopyParams inputParams{1, 0, 0, 0};
        DataCopyParams mergeParams{0, DEPTH_BLOCKS, 0, DEPTH_BLOCKS};
        DataCopyParams outputParams{
            0,
            static_cast<uint16_t>((BLOCK2_CASE3_ROW_ELEMS * sizeof(DT_X)) >> 5),
            0,
            static_cast<uint16_t>((BLOCK2_CASE3_ROW_ELEMS * sizeof(DT_X)) >> 5)};

        uint32_t task = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();
        if (blockNum == BLOCK2_CASE3_AIV_TASKS) {
            CopyCase3Tile(leftRows, rightRows, mergedRows, task, inputParams, mergeParams, outputParams);
            return;
        }

        for (; task < BLOCK2_CASE3_AIV_TASKS; task += blockNum) {
            CopyCase3Tile(leftRows, rightRows, mergedRows, task, inputParams, mergeParams, outputParams);
        }
    }

private:
    __aicore__ inline void CopyCase3Tile(
        LocalTensor<DT_X> &leftRows, LocalTensor<DT_X> &rightRows, LocalTensor<DT_X> &mergedRows,
        uint32_t task, DataCopyParams &inputParams, DataCopyParams &mergeParams,
        DataCopyParams &outputParams) {
        uint32_t outB;
        uint32_t blockH;
        uint32_t inH;
        DecodeTask(task, outB, blockH, inH);
        uint32_t rows = Min(BLOCK2_CASE3_TILE_ROWS, BLOCK2_CASE3_INPUT_HEIGHT - inH);

        inputParams.blockLen = static_cast<uint16_t>(
            (rows * BLOCK2_CASE3_INPUT_ROW_ELEMS * sizeof(DT_X)) >> 5);
        DataCopy(leftRows, xGm[CalcInputBase(outB, blockH, 0, inH)], inputParams);
        DataCopy(rightRows, xGm[CalcInputBase(outB, blockH, 1, inH)], inputParams);
        PipeBarrier<PIPE_ALL>();

        mergeParams.blockCount = static_cast<uint16_t>(rows * BLOCK2_CASE3_INPUT_WIDTH);
        DataCopy(mergedRows, leftRows, mergeParams);
        DataCopy(mergedRows[BLOCK2_CASE3_DEPTH], rightRows, mergeParams);
        PipeBarrier<PIPE_MTE2>();

        outputParams.blockCount = static_cast<uint16_t>(rows);
        DataCopy(yGm[CalcOutputBase(outB, blockH, inH)], mergedRows, outputParams);
        PipeBarrier<PIPE_MTE3>();
    }

    __aicore__ inline void DecodeTask(
        uint32_t task, uint32_t &outB, uint32_t &blockH, uint32_t &inH) {
        uint32_t rowTile = task & 1U;
        uint32_t batchAndPhase = task >> 1U;
        blockH = batchAndPhase & 1U;
        outB = batchAndPhase >> 1U;
        inH = rowTile * BLOCK2_CASE3_TILE_ROWS;
    }

    __aicore__ inline uint64_t CalcInputBase(
        uint32_t outB, uint32_t blockH, uint32_t blockW, uint32_t inH) {
        uint32_t inB = ((blockH << 1U) + blockW) * BLOCK2_CASE3_OUTPUT_BATCH + outB;
        return (static_cast<uint64_t>(inB) * BLOCK2_CASE3_INPUT_HEIGHT + inH) *
               BLOCK2_CASE3_INPUT_ROW_ELEMS;
    }

    __aicore__ inline uint64_t CalcOutputBase(uint32_t outB, uint32_t blockH, uint32_t inH) {
        return (static_cast<uint64_t>(outB) * BLOCK2_CASE3_OUTPUT_HEIGHT +
               blockH + (static_cast<uint64_t>(inH) << 1U)) *
               BLOCK2_CASE3_ROW_ELEMS;
    }

private:
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
};

template <class DT_X>
class KernelBatchToSpaceCase1Aiv {
public:
    __aicore__ inline KernelBatchToSpaceCase1Aiv() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
    }

    __aicore__ inline void Process() {
        constexpr uint32_t TWO_ROW_ELEMS = BLOCK2_CASE1_ROW_ELEMS << 1;
        constexpr uint16_t DEPTH_BLOCKS =
            static_cast<uint16_t>((BLOCK2_CASE1_DEPTH * sizeof(DT_X)) >> 5);
        const DataCopyParams loadToRowParams{BLOCK2_CASE1_LANE_COLS, DEPTH_BLOCKS, 0, DEPTH_BLOCKS};

        uint32_t inH = GetBlockIdx();
        LocalTensor<DT_X> output(TPosition::VECCALC, 0, TWO_ROW_ELEMS);

        LoadOutputRowPair(output, 0, inH, loadToRowParams);
        SetFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
        WaitFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
        DataCopy(yGm[static_cast<uint64_t>(inH << 1) * BLOCK2_CASE1_ROW_ELEMS], output, TWO_ROW_ELEMS);
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));

        LoadOutputRowPair(output, 1, inH, loadToRowParams);
        SetFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
        WaitFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
        DataCopy(yGm[(static_cast<uint64_t>(BLOCK2_CASE1_OUTPUT_HEIGHT) + (inH << 1)) *
            BLOCK2_CASE1_ROW_ELEMS], output, TWO_ROW_ELEMS);
    }

private:
    __aicore__ inline void LoadOutputRowPair(
        LocalTensor<DT_X> &output, uint32_t outB, uint32_t inH, const DataCopyParams &loadToRowParams) {
        constexpr uint64_t BATCH_STRIDE = BLOCK2_CASE1_INPUT_BATCH_ELEMS;
        uint64_t inBase =
            static_cast<uint64_t>(outB) * BATCH_STRIDE +
            static_cast<uint64_t>(inH) * BLOCK2_CASE1_INPUT_ROW_ELEMS;
        DataCopy(output, xGm[inBase], loadToRowParams);
        DataCopy(output[BLOCK2_CASE1_DEPTH],
            xGm[inBase + (BATCH_STRIDE << 1)],
            loadToRowParams);
        DataCopy(output[BLOCK2_CASE1_ROW_ELEMS],
            xGm[inBase + (BATCH_STRIDE << 2)],
            loadToRowParams);
        DataCopy(output[BLOCK2_CASE1_ROW_ELEMS + BLOCK2_CASE1_DEPTH],
            xGm[inBase + BATCH_STRIDE * 6],
            loadToRowParams);
    }

private:
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
};

template <class DT_X>
class KernelBatchToSpaceCase2Aiv {
public:
    __aicore__ inline KernelBatchToSpaceCase2Aiv() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
        pipe.InitBuffer(buffer, BLOCK2_CASE2_GATHER_BUFFER_ELEMS * sizeof(DT_X));
        pipe.InitBuffer(offsetBuffer, BLOCK2_CASE2_ROW_ELEMS * sizeof(uint32_t));
    }

    __aicore__ inline void Process() {
        constexpr uint32_t DEPTH_BYTES = BLOCK2_CASE2_DEPTH * sizeof(DT_X);
        DataCopyExtParams inParams{
            1, static_cast<uint32_t>(BLOCK2_CASE2_COMPACT_LANE_ELEMS * sizeof(DT_X)), 0, 0, 0};
        DataCopyExtParams outParams{
            1, static_cast<uint32_t>(BLOCK2_CASE2_ROW_ELEMS * sizeof(DT_X)), 0, 0, 0};
        DataCopyPadExtParams<DT_X> padParams{false, 0, 0, static_cast<DT_X>(0)};

        uint32_t outH = GetBlockIdx();
        if (outH >= BLOCK2_CASE2_OUTPUT_HEIGHT) {
            return;
        }
        uint32_t inH = (outH >> 1) + 1;
        uint32_t blockH = outH & 1;
        uint64_t inputRowBase = static_cast<uint64_t>(inH) * BLOCK2_CASE2_INPUT_ROW_ELEMS;
        uint64_t rowBase0 = static_cast<uint64_t>(blockH << 1) * BLOCK2_CASE2_INPUT_BATCH_ELEMS + inputRowBase;
        uint64_t rowBase1 = rowBase0 + BLOCK2_CASE2_INPUT_BATCH_ELEMS;
        uint64_t outBase = static_cast<uint64_t>(outH) * BLOCK2_CASE2_ROW_ELEMS;

        auto data = buffer.Get<DT_X>();
        DataCopyPad(data, xGm[rowBase1 + BLOCK2_CASE2_DEPTH], inParams, padParams);
        DataCopyPad(data[BLOCK2_CASE2_COMPACT_LANE_ALIGNED_ELEMS],
            xGm[rowBase0 + (BLOCK2_CASE2_DEPTH << 1)], inParams, padParams);
        auto offsets = offsetBuffer.Get<uint32_t>();
        PrepareGatherOffsets(offsets);
        TEventID eventIdMte2ToV = static_cast<TEventID>(0);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Gather<DT_X>(
            data[BLOCK2_CASE2_GATHER_OUTPUT_OFFSET],
            data,
            offsets,
            0,
            static_cast<uint32_t>(BLOCK2_CASE2_ROW_ELEMS));
        TEventID eventIdVToMte3 = static_cast<TEventID>(1);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopyPad(yGm[outBase], data[BLOCK2_CASE2_GATHER_OUTPUT_OFFSET], outParams);
    }

private:
    __aicore__ inline void PrepareGatherOffsets(LocalTensor<uint32_t> &offsets) {
        constexpr uint32_t DEPTH_BYTES = BLOCK2_CASE2_DEPTH * sizeof(DT_X);
        constexpr uint32_t SECOND_LANE_BYTES = BLOCK2_CASE2_COMPACT_LANE_ALIGNED_ELEMS * sizeof(DT_X);
        for (uint32_t col = 0; col < BLOCK2_CASE2_LANE_COLS; ++col) {
            uint32_t dstBase = (col << 1) * BLOCK2_CASE2_DEPTH;
            uint32_t firstBase = col * DEPTH_BYTES;
            uint32_t secondBase = SECOND_LANE_BYTES + firstBase;
            offsets.SetValue(dstBase, firstBase);
            offsets.SetValue(dstBase + 1, firstBase + sizeof(DT_X));
            offsets.SetValue(dstBase + 2, firstBase + sizeof(DT_X) * 2);
            offsets.SetValue(dstBase + 3, firstBase + sizeof(DT_X) * 3);
            offsets.SetValue(dstBase + 4, firstBase + sizeof(DT_X) * 4);
            offsets.SetValue(dstBase + BLOCK2_CASE2_DEPTH, secondBase);
            offsets.SetValue(dstBase + BLOCK2_CASE2_DEPTH + 1, secondBase + sizeof(DT_X));
            offsets.SetValue(dstBase + BLOCK2_CASE2_DEPTH + 2, secondBase + sizeof(DT_X) * 2);
            offsets.SetValue(dstBase + BLOCK2_CASE2_DEPTH + 3, secondBase + sizeof(DT_X) * 3);
            offsets.SetValue(dstBase + BLOCK2_CASE2_DEPTH + 4, secondBase + sizeof(DT_X) * 4);
        }
        TEventID eventIdStoV = static_cast<TEventID>(2);
        SetFlag<HardEvent::S_V>(eventIdStoV);
        WaitFlag<HardEvent::S_V>(eventIdStoV);
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    TBuf<TPosition::VECCALC> offsetBuffer;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
};

template <class DT_X>
class KernelBatchToSpaceCase4Aiv {
public:
    __aicore__ inline KernelBatchToSpaceCase4Aiv() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
        pipe.InitBuffer(buffer, BLOCK2_CASE4_TASK_BUFFER_ELEMS * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        constexpr uint16_t INPUT_BATCH_BLOCKS =
            static_cast<uint16_t>((BLOCK2_CASE4_INPUT_BATCH_ELEMS * sizeof(DT_X)) >> 5);
        constexpr uint16_t DEPTH_BLOCKS =
            static_cast<uint16_t>((BLOCK2_CASE4_DEPTH * sizeof(DT_X)) >> 5);
        constexpr uint16_t ROW_BLOCKS =
            static_cast<uint16_t>((BLOCK2_CASE4_ROW_ELEMS * sizeof(DT_X)) >> 5);
        constexpr uint32_t TASK_COUNT = BLOCK2_CASE4_OUTPUT_BATCH << 1;
        const DataCopyParams loadParams{1, INPUT_BATCH_BLOCKS, 0, 0};
        const DataCopyParams repackParams{
            static_cast<uint16_t>(BLOCK2_CASE4_INPUT_HEIGHT * BLOCK2_CASE4_INPUT_WIDTH),
            DEPTH_BLOCKS,
            0,
            DEPTH_BLOCKS};
        const DataCopyParams outParams{BLOCK2_CASE4_INPUT_HEIGHT, ROW_BLOCKS, 0, ROW_BLOCKS};

        uint32_t blockIdx = GetBlockIdx();
        if (blockIdx >= TASK_COUNT) {
            return;
        }
        uint32_t outB = blockIdx >> 1;
        uint32_t blockH = blockIdx & 1;

        auto data = buffer.Get<DT_X>();
        LocalTensor<DT_X> input0 = data;
        LocalTensor<DT_X> input1 = data[BLOCK2_CASE4_INPUT_BATCH_ELEMS];
        LocalTensor<DT_X> output = data[BLOCK2_CASE4_TASK_INPUT_ELEMS];
        LoadTaskInput(input0, input1, outB, blockH, loadParams);
        PipeBarrier<PIPE_ALL>();
        RepackTaskRows(output, input0, input1, repackParams);
        PipeBarrier<PIPE_ALL>();
        StoreTaskRows(output, outB, blockH, outParams);
        PipeBarrier<PIPE_MTE3>();
    }

private:
    __aicore__ inline void LoadTaskInput(
        LocalTensor<DT_X> &input0, LocalTensor<DT_X> &input1,
        uint32_t outB, uint32_t blockH, const DataCopyParams &loadParams) {
        uint32_t baseBatch = (blockH << 1) * BLOCK2_CASE4_OUTPUT_BATCH + outB;
        uint32_t phase1Batch = baseBatch + BLOCK2_CASE4_OUTPUT_BATCH;
        DataCopy(input0,
            xGm[static_cast<uint64_t>(baseBatch) * BLOCK2_CASE4_INPUT_BATCH_ELEMS],
            loadParams);
        DataCopy(input1,
            xGm[static_cast<uint64_t>(phase1Batch) * BLOCK2_CASE4_INPUT_BATCH_ELEMS],
            loadParams);
    }

    __aicore__ inline void RepackTaskRows(
        LocalTensor<DT_X> &output, LocalTensor<DT_X> &input0, LocalTensor<DT_X> &input1,
        const DataCopyParams &repackParams) {
        DataCopy(output, input0, repackParams);
        DataCopy(output[BLOCK2_CASE4_DEPTH], input1, repackParams);
    }

    __aicore__ inline void StoreTaskRows(
        LocalTensor<DT_X> &output, uint32_t outB, uint32_t blockH, const DataCopyParams &outParams) {
        uint64_t outBase =
            (static_cast<uint64_t>(outB) * BLOCK2_CASE4_OUTPUT_HEIGHT + blockH) * BLOCK2_CASE4_ROW_ELEMS;
        DataCopy(yGm[outBase], output, outParams);
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
};

template <class DT_X>
class KernelBatchToSpaceCase4Aic {
public:
    __aicore__ inline KernelBatchToSpaceCase4Aic() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
        pipe.InitBuffer(a1Buffer, BLOCK2_CASE4_TWO_ROW_ELEMS * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        constexpr uint16_t DEPTH_BLOCKS =
            static_cast<uint16_t>((BLOCK2_CASE4_DEPTH * sizeof(DT_X)) >> 5);
        const DataCopyParams laneParams{BLOCK2_CASE4_INPUT_WIDTH, DEPTH_BLOCKS, 0, DEPTH_BLOCKS};

        uint32_t blockIdx = GetBlockIdx();
        if (blockIdx >= (BLOCK2_CASE4_OUTPUT_ROWS >> 1)) {
            return;
        }
        uint32_t rowStart = blockIdx << 1;
        uint32_t outB = rowStart >> 3;
        uint32_t outH = rowStart & 7;
        auto data = a1Buffer.Get<DT_X>();
        CopyBatchRowToLocal(data, 0, outB, outH, laneParams);
        CopyBatchRowToLocal(data, 1, outB, outH + 1, laneParams);
        SetFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
        WaitFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
        DataCopy(yGm[static_cast<uint64_t>(rowStart) * BLOCK2_CASE4_ROW_ELEMS],
            data, BLOCK2_CASE4_TWO_ROW_ELEMS);
    }

private:
    __aicore__ inline void CopyBatchRowToLocal(
        LocalTensor<DT_X> &data, uint32_t localRow, uint32_t outB, uint32_t outH,
        const DataCopyParams &laneParams) {
        constexpr uint32_t BATCH5_STRIDE = BLOCK2_CASE4_OUTPUT_BATCH * BLOCK2_CASE4_INPUT_BATCH_ELEMS;
        uint32_t baseBatch = ((outH & 1) << 1) * BLOCK2_CASE4_OUTPUT_BATCH + outB;
        uint64_t rowBase =
            static_cast<uint64_t>(baseBatch) * BLOCK2_CASE4_INPUT_BATCH_ELEMS +
            static_cast<uint64_t>(outH >> 1) * BLOCK2_CASE4_INPUT_ROW_ELEMS;
        uint64_t localBase = static_cast<uint64_t>(localRow) * BLOCK2_CASE4_ROW_ELEMS;

        DataCopy(data[localBase], xGm[rowBase], laneParams);
        DataCopy(data[localBase + BLOCK2_CASE4_DEPTH], xGm[rowBase + BATCH5_STRIDE], laneParams);
    }

private:
    TPipe pipe;
    TBuf<TPosition::A1> a1Buffer;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
};

template <class DT_X>
class KernelBatchToSpaceCase5Aiv {
public:
    __aicore__ inline KernelBatchToSpaceCase5Aiv() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
        pipe.InitBuffer(inQueue, DOUBLE_BUFFER_NUM, BLOCK2_CASE5_TIGHT_ALIGNED_ELEMS * sizeof(DT_X));
        pipe.InitBuffer(outQueue, DOUBLE_BUFFER_NUM, BLOCK2_CASE5_TIGHT_ALIGNED_ELEMS * sizeof(DT_X));
        pipe.InitBuffer(offsetBuffer, BLOCK2_CASE5_COMPACT_OFFSET_ELEMS * sizeof(uint32_t));
    }

    __aicore__ inline void Process() {
        constexpr uint64_t TOTAL_TASKS = static_cast<uint64_t>(BLOCK2_CASE5_OUTPUT_WIDTH) << 1U;
        constexpr uint64_t OUTPUT_ROW_ELEMS =
            static_cast<uint64_t>(BLOCK2_CASE5_OUTPUT_WIDTH) * BLOCK2_PAIR_BURST_DEPTH;
        constexpr uint32_t DEPTH_BYTES = BLOCK2_PAIR_BURST_DEPTH * sizeof(DT_X);
        constexpr uint32_t LANE_A_BYTES = BLOCK2_CASE5_HALF_A_COLS * DEPTH_BYTES;
        constexpr uint32_t LANE_B_BYTES = BLOCK2_CASE5_HALF_B_COLS * DEPTH_BYTES;
        constexpr uint32_t OUT_BYTES = BLOCK2_CASE5_TIGHT_ELEMS * sizeof(DT_X);

        DataCopyPadExtParams<DT_X> padParams{false, 0, 0, static_cast<DT_X>(0)};
        DataCopyExtParams laneAParams{1, LANE_A_BYTES, 0, 0, 0};
        DataCopyExtParams laneBParams{1, LANE_B_BYTES, 0, 0, 0};
        DataCopyExtParams outParams{1, OUT_BYTES, 0, 0, 0};

        auto offsets = offsetBuffer.Get<uint32_t>();
        PrepareCompactOffsets(offsets);

        uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
        uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
        uint64_t tasksPerCore = TOTAL_TASKS / blockNum;
        uint64_t extraTasks = TOTAL_TASKS - tasksPerCore * blockNum;
        uint64_t task = blockIdx * tasksPerCore + Min(blockIdx, extraTasks);
        uint64_t taskEnd = task + tasksPerCore + (blockIdx < extraTasks ? 1U : 0U);
        if (task >= taskEnd) {
            return;
        }

        uint64_t pendingOutBase = PrefetchTask(task, laneAParams, laneBParams, padParams, OUTPUT_ROW_ELEMS);
        ++task;
        for (; task < taskEnd; ++task) {
            uint64_t nextOutBase = PrefetchTask(task, laneAParams, laneBParams, padParams, OUTPUT_ROW_ELEMS);
            StorePending(pendingOutBase, outParams, offsets);
            pendingOutBase = nextOutBase;
        }
        StorePending(pendingOutBase, outParams, offsets);
    }

private:
    __aicore__ inline void PrepareCompactOffsets(LocalTensor<uint32_t> &offsets) {
        LocalTensor<int32_t> offsetsI32 = offsets.template ReinterpretCast<int32_t>();
        constexpr uint32_t DEPTH_BYTES = BLOCK2_PAIR_BURST_DEPTH * sizeof(DT_X);
        constexpr uint32_t GROUP_BYTES = BLOCK2_CASE5_HALF_A_COLS * DEPTH_BYTES;
        constexpr uint32_t PAIR_ELEMS = BLOCK2_PAIR_BURST_DEPTH << 1U;
        constexpr uint32_t PERIOD_PAIRS = 4U;
        constexpr uint32_t PERIOD_ELEMS = PERIOD_PAIRS * PAIR_ELEMS;
        constexpr uint32_t PERIOD_INC_BYTES = PERIOD_PAIRS * DEPTH_BYTES;
        for (uint32_t pair = 0; pair < PERIOD_PAIRS; ++pair) {
            uint32_t pairBase = pair * PAIR_ELEMS;
            uint32_t pairByteBase = pair * DEPTH_BYTES;
            ArithProgression<int32_t>(
                offsetsI32[pairBase],
                static_cast<int32_t>(pairByteBase),
                static_cast<int32_t>(sizeof(DT_X)),
                static_cast<int32_t>(BLOCK2_PAIR_BURST_DEPTH));
            ArithProgression<int32_t>(
                offsetsI32[pairBase + BLOCK2_PAIR_BURST_DEPTH],
                static_cast<int32_t>(GROUP_BYTES + pairByteBase),
                static_cast<int32_t>(sizeof(DT_X)),
                static_cast<int32_t>(BLOCK2_PAIR_BURST_DEPTH));
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t built = PERIOD_ELEMS; built < BLOCK2_CASE5_COMPACT_OFFSET_ELEMS; built += PERIOD_ELEMS) {
            uint32_t count = Min(PERIOD_ELEMS, BLOCK2_CASE5_COMPACT_OFFSET_ELEMS - built);
            Adds(
                offsetsI32[built],
                offsetsI32[built - PERIOD_ELEMS],
                static_cast<int32_t>(PERIOD_INC_BYTES),
                count);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline uint64_t PrefetchTask(
        uint64_t task,
        DataCopyExtParams &laneAParams,
        DataCopyExtParams &laneBParams,
        DataCopyPadExtParams<DT_X> &padParams,
        uint64_t outputRowElems) {
        auto tight = inQueue.AllocTensor<DT_X>();
        LoadHalfRow(tight, task, laneAParams, laneBParams, padParams);
        inQueue.EnQue<DT_X>(tight);
        return CalcOutputBase(task, outputRowElems);
    }

    __aicore__ inline void StorePending(
        uint64_t outBase, DataCopyExtParams &outParams, LocalTensor<uint32_t> &offsets) {
        auto tight = inQueue.DeQue<DT_X>();
        auto packed = outQueue.AllocTensor<DT_X>();
        Gather<DT_X>(
            packed,
            tight,
            offsets,
            0,
            static_cast<uint32_t>(BLOCK2_CASE5_TIGHT_ELEMS));
        inQueue.FreeTensor<DT_X>(tight);
        outQueue.EnQue<DT_X>(packed);

        packed = outQueue.DeQue<DT_X>();
        DataCopyPad(yGm[outBase], packed, outParams);
        outQueue.FreeTensor<DT_X>(packed);
    }

    __aicore__ inline void LoadHalfRow(
        LocalTensor<DT_X> &tight,
        uint64_t task,
        DataCopyExtParams &laneAParams,
        DataCopyExtParams &laneBParams,
        DataCopyPadExtParams<DT_X> &padParams) {
        constexpr uint64_t INPUT_ROW_ELEMS =
            static_cast<uint64_t>(BLOCK2_CASE5_INPUT_SIZE) * BLOCK2_PAIR_BURST_DEPTH;
        constexpr uint64_t INPUT_BATCH_ELEMS =
            static_cast<uint64_t>(BLOCK2_CASE5_INPUT_SIZE) * INPUT_ROW_ELEMS;

        uint64_t row = task >> 1U;
        uint32_t half = static_cast<uint32_t>(task & 1U);
        uint32_t outWStart = half * BLOCK2_CASE5_WIDTH254_LANE_COLS;
        uint64_t paddedH = row + 1U;
        uint32_t blockH = static_cast<uint32_t>(paddedH & 1U);
        uint32_t paddedWStart = outWStart + 1U;
        uint32_t phaseA = paddedWStart & 1U;
        uint32_t phaseB = phaseA ^ 1U;
        uint64_t inH = paddedH >> 1U;
        uint64_t inWA = paddedWStart >> 1U;
        uint64_t inWB = (paddedWStart + 1U) >> 1U;
        uint64_t batchA = static_cast<uint64_t>((blockH << 1U) | phaseA);
        uint64_t batchB = static_cast<uint64_t>((blockH << 1U) | phaseB);
        uint64_t srcA = batchA * INPUT_BATCH_ELEMS + inH * INPUT_ROW_ELEMS +
                        inWA * BLOCK2_PAIR_BURST_DEPTH;
        uint64_t srcB = batchB * INPUT_BATCH_ELEMS + inH * INPUT_ROW_ELEMS +
                        inWB * BLOCK2_PAIR_BURST_DEPTH;

        DataCopyPad(tight, xGm[srcA], laneAParams, padParams);
        DataCopyPad(tight[BLOCK2_CASE5_HALF_A_ELEMS], xGm[srcB], laneBParams, padParams);
    }

    __aicore__ inline uint64_t CalcOutputBase(uint64_t task, uint64_t outputRowElems) {
        uint64_t row = task >> 1U;
        uint64_t half = task & 1U;
        return row * outputRowElems +
               half * BLOCK2_CASE5_WIDTH254_LANE_COLS * BLOCK2_PAIR_BURST_DEPTH;
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, DOUBLE_BUFFER_NUM> inQueue;
    TQue<TPosition::VECOUT, DOUBLE_BUFFER_NUM> outQueue;
    TBuf<TPosition::VECCALC> offsetBuffer;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
};

template <class DT_X>
class KernelBatchToSpace {
public:
    __aicore__ inline KernelBatchToSpace() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling) {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));

        outputPixels = tiling.output_pixels;
        inputHeight = tiling.input_height;
        inputWidth = tiling.input_width;
        depth = tiling.depth;
        outputBatch = tiling.output_batch;
        outputHeight = tiling.output_height;
        outputWidth = tiling.output_width;
        blockSize = tiling.block_size;
        cropTop = tiling.crop_top;
        cropLeft = tiling.crop_left;

        if (IsCase5()) {
            pipe.InitBuffer(case5GatherInQueue, DOUBLE_BUFFER_NUM, BLOCK2_CASE5_TIGHT_ALIGNED_ELEMS * sizeof(DT_X));
            pipe.InitBuffer(case5GatherOutQueue, DOUBLE_BUFFER_NUM, BLOCK2_CASE5_TIGHT_ALIGNED_ELEMS * sizeof(DT_X));
            pipe.InitBuffer(case5GatherOffsetBuffer, BLOCK2_CASE5_COMPACT_OFFSET_ELEMS * sizeof(uint32_t));
            return;
        }
        if (IsCase8()) {
            pipe.InitBuffer(burstQueue, DOUBLE_BUFFER_NUM, BLOCK2_CASE8_TILE_ELEMS * sizeof(DT_X));
            return;
        }
        if (IsCase9()) {
            pipe.InitBuffer(burstQueue, DOUBLE_BUFFER_NUM, BLOCK4_QUAD_BURST_BUFFER_ELEMS * sizeof(DT_X));
            return;
        }
        if (IsCase10()) {
            pipe.InitBuffer(burstQueue, DOUBLE_BUFFER_NUM, BLOCK2_CASE10_ROW_TILE_ELEMS * sizeof(DT_X));
            return;
        }
        pipe.InitBuffer(queue, BUFFER_NUM, QUEUE_ELEMS * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (IsCase1()) {
            ProcessCase1Depth128Shape56();
        } else if (IsCase2()) {
            ProcessCase2Depth5Rows();
        } else if (IsCase3()) {
            ProcessCase3Depth64PairRows();
        } else if (IsCase4()) {
            ProcessCase4Depth32Direct();
        } else if (IsCase5()) {
            ProcessCase5GatherSegment();
        } else if (IsCase7()) {
            ProcessCase7Depth16384();
        } else if (IsCase8()) {
            ProcessCase8Depth256Lanes();
        } else if (IsCase9()) {
            ProcessCase9Block4QuadBurst();
        } else if (IsCase10()) {
            ProcessCase10Depth32Width12();
        }
    }

private:
    __aicore__ inline void CopyChunk(uint64_t inBase, uint64_t outBase, uint32_t elems) {
        auto data = queue.AllocTensor<DT_X>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(elems * sizeof(DT_X)), 0, 0, 0};
        DataCopyPadExtParams<DT_X> padParams{false, 0, 0, static_cast<DT_X>(0)};

        DataCopyPad(data, xGm[inBase], copyParams, padParams);
        queue.EnQue<DT_X>(data);
        data = queue.DeQue<DT_X>();
        DataCopyPad(yGm[outBase], data, copyParams);
        queue.FreeTensor<DT_X>(data);
    }

    __aicore__ inline void DecodePixel(uint64_t pixel, uint32_t &outB, uint32_t &outH, uint32_t &outW) {
        outW = static_cast<uint32_t>(pixel % outputWidth);
        pixel /= outputWidth;
        outH = static_cast<uint32_t>(pixel % outputHeight);
        outB = static_cast<uint32_t>(pixel / outputHeight);
    }

    __aicore__ inline void NextPixel(uint32_t &outB, uint32_t &outH, uint32_t &outW) {
        ++outW;
        if (outW == outputWidth) {
            outW = 0;
            ++outH;
            if (outH == outputHeight) {
                outH = 0;
                ++outB;
            }
        }
    }

    __aicore__ inline void CopyCase1Row(uint64_t row, uint32_t outB, uint32_t outH,
        uint64_t inputRowElems, uint64_t inputBatchElems, DataCopyExtParams &outParams) {
        constexpr uint32_t DEPTH128 = 128;
        constexpr uint32_t OUT_W = 56;
        constexpr uint32_t OUTPUT_BATCH2 = 2;
        constexpr uint32_t LANE_COLS = OUT_W >> 1;
        constexpr uint32_t LANE_ELEMS = LANE_COLS * DEPTH128;

        uint32_t inH = outH >> 1;
        uint32_t inB0 = ((outH & 1) << 1) * OUTPUT_BATCH2 + outB;
        uint32_t inB1 = inB0 + OUTPUT_BATCH2;
        uint64_t inputRowBase = static_cast<uint64_t>(inH) * inputRowElems;
        uint64_t inBase0 = static_cast<uint64_t>(inB0) * inputBatchElems + inputRowBase;
        uint64_t inBase1 = static_cast<uint64_t>(inB1) * inputBatchElems + inputRowBase;
        uint64_t outBase = (row * OUT_W) << 7;

        auto data = queue.AllocTensor<DT_X>();
        DataCopy(data, xGm[inBase0], LANE_ELEMS);
        DataCopy(data[LANE_ELEMS], xGm[inBase1], LANE_ELEMS);
        queue.EnQue<DT_X>(data);
        data = queue.DeQue<DT_X>();
        DataCopyPad(yGm[outBase], data, outParams);
        DataCopyPad(yGm[outBase + DEPTH128], data[LANE_ELEMS], outParams);
        queue.FreeTensor<DT_X>(data);
    }

    __aicore__ inline void ProcessCase1Depth128Shape56() {
        constexpr uint32_t DEPTH128 = 128;
        constexpr uint32_t DEPTH_BYTES = DEPTH128 * sizeof(DT_X);
        constexpr uint32_t OUT_H = 56;
        constexpr uint32_t OUT_W = 56;
        constexpr uint32_t LANE_COLS = OUT_W >> 1;
        constexpr uint64_t TOTAL_WORK = static_cast<uint64_t>(2) * OUT_H;

        uint64_t inputRowElems = static_cast<uint64_t>(inputWidth) * DEPTH128;
        uint64_t inputBatchElems = static_cast<uint64_t>(inputHeight) * inputRowElems;
        DataCopyExtParams outParams{LANE_COLS, DEPTH_BYTES, 0, DEPTH_BYTES, 0};

        if (GetBlockNum() == 28) {
            uint32_t blockIdx = GetBlockIdx();
            CopyCase1Row(blockIdx, 0, blockIdx, inputRowElems, inputBatchElems, outParams);
            CopyCase1Row(static_cast<uint64_t>(blockIdx) + 28, 0, blockIdx + 28,
                inputRowElems, inputBatchElems, outParams);
            CopyCase1Row(static_cast<uint64_t>(blockIdx) + 56, 1, blockIdx,
                inputRowElems, inputBatchElems, outParams);
            CopyCase1Row(static_cast<uint64_t>(blockIdx) + 84, 1, blockIdx + 28,
                inputRowElems, inputBatchElems, outParams);
            return;
        }

        for (uint64_t row = GetBlockIdx(); row < TOTAL_WORK; row += GetBlockNum()) {
            uint32_t outB = row >= OUT_H ? 1 : 0;
            uint32_t outH = static_cast<uint32_t>(row - static_cast<uint64_t>(outB) * OUT_H);
            CopyCase1Row(row, outB, outH, inputRowElems, inputBatchElems, outParams);
        }
    }

    __aicore__ inline void ProcessCase2Depth5Rows() {
        constexpr uint32_t DEPTH5 = 5;
        constexpr uint32_t ALIGNED_DEPTH5 = 8;
        constexpr uint32_t DEPTH_BYTES = DEPTH5 * sizeof(DT_X);
        constexpr uint32_t OUT_H = 17;
        constexpr uint32_t OUT_W = 26;
        constexpr uint32_t LANE_COLS = OUT_W >> 1;
        constexpr uint32_t LANE_OFFSET = LANE_COLS * ALIGNED_DEPTH5;
        constexpr uint32_t INPUT_ROW_ELEMS = 75;
        constexpr uint32_t INPUT_BATCH_ELEMS = 750;
        constexpr uint32_t ROW_ELEMS = OUT_W * DEPTH5;

        DataCopyExtParams inParams{LANE_COLS, DEPTH_BYTES, 0, 0, 0};
        DataCopyExtParams outParams{LANE_COLS, DEPTH_BYTES, 0, DEPTH_BYTES, 0};
        DataCopyPadExtParams<DT_X> padParams{false, 0, 0, static_cast<DT_X>(0)};

        for (uint32_t outH = GetBlockIdx(); outH < OUT_H; outH += GetBlockNum()) {
            uint32_t inH = (outH >> 1) + 1;
            uint32_t blockH = outH & 1;
            uint64_t inputRowBase = static_cast<uint64_t>(inH) * INPUT_ROW_ELEMS;
            uint64_t rowBase0 = static_cast<uint64_t>(blockH << 1) * INPUT_BATCH_ELEMS + inputRowBase;
            uint64_t rowBase1 = rowBase0 + INPUT_BATCH_ELEMS;
            uint64_t outBase = static_cast<uint64_t>(outH) * ROW_ELEMS;

            auto data = queue.AllocTensor<DT_X>();
            DataCopyPad(data, xGm[rowBase1 + DEPTH5], inParams, padParams);
            DataCopyPad(data[LANE_OFFSET], xGm[rowBase0 + (DEPTH5 << 1)], inParams, padParams);
            queue.EnQue<DT_X>(data);
            data = queue.DeQue<DT_X>();
            DataCopyPad(yGm[outBase], data, outParams);
            DataCopyPad(yGm[outBase + DEPTH5], data[LANE_OFFSET], outParams);
            queue.FreeTensor<DT_X>(data);
        }
    }

    __aicore__ inline void CopyCase3RowUnrolled(
        LocalTensor<DT_X> &data, uint64_t localBase, uint64_t rowBase,
        const DataCopyExtParams &pairInParams, const DataCopyPadExtParams<DT_X> &padParams) {
        constexpr uint32_t DEPTH64 = 64;
        constexpr uint32_t PAIR_OUT_STRIDE = DEPTH64 << 1;

        DataCopyPad(data[localBase], xGm[rowBase], pairInParams, padParams);
        DataCopyPad(data[localBase + PAIR_OUT_STRIDE], xGm[rowBase + DEPTH64], pairInParams, padParams);
        DataCopyPad(data[localBase + PAIR_OUT_STRIDE * 2], xGm[rowBase + DEPTH64 * 2], pairInParams, padParams);
        DataCopyPad(data[localBase + PAIR_OUT_STRIDE * 3], xGm[rowBase + DEPTH64 * 3], pairInParams, padParams);
        DataCopyPad(data[localBase + PAIR_OUT_STRIDE * 4], xGm[rowBase + DEPTH64 * 4], pairInParams, padParams);
        DataCopyPad(data[localBase + PAIR_OUT_STRIDE * 5], xGm[rowBase + DEPTH64 * 5], pairInParams, padParams);
        DataCopyPad(data[localBase + PAIR_OUT_STRIDE * 6], xGm[rowBase + DEPTH64 * 6], pairInParams, padParams);
        DataCopyPad(data[localBase + PAIR_OUT_STRIDE * 7], xGm[rowBase + DEPTH64 * 7], pairInParams, padParams);
        DataCopyPad(data[localBase + PAIR_OUT_STRIDE * 8], xGm[rowBase + DEPTH64 * 8], pairInParams, padParams);
        DataCopyPad(data[localBase + PAIR_OUT_STRIDE * 9], xGm[rowBase + DEPTH64 * 9], pairInParams, padParams);
        DataCopyPad(data[localBase + PAIR_OUT_STRIDE * 10], xGm[rowBase + DEPTH64 * 10], pairInParams, padParams);
        DataCopyPad(data[localBase + PAIR_OUT_STRIDE * 11], xGm[rowBase + DEPTH64 * 11], pairInParams, padParams);
        DataCopyPad(data[localBase + PAIR_OUT_STRIDE * 12], xGm[rowBase + DEPTH64 * 12], pairInParams, padParams);
        DataCopyPad(data[localBase + PAIR_OUT_STRIDE * 13], xGm[rowBase + DEPTH64 * 13], pairInParams, padParams);
    }

    __aicore__ inline void ProcessCase3Depth64PairRows() {
        constexpr uint32_t DEPTH64 = 64;
        constexpr uint32_t OUTPUT_WIDTH28 = 28;
        constexpr uint32_t OUTPUT_HEIGHT28 = 28;
        constexpr uint32_t OUTPUT_BATCH4 = 4;
        constexpr uint32_t INPUT_WIDTH14 = 14;
        constexpr uint32_t ROW_ELEMS = OUTPUT_WIDTH28 * DEPTH64;
        constexpr uint32_t MAX_ROWS_PER_TILE = QUEUE_ELEMS / ROW_ELEMS;
        constexpr uint64_t OUTPUT_ROWS112 = 112;
        constexpr uint64_t INPUT_ROW_ELEMS = INPUT_WIDTH14 * DEPTH64;
        constexpr uint64_t INPUT_BATCH_ELEMS = 14 * INPUT_ROW_ELEMS;
        constexpr uint64_t BLOCK_BATCH_STRIDE_ELEMS = OUTPUT_BATCH4 * INPUT_BATCH_ELEMS;
        uint32_t depthBytes = static_cast<uint32_t>(DEPTH64 * sizeof(DT_X));
        uint32_t pairSrcStrideBytes =
            static_cast<uint32_t>((BLOCK_BATCH_STRIDE_ELEMS - DEPTH64) * sizeof(DT_X));
        DataCopyExtParams pairInParams{2, depthBytes, pairSrcStrideBytes, 0, 0};
        DataCopyPadExtParams<DT_X> padParams{false, 0, 0, static_cast<DT_X>(0)};

        uint64_t blockNum = GetBlockNum();
        uint64_t blockIdx = GetBlockIdx();
        uint64_t rowStart = 0;
        uint64_t rowCount = 0;
        if (blockNum == 40) {
            if (blockIdx < 32) {
                rowStart = blockIdx * 3;
                rowCount = 3;
            } else {
                rowStart = 96 + (blockIdx - 32) * 2;
                rowCount = 2;
            }
        } else {
            uint64_t rowsPerCore = OUTPUT_ROWS112 / blockNum;
            uint64_t extraRows = OUTPUT_ROWS112 - rowsPerCore * blockNum;
            rowStart = blockIdx * rowsPerCore + Min(blockIdx, extraRows);
            rowCount = rowsPerCore + (blockIdx < extraRows ? 1 : 0);
        }
        if (rowCount == 0) {
            return;
        }

        for (uint64_t doneRows = 0; doneRows < rowCount; doneRows += MAX_ROWS_PER_TILE) {
            uint32_t currentRows =
                static_cast<uint32_t>(Min(static_cast<uint64_t>(MAX_ROWS_PER_TILE), rowCount - doneRows));
            uint64_t currentRowStart = rowStart + doneRows;
            auto data = queue.AllocTensor<DT_X>();
            uint32_t outB = static_cast<uint32_t>(currentRowStart / OUTPUT_HEIGHT28);
            uint32_t outH =
                static_cast<uint32_t>(currentRowStart - static_cast<uint64_t>(outB) * OUTPUT_HEIGHT28);
            for (uint32_t localRow = 0; localRow < currentRows; ++localRow) {
                uint32_t inH = outH >> 1;
                uint32_t blockH = outH & 1;
                uint32_t baseBatch = (blockH << 1) * OUTPUT_BATCH4 + outB;
                uint64_t rowBase =
                    static_cast<uint64_t>(baseBatch) * INPUT_BATCH_ELEMS +
                    static_cast<uint64_t>(inH) * INPUT_ROW_ELEMS;
                uint64_t localBase = static_cast<uint64_t>(localRow) * ROW_ELEMS;
                CopyCase3RowUnrolled(data, localBase, rowBase, pairInParams, padParams);
                ++outH;
                if (outH == OUTPUT_HEIGHT28) {
                    outH = 0;
                    ++outB;
                }
            }

            queue.EnQue<DT_X>(data);
            data = queue.DeQue<DT_X>();
            DataCopy(yGm[currentRowStart * ROW_ELEMS], data, currentRows * ROW_ELEMS);
            queue.FreeTensor<DT_X>(data);
        }
    }

    __aicore__ inline void ProcessCase4Depth32Direct() {
        constexpr uint32_t DEPTH32 = 32;
        uint32_t pixelsPerTile = TILE_ELEMS / DEPTH32;
        uint64_t pixelsPerCore = CeilDiv(outputPixels, GetBlockNum());
        uint64_t start = static_cast<uint64_t>(GetBlockIdx()) * pixelsPerCore;
        uint64_t end = Min(start + pixelsPerCore, outputPixels);
        DataCopyExtParams inParams{1, static_cast<uint32_t>(DEPTH32 * sizeof(DT_X)), 0, 0, 0};
        DataCopyPadExtParams<DT_X> padParams{false, 0, 0, static_cast<DT_X>(0)};

        for (uint64_t pixel = start; pixel < end; pixel += pixelsPerTile) {
            uint32_t currentPixels = static_cast<uint32_t>(Min(static_cast<uint64_t>(pixelsPerTile), end - pixel));
            auto data = queue.AllocTensor<DT_X>();

            uint32_t outB;
            uint32_t outH;
            uint32_t outW;
            DecodePixel(pixel, outB, outH, outW);
            for (uint32_t idx = 0; idx < currentPixels; ++idx) {
                uint32_t inH = outH >> 1;
                uint32_t inW = outW >> 1;
                uint32_t inB = (((outH & 1) << 1) + (outW & 1)) * outputBatch + outB;
                uint64_t inBase = (((static_cast<uint64_t>(inB) * inputHeight + inH) * inputWidth + inW) << 5);
                DataCopyPad(data[idx << 5], xGm[inBase], inParams, padParams);
                NextPixel(outB, outH, outW);
            }

            queue.EnQue<DT_X>(data);
            data = queue.DeQue<DT_X>();
            DataCopyExtParams outParams{
                1, static_cast<uint32_t>((currentPixels << 5) * sizeof(DT_X)), 0, 0, 0};
            DataCopyPad(yGm[pixel << 5], data, outParams);
            queue.FreeTensor<DT_X>(data);
        }
    }

    __aicore__ inline void PrepareCase5HalfRowCompactOffsets(LocalTensor<uint32_t> &offsets) {
        LocalTensor<int32_t> offsetsI32 = offsets.template ReinterpretCast<int32_t>();
        uint32_t depthBytes = BLOCK2_PAIR_BURST_DEPTH * sizeof(DT_X);
        uint32_t groupBytes = BLOCK2_CASE5_HALF_A_COLS * depthBytes;
        constexpr uint32_t PAIR_ELEMS = BLOCK2_PAIR_BURST_DEPTH << 1U;
        constexpr uint32_t PERIOD_PAIRS = 4U;
        constexpr uint32_t PERIOD_ELEMS = PERIOD_PAIRS * PAIR_ELEMS;
        uint32_t periodIncBytes = PERIOD_PAIRS * depthBytes;
        for (uint32_t pair = 0; pair < PERIOD_PAIRS; ++pair) {
            uint32_t pairBase = pair * PAIR_ELEMS;
            uint32_t pairByteBase = pair * depthBytes;
            for (uint32_t channel = 0; channel < BLOCK2_PAIR_BURST_DEPTH; ++channel) {
                uint32_t channelByte = channel * sizeof(DT_X);
                offsetsI32.SetValue(pairBase + channel, static_cast<int32_t>(pairByteBase + channelByte));
                offsetsI32.SetValue(
                    pairBase + BLOCK2_PAIR_BURST_DEPTH + channel,
                    static_cast<int32_t>(groupBytes + pairByteBase + channelByte));
            }
        }
        SetFlag<HardEvent::S_V>(static_cast<TEventID>(6));
        WaitFlag<HardEvent::S_V>(static_cast<TEventID>(6));
        for (uint32_t built = PERIOD_ELEMS; built < BLOCK2_CASE5_COMPACT_OFFSET_ELEMS; built += PERIOD_ELEMS) {
            uint32_t count = Min(PERIOD_ELEMS, BLOCK2_CASE5_COMPACT_OFFSET_ELEMS - built);
            Adds(
                offsetsI32[built],
                offsetsI32[built - PERIOD_ELEMS],
                static_cast<int32_t>(periodIncBytes),
                count);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadCase5HalfRow(
        LocalTensor<DT_X> &tight,
        uint64_t task,
        DataCopyExtParams &laneALoadParams,
        DataCopyExtParams &laneBLoadParams,
        DataCopyPadExtParams<DT_X> &padParams) {
        uint64_t inputRowElems = static_cast<uint64_t>(BLOCK2_CASE5_INPUT_SIZE) * BLOCK2_PAIR_BURST_DEPTH;
        uint64_t inputBatchElems = static_cast<uint64_t>(BLOCK2_CASE5_INPUT_SIZE) * inputRowElems;
        uint64_t row = task >> 1U;
        uint32_t half = static_cast<uint32_t>(task & 1U);
        uint32_t owStart = half * BLOCK2_CASE5_WIDTH254_LANE_COLS;
        uint64_t paddedH = row + 1;
        uint32_t blockH = static_cast<uint32_t>(paddedH & 1U);
        uint32_t fullWStart = owStart + 1U;
        uint32_t phaseA = fullWStart & 1U;
        uint32_t phaseB = phaseA ^ 1U;
        uint64_t ih = paddedH >> 1U;
        uint64_t iwA = fullWStart >> 1U;
        uint64_t iwB = (fullWStart + 1U) >> 1U;
        uint64_t inputNA = static_cast<uint64_t>((blockH << 1U) | phaseA);
        uint64_t inputNB = static_cast<uint64_t>((blockH << 1U) | phaseB);
        uint64_t srcA = inputNA * inputBatchElems + ih * inputRowElems + iwA * BLOCK2_PAIR_BURST_DEPTH;
        uint64_t srcB = inputNB * inputBatchElems + ih * inputRowElems + iwB * BLOCK2_PAIR_BURST_DEPTH;

        DataCopyPad(tight, xGm[srcA], laneALoadParams, padParams);
        DataCopyPad(tight[BLOCK2_CASE5_HALF_A_ELEMS], xGm[srcB], laneBLoadParams, padParams);
    }

    __aicore__ inline uint64_t CalcCase5OutputBase(uint64_t task, uint64_t outputRowElems) {
        uint64_t row = task >> 1U;
        uint64_t half = task & 1U;
        return row * outputRowElems +
               half * BLOCK2_CASE5_WIDTH254_LANE_COLS * BLOCK2_PAIR_BURST_DEPTH;
    }

    __aicore__ inline void RepackCase5HalfRowCompact(
        LocalTensor<DT_X> &packed, LocalTensor<DT_X> &tight, LocalTensor<uint32_t> &offsets) {
        Gather<DT_X>(
            packed,
            tight,
            offsets,
            0,
            static_cast<uint32_t>(BLOCK2_CASE5_TIGHT_ELEMS));
    }

    __aicore__ inline void ProcessCase5GatherSegment() {
        uint64_t totalTasks = static_cast<uint64_t>(outputBatch) * outputHeight * 2U;
        uint64_t task = static_cast<uint64_t>(GetBlockIdx());
        uint64_t taskStep = static_cast<uint64_t>(GetBlockNum());
        if (taskStep == 0) {
            return;
        }

        DataCopyPadExtParams<DT_X> padParams{false, 0, 0, static_cast<DT_X>(0)};
        uint32_t depthBytes = BLOCK2_PAIR_BURST_DEPTH * sizeof(DT_X);
        DataCopyExtParams laneALoadParams{
            1,
            static_cast<uint32_t>(BLOCK2_CASE5_HALF_A_COLS * depthBytes),
            0,
            0,
            0};
        DataCopyExtParams laneBLoadParams{
            1,
            static_cast<uint32_t>(BLOCK2_CASE5_HALF_B_COLS * depthBytes),
            0,
            0,
            0};
        DataCopyExtParams outParams{
            1,
            static_cast<uint32_t>(BLOCK2_CASE5_TIGHT_ELEMS * sizeof(DT_X)),
            0,
            0,
            0};
        uint64_t outputRowElems =
            static_cast<uint64_t>(BLOCK2_CASE5_OUTPUT_WIDTH) * BLOCK2_PAIR_BURST_DEPTH;

        auto offsets = case5GatherOffsetBuffer.Get<uint32_t>();
        PrepareCase5HalfRowCompactOffsets(offsets);

        uint64_t pendingOutBase = 0;
        bool hasPending = false;

        for (; task < totalTasks; task += taskStep) {
            auto firstTight = case5GatherInQueue.AllocTensor<DT_X>();
            LoadCase5HalfRow(firstTight, task, laneALoadParams, laneBLoadParams, padParams);
            case5GatherInQueue.EnQue<DT_X>(firstTight);
            pendingOutBase = CalcCase5OutputBase(task, outputRowElems);
            hasPending = true;
            task += taskStep;
            break;
        }
        if (!hasPending) {
            return;
        }

        for (; task < totalTasks; task += taskStep) {
            auto nextTight = case5GatherInQueue.AllocTensor<DT_X>();
            LoadCase5HalfRow(nextTight, task, laneALoadParams, laneBLoadParams, padParams);
            case5GatherInQueue.EnQue<DT_X>(nextTight);

            auto readyTight = case5GatherInQueue.DeQue<DT_X>();
            auto packed = case5GatherOutQueue.AllocTensor<DT_X>();
            RepackCase5HalfRowCompact(packed, readyTight, offsets);
            case5GatherInQueue.FreeTensor<DT_X>(readyTight);
            case5GatherOutQueue.EnQue<DT_X>(packed);

            packed = case5GatherOutQueue.DeQue<DT_X>();
            DataCopyPad(yGm[pendingOutBase], packed, outParams);
            case5GatherOutQueue.FreeTensor<DT_X>(packed);

            pendingOutBase = CalcCase5OutputBase(task, outputRowElems);
        }

        auto readyTight = case5GatherInQueue.DeQue<DT_X>();
        auto packed = case5GatherOutQueue.AllocTensor<DT_X>();
        RepackCase5HalfRowCompact(packed, readyTight, offsets);
        case5GatherInQueue.FreeTensor<DT_X>(readyTight);
        case5GatherOutQueue.EnQue<DT_X>(packed);

        packed = case5GatherOutQueue.DeQue<DT_X>();
        DataCopyPad(yGm[pendingOutBase], packed, outParams);
        case5GatherOutQueue.FreeTensor<DT_X>(packed);
    }

    __aicore__ inline void ProcessCase7Depth16384() {
        constexpr uint64_t TOTAL_WORK = 8;
        for (uint64_t work = GetBlockIdx(); work < TOTAL_WORK; work += GetBlockNum()) {
            uint64_t offset = work * TILE_ELEMS;
            CopyChunk(offset, offset, TILE_ELEMS);
        }
    }

    __aicore__ inline void ProcessCase8Depth256Lanes() {
        constexpr uint32_t DEPTH256 = BLOCK2_CASE8_DEPTH;
        constexpr uint32_t COLS_PER_TILE = BLOCK2_CASE8_COLS_PER_TILE;
        constexpr uint32_t LANE_TILES = BLOCK2_CASE8_LANE_TILES;
        constexpr uint32_t WORK_PER_ROW = BLOCK2_CASE8_WORK_PER_ROW;
        constexpr uint64_t TOTAL_WORK = static_cast<uint64_t>(BLOCK2_CASE8_OUTPUT_HEIGHT) * WORK_PER_ROW;
        constexpr uint16_t TILE_BLOCKS =
            static_cast<uint16_t>((BLOCK2_CASE8_TILE_ELEMS * sizeof(DT_X)) >> 5);
        constexpr uint16_t DEPTH_BLOCKS =
            static_cast<uint16_t>((DEPTH256 * sizeof(DT_X)) >> 5);

        DataCopyParams inParams{1, TILE_BLOCKS, 0, 0};
        DataCopyParams outParams{
            static_cast<uint16_t>(COLS_PER_TILE),
            DEPTH_BLOCKS,
            0,
            DEPTH_BLOCKS};

        uint64_t work = GetBlockIdx();
        uint64_t pendingOutBase = 0;
        bool hasPending = false;

        for (; work < TOTAL_WORK; work += GetBlockNum()) {
            uint32_t row = static_cast<uint32_t>(work / WORK_PER_ROW);
            uint32_t laneWork = static_cast<uint32_t>(work - static_cast<uint64_t>(row) * WORK_PER_ROW);
            uint32_t lane = laneWork / LANE_TILES;
            uint32_t tile = laneWork - lane * LANE_TILES;
            uint32_t laneColStart = tile * COLS_PER_TILE;

            uint32_t inH = row >> 1;
            uint32_t blockH = row & 1;
            uint32_t inB = (blockH << 1) + lane;
            uint64_t inBase =
                (((static_cast<uint64_t>(inB) * BLOCK2_CASE8_INPUT_HEIGHT + inH) *
                BLOCK2_CASE8_INPUT_WIDTH + laneColStart) * DEPTH256);
            uint64_t outBase =
                ((static_cast<uint64_t>(row) * BLOCK2_CASE8_OUTPUT_WIDTH + (laneColStart << 1) + lane) *
                DEPTH256);

            auto firstData = burstQueue.AllocTensor<DT_X>();
            DataCopy(firstData, xGm[inBase], inParams);
            burstQueue.EnQue<DT_X>(firstData);
            pendingOutBase = outBase;
            hasPending = true;
            work += GetBlockNum();
            break;
        }
        if (!hasPending) {
            return;
        }

        for (; work < TOTAL_WORK; work += GetBlockNum()) {
            uint32_t row = static_cast<uint32_t>(work / WORK_PER_ROW);
            uint32_t laneWork = static_cast<uint32_t>(work - static_cast<uint64_t>(row) * WORK_PER_ROW);
            uint32_t lane = laneWork / LANE_TILES;
            uint32_t tile = laneWork - lane * LANE_TILES;
            uint32_t laneColStart = tile * COLS_PER_TILE;

            uint32_t inH = row >> 1;
            uint32_t blockH = row & 1;
            uint32_t inB = (blockH << 1) + lane;
            uint64_t inBase =
                (((static_cast<uint64_t>(inB) * BLOCK2_CASE8_INPUT_HEIGHT + inH) *
                BLOCK2_CASE8_INPUT_WIDTH + laneColStart) * DEPTH256);
            uint64_t outBase =
                ((static_cast<uint64_t>(row) * BLOCK2_CASE8_OUTPUT_WIDTH + (laneColStart << 1) + lane) *
                DEPTH256);

            auto nextData = burstQueue.AllocTensor<DT_X>();
            DataCopy(nextData, xGm[inBase], inParams);
            burstQueue.EnQue<DT_X>(nextData);

            auto readyData = burstQueue.DeQue<DT_X>();
            DataCopy(yGm[pendingOutBase], readyData, outParams);
            burstQueue.FreeTensor<DT_X>(readyData);

            pendingOutBase = outBase;
        }

        auto readyData = burstQueue.DeQue<DT_X>();
        DataCopy(yGm[pendingOutBase], readyData, outParams);
        burstQueue.FreeTensor<DT_X>(readyData);
    }

    __aicore__ inline void LoadCase9QuadBurst(
        LocalTensor<DT_X> &data, uint32_t rowOutB, uint32_t rowInH, uint32_t rowBlockH,
        uint32_t startOutW, uint32_t currentPixels, uint32_t stride,
        const DataCopyExtParams &quadInParams, bool enableQuadBurst) {
        uint32_t paddedW = startOutW + cropLeft;
        uint32_t inW = paddedW >> 2;
        uint32_t blockW = paddedW & 3;
        uint32_t baseBatch = (rowBlockH << 2) * outputBatch + rowOutB;

        uint64_t batchStride = static_cast<uint64_t>(inputHeight) * inputWidth * depth;
        uint64_t blockBatchStride = static_cast<uint64_t>(outputBatch) * batchStride;
        uint64_t baseBatchOffset = static_cast<uint64_t>(baseBatch) * batchStride;
        uint64_t rowBaseOffset = static_cast<uint64_t>(rowInH) * inputWidth * depth;
        bool useQuadBurstCopy =
            enableQuadBurst &&
            (quadInParams.blockLen % 32 == 0) &&
            (quadInParams.srcStride % 32 == 0) &&
            ((quadInParams.blockLen >> 5) <= 0xFFFFu) &&
            ((quadInParams.srcStride >> 5) <= 0xFFFFu);
        DataCopyParams quadCopyParams{
            4,
            static_cast<uint16_t>(quadInParams.blockLen >> 5),
            static_cast<uint16_t>(quadInParams.srcStride >> 5),
            0};

        uint32_t idx = 0;
        uint32_t prefixPixels = Min(currentPixels, (4 - blockW) & 3);
        for (; idx < prefixPixels; ++idx) {
            uint64_t spatialOffset = rowBaseOffset + static_cast<uint64_t>(inW) * depth;
            uint64_t inBase = baseBatchOffset + static_cast<uint64_t>(blockW) * blockBatchStride + spatialOffset;
            DataCopy(data[idx * stride], xGm[inBase], depth);
            ++blockW;
            if (blockW == 4) {
                blockW = 0;
                ++inW;
            }
        }

        uint32_t remaining = currentPixels - idx;
        DataCopyPadExtParams<DT_X> padParams{false, 0, 0, static_cast<DT_X>(0)};
        while (remaining >= 4) {
            uint64_t spatialOffset = rowBaseOffset + static_cast<uint64_t>(inW) * depth;
            if (useQuadBurstCopy) {
                DataCopy(data[idx * stride], xGm[baseBatchOffset + spatialOffset], quadCopyParams);
            } else {
                DataCopyPad(data[idx * stride], xGm[baseBatchOffset + spatialOffset], quadInParams, padParams);
            }
            idx += 4;
            remaining -= 4;
            ++inW;
        }

        for (; remaining > 0; --remaining, ++idx) {
            uint64_t spatialOffset = rowBaseOffset + static_cast<uint64_t>(inW) * depth;
            uint64_t inBase = baseBatchOffset + static_cast<uint64_t>(blockW) * blockBatchStride + spatialOffset;
            DataCopy(data[idx * stride], xGm[inBase], depth);
            ++blockW;
            if (blockW == 4) {
                blockW = 0;
                ++inW;
            }
        }
    }

    __aicore__ inline void CalcCase9LaneCopy(
        uint32_t startOutW, uint32_t currentPixels, uint32_t lane,
        uint32_t &firstOffset, uint32_t &lanePixels) {
        uint32_t startBlockW = (startOutW + cropLeft) & 3;
        firstOffset = (lane + 4 - startBlockW) & 3;
        if (firstOffset >= currentPixels) {
            lanePixels = 0;
            return;
        }
        lanePixels = ((currentPixels - 1 - firstOffset) >> 2) + 1;
    }

    __aicore__ inline void LoadCase9LaneBursts(
        LocalTensor<DT_X> &data, uint32_t rowOutB, uint32_t rowInH, uint32_t rowBlockH,
        uint32_t startOutW, uint32_t currentPixels) {
        constexpr uint32_t DEPTH64 = BLOCK4_ROW_SEGMENT_DEPTH;
        constexpr uint32_t LANE_ELEMS = BLOCK4_ROW_SEGMENT_LANE_ELEMS;
        uint32_t baseBatch = (rowBlockH << 2) * outputBatch + rowOutB;
        uint64_t batchStride = static_cast<uint64_t>(inputHeight) * inputWidth * depth;
        uint64_t blockBatchStride = static_cast<uint64_t>(outputBatch) * batchStride;
        uint64_t baseBatchOffset = static_cast<uint64_t>(baseBatch) * batchStride;
        uint64_t rowBaseOffset = static_cast<uint64_t>(rowInH) * inputWidth * depth;

        for (uint32_t lane = 0; lane < 4; ++lane) {
            uint32_t firstOffset;
            uint32_t lanePixels;
            CalcCase9LaneCopy(startOutW, currentPixels, lane, firstOffset, lanePixels);
            if (lanePixels == 0) {
                continue;
            }
            uint32_t inW = (startOutW + firstOffset + cropLeft) >> 2;
            uint64_t inBase =
                baseBatchOffset + static_cast<uint64_t>(lane) * blockBatchStride +
                rowBaseOffset + static_cast<uint64_t>(inW) * depth;
            DataCopy(data[lane * LANE_ELEMS], xGm[inBase], lanePixels * DEPTH64);
        }
    }

    __aicore__ inline void StoreCase9LaneBursts(
        LocalTensor<DT_X> &data, uint32_t row, uint32_t startOutW, uint32_t currentPixels) {
        constexpr uint32_t DEPTH64 = BLOCK4_ROW_SEGMENT_DEPTH;
        constexpr uint32_t LANE_ELEMS = BLOCK4_ROW_SEGMENT_LANE_ELEMS;
        constexpr uint16_t DEPTH_BLOCKS = static_cast<uint16_t>((DEPTH64 * sizeof(DT_X)) >> 5);
        constexpr uint16_t OUT_GAP_BLOCKS = static_cast<uint16_t>(((DEPTH64 * 3) * sizeof(DT_X)) >> 5);

        for (uint32_t lane = 0; lane < 4; ++lane) {
            uint32_t firstOffset;
            uint32_t lanePixels;
            CalcCase9LaneCopy(startOutW, currentPixels, lane, firstOffset, lanePixels);
            if (lanePixels == 0) {
                continue;
            }
            DataCopyParams outParams{static_cast<uint16_t>(lanePixels), DEPTH_BLOCKS, 0, OUT_GAP_BLOCKS};
            uint64_t outBase =
                (static_cast<uint64_t>(row) * outputWidth + startOutW + firstOffset) * depth;
            DataCopy(yGm[outBase], data[lane * LANE_ELEMS], outParams);
        }
    }

    __aicore__ inline void ProcessCase9Block4QuadBurst() {
        uint32_t pixelsPerTile = BLOCK4_ROW_SEGMENT_PIXELS_PER_TILE;
        uint64_t outputRows = static_cast<uint64_t>(outputBatch) * outputHeight;
        uint64_t rowTiles = CeilDiv(static_cast<uint64_t>(outputWidth), static_cast<uint64_t>(pixelsPerTile));
        uint64_t totalWork = outputRows * rowTiles;

        uint32_t depthBytes = depth * sizeof(DT_X);
        uint64_t batchStrideElems = static_cast<uint64_t>(inputHeight) * inputWidth * depth;
        uint64_t blockBatchStrideBytes =
            static_cast<uint64_t>(outputBatch) * batchStrideElems * sizeof(DT_X);
        bool enableQuadBurst =
            blockBatchStrideBytes >= depthBytes &&
            (blockBatchStrideBytes - depthBytes) <= 0xFFFFFFFFull;
        DataCopyExtParams quadInParams{
            4,
            depthBytes,
            static_cast<uint32_t>(enableQuadBurst ? (blockBatchStrideBytes - depthBytes) : 0),
            0,
            0};

        uint64_t work = GetBlockIdx();
        uint64_t pendingOutBase = 0;
        uint32_t pendingPixels = 0;
        bool hasPending = false;

        for (; work < totalWork; work += GetBlockNum()) {
            uint32_t row = static_cast<uint32_t>(work / rowTiles);
            uint32_t tile = static_cast<uint32_t>(work - static_cast<uint64_t>(row) * rowTiles);
            uint32_t startOutW = tile * pixelsPerTile;
            uint32_t currentPixels = Min(pixelsPerTile, outputWidth - startOutW);

            auto firstData = burstQueue.AllocTensor<DT_X>();
            uint32_t outB = row / outputHeight;
            uint32_t outH = row - outB * outputHeight;
            uint32_t paddedH = outH + cropTop;
            uint32_t inH = paddedH >> 2;
            uint32_t blockH = paddedH & 3;
            LoadCase9QuadBurst(
                firstData, outB, inH, blockH, startOutW, currentPixels, BLOCK4_ROW_SEGMENT_DEPTH,
                quadInParams, enableQuadBurst);
            burstQueue.EnQue<DT_X>(firstData);

            pendingOutBase = (static_cast<uint64_t>(row) * outputWidth + startOutW) * depth;
            pendingPixels = currentPixels;
            hasPending = true;
            work += GetBlockNum();
            break;
        }

        if (!hasPending) {
            return;
        }

        for (; work < totalWork; work += GetBlockNum()) {
            uint32_t row = static_cast<uint32_t>(work / rowTiles);
            uint32_t tile = static_cast<uint32_t>(work - static_cast<uint64_t>(row) * rowTiles);
            uint32_t startOutW = tile * pixelsPerTile;
            uint32_t currentPixels = Min(pixelsPerTile, outputWidth - startOutW);

            auto nextData = burstQueue.AllocTensor<DT_X>();
            uint32_t outB = row / outputHeight;
            uint32_t outH = row - outB * outputHeight;
            uint32_t paddedH = outH + cropTop;
            uint32_t inH = paddedH >> 2;
            uint32_t blockH = paddedH & 3;
            LoadCase9QuadBurst(
                nextData, outB, inH, blockH, startOutW, currentPixels, BLOCK4_ROW_SEGMENT_DEPTH,
                quadInParams, enableQuadBurst);
            burstQueue.EnQue<DT_X>(nextData);

            auto readyData = burstQueue.DeQue<DT_X>();
            DataCopy(yGm[pendingOutBase], readyData, pendingPixels * depth);
            burstQueue.FreeTensor<DT_X>(readyData);

            pendingOutBase = (static_cast<uint64_t>(row) * outputWidth + startOutW) * depth;
            pendingPixels = currentPixels;
        }

        auto readyData = burstQueue.DeQue<DT_X>();
        DataCopy(yGm[pendingOutBase], readyData, pendingPixels * depth);
        burstQueue.FreeTensor<DT_X>(readyData);
    }

    __aicore__ inline void LoadCase10RowTile(
        LocalTensor<DT_X> &data, uint32_t startRow, uint32_t currentRows,
        const DataCopyParams &laneParams) {
        constexpr uint32_t DEPTH32 = BLOCK2_DEPTH32_BURST_DEPTH;
        constexpr uint32_t OUTPUT_BATCH4 = 4;
        constexpr uint32_t INPUT_H = 1024;
        constexpr uint32_t INPUT_W = 6;
        constexpr uint32_t ROW_ELEMS = BLOCK2_CASE10_ROW_ELEMS;
        constexpr uint32_t INPUT_ROW_ELEMS = INPUT_W * DEPTH32;
        constexpr uint32_t INPUT_BATCH_ELEMS = INPUT_H * INPUT_ROW_ELEMS;
        constexpr uint32_t ROW_PAIR_ELEMS = ROW_ELEMS << 1;

        uint32_t pairStart = startRow >> 1;
        uint32_t pairRows = currentRows >> 1;
        for (uint32_t localPair = 0; localPair < pairRows; ++localPair) {
            uint32_t pair = pairStart + localPair;
            uint32_t outB = pair >> 10;
            uint32_t inH = pair - (outB << 10);
            uint32_t inB0 = outB;
            uint32_t inB1 = inB0 + OUTPUT_BATCH4;
            uint32_t inB2 = inB1 + OUTPUT_BATCH4;
            uint32_t inB3 = inB2 + OUTPUT_BATCH4;
            uint64_t inputRowBase = static_cast<uint64_t>(inH) * INPUT_ROW_ELEMS;
            uint64_t rowBase0 = static_cast<uint64_t>(inB0) * INPUT_BATCH_ELEMS + inputRowBase;
            uint64_t rowBase1 = static_cast<uint64_t>(inB1) * INPUT_BATCH_ELEMS + inputRowBase;
            uint64_t rowBase2 = static_cast<uint64_t>(inB2) * INPUT_BATCH_ELEMS + inputRowBase;
            uint64_t rowBase3 = static_cast<uint64_t>(inB3) * INPUT_BATCH_ELEMS + inputRowBase;
            uint64_t localBase = static_cast<uint64_t>(localPair) * ROW_PAIR_ELEMS;

            DataCopy(data[localBase], xGm[rowBase0], laneParams);
            DataCopy(data[localBase + DEPTH32], xGm[rowBase1], laneParams);
            DataCopy(data[localBase + ROW_ELEMS], xGm[rowBase2], laneParams);
            DataCopy(data[localBase + ROW_ELEMS + DEPTH32], xGm[rowBase3], laneParams);
        }
    }

    __aicore__ inline void LoadCase10Full20Fast(
        LocalTensor<DT_X> &data, uint32_t startRow, const DataCopyParams &laneParams) {
        constexpr uint32_t DEPTH32 = BLOCK2_DEPTH32_BURST_DEPTH;
        constexpr uint32_t OUTPUT_BATCH4 = 4;
        constexpr uint32_t INPUT_H = 1024;
        constexpr uint32_t INPUT_W = 6;
        constexpr uint32_t ROW_ELEMS = BLOCK2_CASE10_ROW_ELEMS;
        constexpr uint32_t INPUT_ROW_ELEMS = INPUT_W * DEPTH32;
        constexpr uint32_t INPUT_BATCH_ELEMS = INPUT_H * INPUT_ROW_ELEMS;
        constexpr uint32_t PAIR_ROWS = BLOCK2_CASE10_ROW_TILE_ROWS >> 1;
        constexpr uint32_t ROW_PAIR_ELEMS = ROW_ELEMS << 1;

        uint32_t pairStart = startRow >> 1;
        for (uint32_t localPair = 0; localPair < PAIR_ROWS; ++localPair) {
            uint32_t pair = pairStart + localPair;
            uint32_t outB = pair >> 10;
            uint32_t inH = pair & 1023;
            uint64_t inputRowBase = static_cast<uint64_t>(inH) * INPUT_ROW_ELEMS;
            uint64_t batchBase = static_cast<uint64_t>(outB) * INPUT_BATCH_ELEMS + inputRowBase;
            uint64_t localBase = static_cast<uint64_t>(localPair) * ROW_PAIR_ELEMS;

            DataCopy(data[localBase], xGm[batchBase], laneParams);
            DataCopy(data[localBase + DEPTH32],
                xGm[batchBase + static_cast<uint64_t>(OUTPUT_BATCH4) * INPUT_BATCH_ELEMS],
                laneParams);
            DataCopy(data[localBase + ROW_ELEMS],
                xGm[batchBase + static_cast<uint64_t>(OUTPUT_BATCH4 << 1) * INPUT_BATCH_ELEMS],
                laneParams);
            DataCopy(data[localBase + ROW_ELEMS + DEPTH32],
                xGm[batchBase + static_cast<uint64_t>(OUTPUT_BATCH4 * 3) * INPUT_BATCH_ELEMS],
                laneParams);
        }
    }

    __aicore__ inline void LoadCase10TileData(
        LocalTensor<DT_X> &data, uint32_t startRow, uint32_t currentRows,
        const DataCopyParams &laneParams) {
        if (currentRows == BLOCK2_CASE10_ROW_TILE_ROWS) {
            LoadCase10Full20Fast(data, startRow, laneParams);
        } else {
            LoadCase10RowTile(data, startRow, currentRows, laneParams);
        }
    }

    __aicore__ inline void ProcessCase10Depth32Width12() {
        constexpr uint32_t DEPTH32 = BLOCK2_DEPTH32_BURST_DEPTH;
        constexpr uint32_t DEPTH_BYTES = DEPTH32 * sizeof(DT_X);
        constexpr uint32_t OUTPUT_BATCH4 = 4;
        constexpr uint32_t OUT_H = 2048;
        constexpr uint32_t OUT_W = 12;
        constexpr uint32_t ROW_ELEMS = BLOCK2_CASE10_ROW_ELEMS;
        constexpr uint32_t ROWS_PER_TILE = BLOCK2_CASE10_ROW_TILE_ROWS;
        constexpr uint32_t TOTAL_ROWS = OUTPUT_BATCH4 * OUT_H;
        constexpr uint32_t TOTAL_TILES = (TOTAL_ROWS + ROWS_PER_TILE - 1) / ROWS_PER_TILE;
        constexpr uint16_t DEPTH_BLOCKS = static_cast<uint16_t>(DEPTH_BYTES >> 5);

        DataCopyParams laneParams{
            static_cast<uint16_t>(OUT_W >> 1),
            DEPTH_BLOCKS,
            0,
            DEPTH_BLOCKS};

        uint32_t tile = GetBlockIdx();
        uint32_t pendingStartRow = 0;
        uint32_t pendingRows = 0;
        bool hasPending = false;

        for (; tile < TOTAL_TILES; tile += GetBlockNum()) {
            uint32_t startRow = tile * ROWS_PER_TILE;
            uint32_t currentRows = Min(ROWS_PER_TILE, TOTAL_ROWS - startRow);
            auto firstData = burstQueue.AllocTensor<DT_X>();
            LoadCase10TileData(firstData, startRow, currentRows, laneParams);
            burstQueue.EnQue<DT_X>(firstData);
            pendingStartRow = startRow;
            pendingRows = currentRows;
            hasPending = true;
            tile += GetBlockNum();
            break;
        }

        if (!hasPending) {
            return;
        }

        for (; tile < TOTAL_TILES; tile += GetBlockNum()) {
            uint32_t startRow = tile * ROWS_PER_TILE;
            uint32_t currentRows = Min(ROWS_PER_TILE, TOTAL_ROWS - startRow);
            auto nextData = burstQueue.AllocTensor<DT_X>();
            LoadCase10TileData(nextData, startRow, currentRows, laneParams);
            burstQueue.EnQue<DT_X>(nextData);

            auto readyData = burstQueue.DeQue<DT_X>();
            DataCopy(yGm[static_cast<uint64_t>(pendingStartRow) * ROW_ELEMS],
                readyData, pendingRows * ROW_ELEMS);
            burstQueue.FreeTensor<DT_X>(readyData);

            pendingStartRow = startRow;
            pendingRows = currentRows;
        }

        auto readyData = burstQueue.DeQue<DT_X>();
        DataCopy(yGm[static_cast<uint64_t>(pendingStartRow) * ROW_ELEMS],
            readyData, pendingRows * ROW_ELEMS);
        burstQueue.FreeTensor<DT_X>(readyData);
    }

    __aicore__ inline bool IsCase1() {
        return blockSize == 2 && sizeof(DT_X) == 4 && depth == 128 && outputPixels == 6272 &&
               outputBatch == 2 && outputHeight == 56 && outputWidth == 56 &&
               inputHeight == 28 && inputWidth == 28 && cropTop == 0 && cropLeft == 0;
    }

    __aicore__ inline bool IsCase2() {
        return blockSize == 2 && sizeof(DT_X) == 4 && depth == 5 && outputPixels == 442 &&
               outputBatch == 1 && outputHeight == 17 && outputWidth == 26 &&
               inputHeight == 10 && inputWidth == 15 && cropTop == 2 && cropLeft == 3;
    }

    __aicore__ inline bool IsCase3() {
        return blockSize == 2 && sizeof(DT_X) == 4 && depth == 64 &&
               outputPixels == BLOCK2_FP32_DEPTH64_ROW_PIXELS &&
               outputBatch == 4 && outputHeight == 28 && outputWidth == 28 &&
               inputHeight == 14 && inputWidth == 14 && cropTop == 0 && cropLeft == 0;
    }

    __aicore__ inline bool IsCase4() {
        return blockSize == 2 && sizeof(DT_X) == 4 && depth == 32 && outputPixels == 480 &&
               outputBatch == 5 && outputHeight == 8 && outputWidth == 12 &&
               inputHeight == 4 && inputWidth == 6 && cropTop == 0 && cropLeft == 0;
    }

    __aicore__ inline bool IsCase5() {
        return blockSize == 2 && sizeof(DT_X) == 2 && depth == BLOCK2_PAIR_BURST_DEPTH &&
               outputPixels == BLOCK2_CASE5_OUTPUT_PIXELS && outputBatch == 1 &&
               outputHeight == BLOCK2_CASE5_OUTPUT_WIDTH && outputWidth == BLOCK2_CASE5_OUTPUT_WIDTH &&
               inputHeight == BLOCK2_CASE5_INPUT_SIZE && inputWidth == BLOCK2_CASE5_INPUT_SIZE &&
               cropTop == 1 && cropLeft == 1;
    }

    __aicore__ inline bool IsCase7() {
        return blockSize == 2 && sizeof(DT_X) == 2 && depth == 16384 && outputPixels == 4 &&
               outputBatch == 1 && outputHeight == 2 && outputWidth == 2 &&
               inputHeight == 1 && inputWidth == 1 && cropTop == 0 && cropLeft == 0;
    }

    __aicore__ inline bool IsCase8() {
        return blockSize == 2 && sizeof(DT_X) == 2 && depth == 256 && outputPixels == 20480 &&
               outputBatch == 1 && outputHeight == 20 && outputWidth == 1024 &&
               inputHeight == 10 && inputWidth == 512 && cropTop == 0 && cropLeft == 0;
    }

    __aicore__ inline bool IsCase9() {
        return blockSize == 4 && sizeof(DT_X) == 2 && depth == BLOCK4_ROW_SEGMENT_DEPTH &&
               outputPixels == BLOCK4_ROW_SEGMENT_OUTPUT_PIXELS &&
               outputBatch == 1 && outputHeight == 40 && outputWidth == 1535 &&
               inputHeight == 10 && inputWidth == 512 && cropTop == 0 && cropLeft == BLOCK4_ROW_SEGMENT_CROP_LEFT;
    }

    __aicore__ inline bool IsCase10() {
        return blockSize == 2 && sizeof(DT_X) == 2 && depth == BLOCK2_DEPTH32_BURST_DEPTH &&
               outputPixels == BLOCK2_DEPTH32_OUTPUT_PIXELS && outputBatch == 4 &&
               outputHeight == 2048 && outputWidth == 12 &&
               inputHeight == 1024 && inputWidth == 6 && cropTop == 0 && cropLeft == 0;
    }

private:
    TPipe pipe;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> queue;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, DOUBLE_BUFFER_NUM> burstQueue;
    TQue<TPosition::VECIN, DOUBLE_BUFFER_NUM> case5GatherInQueue;
    TQue<TPosition::VECOUT, DOUBLE_BUFFER_NUM> case5GatherOutQueue;
    TBuf<TPosition::VECCALC> case5GatherOffsetBuffer;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;

    uint64_t outputPixels;
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t depth;
    uint32_t outputBatch;
    uint32_t outputHeight;
    uint32_t outputWidth;
    uint32_t blockSize;
    uint32_t cropTop;
    uint32_t cropLeft;
};

template <typename DT_X, uint8_t BTS_SCENE>
 __global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);
    if constexpr (BTS_SCENE == BTS_SCENE_CASE6_AIC) {
        KernelBatchToSpaceCase6Aic<DT_X> op;
        op.Init(x, y);
        op.Process();
    } else if constexpr (BTS_SCENE == BTS_SCENE_CASE7_AIC) {
        KernelBatchToSpaceCase7Aic<DT_X> op;
        op.Init(x, y);
        op.Process();
    } else if constexpr (BTS_SCENE == BTS_SCENE_CASE10_AIC) {
        KernelBatchToSpaceCase10Aic<DT_X> op;
        op.Init(x, y);
        op.Process();
    } else if constexpr (BTS_SCENE == BTS_SCENE_CASE10_AIV) {
        KernelBatchToSpaceCase10Aiv<DT_X> op;
        op.Init(x, y);
        op.Process();
    } else if constexpr (BTS_SCENE == BTS_SCENE_CASE9_AIC) {
        KernelBatchToSpaceCase9Aic<DT_X> op;
        op.Init(x, y);
        op.Process();
    } else if constexpr (BTS_SCENE == BTS_SCENE_CASE9_AIV) {
        KernelBatchToSpaceCase9Aiv<DT_X> op;
        op.Init(x, y);
        op.Process();
    } else if constexpr (BTS_SCENE == BTS_SCENE_CASE3_AIC) {
        KernelBatchToSpaceCase3Aic<DT_X> op;
        op.Init(x, y);
        op.Process();
    } else if constexpr (BTS_SCENE == BTS_SCENE_CASE3_AIV) {
        KernelBatchToSpaceCase3Aiv<DT_X> op;
        op.Init(x, y);
        op.Process();
    } else if constexpr (BTS_SCENE == BTS_SCENE_CASE1_AIV) {
        KernelBatchToSpaceCase1Aiv<DT_X> op;
        op.Init(x, y);
        op.Process();
    } else if constexpr (BTS_SCENE == BTS_SCENE_CASE2_AIV) {
        KernelBatchToSpaceCase2Aiv<DT_X> op;
        op.Init(x, y);
        op.Process();
    } else if constexpr (BTS_SCENE == BTS_SCENE_CASE4_AIV) {
        KernelBatchToSpaceCase4Aiv<DT_X> op;
        op.Init(x, y);
        op.Process();
    } else if constexpr (BTS_SCENE == BTS_SCENE_CASE4_AIC) {
        KernelBatchToSpaceCase4Aic<DT_X> op;
        op.Init(x, y);
        op.Process();
    } else {
        GET_TILING_DATA_WITH_STRUCT(BatchToSpaceTilingData, tiling_data, tiling);
        KernelBatchToSpace<DT_X> op;
        op.Init(x, y, tiling_data);
        op.Process();
    }
}
