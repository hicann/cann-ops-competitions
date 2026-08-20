#include "kernel_operator.h"
#include "kernel.h"
using namespace AscendC;

template <typename T, bool Reverse = false>
__aicore__ inline void unpack_3_small(GM_ADDR input, GM_ADDR output, int32_t coreBlocks, int32_t remainBlocks, int32_t H, int32_t W) {
    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;
    int32_t coreBaseBlockIndex;

    if (GetBlockIdx() < remainBlocks) {
        coreBlocks += 1;
        coreBaseBlockIndex = GetBlockIdx() * coreBlocks;
    } else {
        coreBaseBlockIndex = coreBlocks * GetBlockIdx() + remainBlocks;
    }

    inputGm.SetGlobalBuffer((__gm__ T *)input, coreBlocks);
    constexpr int32_t BufferNum = 1;
    constexpr int32_t MAX_UB_SIZE_KB = 190;

    int32_t local_num = 1; // xLocal
    int32_t max_space_size = MAX_UB_SIZE_KB / local_num / BufferNum * 1024;
    int32_t W_padded = ceil_aligned_32B<T>(W);

    int32_t spaceSize = min((int32_t)(H * W_padded * sizeof(T)), max_space_size);
    int32_t n_rows_per_iter = spaceSize / sizeof(T) / W_padded;
    int32_t smallLoopTimes = (H + n_rows_per_iter - 1) / n_rows_per_iter;
    int32_t tailRows = H - (smallLoopTimes - 1) * n_rows_per_iter;

    int32_t xLocal_bytes = n_rows_per_iter * W_padded * sizeof(T) + 512; // 512B的padding，避免越界访问引起的性能问题

    int32_t offset = 0;
    int32_t xLocal_bytes_offset = offset;
    offset += xLocal_bytes;

    LocalTensor<T> xLocal = LocalTensor<T>(TPosition::VECIN, xLocal_bytes_offset, xLocal_bytes);
    SetFlag<HardEvent::MTE3_MTE2>(0);
    if constexpr (Reverse) {
        for (int32_t d = coreBlocks - 1; d >= 0; d--) {
            int32_t d_index = coreBaseBlockIndex + d;
            int32_t baseRowIndex = d_index * H;
            for (int32_t i = smallLoopTimes - 1; i >= 0; i--) {
                int32_t iterIndex = i;

                WaitFlag<HardEvent::MTE3_MTE2>(0);

                int32_t iterRows = (iterIndex == smallLoopTimes - 1) ? tailRows : n_rows_per_iter;
                // 行下标
                int32_t absoluteBeginRowIndex = baseRowIndex + iterIndex * n_rows_per_iter;

                DataCopyExtParams copyParams{static_cast<uint16_t>(iterRows), static_cast<uint32_t>(W * sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                DataCopyPad(xLocal, inputGm[absoluteBeginRowIndex * W], copyParams, padParams);

                SetFlag<HardEvent::MTE2_MTE3>(0);
                WaitFlag<HardEvent::MTE2_MTE3>(0);

                // 将结果写回全局内存
                // MTE3开始
                int32_t baseRowIndexInBlock = iterIndex * n_rows_per_iter;
                for (int32_t k = 0; k < iterRows; k++) {
                    int32_t rowIndex = baseRowIndexInBlock + k;
                    __gm__ uint8_t *outputAddr = GetTensorListDataPtr(output, rowIndex);
                    outputGm.SetGlobalBuffer((__gm__ T *)outputAddr, coreBlocks);
                    DataCopyExtParams storeParams{1, static_cast<uint32_t>(W * sizeof(T)), 0, 0, 0};
                    DataCopyPad(outputGm[d_index * W], xLocal[k * W_padded], storeParams); // 按列写回全局内存
                }
                SetFlag<HardEvent::MTE3_MTE2>(0);
            }
        }
    } else {
        for (int32_t d = 0; d < coreBlocks; d++) {
            int32_t d_index = coreBaseBlockIndex + d;
            int32_t baseRowIndex = d_index * H;
            for (int32_t i = 0; i < smallLoopTimes; i++) {
                int32_t iterIndex = i;

                WaitFlag<HardEvent::MTE3_MTE2>(0);

                int32_t iterRows = (iterIndex == smallLoopTimes - 1) ? tailRows : n_rows_per_iter;
                // 行下标
                int32_t absoluteBeginRowIndex = baseRowIndex + iterIndex * n_rows_per_iter;

                DataCopyExtParams copyParams{static_cast<uint16_t>(iterRows), static_cast<uint32_t>(W * sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                DataCopyPad(xLocal, inputGm[absoluteBeginRowIndex * W], copyParams, padParams);

                SetFlag<HardEvent::MTE2_MTE3>(0);
                WaitFlag<HardEvent::MTE2_MTE3>(0);

                // 将结果写回全局内存
                // MTE3开始
                int32_t baseRowIndexInBlock = iterIndex * n_rows_per_iter;
                for (int32_t k = 0; k < iterRows; k++) {
                    int32_t rowIndex = baseRowIndexInBlock + k;
                    __gm__ uint8_t *outputAddr = GetTensorListDataPtr(output, rowIndex);
                    outputGm.SetGlobalBuffer((__gm__ T *)outputAddr, coreBlocks);
                    DataCopyExtParams storeParams{1, static_cast<uint32_t>(W * sizeof(T)), 0, 0, 0};
                    DataCopyPad(outputGm[d_index * W], xLocal[k * W_padded], storeParams); // 按列写回全局内存
                }
                SetFlag<HardEvent::MTE3_MTE2>(0);
            }
        }
    }
    WaitFlag<HardEvent::MTE3_MTE2>(0);
}

template <typename T>
__aicore__ inline void unpack_3_big(GM_ADDR input, GM_ADDR output, int32_t coreBlocks, int32_t remainBlocks, int32_t H, int32_t W) {
    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;
    int32_t coreBaseRowIndex;

    if (GetBlockIdx() < remainBlocks) {
        coreBlocks += 1;
        coreBaseRowIndex = GetBlockIdx() * coreBlocks;
    } else {
        coreBaseRowIndex = coreBlocks * GetBlockIdx() + remainBlocks;
    }

    inputGm.SetGlobalBuffer((__gm__ T *)input, coreBlocks);
    // printf("output addr : %p, coreRows: %d\n", firstOutputAddr, coreRows);
    constexpr int32_t BufferNum = 1;
    constexpr int32_t MAX_UB_SIZE_KB = 190;

    int32_t local_num = 1; // xLocal
    int32_t max_space_size = MAX_UB_SIZE_KB / local_num / BufferNum * 1024;

    const int32_t max_elems_per_local = max_space_size / static_cast<int32_t>(sizeof(T));
    int32_t tileCols = max_elems_per_local;
    // 尽量让tileCols是32B对齐的元素数，避免中间tile写回时出现覆盖下一tile的风险
    tileCols = min(tileCols, W);

    int32_t xLocal_bytes = tileCols * sizeof(T);

    int32_t offset = 0;
    int32_t xLocal_bytes_offset = offset;
    offset += xLocal_bytes;

    LocalTensor<T> xLocal = LocalTensor<T>(TPosition::VECIN, xLocal_bytes_offset, xLocal_bytes);
    SetFlag<HardEvent::MTE3_MTE2>(0);
    for (int32_t d = 0; d < coreBlocks; d++) {
        int32_t d_index = coreBaseRowIndex + d;
        int32_t baseRowIndex = d_index * H;
        for (int32_t row = 0; row < H; row++) {
            __gm__ uint8_t *outputAddr = GetTensorListDataPtr(output, row);
            outputGm.SetGlobalBuffer((__gm__ T *)outputAddr, coreBlocks);
            for (int32_t col = 0; col < W; col += tileCols) {
                int32_t currentTileCols = min(tileCols, W - col);
                int32_t currentTileCols32B = ceil_aligned_32B<T>(currentTileCols);
                WaitFlag<HardEvent::MTE3_MTE2>(0);
                DataCopy(xLocal, inputGm[(baseRowIndex + row) * W + col], currentTileCols32B);

                SetFlag<HardEvent::MTE2_MTE3>(0);
                WaitFlag<HardEvent::MTE2_MTE3>(0);

                DataCopy(outputGm[d_index * W + col], xLocal, currentTileCols32B);
                SetFlag<HardEvent::MTE3_MTE2>(0);
            }
        }
    }
    WaitFlag<HardEvent::MTE3_MTE2>(0);
}