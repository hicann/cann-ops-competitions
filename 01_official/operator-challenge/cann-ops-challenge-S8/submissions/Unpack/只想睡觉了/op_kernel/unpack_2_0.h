#include "kernel_operator.h"
#include "kernel.h"
using namespace AscendC;

template <typename T>
__aicore__ inline void unpack_2_0_small(GM_ADDR input, GM_ADDR output, int32_t coreRows, int32_t remainRows, int32_t H, int32_t W) {
    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;
    int32_t coreBaseRowIndex;

    if (GetBlockIdx() < remainRows) {
        coreRows += 1;
        coreBaseRowIndex = GetBlockIdx() * coreRows;
    } else {
        coreBaseRowIndex = coreRows * GetBlockIdx() + remainRows;
    }
    __gm__ uint8_t *firstOutputAddr = GetTensorListDataPtr(output, 0);

    inputGm.SetGlobalBuffer((__gm__ T *)input, coreRows);
    outputGm.SetGlobalBuffer((__gm__ T *)firstOutputAddr, coreRows);
    // printf("output addr : %p, coreRows: %d\n", firstOutputAddr, coreRows);
    constexpr int32_t BufferNum = 1;
    constexpr int32_t MAX_UB_SIZE_KB = 180;

    int32_t local_num = 1 + 1; // xLocal, yLocal
    int32_t max_space_size = MAX_UB_SIZE_KB / local_num / BufferNum * 1024;
    int32_t W_padded = W;

    if (H != 1) {
        __gm__ uint8_t *secondOutputAddr = GetTensorListDataPtr(output, 1);
        W_padded = (secondOutputAddr - firstOutputAddr) / sizeof(T);
    }
    // printf("W: %d, W_padded: %d\n", W, W_padded);
    int32_t spaceSize = min((int32_t)(coreRows * W_padded * sizeof(T)), max_space_size);
    int32_t n_rows_per_iter = spaceSize / sizeof(T) / W_padded;
    int32_t smallLoopTimes = (coreRows + n_rows_per_iter - 1) / n_rows_per_iter;
    int32_t tailRows = coreRows - (smallLoopTimes - 1) * n_rows_per_iter;

    int32_t xLocal_bytes = n_rows_per_iter * W_padded * sizeof(T) + 512; // 512B的padding，避免越界访问引起的性能问题
    int32_t yLocal_bytes = n_rows_per_iter * W_padded * sizeof(T);
    int32_t indexLocal_bytes = W_padded * sizeof(uint32_t) + 256;

    int32_t offset = 0;
    int32_t xLocal_bytes_offset = offset;
    offset += xLocal_bytes;
    int32_t yLocal_bytes_offset = offset;
    offset += yLocal_bytes;
    int32_t indexLocal_bytes_offset = offset;
    offset += indexLocal_bytes;

    LocalTensor<T> xLocal[BufferNum] = {
        LocalTensor<T>(TPosition::VECIN, xLocal_bytes_offset, xLocal_bytes),
    };
    LocalTensor<T> yLocal[BufferNum] = {
        LocalTensor<T>(TPosition::VECOUT, yLocal_bytes_offset, yLocal_bytes),
    };
    LocalTensor<uint32_t> indexLocal[BufferNum] = {
        LocalTensor<uint32_t>(TPosition::VECCALC, indexLocal_bytes_offset, indexLocal_bytes),
    };

    if (W != W_padded) {
        for (int32_t i = 0; i < BufferNum; ++i) {
            MyCreateVecIndex(indexLocal[i].template ReinterpretCast<int32_t>(), 0, W_padded);
            Muls(indexLocal[i].template ReinterpretCast<int32_t>(), indexLocal[i].template ReinterpretCast<int32_t>(), static_cast<int32_t>(sizeof(T)), W_padded);
        }
    }

    for (int32_t i = 0; i < smallLoopTimes; i += BufferNum) {
        for (int32_t j = 0; j < BufferNum; ++j) {
            int32_t iterIndex = i + j;
            if (iterIndex >= smallLoopTimes) {
                break;
            }
            int32_t iterRows = (iterIndex == smallLoopTimes - 1) ? tailRows : n_rows_per_iter;
            // 行下标
            int32_t absoluteBeginRowIndex = coreBaseRowIndex + iterIndex * n_rows_per_iter;
            bool set_backward = (iterIndex + BufferNum < smallLoopTimes);
            bool wait_backward = (iterIndex >= BufferNum);

            if (wait_backward) { // 需要等到前序iter的计算完成，才能进行下一轮的MTE2
                WaitFlag<AscendC::HardEvent::V_MTE2>(j);
            }

            // inputGm不需要广播，可以直接读
            DataCopy(xLocal[j], inputGm[absoluteBeginRowIndex * W], ceil_aligned_32B<T>(iterRows * W_padded));
            SetFlag<HardEvent::MTE2_V>(j);
            WaitFlag<HardEvent::MTE2_V>(j);
            if (wait_backward) { // 需要等到前序iter的MTE3完成，才能进行下一轮的计算
                WaitFlag<AscendC::HardEvent::MTE3_V>(j);
            }

            // 计算开始
            if (W != W_padded) {
                MyPad(yLocal[j], xLocal[j], indexLocal[j], iterRows, W, W_padded);
            } else {
                DataCopy(yLocal[j], xLocal[j], iterRows * W_padded); // 直接复制，无需padding
            }
            if (set_backward) { // 通知下一轮iter的MTE2可以开始了
                SetFlag<AscendC::HardEvent::V_MTE2>(j);
            }
            SetFlag<HardEvent::V_MTE3>(j);
            WaitFlag<HardEvent::V_MTE3>(j);

            // 将结果写回全局内存
            // MTE3开始
            DataCopy(outputGm[absoluteBeginRowIndex * W_padded], yLocal[j], iterRows * W_padded); // 直接写回全局内存，避免使用DataCopyPad时的额外开销
            // MTE3结束
            if (set_backward) { // 通知下一轮iter的计算可以开始了
                SetFlag<AscendC::HardEvent::MTE3_V>(j);
            }
        }
    }
}
template <typename T>
__aicore__ inline void unpack_2_0_big(GM_ADDR input, GM_ADDR output, int32_t coreRows, int32_t remainRows, int32_t H, int32_t W) {
    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;
    int32_t coreBaseRowIndex;

    if (GetBlockIdx() < remainRows) {
        coreRows += 1;
        coreBaseRowIndex = GetBlockIdx() * coreRows;
    } else {
        coreBaseRowIndex = coreRows * GetBlockIdx() + remainRows;
    }

    inputGm.SetGlobalBuffer((__gm__ T *)input, coreRows);
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
    for (int32_t row = 0; row < coreRows; row++) {
        __gm__ uint8_t *outputAddr = GetTensorListDataPtr(output, coreBaseRowIndex + row);
        outputGm.SetGlobalBuffer((__gm__ T *)outputAddr, coreRows);
        for (int32_t col = 0; col < W; col += tileCols) {
            int32_t currentTileCols = min(tileCols, W - col);
            int32_t currentTileCols32B = ceil_aligned_32B<T>(currentTileCols);
            WaitFlag<HardEvent::MTE3_MTE2>(0);
            DataCopy(xLocal, inputGm[(coreBaseRowIndex + row) * W + col], currentTileCols32B);

            SetFlag<HardEvent::MTE2_MTE3>(0);
            WaitFlag<HardEvent::MTE2_MTE3>(0);

            DataCopy(outputGm[col], xLocal, currentTileCols32B);
            SetFlag<HardEvent::MTE3_MTE2>(0);
        }
    }
    WaitFlag<HardEvent::MTE3_MTE2>(0);
}