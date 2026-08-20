#ifndef KERNEL_H
#define KERNEL_H

#include "kernel_operator.h"
using namespace AscendC;
using std::is_same_v;

__aicore__ inline bool needReverse(GM_ADDR workspace) {
    GlobalTensor<int32_t> indexGm;
    indexGm.SetGlobalBuffer((__gm__ int32_t *)workspace + GetBlockIdx() * 256, 1);
    indexGm(0)++;
    return (indexGm(0) / 2) % 2 == 1;
}

template <class T, class... others>
constexpr bool is_one_of_v = (is_same_v<T, others> || ...);
constexpr static int32_t ONE_BLOCK_BYTES = 32;
constexpr static int32_t MAX_UB_SIZE_KB = 192;
constexpr static int32_t BufferNum = 1;
// count是元素个数，dstLocal需要>=ceil(count/8)*8*32B的空间
template <typename T>
__aicore__ inline void MyBrcb(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal, int32_t count) {
    int32_t repeat_num = (count + 7) / 8;
    // 只能用254次， 迭代255次有bug：srcLocal可能没有32B对齐，类型是2B，2*8*255%32！=0，不能满足32B对齐的要求
    constexpr int32_t MAX_REPEAT_TIMES = 254;
    int32_t loopTimes = repeat_num / MAX_REPEAT_TIMES;
    int32_t tail = repeat_num % MAX_REPEAT_TIMES;
    constexpr uint32_t dstStride = MAX_REPEAT_TIMES * 8 * 32 / sizeof(T);
    constexpr uint32_t srcStride = MAX_REPEAT_TIMES * 8 * sizeof(T);
    for (int32_t i = 0; i < loopTimes; ++i) {
        Brcb(dstLocal[i * dstStride], srcLocal[i * srcStride], MAX_REPEAT_TIMES, {1, 8});
    }
    if (tail > 0) {
        Brcb(dstLocal[loopTimes * dstStride], srcLocal[loopTimes * srcStride], tail, {1, 8});
    }
}
// 二元广播接口，srcLocalBC是需要广播的张量，srcLocalNBC是需要非广播的张量，OP是具体的计算操作
// srcLocalBC是[M,K*32B/sizeof(T))]的形状，srcLocalNBC和dstLocal是[M,N]的形状，OP需要支持批量广播的参数设置
// 要保证N是32B对齐
template <typename T, void (*func)(const LocalTensor<T> &, const LocalTensor<T> &, const LocalTensor<T> &, uint64_t, const uint8_t, const BinaryRepeatParams &)>
__aicore__ inline void MyBroadCastOp(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocalBC, const LocalTensor<T> &srcLocalNBC, int32_t M, int32_t N, uint8_t K = 1) {
    constexpr int32_t bytesPerElem = static_cast<int32_t>(sizeof(T));
    constexpr int32_t elemsPer256B = 256 / bytesPerElem;
    constexpr int32_t elemsPer32B = 32 / bytesPerElem;
    constexpr int32_t MAX_REPEAT_TIMES = 255;

    // 假设保证 N 是 32B 对齐的
    const int32_t repStride = N / elemsPer32B;

    // 如果 repStride 超过 255，那么 repeatParams 里的 uint8_t 会溢出，
    // 因此必须走同一行内连续处理的逻辑（Pattern A）。反之，走跨行复用的逻辑（Pattern B）。
    bool loopSameRow = (repStride > 255);

    if (loopSameRow) {// N*sizeof(T) > 255*32B
        // Pattern A: 单行内处理，避免跨行的 repStride 超出 8 bit (255) 限制
        for (int32_t r_idx = 0; r_idx < M; ++r_idx) {
            int32_t c = 0;
            while (c < N) {
                int32_t remain_c = N - c;
                if (remain_c >= elemsPer256B) {
                    uint8_t r = static_cast<uint8_t>(min((int32_t)MAX_REPEAT_TIMES, remain_c / elemsPer256B));
                    uint64_t mask = elemsPer256B;
                    // src0BlkStride=0 因为 srcLocalBC 每行只有 1 个 block 会被广播出 256B。
                    // src0RepStride=0 因为同一行还没换行
                    func(dstLocal[r_idx * N + c], srcLocalBC[r_idx * K * elemsPer32B], srcLocalNBC[r_idx * N + c], mask, r, {1, 0, 1, 8, 0, 8});
                    c += r * elemsPer256B;
                } else {
                    // remaining elements in the row (< 256B), process in 1 repeat
                    uint8_t r = 1;
                    uint64_t mask = remain_c;
                    func(dstLocal[r_idx * N + c], srcLocalBC[r_idx * K * elemsPer32B], srcLocalNBC[r_idx * N + c], mask, r, {1, 0, 1, 8, 0, 8});
                    c += remain_c;
                }
            }
        }
    } else {
        // Pattern B: 按列跨行复用单元素的逻辑，效率更高适合大 M 小 N 场景
        for (int32_t c = 0; c < N; c += elemsPer256B) {
            uint64_t mask = min(elemsPer256B, N - c);
            for (int32_t rowStart = 0; rowStart < M; rowStart += MAX_REPEAT_TIMES) {
                uint8_t r = static_cast<uint8_t>(min((int32_t)MAX_REPEAT_TIMES, M - rowStart));
                // 每次 repeat 向下跨越整行，srcLocalBC 在 repeat 间步进 1 个 block(32B)，目标步进 repStride
                func(dstLocal[rowStart * N + c], srcLocalBC[rowStart * K * elemsPer32B], srcLocalNBC[rowStart * N + c], mask, r,
                     {1, 0, 1, static_cast<uint8_t>(repStride), K, static_cast<uint8_t>(repStride)});
            }
        }
    }
}
// 二元广播接口，srcLocalBC是需要广播的张量，srcLocalNBC是需要非广播的张量，OP是具体的计算操作
// srcLocalBC是[1,N]的形状，srcLocalNBC和dstLocal是[M,N]的形状，OP需要支持批量广播的参数设置
// 要保证N是32B对齐
template <typename T, void (*func)(const LocalTensor<T> &, const LocalTensor<T> &, const LocalTensor<T> &, uint64_t, const uint8_t, const BinaryRepeatParams &)>
__aicore__ inline void MyBroadCastOp1xN(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocalBC, const LocalTensor<T> &srcLocalNBC, int32_t M, int32_t N) {
    constexpr int32_t bytesPerElem = static_cast<int32_t>(sizeof(T));
    constexpr int32_t elemsPer256B = 256 / bytesPerElem;
    constexpr int32_t elemsPer32B = 32 / bytesPerElem;
    constexpr int32_t MAX_REPEAT_TIMES = 255;

    // 假设保证 N 是 32B 对齐的
    const int32_t repStride = N / elemsPer32B;

    // 如果 repStride 超过 255，那么 repeatParams 里的 uint8_t 会溢出，
    // 因此必须走同一行内连续处理的逻辑（Pattern A）。反之，走跨行复用的逻辑（Pattern B）。
    bool loopSameRow = (repStride > 255);

    if (loopSameRow) {
        // Pattern A: 单行内处理，避免跨行的 repStride 超出 8 bit (255) 限制
        for (int32_t r_idx = 0; r_idx < M; ++r_idx) {
            int32_t c = 0;
            while (c < N) {
                int32_t remain_c = N - c;
                if (remain_c >= elemsPer256B) {
                    uint8_t r = static_cast<uint8_t>(min((int32_t)MAX_REPEAT_TIMES, remain_c / elemsPer256B));
                    uint64_t mask = elemsPer256B;
                    // srcLocalBC 一维向量和目标行内计算
                    func(dstLocal[r_idx * N + c], srcLocalBC[c], srcLocalNBC[r_idx * N + c], mask, r, {1, 1, 1, 8, 8, 8});
                    c += r * elemsPer256B;
                } else {
                    // remaining elements in the row (< 256B), process in 1 repeat
                    uint8_t r = 1;
                    uint64_t mask = remain_c;
                    func(dstLocal[r_idx * N + c], srcLocalBC[c], srcLocalNBC[r_idx * N + c], mask, r, {1, 1, 1, 8, 8, 8});
                    c += remain_c;
                }
            }
        }
    } else {
        // Pattern B: 按列跨行复用单元素的逻辑，效率更高适合大 M 小 N 场景
        for (int32_t c = 0; c < N; c += elemsPer256B) {
            uint64_t mask = min(elemsPer256B, N - c);
            for (int32_t rowStart = 0; rowStart < M; rowStart += MAX_REPEAT_TIMES) {
                uint8_t r = static_cast<uint8_t>(min((int32_t)MAX_REPEAT_TIMES, M - rowStart));
                // 每次 repeat 向下跨越整行，srcLocalBC 在 repeat 间步进 0 个 block，因为它是 [1,N] 的向量只复用这一行
                func(dstLocal[rowStart * N + c], srcLocalBC[c], srcLocalNBC[rowStart * N + c], mask, r, {1, 1, 1, static_cast<uint8_t>(repStride), 0, static_cast<uint8_t>(repStride)});
            }
        }
    }
}
__aicore__ inline __gm__ uint8_t *GetTensorListDataPtr(__gm__ void *data, uint32_t index) {
    __gm__ uint64_t *dataAddr = reinterpret_cast<__gm__ uint64_t *>(data);
    uint64_t dataPtrOffset = *dataAddr;
    __gm__ uint64_t *dataPtr_ = dataAddr + (dataPtrOffset >> 3) + index;
    return reinterpret_cast<__gm__ uint8_t *>(*(dataPtr_));
}

template <typename T>
__aicore__ inline int32_t ceil_aligned_512B(int32_t count) {
    return ((count * sizeof(T) + 511) / 512) * 512 / sizeof(T);
}

template <typename T>
__aicore__ inline int32_t ceil_aligned_32B(int32_t count) {
    return ((count * sizeof(T) + 31) / 32) * 32 / sizeof(T);
}

template <typename T>
__aicore__ inline void MyPad(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal, const LocalTensor<uint32_t> &indexLocal, int32_t M, int32_t N, int32_t N_Padded) {
    int32_t N_32B_aligned = ceil_aligned_32B<T>(N);
    for (int32_t i = 0; i < M; ++i) {
        int32_t srcLocalBytesOffset = i * N * sizeof(T);
        if ((srcLocalBytesOffset & 31) == 0) {
            DataCopy(dstLocal[i * N_Padded], srcLocal[i * N], N_32B_aligned);
        } else {
            Gather(dstLocal[i * N_Padded], srcLocal, indexLocal, srcLocalBytesOffset, N_32B_aligned);
        }
    }
}
template <typename T>
__aicore__ inline void MyUnPad(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal, int32_t M, int32_t N) {
    const int32_t N_32B_aligned = ceil_aligned_32B<T>(N);
    GatherMaskParams reducev2Params;
    reducev2Params.repeatTimes = static_cast<uint16_t>(M);
    reducev2Params.src0RepeatStride = static_cast<uint16_t>(N_32B_aligned * sizeof(T) / 32);
    uint64_t rsvdCnt = 0;
    GatherMask(dstLocal, srcLocal, 7, true, N, reducev2Params, rsvdCnt);
}
// 支持fp32和int32
template <typename T>
__aicore__ inline void MyCreateVecIndex(const LocalTensor<T> &src, T initVal, uint32_t count) {
    // 边界检查
    if (count == 0) return;

    // 第一阶段：直接设置前8个元素 [0,1,2,3,4,5,6,7]
    int32_t initCount = min((int32_t)count, 8);
    for (int32_t i = 0; i < initCount; i++) {
        src.SetValue(i, (T)(i + initVal));
    }

    // 如果元素数量不超过8，直接返回
    if (count <= 8) {
        return;
    }
    int32_t currentIndex = 8;
    // 第二阶段：使用向量操作批量生成 [8,15], [16,23], ..., [56,63]
    // 每次处理8个元素，最多处理7次（8到63，共56个元素）
    const int32_t vectorSize = 8;
    const int32_t maxVectorOps = 7; // 限制向量操作次数，避免超出范围

    for (int32_t batch = 0; batch < maxVectorOps && currentIndex < (int32_t)count; batch++) {
        int32_t elementsToProcess = min(vectorSize, (int32_t)count - currentIndex);
        Adds(src[currentIndex], src, (T)currentIndex, elementsToProcess);
        currentIndex += vectorSize;
    }

    // 第三阶段：处理剩余元素，使用更大的批次（64个元素）
    while (currentIndex < (int32_t)count) {
        int32_t elementsToProcess = min(64, (int32_t)count - currentIndex);
        Adds(src[currentIndex], src, (T)currentIndex, elementsToProcess);
        currentIndex += 64;
    }
}

// 支持fp32和half,要求H和W都是16的倍数
template <typename T>
__aicore__ inline void MyTranspose(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, int32_t H, int32_t W) {
    uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];

    uint32_t blockSize = ONE_BLK_SIZE / sizeof(T);
    uint32_t highBlock = H / BLOCK_CUBE;
    uint32_t stride = H;// 字节数是32B*H，因此stride等于H
    uint32_t repeat = W / blockSize;

    TransDataTo5HDParams transDataParams;
    transDataParams.repeatTimes = repeat;
    transDataParams.dstRepStride = transDataParams.repeatTimes > 1 ? stride : 0;
    transDataParams.srcRepStride = transDataParams.repeatTimes > 1 ? 1 : 0;
    for (int32_t i = 0; i < highBlock; i++) {
        if constexpr (sizeof(T) == sizeof(half)) {
            for (int32_t m = 0; m < NCHW_CONV_ADDR_LIST_SIZE; m++) {
                dstLocalList[m] = (uint64_t)dstTensor[i * BLOCK_CUBE + H * m].GetPhyAddr();
            }
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
                srcLocalList[n] = (uint64_t)srcTensor[i * W * BLOCK_CUBE + W * n].GetPhyAddr();
            }
            TransDataTo5HD<T>(dstLocalList, srcLocalList, transDataParams);
        } else if constexpr (sizeof(T) == sizeof(float)) {
            for (int32_t m = 0; m < NCHW_CONV_ADDR_LIST_SIZE; m = m + 2) {
                dstLocalList[m] = (uint64_t)dstTensor[i * BLOCK_CUBE + H * (m / 2)].GetPhyAddr();
                dstLocalList[m + 1] = (uint64_t)dstTensor[i * BLOCK_CUBE + H * (m / 2) + blockSize].GetPhyAddr();
            }
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
                srcLocalList[n] = (uint64_t)srcTensor[i * W * BLOCK_CUBE + W * n].GetPhyAddr();
            }
            TransDataTo5HD<T>(dstLocalList, srcLocalList, transDataParams);
        }
    }
}

#endif // KERNEL_H