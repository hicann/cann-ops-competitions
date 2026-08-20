#ifndef KERNEL_H
#define KERNEL_H

#include "kernel_operator.h"
using namespace AscendC;
using std::is_same_v;

template <class T, class... others>
constexpr bool is_one_of_v = (is_same_v<T, others> || ...);
__aicore__ inline bool needReverse(GM_ADDR workspace) {
    GlobalTensor<int32_t> indexGm;
    indexGm.SetGlobalBuffer((__gm__ int32_t *)workspace + GetBlockIdx() * 256, 1);
    indexGm(0)++;
    return (indexGm(0)) % 2 == 1;
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
    int32_t srcLocalBytesOffset = 0;
    int32_t srcLocalOffset = 0;
    int32_t dstLocalOffset = 0;
    int32_t N_bytes = N * sizeof(T);
    for (int32_t i = 0; i < M; ++i) {
        if ((srcLocalBytesOffset & 31) == 0) {
            DataCopy(dstLocal[dstLocalOffset], srcLocal[srcLocalOffset], N_32B_aligned);
        } else {
            Gather(dstLocal[dstLocalOffset], srcLocal, indexLocal, srcLocalBytesOffset, N_32B_aligned);
        }
        srcLocalBytesOffset += N_bytes;
        dstLocalOffset += N_Padded;
        srcLocalOffset += N;
    }
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