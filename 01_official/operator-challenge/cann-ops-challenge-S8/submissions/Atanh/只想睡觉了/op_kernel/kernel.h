#ifndef KERNEL_H
#define KERNEL_H

#include "kernel_operator.h"
using namespace AscendC;
using std::is_same_v;

template <class T, class... others>
inline constexpr bool is_one_of_v = (is_same_v<T, others> || ...);

constexpr int32_t BufferNum = 2;
constexpr uint32_t MAX_UB_SIZE_KB = 190;
__aicore__ inline bool needReverse(GM_ADDR workspace) {
    GlobalTensor<int32_t> indexGm;
    indexGm.SetGlobalBuffer((__gm__ int32_t *)workspace + GetBlockIdx() * 256, 1);
    indexGm(0)++;
    return (indexGm(0)) % 2 == 1;
}
// 获取不同数据类型下，相较于inputLocal临时变量所需空间的倍数
template <typename T>
__aicore__ inline constexpr uint32_t GetTmpLocalMultiple() {
    if constexpr (is_one_of_v<T, float, half, int32_t>) {
        return 0;
    } else if constexpr (is_one_of_v<T, bfloat16_t, int16_t, int8_t, uint8_t>) {
        return 2; // 需要一个与inputLocal同样大小的临时变量来存放转换后的数据
    } else {
        // 不支持其他类型
        return 0;
    }
}

#endif // KERNEL_H