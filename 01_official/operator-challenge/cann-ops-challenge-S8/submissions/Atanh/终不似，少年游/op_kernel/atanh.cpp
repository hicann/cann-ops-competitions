#include "kernel_operator.h"  // 引入 AscendC kernel 侧张量、搬运、向量指令与事件接口。

#include <type_traits>  // 引入 std::is_same_v，用于模板分派不同 dtype 的计算实现。

using namespace AscendC;  // 直接使用 AscendC 命名空间，简化 kernel 代码书写。

namespace {  // 匿名命名空间开始，限制辅助常量、工具函数和内部 kernel 类的可见性。

// 功能块说明：
// 这里定义 kernel 侧通用常量。
// 它们决定缓冲槽数、对齐粒度，以及 fp32/fp16 静态张量双缓冲路径中为规避 UB bank-group 冲突而引入的偏移量。
constexpr uint32_t kSingleBufferNum = 1;  // 单缓冲路径的缓冲槽数。
constexpr uint32_t kDoubleBufferNum = 2;  // 双缓冲路径的缓冲槽数。
constexpr uint32_t kAlignBytes = 32;  // UB 与 DataCopy 常用基础对齐粒度。
constexpr uint32_t kMtePreferredAlignBytes = 512;
constexpr uint32_t kBankConflictSkewBytes = 256;  // 将 Div 双输入错开 256B，降低 bank-group 冲突。
constexpr uint32_t kTilingKeyFp16 = 1;
constexpr uint32_t kTilingKeyFp32 = 2;
constexpr uint32_t kTilingKeyBf16 = 3;
constexpr uint32_t kTilingKeyInt32 = 4;
constexpr uint32_t kTilingKeyInt16 = 5;
constexpr uint32_t kTilingKeyUint8 = 6;
constexpr uint32_t kTilingKeyInt8 = 7;
constexpr uint32_t kTilingKeyFp16Db = 10;
constexpr uint32_t kTilingKeyBf16Db = 11;
constexpr uint32_t kTilingKeyFp32Db = 20;

#ifndef ATANH_PIPELINE_COPYOUT_COMPUTE_FIRST
#define ATANH_PIPELINE_COPYOUT_COMPUTE_FIRST 0
#endif
#ifndef ATANH_PIPELINE_COPYOUT_COMPUTE_FIRST_FP32
#define ATANH_PIPELINE_COPYOUT_COMPUTE_FIRST_FP32 1
#endif
#ifndef ATANH_PIPELINE_COPYOUT_COMPUTE_FIRST_FP16
#define ATANH_PIPELINE_COPYOUT_COMPUTE_FIRST_FP16 0
#endif
#ifndef ATANH_PIPELINE_COPYOUT_COMPUTE_FIRST_BF16
#define ATANH_PIPELINE_COPYOUT_COMPUTE_FIRST_BF16 1
#endif
#ifndef ATANH_SKIP_FINAL_VEC_BARRIER
#define ATANH_SKIP_FINAL_VEC_BARRIER 0
#endif
#ifndef ATANH_SKIP_FINAL_VEC_BARRIER_FP32
#define ATANH_SKIP_FINAL_VEC_BARRIER_FP32 ATANH_SKIP_FINAL_VEC_BARRIER
#endif
#ifndef ATANH_SKIP_FINAL_VEC_BARRIER_FP16
#define ATANH_SKIP_FINAL_VEC_BARRIER_FP16 ATANH_SKIP_FINAL_VEC_BARRIER
#endif
#if ATANH_PIPELINE_COPYOUT_COMPUTE_FIRST
#undef ATANH_PIPELINE_COPYOUT_COMPUTE_FIRST_FP32
#define ATANH_PIPELINE_COPYOUT_COMPUTE_FIRST_FP32 1
#undef ATANH_PIPELINE_COPYOUT_COMPUTE_FIRST_FP16
#define ATANH_PIPELINE_COPYOUT_COMPUTE_FIRST_FP16 1
#undef ATANH_PIPELINE_COPYOUT_COMPUTE_FIRST_BF16
#define ATANH_PIPELINE_COPYOUT_COMPUTE_FIRST_BF16 1
#endif

// 功能块说明：
// 将元素个数或字节数向上按指定对齐粒度补齐。
// 该函数主要用于计算 UB buffer 大小与静态张量地址布局。
__aicore__ inline uint32_t AlignUpCount(uint32_t value, uint32_t align)
{
    return ((value + align - 1) / align) * align;  // 先补齐再整除，得到不小于 value 的最小对齐值。
}

// 功能块说明：
// 判断相对 offset 与长度是否同时满足 fast copy 对齐要求。
// 若满足，则后续可直接走 DataCopy；否则需要走带 pad 的安全路径。
template <typename T>
__aicore__ inline bool CanUseFastCopy(uint32_t offset, uint32_t len, uint32_t copyAlignBytes)
{
    uint64_t byteOffset = static_cast<uint64_t>(offset) * static_cast<uint64_t>(sizeof(T));  // 将元素偏移转换为字节偏移。
    uint64_t byteLen = static_cast<uint64_t>(len) * static_cast<uint64_t>(sizeof(T));  // 将元素长度转换为字节长度。
    return copyAlignBytes != 0 && (byteOffset % copyAlignBytes == 0) && (byteLen % copyAlignBytes == 0);  // 偏移与长度都对齐时才允许 fast copy。
}

// 功能块说明：
// 判断绝对 GM 偏移是否满足 fast copy 条件。
// 这里会把 blockOffset 纳入判断，避免“tile 内偏移是 0，但全局偏移并未按 copyAlign 对齐”的误判。
template <typename T>
__aicore__ inline bool CanUseFastCopyAbs(uint32_t absoluteOffset, uint32_t len, uint32_t copyAlignBytes)
{
    return CanUseFastCopy<T>(absoluteOffset, len, copyAlignBytes);  // 直接复用相同逻辑，只是输入偏移含义变成绝对偏移。
}


template <typename T>
__aicore__ inline uint32_t GetMteAlignedPrefixElems(uint32_t offset, uint32_t len, uint32_t copyAlignBytes)
{
    if (copyAlignBytes < kMtePreferredAlignBytes) {
        return 0;
    }
    uint64_t byteOffset = static_cast<uint64_t>(offset) * static_cast<uint64_t>(sizeof(T));
    uint64_t byteLen = static_cast<uint64_t>(len) * static_cast<uint64_t>(sizeof(T));
    if ((byteOffset % copyAlignBytes) != 0 || byteLen <= copyAlignBytes || (byteLen % copyAlignBytes) == 0) {
        return 0;
    }
    uint64_t prefixBytes = (byteLen / copyAlignBytes) * copyAlignBytes;
    return static_cast<uint32_t>(prefixBytes / sizeof(T));
}

// 功能块说明：
// 输入搬运的安全兜底路径。
// 当对齐条件不满足时，使用 DataCopyPad 做有 padding 能力的 GM -> UB 搬运。
template <typename T>
__aicore__ inline void CopyInPad(GlobalTensor<T>& inputGm, uint32_t offset, LocalTensor<T>& inputLocal, uint32_t len)
{
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};  // 配置一维连续搬运参数。
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};  // 配置不额外启用显式 pad 值，仅走安全搬运路径。
    DataCopyPad(inputLocal, inputGm[offset], copyParams, padParams);  // 执行带 pad 的输入搬运。
}

// 功能块说明：
// 输出搬运的安全兜底路径。
// 当对齐条件不满足时，使用 DataCopyPad 做 UB -> GM 的安全写回。
template <typename T>
__aicore__ inline void CopyOutPad(GlobalTensor<T>& outputGm, uint32_t offset, LocalTensor<T>& outputLocal, uint32_t len)
{
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};  // 配置一维连续写回参数。
    DataCopyPad(outputGm[offset], outputLocal, copyParams);  // 执行带 pad 的输出搬运。
}

// 功能块说明：
// 输入搬运自动选择 fast copy 或 pad copy。
template <typename T>
__aicore__ inline void CopyInAuto(GlobalTensor<T>& inputGm, uint32_t offset, LocalTensor<T>& inputLocal, uint32_t len,
                                  uint32_t copyAlignBytes)
{
    if (CanUseFastCopy<T>(offset, len, copyAlignBytes)) {  // 若当前 tile 的相对偏移与长度都满足对齐。
        DataCopy(inputLocal, inputGm[offset], len);  // 直接走更轻量的 DataCopy。
    } else {  // 若任一条件不满足。
        uint32_t prefixElems = GetMteAlignedPrefixElems<T>(offset, len, copyAlignBytes);
        if (prefixElems > 0) {
            DataCopy(inputLocal, inputGm[offset], prefixElems);
            LocalTensor<T> inputTail = inputLocal[prefixElems];
            CopyInPad(inputGm, offset + prefixElems, inputTail, len - prefixElems);
            return;
        }
        CopyInPad(inputGm, offset, inputLocal, len);  // 走安全的 pad copy 路径。
    }
}

// 功能块说明：
// 输出搬运自动选择 fast copy 或 pad copy。
template <typename T>
__aicore__ inline void CopyOutAuto(GlobalTensor<T>& outputGm, uint32_t offset, LocalTensor<T>& outputLocal,
                                   uint32_t len, uint32_t copyAlignBytes)
{
    if (CanUseFastCopy<T>(offset, len, copyAlignBytes)) {  // 若当前 tile 的相对偏移与长度都满足对齐。
        DataCopy(outputGm[offset], outputLocal, len);  // 直接走更轻量的 DataCopy。
    } else {  // 若任一条件不满足。
        uint32_t prefixElems = GetMteAlignedPrefixElems<T>(offset, len, copyAlignBytes);
        if (prefixElems > 0) {
            DataCopy(outputGm[offset], outputLocal, prefixElems);
            LocalTensor<T> outputTail = outputLocal[prefixElems];
            CopyOutPad(outputGm, offset + prefixElems, outputTail, len - prefixElems);
            return;
        }
        CopyOutPad(outputGm, offset, outputLocal, len);  // 走安全的 pad copy 路径。
    }
}

// 功能块说明：
// 通用 masked atanh 计算链。
// 该路径主要作为非 float / non-half 类型或需要显式 counter mask 时的保底实现。
// 公式是：atanh(x) = 0.5 * (ln(1 + x) - ln(1 - x))。
template <typename T>
__aicore__ inline void ComputeAtanhMasked(LocalTensor<T>& outputLocal, LocalTensor<T>& inputLocal,
                                          LocalTensor<T>& tempLocal, uint32_t len)
{
    const UnaryRepeatParams unaryParams;  // 初始化单目向量指令参数。
    const BinaryRepeatParams binaryParams;  // 初始化双目向量指令参数。

    SetMaskCount();  // 将向量 mask 切换到按元素个数计数的模式。
    SetVectorMask<T, MaskMode::COUNTER>(0, len);  // 设置本次向量操作只覆盖前 len 个元素。

    Adds<T, false>(tempLocal, inputLocal, static_cast<T>(1.0f), MASK_PLACEHOLDER, 1, unaryParams);  // temp = x + 1。
    Muls<T, false>(outputLocal, inputLocal, static_cast<T>(-1.0f), MASK_PLACEHOLDER, 1, unaryParams);  // output = -x。
    Adds<T, false>(outputLocal, outputLocal, static_cast<T>(1.0f), MASK_PLACEHOLDER, 1, unaryParams);  // output = 1 - x。
    Ln<T, false>(tempLocal, tempLocal, MASK_PLACEHOLDER, 1, unaryParams);  // temp = ln(1 + x)。
    Ln<T, false>(outputLocal, outputLocal, MASK_PLACEHOLDER, 1, unaryParams);  // output = ln(1 - x)。
    Sub<T, false>(outputLocal, tempLocal, outputLocal, MASK_PLACEHOLDER, 1, binaryParams);  // output = ln(1 + x) - ln(1 - x)。
    Muls<T, false>(outputLocal, outputLocal, static_cast<T>(0.5f), MASK_PLACEHOLDER, 1, unaryParams);  // output *= 0.5。

    SetMaskNorm();  // 恢复默认 mask 模式。
    ResetMask();  // 清空当前 mask 状态，避免影响后续指令。
}

// 功能块说明：
// half 单缓冲路径的直接计算链。
// 这里把 1 + x 和 1 - x 分别放在 output/temp 中，再做 Div / Ln / Muls。
__aicore__ inline void ComputeAtanhHalf(LocalTensor<half>& outputLocal, LocalTensor<half>& inputLocal,
                                        LocalTensor<half>& tempLocal, uint32_t len)
{
    Adds(outputLocal, inputLocal, static_cast<half>(1.0f), len);  // output = 1 + x。
    PipeBarrier<PIPE_V>();  // 等待上一条向量指令写回完成。
    Muls(tempLocal, inputLocal, static_cast<half>(-1.0f), len);  // temp = -x。
    PipeBarrier<PIPE_V>();  // 保证 temp 可安全作为下游输入。
    Adds(tempLocal, tempLocal, static_cast<half>(1.0f), len);  // temp = 1 - x。
    PipeBarrier<PIPE_V>();  // 保证 Div 读到更新后的 temp。
    Div(outputLocal, outputLocal, tempLocal, len);  // output = (1 + x) / (1 - x)。
    PipeBarrier<PIPE_V>();  // 保证商结果已落地。
    Ln(outputLocal, outputLocal, len);  // output = ln((1 + x) / (1 - x))。
    PipeBarrier<PIPE_V>();  // 保证 ln 结果已落地。
    Muls(outputLocal, outputLocal, static_cast<half>(0.5f), len);  // output = 0.5 * ln((1 + x) / (1 - x))。
    PipeBarrier<PIPE_V>();  // 结束前显式同步向量流水。
}

// 功能块说明：
// float 单缓冲路径的直接计算链。
// 公式与 half 相同，但使用 float 向量指令。
__aicore__ inline void ComputeAtanhFloat(LocalTensor<float>& outputLocal, LocalTensor<float>& inputLocal,
                                         LocalTensor<float>& tempLocal, uint32_t len)
{
    Adds(tempLocal, inputLocal, 1.0f, len);  // temp = 1 + x。
    PipeBarrier<PIPE_V>();  // 保证 temp 更新完成。
    Muls(outputLocal, inputLocal, -1.0f, len);  // output = -x。
    PipeBarrier<PIPE_V>();  // 保证 output 可用于下一步。
    Adds(outputLocal, outputLocal, 1.0f, len);  // output = 1 - x。
    PipeBarrier<PIPE_V>();  // 保证 Div 看到正确分母。
    Div(outputLocal, tempLocal, outputLocal, len);  // output = (1 + x) / (1 - x)。
    PipeBarrier<PIPE_V>();  // 保证商结果落地。
    Ln(outputLocal, outputLocal, len);  // output = ln((1 + x) / (1 - x))。
    PipeBarrier<PIPE_V>();  // 保证 ln 结果落地。
    Muls(outputLocal, outputLocal, 0.5f, len);  // output = 0.5 * ln((1 + x) / (1 - x))。
    PipeBarrier<PIPE_V>();  // 在返回前同步流水。
}

// 功能块说明：
// float 单临时缓冲的原地计算链。
// 输入与输出复用同一块 dataLocal，可减少一个输出缓冲的额外占用。
__aicore__ inline void ComputeAtanhFloatInplace(LocalTensor<float>& dataLocal, LocalTensor<float>& tempLocal,
                                                uint32_t len)
{
    Adds(tempLocal, dataLocal, 1.0f, len);  // temp = 1 + x。
    PipeBarrier<PIPE_V>();  // 保证 temp 更新完成。
    Muls(dataLocal, dataLocal, -1.0f, len);  // data = -x。
    PipeBarrier<PIPE_V>();  // 保证 data 可用于后续。
    Adds(dataLocal, dataLocal, 1.0f, len);  // data = 1 - x。
    PipeBarrier<PIPE_V>();  // 保证 Div 读到正确分母。
    Div(dataLocal, tempLocal, dataLocal, len);  // data = (1 + x) / (1 - x)。
    PipeBarrier<PIPE_V>();  // 保证商结果落地。
    Ln(dataLocal, dataLocal, len);  // data = ln((1 + x) / (1 - x))。
    PipeBarrier<PIPE_V>();  // 保证 ln 结果落地。
    Muls(dataLocal, dataLocal, 0.5f, len);  // data = 0.5 * ln((1 + x) / (1 - x))。
    PipeBarrier<PIPE_V>();  // 在返回前同步流水。
}

// 功能块说明：
// float 双临时缓冲的原地计算链。
// plusLocal 保存 1 + x，minusLocal 保存 1 - x，能让 Div 的两个输入都来自独立缓冲，
// 结合地址错位布局可降低 UB bank-group 冲突。
__aicore__ inline void ComputeAtanhFloatInplaceDualScratch(LocalTensor<float>& dataLocal, LocalTensor<float>& plusLocal,
                                                           LocalTensor<float>& minusLocal, uint32_t len)
{
    Adds(plusLocal, dataLocal, 1.0f, len);  // plus = 1 + x。
    Muls(minusLocal, dataLocal, -1.0f, len);  // minus = -x。
    PipeBarrier<PIPE_V>();  // 同步前两条向量指令。
    Adds(minusLocal, minusLocal, 1.0f, len);  // minus = 1 - x。
    PipeBarrier<PIPE_V>();  // 保证 plus/minus 都可作为 Div 输入。
    Div(dataLocal, plusLocal, minusLocal, len);  // data = (1 + x) / (1 - x)。
    PipeBarrier<PIPE_V>();  // 保证商结果落地。
    Ln(dataLocal, dataLocal, len);  // data = ln((1 + x) / (1 - x))。
    PipeBarrier<PIPE_V>();  // 保证 ln 结果落地。
    Muls(dataLocal, dataLocal, 0.5f, len);  // data = 0.5 * ln((1 + x) / (1 - x))。
#if !ATANH_SKIP_FINAL_VEC_BARRIER_FP32
    PipeBarrier<PIPE_V>();  // 在返回前同步流水。
#endif
}

// 功能块说明：
// half 双临时缓冲的原地计算链。
// 它与 float 双临时缓冲版本同构，只是操作对象换成了 half。
__aicore__ inline void ComputeAtanhHalfInplaceDualScratch(LocalTensor<half>& dataLocal, LocalTensor<half>& plusLocal,
                                                          LocalTensor<half>& minusLocal, uint32_t len)
{
    Adds(plusLocal, dataLocal, static_cast<half>(1.0f), len);  // plus = 1 + x。
    Muls(minusLocal, dataLocal, static_cast<half>(-1.0f), len);  // minus = -x。
    PipeBarrier<PIPE_V>();  // 同步前两条向量指令。
    Adds(minusLocal, minusLocal, static_cast<half>(1.0f), len);  // minus = 1 - x。
    PipeBarrier<PIPE_V>();  // 保证 plus/minus 都可作为 Div 输入。
    Div(dataLocal, plusLocal, minusLocal, len);  // data = (1 + x) / (1 - x)。
    PipeBarrier<PIPE_V>();  // 保证商结果落地。
    Ln(dataLocal, dataLocal, len);  // data = ln((1 + x) / (1 - x))。
    PipeBarrier<PIPE_V>();  // 保证 ln 结果落地。
    Muls(dataLocal, dataLocal, static_cast<half>(0.5f), len);  // data = 0.5 * ln((1 + x) / (1 - x))。
#if !ATANH_SKIP_FINAL_VEC_BARRIER_FP16
    PipeBarrier<PIPE_V>();  // 在返回前同步流水。
#endif
}

// 功能块说明：
// 通用模板分派入口。
// float / half 走专门优化链，其余类型走 masked 保底链。
template <typename T>
__aicore__ inline void ComputeAtanh(LocalTensor<T>& outputLocal, LocalTensor<T>& inputLocal,
                                    LocalTensor<T>& tempLocal, uint32_t len)
{
    if constexpr (std::is_same_v<T, float>) {  // 若模板类型为 float。
        ComputeAtanhFloat(outputLocal, inputLocal, tempLocal, len);  // 走 float 专用实现。
    } else if constexpr (std::is_same_v<T, half>) {  // 若模板类型为 half。
        ComputeAtanhHalf(outputLocal, inputLocal, tempLocal, len);  // 走 half 专用实现。
    } else {  // 其余类型。
        ComputeAtanhMasked(outputLocal, inputLocal, tempLocal, len);  // 走 masked 保底实现。
    }
}

// 功能块说明：
// tile 级输入搬运封装。
// 若当前 block 的所有满 tile 都满足绝对对齐条件，则满 tile 直接走 DataCopy，
// 否则回退到逐 tile 的 auto copy。
template <typename T>
__aicore__ inline void CopyInTile(GlobalTensor<T>& inputGm, uint32_t offset, LocalTensor<T>& inputLocal, uint32_t len,
                                  uint32_t tileLength, uint32_t copyAlignBytes, bool wholeTileFastCopy)
{
    if (wholeTileFastCopy && len == tileLength) {  // 只有“整 tile 且全局对齐安全”时才走最短路径。
        DataCopy(inputLocal, inputGm[offset], len);  // 直接执行 GM -> UB 连续搬运。
        return;  // 整 tile fast path 完成后直接返回。
    }
    CopyInAuto(inputGm, offset, inputLocal, len, copyAlignBytes);  // 非满 tile 或不满足对齐时走自动选择路径。
}

// 功能块说明：
// tile 级输出搬运封装，与 CopyInTile 对称。
template <typename T>
__aicore__ inline void CopyOutTile(GlobalTensor<T>& outputGm, uint32_t offset, LocalTensor<T>& outputLocal,
                                   uint32_t len, uint32_t tileLength, uint32_t copyAlignBytes, bool wholeTileFastCopy)
{
    if (wholeTileFastCopy && len == tileLength) {  // 只有“整 tile 且全局对齐安全”时才走最短路径。
        DataCopy(outputGm[offset], outputLocal, len);  // 直接执行 UB -> GM 连续写回。
        return;  // 整 tile fast path 完成后直接返回。
    }
    CopyOutAuto(outputGm, offset, outputLocal, len, copyAlignBytes);  // 非满 tile 或不满足对齐时走自动选择路径。
}

// 功能块说明：
// 通用单缓冲 kernel 模板。
// 该实现适用于简单场景：串行执行 CopyIn -> Compute -> CopyOut，不做显式 tile overlap。
template <typename T>
class KernelAtanhSingle {
public:
    // 功能块说明：
    // 初始化单缓冲 kernel 的 GM 视图、tile 参数和 UB 缓冲。
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t blockOffset, uint32_t currentBlockLength,
                                uint32_t tileLength, uint32_t copyAlignBytes)
    {
        blockOffset_ = blockOffset;  // 保存当前 block 在整个张量中的起始元素偏移。
        blockLength_ = currentBlockLength;  // 保存当前 block 需要处理的元素数。
        tileLength_ = tileLength;  // 保存 host 下发的 tile 元素数。
        copyAlignBytes_ = copyAlignBytes;  // 保存 host 下发的 copy 对齐粒度。
        inputGm_.SetGlobalBuffer((__gm__ T*)input + blockOffset, currentBlockLength);  // 绑定当前 block 对应的输入 GM 区间。
        outputGm_.SetGlobalBuffer((__gm__ T*)output + blockOffset, currentBlockLength);  // 绑定当前 block 对应的输出 GM 区间。

        const uint32_t bufferSize = AlignUpCount(tileLength_ * sizeof(T), kAlignBytes);  // 计算单个 tile UB 缓冲所需字节数。
        wholeTileFastCopy_ = CanUseFastCopyAbs<T>(blockOffset_, tileLength_, copyAlignBytes_);  // 判断当前 block 的满 tile 是否都能安全走 fast copy。
        pipe_.InitBuffer(inQueue_, kSingleBufferNum, bufferSize);  // 初始化输入队列缓冲。
        pipe_.InitBuffer(outQueue_, kSingleBufferNum, bufferSize);  // 初始化输出队列缓冲。
        pipe_.InitBuffer(tempBuf_, bufferSize);  // 初始化临时计算缓冲。
    }

    // 功能块说明：
    // 逐 tile 串行处理整个 block。
    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < blockLength_; offset += tileLength_) {  // 以 tileLength 为步长扫描当前 block。
            uint32_t len = blockLength_ - offset;  // 先计算剩余元素数。
            if (len > tileLength_) {  // 若剩余元素数超过一个 tile。
                len = tileLength_;  // 则本轮按满 tile 处理。
            }
            ComputeTile(offset, len);  // 处理当前 tile。
        }
    }

private:
    // 功能块说明：
    // 单个 tile 的完整处理过程：搬入、计算、搬出。
    __aicore__ inline void ComputeTile(uint32_t offset, uint32_t len)
    {
        LocalTensor<T> inputLocalAlloc = inQueue_.AllocTensor<T>();  // 从输入队列申请一块 UB 输入缓冲。
        CopyInTile(inputGm_, offset, inputLocalAlloc, len, tileLength_, copyAlignBytes_, wholeTileFastCopy_);  // 将当前 tile 从 GM 搬入 UB。
        inQueue_.EnQue(inputLocalAlloc);  // 将输入缓冲入队，遵循 AscendC 队列使用约定。
        LocalTensor<T> inputLocal = inQueue_.DeQue<T>();  // 将输入缓冲出队，得到可参与计算的本地张量。

        LocalTensor<T> outputLocalAlloc = outQueue_.AllocTensor<T>();  // 从输出队列申请一块 UB 输出缓冲。
        LocalTensor<T> tempLocal = tempBuf_.Get<T>();  // 获取临时缓冲视图。
        ComputeAtanh(outputLocalAlloc, inputLocal, tempLocal, len);  // 执行 atanh 向量计算链。
        outQueue_.EnQue(outputLocalAlloc);  // 将输出缓冲入队。

        LocalTensor<T> outputLocal = outQueue_.DeQue<T>();  // 将输出缓冲出队。
        CopyOutTile(outputGm_, offset, outputLocal, len, tileLength_, copyAlignBytes_, wholeTileFastCopy_);  // 将结果写回 GM。

        outQueue_.FreeTensor(outputLocal);  // 释放输出缓冲。
        inQueue_.FreeTensor(inputLocal);  // 释放输入缓冲。
    }

private:
    TPipe pipe_;  // AscendC pipe，对队列与缓冲区进行生命周期管理。
    TQue<QuePosition::VECIN, kSingleBufferNum> inQueue_;  // 单缓冲输入队列。
    TQue<QuePosition::VECOUT, kSingleBufferNum> outQueue_;  // 单缓冲输出队列。
    TBuf<QuePosition::VECCALC> tempBuf_;  // 临时计算缓冲。
    GlobalTensor<T> inputGm_;  // 当前 block 的输入 GM 视图。
    GlobalTensor<T> outputGm_;  // 当前 block 的输出 GM 视图。
    uint32_t blockOffset_ = 0;  // 当前 block 在整个张量中的起始偏移。
    uint32_t blockLength_ = 0;  // 当前 block 的元素总数。
    uint32_t tileLength_ = 0;  // 每个 tile 的元素数。
    uint32_t copyAlignBytes_ = kAlignBytes;  // host 下发的 copy 对齐粒度。
    bool wholeTileFastCopy_ = false;  // 满 tile 是否可以稳定走 fast copy。
};

// 功能块说明：
// 通用双缓冲 kernel 模板。
// 该实现采用经典的 tileCount + 2 调度：前推 CopyIn，中间 Compute，后推 CopyOut。
template <typename T>
class KernelAtanhDouble {
public:
    // 功能块说明：
    // 初始化双缓冲 kernel 的 GM 视图、tile 参数和 UB 缓冲。
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t blockOffset, uint32_t currentBlockLength,
                                uint32_t tileLength, uint32_t copyAlignBytes)
    {
        blockOffset_ = blockOffset;  // 保存当前 block 的起始偏移。
        blockLength_ = currentBlockLength;  // 保存当前 block 的元素数。
        tileLength_ = tileLength;  // 保存 host 下发 tile 长度。
        copyAlignBytes_ = copyAlignBytes;  // 保存 host 下发 copy 对齐粒度。
        inputGm_.SetGlobalBuffer((__gm__ T*)input + blockOffset, currentBlockLength);  // 绑定输入 GM 视图。
        outputGm_.SetGlobalBuffer((__gm__ T*)output + blockOffset, currentBlockLength);  // 绑定输出 GM 视图。

        const uint32_t bufferSize = AlignUpCount(tileLength_ * sizeof(T), kAlignBytes);  // 计算每个 tile 缓冲的对齐字节数。
        wholeTileFastCopy_ = CanUseFastCopyAbs<T>(blockOffset_, tileLength_, copyAlignBytes_);  // 判断当前 block 的满 tile 是否都能 fast copy。
        pipe_.InitBuffer(inQueue_, kDoubleBufferNum, bufferSize);  // 初始化双缓冲输入队列。
        pipe_.InitBuffer(outQueue_, kDoubleBufferNum, bufferSize);  // 初始化双缓冲输出队列。
        pipe_.InitBuffer(tempBuf_, bufferSize);  // 初始化双缓冲共用临时缓冲。
    }

    // 功能块说明：
    // 执行双缓冲流水。
    // 调度关系是：
    // 第 i 轮 CopyIn(i)；
    // 第 i 轮 Compute(i-1)；
    // 第 i 轮 CopyOut(i-2)。
    __aicore__ inline void Process()
    {
        uint32_t tileCount = (blockLength_ + tileLength_ - 1) / tileLength_;  // 计算当前 block 共有多少个 tile。
        for (uint32_t tileIdx = 0; tileIdx < tileCount + 2; ++tileIdx) {  // 增加 2 轮用于排空流水。
            if (tileIdx < tileCount) {  // 只要还存在未搬入的 tile。
                CopyIn(tileIdx);  // 执行本轮搬入。
            }
            if (tileIdx > 0 && (tileIdx - 1) < tileCount) {  // 当存在已搬入、待计算的 tile。
                Compute(tileIdx - 1);  // 执行上一个 tile 的计算。
            }
            if (tileIdx > 1 && (tileIdx - 2) < tileCount) {  // 当存在已计算、待写回的 tile。
                CopyOut(tileIdx - 2);  // 执行上上个 tile 的写回。
            }
        }
    }

private:
    // 功能块说明：
    // 将 tile 序号转换为该 tile 在 block 内的起始偏移。
    __aicore__ inline uint32_t GetTileOffset(uint32_t tileIdx) const
    {
        return tileIdx * tileLength_;  // 每个 tile 的起点就是序号乘以标准 tileLength。
    }

    // 功能块说明：
    // 获取 tile 的实际长度。
    // 满 tile 返回 tileLength_，最后一个尾 tile 可能更短。
    __aicore__ inline uint32_t GetTileLength(uint32_t tileIdx) const
    {
        uint32_t offset = GetTileOffset(tileIdx);  // 先计算 tile 起始偏移。
        uint32_t len = blockLength_ - offset;  // 再计算从该偏移到 block 末尾还剩多少元素。
        if (len > tileLength_) {  // 若剩余元素大于标准 tileLength。
            len = tileLength_;  // 说明这是满 tile，长度裁成标准值。
        }
        return len;  // 返回实际 tile 长度。
    }

    // 功能块说明：
    // 执行一个 tile 的搬入阶段。
    __aicore__ inline void CopyIn(uint32_t tileIdx)
    {
        uint32_t offset = GetTileOffset(tileIdx);  // 计算 tile 起始偏移。
        uint32_t len = GetTileLength(tileIdx);  // 计算 tile 实际长度。
        LocalTensor<T> inputLocal = inQueue_.AllocTensor<T>();  // 从输入队列申请缓冲。
        CopyInTile(inputGm_, offset, inputLocal, len, tileLength_, copyAlignBytes_, wholeTileFastCopy_);  // 将 tile 从 GM 搬入 UB。
        inQueue_.EnQue(inputLocal);  // 将搬好的输入缓冲入队。
    }

    // 功能块说明：
    // 执行一个 tile 的计算阶段。
    __aicore__ inline void Compute(uint32_t tileIdx)
    {
        uint32_t len = GetTileLength(tileIdx);  // 读取该 tile 的有效长度。
        LocalTensor<T> inputLocal = inQueue_.DeQue<T>();  // 从输入队列取出一个已搬入 tile。
        LocalTensor<T> outputLocal = outQueue_.AllocTensor<T>();  // 从输出队列申请输出缓冲。
        LocalTensor<T> tempLocal = tempBuf_.Get<T>();  // 获取临时缓冲视图。
        ComputeAtanh(outputLocal, inputLocal, tempLocal, len);  // 执行 atanh 计算链。
        outQueue_.EnQue(outputLocal);  // 将结果缓冲入队。
        inQueue_.FreeTensor(inputLocal);  // 输入 tile 在计算完成后立即释放。
    }

    // 功能块说明：
    // 执行一个 tile 的写回阶段。
    __aicore__ inline void CopyOut(uint32_t tileIdx)
    {
        uint32_t offset = GetTileOffset(tileIdx);  // 计算 tile 起始偏移。
        uint32_t len = GetTileLength(tileIdx);  // 计算 tile 实际长度。
        LocalTensor<T> outputLocal = outQueue_.DeQue<T>();  // 从输出队列取出一个已完成计算的 tile。
        CopyOutTile(outputGm_, offset, outputLocal, len, tileLength_, copyAlignBytes_, wholeTileFastCopy_);  // 将结果写回 GM。
        outQueue_.FreeTensor(outputLocal);  // 写回完成后释放输出缓冲。
    }

private:
    TPipe pipe_;  // pipe 管理器。
    TQue<QuePosition::VECIN, kDoubleBufferNum> inQueue_;  // 双缓冲输入队列。
    TQue<QuePosition::VECOUT, kDoubleBufferNum> outQueue_;  // 双缓冲输出队列。
    TBuf<QuePosition::VECCALC> tempBuf_;  // 计算用临时缓冲。
    GlobalTensor<T> inputGm_;  // 输入 GM 视图。
    GlobalTensor<T> outputGm_;  // 输出 GM 视图。
    uint32_t blockOffset_ = 0;  // 当前 block 的全局起始偏移。
    uint32_t blockLength_ = 0;  // 当前 block 的长度。
    uint32_t tileLength_ = 0;  // 标准 tile 长度。
    uint32_t copyAlignBytes_ = kAlignBytes;  // copy 对齐粒度。
    bool wholeTileFastCopy_ = false;  // 满 tile 是否统一允许 fast copy。
};

// 功能块说明：
// 旧版 fp32 紧凑双缓冲实现。
// 它使用 TBuf + 动态事件 ID 管理 2 个数据槽和 1 个临时缓冲。
// 当前虽然不是主热路径，但仍保留在源码中作为历史可对比实现。
class KernelAtanhFp32CompactDouble {
public:
    // 功能块说明：
    // 初始化紧凑双缓冲 fp32 kernel 的 GM 视图、UB 缓冲和事件。
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t blockOffset, uint32_t currentBlockLength,
                                uint32_t tileLength, uint32_t copyAlignBytes)
    {
        blockOffset_ = blockOffset;  // 保存 block 全局偏移。
        blockLength_ = currentBlockLength;  // 保存 block 长度。
        tileLength_ = tileLength;  // 保存 tile 长度。
        copyAlignBytes_ = copyAlignBytes;  // 保存 copy 对齐粒度。
        inputGm_.SetGlobalBuffer((__gm__ float*)input + blockOffset, currentBlockLength);  // 绑定输入 GM 视图。
        outputGm_.SetGlobalBuffer((__gm__ float*)output + blockOffset, currentBlockLength);  // 绑定输出 GM 视图。

        const uint32_t bufferSize = AlignUpCount(tileLength_ * sizeof(float), kAlignBytes);  // 计算每个 float tile 的 UB 缓冲字节数。
        wholeTileFastCopy_ = CanUseFastCopyAbs<float>(blockOffset_, tileLength_, copyAlignBytes_);  // 判断满 tile 是否都能走 fast copy。
        pipe_.InitBuffer(dataBufs_[0], bufferSize);  // 初始化 slot0 数据缓冲。
        pipe_.InitBuffer(dataBufs_[1], bufferSize);  // 初始化 slot1 数据缓冲。
        pipe_.InitBuffer(tempBuf_, bufferSize);  // 初始化公用临时缓冲。
        for (uint32_t slot = 0; slot < kDoubleBufferNum; ++slot) {  // 为两个 slot 分别申请事件。
            mte2ToVecEvent_[slot] = pipe_.AllocEventID<HardEvent::MTE2_V>();  // MTE2 -> V 事件。
            vecToMte3Event_[slot] = pipe_.AllocEventID<HardEvent::V_MTE3>();  // V -> MTE3 事件。
            mte3ToMte2Event_[slot] = pipe_.AllocEventID<HardEvent::MTE3_MTE2>();  // MTE3 -> MTE2 复用事件。
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_[slot]);  // 初始时标记两个 slot 都是空闲可写的。
        }
    }

    // 功能块说明：
    // 执行紧凑双缓冲流水，覆盖预热、稳态和排空三个阶段。
    __aicore__ inline void Process()
    {
        uint32_t tileCount = (blockLength_ + tileLength_ - 1) / tileLength_;  // 计算 tile 总数。
        if (tileCount == 0) {  // 空 block 直接结束。
            ReleaseEvents();  // 释放已申请的事件。
            return;  // 返回。
        }

        CopyIn(0);  // 先搬入第 0 个 tile，完成预热。
        if (tileCount == 1) {  // 若只有 1 个 tile。
            Compute(0);  // 直接计算第 0 个 tile。
            CopyOut(0);  // 直接写回第 0 个 tile。
            WaitAllSlots();  // 等待 slot 完全回到可复用状态。
            ReleaseEvents();  // 释放事件。
            return;  // 返回。
        }

        for (uint32_t tileIdx = 1; tileIdx < tileCount; ++tileIdx) {  // 从第 1 个 tile 开始进入稳态流水。
            if (tileIdx > 1) {  // 当流水已推进到可写回阶段。
                CopyOut(tileIdx - 2);  // 写回两轮之前完成计算的 tile。
            }
#if ATANH_PIPELINE_COPYOUT_COMPUTE_FIRST_FP32
            if (tileIdx > 1) {
            Compute(tileIdx - 1);  // 计算上一个 tile。
            CopyIn(tileIdx);  // 搬入当前 tile。
            } else {
            CopyIn(tileIdx);  // 搬入当前 tile。
            Compute(tileIdx - 1);  // 计算上一个 tile。
            }
#else
            CopyIn(tileIdx);  // 搬入当前 tile。
            Compute(tileIdx - 1);  // 计算上一个 tile。
#endif
        }

        CopyOut(tileCount - 2);  // 写回倒数第二个 tile。
        Compute(tileCount - 1);  // 计算最后一个 tile。
        CopyOut(tileCount - 1);  // 写回最后一个 tile。
        WaitAllSlots();  // 等待所有 slot 完全排空。
        ReleaseEvents();  // 释放事件 ID。
    }

    // 功能块说明：
    // 等待所有 slot 回到“可再次用于 MTE2 搬入”的状态。
    __aicore__ inline void WaitAllSlots()
    {
        for (uint32_t slot = 0; slot < kDoubleBufferNum; ++slot) {  // 遍历两个 slot。
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_[slot]);  // 等待该 slot 的写回完成。
            pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(mte3ToMte2Event_[slot]);  // 释放该 slot 对应的复用事件 ID。
        }
    }

    // 功能块说明：
    // 根据 tile 序号计算其在 block 内的起始偏移。
    __aicore__ inline uint32_t GetTileOffset(uint32_t tileIdx) const
    {
        return tileIdx * tileLength_;  // tile 起点等于序号乘以 tileLength。
    }

    // 功能块说明：
    // 获取 tile 实际有效长度。
    __aicore__ inline uint32_t GetTileLength(uint32_t tileIdx) const
    {
        uint32_t offset = GetTileOffset(tileIdx);  // 先取 tile 起始偏移。
        uint32_t len = blockLength_ - offset;  // 计算剩余长度。
        if (len > tileLength_) {  // 若是满 tile。
            len = tileLength_;  // 裁成标准 tileLength。
        }
        return len;  // 返回有效长度。
    }

    // 功能块说明：
    // 将一个 tile 搬入指定 slot。
    __aicore__ inline void CopyIn(uint32_t tileIdx)
    {
        uint32_t slot = tileIdx & 1U;  // 双缓冲下用 tileIdx 的最低位选槽。
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_[slot]);  // 等待该 slot 确认已从上一轮写回释放。
        pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(mte3ToMte2Event_[slot]);  // 释放上一次写回对应的事件 ID。
        uint32_t offset = GetTileOffset(tileIdx);  // 计算 tile 起始偏移。
        uint32_t len = GetTileLength(tileIdx);  // 计算 tile 实际长度。
        LocalTensor<float> dataLocal = dataBufs_[slot].Get<float>();  // 取出该 slot 的本地数据缓冲。
        CopyInTile(inputGm_, offset, dataLocal, len, tileLength_, copyAlignBytes_, wholeTileFastCopy_);  // 执行 GM -> UB 搬运。
        SetFlag<HardEvent::MTE2_V>(mte2ToVecEvent_[slot]);  // 通知向量流水：该 slot 的输入数据已可计算。
    }

    // 功能块说明：
    // 在指定 slot 上执行一个 tile 的 fp32 原地计算。
    __aicore__ inline void Compute(uint32_t tileIdx)
    {
        uint32_t slot = tileIdx & 1U;  // 根据 tile 序号选择对应 slot。
        WaitFlag<HardEvent::MTE2_V>(mte2ToVecEvent_[slot]);  // 等待搬入完成。
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(mte2ToVecEvent_[slot]);  // 释放搬入阶段的事件 ID。
        uint32_t len = GetTileLength(tileIdx);  // 获取该 tile 的有效长度。
        LocalTensor<float> dataLocal = dataBufs_[slot].Get<float>();  // 获取该 slot 的数据张量。
        LocalTensor<float> tempLocal = tempBuf_.Get<float>();  // 获取临时缓冲。
        ComputeAtanhFloatInplace(dataLocal, tempLocal, len);  // 执行原地 atanh 计算链。
        SetFlag<HardEvent::V_MTE3>(vecToMte3Event_[slot]);  // 通知写回流水：该 slot 的结果已可写回。
    }

    // 功能块说明：
    // 将指定 slot 的结果 tile 写回 GM。
    __aicore__ inline void CopyOut(uint32_t tileIdx)
    {
        uint32_t slot = tileIdx & 1U;  // 根据 tile 序号选择对应 slot。
        WaitFlag<HardEvent::V_MTE3>(vecToMte3Event_[slot]);  // 等待计算完成。
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(vecToMte3Event_[slot]);  // 释放计算阶段的事件 ID。
        uint32_t offset = GetTileOffset(tileIdx);  // 获取 tile 起始偏移。
        uint32_t len = GetTileLength(tileIdx);  // 获取 tile 有效长度。
        LocalTensor<float> dataLocal = dataBufs_[slot].Get<float>();  // 取出该 slot 的数据缓冲。
        CopyOutTile(outputGm_, offset, dataLocal, len, tileLength_, copyAlignBytes_, wholeTileFastCopy_);  // 执行 UB -> GM 写回。
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_[slot]);  // 通知搬入流水：该 slot 已重新空闲，可再次复用。
    }

    // 功能块说明：
    // 释放本类申请的所有事件 ID。
    __aicore__ inline void ReleaseEvents()
    {
        for (uint32_t slot = 0; slot < kDoubleBufferNum; ++slot) {  // 遍历两个 slot。
            pipe_.ReleaseEventID<HardEvent::MTE2_V>(mte2ToVecEvent_[slot]);  // 释放 MTE2->V 事件。
            pipe_.ReleaseEventID<HardEvent::V_MTE3>(vecToMte3Event_[slot]);  // 释放 V->MTE3 事件。
            pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(mte3ToMte2Event_[slot]);  // 释放 MTE3->MTE2 事件。
        }
    }

private:
    TPipe pipe_;  // pipe 管理器。
    TBuf<QuePosition::VECCALC> dataBufs_[kDoubleBufferNum];  // 两个 slot 的数据缓冲。
    TBuf<QuePosition::VECCALC> tempBuf_;  // 一个公用临时缓冲。
    GlobalTensor<float> inputGm_;  // 输入 GM 视图。
    GlobalTensor<float> outputGm_;  // 输出 GM 视图。
    uint32_t blockOffset_ = 0;  // block 全局偏移。
    uint32_t blockLength_ = 0;  // block 长度。
    uint32_t tileLength_ = 0;  // tile 长度。
    uint32_t copyAlignBytes_ = kAlignBytes;  // copy 对齐粒度。
    bool wholeTileFastCopy_ = false;  // 满 tile 是否都允许走 fast copy。
    TEventID mte2ToVecEvent_[kDoubleBufferNum];  // 搬入完成事件。
    TEventID vecToMte3Event_[kDoubleBufferNum];  // 计算完成事件。
    TEventID mte3ToMte2Event_[kDoubleBufferNum];  // 写回完成、slot 复用事件。
};

// 功能块说明：
// 当前 fp32 主热路径：静态张量双缓冲实现。
// 它不再通过 TPipe/TBuf 队列动态管理数据槽，而是手动规划 UB 地址，
// 用 EVENT_ID0 / EVENT_ID1 显式管理两条 tile 流水。
// 这样可以显著降低 scalar/control 开销，是当前大 shape、2D、fp32 场景的主优化方向。
class KernelAtanhFp32StaticDouble {
public:
    // 功能块说明：
    // 初始化 fp32 静态张量双缓冲 kernel。
    // 这里会手动规划两份 dataLocal 与两份 scratch 的 UB 地址布局。
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t blockOffset, uint32_t currentBlockLength,
                                uint32_t tileLength, uint32_t copyAlignBytes)
    {
        blockOffset_ = blockOffset;  // 保存 block 全局起始偏移。
        blockLength_ = currentBlockLength;  // 保存当前 block 长度。
        tileLength_ = tileLength;  // 保存 tile 长度。
        copyAlignBytes_ = copyAlignBytes;  // 保存 copy 对齐粒度。
        wholeTileFastCopy_ = CanUseFastCopyAbs<float>(blockOffset_, tileLength_, copyAlignBytes_);  // 判断满 tile 是否都能走 fast copy。
        bufferSizeBytes_ = AlignUpCount(tileLength_ * sizeof(float), kAlignBytes);  // 计算每个 float tile 的对齐缓冲字节数。
        dataAddr_[0] = 0;  // slot0 数据缓冲起始地址放在 VECCALC 0 偏移处。
        dataAddr_[1] = bufferSizeBytes_;  // slot1 数据缓冲紧接 slot0 之后。
        // Place the two Div inputs on different UB bank-group offsets.
        tempAddr_ = bufferSizeBytes_ * kDoubleBufferNum + kBankConflictSkewBytes;  // 第一个 scratch 向后错开 256B，规避 bank-group 冲突。
        denomAddr_ = bufferSizeBytes_ * (kDoubleBufferNum + 1U) + (kBankConflictSkewBytes * 2U);  // 第二个 scratch 再继续错开，避免与 temp 重叠且进一步错位。

        inputGm_.SetGlobalBuffer((__gm__ float*)input + blockOffset, currentBlockLength);  // 绑定当前 block 的输入 GM 视图。
        outputGm_.SetGlobalBuffer((__gm__ float*)output + blockOffset, currentBlockLength);  // 绑定当前 block 的输出 GM 视图。

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // 初始时标记 slot0 为空闲可搬入。
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);  // 初始时标记 slot1 为空闲可搬入。
    }

    // 功能块说明：
    // 执行 fp32 静态张量双缓冲流水。
    // 针对 tileCount == 3 的热场景额外提供专门展开版本，减少循环与分支控制开销。
    __aicore__ inline void Process()
    {
        uint32_t tileCount = (blockLength_ + tileLength_ - 1) / tileLength_;  // 计算当前 block 的 tile 数量。
        if (tileCount == 0) {  // 若当前 block 没有有效 tile。
            return;  // 直接返回。
        }
        if (tileCount == 3) {  // 若刚好是热场景 3 tile。
            ProcessThreeTiles();  // 走专门展开的三 tile 流水实现。
            return;  // 处理完成后返回。
        }

        CopyIn(0);  // 预热阶段先搬入第 0 个 tile。
        if (tileCount == 1) {  // 若只有 1 个 tile。
            Compute(0);  // 直接计算第 0 个 tile。
            CopyOut(0);  // 直接写回第 0 个 tile。
            WaitAllSlots();  // 等待两个 slot 的复用标记都回到空闲态。
            return;  // 返回。
        }

        for (uint32_t tileIdx = 1; tileIdx < tileCount; ++tileIdx) {  // 从第 1 个 tile 开始推进稳态流水。
            if (tileIdx > 1) {  // 当已经有 tile 完成计算并可写回。
                CopyOut(tileIdx - 2);  // 写回两轮之前的 tile。
            }
#if ATANH_PIPELINE_COPYOUT_COMPUTE_FIRST_FP32
            if (tileIdx > 1) {
            Compute(tileIdx - 1);  // 计算上一个 tile。
            CopyIn(tileIdx);  // 搬入当前 tile。
            } else {
            CopyIn(tileIdx);  // 搬入当前 tile。
            Compute(tileIdx - 1);  // 计算上一个 tile。
            }
#else
            CopyIn(tileIdx);  // 搬入当前 tile。
            Compute(tileIdx - 1);  // 计算上一个 tile。
#endif
        }

        CopyOut(tileCount - 2);  // 写回倒数第二个 tile。
        Compute(tileCount - 1);  // 计算最后一个 tile。
        CopyOut(tileCount - 1);  // 写回最后一个 tile。
        WaitAllSlots();  // 等待所有 slot 最终回到可复用状态。
    }

private:
    // 功能块说明：
    // 将 slot 索引映射到静态事件 ID。
    __aicore__ inline int32_t GetEventId(uint32_t slot) const
    {
        return slot == 0 ? EVENT_ID0 : EVENT_ID1;  // slot0 用 EVENT_ID0，slot1 用 EVENT_ID1。
    }

    // 功能块说明：
    // 根据 slot 获取对应的数据张量视图。
    __aicore__ inline LocalTensor<float> GetDataTensor(uint32_t slot) const
    {
        return LocalTensor<float>(TPosition::VECCALC, dataAddr_[slot], tileLength_);  // 以预先规划好的地址创建 slot 数据视图。
    }

    // 功能块说明：
    // 获取“1 + x”这一路 scratch 的张量视图。
    __aicore__ inline LocalTensor<float> GetTempTensor() const
    {
        return LocalTensor<float>(TPosition::VECCALC, tempAddr_, tileLength_);  // 返回 temp scratch 视图。
    }

    // 功能块说明：
    // 获取“1 - x”这一路 scratch 的张量视图。
    __aicore__ inline LocalTensor<float> GetDenomTensor() const
    {
        return LocalTensor<float>(TPosition::VECCALC, denomAddr_, tileLength_);  // 返回 denom scratch 视图。
    }

    // 功能块说明：
    // 等待两个 slot 都完成最后一次写回并进入可复用状态。
    __aicore__ inline void WaitAllSlots()
    {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // 等待 slot0 完成写回。
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);  // 等待 slot1 完成写回。
    }

    // 功能块说明：
    // 三 tile 专门展开版本。
    // 该路径针对常见热场景显式排布：
    // tile0 搬入 -> tile1 搬入 -> tile0 计算/写回 -> tile2 搬入 -> tile1 计算/写回 -> tile2 计算/写回。
    // 这样能减少一般循环中的索引判断和控制开销。
    __aicore__ inline void ProcessThreeTiles()
    {
        uint32_t offset0 = 0;  // tile0 起始偏移固定为 0。
        uint32_t len0 = tileLength_;  // tile0 视为满 tile。
        uint32_t offset1 = tileLength_;  // tile1 起始偏移为 1 个 tile。
        uint32_t len1 = tileLength_;  // tile1 视为满 tile。
        uint32_t offset2 = tileLength_ * 2;  // tile2 起始偏移为 2 个 tile。
        uint32_t len2 = blockLength_ > offset2 ? (blockLength_ - offset2) : 0;  // tile2 可能是尾 tile，因此按真实剩余长度计算。
        LocalTensor<float> tempLocal = GetTempTensor();  // 获取 plus scratch 视图。
        LocalTensor<float> denomLocal = GetDenomTensor();  // 获取 minus scratch 视图。
        LocalTensor<float> data0 = GetDataTensor(0);  // 获取 slot0 数据视图。
        LocalTensor<float> data1 = GetDataTensor(1);  // 获取 slot1 数据视图。

        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // 等待 slot0 空闲。
        CopyInTile(inputGm_, offset0, data0, len0, tileLength_, copyAlignBytes_, wholeTileFastCopy_);  // 搬入 tile0 到 slot0。
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);  // 标记 slot0 可计算。

        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);  // 等待 slot1 空闲。
        CopyInTile(inputGm_, offset1, data1, len1, tileLength_, copyAlignBytes_, wholeTileFastCopy_);  // 搬入 tile1 到 slot1。
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);  // 标记 slot1 可计算。

        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);  // 等待 tile0 搬入完成。
        ComputeAtanhFloatInplaceDualScratch(data0, tempLocal, denomLocal, len0);  // 在 slot0 上计算 tile0。
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);  // 标记 tile0 可写回。

        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);  // 等待 tile0 计算完成。
        CopyOutTile(outputGm_, offset0, data0, len0, tileLength_, copyAlignBytes_, wholeTileFastCopy_);  // 写回 tile0。
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // 标记 slot0 再次空闲。

        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // 等待 slot0 确认可复用。
        CopyInTile(inputGm_, offset2, data0, len2, tileLength_, copyAlignBytes_, wholeTileFastCopy_);  // 搬入 tile2 到 slot0。
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);  // 标记 tile2 可计算。

        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);  // 等待 tile1 搬入完成。
        ComputeAtanhFloatInplaceDualScratch(data1, tempLocal, denomLocal, len1);  // 在 slot1 上计算 tile1。
        SetFlag<HardEvent::V_MTE3>(EVENT_ID1);  // 标记 tile1 可写回。

        WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);  // 等待 tile1 计算完成。
        CopyOutTile(outputGm_, offset1, data1, len1, tileLength_, copyAlignBytes_, wholeTileFastCopy_);  // 写回 tile1。
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);  // 标记 slot1 再次空闲。

        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);  // 等待 tile2 搬入完成。
        ComputeAtanhFloatInplaceDualScratch(data0, tempLocal, denomLocal, len2);  // 在 slot0 上计算 tile2。
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);  // 标记 tile2 可写回。

        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);  // 等待 tile2 计算完成。
        CopyOutTile(outputGm_, offset2, data0, len2, tileLength_, copyAlignBytes_, wholeTileFastCopy_);  // 写回 tile2。
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // 标记 slot0 再次空闲。

        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // 等待 slot0 最终空闲。
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);  // 等待 slot1 最终空闲。
    }

    // 功能块说明：
    // 根据 tile 序号计算其在 block 内的起始偏移。
    __aicore__ inline uint32_t GetTileOffset(uint32_t tileIdx) const
    {
        return tileIdx * tileLength_;  // tile 起点 = tileIdx * tileLength。
    }

    // 功能块说明：
    // 获取 tile 的有效长度，尾 tile 可能短于标准长度。
    __aicore__ inline uint32_t GetTileLength(uint32_t tileIdx) const
    {
        uint32_t offset = GetTileOffset(tileIdx);  // 先算起始偏移。
        uint32_t len = blockLength_ - offset;  // 再算剩余长度。
        if (len > tileLength_) {  // 若为满 tile。
            len = tileLength_;  // 裁成标准 tile 长度。
        }
        return len;  // 返回最终长度。
    }

    // 功能块说明：
    // 通用 CopyIn 包装，内部根据 tileIdx 选择双缓冲槽。
    __aicore__ inline void CopyIn(uint32_t tileIdx)
    {
        uint32_t offset = GetTileOffset(tileIdx);  // 计算 tile 起始偏移。
        uint32_t len = GetTileLength(tileIdx);  // 计算 tile 有效长度。
        CopyInSlot(tileIdx & 1U, offset, len);  // 使用 tileIdx 最低位选择 slot。
    }

    // 功能块说明：
    // 通用 Compute 包装，内部根据 tileIdx 选择双缓冲槽。
    __aicore__ inline void Compute(uint32_t tileIdx)
    {
        uint32_t len = GetTileLength(tileIdx);  // 读取 tile 有效长度。
        ComputeSlot(tileIdx & 1U, len);  // 在对应 slot 上执行计算。
    }

    // 功能块说明：
    // 通用 CopyOut 包装，内部根据 tileIdx 选择双缓冲槽。
    __aicore__ inline void CopyOut(uint32_t tileIdx)
    {
        uint32_t offset = GetTileOffset(tileIdx);  // 计算 tile 起始偏移。
        uint32_t len = GetTileLength(tileIdx);  // 计算 tile 有效长度。
        CopyOutSlot(tileIdx & 1U, offset, len);  // 在对应 slot 上执行写回。
    }

    // 功能块说明：
    // 在指定 slot 上执行搬入。
    __aicore__ inline void CopyInSlot(uint32_t slot, uint32_t offset, uint32_t len)
    {
        int32_t eventId = GetEventId(slot);  // 将 slot 映射到静态事件 ID。
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);  // 等待该 slot 已从上一轮写回阶段回收完毕。
        LocalTensor<float> dataLocal = GetDataTensor(slot);  // 获取该 slot 的数据视图。
        CopyInTile(inputGm_, offset, dataLocal, len, tileLength_, copyAlignBytes_, wholeTileFastCopy_);  // 执行 GM -> UB 搬运。
        SetFlag<HardEvent::MTE2_V>(eventId);  // 通知计算阶段：数据已就绪。
    }

    // 功能块说明：
    // 在指定 slot 上执行计算。
    __aicore__ inline void ComputeSlot(uint32_t slot, uint32_t len)
    {
        int32_t eventId = GetEventId(slot);  // 获取对应事件 ID。
        WaitFlag<HardEvent::MTE2_V>(eventId);  // 等待搬入阶段完成。
        LocalTensor<float> dataLocal = GetDataTensor(slot);  // 获取数据视图。
        LocalTensor<float> tempLocal = GetTempTensor();  // 获取 plus scratch。
        LocalTensor<float> denomLocal = GetDenomTensor();  // 获取 minus scratch。
        ComputeAtanhFloatInplaceDualScratch(dataLocal, tempLocal, denomLocal, len);  // 执行双 scratch 原地 atanh 计算。
        SetFlag<HardEvent::V_MTE3>(eventId);  // 通知写回阶段：结果已就绪。
    }

    // 功能块说明：
    // 在指定 slot 上执行写回。
    __aicore__ inline void CopyOutSlot(uint32_t slot, uint32_t offset, uint32_t len)
    {
        int32_t eventId = GetEventId(slot);  // 获取对应事件 ID。
        WaitFlag<HardEvent::V_MTE3>(eventId);  // 等待计算阶段完成。
        LocalTensor<float> dataLocal = GetDataTensor(slot);  // 获取数据视图。
        CopyOutTile(outputGm_, offset, dataLocal, len, tileLength_, copyAlignBytes_, wholeTileFastCopy_);  // 执行 UB -> GM 写回。
        SetFlag<HardEvent::MTE3_MTE2>(eventId);  // 通知搬入阶段：该 slot 已空闲可复用。
    }

private:
    TPipe pipe_;  // pipe 管理器，静态张量视图虽然不显式申请队列，但仍依赖运行时上下文。
    GlobalTensor<float> inputGm_;  // 输入 GM 视图。
    GlobalTensor<float> outputGm_;  // 输出 GM 视图。
    uint32_t blockOffset_ = 0;  // 当前 block 的全局偏移。
    uint32_t blockLength_ = 0;  // 当前 block 的长度。
    uint32_t tileLength_ = 0;  // 标准 tile 长度。
    uint32_t copyAlignBytes_ = kAlignBytes;  // copy 对齐粒度。
    uint32_t bufferSizeBytes_ = 0;  // 每个 data tile 的对齐后缓冲字节数。
    uint32_t dataAddr_[kDoubleBufferNum] = {0, 0};  // 两个 data slot 的起始地址。
    uint32_t tempAddr_ = 0;  // plus scratch 的起始地址。
    uint32_t denomAddr_ = 0;  // minus scratch 的起始地址。
    bool wholeTileFastCopy_ = false;  // 满 tile 是否都允许走 fast copy。
};

// 功能块说明：
// fp16 静态张量双缓冲实现。
// 它与 fp32 静态双缓冲结构基本一致，只是数据类型换成了 half，scratch 名字更直观地命名为 plus/minus。
class KernelAtanhFp16StaticDouble {
public:
    // 功能块说明：
    // 初始化 fp16 静态双缓冲 kernel 的 UB 地址布局与 GM 视图。
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t blockOffset, uint32_t currentBlockLength,
                                uint32_t tileLength, uint32_t copyAlignBytes)
    {
        blockOffset_ = blockOffset;  // 保存 block 全局偏移。
        blockLength_ = currentBlockLength;  // 保存 block 长度。
        tileLength_ = tileLength;  // 保存 tile 长度。
        copyAlignBytes_ = copyAlignBytes;  // 保存 copy 对齐粒度。
        wholeTileFastCopy_ = CanUseFastCopyAbs<half>(blockOffset_, tileLength_, copyAlignBytes_);  // 判断满 tile 是否都可走 fast copy。
        bufferSizeBytes_ = AlignUpCount(tileLength_ * sizeof(half), kAlignBytes);  // 计算每个 half tile 的对齐缓冲字节数。
        dataAddr_[0] = 0;  // slot0 数据起点。
        dataAddr_[1] = bufferSizeBytes_;  // slot1 数据起点。
        plusAddr_ = bufferSizeBytes_ * kDoubleBufferNum + kBankConflictSkewBytes;  // plus scratch 采用 bank-group 错位。
        minusAddr_ = bufferSizeBytes_ * (kDoubleBufferNum + 1U) + (kBankConflictSkewBytes * 2U);  // minus scratch 继续错位。

        inputGm_.SetGlobalBuffer((__gm__ half*)input + blockOffset, currentBlockLength);  // 绑定输入 GM 视图。
        outputGm_.SetGlobalBuffer((__gm__ half*)output + blockOffset, currentBlockLength);  // 绑定输出 GM 视图。

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // 初始时 slot0 空闲。
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);  // 初始时 slot1 空闲。
    }

    // 功能块说明：
    // 执行 fp16 静态双缓冲流水。
    __aicore__ inline void Process()
    {
        uint32_t tileCount = (blockLength_ + tileLength_ - 1) / tileLength_;  // 计算当前 block 的 tile 数量。
        if (tileCount == 0) {  // 若没有有效 tile。
            return;  // 直接返回。
        }
        if (tileCount == 3) {  // 若刚好是热场景 3 tile。
            ProcessThreeTiles();  // 走专门展开的三 tile 版本。
            return;  // 返回。
        }

        CopyIn(0);  // 预热搬入 tile0。
        if (tileCount == 1) {  // 若只有一个 tile。
            Compute(0);  // 直接计算 tile0。
            CopyOut(0);  // 直接写回 tile0。
            WaitAllSlots();  // 等待全部 slot 回空闲。
            return;  // 返回。
        }

        for (uint32_t tileIdx = 1; tileIdx < tileCount; ++tileIdx) {  // 推进稳态双缓冲流水。
            if (tileIdx > 1) {  // 当存在可写回 tile 时。
                CopyOut(tileIdx - 2);  // 写回前两轮的 tile。
            }
#if ATANH_PIPELINE_COPYOUT_COMPUTE_FIRST_FP16
            if (tileIdx > 1) {
            Compute(tileIdx - 1);  // 计算上一轮 tile。
            CopyIn(tileIdx);  // 搬入当前 tile。
            } else {
            CopyIn(tileIdx);  // 搬入当前 tile。
            Compute(tileIdx - 1);  // 计算上一轮 tile。
            }
#else
            CopyIn(tileIdx);  // 搬入当前 tile。
            Compute(tileIdx - 1);  // 计算上一轮 tile。
#endif
        }

        CopyOut(tileCount - 2);  // 写回倒数第二个 tile。
        Compute(tileCount - 1);  // 计算最后一个 tile。
        CopyOut(tileCount - 1);  // 写回最后一个 tile。
        WaitAllSlots();  // 等待最终排空。
    }

private:
    // 功能块说明：
    // slot 到静态事件 ID 的映射。
    __aicore__ inline int32_t GetEventId(uint32_t slot) const
    {
        return slot == 0 ? EVENT_ID0 : EVENT_ID1;  // slot0 对应 EVENT_ID0，slot1 对应 EVENT_ID1。
    }

    // 功能块说明：
    // 返回指定 slot 的数据视图。
    __aicore__ inline LocalTensor<half> GetDataTensor(uint32_t slot) const
    {
        return LocalTensor<half>(TPosition::VECCALC, dataAddr_[slot], tileLength_);  // 创建半精度 data 张量视图。
    }

    // 功能块说明：
    // 返回 plus scratch 的张量视图。
    __aicore__ inline LocalTensor<half> GetPlusTensor() const
    {
        return LocalTensor<half>(TPosition::VECCALC, plusAddr_, tileLength_);  // 创建 plus scratch 视图。
    }

    // 功能块说明：
    // 返回 minus scratch 的张量视图。
    __aicore__ inline LocalTensor<half> GetMinusTensor() const
    {
        return LocalTensor<half>(TPosition::VECCALC, minusAddr_, tileLength_);  // 创建 minus scratch 视图。
    }

    // 功能块说明：
    // 等待两个 slot 都完成最后一次写回。
    __aicore__ inline void WaitAllSlots()
    {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // 等待 slot0 最终空闲。
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);  // 等待 slot1 最终空闲。
    }

    // 功能块说明：
    // 三 tile 专门展开版本，结构与 fp32 的 ProcessThreeTiles 完全同构。
    __aicore__ inline void ProcessThreeTiles()
    {
        uint32_t offset0 = 0;  // tile0 起始偏移。
        uint32_t len0 = tileLength_;  // tile0 视为满 tile。
        uint32_t offset1 = tileLength_;  // tile1 起始偏移。
        uint32_t len1 = tileLength_;  // tile1 视为满 tile。
        uint32_t offset2 = tileLength_ * 2;  // tile2 起始偏移。
        uint32_t len2 = blockLength_ > offset2 ? (blockLength_ - offset2) : 0;  // tile2 为尾 tile 时按真实长度处理。
        LocalTensor<half> plusLocal = GetPlusTensor();  // 获取 plus scratch。
        LocalTensor<half> minusLocal = GetMinusTensor();  // 获取 minus scratch。
        LocalTensor<half> data0 = GetDataTensor(0);  // 获取 slot0 数据视图。
        LocalTensor<half> data1 = GetDataTensor(1);  // 获取 slot1 数据视图。

        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // 等待 slot0 空闲。
        CopyInTile(inputGm_, offset0, data0, len0, tileLength_, copyAlignBytes_, wholeTileFastCopy_);  // 搬入 tile0。
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);  // 标记 tile0 可计算。

        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);  // 等待 slot1 空闲。
        CopyInTile(inputGm_, offset1, data1, len1, tileLength_, copyAlignBytes_, wholeTileFastCopy_);  // 搬入 tile1。
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);  // 标记 tile1 可计算。

        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);  // 等待 tile0 搬入完成。
        ComputeAtanhHalfInplaceDualScratch(data0, plusLocal, minusLocal, len0);  // 计算 tile0。
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);  // 标记 tile0 可写回。

        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);  // 等待 tile0 计算完成。
        CopyOutTile(outputGm_, offset0, data0, len0, tileLength_, copyAlignBytes_, wholeTileFastCopy_);  // 写回 tile0。
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // 标记 slot0 再次空闲。

        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // 等待 slot0 可复用。
        CopyInTile(inputGm_, offset2, data0, len2, tileLength_, copyAlignBytes_, wholeTileFastCopy_);  // 搬入 tile2。
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);  // 标记 tile2 可计算。

        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);  // 等待 tile1 搬入完成。
        ComputeAtanhHalfInplaceDualScratch(data1, plusLocal, minusLocal, len1);  // 计算 tile1。
        SetFlag<HardEvent::V_MTE3>(EVENT_ID1);  // 标记 tile1 可写回。

        WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);  // 等待 tile1 计算完成。
        CopyOutTile(outputGm_, offset1, data1, len1, tileLength_, copyAlignBytes_, wholeTileFastCopy_);  // 写回 tile1。
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);  // 标记 slot1 再次空闲。

        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);  // 等待 tile2 搬入完成。
        ComputeAtanhHalfInplaceDualScratch(data0, plusLocal, minusLocal, len2);  // 计算 tile2。
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);  // 标记 tile2 可写回。

        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);  // 等待 tile2 计算完成。
        CopyOutTile(outputGm_, offset2, data0, len2, tileLength_, copyAlignBytes_, wholeTileFastCopy_);  // 写回 tile2。
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // 标记 slot0 再次空闲。

        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // 等待 slot0 最终空闲。
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);  // 等待 slot1 最终空闲。
    }

    // 功能块说明：
    // 根据 tile 序号计算该 tile 在 block 内的起始偏移。
    __aicore__ inline uint32_t GetTileOffset(uint32_t tileIdx) const
    {
        return tileIdx * tileLength_;  // tile 起点 = tileIdx * tileLength。
    }

    // 功能块说明：
    // 获取 tile 有效长度，尾 tile 可能短于标准长度。
    __aicore__ inline uint32_t GetTileLength(uint32_t tileIdx) const
    {
        uint32_t offset = GetTileOffset(tileIdx);  // 计算起始偏移。
        uint32_t len = blockLength_ - offset;  // 计算剩余长度。
        if (len > tileLength_) {  // 若为满 tile。
            len = tileLength_;  // 裁成标准长度。
        }
        return len;  // 返回有效长度。
    }

    // 功能块说明：
    // 通用 CopyIn 包装。
    __aicore__ inline void CopyIn(uint32_t tileIdx)
    {
        uint32_t offset = GetTileOffset(tileIdx);  // 计算 tile 起始偏移。
        uint32_t len = GetTileLength(tileIdx);  // 计算 tile 有效长度。
        CopyInSlot(tileIdx & 1U, offset, len);  // 按最低位选择 slot 执行搬入。
    }

    // 功能块说明：
    // 通用 Compute 包装。
    __aicore__ inline void Compute(uint32_t tileIdx)
    {
        uint32_t len = GetTileLength(tileIdx);  // 读取 tile 有效长度。
        ComputeSlot(tileIdx & 1U, len);  // 在对应 slot 上执行计算。
    }

    // 功能块说明：
    // 通用 CopyOut 包装。
    __aicore__ inline void CopyOut(uint32_t tileIdx)
    {
        uint32_t offset = GetTileOffset(tileIdx);  // 计算 tile 起始偏移。
        uint32_t len = GetTileLength(tileIdx);  // 计算 tile 有效长度。
        CopyOutSlot(tileIdx & 1U, offset, len);  // 在对应 slot 上执行写回。
    }

    // 功能块说明：
    // 在指定 slot 上执行搬入。
    __aicore__ inline void CopyInSlot(uint32_t slot, uint32_t offset, uint32_t len)
    {
        int32_t eventId = GetEventId(slot);  // 获取 slot 对应的事件 ID。
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);  // 等待该 slot 空闲。
        LocalTensor<half> dataLocal = GetDataTensor(slot);  // 获取该 slot 的数据视图。
        CopyInTile(inputGm_, offset, dataLocal, len, tileLength_, copyAlignBytes_, wholeTileFastCopy_);  // 执行搬入。
        SetFlag<HardEvent::MTE2_V>(eventId);  // 通知计算阶段数据已就绪。
    }

    // 功能块说明：
    // 在指定 slot 上执行计算。
    __aicore__ inline void ComputeSlot(uint32_t slot, uint32_t len)
    {
        int32_t eventId = GetEventId(slot);  // 获取 slot 对应事件 ID。
        WaitFlag<HardEvent::MTE2_V>(eventId);  // 等待搬入完成。
        LocalTensor<half> dataLocal = GetDataTensor(slot);  // 获取数据视图。
        LocalTensor<half> plusLocal = GetPlusTensor();  // 获取 plus scratch。
        LocalTensor<half> minusLocal = GetMinusTensor();  // 获取 minus scratch。
        ComputeAtanhHalfInplaceDualScratch(dataLocal, plusLocal, minusLocal, len);  // 执行 half 双 scratch 原地计算。
        SetFlag<HardEvent::V_MTE3>(eventId);  // 通知写回阶段结果已就绪。
    }

    // 功能块说明：
    // 在指定 slot 上执行写回。
    __aicore__ inline void CopyOutSlot(uint32_t slot, uint32_t offset, uint32_t len)
    {
        int32_t eventId = GetEventId(slot);  // 获取 slot 对应事件 ID。
        WaitFlag<HardEvent::V_MTE3>(eventId);  // 等待计算完成。
        LocalTensor<half> dataLocal = GetDataTensor(slot);  // 获取数据视图。
        CopyOutTile(outputGm_, offset, dataLocal, len, tileLength_, copyAlignBytes_, wholeTileFastCopy_);  // 执行写回。
        SetFlag<HardEvent::MTE3_MTE2>(eventId);  // 标记该 slot 再次空闲。
    }

private:
    TPipe pipe_;  // pipe 管理器。
    GlobalTensor<half> inputGm_;  // 输入 GM 视图。
    GlobalTensor<half> outputGm_;  // 输出 GM 视图。
    uint32_t blockOffset_ = 0;  // 当前 block 的全局偏移。
    uint32_t blockLength_ = 0;  // 当前 block 的长度。
    uint32_t tileLength_ = 0;  // 标准 tile 长度。
    uint32_t copyAlignBytes_ = kAlignBytes;  // copy 对齐粒度。
    uint32_t bufferSizeBytes_ = 0;  // 每个 half tile 的对齐后缓冲字节数。
    uint32_t dataAddr_[kDoubleBufferNum] = {0, 0};  // 两个 data slot 的起始地址。
    uint32_t plusAddr_ = 0;  // plus scratch 起始地址。
    uint32_t minusAddr_ = 0;  // minus scratch 起始地址。
    bool wholeTileFastCopy_ = false;  // 满 tile 是否允许走 fast copy。
};

// 功能块说明：
// bf16 单缓冲实现。
// 数据从 bf16 搬入后先 cast 到 float，在 float 域执行 atanh，再 cast 回 bf16 写回。
class KernelAtanhBf16Single {
public:
    // 功能块说明：
    // 初始化 bf16 单缓冲 kernel 的 GM 视图与 UB 缓冲。
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t blockOffset, uint32_t currentBlockLength,
                                uint32_t tileLength, uint32_t copyAlignBytes)
    {
        blockOffset_ = blockOffset;  // 保存 block 全局偏移。
        blockLength_ = currentBlockLength;  // 保存 block 长度。
        tileLength_ = tileLength;  // 保存 tile 长度。
        copyAlignBytes_ = copyAlignBytes;  // 保存 copy 对齐粒度。
        inputGm_.SetGlobalBuffer((__gm__ __bf16*)input + blockOffset, currentBlockLength);  // 绑定输入 GM 视图。
        outputGm_.SetGlobalBuffer((__gm__ __bf16*)output + blockOffset, currentBlockLength);  // 绑定输出 GM 视图。

        const uint32_t rawBufferSize = AlignUpCount(tileLength_ * sizeof(__bf16), kAlignBytes);  // 计算 bf16 输入/输出缓冲字节数。
        const uint32_t floatBufferSize = AlignUpCount(tileLength_ * sizeof(float), kAlignBytes);  // 计算 float 中间缓冲字节数。
        wholeTileFastCopy_ = CanUseFastCopyAbs<__bf16>(blockOffset_, tileLength_, copyAlignBytes_);  // 判断满 tile 是否都允许走 fast copy。
        pipe_.InitBuffer(inQueue_, kSingleBufferNum, rawBufferSize);  // 初始化 bf16 输入队列。
        pipe_.InitBuffer(outQueue_, kSingleBufferNum, rawBufferSize);  // 初始化 bf16 输出队列。
        pipe_.InitBuffer(floatBuf_, floatBufferSize);  // 初始化 float 中间缓冲。
        pipe_.InitBuffer(tempBuf_, floatBufferSize);  // 初始化 float 临时缓冲。
    }

    // 功能块说明：
    // 逐 tile 串行处理整个 block。
    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < blockLength_; offset += tileLength_) {  // 以 tileLength 为步长扫描整个 block。
            uint32_t len = blockLength_ - offset;  // 先计算剩余元素数。
            if (len > tileLength_) {  // 若剩余元素数超过一个标准 tile。
                len = tileLength_;  // 则本轮按满 tile 处理。
            }
            ComputeTile(offset, len);  // 处理当前 tile。
        }
    }

private:
    // 功能块说明：
    // 单个 bf16 tile 的完整处理链：搬入 -> cast 到 float -> 计算 -> cast 回 bf16 -> 写回。
    __aicore__ inline void ComputeTile(uint32_t offset, uint32_t len)
    {
        LocalTensor<__bf16> inputLocal = inQueue_.AllocTensor<__bf16>();  // 申请 bf16 输入缓冲。
        CopyInTile(inputGm_, offset, inputLocal, len, tileLength_, copyAlignBytes_, wholeTileFastCopy_);  // 搬入当前 tile。
        inQueue_.EnQue(inputLocal);  // 输入缓冲入队。
        inputLocal = inQueue_.DeQue<__bf16>();  // 输入缓冲出队。

        LocalTensor<float> inputFloat = floatBuf_.Get<float>();  // 获取 float 中间缓冲。
        Cast(inputFloat, inputLocal, RoundMode::CAST_NONE, len);  // 将 bf16 输入转换成 float。

        LocalTensor<float> tempLocal = tempBuf_.Get<float>();  // 获取 float 临时缓冲。
        ComputeAtanh(inputFloat, inputFloat, tempLocal, len);  // 直接在 float 缓冲上原地计算 atanh。

        LocalTensor<__bf16> outputLocal = outQueue_.AllocTensor<__bf16>();  // 申请 bf16 输出缓冲。
        Cast(outputLocal, inputFloat, RoundMode::CAST_ROUND, len);  // 将 float 结果 round 回 bf16。
        outQueue_.EnQue(outputLocal);  // 输出缓冲入队。
        outputLocal = outQueue_.DeQue<__bf16>();  // 输出缓冲出队。
        CopyOutTile(outputGm_, offset, outputLocal, len, tileLength_, copyAlignBytes_, wholeTileFastCopy_);  // 将结果写回 GM。

        outQueue_.FreeTensor(outputLocal);  // 释放输出缓冲。
        inQueue_.FreeTensor(inputLocal);  // 释放输入缓冲。
    }

private:
    TPipe pipe_;  // pipe 管理器。
    TQue<QuePosition::VECIN, kSingleBufferNum> inQueue_;  // bf16 输入队列。
    TQue<QuePosition::VECOUT, kSingleBufferNum> outQueue_;  // bf16 输出队列。
    TBuf<QuePosition::VECCALC> floatBuf_;  // bf16 -> float 中间缓冲。
    TBuf<QuePosition::VECCALC> tempBuf_;  // float 计算临时缓冲。
    GlobalTensor<__bf16> inputGm_;  // 输入 GM 视图。
    GlobalTensor<__bf16> outputGm_;  // 输出 GM 视图。
    uint32_t blockOffset_ = 0;  // 当前 block 全局偏移。
    uint32_t blockLength_ = 0;  // 当前 block 长度。
    uint32_t tileLength_ = 0;  // tile 长度。
    uint32_t copyAlignBytes_ = kAlignBytes;  // copy 对齐粒度。
    bool wholeTileFastCopy_ = false;  // 满 tile 是否可统一走 fast copy。
};

// 功能块说明：
// bf16 双缓冲 cast-back 路径。
// 搬入原始 bf16，计算阶段转 float 原地执行 atanh，再 round 回 bf16，最后双缓冲写回。
class KernelAtanhBf16Double {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t blockOffset, uint32_t currentBlockLength,
                                uint32_t tileLength, uint32_t copyAlignBytes)
    {
        blockOffset_ = blockOffset;
        blockLength_ = currentBlockLength;
        tileLength_ = tileLength;
        copyAlignBytes_ = copyAlignBytes;
        wholeTileFastCopy_ = CanUseFastCopyAbs<__bf16>(blockOffset_, tileLength_, copyAlignBytes_);
        inputGm_.SetGlobalBuffer((__gm__ __bf16*)input + blockOffset, currentBlockLength);
        outputGm_.SetGlobalBuffer((__gm__ __bf16*)output + blockOffset, currentBlockLength);

        const uint32_t rawBufferSize = AlignUpCount(tileLength_ * sizeof(__bf16), kAlignBytes);
        const uint32_t floatBufferSize = AlignUpCount(tileLength_ * sizeof(float), kAlignBytes);
        pipe_.InitBuffer(rawBufs_[0], rawBufferSize);
        pipe_.InitBuffer(rawBufs_[1], rawBufferSize);
        pipe_.InitBuffer(floatBuf_, floatBufferSize);
        pipe_.InitBuffer(tempBuf_, floatBufferSize);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void Process()
    {
        uint32_t tileCount = (blockLength_ + tileLength_ - 1) / tileLength_;
        if (tileCount == 0) {
            WaitAllSlots();
            return;
        }

        CopyIn(0);
        if (tileCount == 1) {
            Compute(0);
            CopyOut(0);
            WaitAllSlots();
            return;
        }

        for (uint32_t tileIdx = 1; tileIdx < tileCount; ++tileIdx) {
            if (tileIdx > 1) {
                CopyOut(tileIdx - 2);
            }
#if ATANH_PIPELINE_COPYOUT_COMPUTE_FIRST_BF16
            if (tileIdx > 1) {
            Compute(tileIdx - 1);
            CopyIn(tileIdx);
            } else {
            CopyIn(tileIdx);
            Compute(tileIdx - 1);
            }
#else
            CopyIn(tileIdx);
            Compute(tileIdx - 1);
#endif
        }

        CopyOut(tileCount - 2);
        Compute(tileCount - 1);
        CopyOut(tileCount - 1);
        WaitAllSlots();
    }

private:
    __aicore__ inline int32_t GetEventId(uint32_t slot) const
    {
        return slot == 0 ? EVENT_ID0 : EVENT_ID1;
    }

    __aicore__ inline void WaitAllSlots()
    {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    __aicore__ inline uint32_t GetTileOffset(uint32_t tileIdx) const
    {
        return tileIdx * tileLength_;
    }

    __aicore__ inline uint32_t GetTileLength(uint32_t tileIdx) const
    {
        uint32_t offset = GetTileOffset(tileIdx);
        uint32_t len = blockLength_ - offset;
        if (len > tileLength_) {
            len = tileLength_;
        }
        return len;
    }

    __aicore__ inline void CopyIn(uint32_t tileIdx)
    {
        uint32_t slot = tileIdx & 1U;
        int32_t eventId = GetEventId(slot);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
        uint32_t offset = GetTileOffset(tileIdx);
        uint32_t len = GetTileLength(tileIdx);
        LocalTensor<__bf16> inputLocal = rawBufs_[slot].Get<__bf16>();
        CopyInTile(inputGm_, offset, inputLocal, len, tileLength_, copyAlignBytes_, wholeTileFastCopy_);
        SetFlag<HardEvent::MTE2_V>(eventId);
    }

    __aicore__ inline void Compute(uint32_t tileIdx)
    {
        uint32_t slot = tileIdx & 1U;
        int32_t eventId = GetEventId(slot);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        uint32_t len = GetTileLength(tileIdx);
        LocalTensor<__bf16> inputLocal = rawBufs_[slot].Get<__bf16>();
        LocalTensor<__bf16> outputLocal = rawBufs_[slot].Get<__bf16>();
        LocalTensor<float> inputFloat = floatBuf_.Get<float>();
        Cast(inputFloat, inputLocal, RoundMode::CAST_NONE, len);
        LocalTensor<float> tempLocal = tempBuf_.Get<float>();
        ComputeAtanh(inputFloat, inputFloat, tempLocal, len);
        Cast(outputLocal, inputFloat, RoundMode::CAST_ROUND, len);
        SetFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx)
    {
        uint32_t slot = tileIdx & 1U;
        int32_t eventId = GetEventId(slot);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        uint32_t offset = GetTileOffset(tileIdx);
        uint32_t len = GetTileLength(tileIdx);
        LocalTensor<__bf16> outputLocal = rawBufs_[slot].Get<__bf16>();
        CopyOutTile(outputGm_, offset, outputLocal, len, tileLength_, copyAlignBytes_, wholeTileFastCopy_);
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
    }

private:
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> rawBufs_[kDoubleBufferNum];
    TBuf<QuePosition::VECCALC> floatBuf_;
    TBuf<QuePosition::VECCALC> tempBuf_;
    GlobalTensor<__bf16> inputGm_;
    GlobalTensor<__bf16> outputGm_;
    uint32_t blockOffset_ = 0;
    uint32_t blockLength_ = 0;
    uint32_t tileLength_ = 0;
    uint32_t copyAlignBytes_ = kAlignBytes;
    bool wholeTileFastCopy_ = false;
};

// 功能块说明：
// int/uint 输入转 float 输出的单缓冲路径。
// 对 int8/uint8 先转 half 再转 float，和历史实现保持兼容；其余类型直接转 float。
template <typename InputT>
class KernelAtanhCastFloatSingle {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t blockOffset, uint32_t currentBlockLength,
                                uint32_t tileLength, uint32_t copyAlignBytes)
    {
        blockOffset_ = blockOffset;
        blockLength_ = currentBlockLength;
        tileLength_ = tileLength;
        copyAlignBytes_ = copyAlignBytes;
        wholeTileFastCopy_ = CanUseFastCopyAbs<InputT>(blockOffset_, tileLength_, copyAlignBytes_);
        outputWholeTileFastCopy_ = CanUseFastCopyAbs<float>(blockOffset_, tileLength_, copyAlignBytes_);
        inputGm_.SetGlobalBuffer((__gm__ InputT*)input + blockOffset, currentBlockLength);
        outputGm_.SetGlobalBuffer((__gm__ float*)output + blockOffset, currentBlockLength);

        const uint32_t inputBufferSize = AlignUpCount(tileLength_ * sizeof(InputT), kAlignBytes);
        const uint32_t halfBufferSize = AlignUpCount(tileLength_ * sizeof(half), kAlignBytes);
        const uint32_t floatBufferSize = AlignUpCount(tileLength_ * sizeof(float), kAlignBytes);
        pipe_.InitBuffer(inQueue_, kSingleBufferNum, inputBufferSize);
        pipe_.InitBuffer(floatInputQueue_, kSingleBufferNum, floatBufferSize);
        pipe_.InitBuffer(floatOutputQueue_, kSingleBufferNum, floatBufferSize);
        pipe_.InitBuffer(tempBuf_, floatBufferSize);
        if constexpr (std::is_same_v<InputT, int8_t> || std::is_same_v<InputT, uint8_t>) {
            pipe_.InitBuffer(halfInputQueue_, kSingleBufferNum, halfBufferSize);
        }
    }

    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < blockLength_; offset += tileLength_) {
            uint32_t len = blockLength_ - offset;
            if (len > tileLength_) {
                len = tileLength_;
            }
            ComputeTile(offset, len);
        }
    }

private:
    __aicore__ inline void ComputeTile(uint32_t offset, uint32_t len)
    {
        LocalTensor<InputT> inputLocalRaw = inQueue_.AllocTensor<InputT>();
        CopyInTile(inputGm_, offset, inputLocalRaw, len, tileLength_, copyAlignBytes_, wholeTileFastCopy_);
        inQueue_.EnQue(inputLocalRaw);
        inputLocalRaw = inQueue_.DeQue<InputT>();

        LocalTensor<float> inputLocal = floatInputQueue_.AllocTensor<float>();
        if constexpr (std::is_same_v<InputT, int8_t> || std::is_same_v<InputT, uint8_t>) {
            LocalTensor<half> inputHalf = halfInputQueue_.AllocTensor<half>();
            Cast(inputHalf, inputLocalRaw, RoundMode::CAST_NONE, len);
            Cast(inputLocal, inputHalf, RoundMode::CAST_NONE, len);
            halfInputQueue_.FreeTensor(inputHalf);
        } else {
            Cast(inputLocal, inputLocalRaw, RoundMode::CAST_NONE, len);
        }

        LocalTensor<float> outputLocal = floatOutputQueue_.AllocTensor<float>();
        LocalTensor<float> tempLocal = tempBuf_.Get<float>();
        ComputeAtanh(outputLocal, inputLocal, tempLocal, len);

        floatOutputQueue_.EnQue(outputLocal);
        outputLocal = floatOutputQueue_.DeQue<float>();
        CopyOutTile(outputGm_, offset, outputLocal, len, tileLength_, copyAlignBytes_, outputWholeTileFastCopy_);

        floatOutputQueue_.FreeTensor(outputLocal);
        floatInputQueue_.FreeTensor(inputLocal);
        inQueue_.FreeTensor(inputLocalRaw);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, kSingleBufferNum> inQueue_;
    TQue<QuePosition::VECCALC, kSingleBufferNum> halfInputQueue_;
    TQue<QuePosition::VECCALC, kSingleBufferNum> floatInputQueue_;
    TQue<QuePosition::VECOUT, kSingleBufferNum> floatOutputQueue_;
    TBuf<QuePosition::VECCALC> tempBuf_;
    GlobalTensor<InputT> inputGm_;
    GlobalTensor<float> outputGm_;
    uint32_t blockOffset_ = 0;
    uint32_t blockLength_ = 0;
    uint32_t tileLength_ = 0;
    uint32_t copyAlignBytes_ = kAlignBytes;
    bool wholeTileFastCopy_ = false;
    bool outputWholeTileFastCopy_ = false;
};

// 功能块说明：
// 单缓冲通用运行包装。
// 它负责根据 blockIdx 计算当前核应处理的 blockOffset 和 blockLength，再实例化对应 kernel。
template <typename T>
__aicore__ inline void RunAtanhSingle(GM_ADDR input, GM_ADDR output, uint32_t blockLength, uint32_t lastBlockLength,
                                      uint32_t tileLength, uint32_t copyAlignBytes)
{
    uint32_t blockIdx = GetBlockIdx();  // 读取当前 AIV block 索引。
    uint32_t blockOffset = blockLength * blockIdx;  // 计算当前 block 在全局张量中的起始偏移。
    uint32_t currentBlockLength = (blockIdx + 1 == GetBlockNum()) ? lastBlockLength : blockLength;  // 最后一个 block 用 lastBlockLength，其余 block 用标准 blockLength。
    KernelAtanhSingle<T> kernel;  // 构造单缓冲 kernel 实例。
    kernel.Init(input, output, blockOffset, currentBlockLength, tileLength, copyAlignBytes);  // 初始化 kernel。
    kernel.Process();  // 执行 kernel。
}

// 功能块说明：
// 双缓冲通用运行包装。
template <typename T>
__aicore__ inline void RunAtanhDouble(GM_ADDR input, GM_ADDR output, uint32_t blockLength, uint32_t lastBlockLength,
                                      uint32_t tileLength, uint32_t copyAlignBytes)
{
    uint32_t blockIdx = GetBlockIdx();  // 读取当前 AIV block 索引。
    uint32_t blockOffset = blockLength * blockIdx;  // 计算当前 block 的全局偏移。
    uint32_t currentBlockLength = (blockIdx + 1 == GetBlockNum()) ? lastBlockLength : blockLength;  // 计算当前 block 的实际长度。
    KernelAtanhDouble<T> kernel;  // 构造双缓冲 kernel 实例。
    kernel.Init(input, output, blockOffset, currentBlockLength, tileLength, copyAlignBytes);  // 初始化 kernel。
    kernel.Process();  // 执行 kernel。
}

// 功能块说明：
// 旧版 fp32 紧凑双缓冲运行包装。
__aicore__ inline void RunAtanhFp32CompactDouble(GM_ADDR input, GM_ADDR output, uint32_t blockLength,
                                                 uint32_t lastBlockLength, uint32_t tileLength,
                                                 uint32_t copyAlignBytes)
{
    uint32_t blockIdx = GetBlockIdx();  // 读取当前 AIV block 索引。
    uint32_t blockOffset = blockLength * blockIdx;  // 计算当前 block 的全局偏移。
    uint32_t currentBlockLength = (blockIdx + 1 == GetBlockNum()) ? lastBlockLength : blockLength;  // 计算当前 block 的实际长度。
    KernelAtanhFp32CompactDouble kernel;  // 构造旧版 fp32 紧凑双缓冲 kernel。
    kernel.Init(input, output, blockOffset, currentBlockLength, tileLength, copyAlignBytes);  // 初始化 kernel。
    kernel.Process();  // 执行 kernel。
}

// 功能块说明：
// 当前主热路径 fp32 静态张量双缓冲运行包装。
__aicore__ inline void RunAtanhFp32StaticDouble(GM_ADDR input, GM_ADDR output, uint32_t blockLength,
                                                uint32_t lastBlockLength, uint32_t tileLength,
                                                uint32_t copyAlignBytes)
{
    uint32_t blockIdx = GetBlockIdx();  // 读取当前 AIV block 索引。
    uint32_t blockOffset = blockLength * blockIdx;  // 计算当前 block 的全局偏移。
    uint32_t currentBlockLength = (blockIdx + 1 == GetBlockNum()) ? lastBlockLength : blockLength;  // 计算当前 block 的实际长度。
    KernelAtanhFp32StaticDouble kernel;  // 构造 fp32 静态双缓冲 kernel。
    kernel.Init(input, output, blockOffset, currentBlockLength, tileLength, copyAlignBytes);  // 初始化 kernel。
    kernel.Process();  // 执行 kernel。
}

// 功能块说明：
// fp16 静态张量双缓冲运行包装。
__aicore__ inline void RunAtanhFp16StaticDouble(GM_ADDR input, GM_ADDR output, uint32_t blockLength,
                                                uint32_t lastBlockLength, uint32_t tileLength,
                                                uint32_t copyAlignBytes)
{
    uint32_t blockIdx = GetBlockIdx();  // 读取当前 AIV block 索引。
    uint32_t blockOffset = blockLength * blockIdx;  // 计算当前 block 的全局偏移。
    uint32_t currentBlockLength = (blockIdx + 1 == GetBlockNum()) ? lastBlockLength : blockLength;  // 计算当前 block 的实际长度。
    KernelAtanhFp16StaticDouble kernel;  // 构造 fp16 静态双缓冲 kernel。
    kernel.Init(input, output, blockOffset, currentBlockLength, tileLength, copyAlignBytes);  // 初始化 kernel。
    kernel.Process();  // 执行 kernel。
}

// 功能块说明：
// bf16 单缓冲运行包装。
__aicore__ inline void RunAtanhBf16Single(GM_ADDR input, GM_ADDR output, uint32_t blockLength,
                                          uint32_t lastBlockLength, uint32_t tileLength, uint32_t copyAlignBytes)
{
    uint32_t blockIdx = GetBlockIdx();  // 读取当前 AIV block 索引。
    uint32_t blockOffset = blockLength * blockIdx;  // 计算当前 block 的全局偏移。
    uint32_t currentBlockLength = (blockIdx + 1 == GetBlockNum()) ? lastBlockLength : blockLength;  // 计算当前 block 的实际长度。
    KernelAtanhBf16Single kernel;  // 构造 bf16 单缓冲 kernel。
    kernel.Init(input, output, blockOffset, currentBlockLength, tileLength, copyAlignBytes);  // 初始化 kernel。
    kernel.Process();  // 执行 kernel。
}

// 功能块说明：
// bf16 双缓冲运行包装。
__aicore__ inline void RunAtanhBf16Double(GM_ADDR input, GM_ADDR output, uint32_t blockLength,
                                          uint32_t lastBlockLength, uint32_t tileLength, uint32_t copyAlignBytes)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockOffset = blockLength * blockIdx;
    uint32_t currentBlockLength = (blockIdx + 1 == GetBlockNum()) ? lastBlockLength : blockLength;
    KernelAtanhBf16Double kernel;
    kernel.Init(input, output, blockOffset, currentBlockLength, tileLength, copyAlignBytes);
    kernel.Process();
}

// 功能块说明：
// int/uint 输入转 float 输出的单缓冲运行包装。
template <typename InputT>
__aicore__ inline void RunAtanhCastFloatSingle(GM_ADDR input, GM_ADDR output, uint32_t blockLength,
                                               uint32_t lastBlockLength, uint32_t tileLength,
                                               uint32_t copyAlignBytes)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockOffset = blockLength * blockIdx;
    uint32_t currentBlockLength = (blockIdx + 1 == GetBlockNum()) ? lastBlockLength : blockLength;
    KernelAtanhCastFloatSingle<InputT> kernel;
    kernel.Init(input, output, blockOffset, currentBlockLength, tileLength, copyAlignBytes);
    kernel.Process();
}

}  // namespace  // 匿名命名空间结束。

// 功能块说明：
// 这是算子在 device 侧的统一入口函数。
// 它负责读取 host 下发的 tiling 数据，并根据 tiling key 分发到不同 dtype / buffer 策略的具体 kernel。
extern "C" __global__ __aicore__ void atanh(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;  // 当前算子没有使用独立 workspace，显式消除未使用告警。
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);  // 声明当前 kernel 仅使用 AIV 向量核任务类型。
    GET_TILING_DATA(tilingData, tiling);  // 从运行时传入的 tiling 指针中反序列化 host 侧 tiling 数据。

    if (TILING_KEY_IS(1)) {  // 若当前 tiling key 对应 fp16 单缓冲路径。
        RunAtanhSingle<half>(input, output, tilingData.blockLength, tilingData.lastBlockLength, tilingData.tileLength,
                             tilingData.copyAlignBytes);  // 调度到 half 单缓冲 kernel。
    } else if (TILING_KEY_IS(2)) {  // 若当前 tiling key 对应 fp32 单缓冲路径。
        RunAtanhSingle<float>(input, output, tilingData.blockLength, tilingData.lastBlockLength, tilingData.tileLength,
                              tilingData.copyAlignBytes);  // 调度到 float 单缓冲 kernel。
    } else if (TILING_KEY_IS(3)) {  // 若当前 tiling key 对应 bf16 单缓冲路径。
        RunAtanhBf16Single(input, output, tilingData.blockLength, tilingData.lastBlockLength, tilingData.tileLength,
                           tilingData.copyAlignBytes);  // 调度到 bf16 单缓冲 kernel。
    } else if (TILING_KEY_IS(4)) {
        RunAtanhCastFloatSingle<int32_t>(input, output, tilingData.blockLength, tilingData.lastBlockLength,
                                         tilingData.tileLength, tilingData.copyAlignBytes);
    } else if (TILING_KEY_IS(5)) {
        RunAtanhCastFloatSingle<int16_t>(input, output, tilingData.blockLength, tilingData.lastBlockLength,
                                         tilingData.tileLength, tilingData.copyAlignBytes);
    } else if (TILING_KEY_IS(6)) {
        RunAtanhCastFloatSingle<uint8_t>(input, output, tilingData.blockLength, tilingData.lastBlockLength,
                                         tilingData.tileLength, tilingData.copyAlignBytes);
    } else if (TILING_KEY_IS(7)) {
        RunAtanhCastFloatSingle<int8_t>(input, output, tilingData.blockLength, tilingData.lastBlockLength,
                                        tilingData.tileLength, tilingData.copyAlignBytes);
    } else if (TILING_KEY_IS(10)) {  // 若当前 tiling key 对应 fp16 静态双缓冲路径。
        AscendC::InitSocState();  // 初始化静态事件状态，供 EVENT_ID0 / EVENT_ID1 路径使用。
        RunAtanhFp16StaticDouble(input, output, tilingData.blockLength, tilingData.lastBlockLength,
                                 tilingData.tileLength, tilingData.copyAlignBytes);  // 调度到 fp16 静态双缓冲 kernel。
    } else if (TILING_KEY_IS(11)) {
        AscendC::InitSocState();
        RunAtanhBf16Double(input, output, tilingData.blockLength, tilingData.lastBlockLength, tilingData.tileLength,
                           tilingData.copyAlignBytes);
    } else if (TILING_KEY_IS(20)) {  // 若当前 tiling key 对应 fp32 静态双缓冲主路径。
        AscendC::InitSocState();  // 初始化静态事件状态。
        RunAtanhFp32StaticDouble(input, output, tilingData.blockLength, tilingData.lastBlockLength,
                                 tilingData.tileLength, tilingData.copyAlignBytes);  // 调度到 fp32 静态双缓冲 kernel。
    } else if (TILING_KEY_IS(21)) {  // 若运行时仍请求历史兼容 key21。
        AscendC::InitSocState();  // 同样初始化静态事件状态。
        RunAtanhFp32StaticDouble(input, output, tilingData.blockLength, tilingData.lastBlockLength,
                                 tilingData.tileLength, tilingData.copyAlignBytes);  // 兼容性地仍调度到同一份 fp32 静态双缓冲实现。
    }
}
