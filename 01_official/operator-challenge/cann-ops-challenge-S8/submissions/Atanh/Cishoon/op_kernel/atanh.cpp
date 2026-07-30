#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;

template <typename T>
class KernelAtanh {
    // 是否需要 Cast 到浮点类型再计算
    static constexpr bool kNeedsCast = !(std::is_same_v<T, float> || std::is_same_v<T, half>);
    // 计算类型: int8/uint8 → half, 其他需 Cast 的类型(bf16/int16/int32) → float
    using CompT = std::conditional_t<
        !kNeedsCast, T,
        std::conditional_t<std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>, half, float>
    >;

public:
    // 手写双缓冲: 不用 TPipe/TQue, 直接按字节偏移构造 LocalTensor, 用 SetFlag/WaitFlag 显式编排
    // MTE2(搬入)/V(计算)/MTE3(搬出) 三级流水, 去掉 TQue 每 tile 的 Alloc/EnQue/DeQue/Free 开销
    // 与自动同步的保守性。UB 布局(沿用 host 给的 tileLength, 预算完全一致):
    //   非 cast: in[0] in[1] out[0] out[1]                 (4 × tileLength × sizeof(T))
    //   cast  : 上述 4 块 + 共享 c1 c2                       (+ 2 × tileLength × sizeof(CompT))
    //   c1/c2 是 V 阶段临时, V 串行执行 → 两槽位共享一份即可, 无需双份。
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                                uint32_t coreDataNum, uint32_t globalOffset, uint32_t tileLength)
    {
        InitSocState();
        inputGm.SetGlobalBuffer((__gm__ T*)input + globalOffset, coreDataNum);
        outputGm.SetGlobalBuffer((__gm__ T*)output + globalOffset, coreDataNum);
        this->coreDataNum = coreDataNum;
        this->tileLength  = tileLength;

        uint32_t tBytes = tileLength * sizeof(T);
        inLocal[0]  = LocalTensor<T>(TPosition::VECIN,  0,          tileLength);
        inLocal[1]  = LocalTensor<T>(TPosition::VECIN,  tBytes,     tileLength);
        outLocal[0] = LocalTensor<T>(TPosition::VECOUT, 2 * tBytes, tileLength);
        outLocal[1] = LocalTensor<T>(TPosition::VECOUT, 3 * tBytes, tileLength);
        if constexpr (kNeedsCast) {
            uint32_t cBase = 4 * tBytes;
            c1 = LocalTensor<CompT>(TPosition::VECCALC, cBase, tileLength);
            c2 = LocalTensor<CompT>(TPosition::VECCALC, cBase + tileLength * sizeof(CompT), tileLength);
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t numTiles = (coreDataNum + tileLength - 1) / tileLength;
        for (uint32_t iter = 0; iter < numTiles; iter++) {
            int32_t  j      = iter & 1;                 // ping-pong 槽位
            uint32_t offset = iter * tileLength;
            uint32_t length = (offset + tileLength <= coreDataNum) ? tileLength
                                                                   : (coreDataNum - offset);

            // 等本槽位上一轮(iter-2)的搬出完成, 才能复用 in/out buffer
            if (iter >= 2) WaitFlag<HardEvent::MTE3_MTE2>(j);
            DataCopy(inLocal[j], inputGm[offset], length);
            SetFlag<HardEvent::MTE2_V>(j);
            WaitFlag<HardEvent::MTE2_V>(j);

            Compute(j, length);

            SetFlag<HardEvent::V_MTE3>(j);
            WaitFlag<HardEvent::V_MTE3>(j);
            DataCopy(outputGm[offset], outLocal[j], length);
            // 还有 iter+2 会复用本槽位时才发标志, 末尾两轮不发 → 收支平衡, 事件全部消费
            if (iter + 2 < numTiles) SetFlag<HardEvent::MTE3_MTE2>(j);
        }
    }

private:
    // atanh(x) = 0.5 * ln((1+x)/(1-x))，算子序列与原实现逐一对应
    __aicore__ inline void Compute(int32_t j, uint32_t length)
    {
        if constexpr (!kNeedsCast) {
            // float / half: 直接在原始类型上算, inLocal[j] 复用为 (1-x) 临时
            Adds(outLocal[j], inLocal[j], (T)1, length);     // 1 + x
            Muls(inLocal[j], inLocal[j], (T)(-1), length);   // -x
            Adds(inLocal[j], inLocal[j], (T)1, length);      // 1 - x
            Div(outLocal[j], outLocal[j], inLocal[j], length);
            Ln(outLocal[j], outLocal[j], length);
            Muls(outLocal[j], outLocal[j], (T)0.5, length);
        } else {
            // bf16/int*: Cast → 计算(CompT) → Cast 回
            Cast(c1, inLocal[j], RoundMode::CAST_NONE, length);
            Adds(c2, c1, (CompT)1, length);
            Muls(c1, c1, (CompT)(-1), length);
            Adds(c1, c1, (CompT)1, length);
            Div(c2, c2, c1, length);
            Ln(c2, c2, length);
            Muls(c2, c2, (CompT)0.5, length);
            Cast(outLocal[j], c2, RoundMode::CAST_ROUND, length);
        }
    }

    GlobalTensor<T>    inputGm, outputGm;
    LocalTensor<T>     inLocal[2], outLocal[2];
    LocalTensor<CompT> c1, c2;
    uint32_t coreDataNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void atanh(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    // 手写 1 次 64-bit GM load 替掉 GET_TILING_DATA 自动生成的 2 次 32-bit load。
    // AtanhTilingData 布局 (#pragma pack(1))：[0..3]=smallCoreDataNum(含 tailBlockNum), [4..7]=tileLength
    // aarch64 little-endian：低 32 bit 是 smallCoreDataNum，高 32 bit 是 tileLength
    uint64_t packed = *((const __gm__ uint64_t*)tiling);
    uint32_t packedSmallCoreDataNum = (uint32_t)packed;
    uint32_t tileLength = (uint32_t)(packed >> 32);

    uint32_t tailBlockNum = packedSmallCoreDataNum & 127;
    uint32_t smallCoreDataNum = packedSmallCoreDataNum & (~127u);
    // bigCoreDataNum = smallCoreDataNum + blockDataNum（编译期常量）
    constexpr uint32_t kBlockDataNum = 512 / sizeof(DTYPE_INPUT);
    uint32_t bigCoreDataNum = smallCoreDataNum + kBlockDataNum;

    uint32_t coreIdx = GetBlockIdx();
    uint32_t coreDataNum;
    uint32_t globalOffset;

    if (coreIdx < tailBlockNum) {
        coreDataNum = bigCoreDataNum;
        globalOffset = bigCoreDataNum * coreIdx;
    } else {
        coreDataNum = smallCoreDataNum;
        globalOffset = bigCoreDataNum * tailBlockNum + smallCoreDataNum * (coreIdx - tailBlockNum);
    }

    KernelAtanh<DTYPE_INPUT> op;
    op.Init(input, output, coreDataNum, globalOffset, tileLength);
    op.Process();
}
