/* IsNan kernel —— y[i]=isnan(x[i])。x: fp16/bf16/fp32 → y: bool(int8 0/1)。
 * NaN 位模式：abs=bits&0x7FFF(或0x7FFFFFFF)，diff=abs-INF_bits；NaN⟺diff>0 → clamp[0,1]。
 *   fp16/bf16：int16 reinterpret + And + Adds + (int16→half clamp) + half→int8。
 *   fp32：fp Abs 清符号 + int32 Adds diff + int32→fp32 + clamp + fp32→half→int8。
 *   （整数域 clamp/Cast→int8 实测不可靠，clamp 必须经 half。）
 * 优化：TilingKey 按 dtype 分支（只加载用到的 kernel）；核长 ALN=32 对齐 + DataCopy；
 *      cmask Duplicate 移到 Init；大 case 才开 workspace flip（小 case flip 反增开销）。
 * TilingData 仅 {totalLen, tileLen}，分核从 GetBlockNum() 在 kernel 内重算。 */
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t FLIP_MIN_ELEM = 1u << 20;   // ≥1M 元素（load-bound）才开 flip
}

template <typename T>
class KernelIsNan {
public:
    __aicore__ inline KernelIsNan() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLen, uint32_t tileLen, TPipe* pipe)
    {
        tileLen_ = tileLen;
        const uint32_t ALN = 32;
        uint32_t coreNum = GetBlockNum();
        uint32_t bid = GetBlockIdx();
        uint32_t nBlk = totalLen / ALN;
        uint32_t tailExtra = totalLen % ALN;
        uint32_t perBlk = nBlk / coreNum;
        uint32_t remBlk = nBlk % coreNum;
        uint32_t bigCoreLen = (perBlk + 1) * ALN, smallCoreLen = perBlk * ALN;
        if (bid < remBlk) { coreLen_ = bigCoreLen; coreStart_ = bid * bigCoreLen; }
        else { coreLen_ = smallCoreLen; coreStart_ = remBlk * bigCoreLen + (bid - remBlk) * smallCoreLen; }
        if (bid == coreNum - 1) coreLen_ += tailExtra;   // 末核吃 <ALN 零头

        xGm_.SetGlobalBuffer((__gm__ T*)x + coreStart_, coreLen_);
        yGm_.SetGlobalBuffer((__gm__ int8_t*)y + coreStart_, coreLen_);
        pipe->InitBuffer(qIn_,  BUFFER_NUM, tileLen_ * sizeof(T));
        pipe->InitBuffer(qOut_, BUFFER_NUM, tileLen_ * sizeof(int8_t));
        pipe->InitBuffer(bufConst_, tileLen_ * sizeof(int16_t));
        pipe->InitBuffer(bufWork_,  tileLen_ * sizeof(float));
        pipe->InitBuffer(bufH_,     tileLen_ * sizeof(half));
        cmask_ = bufConst_.Get<int16_t>();
        Duplicate(cmask_, (int16_t)0x7FFF, tileLen_);    // 常量 abs-mask：只 Duplicate 一次
    }

    __aicore__ inline void Process(bool reverse)
    {
        if (coreLen_ == 0) return;
        uint32_t loop = coreLen_ / tileLen_;
        uint32_t tail = coreLen_ - loop * tileLen_;
        if (!reverse) {
            for (uint32_t i = 0; i < loop; ++i) ProcessTile(i * tileLen_, tileLen_);
            if (tail > 0) ProcessTile(loop * tileLen_, tail);
        } else {
            if (tail > 0) ProcessTile(loop * tileLen_, tail);
            for (int32_t i = (int32_t)loop - 1; i >= 0; --i)
                ProcessTile((uint32_t)i * tileLen_, tileLen_);
        }
    }

private:
    __aicore__ inline void ProcessTile(uint32_t off, uint32_t n)
    {
        constexpr uint32_t ALN_IN = 32 / sizeof(T);   // fp16/bf16=16, fp32=8
        LocalTensor<T> xL = qIn_.template AllocTensor<T>();
        if (n % ALN_IN == 0) {
            DataCopy(xL, xGm_[off], n);
        } else {
            DataCopyExtParams cpi{1, n * (uint32_t)sizeof(T), 0, 0, 0};
            DataCopyPadExtParams<T> pad{false, 0, 0, 0};
            DataCopyPad(xL, xGm_[off], cpi, pad);
        }
        qIn_.EnQue(xL);
        xL = qIn_.template DeQue<T>();

        LocalTensor<int8_t> yL = qOut_.template AllocTensor<int8_t>();
        if constexpr (sizeof(T) == 2) {
            const int16_t INF16 = IsSameType<T, half>::value ? (int16_t)0x7C00 : (int16_t)0x7F80;
            LocalTensor<int16_t> xi = xL.template ReinterpretCast<int16_t>();
            And(xi, xi, cmask_, n);                  // abs = bits & 0x7FFF
            Adds(xi, xi, (int16_t)(-INF16), n);      // diff：NaN>0, inf=0, 正常<0
            LocalTensor<half> hf = bufH_.Get<half>();
            Cast(hf, xi, RoundMode::CAST_RINT, n);
            Maxs(hf, hf, (half)0.0, n);
            Mins(hf, hf, (half)1.0, n);              // clamp[0,1]
            Cast(yL, hf, RoundMode::CAST_RINT, n);
        } else {
            LocalTensor<float> xf = xL.template ReinterpretCast<float>();
            Abs(xf, xf, n);                          // |x|：清符号
            LocalTensor<int32_t> xi = xL.template ReinterpretCast<int32_t>();
            Adds(xi, xi, -0x7F800000, n);            // diff
            LocalTensor<float> rf = bufWork_.Get<float>();
            Cast(rf, xi, RoundMode::CAST_RINT, n);   // int32→fp32（大值可表示）
            Maxs(rf, rf, 0.0f, n);
            Mins(rf, rf, 1.0f, n);
            LocalTensor<half> hf = bufH_.Get<half>();
            Cast(hf, rf, RoundMode::CAST_RINT, n);
            Cast(yL, hf, RoundMode::CAST_RINT, n);
        }
        qIn_.FreeTensor(xL);
        qOut_.EnQue(yL);

        yL = qOut_.template DeQue<int8_t>();
        DataCopy(yGm_[off], yL, (n + 31) / 32 * 32);   // int8 输出 32B 向上对齐
        qOut_.FreeTensor(yL);
    }

    GlobalTensor<T> xGm_;
    GlobalTensor<int8_t> yGm_;
    TQue<QuePosition::VECIN,  BUFFER_NUM> qIn_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> qOut_;
    TBuf<QuePosition::VECCALC> bufConst_, bufWork_, bufH_;
    LocalTensor<int16_t> cmask_;
    uint32_t tileLen_{0}, coreLen_{0}, coreStart_{0};
};

extern "C" __global__ __aicore__ void is_nan(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(t, tiling);
    // 大 case 才开 workspace flip（小 case flip 的 workGm 读写反增开销）
    const bool useFlip = (t.totalLen >= FLIP_MIN_ELEM);
    GlobalTensor<int32_t> workGm;
    workGm.SetGlobalBuffer((__gm__ int32_t*)workspace);
    const bool reverse = useFlip && ((workGm(0) % 2) != 0);
    TPipe pipe;
    if (TILING_KEY_IS(1)) {            // fp32
        KernelIsNan<float> op; op.Init(x, y, t.totalLen, t.tileLen, &pipe); op.Process(reverse);
    } else if (TILING_KEY_IS(2)) {     // fp16
        KernelIsNan<half> op; op.Init(x, y, t.totalLen, t.tileLen, &pipe); op.Process(reverse);
    } else if (TILING_KEY_IS(3)) {     // bf16
        KernelIsNan<bfloat16_t> op; op.Init(x, y, t.totalLen, t.tileLen, &pipe); op.Process(reverse);
    }
    if (useFlip && GetBlockIdx() == 0) workGm(0) = workGm(0) + 1;
}
