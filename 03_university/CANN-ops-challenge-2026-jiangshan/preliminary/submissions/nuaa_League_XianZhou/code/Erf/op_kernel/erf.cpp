// Kernel 侧核函数实现
//
// 改造说明（与 op_host/erf.cpp 配套）：
//   A. 自适应铺核：Host 在 mode/perCore/tailCoreLen 中已完成档位决策；
//   B. 小张量关闭双缓冲：mode==0 走精简路径（TBuf + 手写 Set/Wait Flag）；
//   D. 重复标量下放：tile 切分、末核长度、对齐尾部全部从 tiling 读取；
//   C. tanh 近似替代 AscendC::Erf 高阶 API：
//      erf(x) ≈ tanh((2/√π) · x · (1 + 0.08943·x²))
//      6 个矢量指令，无 cancellation，全值域 < 6e-4 误差。
//   E. SMALL 路径去 TQue 化：单 TBuf + Set/Wait Flag 手动同步，
//      省去 TQue 的 Alloc/EnQue/DeQue/Free 4 次调用 + 多次 InitBuffer 开销。
//      只对 mode=0 (SMALL/MEDIUM) 生效；mode=1 (LARGE) 保持双缓冲 TQue 流水。
#include "kernel_operator.h"
 
#include "erf_tiling.h"
#include "tiling_key_erf.h"
 
using namespace AscendC;
 
// ============================================================================
// Erf 多项式实现（minimax 5 阶 + 2.5 clamp）
//   erf(x) ≈ x_clamped · (c0 + c1·z + c2·z² + c3·z³ + c4·z⁴ + c5·z⁵)
//   其中 z = x_clamped², x_clamped = clamp(x, -2.5, 2.5)
//
//   语义（in-place）：xy 同时担任 input 和 output；y_scratch、x2_scratch 是 scratch
//
//   精度：clamp 边界 x=±2.5 误差 ~2e-4；饱和区 (|x|>2.5) clamp 后输出 ≈0.9998，
//        而真实 erf(3)=0.99998 / erf(10)≈1.0，绝对误差 ~2.3e-4，落在 1e-3 阈值内
//
//   ops 数：14（Maxs Mins Mul Muls + 5×Adds + 5×Mul + Mul），全 FMA-style，
//          无 transcendental，但比 Tanh 路径多 9 op
// ============================================================================
template <class DT_X>
__aicore__ inline void EvalErfPolynomial(LocalTensor<DT_X> &xy,
                                          LocalTensor<DT_X> &y_scratch,
                                          LocalTensor<DT_X> &x2_scratch,
                                          uint32_t n)
{
    const DT_X kClampLo = static_cast<DT_X>(-2.50f);
    const DT_X kClampHi = static_cast<DT_X>( 2.50f);
    const DT_X kPolyC0  = static_cast<DT_X>( 1.12773312f);
    const DT_X kPolyC1  = static_cast<DT_X>(-3.69239590e-1f);
    const DT_X kPolyC2  = static_cast<DT_X>( 1.00451917e-1f);
    const DT_X kPolyC3  = static_cast<DT_X>(-1.78319906e-2f);
    const DT_X kPolyC4  = static_cast<DT_X>( 1.79231248e-3f);
    const DT_X kPolyC5  = static_cast<DT_X>(-7.60648791e-5f);
 
    // clamp x ∈ [-2.5, 2.5]，in-place 修改 xy
    Maxs(xy, xy, kClampLo, n);
    Mins(xy, xy, kClampHi, n);
 
    // x2_scratch = x_clamped²
    Mul(x2_scratch, xy, xy, n);
 
    // Horner: y_scratch = (((((c5·z + c4)·z + c3)·z + c2)·z + c1)·z + c0)
    Muls(y_scratch, x2_scratch, kPolyC5, n);
    Adds(y_scratch, y_scratch, kPolyC4, n);
    Mul (y_scratch, y_scratch, x2_scratch, n);
    Adds(y_scratch, y_scratch, kPolyC3, n);
    Mul (y_scratch, y_scratch, x2_scratch, n);
    Adds(y_scratch, y_scratch, kPolyC2, n);
    Mul (y_scratch, y_scratch, x2_scratch, n);
    Adds(y_scratch, y_scratch, kPolyC1, n);
    Mul (y_scratch, y_scratch, x2_scratch, n);
    Adds(y_scratch, y_scratch, kPolyC0, n);
 
    // 最终：xy = y_scratch · x_clamped（in-place 写回 xy）
    Mul(xy, y_scratch, xy, n);
}
 
// ============================================================================
// 手写 Erf 核心：tanh-based 近似（保留供 LARGE 路径使用）
//   erf(x) ≈ tanh(A · x · (1 + B · x²))
//   A = 2/√π ≈ 1.1283791671
//   B = 0.08943（GELU 反推）
// ============================================================================
template <class DT_X>
__aicore__ inline void ComputeErfTanh(LocalTensor<DT_X> &yLocal,
                                      LocalTensor<DT_X> &xLocal,
                                      LocalTensor<DT_X> &tmp1,
                                      uint32_t len)
{
    // 数学折叠：A·x·(1+B·x²) = x·(A + AB·x²)
    //   旧：6 op  (Mul Muls Adds Muls Mul Tanh)
    //   新：5 op  (Mul Muls Adds Mul Tanh)
    // 把 A·B 这个常数预算好，省一条 Muls。
    // 注：yLocal 与 xLocal 允许 alias（SMALL 路径 in-place 用），
    //     Mul/Tanh 标准 in-place 语义安全。
    const DT_X A  = static_cast<DT_X>(1.1283791671f);              // 2/√π
    const DT_X AB = static_cast<DT_X>(0.1009109489f);              // A × 0.08943
 
    Mul (tmp1, xLocal, xLocal, len);    // tmp1 = x²
    Muls(tmp1, tmp1, AB, len);          // tmp1 = AB·x²
    Adds(tmp1, tmp1, A,  len);          // tmp1 = A + AB·x²
    Mul (yLocal, xLocal, tmp1, len);    // yLocal = x · (A + AB·x²)
    Tanh(yLocal, yLocal, len);
}
 
// ============================================================================
// SMALL / MEDIUM 入口：去 TQue 化的精简路径
//   - 单次 InitBuffer 申请 3·len 字节的 UB，切成 in/out/tmp 三个 LocalTensor
//   - 用固定 EVENT_ID0 + Set/Wait Flag 做 MTE2→V→MTE3 跨 pipe 同步
//   - 单 tile 直接串行：DataCopy → Compute → DataCopy，无循环
// ============================================================================
template <class DT_X>
__aicore__ inline void RunErfSmall(GM_ADDR x, GM_ADDR y, const ErfTilingData &td) {
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockNum = GetBlockNum();
    bool     isLast   = (blockIdx + 1u == blockNum);
    uint32_t len      = isLast ? td.tailCoreLen : td.perCore;
 
    if (len == 0u) return;
 
    uint32_t start = blockIdx * td.perCore;
 
    GlobalTensor<DT_X> xGM, yGM;
    xGM.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + start, len);
    yGM.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + start, len);
 
    // UB 布局：xy (in-place 输入输出) + y_scratch + x2_scratch
    //   多项式需要 3 buffer（比 Tanh 多 1 个 x² scratch）
    //   每核 UB = 3·len·4。len ≤ INNER_TILE = 8192 → UB ≤ 96KB ✓
    TPipe pipe;
    TBuf<TPosition::VECCALC> bigBuf;
    pipe.InitBuffer(bigBuf, 3u * len * sizeof(DT_X));
 
    LocalTensor<DT_X> ub         = bigBuf.template Get<DT_X>();
    LocalTensor<DT_X> xy         = ub;                  // in-place input/output
    LocalTensor<DT_X> y_scratch  = ub[len];             // Horner 多项式累加器
    LocalTensor<DT_X> x2_scratch = ub[2u * len];        // x² 中间值
 
    constexpr event_t EID = EVENT_ID0;
 
    // ── Stage 1：GM → UB ──────────────────────────────────────────────
    DataCopy(xy, xGM, len);
 
    // ── 同步：等 MTE2 完成，再让 V 读 xy ─────────────────────────────
    // 注：试过 pipe_barrier(PIPE_MTE2)，跨 pipe 不能正确同步，96% WA。
    SetFlag<HardEvent::MTE2_V>(EID);
    WaitFlag<HardEvent::MTE2_V>(EID);
 
    // ── Stage 2：Compute（多项式，in-place 写回 xy）──────────────────
    EvalErfPolynomial(xy, y_scratch, x2_scratch, len);
 
    // ── 同步：等 V 完成，再让 MTE3 读 xy（已存了结果）───────────────
    SetFlag<HardEvent::V_MTE3>(EID);
    WaitFlag<HardEvent::V_MTE3>(EID);
 
    // ── Stage 3：UB → GM ──────────────────────────────────────────────
    DataCopy(yGM, xy, len);
}
 
// ============================================================================
// SMALL_PIPELINED (mode=2)：单核 + 2 tile + 双缓冲手工流水
//   适用范围：SMALL_NOPIPE_MAX < length ≤ SINGLE_CORE_MAX (即 2K-16K 区间)
//   假设：DataCopy 启动延迟 ≈ Compute 时间，把数据切两半让两条 pipe 并行
//
//   流水时间线（理想情况）：
//     T0   MTE2_A ────┐
//     T1   MTE2_B ────│──┐    V_A 等到 MTE2_A 完成后并行执行
//     T2                │  │    MTE3_A 等到 V_A 完成后并行执行
//     T3                │  V_B  MTE3_A 与 V_B 在不同 pipe 并行
//     T4                       MTE3_B 最后扫尾
//
//   UB 占用：3 × tile_length × 4B（buf_A + buf_B + tmp，A/B 各兼输入输出）
//   核心同步原语：EVENT_ID0 / EVENT_ID1 分别管 A、B 两个 tile 的跨 pipe sync
// ============================================================================
template <class DT_X>
__aicore__ inline void RunErfSmallPipelined(GM_ADDR x, GM_ADDR y, const ErfTilingData &td) {
    // 单核路径：blockIdx 恒为 0，无需读 GetBlockIdx/GetBlockNum
    uint32_t total_len = td.perCore;
    uint32_t tile1_len = td.tileLength;
    uint32_t tile2_len = total_len - tile1_len;     // 第二个 tile，host 保证 > 0 且 ALIGN_NUM 倍数
 
    if (total_len == 0u) return;
 
    GlobalTensor<DT_X> xGM, yGM;
    xGM.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), total_len);
    yGM.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), total_len);
 
    // UB 布局：buf_A | buf_B | y_scratch | x2_scratch
    //   buf_A / buf_B：两 tile 的 in-place 输入输出
    //   y_scratch / x2_scratch：多项式 scratch（两 tile 共用，因为同一时刻只算一个 tile）
    //   每块 tile1_len 元素，总 UB = 4 × tile1_len × 4。
    //   tile1_len ≤ ceil(16384/2) = 8192 → UB ≤ 128KB ✓
    TPipe pipe;
    TBuf<TPosition::VECCALC> bigBuf;
    pipe.InitBuffer(bigBuf, 4u * tile1_len * sizeof(DT_X));
 
    LocalTensor<DT_X> ub         = bigBuf.template Get<DT_X>();
    LocalTensor<DT_X> buf_A      = ub;
    LocalTensor<DT_X> buf_B      = ub[tile1_len];
    LocalTensor<DT_X> y_scratch  = ub[2u * tile1_len];
    LocalTensor<DT_X> x2_scratch = ub[3u * tile1_len];
 
    constexpr event_t EID_A = EVENT_ID0;
    constexpr event_t EID_B = EVENT_ID1;
 
    // ── T0：MTE2 启动 A 搬入 ─────────────────────────────────────────
    DataCopy(buf_A, xGM, tile1_len);
    SetFlag<HardEvent::MTE2_V>(EID_A);
 
    // ── T1：MTE2 启动 B 搬入（并行）  +  V 等 A 数据后开始算 ────────
    DataCopy(buf_B, xGM[tile1_len], tile2_len);
    SetFlag<HardEvent::MTE2_V>(EID_B);
 
    WaitFlag<HardEvent::MTE2_V>(EID_A);
    EvalErfPolynomial(buf_A, y_scratch, x2_scratch, tile1_len);   // 结果 in-place 写回 buf_A
    SetFlag<HardEvent::V_MTE3>(EID_A);
 
    // ── T2：MTE3 启动 A 搬出（并行）  +  V 等 B 数据后算 ────────────
    WaitFlag<HardEvent::V_MTE3>(EID_A);
    DataCopy(yGM, buf_A, tile1_len);
 
    WaitFlag<HardEvent::MTE2_V>(EID_B);
    EvalErfPolynomial(buf_B, y_scratch, x2_scratch, tile2_len);   // 复用 scratch，buf_B in-place
    SetFlag<HardEvent::V_MTE3>(EID_B);
 
    // ── T3：MTE3 搬出 B 收尾 ────────────────────────────────────────
    WaitFlag<HardEvent::V_MTE3>(EID_B);
    DataCopy(yGM[tile1_len], buf_B, tile2_len);
}
 
// ============================================================================
// LARGE：每核多 tile + 双缓冲，BUFFER_NUM = 2（保持不变，已贴近最优）
// ============================================================================
template <class DT_X>
class KernelErfPipeline {
    static constexpr int32_t BUFFER_NUM = 2;
 
public:
    __aicore__ inline KernelErfPipeline() {}
 
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ErfTilingData &td) {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();
        bool isLast       = (blockIdx + 1u == blockNum);
 
        tileLength_ = td.tileLength;
 
        if (isLast) {
            tileNum_        = td.tailCoreFullTiles;
            alignedTailLen_ = td.tailCoreAlignedTail;
        } else {
            tileNum_        = td.perCore / td.tileLength;
            alignedTailLen_ = 0u;
        }
 
        if (tileNum_ == 0u && alignedTailLen_ == 0u) {
            return;
        }
 
        uint32_t start   = blockIdx * td.perCore;
        uint32_t safeLen = tileNum_ * tileLength_ + alignedTailLen_;
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + start, safeLen);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + start, safeLen);
 
        pipe_.InitBuffer(inQ_,  BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(outQ_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(tmp1Buf_, tileLength_ * sizeof(DT_X));
    }
 
    __aicore__ inline void Process() {
        for (uint32_t i = 0u; i < tileNum_; i++) {
            uint32_t offset = i * tileLength_;
            CopyIn (offset, tileLength_);
            Compute(tileLength_);
            CopyOut(offset, tileLength_);
        }
        if (alignedTailLen_ > 0u) {
            uint32_t offset = tileNum_ * tileLength_;
            CopyIn (offset, alignedTailLen_);
            Compute(alignedTailLen_);
            CopyOut(offset, alignedTailLen_);
        }
    }
 
private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t len) {
        LocalTensor<DT_X> xLocal = inQ_.template AllocTensor<DT_X>();
        DataCopy(xLocal, xGM_[offset], len);
        inQ_.EnQue(xLocal);
    }
 
    __aicore__ inline void Compute(uint32_t len) {
        LocalTensor<DT_X> xLocal = inQ_.template DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQ_.template AllocTensor<DT_X>();
        LocalTensor<DT_X> tmp1   = tmp1Buf_.template Get<DT_X>();
        ComputeErfTanh(yLocal, xLocal, tmp1, len);
        outQ_.EnQue(yLocal);
        inQ_.FreeTensor(xLocal);
    }
 
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t len) {
        LocalTensor<DT_X> yLocal = outQ_.template DeQue<DT_X>();
        DataCopy(yGM_[offset], yLocal, len);
        outQ_.FreeTensor(yLocal);
    }
 
    GlobalTensor<DT_X> xGM_;
    GlobalTensor<DT_X> yGM_;
    TPipe pipe_;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQ_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQ_;
    TBuf<TPosition::VECCALC>              tmp1Buf_;
    uint32_t tileLength_     = 0u;
    uint32_t tileNum_        = 0u;
    uint32_t alignedTailLen_ = 0u;
};
 
// ============================================================================
// 入口：按 Host 给出的 mode 字段派发
// ============================================================================
template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y,
                               GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
 
    if (tiling_data.mode == 0u) {
        // VERY_SMALL（< SMALL_NOPIPE_MAX）或 MEDIUM（多核单 tile）
        RunErfSmall<DT_X>(x, y, tiling_data);
    } else if (tiling_data.mode == 2u) {
        // SMALL_PIPELINED：单核 2 tile 双缓冲流水
        RunErfSmallPipelined<DT_X>(x, y, tiling_data);
    } else {
        // LARGE：多核多 tile 双缓冲（mode == 1）
        KernelErfPipeline<DT_X> op;
        op.Init(x, y, tiling_data);
        op.Process();
    }
}