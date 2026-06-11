// y = input + x1 * x2 * value   kernel侧
// fp16/fp32/int32/int8, numpy广播, 非32B对齐地址
//
// 计算模式:
//   mode=0 — 快速路径, 所有输入为标量或与输出同形, 各核均分连续区间
//   mode=1 — 广播路径, 输出视为 nRows×inner 2D矩阵, 两级分核
//
// 性能优化要点:
//   - 双缓冲 (depth=2): MTE搬下一个tile时Vector算当前tile, 隐藏GM延迟
//   - tile最大化: 从UB容量反推, 不设上限, 减少循环轮次→减少指令发射
//   - 编译期分支消去: if constexpr + 广播组合→最少向量指令路径
//   - int8分离: float中间计算防溢出, buffer复用省UB
//   - 二级核分配: 行多→分整行(连续搬), 行少→行内切(细粒度并行)
//   - 标量×向量指令(Muls/Adds)优先于向量×向量(Mul/Add), 前者吞吐更高

#include "kernel_operator.h"

#include "addcmul_tiling.h"
#include "tiling_key_addcmul.h"

using namespace AscendC;

// AscendC向量引擎不支持int8算术指令, int8必须提升到float中间计算再量化回int8.
// CompTy特化让kernel代码通过CT统一处理, if constexpr在编译期消掉分支.
template <typename T> struct CompTy { using type = T; };
template <> struct CompTy<int8_t> { using type = float; };

template <class T>
class KernelAddcmul {
public:
    using CT = typename CompTy<T>::type;
    static constexpr bool kI8 = IsSameType<T, int8_t>::value;

    __aicore__ inline KernelAddcmul() {}

    __aicore__ inline void Init(GM_ADDR inp, GM_ADDR x1, GM_ADDR x2,
                                GM_ADDR val, GM_ADDR y, const AddcmulTilingData &td) {

        gmIn_.SetGlobalBuffer((__gm__ T *)inp);
        gmX1_.SetGlobalBuffer((__gm__ T *)x1);
        gmX2_.SetGlobalBuffer((__gm__ T *)x2);
        gmVal_.SetGlobalBuffer((__gm__ T *)val);
        gmY_.SetGlobalBuffer((__gm__ T *)y);

        // 先拿模式相关的, 再拿buffer相关的 — 顺便做ADDCMUL_MAX_DIM拷贝
        mode_   = td.mode;
        nelts_  = td.totalLength;
        tile_   = td.tileLength;

        inBc_   = td.lastBcIn;
        x1Bc_   = td.lastBcX1;
        x2Bc_   = td.lastBcX2;
        inScl_  = td.scalarIn;
        x1Scl_  = td.scalarX1;
        x2Scl_  = td.scalarX2;

        inner_  = td.lastDim;
        nRows_  = td.outerSize;
        nOuter_ = td.outerCount;

        pipe_.InitBuffer(qScalar_, 1, 32);

        // 双缓冲depth=2: Vector算当前tile时MTE搬下一个, 乒乓节奏
        // depth再大收益递减且多占UB
        if constexpr (kI8) {
            // int8: 双缓冲io + float计算buf + half中转
            // UB节省: cX1_/cX2_在wrapI8()里临时复用, 省额外分配
            pipe_.InitBuffer(bHalf_,   tile_ * sizeof(half));
            pipe_.InitBuffer(bX2_,     tile_ * sizeof(CT));
            pipe_.InitBuffer(bX1_,     tile_ * sizeof(CT));
            pipe_.InitBuffer(bIn_,     tile_ * sizeof(CT));
            pipe_.InitBuffer(qOut_, 2, tile_ * sizeof(T));
            pipe_.InitBuffer(qIn_,  2, tile_ * sizeof(T));
            hf_   = bHalf_.Get<half>();
            cX2_  = bX2_.Get<CT>();
            cX1_  = bX1_.Get<CT>();
            cIn_  = bIn_.Get<CT>();
        } else {
            // 非int8: 直接在队列上算, 3输入+1输出全depth=2
            pipe_.InitBuffer(qOut_, 2, tile_ * sizeof(T));
            pipe_.InitBuffer(qX2_,  2, tile_ * sizeof(T));
            pipe_.InitBuffer(qX1_,  2, tile_ * sizeof(T));
            pipe_.InitBuffer(qIn_,  2, tile_ * sizeof(T));
        }

        // 数组拷贝放最后 — 和上面没依赖, 放哪都行
        uint32_t i = 0;
        do {
            oshape_[i] = td.outerShape[i];
            stIn_[i]   = td.strideIn[i];
            stX1_[i]   = td.strideX1[i];
            stX2_[i]   = td.strideX2[i];
            ++i;
        } while (i < ADDCMUL_MAX_DIM);
    }

    __aicore__ inline void Process() {
        if (nelts_ == 0) return;

        val_ = loadScalar(gmVal_, 0);

        if (mode_ == 0) { doFast(); } else { doRow(); }
    }

private:
    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }

    __aicore__ inline void copyIn(const LocalTensor<T> &dst, const GlobalTensor<T> &gm,
                                  uint32_t off, uint32_t n) {
        DataCopyExtParams ext{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(dst, gm[off], ext, pad);
    }
    __aicore__ inline void copyOut(const GlobalTensor<T> &gm, uint32_t off,
                                   const LocalTensor<T> &src, uint32_t n) {
        DataCopyExtParams ext{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
        DataCopyPad(gm[off], src, ext);
    }

    // half标量乘/加必须升float算再截断 — AscendC标量单元对half支持不全
    // 向量单元(Muls/Adds/Mul/Add)不需要这个
    __aicore__ inline CT s_mul(CT a, CT b) {
        if constexpr (IsSameType<CT, half>::value)
            return static_cast<half>(static_cast<float>(a) * static_cast<float>(b));
        else
            return static_cast<CT>(a * b);
    }
    __aicore__ inline CT s_add(CT a, CT b) {
        if constexpr (IsSameType<CT, half>::value)
            return static_cast<half>(static_cast<float>(a) + static_cast<float>(b));
        else
            return static_cast<CT>(a + b);
    }

    // 从GM读单个标量 — 通过Scalar队列中转保证MTE写完再取
    __aicore__ inline CT loadScalar(const GlobalTensor<T> &gm, uint32_t off) {
        LocalTensor<T> tmp = qScalar_.AllocTensor<T>();
        DataCopyExtParams ext{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(tmp, gm[off], ext, pad);
        qScalar_.EnQue(tmp);
        tmp = qScalar_.DeQue<T>();
        T raw = tmp.GetValue(0);
        qScalar_.FreeTensor(tmp);
        if constexpr (kI8) return static_cast<float>(raw);
        else               return raw;
    }

    // int8→half→float 两级Cast — Cast指令不支持int8↔float直转(硬件ISA限制)
    __aicore__ inline void loadI8(const LocalTensor<float> &dst, const GlobalTensor<T> &gm,
                                  uint32_t off, uint32_t n) {
        LocalTensor<T> buf = qIn_.AllocTensor<T>();
        copyIn(buf, gm, off, n);
        qIn_.EnQue(buf);
        buf = qIn_.DeQue<T>();
        Cast(hf_, buf, RoundMode::CAST_NONE, n);
        Cast(dst, hf_, RoundMode::CAST_NONE, n);
        qIn_.FreeTensor(buf);
    }

    // 补码回绕到int8: w = x - 256 × floor((x + 128) / 256)
    // 把任意浮点值以256为模折叠到[-128,127], 模拟int8溢出
    // 复用cX1_/cX2_做临时buffer, ReinterpretCast改类型不占新UB
    __aicore__ inline void wrapI8(uint32_t n) {
        LocalTensor<float> rf = cIn_;
        Adds(cX1_, rf, 128.0f, n);
        Muls(cX1_, cX1_, static_cast<float>(1.0f / 256.0f), n);
        LocalTensor<int32_t> tmp_i = cX2_.template ReinterpretCast<int32_t>();
        Cast(tmp_i, cX1_, RoundMode::CAST_FLOOR, n);
        Cast(cX1_, tmp_i, RoundMode::CAST_NONE, n);
        Muls(cX1_, cX1_, 256.0f, n);
        Sub(rf, rf, cX1_, n);
    }

    // ---- 非int8计算 ----
    // 按(dX1,dX2,dIn)8种组合选最少向量指令.
    // Muls/Adds(标量×向量)吞吐高于Mul/Add(向量×向量), 因为标量在寄存器不需要多读一个向量端口.
    __aicore__ inline void calcNormal(uint32_t oOff, uint32_t n,
                                      bool dIn, uint32_t iOff, CT si,
                                      bool dX1, uint32_t x1Off, CT sx1,
                                      bool dX2, uint32_t x2Off, CT sx2) {
        LocalTensor<T> tIn, tX1, tX2;
        const CT v = val_;

        // 三个独立DMA引擎并行搬, 广播的跳过
        if (!dX2) { tX2 = qX2_.AllocTensor<T>();   copyIn(tX2, gmX2_, x2Off, n);  qX2_.EnQue(tX2);   }
        if (!dX1) { tX1 = qX1_.AllocTensor<T>();   copyIn(tX1, gmX1_, x1Off, n);  qX1_.EnQue(tX1);   }
        if (!dIn) { tIn = qIn_.AllocTensor<T>();   copyIn(tIn, gmIn_, iOff, n);   qIn_.EnQue(tIn);   }

        if (!dIn) tIn = qIn_.DeQue<T>();
        if (!dX1) tX1 = qX1_.DeQue<T>();
        if (!dX2) tX2 = qX2_.DeQue<T>();

        LocalTensor<T> tOut = qOut_.AllocTensor<T>();

        if (dX1 && dX2) {
            CT coef = s_mul(s_mul(sx1, sx2), v);
            if (dIn) Duplicate(tOut, s_add(si, coef), n);
            else     Adds(tOut, tIn, coef, n);
        } else if (dX2) {
            Muls(tOut, tX1, sx2, n);  Muls(tOut, tOut, v, n);
            if (dIn) Adds(tOut, tOut, si, n);  else Add(tOut, tOut, tIn, n);
        } else if (dX1) {
            Muls(tOut, tX2, sx1, n);  Muls(tOut, tOut, v, n);
            if (dIn) Adds(tOut, tOut, si, n);  else Add(tOut, tOut, tIn, n);
        } else {
            // x1,x2都要搬 — 复用tX1做累加器省一次AllocTensor
            Mul(tX1, tX1, tX2, n);  Muls(tX1, tX1, v, n);
            if (dIn) Adds(tOut, tX1, si, n);  else Add(tOut, tIn, tX1, n);
        }

        // 尽早释放 → 下个tile搬运可提前排队
        if (!dX2) qX2_.FreeTensor(tX2);
        if (!dX1) qX1_.FreeTensor(tX1);
        if (!dIn) qIn_.FreeTensor(tIn);

        qOut_.EnQue(tOut);
        tOut = qOut_.DeQue<T>();
        copyOut(gmY_, oOff, tOut, n);
        qOut_.FreeTensor(tOut);
    }

    // ---- int8计算 ----
    // 数据流: GM(int8)→Cast→half→Cast→float→计算→wrapI8→Cast→half→Cast→int8→GM
    // 绕的原因: (1)没int8算术 (2)Cast不支持int8↔float (3)float结果可能超范围
    __aicore__ inline void calcI8(uint32_t oOff, uint32_t n,
                                  bool dIn, uint32_t iOff, CT si,
                                  bool dX1, uint32_t x1Off, CT sx1,
                                  bool dX2, uint32_t x2Off, CT sx2) {
        if (!dX2) loadI8(cX2_, gmX2_, x2Off, n);
        if (!dX1) loadI8(cX1_, gmX1_, x1Off, n);
        if (!dIn) loadI8(cIn_, gmIn_, iOff, n);

        const float v = val_;

        if (dX1 && dX2) {
            float coef = sx1 * sx2 * v;
            if (dIn) Duplicate(cIn_, si + coef, n);
            else     Adds(cIn_, cIn_, coef, n);
        } else if (!dX1 && !dX2) {
            Mul(cX1_, cX1_, cX2_, n); Muls(cX1_, cX1_, v, n);
            if (dIn) Adds(cIn_, cX1_, si, n);  else Add(cIn_, cIn_, cX1_, n);
        } else if (dX1) {
            Muls(cX1_, cX2_, sx1 * v, n);
            if (dIn) Adds(cIn_, cX1_, si, n);  else Add(cIn_, cIn_, cX1_, n);
        } else {
            Muls(cX1_, cX1_, sx2 * v, n);
            if (dIn) Adds(cIn_, cX1_, si, n);  else Add(cIn_, cIn_, cX1_, n);
        }

        wrapI8(n);

        LocalTensor<T> o = qOut_.AllocTensor<T>();
        Cast(hf_, cIn_, RoundMode::CAST_NONE, n);
        Cast(o, hf_, RoundMode::CAST_RINT, n);
        qOut_.EnQue(o);
        o = qOut_.DeQue<T>();
        copyOut(gmY_, oOff, o, n);
        qOut_.FreeTensor(o);
    }

    __aicore__ inline void calcStore(uint32_t oOff, uint32_t n,
                                     bool dIn, uint32_t iOff, CT si,
                                     bool dX1, uint32_t x1Off, CT sx1,
                                     bool dX2, uint32_t x2Off, CT sx2) {
        if constexpr (kI8)
            calcI8(oOff, n, dIn, iOff, si, dX1, x1Off, sx1, dX2, x2Off, sx2);
        else
            calcNormal(oOff, n, dIn, iOff, si, dX1, x1Off, sx1, dX2, x2Off, sx2);
    }

    // ---- 快速路径: 无广播均匀分核 ----
    __aicore__ inline void doFast() {
        uint32_t tid = GetBlockIdx();
        uint32_t n   = GetBlockNum();

        CT sx2 = x2Scl_ ? loadScalar(gmX2_, 0) : static_cast<CT>(0);
        CT sx1 = x1Scl_ ? loadScalar(gmX1_, 0) : static_cast<CT>(0);
        CT si  = inScl_ ? loadScalar(gmIn_, 0) : static_cast<CT>(0);

        uint32_t blk  = nelts_ / n;
        uint32_t xtra = nelts_ % n;
        uint32_t beg  = tid * blk + (tid < xtra ? tid : xtra);
        uint32_t nLoc = blk + (tid < xtra ? 1 : 0);

        uint32_t off = 0;
        while (off < nLoc) {
            uint32_t sz   = Min(tile_, nLoc - off);
            uint32_t gOff = beg + off;
            calcStore(gOff, sz, inScl_, gOff, si, x1Scl_, gOff, sx1, x2Scl_, gOff, sx2);
            off += tile_;
        }
    }

    // 处理 [rBeg, rBeg+nRow) 连续行, 每行内按 cIdx/cTot 切inner
    // 行间地址: 线性行号→多维坐标(row-major), coord×stride累加得基地址
    __aicore__ inline void doRowRange(uint32_t rBeg, uint32_t nRow,
                                      uint32_t cIdx, uint32_t cTot) {
        uint32_t cSz  = (inner_ + cTot - 1) / cTot;
        uint32_t cBeg = cIdx * cSz;
        uint32_t cEnd = Min(cBeg + cSz, inner_);

        bool simple = (nOuter_ == 1);

        uint32_t ri = 0;
        while (ri < nRow) {
            uint32_t r = rBeg + ri;
            ++ri;

            uint32_t aIn = 0, aX1 = 0, aX2 = 0;
            if (simple) {
                aX2 = r * stX2_[0];  aX1 = r * stX1_[0];  aIn = r * stIn_[0];
            } else {
                uint32_t t = r;
                int d = static_cast<int>(nOuter_) - 1;
                while (d >= 0) {
                    uint32_t sz  = oshape_[d];
                    uint32_t crd = t % sz;
                    t /= sz;
                    aIn  += crd * stIn_[d];
                    aX1 += crd * stX1_[d];
                    aX2 += crd * stX2_[d];
                    --d;
                }
            }

            CT sx2  = x2Bc_ ? loadScalar(gmX2_, aX2) : static_cast<CT>(0);
            CT sx1  = x1Bc_ ? loadScalar(gmX1_, aX1) : static_cast<CT>(0);
            CT sin  = inBc_ ? loadScalar(gmIn_,  aIn)  : static_cast<CT>(0);

            uint32_t base = r * inner_;
            uint32_t cc = cBeg;
            while (cc < cEnd) {
                uint32_t sz = Min(tile_, cEnd - cc);
                calcStore(base + cc, sz, inBc_, aIn+cc, sin, x1Bc_, aX1+cc, sx1, x2Bc_, aX2+cc, sx2);
                cc += tile_;
            }
        }
    }

    // ---- 广播路径: nRows×inner 2D分核 ----
    // nRows<n→行内多核切inner; nRows≥n→每核若干整行
    __aicore__ inline void doRow() {
        uint32_t tid = GetBlockIdx();
        uint32_t n   = GetBlockNum();

        if (nRows_ < n) {
            uint32_t cpR = n / nRows_;
            uint32_t myR = tid / cpR;
            uint32_t myC = tid % cpR;
            if (myR >= nRows_) return;
            doRowRange(myR, 1, myC, cpR);
            return;
        }

        uint32_t each = nRows_ / n;
        uint32_t xtra = nRows_ % n;
        uint32_t rBeg = tid * each + (tid < xtra ? tid : xtra);
        uint32_t nRow = each + (tid < xtra ? 1 : 0);
        if (nRow == 0) return;
        doRowRange(rBeg, nRow, 0, 1);
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECOUT, 2> qOut_;
    TQue<TPosition::VECIN, 1>  qScalar_;
    TQue<TPosition::VECIN, 2>  qX2_, qX1_, qIn_;
    TBuf<TPosition::VECCALC>   bHalf_, bX2_, bX1_, bIn_;

    CT val_;

    LocalTensor<half> hf_;
    LocalTensor<CT>   cX2_, cX1_, cIn_;

    GlobalTensor<T> gmY_, gmVal_, gmX2_, gmX1_, gmIn_;

    uint32_t mode_, nelts_, tile_;
    uint32_t inner_, nRows_, nOuter_;
    uint32_t inBc_, x1Bc_, x2Bc_, inScl_, x1Scl_, x2Scl_;
    uint32_t stX2_[ADDCMUL_MAX_DIM], stX1_[ADDCMUL_MAX_DIM], stIn_[ADDCMUL_MAX_DIM];
    uint32_t oshape_[ADDCMUL_MAX_DIM];
};

template <typename DT>
__global__ __aicore__ void addcmul(GM_ADDR inp, GM_ADDR x1, GM_ADDR x2, GM_ADDR val,
                                   GM_ADDR y, GM_ADDR ws, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(AddcmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddcmulTilingData, td, tiling);
    KernelAddcmul<DT> op;
    op.Init(inp, x1, x2, val, y, td);
    op.Process();
}
