// AscendC ClipByValue Kernel: DoubleBuffer隐藏DMA延迟 + 32B对齐搬运 + 多核切分
#include "kernel_operator.h"

#include "clip_by_value_tiling.h"
#include "tiling_key_clip_by_value.h"

constexpr int32_t DBUF = 2;
constexpr uint32_t ALIGN_SZ = 32;

template <class DT_X>
class KernelClipByValue {
public:
    __aicore__ inline KernelClipByValue() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t totalLength, uint32_t tileLength,
                                float min, float max, uint32_t fastPath) {
        this->lo = min;
        this->hi = max;
        // 每32B含几个元素: half=16, float/int32=8 — 用来判断对齐走快慢路径
        uint32_t epb = ALIGN_SZ / static_cast<uint32_t>(sizeof(DT_X));
        this->epb = epb;

        // 快速路径: 单核单tile兜住全部数据，跳过分核和对齐逻辑
        if (fastPath) {
            this->blk_len = totalLength;
            this->tile_len = totalLength;
            this->n_tile = 1;
            this->tail_len = totalLength;
            xGm.SetGlobalBuffer((__gm__ DT_X *)x, totalLength);
            yGm.SetGlobalBuffer((__gm__ DT_X *)y, totalLength);
            pipe.InitBuffer(inQ, DBUF, totalLength * sizeof(DT_X));
            pipe.InitBuffer(outQ, DBUF, totalLength * sizeof(DT_X));
            return;
        }

        // --- 多核切分: 各核offset必须32B对齐 ---
        uint32_t ncores = AscendC::GetBlockNum();
        uint32_t cid = AscendC::GetBlockIdx();

        // 每核元素数上取整到epb
        uint32_t per = (totalLength + ncores - 1) / ncores;
        per = ((per + epb - 1) / epb) * epb;

        uint32_t off = cid * per;
        if (off >= totalLength) {
            this->blk_len = 0;
            return;
        }
        uint32_t fin = off + per;
        if (fin > totalLength) {
            fin = totalLength;
        }
        this->blk_len = fin - off;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + off, this->blk_len);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + off, this->blk_len);

        // --- 核内tile切分 ---
        if (tileLength >= this->blk_len) {
            this->tile_len = this->blk_len;
            this->n_tile = 1;
            this->tail_len = this->blk_len;
        } else {
            this->tile_len = tileLength;
            this->n_tile = this->blk_len / tileLength;
            // 乘减代替取模，标量单元上更快
            this->tail_len = this->blk_len - this->n_tile * tileLength;
            if (this->tail_len) {
                this->n_tile += 1;
            } else {
                this->tail_len = tileLength;
            }
        }

        // 输入输出各2块(DoubleBuffer)，框架管理ping-pong
        pipe.InitBuffer(inQ, DBUF, this->tile_len * sizeof(DT_X));
        pipe.InitBuffer(outQ, DBUF, this->tile_len * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (!this->blk_len) {
            return;
        }
        // 满tile先跑 (n_tile-1) 个，尾巴单独处理 — 循环内零分支
        int32_t full = static_cast<int32_t>(this->n_tile) - 1;
        for (int32_t t = 0; t < full; t++) {
            CopyIn(t, this->tile_len);
            Compute(this->tile_len);
            CopyOut(t, this->tile_len);
        }
        CopyIn(full, this->tail_len);
        Compute(this->tail_len);
        CopyOut(full, this->tail_len);
    }

private:
    // clamp: y = min(max(x, lo), hi)
    // int32边界向零截断后直接在整型域裁剪，避免经float导致|x|>2^24精度丢失
    __aicore__ inline void Compute(uint32_t len) {
        AscendC::LocalTensor<DT_X> src = inQ.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> dst = outQ.AllocTensor<DT_X>();

        if constexpr (std::is_same_v<DT_X, int32_t>) {
            int32_t ilo = static_cast<int32_t>(this->lo);
            int32_t ihi = static_cast<int32_t>(this->hi);
            AscendC::Maxs(dst, src, ilo, len);
            AscendC::Mins(dst, dst, ihi, len);
        } else {
            DT_X slo = static_cast<DT_X>(this->lo);
            DT_X shi = static_cast<DT_X>(this->hi);
            AscendC::Maxs(dst, src, slo, len);
            AscendC::Mins(dst, dst, shi, len);
        }

        outQ.EnQue<DT_X>(dst);
        inQ.FreeTensor(src);
    }

    // GM → LM: 搬入。对齐走DataCopy，非对齐尾巴走DataCopyPad
    __aicore__ inline void CopyIn(int32_t step, uint32_t len) {
        AscendC::LocalTensor<DT_X> loc = inQ.AllocTensor<DT_X>();
        uint64_t off = static_cast<uint64_t>(step) * this->tile_len;
        if (!(len % this->epb)) {
            AscendC::DataCopy(loc, xGm[off], len);
        } else {
            AscendC::DataCopyExtParams cp;
            cp.blockCount = 1;
            cp.blockLen = len * static_cast<uint32_t>(sizeof(DT_X));
            cp.srcStride = 0;
            cp.dstStride = 0;
            AscendC::DataCopyPadExtParams<DT_X> pp;
            pp.isPad = false;
            pp.leftPadding = 0;
            pp.rightPadding = 0;
            pp.paddingValue = 0;
            AscendC::DataCopyPad(loc, xGm[off], cp, pp);
        }
        inQ.EnQue(loc);
    }

    // LM → GM: 搬出。逻辑同搬入，对齐走快路径省时间
    __aicore__ inline void CopyOut(int32_t step, uint32_t len) {
        AscendC::LocalTensor<DT_X> loc = outQ.DeQue<DT_X>();
        uint64_t off = static_cast<uint64_t>(step) * this->tile_len;
        if (!(len % this->epb)) {
            AscendC::DataCopy(yGm[off], loc, len);
        } else {
            AscendC::DataCopyExtParams cp;
            cp.blockCount = 1;
            cp.blockLen = len * static_cast<uint32_t>(sizeof(DT_X));
            cp.srcStride = 0;
            cp.dstStride = 0;
            AscendC::DataCopyPad(yGm[off], loc, cp);
        }
        outQ.FreeTensor(loc);
    }

private:
    // 标量
    float lo, hi;
    uint32_t epb;
    uint32_t blk_len, tile_len, n_tile, tail_len;
    // 硬件对象
    AscendC::GlobalTensor<DT_X> xGm, yGm;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, DBUF> inQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, DBUF> outQ;
};

template <typename DT_X>
__global__ __aicore__ void clip_by_value(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ClipByValueTilingData);
    GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, td, tiling);
    KernelClipByValue<DT_X> op;
    op.Init(x, y, td.totalLength, td.tileLength, td.min, td.max, td.fastPath);
    op.Process();
}
