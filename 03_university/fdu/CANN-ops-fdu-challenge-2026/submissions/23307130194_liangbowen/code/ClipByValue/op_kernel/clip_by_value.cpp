#include "kernel_operator.h"
#include "clip_by_value_tiling.h"
#include "tiling_key_clip_by_value.h"

constexpr int32_t BUFFER_NUM = 1;

template <class DT_X>
class KernelClipByValue {
public:
    __aicore__ inline KernelClipByValue() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, ClipByValueTilingData &td)
    {
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t basePerCore = td.totalLength / blockNum;
        uint32_t tailCores = td.tailCoreNum;

        this->coreDataNum = basePerCore;
        if (coreIdx < tailCores) {
            this->coreDataNum = basePerCore + 1;
        }

        this->gmOffset = coreIdx * basePerCore + (coreIdx < tailCores ? coreIdx : tailCores);
        this->gmX = (__gm__ DT_X *)x;
        this->gmY = (__gm__ DT_X *)y;

        // 边界值设置
        if constexpr (sizeof(DT_X) == 2) {  
            this->minHalfBits = td.min_half_bits;
            this->maxHalfBits = td.max_half_bits;
        } else if constexpr (std::is_same_v<DT_X, float>) {
            this->minFloat = td.min_val;
            this->maxFloat = td.max_val;
        } else {  
            float minVal = td.min_val;
            float maxVal = td.max_val;
            if (minVal <= static_cast<float>(INT32_MIN)) {
                this->minInt = INT32_MIN;
            } else if (minVal >= static_cast<float>(INT32_MAX)) {
                this->minInt = INT32_MAX;
            } else if (minVal != minVal) { // NaN
                this->minInt = 0;
            } else {
                this->minInt = static_cast<int32_t>(minVal);
            }
            if (maxVal <= static_cast<float>(INT32_MIN)) {
                this->maxInt = INT32_MIN;
            } else if (maxVal >= static_cast<float>(INT32_MAX)) {
                this->maxInt = INT32_MAX;
            } else if (maxVal != maxVal) {
                this->maxInt = 0;
            } else {
                this->maxInt = static_cast<int32_t>(maxVal);
            }
        }

        this->tileDataNum = td.tileDataNum;

        uint32_t bufSize = (this->tileDataNum > 0) ? this->tileDataNum : this->coreDataNum;
        pipe.InitBuffer(inQueueX, BUFFER_NUM, bufSize * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, bufSize * sizeof(DT_X));

        
        if constexpr (std::is_same_v<DT_X, int32_t>) {
            pipe.InitBuffer(workBuf, bufSize * sizeof(int32_t));
        }
    }

    __aicore__ inline void Process()
    {
        if (this->coreDataNum == 0) {
            return;
        }

        uint32_t tileLen = this->tileDataNum;
        if (tileLen == 0 || tileLen >= this->coreDataNum) {
            tileLen = this->coreDataNum;
        }

        uint32_t tileCnt = (this->coreDataNum + tileLen - 1) / tileLen;

        for (uint32_t t = 0; t < tileCnt; t++) {
            uint32_t ts = t * tileLen;
            uint32_t cur = (ts + tileLen <= this->coreDataNum) ? tileLen : (this->coreDataNum - ts);
            CopyIn(ts, cur);
            Compute(cur);
            CopyOut(ts, cur);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t off, uint32_t len)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::GlobalTensor<DT_X> gtx;
        gtx.SetGlobalBuffer(this->gmX + this->gmOffset + off, len);
        AscendC::DataCopy(xLocal, gtx, len);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();

        if constexpr (sizeof(DT_X) == 2) {                    
            half min_h = *reinterpret_cast<const half*>(&this->minHalfBits);
            half max_h = *reinterpret_cast<const half*>(&this->maxHalfBits);
            AscendC::DataCopy(yLocal, xLocal, len);
            AscendC::Maxs(yLocal, yLocal, min_h, len);
            AscendC::Mins(yLocal, yLocal, max_h, len);
        } else if constexpr (std::is_same_v<DT_X, float>) {    // float32
            AscendC::DataCopy(yLocal, xLocal, len);
            AscendC::Maxs(yLocal, yLocal, this->minFloat, len);
            AscendC::Mins(yLocal, yLocal, this->maxFloat, len);
        } else {                                                // int32
            auto work = workBuf.Get<int32_t>();
            AscendC::Duplicate(work, this->minInt, len);
            AscendC::Max(yLocal, xLocal, work, len);
            AscendC::Duplicate(work, this->maxInt, len);
            AscendC::Min(yLocal, yLocal, work, len);
        }

        outQueueY.EnQue<DT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t off, uint32_t len)
    {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        AscendC::GlobalTensor<DT_X> gty;
        gty.SetGlobalBuffer(this->gmY + this->gmOffset + off, len);
        AscendC::DataCopy(gty, yLocal, len);
        outQueueY.FreeTensor(yLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> workBuf;
    __gm__ DT_X *gmX;
    __gm__ DT_X *gmY;
    
    union {
        float minFloat;
        uint32_t minHalfBits;
    };
    union {
        float maxFloat;
        uint32_t maxHalfBits;
    };
    int32_t minInt;
    int32_t maxInt;
    uint32_t coreDataNum;
    uint32_t gmOffset;
    uint32_t tileDataNum;
};

template <typename DT_X>
__global__ __aicore__ void clip_by_value(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ClipByValueTilingData);
    GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tiling_data, tiling);
    KernelClipByValue<DT_X> op;
    op.Init(x, y, tiling_data);
    op.Process();
}