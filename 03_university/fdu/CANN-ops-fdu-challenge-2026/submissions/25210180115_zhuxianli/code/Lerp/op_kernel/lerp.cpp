// Kernel侧核函数实现
#include "kernel_operator.h"

#include "lerp_tiling.h"
#include "tiling_key_lerp.h"

#ifndef FDU_BUFFER_NUM
#define FDU_BUFFER_NUM 2
#endif

template <class T>
class KernelBase {
protected:
    __aicore__ inline void InitBase(GM_ADDR start, GM_ADDR end, GM_ADDR y, const LerpTilingData &tilingData) {
        tileLength = tilingData.tileLength == 0 ? 32 : tilingData.tileLength;
        alignElems = tilingData.alignElems == 0 ? (32 / sizeof(T)) : tilingData.alignElems;
        weight = tilingData.weight;

        uint32_t length = tilingData.length;
        uint32_t perCore = tilingData.lengthPerCore == 0 ? length : tilingData.lengthPerCore;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        startOffset = blockIdx * perCore;
        if (startOffset >= length) {
            coreLength = 0;
            return;
        }
        coreLength = length - startOffset;
        if (coreLength > perCore) {
            coreLength = perCore;
        }

        startGm.SetGlobalBuffer((__gm__ T *)start + startOffset, coreLength);
        endGm.SetGlobalBuffer((__gm__ T *)end + startOffset, coreLength);
        yGm.SetGlobalBuffer((__gm__ T *)y + startOffset, coreLength);
    }

    __aicore__ inline bool IsAligned() const {
#ifdef FDU_FORCE_PAD
        return false;
#else
        return (processLength % alignElems) == 0;
#endif
    }

    __aicore__ inline void LoadTensor(AscendC::LocalTensor<T> &dst, const AscendC::GlobalTensor<T> &src,
                                      uint32_t offset, bool aligned) {
        if (aligned) {
            AscendC::DataCopy(dst, src[offset], processLength);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(processLength * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(dst, src[offset], copyParams, padParams);
        }
    }

    __aicore__ inline void StoreTensor(const AscendC::LocalTensor<T> &src, uint32_t offset, bool aligned) {
        if (aligned) {
            AscendC::DataCopy(yGm[offset], src, processLength);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(processLength * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(yGm[offset], src, copyParams);
        }
    }

protected:
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> startGm;
    AscendC::GlobalTensor<T> endGm;
    AscendC::GlobalTensor<T> yGm;
    uint32_t tileLength;
    uint32_t startOffset;
    uint32_t coreLength;
    uint32_t processLength;
    uint32_t alignElems;
    float weight;
};

template <class T>
class KernelLerpGeneral : public KernelBase<T> {
public:
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR y, const LerpTilingData &tilingData) {
        this->InitBase(start, end, y, tilingData);
        if (this->coreLength == 0) {
            return;
        }
        this->pipe.InitBuffer(inQueueStart, FDU_BUFFER_NUM, this->tileLength * sizeof(T));
        this->pipe.InitBuffer(inQueueEnd, FDU_BUFFER_NUM, this->tileLength * sizeof(T));
        this->pipe.InitBuffer(outQueueY, FDU_BUFFER_NUM, this->tileLength * sizeof(T));
    }

    __aicore__ inline void Process() {
        if (this->coreLength == 0) {
            return;
        }
        for (uint32_t offset = 0; offset < this->coreLength; offset += this->tileLength) {
            uint32_t remain = this->coreLength - offset;
            this->processLength = remain < this->tileLength ? remain : this->tileLength;
            bool aligned = this->IsAligned();

            AscendC::LocalTensor<T> startLocal = inQueueStart.template AllocTensor<T>();
            AscendC::LocalTensor<T> endLocal = inQueueEnd.template AllocTensor<T>();
            this->LoadTensor(startLocal, this->startGm, offset, aligned);
            this->LoadTensor(endLocal, this->endGm, offset, aligned);
            inQueueStart.EnQue(startLocal);
            inQueueEnd.EnQue(endLocal);

            AscendC::LocalTensor<T> yLocal = outQueueY.template AllocTensor<T>();
            startLocal = inQueueStart.template DeQue<T>();
            endLocal = inQueueEnd.template DeQue<T>();
            AscendC::Sub(yLocal, endLocal, startLocal, this->processLength);
            AscendC::Muls(yLocal, yLocal, static_cast<T>(this->weight), this->processLength);
            AscendC::Add(yLocal, startLocal, yLocal, this->processLength);
            inQueueStart.FreeTensor(startLocal);
            inQueueEnd.FreeTensor(endLocal);
            outQueueY.template EnQue<T>(yLocal);

            yLocal = outQueueY.template DeQue<T>();
            this->StoreTensor(yLocal, offset, aligned);
            outQueueY.FreeTensor(yLocal);
        }
    }

private:
    AscendC::TQue<AscendC::QuePosition::VECIN, FDU_BUFFER_NUM> inQueueStart;
    AscendC::TQue<AscendC::QuePosition::VECIN, FDU_BUFFER_NUM> inQueueEnd;
    AscendC::TQue<AscendC::QuePosition::VECOUT, FDU_BUFFER_NUM> outQueueY;
};

template <class T, bool COPY_END>
class KernelLerpCopy : public KernelBase<T> {
public:
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR y, const LerpTilingData &tilingData) {
        this->InitBase(start, end, y, tilingData);
        if (this->coreLength == 0) {
            return;
        }
        this->pipe.InitBuffer(outQueueY, FDU_BUFFER_NUM, this->tileLength * sizeof(T));
    }

    __aicore__ inline void Process() {
        if (this->coreLength == 0) {
            return;
        }
        for (uint32_t offset = 0; offset < this->coreLength; offset += this->tileLength) {
            uint32_t remain = this->coreLength - offset;
            this->processLength = remain < this->tileLength ? remain : this->tileLength;
            bool aligned = this->IsAligned();

            AscendC::LocalTensor<T> yLocal = outQueueY.template AllocTensor<T>();
            if (COPY_END) {
                this->LoadTensor(yLocal, this->endGm, offset, aligned);
            } else {
                this->LoadTensor(yLocal, this->startGm, offset, aligned);
            }
            outQueueY.template EnQue<T>(yLocal);

            yLocal = outQueueY.template DeQue<T>();
            this->StoreTensor(yLocal, offset, aligned);
            outQueueY.FreeTensor(yLocal);
        }
    }

private:
    AscendC::TQue<AscendC::QuePosition::VECOUT, FDU_BUFFER_NUM> outQueueY;
};

template <typename T, uint32_t MODE>
__global__ __aicore__ void lerp(GM_ADDR start, GM_ADDR end, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LerpTilingData);
    GET_TILING_DATA_WITH_STRUCT(LerpTilingData, tiling_data, tiling);
    if constexpr (MODE == 0) {
        KernelLerpGeneral<T> op;
        op.Init(start, end, y, tiling_data);
        op.Process();
    } else if constexpr (MODE == 1) {
        KernelLerpCopy<T, false> op;
        op.Init(start, end, y, tiling_data);
        op.Process();
    } else {
        KernelLerpCopy<T, true> op;
        op.Init(start, end, y, tiling_data);
        op.Process();
    }
}
