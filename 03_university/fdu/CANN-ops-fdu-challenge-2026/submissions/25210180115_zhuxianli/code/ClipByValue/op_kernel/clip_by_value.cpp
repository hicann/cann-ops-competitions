// Kernel侧核函数实现
#include "kernel_operator.h"

#include "clip_by_value_tiling.h"
#include "tiling_key_clip_by_value.h"

#ifndef FDU_BUFFER_NUM
#define FDU_BUFFER_NUM 2
#endif

// y = min(max(x, minValue), maxValue)
// 优化要点(沿用 Lerp 经验):
//  - 32B 对齐时用轻量 DataCopy, 仅末核非对齐尾巴用 DataCopyPad;
//  - host 侧把每核数据量按 512B 对齐, 保证对齐 DataCopy 的 GM 起始落在 512B 边界(910B 上多核安全);
//  - 计算仍为 Maxs + Mins 两条向量指令, 原地累加无额外 buffer。
template <class DT_X>
class KernelClipByValue {
public:
    __aicore__ inline KernelClipByValue() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ClipByValueTilingData &tilingData) {
        blockDim = tilingData.blockDim == 0 ? 1 : tilingData.blockDim;
        tileLength = tilingData.tileLength == 0 ? 32 : tilingData.tileLength;
        minValue = static_cast<DT_X>(tilingData.minValue);
        maxValue = static_cast<DT_X>(tilingData.maxValue);
        alignElems = tilingData.alignElems == 0 ? (32 / sizeof(DT_X)) : tilingData.alignElems;

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

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + startOffset, coreLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + startOffset, coreLength);
        pipe.InitBuffer(inQueueX, FDU_BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, FDU_BUFFER_NUM, tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (coreLength == 0) {
            return;
        }
        for (uint32_t offset = 0; offset < coreLength; offset += tileLength) {
            uint32_t remain = coreLength - offset;
            processLength = remain < tileLength ? remain : tileLength;
            CopyIn(offset);
            Compute();
            CopyOut(offset);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        if ((processLength % alignElems) == 0) {
            AscendC::DataCopy(xLocal, xGm[offset], processLength);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(processLength * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        }
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute() {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        AscendC::Maxs(yLocal, xLocal, minValue, processLength);
        AscendC::Mins(yLocal, yLocal, maxValue, processLength);
        outQueueY.EnQue<DT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset) {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        if ((processLength % alignElems) == 0) {
            AscendC::DataCopy(yGm[offset], yLocal, processLength);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(processLength * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPad(yGm[offset], yLocal, copyParams);
        }
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, FDU_BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, FDU_BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t blockDim;
    uint32_t tileLength;
    uint32_t startOffset;
    uint32_t coreLength;
    uint32_t processLength;
    uint32_t alignElems;
    DT_X minValue;
    DT_X maxValue;
};

template <typename DT_X>
__global__ __aicore__ void clip_by_value(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ClipByValueTilingData);
    GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tiling_data, tiling);
    KernelClipByValue<DT_X> op;
    op.Init(x, y, tiling_data);
    op.Process();
}
