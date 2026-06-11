// Kernel侧核函数实现
#include "kernel_operator.h"

#include "clip_by_value_tiling.h"
#include "tiling_key_clip_by_value.h"

template <class DT_X>
class KernelClipByValue {
public:
    __aicore__ inline KernelClipByValue() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, ClipByValueTilingData *tilingData) {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t totalLength = tilingData->length;
        uint32_t blockDim = tilingData->blockDim;
        uint32_t blockFormer = tilingData->blockFormer;
        uint32_t ubFormer = tilingData->ubFormer;

        // 使用 Host 侧计算的 blockFormer 确定本核偏移（256B 对齐）
        int64_t coreOffset = static_cast<int64_t>(blockFormer) * blockIdx;
        int64_t coreLen = min(static_cast<int64_t>(blockFormer),
                              static_cast<int64_t>(totalLength) - coreOffset);
        if (coreLen < 0) coreLen = 0;

        coreOffset_ = static_cast<uint32_t>(coreOffset);
        coreLength_ = static_cast<uint32_t>(coreLen);
        ubFormer_ = ubFormer;

        // 将浮点属性值转换为算子数据类型
        minVal_ = static_cast<DT_X>(tilingData->min_val);
        maxVal_ = static_cast<DT_X>(tilingData->max_val);

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));

        // 初始化流水线缓冲区 — Double Buffer (2 in + 2 out)，使用动态 ubFormer
        pipe.InitBuffer(inQueueX_, 2, ubFormer_ * sizeof(DT_X));
        pipe.InitBuffer(outQueueY_, 2, ubFormer_ * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (coreLength_ == 0) {
            return;
        }

        uint32_t remaining = coreLength_;
        uint32_t offset = 0;
        while (remaining > 0) {
            uint32_t tileLen = (remaining > ubFormer_) ? ubFormer_ : remaining;
            ProcessTile(offset, tileLen);
            offset += tileLen;
            remaining -= tileLen;
        }
    }

private:
    __aicore__ inline void ProcessTile(uint32_t offset, uint32_t length) {
        // Step 1: CopyIn — 从 GM 搬运一个 tile 到 UB（异步 DMA，MTE2）
        AscendC::LocalTensor<DT_X> xLocal = inQueueX_.AllocTensor<DT_X>();
        AscendC::DataCopyParams copyParams = {1, static_cast<uint16_t>(length * sizeof(DT_X)), 0, 0};
        AscendC::DataCopyPadParams padParams = {false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm_[coreOffset_ + offset], copyParams, padParams);
        inQueueX_.EnQue(xLocal);

        // Step 2: Compute — 裁剪（Vector 流水线 PIPE_V）
        xLocal = inQueueX_.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY_.AllocTensor<DT_X>();

        // y = max(x, min) —— 裁剪下界
        AscendC::Maxs(yLocal, xLocal, minVal_, length);
        // y = min(y, max) —— 裁剪上界（in-place 操作）
        AscendC::Mins(yLocal, yLocal, maxVal_, length);

        outQueueY_.EnQue(yLocal);
        inQueueX_.FreeTensor(xLocal);

        // Step 3: CopyOut — 从 UB 写回 GM（MTE3）
        yLocal = outQueueY_.DeQue<DT_X>();
        AscendC::DataCopyPad(yGm_[coreOffset_ + offset], yLocal, copyParams);
        outQueueY_.FreeTensor(yLocal);
    }

    AscendC::TPipe pipe;
    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> yGm_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY_;
    DT_X minVal_;
    DT_X maxVal_;
    uint32_t coreLength_ = 0;
    uint32_t coreOffset_ = 0;
    uint32_t ubFormer_ = 0;
};

template <typename DT_X>
__global__ __aicore__ void clip_by_value(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ClipByValueTilingData);
    GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tiling_data, tiling);
    KernelClipByValue<DT_X> op;
    op.Init(x, y, &tiling_data);
    op.Process();
}
