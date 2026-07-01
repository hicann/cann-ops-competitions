#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

template <class DT_X>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length, uint32_t blockNum) {
        length_ = length;
        blockNum_ = blockNum == 0 ? 1 : blockNum;
        const uint32_t blockIdx = AscendC::GetBlockIdx();

        const uint32_t fullTileNum = length_ / TILE_LENGTH;
        const uint32_t tailCount = length_ % TILE_LENGTH;
        const uint32_t baseTileNum = fullTileNum / blockNum_;
        const uint32_t extraTileNum = fullTileNum % blockNum_;
        const uint32_t localTileNum = baseTileNum + (blockIdx < extraTileNum ? 1U : 0U);
        const uint32_t startTile = blockIdx * baseTileNum + (blockIdx < extraTileNum ? blockIdx : extraTileNum);

        start_ = startTile * TILE_LENGTH;
        blockLength_ = localTileNum * TILE_LENGTH;
        if (blockIdx + 1 == blockNum_) {
            blockLength_ += tailCount;
        }

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x + start_, blockLength_);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y + start_, blockLength_);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (length_ == 0 || blockLength_ == 0) {
            return;
        }

        const uint32_t fullTileCount = blockLength_ / TILE_LENGTH;
        const uint32_t tailCount = blockLength_ % TILE_LENGTH;

        if (fullTileCount == 0) {
            CopyInTail(0, tailCount);
            Compute(tailCount);
            CopyOutTail(0, tailCount);
            return;
        }

        if (fullTileCount <= 2) {
            for (uint32_t i = 0; i < fullTileCount; ++i) {
                const uint32_t offset = i * TILE_LENGTH;
                CopyInFull(offset);
                Compute(TILE_LENGTH);
                CopyOutFull(offset);
            }
            if (tailCount > 0) {
                const uint32_t tailOffset = fullTileCount * TILE_LENGTH;
                CopyInTail(tailOffset, tailCount);
                Compute(tailCount);
                CopyOutTail(tailOffset, tailCount);
            }
            return;
        }

        for (uint32_t i = 0; i < fullTileCount + 2; ++i) {
            if (i < fullTileCount) {
                CopyInFull(i * TILE_LENGTH);
            }
            if (i >= 1 && (i - 1) < fullTileCount) {
                Compute(TILE_LENGTH);
            }
            if (i >= 2 && (i - 2) < fullTileCount) {
                CopyOutFull((i - 2) * TILE_LENGTH);
            }
        }

        if (tailCount > 0) {
            const uint32_t tailOffset = fullTileCount * TILE_LENGTH;
            CopyInTail(tailOffset, tailCount);
            Compute(tailCount);
            CopyOutTail(tailOffset, tailCount);
        }
    }

private:
    __aicore__ inline void CopyInFull(uint32_t offset) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX_.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm_[offset], TILE_LENGTH);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInTail(uint32_t offset, uint32_t count) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX_.AllocTensor<DT_X>();
        if ((count & (ALIGN_ELEM_NUM - 1U)) == 0) {
            AscendC::DataCopy(xLocal, xGm_[offset], count);
            inQueueX_.EnQue(xLocal);
            return;
        }
        AscendC::DataCopyExtParams copyParams = {
            1,
            static_cast<uint32_t>(count * sizeof(DT_X)),
            0,
            0,
            0
        };
        AscendC::DataCopyPadExtParams<DT_X> padParams = {false, 0, 0, 0};
        AscendC::DataCopyPad<DT_X>(xLocal, xGm_[offset], copyParams, padParams);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX_.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY_.AllocTensor<DT_X>();
        AscendC::Erf<DT_X>(yLocal, xLocal, count);
        outQueueY_.EnQue(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOutFull(uint32_t offset) {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY_.DeQue<DT_X>();
        AscendC::DataCopy(yGm_[offset], yLocal, TILE_LENGTH);
        outQueueY_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutTail(uint32_t offset, uint32_t count) {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY_.DeQue<DT_X>();
        if ((count & (ALIGN_ELEM_NUM - 1U)) == 0) {
            AscendC::DataCopy(yGm_[offset], yLocal, count);
            outQueueY_.FreeTensor(yLocal);
            return;
        }
        AscendC::DataCopyExtParams copyParams = {
            1,
            static_cast<uint32_t>(count * sizeof(DT_X)),
            0,
            0,
            0
        };
        AscendC::DataCopyPad<DT_X>(yGm_[offset], yLocal, copyParams);
        outQueueY_.FreeTensor(yLocal);
    }

private:
    static constexpr uint32_t TILE_LENGTH = 4096;
    static constexpr uint32_t BUFFER_NUM = 2;
    static constexpr uint32_t ALIGN_ELEM_NUM = 8;

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> yGm_;
    uint32_t length_ = 0;
    uint32_t blockNum_ = 1;
    uint32_t start_ = 0;
    uint32_t blockLength_ = 0;
};

template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tilingData, tiling);
    KernelErf<DT_X> op;
    op.Init(x, y, tilingData.length, tilingData.blockNum);
    op.Process();
}
