// Kernel侧核函数实现
#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t ALIGN_NUM = 8;
constexpr uint32_t SMALL_SCALAR_LEN = 16;

template <class DT_X>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t totalLength,
                                uint32_t tileLength,
                                uint32_t blockDim) {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockLength = (totalLength + blockDim - 1) / blockDim;
        blockLength = (blockLength + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
        coreOffset_ = blockIdx * blockLength;
        if (coreOffset_ >= totalLength) {
            coreLength_ = 0;
            return;
        }
        uint32_t remainLength = totalLength - coreOffset_;
        coreLength_ = remainLength < blockLength ? remainLength : blockLength;

        tileLength_ = tileLength;
        tileNum_ = coreLength_ / tileLength_;
        tailLength_ = coreLength_ % tileLength_;

        xGm_.SetGlobalBuffer(
            (reinterpret_cast<__gm__ float *>(x)) + coreOffset_,
            coreLength_);
        yGm_.SetGlobalBuffer(
            (reinterpret_cast<__gm__ float *>(y)) + coreOffset_,
            coreLength_);

        uint32_t bufBytes = tileLength_ * sizeof(float);
        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, bufBytes);
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, bufBytes);
        pipe_.InitBuffer(tmpBuf1_, bufBytes);
    }

    __aicore__ inline void Process() {
        if (coreLength_ == 0) return;
        if (coreLength_ <= SMALL_SCALAR_LEN &&
            coreLength_ % ALIGN_NUM != 0) {
            CopyIn(0, coreLength_);
            ComputeSmall(coreLength_);
            CopyOut(0, coreLength_);
            return;
        }
        for (uint32_t i = 0; i < tileNum_; i++) {
            CopyIn(i, tileLength_);
            Compute(tileLength_);
            CopyOut(i, tileLength_);
        }
        if (tailLength_ > 0) {
            CopyIn(tileNum_, tailLength_);
            Compute(tailLength_);
            CopyOut(tileNum_, tailLength_);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t idx, uint32_t len) {
        AscendC::LocalTensor<float> xLocal =
            inQueueX_.AllocTensor<float>();
        if (len % ALIGN_NUM == 0) {
            AscendC::DataCopy(xLocal, xGm_[idx * tileLength_], len);
        } else {
            uint32_t copyBytes = static_cast<uint32_t>(len * sizeof(float));
            uint8_t rightPadding =
                static_cast<uint8_t>((ALIGN_NUM - len % ALIGN_NUM) %
                                     ALIGN_NUM);
            AscendC::DataCopyExtParams copyParams = {
                1, copyBytes, 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> padParams = {
                true, 0, rightPadding, 0.0f};
            AscendC::DataCopyPad(xLocal, xGm_[idx * tileLength_],
                                 copyParams, padParams);
        }
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len) {
        AscendC::LocalTensor<float> xLocal =
            inQueueX_.DeQue<float>();
        AscendC::LocalTensor<float> yLocal =
            outQueueY_.AllocTensor<float>();

        AscendC::LocalTensor<float> x2 =
            tmpBuf1_.Get<float>();
        AscendC::Mins(xLocal, xLocal, (float)2.75, len);
        AscendC::Maxs(xLocal, xLocal, (float)(-2.75), len);

        AscendC::Mul(x2, xLocal, xLocal, len);

        AscendC::Muls(yLocal, x2, (float)(-0.00000066635636), len);
        AscendC::Adds(yLocal, yLocal, (float)0.000024805751, len);
        AscendC::Mul(yLocal, yLocal, x2, len);
        AscendC::Adds(yLocal, yLocal, (float)(-0.00040413666), len);
        AscendC::Mul(yLocal, yLocal, x2, len);
        AscendC::Adds(yLocal, yLocal, (float)0.0038524622, len);
        AscendC::Mul(yLocal, yLocal, x2, len);
        AscendC::Adds(yLocal, yLocal, (float)(-0.024392316), len);
        AscendC::Mul(yLocal, yLocal, x2, len);
        AscendC::Adds(yLocal, yLocal, (float)0.11053743, len);
        AscendC::Mul(yLocal, yLocal, x2, len);
        AscendC::Adds(yLocal, yLocal, (float)(-0.37532875), len);
        AscendC::Mul(yLocal, yLocal, x2, len);
        AscendC::Adds(yLocal, yLocal, (float)1.1283792, len);
        AscendC::Mul(yLocal, yLocal, xLocal, len);

        outQueueY_.EnQue(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeSmall(uint32_t len) {
        AscendC::LocalTensor<float> xLocal =
            inQueueX_.DeQue<float>();
        AscendC::LocalTensor<float> yLocal =
            outQueueY_.AllocTensor<float>();

        for (uint32_t i = 0; i < len; i++) {
            float x = xLocal.GetValue(i);
            if (x > 2.75f) {
                x = 2.75f;
            } else if (x < -2.75f) {
                x = -2.75f;
            }
            float z = x * x;
            float y = -0.00000066635636f;
            y = y * z + 0.000024805751f;
            y = y * z - 0.00040413666f;
            y = y * z + 0.0038524622f;
            y = y * z - 0.024392316f;
            y = y * z + 0.11053743f;
            y = y * z - 0.37532875f;
            y = y * z + 1.1283792f;
            yLocal.SetValue(i, y * x);
        }

        outQueueY_.EnQue(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t idx, uint32_t len) {
        AscendC::LocalTensor<float> yLocal =
            outQueueY_.DeQue<float>();
        if (len % ALIGN_NUM == 0) {
            AscendC::DataCopy(yGm_[idx * tileLength_], yLocal, len);
        } else {
            AscendC::DataCopyExtParams copyParams = {
                1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(yGm_[idx * tileLength_], yLocal,
                                 copyParams);
        }
        outQueueY_.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf1_;

    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> yGm_;

    uint32_t coreLength_ = 0;
    uint32_t coreOffset_ = 0;
    uint32_t tileLength_ = 0;
    uint32_t tileNum_ = 0;
    uint32_t tailLength_ = 0;
};

template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y,
                               GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    KernelErf<DT_X> op;
    op.Init(x, y,
            tiling_data.totalLength,
            tiling_data.tileLength,
            tiling_data.blockDim);
    op.Process();
}
