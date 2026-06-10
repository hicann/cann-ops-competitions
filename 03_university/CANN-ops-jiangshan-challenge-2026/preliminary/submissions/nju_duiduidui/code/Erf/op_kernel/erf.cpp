#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

namespace {
constexpr int32_t BUFFER_NUM = 2;

constexpr float ERF_CLAMP_MIN = -2.4f;
constexpr float ERF_CLAMP_MAX = 2.4f;
// erf(x) = x * P(z), z = (x / 2.4)^2
constexpr float ERF_Z_SCALE = 1.0f / (ERF_CLAMP_MAX * ERF_CLAMP_MAX);
constexpr float ERF_COEF_0 = 1.1271493f;
constexpr float ERF_COEF_1 = -2.1248968f;
constexpr float ERF_COEF_2 = 3.3525095f;
constexpr float ERF_COEF_3 = -3.5023823f;
constexpr float ERF_COEF_4 = 2.1063435f;
constexpr float ERF_COEF_5 = -0.5424386f;

__aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return (value + align - 1U) / align * align;
}
}  // namespace

template <typename T>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}

    __aicore__ inline void Init(AscendC::TPipe *pipe, GM_ADDR x, GM_ADDR y, uint32_t totalLength,
                                uint32_t usedCoreNum, uint32_t formerLength,
                                uint32_t tailLength, uint32_t tileLength)
    {
        uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        tileLength_ = tileLength;

        if (usedCoreNum <= 1) {
            blockLength_ = totalLength;
            gmOffset_ = 0;
        } else if (blockIdx < usedCoreNum - 1U) {
            blockLength_ = formerLength;
            gmOffset_ = blockIdx * formerLength;
        } else {
            blockLength_ = tailLength;
            gmOffset_ = (usedCoreNum - 1U) * formerLength;
        }

        xGm.SetGlobalBuffer((__gm__ T *)x + gmOffset_, blockLength_);
        yGm.SetGlobalBuffer((__gm__ T *)y + gmOffset_, blockLength_);

        pipe->InitBuffer(inQueueX, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe->InitBuffer(uBuf, tileLength_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        if (blockLength_ == 0) {
            return;
        }

        if (blockLength_ <= tileLength_) {
            CopyIn(0U, blockLength_);
            Compute(blockLength_);
            CopyOut(0U, blockLength_);
            return;
        }

        uint32_t tileNum = (blockLength_ + tileLength_ - 1U) / tileLength_;
        CopyIn(0U, GetTileLength(0U, tileNum));
        for (uint32_t i = 0; i < tileNum; ++i) {
            if (i + 1U < tileNum) {
                CopyIn(i + 1U, GetTileLength(i + 1U, tileNum));
            }
            uint32_t curLength = GetTileLength(i, tileNum);
            Compute(curLength);
            CopyOut(i, curLength);
        }
    }

private:
    __aicore__ inline uint32_t GetTileLength(uint32_t progress, uint32_t tileNum) const
    {
        if (progress + 1U == tileNum) {
            return blockLength_ - progress * tileLength_;
        }
        return tileLength_;
    }

    __aicore__ inline void CopyIn(uint32_t progress, uint32_t curLength)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        uint32_t alignElems = 32 / sizeof(T);
        uint32_t alignedLength = AlignUp(curLength, alignElems);
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(curLength * sizeof(T)),
            static_cast<uint32_t>(0),
            static_cast<uint32_t>(0),
            static_cast<uint32_t>(0)};
        AscendC::DataCopyPadExtParams<T> padParams{
            alignedLength != curLength,
            static_cast<uint8_t>(0),
            static_cast<uint8_t>(alignedLength - curLength),
            static_cast<T>(0)};

        AscendC::DataCopyPad(xLocal, xGm[progress * tileLength_], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t curLength)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        AscendC::LocalTensor<T> uLocal = uBuf.Get<T>();

        AscendC::Maxs(xLocal, xLocal, static_cast<T>(ERF_CLAMP_MIN), curLength);
        AscendC::Mins(xLocal, xLocal, static_cast<T>(ERF_CLAMP_MAX), curLength);
        AscendC::Mul(uLocal, xLocal, xLocal, curLength);
        AscendC::Muls(uLocal, uLocal, static_cast<T>(ERF_Z_SCALE), curLength);

        AscendC::Muls(yLocal, uLocal, static_cast<T>(ERF_COEF_5), curLength);
        AscendC::Adds(yLocal, yLocal, static_cast<T>(ERF_COEF_4), curLength);
        AscendC::Mul(yLocal, yLocal, uLocal, curLength);
        AscendC::Adds(yLocal, yLocal, static_cast<T>(ERF_COEF_3), curLength);
        AscendC::Mul(yLocal, yLocal, uLocal, curLength);
        AscendC::Adds(yLocal, yLocal, static_cast<T>(ERF_COEF_2), curLength);
        AscendC::Mul(yLocal, yLocal, uLocal, curLength);
        AscendC::Adds(yLocal, yLocal, static_cast<T>(ERF_COEF_1), curLength);
        AscendC::Mul(yLocal, yLocal, uLocal, curLength);
        AscendC::Adds(yLocal, yLocal, static_cast<T>(ERF_COEF_0), curLength);
        AscendC::Mul(yLocal, yLocal, xLocal, curLength);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t curLength)
    {
        AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(curLength * sizeof(T)),
            static_cast<uint32_t>(0),
            static_cast<uint32_t>(0),
            static_cast<uint32_t>(0)};

        AscendC::DataCopyPad(yGm[progress * tileLength_], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> uBuf;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    uint32_t tileLength_ = 0;
    uint32_t blockLength_ = 0;
    uint32_t gmOffset_ = 0;
};

template <typename T>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    AscendC::TPipe pipe;
    KernelErf<T> op;
    op.Init(&pipe, x, y, tiling_data.totalLength, tiling_data.usedCoreNum,
            tiling_data.formerLength, tiling_data.tailLength, tiling_data.tileLength);
    op.Process();
}
