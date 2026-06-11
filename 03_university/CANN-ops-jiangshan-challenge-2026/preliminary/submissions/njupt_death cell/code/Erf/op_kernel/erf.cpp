#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t SMALL_TILE_LENGTH = 128;
constexpr uint32_t MID_TILE_LENGTH = 512;
constexpr uint32_t HUGE_MID_TILE_LENGTH = 2048;
constexpr uint32_t TILE_LENGTH = 4096;
constexpr uint32_t TILE_SHIFT = 12;
constexpr uint32_t TILE_MASK = TILE_LENGTH - 1;

template <class DT_X>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ErfTilingData &tiling)
    {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t length = tiling.length;
        uint32_t perBlock = tiling.perBlock;
        if (length == 0) {
            return;
        }

        uint32_t start = blockIdx * perBlock;
        uint32_t currentLength = length - start;
        if (currentLength > perBlock) {
            currentLength = perBlock;
        }
        fullTileNum = currentLength >> TILE_SHIFT;
        tailLength = currentLength & TILE_MASK;
        blockLength = currentLength;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + start, currentLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + start, currentLength);

        uint32_t bufferLength = (currentLength <= SMALL_TILE_LENGTH)
            ? SMALL_TILE_LENGTH
            : ((currentLength <= MID_TILE_LENGTH)
                ? MID_TILE_LENGTH
                : ((currentLength <= HUGE_MID_TILE_LENGTH) ? HUGE_MID_TILE_LENGTH : TILE_LENGTH));
        pipe.InitBuffer(inQueue, BUFFER_NUM, bufferLength * sizeof(DT_X));
        pipe.InitBuffer(outQueue, BUFFER_NUM, bufferLength * sizeof(DT_X));
        pipe.InitBuffer(tmpBuf, bufferLength * sizeof(DT_X));
        if (fullTileNum > 0) {
            pipe.InitBuffer(tmpBuf2, bufferLength * sizeof(DT_X));
            pipe.InitBuffer(tmpBuf3, bufferLength * sizeof(DT_X));
        }
    }

    __aicore__ inline void Process()
    {
        if (blockLength <= MID_TILE_LENGTH) {
            ProcessSingleTile<true>();
            return;
        }

        if (blockLength <= HUGE_MID_TILE_LENGTH) {
            ProcessSingleTile<false>();
            return;
        }

        if (fullTileNum > 0) {
            ProcessFullTiles();
        }

        uint32_t offset = fullTileNum << TILE_SHIFT;
        ProcessTail(offset);
    }

private:
    template <bool CORE_FAST>
    __aicore__ inline void ProcessSingleTile()
    {
        if (blockLength == 0) {
            return;
        }

        if ((blockLength & (ALIGN_NUM - 1)) == 0) {
            CopyInAlignedTail(0, blockLength);
            ComputeSingleTile<CORE_FAST>(blockLength);
            CopyOutAlignedTail(0, blockLength);
        } else {
            CopyInPad(0, blockLength);
            ComputeSingleTile<CORE_FAST>(blockLength);
            CopyOutPad(0, blockLength);
        }
    }

    __aicore__ inline void ProcessFullTiles()
    {
        uint32_t offset = 0;
        CopyInAligned(0);

        for (uint32_t i = 1; i < fullTileNum; ++i) {
            uint32_t nextOffset = offset + TILE_LENGTH;
            CopyInAligned(nextOffset);
            ComputeAligned();
            CopyOutAligned(offset);
            offset = nextOffset;
        }

        ComputeAligned();
        CopyOutAligned(offset);
    }

    template <bool CORE_FAST>
    __aicore__ inline void ComputeSingleTile(uint32_t len)
    {
        if constexpr (CORE_FAST) {
            ComputeCore(len);
        } else {
            ComputeTail(len);
        }
    }

    __aicore__ inline void ProcessTail(uint32_t offset)
    {
        uint32_t len = tailLength;
        if (len == 0) {
            return;
        }
        if ((len & (ALIGN_NUM - 1)) == 0) {
            CopyInAlignedTail(offset, len);
            Compute(len);
            CopyOutAlignedTail(offset, len);
        } else {
            CopyInPad(offset, len);
            Compute(len);
            CopyOutPad(offset, len);
        }
    }

    __aicore__ inline void CopyInPad(uint32_t offset, uint32_t len)
    {
        LocalTensor<DT_X> xLocal = inQueue.template AllocTensor<DT_X>();

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = len * sizeof(DT_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        DataCopyPadExtParams<DT_X> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = static_cast<DT_X>(0);

        DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void CopyInAligned(uint32_t offset)
    {
        LocalTensor<DT_X> xLocal = inQueue.template AllocTensor<DT_X>();
        DataCopy(xLocal, xGm[offset], TILE_LENGTH);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void CopyInAlignedTail(uint32_t offset, uint32_t len)
    {
        LocalTensor<DT_X> xLocal = inQueue.template AllocTensor<DT_X>();
        DataCopy(xLocal, xGm[offset], len);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        LocalTensor<DT_X> xLocal = inQueue.template DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueue.template AllocTensor<DT_X>();

        ComputeTailFast(yLocal, xLocal, len);
        outQueue.EnQue(yLocal);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeAligned()
    {
        LocalTensor<DT_X> xLocal = inQueue.template DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueue.template AllocTensor<DT_X>();

        ComputeFastAligned(yLocal, xLocal);
        outQueue.EnQue(yLocal);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeCore(uint32_t len)
    {
        LocalTensor<DT_X> xLocal = inQueue.template DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueue.template AllocTensor<DT_X>();

        if (len == 512) {
            ComputeCoreFastImpl<512>(yLocal, xLocal);
        } else if (len == 256) {
            ComputeCoreFastImpl<256>(yLocal, xLocal);
        } else if (len == 128) {
            ComputeCoreFastImpl<128>(yLocal, xLocal);
        } else {
            ComputeCoreFast(yLocal, xLocal, len);
        }

        outQueue.EnQue(yLocal);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeTail(uint32_t len)
    {
        LocalTensor<DT_X> xLocal = inQueue.template DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueue.template AllocTensor<DT_X>();

        if (len == 2048) {
            ComputeTailFastImpl<2048>(yLocal, xLocal);
        } else if (len == 1536) {
            ComputeTailFastImpl<1536>(yLocal, xLocal);
        } else if (len == 1024) {
            ComputeTailFastImpl<1024>(yLocal, xLocal);
        } else if (len == 768) {
            ComputeTailFastImpl<768>(yLocal, xLocal);
        } else {
            ComputeTailFast(yLocal, xLocal, len);
        }

        outQueue.EnQue(yLocal);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeFastAligned(LocalTensor<DT_X> yLocal, LocalTensor<DT_X> xLocal)
    {
        ComputeFastEstrin<TILE_LENGTH>(yLocal, xLocal);
    }

    __aicore__ inline void ComputeTailFast(LocalTensor<DT_X> yLocal, LocalTensor<DT_X> xLocal, uint32_t len)
    {
        LocalTensor<DT_X> x2 = tmpBuf.template Get<DT_X>();

        Mins(xLocal, xLocal, static_cast<DT_X>(2.82f), len);
        Maxs(xLocal, xLocal, static_cast<DT_X>(-2.82f), len);
        Mul(x2, xLocal, xLocal, len);
        Muls(yLocal, x2, static_cast<DT_X>(5.17908380e-06f), len);
        Adds(yLocal, yLocal, static_cast<DT_X>(-1.73148240e-04f), len);
        Mul(yLocal, yLocal, x2, len);
        Adds(yLocal, yLocal, static_cast<DT_X>(2.46950591e-03f), len);
        Mul(yLocal, yLocal, x2, len);
        Adds(yLocal, yLocal, static_cast<DT_X>(-1.99616955e-02f), len);
        Mul(yLocal, yLocal, x2, len);
        Adds(yLocal, yLocal, static_cast<DT_X>(1.03248518e-01f), len);
        Mul(yLocal, yLocal, x2, len);
        Adds(yLocal, yLocal, static_cast<DT_X>(-3.70017570e-01f), len);
        Mul(yLocal, yLocal, x2, len);
        Adds(yLocal, yLocal, static_cast<DT_X>(1.12723383e+00f), len);
        Mul(yLocal, yLocal, xLocal, len);
    }

    __aicore__ inline void ComputeCoreFast(LocalTensor<DT_X> yLocal, LocalTensor<DT_X> xLocal, uint32_t len)
    {
        LocalTensor<DT_X> x2 = tmpBuf.template Get<DT_X>();

        Mins(xLocal, xLocal, static_cast<DT_X>(2.4f), len);
        Maxs(xLocal, xLocal, static_cast<DT_X>(-2.4f), len);
        Mul(x2, xLocal, xLocal, len);
        Muls(yLocal, x2, static_cast<DT_X>(-8.53516834e-05f), len);
        Adds(yLocal, yLocal, static_cast<DT_X>(1.90835881e-03f), len);
        Mul(yLocal, yLocal, x2, len);
        Adds(yLocal, yLocal, static_cast<DT_X>(-1.82814188e-02f), len);
        Mul(yLocal, yLocal, x2, len);
        Adds(yLocal, yLocal, static_cast<DT_X>(1.00872477e-01f), len);
        Mul(yLocal, yLocal, x2, len);
        Adds(yLocal, yLocal, static_cast<DT_X>(-3.68623966e-01f), len);
        Mul(yLocal, yLocal, x2, len);
        Adds(yLocal, yLocal, static_cast<DT_X>(1.12700773e+00f), len);
        Mul(yLocal, yLocal, xLocal, len);
    }

    template <uint32_t LEN>
    __aicore__ inline void ComputeFastEstrin(LocalTensor<DT_X> yLocal, LocalTensor<DT_X> xLocal)
    {
        LocalTensor<DT_X> z = tmpBuf.template Get<DT_X>();
        LocalTensor<DT_X> z2 = tmpBuf2.template Get<DT_X>();
        LocalTensor<DT_X> t = tmpBuf3.template Get<DT_X>();

        Mins(xLocal, xLocal, static_cast<DT_X>(2.82f), LEN);
        Maxs(xLocal, xLocal, static_cast<DT_X>(-2.82f), LEN);
        Mul(z, xLocal, xLocal, LEN);
        Mul(z2, z, z, LEN);

        Muls(yLocal, z, static_cast<DT_X>(-3.74389461e-01f), LEN);
        Adds(yLocal, yLocal, static_cast<DT_X>(1.12811932e+00f), LEN);

        Muls(t, z, static_cast<DT_X>(-2.37653824e-02f), LEN);
        Adds(t, t, static_cast<DT_X>(1.09415559e-01f), LEN);
        Mul(t, t, z2, LEN);
        Add(yLocal, yLocal, t, LEN);

        Muls(t, z, static_cast<DT_X>(-5.82005128e-07f), LEN);
        Adds(t, t, static_cast<DT_X>(2.22992836e-05f), LEN);
        Mul(t, t, z2, LEN);

        Muls(z, z, static_cast<DT_X>(-3.74091217e-04f), LEN);
        Adds(z, z, static_cast<DT_X>(3.66673108e-03f), LEN);
        Add(t, t, z, LEN);
        Mul(z2, z2, z2, LEN);
        Mul(t, t, z2, LEN);
        Add(yLocal, yLocal, t, LEN);
        Mul(yLocal, yLocal, xLocal, LEN);
    }

    template <uint32_t LEN>
    __aicore__ inline void ComputeTailFastImpl(LocalTensor<DT_X> yLocal, LocalTensor<DT_X> xLocal)
    {
        LocalTensor<DT_X> x2 = tmpBuf.template Get<DT_X>();

        Mins(xLocal, xLocal, static_cast<DT_X>(2.82f), LEN);
        Maxs(xLocal, xLocal, static_cast<DT_X>(-2.82f), LEN);
        Mul(x2, xLocal, xLocal, LEN);
        Muls(yLocal, x2, static_cast<DT_X>(5.17908380e-06f), LEN);
        Adds(yLocal, yLocal, static_cast<DT_X>(-1.73148240e-04f), LEN);
        Mul(yLocal, yLocal, x2, LEN);
        Adds(yLocal, yLocal, static_cast<DT_X>(2.46950591e-03f), LEN);
        Mul(yLocal, yLocal, x2, LEN);
        Adds(yLocal, yLocal, static_cast<DT_X>(-1.99616955e-02f), LEN);
        Mul(yLocal, yLocal, x2, LEN);
        Adds(yLocal, yLocal, static_cast<DT_X>(1.03248518e-01f), LEN);
        Mul(yLocal, yLocal, x2, LEN);
        Adds(yLocal, yLocal, static_cast<DT_X>(-3.70017570e-01f), LEN);
        Mul(yLocal, yLocal, x2, LEN);
        Adds(yLocal, yLocal, static_cast<DT_X>(1.12723383e+00f), LEN);
        Mul(yLocal, yLocal, xLocal, LEN);
    }

    template <uint32_t LEN>
    __aicore__ inline void ComputeCoreFastImpl(LocalTensor<DT_X> yLocal, LocalTensor<DT_X> xLocal)
    {
        LocalTensor<DT_X> x2 = tmpBuf.template Get<DT_X>();

        Mins(xLocal, xLocal, static_cast<DT_X>(2.4f), LEN);
        Maxs(xLocal, xLocal, static_cast<DT_X>(-2.4f), LEN);
        Mul(x2, xLocal, xLocal, LEN);
        Muls(yLocal, x2, static_cast<DT_X>(-8.53516834e-05f), LEN);
        Adds(yLocal, yLocal, static_cast<DT_X>(1.90835881e-03f), LEN);
        Mul(yLocal, yLocal, x2, LEN);
        Adds(yLocal, yLocal, static_cast<DT_X>(-1.82814188e-02f), LEN);
        Mul(yLocal, yLocal, x2, LEN);
        Adds(yLocal, yLocal, static_cast<DT_X>(1.00872477e-01f), LEN);
        Mul(yLocal, yLocal, x2, LEN);
        Adds(yLocal, yLocal, static_cast<DT_X>(-3.68623966e-01f), LEN);
        Mul(yLocal, yLocal, x2, LEN);
        Adds(yLocal, yLocal, static_cast<DT_X>(1.12700773e+00f), LEN);
        Mul(yLocal, yLocal, xLocal, LEN);
    }

    __aicore__ inline void CopyOutPad(uint32_t offset, uint32_t len)
    {
        LocalTensor<DT_X> yLocal = outQueue.template DeQue<DT_X>();

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = len * sizeof(DT_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueue.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutAligned(uint32_t offset)
    {
        LocalTensor<DT_X> yLocal = outQueue.template DeQue<DT_X>();
        DataCopy(yGm[offset], yLocal, TILE_LENGTH);
        outQueue.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutAlignedTail(uint32_t offset, uint32_t len)
    {
        LocalTensor<DT_X> yLocal = outQueue.template DeQue<DT_X>();
        DataCopy(yGm[offset], yLocal, len);
        outQueue.FreeTensor(yLocal);
    }

private:
    static constexpr uint32_t ALIGN_NUM = 8;

    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueue;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> tmpBuf2;
    TBuf<TPosition::VECCALC> tmpBuf3;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    uint32_t fullTileNum = 0;
    uint32_t tailLength = 0;
    uint32_t blockLength = 0;
};

template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);

    KernelErf<DT_X> op;
    op.Init(x, y, tiling_data);
    op.Process();
}
