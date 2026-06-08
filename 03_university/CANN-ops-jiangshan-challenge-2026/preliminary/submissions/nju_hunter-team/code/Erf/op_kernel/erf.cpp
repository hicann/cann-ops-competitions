// Kernel侧核函数实现
#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

namespace
{
    constexpr uint32_t BUFFER_NUM = 1;
    constexpr uint32_t DATA_ALIGN_BYTES = 32;
    constexpr uint32_t LIGHT_SMALL_UB_ELEMS = 1024;
    constexpr uint32_t FAST_HEADER_SHIFT = 32;
    constexpr uint64_t FAST_HEADER_MODE_MASK = 0xffffffffULL;
    constexpr uint64_t FAST_SMALL_SINGLE_MODE = 1;
    constexpr float ERF_CLIP = 2.25f;
    constexpr float ERF_C0 = 1.1241973028f;
    constexpr float ERF_C1 = -3.5714637328e-1f;
    constexpr float ERF_C2 = 8.7919842189e-2f;
    constexpr float ERF_C3 = -1.2357624117e-2f;
    constexpr float ERF_C4 = 7.2783690333e-4f;

    template <typename T>
    __aicore__ inline void CpGm2Local(const AscendC::LocalTensor<T> &lt,
                                      const AscendC::GlobalTensor<T> &gt, uint32_t len)
    {
        constexpr uint32_t ALIGN_ELEM_MASK = ~(DATA_ALIGN_BYTES / sizeof(T) - 1);
        uint32_t alignElemNum = len & ALIGN_ELEM_MASK;
        uint32_t unAlignLen = (len - alignElemNum) * sizeof(T);

        if (alignElemNum != 0)
        {
            AscendC::DataCopy(lt, gt, alignElemNum);
        }
        if (unAlignLen != 0)
        {
            const AscendC::DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
            const AscendC::DataCopyPadExtParams<T> dataCopyPadExtParams{false, 0, 0, 0};
            AscendC::DataCopyPad(lt[alignElemNum], gt[alignElemNum],
                                 dataCopyExtParams, dataCopyPadExtParams);
        }
    }

    template <typename T>
    __aicore__ inline void CpLocal2Gm(const AscendC::GlobalTensor<T> &gt,
                                      const AscendC::LocalTensor<T> &lt, uint32_t len)
    {
        constexpr uint32_t ALIGN_ELEM_MASK = ~(DATA_ALIGN_BYTES / sizeof(T) - 1);
        uint32_t alignElemNum = len & ALIGN_ELEM_MASK;
        uint32_t unAlignLen = (len - alignElemNum) * sizeof(T);

        if (alignElemNum != 0)
        {
            AscendC::DataCopy(gt, lt, alignElemNum);
        }
        if (unAlignLen != 0)
        {
            const AscendC::DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
            AscendC::DataCopyPad(gt[alignElemNum], lt[alignElemNum],
                                 dataCopyExtParams);
        }
    }

    template <typename T>
    __aicore__ inline void ComputeErfPoly(AscendC::LocalTensor<T> xLocal,
                                          AscendC::LocalTensor<T> yLocal,
                                          AscendC::LocalTensor<T> polyLocal,
                                          uint32_t len)
    {
        AscendC::Mins(xLocal, xLocal, static_cast<T>(ERF_CLIP), len);
        AscendC::Maxs(xLocal, xLocal, static_cast<T>(-ERF_CLIP), len);
        AscendC::Mul(yLocal, xLocal, xLocal, len);
        AscendC::Muls(polyLocal, yLocal, static_cast<T>(ERF_C4), len);
        AscendC::Adds(polyLocal, polyLocal, static_cast<T>(ERF_C3), len);
        AscendC::Mul(polyLocal, polyLocal, yLocal, len);
        AscendC::Adds(polyLocal, polyLocal, static_cast<T>(ERF_C2), len);
        AscendC::Mul(polyLocal, polyLocal, yLocal, len);
        AscendC::Adds(polyLocal, polyLocal, static_cast<T>(ERF_C1), len);
        AscendC::Mul(polyLocal, polyLocal, yLocal, len);
        AscendC::Adds(polyLocal, polyLocal, static_cast<T>(ERF_C0), len);
        AscendC::Mul(yLocal, xLocal, polyLocal, len);
    }

    template <typename T>
    __aicore__ inline bool IsAlignedLen(uint32_t len)
    {
        constexpr uint32_t ALIGN_ELEM_NUM = DATA_ALIGN_BYTES / sizeof(T);
        return (len & (ALIGN_ELEM_NUM - 1)) == 0;
    }

    template <typename T>
    __aicore__ inline void FastErfSmallCore(GM_ADDR x, GM_ADDR y,
                                            uint64_t blockOffset,
                                            uint32_t processDataNum)
    {
        AscendC::GlobalTensor<T> xGm;
        AscendC::GlobalTensor<T> yGm;
        xGm.SetGlobalBuffer((__gm__ T *)x + blockOffset, processDataNum);
        yGm.SetGlobalBuffer((__gm__ T *)y + blockOffset, processDataNum);

        AscendC::LocalTensor<T> xLocal;
        AscendC::TBuffAddr xAddr;
        xAddr.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECCALC);
        xLocal.SetAddr(xAddr);
        xLocal.InitBuffer(0, LIGHT_SMALL_UB_ELEMS);

        AscendC::LocalTensor<T> yLocal;
        AscendC::TBuffAddr yAddr;
        yAddr.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECCALC);
        yLocal.SetAddr(yAddr);
        yLocal.InitBuffer(LIGHT_SMALL_UB_ELEMS * sizeof(T), LIGHT_SMALL_UB_ELEMS);

        AscendC::LocalTensor<T> polyLocal;
        AscendC::TBuffAddr polyAddr;
        polyAddr.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECCALC);
        polyLocal.SetAddr(polyAddr);
        polyLocal.InitBuffer(2 * LIGHT_SMALL_UB_ELEMS * sizeof(T), LIGHT_SMALL_UB_ELEMS);

        bool aligned = IsAlignedLen<T>(processDataNum);
        if (aligned)
        {
            AscendC::DataCopy(xLocal, xGm, processDataNum);
        }
        else
        {
            CpGm2Local(xLocal, xGm, processDataNum);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        ComputeErfPoly(xLocal, yLocal, polyLocal, processDataNum);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        if (aligned)
        {
            AscendC::DataCopy(yGm, yLocal, processDataNum);
            return;
        }

        CpLocal2Gm(yGm, yLocal, processDataNum);
    }

    template <typename T>
    __aicore__ inline void FastErfSmallSingle(GM_ADDR x, GM_ADDR y, uint64_t totalLength)
    {
        FastErfSmallCore<T>(x, y, 0, static_cast<uint32_t>(totalLength));
    }

} // namespace

template <typename DT_X>
class KernelErf
{
public:
    __aicore__ inline KernelErf() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ErfTilingData &tilingData)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        this->tileLength = tilingData.tileLength;
        this->smallMode = tilingData.smallMode;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = tilingData.coreNum;

        uint64_t blockOffset = static_cast<uint64_t>(blockIdx) * tilingData.blockLength;
        if (tilingData.totalLength == 0 || blockOffset >= tilingData.totalLength)
        {
            this->coreLength = 0;
            blockOffset = 0;
        }
        else if (blockIdx + 1 == blockNum)
        {
            this->coreLength = tilingData.totalLength - blockOffset;
        }
        else
        {
            this->coreLength = tilingData.blockLength;
        }

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + blockOffset, this->coreLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + blockOffset, this->coreLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(polyBuf, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (this->coreLength == 0)
        {
            return;
        }

        if (this->smallMode != 0)
        {
            ProcessSmall();
            return;
        }

        uint64_t remaining = this->coreLength;
        uint64_t tileOffset = 0;
        while (remaining >= this->tileLength)
        {
            this->processDataNum = this->tileLength;
            CopyInAligned(tileOffset);
            Compute();
            CopyOutAligned(tileOffset);
            remaining -= this->tileLength;
            tileOffset += this->tileLength;
        }

        if (remaining != 0)
        {
            this->processDataNum = static_cast<uint32_t>(remaining);
            if (IsAligned(this->processDataNum))
            {
                CopyInAligned(tileOffset);
                Compute();
                CopyOutAligned(tileOffset);
                return;
            }

            CopyIn(tileOffset);
            Compute();
            CopyOut(tileOffset);
        }
    }

private:
    __aicore__ inline void ProcessSmall()
    {
        this->processDataNum = static_cast<uint32_t>(this->coreLength);
        if (IsAligned(this->processDataNum))
        {
            CopyInAligned(0);
            Compute();
            CopyOutAligned(0);
            return;
        }

        CopyIn(0);
        Compute();
        CopyOut(0);
    }

    __aicore__ inline bool IsAligned(uint32_t len) const
    {
        return IsAlignedLen<DT_X>(len);
    }

    __aicore__ inline void CopyIn(uint64_t tileOffset)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        CpGm2Local(xLocal, xGm[tileOffset], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyInAligned(uint64_t tileOffset)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[tileOffset], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> polyLocal = polyBuf.Get<DT_X>();

        ComputeLocal(xLocal, yLocal, polyLocal);

        outQueueY.EnQue<DT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeLocal(AscendC::LocalTensor<DT_X> xLocal,
                                        AscendC::LocalTensor<DT_X> yLocal,
                                        AscendC::LocalTensor<DT_X> polyLocal)
    {
        ComputeErfPoly(xLocal, yLocal, polyLocal, this->processDataNum);
    }

    __aicore__ inline void CopyOut(uint64_t tileOffset)
    {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        CpLocal2Gm(yGm[tileOffset], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutAligned(uint64_t tileOffset)
    {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        AscendC::DataCopy(yGm[tileOffset], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> polyBuf;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;

    uint64_t coreLength = 0;
    uint32_t tileLength = 0;
    uint32_t processDataNum = 0;
    uint32_t smallMode = 0;
};

template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ErfTilingData);
    __gm__ uint64_t *tilingHeaderGm = reinterpret_cast<__gm__ uint64_t *>(tiling);
    uint64_t fastHeader = tilingHeaderGm[0];
    uint64_t fastMode = fastHeader & FAST_HEADER_MODE_MASK;
    if (fastMode == FAST_SMALL_SINGLE_MODE)
    {
        FastErfSmallSingle<DT_X>(x, y, fastHeader >> FAST_HEADER_SHIFT);
        return;
    }
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tilingData, tiling);
    KernelErf<DT_X> op;
    op.Init(x, y, tilingData);
    op.Process();
}
