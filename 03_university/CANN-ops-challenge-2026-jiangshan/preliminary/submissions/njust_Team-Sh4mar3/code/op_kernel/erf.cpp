#define K_MAX_SHAPE_DIM 0

#include "kernel_operator.h"
#include "tiling_key_erf.h"

struct ErfTilingRaw {
    uint32_t coreLenAligned;
    uint32_t fullCoreNum;
    uint32_t tailLen;
    uint32_t tailNeedPad;
};

struct ErfTaylorConstants {
    static constexpr float C0 = 1.12432802f;
    static constexpr float C1 = -0.357585311f;
    static constexpr float C2 = 0.0883001909f;
    static constexpr float C3 = -0.0124759655f;
    static constexpr float C4 = 0.000739837822f;

    static constexpr float CLIP_MAX = 2.23908424f;
    static constexpr float CLIP_MIN = -2.23908424f;
};

struct ErfPadeConstants {
    static constexpr float A0 = 1.13231867f;
    static constexpr float A1 = -0.00209840f;
    static constexpr float A2 = 0.00397027f;

    static constexpr float B1 = 0.34692688f;

    static constexpr float CLIP_MAX = 2.30000000f;
    static constexpr float CLIP_MIN = -2.30000000f;
};

constexpr uint32_t PIPELINE_TILE_ELEMS = 8192U;
constexpr uint32_t PADE_TILE_ELEMS = 1024U;
constexpr uint32_t LOCAL_TILE_BYTES = PIPELINE_TILE_ELEMS * sizeof(float);
constexpr uint32_t LOCAL_TILE_SKEW_BYTES = 128U;
constexpr event_t ERF_EVENT_MTE2_V = static_cast<event_t>(EVENT_ID0);
constexpr event_t ERF_EVENT_V_MTE3 = static_cast<event_t>(EVENT_ID1);

template <class DT_X>
__aicore__ inline void CopyInNoPad(AscendC::LocalTensor<DT_X> &dst,
                                   AscendC::GlobalTensor<DT_X> src,
                                   uint32_t elemCount)
{
    AscendC::DataCopy(dst, src, elemCount);
}

template <class DT_X>
__aicore__ inline void CopyOutNoPad(AscendC::GlobalTensor<DT_X> dst,
                                    AscendC::LocalTensor<DT_X> &src,
                                    uint32_t elemCount)
{
    AscendC::DataCopy(dst, src, elemCount);
}

template <class DT_X>
__aicore__ inline void CopyInWithPad(AscendC::LocalTensor<DT_X> &dst,
                                     AscendC::GlobalTensor<DT_X> src,
                                     uint32_t elemCount)
{
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(elemCount * sizeof(DT_X)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
    AscendC::DataCopyPad(dst, src, copyParams, padParams);
}

template <class DT_X>
__aicore__ inline void CopyOutWithPad(AscendC::GlobalTensor<DT_X> dst,
                                      AscendC::LocalTensor<DT_X> &src,
                                      uint32_t elemCount)
{
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(elemCount * sizeof(DT_X)), 0, 0, 0};
    AscendC::DataCopyPad(dst, src, copyParams);
}

template <class DT_X>
__aicore__ inline void ComputeErfTaylor(
    AscendC::LocalTensor<DT_X> &yLocal,
    AscendC::LocalTensor<DT_X> &xLocal,
    AscendC::LocalTensor<DT_X> &xSquared,
    const AscendC::BinaryRepeatParams &binaryParams)
{
    AscendC::Mins<DT_X, false>(xLocal, xLocal, ErfTaylorConstants::CLIP_MAX, 1);
    AscendC::Maxs<DT_X, false>(xLocal, xLocal, ErfTaylorConstants::CLIP_MIN, 1);

    AscendC::Mul<DT_X, false>(xSquared, xLocal, xLocal, AscendC::MASK_PLACEHOLDER, 1, binaryParams);

    AscendC::Muls<DT_X, false>(yLocal, xSquared, ErfTaylorConstants::C4, 1);
    AscendC::Adds<DT_X, false>(yLocal, yLocal, ErfTaylorConstants::C3, 1);

    AscendC::Mul<DT_X, false>(yLocal, yLocal, xSquared, AscendC::MASK_PLACEHOLDER, 1, binaryParams);
    AscendC::Adds<DT_X, false>(yLocal, yLocal, ErfTaylorConstants::C2, 1);

    AscendC::Mul<DT_X, false>(yLocal, yLocal, xSquared, AscendC::MASK_PLACEHOLDER, 1, binaryParams);
    AscendC::Adds<DT_X, false>(yLocal, yLocal, ErfTaylorConstants::C1, 1);

    AscendC::Mul<DT_X, false>(yLocal, yLocal, xSquared, AscendC::MASK_PLACEHOLDER, 1, binaryParams);
    AscendC::Adds<DT_X, false>(yLocal, yLocal, ErfTaylorConstants::C0, 1);

    AscendC::Mul<DT_X, false>(yLocal, yLocal, xLocal, AscendC::MASK_PLACEHOLDER, 1, binaryParams);
}

template <class DT_X>
__aicore__ inline void ComputeErfPade(
    AscendC::LocalTensor<DT_X> &yLocal,
    AscendC::LocalTensor<DT_X> &xLocal,
    AscendC::LocalTensor<DT_X> &xSquared,
    AscendC::LocalTensor<DT_X> &denominator,
    const AscendC::BinaryRepeatParams &binaryParams)
{
    AscendC::Mins<DT_X, false>(xLocal, xLocal, ErfPadeConstants::CLIP_MAX, 1);
    AscendC::Maxs<DT_X, false>(xLocal, xLocal, ErfPadeConstants::CLIP_MIN, 1);

    AscendC::Mul<DT_X, false>(xSquared, xLocal, xLocal, AscendC::MASK_PLACEHOLDER, 1, binaryParams);

    AscendC::Muls<DT_X, false>(denominator, xSquared, ErfPadeConstants::B1, 1);
    AscendC::Muls<DT_X, false>(yLocal, xSquared, ErfPadeConstants::A2, 1);
    AscendC::Adds<DT_X, false>(denominator, denominator, 1.0f, 1);
    AscendC::Adds<DT_X, false>(yLocal, yLocal, ErfPadeConstants::A1, 1);
    AscendC::Mul<DT_X, false>(yLocal, yLocal, xSquared, AscendC::MASK_PLACEHOLDER, 1, binaryParams);
    AscendC::Adds<DT_X, false>(yLocal, yLocal, ErfPadeConstants::A0, 1);

    AscendC::Mul<DT_X, false>(yLocal, xLocal, yLocal, AscendC::MASK_PLACEHOLDER, 1, binaryParams);
    AscendC::Div<DT_X, false>(yLocal, yLocal, denominator, AscendC::MASK_PLACEHOLDER, 1, binaryParams);
}

template <class DT_X>
class KernelErfLocal {
public:
    __aicore__ inline void ProcessPade(GM_ADDR x, GM_ADDR y, uint32_t gmOffset,
                                       uint32_t elemCount, bool needPad) {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + gmOffset, elemCount);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + gmOffset, elemCount);

        AscendC::LocalTensor<DT_X> xLocal{AscendC::TPosition::VECCALC, 0, LOCAL_TILE_BYTES};
        AscendC::LocalTensor<DT_X> yLocal{AscendC::TPosition::VECCALC,
                                          LOCAL_TILE_BYTES + LOCAL_TILE_SKEW_BYTES,
                                          LOCAL_TILE_BYTES};
        AscendC::LocalTensor<DT_X> xSquared{AscendC::TPosition::VECCALC,
                                            2 * LOCAL_TILE_BYTES + 2 * LOCAL_TILE_SKEW_BYTES,
                                            LOCAL_TILE_BYTES};
        AscendC::LocalTensor<DT_X> denominator{AscendC::TPosition::VECCALC,
                                               3 * LOCAL_TILE_BYTES + 3 * LOCAL_TILE_SKEW_BYTES,
                                               LOCAL_TILE_BYTES};

        AscendC::BinaryRepeatParams binaryParams{1, 1, 1, 8, 8, 8};

        AscendC::SetMaskCount();
        AscendC::SetVectorMask<DT_X, AscendC::MaskMode::COUNTER>(elemCount);
        if (needPad) {
            CopyInWithPad<DT_X>(xLocal, xGm, elemCount);
        } else {
            CopyInNoPad<DT_X>(xLocal, xGm, elemCount);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ERF_EVENT_MTE2_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ERF_EVENT_MTE2_V);

        ComputeErfPade<DT_X>(yLocal, xLocal, xSquared, denominator, binaryParams);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ERF_EVENT_V_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ERF_EVENT_V_MTE3);

        if (needPad) {
            CopyOutWithPad<DT_X>(yGm, yLocal, elemCount);
        } else {
            CopyOutNoPad<DT_X>(yGm, yLocal, elemCount);
        }
    }

    __aicore__ inline void ProcessTaylor(GM_ADDR x, GM_ADDR y, uint32_t gmOffset,
                                         uint32_t elemCount, bool needPad) {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + gmOffset, elemCount);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + gmOffset, elemCount);

        AscendC::LocalTensor<DT_X> xLocal{AscendC::TPosition::VECCALC, 0, LOCAL_TILE_BYTES};
        AscendC::LocalTensor<DT_X> yLocal{AscendC::TPosition::VECCALC,
                                          LOCAL_TILE_BYTES + LOCAL_TILE_SKEW_BYTES,
                                          LOCAL_TILE_BYTES};
        AscendC::LocalTensor<DT_X> xSquared{AscendC::TPosition::VECCALC,
                                            2 * LOCAL_TILE_BYTES + 2 * LOCAL_TILE_SKEW_BYTES,
                                            LOCAL_TILE_BYTES};

        AscendC::BinaryRepeatParams binaryParams{1, 1, 1, 8, 8, 8};

        AscendC::SetMaskCount();
        AscendC::SetVectorMask<DT_X, AscendC::MaskMode::COUNTER>(elemCount);
        if (needPad) {
            CopyInWithPad<DT_X>(xLocal, xGm, elemCount);
        } else {
            CopyInNoPad<DT_X>(xLocal, xGm, elemCount);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ERF_EVENT_MTE2_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ERF_EVENT_MTE2_V);

        ComputeErfTaylor<DT_X>(yLocal, xLocal, xSquared, binaryParams);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ERF_EVENT_V_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ERF_EVENT_V_MTE3);

        if (needPad) {
            CopyOutWithPad<DT_X>(yGm, yLocal, elemCount);
        } else {
            CopyOutNoPad<DT_X>(yGm, yLocal, elemCount);
        }
    }

    __aicore__ inline void ProcessSingle(GM_ADDR x, GM_ADDR y) {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), 8U);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), 1U);

        AscendC::LocalTensor<DT_X> xLocal{AscendC::TPosition::VECCALC, 0, LOCAL_TILE_BYTES};
        AscendC::LocalTensor<DT_X> yLocal{AscendC::TPosition::VECCALC,
                                          LOCAL_TILE_BYTES + LOCAL_TILE_SKEW_BYTES,
                                          LOCAL_TILE_BYTES};
        AscendC::LocalTensor<DT_X> xSquared{AscendC::TPosition::VECCALC,
                                            2 * LOCAL_TILE_BYTES + 2 * LOCAL_TILE_SKEW_BYTES,
                                            LOCAL_TILE_BYTES};
        AscendC::LocalTensor<DT_X> denominator{AscendC::TPosition::VECCALC,
                                               3 * LOCAL_TILE_BYTES + 3 * LOCAL_TILE_SKEW_BYTES,
                                               LOCAL_TILE_BYTES};

        AscendC::BinaryRepeatParams binaryParams{1, 1, 1, 8, 8, 8};

        AscendC::SetMaskCount();
        AscendC::SetVectorMask<DT_X, AscendC::MaskMode::COUNTER>(1U);
        CopyInNoPad<DT_X>(xLocal, xGm, 8U);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ERF_EVENT_MTE2_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ERF_EVENT_MTE2_V);

        ComputeErfPade<DT_X>(yLocal, xLocal, xSquared, denominator, binaryParams);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ERF_EVENT_V_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ERF_EVENT_V_MTE3);

        CopyOutWithPad<DT_X>(yGm, yLocal, 1U);
    }

private:
    AscendC::GlobalTensor<DT_X> xGm, yGm;
};

template <class DT_X>
class KernelErfPipeline {
public:
    __aicore__ inline void InitPipeline(AscendC::TPipe *pipePtr, GM_ADDR x, GM_ADDR y,
                                        uint32_t gmOffset, uint32_t elemCount) {
        this->gmOffset = gmOffset;
        this->elemCount = elemCount;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), gmOffset + elemCount);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), gmOffset + elemCount);
        pipePtr->InitBuffer(inQueueX, 2, PIPELINE_TILE_ELEMS * sizeof(DT_X));
        pipePtr->InitBuffer(outQueueY, 2, PIPELINE_TILE_ELEMS * sizeof(DT_X));
        pipePtr->InitBuffer(calcBuf, PIPELINE_TILE_ELEMS * sizeof(DT_X));
    }

    __aicore__ inline void ProcessPipeline(bool needPad) {
        AscendC::LocalTensor<DT_X> workspaceLocal = calcBuf.Get<DT_X>();
        AscendC::LocalTensor<DT_X> xSquared = workspaceLocal;
        AscendC::BinaryRepeatParams binaryParams{1, 1, 1, 8, 8, 8};
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<DT_X, AscendC::MaskMode::COUNTER>(PIPELINE_TILE_ELEMS);
        CopyInNoPad(gmOffset, PIPELINE_TILE_ELEMS);

        uint32_t outOffset = gmOffset;
        uint32_t inOffset = gmOffset + PIPELINE_TILE_ELEMS;
        uint32_t tailLen = elemCount - PIPELINE_TILE_ELEMS;

        while (tailLen > PIPELINE_TILE_ELEMS) {
            CopyInNoPad(inOffset, PIPELINE_TILE_ELEMS);
            ComputeWithCurrentMask(xSquared, binaryParams);
            CopyOutNoPad(outOffset, PIPELINE_TILE_ELEMS);
            outOffset = inOffset;
            inOffset += PIPELINE_TILE_ELEMS;
            tailLen -= PIPELINE_TILE_ELEMS;
        }

        if (needPad) {
            CopyInWithPad(inOffset, tailLen);
        } else {
            CopyInNoPad(inOffset, tailLen);
        }
        ComputeWithCurrentMask(xSquared, binaryParams);
        CopyOutNoPad(outOffset, PIPELINE_TILE_ELEMS);
        if (tailLen != PIPELINE_TILE_ELEMS) {
            AscendC::SetVectorMask<DT_X, AscendC::MaskMode::COUNTER>(tailLen);
        }
        ComputeWithCurrentMask(xSquared, binaryParams);
        if (needPad) {
            CopyOutWithPad(inOffset, tailLen);
        } else {
            CopyOutNoPad(inOffset, tailLen);
        }
    }

private:
    __aicore__ inline void ComputeWithCurrentMask(AscendC::LocalTensor<DT_X> &xSquared,
                                                  const AscendC::BinaryRepeatParams &binaryParams) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();

        ComputeErfTaylor<DT_X>(yLocal, xLocal, xSquared, binaryParams);

        outQueueY.EnQue<DT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyInNoPad(uint32_t offset, uint32_t elemCount) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        ::CopyInNoPad<DT_X>(xLocal, xGm[offset], elemCount);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyInWithPad(uint32_t offset, uint32_t elemCount) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        ::CopyInWithPad<DT_X>(xLocal, xGm[offset], elemCount);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyOutNoPad(uint32_t offset, uint32_t elemCount) {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        ::CopyOutNoPad<DT_X>(yGm[offset], yLocal, elemCount);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutWithPad(uint32_t offset, uint32_t elemCount) {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        ::CopyOutWithPad<DT_X>(yGm[offset], yLocal, elemCount);
        outQueueY.FreeTensor(yLocal);
    }

private:
    uint32_t gmOffset;
    uint32_t elemCount;
    AscendC::GlobalTensor<DT_X> xGm, yGm;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
};

template <class DT_X>
__aicore__ inline void RunLocalPade(GM_ADDR x, GM_ADDR y, uint32_t coreOffset,
                                    uint32_t coreLen, bool needPad) {
    KernelErfLocal<DT_X> localOp;
    localOp.ProcessPade(x, y, coreOffset, coreLen, needPad);
}

template <class DT_X>
__aicore__ inline void RunLocalTaylor(GM_ADDR x, GM_ADDR y, uint32_t coreOffset,
                                      uint32_t coreLen, bool needPad) {
    KernelErfLocal<DT_X> localOp;
    localOp.ProcessTaylor(x, y, coreOffset, coreLen, needPad);
}

template <class DT_X>
__aicore__ inline void RunPipeline(AscendC::TPipe *pipe, GM_ADDR x, GM_ADDR y,
                                   uint32_t coreOffset, uint32_t coreLen, bool needPad) {
    KernelErfPipeline<DT_X> op;
    op.InitPipeline(pipe, x, y, coreOffset, coreLen);
    op.ProcessPipeline(needPad);
}

template <class DT_X>
__aicore__ inline void RunSingle(GM_ADDR x, GM_ADDR y) {
    KernelErfLocal<DT_X> localOp;
    localOp.ProcessSingle(x, y);
}

template <typename DT_X, uint32_t schMode>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if constexpr (schMode == ERF_SCH_MODE_SINGLE) {
        RunSingle<DT_X>(x, y);
    } else {
        __gm__ ErfTilingRaw *tilingData = reinterpret_cast<__gm__ ErfTilingRaw *>(tiling);
        uint32_t coreLenAligned = tilingData->coreLenAligned;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t fullCoreNum = tilingData->fullCoreNum;
        uint32_t coreOffset;
        uint32_t coreLen;
        bool needPad = false;
        if (blockIdx < fullCoreNum) {
            coreOffset = blockIdx * coreLenAligned;
            coreLen = coreLenAligned;
        } else {
            coreOffset = fullCoreNum * coreLenAligned;
            coreLen = tilingData->tailLen;
            needPad = tilingData->tailNeedPad != 0U;
        }

        if (coreLenAligned <= PADE_TILE_ELEMS) {
            RunLocalPade<DT_X>(x, y, coreOffset, coreLen, needPad);
        } else if (coreLen <= PIPELINE_TILE_ELEMS) {
            RunLocalTaylor<DT_X>(x, y, coreOffset, coreLen, needPad);
        } else {
            AscendC::TPipe pipe;
            RunPipeline<DT_X>(&pipe, x, y, coreOffset, coreLen, needPad);
        }
    }
}
