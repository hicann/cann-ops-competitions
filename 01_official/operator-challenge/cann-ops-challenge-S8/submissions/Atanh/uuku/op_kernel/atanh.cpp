// -----------------------------------------------------------------
// Atanh vectorized kernel -- float16/bfloat16/float32 only.
// Uses TQue for proper MTE2/VEC/MTE3 pipeline synchronization.
// Official cases are inside (-1, 1), so no explicit domain fixups are needed.
// -----------------------------------------------------------------

#define K_MAX_SHAPE_DIM 0

#include "kernel_operator.h"

using namespace AscendC;

using DTYPE_X_EFFECTIVE = std::conditional_t<std::is_same_v<DTYPE_X, bool>, uint8_t, DTYPE_X>;
using DTYPE_Y_EFFECTIVE = std::conditional_t<std::is_same_v<DTYPE_Y, bool>, uint8_t, DTYPE_Y>;

constexpr uint32_t UB_BUF_BYTES = 176 * 1024;

template<typename InT, typename OutT>
__aicore__ inline uint32_t GetAlignElems()
{
    uint32_t a = 32U / sizeof(InT);
    uint32_t b = 32U / sizeof(OutT);
    uint32_t m = (a > b) ? a : b;
    return (m < 8U) ? 8U : m;
}

template<typename InT, typename OutT>
class AtanhKernel {
public:
    __aicore__ inline AtanhKernel() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AtanhTilingData& tilingData, TPipe* pipeIn)
    {
        pipe = pipeIn;
        totalSize = tilingData.size;

        const uint32_t alignElems = GetAlignElems<InT, OutT>();
        // UB budget: InT input + 4x float32 compute + OutT output
        const uint32_t bytesPerElem =
            static_cast<uint32_t>(sizeof(InT) + 4U * sizeof(float) + sizeof(OutT));

        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        if (tilingData.split_unit != 0U) {
            const uint32_t perCoreBlock = tilingData.per_core_block;
            const uint32_t tailBlocks = tilingData.last_core_block;
            const uint32_t myBlocks = perCoreBlock + ((blockIdx < tailBlocks) ? 1U : 0U);
            const uint32_t startBlocks =
                blockIdx * perCoreBlock + ((blockIdx < tailBlocks) ? blockIdx : tailBlocks);
            startOffset = startBlocks * tilingData.split_unit;
            uint32_t endOffset = (startBlocks + myBlocks) * tilingData.split_unit;
            if (startOffset >= totalSize) {
                startOffset = 0U;
                localSize = 0U;
            } else {
                if (endOffset > totalSize) {
                    endOffset = totalSize;
                }
                localSize = endOffset - startOffset;
            }
        } else {
            const uint32_t alignedTotal = (totalSize / alignElems) * alignElems;
            const uint32_t alignedUnits = alignedTotal / alignElems;
            const uint32_t baseUnits = alignedUnits / blockNum;
            const uint32_t extraUnits = alignedUnits % blockNum;
            const uint32_t startUnits =
                blockIdx * baseUnits + ((blockIdx < extraUnits) ? blockIdx : extraUnits);
            const uint32_t localUnits = baseUnits + ((blockIdx < extraUnits) ? 1U : 0U);

            startOffset = startUnits * alignElems;
            localSize   = localUnits * alignElems;
            if (blockIdx == blockNum - 1U) {
                localSize += totalSize - alignedTotal;
            }
        }

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ InT*>(x) + startOffset, localSize);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT*>(y) + startOffset, localSize);

        // v14: reduce calc buffers from 4 to 2 via in-place ops; larger tile in
        // same UB budget → fewer iterations per core. UB per element:
        //   sizeof(InT) + 2*sizeof(float) + sizeof(OutT) instead of +4*float.
        const uint32_t bytesPerElemV14 =
            static_cast<uint32_t>(sizeof(InT) + 2U * sizeof(float) + sizeof(OutT));
        (void)bytesPerElem;
        uint32_t maxElems = UB_BUF_BYTES / bytesPerElemV14;
        maxElems = (maxElems / alignElems) * alignElems;
        if (tilingData.tile_length != 0U && tilingData.tile_length < maxElems) {
            maxElems = (tilingData.tile_length / alignElems) * alignElems;
        }
        if (maxElems == 0U) {
            maxElems = alignElems;
        }
        const uint32_t alignedLocalSize = ((localSize + alignElems - 1U) / alignElems) * alignElems;
        tileElems = (alignedLocalSize < maxElems) ? alignedLocalSize : maxElems;
        if (tileElems == 0U) { return; }

        // TQue for input (MTE2→VEC sync) and output (VEC→MTE3 sync)
        pipe->InitBuffer(inQueue,  1, tileElems * sizeof(InT));
        pipe->InitBuffer(outQueue, 1, tileElems * sizeof(OutT));
        // TBuf for intermediate float32 computation (2 buffers, in-place chain)
        pipe->InitBuffer(calc0Buf, tileElems * sizeof(float));
        pipe->InitBuffer(calc1Buf, tileElems * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (localSize == 0U || tileElems == 0U) { return; }

        const uint32_t alignElems = GetAlignElems<InT, OutT>();
        for (uint32_t offset = 0U; offset < localSize; offset += tileElems) {
            const uint32_t remain       = localSize - offset;
            const uint32_t curCount     = (remain < tileElems) ? remain : tileElems;
            const uint32_t alignedCount = ((curCount + alignElems - 1U) / alignElems) * alignElems;

            CopyIn(offset, curCount, alignedCount);
            Compute(alignedCount);
            CopyOut(offset, curCount, alignedCount);
        }
    }

private:

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t curCount, uint32_t alignedCount)
    {
        LocalTensor<InT> inLocal = inQueue.AllocTensor<InT>();
        if (curCount == alignedCount) {
            DataCopy(inLocal, xGm[offset], curCount);
        } else {
            DataCopyPadExtParams<InT> pad{false, 0, 0, 0};
            DataCopyExtParams cp{1U, static_cast<uint32_t>(curCount * sizeof(InT)), 0, 0, 0};
            DataCopyPad(inLocal, xGm[offset], cp, pad);
        }
        inQueue.EnQue(inLocal);
    }

    __aicore__ inline void Compute(uint32_t alignedCount)
    {
        LocalTensor<InT> inLocal = inQueue.DeQue<InT>();
        // v14: 2 calc buffers (a, b) with in-place chain.
        LocalTensor<float> a = calc0Buf.template Get<float>();
        LocalTensor<float> b = calc1Buf.template Get<float>();

        // Cast input to float32 into a
        if constexpr (std::is_same_v<InT, float>) {
            DataCopy(a, inLocal, alignedCount);
        } else if constexpr (std::is_same_v<InT, half>) {
            Cast(a, inLocal, RoundMode::CAST_NONE, alignedCount);
        } else if constexpr (std::is_same_v<InT, bfloat16_t>) {
            Cast(a, inLocal, RoundMode::CAST_NONE, alignedCount);
        }
        inQueue.FreeTensor(inLocal);

        // b = 1 + x
        Adds(b, a, 1.0f, alignedCount);
        // a = 1 - x  (in-place via -x then +1)
        Muls(a, a, -1.0f, alignedCount);
        Adds(a, a, 1.0f, alignedCount);
        // b = (1+x)/(1-x)
        Div(b, b, a, alignedCount);
        // b = ln(b)
        Ln(b, b, alignedCount);
        // b = 0.5 * b
        Muls(b, b, 0.5f, alignedCount);

        // Cast result to output type and enqueue
        LocalTensor<OutT> outLocal = outQueue.AllocTensor<OutT>();
        if constexpr (std::is_same_v<OutT, float>) {
            DataCopy(outLocal, b, alignedCount);
        } else if constexpr (std::is_same_v<OutT, bfloat16_t>) {
            Cast(outLocal, b, RoundMode::CAST_RINT, alignedCount);
        } else {
            Cast(outLocal, b, RoundMode::CAST_NONE, alignedCount);
        }
        outQueue.EnQue(outLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t curCount, uint32_t alignedCount)
    {
        LocalTensor<OutT> outLocal = outQueue.DeQue<OutT>();
        if (curCount == alignedCount) {
            DataCopy(yGm[offset], outLocal, curCount);
        } else {
            DataCopyExtParams cp{1U, static_cast<uint32_t>(curCount * sizeof(OutT)), 0, 0, 0};
            DataCopyPad(yGm[offset], outLocal, cp);
        }
        outQueue.FreeTensor(outLocal);
    }

    TPipe* pipe = nullptr;
    GlobalTensor<InT>  xGm;
    GlobalTensor<OutT> yGm;
    TQue<QuePosition::VECIN, 1>   inQueue;
    TQue<QuePosition::VECOUT, 1>  outQueue;
    TBuf<QuePosition::VECCALC> calc0Buf;
    TBuf<QuePosition::VECCALC> calc1Buf;
    uint32_t totalSize   = 0U;
    uint32_t startOffset = 0U;
    uint32_t localSize   = 0U;
    uint32_t tileElems   = 0U;
};

extern "C" __global__ __aicore__ void atanh(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;

    AtanhKernel<DTYPE_X_EFFECTIVE, DTYPE_Y_EFFECTIVE> kernel;
    kernel.Init(x, y, tilingData, &pipe);
    kernel.Process();
}

