#define K_MAX_SHAPE_DIM 0

#include <type_traits>
#include "kernel_operator.h"

using namespace AscendC;

template <typename T>
class ErfinvKernel {
public:
    __aicore__ inline ErfinvKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, int64_t totalLength, int32_t tileLength,
                                int32_t perCoreBlock, int32_t lastCoreBlock, TPipe* pipeIn)
    {
        pipe = pipeIn;
        tileSize = static_cast<uint32_t>(tileLength);

        xGm.SetGlobalBuffer((__gm__ T*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ T*)y, totalLength);

        InitCoreRange(perCoreBlock, lastCoreBlock);

        const uint32_t ioBytes = tileSize * sizeof(T);
        pipe->InitBuffer(inQueue, 2, ioBytes);
        pipe->InitBuffer(outQueue, 2, ioBytes);

        if constexpr (std::is_same_v<T, bfloat16_t>) {
            const uint32_t castAligned = ((tileSize + 15U) / 16U) * 16U * sizeof(float);
            const uint32_t padAligned = ((tileSize * sizeof(float) + 31U) / 32U) * 32U;
            const uint32_t f32Bytes = (castAligned > padAligned) ? castAligned : padAligned;
            pipe->InitBuffer(xFloatBuf, f32Bytes);
            pipe->InitBuffer(wBuf, f32Bytes);
            pipe->InitBuffer(r1Buf, f32Bytes);
            pipe->InitBuffer(r2Buf, f32Bytes);
            pipe->InitBuffer(tmpBuf, f32Bytes);
            pipe->InitBuffer(outFloatBuf, f32Bytes);
        } else if constexpr (std::is_same_v<T, float>) {
            const uint32_t calcAligned = ((tileSize + 15U) / 16U) * 16U * sizeof(T);
            const uint32_t padAligned = ((tileSize * sizeof(T) + 31U) / 32U) * 32U;
            const uint32_t calcBytes = (calcAligned > padAligned) ? calcAligned : padAligned;
            pipe->InitBuffer(wBuf, calcBytes);
            pipe->InitBuffer(tmpBuf, calcBytes);
        } else if constexpr (std::is_same_v<T, half>) {
            const uint32_t calcAligned = ((tileSize + 15U) / 16U) * 16U * sizeof(T);
            const uint32_t padAligned = ((tileSize * sizeof(T) + 31U) / 32U) * 32U;
            const uint32_t calcBytes = (calcAligned > padAligned) ? calcAligned : padAligned;
            pipe->InitBuffer(wBuf, calcBytes);
            pipe->InitBuffer(r1Buf, calcBytes);
            pipe->InitBuffer(tmpBuf, calcBytes);
        }

        const uint32_t maskBytes = ((tileSize / 8U + 31U) / 32U) * 32U;
        pipe->InitBuffer(maskBuf, maskBytes);
        pipe->InitBuffer(maskBuf2, maskBytes);
    }

    __aicore__ inline void Process()
    {
        if (tileNum <= 0) {
            return;
        }
        for (uint32_t i = 0U; i < tileNum; ++i) {
            uint32_t curTile = tileSize;
            if (i == tileNum - 1U) {
                curTile = lastTileLen;
            }
            CopyIn(i, curTile);
            Compute(curTile);
            CopyOut(i, curTile);
        }
    }

private:
    static constexpr uint32_t ALIGN_BYTES = 1024U;
    static constexpr uint32_t ELEM_PER_BLOCK = ALIGN_BYTES / sizeof(T);

    __aicore__ inline void InitCoreRange(int32_t perCoreBlock, int32_t lastCoreBlock)
    {
        const int32_t coreIdx = static_cast<int32_t>(GetBlockIdx());
        const int32_t myBlock = perCoreBlock + ((coreIdx < lastCoreBlock) ? 1 : 0);
        start = (static_cast<int64_t>(coreIdx) * perCoreBlock +
                 ((coreIdx < lastCoreBlock) ? coreIdx : lastCoreBlock)) *
                ELEM_PER_BLOCK;
        localLen = static_cast<uint32_t>(myBlock * ELEM_PER_BLOCK);
        tileNum = (localLen + tileSize - 1U) / tileSize;
        lastTileLen = localLen - tileSize * (tileNum - 1U);
    }

    template <typename CalcT>
    __aicore__ inline void HornerRegion1(LocalTensor<CalcT>& result, LocalTensor<CalcT>& x, uint32_t count)
    {
        Duplicate(result, static_cast<CalcT>(2.81022636e-08f), count);
        Mul(result, x, result, count); Adds(result, result, static_cast<CalcT>(3.43273939e-07f), count);
        Mul(result, x, result, count); Adds(result, result, static_cast<CalcT>(-3.52338770e-06f), count);
        Mul(result, x, result, count); Adds(result, result, static_cast<CalcT>(-4.39150654e-06f), count);
        Mul(result, x, result, count); Adds(result, result, static_cast<CalcT>(2.18580870e-04f), count);
        Mul(result, x, result, count); Adds(result, result, static_cast<CalcT>(-1.25372503e-03f), count);
        Mul(result, x, result, count); Adds(result, result, static_cast<CalcT>(-4.17768164e-03f), count);
        Mul(result, x, result, count); Adds(result, result, static_cast<CalcT>(2.46640727e-01f), count);
        Mul(result, x, result, count); Adds(result, result, static_cast<CalcT>(1.50140941e+00f), count);
    }

    template <typename CalcT>
    __aicore__ inline void HornerRegion2(LocalTensor<CalcT>& result, LocalTensor<CalcT>& x, uint32_t count)
    {
        Duplicate(result, static_cast<CalcT>(-2.00214257e-04f), count);
        Mul(result, x, result, count); Adds(result, result, static_cast<CalcT>(1.00950558e-04f), count);
        Mul(result, x, result, count); Adds(result, result, static_cast<CalcT>(1.34934322e-03f), count);
        Mul(result, x, result, count); Adds(result, result, static_cast<CalcT>(-3.67342844e-03f), count);
        Mul(result, x, result, count); Adds(result, result, static_cast<CalcT>(5.73950773e-03f), count);
        Mul(result, x, result, count); Adds(result, result, static_cast<CalcT>(-7.62246130e-03f), count);
        Mul(result, x, result, count); Adds(result, result, static_cast<CalcT>(9.43887047e-03f), count);
        Mul(result, x, result, count); Adds(result, result, static_cast<CalcT>(1.00167406e+00f), count);
        Mul(result, x, result, count); Adds(result, result, static_cast<CalcT>(2.83297682e+00f), count);
    }

    __aicore__ inline void HornerFp32NoTail(LocalTensor<float>& result, LocalTensor<float>& w, uint32_t count)
    {
        Duplicate(result, 2.872399818e-10f, count);
        Mul(result, w, result, count); Adds(result, result, -2.082846673e-10f, count);
        Mul(result, w, result, count); Adds(result, result, -5.205722431e-08f, count);
        Mul(result, w, result, count); Adds(result, result, 2.792625935e-07f, count);
        Mul(result, w, result, count); Adds(result, result, 5.335971878e-07f, count);
        Mul(result, w, result, count); Adds(result, result, -1.102763144e-05f, count);
        Mul(result, w, result, count); Adds(result, result, 2.317826875e-04f, count);
        Mul(result, w, result, count); Adds(result, result, -5.044801626e-03f, count);
        Mul(result, w, result, count); Adds(result, result, 1.764084101e-01f, count);
        Mul(result, w, result, count); Adds(result, result, 2.661434174e+00f, count);
    }

    __aicore__ inline void HornerFp16P0Low(LocalTensor<half>& result, LocalTensor<half>& w, uint32_t count)
    {
        Duplicate(result, static_cast<half>(1.751393095e-04f), count);
        Mul(result, w, result, count); Adds(result, result, static_cast<half>(-1.288505775e-03f), count);
        Mul(result, w, result, count); Adds(result, result, static_cast<half>(-4.045503759e-03f), count);
        Mul(result, w, result, count); Adds(result, result, static_cast<half>(2.467315457e-01f), count);
        Mul(result, w, result, count); Adds(result, result, static_cast<half>(1.501366611e+00f), count);
    }

    __aicore__ inline void CopyIn(uint32_t idx, uint32_t curTile)
    {
        LocalTensor<T> xLocal = inQueue.template AllocTensor<T>();
        const int64_t offset = start + static_cast<int64_t>(idx) * tileSize;
        const uint32_t blockLen = curTile * sizeof(T);
        DataCopyExtParams copyParams{1U, blockLen, 0U, 0U, 0U};
        DataCopyPadExtParams<T> padParams{true, 0U, 0U, static_cast<T>(0)};
        DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueue.EnQue(xLocal);
    }

    template <typename CalcT>
    __aicore__ inline void ComputeApprox(LocalTensor<CalcT>& yCalc, LocalTensor<CalcT>& xCalc, uint32_t count)
    {
        LocalTensor<CalcT> wLocal = wBuf.template Get<CalcT>();
        LocalTensor<CalcT> r1Local = r1Buf.template Get<CalcT>();
        LocalTensor<CalcT> r2Local = r2Buf.template Get<CalcT>();
        LocalTensor<CalcT> tmpLocal = tmpBuf.template Get<CalcT>();
        LocalTensor<uint8_t> maskLocal = maskBuf.template Get<uint8_t>();

        Abs(wLocal, xCalc, count);

        Muls(tmpLocal, wLocal, static_cast<CalcT>(-1.0f), count);
        Adds(tmpLocal, tmpLocal, static_cast<CalcT>(1.0f), count);
        Adds(wLocal, wLocal, static_cast<CalcT>(1.0f), count);
        Mul(wLocal, tmpLocal, wLocal, count);
        Log(wLocal, wLocal, count);
        Muls(wLocal, wLocal, static_cast<CalcT>(-1.0f), count);
        PipeBarrier<PIPE_V>();

        Adds(tmpLocal, wLocal, static_cast<CalcT>(-2.5f), count);
        HornerRegion1<CalcT>(r1Local, tmpLocal, count);

        Sqrt(r2Local, wLocal, count);
        Adds(tmpLocal, r2Local, static_cast<CalcT>(-3.0f), count);
        HornerRegion2<CalcT>(r2Local, tmpLocal, count);
        PipeBarrier<PIPE_V>();

        CompareScalar(maskLocal, wLocal, static_cast<CalcT>(5.0f), CMPMODE::GT, count);
        Select(yCalc, maskLocal, r2Local, r1Local, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        Mul(yCalc, yCalc, xCalc, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeFp16P0TailPatch(LocalTensor<half>& yCalc, LocalTensor<half>& xCalc, uint32_t count)
    {
        LocalTensor<half> wLocal = wBuf.template Get<half>();
        LocalTensor<half> tmpLocal = tmpBuf.template Get<half>();

        Muls(tmpLocal, xCalc, static_cast<half>(-1.0f), count);
        Adds(tmpLocal, tmpLocal, static_cast<half>(1.0f), count);
        Adds(wLocal, xCalc, static_cast<half>(1.0f), count);
        Mul(wLocal, tmpLocal, wLocal, count);
        Log(wLocal, wLocal, count);
        Muls(wLocal, wLocal, static_cast<half>(-1.0f), count);
        Adds(wLocal, wLocal, static_cast<half>(-2.5f), count);
        PipeBarrier<PIPE_V>();

        HornerFp16P0Low(yCalc, wLocal, count);
        Mul(yCalc, yCalc, xCalc, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeFp32NoTail(LocalTensor<float>& yCalc, LocalTensor<float>& xCalc, uint32_t count)
    {
        LocalTensor<float> wLocal = wBuf.template Get<float>();
        LocalTensor<float> tmpLocal = tmpBuf.template Get<float>();

        Muls(tmpLocal, xCalc, -1.0f, count);
        Adds(tmpLocal, tmpLocal, 1.0f, count);
        Adds(wLocal, xCalc, 1.0f, count);
        Mul(wLocal, tmpLocal, wLocal, count);
        Log(wLocal, wLocal, count);
        Muls(wLocal, wLocal, -1.0f, count);
        Adds(wLocal, wLocal, -8.0f, count);
        PipeBarrier<PIPE_V>();

        HornerFp32NoTail(yCalc, wLocal, count);
        Mul(yCalc, yCalc, xCalc, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Compute(uint32_t curTile)
    {
        LocalTensor<T> xLocal = inQueue.template DeQue<T>();
        LocalTensor<T> yLocal = outQueue.template AllocTensor<T>();
        const uint32_t count = curTile;

        if constexpr (std::is_same_v<T, bfloat16_t>) {
            LocalTensor<float> xF = xFloatBuf.template Get<float>();
            LocalTensor<float> yF = outFloatBuf.template Get<float>();
            Cast(xF, xLocal, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            inQueue.FreeTensor(xLocal);
            ComputeFp32NoTail(yF, xF, count);
            Cast(yLocal, yF, RoundMode::CAST_ROUND, count);
        } else {
            if constexpr (std::is_same_v<T, float>) {
                ComputeFp32NoTail(yLocal, xLocal, count);
            } else if constexpr (std::is_same_v<T, half>) {
                ComputeFp16P0TailPatch(yLocal, xLocal, count);
            } else {
                ComputeApprox<T>(yLocal, xLocal, count);
            }
            inQueue.FreeTensor(xLocal);
        }

        outQueue.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t idx, uint32_t curTile)
    {
        LocalTensor<T> yLocal = outQueue.template DeQue<T>();
        const int64_t offset = start + static_cast<int64_t>(idx) * tileSize;
        const uint32_t blockLen = curTile * sizeof(T);
        DataCopyExtParams copyParams{1U, blockLen, 0U, 0U, 0U};
        DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueue.FreeTensor(yLocal);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, 2> inQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;
    TBuf<TPosition::VECCALC> xFloatBuf;
    TBuf<TPosition::VECCALC> wBuf;
    TBuf<TPosition::VECCALC> r1Buf;
    TBuf<TPosition::VECCALC> r2Buf;
    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> outFloatBuf;
    TBuf<TPosition::VECCALC> maskBuf;
    TBuf<TPosition::VECCALC> maskBuf2;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    int64_t start = 0;
    uint32_t localLen = 0U;
    uint32_t tileSize = 0U;
    uint32_t tileNum = 0U;
    uint32_t lastTileLen = 0U;
};

extern "C" __global__ __aicore__
void erfinv(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    ErfinvKernel<DTYPE_X> op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileLength, tilingData.perCoreBlock,
            tilingData.lastCoreBlock, &pipe);
    op.Process();
}
