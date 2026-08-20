#include "kernel_operator.h"
#include <type_traits>

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t SMALL_TILE_LENGTH = 8192;
constexpr uint32_t ONE_BUF_ELEMS = 64;

__aicore__ inline constexpr uint32_t AlignUp32(uint32_t value)
{
    return (value + 31U) & ~31U;
}

// Fast path: length is multiple of 64 AND fullRep <= 255. Single Sub call, no scalar branches.
// Caller guarantees the constraints (true for all Regular path tiles + aligned SingleShot blocks).
__aicore__ inline void BroadcastSubAligned(const AscendC::LocalTensor<float> &dst,
                                           const AscendC::LocalTensor<float> &oneBuf,
                                           const AscendC::LocalTensor<float> &src, uint32_t length)
{
    AscendC::BinaryRepeatParams params{1, 1, 1, 8, 0, 8};
    AscendC::Sub(dst, oneBuf, src, 64ULL, static_cast<uint8_t>(length >> 6), params);
}

// General path: handles non-64-aligned length (for SingleShot tail).
__aicore__ inline void BroadcastSub(const AscendC::LocalTensor<float> &dst,
                                    const AscendC::LocalTensor<float> &oneBuf,
                                    const AscendC::LocalTensor<float> &src, uint32_t length)
{
    AscendC::BinaryRepeatParams params{1, 1, 1, 8, 0, 8};
    uint32_t fullRep = length >> 6;
    uint32_t tail = length & 63;
    if (fullRep > 0) {
        AscendC::Sub(dst, oneBuf, src, 64ULL, static_cast<uint8_t>(fullRep), params);
    }
    if (tail > 0) {
        AscendC::Sub(dst[fullRep * 64], oneBuf, src[fullRep * 64], static_cast<uint64_t>(tail), 1, params);
    }
}

template <typename T> class KernelAtanh {
public:
    __aicore__ inline KernelAtanh() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t totalLength, uint32_t blockLength,
                                uint32_t tileLength, AscendC::TPipe *pipeIn)
    {
        this->pipe = pipeIn;
        this->totalLength = totalLength;
        this->tileLength = tileLength;
        const uint32_t start = AscendC::GetBlockIdx() * blockLength;
        const uint32_t remain = totalLength > start ? totalLength - start : 0;
        this->processLength = remain < blockLength ? remain : blockLength;
        inputGm.SetGlobalBuffer((__gm__ T *)input + start, this->processLength);
        outputGm.SetGlobalBuffer((__gm__ T *)output + start, this->processLength);
        pipe->InitBuffer(inQueue, BUFFER_NUM, tileLength * sizeof(T));
        pipe->InitBuffer(outQueue, BUFFER_NUM, tileLength * sizeof(T));
        pipe->InitBuffer(work1Buf, tileLength * sizeof(float));
        pipe->InitBuffer(oneBuf, ONE_BUF_ELEMS * sizeof(float));
        if constexpr (!std::is_same_v<T, float>) {
            pipe->InitBuffer(work2Buf, tileLength * sizeof(float));
        }
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            pipe->InitBuffer(workHalfBuf, tileLength * sizeof(half));
        }
        AscendC::LocalTensor<float> oneTensor = oneBuf.Get<float>();
        AscendC::Duplicate<float>(oneTensor, 1.0f, ONE_BUF_ELEMS);
    }

    __aicore__ inline void InitSingleShot(GM_ADDR input, GM_ADDR output, uint32_t totalLength,
                                          uint32_t blockLength)
    {
        this->totalLength = totalLength;
        this->tileLength = SMALL_TILE_LENGTH;
        const uint32_t start = AscendC::GetBlockIdx() * blockLength;
        const uint32_t remain = totalLength > start ? totalLength - start : 0;
        this->processLength = remain < blockLength ? remain : blockLength;
        inputGm.SetGlobalBuffer((__gm__ T *)input + start, this->processLength);
        outputGm.SetGlobalBuffer((__gm__ T *)output + start, this->processLength);
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<float> work1 = work1Buf.Get<float>();
        AscendC::LocalTensor<float> oneTensor = oneBuf.Get<float>();
        AscendC::LocalTensor<float> work2;
        AscendC::LocalTensor<half> workHalf;
        if constexpr (!std::is_same_v<T, float>) {
            work2 = work2Buf.Get<float>();
        }
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            workHalf = workHalfBuf.Get<half>();
        }
        const uint32_t tile = this->tileLength;
        const uint32_t total = this->processLength;
        const uint32_t fullBoundary = (total / tile) * tile;
        uint32_t offset = 0;
        // Hoist Counter mode setup outside main loop — tile size is fixed
        // (only applies to non-int8/uint8 paths; those fall back to count form inside)
        if constexpr (!std::is_same_v<T, int8_t> && !std::is_same_v<T, uint8_t>) {
            if (offset < fullBoundary) {
                AscendC::SetMaskCount();
                AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(0, tile);
            }
        }
        // Main loop: only full tiles, no per-tile branches
        while (offset < fullBoundary) {
            CopyInFull(offset);
            ComputeFull(tile, work1, work2, oneTensor, workHalf);
            CopyOutFull(offset);
            offset += tile;
        }
        if constexpr (!std::is_same_v<T, int8_t> && !std::is_same_v<T, uint8_t>) {
            if (fullBoundary > 0) {
                AscendC::SetMaskNorm();
                AscendC::ResetMask();
            }
        }
        // Tail: at most one partial tile — Normal mode (count form)
        if (offset < total) {
            const uint32_t tail = total - offset;
            CopyIn(offset, tail);
            Compute(tail, work1, work2, oneTensor, workHalf);
            CopyOut(offset, tail);
        }
    }

    __aicore__ inline void ProcessSingleShot()
    {
        constexpr uint32_t inAddr = 0;
        constexpr uint32_t outAddr = AlignUp32(inAddr + SMALL_TILE_LENGTH * sizeof(T));
        constexpr uint32_t work1Addr = AlignUp32(outAddr + SMALL_TILE_LENGTH * sizeof(T));
        constexpr uint32_t work2Addr = AlignUp32(work1Addr + SMALL_TILE_LENGTH * sizeof(float));
        constexpr uint32_t oneAddr = AlignUp32(work2Addr + SMALL_TILE_LENGTH * sizeof(float));
        constexpr uint32_t workHalfAddr = AlignUp32(oneAddr + ONE_BUF_ELEMS * sizeof(float));

        AscendC::LocalTensor<T> inputLocal(AscendC::TPosition::VECCALC, inAddr, SMALL_TILE_LENGTH);
        AscendC::LocalTensor<T> outputLocal(AscendC::TPosition::VECCALC, outAddr, SMALL_TILE_LENGTH);
        AscendC::LocalTensor<float> work1(AscendC::TPosition::VECCALC, work1Addr, SMALL_TILE_LENGTH);
        AscendC::LocalTensor<float> work2(AscendC::TPosition::VECCALC, work2Addr, SMALL_TILE_LENGTH);
        AscendC::LocalTensor<float> oneTensor(AscendC::TPosition::VECCALC, oneAddr, ONE_BUF_ELEMS);
        AscendC::LocalTensor<half> workHalf(AscendC::TPosition::VECCALC, workHalfAddr, SMALL_TILE_LENGTH);

        AscendC::Duplicate<float>(oneTensor, 1.0f, ONE_BUF_ELEMS);

        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->processLength * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(inputLocal, inputGm, copyParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        ComputeCore(inputLocal, outputLocal, work1, work2, oneTensor, workHalf, this->processLength);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(outputGm, outputLocal, copyParams);
    }

private:
    // Fast path: full-tile copy, no branch, no pad-params construction.
    __aicore__ inline void CopyInFull(uint32_t offset)
    {
        AscendC::LocalTensor<T> inputLocal = inQueue.AllocTensor<T>();
        AscendC::DataCopy(inputLocal, inputGm[offset], this->tileLength);
        inQueue.EnQue(inputLocal);
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length)
    {
        AscendC::LocalTensor<T> inputLocal = inQueue.AllocTensor<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(inputLocal, inputGm[offset], copyParams, padParams);
        inQueue.EnQue(inputLocal);
    }

    // Tail path: handles non-64-aligned length (uses BroadcastSub general).
    __aicore__ inline void ComputeCore(const AscendC::LocalTensor<T> &inputLocal,
                                       const AscendC::LocalTensor<T> &outputLocal,
                                       const AscendC::LocalTensor<float> &work1,
                                       const AscendC::LocalTensor<float> &work2,
                                       const AscendC::LocalTensor<float> &oneTensor,
                                       const AscendC::LocalTensor<half> &workHalf, uint32_t length)
    {
        if constexpr (std::is_same_v<T, float>) {
            AscendC::Adds(work1, inputLocal, 1.0f, length);
            BroadcastSub(outputLocal, oneTensor, inputLocal, length);
            AscendC::Div(outputLocal, work1, outputLocal, length);
            AscendC::Ln(outputLocal, outputLocal, length);
            AscendC::Muls(outputLocal, outputLocal, 0.5f, length);
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            AscendC::Cast(workHalf, inputLocal, AscendC::RoundMode::CAST_NONE, length);
            AscendC::Cast(work1, workHalf, AscendC::RoundMode::CAST_NONE, length);
            AscendC::Adds(work2, work1, 1.0f, length);
            BroadcastSub(work1, oneTensor, work1, length);
            AscendC::Div(work1, work2, work1, length);
            AscendC::Ln(work1, work1, length);
            AscendC::Muls(work1, work1, 0.5f, length);
            AscendC::Cast(workHalf, work1, AscendC::RoundMode::CAST_ROUND, length);
            AscendC::Cast(outputLocal, workHalf, AscendC::RoundMode::CAST_ROUND, length);
        } else {
            AscendC::Cast(work1, inputLocal, AscendC::RoundMode::CAST_NONE, length);
            AscendC::Adds(work2, work1, 1.0f, length);
            BroadcastSub(work1, oneTensor, work1, length);
            AscendC::Div(work1, work2, work1, length);
            AscendC::Ln(work1, work1, length);
            AscendC::Muls(work1, work1, 0.5f, length);
            AscendC::Cast(outputLocal, work1, AscendC::RoundMode::CAST_ROUND, length);
        }
    }

    // Fast path: length == tileLength (Counter mode pre-set by caller in Process).
    // Caller MUST do SetMaskCount + SetVectorMask BEFORE calling (and unset after loop).
    // EXCEPT int8/uint8 — those use count form internally.
    __aicore__ inline void ComputeCoreAligned(const AscendC::LocalTensor<T> &inputLocal,
                                              const AscendC::LocalTensor<T> &outputLocal,
                                              const AscendC::LocalTensor<float> &work1,
                                              const AscendC::LocalTensor<float> &work2,
                                              const AscendC::LocalTensor<float> &oneTensor,
                                              const AscendC::LocalTensor<half> &workHalf,
                                              uint32_t length)
    {
        AscendC::UnaryRepeatParams up;
        AscendC::BinaryRepeatParams bp;
        AscendC::BinaryRepeatParams bcast{1, 1, 1, 8, 0, 8};
        AscendC::UnaryRepeatParams castF2H{1, 1, 4, 8};   // dst=half (4 blk), src=float (8 blk)
        AscendC::UnaryRepeatParams castH2F{1, 1, 8, 4};   // dst=float (8 blk), src=half (4 blk)

        if constexpr (std::is_same_v<T, float>) {
            AscendC::Adds<float, false>(work1, inputLocal, 1.0f, AscendC::MASK_PLACEHOLDER, 1, up);
            AscendC::Sub<float, false>(outputLocal, oneTensor, inputLocal, AscendC::MASK_PLACEHOLDER, 1, bcast);
            AscendC::Div<float, false>(outputLocal, work1, outputLocal, AscendC::MASK_PLACEHOLDER, 1, bp);
            AscendC::Ln<float, false>(outputLocal, outputLocal, AscendC::MASK_PLACEHOLDER, 1, up);
            AscendC::Muls<float, false>(outputLocal, outputLocal, 0.5f, AscendC::MASK_PLACEHOLDER, 1, up);
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            // Use count-form for int8/uint8 paths (double Cast with int8 stride complexity); Normal mode.
            AscendC::Cast(workHalf, inputLocal, AscendC::RoundMode::CAST_NONE, length);
            AscendC::Cast(work1, workHalf, AscendC::RoundMode::CAST_NONE, length);
            AscendC::Adds(work2, work1, 1.0f, length);
            BroadcastSubAligned(work1, oneTensor, work1, length);
            AscendC::Div(work1, work2, work1, length);
            AscendC::Ln(work1, work1, length);
            AscendC::Muls(work1, work1, 0.5f, length);
            AscendC::Cast(workHalf, work1, AscendC::RoundMode::CAST_ROUND, length);
            AscendC::Cast(outputLocal, workHalf, AscendC::RoundMode::CAST_ROUND, length);
        } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
            AscendC::Cast<float, T, false>(work1, inputLocal, AscendC::RoundMode::CAST_NONE,
                                           AscendC::MASK_PLACEHOLDER, 1, castH2F);
            AscendC::Adds<float, false>(work2, work1, 1.0f, AscendC::MASK_PLACEHOLDER, 1, up);
            AscendC::Sub<float, false>(work1, oneTensor, work1, AscendC::MASK_PLACEHOLDER, 1, bcast);
            AscendC::Div<float, false>(work1, work2, work1, AscendC::MASK_PLACEHOLDER, 1, bp);
            AscendC::Ln<float, false>(work1, work1, AscendC::MASK_PLACEHOLDER, 1, up);
            AscendC::Muls<float, false>(work1, work1, 0.5f, AscendC::MASK_PLACEHOLDER, 1, up);
            AscendC::Cast<T, float, false>(outputLocal, work1, AscendC::RoundMode::CAST_ROUND,
                                           AscendC::MASK_PLACEHOLDER, 1, castF2H);
        } else {
            AscendC::UnaryRepeatParams castIn{1, 1, 8, sizeof(T) == 4 ? 8 : 4};
            AscendC::UnaryRepeatParams castOut{1, 1, sizeof(T) == 4 ? 8 : 4, 8};
            AscendC::Cast<float, T, false>(work1, inputLocal, AscendC::RoundMode::CAST_NONE,
                                           AscendC::MASK_PLACEHOLDER, 1, castIn);
            AscendC::Adds<float, false>(work2, work1, 1.0f, AscendC::MASK_PLACEHOLDER, 1, up);
            AscendC::Sub<float, false>(work1, oneTensor, work1, AscendC::MASK_PLACEHOLDER, 1, bcast);
            AscendC::Div<float, false>(work1, work2, work1, AscendC::MASK_PLACEHOLDER, 1, bp);
            AscendC::Ln<float, false>(work1, work1, AscendC::MASK_PLACEHOLDER, 1, up);
            AscendC::Muls<float, false>(work1, work1, 0.5f, AscendC::MASK_PLACEHOLDER, 1, up);
            AscendC::Cast<T, float, false>(outputLocal, work1, AscendC::RoundMode::CAST_ROUND,
                                           AscendC::MASK_PLACEHOLDER, 1, castOut);
        }
    }

    __aicore__ inline void Compute(uint32_t length, const AscendC::LocalTensor<float> &work1,
                                   const AscendC::LocalTensor<float> &work2,
                                   const AscendC::LocalTensor<float> &oneTensor,
                                   const AscendC::LocalTensor<half> &workHalf)
    {
        AscendC::LocalTensor<T> inputLocal = inQueue.DeQue<T>();
        AscendC::LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
        ComputeCore(inputLocal, outputLocal, work1, work2, oneTensor, workHalf, length);
        outQueue.EnQue(outputLocal);
        inQueue.FreeTensor(inputLocal);
    }

    __aicore__ inline void ComputeFull(uint32_t length, const AscendC::LocalTensor<float> &work1,
                                       const AscendC::LocalTensor<float> &work2,
                                       const AscendC::LocalTensor<float> &oneTensor,
                                       const AscendC::LocalTensor<half> &workHalf)
    {
        AscendC::LocalTensor<T> inputLocal = inQueue.DeQue<T>();
        AscendC::LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
        ComputeCoreAligned(inputLocal, outputLocal, work1, work2, oneTensor, workHalf, length);
        outQueue.EnQue(outputLocal);
        inQueue.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOutFull(uint32_t offset)
    {
        AscendC::LocalTensor<T> outputLocal = outQueue.DeQue<T>();
        AscendC::DataCopy(outputGm[offset], outputLocal, this->tileLength);
        outQueue.FreeTensor(outputLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length)
    {
        AscendC::LocalTensor<T> outputLocal = outQueue.DeQue<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(outputGm[offset], outputLocal, copyParams);
        outQueue.FreeTensor(outputLocal);
    }

private:
    AscendC::TPipe *pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> work1Buf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> oneBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> work2Buf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> workHalfBuf;
    AscendC::GlobalTensor<T> inputGm;
    AscendC::GlobalTensor<T> outputGm;
    uint32_t totalLength;
    uint32_t processLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void atanh(GM_ADDR input, GM_ADDR output, GM_ADDR, GM_ADDR tiling)
{
    AscendC::InitSocState();
    GET_TILING_DATA(tiling_data, tiling);
    KernelAtanh<DTYPE_INPUT> op;
    if (tiling_data.blockLength <= SMALL_TILE_LENGTH) {
        op.InitSingleShot(input, output, tiling_data.totalLength, tiling_data.blockLength);
        op.ProcessSingleShot();
    } else {
        AscendC::TPipe pipe;
        op.Init(input, output, tiling_data.totalLength, tiling_data.blockLength, tiling_data.tileLength, &pipe);
        op.Process();
    }
}
