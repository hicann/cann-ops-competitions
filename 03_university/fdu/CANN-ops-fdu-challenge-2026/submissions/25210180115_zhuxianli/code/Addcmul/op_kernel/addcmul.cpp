// Kernel侧核函数实现
#include "kernel_operator.h"
#include <type_traits>

#include "addcmul_tiling.h"
#include "tiling_key_addcmul.h"

#ifndef FDU_BUFFER_NUM
#define FDU_BUFFER_NUM 2
#endif

template <class DT_INPUT_DATA>
class KernelAddcmul {
public:
    __aicore__ inline KernelAddcmul() {}
    __aicore__ inline void Init(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
                                const AddcmulTilingData &tilingData) {
        this->tilingData = tilingData;
        blockDim = tilingData.blockDim == 0 ? 1 : tilingData.blockDim;
        tileLength = tilingData.tileLength == 0 ? 32 : tilingData.tileLength;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        // Broadcast paths pass a K-aligned per-core stride so every core handles whole rows.
        uint32_t lengthPerCore = (tilingData.coreStride != 0)
                                     ? tilingData.coreStride
                                     : (tilingData.length + blockDim - 1) / blockDim;
        startOffset = blockIdx * lengthPerCore;
        coreLength = startOffset < tilingData.length ? tilingData.length - startOffset : 0;
        if (coreLength > lengthPerCore) {
            coreLength = lengthPerCore;
        }

        inputGm.SetGlobalBuffer((__gm__ DT_INPUT_DATA *)input_data, tilingData.length);
        x1Gm.SetGlobalBuffer((__gm__ DT_INPUT_DATA *)x1, tilingData.length);
        x2Gm.SetGlobalBuffer((__gm__ DT_INPUT_DATA *)x2, tilingData.length);
        valueGm.SetGlobalBuffer((__gm__ DT_INPUT_DATA *)value, 1);
        yGm.SetGlobalBuffer((__gm__ DT_INPUT_DATA *)y, tilingData.length);
        valueScalar = valueGm.GetValue(0);

        pipe.InitBuffer(inQueueInput, FDU_BUFFER_NUM, tileLength * sizeof(DT_INPUT_DATA));
        pipe.InitBuffer(inQueueX1, FDU_BUFFER_NUM, tileLength * sizeof(DT_INPUT_DATA));
        pipe.InitBuffer(inQueueX2, FDU_BUFFER_NUM, tileLength * sizeof(DT_INPUT_DATA));
        pipe.InitBuffer(outQueueY, FDU_BUFFER_NUM, tileLength * sizeof(DT_INPUT_DATA));

        // int8 vectorized paths (contiguous AND structured broadcast) need int32/half scratch.
        // Host caps int8 tileLength so this fits UB. Skip only the GENERAL per-element fallback.
        if constexpr (std::is_same_v<DT_INPUT_DATA, int8_t>) {
            bool anyGeneral = (tilingData.inputKind == 4) || (tilingData.x1Kind == 4) || (tilingData.x2Kind == 4);
            if (!anyGeneral) {
                pipe.InitBuffer(tbufHalf, tileLength * sizeof(half));
                pipe.InitBuffer(tbufW0, tileLength * sizeof(int32_t));
                pipe.InitBuffer(tbufW1, tileLength * sizeof(int32_t));
            }
        }

        // Row-vector fast path (fp/half/int32): persistent UB buffers holding a broadcast row-vector
        // operand replicated across the tile. Only allocated for the operands that are row-vectors.
        if constexpr (!std::is_same_v<DT_INPUT_DATA, int8_t>) {
            if (tilingData.fastBcast != 0) {
                if (tilingData.inputKind == 2) pipe.InitBuffer(tbufRepInput, tileLength * sizeof(DT_INPUT_DATA));
                if (tilingData.x1Kind == 2) pipe.InitBuffer(tbufRepX1, tileLength * sizeof(DT_INPUT_DATA));
                if (tilingData.x2Kind == 2) pipe.InitBuffer(tbufRepX2, tileLength * sizeof(DT_INPUT_DATA));
            }
        }
    }

    __aicore__ inline void Process() {
        if (coreLength == 0) {
            return;
        }
        // AscendC's Muls / Adds / Duplicate only support half / float / int16_t / int32_t (no int8).
        // Gate the vectorized paths via `if constexpr` so the int8 instantiation
        // does not pull in the unsupported intrinsics.
        constexpr bool isFloatLike = std::is_same_v<DT_INPUT_DATA, float> || std::is_same_v<DT_INPUT_DATA, half>;
        constexpr bool isInt32 = std::is_same_v<DT_INPUT_DATA, int32_t>;

        if constexpr (isFloatLike) {
            if (tilingData.contiguous != 0) {
                ProcessContiguous();
                return;
            }
            if (tilingData.fastBcast != 0) {
                ProcessBroadcastRowVec();
                return;
            }
            bool anyGeneral = (tilingData.inputKind == 4) || (tilingData.x1Kind == 4) || (tilingData.x2Kind == 4);
            uint32_t chunkLen = ComputeChunkLen();
            if (!anyGeneral && chunkLen >= 16) {
                ProcessBroadcastChunked(chunkLen);
                return;
            }
            ProcessBroadcastVectorized();
        } else if constexpr (isInt32) {
            if (tilingData.fastBcast != 0) {
                ProcessBroadcastRowVec();
                return;
            }
            bool anyGeneral = (tilingData.inputKind == 4) || (tilingData.x1Kind == 4) || (tilingData.x2Kind == 4);
            uint32_t chunkLen = ComputeChunkLen();
            if (!anyGeneral && chunkLen >= 16) {
                ProcessBroadcastChunked(chunkLen);
                return;
            }
            ProcessBroadcastInteger();
        } else {
            // int8: vector Muls/Adds/Duplicate are unsupported on int8, but we can vectorize by
            // computing in int32 (exact for int8-derived values) and doing the int8 two's-complement
            // wrap with bit-shifts. Contiguous uses the flat path; structured broadcast uses the
            // chunked int32 path; only GENERAL broadcasts fall back to per-element int64.
            if (tilingData.contiguous != 0) {
                ProcessContiguousInt8Vec();
                return;
            }
            bool anyGeneral = (tilingData.inputKind == 4) || (tilingData.x1Kind == 4) || (tilingData.x2Kind == 4);
            if (!anyGeneral) {
                ProcessBroadcastChunkedInt8(ComputeChunkLen());
            } else {
                ProcessBroadcastInteger();
            }
        }
    }

private:
    // The inner broadcast/contig "run" length: the smallest contigLen among the
    // INNER_CONTIG / INNER_BCAST operands. Because every contigLen is a product of a
    // trailing suffix of the output dims, these runs are nested (the smallest divides
    // all the others), so a chunk that stays within one Cmin-run also stays within one
    // run of every other constrained operand. If no operand is constrained, the whole
    // output is a single run (FULL operands copy contiguously, SCALAR fold to a coef).
    __aicore__ inline uint32_t ComputeBcastRun() {
        uint32_t run = 0;  // 0 == unset (no constrained operand)
        if (tilingData.inputKind == 2 || tilingData.inputKind == 3) {
            if (run == 0 || tilingData.inputContigLen < run) run = tilingData.inputContigLen;
        }
        if (tilingData.x1Kind == 2 || tilingData.x1Kind == 3) {
            if (run == 0 || tilingData.x1ContigLen < run) run = tilingData.x1ContigLen;
        }
        if (tilingData.x2Kind == 2 || tilingData.x2Kind == 3) {
            if (run == 0 || tilingData.x2ContigLen < run) run = tilingData.x2ContigLen;
        }
        if (run == 0) run = tilingData.length;  // no constraint -> never clamps
        return run;
    }

    __aicore__ inline uint32_t ComputeChunkLen() {
        uint32_t chunkLen = ComputeBcastRun();
        if (chunkLen > coreLength) chunkLen = coreLength;
        if (chunkLen > tileLength) chunkLen = tileLength;
        if (chunkLen == 0) chunkLen = 1;
        return chunkLen;
    }

    __aicore__ inline void ProcessContiguous() {
        for (uint32_t offset = 0; offset < coreLength; offset += tileLength) {
            processLength = coreLength - offset < tileLength ? coreLength - offset : tileLength;
            uint32_t globalOffset = startOffset + offset;
            CopyInContiguous(globalOffset);
            ComputeContiguous();
            CopyOut(globalOffset);
        }
    }

    __aicore__ inline void CopyInContiguous(uint32_t globalOffset) {
        AscendC::LocalTensor<DT_INPUT_DATA> inputLocal = inQueueInput.AllocTensor<DT_INPUT_DATA>();
        AscendC::LocalTensor<DT_INPUT_DATA> x1Local = inQueueX1.AllocTensor<DT_INPUT_DATA>();
        AscendC::LocalTensor<DT_INPUT_DATA> x2Local = inQueueX2.AllocTensor<DT_INPUT_DATA>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(processLength * sizeof(DT_INPUT_DATA)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT_INPUT_DATA> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(inputLocal, inputGm[globalOffset], copyParams, padParams);
        AscendC::DataCopyPad(x1Local, x1Gm[globalOffset], copyParams, padParams);
        AscendC::DataCopyPad(x2Local, x2Gm[globalOffset], copyParams, padParams);
        inQueueInput.EnQue(inputLocal);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    __aicore__ inline void ComputeContiguous() {
        AscendC::LocalTensor<DT_INPUT_DATA> inputLocal = inQueueInput.DeQue<DT_INPUT_DATA>();
        AscendC::LocalTensor<DT_INPUT_DATA> x1Local = inQueueX1.DeQue<DT_INPUT_DATA>();
        AscendC::LocalTensor<DT_INPUT_DATA> x2Local = inQueueX2.DeQue<DT_INPUT_DATA>();
        AscendC::LocalTensor<DT_INPUT_DATA> yLocal = outQueueY.AllocTensor<DT_INPUT_DATA>();
        // Accumulate in yLocal directly to avoid a separate tmp UB buffer.
        AscendC::Mul(yLocal, x1Local, x2Local, processLength);
        AscendC::Muls(yLocal, yLocal, valueScalar, processLength);
        AscendC::Add(yLocal, yLocal, inputLocal, processLength);
        outQueueY.EnQue<DT_INPUT_DATA>(yLocal);
        inQueueInput.FreeTensor(inputLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }

    __aicore__ inline void CopyOut(uint32_t globalOffset) {
        AscendC::LocalTensor<DT_INPUT_DATA> yLocal = outQueueY.DeQue<DT_INPUT_DATA>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(processLength * sizeof(DT_INPUT_DATA)), 0, 0, 0};
        AscendC::DataCopyPad(yGm[globalOffset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline uint32_t GetBroadcastOffset(uint32_t linear, const uint32_t *stride) {
        uint32_t offset = 0;
        uint32_t residual = linear;
        for (int32_t dim = static_cast<int32_t>(tilingData.rank) - 1; dim >= 0; --dim) {
            uint32_t coord = residual % tilingData.outputShape[dim];
            residual = residual / tilingData.outputShape[dim];
            offset += coord * stride[dim];
        }
        return offset;
    }

    __aicore__ inline DT_INPUT_DATA CombineCoef(bool x1Scalar, DT_INPUT_DATA x1Val, bool x2Scalar, DT_INPUT_DATA x2Val) {
        if constexpr (std::is_integral_v<DT_INPUT_DATA>) {
            int64_t c = static_cast<int64_t>(valueScalar);
            if (x1Scalar) c *= static_cast<int64_t>(x1Val);
            if (x2Scalar) c *= static_cast<int64_t>(x2Val);
            return static_cast<DT_INPUT_DATA>(c);
        } else {
            float c = static_cast<float>(valueScalar);
            if (x1Scalar) c *= static_cast<float>(x1Val);
            if (x2Scalar) c *= static_cast<float>(x2Val);
            return static_cast<DT_INPUT_DATA>(c);
        }
    }

    // Chunked broadcast: every tensor is either fully loaded for the chunk (FULL / INNER_CONTIG) or
    // contributes a single scalar value (SCALAR / INNER_BCAST). All compute is via vector ops.
    // Accumulates into yLocal in place, no scratch UB needed.
    __aicore__ inline void ProcessBroadcastChunked(uint32_t chunkLen) {
        uint32_t inputKind = tilingData.inputKind;
        uint32_t x1Kind = tilingData.x1Kind;
        uint32_t x2Kind = tilingData.x2Kind;
        bool inputAsScalar = (inputKind == 1) || (inputKind == 3);
        bool x1AsScalar = (x1Kind == 1) || (x1Kind == 3);
        bool x2AsScalar = (x2Kind == 1) || (x2Kind == 3);

        DT_INPUT_DATA inputScalarBase = (inputKind == 1) ? inputGm.GetValue(0) : DT_INPUT_DATA(0);
        DT_INPUT_DATA x1ScalarBase = (x1Kind == 1) ? x1Gm.GetValue(0) : DT_INPUT_DATA(0);
        DT_INPUT_DATA x2ScalarBase = (x2Kind == 1) ? x2Gm.GetValue(0) : DT_INPUT_DATA(0);

        // Cmin run boundary: a chunk must never cross it, or a constrained operand's DataCopyPad
        // would read past its broadcast source row (the multi-core straddle bug). When cores are
        // K-aligned (coreStride != 0) AND chunkLen == bcastRun (the common case: one whole row per
        // chunk), a chunk can never straddle, so we skip the per-chunk modulo — it cost O(chunks)
        // and regressed many-chunk broadcasts. Clamp only when a chunk subdivides a row (tile < K)
        // or cores are not K-aligned.
        uint32_t bcastRun = ComputeBcastRun();
        bool needClamp = (chunkLen < bcastRun) || (tilingData.coreStride == 0);
        for (uint32_t offset = 0; offset < coreLength; offset += processLength) {
            uint32_t globalOffset = startOffset + offset;
            processLength = chunkLen;
            if (processLength > coreLength - offset) processLength = coreLength - offset;
            if (needClamp) {
                uint32_t toRowEnd = bcastRun - (globalOffset % bcastRun);
                if (processLength > toRowEnd) processLength = toRowEnd;
            }

            DT_INPUT_DATA inputScalar = inputScalarBase;
            DT_INPUT_DATA x1Scalar = x1ScalarBase;
            DT_INPUT_DATA x2Scalar = x2ScalarBase;
            if (inputKind == 3) {
                inputScalar = inputGm.GetValue(GetBroadcastOffset(globalOffset, tilingData.inputStride));
            }
            if (x1Kind == 3) {
                x1Scalar = x1Gm.GetValue(GetBroadcastOffset(globalOffset, tilingData.x1Stride));
            }
            if (x2Kind == 3) {
                x2Scalar = x2Gm.GetValue(GetBroadcastOffset(globalOffset, tilingData.x2Stride));
            }
            DT_INPUT_DATA coef = CombineCoef(x1AsScalar, x1Scalar, x2AsScalar, x2Scalar);

            // Load tensor inputs (FULL or INNER_CONTIG)
            if (!x1AsScalar) {
                uint32_t srcOff = (x1Kind == 0) ? globalOffset : GetBroadcastOffset(globalOffset, tilingData.x1Stride);
                AscendC::LocalTensor<DT_INPUT_DATA> x1Local = inQueueX1.AllocTensor<DT_INPUT_DATA>();
                AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(processLength * sizeof(DT_INPUT_DATA)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<DT_INPUT_DATA> pp{false, 0, 0, 0};
                AscendC::DataCopyPad(x1Local, x1Gm[srcOff], cp, pp);
                inQueueX1.EnQue(x1Local);
            }
            if (!x2AsScalar) {
                uint32_t srcOff = (x2Kind == 0) ? globalOffset : GetBroadcastOffset(globalOffset, tilingData.x2Stride);
                AscendC::LocalTensor<DT_INPUT_DATA> x2Local = inQueueX2.AllocTensor<DT_INPUT_DATA>();
                AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(processLength * sizeof(DT_INPUT_DATA)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<DT_INPUT_DATA> pp{false, 0, 0, 0};
                AscendC::DataCopyPad(x2Local, x2Gm[srcOff], cp, pp);
                inQueueX2.EnQue(x2Local);
            }
            if (!inputAsScalar) {
                uint32_t srcOff = (inputKind == 0) ? globalOffset : GetBroadcastOffset(globalOffset, tilingData.inputStride);
                AscendC::LocalTensor<DT_INPUT_DATA> inputLocal = inQueueInput.AllocTensor<DT_INPUT_DATA>();
                AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(processLength * sizeof(DT_INPUT_DATA)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<DT_INPUT_DATA> pp{false, 0, 0, 0};
                AscendC::DataCopyPad(inputLocal, inputGm[srcOff], cp, pp);
                inQueueInput.EnQue(inputLocal);
            }

            AscendC::LocalTensor<DT_INPUT_DATA> yLocal = outQueueY.AllocTensor<DT_INPUT_DATA>();

            // Compute the product term (x1 * x2 * value) into yLocal in place.
            if (!x1AsScalar && !x2AsScalar) {
                AscendC::LocalTensor<DT_INPUT_DATA> x1Local = inQueueX1.DeQue<DT_INPUT_DATA>();
                AscendC::LocalTensor<DT_INPUT_DATA> x2Local = inQueueX2.DeQue<DT_INPUT_DATA>();
                AscendC::Mul(yLocal, x1Local, x2Local, processLength);
                AscendC::Muls(yLocal, yLocal, valueScalar, processLength);
                inQueueX1.FreeTensor(x1Local);
                inQueueX2.FreeTensor(x2Local);
            } else if (!x1AsScalar && x2AsScalar) {
                AscendC::LocalTensor<DT_INPUT_DATA> x1Local = inQueueX1.DeQue<DT_INPUT_DATA>();
                AscendC::Muls(yLocal, x1Local, coef, processLength);
                inQueueX1.FreeTensor(x1Local);
            } else if (x1AsScalar && !x2AsScalar) {
                AscendC::LocalTensor<DT_INPUT_DATA> x2Local = inQueueX2.DeQue<DT_INPUT_DATA>();
                AscendC::Muls(yLocal, x2Local, coef, processLength);
                inQueueX2.FreeTensor(x2Local);
            }
            // else: x1 and x2 both scalar — yLocal is still uninitialised; handled below.

            // Combine with the input contribution.
            if (!inputAsScalar) {
                AscendC::LocalTensor<DT_INPUT_DATA> inputLocal = inQueueInput.DeQue<DT_INPUT_DATA>();
                if (x1AsScalar && x2AsScalar) {
                    AscendC::Adds(yLocal, inputLocal, coef, processLength);
                } else {
                    AscendC::Add(yLocal, yLocal, inputLocal, processLength);
                }
                inQueueInput.FreeTensor(inputLocal);
            } else {
                if (x1AsScalar && x2AsScalar) {
                    DT_INPUT_DATA total;
                    if constexpr (std::is_integral_v<DT_INPUT_DATA>) {
                        total = static_cast<DT_INPUT_DATA>(static_cast<int64_t>(inputScalar) + static_cast<int64_t>(coef));
                    } else {
                        total = static_cast<DT_INPUT_DATA>(static_cast<float>(inputScalar) + static_cast<float>(coef));
                    }
                    AscendC::Duplicate(yLocal, total, processLength);
                } else {
                    AscendC::Adds(yLocal, yLocal, inputScalar, processLength);
                }
            }

            outQueueY.EnQue<DT_INPUT_DATA>(yLocal);
            CopyOut(globalOffset);
        }
    }

    // Fallback: per-element gather for fp/half when the broadcast pattern is GENERAL.
    __aicore__ inline void ProcessBroadcastVectorized() {
        for (uint32_t offset = 0; offset < coreLength; offset += tileLength) {
            processLength = coreLength - offset < tileLength ? coreLength - offset : tileLength;
            uint32_t globalBase = startOffset + offset;

            AscendC::LocalTensor<DT_INPUT_DATA> inputLocal = inQueueInput.AllocTensor<DT_INPUT_DATA>();
            AscendC::LocalTensor<DT_INPUT_DATA> x1Local = inQueueX1.AllocTensor<DT_INPUT_DATA>();
            AscendC::LocalTensor<DT_INPUT_DATA> x2Local = inQueueX2.AllocTensor<DT_INPUT_DATA>();

            for (uint32_t j = 0; j < processLength; ++j) {
                uint32_t g = globalBase + j;
                uint32_t inputOffset = GetBroadcastOffset(g, tilingData.inputStride);
                uint32_t x1Offset = GetBroadcastOffset(g, tilingData.x1Stride);
                uint32_t x2Offset = GetBroadcastOffset(g, tilingData.x2Stride);
                inputLocal.SetValue(j, inputGm.GetValue(inputOffset));
                x1Local.SetValue(j, x1Gm.GetValue(x1Offset));
                x2Local.SetValue(j, x2Gm.GetValue(x2Offset));
            }

            inQueueInput.EnQue(inputLocal);
            inQueueX1.EnQue(x1Local);
            inQueueX2.EnQue(x2Local);

            ComputeContiguous();
            CopyOut(globalBase);
        }
    }

    // Vectorized int8 contiguous path. Computes y = input + x1*x2*value with exact int32
    // arithmetic (int8-derived products never overflow int32), then applies torch's int8
    // two's-complement wrap via bitwise ops. All per-tile, no per-element scalar loop.
    __aicore__ inline void ProcessContiguousInt8Vec() {
        if constexpr (std::is_same_v<DT_INPUT_DATA, int8_t>) {
            AscendC::LocalTensor<half> hbuf = tbufHalf.Get<half>();
            AscendC::LocalTensor<int32_t> w0 = tbufW0.Get<int32_t>();
            AscendC::LocalTensor<int32_t> w1 = tbufW1.Get<int32_t>();
            int32_t valueI = static_cast<int32_t>(valueScalar);

            for (uint32_t offset = 0; offset < coreLength; offset += tileLength) {
                processLength = coreLength - offset < tileLength ? coreLength - offset : tileLength;
                uint32_t globalOffset = startOffset + offset;

                AscendC::LocalTensor<int8_t> inputLocal = inQueueInput.AllocTensor<int8_t>();
                AscendC::LocalTensor<int8_t> x1Local = inQueueX1.AllocTensor<int8_t>();
                AscendC::LocalTensor<int8_t> x2Local = inQueueX2.AllocTensor<int8_t>();
                AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(processLength * sizeof(int8_t)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<int8_t> pp{false, 0, 0, 0};
                AscendC::DataCopyPad(inputLocal, inputGm[globalOffset], cp, pp);
                AscendC::DataCopyPad(x1Local, x1Gm[globalOffset], cp, pp);
                AscendC::DataCopyPad(x2Local, x2Gm[globalOffset], cp, pp);
                inQueueInput.EnQue(inputLocal);
                inQueueX1.EnQue(x1Local);
                inQueueX2.EnQue(x2Local);

                AscendC::LocalTensor<int8_t> inL = inQueueInput.DeQue<int8_t>();
                AscendC::LocalTensor<int8_t> x1L = inQueueX1.DeQue<int8_t>();
                AscendC::LocalTensor<int8_t> x2L = inQueueX2.DeQue<int8_t>();
                AscendC::LocalTensor<int8_t> yL = outQueueY.AllocTensor<int8_t>();

                // x1 -> int32 (via half), x2 -> int32
                AscendC::Cast(hbuf, x1L, AscendC::RoundMode::CAST_NONE, processLength);
                AscendC::Cast(w0, hbuf, AscendC::RoundMode::CAST_RINT, processLength);
                AscendC::Cast(hbuf, x2L, AscendC::RoundMode::CAST_NONE, processLength);
                AscendC::Cast(w1, hbuf, AscendC::RoundMode::CAST_RINT, processLength);
                // w0 = x1 * x2 * value   (exact in int32: |x1*x2*value| <= 127^3 < 2^31)
                AscendC::Mul(w0, w0, w1, processLength);
                AscendC::Muls(w0, w0, valueI, processLength);
                // w0 += input
                AscendC::Cast(hbuf, inL, AscendC::RoundMode::CAST_NONE, processLength);
                AscendC::Cast(w1, hbuf, AscendC::RoundMode::CAST_RINT, processLength);
                AscendC::Add(w0, w0, w1, processLength);
                // two's-complement wrap to int8 via arithmetic shifts: v = (r << 24) >> 24
                // (sign-extends the low byte; matches torch's int8 wrap exactly).
                AscendC::ShiftLeft(w0, w0, static_cast<int32_t>(24), processLength);
                AscendC::ShiftRight(w0, w0, static_cast<int32_t>(24), processLength);
                // Narrow int32 -> int8. NOTE: int32->half Cast is a dequant op (needs a scale),
                // so go via float: int32 -> float -> half -> int8 (v is small, all exact).
                AscendC::LocalTensor<float> fbuf = tbufW1.Get<float>();
                AscendC::Cast(fbuf, w0, AscendC::RoundMode::CAST_RINT, processLength);
                AscendC::Cast(hbuf, fbuf, AscendC::RoundMode::CAST_RINT, processLength);
                AscendC::Cast(yL, hbuf, AscendC::RoundMode::CAST_RINT, processLength);

                outQueueY.EnQue<int8_t>(yL);
                inQueueInput.FreeTensor(inL);
                inQueueX1.FreeTensor(x1L);
                inQueueX2.FreeTensor(x2L);
                CopyOut(globalOffset);
            }
        }
    }

    // Replicate a pure row-vector operand x[0:K] across `wide` UB elements: rep[i] = x[i % K].
    // Loads K elements once via a VECIN queue, then doubles the run with element-wise copies.
    __aicore__ inline void ReplicateRowVec(AscendC::LocalTensor<DT_INPUT_DATA> rep,
                                           AscendC::GlobalTensor<DT_INPUT_DATA> &gm,
                                           AscendC::TQue<AscendC::QuePosition::VECIN, FDU_BUFFER_NUM> &q,
                                           uint32_t K, uint32_t wide) {
        AscendC::LocalTensor<DT_INPUT_DATA> t = q.AllocTensor<DT_INPUT_DATA>();
        AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(K * sizeof(DT_INPUT_DATA)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT_INPUT_DATA> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(t, gm[0], cp, pp);
        q.EnQue(t);
        AscendC::LocalTensor<DT_INPUT_DATA> td = q.DeQue<DT_INPUT_DATA>();
        AscendC::Adds(rep, td, static_cast<DT_INPUT_DATA>(0), K);  // rep[0:K] = x[0:K]
        q.FreeTensor(td);
        uint32_t filled = K;
        while (filled < wide) {
            uint32_t cp2 = filled < (wide - filled) ? filled : (wide - filled);
            AscendC::Adds(rep[filled], rep, static_cast<DT_INPUT_DATA>(0), cp2);
            filled += cp2;
        }
    }

    // Row-vector fast broadcast path (fp/half/int32). Each broadcast operand is a pure row vector
    // [1,K]/[K]; replicate it once across a wide tile, stream FULL operands, and run FULL-WIDTH vector
    // ops over many rows per tile instead of K-wide per-row chunks. Mirrors ProcessBroadcastChunked's
    // compute combinations (scalars folded into a coef). Cores are K-aligned so rep phase 0 is correct.
    __aicore__ inline void ProcessBroadcastRowVec() {
        if constexpr (!std::is_same_v<DT_INPUT_DATA, int8_t>) {
            uint32_t K = tilingData.rowLen == 0 ? 1 : tilingData.rowLen;
            uint32_t rowsPerTile = tileLength / K;
            if (rowsPerTile == 0) rowsPerTile = 1;
            uint32_t wide = rowsPerTile * K;

            uint32_t inputKind = tilingData.inputKind, x1Kind = tilingData.x1Kind, x2Kind = tilingData.x2Kind;
            bool inputRV = (inputKind == 2), x1RV = (x1Kind == 2), x2RV = (x2Kind == 2);
            bool inputFull = (inputKind == 0), x1Full = (x1Kind == 0), x2Full = (x2Kind == 0);
            bool x1AsScalar = (x1Kind == 1), x2AsScalar = (x2Kind == 1), inputAsScalar = (inputKind == 1);

            // replicate row-vec operands once (only up to what this core uses)
            uint32_t repWide = wide;
            if (repWide > coreLength) {
                uint32_t rows = coreLength / K;
                if (rows == 0) rows = 1;
                repWide = rows * K;
                if (repWide > wide) repWide = wide;
            }
            AscendC::LocalTensor<DT_INPUT_DATA> repInput, repX1, repX2;
            if (inputRV) { repInput = tbufRepInput.Get<DT_INPUT_DATA>(); ReplicateRowVec(repInput, inputGm, inQueueInput, K, repWide); }
            if (x1RV) { repX1 = tbufRepX1.Get<DT_INPUT_DATA>(); ReplicateRowVec(repX1, x1Gm, inQueueX1, K, repWide); }
            if (x2RV) { repX2 = tbufRepX2.Get<DT_INPUT_DATA>(); ReplicateRowVec(repX2, x2Gm, inQueueX2, K, repWide); }

            DT_INPUT_DATA x1Scalar = x1AsScalar ? x1Gm.GetValue(0) : DT_INPUT_DATA(0);
            DT_INPUT_DATA x2Scalar = x2AsScalar ? x2Gm.GetValue(0) : DT_INPUT_DATA(0);
            DT_INPUT_DATA inputScalar = inputAsScalar ? inputGm.GetValue(0) : DT_INPUT_DATA(0);
            DT_INPUT_DATA coef = CombineCoef(x1AsScalar, x1Scalar, x2AsScalar, x2Scalar);

            for (uint32_t offset = 0; offset < coreLength; offset += wide) {
                processLength = (coreLength - offset) < wide ? (coreLength - offset) : wide;
                uint32_t g = startOffset + offset;

                // load FULL operands (contiguous)
                if (!x1AsScalar && x1Full) {
                    AscendC::LocalTensor<DT_INPUT_DATA> x1Local = inQueueX1.AllocTensor<DT_INPUT_DATA>();
                    AscendC::DataCopyExtParams cpp{1, static_cast<uint32_t>(processLength * sizeof(DT_INPUT_DATA)), 0, 0, 0};
                    AscendC::DataCopyPadExtParams<DT_INPUT_DATA> ppp{false, 0, 0, 0};
                    AscendC::DataCopyPad(x1Local, x1Gm[g], cpp, ppp);
                    inQueueX1.EnQue(x1Local);
                }
                if (!x2AsScalar && x2Full) {
                    AscendC::LocalTensor<DT_INPUT_DATA> x2Local = inQueueX2.AllocTensor<DT_INPUT_DATA>();
                    AscendC::DataCopyExtParams cpp{1, static_cast<uint32_t>(processLength * sizeof(DT_INPUT_DATA)), 0, 0, 0};
                    AscendC::DataCopyPadExtParams<DT_INPUT_DATA> ppp{false, 0, 0, 0};
                    AscendC::DataCopyPad(x2Local, x2Gm[g], cpp, ppp);
                    inQueueX2.EnQue(x2Local);
                }
                if (!inputAsScalar && inputFull) {
                    AscendC::LocalTensor<DT_INPUT_DATA> inLocal = inQueueInput.AllocTensor<DT_INPUT_DATA>();
                    AscendC::DataCopyExtParams cpp{1, static_cast<uint32_t>(processLength * sizeof(DT_INPUT_DATA)), 0, 0, 0};
                    AscendC::DataCopyPadExtParams<DT_INPUT_DATA> ppp{false, 0, 0, 0};
                    AscendC::DataCopyPad(inLocal, inputGm[g], cpp, ppp);
                    inQueueInput.EnQue(inLocal);
                }

                AscendC::LocalTensor<DT_INPUT_DATA> yLocal = outQueueY.AllocTensor<DT_INPUT_DATA>();

                // product term x1*x2*value into yLocal
                if (!x1AsScalar && !x2AsScalar) {
                    AscendC::LocalTensor<DT_INPUT_DATA> x1t = x1RV ? repX1 : inQueueX1.DeQue<DT_INPUT_DATA>();
                    AscendC::LocalTensor<DT_INPUT_DATA> x2t = x2RV ? repX2 : inQueueX2.DeQue<DT_INPUT_DATA>();
                    AscendC::Mul(yLocal, x1t, x2t, processLength);
                    AscendC::Muls(yLocal, yLocal, valueScalar, processLength);
                    if (x1Full) inQueueX1.FreeTensor(x1t);
                    if (x2Full) inQueueX2.FreeTensor(x2t);
                } else if (!x1AsScalar && x2AsScalar) {
                    AscendC::LocalTensor<DT_INPUT_DATA> x1t = x1RV ? repX1 : inQueueX1.DeQue<DT_INPUT_DATA>();
                    AscendC::Muls(yLocal, x1t, coef, processLength);
                    if (x1Full) inQueueX1.FreeTensor(x1t);
                } else if (x1AsScalar && !x2AsScalar) {
                    AscendC::LocalTensor<DT_INPUT_DATA> x2t = x2RV ? repX2 : inQueueX2.DeQue<DT_INPUT_DATA>();
                    AscendC::Muls(yLocal, x2t, coef, processLength);
                    if (x2Full) inQueueX2.FreeTensor(x2t);
                }

                // input term
                if (!inputAsScalar) {
                    AscendC::LocalTensor<DT_INPUT_DATA> inT = inputRV ? repInput : inQueueInput.DeQue<DT_INPUT_DATA>();
                    if (x1AsScalar && x2AsScalar) {
                        AscendC::Adds(yLocal, inT, coef, processLength);
                    } else {
                        AscendC::Add(yLocal, yLocal, inT, processLength);
                    }
                    if (inputFull) inQueueInput.FreeTensor(inT);
                } else {
                    if (x1AsScalar && x2AsScalar) {
                        DT_INPUT_DATA total;
                        if constexpr (std::is_integral_v<DT_INPUT_DATA>) {
                            total = static_cast<DT_INPUT_DATA>(static_cast<int64_t>(inputScalar) + static_cast<int64_t>(coef));
                        } else {
                            total = static_cast<DT_INPUT_DATA>(static_cast<float>(inputScalar) + static_cast<float>(coef));
                        }
                        AscendC::Duplicate(yLocal, total, processLength);
                    } else {
                        AscendC::Adds(yLocal, yLocal, inputScalar, processLength);
                    }
                }

                outQueueY.EnQue<DT_INPUT_DATA>(yLocal);
                CopyOut(g);
            }
        }
    }

    // Load `len` int8 values from gm[srcOff] and widen to int32 in `dst` (via half, both exact
    // for int8 magnitudes). Uses one of the VECIN queues for the staging int8 tile.
    __aicore__ inline void LoadInt8ToInt32(AscendC::LocalTensor<int32_t> dst,
                                           AscendC::GlobalTensor<int8_t> &gm, uint32_t srcOff,
                                           uint32_t len, AscendC::TQue<AscendC::QuePosition::VECIN, FDU_BUFFER_NUM> &q,
                                           AscendC::LocalTensor<half> hbuf) {
        if constexpr (std::is_same_v<DT_INPUT_DATA, int8_t>) {
            AscendC::LocalTensor<int8_t> l = q.AllocTensor<int8_t>();
            AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(len * sizeof(int8_t)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<int8_t> pp{false, 0, 0, 0};
            AscendC::DataCopyPad(l, gm[srcOff], cp, pp);
            q.EnQue(l);
            AscendC::LocalTensor<int8_t> ld = q.DeQue<int8_t>();
            AscendC::Cast(hbuf, ld, AscendC::RoundMode::CAST_NONE, len);
            AscendC::Cast(dst, hbuf, AscendC::RoundMode::CAST_RINT, len);
            q.FreeTensor(ld);
        }
    }

    // Vectorized int8 structured-broadcast path. Mirrors ProcessBroadcastChunked but computes in
    // int32 (exact: |x1*x2*value| <= 127^3 < 2^31) and applies torch's int8 wrap via bit-shifts.
    // Every operand is materialized into a full int32 tile (bulk DataCopy+widen for FULL/INNER_CONTIG,
    // Duplicate for SCALAR/INNER_BCAST), then a uniform Mul/Muls/Add/wrap/narrow. Row-clamped so a
    // chunk never crosses a broadcast-run boundary (same correctness invariant as the fp path).
    __aicore__ inline void ProcessBroadcastChunkedInt8(uint32_t chunkLen) {
        if constexpr (std::is_same_v<DT_INPUT_DATA, int8_t>) {
            AscendC::LocalTensor<half> hbuf = tbufHalf.Get<half>();
            AscendC::LocalTensor<int32_t> w = tbufW0.Get<int32_t>();
            AscendC::LocalTensor<int32_t> tmp = tbufW1.Get<int32_t>();
            int32_t valueI = static_cast<int32_t>(valueScalar);

            uint32_t inputKind = tilingData.inputKind;
            uint32_t x1Kind = tilingData.x1Kind;
            uint32_t x2Kind = tilingData.x2Kind;
            bool inputAsScalar = (inputKind == 1) || (inputKind == 3);
            bool x1AsScalar = (x1Kind == 1) || (x1Kind == 3);
            bool x2AsScalar = (x2Kind == 1) || (x2Kind == 3);
            int8_t inputScalarBase = (inputKind == 1) ? inputGm.GetValue(0) : int8_t(0);
            int8_t x1ScalarBase = (x1Kind == 1) ? x1Gm.GetValue(0) : int8_t(0);
            int8_t x2ScalarBase = (x2Kind == 1) ? x2Gm.GetValue(0) : int8_t(0);

            uint32_t bcastRun = ComputeBcastRun();
            bool needClamp = (chunkLen < bcastRun) || (tilingData.coreStride == 0);
            for (uint32_t offset = 0; offset < coreLength; offset += processLength) {
                uint32_t globalOffset = startOffset + offset;
                processLength = chunkLen;
                if (processLength > coreLength - offset) processLength = coreLength - offset;
                if (needClamp) {
                    uint32_t toRowEnd = bcastRun - (globalOffset % bcastRun);
                    if (processLength > toRowEnd) processLength = toRowEnd;
                }

                int8_t x1S = x1ScalarBase, x2S = x2ScalarBase, inS = inputScalarBase;
                if (x1Kind == 3) x1S = x1Gm.GetValue(GetBroadcastOffset(globalOffset, tilingData.x1Stride));
                if (x2Kind == 3) x2S = x2Gm.GetValue(GetBroadcastOffset(globalOffset, tilingData.x2Stride));
                if (inputKind == 3) inS = inputGm.GetValue(GetBroadcastOffset(globalOffset, tilingData.inputStride));

                // w = x1 (int32)
                if (!x1AsScalar) {
                    uint32_t srcOff = (x1Kind == 0) ? globalOffset : GetBroadcastOffset(globalOffset, tilingData.x1Stride);
                    LoadInt8ToInt32(w, x1Gm, srcOff, processLength, inQueueX1, hbuf);
                } else {
                    AscendC::Duplicate(w, static_cast<int32_t>(x1S), processLength);
                }
                // tmp = x2 (int32)
                if (!x2AsScalar) {
                    uint32_t srcOff = (x2Kind == 0) ? globalOffset : GetBroadcastOffset(globalOffset, tilingData.x2Stride);
                    LoadInt8ToInt32(tmp, x2Gm, srcOff, processLength, inQueueX2, hbuf);
                } else {
                    AscendC::Duplicate(tmp, static_cast<int32_t>(x2S), processLength);
                }
                // w = x1 * x2 * value
                AscendC::Mul(w, w, tmp, processLength);
                AscendC::Muls(w, w, valueI, processLength);
                // tmp = input (int32); w += input
                if (!inputAsScalar) {
                    uint32_t srcOff = (inputKind == 0) ? globalOffset : GetBroadcastOffset(globalOffset, tilingData.inputStride);
                    LoadInt8ToInt32(tmp, inputGm, srcOff, processLength, inQueueInput, hbuf);
                } else {
                    AscendC::Duplicate(tmp, static_cast<int32_t>(inS), processLength);
                }
                AscendC::Add(w, w, tmp, processLength);
                // two's-complement wrap to int8: v = (r << 24) >> 24
                AscendC::ShiftLeft(w, w, static_cast<int32_t>(24), processLength);
                AscendC::ShiftRight(w, w, static_cast<int32_t>(24), processLength);
                // narrow int32 -> int8 via float (int32->half is a dequant op): int32->float->half->int8
                AscendC::LocalTensor<int8_t> yL = outQueueY.AllocTensor<int8_t>();
                AscendC::LocalTensor<float> fbuf = tbufW1.Get<float>();
                AscendC::Cast(fbuf, w, AscendC::RoundMode::CAST_RINT, processLength);
                AscendC::Cast(hbuf, fbuf, AscendC::RoundMode::CAST_RINT, processLength);
                AscendC::Cast(yL, hbuf, AscendC::RoundMode::CAST_RINT, processLength);
                outQueueY.EnQue<int8_t>(yL);
                CopyOut(globalOffset);
            }
        }
    }

    __aicore__ inline void ProcessBroadcastInteger() {
        int64_t valueScalar64 = static_cast<int64_t>(valueScalar);
        for (uint32_t offset = 0; offset < coreLength; offset += tileLength) {
            processLength = coreLength - offset < tileLength ? coreLength - offset : tileLength;
            uint32_t globalBase = startOffset + offset;

            AscendC::LocalTensor<DT_INPUT_DATA> yLocal = outQueueY.AllocTensor<DT_INPUT_DATA>();
            for (uint32_t j = 0; j < processLength; ++j) {
                uint32_t g = globalBase + j;
                uint32_t inputOffset = GetBroadcastOffset(g, tilingData.inputStride);
                uint32_t x1Offset = GetBroadcastOffset(g, tilingData.x1Stride);
                uint32_t x2Offset = GetBroadcastOffset(g, tilingData.x2Stride);
                int64_t inputValue = static_cast<int64_t>(inputGm.GetValue(inputOffset));
                int64_t x1Value = static_cast<int64_t>(x1Gm.GetValue(x1Offset));
                int64_t x2Value = static_cast<int64_t>(x2Gm.GetValue(x2Offset));
                yLocal.SetValue(j, static_cast<DT_INPUT_DATA>(inputValue + x1Value * x2Value * valueScalar64));
            }
            outQueueY.EnQue<DT_INPUT_DATA>(yLocal);
            CopyOut(globalBase);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, FDU_BUFFER_NUM> inQueueInput;
    AscendC::TQue<AscendC::QuePosition::VECIN, FDU_BUFFER_NUM> inQueueX1;
    AscendC::TQue<AscendC::QuePosition::VECIN, FDU_BUFFER_NUM> inQueueX2;
    AscendC::TQue<AscendC::QuePosition::VECOUT, FDU_BUFFER_NUM> outQueueY;
    // Scratch for the int8 contiguous vector path (only InitBuffer'd for int8 + contiguous).
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tbufHalf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tbufW0;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tbufW1;
    // Persistent replicated row-vector buffers for the fp/int32 fast broadcast path.
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tbufRepInput;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tbufRepX1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tbufRepX2;
    AscendC::GlobalTensor<DT_INPUT_DATA> inputGm;
    AscendC::GlobalTensor<DT_INPUT_DATA> x1Gm;
    AscendC::GlobalTensor<DT_INPUT_DATA> x2Gm;
    AscendC::GlobalTensor<DT_INPUT_DATA> valueGm;
    AscendC::GlobalTensor<DT_INPUT_DATA> yGm;
    AddcmulTilingData tilingData;
    uint32_t blockDim;
    uint32_t tileLength;
    uint32_t startOffset;
    uint32_t coreLength;
    uint32_t processLength;
    DT_INPUT_DATA valueScalar;
};

template <typename DT_INPUT_DATA>
__global__ __aicore__ void addcmul(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(AddcmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddcmulTilingData, tiling_data, tiling);
    KernelAddcmul<DT_INPUT_DATA> op;
    op.Init(input_data, x1, x2, value, y, tiling_data);
    op.Process();
}
