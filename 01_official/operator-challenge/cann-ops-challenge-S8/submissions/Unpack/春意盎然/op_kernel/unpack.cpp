#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include <type_traits>

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t MAX_BLOCK_COUNT = 4095;

// Path-A gather (sizeof(T) >= 2): DOUBLE-BUFFERED input for chunk pipelining.
// UB budget check (worst case fp32, outer ≥ PATHA_MAX_ROWS):
//   input   = 2 * PATHA_INPUT_BYTES                              = 128 KB (2*64KB)
//   offset  = PATHA_MAX_ROWS * sizeof(uint32_t)                  =  16 KB (4096*4)
//   packed  = BUFFER_NUM * PATHA_MAX_ROWS * sizeof(fp32)         =  32 KB (2*4096*4)
//   TOTAL                                                          = 176 KB  ✓ < 192 KB UB
// Double buffer = chunk N+1 MTE2 overlaps with chunk N V/MTE3 for multi-chunk cases.
constexpr uint32_t PATHA_INPUT_BYTES = 65536;           //  64 KB per buffer (× 2 BUFFER_NUM)
constexpr uint32_t PATHA_MAX_ROWS    = 4096;

// Path-A ROW-MAJOR (v9.32): single (NOT double) input buffer, made large so each chunk
// holds more rows → bigger MTE3 writes (rows*sizeof(T)). Row-major's MTE2 is only ~5-15%
// busy (per profiling), so dropping input double-buffering costs little, while the MTE3
// pipe — the row-major WALL at 79-97% for mid axis — benefits from fewer/larger writes.
// packedQueue stays double-buffered so gather(n+1) overlaps the dominant MTE3(n).
//
// Input size is dtype-aware: total UB budget = 176KB (16KB margin under the 192KB UB,
// matching the proven-safe online footprint). Layout = input + offset(4096*4=16KB) +
// packed(2*4096*sizeof(T)). So input = 176KB - 16KB - 2*4096*sizeof(T):
//   fp16/bf16/int16 (2B): 144KB ;  fp32/int32 (4B): 128KB.
constexpr uint32_t PATHA_RM_TOTAL_BYTES = 180224;       // 176 KB total budget

// Path-A cast-gather (sizeof(T) == 1): smaller int8 input + 2x fp16 work buffer.
constexpr uint32_t PATHA_CG_IN_BYTES  = 32768;          // 32 KB int8 input
constexpr uint32_t PATHA_CG_FP16_BYTES = 65536;         // 64 KB fp16 work (2x the int8 chunk)
constexpr uint32_t PATHA_CG_MAX_ROWS  = 4096;           // ceiling

// ============================================================
// Path-A TRANSPOSE variant (v9.40): replace the per-axis strided vector Gather with a
// repeat-batched hardware vnchwconv (TransDataTo5HD). Maps the deinterleave directly:
//   input [rows, axis] --transpose--> [axis, rows]  (output[a] = row a = contiguous write).
// Fixes BOTH Path-A walls: slow per-element Gather (vec-bound) AND scattered small writes.
//
// v9.40 ARBITRARY AXIS: rows must be 16-aligned (vnchwconv tile) AND each row's UB start
// must be 32B-aligned. For axis NOT a multiple of 16 the GM rows aren't 32B-strided, so we
// right-pad each row up to axisPad = roundup(axis,16) on load via DataCopyPad rightPadding
// (<=15 elems = <=30B <= the 32B padding limit). Then transpose [rows, axisPad] -> [axisPad,
// rows] and write only the first `axis` real outputs. Covers ALL 2-byte axis in [2,128].
constexpr uint32_t TRANSPOSE_MAX_AXIS  = 128;          // = row-major axis cap (PTR_CACHE)
constexpr uint32_t TRANSPOSE_UB_BUDGET = 180224;       // 176KB for input+dst (proven-safe footprint)
constexpr uint32_t TRANSPOSE_MAX_W     = 4095;          // wSize (=rows/chunk) hard cap

__aicore__ inline uint32_t AlignUpU32(uint32_t value, uint32_t align)
{
    return (value + align - 1U) / align * align;
}

__aicore__ inline uint32_t GcdU32(uint32_t a, uint32_t b)
{
    while (b != 0U) { uint32_t t = a % b; a = b; b = t; }
    return a;
}

// Compute the chunk-row alignment such that chunk_rows * axisBytes is multiple of 32B.
// Required because GM MTE2 reads at chunk boundaries (rowStart * axisBytes) MUST land on
// 32-byte aligned addresses — otherwise the vector engine raises a hardware exception.
__aicore__ inline uint32_t ChunkRowAlign(uint32_t axisBytes)
{
    return 32U / GcdU32(32U, axisBytes);
}

template <typename T> class KernelUnpack {
public:
    __aicore__ inline KernelUnpack() {}

    // v9.32: dtype-aware single input-buffer size for row-major Path A.
    // = 176KB total - 16KB offset - 2*PATHA_MAX_ROWS*sizeof(T) packed.
    static constexpr uint32_t RmInputBytes()
    {
        return PATHA_RM_TOTAL_BYTES - PATHA_MAX_ROWS * sizeof(uint32_t)
               - 2u * PATHA_MAX_ROWS * sizeof(T);
    }

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR outputList, uint32_t totalLength, uint32_t outerLength,
                                uint32_t axisLength, uint32_t innerLength, uint32_t blockLength, uint32_t tileLength,
                                AscendC::TPipe *pipeIn)
    {
        this->pipe = pipeIn;
        this->totalLength = totalLength;
        this->outerLength = outerLength;
        this->axisLength = axisLength;
        this->innerLength = innerLength;
        this->outputLength = outerLength * innerLength;
        this->tileLength = tileLength;
        this->startOffset = AscendC::GetBlockIdx() * blockLength;

        const uint32_t innerBytes = innerLength * sizeof(T);
        const bool pathA = (innerLength == 1);
        const bool pathB = (!pathA) && (innerBytes <= tileLength);

        uint32_t spaceSize;
        if (pathA) {
            // Path A hybrid (v9.26): row-major if (outer ≥ 4096 && axis ≤ 128 && !bool),
            // else axis-major. Must match Process() and host tiling rule.
            const bool useRowMajor = !std::is_same_v<T, bool> &&
                                     (outerLength >= 4096u) && (axisLength <= 128u);
            spaceSize = useRowMajor ? outerLength : axisLength;
        } else if (pathB) {
            spaceSize = outerLength * axisLength;
        } else {
            spaceSize = totalLength;
        }
        uint32_t remain = spaceSize > this->startOffset ? spaceSize - this->startOffset : 0;
        this->processLength = remain < blockLength ? remain : blockLength;

        inputGm.SetGlobalBuffer((__gm__ T *)input, totalLength);
        outputs.Init((__gm__ void *)outputList);
        this->pathAMode = 0;  // 0=scalar, 1=gather, 2=cast-gather

        if (pathA) {
            const uint32_t axisBytesA = axisLength * sizeof(T);
            const bool gatherOk = (sizeof(T) >= 2) && (axisBytesA <= PATHA_INPUT_BYTES);
            // cast-gather: int8/uint8 (NOT bool — bool has no Cast↔half intrinsic) with
            // axisBytes (as fp16) fits in fp16 work buffer
            constexpr bool isCastable = (sizeof(T) == 1) && !std::is_same_v<T, bool>;
            const bool castGatherOk = isCastable && (axisLength * sizeof(half) <= PATHA_CG_FP16_BYTES);
            // v9.39b: row-major + small axis → repeat-batched TransDataTo5HD (vnchwconv),
            // replacing the strided Gather. MUST match Process()/host useRowMajor rule.
            const bool useRowMajorA = !std::is_same_v<T, bool> &&
                                      (outerLength >= 4096u) && (axisLength <= 128u);
            // v9.40: TransDataTo5HD for ALL 2-byte types, ANY axis in [2,128]. Needs NO tmp
            // buffer (src/dst datablock-address lists are on the stack); repeatTimes batches all
            // rows/16 tiles in ONE call (scalar setup once, hardware loops) — unlike the
            // high-level Transpose which software-looped (sca=99%). Non-mult-16 axis is right-
            // padded on load to axisPad=roundup(axis,16); buffers are sized for axisPad.
            const uint32_t axisPadA = AlignUpU32(axisLength, 16u);
            const uint32_t axisPadBytesA = axisPadA * sizeof(T);
            // v9.41: TransDataTo5HD transpose for BOTH 2-byte (axis%16==0, 16-elem datablocks)
            // AND 4-byte fp32/int32 (axis%8==0, 8-elem datablocks). Aligned axis only — non-
            // mult axis needs per-row pad load → MTE2-bound (1M*63=776us, WORSE), keep on gather.
            // axis<=128 (row-major cap). Buffers sized by axisBytes (axis is aligned, no padding).
            (void)axisPadA; (void)axisPadBytesA;
            const bool transOk2 = (sizeof(T) == 2u) && (axisLength % 16u == 0u);
            const bool transOk4 = (sizeof(T) == 4u) && (axisLength % 8u == 0u);
            const bool useTranspose = gatherOk && useRowMajorA && (transOk2 || transOk4) &&
                                      (axisLength <= TRANSPOSE_MAX_AXIS);
            if (useTranspose) {
                this->pathAMode = 3;
                // ADAPTIVE buffering. BN2 (double-buffer input+dst) overlaps chunk n+1's MTE2
                // load with chunk n's vnchwconv+MTE3. WINS only for tiny 2-byte axis (16) with
                // many chunks (1M*16 34.2->24.9us -27%); axis>=32 / few chunks / 4-byte REGRESS
                // → keep BN1 (bigger rows). rows multiple of 16 (vnchwconv tile); axisBytes a
                // multiple of 32 (axis%16==0 2B, axis%8==0 4B) keeps chunk-starts 32B-aligned.
                const uint32_t bn1Rows = ((TRANSPOSE_UB_BUDGET / (2u * axisBytesA)) / 16u) * 16u;
                const bool manyChunks = (bn1Rows != 0u) && (blockLength >= 3u * bn1Rows);
                const uint32_t bn = (transOk2 && (axisLength <= 16u) && manyChunks) ? BUFFER_NUM : 1u;
                uint32_t maxRowsT = TRANSPOSE_UB_BUDGET / (2u * bn * axisBytesA);
                maxRowsT = (maxRowsT / 16u) * 16u;
                if (maxRowsT == 0u) maxRowsT = 16u;
                this->transMaxRows = maxRowsT;
                pipe->InitBuffer(inputQueue, bn, maxRowsT * axisBytesA);    // input [rows, axis]
                pipe->InitBuffer(transDstQueue, bn, maxRowsT * axisBytesA); // dst   [axis, rows]
            } else if (gatherOk) {
                this->pathAMode = 1;
                // v9.32: row-major uses a SINGLE large input buffer (bigger writes);
                // axis-major keeps the double-buffered 64KB (it re-reads, so MTE2 overlap
                // matters). Selection must match Process()/host useRowMajor rule.
                const bool useRowMajor = useRowMajorA;
                const uint32_t inputBytes = useRowMajor ? RmInputBytes() : PATHA_INPUT_BYTES;
                if (useRowMajor) {
                    pipe->InitBuffer(inputQueue, 1, RmInputBytes());        // single buffer
                } else {
                    pipe->InitBuffer(inputQueue, BUFFER_NUM, PATHA_INPUT_BYTES);
                }
                pipe->InitBuffer(offsetBuf, PATHA_MAX_ROWS * sizeof(uint32_t));
                const uint32_t packedRows = outerLength < PATHA_MAX_ROWS ? outerLength : PATHA_MAX_ROWS;
                pipe->InitBuffer(packedQueue, BUFFER_NUM, AlignUpU32(packedRows * sizeof(T), BLOCK_BYTES));

                AscendC::LocalTensor<uint32_t> offsetTensor = offsetBuf.Get<uint32_t>();
                uint32_t maxRows = inputBytes / axisBytesA;
                if (maxRows > PATHA_MAX_ROWS) maxRows = PATHA_MAX_ROWS;
                // Multi-chunk source MTE2 offsets must be 32B-aligned — round chunk-row
                // cap down to a multiple of the alignment, so chunk2/3/... start on aligned
                // GM bytes. (Single-chunk cases are unaffected: first chunk always starts
                // at byte 0.)
                const uint32_t rowAlignA = ChunkRowAlign(axisBytesA);
                if (maxRows >= rowAlignA) {
                    maxRows = (maxRows / rowAlignA) * rowAlignA;
                }
                const uint32_t rowsNeeded = outerLength < maxRows ? outerLength : maxRows;
                BuildOffsetTable(offsetTensor, rowsNeeded, axisBytesA);
            } else if (castGatherOk) {
                this->pathAMode = 2;
                pipe->InitBuffer(inputQueue, 1, PATHA_CG_IN_BYTES);              // int8 input
                pipe->InitBuffer(workFp16Buf, PATHA_CG_FP16_BYTES);              // fp16 work
                pipe->InitBuffer(offsetBuf, PATHA_CG_MAX_ROWS * sizeof(uint32_t));
                const uint32_t packedRows = outerLength < PATHA_CG_MAX_ROWS ? outerLength : PATHA_CG_MAX_ROWS;
                // Separate fp16 and int8 packed queues — safer than ReinterpretCast aliasing
                // which appeared to introduce sporadic failures online (Case4).
                pipe->InitBuffer(packedQueue, BUFFER_NUM, AlignUpU32(packedRows * sizeof(half), BLOCK_BYTES));
                pipe->InitBuffer(int8PackedQueue, BUFFER_NUM, AlignUpU32(packedRows * sizeof(T), BLOCK_BYTES));

                // Offsets index into fp16 work buffer (byte offsets, stride = axis * sizeof(half))
                AscendC::LocalTensor<uint32_t> offsetTensor = offsetBuf.Get<uint32_t>();
                const uint32_t axisBytesFp16 = axisLength * sizeof(half);
                const uint32_t axisBytesInt = axisLength * sizeof(T);  // int8 input stride
                uint32_t maxRowsCG = PATHA_CG_FP16_BYTES / axisBytesFp16;
                if (maxRowsCG > PATHA_CG_MAX_ROWS) maxRowsCG = PATHA_CG_MAX_ROWS;
                // Align so BOTH the int8 input MTE2 and the fp16 work region access
                // 32B-aligned GM/UB offsets at chunk boundaries.
                const uint32_t rowAlignInt = ChunkRowAlign(axisBytesInt);
                const uint32_t rowAlignFp16 = ChunkRowAlign(axisBytesFp16);
                uint32_t rowAlignCG = rowAlignInt > rowAlignFp16 ? rowAlignInt : rowAlignFp16;
                if (maxRowsCG >= rowAlignCG) {
                    maxRowsCG = (maxRowsCG / rowAlignCG) * rowAlignCG;
                }
                const uint32_t rowsNeededCG = outerLength < maxRowsCG ? outerLength : maxRowsCG;
                BuildOffsetTable(offsetTensor, rowsNeededCG, axisBytesFp16);
            } else {
                this->pathAMode = 0;
                // Scalar fallback uses BUFFER_NUM=2 queues. Cap each at 32 KB so the
                // queue pair stays under 64 KB even when the host's tileLength is
                // tuned up to 96 KB for Path B/C benefit.
                uint32_t paFallbackBytes = tileLength < 32768u ? tileLength : 32768u;
                pipe->InitBuffer(gatherQueue, BUFFER_NUM, paFallbackBytes);
                uint32_t packedBytes = AlignUpU32(paFallbackBytes / BLOCK_BYTES * sizeof(T), BLOCK_BYTES);
                if (packedBytes < BLOCK_BYTES) packedBytes = BLOCK_BYTES;
                pipe->InitBuffer(copyQueue, BUFFER_NUM, packedBytes);
            }
        } else {
            pipe->InitBuffer(copyQueue, BUFFER_NUM, tileLength);
        }
    }

    // Hybrid Path A selection (v9.26). MUST match the host's selection rule in
    // unpack.cpp tiling. For non-bool types: use row-major when the dataset is large
    // enough to make MTE2 redundancy matter (outer ≥ 4096) AND axis is small enough
    // that the per-chunk "all axes" V/MTE3 loop stays cheap (axis ≤ 128).
    __aicore__ inline bool UseRowMajorPathA() const
    {
        if constexpr (std::is_same_v<T, bool>) return false;
        return (this->outerLength >= 4096u) && (this->axisLength <= 128u);
    }

    __aicore__ inline void Process()
    {
        if (this->innerLength == 1) {
            if (UseRowMajorPathA()) {
                // Row-major: each core covers a slice of OUTER rows × ALL axes.
                switch (this->pathAMode) {
                    case 3: ProcessInnerOneTransposeRowMajor(); break;
                    case 1: ProcessInnerOneGatherRowMajor(); break;
                    case 2: ProcessInnerOneCastGatherRowMajor(); break;
                    default: ProcessInnerOneScalar(); break;
                }
            } else {
                // Axis-major: each core covers a slice of axes × ALL rows.
                switch (this->pathAMode) {
                    case 1: ProcessInnerOneGather(); break;
                    case 2: ProcessInnerOneCastGather(); break;
                    default: ProcessInnerOneScalar(); break;
                }
            }
            return;
        }
        const uint32_t innerBytes = this->innerLength * sizeof(T);
        if (innerBytes <= this->tileLength) {
            ProcessInnerSmall(innerBytes);
        } else {
            ProcessInnerHuge(innerBytes);
        }
    }

private:
    // ============================================================
    // Build offset table [0, stride, 2*stride, ..., (count-1)*stride] in offsetTensor.
    //
    // Strategy: seed first SEED_N entries via scalar SetValue, then DOUBLE via vector
    // Adds — total scalar cost = O(SEED_N), vector cost = O(log N).
    // The earlier "vector core exception" misattributed to this routine was actually a UB
    // overflow elsewhere (PATHA_MAX_ROWS=8192 made packed queue too large). With the UB
    // fix in place this doubling routine is safe and a real ~5us win on big-outer cases.
    // ============================================================
    __aicore__ inline void BuildOffsetTable(AscendC::LocalTensor<uint32_t>& offsetTensor,
                                            uint32_t count, uint32_t stride)
    {
        constexpr uint32_t SEED_N = 16;
        const uint32_t seedCount = count < SEED_N ? count : SEED_N;
        uint32_t off = 0;
        for (uint32_t i = 0; i < seedCount; ++i) {
            offsetTensor.SetValue(i, off);
            off += stride;
        }
        if (count <= SEED_N) {
            return;
        }
        // S → V sync: subsequent Adds reads the SetValue results.
        int32_t eventId = GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventId);

        AscendC::LocalTensor<int32_t> offsetAsInt = offsetTensor.ReinterpretCast<int32_t>();
        uint32_t n = SEED_N;
        while (n < count) {
            uint32_t chunk = (count - n) < n ? (count - n) : n;
            AscendC::Adds(offsetAsInt[n], offsetAsInt, static_cast<int32_t>(n * stride), chunk);
            AscendC::PipeBarrier<PIPE_V>();
            n *= 2;
        }
    }

    // ============================================================
    // Path A — TRANSPOSE variant, ROW-MAJOR distribution (v9.40, ARBITRARY 2-byte axis)
    //
    // Per chunk of `rows` outer-rows: load [rows, axis] (one MTE2; non-mult-16 axis is
    // right-padded to axisPad on load), repeat-batched vnchwconv transpose [rows, axisPad] ->
    // [axisPad, rows] in UB, then write each REAL output[a] = dst[a*rowsW .. +rows] (one
    // contiguous MTE3). Replaces the per-element strided Gather (vec-bound) + scattered writes.
    //
    // vnchwconv (2-byte): 1 datablock = 16 elems. Per 16-column tile b, set 16 src/dst
    // datablock addresses ONCE, repeatTimes=(sub/16) tiles loop in hardware (scalar setup
    // once). srcList[i] = padded input row (s0+i), cols b*16..  ; srcRepStride = axisPad
    // datablocks. dstList[i] = output col (b*16+i), pos s0      ; dstRepStride = 1 datablock.
    // 2B (fp16/bf16/int16) reinterpret to uint16 — transpose is pure bit-movement.
    // ============================================================
    __aicore__ inline void ProcessInnerOneTransposeRowMajor()
    {
        if constexpr (sizeof(T) == 2 && !std::is_same_v<T, bool>) {
            if (this->processLength == 0) return;
            const uint32_t outer = this->outerLength;
            const uint32_t axis  = this->axisLength;            // [2, 128]
            const uint32_t axisPad = AlignUpU32(axis, 16u);     // 16-aligned column count
            const uint32_t startRow = this->startOffset;
            const uint32_t rowEnd   = startRow + this->processLength;
            const uint32_t axisBytes = axis * sizeof(T);
            const bool needPad = (axis != axisPad);
            const uint32_t rowsStep  = this->transMaxRows;      // chunk rows (multiple of 16)
            const uint32_t colBlocks = axisPad / 16u;           // # of 16-column tiles

            // Cache output GM pointers (axis <= TRANSPOSE_MAX_AXIS = 128 = PTR_CACHE).
            constexpr uint32_t PTR_CACHE = 128;
            __gm__ T* outPtrs[PTR_CACHE];
            for (uint32_t j = 0; j < axis; ++j) outPtrs[j] = outputs.GetDataPtr<T>(j);

            AscendC::GlobalTensor<T> outGm;
            // Right-pad each row to axisPad on load (pad bytes = (axisPad-axis)*2 <= 30 <= 32B).
            AscendC::DataCopyPadExtParams<T> padPad{true, 0, static_cast<uint8_t>(axisPad - axis),
                                                    static_cast<T>(0)};
            AscendC::DataCopyPadExtParams<T> padNone{false, 0, 0, static_cast<T>(0)};

            uint32_t row = startRow;
            while (row < rowEnd) {
                uint32_t rows = rowEnd - row;
                if (rows > rowsStep) rows = rowsStep;
                // rowsW = rows padded to the 16-row vnchwconv tile. Full chunks are already
                // multiples of 16; only the last partial chunk pads up. Padding rows are read
                // as garbage by the transpose but never written out.
                const uint32_t rowsW = AlignUpU32(rows, 16u);

                // --- MTE2: load into [rows, axisPad] UB ---
                AscendC::LocalTensor<T> inputLocal = inputQueue.AllocTensor<T>();
                if (needPad) {
                    // rows blocks of axisBytes each, right-padded to axisPad (32B-aligned rows).
                    AscendC::DataCopyExtParams loadParams{
                        static_cast<uint16_t>(rows), axisBytes, 0, 0, 0};
                    AscendC::DataCopyPad(inputLocal, inputGm[static_cast<uint64_t>(row) * axis],
                                         loadParams, padPad);
                } else {
                    // axis already 16-aligned: one contiguous DMA (axisPad == axis).
                    AscendC::DataCopyExtParams loadParams{
                        1, static_cast<uint32_t>(rows * axisBytes), 0, 0, 0};
                    AscendC::DataCopyPad(inputLocal, inputGm[static_cast<uint64_t>(row) * axis],
                                         loadParams, padNone);
                }
                inputQueue.EnQue(inputLocal);
                inputLocal = inputQueue.DeQue<T>();

                // --- transpose [rowsW, axisPad] -> [axisPad, rowsW] via repeat-batched vnchwconv ---
                AscendC::LocalTensor<T> dstLocal = transDstQueue.AllocTensor<T>();
                AscendC::LocalTensor<uint16_t> srcU = inputLocal.template ReinterpretCast<uint16_t>();
                AscendC::LocalTensor<uint16_t> dstU = dstLocal.template ReinterpretCast<uint16_t>();
                for (uint32_t b = 0; b < colBlocks; ++b) {
                    uint32_t s0 = 0;
                    while (s0 < rowsW) {
                        uint32_t sub = rowsW - s0;
                        if (sub > 4080u) sub = 4080u;          // repeatTimes <= 255
                        AscendC::TransDataTo5HDParams tp;
                        tp.dstHighHalf = false;
                        tp.srcHighHalf = false;
                        tp.repeatTimes  = static_cast<uint8_t>(sub / 16u);
                        tp.srcRepStride = static_cast<uint16_t>(axisPad); // 16 rows = axisPad datablocks
                        tp.dstRepStride = 1;                              // +16 elems = 1 datablock
                        uint64_t srcList[16];
                        uint64_t dstList[16];
                        for (uint32_t i = 0; i < 16u; ++i) {
                            srcList[i] = (uint64_t)(srcU[(s0 + i) * axisPad + b * 16u].GetPhyAddr());
                            dstList[i] = (uint64_t)(dstU[(b * 16u + i) * rowsW + s0].GetPhyAddr());
                        }
                        AscendC::TransDataTo5HD<uint16_t>(dstList, srcList, tp);
                        s0 += sub;
                    }
                }
                transDstQueue.EnQue(dstLocal);
                dstLocal = transDstQueue.DeQue<T>();

                // --- MTE3: write each REAL output[a] = dst[a*rowsW .. +rows] (contiguous) ---
                AscendC::DataCopyExtParams writeParams{
                    1, static_cast<uint32_t>(rows * sizeof(T)), 0, 0, 0};
                uint32_t srcOff = 0;
                for (uint32_t a = 0; a < axis; ++a) {
                    outGm.SetGlobalBuffer(outPtrs[a], outer);
                    AscendC::DataCopyPad(outGm[row], dstLocal[srcOff], writeParams);
                    srcOff += rowsW;
                }

                transDstQueue.FreeTensor(dstLocal);
                inputQueue.FreeTensor(inputLocal);
                row += rows;
            }
        } else if constexpr (sizeof(T) == 4) {
            // 4-byte (fp32/int32) — 32-bit vnchwconv: 1 datablock = 8 elems. The instr takes 16
            // input datablocks (16 rows, cols b*8..b*8+7) and emits 16 output datablocks where
            // db[2p] = col p of rows 0-7, db[2p+1] = col p of rows 8-15. So output col (b*8+p)
            // for a 16-row group = db[2p](rows0-7) ++ db[2p+1](rows8-15), contiguous. axis%8==0.
            if (this->processLength == 0) return;
            const uint32_t outer = this->outerLength;
            const uint32_t axis  = this->axisLength;            // mult of 8, [8,128]
            const uint32_t startRow = this->startOffset;
            const uint32_t rowEnd   = startRow + this->processLength;
            const uint32_t axisBytes = axis * sizeof(T);
            const uint32_t rowsStep  = this->transMaxRows;      // chunk rows (multiple of 16)
            const uint32_t colBlocks = axis / 8u;               // # of 8-column tiles

            constexpr uint32_t PTR_CACHE = 128;
            __gm__ T* outPtrs[PTR_CACHE];
            for (uint32_t j = 0; j < axis; ++j) outPtrs[j] = outputs.GetDataPtr<T>(j);

            AscendC::GlobalTensor<T> outGm;
            AscendC::DataCopyPadExtParams<T> padNone{false, 0, 0, static_cast<T>(0)};

            uint32_t row = startRow;
            while (row < rowEnd) {
                uint32_t rows = rowEnd - row;
                if (rows > rowsStep) rows = rowsStep;
                const uint32_t rowsW = AlignUpU32(rows, 16u);

                // --- MTE2: load [rows, axis] contiguous (axis%8==0 → axisBytes mult of 32) ---
                AscendC::LocalTensor<T> inputLocal = inputQueue.AllocTensor<T>();
                AscendC::DataCopyExtParams loadParams{
                    1, static_cast<uint32_t>(rows * axisBytes), 0, 0, 0};
                AscendC::DataCopyPad(inputLocal, inputGm[static_cast<uint64_t>(row) * axis],
                                     loadParams, padNone);
                inputQueue.EnQue(inputLocal);
                inputLocal = inputQueue.DeQue<T>();

                // --- transpose [rowsW, axis] -> [axis, rowsW] via repeat-batched 32-bit vnchwconv ---
                AscendC::LocalTensor<T> dstLocal = transDstQueue.AllocTensor<T>();
                AscendC::LocalTensor<uint32_t> srcU = inputLocal.template ReinterpretCast<uint32_t>();
                AscendC::LocalTensor<uint32_t> dstU = dstLocal.template ReinterpretCast<uint32_t>();
                for (uint32_t b = 0; b < colBlocks; ++b) {
                    uint32_t s0 = 0;
                    while (s0 < rowsW) {
                        uint32_t sub = rowsW - s0;
                        if (sub > 4080u) sub = 4080u;          // repeatTimes <= 255
                        AscendC::TransDataTo5HDParams tp;
                        tp.dstHighHalf = false;
                        tp.srcHighHalf = false;
                        tp.repeatTimes  = static_cast<uint8_t>(sub / 16u);
                        tp.srcRepStride = static_cast<uint16_t>(2u * axis); // 16 rows = 2*axis datablocks(8 each)
                        tp.dstRepStride = 2;                                // 16 rows = 2 datablocks
                        uint64_t srcList[16];
                        uint64_t dstList[16];
                        for (uint32_t i = 0; i < 16u; ++i) {
                            srcList[i] = (uint64_t)(srcU[(s0 + i) * axis + b * 8u].GetPhyAddr());
                        }
                        for (uint32_t p = 0; p < 8u; ++p) {
                            const uint32_t col = b * 8u + p;
                            dstList[2u * p]      = (uint64_t)(dstU[col * rowsW + s0 + 0u].GetPhyAddr());
                            dstList[2u * p + 1u] = (uint64_t)(dstU[col * rowsW + s0 + 8u].GetPhyAddr());
                        }
                        AscendC::TransDataTo5HD<uint32_t>(dstList, srcList, tp);
                        s0 += sub;
                    }
                }
                transDstQueue.EnQue(dstLocal);
                dstLocal = transDstQueue.DeQue<T>();

                // --- MTE3: write each output[a] = dst[a*rowsW .. +rows] (contiguous) ---
                AscendC::DataCopyExtParams writeParams{
                    1, static_cast<uint32_t>(rows * sizeof(T)), 0, 0, 0};
                uint32_t srcOff = 0;
                for (uint32_t a = 0; a < axis; ++a) {
                    outGm.SetGlobalBuffer(outPtrs[a], outer);
                    AscendC::DataCopyPad(outGm[row], dstLocal[srcOff], writeParams);
                    srcOff += rowsW;
                }

                transDstQueue.FreeTensor(dstLocal);
                inputQueue.FreeTensor(inputLocal);
                row += rows;
            }
        }
    }

    // ============================================================
    // Path A — gather variant, ROW-MAJOR distribution (v9.25)
    //
    // Each core covers `processLength` consecutive OUTER rows × ALL axes. MTE2 reads
    // ONLY this core's rows (1/coreNum of the input). Then for each axisIdx in [0,axis):
    // Gather extracts the column, MTE3 writes to outputs[axisIdx][startRow..endRow].
    //
    // Why this beats axis-major for large-outer cases (e.g. Case4):
    //   - Axis-major: per core does outer/maxRowsPerChunk chunks, each reading the FULL
    //     slice (rows × ALL axes) via MTE2 even though only axisCount/axis fraction is
    //     used. Effective HBM = coreNum × tensor_size (L2 absorbs some duplication).
    //   - Row-major: per core does (outer/coreNum)/maxRowsPerChunk chunks, each MTE2
    //     reads only its own rows. Effective HBM = 1 × tensor_size.
    // Per-chunk work increases (all axes processed vs axisCount) but the chunk COUNT
    // per core drops ~coreNum-fold for big outer — net big win when axis ≤ 128.
    // For high axis (>128) the per-chunk V/MTE3 cost dominates → keep axis-major.
    // ============================================================
    __aicore__ inline void ProcessInnerOneGatherRowMajor()
    {
        if (this->processLength == 0) return;
        const uint32_t outer = this->outerLength;
        const uint32_t axis = this->axisLength;
        const uint32_t startRow = this->startOffset;       // first row idx for this core
        const uint32_t rowsThisCore = this->processLength; // rows assigned to this core
        const uint32_t rowEnd = startRow + rowsThisCore;
        const uint32_t axisBytes = axis * sizeof(T);
        // v9.32: row-major uses the LARGE single buffer → more rows/chunk → bigger writes.
        uint32_t maxRowsPerChunk = RmInputBytes() / axisBytes;
        if (maxRowsPerChunk > PATHA_MAX_ROWS) maxRowsPerChunk = PATHA_MAX_ROWS;
        // Round DOWN to multiple of ChunkRowAlign so chunk-2 MTE2 source offset is 32B-aligned.
        const uint32_t rowAlign = ChunkRowAlign(axisBytes);
        if (maxRowsPerChunk >= rowAlign) {
            maxRowsPerChunk = (maxRowsPerChunk / rowAlign) * rowAlign;
        }

        AscendC::LocalTensor<uint32_t> offsetTensor = offsetBuf.Get<uint32_t>();

        // Cache ALL `axis` output GM pointers (≤ 256 by spec; PTR_CACHE=32 covers most).
        constexpr uint32_t PTR_CACHE = 32;
        __gm__ T* outPtrs[PTR_CACHE];
        const bool useCache = axis <= PTR_CACHE;
        if (useCache) {
            for (uint32_t j = 0; j < axis; ++j) {
                outPtrs[j] = outputs.GetDataPtr<T>(j);
            }
        }

        // Hoist outGm and padParams out of all loops (re-bind outGm per axis change).
        AscendC::GlobalTensor<T> outGm;
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

        uint32_t row = startRow;
        while (row < rowEnd) {
            uint32_t rows = rowEnd - row;
            if (rows > maxRowsPerChunk) rows = maxRowsPerChunk;

            AscendC::LocalTensor<T> inputLocal = inputQueue.AllocTensor<T>();
            AscendC::DataCopyExtParams loadParams{
                1, static_cast<uint32_t>(rows * axisBytes), 0, 0, 0};
            AscendC::DataCopyPad(inputLocal,
                                 inputGm[static_cast<uint64_t>(row) * axis],
                                 loadParams, padParams);
            inputQueue.EnQue(inputLocal);
            inputLocal = inputQueue.DeQue<T>();

            // v9.29 scalar reduction: writeParams is constant across the axis loop
            // (rows fixed for this chunk) — hoist outside the per-axis iteration.
            // Maintain srcBaseAddrBytes incrementally to avoid axisIdx*sizeof(T) mul.
            AscendC::DataCopyExtParams writeParams{
                1, static_cast<uint32_t>(rows * sizeof(T)), 0, 0, 0};
            uint32_t srcBaseAddrBytes = 0;
            for (uint32_t axisIdx = 0; axisIdx < axis; ++axisIdx) {
                outGm.SetGlobalBuffer(useCache ? outPtrs[axisIdx] : outputs.GetDataPtr<T>(axisIdx),
                                      outer);

                AscendC::LocalTensor<T> packed = packedQueue.AllocTensor<T>();
                AscendC::Gather<T>(packed, inputLocal, offsetTensor, srcBaseAddrBytes, rows);
                packedQueue.EnQue(packed);
                packed = packedQueue.DeQue<T>();
                AscendC::DataCopyPad(outGm[row], packed, writeParams);
                packedQueue.FreeTensor(packed);
                srcBaseAddrBytes += sizeof(T);
            }

            inputQueue.FreeTensor(inputLocal);
            row += rows;
        }
    }

    // ============================================================
    // Path A — cast-gather variant (sizeof(T) == 1)
    //   int8/uint8 → Cast to fp16 → vgather → Cast back → MTE3
    //
    // Wrapped in `if constexpr` so the compiler skips this body for non-1-byte
    // types (Cast<half,half> would not compile there).
    // ============================================================
    __aicore__ inline void ProcessInnerOneCastGatherRowMajor()
    {
        if constexpr (sizeof(T) == 1 && !std::is_same_v<T, bool>) {
            if (this->processLength == 0) return;
            const uint32_t outer = this->outerLength;
            const uint32_t axis = this->axisLength;
            const uint32_t startRow = this->startOffset;       // v9.25 row-major
            const uint32_t rowsThisCore = this->processLength;
            const uint32_t rowEnd = startRow + rowsThisCore;
            const uint32_t axisBytesInt = axis * sizeof(T);
            const uint32_t axisBytesFp16 = axis * sizeof(half);
            uint32_t maxRowsPerChunk = PATHA_CG_FP16_BYTES / axisBytesFp16;
            if (maxRowsPerChunk > PATHA_CG_MAX_ROWS) maxRowsPerChunk = PATHA_CG_MAX_ROWS;
            // Align for both int8 input and fp16 work chunk-start to be 32B-aligned.
            const uint32_t rowAlignInt = ChunkRowAlign(axisBytesInt);
            const uint32_t rowAlignFp16 = ChunkRowAlign(axisBytesFp16);
            uint32_t rowAlign = rowAlignInt > rowAlignFp16 ? rowAlignInt : rowAlignFp16;
            if (maxRowsPerChunk >= rowAlign) {
                maxRowsPerChunk = (maxRowsPerChunk / rowAlign) * rowAlign;
            }

            AscendC::LocalTensor<uint32_t> offsetTensor = offsetBuf.Get<uint32_t>();
            AscendC::LocalTensor<half> workFp16 = workFp16Buf.Get<half>();

            // Cache ALL `axis` output GM pointers (≤ 256; PTR_CACHE=32 covers most).
            constexpr uint32_t PTR_CACHE = 32;
            __gm__ T* outPtrs[PTR_CACHE];
            const bool useCache = axis <= PTR_CACHE;
            if (useCache) {
                for (uint32_t j = 0; j < axis; ++j) {
                    outPtrs[j] = outputs.GetDataPtr<T>(j);
                }
            }

            uint32_t row = startRow;
            while (row < rowEnd) {
                uint32_t rows = rowEnd - row;
                if (rows > maxRowsPerChunk) rows = maxRowsPerChunk;

                // 1) Load int8 chunk (this core's rows only)
                AscendC::LocalTensor<T> inputLocal = inputQueue.AllocTensor<T>();
                AscendC::DataCopyExtParams loadParams{
                    1, static_cast<uint32_t>(rows * axisBytesInt), 0, 0, 0};
                AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
                AscendC::DataCopyPad(inputLocal,
                                     inputGm[static_cast<uint64_t>(row) * axis],
                                     loadParams, padParams);
                inputQueue.EnQue(inputLocal);
                inputLocal = inputQueue.DeQue<T>();

                // 2) Cast int8/uint8 → fp16 (whole chunk)
                AscendC::Cast(workFp16, inputLocal, AscendC::RoundMode::CAST_NONE, rows * axis);
                AscendC::PipeBarrier<PIPE_V>();
                inputQueue.FreeTensor(inputLocal);

                // 3) Per axis: vgather (fp16) → cast to int8 → MTE3. ALL axes per chunk.
                for (uint32_t axisIdx = 0; axisIdx < axis; ++axisIdx) {
                    AscendC::GlobalTensor<T> outGm;
                    outGm.SetGlobalBuffer(useCache ? outPtrs[axisIdx] : outputs.GetDataPtr<T>(axisIdx), outer);

                    AscendC::LocalTensor<half> packedFp16 = packedQueue.AllocTensor<half>();
                    const uint32_t srcBaseAddrBytes = axisIdx * sizeof(half);
                    AscendC::Gather<half>(packedFp16, workFp16, offsetTensor, srcBaseAddrBytes, rows);
                    packedQueue.EnQue(packedFp16);
                    packedFp16 = packedQueue.DeQue<half>();

                    AscendC::LocalTensor<T> packedInt = int8PackedQueue.AllocTensor<T>();
                    AscendC::Cast(packedInt, packedFp16, AscendC::RoundMode::CAST_RINT, rows);
                    packedQueue.FreeTensor(packedFp16);
                    int8PackedQueue.EnQue(packedInt);
                    packedInt = int8PackedQueue.DeQue<T>();

                    AscendC::DataCopyExtParams writeParams{
                        1, static_cast<uint32_t>(rows * sizeof(T)), 0, 0, 0};
                    AscendC::DataCopyPad(outGm[row], packedInt, writeParams);
                    int8PackedQueue.FreeTensor(packedInt);
                }

                row += rows;
            }
        }
    }

    // ============================================================
    // Path A — gather variant, AXIS-MAJOR distribution (original/v9.22b)
    // Each core handles `axisCount = processLength` axes × ALL rows. Per chunk reads
    // full input slice (rows × ALL axes) via MTE2, then Gather extracts only this
    // core's axes. Works well when axis is large enough that the Gather/MTE3 cost
    // dominates and HBM redundancy is absorbed by L2 cache.
    // ============================================================
    __aicore__ inline void ProcessInnerOneGather()
    {
        if (this->processLength == 0) return;
        const uint32_t outer = this->outerLength;
        const uint32_t axis = this->axisLength;
        const uint32_t axisCount = this->processLength;
        const uint32_t axisBytes = axis * sizeof(T);
        uint32_t maxRowsPerChunk = PATHA_INPUT_BYTES / axisBytes;
        if (maxRowsPerChunk > PATHA_MAX_ROWS) maxRowsPerChunk = PATHA_MAX_ROWS;
        const uint32_t rowAlign = ChunkRowAlign(axisBytes);
        if (maxRowsPerChunk >= rowAlign) {
            maxRowsPerChunk = (maxRowsPerChunk / rowAlign) * rowAlign;
        }

        AscendC::LocalTensor<uint32_t> offsetTensor = offsetBuf.Get<uint32_t>();

        constexpr uint32_t PTR_CACHE = 32;
        __gm__ T* outPtrs[PTR_CACHE];
        const bool useCache = axisCount <= PTR_CACHE;
        if (useCache) {
            for (uint32_t j = 0; j < axisCount; ++j) {
                outPtrs[j] = outputs.GetDataPtr<T>(this->startOffset + j);
            }
        }

        uint32_t rowStart = 0;
        while (rowStart < outer) {
            uint32_t rows = outer - rowStart;
            if (rows > maxRowsPerChunk) rows = maxRowsPerChunk;

            AscendC::LocalTensor<T> inputLocal = inputQueue.AllocTensor<T>();
            AscendC::DataCopyExtParams loadParams{
                1, static_cast<uint32_t>(rows * axisBytes), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(inputLocal,
                                 inputGm[static_cast<uint64_t>(rowStart) * axis],
                                 loadParams, padParams);
            inputQueue.EnQue(inputLocal);
            inputLocal = inputQueue.DeQue<T>();

            for (uint32_t j = 0; j < axisCount; ++j) {
                const uint32_t axisIdx = this->startOffset + j;
                AscendC::GlobalTensor<T> outGm;
                outGm.SetGlobalBuffer(useCache ? outPtrs[j] : outputs.GetDataPtr<T>(axisIdx),
                                      outer);

                AscendC::LocalTensor<T> packed = packedQueue.AllocTensor<T>();
                const uint32_t srcBaseAddrBytes = axisIdx * sizeof(T);
                AscendC::Gather<T>(packed, inputLocal, offsetTensor, srcBaseAddrBytes, rows);
                packedQueue.EnQue(packed);

                packed = packedQueue.DeQue<T>();
                AscendC::DataCopyExtParams writeParams{
                    1, static_cast<uint32_t>(rows * sizeof(T)), 0, 0, 0};
                AscendC::DataCopyPad(outGm[rowStart], packed, writeParams);
                packedQueue.FreeTensor(packed);
            }

            inputQueue.FreeTensor(inputLocal);
            rowStart += rows;
        }
    }

    // ============================================================
    // Path A — cast-gather variant, AXIS-MAJOR (original/v9.22b)
    // ============================================================
    __aicore__ inline void ProcessInnerOneCastGather()
    {
        if constexpr (sizeof(T) == 1 && !std::is_same_v<T, bool>) {
            if (this->processLength == 0) return;
            const uint32_t outer = this->outerLength;
            const uint32_t axis = this->axisLength;
            const uint32_t axisCount = this->processLength;
            const uint32_t axisBytesInt = axis * sizeof(T);
            const uint32_t axisBytesFp16 = axis * sizeof(half);
            uint32_t maxRowsPerChunk = PATHA_CG_FP16_BYTES / axisBytesFp16;
            if (maxRowsPerChunk > PATHA_CG_MAX_ROWS) maxRowsPerChunk = PATHA_CG_MAX_ROWS;
            const uint32_t rowAlignInt = ChunkRowAlign(axisBytesInt);
            const uint32_t rowAlignFp16 = ChunkRowAlign(axisBytesFp16);
            uint32_t rowAlign = rowAlignInt > rowAlignFp16 ? rowAlignInt : rowAlignFp16;
            if (maxRowsPerChunk >= rowAlign) {
                maxRowsPerChunk = (maxRowsPerChunk / rowAlign) * rowAlign;
            }

            AscendC::LocalTensor<uint32_t> offsetTensor = offsetBuf.Get<uint32_t>();
            AscendC::LocalTensor<half> workFp16 = workFp16Buf.Get<half>();

            uint32_t rowStart = 0;
            while (rowStart < outer) {
                uint32_t rows = outer - rowStart;
                if (rows > maxRowsPerChunk) rows = maxRowsPerChunk;

                AscendC::LocalTensor<T> inputLocal = inputQueue.AllocTensor<T>();
                AscendC::DataCopyExtParams loadParams{
                    1, static_cast<uint32_t>(rows * axisBytesInt), 0, 0, 0};
                AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
                AscendC::DataCopyPad(inputLocal,
                                     inputGm[static_cast<uint64_t>(rowStart) * axis],
                                     loadParams, padParams);
                inputQueue.EnQue(inputLocal);
                inputLocal = inputQueue.DeQue<T>();

                AscendC::Cast(workFp16, inputLocal, AscendC::RoundMode::CAST_NONE, rows * axis);
                AscendC::PipeBarrier<PIPE_V>();
                inputQueue.FreeTensor(inputLocal);

                for (uint32_t j = 0; j < axisCount; ++j) {
                    const uint32_t axisIdx = this->startOffset + j;
                    AscendC::GlobalTensor<T> outGm;
                    outGm.SetGlobalBuffer(outputs.GetDataPtr<T>(axisIdx), outer);

                    AscendC::LocalTensor<half> packedFp16 = packedQueue.AllocTensor<half>();
                    const uint32_t srcBaseAddrBytes = axisIdx * sizeof(half);
                    AscendC::Gather<half>(packedFp16, workFp16, offsetTensor, srcBaseAddrBytes, rows);
                    packedQueue.EnQue(packedFp16);
                    packedFp16 = packedQueue.DeQue<half>();

                    AscendC::LocalTensor<T> packedInt = int8PackedQueue.AllocTensor<T>();
                    AscendC::Cast(packedInt, packedFp16, AscendC::RoundMode::CAST_RINT, rows);
                    packedQueue.FreeTensor(packedFp16);
                    int8PackedQueue.EnQue(packedInt);
                    packedInt = int8PackedQueue.DeQue<T>();

                    AscendC::DataCopyExtParams writeParams{
                        1, static_cast<uint32_t>(rows * sizeof(T)), 0, 0, 0};
                    AscendC::DataCopyPad(outGm[rowStart], packedInt, writeParams);
                    int8PackedQueue.FreeTensor(packedInt);
                }

                rowStart += rows;
            }
        }
    }

    // ============================================================
    // Path A — scalar fallback (very large axisBytes)
    // ============================================================
    __aicore__ inline void ProcessInnerOneScalar()
    {
        if (this->processLength == 0) return;
        const uint32_t outer = this->outerLength;
        const uint32_t axis = this->axisLength;
        const uint32_t axisCount = this->processLength;

        const uint32_t batchBytes = axisCount * sizeof(T);
        const bool batched = (batchBytes <= BLOCK_BYTES);
        const uint32_t blockLenBytes = batched ? batchBytes : static_cast<uint32_t>(sizeof(T));
        const uint32_t paddedRowBytes = AlignUpU32(blockLenBytes, BLOCK_BYTES);
        const uint32_t blockElems = paddedRowBytes / sizeof(T);
        // Use the SAME capacity Init used (capped to keep queue UB under 64 KB).
        const uint32_t fallbackTile = this->tileLength < 32768u ? this->tileLength : 32768u;
        uint32_t rowsPerTile = fallbackTile / paddedRowBytes;
        if (rowsPerTile == 0) rowsPerTile = 1;
        if (rowsPerTile > MAX_BLOCK_COUNT) rowsPerTile = MAX_BLOCK_COUNT;
        const uint32_t innerPerBatch = batched ? axisCount : 1u;
        const uint32_t srcStrideBytes = (axis - innerPerBatch) * sizeof(T);

        for (uint32_t outerLoopStart = 0; outerLoopStart < (batched ? 1u : axisCount); ++outerLoopStart) {
            const uint32_t axesInPass = batched ? axisCount : 1u;
            const uint32_t firstAxisInPass = batched ? 0u : outerLoopStart;

            uint32_t row = 0;
            while (row < outer) {
                uint32_t rows = outer - row;
                if (rows > rowsPerTile) rows = rowsPerTile;

                AscendC::LocalTensor<T> gatherLocal = gatherQueue.AllocTensor<T>();
                AscendC::DataCopyExtParams inParams{
                    static_cast<uint16_t>(rows), blockLenBytes, srcStrideBytes, 0, 0};
                AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
                const uint64_t inOffset =
                    static_cast<uint64_t>(row) * axis + this->startOffset + firstAxisInPass;
                AscendC::DataCopyPad(gatherLocal, inputGm[inOffset], inParams, padParams);
                gatherQueue.EnQue(gatherLocal);
                gatherLocal = gatherQueue.DeQue<T>();

                for (uint32_t j = 0; j < axesInPass; ++j) {
                    const uint32_t axisIdx = this->startOffset + firstAxisInPass + j;
                    AscendC::GlobalTensor<T> outGm;
                    outGm.SetGlobalBuffer(outputs.GetDataPtr<T>(axisIdx), outer);

                    AscendC::LocalTensor<T> packed = copyQueue.AllocTensor<T>();
                    for (uint32_t r = 0; r < rows; ++r) {
                        packed.SetValue(r, gatherLocal.GetValue(r * blockElems + j));
                    }
                    copyQueue.EnQue(packed);
                    packed = copyQueue.DeQue<T>();

                    AscendC::DataCopyExtParams outParams{
                        1, static_cast<uint32_t>(rows * sizeof(T)), 0, 0, 0};
                    AscendC::DataCopyPad(outGm[row], packed, outParams);
                    copyQueue.FreeTensor(packed);
                }

                gatherQueue.FreeTensor(gatherLocal);
                row += rows;
            }
        }
    }

    // ============================================================
    // Path B — innerLength > 1, single row fits in tile
    // ============================================================
    // ============================================================
    // Path B — innerLength > 1, single row fits in tile (v9.27 optimized)
    //
    // v9.27 changes:
    //   - INCREMENTAL (axisIdx, outerStart) state instead of pos / outer + pos % outer
    //     per iter — eliminates an integer division per chunk (was one of the dominant
    //     scalar costs for big-data Case5 with 320+ chunks per core).
    //   - OUTPUT GM pointer CACHE (up to 32 axisIdx values) — avoids GetDataPtr call
    //     per chunk; only re-bind on axisIdx change.
    //   - inOffset maintained incrementally (was a uint64 multiply per chunk).
    //   - Hoist constants out of loop (inOffsetStride, outSrcStrideDB, padParams).
    // ============================================================
    __aicore__ inline void ProcessInnerSmall(uint32_t innerBytes)
    {
        if (this->processLength == 0) return;
        const uint32_t outer = this->outerLength;
        const uint32_t inner = this->innerLength;
        const uint32_t axis = this->axisLength;
        const uint32_t paddedRowBytes = AlignUpU32(innerBytes, BLOCK_BYTES);
        uint32_t rowsPerTile = this->tileLength / paddedRowBytes;
        if (rowsPerTile == 0) rowsPerTile = 1;
        if (rowsPerTile > MAX_BLOCK_COUNT) rowsPerTile = MAX_BLOCK_COUNT;
        const uint32_t outSrcStrideDB = (paddedRowBytes - innerBytes) / BLOCK_BYTES;
        const uint32_t inSrcStrideBytes = (axis - 1) * innerBytes;
        const uint64_t outBufSize = static_cast<uint64_t>(outer) * inner;
        const uint64_t axisInnerStride = static_cast<uint64_t>(axis) * inner;
        // Stride in INPUT GM advanced by `rows` consecutive (outerStart) rows of same axisIdx.
        // inOffset advances by rows * axis * inner each chunk within the same axisIdx.

        // Cache output GM pointers (≤32 axisIdx covered; falls back to GetDataPtr beyond).
        constexpr uint32_t PTR_CACHE = 32;
        __gm__ T* outPtrs[PTR_CACHE];
        const bool useCache = axis <= PTR_CACHE;
        if (useCache) {
            for (uint32_t i = 0; i < axis; ++i) {
                outPtrs[i] = outputs.GetDataPtr<T>(i);
            }
        }

        // Initial decomposition (one division on entry only).
        const uint32_t pos0 = this->startOffset;
        const uint32_t endPos = pos0 + this->processLength;
        uint32_t axisIdx = pos0 / outer;
        uint32_t outerStart = pos0 - axisIdx * outer;
        uint64_t inOffset = static_cast<uint64_t>(outerStart) * axisInnerStride
                            + static_cast<uint64_t>(axisIdx) * inner;

        AscendC::GlobalTensor<T> outGm;
        __gm__ T* curOutBase = useCache ? outPtrs[axisIdx] : outputs.GetDataPtr<T>(axisIdx);
        outGm.SetGlobalBuffer(curOutBase, outBufSize);
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

        uint32_t pos = pos0;
        while (pos < endPos) {
            const uint32_t remainInAxis = outer - outerStart;
            const uint32_t remainTotal = endPos - pos;
            uint32_t rows = rowsPerTile < remainInAxis ? rowsPerTile : remainInAxis;
            if (rows > remainTotal) rows = remainTotal;

            AscendC::LocalTensor<T> local = copyQueue.AllocTensor<T>();
            AscendC::DataCopyExtParams inParams{
                static_cast<uint16_t>(rows), innerBytes, inSrcStrideBytes, 0, 0};
            AscendC::DataCopyPad(local, inputGm[inOffset], inParams, padParams);
            copyQueue.EnQue(local);

            local = copyQueue.DeQue<T>();
            AscendC::DataCopyExtParams outParams{
                static_cast<uint16_t>(rows), innerBytes, outSrcStrideDB, 0, 0};
            AscendC::DataCopyPad(outGm[static_cast<uint64_t>(outerStart) * inner], local, outParams);
            copyQueue.FreeTensor(local);

            // Advance state incrementally — no division.
            pos += rows;
            outerStart += rows;
            inOffset += static_cast<uint64_t>(rows) * axisInnerStride;
            if (outerStart == outer) {
                // Crossed into next axisIdx.
                outerStart = 0;
                ++axisIdx;
                inOffset = static_cast<uint64_t>(axisIdx) * inner;
                if (pos < endPos) {
                    curOutBase = useCache ? outPtrs[axisIdx] : outputs.GetDataPtr<T>(axisIdx);
                    outGm.SetGlobalBuffer(curOutBase, outBufSize);
                }
            }
        }
    }

    // ============================================================
    // Path C — innerLength > 1, inner row exceeds tile
    //
    // Optimization: incremental (axisIdx, outerIdx, innerIdx, outOff) state replaces
    // two integer divisions per chunk. Output GM pointer cached, refetched only on axis
    // crossing — saves ~50 GetDataPtr calls per core for large cases.
    // ============================================================
    __aicore__ inline void ProcessInnerHuge(uint32_t innerBytes)
    {
        if (this->processLength == 0) return;
        const uint32_t outer = this->outerLength;
        const uint32_t inner = this->innerLength;
        const uint32_t axis = this->axisLength;
        const uint32_t elemsPerTile = this->tileLength / sizeof(T);

        // INPUT-linear tiling: host gives us startOffset as a linear INPUT position.
        // MTE2 reads are perfectly sequential in HBM (input layout [outer, axis, inner]
        // advances by +1 byte each step). L2 prefetch streams. When we cross an axis
        // boundary, output tensor changes — cache pointers up to PTR_CACHE so the
        // switch is just an array lookup (kept small to avoid UB-spill of stack array).
        constexpr uint32_t PTR_CACHE = 32;
        __gm__ T* outPtrs[PTR_CACHE];
        const bool useCache = axis <= PTR_CACHE;
        if (useCache) {
            for (uint32_t i = 0; i < axis; ++i) {
                outPtrs[i] = outputs.GetDataPtr<T>(i);
            }
        }

        const uint64_t startInput = this->startOffset;
        const uint64_t axisInner = static_cast<uint64_t>(axis) * inner;
        uint32_t outerIdx = static_cast<uint32_t>(startInput / axisInner);
        uint64_t rem = startInput - static_cast<uint64_t>(outerIdx) * axisInner;
        uint32_t axisIdx = static_cast<uint32_t>(rem / inner);
        uint32_t innerIdx = static_cast<uint32_t>(rem - static_cast<uint64_t>(axisIdx) * inner);
        uint64_t inOffset = startInput;
        uint32_t outOff = outerIdx * inner + innerIdx;
        const uint32_t outBufSize = outer * inner;

        AscendC::GlobalTensor<T> outGm;
        outGm.SetGlobalBuffer(useCache ? outPtrs[axisIdx] : outputs.GetDataPtr<T>(axisIdx),
                              outBufSize);
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

        uint32_t done = 0;
        while (done < this->processLength) {
            uint32_t chunk = elemsPerTile;
            const uint32_t remainInner = inner - innerIdx;
            if (chunk > remainInner) chunk = remainInner;
            const uint32_t remainTotal = this->processLength - done;
            if (chunk > remainTotal) chunk = remainTotal;
            AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(chunk * sizeof(T)), 0, 0, 0};

            AscendC::LocalTensor<T> local = copyQueue.AllocTensor<T>();
            AscendC::DataCopyPad(local, inputGm[inOffset], cp, padParams);
            copyQueue.EnQue(local);
            local = copyQueue.DeQue<T>();
            AscendC::DataCopyPad(outGm[outOff], local, cp);
            copyQueue.FreeTensor(local);

            done += chunk;
            innerIdx += chunk;
            inOffset += chunk;
            outOff += chunk;
            if (innerIdx == inner) {
                innerIdx = 0;
                if (++axisIdx == axis) {
                    axisIdx = 0;
                    ++outerIdx;
                }
                outOff = outerIdx * inner;
                if (done < this->processLength) {
                    outGm.SetGlobalBuffer(useCache ? outPtrs[axisIdx]
                                                   : outputs.GetDataPtr<T>(axisIdx),
                                          outBufSize);
                }
            }
        }
    }

private:
    AscendC::TPipe *pipe;
    // Path B/C + Path-A scalar fallback
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, BUFFER_NUM> copyQueue;
    // Path-A scalar fallback only
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> gatherQueue;
    // Path-A gather variant (in: MTE2, out: vgather/V).
    // BUFFER_NUM=2 enables chunk pipelining when outer has multi-chunk; for single-chunk
    // cases the second buffer is unused but free (UB is also used by cast-gather path).
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inputQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> offsetBuf;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> packedQueue;
    // Path-A cast-gather variant extras
    AscendC::TBuf<AscendC::QuePosition::VECCALC> workFp16Buf;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> int8PackedQueue;
    // Path-A transpose variant (v9.39): dst holds the whole [axis, rows] result, tmp is the
    // NHWC2NCHW scratch. BN1 (no chunk overlap) for first-correctness; revisit for overlap.
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> transDstQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> transTmpBuf;
    uint32_t transMaxRows;
    uint32_t transRowMult;
    AscendC::GlobalTensor<T> inputGm;
    AscendC::ListTensorDesc outputs;
    uint32_t totalLength;
    uint32_t outerLength;
    uint32_t axisLength;
    uint32_t innerLength;
    uint32_t outputLength;
    uint32_t tileLength;
    uint32_t startOffset;
    uint32_t processLength;
    uint8_t pathAMode;  // 0=scalar, 1=gather, 2=cast-gather, 3=transpose (v9.39)
};

extern "C" __global__ __aicore__ void unpack(GM_ADDR input, GM_ADDR output, GM_ADDR, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    AscendC::TPipe pipe;
    KernelUnpack<DTYPE_INPUT> op;
    op.Init(input, output, tiling_data.totalLength, tiling_data.outerLength, tiling_data.axisLength,
            tiling_data.innerLength, tiling_data.blockLength, tiling_data.tileLength, &pipe);
    op.Process();
}
