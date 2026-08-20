#include "kernel_operator.h"
#include <type_traits>

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t ASSIGN_MAX_DIMS = 8;
constexpr uint32_t SMALL_COPY_BYTES = 8 * 1024;
// UB budget for KernelAssignLastDimScalarBroadcast (CONDITIONAL init — only
// one set of buffers is allocated at a time).
//
// Buffer sizes are picked so that AscendC::Broadcast's full internal workspace
// fits (per GetBroadCastMaxMinTmpSize this can be large for medium k with
// many rows).  We give 96KB to bcTmpBuf which empirically lets K>1 work for
// k up to ~512 elements (any dtype).
constexpr uint32_t BC_IN_BYTES  = 4U * 1024U;     // 4 KB  (src scalars; K ≤ 2040)
constexpr uint32_t BC_OUT_BYTES = 32U * 1024U;    // 32 KB (K * N * sizeof(T) per batch)
constexpr uint32_t BC_TMP_BYTES = 96U * 1024U;    // 96 KB (Broadcast workspace)

// Larger per-buffer tile size for k1/k2/k5/k7 kernels (which only use 1-2
// buffers).  Host computes tileLength = ubSize/8/typeSize ≈ 24KB which wastes
// 75% of UB.  Override to 80KB per buffer → 160KB total with BUFFER_NUM=2,
// leaving 32KB UB headroom.  Reduces per-MTE3 overhead for large transfers.
constexpr uint32_t BIG_TILE_BYTES = 80U * 1024U;

template <typename T> class KernelAssignSmall {
public:
    __aicore__ inline KernelAssignSmall() {}

    __aicore__ inline void Init(GM_ADDR other, GM_ADDR output, uint32_t totalLength)
    {
        this->processLength = totalLength;
        otherGm.SetGlobalBuffer((__gm__ T *)other, this->processLength);
        inputGm.SetGlobalBuffer((__gm__ T *)output, this->processLength);
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t smallElems = SMALL_COPY_BYTES / sizeof(T);
        AscendC::LocalTensor<T> local(AscendC::TPosition::VECCALC, 0, smallElems);
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->processLength * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(local, otherGm, copyParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(inputGm, local, copyParams);
    }

private:
    AscendC::GlobalTensor<T> otherGm;
    AscendC::GlobalTensor<T> inputGm;
    uint32_t processLength;
};

template <typename T> class KernelAssignScalarBroadcast {
public:
    __aicore__ inline KernelAssignScalarBroadcast() {}

    __aicore__ inline void Init(GM_ADDR other, GM_ADDR output, uint32_t totalLength, uint32_t blockLength,
                                uint32_t tileLength, AscendC::TPipe *pipeIn)
    {
        this->pipe = pipeIn;
        this->startOffset = AscendC::GetBlockIdx() * blockLength;
        uint32_t remaining = totalLength > this->startOffset ? totalLength - this->startOffset : 0;
        this->processLength = remaining < blockLength ? remaining : blockLength;
        // Big tile when worthwhile: if this core's data exceeds host's tile,
        // use BIG_TILE to reduce chunk count (fewer per-MTE3 calls).
        const uint32_t bigTile = BIG_TILE_BYTES / sizeof(T);
        this->tileLength = (this->processLength > tileLength && bigTile > tileLength) ? bigTile : tileLength;
        otherGm.SetGlobalBuffer((__gm__ T *)other, 1);
        inputGm.SetGlobalBuffer((__gm__ T *)output + this->startOffset, this->processLength);
        pipe->InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe->InitBuffer(scalarBuf, sizeof(T) * 32);
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<T> scalarLocal = scalarBuf.Get<T>();
        AscendC::DataCopyExtParams scalarCopyParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(scalarLocal, otherGm, scalarCopyParams, padParams);
        event_t eventIdMte2ToS = static_cast<event_t>(pipe->FetchEventID(AscendC::HardEvent::MTE2_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventIdMte2ToS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventIdMte2ToS);
        const T value = scalarLocal.GetValue(0);
        event_t eventIdSToV = static_cast<event_t>(pipe->FetchEventID(AscendC::HardEvent::S_V));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIdSToV);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIdSToV);

        uint32_t offset = 0;
        while (offset < this->processLength) {
            uint32_t curLength = this->processLength - offset;
            curLength = curLength < this->tileLength ? curLength : this->tileLength;
            AscendC::LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
            FillLocal(outputLocal, value, curLength);
            outQueue.EnQue(outputLocal);
            CopyOut(offset, curLength);
            offset += curLength;
        }
    }

private:
    __aicore__ inline void FillLocal(const AscendC::LocalTensor<T> &outputLocal, T value, uint32_t length)
    {
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, bool>) {
            const uint8_t byteValue = static_cast<uint8_t>(value);
            const uint32_t fillValue = static_cast<uint32_t>(byteValue) * 0x01010101U;
            AscendC::LocalTensor<uint32_t> outputWords = outputLocal.template ReinterpretCast<uint32_t>();
            AscendC::Duplicate(outputWords, fillValue, (length + sizeof(uint32_t) - 1) / sizeof(uint32_t));
        } else {
            AscendC::Duplicate(outputLocal, value, length);
        }
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length)
    {
        AscendC::LocalTensor<T> outputLocal = outQueue.DeQue<T>();
        if (length == this->tileLength) {
            AscendC::DataCopy(inputGm[offset], outputLocal, length);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(inputGm[offset], outputLocal, copyParams);
        }
        outQueue.FreeTensor(outputLocal);
    }

private:
    AscendC::TPipe *pipe;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scalarBuf;
    AscendC::GlobalTensor<T> otherGm;
    AscendC::GlobalTensor<T> inputGm;
    uint32_t tileLength;
    uint32_t startOffset;
    uint32_t processLength;
};

template <typename T> class KernelAssignScalarBroadcastSmall {
public:
    __aicore__ inline KernelAssignScalarBroadcastSmall() {}

    __aicore__ inline void Init(GM_ADDR other, GM_ADDR output, uint32_t totalLength)
    {
        this->processLength = totalLength;
        otherGm.SetGlobalBuffer((__gm__ T *)other, 1);
        inputGm.SetGlobalBuffer((__gm__ T *)output, this->processLength);
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t smallElems = SMALL_COPY_BYTES / sizeof(T);
        AscendC::LocalTensor<T> local(AscendC::TPosition::VECCALC, 0, smallElems);
        AscendC::DataCopyExtParams scalarCopyParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(local, otherGm, scalarCopyParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

        const T value = local.GetValue(0);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        FillLocal(local, value, this->processLength);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->processLength * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(inputGm, local, copyParams);
    }

private:
    __aicore__ inline void FillLocal(const AscendC::LocalTensor<T> &local, T value, uint32_t length)
    {
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, bool>) {
            const uint8_t byteValue = static_cast<uint8_t>(value);
            const uint32_t fillValue = static_cast<uint32_t>(byteValue) * 0x01010101U;
            AscendC::LocalTensor<uint32_t> localWords = local.template ReinterpretCast<uint32_t>();
            AscendC::Duplicate(localWords, fillValue, (length + sizeof(uint32_t) - 1) / sizeof(uint32_t));
        } else {
            AscendC::Duplicate(local, value, length);
        }
    }

private:
    AscendC::GlobalTensor<T> otherGm;
    AscendC::GlobalTensor<T> inputGm;
    uint32_t processLength;
};

template <typename T> class KernelAssignRowRepeat {
public:
    __aicore__ inline KernelAssignRowRepeat() {}

    __aicore__ inline void Init(GM_ADDR other, GM_ADDR output, uint32_t totalLength, uint32_t blockLength,
                                uint32_t tileLength, uint32_t innerLength, AscendC::TPipe *pipeIn)
    {
        this->pipe = pipeIn;
        this->innerLength = innerLength;
        this->rowCount = this->innerLength == 0 ? 0 : totalLength / this->innerLength;
        this->blockRows = (this->rowCount + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
        this->startRow = AscendC::GetBlockIdx() * this->blockRows;
        uint32_t remainingRows = this->rowCount > this->startRow ? this->rowCount - this->startRow : 0;
        this->processRows = remainingRows < this->blockRows ? remainingRows : this->blockRows;
        this->startOffset = AscendC::GetBlockIdx() * blockLength;
        uint32_t remaining = totalLength > this->startOffset ? totalLength - this->startOffset : 0;
        this->processLength = remaining < blockLength ? remaining : blockLength;
        // Big tile when this core has more than one host-tile of data.
        const uint32_t bigTile = BIG_TILE_BYTES / sizeof(T);
        this->tileLength = (this->processLength > tileLength && bigTile > tileLength) ? bigTile : tileLength;
        otherGm.SetGlobalBuffer((__gm__ T *)other, this->innerLength);
        inputGm.SetGlobalBuffer((__gm__ T *)output, totalLength);
        pipe->InitBuffer(copyQueue, BUFFER_NUM, this->tileLength * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        if (CanBatchWholeRows()) {
            ProcessWholeRows();
            return;
        }
        ProcessByElementRange();
    }

private:
    __aicore__ inline bool CanBatchWholeRows() const
    {
        return this->innerLength != 0 && this->innerLength <= this->tileLength;
    }

    __aicore__ inline void ProcessWholeRows()
    {
        CopyIn(0, this->innerLength);
        CopyOutRepeatedRows(this->startRow * this->innerLength, this->innerLength, this->processRows);
    }

    __aicore__ inline void ProcessByElementRange()
    {
        uint32_t offset = 0;
        while (offset < this->processLength) {
            const uint32_t globalOffset = this->startOffset + offset;
            const uint32_t col = this->innerLength == 0 ? 0 : globalOffset % this->innerLength;
            uint32_t curLength = this->processLength - offset;
            const uint32_t rowRemain = this->innerLength - col;
            curLength = curLength < this->tileLength ? curLength : this->tileLength;
            curLength = curLength < rowRemain ? curLength : rowRemain;
            CopyIn(col, curLength);
            CopyOut(globalOffset, curLength);
            offset += curLength;
        }
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length)
    {
        AscendC::LocalTensor<T> outputLocal = copyQueue.AllocTensor<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(outputLocal, otherGm[offset], copyParams, padParams);
        copyQueue.EnQue(outputLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length)
    {
        AscendC::LocalTensor<T> outputLocal = copyQueue.DeQue<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(inputGm[offset], outputLocal, copyParams);
        copyQueue.FreeTensor(outputLocal);
    }

    __aicore__ inline void CopyOutRepeatedRows(uint32_t offset, uint32_t length, uint32_t batchRows)
    {
        AscendC::LocalTensor<T> outputLocal = copyQueue.DeQue<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
        for (uint32_t row = 0; row < batchRows; ++row) {
            AscendC::DataCopyPad(inputGm[offset + row * this->innerLength], outputLocal, copyParams);
        }
        copyQueue.FreeTensor(outputLocal);
    }

private:
    AscendC::TPipe *pipe;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, BUFFER_NUM> copyQueue;
    AscendC::GlobalTensor<T> otherGm;
    AscendC::GlobalTensor<T> inputGm;
    uint32_t tileLength;
    uint32_t innerLength;
    uint32_t rowCount;
    uint32_t blockRows;
    uint32_t startRow;
    uint32_t processRows;
    uint32_t startOffset;
    uint32_t processLength;
};

template <typename T> class KernelAssignBlockRepeat {
public:
    __aicore__ inline KernelAssignBlockRepeat() {}

    __aicore__ inline void Init(GM_ADDR other, GM_ADDR output, uint32_t totalLength, uint32_t blockLength,
                                uint32_t tileLength, uint32_t repeatLength, AscendC::TPipe *pipeIn)
    {
        this->pipe = pipeIn;
        this->repeatLength = repeatLength;
        this->startOffset = AscendC::GetBlockIdx() * blockLength;
        uint32_t remaining = totalLength > this->startOffset ? totalLength - this->startOffset : 0;
        this->processLength = remaining < blockLength ? remaining : blockLength;
        // Big tile when worthwhile: more than one host-tile of data per core.
        const uint32_t bigTile = BIG_TILE_BYTES / sizeof(T);
        this->tileLength = (this->processLength > tileLength && bigTile > tileLength) ? bigTile : tileLength;
        otherGm.SetGlobalBuffer((__gm__ T *)other, this->repeatLength);
        inputGm.SetGlobalBuffer((__gm__ T *)output, totalLength);
        pipe->InitBuffer(copyQueue, BUFFER_NUM, this->tileLength * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        uint32_t offset = 0;
        while (offset < this->processLength) {
            const uint32_t globalOffset = this->startOffset + offset;
            const uint32_t sourceOffset = this->repeatLength == 0 ? 0 : globalOffset % this->repeatLength;
            uint32_t curLength = this->processLength - offset;
            const uint32_t repeatRemain = this->repeatLength - sourceOffset;
            curLength = curLength < this->tileLength ? curLength : this->tileLength;
            curLength = curLength < repeatRemain ? curLength : repeatRemain;
            CopyIn(sourceOffset, curLength);
            CopyOut(globalOffset, curLength);
            offset += curLength;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length)
    {
        AscendC::LocalTensor<T> outputLocal = copyQueue.AllocTensor<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(outputLocal, otherGm[offset], copyParams, padParams);
        copyQueue.EnQue(outputLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length)
    {
        AscendC::LocalTensor<T> outputLocal = copyQueue.DeQue<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(inputGm[offset], outputLocal, copyParams);
        copyQueue.FreeTensor(outputLocal);
    }

private:
    AscendC::TPipe *pipe;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, BUFFER_NUM> copyQueue;
    AscendC::GlobalTensor<T> otherGm;
    AscendC::GlobalTensor<T> inputGm;
    uint32_t tileLength;
    uint32_t repeatLength;
    uint32_t startOffset;
    uint32_t processLength;
};

template <typename T> class KernelAssignLastDimScalarBroadcast {
public:
    __aicore__ inline KernelAssignLastDimScalarBroadcast() {}

    __aicore__ inline void Init(GM_ADDR other, GM_ADDR output, uint32_t totalLength, uint32_t blockLength,
                                uint32_t tileLength, uint32_t innerLength, uint32_t otherLength,
                                AscendC::TPipe *pipeIn)
    {
        this->pipe = pipeIn;
        this->tileLength = tileLength;
        this->innerLength = innerLength;
        this->otherLength = otherLength;
        this->rowCount = this->innerLength == 0 ? 0 : totalLength / this->innerLength;
        this->repeatRows = this->otherLength == 0 ? 0 : this->rowCount / this->otherLength;
        this->useRowOffset = this->otherLength == this->rowCount ? 1U : 0U;
        this->blockRows = (this->rowCount + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
        this->startRow = AscendC::GetBlockIdx() * this->blockRows;
        uint32_t remainingRows = this->rowCount > this->startRow ? this->rowCount - this->startRow : 0;
        this->processRows = remainingRows < this->blockRows ? remainingRows : this->blockRows;
        this->startOffset = AscendC::GetBlockIdx() * blockLength;
        uint32_t remaining = totalLength > this->startOffset ? totalLength - this->startOffset : 0;
        this->processLength = remaining < blockLength ? remaining : blockLength;
        otherGm.SetGlobalBuffer((__gm__ T *)other, this->otherLength);
        inputGm.SetGlobalBuffer((__gm__ T *)output, totalLength);
        // Decide which buffer set to allocate.  The Broadcast path and the
        // scalar-Duplicate path share the same UB; allocating both would
        // exceed ~192 KB.  Since the paths are mutually exclusive at runtime,
        // we only initialise the one that will actually be used.
        // Broadcast path handles ANY N via inner-N chunking (per task spec N up to 10000).
        this->willUseBroadcast = (this->useRowOffset != 0) && (this->innerLength > 0);
        if (this->willUseBroadcast) {
            // AscendC::Broadcast K-batch pipeline (all-vector, no scalar loop):
            //   bcInQueue (MTE2 → V), bcOutQueue (V → MTE3), bcTmpBuf (Broadcast workspace).
            pipe->InitBuffer(bcInQueue,  2, BC_IN_BYTES);
            pipe->InitBuffer(bcOutQueue, 2, BC_OUT_BYTES);
            pipe->InitBuffer(bcTmpBuf,   BC_TMP_BYTES);
        } else {
            // Scalar-Duplicate path (per-row or repeated-row).
            pipe->InitBuffer(scalarQueue, BUFFER_NUM, this->tileLength * sizeof(T));
            pipe->InitBuffer(scalarBuf, sizeof(T) * 32);
            pipe->InitBuffer(otherBuf, this->tileLength * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        if (this->willUseBroadcast) {
            ProcessUniqueRowsBroadcast();
            return;
        }
        if (this->useRowOffset == 0 && this->otherLength > 0 && this->repeatRows > 1) {
            ProcessRepeatedRows();
            return;
        }
        if (this->useRowOffset != 0) {
            // Very large N that doesn't fit Broadcast output buffer;
            // fall through to scalar path.
        }
        ProcessByOutputRows();
    }

private:
    __aicore__ inline void ProcessByOutputRows()
    {
        uint32_t offset = 0;
        while (offset < this->processLength) {
            const uint32_t globalOffset = this->startOffset + offset;
            const uint32_t row = this->innerLength == 0 ? 0 : globalOffset / this->innerLength;
            const uint32_t col = this->innerLength == 0 ? 0 : globalOffset % this->innerLength;
            const uint32_t otherOffset =
                this->useRowOffset != 0 ? row : (this->otherLength == 0 ? 0 : row % this->otherLength);
            uint32_t curLength = this->processLength - offset;
            const uint32_t rowRemain = this->innerLength - col;
            curLength = curLength < this->tileLength ? curLength : this->tileLength;
            curLength = curLength < rowRemain ? curLength : rowRemain;
            const T value = LoadScalar(otherOffset);
            CopyScalarRows(value, globalOffset, curLength, 1, 1);
            offset += curLength;
        }
    }

    // K-batch fast path: hybrid algorithm avoiding both scalar loops AND
    // AscendC::Broadcast's empirical k≤8192 element limit.
    //
    // Two regimes:
    //   1) Multi-row Broadcast (when N ≤ 8192 AND rowBytesAligned ≤ BC_OUT_BYTES):
    //      Use AscendC::Broadcast<BT, 2, 1>(K, N) on K rows at once.  Pure
    //      vector op — brcb → Copy → GatherMask internally, zero scalar loops.
    //   2) Single-row Duplicate (when N > 8192 OR row too big for K>1 packing):
    //      K=1, load one scalar, Duplicate(N or chunkN) per row.  ONE scalar
    //      GetValue per row (not a per-element scalar loop), then a single
    //      vector Duplicate fills the row.
    //
    // For Broadcast, dtypes outside its native set (int16/bf16/int32) get a
    // bit-exact same-width reinterpret_cast.  Broadcast just replicates bit
    // patterns so this is correct for all 8 Assign dtypes.
    __aicore__ inline void ProcessUniqueRowsBroadcast()
    {
        if (this->processRows == 0 || this->innerLength == 0) {
            return;
        }

        constexpr uint32_t TYPE_BYTES = sizeof(T);
        constexpr uint32_t BLOCK_32 = 32U;
        // Threshold for routing to multi-row Broadcast vs single-row Duplicate.
        // Larger N → more chance Broadcast crashes; smaller N → less batching benefit.
        constexpr uint32_t BROADCAST_SAFE_N = 512U;
        const uint32_t N = this->innerLength;
        const uint32_t rowBytes = N * TYPE_BYTES;
        const uint32_t rowBytesAligned = (rowBytes + BLOCK_32 - 1U) / BLOCK_32 * BLOCK_32;

        if (N <= BROADCAST_SAFE_N && rowBytesAligned <= BC_OUT_BYTES) {
            ProcessMultiRowBroadcast(N, rowBytesAligned);
        } else {
            ProcessRowByDuplicate(N);
        }
    }

    // Multi-row Broadcast (small/medium N).
    __aicore__ inline void ProcessMultiRowBroadcast(uint32_t N, uint32_t rowBytesAligned)
    {
        using BT = std::conditional_t<sizeof(T) == 1U, uint8_t,
                   std::conditional_t<sizeof(T) == 2U, half, float>>;
        constexpr uint32_t TYPE_BYTES = sizeof(T);
        const uint32_t rowBytes = N * TYPE_BYTES;

        uint32_t kMax = BC_OUT_BYTES / rowBytesAligned;
        // Brcb limit: K ≤ 2040. With BROADCAST_SAFE_N=512, this is also satisfied
        // (since BC_OUT_BYTES/64 < 2040 ... wait actually for tiny rowBytesAligned
        // like 32 we'd get K=1024, still under 2040).
        if (kMax > 2040U) kMax = 2040U;
        const uint32_t srcCapacity = BC_IN_BYTES / TYPE_BYTES;
        if (kMax > srcCapacity) kMax = srcCapacity;
        if (kMax == 0U) return;

        uint32_t rowOffset = 0;
        while (rowOffset < this->processRows) {
            uint32_t rows = this->processRows - rowOffset;
            rows = rows < kMax ? rows : kMax;

            AscendC::LocalTensor<T> srcT = bcInQueue.template AllocTensor<T>();
            AscendC::DataCopyExtParams loadParams{
                1U, static_cast<uint32_t>(rows * TYPE_BYTES), 0U, 0U, 0U};
            AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(srcT, otherGm[this->startRow + rowOffset], loadParams, padParams);
            bcInQueue.EnQue(srcT);

            AscendC::LocalTensor<T> srcDQ = bcInQueue.template DeQue<T>();
            AscendC::LocalTensor<T> dstT  = bcOutQueue.template AllocTensor<T>();
            AscendC::LocalTensor<BT> srcBT = srcDQ.template ReinterpretCast<BT>();
            AscendC::LocalTensor<BT> dstBT = dstT.template ReinterpretCast<BT>();
            AscendC::LocalTensor<uint8_t> tmpU8 = bcTmpBuf.template Get<uint8_t>();
            const uint32_t srcShape[2] = {rows, 1U};
            const uint32_t dstShape[2] = {rows, N};
            AscendC::Broadcast<BT, 2, 1>(dstBT, srcBT, dstShape, srcShape, tmpU8);
            bcOutQueue.EnQue(dstT);
            bcInQueue.FreeTensor(srcDQ);

            AscendC::LocalTensor<T> dstDQ = bcOutQueue.template DeQue<T>();
            const uint32_t outBase = (this->startRow + rowOffset) * N;
            AscendC::DataCopyExtParams stParams{
                1U, static_cast<uint32_t>(rows * rowBytes), 0U, 0U, 0U};
            AscendC::DataCopyPad(inputGm[outBase], dstDQ, stParams);
            bcOutQueue.FreeTensor(dstDQ);

            rowOffset += rows;
        }
    }

    // Single-row Duplicate path for large N (N > BROADCAST_SAFE_N).
    // Per row: 1 GetValue + 1 Duplicate (or chunked Duplicates for very large N) + 1 MTE3.
    __aicore__ inline void ProcessRowByDuplicate(uint32_t N)
    {
        constexpr uint32_t TYPE_BYTES = sizeof(T);
        constexpr uint32_t BLOCK_32 = 32U;
        const uint32_t elemsPer32B = BLOCK_32 / TYPE_BYTES;
        // Max elements per Duplicate chunk: fits in bcOutQueue buffer.
        const uint32_t maxChunkElems = (BC_OUT_BYTES / TYPE_BYTES / elemsPer32B) * elemsPer32B;

        if (this->processRows == 0) return;

        // Bulk-load scalars (up to BC_TMP_BYTES/sizeof(T) at a time).
        const uint32_t maxLoadRows = BC_TMP_BYTES / TYPE_BYTES;
        AscendC::LocalTensor<T> scalarBuf = bcTmpBuf.template Get<T>();

        uint32_t rowOffset = 0;
        while (rowOffset < this->processRows) {
            const uint32_t loadRows = (this->processRows - rowOffset) < maxLoadRows
                ? (this->processRows - rowOffset) : maxLoadRows;

            // MTE2: bulk-load loadRows scalars
            AscendC::DataCopyExtParams loadParams{
                1U, static_cast<uint32_t>(loadRows * TYPE_BYTES), 0U, 0U, 0U};
            AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(scalarBuf, otherGm[this->startRow + rowOffset], loadParams, padParams);
            event_t mte2s = static_cast<event_t>(pipe->FetchEventID(AscendC::HardEvent::MTE2_S));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(mte2s);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(mte2s);

            for (uint32_t i = 0; i < loadRows; ++i) {
                const T val = scalarBuf.GetValue(i);
                event_t sv = static_cast<event_t>(pipe->FetchEventID(AscendC::HardEvent::S_V));
                AscendC::SetFlag<AscendC::HardEvent::S_V>(sv);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(sv);

                // Fill this row in chunks of up to maxChunkElems.
                uint32_t colOffset = 0;
                while (colOffset < N) {
                    uint32_t thisChunk = N - colOffset;
                    if (thisChunk > maxChunkElems) thisChunk = maxChunkElems;

                    AscendC::LocalTensor<T> dstT = bcOutQueue.template AllocTensor<T>();
                    FillLocal(dstT, val, thisChunk);
                    bcOutQueue.EnQue(dstT);

                    AscendC::LocalTensor<T> dstDQ = bcOutQueue.template DeQue<T>();
                    const uint32_t outBase = (this->startRow + rowOffset + i) * N + colOffset;
                    AscendC::DataCopyExtParams stParams{
                        1U, static_cast<uint32_t>(thisChunk * TYPE_BYTES), 0U, 0U, 0U};
                    AscendC::DataCopyPad(inputGm[outBase], dstDQ, stParams);
                    bcOutQueue.FreeTensor(dstDQ);

                    colOffset += thisChunk;
                }
            }
            rowOffset += loadRows;
        }
    }



    __aicore__ inline void ProcessRepeatedRows()
    {
        const uint32_t blockSourceRows = (this->otherLength + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
        const uint32_t startSourceRow = AscendC::GetBlockIdx() * blockSourceRows;
        uint32_t remainingSourceRows = this->otherLength > startSourceRow ? this->otherLength - startSourceRow : 0;
        const uint32_t processSourceRows =
            remainingSourceRows < blockSourceRows ? remainingSourceRows : blockSourceRows;
        for (uint32_t sourceRowOffset = 0; sourceRowOffset < processSourceRows; ++sourceRowOffset) {
            const uint32_t sourceRow = startSourceRow + sourceRowOffset;
            const T value = LoadScalar(sourceRow);
            uint32_t col = 0;
            while (col < this->innerLength) {
                const uint32_t curLength = GetCopyLength(this->innerLength - col);
                const uint32_t maxBatchRows = GetMaxBatchRows(curLength);
                uint32_t repeatOffset = 0;
                while (repeatOffset < this->repeatRows) {
                    uint32_t batchRows = this->repeatRows - repeatOffset;
                    batchRows = batchRows < maxBatchRows ? batchRows : maxBatchRows;
                    const uint32_t outRow = sourceRow + repeatOffset * this->otherLength;
                    CopyScalarRows(value, outRow * this->innerLength + col, curLength, batchRows, this->otherLength);
                    repeatOffset += batchRows;
                }
                col += curLength;
            }
        }
    }

    __aicore__ inline T LoadScalar(uint32_t otherOffset)
    {
        AscendC::LocalTensor<T> scalarLocal = scalarBuf.Get<T>();
        AscendC::DataCopyExtParams scalarCopyParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(scalarLocal, otherGm[otherOffset], scalarCopyParams, padParams);
        event_t eventIdMte2ToS = static_cast<event_t>(pipe->FetchEventID(AscendC::HardEvent::MTE2_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventIdMte2ToS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventIdMte2ToS);
        const T value = scalarLocal.GetValue(0);
        SyncScalarToVector();
        return value;
    }

    __aicore__ inline void SyncScalarToVector()
    {
        event_t eventIdSToV = static_cast<event_t>(pipe->FetchEventID(AscendC::HardEvent::S_V));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIdSToV);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIdSToV);
    }

    __aicore__ inline uint32_t GetCopyLength(uint32_t remaining) const
    {
        constexpr uint32_t alignElems = 32 / sizeof(T);
        uint32_t curLength = remaining < this->tileLength ? remaining : this->tileLength;
        if (curLength > alignElems) {
            const uint32_t alignedLength = curLength / alignElems * alignElems;
            if (alignedLength > 0) {
                return alignedLength;
            }
        }
        return curLength;
    }

    __aicore__ inline uint32_t GetMaxBatchRows(uint32_t length) const
    {
        if (length == 0) {
            return 1;
        }
        uint32_t maxBatchRows = this->tileLength / length;
        if (maxBatchRows == 0) {
            maxBatchRows = 1;
        }
        maxBatchRows = maxBatchRows < 4095U ? maxBatchRows : 4095U;
        return maxBatchRows;
    }

    __aicore__ inline void CopyScalarRows(T value, uint32_t outOffset, uint32_t length, uint32_t batchRows,
                                          uint32_t rowStride)
    {
        AscendC::LocalTensor<T> outputLocal = scalarQueue.AllocTensor<T>();
        constexpr uint32_t alignElems = 32 / sizeof(T);
        const bool canPackRows = length % alignElems == 0;
        FillLocal(outputLocal, value, canPackRows ? length * batchRows : length);
        scalarQueue.EnQue(outputLocal);
        if (canPackRows || batchRows == 1) {
            CopyOutRows(outOffset, length, batchRows, rowStride);
        } else {
            CopyOutSameRows(outOffset, length, batchRows, rowStride);
        }
    }

    __aicore__ inline void FillLocal(const AscendC::LocalTensor<T> &outputLocal, T value, uint32_t length)
    {
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, bool>) {
            const uint8_t byteValue = static_cast<uint8_t>(value);
            const uint32_t fillValue = static_cast<uint32_t>(byteValue) * 0x01010101U;
            AscendC::LocalTensor<uint32_t> outputWords = outputLocal.template ReinterpretCast<uint32_t>();
            AscendC::Duplicate(outputWords, fillValue, (length + sizeof(uint32_t) - 1) / sizeof(uint32_t));
        } else {
            AscendC::Duplicate(outputLocal, value, length);
        }
    }

    __aicore__ inline void CopyOutRows(uint32_t offset, uint32_t length, uint32_t batchRows, uint32_t rowStride)
    {
        AscendC::LocalTensor<T> outputLocal = scalarQueue.DeQue<T>();
        const uint32_t dstStride = (rowStride * this->innerLength - length) * sizeof(T);
        AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(batchRows),
                                              static_cast<uint32_t>(length * sizeof(T)), 0, dstStride, 0};
        AscendC::DataCopyPad(inputGm[offset], outputLocal, copyParams);
        scalarQueue.FreeTensor(outputLocal);
    }

    __aicore__ inline void CopyOutSameRows(uint32_t offset, uint32_t length, uint32_t batchRows, uint32_t rowStride)
    {
        AscendC::LocalTensor<T> outputLocal = scalarQueue.DeQue<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
        const uint32_t rowJump = rowStride * this->innerLength;
        for (uint32_t row = 0; row < batchRows; ++row) {
            AscendC::DataCopyPad(inputGm[offset + row * rowJump], outputLocal, copyParams);
        }
        scalarQueue.FreeTensor(outputLocal);
    }

private:
    AscendC::TPipe *pipe;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> scalarQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scalarBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> otherBuf;
    // AscendC::Broadcast K-batch pipeline (all vector ops, no scalar loop):
    //   bcInQueue (MTE2 → V), bcOutQueue (V → MTE3), bcTmpBuf (Broadcast workspace).
    AscendC::TQue<AscendC::QuePosition::VECIN,  2> bcInQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> bcOutQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC>   bcTmpBuf;
    AscendC::GlobalTensor<T> otherGm;
    AscendC::GlobalTensor<T> inputGm;
    uint32_t tileLength;
    uint32_t innerLength;
    uint32_t otherLength;
    uint32_t rowCount;
    uint32_t repeatRows;
    uint32_t useRowOffset;
    bool willUseBroadcast;
    uint32_t blockRows;
    uint32_t startRow;
    uint32_t processRows;
    uint32_t startOffset;
    uint32_t processLength;
};

template <typename T> class KernelAssignMiddleLastDimScalarBroadcast {
public:
    __aicore__ inline KernelAssignMiddleLastDimScalarBroadcast() {}

    __aicore__ inline void Init(GM_ADDR other, GM_ADDR output, uint32_t totalLength, uint32_t blockLength,
                                uint32_t tileLength, uint32_t innerLength, uint32_t otherLength, uint32_t dimNum,
                                const uint32_t *outShape, const uint32_t *otherShape, AscendC::TPipe *pipeIn)
    {
        this->pipe = pipeIn;
        this->tileLength = tileLength;
        this->innerLength = innerLength;
        this->otherLength = otherLength;
        this->suffixRows = 1;
        this->broadcastRows = 1;
        for (int32_t dim = static_cast<int32_t>(dimNum) - 2; dim >= 0; --dim) {
            if (otherShape[dim] == outShape[dim]) {
                this->suffixRows *= outShape[dim];
                continue;
            }
            for (; dim >= 0; --dim) {
                if (otherShape[dim] == 1 && outShape[dim] != 1) {
                    this->broadcastRows *= outShape[dim];
                } else {
                    break;
                }
            }
            break;
        }
        this->outerSpan = this->suffixRows * this->broadcastRows;
        this->sourceBlockRows = (this->otherLength + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
        this->startSourceRow = AscendC::GetBlockIdx() * this->sourceBlockRows;
        uint32_t remainingSourceRows =
            this->otherLength > this->startSourceRow ? this->otherLength - this->startSourceRow : 0;
        this->processSourceRows =
            remainingSourceRows < this->sourceBlockRows ? remainingSourceRows : this->sourceBlockRows;
        otherGm.SetGlobalBuffer((__gm__ T *)other, this->otherLength);
        inputGm.SetGlobalBuffer((__gm__ T *)output, totalLength);
        pipe->InitBuffer(scalarQueue, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe->InitBuffer(scalarBuf, sizeof(T) * 32);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t sourceRowOffset = 0; sourceRowOffset < this->processSourceRows; ++sourceRowOffset) {
            const uint32_t sourceRow = this->startSourceRow + sourceRowOffset;
            const uint32_t suffixOffset = this->suffixRows == 0 ? 0 : sourceRow % this->suffixRows;
            const uint32_t prefixOffset = this->suffixRows == 0 ? 0 : sourceRow / this->suffixRows;
            const T value = LoadScalar(sourceRow);
            uint32_t col = 0;
            while (col < this->innerLength) {
                const uint32_t curLength = GetCopyLength(this->innerLength - col);
                const uint32_t maxBatch = GetMaxBatchRows(curLength);
                uint32_t broadcastOffset = 0;
                while (broadcastOffset < this->broadcastRows) {
                    uint32_t batchRows = this->broadcastRows - broadcastOffset;
                    batchRows = batchRows < maxBatch ? batchRows : maxBatch;
                    const uint32_t outRow =
                        prefixOffset * this->outerSpan + broadcastOffset * this->suffixRows + suffixOffset;
                    CopyScalarRows(value, outRow * this->innerLength + col, curLength, batchRows);
                    broadcastOffset += batchRows;
                }
                col += curLength;
            }
        }
    }

private:
    __aicore__ inline T LoadScalar(uint32_t otherOffset)
    {
        AscendC::LocalTensor<T> scalarLocal = scalarBuf.Get<T>();
        AscendC::DataCopyExtParams scalarCopyParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(scalarLocal, otherGm[otherOffset], scalarCopyParams, padParams);
        event_t eventIdMte2ToS = static_cast<event_t>(pipe->FetchEventID(AscendC::HardEvent::MTE2_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventIdMte2ToS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventIdMte2ToS);
        const T value = scalarLocal.GetValue(0);
        event_t eventIdSToV = static_cast<event_t>(pipe->FetchEventID(AscendC::HardEvent::S_V));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIdSToV);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIdSToV);
        return value;
    }

    __aicore__ inline uint32_t GetCopyLength(uint32_t remaining) const
    {
        constexpr uint32_t alignElems = 32 / sizeof(T);
        uint32_t curLength = remaining < this->tileLength ? remaining : this->tileLength;
        if (curLength > alignElems) {
            const uint32_t alignedLength = curLength / alignElems * alignElems;
            if (alignedLength > 0) {
                return alignedLength;
            }
        }
        return curLength;
    }

    __aicore__ inline uint32_t GetMaxBatchRows(uint32_t length) const
    {
        if (length == 0) {
            return 1;
        }
        uint32_t maxBatchRows = this->tileLength / length;
        if (maxBatchRows == 0) {
            maxBatchRows = 1;
        }
        maxBatchRows = maxBatchRows < 4095U ? maxBatchRows : 4095U;
        return maxBatchRows;
    }

    __aicore__ inline void CopyScalarRows(T value, uint32_t outOffset, uint32_t length, uint32_t batchRows)
    {
        AscendC::LocalTensor<T> outputLocal = scalarQueue.AllocTensor<T>();
        constexpr uint32_t alignElems = 32 / sizeof(T);
        const bool canPackRows = length % alignElems == 0;
        FillLocal(outputLocal, value, canPackRows ? length * batchRows : length);
        scalarQueue.EnQue(outputLocal);
        if (canPackRows || batchRows == 1) {
            CopyOutRows(outOffset, length, batchRows);
        } else {
            CopyOutSameRows(outOffset, length, batchRows);
        }
    }

    __aicore__ inline void FillLocal(const AscendC::LocalTensor<T> &outputLocal, T value, uint32_t length)
    {
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, bool>) {
            const uint8_t byteValue = static_cast<uint8_t>(value);
            const uint32_t fillValue = static_cast<uint32_t>(byteValue) * 0x01010101U;
            AscendC::LocalTensor<uint32_t> outputWords = outputLocal.template ReinterpretCast<uint32_t>();
            AscendC::Duplicate(outputWords, fillValue, (length + sizeof(uint32_t) - 1) / sizeof(uint32_t));
        } else {
            AscendC::Duplicate(outputLocal, value, length);
        }
    }

    __aicore__ inline void CopyOutRows(uint32_t offset, uint32_t length, uint32_t batchRows)
    {
        AscendC::LocalTensor<T> outputLocal = scalarQueue.DeQue<T>();
        const uint32_t dstStride = (this->suffixRows * this->innerLength - length) * sizeof(T);
        AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(batchRows),
                                              static_cast<uint32_t>(length * sizeof(T)), 0, dstStride, 0};
        AscendC::DataCopyPad(inputGm[offset], outputLocal, copyParams);
        scalarQueue.FreeTensor(outputLocal);
    }

    __aicore__ inline void CopyOutSameRows(uint32_t offset, uint32_t length, uint32_t batchRows)
    {
        AscendC::LocalTensor<T> outputLocal = scalarQueue.DeQue<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
        const uint32_t rowJump = this->suffixRows * this->innerLength;
        for (uint32_t row = 0; row < batchRows; ++row) {
            AscendC::DataCopyPad(inputGm[offset + row * rowJump], outputLocal, copyParams);
        }
        scalarQueue.FreeTensor(outputLocal);
    }

private:
    AscendC::TPipe *pipe;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> scalarQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scalarBuf;
    AscendC::GlobalTensor<T> otherGm;
    AscendC::GlobalTensor<T> inputGm;
    uint32_t tileLength;
    uint32_t innerLength;
    uint32_t otherLength;
    uint32_t suffixRows;
    uint32_t broadcastRows;
    uint32_t outerSpan;
    uint32_t sourceBlockRows;
    uint32_t startSourceRow;
    uint32_t processSourceRows;
};

template <typename T> class KernelAssignContiguous {
public:
    __aicore__ inline KernelAssignContiguous() {}

    __aicore__ inline void Init(GM_ADDR other, GM_ADDR output, uint32_t totalLength, uint32_t blockLength,
                                uint32_t tileLength, AscendC::TPipe *pipeIn)
    {
        this->pipe = pipeIn;
        this->totalLength = totalLength;
        this->blockLength = blockLength;
        this->startOffset = AscendC::GetBlockIdx() * blockLength;
        uint32_t remaining = this->totalLength > this->startOffset ? this->totalLength - this->startOffset : 0;
        this->processLength = remaining < blockLength ? remaining : blockLength;
        // Big tile when this core has more than one host-tile of data.
        const uint32_t bigTile = BIG_TILE_BYTES / sizeof(T);
        this->tileLength = (this->processLength > tileLength && bigTile > tileLength) ? bigTile : tileLength;

        otherGm.SetGlobalBuffer((__gm__ T *)other + this->startOffset, this->processLength);
        inputGm.SetGlobalBuffer((__gm__ T *)output + this->startOffset, this->processLength);
        pipe->InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        uint32_t offset = 0;
        while (offset < this->processLength) {
            uint32_t curLength = this->processLength - offset;
            curLength = curLength < this->tileLength ? curLength : this->tileLength;
            CopyIn(offset, curLength);
            CopyOut(offset, curLength);
            offset += curLength;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length)
    {
        AscendC::LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
        if (length == this->tileLength) {
            AscendC::DataCopy(outputLocal, otherGm[offset], length);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(outputLocal, otherGm[offset], copyParams, padParams);
        }
        outQueue.EnQue(outputLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length)
    {
        AscendC::LocalTensor<T> outputLocal = outQueue.DeQue<T>();
        if (length == this->tileLength) {
            AscendC::DataCopy(inputGm[offset], outputLocal, length);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(inputGm[offset], outputLocal, copyParams);
        }
        outQueue.FreeTensor(outputLocal);
    }

private:
    AscendC::TPipe *pipe;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::GlobalTensor<T> otherGm;
    AscendC::GlobalTensor<T> inputGm;
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t startOffset;
    uint32_t processLength;
};

template <typename T> class KernelAssignBroadcast {
public:
    __aicore__ inline KernelAssignBroadcast() {}

    __aicore__ inline void Init(GM_ADDR other, GM_ADDR output, uint32_t totalLength, uint32_t otherLength,
                                uint32_t tileLength, uint32_t dimNum, const uint32_t *outShape,
                                const uint32_t *otherShape, const uint32_t *otherStride, AscendC::TPipe *pipeIn)
    {
        this->pipe = pipeIn;
        this->totalLength = totalLength;
        this->otherLength = otherLength;
        this->tileLength = tileLength;
        this->dimNum = dimNum;
        for (uint32_t i = 0; i < ASSIGN_MAX_DIMS; ++i) {
            this->outShape[i] = outShape[i];
            this->otherShape[i] = otherShape[i];
            this->otherStride[i] = otherStride[i];
        }

        this->innerLength = this->outShape[this->dimNum - 1];
        this->otherInnerLength = this->otherShape[this->dimNum - 1];
        this->rowCount = this->innerLength == 0 ? 0 : this->totalLength / this->innerLength;
        this->blockRows = (this->rowCount + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
        this->startRow = AscendC::GetBlockIdx() * this->blockRows;
        uint32_t remainingRows = this->rowCount > this->startRow ? this->rowCount - this->startRow : 0;
        this->processRows = remainingRows < this->blockRows ? remainingRows : this->blockRows;

        otherGm.SetGlobalBuffer((__gm__ T *)other, this->otherLength);
        inputGm.SetGlobalBuffer((__gm__ T *)output, this->totalLength);
        pipe->InitBuffer(copyQueue, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe->InitBuffer(scalarQueue, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe->InitBuffer(scalarBuf, sizeof(T) * 32);
        pipe->InitBuffer(otherBuf, this->tileLength * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        // Path A: scalar-row broadcast with cacheable `other` → scalar-row fast path.
        if (this->otherInnerLength == 1 && this->innerLength != 1 && this->otherLength <= this->tileLength) {
            ProcessCachedScalarRows();
            return;
        }
        // Path B: `other` fits in UB → cache it once, then per-row only MTE3.
        // Eliminates per-row MTE2 (which is the main cost for small innerLength).
        if (this->otherLength <= this->tileLength) {
            ProcessCachedContiguousRows();
            return;
        }
        // Path C: fallback — per-row MTE2 + MTE3.
        for (uint32_t row = 0; row < this->processRows; ++row) {
            const uint32_t outRow = this->startRow + row;
            const uint32_t otherBase = CalcOtherBase(outRow);
            const uint32_t outBase = outRow * this->innerLength;
            if (this->otherInnerLength == 1 && this->innerLength != 1) {
                CopyScalarRow(otherBase, outBase);
            } else {
                CopyContiguousRow(otherBase, outBase);
            }
        }
    }

    // Cache the source rows of `other` in UB with each row 32B-aligned, so that
    // per-output-row MTE3 reads from a 32B-aligned UB position.  Eliminates
    // per-row MTE2 (the main cost for small innerLength k3 broadcasts).
    __aicore__ inline void ProcessCachedContiguousRows()
    {
        constexpr uint32_t TYPE_BYTES = sizeof(T);
        constexpr uint32_t BLOCK_32 = 32U;
        const uint32_t rowBytes = this->otherInnerLength * TYPE_BYTES;
        const uint32_t paddedRowBytes = (rowBytes + BLOCK_32 - 1U) / BLOCK_32 * BLOCK_32;
        const uint32_t paddedRowElems = paddedRowBytes / TYPE_BYTES;
        const uint32_t srcRowCount =
            this->otherInnerLength == 0 ? 0 : this->otherLength / this->otherInnerLength;

        // Check the padded layout fits in otherBuf (tileLength elements).
        if (srcRowCount == 0 ||
            srcRowCount * paddedRowElems > this->tileLength) {
            // Doesn't fit — fall back to per-row MTE2 + MTE3.
            for (uint32_t row = 0; row < this->processRows; ++row) {
                const uint32_t outRow = this->startRow + row;
                const uint32_t otherBase = CalcOtherBase(outRow);
                const uint32_t outBase = outRow * this->innerLength;
                CopyContiguousRow(otherBase, outBase);
            }
            return;
        }

        AscendC::LocalTensor<T> cached = otherBuf.Get<T>();
        // Multi-block load: each source row goes to a 32B-aligned UB offset.
        // srcStride=0 (GM bytes, packed source).
        // dstStride=0 (UB datablocks, consecutive 32B-aligned UB positions).
        AscendC::DataCopyExtParams loadParams{
            static_cast<uint16_t>(srcRowCount),
            static_cast<uint32_t>(rowBytes),
            0U, 0U, 0U};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(cached, otherGm, loadParams, padParams);

        // Sync MTE2 → MTE3.
        event_t e = static_cast<event_t>(pipe->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(e);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(e);

        // For each output row: MTE3 innerLength elements from cached[srcRowIdx *
        // paddedRowElems] to GM[outRow * innerLength].  srcRowIdx position is
        // 32B-aligned in UB.
        AscendC::DataCopyExtParams stParams{
            1U, static_cast<uint32_t>(this->innerLength * TYPE_BYTES), 0U, 0U, 0U};
        for (uint32_t row = 0; row < this->processRows; ++row) {
            const uint32_t outRow = this->startRow + row;
            const uint32_t otherBase = CalcOtherBase(outRow);
            const uint32_t srcRowIdx = this->otherInnerLength == 0 ? 0 :
                                       otherBase / this->otherInnerLength;
            AscendC::DataCopyPad(inputGm[outRow * this->innerLength],
                                 cached[srcRowIdx * paddedRowElems], stParams);
        }
    }

private:
    __aicore__ inline uint32_t CalcOtherBase(uint32_t outRow) const
    {
        uint32_t base = 0;
        uint32_t residue = outRow;
        for (int32_t dim = static_cast<int32_t>(this->dimNum) - 2; dim >= 0; --dim) {
            const uint32_t dimSize = this->outShape[dim];
            const uint32_t index = dimSize == 0 ? 0 : residue % dimSize;
            residue = dimSize == 0 ? 0 : residue / dimSize;
            base += index * this->otherStride[dim];
        }
        return base;
    }

    __aicore__ inline void ProcessCachedScalarRows()
    {
        AscendC::LocalTensor<T> otherLocal = otherBuf.Get<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->otherLength * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(otherLocal, otherGm, copyParams, padParams);
        event_t eventIdMte2ToS = static_cast<event_t>(pipe->FetchEventID(AscendC::HardEvent::MTE2_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventIdMte2ToS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventIdMte2ToS);

        for (uint32_t row = 0; row < this->processRows; ++row) {
            const uint32_t outRow = this->startRow + row;
            const uint32_t otherBase = CalcOtherBase(outRow);
            const T value = otherLocal.GetValue(otherBase);
            SyncScalarToVector();
            CopyScalarValue(value, outRow * this->innerLength);
        }
    }

    __aicore__ inline void CopyContiguousRow(uint32_t otherBase, uint32_t outBase)
    {
        uint32_t col = 0;
        while (col < this->innerLength) {
            uint32_t curLength = this->innerLength - col;
            curLength = curLength < this->tileLength ? curLength : this->tileLength;
            CopyIn(otherBase + col, curLength);
            CopyOut(outBase + col, curLength);
            col += curLength;
        }
    }

    __aicore__ inline void CopyScalarRow(uint32_t otherBase, uint32_t outBase)
    {
        AscendC::LocalTensor<T> scalarLocal = scalarBuf.Get<T>();
        AscendC::DataCopyExtParams scalarCopyParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(scalarLocal, otherGm[otherBase], scalarCopyParams, padParams);
        event_t eventIdMte2ToS = static_cast<event_t>(pipe->FetchEventID(AscendC::HardEvent::MTE2_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventIdMte2ToS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventIdMte2ToS);
        const T value = scalarLocal.GetValue(0);
        SyncScalarToVector();
        CopyScalarValue(value, outBase);
    }

    __aicore__ inline void CopyScalarValue(T value, uint32_t outBase)
    {
        uint32_t col = 0;
        while (col < this->innerLength) {
            uint32_t curLength = this->innerLength - col;
            curLength = curLength < this->tileLength ? curLength : this->tileLength;
            AscendC::LocalTensor<T> outputLocal = scalarQueue.AllocTensor<T>();
            FillLocal(outputLocal, value, curLength);
            scalarQueue.EnQue(outputLocal);
            CopyOutScalar(outBase + col, curLength);
            col += curLength;
        }
    }

    __aicore__ inline void SyncScalarToVector()
    {
        event_t eventIdSToV = static_cast<event_t>(pipe->FetchEventID(AscendC::HardEvent::S_V));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIdSToV);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIdSToV);
    }

    __aicore__ inline void FillLocal(const AscendC::LocalTensor<T> &outputLocal, T value, uint32_t length)
    {
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, bool>) {
            const uint8_t byteValue = static_cast<uint8_t>(value);
            const uint32_t fillValue = static_cast<uint32_t>(byteValue) * 0x01010101U;
            AscendC::LocalTensor<uint32_t> outputWords = outputLocal.template ReinterpretCast<uint32_t>();
            AscendC::Duplicate(outputWords, fillValue, (length + sizeof(uint32_t) - 1) / sizeof(uint32_t));
        } else {
            AscendC::Duplicate(outputLocal, value, length);
        }
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length)
    {
        AscendC::LocalTensor<T> outputLocal = copyQueue.AllocTensor<T>();
        if (length == this->tileLength) {
            AscendC::DataCopy(outputLocal, otherGm[offset], length);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(outputLocal, otherGm[offset], copyParams, padParams);
        }
        copyQueue.EnQue(outputLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length)
    {
        AscendC::LocalTensor<T> outputLocal = copyQueue.DeQue<T>();
        if (length == this->tileLength) {
            AscendC::DataCopy(inputGm[offset], outputLocal, length);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(inputGm[offset], outputLocal, copyParams);
        }
        copyQueue.FreeTensor(outputLocal);
    }

    __aicore__ inline void CopyOutScalar(uint32_t offset, uint32_t length)
    {
        AscendC::LocalTensor<T> outputLocal = scalarQueue.DeQue<T>();
        if (length == this->tileLength) {
            AscendC::DataCopy(inputGm[offset], outputLocal, length);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(inputGm[offset], outputLocal, copyParams);
        }
        scalarQueue.FreeTensor(outputLocal);
    }

private:
    AscendC::TPipe *pipe;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, BUFFER_NUM> copyQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> scalarQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scalarBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> otherBuf;
    AscendC::GlobalTensor<T> otherGm;
    AscendC::GlobalTensor<T> inputGm;
    uint32_t totalLength;
    uint32_t otherLength;
    uint32_t tileLength;
    uint32_t dimNum;
    uint32_t outShape[ASSIGN_MAX_DIMS];
    uint32_t otherShape[ASSIGN_MAX_DIMS];
    uint32_t otherStride[ASSIGN_MAX_DIMS];
    uint32_t innerLength;
    uint32_t otherInnerLength;
    uint32_t rowCount;
    uint32_t blockRows;
    uint32_t startRow;
    uint32_t processRows;
};

extern "C" __global__ __aicore__ void assign(GM_ADDR input, GM_ADDR other, GM_ADDR, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(0)) {
        KernelAssignSmall<DTYPE_INPUT> op;
        op.Init(other, input, tiling_data.totalLength);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        AscendC::TPipe pipe;
        KernelAssignContiguous<DTYPE_INPUT> op;
        op.Init(other, input, tiling_data.totalLength, tiling_data.blockLength, tiling_data.tileLength, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        AscendC::TPipe pipe;
        KernelAssignScalarBroadcast<DTYPE_INPUT> op;
        op.Init(other, input, tiling_data.totalLength, tiling_data.blockLength, tiling_data.tileLength, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        AscendC::TPipe pipe;
        KernelAssignBroadcast<DTYPE_INPUT> op;
        op.Init(other, input, tiling_data.totalLength, tiling_data.otherLength, tiling_data.tileLength,
                tiling_data.dimNum, tiling_data.outShape, tiling_data.otherShape, tiling_data.otherStride, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        KernelAssignScalarBroadcastSmall<DTYPE_INPUT> op;
        op.Init(other, input, tiling_data.totalLength);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        AscendC::TPipe pipe;
        KernelAssignRowRepeat<DTYPE_INPUT> op;
        op.Init(other, input, tiling_data.totalLength, tiling_data.blockLength, tiling_data.tileLength,
                tiling_data.outShape[tiling_data.dimNum - 1], &pipe);
        op.Process();
    } else if (TILING_KEY_IS(7)) {
        AscendC::TPipe pipe;
        KernelAssignBlockRepeat<DTYPE_INPUT> op;
        op.Init(other, input, tiling_data.totalLength, tiling_data.blockLength, tiling_data.tileLength,
                tiling_data.otherLength, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(6)) {
        return;  // POLLUTED-6
        AscendC::TPipe pipe;
        KernelAssignLastDimScalarBroadcast<DTYPE_INPUT> op;
        op.Init(other, input, tiling_data.totalLength, tiling_data.blockLength, tiling_data.tileLength,
                tiling_data.outShape[tiling_data.dimNum - 1], tiling_data.otherLength, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(8)) {
        AscendC::TPipe pipe;
        KernelAssignMiddleLastDimScalarBroadcast<DTYPE_INPUT> op;
        op.Init(other, input, tiling_data.totalLength, tiling_data.blockLength, tiling_data.tileLength,
                tiling_data.outShape[tiling_data.dimNum - 1], tiling_data.otherLength, tiling_data.dimNum,
                tiling_data.outShape, tiling_data.otherShape, &pipe);
        op.Process();
    } else {
        return;
    }
}
