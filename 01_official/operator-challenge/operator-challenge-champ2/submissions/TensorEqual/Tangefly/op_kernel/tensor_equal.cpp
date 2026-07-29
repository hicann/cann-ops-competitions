#include "kernel_inc.h"

constexpr uint32_t CMP_ALIGN_BYTES = 256;
constexpr uint32_t DATA_COPY_ALIGN_BYTES = 32;
constexpr uint32_t CMP_MASK_ELEMS_PER_BYTE = 8;
constexpr uint32_t BATCH_SIZE = 256;
constexpr uint32_t CONTIG_BUFFER_NUM = 2;

template<typename DATA, class TS>
class TensorEqualKernelCommon {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const TS& tilingData, TPipe* pipe)
    {
        x1Addr = x1;
        x2Addr = x2;
        ti = tilingData;
        this->pipe = pipe;
        x1Gm.SetGlobalBuffer((__gm__ DATA*)x1);
        x2Gm.SetGlobalBuffer((__gm__ DATA*)x2);
        yGm.SetGlobalBuffer((__gm__ uint8_t*)y);
    }

protected:
    static constexpr bool NEED_FP16_WORK = is_same_v<DATA, int8_t> || is_same_v<DATA, uint8_t>;
    static constexpr bool NEED_FP32_WORK = is_same_v<DATA, bfloat16_t> || is_same_v<DATA, int16_t>;

    __aicore__ inline uint32_t GetCmpAlignedCount(uint32_t count) const
    {
        constexpr uint32_t ALIGN_ELEMS = CMP_ALIGN_BYTES / sizeof(DATA);
        return CEIL(count, ALIGN_ELEMS);
    }

    __aicore__ inline uint32_t GetSmallInnerAlignedCount() const
    {
        return ti.tk6AlignedInnerLength;
    }

    __aicore__ inline uint32_t GetCmpRepeatLength() const
    {
        return ti.tk6CmpRepeatLength;
    }

    __aicore__ inline bool IsDataCopyAligned(uint32_t count) const
    {
        return (count * sizeof(DATA)) % DATA_COPY_ALIGN_BYTES == 0U;
    }

    __aicore__ inline bool IsByteCopyAligned(uint32_t count) const
    {
        return count % DATA_COPY_ALIGN_BYTES == 0U;
    }

    __aicore__ inline void CompareEqual(LocalTensor<uint8_t> yLocal,
                                        LocalTensor<DATA> x1Local,
                                        LocalTensor<DATA> x2Local,
                                        LocalTensor<half> x1Fp16,
                                        LocalTensor<half> x2Fp16,
                                        LocalTensor<float> x1Fp32,
                                        LocalTensor<float> x2Fp32,
                                        uint32_t count) const
    {
        if constexpr (is_same_v<DATA, half> || is_same_v<DATA, float>) {
            Compare(yLocal, x1Local, x2Local, CMPMODE::EQ, count);
        } else if constexpr (is_same_v<DATA, int32_t> || is_same_v<DATA, int16_t>) {
            Sub(x1Local, x1Local, x2Local, count);
            CompareIntegerDiffEqual(yLocal, x1Local, x1Fp32, count);
        } else if constexpr (is_same_v<DATA, int8_t> || is_same_v<DATA, uint8_t>) {
            Cast(x1Fp16, x1Local, RoundMode::CAST_NONE, count);
            Cast(x2Fp16, x2Local, RoundMode::CAST_NONE, count);
            Compare(yLocal, x1Fp16, x2Fp16, CMPMODE::EQ, count);
        } else {
            Cast(x1Fp32, x1Local, RoundMode::CAST_NONE, count);
            Cast(x2Fp32, x2Local, RoundMode::CAST_NONE, count);
            Compare(yLocal, x1Fp32, x2Fp32, CMPMODE::EQ, count);
        }
    }

    __aicore__ inline void CompareIntegerDiffEqual(LocalTensor<uint8_t> yLocal,
                                                   LocalTensor<DATA> diffLocal,
                                                   LocalTensor<float> diffFp32,
                                                   uint32_t count) const
    {
        if constexpr (is_same_v<DATA, int32_t>) {
            CompareScalar(yLocal, diffLocal, static_cast<int32_t>(0), CMPMODE::EQ, count);
        } else if constexpr (is_same_v<DATA, int16_t>) {
            Cast(diffFp32, diffLocal, RoundMode::CAST_NONE, count);
            CompareScalar(yLocal, diffFp32, 0.0f, CMPMODE::EQ, count);
        }
    }

    __aicore__ inline void CompareEqualScalar(LocalTensor<uint8_t> yLocal,
                                              LocalTensor<DATA> xLocal,
                                              GM_ADDR scalarAddr,
                                              uint32_t scalarIndex,
                                              LocalTensor<half> xFp16,
                                              LocalTensor<float> xFp32,
                                              uint32_t count) const
    {
        if constexpr (is_same_v<DATA, half>) {
            const half scalar = *(CAST(__gm__ half*, scalarAddr) + scalarIndex);
            CompareScalar(yLocal, xLocal, scalar, CMPMODE::EQ, count);
        } else if constexpr (is_same_v<DATA, float>) {
            const float scalar = *(CAST(__gm__ float*, scalarAddr) + scalarIndex);
            CompareScalar(yLocal, xLocal, scalar, CMPMODE::EQ, count);
        } else if constexpr (is_same_v<DATA, int8_t>) {
            Cast(xFp16, xLocal, RoundMode::CAST_NONE, count);
            const int8_t scalarValue = *(CAST(__gm__ int8_t*, scalarAddr) + scalarIndex);
            const int32_t scalarI32 = static_cast<int32_t>(scalarValue);
            CompareScalar(yLocal, xFp16, half(static_cast<float>(scalarI32)), CMPMODE::EQ, count);
        } else if constexpr (is_same_v<DATA, uint8_t>) {
            Cast(xFp16, xLocal, RoundMode::CAST_NONE, count);
            const uint8_t scalarValue = *(CAST(__gm__ uint8_t*, scalarAddr) + scalarIndex);
            const int32_t scalarI32 = static_cast<int32_t>(scalarValue);
            CompareScalar(yLocal, xFp16, half(static_cast<float>(scalarI32)), CMPMODE::EQ, count);
        } else if constexpr (is_same_v<DATA, int16_t>) {
            Cast(xFp32, xLocal, RoundMode::CAST_NONE, count);
            const int16_t scalarValue = *(CAST(__gm__ int16_t*, scalarAddr) + scalarIndex);
            const int32_t scalarI32 = static_cast<int32_t>(scalarValue);
            CompareScalar(yLocal, xFp32, static_cast<float>(scalarI32), CMPMODE::EQ, count);
        } else if constexpr (is_same_v<DATA, int32_t>) {
            const int32_t scalarValue = *(CAST(__gm__ int32_t*, scalarAddr) + scalarIndex);
            CompareScalar(yLocal, xLocal, scalarValue, CMPMODE::EQ, count);
        } else {
            Cast(xFp32, xLocal, RoundMode::CAST_NONE, count);
            const int32_t scalarValue = *(CAST(__gm__ int32_t*, scalarAddr) + scalarIndex);
            CompareScalar(yLocal, xFp32, static_cast<float>(scalarValue), CMPMODE::EQ, count);
        }
    }

    __aicore__ inline void InitSelectConstants(LocalTensor<half> oneLocal,
                                               LocalTensor<half> zeroLocal,
                                               uint32_t count) const
    {
        Duplicate(oneLocal, half(1.0), count);
        Duplicate(zeroLocal, half(0.0), count);
    }

    __aicore__ inline void ExpandCompareMaskBySelect(LocalTensor<uint8_t> yLocal,
                                                     LocalTensor<uint8_t> cmpLocal,
                                                     LocalTensor<half> oneLocal,
                                                     LocalTensor<half> zeroLocal,
                                                     LocalTensor<half> selectedLocal,
                                                     uint32_t count) const
    {
        Select(selectedLocal, cmpLocal, oneLocal, zeroLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        Cast(yLocal, selectedLocal, RoundMode::CAST_RINT, count);
    }

    __aicore__ inline void DuplicateZero(LocalTensor<DATA> local, uint32_t length) const
    {
        if constexpr (is_same_v<DATA, bfloat16_t>) {
            LocalTensor<uint16_t> localU16 = local.template ReinterpretCast<uint16_t>();
            Duplicate<uint16_t>(localU16, 0U, length);
        } else if constexpr (is_same_v<DATA, int8_t> || is_same_v<DATA, uint8_t>) {
            LocalTensor<uint16_t> localU16 = local.template ReinterpretCast<uint16_t>();
            Duplicate<uint16_t>(localU16, 0U, (length + 1U) >> 1U);
        } else {
            Duplicate<DATA>(local, static_cast<DATA>(0), length);
        }
    }

    __aicore__ inline void DuplicateGmValue(LocalTensor<DATA> local,
                                            GM_ADDR addr,
                                            uint32_t gmIndex,
                                            uint32_t length) const
    {
        if constexpr (is_same_v<DATA, bfloat16_t>) {
            LocalTensor<uint16_t> localU16 = local.template ReinterpretCast<uint16_t>();
            const uint16_t value = *(CAST(__gm__ uint16_t*, addr) + gmIndex);
            Duplicate<uint16_t>(localU16, value, length);
        } else if constexpr (is_same_v<DATA, int8_t> || is_same_v<DATA, uint8_t>) {
            LocalTensor<uint16_t> localU16 = local.template ReinterpretCast<uint16_t>();
            const uint16_t valueByte = *(CAST(__gm__ uint8_t*, addr) + gmIndex);
            const uint16_t valuePack = valueByte | (valueByte << 8U);
            Duplicate<uint16_t>(localU16, valuePack, (length + 1U) >> 1U);
        } else {
            const DATA value = *(CAST(__gm__ DATA*, addr) + gmIndex);
            Duplicate<DATA>(local, value, length);
        }
    }
    GM_ADDR x1Addr;
    GM_ADDR x2Addr;
    TS ti;
    TPipe* pipe;
    GlobalTensor<DATA> x1Gm;
    GlobalTensor<DATA> x2Gm;
    GlobalTensor<uint8_t> yGm;

    TBuf<TPosition::VECCALC> x1Buf;
    TBuf<TPosition::VECCALC> x2Buf;
    TBuf<TPosition::VECCALC> x1Fp16Buf;
    TBuf<TPosition::VECCALC> x2Fp16Buf;
    TBuf<TPosition::VECCALC> x1Fp32Buf;
    TBuf<TPosition::VECCALC> x2Fp32Buf;
    TBuf<TPosition::VECCALC> cmpBuf;
    TBuf<TPosition::VECCALC> oneBuf;
    TBuf<TPosition::VECCALC> zeroBuf;
    TBuf<TPosition::VECCALC> selectedBuf;
    TBuf<TPosition::VECCALC> yBuf;

    TQue<QuePosition::VECIN, CONTIG_BUFFER_NUM> qX1;
    TQue<QuePosition::VECIN, CONTIG_BUFFER_NUM> qX2;
    TQue<QuePosition::VECOUT, CONTIG_BUFFER_NUM> qY;
    TBuf<TPosition::VECCALC> contigX1Fp16Buf;
    TBuf<TPosition::VECCALC> contigX2Fp16Buf;
    TBuf<TPosition::VECCALC> contigX1Fp32Buf;
    TBuf<TPosition::VECCALC> contigX2Fp32Buf;
    TBuf<TPosition::VECCALC> contigCmpBuf;
    TBuf<TPosition::VECCALC> contigOneBuf;
    TBuf<TPosition::VECCALC> contigZeroBuf;
    TBuf<TPosition::VECCALC> contigSelectedBuf;
    bool currentUseScalarCompare = false;
    bool currentX1Contiguous = true;
    bool currentX2Contiguous = true;
    uint32_t currentX1Offset = 0;
    uint32_t currentX2Offset = 0;

    __aicore__ inline uint32_t GetX1Index(uint32_t yIdx) const
    {
        uint32_t x1Idx = 0;
        for (uint32_t axis = 0; axis < ti.shapeSize; ++axis) {
            const uint32_t coord = yIdx / ti.yShapeRSum[axis] % ti.yShape[axis];
            x1Idx += (ti.x1Shape[axis] == 1U) ? 0U : ti.x1ShapeRSum[axis] * coord;
        }
        return x1Idx;
    }

    __aicore__ inline uint32_t GetX2Index(uint32_t yIdx) const
    {
        uint32_t x2Idx = 0;
        for (uint32_t axis = 0; axis < ti.shapeSize; ++axis) {
            const uint32_t coord = yIdx / ti.yShapeRSum[axis] % ti.yShape[axis];
            x2Idx += (ti.x2Shape[axis] == 1U) ? 0U : ti.x2ShapeRSum[axis] * coord;
        }
        return x2Idx;
    }

    __aicore__ inline bool IsFullShape(const uint32_t* shape) const
    {
        for (uint32_t axis = 0; axis < ti.shapeSize; ++axis) {
            if (shape[axis] != ti.yShape[axis]) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline bool IsGlobalScalar(const uint32_t* shape) const
    {
        for (uint32_t axis = 0; axis < ti.shapeSize; ++axis) {
            if (shape[axis] != 1U) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline uint32_t GetContiguousLimit(const uint32_t* shape, uint32_t yOffset) const
    {
        uint32_t limit = ti.dataLength - yOffset;
        for (uint32_t axis = 0; axis < ti.shapeSize; ++axis) {
            if (shape[axis] == 1U && ti.yShape[axis] != 1U) {
                const uint32_t span = ti.yShapeRSum[axis];
                const uint32_t remain = span - (yOffset % span);
                limit = remain < limit ? remain : limit;
            }
        }
        return limit;
    }

    __aicore__ inline uint32_t GetRowRepeatLimit(const uint32_t* shape, uint32_t yOffset) const
    {
        uint32_t limit = ti.dataLength - yOffset;
        for (uint32_t axis = 1; axis < ti.shapeSize; ++axis) {
            if (shape[axis] != 1U && ti.yShape[axis] != 1U) {
                const uint32_t span = ti.yShapeRSum[axis];
                const uint32_t remain = span - (yOffset % span);
                limit = remain < limit ? remain : limit;
            }
        }
        return limit;
    }

    __aicore__ inline uint32_t Min3(uint32_t a, uint32_t b, uint32_t c) const
    {
        uint32_t result = a < b ? a : b;
        return result < c ? result : c;
    }

    __aicore__ inline void InitGenericBuffers(uint32_t batchLength)
    {
        pipe->InitBuffer(x1Buf, batchLength * sizeof(DATA));
        pipe->InitBuffer(x2Buf, batchLength * sizeof(DATA));
        if constexpr (NEED_FP16_WORK) {
            pipe->InitBuffer(x1Fp16Buf, batchLength * sizeof(half));
            pipe->InitBuffer(x2Fp16Buf, batchLength * sizeof(half));
        }
        if constexpr (NEED_FP32_WORK) {
            pipe->InitBuffer(x1Fp32Buf, batchLength * sizeof(float));
            pipe->InitBuffer(x2Fp32Buf, batchLength * sizeof(float));
        }
        pipe->InitBuffer(cmpBuf, batchLength * sizeof(uint8_t));
        pipe->InitBuffer(oneBuf, batchLength * sizeof(half));
        pipe->InitBuffer(zeroBuf, batchLength * sizeof(half));
        pipe->InitBuffer(selectedBuf, batchLength * sizeof(half));
        pipe->InitBuffer(yBuf, batchLength * sizeof(uint8_t));
    }

    __aicore__ inline void ProcessGeneric(uint32_t batchLength)
    {
        LocalTensor<DATA> x1Local = x1Buf.template Get<DATA>();
        LocalTensor<DATA> x2Local = x2Buf.template Get<DATA>();
        LocalTensor<half> x1Fp16;
        LocalTensor<half> x2Fp16;
        LocalTensor<float> x1Fp32;
        LocalTensor<float> x2Fp32;
        LocalTensor<uint8_t> cmpLocal = cmpBuf.template Get<uint8_t>();
        LocalTensor<half> oneLocal = oneBuf.template Get<half>();
        LocalTensor<half> zeroLocal = zeroBuf.template Get<half>();
        LocalTensor<half> selectedLocal = selectedBuf.template Get<half>();
        LocalTensor<uint8_t> yLocal = yBuf.template Get<uint8_t>();
        if constexpr (NEED_FP16_WORK) {
            x1Fp16 = x1Fp16Buf.template Get<half>();
            x2Fp16 = x2Fp16Buf.template Get<half>();
        }
        if constexpr (NEED_FP32_WORK) {
            x1Fp32 = x1Fp32Buf.template Get<float>();
            x2Fp32 = x2Fp32Buf.template Get<float>();
        }

        this->InitSelectConstants(oneLocal, zeroLocal, batchLength);
        for (uint32_t base = 0; base < ti.dataLength; base += batchLength) {
            const uint32_t validCount = ((ti.dataLength - base) < batchLength) ? (ti.dataLength - base) : batchLength;
            const uint32_t cmpCount = this->GetCmpAlignedCount(validCount);

            for (uint32_t localIdx = 0; localIdx < cmpCount; ++localIdx) {
                const uint32_t yIdx = base + ((localIdx < validCount) ? localIdx : 0U);
                x1Local.SetValue(localIdx, *(CAST(__gm__ DATA*, x1Addr) + GetX1Index(yIdx)));
                x2Local.SetValue(localIdx, *(CAST(__gm__ DATA*, x2Addr) + GetX2Index(yIdx)));
            }

            this->CompareEqual(cmpLocal, x1Local, x2Local, x1Fp16, x2Fp16, x1Fp32, x2Fp32, cmpCount);
            this->ExpandCompareMaskBySelect(yLocal, cmpLocal, oneLocal, zeroLocal, selectedLocal, cmpCount);
            DataCopyPad(yGm[base], yLocal, {1, static_cast<uint32_t>(validCount * sizeof(uint8_t)), 0, 0, 0});
        }
    }

    __aicore__ inline void InitContiguousBuffers()
    {
        pipe->InitBuffer(qX1, CONTIG_BUFFER_NUM, ti.tileLength * sizeof(DATA));
        pipe->InitBuffer(qX2, CONTIG_BUFFER_NUM, ti.tileLength * sizeof(DATA));
        pipe->InitBuffer(qY, CONTIG_BUFFER_NUM, ti.tileLength * sizeof(uint8_t));
        if constexpr (NEED_FP16_WORK) {
            pipe->InitBuffer(contigX1Fp16Buf, ti.tileLength * sizeof(half));
            pipe->InitBuffer(contigX2Fp16Buf, ti.tileLength * sizeof(half));
        }
        if constexpr (NEED_FP32_WORK) {
            pipe->InitBuffer(contigX1Fp32Buf, ti.tileLength * sizeof(float));
            pipe->InitBuffer(contigX2Fp32Buf, ti.tileLength * sizeof(float));
        }
        pipe->InitBuffer(contigCmpBuf, ti.tileLength * sizeof(uint8_t));
        pipe->InitBuffer(contigOneBuf, ti.tileLength * sizeof(half));
        pipe->InitBuffer(contigZeroBuf, ti.tileLength * sizeof(half));
        pipe->InitBuffer(contigSelectedBuf, ti.tileLength * sizeof(half));
    }

    __aicore__ inline void InitContiguousSelectConstants()
    {
        LocalTensor<half> oneLocal = contigOneBuf.template Get<half>();
        LocalTensor<half> zeroLocal = contigZeroBuf.template Get<half>();
        this->InitSelectConstants(oneLocal, zeroLocal, ti.tileLength);
    }

    __aicore__ inline void CopyInOne(LocalTensor<DATA>& local,
                                     GlobalTensor<DATA>& gm,
                                     GM_ADDR addr,
                                     uint32_t gmOffset,
                                     bool contiguous,
                                     bool needLocal,
                                     uint32_t validLength,
                                     uint32_t calcLength)
    {
        if (!needLocal) {
            return;
        }
        if (!contiguous) {
            this->DuplicateGmValue(local, addr, gmOffset, calcLength);
            return;
        }
        if (validLength != calcLength) {
            this->DuplicateZero(local, calcLength);
            PipeBarrier<PIPE_ALL>();
        }
        if (validLength == calcLength && this->IsDataCopyAligned(validLength)) {
            DataCopy(local, gm[gmOffset], validLength);
        } else {
            const DataCopyParams copyParams{1, static_cast<uint16_t>(validLength * sizeof(DATA)), 0, 0};
            const DataCopyPadParams padParams{false, 0, 0, 0};
            DataCopyPad(local, gm[gmOffset], copyParams, padParams);
        }
    }

    __aicore__ inline void CopyInDirect(uint32_t x1Offset,
                                        uint32_t x2Offset,
                                        bool x1Contiguous,
                                        bool x2Contiguous,
                                        uint32_t validLength,
                                        uint32_t calcLength)
    {
        currentX1Offset = x1Offset;
        currentX2Offset = x2Offset;
        currentX1Contiguous = x1Contiguous;
        currentX2Contiguous = x2Contiguous;
        constexpr bool SUPPORTS_SCALAR_COMPARE = !is_same_v<DATA, bfloat16_t>;
        currentUseScalarCompare = SUPPORTS_SCALAR_COMPARE && (x1Contiguous != x2Contiguous);
        const bool needX1Local = x1Contiguous || !currentUseScalarCompare;
        const bool needX2Local = x2Contiguous || !currentUseScalarCompare;

        if (currentUseScalarCompare) {
            if (needX1Local) {
                LocalTensor<DATA> x1Local = qX1.template AllocTensor<DATA>();
                CopyInOne(x1Local, x1Gm, x1Addr, x1Offset, x1Contiguous, true, validLength, calcLength);
                qX1.template EnQue<DATA>(x1Local);
            } else {
                LocalTensor<DATA> x2Local = qX2.template AllocTensor<DATA>();
                CopyInOne(x2Local, x2Gm, x2Addr, x2Offset, x2Contiguous, true, validLength, calcLength);
                qX2.template EnQue<DATA>(x2Local);
            }
            return;
        }

        LocalTensor<DATA> x1Local = qX1.template AllocTensor<DATA>();
        LocalTensor<DATA> x2Local = qX2.template AllocTensor<DATA>();
        CopyInOne(x1Local, x1Gm, x1Addr, x1Offset, x1Contiguous, needX1Local, validLength, calcLength);
        CopyInOne(x2Local, x2Gm, x2Addr, x2Offset, x2Contiguous, needX2Local, validLength, calcLength);
        qX1.template EnQue<DATA>(x1Local);
        qX2.template EnQue<DATA>(x2Local);
    }

    __aicore__ inline void ComputeContiguous(uint32_t calcLength)
    {
        LocalTensor<uint8_t> yLocal = qY.template AllocTensor<uint8_t>();
        LocalTensor<uint8_t> cmpLocal = contigCmpBuf.template Get<uint8_t>();
        LocalTensor<half> x1Fp16;
        LocalTensor<half> x2Fp16;
        LocalTensor<float> x1Fp32;
        LocalTensor<float> x2Fp32;
        LocalTensor<half> oneLocal = contigOneBuf.template Get<half>();
        LocalTensor<half> zeroLocal = contigZeroBuf.template Get<half>();
        LocalTensor<half> selectedLocal = contigSelectedBuf.template Get<half>();
        if constexpr (NEED_FP16_WORK) {
            x1Fp16 = contigX1Fp16Buf.template Get<half>();
            x2Fp16 = contigX2Fp16Buf.template Get<half>();
        }
        if constexpr (NEED_FP32_WORK) {
            x1Fp32 = contigX1Fp32Buf.template Get<float>();
            x2Fp32 = contigX2Fp32Buf.template Get<float>();
        }

        if constexpr (!is_same_v<DATA, bfloat16_t>) {
            if (currentUseScalarCompare) {
                if (currentX1Contiguous) {
                    LocalTensor<DATA> x1Local = qX1.template DeQue<DATA>();
                    this->CompareEqualScalar(cmpLocal, x1Local, x2Addr, currentX2Offset, x1Fp16, x1Fp32, calcLength);
                    qX1.FreeTensor(x1Local);
                } else {
                    LocalTensor<DATA> x2Local = qX2.template DeQue<DATA>();
                    this->CompareEqualScalar(cmpLocal, x2Local, x1Addr, currentX1Offset, x2Fp16, x2Fp32, calcLength);
                    qX2.FreeTensor(x2Local);
                }
                this->ExpandCompareMaskBySelect(yLocal, cmpLocal, oneLocal, zeroLocal, selectedLocal, calcLength);
                qY.template EnQue<uint8_t>(yLocal);
                return;
            } else {
                LocalTensor<DATA> x1Local = qX1.template DeQue<DATA>();
                LocalTensor<DATA> x2Local = qX2.template DeQue<DATA>();
                this->CompareEqual(cmpLocal, x1Local, x2Local, x1Fp16, x2Fp16, x1Fp32, x2Fp32, calcLength);
                qX1.FreeTensor(x1Local);
                qX2.FreeTensor(x2Local);
            }
        } else {
            LocalTensor<DATA> x1Local = qX1.template DeQue<DATA>();
            LocalTensor<DATA> x2Local = qX2.template DeQue<DATA>();
            this->CompareEqual(cmpLocal, x1Local, x2Local, x1Fp16, x2Fp16, x1Fp32, x2Fp32, calcLength);
            qX1.FreeTensor(x1Local);
            qX2.FreeTensor(x2Local);
        }
        this->ExpandCompareMaskBySelect(yLocal, cmpLocal, oneLocal, zeroLocal, selectedLocal, calcLength);
        qY.template EnQue<uint8_t>(yLocal);
    }

    __aicore__ inline void CopyOutContiguous(uint32_t yOffset, uint32_t validLength)
    {
        LocalTensor<uint8_t> yLocal = qY.template DeQue<uint8_t>();
        if (this->IsByteCopyAligned(validLength)) {
            DataCopy(yGm[yOffset], yLocal, validLength);
        } else {
            DataCopyPad(yGm[yOffset], yLocal, {1, static_cast<uint32_t>(validLength * sizeof(uint8_t)), 0, 0, 0});
        }
        qY.FreeTensor(yLocal);
    }

    __aicore__ inline void ProcessDirectTail()
    {
        const uint32_t innerLength = ti.yShape[0];
        const uint32_t outerLength = ti.dataLength / innerLength;
        const bool x1Contiguous = ti.x1Shape[0] == ti.yShape[0];
        const bool x2Contiguous = ti.x2Shape[0] == ti.yShape[0];
        for (uint32_t outer = 0; outer < outerLength; ++outer) {
            const uint32_t yBase = outer * innerLength;
            const uint32_t x1Base = GetX1Index(yBase);
            const uint32_t x2Base = GetX2Index(yBase);
            for (uint32_t inner = 0; inner < innerLength; inner += ti.tileLength) {
                const uint32_t validLength = ((innerLength - inner) < ti.tileLength) ? (innerLength - inner) : ti.tileLength;
                const uint32_t calcLength = this->GetCmpAlignedCount(validLength);
                const uint32_t x1Offset = x1Contiguous ? (x1Base + inner) : x1Base;
                const uint32_t x2Offset = x2Contiguous ? (x2Base + inner) : x2Base;
                CopyInDirect(x1Offset, x2Offset, x1Contiguous, x2Contiguous, validLength, calcLength);
                ComputeContiguous(calcLength);
                CopyOutContiguous(yBase + inner, validLength);
            }
        }
    }
};

template<typename DATA, class TS>
class TensorEqualGenericKernel : public TensorEqualKernelCommon<DATA, TS> {
public:
    __aicore__ inline void Process()
    {
        if (this->ti.dataLength == 0U) {
            return;
        }
        this->InitGenericBuffers(BATCH_SIZE);
        this->ProcessGeneric(BATCH_SIZE);
    }
};

template<typename DATA, class TS>
class TensorEqualLinearKernel : public TensorEqualKernelCommon<DATA, TS> {
public:
    __aicore__ inline void Process()
    {
        if (this->ti.dataLength == 0U) {
            return;
        }
        this->InitContiguousBuffers();
        this->InitContiguousSelectConstants();
        for (uint32_t base = 0; base < this->ti.dataLength; base += this->ti.tileLength) {
            const uint32_t validLength = ((this->ti.dataLength - base) < this->ti.tileLength) ?
                                         (this->ti.dataLength - base) : this->ti.tileLength;
            const uint32_t calcLength = this->GetCmpAlignedCount(validLength);
            this->CopyInDirect(base, base, true, true, validLength, calcLength);
            this->ComputeContiguous(calcLength);
            this->CopyOutContiguous(base, validLength);
        }
    }
};

template<typename DATA, class TS>
class TensorEqualBroadcastKernel : public TensorEqualKernelCommon<DATA, TS> {
public:
    __aicore__ inline void Process()
    {
        if (this->ti.dataLength == 0U) {
            return;
        }
        if (this->ti.shapeSize > 0U) {
            this->InitContiguousBuffers();
            this->InitContiguousSelectConstants();
            this->ProcessDirectTail();
            return;
        }
        this->InitGenericBuffers(BATCH_SIZE);
        this->ProcessGeneric(BATCH_SIZE);
    }
};

template<typename DATA, class TS>
class TensorEqualGlobalScalarKernel : public TensorEqualKernelCommon<DATA, TS> {
public:
    __aicore__ inline void Process()
    {
        if (this->ti.dataLength == 0U) {
            return;
        }
        this->InitContiguousBuffers();
        this->InitContiguousSelectConstants();
        const bool x1Scalar = this->IsGlobalScalar(this->ti.x1Shape);
        for (uint32_t base = 0; base < this->ti.dataLength; base += this->ti.tileLength) {
            const uint32_t validLength = ((this->ti.dataLength - base) < this->ti.tileLength) ?
                                         (this->ti.dataLength - base) : this->ti.tileLength;
            const uint32_t calcLength = this->GetCmpAlignedCount(validLength);
            if (x1Scalar) {
                this->CopyInDirect(0U, base, false, true, validLength, calcLength);
            } else {
                this->CopyInDirect(base, 0U, true, false, validLength, calcLength);
            }
            this->ComputeContiguous(calcLength);
            this->CopyOutContiguous(base, validLength);
        }
    }
};

template<typename DATA, class TS>
class TensorEqualLastDimScalarKernel : public TensorEqualKernelCommon<DATA, TS> {
public:
    __aicore__ inline void Process()
    {
        if (this->ti.dataLength == 0U) {
            return;
        }
        this->InitContiguousBuffers();
        this->InitContiguousSelectConstants();

        const uint32_t innerLength = this->ti.innerLength;
        const uint32_t outerLength = this->ti.dataLength / innerLength;
        const uint32_t rowsPerTile = this->ti.tileLength / innerLength;
        const bool x1Contiguous = this->IsFullShape(this->ti.x1Shape) && this->ti.x2Shape[0] == 1U;
        const bool x2Contiguous = this->IsFullShape(this->ti.x2Shape) && this->ti.x1Shape[0] == 1U;
        if (rowsPerTile <= 1U || (!x1Contiguous && !x2Contiguous) ||
            this->GetCmpAlignedCount(innerLength) != innerLength) {
            this->ProcessDirectTail();
            return;
        }

        for (uint32_t outer = 0; outer < outerLength; outer += rowsPerTile) {
            const uint32_t rowsThis = ((outerLength - outer) < rowsPerTile) ? (outerLength - outer) : rowsPerTile;
            const uint32_t yBase = outer * innerLength;
            const uint32_t batchLength = rowsThis * innerLength;
            CopyInContiguousBatch(yBase, batchLength, x1Contiguous);
            ComputeLastDimScalarBatch(outer, rowsThis, innerLength, x1Contiguous);
            this->CopyOutContiguous(yBase, batchLength);
        }
    }

private:
    __aicore__ inline void CopyInContiguousBatch(uint32_t gmOffset, uint32_t batchLength, bool useX1)
    {
        if (useX1) {
            LocalTensor<DATA> x1Local = this->qX1.template AllocTensor<DATA>();
            this->CopyInOne(x1Local, this->x1Gm, this->x1Addr, gmOffset, true, true, batchLength, batchLength);
            this->qX1.template EnQue<DATA>(x1Local);
        } else {
            LocalTensor<DATA> x2Local = this->qX2.template AllocTensor<DATA>();
            this->CopyInOne(x2Local, this->x2Gm, this->x2Addr, gmOffset, true, true, batchLength, batchLength);
            this->qX2.template EnQue<DATA>(x2Local);
        }
    }

    __aicore__ inline void ComputeLastDimScalarBatch(uint32_t outerBase,
                                                     uint32_t rowsThis,
                                                     uint32_t innerLength,
                                                     bool x1Contiguous)
    {
        LocalTensor<uint8_t> yLocal = this->qY.template AllocTensor<uint8_t>();
        LocalTensor<uint8_t> cmpLocal = this->contigCmpBuf.template Get<uint8_t>();
        LocalTensor<half> x1Fp16;
        LocalTensor<half> x2Fp16;
        LocalTensor<float> x1Fp32;
        LocalTensor<float> x2Fp32;
        LocalTensor<half> oneLocal = this->contigOneBuf.template Get<half>();
        LocalTensor<half> zeroLocal = this->contigZeroBuf.template Get<half>();
        LocalTensor<half> selectedLocal = this->contigSelectedBuf.template Get<half>();
        const uint32_t batchLength = rowsThis * innerLength;
        if constexpr (TensorEqualKernelCommon<DATA, TS>::NEED_FP16_WORK) {
            x1Fp16 = this->contigX1Fp16Buf.template Get<half>();
            x2Fp16 = this->contigX2Fp16Buf.template Get<half>();
        }
        if constexpr (TensorEqualKernelCommon<DATA, TS>::NEED_FP32_WORK) {
            x1Fp32 = this->contigX1Fp32Buf.template Get<float>();
            x2Fp32 = this->contigX2Fp32Buf.template Get<float>();
        }

        if (x1Contiguous) {
            LocalTensor<DATA> xLocal = this->qX1.template DeQue<DATA>();
            for (uint32_t row = 0; row < rowsThis; ++row) {
                const uint32_t rowOffset = row * innerLength;
                const uint32_t maskOffset = rowOffset / CMP_MASK_ELEMS_PER_BYTE;
                const uint32_t yBase = (outerBase + row) * innerLength;
                this->CompareEqualScalar(cmpLocal[maskOffset], xLocal[rowOffset],
                                         this->x2Addr, this->GetX2Index(yBase),
                                         x1Fp16[rowOffset], x1Fp32[rowOffset], innerLength);
            }
            this->qX1.FreeTensor(xLocal);
        } else {
            LocalTensor<DATA> xLocal = this->qX2.template DeQue<DATA>();
            for (uint32_t row = 0; row < rowsThis; ++row) {
                const uint32_t rowOffset = row * innerLength;
                const uint32_t maskOffset = rowOffset / CMP_MASK_ELEMS_PER_BYTE;
                const uint32_t yBase = (outerBase + row) * innerLength;
                this->CompareEqualScalar(cmpLocal[maskOffset], xLocal[rowOffset],
                                         this->x1Addr, this->GetX1Index(yBase),
                                         x2Fp16[rowOffset], x2Fp32[rowOffset], innerLength);
            }
            this->qX2.FreeTensor(xLocal);
        }
        this->ExpandCompareMaskBySelect(yLocal, cmpLocal, oneLocal, zeroLocal, selectedLocal, batchLength);
        this->qY.template EnQue<uint8_t>(yLocal);
    }
};

template<typename DATA, class TS>
class TensorEqualBroadcastContiguousKernel : public TensorEqualKernelCommon<DATA, TS> {
public:
    __aicore__ inline void Process()
    {
        if (this->ti.dataLength == 0U) {
            return;
        }
        this->InitContiguousBuffers();
        this->InitContiguousSelectConstants();

        uint32_t base = 0;
        while (base < this->ti.dataLength) {
            const uint32_t x1Limit = this->GetContiguousLimit(this->ti.x1Shape, base);
            const uint32_t x2Limit = this->GetContiguousLimit(this->ti.x2Shape, base);
            const uint32_t remaining = this->ti.dataLength - base;
            uint32_t validLength = this->ti.tileLength < remaining ? this->ti.tileLength : remaining;
            validLength = x1Limit < validLength ? x1Limit : validLength;
            validLength = x2Limit < validLength ? x2Limit : validLength;
            if (validLength == 0U) {
                validLength = 1U;
            }

            const uint32_t calcLength = this->GetCmpAlignedCount(validLength);
            this->CopyInDirect(this->GetX1Index(base), this->GetX2Index(base), true, true, validLength, calcLength);
            this->ComputeContiguous(calcLength);
            this->CopyOutContiguous(base, validLength);
            base += validLength;
        }
    }
};

template<typename DATA, class TS>
class TensorEqualSmallInnerBroadcastKernel : public TensorEqualKernelCommon<DATA, TS> {
public:
    __aicore__ inline void Process()
    {
        if (this->ti.dataLength == 0U) {
            return;
        }
        this->InitContiguousBuffers();
        this->InitContiguousSelectConstants();

        const uint32_t innerLength = this->ti.innerLength;
        const uint32_t alignedInnerLength = this->GetSmallInnerAlignedCount();
        if (innerLength == 0U || alignedInnerLength == 0U || alignedInnerLength > this->ti.tileLength ||
            this->ti.tk6RowsPerTile == 0U) {
            this->ProcessDirectTail();
            return;
        }

        const uint32_t outerLength = this->ti.outerLength;
        for (uint32_t outer = 0; outer < outerLength;) {
            if (TryProcessSmallInnerRepeatBatch(outer, outerLength)) {
                continue;
            }

            const uint32_t yBase = outer * innerLength;
            const uint32_t calcLength = this->GetCmpAlignedCount(innerLength);
            this->CopyInDirect(this->GetX1Index(yBase), this->GetX2Index(yBase), true, true, innerLength, calcLength);
            this->ComputeContiguous(calcLength);
            this->CopyOutContiguous(yBase, innerLength);
            ++outer;
        }
    }

private:
    __aicore__ inline void CopyInSmallInnerRepeatBatch(uint32_t repeatOffset,
                                                       uint32_t contigOffset,
                                                       uint32_t rowsThis,
                                                       bool x1Repeated)
    {
        const uint32_t innerLength = this->ti.innerLength;
        const uint32_t alignedInnerLength = this->GetSmallInnerAlignedCount();
        const uint32_t rowBytes = innerLength * sizeof(DATA);
        const DataCopyParams rowParams{1, static_cast<uint16_t>(rowBytes), 0, 0};
        const DataCopyPadParams padParams{false, 0, 0, 0};

        LocalTensor<DATA> repeatLocal = x1Repeated ? this->qX1.template AllocTensor<DATA>() :
                                                     this->qX2.template AllocTensor<DATA>();
        LocalTensor<DATA> contigLocal = x1Repeated ? this->qX2.template AllocTensor<DATA>() :
                                                     this->qX1.template AllocTensor<DATA>();
        this->DuplicateZero(repeatLocal, alignedInnerLength);
        this->DuplicateZero(contigLocal, rowsThis * alignedInnerLength);
        PipeBarrier<PIPE_ALL>();
        if (x1Repeated) {
            DataCopyPad(repeatLocal, this->x1Gm[repeatOffset], rowParams, padParams);
            for (uint32_t row = 0; row < rowsThis; ++row) {
                DataCopyPad(contigLocal[row * alignedInnerLength],
                            this->x2Gm[contigOffset + row * innerLength], rowParams, padParams);
            }
            this->qX1.template EnQue<DATA>(repeatLocal);
            this->qX2.template EnQue<DATA>(contigLocal);
        } else {
            DataCopyPad(repeatLocal, this->x2Gm[repeatOffset], rowParams, padParams);
            for (uint32_t row = 0; row < rowsThis; ++row) {
                DataCopyPad(contigLocal[row * alignedInnerLength],
                            this->x1Gm[contigOffset + row * innerLength], rowParams, padParams);
            }
            this->qX2.template EnQue<DATA>(repeatLocal);
            this->qX1.template EnQue<DATA>(contigLocal);
        }
    }

    __aicore__ inline void CopyInSmallInnerMaterializedBatch(uint32_t repeatOffset,
                                                             uint32_t contigOffset,
                                                             uint32_t rowsThis,
                                                             bool x1Repeated)
    {
        const uint32_t innerLength = this->ti.innerLength;
        const uint32_t alignedInnerLength = this->GetSmallInnerAlignedCount();
        const uint32_t batchLength = rowsThis * alignedInnerLength;
        const uint32_t rowBytes = innerLength * sizeof(DATA);
        const DataCopyParams rowParams{1, static_cast<uint16_t>(rowBytes), 0, 0};
        const DataCopyPadParams padParams{false, 0, 0, 0};

        LocalTensor<DATA> x1Local = this->qX1.template AllocTensor<DATA>();
        LocalTensor<DATA> x2Local = this->qX2.template AllocTensor<DATA>();
        this->DuplicateZero(x1Local, batchLength);
        this->DuplicateZero(x2Local, batchLength);
        PipeBarrier<PIPE_ALL>();

        for (uint32_t row = 0; row < rowsThis; ++row) {
            const uint32_t localOffset = row * alignedInnerLength;
            const uint32_t contigRowOffset = contigOffset + row * innerLength;
            if (x1Repeated) {
                DataCopyPad(x1Local[localOffset], this->x1Gm[repeatOffset], rowParams, padParams);
                DataCopyPad(x2Local[localOffset], this->x2Gm[contigRowOffset], rowParams, padParams);
            } else {
                DataCopyPad(x1Local[localOffset], this->x1Gm[contigRowOffset], rowParams, padParams);
                DataCopyPad(x2Local[localOffset], this->x2Gm[repeatOffset], rowParams, padParams);
            }
        }
        this->qX1.template EnQue<DATA>(x1Local);
        this->qX2.template EnQue<DATA>(x2Local);
    }

    __aicore__ inline void CompareSmallInnerRepeatBatch(LocalTensor<uint8_t> cmpLocal,
                                                        LocalTensor<DATA> x1Local,
                                                        LocalTensor<DATA> x2Local,
                                                        LocalTensor<half> x1Fp16,
                                                        LocalTensor<half> x2Fp16,
                                                        LocalTensor<float> x1Fp32,
                                                        LocalTensor<float> x2Fp32,
                                                        uint32_t rowsThis,
                                                        bool x1Repeated)
    {
        const uint32_t alignedInnerLength = this->GetSmallInnerAlignedCount();
        const uint32_t batchLength = rowsThis * alignedInnerLength;
        BinaryRepeatParams repeatParams;
        repeatParams.src0RepStride = static_cast<uint8_t>(x1Repeated ? 0U : this->ti.tk6RepeatStrideBlocks);
        repeatParams.src1RepStride = static_cast<uint8_t>(x1Repeated ? this->ti.tk6RepeatStrideBlocks : 0U);

        if constexpr (is_same_v<DATA, half> || is_same_v<DATA, float>) {
            Compare(cmpLocal, x1Local, x2Local, CMPMODE::EQ, static_cast<uint64_t>(alignedInnerLength),
                    static_cast<uint8_t>(rowsThis), repeatParams);
        } else if constexpr (is_same_v<DATA, int32_t> || is_same_v<DATA, int16_t>) {
            const uint8_t dataRepeatStrideBlocks = static_cast<uint8_t>((alignedInnerLength * sizeof(DATA)) /
                                                                        DATA_COPY_ALIGN_BYTES);
            BinaryRepeatParams subParams;
            subParams.dstBlkStride = 1;
            subParams.src0BlkStride = 1;
            subParams.src1BlkStride = 1;
            subParams.dstRepStride = dataRepeatStrideBlocks;
            subParams.src0RepStride = static_cast<uint8_t>(x1Repeated ? 0U : dataRepeatStrideBlocks);
            subParams.src1RepStride = static_cast<uint8_t>(x1Repeated ? dataRepeatStrideBlocks : 0U);
            LocalTensor<DATA> diffLocal = x1Repeated ? x2Local : x1Local;
            Sub(diffLocal, x1Local, x2Local, static_cast<uint64_t>(alignedInnerLength),
                static_cast<uint8_t>(rowsThis), subParams);
            this->CompareIntegerDiffEqual(cmpLocal, diffLocal, x1Fp32, batchLength);
        } else if constexpr (is_same_v<DATA, int8_t> || is_same_v<DATA, uint8_t>) {
            Cast(x1Fp16, x1Local, RoundMode::CAST_NONE, batchLength);
            Cast(x2Fp16, x2Local, RoundMode::CAST_NONE, batchLength);
            Compare(cmpLocal, x1Fp16, x2Fp16, CMPMODE::EQ, static_cast<uint64_t>(alignedInnerLength),
                    static_cast<uint8_t>(rowsThis), repeatParams);
        } else {
            Cast(x1Fp32, x1Local, RoundMode::CAST_NONE, batchLength);
            Cast(x2Fp32, x2Local, RoundMode::CAST_NONE, batchLength);
            Compare(cmpLocal, x1Fp32, x2Fp32, CMPMODE::EQ, static_cast<uint64_t>(alignedInnerLength),
                    static_cast<uint8_t>(rowsThis), repeatParams);
        }
    }

    __aicore__ inline void ComputeSmallInnerRepeatBatch(uint32_t rowsThis, bool x1Repeated)
    {
        LocalTensor<uint8_t> yLocal = this->qY.template AllocTensor<uint8_t>();
        LocalTensor<uint8_t> cmpLocal = this->contigCmpBuf.template Get<uint8_t>();
        LocalTensor<half> x1Fp16;
        LocalTensor<half> x2Fp16;
        LocalTensor<float> x1Fp32;
        LocalTensor<float> x2Fp32;
        LocalTensor<half> oneLocal = this->contigOneBuf.template Get<half>();
        LocalTensor<half> zeroLocal = this->contigZeroBuf.template Get<half>();
        LocalTensor<half> selectedLocal = this->contigSelectedBuf.template Get<half>();
        const uint32_t batchLength = rowsThis * this->GetSmallInnerAlignedCount();
        if constexpr (TensorEqualKernelCommon<DATA, TS>::NEED_FP16_WORK) {
            x1Fp16 = this->contigX1Fp16Buf.template Get<half>();
            x2Fp16 = this->contigX2Fp16Buf.template Get<half>();
        }
        if constexpr (TensorEqualKernelCommon<DATA, TS>::NEED_FP32_WORK) {
            x1Fp32 = this->contigX1Fp32Buf.template Get<float>();
            x2Fp32 = this->contigX2Fp32Buf.template Get<float>();
        }

        LocalTensor<DATA> x1Local = this->qX1.template DeQue<DATA>();
        LocalTensor<DATA> x2Local = this->qX2.template DeQue<DATA>();
        CompareSmallInnerRepeatBatch(cmpLocal, x1Local, x2Local, x1Fp16, x2Fp16, x1Fp32, x2Fp32,
                                     rowsThis, x1Repeated);
        this->qX1.FreeTensor(x1Local);
        this->qX2.FreeTensor(x2Local);

        this->ExpandCompareMaskBySelect(yLocal, cmpLocal, oneLocal, zeroLocal, selectedLocal, batchLength);
        this->qY.template EnQue<uint8_t>(yLocal);
    }

    __aicore__ inline void ComputeSmallInnerMaterializedBatch(uint32_t rowsThis)
    {
        LocalTensor<uint8_t> yLocal = this->qY.template AllocTensor<uint8_t>();
        LocalTensor<uint8_t> cmpLocal = this->contigCmpBuf.template Get<uint8_t>();
        LocalTensor<half> x1Fp16;
        LocalTensor<half> x2Fp16;
        LocalTensor<float> x1Fp32;
        LocalTensor<float> x2Fp32;
        LocalTensor<half> oneLocal = this->contigOneBuf.template Get<half>();
        LocalTensor<half> zeroLocal = this->contigZeroBuf.template Get<half>();
        LocalTensor<half> selectedLocal = this->contigSelectedBuf.template Get<half>();
        const uint32_t batchLength = rowsThis * this->GetSmallInnerAlignedCount();
        if constexpr (TensorEqualKernelCommon<DATA, TS>::NEED_FP16_WORK) {
            x1Fp16 = this->contigX1Fp16Buf.template Get<half>();
            x2Fp16 = this->contigX2Fp16Buf.template Get<half>();
        }
        if constexpr (TensorEqualKernelCommon<DATA, TS>::NEED_FP32_WORK) {
            x1Fp32 = this->contigX1Fp32Buf.template Get<float>();
            x2Fp32 = this->contigX2Fp32Buf.template Get<float>();
        }

        LocalTensor<DATA> x1Local = this->qX1.template DeQue<DATA>();
        LocalTensor<DATA> x2Local = this->qX2.template DeQue<DATA>();
        this->CompareEqual(cmpLocal, x1Local, x2Local, x1Fp16, x2Fp16, x1Fp32, x2Fp32, batchLength);
        this->qX1.FreeTensor(x1Local);
        this->qX2.FreeTensor(x2Local);

        this->ExpandCompareMaskBySelect(yLocal, cmpLocal, oneLocal, zeroLocal, selectedLocal, batchLength);
        this->qY.template EnQue<uint8_t>(yLocal);
    }

    __aicore__ inline void CopyOutSmallInnerRepeatBatch(uint32_t yBase, uint32_t rowsThis)
    {
        LocalTensor<uint8_t> yLocal = this->qY.template DeQue<uint8_t>();
        const uint32_t innerLength = this->ti.innerLength;
        const uint32_t alignedInnerLength = this->GetSmallInnerAlignedCount();
        const DataCopyParams copyParams{1, static_cast<uint16_t>(innerLength), 0, 0};
        for (uint32_t row = 0; row < rowsThis; ++row) {
            DataCopyPad(this->yGm[yBase + row * innerLength], yLocal[row * alignedInnerLength], copyParams);
        }
        this->qY.FreeTensor(yLocal);
    }

    __aicore__ inline bool TryProcessSmallInnerRepeatBatch(uint32_t& outer, uint32_t outerLength)
    {
        const uint32_t innerLength = this->ti.innerLength;
        const uint32_t maxRows = this->ti.tk6RowsPerTile;
        if (maxRows == 0U) {
            return false;
        }

        const uint32_t yBase = outer * innerLength;
        const uint32_t remainingRows = outerLength - outer;
        const uint32_t x1RepeatRows = this->GetRowRepeatLimit(this->ti.x1Shape, yBase) / innerLength;
        const uint32_t x2RepeatRows = this->GetRowRepeatLimit(this->ti.x2Shape, yBase) / innerLength;
        const uint32_t x1ContigRows = this->GetContiguousLimit(this->ti.x1Shape, yBase) / innerLength;
        const uint32_t x2ContigRows = this->GetContiguousLimit(this->ti.x2Shape, yBase) / innerLength;

        if (x1RepeatRows > 1U && x2ContigRows > 1U) {
            const uint32_t rowsThis = this->Min3(remainingRows, x1RepeatRows, x2ContigRows) < maxRows ?
                                      this->Min3(remainingRows, x1RepeatRows, x2ContigRows) : maxRows;
            ProcessSmallInnerRepeatOrMaterializedBatch(this->GetX1Index(yBase), this->GetX2Index(yBase),
                                                       rowsThis, true);
            CopyOutSmallInnerRepeatBatch(yBase, rowsThis);
            outer += rowsThis;
            return true;
        }

        if (x2RepeatRows > 1U && x1ContigRows > 1U) {
            const uint32_t rowsThis = this->Min3(remainingRows, x2RepeatRows, x1ContigRows) < maxRows ?
                                      this->Min3(remainingRows, x2RepeatRows, x1ContigRows) : maxRows;
            ProcessSmallInnerRepeatOrMaterializedBatch(this->GetX2Index(yBase), this->GetX1Index(yBase),
                                                       rowsThis, false);
            CopyOutSmallInnerRepeatBatch(yBase, rowsThis);
            outer += rowsThis;
            return true;
        }

        return false;
    }

    __aicore__ inline void ProcessSmallInnerRepeatOrMaterializedBatch(uint32_t repeatOffset,
                                                                      uint32_t contigOffset,
                                                                      uint32_t rowsThis,
                                                                      bool x1Repeated)
    {
        if (this->GetSmallInnerAlignedCount() == this->GetCmpRepeatLength()) {
            CopyInSmallInnerRepeatBatch(repeatOffset, contigOffset, rowsThis, x1Repeated);
            ComputeSmallInnerRepeatBatch(rowsThis, x1Repeated);
            return;
        }

        CopyInSmallInnerMaterializedBatch(repeatOffset, contigOffset, rowsThis, x1Repeated);
        ComputeSmallInnerMaterializedBatch(rowsThis);
    }
};

extern "C" __global__ __aicore__ void tensor_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        TensorEqualLinearKernel<DTYPE_X1, decltype(tiling_data)> op;
        op.Init(x1, x2, y, tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        TensorEqualBroadcastKernel<DTYPE_X1, decltype(tiling_data)> op;
        op.Init(x1, x2, y, tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        TensorEqualGlobalScalarKernel<DTYPE_X1, decltype(tiling_data)> op;
        op.Init(x1, x2, y, tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        TensorEqualLastDimScalarKernel<DTYPE_X1, decltype(tiling_data)> op;
        op.Init(x1, x2, y, tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        TensorEqualBroadcastContiguousKernel<DTYPE_X1, decltype(tiling_data)> op;
        op.Init(x1, x2, y, tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(6)) {
        TensorEqualSmallInnerBroadcastKernel<DTYPE_X1, decltype(tiling_data)> op;
        op.Init(x1, x2, y, tiling_data, &pipe);
        op.Process();
    } else {
        TensorEqualGenericKernel<DTYPE_X1, decltype(tiling_data)> op;
        op.Init(x1, x2, y, tiling_data, &pipe);
        op.Process();
    }
}
