// Kernel侧核函数实现
#include "kernel_operator.h"

#include "addcmul_tiling.h"
#include "tiling_key_addcmul.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t ALIGN_BYTES = 32;

template <class DT_INPUT_DATA>
class KernelAddcmul {
public:
    __aicore__ inline KernelAddcmul() {}

    __aicore__ inline void Init(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
        const AddcmulTilingData *tiling) {
        this->length = tiling->length;
        this->tileLength = tiling->tileLength == 0 ? 1 : tiling->tileLength;
        this->perCoreLength = tiling->perCoreLength == 0 ? 1 : tiling->perCoreLength;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        this->start = blockIdx * this->perCoreLength;
        if (this->start >= this->length) {
            this->coreLength = 0;
            return;
        }
        uint32_t remain = this->length - this->start;
        this->coreLength = remain < this->perCoreLength ? remain : this->perCoreLength;
        if (this->coreLength == 0) {
            return;
        }

        inputGm.SetGlobalBuffer((__gm__ DT_INPUT_DATA *)input_data, this->length);
        x1Gm.SetGlobalBuffer((__gm__ DT_INPUT_DATA *)x1, this->length);
        x2Gm.SetGlobalBuffer((__gm__ DT_INPUT_DATA *)x2, this->length);
        valueGm.SetGlobalBuffer((__gm__ DT_INPUT_DATA *)value, 1);
        yGm.SetGlobalBuffer((__gm__ DT_INPUT_DATA *)y, this->length);
        this->valueScalar = valueGm.GetValue(0);

        pipe.InitBuffer(inQueueInput, BUFFER_NUM, this->tileLength * sizeof(DT_INPUT_DATA));
        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileLength * sizeof(DT_INPUT_DATA));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileLength * sizeof(DT_INPUT_DATA));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_INPUT_DATA));
    }

    __aicore__ inline void Process() {
        if (this->coreLength == 0) {
            return;
        }
        ProcessContiguous();
    }

private:
    __aicore__ inline bool IsAlignedCopy(uint32_t count) const {
        return (count * sizeof(DT_INPUT_DATA)) % ALIGN_BYTES == 0;
    }

    __aicore__ inline void CopyOneContiguous(AscendC::LocalTensor<DT_INPUT_DATA> local,
        AscendC::GlobalTensor<DT_INPUT_DATA> gm, uint32_t gmOffset, uint32_t count) {
        if (IsAlignedCopy(count)) {
            AscendC::DataCopy(local, gm[gmOffset], count);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(DT_INPUT_DATA)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_INPUT_DATA> padParams{false, 0, 0, static_cast<DT_INPUT_DATA>(0)};
            AscendC::DataCopyPad(local, gm[gmOffset], copyParams, padParams);
        }
    }

    __aicore__ inline void CopyInContiguousTile(uint32_t offset, uint32_t count) {
        AscendC::LocalTensor<DT_INPUT_DATA> inputLocal = inQueueInput.AllocTensor<DT_INPUT_DATA>();
        AscendC::LocalTensor<DT_INPUT_DATA> x1Local = inQueueX1.AllocTensor<DT_INPUT_DATA>();
        AscendC::LocalTensor<DT_INPUT_DATA> x2Local = inQueueX2.AllocTensor<DT_INPUT_DATA>();
        uint32_t gmOffset = this->start + offset;
        CopyOneContiguous(inputLocal, inputGm, gmOffset, count);
        CopyOneContiguous(x1Local, x1Gm, gmOffset, count);
        CopyOneContiguous(x2Local, x2Gm, gmOffset, count);
        inQueueInput.EnQue(inputLocal);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    __aicore__ inline void ProcessContiguous() {
        uint32_t offset = 0;
        uint32_t fullTiles = this->coreLength / this->tileLength;
        for (uint32_t i = 0; i < fullTiles; ++i) {
            CopyInContiguousTile(offset, this->tileLength);
            Compute(this->tileLength);
            CopyOut(offset, this->tileLength);
            offset += this->tileLength;
        }
        uint32_t tail = this->coreLength - offset;
        if (tail != 0) {
            CopyInContiguousTile(offset, tail);
            Compute(tail);
            CopyOut(offset, tail);
        }
    }

    __aicore__ inline void Compute(uint32_t count) {
        AscendC::LocalTensor<DT_INPUT_DATA> inputLocal = inQueueInput.DeQue<DT_INPUT_DATA>();
        AscendC::LocalTensor<DT_INPUT_DATA> x1Local = inQueueX1.DeQue<DT_INPUT_DATA>();
        AscendC::LocalTensor<DT_INPUT_DATA> x2Local = inQueueX2.DeQue<DT_INPUT_DATA>();
        AscendC::LocalTensor<DT_INPUT_DATA> yLocal = outQueueY.AllocTensor<DT_INPUT_DATA>();

        AscendC::Mul(x1Local, x1Local, x2Local, static_cast<int32_t>(count));
        AscendC::Muls(x1Local, x1Local, this->valueScalar, static_cast<int32_t>(count));
        AscendC::Add(yLocal, inputLocal, x1Local, static_cast<int32_t>(count));

        outQueueY.EnQue(yLocal);
        inQueueInput.FreeTensor(inputLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count) {
        AscendC::LocalTensor<DT_INPUT_DATA> yLocal = outQueueY.DeQue<DT_INPUT_DATA>();
        uint32_t gmOffset = this->start + offset;
        if (IsAlignedCopy(count)) {
            AscendC::DataCopy(yGm[gmOffset], yLocal, count);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(DT_INPUT_DATA)), 0, 0, 0};
            AscendC::DataCopyPad(yGm[gmOffset], yLocal, copyParams);
        }
        outQueueY.FreeTensor(yLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueInput;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX1;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX2;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DT_INPUT_DATA> inputGm;
    AscendC::GlobalTensor<DT_INPUT_DATA> x1Gm;
    AscendC::GlobalTensor<DT_INPUT_DATA> x2Gm;
    AscendC::GlobalTensor<DT_INPUT_DATA> valueGm;
    AscendC::GlobalTensor<DT_INPUT_DATA> yGm;
    uint32_t length;
    uint32_t tileLength;
    uint32_t perCoreLength;
    uint32_t start;
    uint32_t coreLength;
    DT_INPUT_DATA valueScalar;
};

template <>
class KernelAddcmul<int8_t> {
public:
    __aicore__ inline KernelAddcmul() {}

    __aicore__ inline void Init(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
        const AddcmulTilingData *tiling) {
        this->length = tiling->length;
        this->tileLength = tiling->tileLength == 0 ? 1 : tiling->tileLength;
        this->perCoreLength = tiling->perCoreLength == 0 ? 1 : tiling->perCoreLength;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        this->start = blockIdx * this->perCoreLength;
        if (this->start >= this->length) {
            this->coreLength = 0;
            return;
        }
        uint32_t remain = this->length - this->start;
        this->coreLength = remain < this->perCoreLength ? remain : this->perCoreLength;
        if (this->coreLength == 0) {
            return;
        }

        inputGm.SetGlobalBuffer((__gm__ int8_t *)input_data, this->length);
        x1Gm.SetGlobalBuffer((__gm__ int8_t *)x1, this->length);
        x2Gm.SetGlobalBuffer((__gm__ int8_t *)x2, this->length);
        valueGm.SetGlobalBuffer((__gm__ int8_t *)value, 1);
        yGm.SetGlobalBuffer((__gm__ int8_t *)y, this->length);
        this->valueScalar = static_cast<half>(valueGm.GetValue(0));

        pipe.InitBuffer(inQueueInput, BUFFER_NUM, this->tileLength * sizeof(int8_t));
        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileLength * sizeof(int8_t));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileLength * sizeof(int8_t));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(int8_t));
        pipe.InitBuffer(inputHalfBuf, this->tileLength * sizeof(half));
        pipe.InitBuffer(x1HalfBuf, this->tileLength * sizeof(half));
        pipe.InitBuffer(x2HalfBuf, this->tileLength * sizeof(half));
        pipe.InitBuffer(tmpHalfBuf, this->tileLength * sizeof(half));
    }

    __aicore__ inline void Process() {
        if (this->coreLength == 0) {
            return;
        }
        ProcessContiguous();
    }

private:
    __aicore__ inline bool IsAlignedCopy(uint32_t count) const {
        return (count * sizeof(int8_t)) % ALIGN_BYTES == 0;
    }

    __aicore__ inline void CopyOneContiguous(AscendC::LocalTensor<int8_t> local,
        AscendC::GlobalTensor<int8_t> gm, uint32_t gmOffset, uint32_t count) {
        if (IsAlignedCopy(count)) {
            AscendC::DataCopy(local, gm[gmOffset], count);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(int8_t)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<int8_t> padParams{false, 0, 0, static_cast<int8_t>(0)};
            AscendC::DataCopyPad(local, gm[gmOffset], copyParams, padParams);
        }
    }

    __aicore__ inline void CopyInContiguousTile(uint32_t offset, uint32_t count) {
        AscendC::LocalTensor<int8_t> inputLocal = inQueueInput.AllocTensor<int8_t>();
        AscendC::LocalTensor<int8_t> x1Local = inQueueX1.AllocTensor<int8_t>();
        AscendC::LocalTensor<int8_t> x2Local = inQueueX2.AllocTensor<int8_t>();
        uint32_t gmOffset = this->start + offset;
        CopyOneContiguous(inputLocal, inputGm, gmOffset, count);
        CopyOneContiguous(x1Local, x1Gm, gmOffset, count);
        CopyOneContiguous(x2Local, x2Gm, gmOffset, count);
        inQueueInput.EnQue(inputLocal);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    __aicore__ inline void ProcessContiguous() {
        uint32_t offset = 0;
        uint32_t fullTiles = this->coreLength / this->tileLength;
        for (uint32_t i = 0; i < fullTiles; ++i) {
            CopyInContiguousTile(offset, this->tileLength);
            Compute(this->tileLength);
            CopyOut(offset, this->tileLength);
            offset += this->tileLength;
        }
        uint32_t tail = this->coreLength - offset;
        if (tail != 0) {
            CopyInContiguousTile(offset, tail);
            Compute(tail);
            CopyOut(offset, tail);
        }
    }

    __aicore__ inline void Compute(uint32_t count) {
        AscendC::LocalTensor<int8_t> inputLocal = inQueueInput.DeQue<int8_t>();
        AscendC::LocalTensor<int8_t> x1Local = inQueueX1.DeQue<int8_t>();
        AscendC::LocalTensor<int8_t> x2Local = inQueueX2.DeQue<int8_t>();
        AscendC::LocalTensor<int8_t> yLocal = outQueueY.AllocTensor<int8_t>();

        AscendC::LocalTensor<half> inputHalf = inputHalfBuf.Get<half>();
        AscendC::LocalTensor<half> x1Half = x1HalfBuf.Get<half>();
        AscendC::LocalTensor<half> x2Half = x2HalfBuf.Get<half>();
        AscendC::LocalTensor<half> tmpHalf = tmpHalfBuf.Get<half>();
        uint32_t calcCount = static_cast<uint32_t>(count);

        AscendC::Cast(inputHalf, inputLocal, AscendC::RoundMode::CAST_NONE, calcCount);
        AscendC::Cast(x1Half, x1Local, AscendC::RoundMode::CAST_NONE, calcCount);
        AscendC::Cast(x2Half, x2Local, AscendC::RoundMode::CAST_NONE, calcCount);
        AscendC::Mul(tmpHalf, x1Half, x2Half, static_cast<int32_t>(count));
        AscendC::Muls(tmpHalf, tmpHalf, this->valueScalar, static_cast<int32_t>(count));
        AscendC::Add(inputHalf, inputHalf, tmpHalf, static_cast<int32_t>(count));
        AscendC::Cast(yLocal, inputHalf, AscendC::RoundMode::CAST_TRUNC, calcCount);

        outQueueY.EnQue(yLocal);
        inQueueInput.FreeTensor(inputLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count) {
        AscendC::LocalTensor<int8_t> yLocal = outQueueY.DeQue<int8_t>();
        uint32_t gmOffset = this->start + offset;
        if (IsAlignedCopy(count)) {
            AscendC::DataCopy(yGm[gmOffset], yLocal, count);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(int8_t)), 0, 0, 0};
            AscendC::DataCopyPad(yGm[gmOffset], yLocal, copyParams);
        }
        outQueueY.FreeTensor(yLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueInput;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX1;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX2;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputHalfBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x1HalfBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x2HalfBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpHalfBuf;
    AscendC::GlobalTensor<int8_t> inputGm;
    AscendC::GlobalTensor<int8_t> x1Gm;
    AscendC::GlobalTensor<int8_t> x2Gm;
    AscendC::GlobalTensor<int8_t> valueGm;
    AscendC::GlobalTensor<int8_t> yGm;
    uint32_t length;
    uint32_t tileLength;
    uint32_t perCoreLength;
    uint32_t start;
    uint32_t coreLength;
    half valueScalar;
};

template <typename DT_INPUT_DATA>
__global__ __aicore__ void addcmul(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(AddcmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddcmulTilingData, tiling_data, tiling);
    KernelAddcmul<DT_INPUT_DATA> op;
    op.Init(input_data, x1, x2, value, y, &tiling_data);
    op.Process();
}
