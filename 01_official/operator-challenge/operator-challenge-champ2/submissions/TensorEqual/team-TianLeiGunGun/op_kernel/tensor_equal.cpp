#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

__aicore__ inline bool Bf16RawEqual(uint16_t x1, uint16_t x2)
{
    uint16_t absX1 = x1 & static_cast<uint16_t>(0x7fff);
    uint16_t absX2 = x2 & static_cast<uint16_t>(0x7fff);
    bool x1IsNan = (absX1 > static_cast<uint16_t>(0x7f80));
    bool x2IsNan = (absX2 > static_cast<uint16_t>(0x7f80));
    if (x1IsNan || x2IsNan)
    {
        return false;
    }
    if (absX1 == 0 && absX2 == 0)
    {
        return true;
    }
    return x1 == x2;
}

class KernelTensorEqualHalf
{
    using T = half;

public:
    __aicore__ inline KernelTensorEqualHalf() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                uint32_t formerLength, uint32_t formerNum,
                                uint32_t tailLength,
                                uint32_t totalLength, uint32_t partitionLength, TPipe *pipeIn)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        const uint32_t weight = sizeof(T);
        this->formerLength = formerLength;
        this->formerNum = formerNum;
        this->tailLength = tailLength;
        this->totalLength = totalLength;

        x1Gm.SetGlobalBuffer((__gm__ T *)x1, totalLength);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2, totalLength);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y, totalLength);
        pipe = pipeIn;
        pipe->InitBuffer(inQueueX1, BUFFER_NUM, partitionLength * weight * sizeof(T));
        pipe->InitBuffer(inQueueX2, BUFFER_NUM, partitionLength * weight * sizeof(T));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, partitionLength * weight * sizeof(uint8_t));

        pipe->InitBuffer(tmpSelMask, partitionLength * weight * sizeof(uint8_t));
    }
    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->formerNum; ++i)
        {
            CopyIn(i, this->formerLength);
            Compute(i, this->formerLength);
            CopyOut(i, this->formerLength);
        }
        // 对于 小Tiling
        if (this->tailLength)
        {
            uint32_t tailLength_AC_32 = ((this->tailLength * sizeof(T) + 31) / 32 * 32) / sizeof(T);
            CopyIn(this->formerNum, tailLength_AC_32);
            Compute(this->formerNum, this->tailLength);
            CopyOut(this->formerNum, tailLength_AC_32);
        }
    }

private:
    __aicore__ inline void CopyIn(const uint32_t _progress_, const uint32_t _len_)
    {
        LocalTensor<T> x1Local = inQueueX1.AllocTensor<T>();
        LocalTensor<T> x2Local = inQueueX2.AllocTensor<T>();
        DataCopy(x1Local, x1Gm[_progress_ * this->formerLength], _len_);
        DataCopy(x2Local, x2Gm[_progress_ * this->formerLength], _len_);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }
    __aicore__ inline void Compute(const uint32_t _progress_, const uint32_t _len_)
    {
        LocalTensor<T> x1Local = inQueueX1.DeQue<T>();
        LocalTensor<T> x2Local = inQueueX2.DeQue<T>();
        LocalTensor<uint8_t> yLocal = outQueueY.AllocTensor<uint8_t>();
        auto selMask = tmpSelMask.Get<uint8_t>();

        Sub(x1Local, x1Local, x2Local, _len_);

        CompareScalar(selMask, x1Local, static_cast<T>(0), CMPMODE::EQ, (_len_ + 255) / 256 * 256);
        Duplicate(x1Local, (T)1, _len_);
        Select(x1Local, selMask, x1Local, (T)0, SELMODE::VSEL_TENSOR_SCALAR_MODE, _len_);
        Cast(yLocal, x1Local, RoundMode::CAST_NONE, _len_);

        outQueueY.EnQue<uint8_t>(yLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }
    __aicore__ inline void CopyOut(const uint32_t _progress_, const uint32_t _len_)
    {
        LocalTensor<uint8_t> yLocal = outQueueY.DeQue<uint8_t>();
        DataCopyPad(yGm[_progress_ * this->formerLength], yLocal, {1, _len_, 0, 0, 0});
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> inQueueX1, inQueueX2;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<QuePosition::VECCALC> tmpSelMask;

    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<uint8_t> yGm;

    uint32_t formerLength;
    uint32_t formerNum;
    uint32_t tailLength;
    uint32_t totalLength;
};

class KernelTensorEqualFloat
{
    using T = float;

public:
    __aicore__ inline KernelTensorEqualFloat() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                uint32_t formerLength, uint32_t formerNum,
                                uint32_t tailLength,
                                uint32_t totalLength, uint32_t partitionLength, TPipe *pipeIn)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        const uint32_t weight = sizeof(T);
        this->formerLength = formerLength;
        this->formerNum = formerNum;
        this->tailLength = tailLength;
        this->totalLength = totalLength;

        x1Gm.SetGlobalBuffer((__gm__ T *)x1, totalLength);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2, totalLength);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y, totalLength);
        pipe = pipeIn;
        pipe->InitBuffer(inQueueX1, BUFFER_NUM, partitionLength * weight * sizeof(T));
        pipe->InitBuffer(inQueueX2, BUFFER_NUM, partitionLength * weight * sizeof(T));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, partitionLength * weight * sizeof(uint8_t));

        pipe->InitBuffer(tmpSelMask, partitionLength * weight * sizeof(uint8_t));
        pipe->InitBuffer(tmpBuff, partitionLength * weight * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->formerNum; ++i)
        {
            CopyIn(i, this->formerLength);
            Compute(i, this->formerLength);
            CopyOut(i, this->formerLength);
        }
        // 对于 小Tiling
        if (this->tailLength)
        {
            uint32_t tailLength_AC_32 = ((this->tailLength * sizeof(T) + 31) / 32 * 32) / sizeof(T);
            CopyIn(this->formerNum, tailLength_AC_32);
            Compute(this->formerNum, this->tailLength);
            CopyOut(this->formerNum, tailLength_AC_32);
        }
    }

private:
    __aicore__ inline void CopyIn(const uint32_t _progress_, const uint32_t _len_)
    {
        LocalTensor<T> x1Local = inQueueX1.AllocTensor<T>();
        LocalTensor<T> x2Local = inQueueX2.AllocTensor<T>();
        DataCopy(x1Local, x1Gm[_progress_ * this->formerLength], _len_);
        DataCopy(x2Local, x2Gm[_progress_ * this->formerLength], _len_);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }
    __aicore__ inline void Compute(const uint32_t _progress_, const uint32_t _len_)
    {
        LocalTensor<T> x1Local = inQueueX1.DeQue<T>();
        LocalTensor<T> x2Local = inQueueX2.DeQue<T>();
        LocalTensor<uint8_t> yLocal = outQueueY.AllocTensor<uint8_t>();
        auto selMask = tmpSelMask.Get<uint8_t>();
        auto buff = tmpBuff.Get<half>();

        Sub(x1Local, x1Local, x2Local, _len_);

        CompareScalar(selMask, x1Local, static_cast<T>(0), CMPMODE::EQ, (_len_ + 255) / 256 * 256);
        Duplicate(x1Local, (T)1, _len_);
        Select(x1Local, selMask, x1Local, (T)0, SELMODE::VSEL_TENSOR_SCALAR_MODE, _len_);

        Cast(buff, x1Local, RoundMode::CAST_NONE, _len_);
        Cast(yLocal, buff, RoundMode::CAST_NONE, _len_);

        outQueueY.EnQue<uint8_t>(yLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }
    __aicore__ inline void CopyOut(const uint32_t _progress_, const uint32_t _len_)
    {
        LocalTensor<uint8_t> yLocal = outQueueY.DeQue<uint8_t>();
        DataCopyPad(yGm[_progress_ * this->formerLength], yLocal, {1, _len_, 0, 0, 0});
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> inQueueX1, inQueueX2;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<QuePosition::VECCALC> tmpSelMask, tmpBuff;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<uint8_t> yGm;

    uint32_t formerLength;
    uint32_t formerNum;
    uint32_t tailLength;
    uint32_t totalLength;
};

class KernelTensorEqualInt32
{
    using T = int32_t;

public:
    __aicore__ inline KernelTensorEqualInt32() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                uint32_t formerLength, uint32_t formerNum,
                                uint32_t tailLength,
                                uint32_t totalLength, uint32_t partitionLength, TPipe *pipeIn)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        const uint32_t weight = sizeof(T);
        this->formerLength = formerLength;
        this->formerNum = formerNum;
        this->tailLength = tailLength;
        this->totalLength = totalLength;

        x1Gm.SetGlobalBuffer((__gm__ T *)x1, totalLength);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2, totalLength);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y, totalLength);
        pipe = pipeIn;
        pipe->InitBuffer(inQueueX1, BUFFER_NUM, partitionLength * weight * sizeof(T));
        pipe->InitBuffer(inQueueX2, BUFFER_NUM, partitionLength * weight * sizeof(T));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, partitionLength * weight * sizeof(uint8_t));

        pipe->InitBuffer(tmp1, partitionLength * weight * sizeof(float));
        pipe->InitBuffer(tmp2, partitionLength * weight * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->formerNum; ++i)
        {
            CopyIn(i, this->formerLength);
            Compute(i, this->formerLength);
            CopyOut(i, this->formerLength);
        }
        if (this->tailLength)
        {
            uint32_t tailLength_AC_32 = ((this->tailLength * sizeof(T) + 31) / 32 * 32) / sizeof(T);
            CopyIn(this->formerNum, tailLength_AC_32);
            Compute(this->formerNum, this->tailLength);
            CopyOut(this->formerNum, tailLength_AC_32);
        }
    }

private:
    __aicore__ inline void CopyIn(const uint32_t _progress_, const uint32_t _len_)
    {
        LocalTensor<T> x1Local = inQueueX1.AllocTensor<T>();
        LocalTensor<T> x2Local = inQueueX2.AllocTensor<T>();
        DataCopy(x1Local, x1Gm[_progress_ * this->formerLength], _len_);
        DataCopy(x2Local, x2Gm[_progress_ * this->formerLength], _len_);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }
    __aicore__ inline void Compute(const uint32_t _progress_, const uint32_t _len_)
    {
        LocalTensor<T> x1Local = inQueueX1.DeQue<T>();
        LocalTensor<T> x2Local = inQueueX2.DeQue<T>();
        LocalTensor<uint8_t> yLocal = outQueueY.AllocTensor<uint8_t>();
        auto tmpF = tmp1.Get<float>();
        auto tmpH = tmp2.Get<half>();

        Sub(x1Local, x1Local, x2Local, _len_);

        CompareScalar(yLocal, x1Local, static_cast<T>(0), CMPMODE::EQ, (_len_ + 255) / 256 * 256);
        Duplicate(tmpF, static_cast<float>(1), _len_);
        Select(tmpF, yLocal, tmpF, static_cast<float>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, _len_);
        Cast(tmpH, tmpF, RoundMode::CAST_NONE, _len_);
        Cast(yLocal, tmpH, RoundMode::CAST_NONE, _len_);

        outQueueY.EnQue<uint8_t>(yLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }
    __aicore__ inline void CopyOut(const uint32_t _progress_, const uint32_t _len_)
    {
        LocalTensor<uint8_t> yLocal = outQueueY.DeQue<uint8_t>();
        DataCopyPad(yGm[_progress_ * this->formerLength], yLocal, {1, _len_, 0, 0, 0});
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> inQueueX1, inQueueX2;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<QuePosition::VECCALC> tmp1, tmp2;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<uint8_t> yGm;
    uint32_t formerLength;
    uint32_t formerNum;
    uint32_t tailLength;
    uint32_t totalLength;
};

class KernelTensorEqualInt8
{
    using T = int8_t;

public:
    __aicore__ inline KernelTensorEqualInt8() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                uint32_t formerLength, uint32_t formerNum,
                                uint32_t tailLength,
                                uint32_t totalLength, uint32_t partitionLength, TPipe *pipeIn)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        const uint32_t weight = sizeof(T);
        this->formerLength = formerLength;
        this->formerNum = formerNum;
        this->tailLength = tailLength;
        this->totalLength = totalLength;

        x1Gm.SetGlobalBuffer((__gm__ T *)x1, totalLength);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2, totalLength);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y, totalLength);
        pipe = pipeIn;
        pipe->InitBuffer(inQueueX1, BUFFER_NUM, partitionLength * weight * sizeof(T));
        pipe->InitBuffer(inQueueX2, BUFFER_NUM, partitionLength * weight * sizeof(T));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, partitionLength * weight * sizeof(uint8_t));

        pipe->InitBuffer(tmpSelMask, partitionLength * weight * sizeof(uint8_t));
        pipe->InitBuffer(tmp1, partitionLength * weight * sizeof(half));
        pipe->InitBuffer(tmp2, partitionLength * weight * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->formerNum; ++i)
        {
            CopyIn(i, this->formerLength);
            Compute(i, this->formerLength);
            CopyOut(i, this->formerLength);
        }
        if (this->tailLength)
        {
            uint32_t tailLength_AC_32 = ((this->tailLength * sizeof(T) + 31) / 32 * 32) / sizeof(T);
            CopyIn(this->formerNum, tailLength_AC_32);
            Compute(this->formerNum, this->tailLength);
            CopyOut(this->formerNum, tailLength_AC_32);
        }
    }

private:
    __aicore__ inline void CopyIn(const uint32_t _progress_, const uint32_t _len_)
    {
        LocalTensor<T> x1Local = inQueueX1.AllocTensor<T>();
        LocalTensor<T> x2Local = inQueueX2.AllocTensor<T>();
        DataCopy(x1Local, x1Gm[_progress_ * this->formerLength], _len_);
        DataCopy(x2Local, x2Gm[_progress_ * this->formerLength], _len_);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }
    __aicore__ inline void Compute(const uint32_t _progress_, const uint32_t _len_)
    {
        LocalTensor<T> x1Local = inQueueX1.DeQue<T>();
        LocalTensor<T> x2Local = inQueueX2.DeQue<T>();
        LocalTensor<uint8_t> yLocal = outQueueY.AllocTensor<uint8_t>();
        auto selMask = tmpSelMask.Get<uint8_t>();
        auto tmpX1 = tmp1.Get<half>();
        auto tmpX2 = tmp2.Get<half>();

        Cast(tmpX1, x1Local, RoundMode::CAST_NONE, _len_);
        Cast(tmpX2, x2Local, RoundMode::CAST_NONE, _len_);
        Sub(tmpX1, tmpX1, tmpX2, _len_);

        CompareScalar(selMask, tmpX1, static_cast<half>(0), CMPMODE::EQ, (_len_ + 255) / 256 * 256);

        Duplicate(tmpX1, (half)1, _len_);
        Select(tmpX1, selMask, tmpX1, (half)0, SELMODE::VSEL_TENSOR_SCALAR_MODE, _len_);

        Cast(yLocal, tmpX1, RoundMode::CAST_NONE, _len_);

        outQueueY.EnQue<uint8_t>(yLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }
    __aicore__ inline void CopyOut(const uint32_t _progress_, const uint32_t _len_)
    {
        LocalTensor<uint8_t> yLocal = outQueueY.DeQue<uint8_t>();
        DataCopy(yGm[_progress_ * this->formerLength], yLocal, _len_);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> inQueueX1, inQueueX2;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<QuePosition::VECCALC> tmpSelMask, tmp1, tmp2;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<uint8_t> yGm;
    uint32_t formerLength;
    uint32_t formerNum;
    uint32_t tailLength;
    uint32_t totalLength;
};

template <typename T>
class KernelTensorEqualCompareVector
{
public:
    __aicore__ inline KernelTensorEqualCompareVector() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                uint32_t formerLength, uint32_t formerNum,
                                uint32_t tailLength,
                                uint32_t totalLength, uint32_t partitionLength, TPipe *pipeIn)
    {
        this->formerLength = formerLength;
        this->formerNum = formerNum;
        this->tailLength = tailLength;
        this->totalLength = totalLength;
        x1Gm.SetGlobalBuffer((__gm__ T *)x1, totalLength);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2, totalLength);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y, totalLength);
        pipe = pipeIn;
        pipe->InitBuffer(inQueueX1, BUFFER_NUM, partitionLength * sizeof(T));
        pipe->InitBuffer(inQueueX2, BUFFER_NUM, partitionLength * sizeof(T));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, partitionLength * sizeof(uint8_t));
        pipe->InitBuffer(tmpSelMask, partitionLength * sizeof(uint8_t));
        pipe->InitBuffer(tmpHalf, partitionLength * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->formerNum; ++i)
        {
            CopyIn(i, this->formerLength);
            Compute(this->formerLength);
            CopyOut(i, this->formerLength);
        }
        if (this->tailLength)
        {
            uint32_t tailLength_AC_32 = ((this->tailLength * sizeof(T) + 31) / 32 * 32) / sizeof(T);
            CopyIn(this->formerNum, tailLength_AC_32);
            Compute(this->tailLength);
            CopyOut(this->formerNum, ((this->tailLength + 31) / 32 * 32));
        }
    }

private:
    __aicore__ inline void CopyIn(const uint32_t progress, const uint32_t len)
    {
        LocalTensor<T> x1Local = inQueueX1.AllocTensor<T>();
        LocalTensor<T> x2Local = inQueueX2.AllocTensor<T>();
        DataCopy(x1Local, x1Gm[progress * this->formerLength], len);
        DataCopy(x2Local, x2Gm[progress * this->formerLength], len);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }
    __aicore__ inline void Compute(const uint32_t len)
    {
        LocalTensor<T> x1Local = inQueueX1.DeQue<T>();
        LocalTensor<T> x2Local = inQueueX2.DeQue<T>();
        LocalTensor<uint8_t> yLocal = outQueueY.AllocTensor<uint8_t>();
        auto selMask = tmpSelMask.Get<uint8_t>();
        auto tmpH = tmpHalf.Get<half>();

        Compare(selMask, x1Local, x2Local, CMPMODE::EQ, (len + 255) / 256 * 256);
        Duplicate(tmpH, static_cast<half>(1), len);
        Select(tmpH, selMask, tmpH, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
        Cast(yLocal, tmpH, RoundMode::CAST_NONE, len);

        outQueueY.EnQue<uint8_t>(yLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }
    __aicore__ inline void CopyOut(const uint32_t progress, const uint32_t len)
    {
        LocalTensor<uint8_t> yLocal = outQueueY.DeQue<uint8_t>();
        DataCopyPad(yGm[progress * this->formerLength], yLocal, {1, len, 0, 0, 0});
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> inQueueX1, inQueueX2;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<QuePosition::VECCALC> tmpSelMask, tmpHalf;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<uint8_t> yGm;
    uint32_t formerLength;
    uint32_t formerNum;
    uint32_t tailLength;
    uint32_t totalLength;
};

template <typename T>
class KernelTensorEqualScalar
{

public:
    __aicore__ inline KernelTensorEqualScalar() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                uint32_t formerLength, uint32_t formerNum,
                                uint32_t tailLength,
                                uint32_t totalLength, uint32_t partitionLength, TPipe *pipeIn)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->totalLength = totalLength;

        x1Gm.SetGlobalBuffer((__gm__ T *)x1, totalLength);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2, totalLength);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y, totalLength);
    }
    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->totalLength; ++i)
        {
            T x1Value = x1Gm.GetValue(i);
            T x2Value = x2Gm.GetValue(i);
            yGm.SetValue(i, static_cast<uint8_t>(x1Value == x2Value));
        }
    }

private:
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<uint8_t> yGm;
    uint32_t totalLength;
};

class KernelTensorEqualBf16Raw
{

public:
    __aicore__ inline KernelTensorEqualBf16Raw() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                uint32_t formerLength, uint32_t formerNum,
                                uint32_t tailLength,
                                uint32_t totalLength, uint32_t partitionLength, TPipe *pipeIn)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->totalLength = totalLength;
        x1Gm.SetGlobalBuffer((__gm__ uint16_t *)x1, totalLength);
        x2Gm.SetGlobalBuffer((__gm__ uint16_t *)x2, totalLength);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y, totalLength);
    }
    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->totalLength; ++i)
        {
            yGm.SetValue(i, static_cast<uint8_t>(Bf16RawEqual(x1Gm.GetValue(i), x2Gm.GetValue(i))));
        }
    }

private:
    GlobalTensor<uint16_t> x1Gm;
    GlobalTensor<uint16_t> x2Gm;
    GlobalTensor<uint8_t> yGm;
    uint32_t totalLength;
};

template <typename T>
class KernelTensorEqualScalarBroadCast
{

public:
    __aicore__ inline KernelTensorEqualScalarBroadCast() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                uint32_t formerLength, uint32_t formerNum,
                                uint32_t tailLength,
                                uint32_t totalLength, uint32_t partitionLength,
                                int32_t scalarSide, TPipe *pipeIn)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->formerLength = formerLength;
        this->formerNum = formerNum;
        this->tailLength = tailLength;
        this->totalLength = totalLength;
        this->scalarSide = scalarSide;
        x1Gm.SetGlobalBuffer((__gm__ T *)x1, totalLength);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2, totalLength);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y, totalLength);
        pipe = pipeIn;
        pipe->InitBuffer(inQueueX, BUFFER_NUM, partitionLength * sizeof(T));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, partitionLength * sizeof(uint8_t));
        pipe->InitBuffer(tmpSelMask, partitionLength * sizeof(uint8_t));
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int32_t>)
        {
            pipe->InitBuffer(tmpFloat, partitionLength * sizeof(float));
            pipe->InitBuffer(tmpHalf, partitionLength * sizeof(half));
        }
        if constexpr (std::is_same_v<T, int8_t>)
        {
            pipe->InitBuffer(tmpHalf, partitionLength * sizeof(half));
        }
    }
    __aicore__ inline void Process()
    {
        if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint8_t>)
        {
            ProcessScalar();
        }
        else
        {
            for (uint32_t i = 0; i < this->formerNum; ++i)
            {
                CopyIn(i, this->formerLength);
                Compute(this->formerLength);
                CopyOut(i, this->formerLength);
            }
            if (this->tailLength)
            {
                uint32_t tailLength_AC_32 = ((this->tailLength * sizeof(T) + 31) / 32 * 32) / sizeof(T);
                CopyIn(this->formerNum, tailLength_AC_32);
                Compute(this->tailLength);
                CopyOut(this->formerNum, ((this->tailLength + 31) / 32 * 32));
            }
        }
    }

private:
    __aicore__ inline T GetScalar()
    {
        return this->scalarSide == 1 ? x1Gm.GetValue(0) : x2Gm.GetValue(0);
    }
    __aicore__ inline void CopyIn(const uint32_t progress, const uint32_t len)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        if (this->scalarSide == 1)
        {
            DataCopy(xLocal, x2Gm[progress * this->formerLength], len);
        }
        else
        {
            DataCopy(xLocal, x1Gm[progress * this->formerLength], len);
        }
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(const uint32_t len)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<uint8_t> yLocal = outQueueY.AllocTensor<uint8_t>();
        auto selMask = tmpSelMask.Get<uint8_t>();
        T scalar = GetScalar();

        if constexpr (std::is_same_v<T, half>)
        {
            CompareScalar(selMask, xLocal, scalar, CMPMODE::EQ, (len + 255) / 256 * 256);
            Duplicate(xLocal, static_cast<T>(1), len);
            Select(xLocal, selMask, xLocal, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
            Cast(yLocal, xLocal, RoundMode::CAST_NONE, len);
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            auto tmpF = tmpFloat.Get<float>();
            auto tmpH = tmpHalf.Get<half>();
            CompareScalar(selMask, xLocal, scalar, CMPMODE::EQ, (len + 255) / 256 * 256);
            Duplicate(tmpF, static_cast<float>(1), len);
            Select(tmpF, selMask, tmpF, static_cast<float>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
            Cast(tmpH, tmpF, RoundMode::CAST_NONE, len);
            Cast(yLocal, tmpH, RoundMode::CAST_NONE, len);
        }
        else if constexpr (std::is_same_v<T, int32_t>)
        {
            auto tmpF = tmpFloat.Get<float>();
            auto tmpH = tmpHalf.Get<half>();
            CompareScalar(selMask, xLocal, scalar, CMPMODE::EQ, (len + 255) / 256 * 256);
            Duplicate(tmpF, static_cast<float>(1), len);
            Select(tmpF, selMask, tmpF, static_cast<float>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
            Cast(tmpH, tmpF, RoundMode::CAST_NONE, len);
            Cast(yLocal, tmpH, RoundMode::CAST_NONE, len);
        }
        else if constexpr (std::is_same_v<T, int8_t>)
        {
            auto tmpH = tmpHalf.Get<half>();
            Cast(tmpH, xLocal, RoundMode::CAST_NONE, len);
            CompareScalar(selMask, tmpH, static_cast<half>(scalar), CMPMODE::EQ, (len + 255) / 256 * 256);
            Duplicate(tmpH, static_cast<half>(1), len);
            Select(tmpH, selMask, tmpH, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
            Cast(yLocal, tmpH, RoundMode::CAST_NONE, len);
        }

        outQueueY.EnQue<uint8_t>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(const uint32_t progress, const uint32_t len)
    {
        LocalTensor<uint8_t> yLocal = outQueueY.DeQue<uint8_t>();
        DataCopyPad(yGm[progress * this->formerLength], yLocal, {1, len, 0, 0, 0});
        outQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void ProcessScalar()
    {
        T scalar = GetScalar();
        for (uint32_t i = 0; i < this->totalLength; ++i)
        {
            T value = this->scalarSide == 1 ? x2Gm.GetValue(i) : x1Gm.GetValue(i);
            yGm.SetValue(i, static_cast<uint8_t>(value == scalar));
        }
    }

private:
    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<QuePosition::VECCALC> tmpSelMask, tmpFloat, tmpHalf;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<uint8_t> yGm;
    uint32_t formerLength;
    uint32_t formerNum;
    uint32_t tailLength;
    uint32_t totalLength;
    int32_t scalarSide;
};

class KernelTensorEqualScalarBroadCastBf16Raw
{
public:
    __aicore__ inline KernelTensorEqualScalarBroadCastBf16Raw() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t totalLength, int32_t scalarSide)
    {
        this->totalLength = totalLength;
        this->scalarSide = scalarSide;
        x1Gm.SetGlobalBuffer((__gm__ uint16_t *)x1, totalLength);
        x2Gm.SetGlobalBuffer((__gm__ uint16_t *)x2, totalLength);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y, totalLength);
    }
    __aicore__ inline void Process()
    {
        uint16_t scalar = this->scalarSide == 1 ? x1Gm.GetValue(0) : x2Gm.GetValue(0);
        for (uint32_t i = 0; i < this->totalLength; ++i)
        {
            uint16_t value = this->scalarSide == 1 ? x2Gm.GetValue(i) : x1Gm.GetValue(i);
            yGm.SetValue(i, static_cast<uint8_t>(Bf16RawEqual(value, scalar)));
        }
    }

private:
    GlobalTensor<uint16_t> x1Gm;
    GlobalTensor<uint16_t> x2Gm;
    GlobalTensor<uint8_t> yGm;
    uint32_t totalLength;
    int32_t scalarSide;
};

template <typename T>
class KernelTensorEqualLastDimBroadCast
{
public:
    __aicore__ inline KernelTensorEqualLastDimBroadCast() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                uint32_t innerLength, uint32_t outerNum,
                                uint32_t partitionLength, int32_t scalarSide, TPipe *pipeIn)
    {
        this->innerLength = innerLength;
        this->outerNum = outerNum;
        this->partitionLength = partitionLength;
        this->scalarSide = scalarSide;
        x1Gm.SetGlobalBuffer((__gm__ T *)x1, innerLength * outerNum);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2, innerLength * outerNum);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y, innerLength * outerNum);
        pipe = pipeIn;
        pipe->InitBuffer(inQueueX, BUFFER_NUM, partitionLength * sizeof(T));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, partitionLength * sizeof(uint8_t));
        pipe->InitBuffer(tmpSelMask, partitionLength * sizeof(uint8_t));
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int32_t>)
        {
            pipe->InitBuffer(tmpFloat, partitionLength * sizeof(float));
            pipe->InitBuffer(tmpHalf, partitionLength * sizeof(half));
        }
        if constexpr (std::is_same_v<T, int8_t>)
        {
            pipe->InitBuffer(tmpHalf, partitionLength * sizeof(half));
        }
    }
    __aicore__ inline void Process()
    {
        for (uint32_t row = 0; row < this->outerNum; ++row)
        {
            T scalar = this->scalarSide == 1 ? x1Gm.GetValue(row) : x2Gm.GetValue(row);
            for (uint32_t offset = 0; offset < this->innerLength; offset += this->partitionLength)
            {
                uint32_t len = this->innerLength - offset;
                if (len > this->partitionLength)
                {
                    len = this->partitionLength;
                }
                uint32_t copyLen = ((len * sizeof(T) + 31) / 32 * 32) / sizeof(T);
                uint32_t gmOffset = row * this->innerLength + offset;
                CopyIn(gmOffset, copyLen);
                Compute(len, scalar);
                CopyOut(gmOffset, len);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(const uint32_t gmOffset, const uint32_t len)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        if (this->scalarSide == 1)
        {
            DataCopy(xLocal, x2Gm[gmOffset], len);
        }
        else
        {
            DataCopy(xLocal, x1Gm[gmOffset], len);
        }
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(const uint32_t len, T scalar)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<uint8_t> yLocal = outQueueY.AllocTensor<uint8_t>();
        auto selMask = tmpSelMask.Get<uint8_t>();

        if constexpr (std::is_same_v<T, half>)
        {
            CompareScalar(selMask, xLocal, scalar, CMPMODE::EQ, (len + 255) / 256 * 256);
            Duplicate(xLocal, static_cast<T>(1), len);
            Select(xLocal, selMask, xLocal, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
            Cast(yLocal, xLocal, RoundMode::CAST_NONE, len);
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            auto tmpF = tmpFloat.Get<float>();
            auto tmpH = tmpHalf.Get<half>();
            CompareScalar(selMask, xLocal, scalar, CMPMODE::EQ, (len + 255) / 256 * 256);
            Duplicate(tmpF, static_cast<float>(1), len);
            Select(tmpF, selMask, tmpF, static_cast<float>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
            Cast(tmpH, tmpF, RoundMode::CAST_NONE, len);
            Cast(yLocal, tmpH, RoundMode::CAST_NONE, len);
        }
        else if constexpr (std::is_same_v<T, int32_t>)
        {
            auto tmpF = tmpFloat.Get<float>();
            auto tmpH = tmpHalf.Get<half>();
            CompareScalar(selMask, xLocal, scalar, CMPMODE::EQ, (len + 255) / 256 * 256);
            Duplicate(tmpF, static_cast<float>(1), len);
            Select(tmpF, selMask, tmpF, static_cast<float>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
            Cast(tmpH, tmpF, RoundMode::CAST_NONE, len);
            Cast(yLocal, tmpH, RoundMode::CAST_NONE, len);
        }
        else if constexpr (std::is_same_v<T, int8_t>)
        {
            auto tmpH = tmpHalf.Get<half>();
            Cast(tmpH, xLocal, RoundMode::CAST_NONE, len);
            CompareScalar(selMask, tmpH, static_cast<half>(scalar), CMPMODE::EQ, (len + 255) / 256 * 256);
            Duplicate(tmpH, static_cast<half>(1), len);
            Select(tmpH, selMask, tmpH, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
            Cast(yLocal, tmpH, RoundMode::CAST_NONE, len);
        }

        outQueueY.EnQue<uint8_t>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(const uint32_t gmOffset, const uint32_t len)
    {
        LocalTensor<uint8_t> yLocal = outQueueY.DeQue<uint8_t>();
        DataCopyPad(yGm[gmOffset], yLocal, {1, len, 0, 0, 0});
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<QuePosition::VECCALC> tmpSelMask, tmpFloat, tmpHalf;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<uint8_t> yGm;
    uint32_t innerLength;
    uint32_t outerNum;
    uint32_t partitionLength;
    int32_t scalarSide;
};

template <typename T>
class KernelTensorEqualRepeatBlockBroadCast
{
public:
    __aicore__ inline KernelTensorEqualRepeatBlockBroadCast() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                uint32_t blockLength, uint32_t repeatNum,
                                uint32_t partitionLength, int32_t repeatSide, TPipe *pipeIn)
    {
        this->blockLength = blockLength;
        this->repeatNum = repeatNum;
        this->partitionLength = partitionLength;
        this->repeatSide = repeatSide;
        x1Gm.SetGlobalBuffer((__gm__ T *)x1, blockLength * repeatNum);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2, blockLength * repeatNum);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y, blockLength * repeatNum);
        pipe = pipeIn;
        pipe->InitBuffer(inQueueX1, BUFFER_NUM, partitionLength * sizeof(T));
        pipe->InitBuffer(inQueueX2, BUFFER_NUM, partitionLength * sizeof(T));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, partitionLength * sizeof(uint8_t));
        pipe->InitBuffer(tmpSelMask, partitionLength * sizeof(uint8_t));
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int32_t>)
        {
            pipe->InitBuffer(tmpFloat, partitionLength * sizeof(float));
            pipe->InitBuffer(tmpHalf, partitionLength * sizeof(half));
        }
        if constexpr (std::is_same_v<T, int8_t>)
        {
            pipe->InitBuffer(tmpHalf, partitionLength * sizeof(half));
            pipe->InitBuffer(tmpHalf2, partitionLength * sizeof(half));
        }
    }
    __aicore__ inline void Process()
    {
        for (uint32_t r = 0; r < this->repeatNum; ++r)
        {
            for (uint32_t offset = 0; offset < this->blockLength; offset += this->partitionLength)
            {
                uint32_t len = this->blockLength - offset;
                if (len > this->partitionLength)
                {
                    len = this->partitionLength;
                }
                uint32_t copyLen = ((len * sizeof(T) + 31) / 32 * 32) / sizeof(T);
                uint32_t outOffset = r * this->blockLength + offset;
                uint32_t x1Offset = this->repeatSide == 1 ? offset : outOffset;
                uint32_t x2Offset = this->repeatSide == 2 ? offset : outOffset;
                CopyIn(x1Offset, x2Offset, copyLen);
                Compute(len);
                CopyOut(outOffset, len);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(const uint32_t x1Offset, const uint32_t x2Offset, const uint32_t len)
    {
        LocalTensor<T> x1Local = inQueueX1.AllocTensor<T>();
        LocalTensor<T> x2Local = inQueueX2.AllocTensor<T>();
        DataCopy(x1Local, x1Gm[x1Offset], len);
        DataCopy(x2Local, x2Gm[x2Offset], len);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }
    __aicore__ inline void Compute(const uint32_t len)
    {
        LocalTensor<T> x1Local = inQueueX1.DeQue<T>();
        LocalTensor<T> x2Local = inQueueX2.DeQue<T>();
        LocalTensor<uint8_t> yLocal = outQueueY.AllocTensor<uint8_t>();
        auto selMask = tmpSelMask.Get<uint8_t>();

        if constexpr (std::is_same_v<T, half>)
        {
            Sub(x1Local, x1Local, x2Local, len);
            CompareScalar(selMask, x1Local, static_cast<T>(0), CMPMODE::EQ, (len + 255) / 256 * 256);
            Duplicate(x1Local, static_cast<T>(1), len);
            Select(x1Local, selMask, x1Local, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
            Cast(yLocal, x1Local, RoundMode::CAST_NONE, len);
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            auto tmpF = tmpFloat.Get<float>();
            auto tmpH = tmpHalf.Get<half>();
            Sub(x1Local, x1Local, x2Local, len);
            CompareScalar(selMask, x1Local, static_cast<T>(0), CMPMODE::EQ, (len + 255) / 256 * 256);
            Duplicate(tmpF, static_cast<float>(1), len);
            Select(tmpF, selMask, tmpF, static_cast<float>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
            Cast(tmpH, tmpF, RoundMode::CAST_NONE, len);
            Cast(yLocal, tmpH, RoundMode::CAST_NONE, len);
        }
        else if constexpr (std::is_same_v<T, int32_t>)
        {
            auto tmpF = tmpFloat.Get<float>();
            auto tmpH = tmpHalf.Get<half>();
            Sub(x1Local, x1Local, x2Local, len);
            CompareScalar(selMask, x1Local, static_cast<T>(0), CMPMODE::EQ, (len + 255) / 256 * 256);
            Duplicate(tmpF, static_cast<float>(1), len);
            Select(tmpF, selMask, tmpF, static_cast<float>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
            Cast(tmpH, tmpF, RoundMode::CAST_NONE, len);
            Cast(yLocal, tmpH, RoundMode::CAST_NONE, len);
        }
        else if constexpr (std::is_same_v<T, int8_t>)
        {
            auto tmpH = tmpHalf.Get<half>();
            auto tmpX2 = tmpHalf2.Get<half>();
            Cast(tmpH, x1Local, RoundMode::CAST_NONE, len);
            Cast(tmpX2, x2Local, RoundMode::CAST_NONE, len);
            Sub(tmpH, tmpH, tmpX2, len);
            CompareScalar(selMask, tmpH, static_cast<half>(0), CMPMODE::EQ, (len + 255) / 256 * 256);
            Duplicate(tmpH, static_cast<half>(1), len);
            Select(tmpH, selMask, tmpH, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
            Cast(yLocal, tmpH, RoundMode::CAST_NONE, len);
        }

        outQueueY.EnQue<uint8_t>(yLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }
    __aicore__ inline void CopyOut(const uint32_t outOffset, const uint32_t len)
    {
        LocalTensor<uint8_t> yLocal = outQueueY.DeQue<uint8_t>();
        DataCopyPad(yGm[outOffset], yLocal, {1, len, 0, 0, 0});
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> inQueueX1, inQueueX2;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<QuePosition::VECCALC> tmpSelMask, tmpFloat, tmpHalf, tmpHalf2;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<uint8_t> yGm;
    uint32_t blockLength;
    uint32_t repeatNum;
    uint32_t partitionLength;
    int32_t repeatSide;
};

template <typename T>
class KernelTensorEqualContiguousVector
{
public:
    __aicore__ inline KernelTensorEqualContiguousVector() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                uint32_t totalLength, uint32_t partitionLength, TPipe *pipeIn)
    {
        this->totalLength = totalLength;
        this->partitionLength = partitionLength;
        x1Gm.SetGlobalBuffer((__gm__ T *)x1, totalLength);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2, totalLength);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y, totalLength);
        pipe = pipeIn;
        pipe->InitBuffer(inQueueX1, BUFFER_NUM, partitionLength * sizeof(T));
        pipe->InitBuffer(inQueueX2, BUFFER_NUM, partitionLength * sizeof(T));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, partitionLength * sizeof(uint8_t));
        pipe->InitBuffer(tmpSelMask, partitionLength * sizeof(uint8_t));
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int32_t>)
        {
            pipe->InitBuffer(tmpFloat, partitionLength * sizeof(float));
            pipe->InitBuffer(tmpHalf, partitionLength * sizeof(half));
        }
        if constexpr (std::is_same_v<T, int8_t>)
        {
            pipe->InitBuffer(tmpHalf, partitionLength * sizeof(half));
            pipe->InitBuffer(tmpHalf2, partitionLength * sizeof(half));
        }
    }
    __aicore__ inline void Process()
    {
        uint32_t blockLength = (this->totalLength + GetBlockNum() - 1) / GetBlockNum();
        blockLength = (blockLength + 31) / 32 * 32;
        uint32_t start = GetBlockIdx() * blockLength;
        uint32_t end = start + blockLength;
        if (end > this->totalLength)
        {
            end = this->totalLength;
        }
        for (uint32_t offset = start; offset < end; offset += this->partitionLength)
        {
            uint32_t len = end - offset;
            if (len > this->partitionLength)
            {
                len = this->partitionLength;
            }
            uint32_t copyLen = ((len * sizeof(T) + 31) / 32 * 32) / sizeof(T);
            CopyIn(offset, copyLen);
            Compute(len);
            CopyOut(offset, len);
        }
    }

private:
    __aicore__ inline void CopyIn(const uint32_t offset, const uint32_t len)
    {
        LocalTensor<T> x1Local = inQueueX1.AllocTensor<T>();
        LocalTensor<T> x2Local = inQueueX2.AllocTensor<T>();
        DataCopy(x1Local, x1Gm[offset], len);
        DataCopy(x2Local, x2Gm[offset], len);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }
    __aicore__ inline void Compute(const uint32_t len)
    {
        LocalTensor<T> x1Local = inQueueX1.DeQue<T>();
        LocalTensor<T> x2Local = inQueueX2.DeQue<T>();
        LocalTensor<uint8_t> yLocal = outQueueY.AllocTensor<uint8_t>();
        auto selMask = tmpSelMask.Get<uint8_t>();
        if constexpr (std::is_same_v<T, half>)
        {
            Sub(x1Local, x1Local, x2Local, len);
            CompareScalar(selMask, x1Local, static_cast<T>(0), CMPMODE::EQ, (len + 255) / 256 * 256);
            Duplicate(x1Local, static_cast<T>(1), len);
            Select(x1Local, selMask, x1Local, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
            Cast(yLocal, x1Local, RoundMode::CAST_NONE, len);
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            auto tmpF = tmpFloat.Get<float>();
            auto tmpH = tmpHalf.Get<half>();
            Sub(x1Local, x1Local, x2Local, len);
            CompareScalar(selMask, x1Local, static_cast<T>(0), CMPMODE::EQ, (len + 255) / 256 * 256);
            Duplicate(tmpF, static_cast<float>(1), len);
            Select(tmpF, selMask, tmpF, static_cast<float>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
            Cast(tmpH, tmpF, RoundMode::CAST_NONE, len);
            Cast(yLocal, tmpH, RoundMode::CAST_NONE, len);
        }
        else if constexpr (std::is_same_v<T, int32_t>)
        {
            auto tmpF = tmpFloat.Get<float>();
            auto tmpH = tmpHalf.Get<half>();
            Sub(x1Local, x1Local, x2Local, len);
            CompareScalar(selMask, x1Local, static_cast<T>(0), CMPMODE::EQ, (len + 255) / 256 * 256);
            Duplicate(tmpF, static_cast<float>(1), len);
            Select(tmpF, selMask, tmpF, static_cast<float>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
            Cast(tmpH, tmpF, RoundMode::CAST_NONE, len);
            Cast(yLocal, tmpH, RoundMode::CAST_NONE, len);
        }
        else if constexpr (std::is_same_v<T, int8_t>)
        {
            auto tmpH = tmpHalf.Get<half>();
            auto tmpX2 = tmpHalf2.Get<half>();
            Cast(tmpH, x1Local, RoundMode::CAST_NONE, len);
            Cast(tmpX2, x2Local, RoundMode::CAST_NONE, len);
            Sub(tmpH, tmpH, tmpX2, len);
            CompareScalar(selMask, tmpH, static_cast<half>(0), CMPMODE::EQ, (len + 255) / 256 * 256);
            Duplicate(tmpH, static_cast<half>(1), len);
            Select(tmpH, selMask, tmpH, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
            Cast(yLocal, tmpH, RoundMode::CAST_NONE, len);
        }
        outQueueY.EnQue<uint8_t>(yLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }
    __aicore__ inline void CopyOut(const uint32_t offset, const uint32_t len)
    {
        LocalTensor<uint8_t> yLocal = outQueueY.DeQue<uint8_t>();
        DataCopyPad(yGm[offset], yLocal, {1, len, 0, 0, 0});
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> inQueueX1, inQueueX2;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<QuePosition::VECCALC> tmpSelMask, tmpFloat, tmpHalf, tmpHalf2;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<uint8_t> yGm;
    uint32_t totalLength;
    uint32_t partitionLength;
};

template <typename T>
class KernelTensorEqualInnerVectorBroadCast
{
public:
    __aicore__ inline KernelTensorEqualInnerVectorBroadCast() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                uint32_t innerLength, uint32_t outerNum,
                                uint32_t partitionLength, int32_t scalarSide,
                                int32_t outerRank,
                                int32_t *y_ndarray, int32_t *x1_ndarray, int32_t *x2_ndarray,
                                int32_t *x1_stride, int32_t *x2_stride, TPipe *pipeIn)
    {
        this->innerLength = innerLength;
        this->outerNum = outerNum;
        this->partitionLength = partitionLength;
        this->scalarSide = scalarSide;
        this->outerRank = outerRank;
        this->y_ndarray = y_ndarray;
        this->x1_ndarray = x1_ndarray;
        this->x2_ndarray = x2_ndarray;
        this->x1_stride = x1_stride;
        this->x2_stride = x2_stride;
        x1Gm.SetGlobalBuffer((__gm__ T *)x1, innerLength * outerNum);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2, innerLength * outerNum);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y, innerLength * outerNum);
        pipe = pipeIn;
        pipe->InitBuffer(inQueueX, BUFFER_NUM, partitionLength * sizeof(T));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, partitionLength * sizeof(uint8_t));
        pipe->InitBuffer(tmpSelMask, partitionLength * sizeof(uint8_t));
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int32_t>)
        {
            pipe->InitBuffer(tmpFloat, partitionLength * sizeof(float));
            pipe->InitBuffer(tmpHalf, partitionLength * sizeof(half));
        }
        if constexpr (std::is_same_v<T, int8_t>)
        {
            pipe->InitBuffer(tmpHalf, partitionLength * sizeof(half));
        }
    }
    __aicore__ inline void Process()
    {
        for (uint32_t outer = 0; outer < this->outerNum; ++outer)
        {
            uint32_t x1Base = 0;
            uint32_t x2Base = 0;
            uint32_t tmp = outer;
            for (int32_t d = this->outerRank - 1; d >= 0; --d)
            {
                uint32_t coord = tmp % this->y_ndarray[d];
                tmp /= this->y_ndarray[d];
                if (this->x1_ndarray[d] != 1)
                {
                    x1Base += coord * this->x1_stride[d];
                }
                if (this->x2_ndarray[d] != 1)
                {
                    x2Base += coord * this->x2_stride[d];
                }
            }
            uint32_t outBase = outer * this->innerLength;
            T scalar = this->scalarSide == 1 ? x1Gm.GetValue(x1Base) : x2Gm.GetValue(x2Base);
            uint32_t vectorBase = this->scalarSide == 1 ? x2Base : x1Base;
            for (uint32_t offset = 0; offset < this->innerLength; offset += this->partitionLength)
            {
                uint32_t len = this->innerLength - offset;
                if (len > this->partitionLength)
                {
                    len = this->partitionLength;
                }
                uint32_t copyLen = ((len * sizeof(T) + 31) / 32 * 32) / sizeof(T);
                CopyIn(vectorBase + offset, copyLen);
                Compute(len, scalar);
                CopyOut(outBase + offset, len);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(const uint32_t vectorOffset, const uint32_t len)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        if (this->scalarSide == 1)
        {
            DataCopy(xLocal, x2Gm[vectorOffset], len);
        }
        else
        {
            DataCopy(xLocal, x1Gm[vectorOffset], len);
        }
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(const uint32_t len, T scalar)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<uint8_t> yLocal = outQueueY.AllocTensor<uint8_t>();
        auto selMask = tmpSelMask.Get<uint8_t>();
        if constexpr (std::is_same_v<T, half>)
        {
            CompareScalar(selMask, xLocal, scalar, CMPMODE::EQ, (len + 255) / 256 * 256);
            Duplicate(xLocal, static_cast<T>(1), len);
            Select(xLocal, selMask, xLocal, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
            Cast(yLocal, xLocal, RoundMode::CAST_NONE, len);
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            auto tmpF = tmpFloat.Get<float>();
            auto tmpH = tmpHalf.Get<half>();
            CompareScalar(selMask, xLocal, scalar, CMPMODE::EQ, (len + 255) / 256 * 256);
            Duplicate(tmpF, static_cast<float>(1), len);
            Select(tmpF, selMask, tmpF, static_cast<float>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
            Cast(tmpH, tmpF, RoundMode::CAST_NONE, len);
            Cast(yLocal, tmpH, RoundMode::CAST_NONE, len);
        }
        else if constexpr (std::is_same_v<T, int32_t>)
        {
            auto tmpF = tmpFloat.Get<float>();
            auto tmpH = tmpHalf.Get<half>();
            CompareScalar(selMask, xLocal, scalar, CMPMODE::EQ, (len + 255) / 256 * 256);
            Duplicate(tmpF, static_cast<float>(1), len);
            Select(tmpF, selMask, tmpF, static_cast<float>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
            Cast(tmpH, tmpF, RoundMode::CAST_NONE, len);
            Cast(yLocal, tmpH, RoundMode::CAST_NONE, len);
        }
        else if constexpr (std::is_same_v<T, int8_t>)
        {
            auto tmpH = tmpHalf.Get<half>();
            Cast(tmpH, xLocal, RoundMode::CAST_NONE, len);
            CompareScalar(selMask, tmpH, static_cast<half>(scalar), CMPMODE::EQ, (len + 255) / 256 * 256);
            Duplicate(tmpH, static_cast<half>(1), len);
            Select(tmpH, selMask, tmpH, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
            Cast(yLocal, tmpH, RoundMode::CAST_NONE, len);
        }
        outQueueY.EnQue<uint8_t>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(const uint32_t outOffset, const uint32_t len)
    {
        LocalTensor<uint8_t> yLocal = outQueueY.DeQue<uint8_t>();
        DataCopyPad(yGm[outOffset], yLocal, {1, len, 0, 0, 0});
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<QuePosition::VECCALC> tmpSelMask, tmpFloat, tmpHalf;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<uint8_t> yGm;
    uint32_t innerLength;
    uint32_t outerNum;
    uint32_t partitionLength;
    int32_t scalarSide;
    int32_t outerRank;
    int32_t *y_ndarray;
    int32_t *x1_ndarray;
    int32_t *x2_ndarray;
    int32_t *x1_stride;
    int32_t *x2_stride;
};

template <typename T>
class KernelTensorEqualBroadCast
{

public:
    __aicore__ inline KernelTensorEqualBroadCast() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                int32_t y_dimensional,
                                int32_t *y_ndarray, int32_t *x1_ndarray, int32_t *x2_ndarray,
                                int32_t *y_sumndarray, int32_t *x1_sumndarray, int32_t *x2_sumndarray)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        this->y_dimensional = y_dimensional;

        this->y_ndarray = y_ndarray;
        this->x1_ndarray = x1_ndarray;
        this->x2_ndarray = x2_ndarray;
        this->y_sumndarray = y_sumndarray;
        this->x1_sumndarray = x1_sumndarray;
        this->x2_sumndarray = x2_sumndarray;

        x1Gm.SetGlobalBuffer((__gm__ T *)x1, 1);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2, 1);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y, 1);
    }
    __aicore__ inline void Process()
    {
        int dim = this->y_dimensional;
        for (int j = 0; j < this->y_sumndarray[dim]; j++)
        {
            int x1_start = 0, x2_start = 0;
            for (int k = 0; k < dim; k++)
            {
                if (this->x1_ndarray[k] != 1)
                {
                    x1_start += this->x1_sumndarray[k] * (j / this->y_sumndarray[k] % this->y_ndarray[k]);
                }
                if (this->x2_ndarray[k] != 1)
                {
                    x2_start += this->x2_sumndarray[k] * (j / this->y_sumndarray[k] % this->y_ndarray[k]);
                }
            }
            if constexpr (std::is_same_v<T, uint8_t>)
            {
                T x1 = x1Gm.GetValue(x1_start);
                T x2 = x2Gm.GetValue(x2_start);
                yGm.SetValue(j, static_cast<uint8_t>(x1 == x2));
            }
            else
            {
                float x1 = static_cast<float>(x1Gm.GetValue(x1_start));
                float x2 = static_cast<float>(x2Gm.GetValue(x2_start));
                yGm.SetValue(j, static_cast<uint8_t>(x1 == x2));
            }
        }
    }

private:
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<uint8_t> yGm;

    int32_t y_dimensional;
    int32_t *y_ndarray;
    int32_t *x1_ndarray;
    int32_t *x2_ndarray;

    int32_t *y_sumndarray;
    int32_t *x1_sumndarray;
    int32_t *x2_sumndarray;
};

class KernelTensorEqualBroadCastBf16Raw
{

public:
    __aicore__ inline KernelTensorEqualBroadCastBf16Raw() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                int32_t y_dimensional,
                                int32_t *y_ndarray, int32_t *x1_ndarray, int32_t *x2_ndarray,
                                int32_t *y_sumndarray, int32_t *x1_sumndarray, int32_t *x2_sumndarray)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->y_dimensional = y_dimensional;
        this->y_ndarray = y_ndarray;
        this->x1_ndarray = x1_ndarray;
        this->x2_ndarray = x2_ndarray;
        this->y_sumndarray = y_sumndarray;
        this->x1_sumndarray = x1_sumndarray;
        this->x2_sumndarray = x2_sumndarray;
        x1Gm.SetGlobalBuffer((__gm__ uint16_t *)x1, 1);
        x2Gm.SetGlobalBuffer((__gm__ uint16_t *)x2, 1);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y, 1);
    }
    __aicore__ inline void Process()
    {
        int dim = this->y_dimensional;
        for (int j = 0; j < this->y_sumndarray[dim]; j++)
        {
            int x1_start = 0, x2_start = 0;
            for (int k = 0; k < dim; k++)
            {
                if (this->x1_ndarray[k] != 1)
                {
                    x1_start += this->x1_sumndarray[k] * (j / this->y_sumndarray[k] % this->y_ndarray[k]);
                }
                if (this->x2_ndarray[k] != 1)
                {
                    x2_start += this->x2_sumndarray[k] * (j / this->y_sumndarray[k] % this->y_ndarray[k]);
                }
            }
            yGm.SetValue(j, static_cast<uint8_t>(Bf16RawEqual(x1Gm.GetValue(x1_start), x2Gm.GetValue(x2_start))));
        }
    }

private:
    GlobalTensor<uint16_t> x1Gm;
    GlobalTensor<uint16_t> x2Gm;
    GlobalTensor<uint8_t> yGm;
    int32_t y_dimensional;
    int32_t *y_ndarray;
    int32_t *x1_ndarray;
    int32_t *x2_ndarray;
    int32_t *y_sumndarray;
    int32_t *x1_sumndarray;
    int32_t *x2_sumndarray;
};

extern "C" __global__ __aicore__ void tensor_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(3))
    {
        KernelTensorEqualInt32 op;
        op.Init(x1, x2, y,
                tiling_data.formerLength, tiling_data.formerNum,
                tiling_data.tailLength,
                tiling_data.totalLength, tiling_data.partitionLength, &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(1))
    {
        KernelTensorEqualHalf op;
        op.Init(x1, x2, y,
                tiling_data.formerLength, tiling_data.formerNum,
                tiling_data.tailLength,
                tiling_data.totalLength, tiling_data.partitionLength, &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(2))
    {
        KernelTensorEqualFloat op;
        op.Init(x1, x2, y,
                tiling_data.formerLength, tiling_data.formerNum,
                tiling_data.tailLength,
                tiling_data.totalLength, tiling_data.partitionLength, &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(4))
    {
        KernelTensorEqualInt8 op;
        op.Init(x1, x2, y,
                tiling_data.formerLength, tiling_data.formerNum,
                tiling_data.tailLength,
                tiling_data.totalLength, tiling_data.partitionLength, &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(5))
    {
        if constexpr (std::is_same_v<DTYPE_X1, bfloat16_t>)
        {
            KernelTensorEqualBroadCastBf16Raw op;
            op.Init(x1, x2, y,
                    tiling_data.y_dimensional,
                    tiling_data.y_ndarray, tiling_data.x1_ndarray, tiling_data.x2_ndarray,
                    tiling_data.y_sumndarray, tiling_data.x1_sumndarray, tiling_data.x2_sumndarray);
            op.Process();
        }
        else
        {
            KernelTensorEqualBroadCast<DTYPE_X1> op;
            op.Init(x1, x2, y,
                    tiling_data.y_dimensional,
                    tiling_data.y_ndarray, tiling_data.x1_ndarray, tiling_data.x2_ndarray,
                    tiling_data.y_sumndarray, tiling_data.x1_sumndarray, tiling_data.x2_sumndarray);
            op.Process();
        }
    }
    else if (TILING_KEY_IS(6))
    {
        KernelTensorEqualBf16Raw op;
        op.Init(x1, x2, y,
                tiling_data.formerLength, tiling_data.formerNum,
                tiling_data.tailLength,
                tiling_data.totalLength, tiling_data.partitionLength, &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(7))
    {
        KernelTensorEqualScalar<int16_t> op;
        op.Init(x1, x2, y,
                tiling_data.formerLength, tiling_data.formerNum,
                tiling_data.tailLength,
                tiling_data.totalLength, tiling_data.partitionLength, &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(8))
    {
        KernelTensorEqualScalar<uint8_t> op;
        op.Init(x1, x2, y,
                tiling_data.formerLength, tiling_data.formerNum,
                tiling_data.tailLength,
                tiling_data.totalLength, tiling_data.partitionLength, &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(9))
    {
        if constexpr (std::is_same_v<DTYPE_X1, bfloat16_t>)
        {
            KernelTensorEqualScalarBroadCastBf16Raw op;
            op.Init(x1, x2, y, tiling_data.totalLength, tiling_data.y_dimensional);
            op.Process();
        }
        else
        {
            KernelTensorEqualScalarBroadCast<DTYPE_X1> op;
            op.Init(x1, x2, y,
                    tiling_data.formerLength, tiling_data.formerNum,
                    tiling_data.tailLength,
                    tiling_data.totalLength, tiling_data.partitionLength,
                    tiling_data.y_dimensional, &pipe);
            op.Process();
        }
    }
    else if (TILING_KEY_IS(10))
    {
        if constexpr (std::is_same_v<DTYPE_X1, half> || std::is_same_v<DTYPE_X1, float> ||
                      std::is_same_v<DTYPE_X1, int32_t> || std::is_same_v<DTYPE_X1, int8_t>)
        {
            KernelTensorEqualLastDimBroadCast<DTYPE_X1> op;
            op.Init(x1, x2, y,
                    tiling_data.formerLength, tiling_data.formerNum,
                    tiling_data.partitionLength, tiling_data.y_dimensional, &pipe);
            op.Process();
        }
    }
    else if (TILING_KEY_IS(11))
    {
        if constexpr (std::is_same_v<DTYPE_X1, half> || std::is_same_v<DTYPE_X1, float> ||
                      std::is_same_v<DTYPE_X1, int32_t> || std::is_same_v<DTYPE_X1, int8_t>)
        {
            KernelTensorEqualRepeatBlockBroadCast<DTYPE_X1> op;
            op.Init(x1, x2, y,
                    tiling_data.formerLength, tiling_data.formerNum,
                    tiling_data.partitionLength, tiling_data.y_dimensional, &pipe);
            op.Process();
        }
    }
    else if (TILING_KEY_IS(12))
    {
        if constexpr (std::is_same_v<DTYPE_X1, half> || std::is_same_v<DTYPE_X1, float> ||
                      std::is_same_v<DTYPE_X1, int32_t> || std::is_same_v<DTYPE_X1, int8_t>)
        {
            KernelTensorEqualInnerVectorBroadCast<DTYPE_X1> op;
            op.Init(x1, x2, y,
                    tiling_data.formerLength, tiling_data.formerNum,
                    tiling_data.partitionLength, tiling_data.tailLength,
                    tiling_data.y_dimensional,
                    tiling_data.y_ndarray, tiling_data.x1_ndarray, tiling_data.x2_ndarray,
                    tiling_data.x1_sumndarray, tiling_data.x2_sumndarray, &pipe);
            op.Process();
        }
    }
    else if (TILING_KEY_IS(13))
    {
        if constexpr (std::is_same_v<DTYPE_X1, half> || std::is_same_v<DTYPE_X1, float> ||
                      std::is_same_v<DTYPE_X1, int32_t> || std::is_same_v<DTYPE_X1, int8_t>)
        {
            KernelTensorEqualContiguousVector<DTYPE_X1> op;
            op.Init(x1, x2, y, tiling_data.totalLength, tiling_data.partitionLength, &pipe);
            op.Process();
        }
    }
}
