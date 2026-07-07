#include "kernel_operator.h"

#include <type_traits>

using namespace AscendC;

namespace {

constexpr uint32_t kBufferNum = 2;
constexpr uint32_t kAlignBytes = 32;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kSqrtPiOver2 = 0.886226925452758f;
constexpr float kTwoOverSqrtPi = 1.1283791670955126f;
constexpr float kSimpson32Scale = 0.011753949657244923f;  // 2/sqrt(pi)/(3*32)
constexpr float kDomainEps = 1.0e-7f;
constexpr float kSignEps = 1.0e-30f;
constexpr float kLutMaxX = 0.99999f;
constexpr float kWinitzkiA = 0.147f;
constexpr float kWinitzkiC = 4.330746750799873f;  // 2 / (pi * a)
constexpr uint32_t kFp32GilesMinTotalLength = 1000000U;
constexpr uint32_t kFastSpecialMinTotalLength = 4096U;

__aicore__ inline uint32_t AlignUpCount(uint32_t value, uint32_t align)
{
    return ((value + align - 1) / align) * align;
}

__aicore__ inline uint32_t AlignUpBytes(uint32_t bytes)
{
    return AlignUpCount(bytes, kAlignBytes);
}

template <typename T>
__aicore__ inline bool CanUseFastCopy(uint32_t offset, uint32_t len, uint32_t copyAlignBytes)
{
    uint64_t byteOffset = static_cast<uint64_t>(offset) * static_cast<uint64_t>(sizeof(T));
    uint64_t byteLen = static_cast<uint64_t>(len) * static_cast<uint64_t>(sizeof(T));
    return copyAlignBytes != 0 && (byteOffset % copyAlignBytes == 0) && (byteLen % copyAlignBytes == 0);
}

template <typename T>
__aicore__ inline void CopyInPad(GlobalTensor<T>& inputGm, uint32_t offset, LocalTensor<T>& inputLocal, uint32_t len)
{
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(inputLocal, inputGm[offset], copyParams, padParams);
}

template <typename T>
__aicore__ inline void CopyOutPad(GlobalTensor<T>& outputGm, uint32_t offset, LocalTensor<T>& outputLocal, uint32_t len)
{
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGm[offset], outputLocal, copyParams);
}

template <typename T>
__aicore__ inline void CopyInAuto(GlobalTensor<T>& inputGm, uint32_t offset, LocalTensor<T>& inputLocal, uint32_t len,
                                  uint32_t copyAlignBytes)
{
    if (CanUseFastCopy<T>(offset, len, copyAlignBytes)) {
        DataCopy(inputLocal, inputGm[offset], len);
    } else {
        CopyInPad(inputGm, offset, inputLocal, len);
    }
}

template <typename T>
__aicore__ inline void CopyOutAuto(GlobalTensor<T>& outputGm, uint32_t offset, LocalTensor<T>& outputLocal,
                                   uint32_t len, uint32_t copyAlignBytes)
{
    if (CanUseFastCopy<T>(offset, len, copyAlignBytes)) {
        DataCopy(outputGm[offset], outputLocal, len);
    } else {
        CopyOutPad(outputGm, offset, outputLocal, len);
    }
}

class TileSchedule {
public:
    __aicore__ inline void Init(uint32_t totalLength, uint32_t tileLength)
    {
        tileLength_ = tileLength;
        if (totalLength == 0 || tileLength == 0) {
            tileNum_ = 0;
            lastTileLength_ = 0;
            return;
        }
        tileNum_ = (totalLength + tileLength - 1) / tileLength;
        lastTileLength_ = totalLength - (tileNum_ - 1) * tileLength;
    }

    __aicore__ inline uint32_t GetTileNum() const
    {
        return tileNum_;
    }

    __aicore__ inline uint32_t GetTileOffset(uint32_t tileIdx) const
    {
        return tileIdx * tileLength_;
    }

    __aicore__ inline uint32_t GetTileLength(uint32_t tileIdx) const
    {
        return (tileIdx + 1 == tileNum_) ? lastTileLength_ : tileLength_;
    }

private:
    uint32_t tileLength_ = 0;
    uint32_t tileNum_ = 0;
    uint32_t lastTileLength_ = 0;
};

__aicore__ inline void ComputeErfApproxFloat(LocalTensor<float>& erf, LocalTensor<float>& src,
                                             LocalTensor<float>& absSrc, LocalTensor<float>& work,
                                             uint32_t len)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;

    Mul<float, false>(absSrc, src, src, MASK_PLACEHOLDER, 1, binaryParams);
    Adds<float, false>(absSrc, absSrc, kSignEps, MASK_PLACEHOLDER, 1, unaryParams);
    Sqrt<float, false>(absSrc, absSrc, MASK_PLACEHOLDER, 1, unaryParams);

    Muls<float, false>(work, absSrc, 0.3275911f, MASK_PLACEHOLDER, 1, unaryParams);
    Adds<float, false>(work, work, 1.0f, MASK_PLACEHOLDER, 1, unaryParams);
    Muls<float, false>(erf, work, 0.0f, MASK_PLACEHOLDER, 1, unaryParams);
    Adds<float, false>(erf, erf, 1.0f, MASK_PLACEHOLDER, 1, unaryParams);
    Div<float, false>(work, erf, work, MASK_PLACEHOLDER, 1, binaryParams);

    // Abramowitz-Stegun erf approximation:
    // erf(x) ~= sign(x) * (1 - P(t) * exp(-abs(x)^2)), t = 1 / (1 + p * abs(x)).
    Muls<float, false>(erf, work, 1.061405429f, MASK_PLACEHOLDER, 1, unaryParams);
    Adds<float, false>(erf, erf, -1.453152027f, MASK_PLACEHOLDER, 1, unaryParams);
    // Keep work as exp(-abs(x)^2). Newton reuses it for the derivative.
    Mul<float, false>(erf, erf, work, MASK_PLACEHOLDER, 1, binaryParams);
    Adds<float, false>(erf, erf, 1.421413741f, MASK_PLACEHOLDER, 1, unaryParams);
    Mul<float, false>(erf, erf, work, MASK_PLACEHOLDER, 1, binaryParams);
    Adds<float, false>(erf, erf, -0.284496736f, MASK_PLACEHOLDER, 1, unaryParams);
    Mul<float, false>(erf, erf, work, MASK_PLACEHOLDER, 1, binaryParams);
    Adds<float, false>(erf, erf, 0.254829592f, MASK_PLACEHOLDER, 1, unaryParams);
    Mul<float, false>(erf, erf, work, MASK_PLACEHOLDER, 1, binaryParams);

    Mul<float, false>(work, absSrc, absSrc, MASK_PLACEHOLDER, 1, binaryParams);
    Muls<float, false>(work, work, -1.0f, MASK_PLACEHOLDER, 1, unaryParams);
    Exp<float, false>(work, work, MASK_PLACEHOLDER, 1, unaryParams);
    Mul<float, false>(erf, erf, work, MASK_PLACEHOLDER, 1, binaryParams);
    Muls<float, false>(erf, erf, -1.0f, MASK_PLACEHOLDER, 1, unaryParams);
    Adds<float, false>(erf, erf, 1.0f, MASK_PLACEHOLDER, 1, unaryParams);

    Div<float, false>(absSrc, src, absSrc, MASK_PLACEHOLDER, 1, binaryParams);
    Mul<float, false>(erf, erf, absSrc, MASK_PLACEHOLDER, 1, binaryParams);
}

__aicore__ inline void AddLutSegment(LocalTensor<float>& yAbs, LocalTensor<float>& absX,
                                     LocalTensor<float>& work, float x0, float width, float slope)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;

    Adds<float, false>(work, absX, -x0, MASK_PLACEHOLDER, 1, unaryParams);
    Maxs<float, false>(work, work, 0.0f, MASK_PLACEHOLDER, 1, unaryParams);
    Mins<float, false>(work, work, width, MASK_PLACEHOLDER, 1, unaryParams);
    Muls<float, false>(work, work, slope, MASK_PLACEHOLDER, 1, unaryParams);
    Add<float, false>(yAbs, yAbs, work, MASK_PLACEHOLDER, 1, binaryParams);
}

__aicore__ inline void ComputeLutInitFloat(LocalTensor<float>& y, LocalTensor<float>& target,
                                           LocalTensor<float>& initX, LocalTensor<float>& absX,
                                           LocalTensor<float>& work, uint32_t len)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;

    Maxs<float, false>(initX, target, -kLutMaxX, MASK_PLACEHOLDER, 1, unaryParams);
    Mins<float, false>(initX, initX, kLutMaxX, MASK_PLACEHOLDER, 1, unaryParams);
    Mul<float, false>(absX, initX, initX, MASK_PLACEHOLDER, 1, binaryParams);
    Adds<float, false>(absX, absX, kSignEps, MASK_PLACEHOLDER, 1, unaryParams);
    Sqrt<float, false>(absX, absX, MASK_PLACEHOLDER, 1, unaryParams);
    Muls<float, false>(y, absX, 0.0f, MASK_PLACEHOLDER, 1, unaryParams);

    // LUT linear interpolation in vector form:
    // y_abs = sum_i slope_i * clamp(abs(x) - x_i, 0, x_{i+1} - x_i).
    AddLutSegment(y, absX, work, 0.0f, 0.1f, 0.888559905f);
    AddLutSegment(y, absX, work, 0.1f, 0.1f, 0.902874641f);
    AddLutSegment(y, absX, work, 0.2f, 0.1f, 0.933192601f);
    AddLutSegment(y, absX, work, 0.3f, 0.1f, 0.983444439f);
    AddLutSegment(y, absX, work, 0.4f, 0.1f, 1.061291176f);
    AddLutSegment(y, absX, work, 0.5f, 0.1f, 1.181798052f);
    AddLutSegment(y, absX, work, 0.6f, 0.1f, 1.377529966f);
    AddLutSegment(y, absX, work, 0.7f, 0.1f, 1.733247244f);
    AddLutSegment(y, absX, work, 0.8f, 0.1f, 2.568933513f);
    AddLutSegment(y, absX, work, 0.9f, 0.05f, 4.456334114f);
    AddLutSegment(y, absX, work, 0.95f, 0.03f, 8.635763327f);
    AddLutSegment(y, absX, work, 0.98f, 0.01f, 17.641007900f);
    AddLutSegment(y, absX, work, 0.99f, 0.005f, 32.697224617f);
    AddLutSegment(y, absX, work, 0.995f, 0.004f, 85.470974445f);
    AddLutSegment(y, absX, work, 0.999f, 0.0009f, 471.421877543f);
    AddLutSegment(y, absX, work, 0.9999f, 0.00009f, 4135.274887083f);

    Div<float, false>(work, initX, absX, MASK_PLACEHOLDER, 1, binaryParams);
    Mul<float, false>(y, y, work, MASK_PLACEHOLDER, 1, binaryParams);
}

__aicore__ inline void ComputeLutInitDenseFloat(LocalTensor<float>& y, LocalTensor<float>& target,
                                                LocalTensor<float>& initX, LocalTensor<float>& absX,
                                                LocalTensor<float>& work, uint32_t len)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;

    Maxs<float, false>(initX, target, -kLutMaxX, MASK_PLACEHOLDER, 1, unaryParams);
    Mins<float, false>(initX, initX, kLutMaxX, MASK_PLACEHOLDER, 1, unaryParams);
    Mul<float, false>(absX, initX, initX, MASK_PLACEHOLDER, 1, binaryParams);
    Adds<float, false>(absX, absX, kSignEps, MASK_PLACEHOLDER, 1, unaryParams);
    Sqrt<float, false>(absX, absX, MASK_PLACEHOLDER, 1, unaryParams);
    Muls<float, false>(y, absX, 0.0f, MASK_PLACEHOLDER, 1, unaryParams);

    // Half/bfloat16 only need extra initial accuracy in the steep tail. Keeping
    // the main range sparse avoids spending more vector ops than a Newton step.
    AddLutSegment(y, absX, work, 0.0f, 0.1f, 0.888559905f);
    AddLutSegment(y, absX, work, 0.1f, 0.1f, 0.902874641f);
    AddLutSegment(y, absX, work, 0.2f, 0.1f, 0.933192601f);
    AddLutSegment(y, absX, work, 0.3f, 0.1f, 0.983444439f);
    AddLutSegment(y, absX, work, 0.4f, 0.1f, 1.061291176f);
    AddLutSegment(y, absX, work, 0.5f, 0.1f, 1.181798052f);
    AddLutSegment(y, absX, work, 0.6f, 0.1f, 1.377529966f);
    AddLutSegment(y, absX, work, 0.7f, 0.1f, 1.733247244f);
    AddLutSegment(y, absX, work, 0.8f, 0.1f, 2.568933513f);
    AddLutSegment(y, absX, work, 0.9f, 0.05f, 4.456334114f);
    AddLutSegment(y, absX, work, 0.95f, 0.03f, 8.635763327f);
    AddLutSegment(y, absX, work, 0.98f, 0.01f, 17.641007900f);
    AddLutSegment(y, absX, work, 0.99f, 0.005f, 32.69722462f);
    AddLutSegment(y, absX, work, 0.995f, 0.0025f, 61.18106842f);
    AddLutSegment(y, absX, work, 0.9975f, 0.0015f, 125.9541512f);
    AddLutSegment(y, absX, work, 0.999f, 0.0005f, 269.0014839f);
    AddLutSegment(y, absX, work, 0.9995f, 0.0004f, 724.4473696f);
    AddLutSegment(y, absX, work, 0.9999f, 0.00009f, 4135.274887f);

    Div<float, false>(work, initX, absX, MASK_PLACEHOLDER, 1, binaryParams);
    Mul<float, false>(y, y, work, MASK_PLACEHOLDER, 1, binaryParams);
}

template <int Steps, bool UseDenseLut>
__aicore__ inline void ComputeErfinvFloat(LocalTensor<float>& y, LocalTensor<float>& x, LocalTensor<float>& tmp1,
                                          LocalTensor<float>& tmp2, LocalTensor<float>& tmp3, uint32_t len)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, len);

    // Clamp in-place to avoid carrying a separate target buffer through Newton.
    // The original input is recast from inputLocal before special-value fixup.
    Maxs<float, false>(x, x, -1.0f + kDomainEps, MASK_PLACEHOLDER, 1, unaryParams);
    Mins<float, false>(x, x, 1.0f - kDomainEps, MASK_PLACEHOLDER, 1, unaryParams);
    if constexpr (UseDenseLut) {
        ComputeLutInitDenseFloat(y, x, tmp1, tmp2, tmp3, len);
    } else {
        ComputeLutInitFloat(y, x, tmp1, tmp2, tmp3, len);
    }

    for (int i = 0; i < Steps; ++i) {
        ComputeErfApproxFloat(tmp1, y, tmp2, tmp3, len);
        Sub<float, false>(tmp1, tmp1, x, MASK_PLACEHOLDER, 1, binaryParams);
        Div<float, false>(tmp1, tmp1, tmp3, MASK_PLACEHOLDER, 1, binaryParams);
        Muls<float, false>(tmp1, tmp1, kSqrtPiOver2, MASK_PLACEHOLDER, 1, unaryParams);
        Sub<float, false>(y, y, tmp1, MASK_PLACEHOLDER, 1, binaryParams);
    }

    SetMaskNorm();
    ResetMask();
}

__aicore__ inline void ComputeErfinvWinitzkiFloat(LocalTensor<float>& y, LocalTensor<float>& x,
                                                  LocalTensor<float>& tmp1, LocalTensor<float>& tmp2,
                                                  LocalTensor<float>& tmp3, LocalTensor<uint8_t>& mask,
                                                  uint32_t len)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, len);

    Mul<float, false>(tmp1, x, x, MASK_PLACEHOLDER, 1, binaryParams);
    Muls<float, false>(tmp1, tmp1, -1.0f, MASK_PLACEHOLDER, 1, unaryParams);
    Adds<float, false>(tmp1, tmp1, 1.0f, MASK_PLACEHOLDER, 1, unaryParams);
    Maxs<float, false>(tmp1, tmp1, kDomainEps, MASK_PLACEHOLDER, 1, unaryParams);
    Ln<float, false>(tmp1, tmp1, MASK_PLACEHOLDER, 1, unaryParams);

    Muls<float, false>(tmp2, tmp1, 0.5f, MASK_PLACEHOLDER, 1, unaryParams);
    Adds<float, false>(tmp2, tmp2, kWinitzkiC, MASK_PLACEHOLDER, 1, unaryParams);

    Mul<float, false>(tmp3, tmp2, tmp2, MASK_PLACEHOLDER, 1, binaryParams);
    Muls<float, false>(tmp1, tmp1, -1.0f / kWinitzkiA, MASK_PLACEHOLDER, 1, unaryParams);
    Add<float, false>(tmp3, tmp3, tmp1, MASK_PLACEHOLDER, 1, binaryParams);
    Sqrt<float, false>(tmp3, tmp3, MASK_PLACEHOLDER, 1, unaryParams);
    Sub<float, false>(tmp3, tmp3, tmp2, MASK_PLACEHOLDER, 1, binaryParams);
    Maxs<float, false>(tmp3, tmp3, 0.0f, MASK_PLACEHOLDER, 1, unaryParams);
    Sqrt<float, false>(y, tmp3, MASK_PLACEHOLDER, 1, unaryParams);

    Mul<float, false>(tmp1, x, x, MASK_PLACEHOLDER, 1, binaryParams);
    Adds<float, false>(tmp1, tmp1, kSignEps, MASK_PLACEHOLDER, 1, unaryParams);
    Sqrt<float, false>(tmp1, tmp1, MASK_PLACEHOLDER, 1, unaryParams);
    Adds<float, false>(tmp2, tmp1, -0.95f, MASK_PLACEHOLDER, 1, unaryParams);
    Maxs<float, false>(tmp2, tmp2, 0.0f, MASK_PLACEHOLDER, 1, unaryParams);
    Mins<float, false>(tmp2, tmp2, 0.05f, MASK_PLACEHOLDER, 1, unaryParams);
    Muls<float, false>(tmp2, tmp2, 0.04f, MASK_PLACEHOLDER, 1, unaryParams);
    Adds<float, false>(tmp2, tmp2, 1.0f, MASK_PLACEHOLDER, 1, unaryParams);
    Mul<float, false>(y, y, tmp2, MASK_PLACEHOLDER, 1, binaryParams);

    Muls<float, false>(tmp1, y, -1.0f, MASK_PLACEHOLDER, 1, unaryParams);
    uint32_t compareLen = AlignUpCount(len, 64U);
    CompareScalar(mask, x, 0.0f, CMPMODE::LT, compareLen);
    PipeBarrier<PIPE_V>();
    Select(y, mask, tmp1, y, SELMODE::VSEL_TENSOR_TENSOR_MODE, len);
    PipeBarrier<PIPE_V>();

    SetMaskNorm();
    ResetMask();
}

__aicore__ inline void ComputeErfinvGilesFloat(LocalTensor<float>& y, LocalTensor<float>& x,
                                               LocalTensor<float>& tmp1, LocalTensor<float>& tmp2,
                                               LocalTensor<float>& tmp3, LocalTensor<uint8_t>& mask,
                                               uint32_t len)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, len);

    Muls<float, false>(tmp1, x, -1.0f, MASK_PLACEHOLDER, 1, unaryParams);
    Adds<float, false>(tmp1, tmp1, 1.0f, MASK_PLACEHOLDER, 1, unaryParams);
    Adds<float, false>(tmp2, x, 1.0f, MASK_PLACEHOLDER, 1, unaryParams);
    Mul<float, false>(tmp1, tmp1, tmp2, MASK_PLACEHOLDER, 1, binaryParams);
    Maxs<float, false>(tmp1, tmp1, kSignEps, MASK_PLACEHOLDER, 1, unaryParams);
    Ln<float, false>(tmp1, tmp1, MASK_PLACEHOLDER, 1, unaryParams);
    Muls<float, false>(tmp1, tmp1, -1.0f, MASK_PLACEHOLDER, 1, unaryParams);

    // Mike Giles' single-precision erfinv approximation. It replaces the fp32
    // LUT + three Newton steps with one log, one sqrt, and two polynomial paths.
    Adds<float, false>(tmp2, tmp1, -2.5f, MASK_PLACEHOLDER, 1, unaryParams);
    Muls<float, false>(y, tmp2, 2.81022636e-08f, MASK_PLACEHOLDER, 1, unaryParams);
    Adds<float, false>(y, y, 3.43273939e-07f, MASK_PLACEHOLDER, 1, unaryParams);
    Mul<float, false>(y, y, tmp2, MASK_PLACEHOLDER, 1, binaryParams);
    Adds<float, false>(y, y, -3.5233877e-06f, MASK_PLACEHOLDER, 1, unaryParams);
    Mul<float, false>(y, y, tmp2, MASK_PLACEHOLDER, 1, binaryParams);
    Adds<float, false>(y, y, -4.39150654e-06f, MASK_PLACEHOLDER, 1, unaryParams);
    Mul<float, false>(y, y, tmp2, MASK_PLACEHOLDER, 1, binaryParams);
    Adds<float, false>(y, y, 0.00021858087f, MASK_PLACEHOLDER, 1, unaryParams);
    Mul<float, false>(y, y, tmp2, MASK_PLACEHOLDER, 1, binaryParams);
    Adds<float, false>(y, y, -0.00125372503f, MASK_PLACEHOLDER, 1, unaryParams);
    Mul<float, false>(y, y, tmp2, MASK_PLACEHOLDER, 1, binaryParams);
    Adds<float, false>(y, y, -0.00417768164f, MASK_PLACEHOLDER, 1, unaryParams);
    Mul<float, false>(y, y, tmp2, MASK_PLACEHOLDER, 1, binaryParams);
    Adds<float, false>(y, y, 0.246640727f, MASK_PLACEHOLDER, 1, unaryParams);
    Mul<float, false>(y, y, tmp2, MASK_PLACEHOLDER, 1, binaryParams);
    Adds<float, false>(y, y, 1.50140941f, MASK_PLACEHOLDER, 1, unaryParams);

    Sqrt<float, false>(tmp2, tmp1, MASK_PLACEHOLDER, 1, unaryParams);
    Adds<float, false>(tmp2, tmp2, -3.0f, MASK_PLACEHOLDER, 1, unaryParams);
    Muls<float, false>(tmp3, tmp2, -0.000200214257f, MASK_PLACEHOLDER, 1, unaryParams);
    Adds<float, false>(tmp3, tmp3, 0.000100950558f, MASK_PLACEHOLDER, 1, unaryParams);
    Mul<float, false>(tmp3, tmp3, tmp2, MASK_PLACEHOLDER, 1, binaryParams);
    Adds<float, false>(tmp3, tmp3, 0.00134934322f, MASK_PLACEHOLDER, 1, unaryParams);
    Mul<float, false>(tmp3, tmp3, tmp2, MASK_PLACEHOLDER, 1, binaryParams);
    Adds<float, false>(tmp3, tmp3, -0.00367342844f, MASK_PLACEHOLDER, 1, unaryParams);
    Mul<float, false>(tmp3, tmp3, tmp2, MASK_PLACEHOLDER, 1, binaryParams);
    Adds<float, false>(tmp3, tmp3, 0.00573950773f, MASK_PLACEHOLDER, 1, unaryParams);
    Mul<float, false>(tmp3, tmp3, tmp2, MASK_PLACEHOLDER, 1, binaryParams);
    Adds<float, false>(tmp3, tmp3, -0.0076224613f, MASK_PLACEHOLDER, 1, unaryParams);
    Mul<float, false>(tmp3, tmp3, tmp2, MASK_PLACEHOLDER, 1, binaryParams);
    Adds<float, false>(tmp3, tmp3, 0.00943887047f, MASK_PLACEHOLDER, 1, unaryParams);
    Mul<float, false>(tmp3, tmp3, tmp2, MASK_PLACEHOLDER, 1, binaryParams);
    Adds<float, false>(tmp3, tmp3, 1.00167406f, MASK_PLACEHOLDER, 1, unaryParams);
    Mul<float, false>(tmp3, tmp3, tmp2, MASK_PLACEHOLDER, 1, binaryParams);
    Adds<float, false>(tmp3, tmp3, 2.83297682f, MASK_PLACEHOLDER, 1, unaryParams);

    Mul<float, false>(tmp2, y, x, MASK_PLACEHOLDER, 1, binaryParams);
    Adds<float, false>(y, tmp2, 0.0f, MASK_PLACEHOLDER, 1, unaryParams);
    Mul<float, false>(tmp2, tmp3, x, MASK_PLACEHOLDER, 1, binaryParams);
    Adds<float, false>(tmp3, tmp2, 0.0f, MASK_PLACEHOLDER, 1, unaryParams);

    uint32_t compareLen = AlignUpCount(len, 64U);
    CompareScalar(mask, tmp1, 5.0f, CMPMODE::GE, compareLen);
    PipeBarrier<PIPE_V>();
    Select(y, mask, tmp3, y, SELMODE::VSEL_TENSOR_TENSOR_MODE, len);

    SetMaskNorm();
    ResetMask();
}

__aicore__ inline void FixSpecialValuesInRange(LocalTensor<float>& y, LocalTensor<float>& x,
                                               LocalTensor<uint8_t>& mask, LocalTensor<float>& special,
                                               uint32_t len)
{
    uint32_t compareLen = AlignUpCount(len, 64U);
    float zero = 0.0f;
    float posInf = 1.0f / zero;
    float negInf = -1.0f / zero;

    Duplicate(special, posInf, len);
    CompareScalar(mask, x, 1.0f, CMPMODE::EQ, compareLen);
    PipeBarrier<PIPE_V>();
    Select(y, mask, special, y, SELMODE::VSEL_TENSOR_TENSOR_MODE, len);

    Duplicate(special, negInf, len);
    CompareScalar(mask, x, -1.0f, CMPMODE::EQ, compareLen);
    PipeBarrier<PIPE_V>();
    Select(y, mask, special, y, SELMODE::VSEL_TENSOR_TENSOR_MODE, len);
}

__aicore__ inline void FixSpecialValues(LocalTensor<float>& y, LocalTensor<float>& x,
                                        LocalTensor<uint8_t>& mask, LocalTensor<float>& special, uint32_t len)
{
    uint32_t compareLen = AlignUpCount(len, 64U);
    float zero = 0.0f;
    float posInf = 1.0f / zero;
    float negInf = -1.0f / zero;
    float nanValue = zero / zero;

    Duplicate(special, posInf, len);
    PipeBarrier<PIPE_V>();
    CompareScalar(mask, x, 1.0f, CMPMODE::EQ, compareLen);
    PipeBarrier<PIPE_V>();
    Select(y, mask, special, y, SELMODE::VSEL_TENSOR_TENSOR_MODE, len);
    PipeBarrier<PIPE_V>();

    Duplicate(special, negInf, len);
    PipeBarrier<PIPE_V>();
    CompareScalar(mask, x, -1.0f, CMPMODE::EQ, compareLen);
    PipeBarrier<PIPE_V>();
    Select(y, mask, special, y, SELMODE::VSEL_TENSOR_TENSOR_MODE, len);
    PipeBarrier<PIPE_V>();

    Duplicate(special, nanValue, len);
    PipeBarrier<PIPE_V>();
    CompareScalar(mask, x, 1.0f, CMPMODE::GT, compareLen);
    PipeBarrier<PIPE_V>();
    Select(y, mask, special, y, SELMODE::VSEL_TENSOR_TENSOR_MODE, len);
    PipeBarrier<PIPE_V>();

    CompareScalar(mask, x, -1.0f, CMPMODE::LT, compareLen);
    PipeBarrier<PIPE_V>();
    Select(y, mask, special, y, SELMODE::VSEL_TENSOR_TENSOR_MODE, len);
    PipeBarrier<PIPE_V>();

    Compare(mask, x, x, CMPMODE::LE, compareLen);
    PipeBarrier<PIPE_V>();
    Select(y, mask, y, special, SELMODE::VSEL_TENSOR_TENSOR_MODE, len);
    PipeBarrier<PIPE_V>();
}

template <typename T>
class KernelErfinv {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t blockOffset, uint32_t currentBlockLength,
                                uint32_t totalLength, uint32_t tileLength, uint32_t copyAlignBytes)
    {
        tileLength_ = tileLength;
        totalLength_ = totalLength;
        copyAlignBytes_ = copyAlignBytes;
        tileSchedule_.Init(currentBlockLength, tileLength);
        inputGm_.SetGlobalBuffer((__gm__ T*)input + blockOffset, currentBlockLength);
        outputGm_.SetGlobalBuffer((__gm__ T*)output + blockOffset, currentBlockLength);

        const uint32_t rawBufferSize = AlignUpBytes(tileLength_ * sizeof(T));
        const uint32_t floatBufferSize = AlignUpBytes(tileLength_ * sizeof(float));
        pipe_.InitBuffer(inputQueue_, kBufferNum, rawBufferSize);
        pipe_.InitBuffer(outputQueue_, kBufferNum, rawBufferSize);
        pipe_.InitBuffer(xBuf_, floatBufferSize);
        pipe_.InitBuffer(yBuf_, floatBufferSize);
        pipe_.InitBuffer(tmp1Buf_, floatBufferSize);
        pipe_.InitBuffer(tmp2Buf_, floatBufferSize);
        pipe_.InitBuffer(tmp3Buf_, floatBufferSize);
        pipe_.InitBuffer(maskBuf_, AlignUpBytes(AlignUpCount(tileLength_, 64U)));
    }

    __aicore__ inline void Process()
    {
        uint32_t tileNum = tileSchedule_.GetTileNum();
        if (tileNum == 0) {
            return;
        }

        CopyInTile(0);
        for (uint32_t tileIdx = 0; tileIdx < tileNum; ++tileIdx) {
            uint32_t offset = tileSchedule_.GetTileOffset(tileIdx);
            uint32_t len = tileSchedule_.GetTileLength(tileIdx);
            LocalTensor<T> inputLocal = inputQueue_.DeQue<T>();

            if (tileIdx + 1 < tileNum) {
                CopyInTile(tileIdx + 1);
            }

            LocalTensor<T> outputLocal = outputQueue_.AllocTensor<T>();
            ComputeLocal(inputLocal, outputLocal, len);
            outputQueue_.EnQue(outputLocal);
            inputQueue_.FreeTensor(inputLocal);

            outputLocal = outputQueue_.DeQue<T>();
            CopyOutAuto(outputGm_, offset, outputLocal, len, copyAlignBytes_);
            outputQueue_.FreeTensor(outputLocal);
        }
    }

private:
    __aicore__ inline void CopyInTile(uint32_t tileIdx)
    {
        uint32_t offset = tileSchedule_.GetTileOffset(tileIdx);
        uint32_t len = tileSchedule_.GetTileLength(tileIdx);
        LocalTensor<T> inputLocal = inputQueue_.AllocTensor<T>();
        CopyInAuto(inputGm_, offset, inputLocal, len, copyAlignBytes_);
        inputQueue_.EnQue(inputLocal);
    }

    __aicore__ inline void ComputeLocal(LocalTensor<T>& inputLocal, LocalTensor<T>& outputLocal, uint32_t len)
    {
        LocalTensor<float> xLocal = xBuf_.Get<float>();
        if constexpr (std::is_same_v<T, float>) {
            Adds(xLocal, inputLocal, 0.0f, len);
        } else {
            Cast(xLocal, inputLocal, RoundMode::CAST_NONE, len);
        }

        LocalTensor<float> yLocal = yBuf_.Get<float>();
        LocalTensor<float> tmp1Local = tmp1Buf_.Get<float>();
        LocalTensor<float> tmp2Local = tmp2Buf_.Get<float>();
        LocalTensor<float> tmp3Local = tmp3Buf_.Get<float>();
        LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();
        if constexpr (std::is_same_v<T, float>) {
            if (totalLength_ >= kFp32GilesMinTotalLength) {
                ComputeErfinvGilesFloat(yLocal, xLocal, tmp1Local, tmp2Local, tmp3Local, maskLocal, len);
            } else {
                ComputeErfinvFloat<3, false>(yLocal, xLocal, tmp1Local, tmp2Local, tmp3Local, len);
            }
        } else if constexpr (std::is_same_v<T, half>) {
            ComputeErfinvWinitzkiFloat(yLocal, xLocal, tmp1Local, tmp2Local, tmp3Local, maskLocal, len);
        } else {
            ComputeErfinvGilesFloat(yLocal, xLocal, tmp1Local, tmp2Local, tmp3Local, maskLocal, len);
        }

        if constexpr (std::is_same_v<T, half>) {
            if (totalLength_ < kFastSpecialMinTotalLength) {
                Cast(xLocal, inputLocal, RoundMode::CAST_NONE, len);
                FixSpecialValues(yLocal, xLocal, maskLocal, tmp3Local, len);
            }
        } else if constexpr (std::is_same_v<T, float>) {
            if (totalLength_ < kFp32GilesMinTotalLength) {
                Adds(xLocal, inputLocal, 0.0f, len);
                FixSpecialValues(yLocal, xLocal, maskLocal, tmp3Local, len);
            }
        } else {
            Cast(xLocal, inputLocal, RoundMode::CAST_NONE, len);
            FixSpecialValues(yLocal, xLocal, maskLocal, tmp3Local, len);
        }

        if constexpr (std::is_same_v<T, float>) {
            Adds(outputLocal, yLocal, 0.0f, len);
        } else {
            Cast(outputLocal, yLocal, RoundMode::CAST_ROUND, len);
        }
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, kBufferNum> inputQueue_;
    TQue<QuePosition::VECOUT, kBufferNum> outputQueue_;
    TBuf<QuePosition::VECCALC> xBuf_;
    TBuf<QuePosition::VECCALC> yBuf_;
    TBuf<QuePosition::VECCALC> tmp1Buf_;
    TBuf<QuePosition::VECCALC> tmp2Buf_;
    TBuf<QuePosition::VECCALC> tmp3Buf_;
    TBuf<QuePosition::VECCALC> maskBuf_;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
    uint32_t tileLength_ = 0;
    uint32_t totalLength_ = 0;
    uint32_t copyAlignBytes_ = kAlignBytes;
    TileSchedule tileSchedule_;
};

template <typename T>
__aicore__ inline void RunErfinv(GM_ADDR input, GM_ADDR output, uint32_t totalLength, uint32_t blockLength,
                                 uint32_t lastBlockLength, uint32_t tileLength, uint32_t copyAlignBytes)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockOffset = blockLength * blockIdx;
    uint32_t currentBlockLength = (blockIdx + 1 == GetBlockNum()) ? lastBlockLength : blockLength;
    KernelErfinv<T> kernel;
    kernel.Init(input, output, blockOffset, currentBlockLength, totalLength, tileLength, copyAlignBytes);
    kernel.Process();
}

}  // namespace

extern "C" __global__ __aicore__ void erfinv(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(0)) {
        RunErfinv<half>(input, output, tilingData.totalLength, tilingData.blockLength, tilingData.lastBlockLength,
                        tilingData.tileLength, tilingData.copyAlignBytes);
    } else if (TILING_KEY_IS(1)) {
        RunErfinv<bfloat16_t>(input, output, tilingData.totalLength, tilingData.blockLength, tilingData.lastBlockLength,
                              tilingData.tileLength, tilingData.copyAlignBytes);
    } else if (TILING_KEY_IS(2)) {
        RunErfinv<float>(input, output, tilingData.totalLength, tilingData.blockLength, tilingData.lastBlockLength,
                         tilingData.tileLength, tilingData.copyAlignBytes);
    }
}
