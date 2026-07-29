#include "../op_host/gelu_v2_tiling.h"
#include "kernel_operator.h"
#include <type_traits>

using namespace AscendC;

namespace {
constexpr uint32_t BUFFER_NUM = 2;
constexpr float GELU_CUBIC_COEF = 0.044715f;
constexpr float GELU_NEG_SQRT_EIGHT_OVER_PI = -1.5957691216057308f;
constexpr float GELU_ERF_FIT4_PARAM1 = -1.389389187e-7f;
constexpr float GELU_ERF_FIT4_PARAM2 = 0.0007362028982f;
constexpr float GELU_ERF_FIT4_PARAM3 = -0.07400324227f;
constexpr float GELU_ERF_FIT4_PARAM4 = -1.595424403f;
constexpr float GELU_ERF_FIT5_PARAM1 = -9.87661083e-7f;
constexpr float GELU_ERF_FIT5_PARAM2 = 3.16859727e-5f;
constexpr float GELU_ERF_FIT5_PARAM3 = 0.000443164981f;
constexpr float GELU_ERF_FIT5_PARAM4 = -0.0732606772f;
constexpr float GELU_ERF_FIT5_PARAM5 = -1.59564588f;
}

template <typename T, bool APPROXIMATE_TANH>
class KernelGeluV2 {
public:
    __aicore__ inline KernelGeluV2() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t coreDataNum,
                                uint32_t globalOffset, uint32_t tileLength)
    {
        InitSocState();
        this->coreDataNum = coreDataNum;
        this->tileLength = tileLength;
        xGm.SetGlobalBuffer((__gm__ T*)x + globalOffset, coreDataNum);
        yGm.SetGlobalBuffer((__gm__ T*)y + globalOffset, coreDataNum);
    }

    __aicore__ inline void Process()
    {
        if (coreDataNum == 0) {
            return;
        }

        uint32_t ioTileBytes = tileLength * sizeof(T);
        uint32_t calcTileBytes = tileLength * sizeof(float);
        uint32_t calcBase = 2 * BUFFER_NUM * ioTileBytes;

        LocalTensor<T> xLocal[BUFFER_NUM] = {
            LocalTensor<T>(TPosition::VECIN, 0, tileLength),
            LocalTensor<T>(TPosition::VECIN, 2 * ioTileBytes, tileLength),
        };
        LocalTensor<T> yLocal[BUFFER_NUM] = {
            LocalTensor<T>(TPosition::VECOUT, ioTileBytes, tileLength),
            LocalTensor<T>(TPosition::VECOUT, 3 * ioTileBytes, tileLength),
        };
        LocalTensor<float> calc0(TPosition::VECCALC, calcBase, tileLength);
        LocalTensor<float> calc1(TPosition::VECCALC, calcBase + calcTileBytes, tileLength);
        LocalTensor<float> calc2(TPosition::VECCALC, calcBase + 2 * calcTileBytes, tileLength);

        uint32_t totalTiles = (coreDataNum + tileLength - 1) / tileLength;
        for (uint32_t tileBase = 0; tileBase < totalTiles; tileBase += BUFFER_NUM) {
            for (uint32_t j = 0; j < BUFFER_NUM; ++j) {
                uint32_t tileIndex = tileBase + j;
                if (tileIndex >= totalTiles) {
                    break;
                }

                uint32_t offset = tileIndex * tileLength;
                uint32_t validLen = coreDataNum - offset;
                if (validLen > tileLength) {
                    validLen = tileLength;
                }
                constexpr uint32_t alignNum = 32 / sizeof(T);
                uint32_t alignedLen = (validLen + alignNum - 1) / alignNum * alignNum;
                bool waitBackward = tileIndex >= BUFFER_NUM;
                bool setBackward = tileIndex + BUFFER_NUM < totalTiles;

                if (waitBackward) {
                    WaitFlag<HardEvent::V_MTE2>(j);
                }
                if (validLen == alignedLen) {
                    DataCopy(xLocal[j], xGm[offset], validLen);
                } else {
                    DataCopyExtParams copyInParams{
                        1, static_cast<uint32_t>(validLen * sizeof(T)), 0, 0, 0};
                    DataCopyPadExtParams<T> padParams{
                        true, 0, static_cast<uint8_t>(alignedLen - validLen), static_cast<T>(0)};
                    DataCopyPad(xLocal[j], xGm[offset], copyInParams, padParams);
                }
                SetFlag<HardEvent::MTE2_V>(j);

                if (waitBackward) {
                    WaitFlag<HardEvent::MTE3_V>(j);
                }
                WaitFlag<HardEvent::MTE2_V>(j);
                if constexpr (std::is_same_v<T, float>) {
                    LocalTensor<float> xFloat = xLocal[j];
                    LocalTensor<float> yFloat = yLocal[j];

                    if constexpr (APPROXIMATE_TANH) {
                        ComputeTanh(xFloat, calc0, yFloat, alignedLen);
                    } else {
                        ComputeGeluErf4Coeff(yFloat, xFloat, calc0, alignedLen);
                    }
                    if (setBackward) {
                        SetFlag<HardEvent::V_MTE2>(j);
                    }
                } else {
                    Cast(calc0, xLocal[j], RoundMode::CAST_NONE, alignedLen);
                    if (setBackward) {
                        SetFlag<HardEvent::V_MTE2>(j);
                    }

                    if constexpr (APPROXIMATE_TANH) {
                        ComputeTanh(calc0, calc1, calc0, alignedLen);
                        if constexpr (std::is_same_v<T, bfloat16_t>) {
                            Cast(yLocal[j], calc0, RoundMode::CAST_RINT, alignedLen);
                        } else {
                            Cast(yLocal[j], calc0, RoundMode::CAST_ROUND, alignedLen);
                        }
                    } else {
                        ComputeGeluErf5Coeff(calc2, calc0, calc1, alignedLen);

                        if constexpr (std::is_same_v<T, bfloat16_t>) {
                            Cast(yLocal[j], calc2, RoundMode::CAST_RINT, alignedLen);
                        } else {
                            Cast(yLocal[j], calc2, RoundMode::CAST_ROUND, alignedLen);
                        }
                    }
                }

                SetFlag<HardEvent::V_MTE3>(j);
                WaitFlag<HardEvent::V_MTE3>(j);
                if (validLen == alignedLen) {
                    DataCopy(yGm[offset], yLocal[j], validLen);
                } else {
                    DataCopyExtParams copyOutParams{
                        1, static_cast<uint32_t>(validLen * sizeof(T)), 0, 0, 0};
                    DataCopyPad(yGm[offset], yLocal[j], copyOutParams);
                }
                if (setBackward) {
                    SetFlag<HardEvent::MTE3_V>(j);
                }
            }
        }
    }

private:
    __aicore__ inline void ComputeTanh(LocalTensor<float>& xFloat,
                                       LocalTensor<float>& tmp0,
                                       LocalTensor<float>& tmp1,
                                       uint32_t len)
    {
        Mul(tmp0, xFloat, xFloat, len);
        Muls(tmp0, tmp0, GELU_CUBIC_COEF, len);
        FusedMulAdd(tmp0, xFloat, xFloat, len);
        Muls(tmp0, tmp0, GELU_NEG_SQRT_EIGHT_OVER_PI, len);
        Exp(tmp0, tmp0, len);
        Adds(tmp0, tmp0, 1.0f, len);
        Div(tmp1, xFloat, tmp0, len);
    }

    __aicore__ inline void ComputeGeluErf4Coeff(LocalTensor<float>& out,
                                                LocalTensor<float>& x,
                                                LocalTensor<float>& xPow,
                                                uint32_t len)
    {
        Mul(xPow, x, x, len);
        Muls(out, xPow, GELU_ERF_FIT4_PARAM1, len);
        Adds(out, out, GELU_ERF_FIT4_PARAM2, len);
        Mul(out, out, xPow, len);
        Adds(out, out, GELU_ERF_FIT4_PARAM3, len);
        Mul(out, out, xPow, len);
        Adds(out, out, GELU_ERF_FIT4_PARAM4, len);
        Mul(out, out, x, len);
        Exp(out, out, len);
        Adds(out, out, 1.0f, len);
        Div(out, x, out, len);
    }

    __aicore__ inline void ComputeGeluErf5Coeff(LocalTensor<float>& out,
                                                LocalTensor<float>& x,
                                                LocalTensor<float>& xPow,
                                                uint32_t len)
    {
        Mul(xPow, x, x, len);
        Muls(out, xPow, GELU_ERF_FIT5_PARAM1, len);
        Adds(out, out, GELU_ERF_FIT5_PARAM2, len);
        Mul(out, out, xPow, len);
        Adds(out, out, GELU_ERF_FIT5_PARAM3, len);
        Mul(out, out, xPow, len);
        Adds(out, out, GELU_ERF_FIT5_PARAM4, len);
        Mul(out, out, xPow, len);
        Adds(out, out, GELU_ERF_FIT5_PARAM5, len);
        Mul(out, out, x, len);
        Exp(out, out, len);
        Adds(out, out, 1.0f, len);
        Div(out, x, out, len);
    }

private:
    uint32_t coreDataNum;
    uint32_t tileLength;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
};

extern "C" __global__ __aicore__ void gelu_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    // 手写 1 次 64-bit GM load 替掉 GET_TILING_DATA 自动生成的多次 32-bit load。
    // GeluV2TilingData 布局：[0..3]=totalLength, [4..7]=coreDataNum, [8..11]=tileLength
    uint64_t packed = *((const __gm__ uint64_t*)tiling);
    uint32_t totalLength = static_cast<uint32_t>(packed);
    uint32_t coreDataNum = static_cast<uint32_t>(packed >> 32);
    uint32_t tileLength = *((const __gm__ uint32_t*)tiling + 2);

    uint32_t globalOffset = GetBlockIdx() * coreDataNum;
    if (globalOffset >= totalLength) {
        coreDataNum = 0;
    } else {
        uint32_t remaining = totalLength - globalOffset;
        if (remaining < coreDataNum) {
            coreDataNum = remaining;
        }
    }

    using input_t = DTYPE_X;
    if (TILING_KEY_IS(1)) {
        KernelGeluV2<input_t, true> op;
        op.Init(x, y, coreDataNum, globalOffset, tileLength);
        op.Process();
    } else if (TILING_KEY_IS(0)) {
        KernelGeluV2<input_t, false> op;
        op.Init(x, y, coreDataNum, globalOffset, tileLength);
        op.Process();
    }
}
