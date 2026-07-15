/*!
 * \file hard_swish.cpp
 * \brief HardSwish 算子 kernel 入口
 */

#include "hard_swish.h"

using namespace AscendC;

constexpr uint32_t TINY_DIRECT_NUM = 15;
constexpr uint32_t TINY_DIRECT_CALC_NUM = 16;
constexpr uint32_t TINY_X_ADDR = 0;
constexpr uint32_t TINY_Y_ADDR = TINY_DIRECT_CALC_NUM * sizeof(float);
constexpr int32_t EVENT_ID_MTE2_V = 0;
constexpr int32_t EVENT_ID_V_MTE3 = 1;

enum class HardSwishTilingKey : uint32_t
{
    TILING_KEY_HARDSWISH_MODE_0 = 0,
    TILING_KEY_HARDSWISH_MODE_1 = 1,
    TILING_KEY_HARDSWISH_MODE_2 = 2,
    TILING_KEY_HARDSWISH_MODE_3 = 3,
};

__aicore__ inline float CalcHardSwishScalar(float xVal)
{
    float relu6 = xVal + 3.0f;
    if (relu6 < 0.0f)
    {
        relu6 = 0.0f;
    }
    if (relu6 > 6.0f)
    {
        relu6 = 6.0f;
    }
    return xVal * relu6 * 0.16666666666666666f;
}

__aicore__ inline void StoreHardSwishScalar(__gm__ float *inputGMX, __gm__ float *outputGMY, uint32_t index)
{
    const float xVal = inputGMX[index];
    outputGMY[index] = CalcHardSwishScalar(xVal);
}

__aicore__ inline void StoreHardSwishScalar(GlobalTensor<float> &inputGMX, GlobalTensor<float> &outputGMY, uint32_t index)
{
    const float xVal = inputGMX.GetValue(index);
    outputGMY.SetValue(index, CalcHardSwishScalar(xVal));
}

__aicore__ inline void ProcessHardSwishTinyDirect(GM_ADDR x, GM_ADDR y)
{
    GlobalTensor<float> inputGMX;
    GlobalTensor<float> outputGMY;
    inputGMX.SetGlobalBuffer((__gm__ float *)x, TINY_DIRECT_NUM);
    outputGMY.SetGlobalBuffer((__gm__ float *)y, TINY_DIRECT_CALC_NUM);

    LocalTensor<float> xLocal(TPosition::VECIN, TINY_X_ADDR, TINY_DIRECT_CALC_NUM);
    LocalTensor<float> yLocal(TPosition::VECCALC, TINY_Y_ADDR, TINY_DIRECT_CALC_NUM);

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(TINY_DIRECT_NUM * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
    DataCopyPad(xLocal, inputGMX[0], copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_V);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_V);

    Adds(yLocal, xLocal, 3.0f, TINY_DIRECT_CALC_NUM);
    Maxs(yLocal, yLocal, 0.0f, TINY_DIRECT_CALC_NUM);
    Mins(yLocal, yLocal, 6.0f, TINY_DIRECT_CALC_NUM);
    Mul(yLocal, yLocal, xLocal, TINY_DIRECT_CALC_NUM);
    Muls(yLocal, yLocal, 0.16666666666666666f, TINY_DIRECT_CALC_NUM);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID_V_MTE3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID_V_MTE3);

    DataCopy(outputGMY[0], yLocal, TINY_DIRECT_CALC_NUM);
}

__aicore__ inline void ProcessHardSwishScalarMode0(GM_ADDR x, GM_ADDR y)
{
    GlobalTensor<float> inputGMX;
    GlobalTensor<float> outputGMY;
    inputGMX.SetGlobalBuffer((__gm__ float*)x, 1);
    outputGMY.SetGlobalBuffer((__gm__ float*)y, 1);

    float xVal = inputGMX.GetValue(0);
    float yVal = xVal;
    if (xVal < 3.0f) {
        float relu6 = xVal + 3.0f;
        if (relu6 < 0.0f) {
            relu6 = 0.0f;
        }
        yVal = xVal * relu6 * 0.16666666666666666f;
    }
    outputGMY.SetValue(0, yVal);
}

template <uint32_t schMode>
__global__ __aicore__ void hard_swish(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (schMode == static_cast<uint32_t>(HardSwishTilingKey::TILING_KEY_HARDSWISH_MODE_0)) {
        REGISTER_TILING_DEFAULT(HardSwishTilingData);
        GET_TILING_DATA_WITH_STRUCT(HardSwishTilingData, tilingData, tiling);
        TPipe pipe;
        NsHardSwish::HardSwish<float, false> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(HardSwishTilingKey::TILING_KEY_HARDSWISH_MODE_1)) {
        ProcessHardSwishScalarMode0(x, y);
    } else if constexpr (schMode == static_cast<uint32_t>(HardSwishTilingKey::TILING_KEY_HARDSWISH_MODE_2)) {
        ProcessHardSwishTinyDirect(x, y);
    } else if constexpr (schMode == static_cast<uint32_t>(HardSwishTilingKey::TILING_KEY_HARDSWISH_MODE_3)) {
        ProcessHardSwishScalarMode0(x, y);
    }
}
