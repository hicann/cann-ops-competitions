/*!
 * \file selu.cpp
 * \brief Selu 算子 kernel 入口
 */

#include "selu.h"

enum class SeluTilingKey : uint32_t
{
    TILING_KEY_SELU_MODE_0 = 0,
    TILING_KEY_SELU_MODE_1 = 1,
    TILING_KEY_SELU_MODE_2 = 2,
    TILING_KEY_SELU_MODE_3 = 3,
    TILING_KEY_SELU_MODE_4 = 4,
    TILING_KEY_SELU_MODE_5 = 5,
    TILING_KEY_SELU_MODE_6 = 6,
    TILING_KEY_SELU_MODE_7 = 7,
    TILING_KEY_SELU_MODE_8 = 8,
};

template <uint32_t schMode>
__global__ __aicore__ void selu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SeluTilingData);
    if constexpr (schMode == static_cast<uint32_t>(SeluTilingKey::TILING_KEY_SELU_MODE_2)) {
        NsSelu::SeluScalarNegative op;
        op.InitAndProcess(x, y);
        return;
    }
    if constexpr (schMode == static_cast<uint32_t>(SeluTilingKey::TILING_KEY_SELU_MODE_8)) {
        NsSelu::SeluFixed15Scalar op;
        op.InitAndProcess(x, y);
        return;
    }
    GET_TILING_DATA_WITH_STRUCT(SeluTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(SeluTilingKey::TILING_KEY_SELU_MODE_0)) {
        NsSelu::Selu<float, NsSelu::DOUBLE_BUFFER_NUM, true, false, true> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(SeluTilingKey::TILING_KEY_SELU_MODE_1)) {
        NsSelu::Selu<float, NsSelu::SINGLE_BUFFER_NUM> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(SeluTilingKey::TILING_KEY_SELU_MODE_3)) {
        NsSelu::Selu<float, NsSelu::SINGLE_BUFFER_NUM, true> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(SeluTilingKey::TILING_KEY_SELU_MODE_4)) {
        NsSelu::SeluSmallScalar op;
        op.InitAndProcess(x, y, &tilingData);
    }
    if constexpr (schMode == static_cast<uint32_t>(SeluTilingKey::TILING_KEY_SELU_MODE_5)) {
        NsSelu::Selu<float, NsSelu::DOUBLE_BUFFER_NUM> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(SeluTilingKey::TILING_KEY_SELU_MODE_6)) {
        NsSelu::Selu<float, NsSelu::DOUBLE_BUFFER_NUM, true, false> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(SeluTilingKey::TILING_KEY_SELU_MODE_7)) {
        NsSelu::Selu<float, NsSelu::DOUBLE_BUFFER_NUM, true, true> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
