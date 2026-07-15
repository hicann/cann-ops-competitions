/*!
 * \file celu.cpp
 * \brief Celu 算子 kernel 入口
 */

#include "celu.h"

enum class CeluTilingKey : uint32_t
{
    TILING_KEY_CELU_MODE_0 = 0,
    TILING_KEY_CELU_MODE_1 = 1,
    TILING_KEY_CELU_MODE_2 = 2,
    TILING_KEY_CELU_MODE_3 = 3,
    TILING_KEY_CELU_MODE_4 = 4,
    TILING_KEY_CELU_MODE_5 = 5,
    TILING_KEY_CELU_MODE_6 = 6,
    TILING_KEY_CELU_MODE_7 = 7,
    TILING_KEY_CELU_MODE_8 = 8,
    TILING_KEY_CELU_MODE_9 = 9,
    TILING_KEY_CELU_MODE_10 = 10,
    TILING_KEY_CELU_MODE_11 = 11,
    TILING_KEY_CELU_MODE_12 = 12,
};

template <uint32_t schMode>
__global__ __aicore__ void celu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CeluTilingData);
    if constexpr (schMode == static_cast<uint32_t>(CeluTilingKey::TILING_KEY_CELU_MODE_4)) {
        NsCelu::CeluScalarPolyOne op;
        op.InitAndProcess(x, y);
        return;
    }
    if constexpr (schMode == static_cast<uint32_t>(CeluTilingKey::TILING_KEY_CELU_MODE_12)) {
        NsCelu::CeluTinyScalarAlpha op;
        op.InitAndProcess(x, y, 15, 0.5f);
        return;
    }
    GET_TILING_DATA_WITH_STRUCT(CeluTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(CeluTilingKey::TILING_KEY_CELU_MODE_0)) {
        NsCelu::Celu<half, NsCelu::DOUBLE_BUFFER_NUM> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(CeluTilingKey::TILING_KEY_CELU_MODE_1)) {
        NsCelu::Celu<float, NsCelu::SINGLE_BUFFER_NUM> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(CeluTilingKey::TILING_KEY_CELU_MODE_2)) {
        NsCelu::CeluScalar<float> op;
        op.InitAndProcess(x, y, &tilingData);
    }
    if constexpr (schMode == static_cast<uint32_t>(CeluTilingKey::TILING_KEY_CELU_MODE_3)) {
        NsCelu::Celu<float, NsCelu::SINGLE_BUFFER_NUM, true> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(CeluTilingKey::TILING_KEY_CELU_MODE_5)) {
        NsCelu::Celu<
            float,
            NsCelu::DOUBLE_BUFFER_NUM,
            false,
            false,
            NsCelu::NON_PIPELINE_DOUBLE_TQUE_DEPTH> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(CeluTilingKey::TILING_KEY_CELU_MODE_6)) {
        NsCelu::Celu<
            float,
            NsCelu::DOUBLE_BUFFER_NUM,
            true,
            false,
            NsCelu::NON_PIPELINE_DOUBLE_TQUE_DEPTH> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(CeluTilingKey::TILING_KEY_CELU_MODE_7)) {
        NsCelu::Celu<
            float,
            NsCelu::DOUBLE_BUFFER_NUM,
            true,
            true,
            NsCelu::PIPELINE_DOUBLE_TQUE_DEPTH> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(CeluTilingKey::TILING_KEY_CELU_MODE_8)) {
        NsCelu::Celu<
            float,
            NsCelu::DOUBLE_BUFFER_NUM,
            false,
            false,
            1> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(CeluTilingKey::TILING_KEY_CELU_MODE_9)) {
        NsCelu::CeluSmallScalarAlpha op;
        op.InitAndProcess(x, y, &tilingData);
    }
    if constexpr (schMode == static_cast<uint32_t>(CeluTilingKey::TILING_KEY_CELU_MODE_10)) {
        NsCelu::Celu<
            float,
            NsCelu::DOUBLE_BUFFER_NUM,
            true,
            true,
            NsCelu::PIPELINE_DOUBLE_TQUE_DEPTH,
            true> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(CeluTilingKey::TILING_KEY_CELU_MODE_11)) {
        NsCelu::CeluSmallBlockScalarAlpha op;
        op.InitAndProcess(x, y, &tilingData);
    }
}
