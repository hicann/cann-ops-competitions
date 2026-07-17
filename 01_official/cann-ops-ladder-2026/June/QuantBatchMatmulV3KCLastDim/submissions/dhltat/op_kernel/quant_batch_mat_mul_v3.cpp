/*!
 * \file quant_batch_mat_mul_v3.cpp
 * \brief QuantBatchMatMulV3 算子 kernel 入口
 */

#include "quant_batch_mat_mul_v3.h"

enum class QuantBatchMatMulV3TilingKey : uint32_t
{
    TILING_KEY_QUANTBATCHMATMULV3_MODE_0 = 0,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_1 = 1,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_2 = 2,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_3 = 3,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_4 = 4,
};

template <uint32_t schMode>
__global__ __aicore__ void quant_batch_mat_mul_v3(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR pertokenScale, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }
    AscendC::TPipe tPipe;
    GET_TILING_DATA(tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_0)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3Matmul<false, false> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, &tilingData.matmulTiling);
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_1)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3Matmul<false, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, &tilingData.matmulTiling);
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_2)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3Matmul<true, false> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, &tilingData.matmulTiling);
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_3)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3Matmul<true, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, &tilingData.matmulTiling);
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_4)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3Matmul<true, false, true> op;
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
}
