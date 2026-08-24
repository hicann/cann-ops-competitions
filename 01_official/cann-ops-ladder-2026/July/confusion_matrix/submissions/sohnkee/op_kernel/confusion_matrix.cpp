/*!
 * \file confusion_matrix.cpp
 * \brief ConfusionMatrix 算子 kernel 入口
 */

#include "confusion_matrix.h"

template <uint32_t schMode>
__global__ __aicore__ void confusion_matrix(GM_ADDR labels, GM_ADDR predictions, GM_ADDR weights, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    // The zero-argument SyncAll in tiling key 6 needs the mixed AIV task
    // metadata.  Keep every other key on the lean AIV-only launch path.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(6, KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(ConfusionMatrixTilingData);
    if constexpr (schMode == CONFUSIONMATRIX_TPL_SCH_MODE_0) {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        GET_TILING_DATA_WITH_STRUCT(ConfusionMatrixTilingData, tilingData, tiling);
        NsConfusionMatrix::ProcessTinyInt32NoWeight(labels, predictions, y, &tilingData);
        return;
    }
    if constexpr (schMode == CONFUSIONMATRIX_TPL_SCH_MODE_2) {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        GET_TILING_DATA_WITH_STRUCT(ConfusionMatrixTilingData, tilingData, tiling);
        NsConfusionMatrix::ProcessTinyInt32WeightInt32(labels, predictions, weights, y, &tilingData);
        return;
    }
    if constexpr (schMode == CONFUSIONMATRIX_TPL_SCH_MODE_3) {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        GET_TILING_DATA_WITH_STRUCT(ConfusionMatrixTilingData, tilingData, tiling);
        NsConfusionMatrix::ProcessTinyInt64WeightFloat(labels, predictions, weights, y, &tilingData);
        return;
    }
    if constexpr (schMode == CONFUSIONMATRIX_TPL_SCH_MODE_4) {
        if (AscendC::GetBlockIdx() >= 3) {
            return;
        }
        GET_TILING_DATA_WITH_STRUCT(ConfusionMatrixTilingData, tilingData, tiling);
        NsConfusionMatrix::ProcessC3VectorHistogram<false>(labels, predictions, y, &tilingData);
        return;
    }
    if constexpr (schMode == CONFUSIONMATRIX_TPL_SCH_MODE_5) {
        if (AscendC::GetBlockIdx() >= 3) {
            return;
        }
        GET_TILING_DATA_WITH_STRUCT(ConfusionMatrixTilingData, tilingData, tiling);
        NsConfusionMatrix::ProcessC3VectorHistogram<true>(labels, predictions, y, &tilingData);
        return;
    }
#if TILING_KEY_VAR == CONFUSIONMATRIX_TPL_SCH_MODE_6
    if constexpr (schMode == CONFUSIONMATRIX_TPL_SCH_MODE_6) {
        GET_TILING_DATA_WITH_STRUCT(ConfusionMatrixTilingData, tilingData, tiling);
        NsConfusionMatrix::ConfusionMatrix op;
        op.Init(labels, predictions, weights, y, &tilingData);
        op.ProcessAtomicSparse();
        return;
    }
#endif
    GET_TILING_DATA_WITH_STRUCT(ConfusionMatrixTilingData, tilingData, tiling);
    NsConfusionMatrix::ConfusionMatrix op;
    op.Init(labels, predictions, weights, y, &tilingData);
    op.Process();
}
