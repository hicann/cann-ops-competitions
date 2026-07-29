#ifndef TENSOR_EQUAL_TILING_H
#define TENSOR_EQUAL_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TensorEqualTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, smallCoreDataNum);
    TILING_DATA_FIELD_DEF(uint32_t, bigCoreDataNum);
    TILING_DATA_FIELD_DEF(uint32_t, finalBigTileNum);
    TILING_DATA_FIELD_DEF(uint32_t, finalSmallTileNum);
    TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);
    TILING_DATA_FIELD_DEF(uint32_t, smallTailDataNum);
    TILING_DATA_FIELD_DEF(uint32_t, bigTailDataNum);
    TILING_DATA_FIELD_DEF(uint32_t, tailBlockNum);
    TILING_DATA_FIELD_DEF(uint32_t, bigprocessDataNum_computes);
    TILING_DATA_FIELD_DEF(uint32_t, smallprocessDataNum_computes);
    TILING_DATA_FIELD_DEF(uint32_t, tailbigprocessDataNum_computes);
    TILING_DATA_FIELD_DEF(uint32_t, tailsmallprocessDataNum_computes);
    TILING_DATA_FIELD_DEF(uint32_t, dataType);
    TILING_DATA_FIELD_DEF(uint32_t, typeLength);
    TILING_DATA_FIELD_DEF(uint32_t, rank);
    TILING_DATA_FIELD_DEF(uint32_t, broadcastNeeded);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, outDims);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, selfStrides);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, otherStrides);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TensorEqual, TensorEqualTilingData)
} // namespace optiling

#endif // TENSOR_EQUAL_TILING_H
