
#include "register/tilingdata_base.h"

constexpr uint8_t MAX_SHAPE_SIZE = 25;

namespace optiling {
BEGIN_TILING_DATA_DEF(TensorEqualTilingData)
  TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_SHAPE_SIZE, x1Shape)
  TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_SHAPE_SIZE + 1, x1ShapeRSum)
  TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_SHAPE_SIZE, x2Shape)
  TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_SHAPE_SIZE + 1, x2ShapeRSum)
  TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_SHAPE_SIZE, yShape)
  TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_SHAPE_SIZE + 1, yShapeRSum)
  TILING_DATA_FIELD_DEF(uint32_t, shapeSize)
  TILING_DATA_FIELD_DEF(uint32_t, dataLength)
  TILING_DATA_FIELD_DEF(uint32_t, tileLength)
  TILING_DATA_FIELD_DEF(uint32_t, innerLength)
  TILING_DATA_FIELD_DEF(uint32_t, outerLength)
  TILING_DATA_FIELD_DEF(uint32_t, tk6AlignedInnerLength)
  TILING_DATA_FIELD_DEF(uint32_t, tk6RowsPerTile)
  TILING_DATA_FIELD_DEF(uint32_t, tk6CmpRepeatLength)
  TILING_DATA_FIELD_DEF(uint32_t, tk6RepeatStrideBlocks)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TensorEqual, TensorEqualTilingData)
}
