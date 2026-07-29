
#include "register/tilingdata_base.h"

constexpr int MAX_SHAPE(4);
constexpr int DIM_LAST(MAX_SHAPE - 1);
namespace optiling {
BEGIN_TILING_DATA_DEF(TensorEqualTilingData)
  TILING_DATA_FIELD_DEF(int64_t, iterations);
  TILING_DATA_FIELD_DEF(int64_t, tile_length);
  TILING_DATA_FIELD_DEF(int64_t, iter_per_vector);
  TILING_DATA_FIELD_DEF(int64_t, vector_per_iter);
  TILING_DATA_FIELD_DEF(int64_t, real_residue);
  TILING_DATA_FIELD_DEF(int64_t, pad_residue);

  TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_SHAPE, input_shape);
  TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_SHAPE, other_shape);
  TILING_DATA_FIELD_DEF(int, iter_idx);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TensorEqual, TensorEqualTilingData)
}
