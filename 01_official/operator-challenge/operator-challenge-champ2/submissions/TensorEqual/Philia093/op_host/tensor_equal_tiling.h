#ifndef TENSOR_EQUAL_TILING_H
#define TENSOR_EQUAL_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(TensorEqualTilingData)
  TILING_DATA_FIELD_DEF(int64_t, total_elems);
  TILING_DATA_FIELD_DEF(int64_t, elems_per_core);
  TILING_DATA_FIELD_DEF(int64_t, tile_elems);
  TILING_DATA_FIELD_DEF(int64_t, rank);
  TILING_DATA_FIELD_DEF(int64_t, same_shape);
  TILING_DATA_FIELD_DEF(int64_t, out_dim0);
  TILING_DATA_FIELD_DEF(int64_t, out_dim1);
  TILING_DATA_FIELD_DEF(int64_t, out_dim2);
  TILING_DATA_FIELD_DEF(int64_t, out_dim3);
  TILING_DATA_FIELD_DEF(int64_t, out_dim4);
  TILING_DATA_FIELD_DEF(int64_t, out_dim5);
  TILING_DATA_FIELD_DEF(int64_t, out_dim6);
  TILING_DATA_FIELD_DEF(int64_t, out_dim7);
  TILING_DATA_FIELD_DEF(int64_t, x1_dim0);
  TILING_DATA_FIELD_DEF(int64_t, x1_dim1);
  TILING_DATA_FIELD_DEF(int64_t, x1_dim2);
  TILING_DATA_FIELD_DEF(int64_t, x1_dim3);
  TILING_DATA_FIELD_DEF(int64_t, x1_dim4);
  TILING_DATA_FIELD_DEF(int64_t, x1_dim5);
  TILING_DATA_FIELD_DEF(int64_t, x1_dim6);
  TILING_DATA_FIELD_DEF(int64_t, x1_dim7);
  TILING_DATA_FIELD_DEF(int64_t, x2_dim0);
  TILING_DATA_FIELD_DEF(int64_t, x2_dim1);
  TILING_DATA_FIELD_DEF(int64_t, x2_dim2);
  TILING_DATA_FIELD_DEF(int64_t, x2_dim3);
  TILING_DATA_FIELD_DEF(int64_t, x2_dim4);
  TILING_DATA_FIELD_DEF(int64_t, x2_dim5);
  TILING_DATA_FIELD_DEF(int64_t, x2_dim6);
  TILING_DATA_FIELD_DEF(int64_t, x2_dim7);
  TILING_DATA_FIELD_DEF(int64_t, out_stride0);
  TILING_DATA_FIELD_DEF(int64_t, out_stride1);
  TILING_DATA_FIELD_DEF(int64_t, out_stride2);
  TILING_DATA_FIELD_DEF(int64_t, out_stride3);
  TILING_DATA_FIELD_DEF(int64_t, out_stride4);
  TILING_DATA_FIELD_DEF(int64_t, out_stride5);
  TILING_DATA_FIELD_DEF(int64_t, out_stride6);
  TILING_DATA_FIELD_DEF(int64_t, out_stride7);
  TILING_DATA_FIELD_DEF(int64_t, x1_stride0);
  TILING_DATA_FIELD_DEF(int64_t, x1_stride1);
  TILING_DATA_FIELD_DEF(int64_t, x1_stride2);
  TILING_DATA_FIELD_DEF(int64_t, x1_stride3);
  TILING_DATA_FIELD_DEF(int64_t, x1_stride4);
  TILING_DATA_FIELD_DEF(int64_t, x1_stride5);
  TILING_DATA_FIELD_DEF(int64_t, x1_stride6);
  TILING_DATA_FIELD_DEF(int64_t, x1_stride7);
  TILING_DATA_FIELD_DEF(int64_t, x2_stride0);
  TILING_DATA_FIELD_DEF(int64_t, x2_stride1);
  TILING_DATA_FIELD_DEF(int64_t, x2_stride2);
  TILING_DATA_FIELD_DEF(int64_t, x2_stride3);
  TILING_DATA_FIELD_DEF(int64_t, x2_stride4);
  TILING_DATA_FIELD_DEF(int64_t, x2_stride5);
  TILING_DATA_FIELD_DEF(int64_t, x2_stride6);
  TILING_DATA_FIELD_DEF(int64_t, x2_stride7);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TensorEqual, TensorEqualTilingData)

}  // namespace optiling

#endif
