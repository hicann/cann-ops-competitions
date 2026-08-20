#ifndef ASSIGN_TILING_H
#define ASSIGN_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AssignTilingData)
  TILING_DATA_FIELD_DEF(int64_t, total_elems);
  TILING_DATA_FIELD_DEF(int64_t, total_bytes);
  TILING_DATA_FIELD_DEF(int64_t, elem_size);
  TILING_DATA_FIELD_DEF(int64_t, rank);
  TILING_DATA_FIELD_DEF(int64_t, same_shape);
  TILING_DATA_FIELD_DEF(int64_t, block_dim);
  TILING_DATA_FIELD_DEF(int64_t, total_core_num);
  TILING_DATA_FIELD_DEF(int64_t, used_core_num);
  TILING_DATA_FIELD_DEF(int64_t, ub_factor);
  TILING_DATA_FIELD_DEF(int64_t, block_factor);
  TILING_DATA_FIELD_DEF(int64_t, tail_block_factor);
  TILING_DATA_FIELD_DEF(int64_t, tail_block_tail_ub_factor);
  TILING_DATA_FIELD_DEF(int64_t, elems_per_core);
  TILING_DATA_FIELD_DEF(int64_t, bytes_per_core);
  TILING_DATA_FIELD_DEF(int64_t, dim0);
  TILING_DATA_FIELD_DEF(int64_t, dim1);
  TILING_DATA_FIELD_DEF(int64_t, dim2);
  TILING_DATA_FIELD_DEF(int64_t, dim3);
  TILING_DATA_FIELD_DEF(int64_t, dim4);
  TILING_DATA_FIELD_DEF(int64_t, dim5);
  TILING_DATA_FIELD_DEF(int64_t, dim6);
  TILING_DATA_FIELD_DEF(int64_t, dim7);
  TILING_DATA_FIELD_DEF(int64_t, other_dim0);
  TILING_DATA_FIELD_DEF(int64_t, other_dim1);
  TILING_DATA_FIELD_DEF(int64_t, other_dim2);
  TILING_DATA_FIELD_DEF(int64_t, other_dim3);
  TILING_DATA_FIELD_DEF(int64_t, other_dim4);
  TILING_DATA_FIELD_DEF(int64_t, other_dim5);
  TILING_DATA_FIELD_DEF(int64_t, other_dim6);
  TILING_DATA_FIELD_DEF(int64_t, other_dim7);
  TILING_DATA_FIELD_DEF(int64_t, out_stride0);
  TILING_DATA_FIELD_DEF(int64_t, out_stride1);
  TILING_DATA_FIELD_DEF(int64_t, out_stride2);
  TILING_DATA_FIELD_DEF(int64_t, out_stride3);
  TILING_DATA_FIELD_DEF(int64_t, out_stride4);
  TILING_DATA_FIELD_DEF(int64_t, out_stride5);
  TILING_DATA_FIELD_DEF(int64_t, out_stride6);
  TILING_DATA_FIELD_DEF(int64_t, out_stride7);
  TILING_DATA_FIELD_DEF(int64_t, other_stride0);
  TILING_DATA_FIELD_DEF(int64_t, other_stride1);
  TILING_DATA_FIELD_DEF(int64_t, other_stride2);
  TILING_DATA_FIELD_DEF(int64_t, other_stride3);
  TILING_DATA_FIELD_DEF(int64_t, other_stride4);
  TILING_DATA_FIELD_DEF(int64_t, other_stride5);
  TILING_DATA_FIELD_DEF(int64_t, other_stride6);
  TILING_DATA_FIELD_DEF(int64_t, other_stride7);
  TILING_DATA_FIELD_DEF(int64_t, broadcast_mode);
  TILING_DATA_FIELD_DEF(int64_t, outer_elems);
  TILING_DATA_FIELD_DEF(int64_t, outer_elems_per_core);
  TILING_DATA_FIELD_DEF(int64_t, out_last_dim);
  TILING_DATA_FIELD_DEF(int64_t, other_last_dim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Assign, AssignTilingData)
} // namespace optiling

#endif
