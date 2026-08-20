#ifndef UNPACK_TILING_H
#define UNPACK_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(UnpackTilingData)
  TILING_DATA_FIELD_DEF(int64_t, axis);
  TILING_DATA_FIELD_DEF(int64_t, num);
  TILING_DATA_FIELD_DEF(int64_t, elem_size);
  TILING_DATA_FIELD_DEF(int64_t, outer);
  TILING_DATA_FIELD_DEF(int64_t, inner);
  TILING_DATA_FIELD_DEF(int64_t, output_elems);
  TILING_DATA_FIELD_DEF(int64_t, task_mode);
  TILING_DATA_FIELD_DEF(int64_t, total_tasks);
  TILING_DATA_FIELD_DEF(int64_t, tasks_per_core);
  TILING_DATA_FIELD_DEF(int64_t, tile_elems);
  TILING_DATA_FIELD_DEF(int64_t, direct_row_batch);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Unpack, UnpackTilingData)
} // namespace optiling

#endif
