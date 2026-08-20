#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(UnpackTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, outer);
  TILING_DATA_FIELD_DEF(uint64_t, num);
  TILING_DATA_FIELD_DEF(uint64_t, inner);

  TILING_DATA_FIELD_DEF(uint32_t, coreNum);
  TILING_DATA_FIELD_DEF(uint32_t, tile_outer);
  TILING_DATA_FIELD_DEF(uint32_t, tile_inner);
  TILING_DATA_FIELD_DEF(uint32_t, tile_outer_num);
  TILING_DATA_FIELD_DEF(uint32_t, tile_inner_num);
  TILING_DATA_FIELD_DEF(uint32_t, ub_buffer_size);
  TILING_DATA_FIELD_DEF(uint32_t, ub_out_buffer_size);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Unpack, UnpackTilingData)
}
