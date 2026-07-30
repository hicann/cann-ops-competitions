
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(UnpackTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, M);
  TILING_DATA_FIELD_DEF(uint32_t, K);
  TILING_DATA_FIELD_DEF(uint32_t, L);
  TILING_DATA_FIELD_DEF(uint32_t, tile_length);
  TILING_DATA_FIELD_DEF(uint32_t, iter_per_row);
  TILING_DATA_FIELD_DEF(uint32_t, real_residue);
  TILING_DATA_FIELD_DEF(uint32_t, pad_residue);
  TILING_DATA_FIELD_DEF(uint32_t, iterations);
  TILING_DATA_FIELD_DEF(uint32_t, pack_mode);
  TILING_DATA_FIELD_DEF(uint32_t, pack_tile_m);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Unpack, UnpackTilingData)
}
