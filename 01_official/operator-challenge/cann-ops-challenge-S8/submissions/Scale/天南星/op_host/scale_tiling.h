#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScaleTilingData)
    TILING_DATA_FIELD_DEF(int, size);
    TILING_DATA_FIELD_DEF(int, low_size);
    TILING_DATA_FIELD_DEF(int, high_size);
    TILING_DATA_FIELD_DEF(int, has_bias);  
    //TILING_DATA_FIELD_DEF(bool, swap_input_other);
    TILING_DATA_FIELD_DEF_ARR(int, 5, input_n);
    TILING_DATA_FIELD_DEF_ARR(int, 5, scale_n);
    TILING_DATA_FIELD_DEF_ARR(int, 5, bias_n);
    TILING_DATA_FIELD_DEF_ARR(int, 5, out_n);
    TILING_DATA_FIELD_DEF_ARR(long, 3, broadcast_offset);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Scale, ScaleTilingData)

BEGIN_TILING_DATA_DEF(ScaleTilingDataFloat)
  TILING_DATA_FIELD_DEF(uint32_t, total_length);
  TILING_DATA_FIELD_DEF(uint32_t, start_length);
  TILING_DATA_FIELD_DEF(uint32_t, end_length);
  TILING_DATA_FIELD_DEF(uint32_t, weight_length);
  TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
  TILING_DATA_FIELD_DEF(uint32_t, tiling_size);
  TILING_DATA_FIELD_DEF(uint32_t, block_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_remain);
  TILING_DATA_FIELD_DEF(uint32_t, has_bias);

  TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, shape);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, reduce1);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, reduce2);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, reduce3);
  TILING_DATA_FIELD_DEF(uint32_t, dim); 
  
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Scale_13, ScaleTilingDataFloat)

// REGISTER_TILING_DATA_CLASS(Scale, ScaleTilingData)

BEGIN_TILING_DATA_DEF(ScaleTilingDataFloatBroadCast)
  TILING_DATA_FIELD_DEF(uint32_t, total_length);
  TILING_DATA_FIELD_DEF(uint32_t, start_length);
  TILING_DATA_FIELD_DEF(uint32_t, end_length);
  TILING_DATA_FIELD_DEF(uint32_t, weight_length);
  TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
  TILING_DATA_FIELD_DEF(uint32_t, tiling_size);
  TILING_DATA_FIELD_DEF(uint32_t, block_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_remain);
  TILING_DATA_FIELD_DEF(uint32_t, has_bias);

  TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, shape);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, reduce1);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, reduce2);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, reduce3);
  TILING_DATA_FIELD_DEF(uint32_t, dim); 
  
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Scale_14, ScaleTilingDataFloatBroadCast)


}