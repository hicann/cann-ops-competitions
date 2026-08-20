#ifndef UNPACK_TILING_H
#define UNPACK_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(UnpackTilingData)
  // 通用三段式形状：原输入统一展开为 [outer_size, axis_size, inner_size]。
  // total_elem 用于绑定输入 GM，output_num 为动态输出个数且应等于 axis_size。
  TILING_DATA_FIELD_DEF(uint32_t, total_elem);
  TILING_DATA_FIELD_DEF(uint32_t, output_num);
  TILING_DATA_FIELD_DEF(uint32_t, outer_size);
  TILING_DATA_FIELD_DEF(uint32_t, axis_size);
  TILING_DATA_FIELD_DEF(uint32_t, inner_size);

  // 路径相关的 tile 尺寸。中轴路径中 tile_o 表示每 tile 的 line 数、tile_r 固定为 1；
  // 末轴路径会复用这两个字段记录列宽和基础行分组信息。
  TILING_DATA_FIELD_DEF(uint32_t, tile_r);
  TILING_DATA_FIELD_DEF(uint32_t, tile_o);

  // row_bytes 是一行的有效字节数；非对齐行在 UB 中按 ub_row_pitch_bytes 跨步存放。
  // row_aligned 为 1 时可使用不带 Pad 的连续/二维 DataCopy。
  TILING_DATA_FIELD_DEF(uint32_t, row_bytes);
  TILING_DATA_FIELD_DEF(uint32_t, ub_row_pitch_bytes);
  TILING_DATA_FIELD_DEF(uint32_t, row_aligned);

  // 首轴路径：mode 0 按完整输出分核，mode 1 将单个大输出交给多个核协作。
  // mode 1 中每组由 cores_per_tensor 个 lane 组成，并以 tensor_group_count 轮转领取输出。
  TILING_DATA_FIELD_DEF(uint32_t, axis0_mode);
  TILING_DATA_FIELD_DEF(uint32_t, axis0_inner_tile_elems);
  TILING_DATA_FIELD_DEF(uint32_t, axis0_cores_per_tensor);
  TILING_DATA_FIELD_DEF(uint32_t, axis0_tensor_group_count);

  // 末轴路径把输入视为 [outer, output_num]：c_tile 为一次读入的列数，
  // rows_per_tile 为一次处理的行数，col_tile_count/repeats 为 kernel 热点循环的派生量。
  TILING_DATA_FIELD_DEF(uint32_t, axis_last_c_tile);
  TILING_DATA_FIELD_DEF(uint32_t, axis_last_groups);
  TILING_DATA_FIELD_DEF(uint32_t, axis_last_rows_per_tile);
  TILING_DATA_FIELD_DEF(uint32_t, axis_last_col_tile_count);
  TILING_DATA_FIELD_DEF(uint32_t, axis_last_full_repeats);
  TILING_DATA_FIELD_DEF(uint32_t, axis_last_tail_repeats);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Unpack, UnpackTilingData)
}  // namespace optiling

#endif
