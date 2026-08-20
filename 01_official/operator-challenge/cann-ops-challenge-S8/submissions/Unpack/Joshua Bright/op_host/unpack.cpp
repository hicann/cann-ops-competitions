#include "unpack_tiling.h"
#include "register/op_def_registry.h"

#include <cstring>
#include <cstdint>
#include <algorithm>
#include <vector>

// Host 侧职责
// -----------
// 将输入统一表示为 [outer, axis, inner]，校验 num/axis，按规范化后的轴位置选择 kernel，
// 并在 UB 容量、DMA 对齐、TransData repeat 上限和核间负载之间选择 tile。
//
// TILING_KEY：1=首轴单输出分核；2=首轴大输出多核；7=中轴 line stream；
// 5=末轴 TransData；8=末轴 2B/4B 非对齐行 Gather。
namespace optiling {

// 910B 单 AIV 核按 192KB UB 规划；双缓冲路径通常每份使用 96KB。
static constexpr uint32_t kUbDoubleBytes = 192U * 1024U;
static constexpr uint32_t kUbBytes = 96U * 1024U;
// 中轴 host 使用 64KB 保守预算，kernel 静态缓冲仍为两份 96KB；较小预算只限制 tile 大小。
static constexpr uint32_t kMidStreamBufBytes = kUbDoubleBytes / 3U;
// axis=-1 且非1B类型时的目标tile大小（优先做大连续写段，减少小包写GM）。
static constexpr uint32_t kAxisLastTargetBytes = 96U * 1024U;
// axis=-1 转置快路径：单buffer预算为96KB。
// 现在dst只保留“单次32B列块转置结果”的scratch，剩余空间尽量让给src。
static constexpr uint32_t kAxisLastTransPhaseBias = 32U;
static constexpr uint32_t kAxisLastTransGuardBytes = 2U * 1024U;
static constexpr uint32_t kAxisLastTransAvailBytes =
    kUbBytes - kAxisLastTransPhaseBias - kAxisLastTransGuardBytes;
static constexpr uint32_t kAxisLastColBlockBytes = 32U;
static constexpr uint32_t kAxisLastGatherBufBytes = 32U * 1024U;
static constexpr uint32_t kAxisLastGatherColChunk = 4U;
static constexpr uint32_t kAxisLastGatherIndexPhasePadBytes = 32U;
static constexpr uint32_t kTransRepeatMax = 240U;

// Kernel 仅按位搬运，host 只需下发元素字宽。
static uint32_t GetElemSize(const ge::DataType dt) {
  switch (dt) {
    case ge::DT_FLOAT16:
    case ge::DT_BF16:
    case ge::DT_INT16:
      return 2;
    case ge::DT_FLOAT:
    case ge::DT_INT32:
      return 4;
    case ge::DT_UINT8:
    case ge::DT_INT8:
    case ge::DT_BOOL:
      return 1;
    default:
      return 1;
  }
}

static uint32_t Align32(uint32_t bytes) {
  return (bytes + 31U) & (~31U);
}

// 末轴 dst 每个输出列的 UB pitch：先对齐，再强制为奇数个 datablock 以缓解 bank 冲突。
static uint32_t GetAxisLastTransDstRowPitchBytes(uint32_t rows, uint32_t elem_size) {
  uint32_t pitch_bytes = Align32(rows * elem_size);
  uint32_t datablocks = pitch_bytes / 32U;
  if ((datablocks & 1U) == 0U) {
    pitch_bytes += 32U;
  }
  return pitch_bytes;
}

static uint32_t GetAxisLastTransDstScratchBytes(uint32_t rows, uint32_t elem_size) {
  // 一个 stage 处理 32B 列宽，乘 2 表示 stage0/stage1。
  const uint32_t block_cols = 32U / elem_size;
  return 2U * GetAxisLastTransDstRowPitchBytes(rows, elem_size) * block_cols;
}

static uint32_t GetAxisLastGatherDstRowPitchBytes(uint32_t rows, uint32_t elem_size) {
  // Gather dst 使用与 TransData dst 相同的奇数 datablock pitch 原则。
  uint32_t pitch_bytes = Align32(rows * elem_size);
  const uint32_t datablocks = pitch_bytes / 32U;
  if ((datablocks & 1U) == 0U) {
    pitch_bytes += 32U;
  }
  return pitch_bytes;
}

static uint32_t GetAxisLastRowAlignScore(uint32_t row_bytes) {
  // 对齐分数仅用于相同并行度候选间的择优。
  if ((row_bytes % 512U) == 0U) {
    return 3U;
  }
  if ((row_bytes % 256U) == 0U) {
    return 2U;
  }
  if ((row_bytes % 128U) == 0U) {
    return 1U;
  }
  return 0U;
}

static uint32_t FloorSqrt(uint32_t v) {
  // 除法写法避免平方产生 uint32 溢出。
  uint32_t x = 0;
  while ((x + 1U) <= (v / (x + 1U))) {
    ++x;
  }
  return x;
}

static uint64_t AbsDiffU64(uint64_t a, uint64_t b) {
  return (a > b) ? (a - b) : (b - a);
}

static uint32_t FloorPow2(uint32_t v) {
  // 首轴协作核数取 2 的幂，便于均匀映射 tensor/lane。
  uint32_t p = 1U;
  while ((p << 1U) != 0U && (p << 1U) <= v) {
    p <<= 1U;
  }
  return p;
}

// 中轴切分：将 [outer, axis] 展平为 total_lines，每条 line 含 inner 个元素。
// tile_o 在 key7 中表示每 tile 的 line 数，tile_r 固定为 1。候选优先级为：
// 并行核数 > tile 数整除并行核数 > 负载余数更小 > tile_o 更大。
static void SelectMidAxisStreamTile(uint32_t outer, uint32_t out_num, uint32_t inner, uint32_t elem_size,
                                    uint32_t &tile_r, uint32_t &tile_o, uint32_t &row_bytes,
                                    uint32_t &ub_row_pitch_bytes, uint32_t &row_aligned) {
  constexpr uint32_t kMidStreamCoreCount = 40U;
  row_bytes = inner * elem_size;
  row_aligned = ((row_bytes & 31U) == 0U) ? 1U : 0U;
  ub_row_pitch_bytes = (row_aligned != 0U) ? row_bytes : Align32(row_bytes);
  tile_r = 1U;
  const uint32_t max_lines = kMidStreamBufBytes / ub_row_pitch_bytes;
  const uint32_t total_lines = outer * out_num;
  if (total_lines == 0U || max_lines == 0U) {
    tile_o = 1U;
    return;
  }
  const uint32_t desired_parallel = (total_lines < kMidStreamCoreCount) ? total_lines : kMidStreamCoreCount;
  uint32_t best_tile_o = (total_lines < max_lines) ? total_lines : max_lines;
  uint32_t best_parallel = 0U;
  bool best_exact_balance = false;
  uint32_t best_balance_rem = ~0U;
  for (uint32_t try_tile_o = 1U; try_tile_o <= max_lines && try_tile_o <= total_lines; ++try_tile_o) {
    // kernel 以 tile_idx=core_idx+n*blockDim 领取 line tile。
    const uint32_t line_tile_count = (total_lines + try_tile_o - 1U) / try_tile_o;
    const uint32_t parallel = (line_tile_count < kMidStreamCoreCount) ? line_tile_count : kMidStreamCoreCount;
    const bool exact_balance = (parallel != 0U) && ((line_tile_count % parallel) == 0U);
    const uint32_t balance_rem = (parallel == 0U) ? line_tile_count : (line_tile_count % parallel);
    if (parallel < desired_parallel) {
      continue;
    }
    if (parallel > best_parallel ||
        (parallel == best_parallel &&
         (exact_balance > best_exact_balance ||
          (exact_balance == best_exact_balance &&
           (balance_rem < best_balance_rem ||
            (balance_rem == best_balance_rem && try_tile_o > best_tile_o)))))) {
      best_tile_o = try_tile_o;
      best_parallel = parallel;
      best_exact_balance = exact_balance;
      best_balance_rem = balance_rem;
    }
  }
  tile_o = best_tile_o;
}
// 末轴拆分策略（axis=-1）：
// 该场景本质是把 [outer, axis] 做转置后散写到多个输出。
// c_tile 为输入 tile 的列数，rows_per_tile 为 outer 行数，groups 为基本 row block 数；
// full/tail_repeats 描述完整列 tile 和尾列 tile 的 TransData 次数。
static void SelectAxisLastTile(uint32_t outer, uint32_t out_num, uint32_t elem_size, uint32_t &tile_r,
                               uint32_t &tile_o, uint32_t &row_bytes, uint32_t &ub_row_pitch_bytes,
                               uint32_t &row_aligned, uint32_t &axis_last_c_tile, uint32_t &axis_last_groups,
                               uint32_t &axis_last_rows_per_tile, uint32_t &axis_last_col_tile_count,
                               uint32_t &axis_last_full_repeats, uint32_t &axis_last_tail_repeats) {
  constexpr uint32_t kAxisLastCoreCount = 40U;
  const uint32_t r_cap = (out_num == 0U) ? 1U : out_num;
  const uint32_t o_cap = (outer == 0U) ? 1U : outer;
  axis_last_c_tile = 0U;
  axis_last_groups = 0U;
  axis_last_rows_per_tile = 0U;
  axis_last_col_tile_count = 0U;
  axis_last_full_repeats = 0U;
  axis_last_tail_repeats = 0U;

  // int8 基本块为 32 行；2B/4B 基本块为 16 行。一个列单元始终覆盖 32B。
  const uint32_t kBlockRows = (elem_size == 1U) ? 32U : 16U;
  const uint32_t cols_unit_raw = 32U / elem_size;
  uint32_t cols_unit = (cols_unit_raw == 0U) ? 1U : cols_unit_raw;
  if (cols_unit > r_cap) {
    cols_unit = r_cap;
  }
  if (cols_unit == 0U) {
    cols_unit = 1U;
  }
  const uint32_t kMaxGroups = 8U;
  tile_r = cols_unit;
  tile_o = 1U;

  const uint32_t one_row_bytes = tile_r * elem_size;
  const uint32_t one_row_pitch_bytes = Align32(one_row_bytes);
  if (one_row_pitch_bytes == 0U || one_row_pitch_bytes > kUbBytes) {
    tile_r = 1U;
    tile_o = 1U;
    row_bytes = elem_size;
    row_aligned = 0U;
    ub_row_pitch_bytes = 0U;
    return;
  }

  // 目标 row tile 接近每核平均行数，并向上取整到基本 row block。
  const uint32_t target_rows = (outer == 0U) ? kBlockRows : ((outer + kAxisLastCoreCount - 1U) / kAxisLastCoreCount);

  auto choose_groups_for_c_tile = [&](uint32_t c_tile) -> uint32_t {
    // 同时计入 src tile 与双 stage dst scratch，保证固定 UB 地址不重叠。
    uint32_t groups_cap = 0U;
    for (uint32_t groups = 1U; groups <= kMaxGroups; ++groups) {
      const uint32_t rows = groups * kBlockRows;
      const uint32_t src_bytes = rows * c_tile * elem_size;
      const uint32_t dst_bytes = GetAxisLastTransDstScratchBytes(rows, elem_size);
      if ((src_bytes + dst_bytes) > kAxisLastTransAvailBytes) {
        break;
      }
      groups_cap = groups;
    }
    if (groups_cap == 0U) {
      return 0U;
    }
    uint32_t target_groups = (target_rows + kBlockRows - 1U) / kBlockRows;
    if (target_groups == 0U) {
      target_groups = 1U;
    }
    if (target_groups > kMaxGroups) {
      target_groups = kMaxGroups;
    }
    if (target_groups > groups_cap) {
      target_groups = groups_cap;
    }
    return target_groups;
  };

  auto commit_axis_last = [&](uint32_t c_tile, uint32_t groups) {
    // 将二维 tile 写成 kernel 热点循环直接使用的派生字段。
    axis_last_c_tile = c_tile;
    axis_last_groups = groups;
    axis_last_rows_per_tile = groups * kBlockRows;
    axis_last_full_repeats = axis_last_c_tile / cols_unit;
    axis_last_col_tile_count = (out_num + axis_last_c_tile - 1U) / axis_last_c_tile;
    const uint32_t tail_cols = out_num - (out_num / axis_last_c_tile) * axis_last_c_tile;
    axis_last_tail_repeats = (tail_cols == 0U) ? axis_last_full_repeats : ((tail_cols + cols_unit - 1U) / cols_unit);
    row_bytes = c_tile * elem_size;
    ub_row_pitch_bytes = Align32(row_bytes);
    row_aligned = ((row_bytes & 31U) == 0U) ? 1U : 0U;
  };

  // 列宽同时受输出列数、UB 容量和 TransData repeat 上限约束。
  const uint32_t c_limit_rep = cols_unit * kTransRepeatMax;
  uint32_t c_tile = (r_cap < c_limit_rep) ? r_cap : c_limit_rep;
  c_tile = (c_tile / cols_unit) * cols_unit;
  if (c_tile < cols_unit) {
    c_tile = cols_unit;
  }

  for (uint32_t try_c = c_tile;; try_c -= cols_unit) {
    // 优先保留宽列 tile，放不下时按一个 32B 列单元递减。
    const uint32_t groups = choose_groups_for_c_tile(try_c);
    if (groups != 0U) {
      commit_axis_last(try_c, groups);
      return;
    }
    if (try_c == cols_unit) {
      break;
    }
  }

  commit_axis_last(cols_unit, 1U);
}

// 非对齐末轴 Gather tiling：src 为紧凑 [rows, out_num]，index 保存 row 字节偏移，
// dst 一次容纳 chunk_cols 个连续输出列。
static void SelectAxisLastGatherTile(uint32_t outer, uint32_t out_num, uint32_t elem_size, uint32_t &tile_r,
                                     uint32_t &tile_o, uint32_t &row_bytes, uint32_t &ub_row_pitch_bytes,
                                     uint32_t &row_aligned, uint32_t &axis_last_c_tile,
                                     uint32_t &axis_last_groups, uint32_t &axis_last_rows_per_tile,
                                     uint32_t &axis_last_col_tile_count, uint32_t &axis_last_full_repeats,
                                     uint32_t &axis_last_tail_repeats) {
  // src+index 使用最多 64KB，dst 双 stage 从剩余预算中分配。
  constexpr uint32_t kAxisLastGatherSrcIndexBudgetBytes = 64U * 1024U;
  constexpr uint32_t kAxisLastGatherDstBudgetBytes = 16U * 1024U - kAxisLastGatherIndexPhasePadBytes;
  row_bytes = out_num * elem_size;
  ub_row_pitch_bytes = Align32(row_bytes);
  row_aligned = ((row_bytes & 31U) == 0U) ? 1U : 0U;
  tile_r = (out_num == 0U) ? 1U : out_num;
  tile_o = 1U;
  axis_last_c_tile = 1U;
  axis_last_groups = 1U;
  axis_last_col_tile_count = 1U;
  axis_last_full_repeats = 0U;
  axis_last_tail_repeats = 0U;
  axis_last_rows_per_tile = 1U;

  if (ub_row_pitch_bytes == 0U) {
    return;
  }

  // 先由紧凑 src tile 给出行数上界；候选循环中再扣除 index 模板和 dst stage 的占用。
  const uint32_t max_rows_by_src = (row_bytes == 0U) ? 0U : (kUbBytes / row_bytes);
  uint32_t max_rows = max_rows_by_src;
  if (outer != 0U && max_rows > outer) {
    max_rows = outer;
  }
  if (max_rows == 0U) {
    return;
  }

  constexpr uint32_t kAxisLastGatherCoreCount = 40U;
  uint32_t best_rows = 0U;
  uint32_t best_chunk_cols = 0U;
  uint32_t best_parallel = 0U;
  uint32_t best_tiles_per_core = ~0U;
  bool best_exact_balance = false;
  uint32_t best_balance_rem = ~0U;
  uint32_t best_align_score = 0U;
  bool best_hits_src_target = false;
  for (uint32_t rows = max_rows; rows >= 1U; --rows) {
    // 每个候选必须容纳源矩阵、uint32 offset 模板和至少一列 Gather dst。
    const uint64_t src_total_bytes = Align32(rows * row_bytes);
    const uint32_t row_template_bytes = Align32(rows * static_cast<uint32_t>(sizeof(uint32_t)));
    if ((src_total_bytes + row_template_bytes) > kAxisLastGatherSrcIndexBudgetBytes) {
      if (rows == 1U) {
        break;
      }
      continue;
    }
    const uint32_t slice_rows = rows;
    const uint32_t per_col_dst_bytes = GetAxisLastGatherDstRowPitchBytes(slice_rows, elem_size);
    const uint32_t remain_bytes = kUbBytes - static_cast<uint32_t>(src_total_bytes) - row_template_bytes;
    if (per_col_dst_bytes == 0U || remain_bytes < per_col_dst_bytes) {
      if (rows == 1U) {
        break;
      }
      continue;
    }
    uint32_t chunk_cols = remain_bytes / per_col_dst_bytes;
    const uint32_t max_chunk_cols_by_dst =
        (per_col_dst_bytes == 0U) ? 0U : (kAxisLastGatherDstBudgetBytes / per_col_dst_bytes);
    if (chunk_cols > max_chunk_cols_by_dst) {
      chunk_cols = max_chunk_cols_by_dst;
    }
    if (chunk_cols > out_num) {
      chunk_cols = out_num;
    }
    if (chunk_cols == 0U) {
      if (rows == 1U) {
        break;
      }
      continue;
    }
    const uint32_t row_tile_count = (outer + rows - 1U) / rows;
    const uint32_t parallel = (row_tile_count < kAxisLastGatherCoreCount) ? row_tile_count : kAxisLastGatherCoreCount;
    const uint32_t tiles_per_core = (parallel == 0U) ? row_tile_count : ((row_tile_count + parallel - 1U) / parallel);
    const uint32_t balance_rem = (parallel == 0U) ? row_tile_count : (row_tile_count % parallel);
    const bool exact_balance = (balance_rem == 0U);
    const uint32_t row_vec_bytes = rows * elem_size;
    const bool hits_src_target = ((src_total_bytes + row_template_bytes) >= kAxisLastGatherSrcIndexBudgetBytes);
    uint32_t align_score = 0U;
    if ((row_vec_bytes % 512U) == 0U) {
      align_score = 2U;
    } else if ((row_vec_bytes % 256U) == 0U) {
      align_score = 1U;
    }
    // 评分：并行度 > 每核 tile 数 > 均衡 > 向量长度对齐 > 预算利用 > chunk 列数 > rows。
    bool better = false;
    if (best_rows == 0U) {
      better = true;
    } else if (parallel != best_parallel) {
      better = (parallel > best_parallel);
    } else if (tiles_per_core != best_tiles_per_core) {
      better = (tiles_per_core < best_tiles_per_core);
    } else if (exact_balance != best_exact_balance) {
      better = (exact_balance > best_exact_balance);
    } else if (balance_rem != best_balance_rem) {
      better = (balance_rem < best_balance_rem);
    } else if (align_score != best_align_score) {
      better = (align_score > best_align_score);
    } else if (hits_src_target != best_hits_src_target) {
      better = (hits_src_target > best_hits_src_target);
    } else if (chunk_cols != best_chunk_cols) {
      better = (chunk_cols > best_chunk_cols);
    } else if (rows != best_rows) {
      better = (rows > best_rows);
    }
    if (better) {
      best_rows = rows;
      best_chunk_cols = chunk_cols;
      best_parallel = parallel;
      best_tiles_per_core = tiles_per_core;
      best_exact_balance = exact_balance;
      best_balance_rem = balance_rem;
      best_align_score = align_score;
      best_hits_src_target = hits_src_target;
    }
    if (rows == 1U) {
      break;
    }
  }
  if (best_rows != 0U) {
    // Gather kernel 将这两个通用末轴字段分别解释为 row tile 高度和每个 dst stage 的列数。
    axis_last_rows_per_tile = best_rows;
    axis_last_c_tile = best_chunk_cols;
  }
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  // rank 限制为 1..4，使 kernel 可使用固定 tiling 字段和 uint32 地址计算。
  UnpackTilingData t;
  const auto &shape = context->GetInputShape(0)->GetStorageShape();

  uint32_t ndim = static_cast<uint32_t>(shape.GetDimNum());
  if (ndim == 0 || ndim > 4) {
    return ge::GRAPH_FAILED;
  }

  // 未使用的尾维填 1，后续乘积无需按 rank 分支。
  uint32_t dims[4] = {1, 1, 1, 1};
  uint32_t total = 1;
  for (uint32_t i = 0; i < ndim; ++i) {
    dims[i] = static_cast<uint32_t>(shape.GetDim(i));
    total *= dims[i];
  }

  // 属性顺序为 num(0)、axis(1)；axis 支持 Python 风格负下标。
  int64_t axis_attr = 0;
  if (context->GetAttrs()) {
    axis_attr = * context->GetAttrs()->GetAttrPointer<int32_t>(1);
  }
  int32_t axis_raw = static_cast<int32_t>(axis_attr);
  int32_t axis_pos = axis_raw;
  if (axis_pos < 0) {
    axis_pos += static_cast<int32_t>(ndim);
  }
  if (axis_pos < 0 || axis_pos >= static_cast<int32_t>(ndim)) {
    return ge::GRAPH_FAILED;
  }
  if (shape.GetDim(axis_pos) <= 0) {
    return ge::GRAPH_FAILED;
  }

  // 动态输出数量必须严格等于被拆维长度，否则 output list 与 kernel 循环上限不一致。
  int64_t num_attr = shape.GetDim(axis_pos);
  if (context->GetAttrs()) {
    num_attr = * context->GetAttrs()->GetAttrPointer<int32_t>(0);
  }
  if (num_attr <= 0 || num_attr != shape.GetDim(axis_pos)) {
    return ge::GRAPH_FAILED;
  }
  const uint32_t output_num = static_cast<uint32_t>(num_attr);

  uint32_t out_dims[4] = {1, 1, 1, 1};
  uint32_t out_i = 0;
  for (uint32_t i = 0; i < ndim; ++i) {
    if (static_cast<int32_t>(i) == axis_pos) {
      continue;
    }
    out_dims[out_i++] = dims[i];
  }

  // 展开通用索引参数：outer * axis * inner。
  // kernel所有路径都基于这组三段式索引做地址计算。
  // 这里的数值按原始shape前后乘积得到；去掉长度为1的维度只影响“首/中/末轴”的路径判定，
  // 不改变outer/axis/inner本身的乘积结果。
  uint32_t outer = 1;
  for (int32_t i = 0; i < axis_pos; ++i) {
    outer *= dims[i];
  }
  uint32_t axis_sz = dims[axis_pos];
  uint32_t inner = 1;
  for (uint32_t i = static_cast<uint32_t>(axis_pos + 1); i < ndim; ++i) {
    inner *= dims[i];
  }

  bool has_non1_before_axis = false;
  for (int32_t i = 0; i < axis_pos; ++i) {
    if (dims[i] != 1U) {
      has_non1_before_axis = true;
      break;
    }
  }
  bool has_non1_after_axis = false;
  for (uint32_t i = static_cast<uint32_t>(axis_pos + 1); i < ndim; ++i) {
    if (dims[i] != 1U) {
      has_non1_after_axis = true;
      break;
    }
  }
  // 规范化规则：
  // 1. 除axis本身外，去掉所有长度为1的维度；
  // 2. 再按剩余维度判断axis位于首/中/末；
  // 3. 若规范化后只剩axis一个维度，则优先复用axis0路径。
  const bool axis_is_normalized_first = !has_non1_before_axis;
  const bool axis_is_normalized_last = !has_non1_after_axis;

  const ge::DataType dt = context->GetInputDesc(0)->GetDataType();
  const uint32_t elem_size = GetElemSize(dt);

  // row_bytes 是有效字节数，ub_row_pitch_bytes 是非对齐行在 UB 中的物理跨度；
  // tile_r/tile_o 的具体含义随路径变化。
  uint32_t tile_r = 1U;
  uint32_t tile_o = 1U;
  uint32_t row_bytes = inner * elem_size;
  uint32_t row_aligned = ((row_bytes & 31U) == 0U) ? 1U : 0U;
  uint32_t ub_row_pitch_bytes = row_aligned ? row_bytes : Align32(row_bytes);
  uint32_t axis0_mode = 0U;
  uint32_t axis0_inner_tile_elems = 0U;
  uint32_t axis0_cores_per_tensor = 1U;
  uint32_t axis0_tensor_group_count = 1U;
  uint32_t axis_last_c_tile = 0U;
  uint32_t axis_last_groups = 0U;
  uint32_t axis_last_rows_per_tile = 0U;
  uint32_t axis_last_col_tile_count = 0U;
  uint32_t axis_last_full_repeats = 0U;
  uint32_t axis_last_tail_repeats = 0U;
  uint32_t block_dim = 1U;

  // host侧按规范化后的axis位置下发不同切分策略：
  // - 首轴：kernel走直接搬运快路径；
  // - 中轴：按“平衡块”策略切分；
  // - 末轴：按“转置块”策略切分（再由kernel按字长分发具体实现）。
  if (axis_is_normalized_first) {
    // mode0 按完整输出分核；mode1 为少量大输出进一步切分 inner。
    constexpr uint32_t kAxis0CoreCount = 40U;
    constexpr uint32_t kAxis0SinkCoreCount = 32U;
    // 大输出协作路径保留 32 核：每个 lane 至少分到约 16KB，且仅在单输出达到 32KB 时启用。
    constexpr uint32_t kAxis0SaturateBytes = 16U * 1024U;
    constexpr uint32_t kAxis0SinkThresholdBytes = 32U * 1024U;
    const uint32_t one_out_bytes = inner * elem_size;
    if (output_num > 0U && output_num < kAxis0SinkCoreCount && one_out_bytes >= kAxis0SinkThresholdBytes) {
      axis0_mode = 1U;
      // 每输出协作核数同时受总核数和“每核至少约 16KB”约束。
      uint32_t max_cores_by_tensor = kAxis0SinkCoreCount / output_num;
      if (max_cores_by_tensor == 0U) {
        max_cores_by_tensor = 1U;
      }
      max_cores_by_tensor = FloorPow2(max_cores_by_tensor);
      uint32_t max_cores_by_bytes = one_out_bytes / kAxis0SaturateBytes;
      if (max_cores_by_bytes == 0U) {
        max_cores_by_bytes = 1U;
      } else {
        max_cores_by_bytes = FloorPow2(max_cores_by_bytes);
      }
      axis0_cores_per_tensor = (max_cores_by_tensor < max_cores_by_bytes) ? max_cores_by_tensor : max_cores_by_bytes;
      if (axis0_cores_per_tensor == 0U) {
        axis0_cores_per_tensor = 1U;
      }
      if (axis0_cores_per_tensor > kAxis0SinkCoreCount) {
        axis0_cores_per_tensor = kAxis0SinkCoreCount;
      }
      // 可并行 tensor 组数；同一核组领取后续输出时以此为步长。
      axis0_tensor_group_count = kAxis0SinkCoreCount / axis0_cores_per_tensor;
      if (axis0_tensor_group_count == 0U) {
        axis0_tensor_group_count = 1U;
      }
      // 最后一 lane 在 kernel 内吸收整除余数。
      axis0_inner_tile_elems = inner / axis0_cores_per_tensor;
      if (axis0_inner_tile_elems == 0U) {
        axis0_inner_tile_elems = 1U;
      }
      block_dim = kAxis0SinkCoreCount;
    } else {
      axis0_mode = 0U;
      block_dim = (output_num < kAxis0CoreCount) ? output_num : kAxis0CoreCount;
      if (block_dim == 0U) {
        block_dim = 1U;
      }
    }
    context->SetTilingKey((axis0_mode == 0U) ? 1U : 2U);
  } else if (!axis_is_normalized_last) {
    // kernel 将 [outer, axis] 视为循环输出编号的 line 流。
    SelectMidAxisStreamTile(outer, output_num, inner, elem_size, tile_r, tile_o, row_bytes, ub_row_pitch_bytes,
                            row_aligned);
    constexpr uint32_t kMidCoreCount = 40U;
    const uint32_t outer_tile_count = (outer + tile_o - 1U) / tile_o;
    const uint32_t row_tile_count = (output_num + tile_r - 1U) / tile_r;
    const uint32_t total_tiles = outer_tile_count * row_tile_count;
    block_dim = (total_tiles < kMidCoreCount) ? total_tiles : kMidCoreCount;
    if (block_dim == 0U) {
      block_dim = 1U;
    }
    context->SetTilingKey(7U);
  } else {
    // 末轴等价于把 [outer, output_num] 转置为 output_num 个连续列张量。
    constexpr uint32_t kAxisLastCoreCount = 40U;
    if (elem_size == 1U || elem_size == 2U || elem_size == 4U) {
      const uint32_t axis_last_row_bytes = output_num * elem_size;
      const bool axis_last_row_aligned = ((axis_last_row_bytes & 31U) == 0U);
      // 本版本所有支持字宽在输入整行非 32B 对齐时走 Gather，否则走 TransData。
      const bool use_axis_last_gather = !axis_last_row_aligned;
      if (use_axis_last_gather) {
        SelectAxisLastGatherTile(outer, output_num, elem_size, tile_r, tile_o, row_bytes, ub_row_pitch_bytes,
                                 row_aligned, axis_last_c_tile, axis_last_groups, axis_last_rows_per_tile,
                                 axis_last_col_tile_count, axis_last_full_repeats, axis_last_tail_repeats);
      } else {
        SelectAxisLastTile(outer, output_num, elem_size, tile_r, tile_o, row_bytes, ub_row_pitch_bytes, row_aligned,
                           axis_last_c_tile, axis_last_groups, axis_last_rows_per_tile, axis_last_col_tile_count,
                           axis_last_full_repeats, axis_last_tail_repeats);
      }
      if (axis_last_rows_per_tile == 0U) {
        axis_last_rows_per_tile = 1U;
      }
      const uint32_t row_tile_count = (outer + axis_last_rows_per_tile - 1U) / axis_last_rows_per_tile;
      block_dim = (row_tile_count < kAxisLastCoreCount) ? row_tile_count : kAxisLastCoreCount;
      if (block_dim == 0U) {
        block_dim = 1U;
      }
      context->SetTilingKey(use_axis_last_gather ? 8U : 5U);
    }
  }

  // 所有字段写入确定值；kernel 仅读取当前 key 所需子集。
  t.set_total_elem(total);
  t.set_output_num(output_num);
  t.set_outer_size(outer);
  t.set_axis_size(axis_sz);
  t.set_inner_size(inner);
  t.set_tile_r(tile_r);
  t.set_tile_o(tile_o);
  t.set_row_bytes(row_bytes);
  t.set_ub_row_pitch_bytes(ub_row_pitch_bytes);
  t.set_row_aligned(row_aligned);
  t.set_axis0_mode(axis0_mode);
  t.set_axis0_inner_tile_elems(axis0_inner_tile_elems);
  t.set_axis0_cores_per_tensor(axis0_cores_per_tensor);
  t.set_axis0_tensor_group_count(axis0_tensor_group_count);
  t.set_axis_last_c_tile(axis_last_c_tile);
  t.set_axis_last_groups(axis_last_groups);
  t.set_axis_last_rows_per_tile(axis_last_rows_per_tile);
  t.set_axis_last_col_tile_count(axis_last_col_tile_count);
  t.set_axis_last_full_repeats(axis_last_full_repeats);
  t.set_axis_last_tail_repeats(axis_last_tail_repeats);

  // kernel 用 GetBlockNum() 作为 tile 循环步长，必须与 host 的任务数一致。
  context->SetBlockDim(block_dim);

  t.SaveToBuffer(context->GetRawTilingData()->GetData(),
                 context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(t.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

// 每个动态输出删除 axis 维，其余维度顺序保持不变。
static ge::graphStatus InferShape(gert::InferShapeContext *context) {
  const gert::Shape *in = context->GetInputShape(0);

  const uint32_t ndim = in->GetDimNum();
  if (ndim == 0 || ndim > 4) {
    return GRAPH_FAILED;
  }

  int64_t axis_attr = 0;
  if (context->GetAttrs()) {
    axis_attr = * context->GetAttrs()->GetAttrPointer<int32_t>(1);
  }
  int32_t axis_pos = static_cast<int32_t>(axis_attr);
  if (axis_pos < 0) {
    axis_pos += static_cast<int32_t>(ndim);
  }
  if (axis_pos < 0 || axis_pos >= static_cast<int32_t>(ndim)) {
    return GRAPH_FAILED;
  }
  if (in->GetDim(axis_pos) <= 0) {
    return GRAPH_FAILED;
  }

  int64_t num_attr = in->GetDim(axis_pos);
  if (context->GetAttrs()) {
    num_attr = * context->GetAttrs()->GetAttrPointer<int32_t>(0);
  }
  if (num_attr <= 0 || num_attr != in->GetDim(axis_pos)) {
    return GRAPH_FAILED;
  }
  const uint32_t out_num = static_cast<uint32_t>(num_attr);

  uint32_t out_ndim = ndim - 1;

  for (uint32_t out_idx = 0; out_idx < out_num; ++out_idx) {
    gert::Shape *out = context->GetOutputShape(out_idx);
    out->SetDimNum(out_ndim);
    uint32_t out_i = 0;
    for (uint32_t i = 0; i < ndim; ++i) {
      if (static_cast<int32_t>(i) == axis_pos) {
        continue;
      }
      out->SetDim(out_i++, in->GetDim(i));
    }
  }
  return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
  // Unpack 只重排数据，全部输出继承输入 dtype。
  int64_t num_attr = 0;
  if (context->GetAttrs()) {
    num_attr = * context->GetAttrs()->GetAttrPointer<int32_t>(0);
  }
  if (num_attr <= 0) {
    return GRAPH_FAILED;
  }

  for (uint32_t i = 0; i < static_cast<uint32_t>(num_attr); ++i) {
    context->SetOutputDataType(i, context->GetInputDataType(0));
  }
  return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class Unpack : public OpDef {
 public:
  explicit Unpack(const char *name) : OpDef(name) {
    // 单个 ND 输入，num 个动态 ND 输出。
    this->Input("input")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32,
                   ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});

    this->Output("output")
        .ParamType(DYNAMIC)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32,
                   ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});

    this->Attr("num").Int();
    this->Attr("axis").Int();

    // 图构建阶段推导 shape/type，运行前由 TilingFunc 选择 kernel key。
    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(Unpack);

}  // namespace ops


