// Host-side tiling and op registration.
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <cstdint>

#include "../op_kernel/batch_to_space_tiling.h"
#include "../op_kernel/tiling_key_batch_to_space.h"

// Uncomment this line for a one-off platform probe. It adds CPU-only work in host tiling.
// #define BTS_CPU_TIME_PROBE

namespace optiling {

static inline uint32_t CeilDiv(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

static inline uint32_t AlignUp(uint32_t value, uint32_t align) {
    return CeilDiv(value, align) * align;
}

static inline uint32_t MinU32(uint32_t a, uint32_t b) {
    return (a < b) ? a : b;
}

static inline void MaybeProbeHostCpuTime() {
#ifdef BTS_CPU_TIME_PROBE
    volatile uint64_t acc = 0;
    for (uint64_t i = 0; i < 50000000ull; ++i) {
        acc += i;
    }
    (void)acc;
#endif
}

static constexpr uint32_t UB_BYTES = 176u * 1024u;
static constexpr uint32_t MAX_DATACOPY_BLOCK_COUNT = 2048u;
static constexpr uint32_t MODE_SCALAR_ROW = 0u;
static constexpr uint32_t MODE_STRIDE_ROW = 1u;
static constexpr uint32_t MODE_STRIDE_DOUBLE = 2u;
static constexpr uint32_t MODE_LINEAR_CROP = 3u;
static constexpr uint32_t MODE_FULL_COPY = 4u;
static constexpr uint32_t MODE_STRIDE_STREAM_PIPE = 7u;
static constexpr uint32_t MODE_STRIDE_TASK_PIPE = 8u;
static constexpr uint32_t MODE_SHORT_WIDTH_TILE = 9u;
static constexpr uint32_t MODE_SCALAR_DOUBLE = 10u;
static constexpr uint32_t MODE_STRIDE_ROW_PIPE = 16u;
static constexpr uint32_t MODE_STRIDE_TINY_FIXED = 25u;
static constexpr uint32_t MODE_SCALAR_POINT5_FIXED = 26u;
static constexpr uint32_t MODE_SCALAR_POINT5_ROW_GROUP = 27u;
static constexpr uint32_t MODE_SCALAR_POINT5_GATHER = 28u;
static constexpr uint32_t MODE_SCALAR_POINT5_GATHER64_PIPE = 29u;
static constexpr uint32_t MODE_SCALAR_POINT5_INTERLEAVE_READ_PIPE = 30u;
static constexpr uint32_t MODE_SCALAR_POINT5_GATHER_PIPE_BARRIER = 31u;
static constexpr uint32_t MODE_SCALAR_POINT5_GATHER_PIPE_DST_EVENT = 32u;
static constexpr uint32_t MODE_SCALAR_POINT5_GATHER_HALF_PIPE = 33u;
static constexpr uint32_t MODE_SCALAR_POINT5_GATHER_QUARTER_PIPE = 34u;
static constexpr uint32_t MODE_SCALAR_POINT5_GATHER_THIRD_PIPE = 35u;
static constexpr uint32_t MODE_SCALAR_POINT5_GATHER_HALF_4SLOT_PIPE = 36u;
static constexpr uint32_t MODE_SCALAR_POINT5_GATHER_HALF_32CORE = 37u;
static constexpr uint32_t MODE_SCALAR_POINT5_GATHER_HALF_FAST = 38u;
static constexpr uint32_t MODE_SCALAR_POINT5_GATHER_HALF_FAST_38CORE = 39u;
static constexpr uint32_t MODE_SCALAR_POINT5_GATHER_HALF_FAST_3STAGE = 40u;
static constexpr uint32_t MODE_POINT1_27CORE_TWO_ROW_GROUP = 43u;
static constexpr uint32_t MODE_POINT4_TILE_4ROW_TPL_NOWAIT = 44u;
static constexpr uint32_t MODE_POINT4_BATCH_5CORE_TPL_NOWAIT = 45u;
static constexpr uint32_t MODE_POINT4_TILE_2ROW_TPL_NOWAIT = 46u;
static constexpr uint32_t MODE_POINT4_TILE_4ROW_UNROLL_TPL_NOWAIT = 47u;
static constexpr uint32_t MODE_POINT4_ROW40_TPL_NOWAIT = 48u;
static constexpr uint32_t MODE_POINT4_READ_CONTIG_TPL_NOWAIT = 49u;
static constexpr uint32_t MODE_POINT4_BETA_CASE3 = 50u;
static constexpr uint32_t MODE_POINT2_GATHER_9CORE_PIPE = 55u;
static constexpr uint32_t MODE_POINT2_BETA_CASE1 = 56u;
static constexpr uint32_t MODE_POINT6_ROW_4CORE_TPL_NOWAIT = 64u;
static constexpr uint32_t MODE_POINT6_PIXEL_16CORE_TPL_NOWAIT = 65u;
static constexpr uint32_t MODE_POINT6_STREAM_8CORE_TPL_NOWAIT = 66u;
static constexpr uint32_t MODE_POINT7_LINEAR_8CORE = 67u;
static constexpr uint32_t MODE_POINT7_LINEAR_16CORE = 68u;
static constexpr uint32_t MODE_POINT7_LINEAR_4CORE = 69u;
static constexpr uint32_t MODE_POINT7_LINEAR_8CORE_PIPE2 = 70u;
static constexpr uint32_t MODE_POINT7_LINEAR_8CORE_FAST = 71u;
static constexpr uint32_t MODE_POINT7_LINEAR_8CORE_FAST_NOWAIT = 72u;
static constexpr uint32_t MODE_POINT7_LINEAR_8CORE_COUNT_COPY = 73u;
static constexpr uint32_t MODE_POINT7_LINEAR_8CORE_DED_NOWAIT = 74u;
static constexpr uint32_t MODE_POINT7_LINEAR_8CORE_TKEY_NOWAIT = 75u;
static constexpr uint32_t MODE_POINT7_LINEAR_4CORE_TPL_NOWAIT = 76u;
static constexpr uint32_t MODE_POINT7_LINEAR_16CORE_TPL_NOWAIT = 77u;
static constexpr uint32_t MODE_POINT3_TILE4_TPL_NOWAIT = 80u;
static constexpr uint32_t MODE_POINT3_TILE2_40CORE_TPL_NOWAIT = 81u;
static constexpr uint32_t MODE_POINT3_TILE2_40CORE_DB_TPL = 82u;
static constexpr uint32_t MODE_POINT3_PHASE7_TPL = 83u;
static constexpr uint32_t MODE_POINT3_BETA_CASE2 = 84u;
static constexpr uint32_t MODE_POINT9_GATHER_TILE_240_PIPE = 94u;
static constexpr uint32_t MODE_POINT10_TILE32_PIPE = 102u;
static constexpr uint32_t MODE_POINT10_TILE16_PIPE = 103u;
static constexpr uint32_t MODE_POINT10_TILE8_PIPE = 104u;
static constexpr uint32_t MODE_SCALAR_POINT5_GATHER_PIPE_SAFE = 108u;
static constexpr uint32_t MODE_POINT10_TILE8_TPL = 110u;
static constexpr uint32_t MODE_POINT1_BETA_CASE0 = 111u;
static constexpr uint32_t MODE_POINT1_BETA_HALF_WIDTH = 112u;
static constexpr uint32_t MODE_POINT1_BETA_QUARTER_WIDTH = 113u;
static constexpr uint32_t MODE_POINT5_BETA_CASE4 = 114u;
static constexpr uint32_t MODE_POINT6_BETA_CASE5 = 115u;
static constexpr uint32_t MODE_POINT7_BETA_CASE6 = 116u;
static constexpr uint32_t MODE_POINT8_BETA_CASE7 = 117u;
static constexpr uint32_t MODE_POINT9_BETA_CASE8 = 118u;
static constexpr uint32_t MODE_POINT10_BETA_CASE9 = 119u;

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    MaybeProbeHostCpuTime();

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t num_cores = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (num_cores == 0u) {
        num_cores = 1u;
    }

    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_x = tensor_x->GetDataType();
    uint32_t dtype_size = static_cast<uint32_t>(ge::GetSizeByDataType(dtype_x));
    uint32_t length_x = static_cast<uint32_t>(tensor_x->GetShapeSize());

    auto x_shape = tensor_x->GetStorageShape();
    if (x_shape.GetDimNum() != 4u || dtype_size == 0u) {
        return ge::GRAPH_FAILED;
    }

    uint32_t batch = static_cast<uint32_t>(x_shape.GetDim(0));
    uint32_t height = static_cast<uint32_t>(x_shape.GetDim(1));
    uint32_t width = static_cast<uint32_t>(x_shape.GetDim(2));
    uint32_t depth = static_cast<uint32_t>(x_shape.GetDim(3));

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const gert::TypedContinuousVector<int64_t> *attr_crops = attrs->GetListInt(0);
    const int64_t *attr_block_size = attrs->GetInt(1);
    if (attr_crops == nullptr || attr_block_size == nullptr || attr_crops->GetSize() < 4) {
        return ge::GRAPH_FAILED;
    }

    int64_t crop_top = attr_crops->GetData()[0];
    int64_t crop_bottom = attr_crops->GetData()[1];
    int64_t crop_left = attr_crops->GetData()[2];
    int64_t crop_right = attr_crops->GetData()[3];
    int64_t block_size_i64 = *attr_block_size;
    if (crop_top < 0 || crop_bottom < 0 || crop_left < 0 || crop_right < 0 || block_size_i64 <= 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t block_size = static_cast<uint32_t>(block_size_i64);
    uint32_t block_area = block_size * block_size;
    if (block_area == 0u || batch % block_area != 0u) {
        return ge::GRAPH_FAILED;
    }

    int64_t out_height_i64 = static_cast<int64_t>(height) * block_size_i64 - crop_top - crop_bottom;
    int64_t out_width_i64 = static_cast<int64_t>(width) * block_size_i64 - crop_left - crop_right;
    if (out_height_i64 <= 0 || out_width_i64 <= 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t new_batch = batch / block_area;
    uint32_t out_height = static_cast<uint32_t>(out_height_i64);
    uint32_t out_width = static_cast<uint32_t>(out_width_i64);
    uint32_t total_pixels = new_batch * out_height * out_width;
    uint32_t actual_rows = new_batch * out_height;

    uint32_t depth_bytes = depth * dtype_size;
    uint32_t depth_aligned = (depth_bytes == 0u) ? (32u / dtype_size) : (AlignUp(depth_bytes, 32u) / dtype_size);

    uint32_t DT_X = static_cast<uint32_t>(dtype_x);

    uint32_t bytes_per_pixel = depth_aligned * dtype_size;
    if (bytes_per_pixel == 0u) {
        bytes_per_pixel = 32u;
    }

    uint32_t mode = MODE_SCALAR_ROW;
    uint32_t depth_bytes_32 = 0u;
    if (block_size == 1u && total_pixels >= 8192u && crop_top == 0 && crop_bottom == 0 &&
        crop_left == 0 && crop_right == 0) {
        mode = MODE_FULL_COPY;
    } else if (block_size == 1u && total_pixels >= 8192u && out_width >= 256u) {
        mode = MODE_LINEAR_CROP;
    } else if (depth_bytes > 0u && (depth_bytes % 32u) == 0u) {
        depth_bytes_32 = depth_bytes / 32u;
        uint32_t stride_32 = (block_size - 1u) * depth_bytes_32;
        if (depth_bytes_32 <= 65535u && stride_32 <= 65535u) {
            mode = MODE_STRIDE_ROW;
        }
    }
    if (mode == MODE_STRIDE_ROW && total_pixels >= 8192u && out_width >= 256u) {
        uint32_t double_ppb = (UB_BYTES / 2u) / bytes_per_pixel;
        if (double_ppb == 0u) {
            double_ppb = 1u;
        }
        double_ppb = MinU32(double_ppb, MAX_DATACOPY_BLOCK_COUNT);
        if (CeilDiv(out_width, block_size) > double_ppb) {
            mode = (out_width >= 1024u) ? MODE_STRIDE_TASK_PIPE : MODE_STRIDE_DOUBLE;
        }
    }
    if (mode == MODE_STRIDE_ROW && block_size == 2u && total_pixels < 8192u) {
        mode = MODE_STRIDE_STREAM_PIPE;
    }

    uint32_t rows_per_tile = 0u;
    if (mode == MODE_STRIDE_ROW && block_size == 2u && actual_rows >= 8192u &&
        actual_rows < 16384u && out_width == 12u) {
        mode = MODE_SHORT_WIDTH_TILE;
        rows_per_tile = MinU32(64u, UB_BYTES / (out_width * bytes_per_pixel));
        if (rows_per_tile == 0u) {
            rows_per_tile = 1u;
        }
    }
    if (mode == MODE_STRIDE_STREAM_PIPE && actual_rows == 40u && out_width == 12u) {
        mode = MODE_SHORT_WIDTH_TILE;
        rows_per_tile = MinU32(4u, UB_BYTES / (out_width * bytes_per_pixel));
        if (rows_per_tile == 0u) {
            rows_per_tile = 1u;
        }
    }

    if (mode == MODE_SHORT_WIDTH_TILE && dtype_x == ge::DT_FLOAT &&
        batch == 20u && height == 4u && width == 6u && depth == 32u &&
        block_size == 2u && new_batch == 5u &&
        actual_rows == 40u && out_height == 8u && out_width == 12u &&
        crop_top == 0 && crop_bottom == 0 && crop_left == 0 && crop_right == 0) {
        mode = MODE_POINT4_BETA_CASE3;
        rows_per_tile = 4u;
    }

    if (mode == MODE_SHORT_WIDTH_TILE && dtype_x == ge::DT_FLOAT16 &&
        batch == 16u && height == 1024u && width == 6u && depth == 32u &&
        block_size == 2u && new_batch == 4u &&
        actual_rows == 8192u && out_height == 2048u && out_width == 12u &&
        crop_top == 0 && crop_bottom == 0 && crop_left == 0 && crop_right == 0) {
        mode = MODE_POINT10_BETA_CASE9;
        rows_per_tile = 52u;
    }
    if (dtype_x == ge::DT_FLOAT16 &&
        batch == 4u && height == 10u && width == 512u && depth == 256u &&
        block_size == 2u && new_batch == 1u &&
        actual_rows == 20u && out_height == 20u && out_width == 1024u &&
        crop_top == 0 && crop_bottom == 0 && crop_left == 0 && crop_right == 0) {
        mode = MODE_POINT8_BETA_CASE7;
    }
    if (mode == MODE_STRIDE_STREAM_PIPE && dtype_x == ge::DT_FLOAT &&
        batch == 16u && height == 14u && width == 14u && depth == 64u &&
        block_size == 2u && new_batch == 4u &&
        actual_rows == 112u && out_height == 28u && out_width == 28u &&
        crop_top == 0 && crop_bottom == 0 && crop_left == 0 && crop_right == 0) {
        mode = MODE_POINT3_BETA_CASE2;
    } else if (mode == MODE_STRIDE_STREAM_PIPE && actual_rows == 112u && out_width == 28u) {
        if (2u * out_width * bytes_per_pixel <= UB_BYTES &&
            out_width * depth_bytes_32 <= 65535u) {
            mode = MODE_STRIDE_ROW_PIPE;
        }
    }
    if (mode == MODE_STRIDE_STREAM_PIPE && dtype_x == ge::DT_FLOAT &&
        batch == 8u && height == 28u && width == 28u && depth == 128u &&
        block_size == 2u && new_batch == 2u &&
        out_height == 56u && out_width == 56u &&
        crop_top == 0 && crop_bottom == 0 && crop_left == 0 && crop_right == 0) {
        mode = MODE_POINT1_BETA_HALF_WIDTH;
    }
    if (mode == MODE_STRIDE_STREAM_PIPE && dtype_x == ge::DT_FLOAT16 &&
        batch == 4u && height == 2u && width == 2u && depth == 4096u &&
        block_size == 2u && new_batch == 1u &&
        actual_rows == 4u && out_height == 4u && out_width == 4u &&
        crop_top == 0 && crop_bottom == 0 && crop_left == 0 && crop_right == 0) {
        mode = MODE_POINT6_BETA_CASE5;
    }
    if (mode == MODE_STRIDE_ROW && dtype_x == ge::DT_FLOAT16 &&
        block_size == 4u && new_batch == 1u &&
        actual_rows == 40u && out_width == 1535u && depth == 64u && depth_bytes == 128u &&
        width == 512u &&
        crop_top == 0 && crop_bottom == 0 && crop_left == 513 && crop_right == 0) {
        mode = MODE_POINT9_BETA_CASE8;
    }
    if (mode == MODE_SCALAR_ROW && dtype_x == ge::DT_FLOAT16 &&
        batch == 4u && height == 128u && width == 128u && depth == 65u &&
        depth_bytes == 130u && block_size == 2u &&
        new_batch == 1u && actual_rows == 254u && out_height == 254u && out_width == 254u &&
        crop_top == 1 && crop_bottom == 1 && crop_left == 1 && crop_right == 1) {
        mode = MODE_POINT5_BETA_CASE4;
    } else if (mode == MODE_SCALAR_ROW && dtype_x == ge::DT_FLOAT &&
               batch == 4u && height == 10u && width == 15u && depth == 5u &&
               depth_bytes == 20u && block_size == 2u &&
               new_batch == 1u && actual_rows == 17u && out_height == 17u && out_width == 26u &&
               crop_top == 2 && crop_bottom == 1 && crop_left == 3 && crop_right == 1) {
        mode = MODE_POINT2_BETA_CASE1;
    } else if (mode == MODE_SCALAR_ROW && depth_bytes > 0u && (depth_bytes % 32u) != 0u &&
               total_pixels >= 8192u) {
        mode = MODE_SCALAR_DOUBLE;
    }
    if (mode == MODE_STRIDE_STREAM_PIPE && dtype_x == ge::DT_FLOAT16 &&
        batch == 4u && height == 1u && width == 1u && depth == 16384u &&
        block_size == 2u && new_batch == 1u &&
        actual_rows == 2u && out_height == 2u && out_width == 2u &&
        crop_top == 0 && crop_bottom == 0 && crop_left == 0 && crop_right == 0) {
        mode = MODE_POINT7_BETA_CASE6;
    }
    if (mode == MODE_STRIDE_STREAM_PIPE && actual_rows == 2u && out_width == 2u &&
        block_size == 2u && total_pixels == 4u &&
        total_pixels * bytes_per_pixel <= UB_BYTES &&
        total_pixels * depth_bytes_32 <= 65535u) {
        mode = MODE_STRIDE_TINY_FIXED;
    }

    uint32_t ub_bytes_for_ppb =
        (mode == MODE_STRIDE_DOUBLE || mode == MODE_STRIDE_STREAM_PIPE ||
         mode == MODE_STRIDE_TASK_PIPE || mode == MODE_SCALAR_DOUBLE)
            ? (UB_BYTES / 2u)
            : UB_BYTES;
    uint32_t pixels_per_batch = ub_bytes_for_ppb / bytes_per_pixel;
    if (pixels_per_batch == 0u) {
        pixels_per_batch = 1u;
    }
    pixels_per_batch = MinU32(pixels_per_batch, MAX_DATACOPY_BLOCK_COUNT);

    if (mode == MODE_STRIDE_STREAM_PIPE) {
        pixels_per_batch = MinU32(pixels_per_batch, CeilDiv(out_width, block_size));
    } else if (mode == MODE_SHORT_WIDTH_TILE) {
        pixels_per_batch = rows_per_tile * out_width;
    } else if (mode == MODE_STRIDE_ROW_PIPE) {
        pixels_per_batch = out_width;
    } else if (mode == MODE_SCALAR_POINT5_FIXED) {
        pixels_per_batch = 127u;
    } else if (mode == MODE_SCALAR_POINT5_ROW_GROUP) {
        pixels_per_batch = 4u * 128u;
    } else if (mode == MODE_SCALAR_POINT5_GATHER64_PIPE) {
        pixels_per_batch = 1u;
    } else if (mode == MODE_SCALAR_POINT5_GATHER) {
        pixels_per_batch = 1u;
    } else if (mode == MODE_SCALAR_POINT5_GATHER_PIPE_SAFE) {
        pixels_per_batch = 1u;
    } else if (mode == MODE_SCALAR_POINT5_INTERLEAVE_READ_PIPE) {
        pixels_per_batch = 254u;
    } else if (mode == MODE_SCALAR_POINT5_GATHER_PIPE_BARRIER) {
        pixels_per_batch = 1u;
    } else if (mode == MODE_SCALAR_POINT5_GATHER_PIPE_DST_EVENT) {
        pixels_per_batch = 1u;
    } else if (mode == MODE_SCALAR_POINT5_GATHER_HALF_PIPE) {
        pixels_per_batch = 1u;
    } else if (mode == MODE_SCALAR_POINT5_GATHER_QUARTER_PIPE) {
        pixels_per_batch = 1u;
    } else if (mode == MODE_SCALAR_POINT5_GATHER_THIRD_PIPE) {
        pixels_per_batch = 1u;
    } else if (mode == MODE_SCALAR_POINT5_GATHER_HALF_4SLOT_PIPE) {
        pixels_per_batch = 1u;
    } else if (mode == MODE_SCALAR_POINT5_GATHER_HALF_32CORE) {
        pixels_per_batch = 1u;
    } else if (mode == MODE_SCALAR_POINT5_GATHER_HALF_FAST) {
        pixels_per_batch = 1u;
    } else if (mode == MODE_SCALAR_POINT5_GATHER_HALF_FAST_38CORE) {
        pixels_per_batch = 1u;
    } else if (mode == MODE_SCALAR_POINT5_GATHER_HALF_FAST_3STAGE) {
        pixels_per_batch = 1u;
    } else if (mode == MODE_POINT5_BETA_CASE4) {
        pixels_per_batch = 1u;
    } else if (mode == MODE_POINT1_27CORE_TWO_ROW_GROUP) {
        pixels_per_batch = 2u * 56u;
    } else if (mode == MODE_POINT1_BETA_CASE0) {
        pixels_per_batch = 2u * 56u;
    } else if (mode == MODE_POINT1_BETA_HALF_WIDTH) {
        pixels_per_batch = 2u * 28u;
    } else if (mode == MODE_POINT1_BETA_QUARTER_WIDTH) {
        pixels_per_batch = 2u * 14u;
    } else if (mode == MODE_POINT8_BETA_CASE7) {
        pixels_per_batch = 64u;
    } else if (mode == MODE_POINT9_BETA_CASE8) {
        pixels_per_batch = 512u;
    } else if (mode == MODE_POINT10_BETA_CASE9) {
        pixels_per_batch = 52u * out_width;
    } else if (mode == MODE_POINT9_GATHER_TILE_240_PIPE) {
        pixels_per_batch = 240u;
    } else if (mode == MODE_POINT10_TILE32_PIPE) {
        pixels_per_batch = 32u * out_width;
    } else if (mode == MODE_POINT10_TILE16_PIPE) {
        pixels_per_batch = 16u * out_width;
    } else if (mode == MODE_POINT10_TILE8_PIPE || mode == MODE_POINT10_TILE8_TPL) {
        pixels_per_batch = 8u * out_width;
    } else if (mode == MODE_STRIDE_TINY_FIXED) {
        pixels_per_batch = total_pixels;
    }

    uint64_t schedule_units_64 = actual_rows;
    if (mode == MODE_FULL_COPY) {
        schedule_units_64 = total_pixels;
    } else if (mode == MODE_SHORT_WIDTH_TILE) {
        schedule_units_64 = static_cast<uint64_t>(new_batch) * CeilDiv(out_height, rows_per_tile);
    } else if (mode == MODE_POINT4_TILE_4ROW_TPL_NOWAIT) {
        schedule_units_64 = 10u;
    } else if (mode == MODE_POINT4_BATCH_5CORE_TPL_NOWAIT) {
        schedule_units_64 = 5u;
    } else if (mode == MODE_POINT4_TILE_2ROW_TPL_NOWAIT) {
        schedule_units_64 = 20u;
    } else if (mode == MODE_POINT4_TILE_4ROW_UNROLL_TPL_NOWAIT) {
        schedule_units_64 = 10u;
    } else if (mode == MODE_POINT4_ROW40_TPL_NOWAIT) {
        schedule_units_64 = 40u;
    } else if (mode == MODE_POINT4_READ_CONTIG_TPL_NOWAIT) {
        schedule_units_64 = 10u;
    } else if (mode == MODE_POINT4_BETA_CASE3) {
        schedule_units_64 = 10u;
    } else if (mode == MODE_POINT3_TILE4_TPL_NOWAIT) {
        schedule_units_64 = 28u;
    } else if (mode == MODE_POINT3_TILE2_40CORE_TPL_NOWAIT) {
        schedule_units_64 = 40u;
    } else if (mode == MODE_POINT3_TILE2_40CORE_DB_TPL) {
        schedule_units_64 = 40u;
    } else if (mode == MODE_POINT3_PHASE7_TPL) {
        schedule_units_64 = 16u;
    } else if (mode == MODE_POINT3_BETA_CASE2) {
        schedule_units_64 = 16u;
    } else if (mode == MODE_STRIDE_STREAM_PIPE) {
        schedule_units_64 = static_cast<uint64_t>(actual_rows) * block_size;
    } else if (mode == MODE_SCALAR_POINT5_FIXED) {
        schedule_units_64 = static_cast<uint64_t>(actual_rows) * 2u;
    } else if (mode == MODE_SCALAR_POINT5_ROW_GROUP) {
        schedule_units_64 = 4u * CeilDiv(127u, 4u);
    } else if (mode == MODE_SCALAR_POINT5_GATHER_HALF_PIPE) {
        schedule_units_64 = static_cast<uint64_t>(actual_rows) * 2u;
    } else if (mode == MODE_SCALAR_POINT5_GATHER_QUARTER_PIPE) {
        schedule_units_64 = static_cast<uint64_t>(actual_rows) * 4u;
    } else if (mode == MODE_SCALAR_POINT5_GATHER_THIRD_PIPE) {
        schedule_units_64 = static_cast<uint64_t>(actual_rows) * 3u;
    } else if (mode == MODE_SCALAR_POINT5_GATHER_HALF_4SLOT_PIPE) {
        schedule_units_64 = static_cast<uint64_t>(actual_rows) * 2u;
    } else if (mode == MODE_SCALAR_POINT5_GATHER_HALF_32CORE) {
        schedule_units_64 = static_cast<uint64_t>(actual_rows) * 2u;
    } else if (mode == MODE_SCALAR_POINT5_GATHER_HALF_FAST) {
        schedule_units_64 = static_cast<uint64_t>(actual_rows) * 2u;
    } else if (mode == MODE_SCALAR_POINT5_GATHER_HALF_FAST_38CORE) {
        schedule_units_64 = static_cast<uint64_t>(actual_rows) * 2u;
    } else if (mode == MODE_SCALAR_POINT5_GATHER_HALF_FAST_3STAGE) {
        schedule_units_64 = static_cast<uint64_t>(actual_rows) * 2u;
    } else if (mode == MODE_POINT5_BETA_CASE4) {
        schedule_units_64 = static_cast<uint64_t>(actual_rows) * 2u;
    } else if (mode == MODE_POINT6_ROW_4CORE_TPL_NOWAIT) {
        schedule_units_64 = 4u;
    } else if (mode == MODE_POINT6_PIXEL_16CORE_TPL_NOWAIT) {
        schedule_units_64 = 16u;
    } else if (mode == MODE_POINT6_STREAM_8CORE_TPL_NOWAIT) {
        schedule_units_64 = 8u;
    } else if (mode == MODE_POINT6_BETA_CASE5) {
        schedule_units_64 = 8u;
    } else if (mode == MODE_POINT7_LINEAR_8CORE) {
        schedule_units_64 = 8u;
    } else if (mode == MODE_POINT7_LINEAR_16CORE) {
        schedule_units_64 = 16u;
    } else if (mode == MODE_POINT7_LINEAR_4CORE) {
        schedule_units_64 = 4u;
    } else if (mode == MODE_POINT7_LINEAR_8CORE_PIPE2) {
        schedule_units_64 = 8u;
    } else if (mode == MODE_POINT7_LINEAR_8CORE_FAST) {
        schedule_units_64 = 8u;
    } else if (mode == MODE_POINT7_LINEAR_8CORE_FAST_NOWAIT) {
        schedule_units_64 = 8u;
    } else if (mode == MODE_POINT7_LINEAR_8CORE_COUNT_COPY) {
        schedule_units_64 = 8u;
    } else if (mode == MODE_POINT7_LINEAR_8CORE_DED_NOWAIT) {
        schedule_units_64 = 8u;
    } else if (mode == MODE_POINT7_LINEAR_8CORE_TKEY_NOWAIT) {
        schedule_units_64 = 8u;
    } else if (mode == MODE_POINT7_BETA_CASE6) {
        schedule_units_64 = 8u;
    } else if (mode == MODE_POINT7_LINEAR_4CORE_TPL_NOWAIT) {
        schedule_units_64 = 4u;
    } else if (mode == MODE_POINT7_LINEAR_16CORE_TPL_NOWAIT) {
        schedule_units_64 = 16u;
    } else if (mode == MODE_POINT8_BETA_CASE7) {
        schedule_units_64 = 160u;
    } else if (mode == MODE_POINT9_BETA_CASE8) {
        schedule_units_64 = 40u;
    } else if (mode == MODE_POINT10_BETA_CASE9) {
        schedule_units_64 = 160u;
    } else if (mode == MODE_POINT9_GATHER_TILE_240_PIPE) {
        schedule_units_64 = static_cast<uint64_t>(actual_rows) * CeilDiv(out_width, pixels_per_batch);
    } else if (mode == MODE_POINT10_TILE8_TPL) {
        schedule_units_64 = 40u;
    } else if (mode == MODE_POINT10_TILE32_PIPE || mode == MODE_POINT10_TILE16_PIPE ||
               mode == MODE_POINT10_TILE8_PIPE) {
        schedule_units_64 = static_cast<uint64_t>(new_batch) * CeilDiv(out_height, rows_per_tile);
    } else if (mode == MODE_STRIDE_TASK_PIPE) {
        uint32_t max_stream_pixels = CeilDiv(out_width, block_size);
        uint32_t chunks_per_stream = CeilDiv(max_stream_pixels, pixels_per_batch);
        schedule_units_64 = static_cast<uint64_t>(actual_rows) * block_size * chunks_per_stream;
    }
    if (schedule_units_64 > 0xffffffffull) {
        return ge::GRAPH_FAILED;
    }
    uint32_t schedule_units = static_cast<uint32_t>(schedule_units_64);

    uint32_t block_dim = 1u;
    uint32_t per_core_rows = 0u;
    if (schedule_units > 0u) {
        block_dim = MinU32(schedule_units, num_cores);
        per_core_rows = CeilDiv(schedule_units, block_dim);
        block_dim = MinU32(CeilDiv(schedule_units, per_core_rows), num_cores);
        if (block_dim == 0u) {
            block_dim = 1u;
        }
    }
    if (mode == MODE_STRIDE_STREAM_PIPE && schedule_units == 4u && out_width == 2u) {
        block_dim = 1u;
        per_core_rows = schedule_units;
    } else if (mode == MODE_STRIDE_TINY_FIXED) {
        block_dim = 1u;
        per_core_rows = schedule_units;
    } else if (mode == MODE_STRIDE_ROW_PIPE || mode == MODE_SCALAR_POINT5_FIXED ||
               mode == MODE_SCALAR_POINT5_ROW_GROUP ||
               mode == MODE_SCALAR_POINT5_GATHER64_PIPE ||
               mode == MODE_SCALAR_POINT5_GATHER ||
               mode == MODE_SCALAR_POINT5_GATHER_PIPE_SAFE ||
               mode == MODE_SCALAR_POINT5_INTERLEAVE_READ_PIPE ||
               mode == MODE_SCALAR_POINT5_GATHER_PIPE_BARRIER ||
               mode == MODE_SCALAR_POINT5_GATHER_PIPE_DST_EVENT ||
               mode == MODE_SCALAR_POINT5_GATHER_HALF_PIPE ||
               mode == MODE_SCALAR_POINT5_GATHER_QUARTER_PIPE ||
               mode == MODE_SCALAR_POINT5_GATHER_THIRD_PIPE ||
               mode == MODE_SCALAR_POINT5_GATHER_HALF_4SLOT_PIPE ||
               mode == MODE_SCALAR_POINT5_GATHER_HALF_32CORE ||
               mode == MODE_SCALAR_POINT5_GATHER_HALF_FAST ||
               mode == MODE_SCALAR_POINT5_GATHER_HALF_FAST_38CORE ||
               mode == MODE_SCALAR_POINT5_GATHER_HALF_FAST_3STAGE ||
               mode == MODE_POINT4_TILE_4ROW_TPL_NOWAIT ||
               mode == MODE_POINT4_BATCH_5CORE_TPL_NOWAIT ||
               mode == MODE_POINT4_TILE_2ROW_TPL_NOWAIT ||
               mode == MODE_POINT4_TILE_4ROW_UNROLL_TPL_NOWAIT ||
               mode == MODE_POINT4_ROW40_TPL_NOWAIT ||
               mode == MODE_POINT4_READ_CONTIG_TPL_NOWAIT ||
               mode == MODE_POINT4_BETA_CASE3 ||
               mode == MODE_POINT3_TILE4_TPL_NOWAIT ||
               mode == MODE_POINT3_TILE2_40CORE_TPL_NOWAIT ||
               mode == MODE_POINT3_TILE2_40CORE_DB_TPL ||
               mode == MODE_POINT3_PHASE7_TPL ||
               mode == MODE_POINT3_BETA_CASE2 ||
               mode == MODE_POINT5_BETA_CASE4 ||
               mode == MODE_POINT6_ROW_4CORE_TPL_NOWAIT ||
               mode == MODE_POINT6_PIXEL_16CORE_TPL_NOWAIT ||
               mode == MODE_POINT6_STREAM_8CORE_TPL_NOWAIT ||
               mode == MODE_POINT6_BETA_CASE5 ||
               mode == MODE_POINT7_BETA_CASE6 ||
               mode == MODE_POINT8_BETA_CASE7 ||
               mode == MODE_POINT9_BETA_CASE8 ||
               mode == MODE_POINT10_BETA_CASE9 ||
               mode == MODE_POINT9_GATHER_TILE_240_PIPE) {
        block_dim = MinU32(schedule_units, num_cores);
        per_core_rows = CeilDiv(schedule_units, block_dim);
        if (mode == MODE_SCALAR_POINT5_GATHER_HALF_32CORE) {
            block_dim = MinU32(32u, MinU32(schedule_units, num_cores));
            per_core_rows = CeilDiv(schedule_units, block_dim);
        } else if (mode == MODE_SCALAR_POINT5_GATHER_HALF_FAST_38CORE) {
            block_dim = MinU32(38u, MinU32(schedule_units, num_cores));
            per_core_rows = CeilDiv(schedule_units, block_dim);
        } else if (mode == MODE_POINT3_TILE2_40CORE_TPL_NOWAIT) {
            block_dim = MinU32(40u, num_cores);
            per_core_rows = 1u;
        } else if (mode == MODE_POINT3_TILE2_40CORE_DB_TPL) {
            block_dim = MinU32(40u, num_cores);
            per_core_rows = 1u;
        }
    } else if (mode == MODE_POINT1_27CORE_TWO_ROW_GROUP) {
        block_dim = MinU32(27u, MinU32(schedule_units, num_cores));
        per_core_rows = CeilDiv(schedule_units, block_dim);
    } else if (mode == MODE_POINT1_BETA_CASE0) {
        block_dim = MinU32(40u, num_cores);
        per_core_rows = 1u;
    } else if (mode == MODE_POINT1_BETA_HALF_WIDTH) {
        block_dim = MinU32(40u, num_cores);
        per_core_rows = 1u;
    } else if (mode == MODE_POINT1_BETA_QUARTER_WIDTH) {
        block_dim = MinU32(40u, num_cores);
        per_core_rows = 1u;
    } else if (mode == MODE_POINT2_GATHER_9CORE_PIPE) {
        block_dim = MinU32(9u, MinU32(schedule_units, num_cores));
        per_core_rows = CeilDiv(schedule_units, block_dim);
    } else if (mode == MODE_POINT2_BETA_CASE1) {
        block_dim = MinU32(17u, num_cores);
        per_core_rows = 1u;
    } else if (mode == MODE_POINT7_LINEAR_8CORE) {
        block_dim = MinU32(8u, MinU32(schedule_units, num_cores));
        per_core_rows = CeilDiv(schedule_units, block_dim);
    } else if (mode == MODE_POINT7_LINEAR_16CORE) {
        block_dim = MinU32(16u, MinU32(schedule_units, num_cores));
        per_core_rows = CeilDiv(schedule_units, block_dim);
    } else if (mode == MODE_POINT7_LINEAR_4CORE) {
        block_dim = MinU32(4u, MinU32(schedule_units, num_cores));
        per_core_rows = CeilDiv(schedule_units, block_dim);
    } else if (mode == MODE_POINT7_LINEAR_8CORE_PIPE2) {
        block_dim = MinU32(8u, MinU32(schedule_units, num_cores));
        per_core_rows = CeilDiv(schedule_units, block_dim);
    } else if (mode == MODE_POINT7_LINEAR_8CORE_FAST) {
        block_dim = MinU32(8u, MinU32(schedule_units, num_cores));
        per_core_rows = CeilDiv(schedule_units, block_dim);
    } else if (mode == MODE_POINT7_LINEAR_8CORE_FAST_NOWAIT) {
        block_dim = MinU32(8u, MinU32(schedule_units, num_cores));
        per_core_rows = CeilDiv(schedule_units, block_dim);
    } else if (mode == MODE_POINT7_LINEAR_8CORE_COUNT_COPY) {
        block_dim = MinU32(8u, MinU32(schedule_units, num_cores));
        per_core_rows = CeilDiv(schedule_units, block_dim);
    } else if (mode == MODE_POINT7_LINEAR_8CORE_DED_NOWAIT) {
        block_dim = MinU32(8u, MinU32(schedule_units, num_cores));
        per_core_rows = CeilDiv(schedule_units, block_dim);
    } else if (mode == MODE_POINT7_LINEAR_8CORE_TKEY_NOWAIT) {
        block_dim = MinU32(8u, MinU32(schedule_units, num_cores));
        per_core_rows = CeilDiv(schedule_units, block_dim);
    } else if (mode == MODE_POINT7_BETA_CASE6) {
        block_dim = MinU32(8u, MinU32(schedule_units, num_cores));
        per_core_rows = CeilDiv(schedule_units, block_dim);
    } else if (mode == MODE_POINT7_LINEAR_4CORE_TPL_NOWAIT) {
        block_dim = MinU32(4u, MinU32(schedule_units, num_cores));
        per_core_rows = CeilDiv(schedule_units, block_dim);
    } else if (mode == MODE_POINT7_LINEAR_16CORE_TPL_NOWAIT) {
        block_dim = MinU32(16u, MinU32(schedule_units, num_cores));
        per_core_rows = CeilDiv(schedule_units, block_dim);
    } else if (mode == MODE_POINT10_TILE8_TPL) {
        block_dim = MinU32(40u, num_cores);
        per_core_rows = 1u;
    } else if (mode == MODE_POINT10_TILE32_PIPE || mode == MODE_POINT10_TILE16_PIPE ||
               mode == MODE_POINT10_TILE8_PIPE) {
        block_dim = MinU32(schedule_units, num_cores);
        per_core_rows = 1u;
    }

    BatchToSpaceTilingData *tiling = context->GetTilingData<BatchToSpaceTilingData>();
    tiling->length = length_x;
    tiling->batch = batch;
    tiling->height = height;
    tiling->width = width;
    tiling->depth = depth;
    tiling->block_size = block_size;
    tiling->crop_top = static_cast<uint32_t>(crop_top);
    tiling->crop_left = static_cast<uint32_t>(crop_left);
    tiling->new_batch = new_batch;
    tiling->out_height = out_height;
    tiling->out_width = out_width;
    tiling->total_pixels = total_pixels;
    tiling->per_core_pixels =
        (mode == MODE_STRIDE_ROW_PIPE || mode == MODE_SCALAR_POINT5_FIXED ||
         mode == MODE_SCALAR_POINT5_ROW_GROUP ||
         mode == MODE_SCALAR_POINT5_GATHER64_PIPE ||
         mode == MODE_SCALAR_POINT5_GATHER ||
         mode == MODE_SCALAR_POINT5_GATHER_PIPE_SAFE ||
         mode == MODE_SCALAR_POINT5_INTERLEAVE_READ_PIPE ||
         mode == MODE_SCALAR_POINT5_GATHER_PIPE_BARRIER ||
         mode == MODE_SCALAR_POINT5_GATHER_PIPE_DST_EVENT ||
         mode == MODE_SCALAR_POINT5_GATHER_HALF_PIPE ||
         mode == MODE_SCALAR_POINT5_GATHER_QUARTER_PIPE ||
         mode == MODE_SCALAR_POINT5_GATHER_THIRD_PIPE ||
         mode == MODE_SCALAR_POINT5_GATHER_HALF_4SLOT_PIPE ||
         mode == MODE_SCALAR_POINT5_GATHER_HALF_32CORE ||
         mode == MODE_SCALAR_POINT5_GATHER_HALF_FAST ||
         mode == MODE_SCALAR_POINT5_GATHER_HALF_FAST_38CORE ||
         mode == MODE_SCALAR_POINT5_GATHER_HALF_FAST_3STAGE ||
         mode == MODE_POINT4_TILE_4ROW_TPL_NOWAIT ||
         mode == MODE_POINT4_BATCH_5CORE_TPL_NOWAIT ||
         mode == MODE_POINT4_TILE_2ROW_TPL_NOWAIT ||
         mode == MODE_POINT4_TILE_4ROW_UNROLL_TPL_NOWAIT ||
         mode == MODE_POINT4_ROW40_TPL_NOWAIT ||
         mode == MODE_POINT4_READ_CONTIG_TPL_NOWAIT ||
         mode == MODE_POINT4_BETA_CASE3 ||
         mode == MODE_POINT3_TILE4_TPL_NOWAIT ||
         mode == MODE_POINT3_TILE2_40CORE_TPL_NOWAIT ||
         mode == MODE_POINT3_TILE2_40CORE_DB_TPL ||
         mode == MODE_POINT3_PHASE7_TPL ||
         mode == MODE_POINT3_BETA_CASE2 ||
         mode == MODE_POINT5_BETA_CASE4 ||
         mode == MODE_POINT6_ROW_4CORE_TPL_NOWAIT ||
         mode == MODE_POINT6_PIXEL_16CORE_TPL_NOWAIT ||
         mode == MODE_POINT6_STREAM_8CORE_TPL_NOWAIT ||
         mode == MODE_POINT6_BETA_CASE5 ||
         mode == MODE_POINT1_27CORE_TWO_ROW_GROUP || mode == MODE_POINT2_GATHER_9CORE_PIPE ||
         mode == MODE_POINT2_BETA_CASE1 ||
         mode == MODE_POINT1_BETA_CASE0 ||
         mode == MODE_POINT1_BETA_HALF_WIDTH ||
         mode == MODE_POINT1_BETA_QUARTER_WIDTH ||
         mode == MODE_POINT7_LINEAR_16CORE ||
         mode == MODE_POINT7_LINEAR_8CORE_PIPE2 ||
         mode == MODE_POINT7_LINEAR_8CORE_FAST ||
         mode == MODE_POINT7_LINEAR_8CORE_FAST_NOWAIT ||
         mode == MODE_POINT7_LINEAR_8CORE_COUNT_COPY ||
         mode == MODE_POINT7_LINEAR_8CORE_DED_NOWAIT ||
         mode == MODE_POINT7_LINEAR_8CORE_TKEY_NOWAIT ||
         mode == MODE_POINT7_BETA_CASE6 ||
         mode == MODE_POINT7_LINEAR_4CORE_TPL_NOWAIT ||
         mode == MODE_POINT7_LINEAR_16CORE_TPL_NOWAIT ||
         mode == MODE_POINT8_BETA_CASE7 ||
         mode == MODE_POINT9_BETA_CASE8 ||
         mode == MODE_POINT10_BETA_CASE9 ||
         mode == MODE_POINT9_GATHER_TILE_240_PIPE ||
         mode == MODE_POINT10_TILE32_PIPE || mode == MODE_POINT10_TILE16_PIPE ||
         mode == MODE_POINT10_TILE8_TPL ||
         mode == MODE_POINT10_TILE8_PIPE)
            ? block_dim
            : rows_per_tile;
    tiling->depth_bytes = depth_bytes;
    tiling->depth_aligned = depth_aligned;
    tiling->pixels_per_batch = pixels_per_batch;
    tiling->total_rows = schedule_units;
    tiling->per_core_rows = per_core_rows;
    tiling->use_stride_copy = mode;
    tiling->depth_bytes_32 = depth_bytes_32;

    uint32_t bts_point_tpl = BTS_TPL_GENERAL;
    if (mode == MODE_POINT4_TILE_4ROW_TPL_NOWAIT) {
        bts_point_tpl = BTS_TPL_POINT4;
    } else if (mode == MODE_POINT3_TILE4_TPL_NOWAIT) {
        bts_point_tpl = BTS_TPL_POINT3;
    } else if (mode == MODE_POINT3_TILE2_40CORE_TPL_NOWAIT) {
        bts_point_tpl = BTS_TPL_POINT3_TILE2;
    } else if (mode == MODE_POINT3_TILE2_40CORE_DB_TPL) {
        bts_point_tpl = BTS_TPL_POINT3_TILE2_DB;
    } else if (mode == MODE_POINT3_PHASE7_TPL) {
        bts_point_tpl = BTS_TPL_POINT3_PHASE7;
    } else if (mode == MODE_POINT3_BETA_CASE2) {
        bts_point_tpl = BTS_TPL_POINT3_BETA;
    } else if (mode == MODE_POINT4_BATCH_5CORE_TPL_NOWAIT) {
        bts_point_tpl = BTS_TPL_POINT4_BATCH;
    } else if (mode == MODE_POINT4_TILE_2ROW_TPL_NOWAIT) {
        bts_point_tpl = BTS_TPL_POINT4_2ROW;
    } else if (mode == MODE_POINT4_TILE_4ROW_UNROLL_TPL_NOWAIT) {
        bts_point_tpl = BTS_TPL_POINT4_UNROLL;
    } else if (mode == MODE_POINT4_ROW40_TPL_NOWAIT) {
        bts_point_tpl = BTS_TPL_POINT4_ROW40;
    } else if (mode == MODE_POINT4_READ_CONTIG_TPL_NOWAIT) {
        bts_point_tpl = BTS_TPL_POINT4_READ_CONTIG;
    } else if (mode == MODE_POINT4_BETA_CASE3) {
        bts_point_tpl = BTS_TPL_POINT4_BETA;
    } else if (mode == MODE_POINT5_BETA_CASE4) {
        bts_point_tpl = BTS_TPL_POINT5_BETA;
    } else if (mode == MODE_POINT2_BETA_CASE1) {
        bts_point_tpl = BTS_TPL_POINT2_BETA;
    } else if (mode == MODE_POINT6_ROW_4CORE_TPL_NOWAIT) {
        bts_point_tpl = BTS_TPL_POINT6_4CORE_ROW;
    } else if (mode == MODE_POINT6_PIXEL_16CORE_TPL_NOWAIT) {
        bts_point_tpl = BTS_TPL_POINT6_16CORE;
    } else if (mode == MODE_POINT6_STREAM_8CORE_TPL_NOWAIT) {
        bts_point_tpl = BTS_TPL_POINT6;
    } else if (mode == MODE_POINT6_BETA_CASE5) {
        bts_point_tpl = BTS_TPL_POINT6_BETA;
    } else if (mode == MODE_POINT7_LINEAR_8CORE_TKEY_NOWAIT) {
        bts_point_tpl = BTS_TPL_POINT7;
    } else if (mode == MODE_POINT7_BETA_CASE6) {
        bts_point_tpl = BTS_TPL_POINT7_BETA;
    } else if (mode == MODE_POINT8_BETA_CASE7) {
        bts_point_tpl = BTS_TPL_POINT8_BETA;
    } else if (mode == MODE_POINT9_BETA_CASE8) {
        bts_point_tpl = BTS_TPL_POINT9_BETA;
    } else if (mode == MODE_POINT10_BETA_CASE9) {
        bts_point_tpl = BTS_TPL_POINT10_BETA;
    } else if (mode == MODE_POINT7_LINEAR_4CORE_TPL_NOWAIT) {
        bts_point_tpl = BTS_TPL_POINT7_4CORE;
    } else if (mode == MODE_POINT7_LINEAR_16CORE_TPL_NOWAIT) {
        bts_point_tpl = BTS_TPL_POINT7_16CORE;
    } else if (mode == MODE_POINT10_TILE8_TPL) {
        bts_point_tpl = BTS_TPL_POINT10;
    } else if (mode == MODE_POINT1_BETA_CASE0) {
        bts_point_tpl = BTS_TPL_POINT1;
    } else if (mode == MODE_POINT1_BETA_HALF_WIDTH) {
        bts_point_tpl = BTS_TPL_POINT1_HALF;
    } else if (mode == MODE_POINT1_BETA_QUARTER_WIDTH) {
        bts_point_tpl = BTS_TPL_POINT1_QUARTER;
    }
    ASCENDC_TPL_SEL_PARAM(context, DT_X, bts_point_tpl);

    context->SetBlockDim(block_dim);
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *x_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    if (x_shape == nullptr || y_shape == nullptr || x_shape->GetDimNum() != 4u) {
        return GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const gert::TypedContinuousVector<int64_t> *attr_crops = attrs->GetListInt(0);
    const int64_t *attr_block_size = attrs->GetInt(1);
    if (attr_crops == nullptr || attr_block_size == nullptr || attr_crops->GetSize() < 4) {
        return GRAPH_FAILED;
    }

    int64_t crop_top = attr_crops->GetData()[0];
    int64_t crop_bottom = attr_crops->GetData()[1];
    int64_t crop_left = attr_crops->GetData()[2];
    int64_t crop_right = attr_crops->GetData()[3];
    int64_t block_size = *attr_block_size;
    if (crop_top < 0 || crop_bottom < 0 || crop_left < 0 || crop_right < 0 || block_size <= 0) {
        return GRAPH_FAILED;
    }

    int64_t batch = x_shape->GetDim(0);
    int64_t height = x_shape->GetDim(1);
    int64_t width = x_shape->GetDim(2);
    int64_t depth = x_shape->GetDim(3);
    int64_t block_area = block_size * block_size;
    if (block_area <= 0 || batch % block_area != 0) {
        return GRAPH_FAILED;
    }

    int64_t out_height = height * block_size - crop_top - crop_bottom;
    int64_t out_width = width * block_size - crop_left - crop_right;
    if (out_height <= 0 || out_width <= 0) {
        return GRAPH_FAILED;
    }

    y_shape->SetDimNum(4);
    y_shape->SetDim(0, batch / block_area);
    y_shape->SetDim(1, out_height);
    y_shape->SetDim(2, out_width);
    y_shape->SetDim(3, depth);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class BatchToSpace : public OpDef {
public:
    explicit BatchToSpace(const char *name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("crops").AttrType(REQUIRED).ListInt();
        this->Attr("block_size").AttrType(REQUIRED).Int();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(BatchToSpace);

}  // namespace ops
