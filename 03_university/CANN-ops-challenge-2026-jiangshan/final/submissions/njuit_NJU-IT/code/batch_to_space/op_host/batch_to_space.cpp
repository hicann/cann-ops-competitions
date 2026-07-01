#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/batch_to_space_tiling.h"
#include "../op_kernel/tiling_key_batch_to_space.h"

namespace optiling {
static uint32_t GcdU32(uint32_t a, uint32_t b) {
    while (b != 0) {
        uint32_t t = a % b;
        a = b;
        b = t;
    }
    return a == 0 ? 1 : a;
}

static uint32_t AlignUp32(uint32_t value) {
    return ((value + 31U) / 32U) * 32U;
}

static uint32_t AdaptiveChunkWidth(
    uint32_t full_width,
    uint32_t max_copy_width,
    uint32_t parallel_units,
    uint32_t max_core_num,
    uint32_t min_chunk_width = 8U) {
    if (full_width == 0 || max_copy_width == 0 || parallel_units == 0 || parallel_units >= max_core_num) {
        return max_copy_width;
    }
    if (parallel_units * 2U > max_core_num) {
        return max_copy_width;
    }
    uint32_t target_chunks = (max_core_num + parallel_units - 1U) / parallel_units;
    if (target_chunks <= 1U) {
        return max_copy_width;
    }
    if (target_chunks > full_width) {
        target_chunks = full_width;
    }
    uint32_t candidate = (full_width + target_chunks - 1U) / target_chunks;
    if (candidate < min_chunk_width) {
        candidate = min_chunk_width;
    }
    if (candidate > max_copy_width) {
        candidate = max_copy_width;
    }
    return candidate;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t num_cores_aiv = platform.GetCoreNumAiv();

    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_x = tensor_x->GetDataType();
    int dtype_size_x = ge::GetSizeByDataType(dtype_x);
    uint32_t input_length = static_cast<uint32_t>(tensor_x->GetShapeSize());
    const gert::Shape &x_shape = tensor_x->GetStorageShape();
    uint32_t batch = static_cast<uint32_t>(x_shape.GetDim(0));
    uint32_t height = static_cast<uint32_t>(x_shape.GetDim(1));
    uint32_t width = static_cast<uint32_t>(x_shape.GetDim(2));
    uint32_t depth = static_cast<uint32_t>(x_shape.GetDim(3));

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const gert::TypedContinuousVector<int64_t> *attr_crops = attrs->GetListInt(0);
    const int64_t *attr_block_size = attrs->GetInt(1);
    uint32_t crop_top = static_cast<uint32_t>(attr_crops->GetData()[0]);
    uint32_t crop_bottom = static_cast<uint32_t>(attr_crops->GetData()[1]);
    uint32_t crop_left = static_cast<uint32_t>(attr_crops->GetData()[2]);
    uint32_t crop_right = static_cast<uint32_t>(attr_crops->GetData()[3]);
    uint32_t block_size = static_cast<uint32_t>(*attr_block_size);

    uint32_t out_batch = batch / (block_size * block_size);
    uint32_t out_height = height * block_size - crop_top - crop_bottom;
    uint32_t out_width = width * block_size - crop_left - crop_right;
    uint32_t output_length = out_batch * out_height * out_width * depth;
    uint32_t output_pixels = out_batch * out_height * out_width;
    uint32_t depth_bytes = depth * static_cast<uint32_t>(dtype_size_x);
    uint32_t contig_max_copy_elems = 32768U / static_cast<uint32_t>(dtype_size_x);
    contig_max_copy_elems = contig_max_copy_elems == 0 ? 1U : contig_max_copy_elems;
    uint32_t contig_elems_per_batch = out_height * out_width * depth;
    uint32_t contig_chunks_per_batch =
        (contig_elems_per_batch + contig_max_copy_elems - 1U) / contig_max_copy_elems;

    uint32_t DT_X = static_cast<uint32_t>(dtype_x);

    BatchToSpaceTilingData *tiling = context->GetTilingData<BatchToSpaceTilingData>();
    tiling->inputLength = input_length;
    tiling->outputLength = output_length;
    tiling->batch = batch;
    tiling->height = height;
    tiling->width = width;
    tiling->depth = depth;
    tiling->outBatch = out_batch;
    tiling->outHeight = out_height;
    tiling->outWidth = out_width;
    tiling->outDepth = depth;
    tiling->cropTop = crop_top;
    tiling->cropBottom = crop_bottom;
    tiling->cropLeft = crop_left;
    tiling->cropRight = crop_right;
    tiling->blockSize = block_size;

    uint32_t elements_per_cache_line = dtype_size_x > 0 ? static_cast<uint32_t>(64 / dtype_size_x) : 1;
    elements_per_cache_line = elements_per_cache_line == 0 ? 1 : elements_per_cache_line;
    uint32_t max_core_num = num_cores_aiv > 0 ? static_cast<uint32_t>(num_cores_aiv) : 1;
    uint32_t copy_threshold = 1;
    bool use_b2_local_interleave = block_size == 2 && depth_bytes != 0 && depth_bytes <= 64 &&
        depth_bytes % 32 == 0 && output_length >= 262144U && out_width <= 1024U &&
        (out_width * depth_bytes) > 8192U && (out_width * depth_bytes) <= 32768U;
    bool use_b2_gather_interleave = false && block_size == 2 && depth_bytes != 0 && depth_bytes <= 256 &&
        depth_bytes % 32 == 0 && output_length >= 131072;
    uint32_t row_bytes = out_width * depth_bytes;
    uint32_t b2_max_lane_width = (out_width + 1U) / 2U;
    uint32_t b2_row_copy_max_width = 1;
    uint32_t b2_row_copy_chunks_per_lane = 0;
    uint32_t b2_row_copy_unit_count = 0;
    uint32_t b2_depth_blocks = depth_bytes / 32U;
    uint32_t b2_depth_chunks = b2_depth_blocks >= 8 ? 4U : 2U;
    if (b2_depth_chunks > b2_depth_blocks) {
        b2_depth_chunks = b2_depth_blocks;
    }
    uint32_t b2_blocks_per_chunk = b2_depth_chunks == 0 ? 0 :
        (b2_depth_blocks + b2_depth_chunks - 1) / b2_depth_chunks;
    uint32_t b2_depth_split_max_width = 1;
    uint32_t b2_depth_split_chunks_per_lane = 0;
    uint32_t b2_depth_split_unit_count = 0;
    if (depth_bytes != 0) {
        b2_row_copy_max_width = 32768 / depth_bytes;
        if (b2_row_copy_max_width == 0) {
            b2_row_copy_max_width = 1;
        }
        if (b2_row_copy_max_width > 4095) {
            b2_row_copy_max_width = 4095;
        }
        b2_row_copy_chunks_per_lane = (b2_max_lane_width + b2_row_copy_max_width - 1) / b2_row_copy_max_width;
        b2_row_copy_unit_count = out_batch * out_height * 2U * b2_row_copy_chunks_per_lane;
    }
    if (b2_blocks_per_chunk != 0) {
        b2_depth_split_max_width = 32768 / (b2_blocks_per_chunk * 32U);
        if (b2_depth_split_max_width == 0) {
            b2_depth_split_max_width = 1;
        }
        if (b2_depth_split_max_width > 4095) {
            b2_depth_split_max_width = 4095;
        }
        b2_depth_split_chunks_per_lane =
            (b2_max_lane_width + b2_depth_split_max_width - 1) / b2_depth_split_max_width;
        b2_depth_split_unit_count = out_batch * out_height * 2U *
            b2_depth_split_chunks_per_lane * b2_depth_chunks;
    }
    bool b2_row_copy_underfills_cores = b2_row_copy_unit_count != 0 &&
        b2_row_copy_unit_count * 2U <= max_core_num;
    bool b2_depth_split_coarse_enough = b2_blocks_per_chunk >= 16U &&
        output_length >= 32768U && output_length <= 65536U;
    bool b2_depth_split_fills_cores = b2_depth_split_unit_count >= max_core_num;
    uint32_t b2_rowtile_target_width = depth_bytes == 0 ? 1U : (4096U / depth_bytes);
    if (b2_rowtile_target_width == 0) {
        b2_rowtile_target_width = 1U;
    }
    if (b2_rowtile_target_width > out_width) {
        b2_rowtile_target_width = out_width;
    }
    uint32_t b2_rowtile_rows_per_group = 0;
    if (depth_bytes != 0 && b2_rowtile_target_width != 0) {
        b2_rowtile_rows_per_group = 32768U / (b2_rowtile_target_width * depth_bytes);
        if (b2_rowtile_rows_per_group > 8U) {
            b2_rowtile_rows_per_group = 8U;
        }
        if (row_bytes > 16384U && b2_rowtile_rows_per_group > 4U) {
            b2_rowtile_rows_per_group = 4U;
        }
    }
    bool use_b2_rowtile_assemble_copy = block_size == 2 && !use_b2_local_interleave && !use_b2_gather_interleave &&
        depth_bytes != 0 && depth_bytes <= 256 && depth_bytes % 32 == 0 &&
        output_length >= 262144U && row_bytes > 4096U &&
        b2_rowtile_target_width < out_width && b2_rowtile_rows_per_group >= 2U;
    bool use_b2_rows_assemble_copy = !use_b2_rowtile_assemble_copy &&
        block_size == 2 && !use_b2_local_interleave && !use_b2_gather_interleave &&
        depth_bytes != 0 && depth_bytes <= 256 && depth_bytes % 32 == 0 && output_length >= 131072 &&
        row_bytes != 0 && row_bytes <= 4096;
    bool use_b2_assemble_copy = block_size == 2 && !use_b2_rowtile_assemble_copy && !use_b2_rows_assemble_copy &&
        !use_b2_local_interleave && !use_b2_gather_interleave && depth_bytes != 0 &&
        depth_bytes <= 32768 && depth_bytes % 32 == 0 && output_length >= 131072;
    bool use_b2_pad_assemble_copy = false && block_size == 2 && !use_b2_assemble_copy && depth_bytes != 0 &&
        depth_bytes <= 32768 && depth_bytes % 32 != 0 && output_length >= 65536;
    bool b2_depth_split_width_gate = (out_width >= 1 && out_width < 4) ||
        (out_width >= 4 && out_width <= 16 && b2_depth_split_fills_cores);
    bool use_b2_depth_split_row_copy = block_size == 2 && b2_depth_split_width_gate &&
        b2_row_copy_underfills_cores && b2_depth_split_coarse_enough &&
        depth_bytes > 256 && depth_bytes <= 32768 &&
        depth_bytes % 32 == 0 && output_length < 131072 && !use_b2_assemble_copy &&
        !use_b2_pad_assemble_copy;
    bool use_b2_row_copy = block_size == 2 && !use_b2_depth_split_row_copy &&
        depth_bytes != 0 && depth_bytes <= 32768 &&
        depth_bytes % 32 == 0 && !use_b2_assemble_copy && !use_b2_pad_assemble_copy;
    bool use_b2_pad_copy = block_size == 2 && !use_b2_assemble_copy && !use_b2_pad_assemble_copy &&
        !use_b2_depth_split_row_copy && !use_b2_row_copy && depth_bytes != 0 &&
        depth_bytes <= 32768 && output_length >= copy_threshold;
    bool use_b2_case5_gather32SmallUbInputPipelineNoFlush_chunks = use_b2_pad_copy && depth_bytes == 130U &&
        out_batch == 1U && out_height == 254U && out_width == 254U &&
        crop_left == 1U && crop_top == 1U;
    bool use_large_depth_copy = depth_bytes > 32768 && output_length >= copy_threshold;
    bool use_b1_contig_copy = block_size == 1 && crop_left == 0 && crop_right == 0 &&
        depth_bytes != 0 && depth_bytes <= 32768 && depth_bytes % 32 == 0 &&
        output_length >= copy_threshold && !use_large_depth_copy && contig_chunks_per_batch <= 2U;
    bool use_b1_contig_pad_copy = block_size == 1 && crop_left == 0 && crop_right == 0 &&
        depth_bytes != 0 && depth_bytes <= 32768 &&
        (depth_bytes % 32 != 0 || contig_chunks_per_batch > 2U) &&
        output_length >= copy_threshold;
    bool use_b1_row_copy = block_size == 1 && !use_b1_contig_copy && depth_bytes != 0 && depth_bytes <= 32768 &&
        depth_bytes % 32 == 0 && output_length >= copy_threshold && !use_large_depth_copy &&
        !use_b1_contig_pad_copy;
    bool use_b1_pad_copy = block_size == 1 && !use_b1_contig_copy && !use_b1_row_copy && !use_b1_contig_pad_copy &&
        depth_bytes != 0 &&
        depth_bytes <= 32768 && output_length >= copy_threshold && !use_large_depth_copy;
    bool use_bn_assemble_row_copy = block_size > 2 && depth_bytes != 0 && depth_bytes <= 256U &&
        depth_bytes % 32U == 0U && output_length >= 262144U && row_bytes != 0U &&
        !use_large_depth_copy;
    bool use_bn_row_copy = block_size > 2 && !use_bn_assemble_row_copy &&
        depth_bytes != 0 && depth_bytes <= 32768 &&
        depth_bytes % 32 == 0 && output_length >= copy_threshold && !use_large_depth_copy;
    bool use_bn_pad_copy = block_size > 2 && !use_bn_assemble_row_copy && !use_bn_row_copy && depth_bytes != 0 &&
        depth_bytes <= 32768 && output_length >= copy_threshold && !use_large_depth_copy;
    bool use_depth1_row_scalar = !use_b2_depth_split_row_copy && !use_b2_row_copy && !use_b2_pad_copy && !use_b1_row_copy &&
        !use_b1_pad_copy && !use_bn_row_copy && !use_bn_pad_copy && depth == 1 &&
        output_length >= copy_threshold;
    bool use_pixel_scalar = !use_b2_depth_split_row_copy && !use_b2_row_copy && !use_b2_pad_copy && !use_b1_row_copy &&
        !use_b1_pad_copy && !use_bn_row_copy && !use_bn_pad_copy && !use_depth1_row_scalar &&
        depth > 1 && output_pixels >= max_core_num;
    uint32_t used_core_num;
    uint32_t core_stride = 0;
    uint32_t core_pixel_stride = 0;
    uint32_t aux_value = 0;
    uint32_t copy_mode = use_b2_rowtile_assemble_copy ? 20U :
        (use_b2_rows_assemble_copy ? 16U :
        (use_b2_gather_interleave ? 15U :
        (use_b2_local_interleave ? 14U :
        (use_b2_assemble_copy ? 11U :
        (use_b2_pad_assemble_copy ? 12U :
        (use_b2_depth_split_row_copy ? 17U :
        (use_b2_row_copy ? 2U :
        (use_b2_pad_copy ? 5U :
        (use_b1_contig_copy ? 18U :
        (use_b1_contig_pad_copy ? 10U :
        (use_b1_row_copy ? 3U :
        (use_b1_pad_copy ? 6U :
        (use_bn_row_copy ? 7U :
        (use_bn_pad_copy ? 8U :
        (use_large_depth_copy ? 9U :
        (use_depth1_row_scalar ? 4U : (use_pixel_scalar ? 1U : 0U)))))))))))))))));
    uint32_t row_group_length = 0;

    if (use_bn_assemble_row_copy) {
        uint32_t row_count = out_batch * out_height;
        uint32_t max_copy_width = 32768U / depth_bytes;
        if (max_copy_width == 0U) {
            max_copy_width = 1U;
        }
        if (max_copy_width > 4095U) {
            max_copy_width = 4095U;
        }
        uint32_t chunks_per_row = (out_width + max_copy_width - 1U) / max_copy_width;
        uint32_t unit_count = row_count * chunks_per_row;
        used_core_num = unit_count < max_core_num ? unit_count : max_core_num;
        core_pixel_stride = (unit_count + used_core_num - 1U) / used_core_num;
        row_group_length = chunks_per_row;
        copy_mode = 77U;
    } else if (use_b2_case5_gather32SmallUbInputPipelineNoFlush_chunks) {
        uint32_t chunks_per_row = 8U;
        uint32_t unit_count = out_height * chunks_per_row;
        used_core_num = unit_count == 0 ? 1 : (unit_count < max_core_num ? unit_count : max_core_num);
        core_pixel_stride = (unit_count + used_core_num - 1U) / used_core_num;
        row_group_length = chunks_per_row;
        copy_mode = 78U;
    } else if (use_b2_rowtile_assemble_copy) {
        uint32_t groups_per_batch = (out_height + b2_rowtile_rows_per_group - 1U) / b2_rowtile_rows_per_group;
        uint32_t chunks_per_row = (out_width + b2_rowtile_target_width - 1U) / b2_rowtile_target_width;
        uint32_t unit_count = out_batch * groups_per_batch * chunks_per_row;
        used_core_num = unit_count == 0 ? 1 : (unit_count < max_core_num ? unit_count : max_core_num);
        core_pixel_stride = (unit_count + used_core_num - 1U) / used_core_num;
        row_group_length = b2_rowtile_rows_per_group;
        aux_value = b2_rowtile_target_width;
    } else if (use_b2_rows_assemble_copy) {
        uint32_t rows_per_group = 32768 / row_bytes;
        if (rows_per_group == 0) {
            rows_per_group = 1;
        }
        if (rows_per_group > 16) {
            rows_per_group = 16;
        }
        uint32_t groups_per_batch = (out_height + rows_per_group - 1) / rows_per_group;
        uint32_t unit_count = out_batch * groups_per_batch;
        used_core_num = unit_count == 0 ? 1 : (unit_count < max_core_num ? unit_count : max_core_num);
        core_pixel_stride = (unit_count + used_core_num - 1) / used_core_num;
        row_group_length = rows_per_group;
    } else if (use_b2_gather_interleave) {
        uint32_t row_count = out_batch * out_height;
        uint32_t max_copy_width = 32768 / depth_bytes;
        uint32_t max_offset_width = 32768 / (depth * 4U);
        if (max_copy_width > max_offset_width) {
            max_copy_width = max_offset_width;
        }
        if (max_copy_width == 0) {
            max_copy_width = 1;
        }
        if (max_copy_width > 1024) {
            max_copy_width = 1024;
        }
        uint32_t chunks_per_row = (out_width + max_copy_width - 1) / max_copy_width;
        uint32_t unit_count = row_count * chunks_per_row;
        used_core_num = unit_count == 0 ? 1 : (unit_count < max_core_num ? unit_count : max_core_num);
        core_pixel_stride = (unit_count + used_core_num - 1) / used_core_num;
        row_group_length = chunks_per_row;
    } else if (use_b2_local_interleave) {
        uint32_t row_count = out_batch * out_height;
        uint32_t max_copy_width = 32768 / (3U * depth_bytes);
        if (max_copy_width == 0) {
            max_copy_width = 1;
        }
        if (max_copy_width > 1024) {
            max_copy_width = 1024;
        }
        uint32_t chunks_per_row = (out_width + max_copy_width - 1) / max_copy_width;
        uint32_t unit_count = row_count * chunks_per_row;
        used_core_num = unit_count == 0 ? 1 : (unit_count < max_core_num ? unit_count : max_core_num);
        core_pixel_stride = (unit_count + used_core_num - 1) / used_core_num;
        row_group_length = chunks_per_row;
    } else if (use_b2_assemble_copy) {
        uint32_t row_count = out_batch * out_height;
        uint32_t max_copy_width = 32768 / depth_bytes;
        if (max_copy_width > 4095) {
            max_copy_width = 4095;
        }
        if (output_length >= 262144U && out_width >= 16U) {
            max_copy_width = AdaptiveChunkWidth(out_width, max_copy_width, row_count, max_core_num, 8U);
        }
        uint32_t chunks_per_row = (out_width + max_copy_width - 1) / max_copy_width;
        uint32_t unit_count = row_count * chunks_per_row;
        used_core_num = unit_count == 0 ? 1 : (unit_count < max_core_num ? unit_count : max_core_num);
        core_pixel_stride = (unit_count + used_core_num - 1) / used_core_num;
        row_group_length = chunks_per_row;
    } else if (use_b2_pad_assemble_copy) {
        uint32_t row_count = out_batch * out_height;
        uint32_t max_copy_width = 32768 / AlignUp32(depth_bytes);
        if (max_copy_width > 4095) {
            max_copy_width = 4095;
        }
        uint32_t chunks_per_row = (out_width + max_copy_width - 1) / max_copy_width;
        uint32_t unit_count = row_count * chunks_per_row;
        used_core_num = unit_count == 0 ? 1 : (unit_count < max_core_num ? unit_count : max_core_num);
        core_pixel_stride = (unit_count + used_core_num - 1) / used_core_num;
        row_group_length = chunks_per_row;
    } else if (use_b2_depth_split_row_copy) {
        uint32_t row_count = out_batch * out_height;
        uint32_t depth_blocks = depth_bytes / 32U;
        uint32_t depth_chunks = depth_blocks >= 8 ? 4U : 2U;
        if (depth_chunks > depth_blocks) {
            depth_chunks = depth_blocks;
        }
        uint32_t blocks_per_chunk = (depth_blocks + depth_chunks - 1) / depth_chunks;
        uint32_t max_copy_width = 32768 / (blocks_per_chunk * 32U);
        if (max_copy_width == 0) {
            max_copy_width = 1;
        }
        if (max_copy_width > 4095) {
            max_copy_width = 4095;
        }
        uint32_t max_lane_width = (out_width + 1) / 2;
        uint32_t chunks_per_lane = (max_lane_width + max_copy_width - 1) / max_copy_width;
        row_group_length = chunks_per_lane * depth_chunks;
        uint32_t unit_count = row_count * 2 * row_group_length;
        used_core_num = unit_count == 0 ? 1 : (unit_count < max_core_num ? unit_count : max_core_num);
        core_pixel_stride = (unit_count + used_core_num - 1) / used_core_num;
        aux_value = depth_chunks;
    } else if (use_b2_row_copy) {
        uint32_t row_count = out_batch * out_height;
        uint32_t max_copy_width = 32768 / depth_bytes;
        if (max_copy_width > 4095) {
            max_copy_width = 4095;
        }
        uint32_t max_lane_width = (out_width + 1) / 2;
        uint32_t chunks_per_lane = (max_lane_width + max_copy_width - 1) / max_copy_width;
        uint32_t unit_count = row_count * 2 * chunks_per_lane;
        used_core_num = unit_count == 0 ? 1 : (unit_count < max_core_num ? unit_count : max_core_num);
        core_pixel_stride = (unit_count + used_core_num - 1) / used_core_num;
        row_group_length = chunks_per_lane;
    } else if (use_b2_pad_copy) {
        uint32_t row_count = out_batch * out_height;
        uint32_t max_copy_width = 32768 / AlignUp32(depth_bytes);
        if (max_copy_width > 4095) {
            max_copy_width = 4095;
        }
        uint32_t max_lane_width = (out_width + 1) / 2;
        uint32_t chunks_per_lane = (max_lane_width + max_copy_width - 1) / max_copy_width;
        uint32_t unit_count = row_count * 2 * chunks_per_lane;
        used_core_num = unit_count == 0 ? 1 : (unit_count < max_core_num ? unit_count : max_core_num);
        core_pixel_stride = (unit_count + used_core_num - 1) / used_core_num;
        row_group_length = chunks_per_lane;
    } else if (use_b1_row_copy) {
        uint32_t row_count = out_batch * out_height;
        uint32_t max_copy_width = 32768 / depth_bytes;
        if (max_copy_width > 4095) {
            max_copy_width = 4095;
        }
        uint32_t chunks_per_row = (out_width + max_copy_width - 1) / max_copy_width;
        uint32_t unit_count = row_count * chunks_per_row;
        used_core_num = unit_count == 0 ? 1 : (unit_count < max_core_num ? unit_count : max_core_num);
        core_pixel_stride = (unit_count + used_core_num - 1) / used_core_num;
        row_group_length = chunks_per_row;
    } else if (use_b1_contig_copy || use_b1_contig_pad_copy) {
        uint32_t max_copy_elems = 32768 / static_cast<uint32_t>(dtype_size_x);
        max_copy_elems = max_copy_elems == 0 ? 1 : max_copy_elems;
        uint32_t elems_per_batch = out_height * out_width * depth;
        uint32_t chunks_per_batch = (elems_per_batch + max_copy_elems - 1) / max_copy_elems;
        uint32_t unit_count = out_batch * chunks_per_batch;
        used_core_num = unit_count == 0 ? 1 : (unit_count < max_core_num ? unit_count : max_core_num);
        core_pixel_stride = (unit_count + used_core_num - 1) / used_core_num;
        row_group_length = chunks_per_batch;
    } else if (use_b1_pad_copy) {
        uint32_t row_count = out_batch * out_height;
        uint32_t max_copy_width = 32768 / depth_bytes;
        uint32_t chunks_per_row = (out_width + max_copy_width - 1) / max_copy_width;
        uint32_t unit_count = row_count * chunks_per_row;
        used_core_num = unit_count == 0 ? 1 : (unit_count < max_core_num ? unit_count : max_core_num);
        core_pixel_stride = (unit_count + used_core_num - 1) / used_core_num;
        row_group_length = chunks_per_row;
    } else if (use_bn_row_copy) {
        uint32_t row_count = out_batch * out_height;
        uint32_t max_copy_width = 32768 / depth_bytes;
        if (max_copy_width > 4095) {
            max_copy_width = 4095;
        }
        uint32_t max_lane_width = (out_width + block_size - 1) / block_size;
        if (output_length >= 262144U && max_lane_width >= 16U) {
            max_copy_width = AdaptiveChunkWidth(
                max_lane_width,
                max_copy_width,
                row_count * block_size,
                max_core_num,
                8U
            );
        }
        uint32_t chunks_per_lane = (max_lane_width + max_copy_width - 1) / max_copy_width;
        uint32_t unit_count = row_count * block_size * chunks_per_lane;
        used_core_num = unit_count == 0 ? 1 : (unit_count < max_core_num ? unit_count : max_core_num);
        core_pixel_stride = (unit_count + used_core_num - 1) / used_core_num;
        row_group_length = chunks_per_lane;
    } else if (use_bn_pad_copy) {
        uint32_t row_count = out_batch * out_height;
        uint32_t max_copy_width = 32768 / AlignUp32(depth_bytes);
        if (max_copy_width > 4095) {
            max_copy_width = 4095;
        }
        uint32_t max_lane_width = (out_width + block_size - 1) / block_size;
        if (output_length >= 262144U && max_lane_width >= 16U) {
            max_copy_width = AdaptiveChunkWidth(
                max_lane_width,
                max_copy_width,
                row_count * block_size,
                max_core_num,
                8U
            );
        }
        uint32_t chunks_per_lane = (max_lane_width + max_copy_width - 1) / max_copy_width;
        uint32_t unit_count = row_count * block_size * chunks_per_lane;
        used_core_num = unit_count == 0 ? 1 : (unit_count < max_core_num ? unit_count : max_core_num);
        core_pixel_stride = (unit_count + used_core_num - 1) / used_core_num;
        row_group_length = chunks_per_lane;
    } else if (use_large_depth_copy) {
        uint32_t max_copy_elems = 32768 / static_cast<uint32_t>(dtype_size_x);
        max_copy_elems = max_copy_elems == 0 ? 1 : max_copy_elems;
        uint32_t depth_chunks = (depth + max_copy_elems - 1) / max_copy_elems;
        uint32_t unit_count = output_pixels * depth_chunks;
        used_core_num = unit_count == 0 ? 1 : (unit_count < max_core_num ? unit_count : max_core_num);
        core_pixel_stride = (unit_count + used_core_num - 1) / used_core_num;
        row_group_length = depth_chunks;
    } else if (use_depth1_row_scalar) {
        uint32_t cache_line_count = (output_length + elements_per_cache_line - 1) / elements_per_cache_line;
        used_core_num = cache_line_count == 0 ? 1 :
            (cache_line_count < max_core_num ? cache_line_count : max_core_num);
        core_stride = (output_length + used_core_num - 1) / used_core_num;
        core_stride = ((core_stride + elements_per_cache_line - 1) / elements_per_cache_line) *
            elements_per_cache_line;
    } else if (use_pixel_scalar) {
        used_core_num = output_pixels < max_core_num ? output_pixels : max_core_num;
        core_pixel_stride = (output_pixels + used_core_num - 1) / used_core_num;
        uint32_t pixel_align = elements_per_cache_line / GcdU32(elements_per_cache_line, depth);
        core_pixel_stride = ((core_pixel_stride + pixel_align - 1) / pixel_align) * pixel_align;
    } else {
        uint32_t cache_line_count = (output_length + elements_per_cache_line - 1) / elements_per_cache_line;
        used_core_num = cache_line_count == 0 ? 1 :
            (cache_line_count < max_core_num ? cache_line_count : max_core_num);
        core_stride = (output_length + used_core_num - 1) / used_core_num;
        core_stride = ((core_stride + elements_per_cache_line - 1) / elements_per_cache_line) *
            elements_per_cache_line;
    }

    if (output_length < 32768U && used_core_num > 16U) {
        uint32_t old_core_num = used_core_num;
        used_core_num = 16U;
        if (core_pixel_stride != 0U) {
            uint32_t approx_units = core_pixel_stride * old_core_num;
            core_pixel_stride = (approx_units + used_core_num - 1U) / used_core_num;
        }
        if (core_stride != 0U) {
            uint32_t approx_elems = core_stride * old_core_num;
            core_stride = (approx_elems + used_core_num - 1U) / used_core_num;
            core_stride = ((core_stride + elements_per_cache_line - 1U) / elements_per_cache_line) *
                elements_per_cache_line;
        }
    }

    ASCENDC_TPL_SEL_PARAM(context, DT_X, copy_mode);

    tiling->usedCoreNum = used_core_num;
    tiling->coreStride = core_stride;
    tiling->pixelLength = depth;
    tiling->corePixelStride = core_pixel_stride;
    tiling->rowGroupLength = row_group_length;
    tiling->canUseBlockCopy = aux_value;
    tiling->copyMode = copy_mode;

    context->SetBlockDim(used_core_num);
    size_t *current_workspace = context->GetWorkspaceSizes(1);
    current_workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *x_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const gert::TypedContinuousVector<int64_t> *attr_crops = attrs->GetListInt(0);
    const int64_t *attr_block_size = attrs->GetInt(1);
    int64_t batch = x_shape->GetDim(0);
    int64_t height = x_shape->GetDim(1);
    int64_t width = x_shape->GetDim(2);
    int64_t depth = x_shape->GetDim(3);
    int64_t block_size = *attr_block_size;
    int64_t out_batch = batch / (block_size * block_size);
    int64_t out_height = height * block_size - attr_crops->GetData()[0] - attr_crops->GetData()[1];
    int64_t out_width = width * block_size - attr_crops->GetData()[2] - attr_crops->GetData()[3];

    y_shape->SetDimNum(4);
    y_shape->SetDim(0, out_batch);
    y_shape->SetDim(1, out_height);
    y_shape->SetDim(2, out_width);
    y_shape->SetDim(3, depth);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
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
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(BatchToSpace);
}  // namespace ops
