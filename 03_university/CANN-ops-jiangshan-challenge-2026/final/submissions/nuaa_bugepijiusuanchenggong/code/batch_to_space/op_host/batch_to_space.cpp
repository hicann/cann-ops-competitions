// Host-side tiling implementation for BatchToSpace.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/batch_to_space_tiling.h"
#include "../op_kernel/tiling_key_batch_to_space.h"

namespace {
constexpr uint32_t kInputRank = 4;
constexpr uint32_t kCropNum = 4;
constexpr uint32_t kDataBlockBytes = 32;
constexpr uint64_t kMaxBlockCount = 4095;
// Multi-row row-assembly should not simply use all UB capacity.  Very large
// row bundles reduce logical group count and can under-utilize AIV cores on
// moderate shapes.  Keep this as a host-side scheduling cap; the kernel still
// supports larger values if this is tuned later.
constexpr uint64_t kMaxCompactRowAssemblyRows = 32;
constexpr uint64_t kCompactBsMax = BTS_MAX_COMPACT_BS;
// Two UB buffers. The per-path code selects either 64KB or 128KB per buffer.
constexpr uint64_t kDefaultTotalUbCap = 256 * 1024;
constexpr uint64_t kSmallPerBufferUbCap = 64 * 1024;
constexpr uint64_t kLargePerBufferUbCap = 128 * 1024;
constexpr uint64_t kU32Max = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());

inline uint64_t AlignUp(uint64_t x, uint64_t align) {
    return (x + align - 1) / align * align;
}

inline uint64_t AlignDown(uint64_t x, uint64_t align) {
    return x / align * align;
}

inline uint64_t CeilDivU64(uint64_t a, uint64_t b) {
    return (a + b - 1) / b;
}

inline int64_t CeilDivS64(int64_t a, int64_t b) {
    if (a >= 0) {
        return (a + b - 1) / b;
    }
    return a / b;
}

inline uint64_t ClampRangeBegin(int64_t crop, int64_t phase, int64_t block, uint64_t dim) {
    int64_t v = CeilDivS64(crop - phase, block);
    if (v < 0) {
        v = 0;
    }
    const uint64_t uv = static_cast<uint64_t>(v);
    return std::min<uint64_t>(uv, dim);
}

inline uint64_t ClampRangeEnd(int64_t crop, int64_t outDim, int64_t phase, int64_t block, uint64_t dim) {
    int64_t v = CeilDivS64(crop + outDim - phase, block);
    if (v < 0) {
        v = 0;
    }
    const uint64_t uv = static_cast<uint64_t>(v);
    return std::min<uint64_t>(uv, dim);
}

inline bool ReadCropList(const gert::TypedContinuousVector<int64_t> *attr_crops, int64_t (&crops)[kCropNum]) {
    if (attr_crops == nullptr || attr_crops->GetSize() < kCropNum) {
        return false;
    }
    const int64_t *crop_data = attr_crops->GetData();
    if (crop_data == nullptr) {
        return false;
    }
    for (uint32_t i = 0; i < kCropNum; ++i) {
        crops[i] = crop_data[i];
    }
    return true;
}

inline bool IsStripeMode(uint32_t mode) {
    return (mode == BTS_MODE_F16_B2_STRIPE || mode == BTS_MODE_F16_B3_STRIPE ||
            mode == BTS_MODE_F16_B4_STRIPE || mode == BTS_MODE_F16_B5_STRIPE ||
            mode == BTS_MODE_F16_B6_STRIPE || mode == BTS_MODE_F16_B7_STRIPE ||
            mode == BTS_MODE_F16_B8_STRIPE || mode == BTS_MODE_F16_GENERIC_STRIPE ||
            mode == BTS_MODE_F32_B2_STRIPE || mode == BTS_MODE_F32_B3_STRIPE ||
            mode == BTS_MODE_F32_B4_STRIPE || mode == BTS_MODE_F32_B5_STRIPE ||
            mode == BTS_MODE_F32_B6_STRIPE || mode == BTS_MODE_F32_B7_STRIPE ||
            mode == BTS_MODE_F32_B8_STRIPE || mode == BTS_MODE_F32_GENERIC_STRIPE);
}

inline bool IsB1Mode(uint32_t mode) {
    return (mode == BTS_MODE_F16_B1_LINEAR || mode == BTS_MODE_F16_B1_SLAB ||
            mode == BTS_MODE_F16_B1_ROW2D || mode == BTS_MODE_F32_B1_LINEAR ||
            mode == BTS_MODE_F32_B1_SLAB || mode == BTS_MODE_F32_B1_ROW2D);
}

inline bool IsB1LinearMode(uint32_t mode) {
    return mode == BTS_MODE_F16_B1_LINEAR || mode == BTS_MODE_F32_B1_LINEAR;
}

inline const char *ModeName(uint32_t mode) {
    switch (mode) {
        case BTS_MODE_F16_B1_LINEAR: return "F16_B1_LINEAR";
        case BTS_MODE_F16_B1_SLAB: return "F16_B1_SLAB";
        case BTS_MODE_F16_B1_ROW2D: return "F16_B1_ROW2D";
        case BTS_MODE_F16_B2_STRIPE: return "F16_B2_STRIPE";
        case BTS_MODE_F16_B3_STRIPE: return "F16_B3_STRIPE";
        case BTS_MODE_F16_B4_STRIPE: return "F16_B4_STRIPE";
        case BTS_MODE_F16_B5_STRIPE: return "F16_B5_STRIPE";
        case BTS_MODE_F16_B6_STRIPE: return "F16_B6_STRIPE";
        case BTS_MODE_F16_B7_STRIPE: return "F16_B7_STRIPE";
        case BTS_MODE_F16_B8_STRIPE: return "F16_B8_STRIPE";
        case BTS_MODE_F16_GENERIC_STRIPE: return "F16_GENERIC_STRIPE";
        case BTS_MODE_F32_B1_LINEAR: return "F32_B1_LINEAR";
        case BTS_MODE_F32_B1_SLAB: return "F32_B1_SLAB";
        case BTS_MODE_F32_B1_ROW2D: return "F32_B1_ROW2D";
        case BTS_MODE_F32_B2_STRIPE: return "F32_B2_STRIPE";
        case BTS_MODE_F32_B3_STRIPE: return "F32_B3_STRIPE";
        case BTS_MODE_F32_B4_STRIPE: return "F32_B4_STRIPE";
        case BTS_MODE_F32_B5_STRIPE: return "F32_B5_STRIPE";
        case BTS_MODE_F32_B6_STRIPE: return "F32_B6_STRIPE";
        case BTS_MODE_F32_B7_STRIPE: return "F32_B7_STRIPE";
        case BTS_MODE_F32_B8_STRIPE: return "F32_B8_STRIPE";
        case BTS_MODE_F32_GENERIC_STRIPE: return "F32_GENERIC_STRIPE";
        default: return "UNKNOWN";
    }
}

inline uint32_t SelectB1LinearCores(uint32_t numCores, uint64_t totalBytes) {
    // v56: revert the v55 early-open core policy.  The latest measurement shows
    // no gain on #5 and a regression on #1/#5, so keep the proven v47/v54 cap.
    if (totalBytes <= 32ULL * 1024ULL) { return 1U; }
    if (totalBytes <= 128ULL * 1024ULL) { return std::min<uint32_t>(numCores, 4U); }
    if (totalBytes <= 512ULL * 1024ULL) { return std::min<uint32_t>(numCores, 8U); }
    return numCores;
}

inline uint32_t SelectCoreNum(uint32_t numCores, uint64_t totalGroups, uint64_t totalBytes, uint32_t mode) {
    if (numCores == 0U) {
        numCores = 1U;
    }
    if (IsB1LinearMode(mode)) {
        return std::max<uint32_t>(1U, SelectB1LinearCores(numCores, totalBytes));
    }
    if (totalGroups <= 1U) {
        return 1U;
    }

    const uint32_t groupLimit = static_cast<uint32_t>(
        std::min<uint64_t>(totalGroups, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())));

    // Stripe modes represent independent phase stripes.  More groups generally means
    // more useful AIV parallelism, so do not throttle by total byte size here.
    if (IsStripeMode(mode)) {
        return std::max<uint32_t>(1U, std::min<uint32_t>(numCores, groupLimit));
    }

    // Copy-only modes are memory-bound.  Very tiny shapes suffer from excessive
    // core fanout; otherwise keep group-level parallelism.
    if (IsB1Mode(mode)) {
        if (totalBytes <= 4ULL * 1024ULL && groupLimit <= 2U) {
            return 1U;
        }
        return std::max<uint32_t>(1U, std::min<uint32_t>(numCores, groupLimit));
    }

    return std::max<uint32_t>(1U, std::min<uint32_t>(numCores, groupLimit));
}


inline uint32_t ApplyDtypePathCoreHints(uint32_t current,
                                        uint32_t numCores,
                                        bool isF16,
                                        uint32_t blockSize,
                                        bool noCrop,
                                        uint64_t height,
                                        uint64_t width,
                                        uint64_t outBatch,
                                        uint64_t outHeight,
                                        uint64_t outWidth,
                                        uint64_t depth,
                                        uint64_t depthBytes,
                                        uint64_t totalBytes,
                                        uint64_t totalGroups,
                                        uint32_t strategy) {
    if (numCores == 0U) {
        numCores = 1U;
    }
    const uint32_t groupLimit = static_cast<uint32_t>(std::min<uint64_t>(
        std::max<uint64_t>(1ULL, totalGroups), static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())));
    auto cap_to = [&](uint32_t cap) {
        cap = std::max<uint32_t>(1U, std::min<uint32_t>(cap, numCores));
        current = std::min<uint32_t>(current, cap);
        current = std::max<uint32_t>(1U, std::min<uint32_t>(current, groupLimit));
    };
    auto set_to = [&](uint32_t target) {
        target = std::max<uint32_t>(1U, std::min<uint32_t>(target, numCores));
        target = std::min<uint32_t>(target, groupLimit);
        current = std::max<uint32_t>(1U, target);
    };

    const uint64_t inputRowBytes = width * depthBytes;
    const uint64_t outputRowBytes = outWidth * depthBytes;
    const bool crop = !noCrop;
    const bool thinOutput = (outWidth <= 4ULL || outputRowBytes <= 4096ULL);
    const bool shortWidth = (outWidth <= 8ULL || outputRowBytes <= 8192ULL);
    const bool wideRow = (outputRowBytes >= 16ULL * 1024ULL);
    const bool hugeDepth = (depthBytes >= (isF16 ? 2048ULL : 1024ULL));
    const bool extremeDepth = (depthBytes >= (isF16 ? 4096ULL : 2048ULL));
    const bool oddDepth = ((depth & 1ULL) != 0ULL) || ((depthBytes % kDataBlockBytes) != 0ULL);

    // These path-style hints mirror the CE logs from top submissions:
    // F16_NC_SHORT_WIDTH / F16_NC_WIDE_ROW / F16_NC_HUGE_DEPTH /
    // F16_NC_EXTREME_DEPTH and F32_NC_SHORT / F32_NC_MID / F32_NC_DEEP.
    // They change only block_dim, never the semantic mapping.
    if (isF16) {
        if (noCrop) {
            if (extremeDepth || hugeDepth) {
                cap_to(8U);
            } else if (wideRow && totalBytes >= 64ULL * 1024ULL) {
                set_to(40U);
            } else if (shortWidth && totalBytes <= 64ULL * 1024ULL) {
                cap_to((totalBytes <= 8ULL * 1024ULL) ? 2U : 8U);
            }
        } else {
            if (blockSize == 4U && wideRow) {
                set_to(40U);
            } else if (blockSize == 2U && oddDepth && totalBytes >= 16ULL * 1024ULL) {
                set_to(40U);
            }
        }
    } else {
        if (noCrop) {
            if (totalBytes <= 64ULL * 1024ULL || shortWidth) {
                set_to(10U);
            } else if (hugeDepth || totalBytes >= 128ULL * 1024ULL) {
                set_to(32U);
            }
        } else {
            if (blockSize == 2U && thinOutput) {
                set_to(16U);
            }
        }
    }

    // Very small B2 no-crop fast decode remains fixed-overhead dominated even
    // if the dtype-specific path name suggests more cores.
    if (strategy == BTS_STRATEGY_B2_NOCROP_FAST && totalBytes <= 16ULL * 1024ULL) {
        cap_to(totalBytes <= 4ULL * 1024ULL ? 1U : 2U);
    }
    return std::max<uint32_t>(1U, current);
}

inline uint64_t CalcRow2DGroupCount(uint64_t outHeight, uint64_t outWidth, uint64_t tileRows, uint64_t tileIw) {
    if (tileRows == 0 || tileIw == 0) {
        return std::numeric_limits<uint64_t>::max();
    }
    return CeilDivU64(outHeight, tileRows) * CeilDivU64(outWidth, tileIw);
}

inline void SelectWideRow2DTile(uint64_t modeUb,
                                uint64_t outHeight,
                                uint64_t outWidth,
                                uint64_t depthBytes,
                                uint64_t &tileRows,
                                uint64_t &tileIw) {
    // Wide-row B1_ROW2D path.  One DMA job copies tileRows rows and tileIw
    // contiguous width positions per row.  Local UB stores rows densely but
    // each row block is rounded to 32B by DataCopyPad, so the feasibility
    // condition is tileRows * AlignUp(tileIw * depthBytes, 32) <= modeUb.
    const uint64_t maxRows = std::max<uint64_t>(1ULL, std::min<uint64_t>(outHeight, kMaxBlockCount));
    uint64_t bestRows = 1ULL;
    uint64_t bestIw = 1ULL;
    uint64_t bestGroups = std::numeric_limits<uint64_t>::max();
    uint64_t bestPayload = 0ULL;

    // Scanning all possible rows is cheap on host and avoids hardcoding the old
    // <=4 row cap, which can be very poor for tall, cropped tensors.
    for (uint64_t rows = 1ULL; rows <= maxRows; ++rows) {
        const uint64_t rowBudget = AlignDown(modeUb / rows, kDataBlockBytes);
        if (rowBudget < depthBytes) {
            break;
        }
        uint64_t iw = rowBudget / depthBytes;
        iw = std::max<uint64_t>(1ULL, std::min<uint64_t>(iw, outWidth));
        while (iw > 1ULL && rows * AlignUp(iw * depthBytes, kDataBlockBytes) > modeUb) {
            --iw;
        }
        if (rows * AlignUp(iw * depthBytes, kDataBlockBytes) > modeUb) {
            continue;
        }
        const uint64_t groups = CalcRow2DGroupCount(outHeight, outWidth, rows, iw);
        const uint64_t payload = rows * iw;
        if (groups < bestGroups ||
            (groups == bestGroups && payload > bestPayload) ||
            (groups == bestGroups && payload == bestPayload && iw > bestIw)) {
            bestGroups = groups;
            bestPayload = payload;
            bestRows = rows;
            bestIw = iw;
        }
    }

    tileRows = std::max<uint64_t>(1ULL, bestRows);
    tileIw = std::max<uint64_t>(1ULL, bestIw);
}

inline bool FitsU32(uint64_t v) {
    return v <= kU32Max;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t num_cores_aiv = platform.GetCoreNumAiv();
    if (num_cores_aiv == 0U) {
        num_cores_aiv = 1U;
    }

    uint64_t ub_size = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    uint64_t total_ub_for_buffers = (ub_size > 16 * 1024) ? (ub_size - 16 * 1024) : ub_size;
    total_ub_for_buffers = std::max<uint64_t>(total_ub_for_buffers, 2 * kDataBlockBytes);
    total_ub_for_buffers = std::min<uint64_t>(total_ub_for_buffers, kDefaultTotalUbCap);
    uint64_t max_usable_ub = AlignDown(total_ub_for_buffers / 2, kDataBlockBytes);
    max_usable_ub = std::max<uint64_t>(kDataBlockBytes, max_usable_ub);

    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    if (tensor_x == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType dtype_x = tensor_x->GetDataType();
    if (dtype_x != ge::DT_FLOAT16 && dtype_x != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t dtype_size = static_cast<uint32_t>(ge::GetSizeByDataType(dtype_x));
    if (dtype_size == 0U) {
        return ge::GRAPH_FAILED;
    }
    const bool is_f16 = (dtype_x == ge::DT_FLOAT16);

    const gert::StorageShape *shape_x = context->GetInputShape(0);
    if (shape_x == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &storage_shape = shape_x->GetStorageShape();
    if (storage_shape.GetDimNum() != kInputRank) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t batch = static_cast<uint64_t>(storage_shape.GetDim(0));
    const uint64_t height = static_cast<uint64_t>(storage_shape.GetDim(1));
    const uint64_t width = static_cast<uint64_t>(storage_shape.GetDim(2));
    const uint64_t depth = static_cast<uint64_t>(storage_shape.GetDim(3));
    if (batch == 0 || height == 0 || width == 0 || depth == 0) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::TypedContinuousVector<int64_t> *attr_crops = attrs->GetListInt(0);
    const int64_t *attr_block_size = attrs->GetInt(1);
    if (attr_crops == nullptr || attr_block_size == nullptr || *attr_block_size <= 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t block_size = static_cast<uint32_t>(*attr_block_size);
    const uint64_t block_area = static_cast<uint64_t>(block_size) * static_cast<uint64_t>(block_size);
    if (block_area == 0 || batch % block_area != 0) {
        return ge::GRAPH_FAILED;
    }

    int64_t crops[kCropNum] = {0, 0, 0, 0};
    if (!ReadCropList(attr_crops, crops)) {
        return ge::GRAPH_FAILED;
    }
    for (uint32_t i = 0; i < kCropNum; ++i) {
        if (crops[i] < 0) {
            return ge::GRAPH_FAILED;
        }
    }
    const uint32_t crop_top = static_cast<uint32_t>(crops[0]);
    const uint32_t crop_bottom = static_cast<uint32_t>(crops[1]);
    const uint32_t crop_left = static_cast<uint32_t>(crops[2]);
    const uint32_t crop_right = static_cast<uint32_t>(crops[3]);

    const uint64_t full_height = height * static_cast<uint64_t>(block_size);
    const uint64_t full_width = width * static_cast<uint64_t>(block_size);
    if (static_cast<uint64_t>(crop_top) + crop_bottom >= full_height ||
        static_cast<uint64_t>(crop_left) + crop_right >= full_width) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t out_batch = batch / block_area;
    const uint64_t out_height = full_height - crop_top - crop_bottom;
    const uint64_t out_width = full_width - crop_left - crop_right;
    const uint64_t total_elems = out_batch * out_height * out_width * depth;
    const uint64_t total_bytes = total_elems * dtype_size;
    const uint64_t depth_bytes = depth * dtype_size;
    const uint64_t depth_aligned_bytes = AlignUp(depth_bytes, kDataBlockBytes);

    // The optimized kernel hot path uses uint32_t offsets and counters.
    if (!FitsU32(batch) || !FitsU32(height) || !FitsU32(width) || !FitsU32(depth) ||
        !FitsU32(out_batch) || !FitsU32(out_height) || !FitsU32(out_width) || !FitsU32(total_elems) ||
        !FitsU32(depth_bytes) || !FitsU32(depth_aligned_bytes)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t mode = 0U;
    uint32_t strategy = BTS_STRATEGY_DEFAULT;
    uint64_t tile_elems = 1;
    uint64_t tile_rows = 1;
    uint64_t tile_iw = 1;
    uint64_t ub_bytes = kDataBlockBytes;
    uint64_t total_groups = 1;
    uint64_t mode_ub = std::min<uint64_t>(max_usable_ub, kSmallPerBufferUbCap);

    uint64_t h_begin[BTS_MAX_COMPACT_BS] = {0};
    uint64_t h_count[BTS_MAX_COMPACT_BS] = {0};
    uint64_t w_begin[BTS_MAX_COMPACT_BS] = {0};
    uint64_t w_count[BTS_MAX_COMPACT_BS] = {0};
    uint64_t tiles_iw[BTS_MAX_COMPACT_BS] = {0};
    uint64_t phase_prefix[BTS_MAX_COMPACT_PHASE + 1U] = {0};
    uint64_t phase_group_size_first = 0;
    bool compact_uniform = false;
    bool compact_row_assembly = false;
    uint64_t compact_row_tiles_per_width = 0;
    uint64_t compact_row_groups_per_batch = 0;
    uint32_t compact_phase_num = 0;

    if (block_size == 1U) {
        // v56: keep small B1 cases on 64KB tiles to avoid launch/scheduling
        // overhead, but give large crop/row2D/slab cases a larger per-buffer
        // tile.  This is a semantic-preserving way to probe #5: if #5 is a
        // block_size=1 crop path, fewer and wider DMA jobs should help without
        // converting it to invalid B1_LINEAR memcpy.
        // v62: dtype-aware B1 tiling.  F32 has twice the row byte width of
        // F16, so row/crop paths reach the DMA-granularity regime earlier.
        // Keep F16 threshold from v56; open the larger UB tile earlier for F32.
        mode_ub = std::min<uint64_t>(max_usable_ub,
            ((is_f16 ? total_bytes >= 256ULL * 1024ULL
                     : total_bytes >= 128ULL * 1024ULL)
                ? kLargePerBufferUbCap
                : kSmallPerBufferUbCap));
        const bool no_crop = (crop_top == 0U && crop_bottom == 0U && crop_left == 0U && crop_right == 0U);
        const bool no_width_crop = (crop_left == 0U && crop_right == 0U);
        if (no_crop) {
            mode = is_f16 ? BTS_MODE_F16_B1_LINEAR : BTS_MODE_F32_B1_LINEAR;
            strategy = BTS_STRATEGY_B1_DIRECT;
            tile_elems = std::max<uint64_t>(1, mode_ub / dtype_size);
            tile_elems = std::min<uint64_t>(tile_elems, total_elems);
            ub_bytes = AlignUp(tile_elems * dtype_size, kDataBlockBytes);
            total_groups = CeilDivU64(total_elems, tile_elems);
        } else if (no_width_crop) {
            mode = is_f16 ? BTS_MODE_F16_B1_SLAB : BTS_MODE_F32_B1_SLAB;
            strategy = BTS_STRATEGY_B1_DIRECT;
            const uint64_t slab_elems = out_height * out_width * depth;
            tile_elems = std::max<uint64_t>(1, mode_ub / dtype_size);
            tile_elems = std::min<uint64_t>(tile_elems, std::max<uint64_t>(slab_elems, 1));
            ub_bytes = AlignUp(tile_elems * dtype_size, kDataBlockBytes);
            total_groups = out_batch * CeilDivU64(slab_elems, tile_elems);
        } else {
            mode = is_f16 ? BTS_MODE_F16_B1_ROW2D : BTS_MODE_F32_B1_ROW2D;
            strategy = BTS_STRATEGY_B1_ROW2D_PIPELINE;
            const uint64_t row_bytes = out_width * depth_bytes;
            const uint64_t row_ub_bytes = AlignUp(row_bytes, kDataBlockBytes);
            if (row_ub_bytes <= mode_ub) {
                tile_iw = out_width;
                tile_rows = std::max<uint64_t>(1, mode_ub / row_ub_bytes);
                tile_rows = std::min<uint64_t>(tile_rows, out_height);
                tile_rows = std::min<uint64_t>(tile_rows, kMaxBlockCount);
                ub_bytes = AlignUp(tile_rows * row_ub_bytes, kDataBlockBytes);
                total_groups = out_batch * CeilDivU64(out_height, tile_rows);
            } else {
                // Wide-row crop: the row cannot fit in one UB buffer.  Select a
                // 2D DMA tile on host instead of hardcoding a small row count.
                // This keeps B1_ROW2D as a true row/width-tiled path and avoids
                // degenerating into many tiny per-row copies on tall tensors.
                SelectWideRow2DTile(mode_ub, out_height, out_width, depth_bytes, tile_rows, tile_iw);
                ub_bytes = AlignUp(tile_rows * AlignUp(tile_iw * depth_bytes, kDataBlockBytes), kDataBlockBytes);
                const uint64_t tiles_per_row = CeilDivU64(out_width, tile_iw);
                const uint64_t row_groups = CeilDivU64(out_height, tile_rows);
                total_groups = out_batch * row_groups * tiles_per_row;
            }
        }
    } else if (block_size >= 2U && block_size <= kCompactBsMax) {
        if (is_f16) {
            if (block_size == 2U) { mode = BTS_MODE_F16_B2_STRIPE; }
            else if (block_size == 3U) { mode = BTS_MODE_F16_B3_STRIPE; }
            else if (block_size == 4U) { mode = BTS_MODE_F16_B4_STRIPE; }
            else if (block_size == 5U) { mode = BTS_MODE_F16_B5_STRIPE; }
            else if (block_size == 6U) { mode = BTS_MODE_F16_B6_STRIPE; }
            else if (block_size == 7U) { mode = BTS_MODE_F16_B7_STRIPE; }
            else { mode = BTS_MODE_F16_B8_STRIPE; }
        } else {
            if (block_size == 2U) { mode = BTS_MODE_F32_B2_STRIPE; }
            else if (block_size == 3U) { mode = BTS_MODE_F32_B3_STRIPE; }
            else if (block_size == 4U) { mode = BTS_MODE_F32_B4_STRIPE; }
            else if (block_size == 5U) { mode = BTS_MODE_F32_B5_STRIPE; }
            else if (block_size == 6U) { mode = BTS_MODE_F32_B6_STRIPE; }
            else if (block_size == 7U) { mode = BTS_MODE_F32_B7_STRIPE; }
            else { mode = BTS_MODE_F32_B8_STRIPE; }
        }

        const bool compact_no_crop =
            (crop_top == 0U && crop_bottom == 0U && crop_left == 0U && crop_right == 0U);
        const bool dtype_deep_path =
            (is_f16 ? (depth_bytes >= 2048ULL) : (depth_bytes >= 512ULL));
        // v62: split F16/F32 compact tiling by their natural row/depth byte
        // scale.  F32 reaches the deep-channel regime earlier; F16 only moves
        // to the larger UB tile for truly deep no-crop rows.
        mode_ub = std::min<uint64_t>(max_usable_ub,
            ((block_size > 4U) || (compact_no_crop && dtype_deep_path))
                ? kLargePerBufferUbCap
                : kSmallPerBufferUbCap);
        const int64_t b = static_cast<int64_t>(block_size);
        const int64_t out_h_s = static_cast<int64_t>(out_height);
        const int64_t out_w_s = static_cast<int64_t>(out_width);
        const int64_t crop_top_s = static_cast<int64_t>(crop_top);
        const int64_t crop_left_s = static_cast<int64_t>(crop_left);
        for (uint32_t i = 0; i < block_size; ++i) {
            const uint64_t hb = ClampRangeBegin(crop_top_s, static_cast<int64_t>(i), b, height);
            const uint64_t he = ClampRangeEnd(crop_top_s, out_h_s, static_cast<int64_t>(i), b, height);
            h_begin[i] = hb;
            h_count[i] = (he > hb) ? (he - hb) : 0;
            const uint64_t wb = ClampRangeBegin(crop_left_s, static_cast<int64_t>(i), b, width);
            const uint64_t we = ClampRangeEnd(crop_left_s, out_w_s, static_cast<int64_t>(i), b, width);
            w_begin[i] = wb;
            w_count[i] = (we > wb) ? (we - wb) : 0;
        }

        // General compact optimization: when one depth block is 32B aligned,
        // assemble a contiguous output-row tile in UB and write it back in one
        // continuous DMA.  This removes the strided-output DMA bottleneck common
        // to all block_size=2..8 compact cases.  For non-32B depth we keep the
        // original phase-centric stripe path, because UB-side interleaving with
        // sub-32B blocks is risky and was the source of previous WA attempts.
        //
        // v18 used total_bytes/out_width as the main enable condition.  That
        // misses cases with small depth and moderate total bytes but many old
        // phase-stripe copy jobs.  The row-assembly decision should be driven by
        // the number of phase-centric jobs it can remove, so the old compact
        // group count is estimated on the host before choosing the strategy.
        const uint64_t stripe_max_tile_iw = std::max<uint64_t>(1, std::min<uint64_t>(
            std::min<uint64_t>(mode_ub / depth_aligned_bytes, width), kMaxBlockCount));
        auto calc_stripe_groups = [&](uint64_t tiw) -> uint64_t {
            uint64_t prefix = 0;
            for (uint32_t bw = 0; bw < block_size; ++bw) {
                const uint64_t tw = (w_count[bw] == 0) ? 0 : CeilDivU64(w_count[bw], tiw);
                for (uint32_t bh = 0; bh < block_size; ++bh) {
                    prefix += out_batch * h_count[bh] * tw;
                }
            }
            return prefix;
        };
        const uint64_t stripe_groups_at_max_tile = calc_stripe_groups(stripe_max_tile_iw);
        const uint64_t row_group_threshold = std::max<uint64_t>(512ULL, static_cast<uint64_t>(num_cores_aiv) * 16ULL);
        const bool can_row_assemble = (depth_bytes % kDataBlockBytes == 0ULL) &&
            ((total_bytes >= 262144ULL) || (stripe_groups_at_max_tile >= row_group_threshold));

        if (can_row_assemble) {
            compact_row_assembly = true;
            strategy = BTS_STRATEGY_COMPACT_ROWASM;
            mode_ub = max_usable_ub;
            const uint64_t max_tile_ow_by_ub = std::max<uint64_t>(1ULL, mode_ub / depth_bytes);
            const uint64_t max_tile_ow_by_block = static_cast<uint64_t>(block_size) * kMaxBlockCount;
            tile_iw = std::max<uint64_t>(1ULL,
                std::min<uint64_t>(out_width, std::min<uint64_t>(max_tile_ow_by_ub, max_tile_ow_by_block)));

            // v20: when one whole output-row tile fits in UB, assemble multiple
            // output rows per logical job.  This keeps output writes contiguous
            // and moves the row tiling decision to host side.  For partial-width
            // row tiles we keep tile_rows=1 to avoid introducing UB-side source
            // stride on CopyOut.
            const uint64_t row_tile_bytes = tile_iw * depth_bytes;
            tile_rows = 1ULL;
            if (tile_iw == out_width && row_tile_bytes > 0ULL && row_tile_bytes <= mode_ub) {
                const uint64_t max_rows_by_ub = std::max<uint64_t>(1ULL, mode_ub / row_tile_bytes);
                uint64_t row_cap = std::min<uint64_t>(max_rows_by_ub, out_height);
                row_cap = std::min<uint64_t>(row_cap, kMaxBlockCount);

                // v21: cap multi-row aggregation to preserve enough independent
                // jobs for AIV parallelism.  v20 packed as many rows as UB could
                // hold; that helped very large strided-output cases, but it could
                // collapse moderate cases to too few groups and hurt badly.
                row_cap = std::min<uint64_t>(row_cap, kMaxCompactRowAssemblyRows);

                const uint64_t tiles_per_width = CeilDivU64(out_width, tile_iw);
                const uint64_t groups_per_row_tile = std::max<uint64_t>(1ULL, out_batch * tiles_per_width);
                const uint64_t target_groups = std::max<uint64_t>(static_cast<uint64_t>(num_cores_aiv) * 4ULL, 64ULL);
                const uint64_t needed_row_groups = CeilDivU64(target_groups, groups_per_row_tile);
                if (needed_row_groups > 0ULL && needed_row_groups < out_height) {
                    const uint64_t parallel_cap = std::max<uint64_t>(1ULL, out_height / needed_row_groups);
                    row_cap = std::min<uint64_t>(row_cap, parallel_cap);
                } else if (needed_row_groups >= out_height) {
                    row_cap = 1ULL;
                }

                tile_rows = std::max<uint64_t>(1ULL, row_cap);
            }
            ub_bytes = AlignUp(tile_rows * row_tile_bytes, kDataBlockBytes);
            compact_row_tiles_per_width = CeilDivU64(out_width, tile_iw);
            compact_row_groups_per_batch = CeilDivU64(out_height, tile_rows) * compact_row_tiles_per_width;
            total_groups = out_batch * compact_row_groups_per_batch;
            total_groups = std::max<uint64_t>(total_groups, 1ULL);
            compact_phase_num = block_size * block_size;
            compact_uniform = false;
            phase_group_size_first = 0;
        } else {
            strategy = BTS_STRATEGY_COMPACT_STRIPE;
            tile_iw = stripe_max_tile_iw;
            const uint64_t target_groups = static_cast<uint64_t>(num_cores_aiv) * ((block_size <= 4U) ? 1ULL : 2ULL);
            while (tile_iw > 1U && calc_stripe_groups(tile_iw) < target_groups) {
                tile_iw = (tile_iw + 1U) / 2U;
            }
            tile_iw = std::max<uint64_t>(1, tile_iw);
            ub_bytes = AlignUp(tile_iw * depth_aligned_bytes, kDataBlockBytes);

            for (uint32_t i = 0; i < block_size; ++i) {
                tiles_iw[i] = (w_count[i] == 0) ? 0 : CeilDivU64(w_count[i], tile_iw);
            }
            phase_prefix[0] = 0;
            uint64_t prefix = 0;
            const uint32_t phase_num = block_size * block_size;
            compact_phase_num = phase_num;
            compact_uniform = true;
            phase_group_size_first = 0;
            for (uint32_t phase = 0; phase < phase_num; ++phase) {
                const uint32_t bh = phase / block_size;
                const uint32_t bw = phase - bh * block_size;
                const uint64_t phase_groups = out_batch * h_count[bh] * tiles_iw[bw];
                if (phase == 0U) {
                    phase_group_size_first = phase_groups;
                } else if (phase_groups != phase_group_size_first) {
                    compact_uniform = false;
                }
                prefix += phase_groups;
                phase_prefix[phase + 1] = prefix;
            }
            if (phase_group_size_first == 0U) {
                compact_uniform = false;
            }
            const uint64_t compact_stripe_group_count = prefix;
            total_groups = std::max<uint64_t>(compact_stripe_group_count, 1ULL);

            // v54: remove the v2_01 tiny-phase schedule from the active
            // selector.  Previous tests showed that phase-level work units do
            // not improve #7 and can hurt #2/#8/#9.  Keep the correct compact
            // stripe work definition and only use COMPACT_SMALL as a low-overhead
            // single-buffer variant when the ordinary group count is already low.
            const uint64_t avg_job_bytes = tile_iw * depth_bytes;
            const uint64_t small_group_limit =
                std::max<uint64_t>(16ULL, static_cast<uint64_t>(num_cores_aiv) * 2ULL);
            if (total_groups <= small_group_limit &&
                avg_job_bytes <= 256ULL &&
                total_bytes <= 64ULL * 1024ULL) {
                strategy = BTS_STRATEGY_COMPACT_SMALL;
            }

            // v55: fast semantic-preserving schedule for the common B2 no-crop
            // compact case.  This does NOT convert BatchToSpace into B1_LINEAR;
            // it keeps the four phase stripes but uses a direct B2/no-crop job
            // decoder in the kernel instead of the general compact decoder.
            if (block_size == 2U &&
                crop_top == 0U && crop_bottom == 0U &&
                crop_left == 0U && crop_right == 0U) {
                strategy = BTS_STRATEGY_B2_NOCROP_FAST;
            }
        }
    } else {
        mode = is_f16 ? BTS_MODE_F16_GENERIC_STRIPE : BTS_MODE_F32_GENERIC_STRIPE;
        strategy = BTS_STRATEGY_GENERIC;
        mode_ub = std::min<uint64_t>(max_usable_ub, kLargePerBufferUbCap);
        tile_iw = std::max<uint64_t>(1, mode_ub / depth_aligned_bytes);
        tile_iw = std::min<uint64_t>(tile_iw, width);
        tile_iw = std::min<uint64_t>(tile_iw, kMaxBlockCount);
        tile_iw = std::max<uint64_t>(tile_iw, 1);
        ub_bytes = AlignUp(tile_iw * depth_aligned_bytes, kDataBlockBytes);
        const uint64_t tiles_per_input_row = CeilDivU64(width, tile_iw);
        total_groups = out_batch * block_area * height * tiles_per_input_row;
        total_groups = std::max<uint64_t>(total_groups, 1);
    }

    if (!FitsU32(tile_elems) || !FitsU32(tile_rows) || !FitsU32(tile_iw) || !FitsU32(ub_bytes) ||
        !FitsU32(total_groups) || !FitsU32(compact_row_tiles_per_width) ||
        !FitsU32(compact_row_groups_per_batch)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t used_core_num = SelectCoreNum(num_cores_aiv, total_groups, total_bytes, mode);
    const bool global_no_crop =
        (crop_top == 0U && crop_bottom == 0U && crop_left == 0U && crop_right == 0U);
    used_core_num = ApplyDtypePathCoreHints(used_core_num,
                                            num_cores_aiv,
                                            is_f16,
                                            block_size,
                                            global_no_crop,
                                            height,
                                            width,
                                            out_batch,
                                            out_height,
                                            out_width,
                                            depth,
                                            depth_bytes,
                                            total_bytes,
                                            total_groups,
                                            strategy);

    // v47: safe corner-oriented core selection.
    // Keep all data mapping in the proven compact/row-assembly kernels.  Do not
    // redirect B2 corner cases to B1_LINEAR or force them into row-assembly: both
    // variants produced WA.  The only safe optimization here is to reduce excessive
    // AIV fan-out for shapes whose work units are few and small.
    if (strategy == BTS_STRATEGY_COMPACT_STRIPE) {
        const bool no_crop =
            (crop_top == 0U && crop_bottom == 0U && crop_left == 0U && crop_right == 0U);

        // Tiny B2 no-crop spatial case.  The correct mapping is still compact
        // phase-stripe, but opening many AIVs for only a few phase groups is pure
        // launch/scheduling overhead.  This mirrors the useful idea seen in the
        // competing CE logs while preserving the original algorithm.
        const bool b2_tiny_nocrop =
            (block_size == 2U && no_crop && height == 1ULL && width == 1ULL &&
             out_batch == 1ULL && out_height == 2ULL && out_width == 2ULL);
        if (b2_tiny_nocrop) {
            used_core_num = std::min<uint32_t>(used_core_num, 8U);
            used_core_num = std::max<uint32_t>(used_core_num, 1U);
        }

        // Medium-small compact stripe fallback.  This is the v38 rule that
        // consistently improved the high-ratio compact small case.  Keep it after
        // the tiny rule; both are only core-count changes, not semantic changes.
        if (total_bytes > 16ULL * 1024ULL &&
            total_bytes <= 64ULL * 1024ULL &&
            total_groups <= std::max<uint64_t>(16ULL, static_cast<uint64_t>(num_cores_aiv) * 2ULL)) {
            used_core_num = std::min<uint32_t>(used_core_num, 8U);
            used_core_num = std::max<uint32_t>(used_core_num, 1U);
        }

        // B2 no-crop aligned row-size corner.  The unsafe v44 experiment forced
        // this into generic row-assembly and caused WA.  Here it remains in the
        // correct compact-stripe kernel; only excessive core fan-out is capped.
        // Use a softer cap than the tiny/medium-small cases to avoid hurting
        // bandwidth-oriented aligned rows.
        const uint64_t source_row_bytes = width * depth_bytes;
        const uint64_t output_row_bytes = out_width * depth_bytes;
        const bool b2_nocrop_aligned_smallrow =
            (block_size == 2U && no_crop &&
             (depth_bytes % kDataBlockBytes == 0ULL) &&
             source_row_bytes <= 32ULL * 1024ULL &&
             output_row_bytes <= 32ULL * 1024ULL &&
             total_bytes <= 128ULL * 1024ULL &&
             total_groups <= std::max<uint64_t>(32ULL, static_cast<uint64_t>(num_cores_aiv) * 4ULL));
        if (b2_nocrop_aligned_smallrow) {
            used_core_num = std::min<uint32_t>(used_core_num, 16U);
            used_core_num = std::max<uint32_t>(used_core_num, 1U);
        }
    }

    // v54: low-overhead core caps for COMPACT_SMALL.  These are intentionally
    // host-only scheduling changes; they do not alter phase mapping or copy
    // semantics.  The aim is to reduce fixed AIV fan-out overhead on the small
    // high-ratio points (#4/#6/#7) without touching ROWASM or large-copy paths.
    if (strategy == BTS_STRATEGY_COMPACT_SMALL) {
        if (total_bytes <= 2ULL * 1024ULL && total_groups <= 16ULL) {
            used_core_num = std::min<uint32_t>(used_core_num, 1U);
        } else if (total_bytes <= 8ULL * 1024ULL && total_groups <= 32ULL) {
            used_core_num = std::min<uint32_t>(used_core_num, 2U);
        } else if (total_bytes <= 16ULL * 1024ULL && total_groups <= 64ULL) {
            used_core_num = std::min<uint32_t>(used_core_num, 4U);
        }
        used_core_num = std::max<uint32_t>(used_core_num, 1U);
    }

    // v55: B2 no-crop fast decode still has the same logical work units as
    // compact stripe, but for very small payloads excessive core fan-out costs
    // more than it saves.  Use a smooth cap; larger cases keep enough cores for
    // bandwidth.
    if (strategy == BTS_STRATEGY_B2_NOCROP_FAST) {
        if (total_bytes <= 4ULL * 1024ULL) {
            used_core_num = std::min<uint32_t>(used_core_num, 1U);
        } else if (total_bytes <= 16ULL * 1024ULL) {
            used_core_num = std::min<uint32_t>(used_core_num, 2U);
        } else if (total_bytes <= 64ULL * 1024ULL) {
            used_core_num = std::min<uint32_t>(used_core_num, 8U);
        } else if (total_bytes <= 128ULL * 1024ULL) {
            used_core_num = std::min<uint32_t>(used_core_num, 16U);
        }
        used_core_num = std::max<uint32_t>(used_core_num, 1U);
    }

    const uint64_t work_num = IsB1LinearMode(mode) ? total_elems : total_groups;
    used_core_num = std::min<uint32_t>(
        used_core_num,
        static_cast<uint32_t>(std::min<uint64_t>(work_num, static_cast<uint64_t>(num_cores_aiv))));
    used_core_num = std::max<uint32_t>(used_core_num, 1U);

    const uint64_t base_work = (used_core_num == 0U) ? work_num : (work_num / used_core_num);
    const uint64_t rem_work = (used_core_num == 0U) ? 0U : (work_num - base_work * used_core_num);
    const uint64_t big_core_num = rem_work;
    const uint64_t big_core_work_num = base_work + ((rem_work != 0U) ? 1U : 0U);
    const uint64_t small_core_work_num = base_work;
    if (!FitsU32(work_num) || !FitsU32(big_core_num) || !FitsU32(big_core_work_num) ||
        !FitsU32(small_core_work_num)) {
        return ge::GRAPH_FAILED;
    }

    BatchToSpaceTilingData *tiling = context->GetTilingData<BatchToSpaceTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->batch = static_cast<uint32_t>(batch);
    tiling->height = static_cast<uint32_t>(height);
    tiling->width = static_cast<uint32_t>(width);
    tiling->depth = static_cast<uint32_t>(depth);
    tiling->out_batch = static_cast<uint32_t>(out_batch);
    tiling->out_height = static_cast<uint32_t>(out_height);
    tiling->out_width = static_cast<uint32_t>(out_width);
    tiling->total_elems = static_cast<uint32_t>(total_elems);
    tiling->total_groups = static_cast<uint32_t>(total_groups);
    tiling->block_size = block_size;
    tiling->crop_top = crop_top;
    tiling->crop_bottom = crop_bottom;
    tiling->crop_left = crop_left;
    tiling->crop_right = crop_right;
    tiling->mode = mode;
    tiling->strategy = strategy;
    tiling->dtype_size = dtype_size;
    tiling->depth_bytes = static_cast<uint32_t>(depth_bytes);
    tiling->depth_aligned_bytes = static_cast<uint32_t>(depth_aligned_bytes);
    tiling->tile_elems = static_cast<uint32_t>(tile_elems);
    tiling->tile_rows = static_cast<uint32_t>(tile_rows);
    tiling->tile_iw = static_cast<uint32_t>(tile_iw);
    tiling->ub_bytes = static_cast<uint32_t>(ub_bytes);
    tiling->used_core_num = used_core_num;
    tiling->big_core_num = static_cast<uint32_t>(big_core_num);
    tiling->big_core_work_num = static_cast<uint32_t>(big_core_work_num);
    tiling->small_core_work_num = static_cast<uint32_t>(small_core_work_num);
    for (uint32_t i = 0; i < BTS_MAX_COMPACT_BS; ++i) {
        tiling->h_begin[i] = static_cast<uint32_t>(h_begin[i]);
        tiling->h_count[i] = static_cast<uint32_t>(h_count[i]);
        tiling->w_begin[i] = static_cast<uint32_t>(w_begin[i]);
        tiling->w_count[i] = static_cast<uint32_t>(w_count[i]);
        tiling->tiles_iw[i] = static_cast<uint32_t>(tiles_iw[i]);
    }
    for (uint32_t i = 0; i < BTS_MAX_COMPACT_PHASE + 1U; ++i) {
        tiling->phase_prefix[i] = static_cast<uint32_t>(phase_prefix[i]);
    }
    tiling->compact_phase_num = compact_phase_num;
    tiling->compact_uniform = compact_uniform ? 1U : 0U;
    tiling->compact_phase_group_size = static_cast<uint32_t>(phase_group_size_first);
    tiling->compact_row_assembly = compact_row_assembly ? 1U : 0U;
    tiling->compact_row_tiles_per_width = static_cast<uint32_t>(compact_row_tiles_per_width);
    tiling->compact_row_groups_per_batch = static_cast<uint32_t>(compact_row_groups_per_batch);

    uint32_t DT_X = is_f16 ? C_DT_FLOAT16 : C_DT_FLOAT;
    uint32_t MODE_KIND = mode;
    ASCENDC_TPL_SEL_PARAM(context, DT_X, MODE_KIND);

    if (std::getenv("BTS_TILING_TRACE") != nullptr) {
        std::fprintf(stderr,
            "[BTS_TILING]{\"shape\":\"%llux%llux%llux%llu\","
            "\"dtype_size\":%u,\"block_size\":%u,"
            "\"crops\":\"%u,%u,%u,%u\","
            "\"out_shape\":\"%llux%llux%llux%llu\","
            "\"mode\":%u,\"mode_name\":\"%s\",\"strategy\":%u,\"total_groups\":%llu,\"total_elems\":%llu,"
            "\"tile_elems\":%llu,\"tile_rows\":%llu,\"tile_iw\":%llu,"
            "\"ub_bytes\":%llu,\"used_core_num\":%u,"
            "\"big_core_num\":%llu,\"big_core_work_num\":%llu,"
            "\"small_core_work_num\":%llu,\"compact_uniform\":%u,\"compact_row_assembly\":%u}\n",
            static_cast<unsigned long long>(batch),
            static_cast<unsigned long long>(height),
            static_cast<unsigned long long>(width),
            static_cast<unsigned long long>(depth),
            dtype_size,
            block_size,
            crop_top, crop_bottom, crop_left, crop_right,
            static_cast<unsigned long long>(out_batch),
            static_cast<unsigned long long>(out_height),
            static_cast<unsigned long long>(out_width),
            static_cast<unsigned long long>(depth),
            mode,
            ModeName(mode),
            strategy,
            static_cast<unsigned long long>(total_groups),
            static_cast<unsigned long long>(total_elems),
            static_cast<unsigned long long>(tile_elems),
            static_cast<unsigned long long>(tile_rows),
            static_cast<unsigned long long>(tile_iw),
            static_cast<unsigned long long>(ub_bytes),
            used_core_num,
            static_cast<unsigned long long>(big_core_num),
            static_cast<unsigned long long>(big_core_work_num),
            static_cast<unsigned long long>(small_core_work_num),
            compact_uniform ? 1U : 0U,
            compact_row_assembly ? 1U : 0U);
    }

    context->SetBlockDim(used_core_num);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *x_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    if (x_shape == nullptr || y_shape == nullptr || x_shape->GetDimNum() != kInputRank) {
        return GRAPH_FAILED;
    }

    const int64_t batch = x_shape->GetDim(0);
    const int64_t height = x_shape->GetDim(1);
    const int64_t width = x_shape->GetDim(2);
    const int64_t depth = x_shape->GetDim(3);

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::TypedContinuousVector<int64_t> *attr_crops = attrs->GetListInt(0);
    const int64_t *attr_block_size = attrs->GetInt(1);
    if (attr_crops == nullptr || attr_block_size == nullptr || *attr_block_size <= 0) {
        return GRAPH_FAILED;
    }

    int64_t crops[kCropNum] = {0, 0, 0, 0};
    if (!ReadCropList(attr_crops, crops)) {
        return GRAPH_FAILED;
    }
    for (uint32_t i = 0; i < kCropNum; ++i) {
        if (crops[i] < 0) {
            return GRAPH_FAILED;
        }
    }
    const int64_t block_size = *attr_block_size;
    const int64_t block_area = block_size * block_size;
    if (block_area <= 0 || batch % block_area != 0) {
        return GRAPH_FAILED;
    }
    const int64_t out_batch = batch / block_area;
    const int64_t out_height = height * block_size - crops[0] - crops[1];
    const int64_t out_width = width * block_size - crops[2] - crops[3];
    if (out_height <= 0 || out_width <= 0) {
        return GRAPH_FAILED;
    }

    y_shape->SetDimNum(kInputRank);
    y_shape->SetDim(0, out_batch);
    y_shape->SetDim(1, out_height);
    y_shape->SetDim(2, out_width);
    y_shape->SetDim(3, depth);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    const auto input_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, input_dtype);
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
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(BatchToSpace);
}  // namespace ops
