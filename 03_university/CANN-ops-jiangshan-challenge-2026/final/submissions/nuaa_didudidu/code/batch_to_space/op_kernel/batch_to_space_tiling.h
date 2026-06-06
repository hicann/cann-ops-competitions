// Refactored tiling plan shared by host validation and the generic device path.
#pragma once

#include <cstdint>

// The operator uses NHWC indexing throughout host inference and device kernels.
struct B2SLayout4D {
    uint32_t batch_count;
    uint32_t height_extent;
    uint32_t width_extent;
    uint32_t channel_count;
};

// Crop order matches the public op attribute: top, bottom, left, right.
struct B2SCropBox {
    uint32_t trim_top;
    uint32_t trim_bottom;
    uint32_t trim_left;
    uint32_t trim_right;
};

// Host selects one fallback traversal policy; specialized schedules ignore it.
static constexpr uint32_t B2S_RUNTIME_ROW_PHASE = 0U;
static constexpr uint32_t B2S_RUNTIME_FLAT_COPY = 1U;
static constexpr uint32_t B2S_RUNTIME_PIXEL = 2U;
static constexpr uint32_t B2S_RUNTIME_PIXEL_CHANNEL = 3U;
static constexpr uint32_t B2S_RUNTIME_VERTICAL = 4U;
static constexpr uint32_t B2S_RUNTIME_PACKED_ROW = 5U;
static constexpr uint32_t B2S_RUNTIME_WIDE_PIXEL = 6U;

// Specialized schedules bake their constants into templates; the fallback path
// reads this compact plan to cover all other legal shapes.
struct B2SDispatchPlan {
    B2SLayout4D input;
    B2SLayout4D output;
    B2SCropBox crop;
    uint32_t block_extent;
    uint32_t element_total;
    uint32_t buffer_element_quota;
    uint32_t runtime_path;
};
