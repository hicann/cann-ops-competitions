// Tiling data structure definition.
#pragma once

#include <cstdint>

enum BatchToSpaceRouteId : uint32_t {
    PATH_F32_NC_DEEP = 1,
    PATH_F32_CROP_THIN = 2,
    PATH_F32_NC_MID = 3,
    PATH_F32_NC_SHORT = 4,
    PATH_F16_CROP_ODD_DEPTH = 5,
    PATH_F16_NC_HUGE_DEPTH = 6,
    PATH_F16_NC_EXTREME_DEPTH = 7,
    PATH_F16_NC_WIDE_ROW = 8,
    PATH_F16_CROP_BS4_WIDE = 9,
    PATH_F16_NC_SHORT_WIDTH = 10,
};

struct BatchToSpaceTilingData {
    uint32_t reserved;
};

static_assert(sizeof(BatchToSpaceTilingData) == 4,
              "BatchToSpaceTilingData must stay compact");
