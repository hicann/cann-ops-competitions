#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScaleTilingData)
// Total number of input/output elements processed by the operator.
TILING_DATA_FIELD_DEF(uint32_t, totalSize);
// Number of scale values. Bias has the same element count when present.
TILING_DATA_FIELD_DEF(uint32_t, scaleElemCount);
// Number of contiguous elements that share one scale/bias value. This is the
// product of dimensions after the scaled axis range.
TILING_DATA_FIELD_DEF(uint32_t, suffixElemCount);
// Element-based core split for the generic scalar/vector paths.
TILING_DATA_FIELD_DEF(uint32_t, baseElems);
TILING_DATA_FIELD_DEF(uint32_t, formerElems);
// Group-based core split for the expanded broadcast path. A group is one
// suffixElemCount-sized region that uses one scale/bias value.
TILING_DATA_FIELD_DEF(uint32_t, baseGroups);
TILING_DATA_FIELD_DEF(uint32_t, formerGroups);
// UB tile size in elements. For expanded broadcast this includes alignment
// padding between groups.
TILING_DATA_FIELD_DEF(uint32_t, tileSize);
// Padded group stride used by expanded broadcast to keep duplicated scale/bias
// vectors aligned for vector instructions.
TILING_DATA_FIELD_DEF(uint32_t, expandedGroupStride);
// Optional execution flags consumed by the kernel.
TILING_DATA_FIELD_DEF(uint8_t, hasBias);
TILING_DATA_FIELD_DEF(uint8_t, useExpandedBroadcast);
TILING_DATA_FIELD_DEF(uint8_t, useRelaxedExpandedFp16);
TILING_DATA_FIELD_DEF(uint8_t, useGroupAlignedExpandedSplit);
// xt5-style middle-axis broadcast path fields. These are used only when the
// host selects tiling key 8; the baseline path above ignores them.
TILING_DATA_FIELD_DEF(uint32_t, smallSize);
TILING_DATA_FIELD_DEF(uint32_t, incSize);
TILING_DATA_FIELD_DEF(uint32_t, formerNum);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, mmInputDims);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, mmOtherDims);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, mmOutputDims);
TILING_DATA_FIELD_DEF(uint8_t, nOutputDims);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Scale, ScaleTilingData)
} // namespace optiling
