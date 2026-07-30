
#include "register/tilingdata_base.h"
constexpr int MAX_SHAPE(4);
constexpr int DIM_LAST(MAX_SHAPE - 1);

namespace optiling {

struct AssignOneShapeTilingData {
    int32_t tile_length;
    int32_t out_shape;
};

template <int COMPACT_SHAPE_NUM>
struct AssignCompactTilingData {
    // Kernel 端由 tile_length + out_shape[DIM_LAST] 复原 iter_per_vector/residue/iterations。
    // COMPACT_SHAPE_NUM 只保存压缩后 shape 的有效后缀。
    int32_t tile_length;
    int32_t other_shape[COMPACT_SHAPE_NUM];
    int32_t out_shape[COMPACT_SHAPE_NUM];
};

BEGIN_TILING_DATA_DEF(AssignTilingData)
TILING_DATA_FIELD_DEF(int32_t, tile_length);

TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_SHAPE, other_shape);
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_SHAPE, out_shape);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Assign, AssignTilingData)
} // namespace optiling
