#ifndef ASSIGN_TILING_H
#define ASSIGN_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(AssignTilingData)
TILING_DATA_FIELD_DEF(uint32_t, bytesPerCore);
TILING_DATA_FIELD_DEF(uint32_t, tailBytes);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Assign, AssignTilingData)
}
#endif
