#ifndef UNPACK_TILING_H
#define UNPACK_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(UnpackTilingData)
  TILING_DATA_FIELD_DEF(int64_t, cutAxis);
  TILING_DATA_FIELD_DEF(int64_t, splitCount);
  TILING_DATA_FIELD_DEF(int64_t, elemBytes);
  TILING_DATA_FIELD_DEF(int64_t, frontSpan);
  TILING_DATA_FIELD_DEF(int64_t, laneWidth);
  TILING_DATA_FIELD_DEF(int64_t, outputElems);
  TILING_DATA_FIELD_DEF(int64_t, route);
  TILING_DATA_FIELD_DEF(int64_t, ticketCount);
  TILING_DATA_FIELD_DEF(int64_t, ticketsPerCore);
  TILING_DATA_FIELD_DEF(int64_t, stageElems);
  TILING_DATA_FIELD_DEF(int64_t, directLaneRelay);
  TILING_DATA_FIELD_DEF(int64_t, bodyTailCores);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Unpack, UnpackTilingData)
} // namespace optiling

#endif
