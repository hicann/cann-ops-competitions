#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <limits>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "../op_kernel/batch_to_space_tiling.h"
#include "../op_kernel/tiling_key_batch_to_space.h"

namespace optiling {
namespace {

bool TuningEnabled() {
  static const bool enabled = []() {
    const char *value = std::getenv("BTS_TUNE_ENABLE");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
  }();
  return enabled;
}

bool ReadTuningU32(const char *name, uint32_t &result) {
  if (!TuningEnabled()) {
    return false;
  }
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value || *end != '\0' ||
      parsed > static_cast<unsigned long>(std::numeric_limits<uint32_t>::max())) {
    return false;
  }
  result = static_cast<uint32_t>(parsed);
  return true;
}

uint32_t TunableU32(const char *name, uint32_t fallback) {
  uint32_t value = fallback;
  return ReadTuningU32(name, value) ? value : fallback;
}

void SetBatchToSpaceTilingKey(gert::TilingContext *context,
                              uint32_t staticVariant =
                                  BTS_STATIC_VARIANT_NONE) {
  const ge::DataType dtype_x =
      context->GetRequiredInputTensor(0)->GetDataType();
  const uint32_t dtX = static_cast<uint32_t>(dtype_x);
  const uint32_t tuning = TuningEnabled() ? 1U : 0U;
  ASCENDC_TPL_SEL_PARAM(context, dtX, tuning, staticVariant);
}

// BTS_DISPATCH_BEGIN
// Pure dispatch core: depends only on <cstdint>/<algorithm> plus the env
// helpers above (TuningEnabled/ReadTuningU32/TunableU32). When this region is
// extracted into the standalone verify harness, those helpers are stubbed to
// their "tuning off" identities, so the region stays pure for verification.
constexpr uint32_t BTS_UB_FALLBACK_BYTES = 192 * 1024;
constexpr uint32_t BTS_DOUBLE_BUFFER_NUM = 2;
constexpr uint32_t BTS_DATA_BLOCK_BYTES = 32;
constexpr uint32_t BTS_GM_ALIGN_BYTES = 512;
constexpr uint32_t BTS_LARGE_DEPTH_MIN_BYTES = 512;
constexpr uint32_t BTS_MAX_BLOCK_COUNT = 4095;
constexpr uint32_t BTS_MAX_UINT16 = 65535;
constexpr uint64_t BTS_MAX_UINT32 = 0xFFFFFFFFULL;
constexpr uint32_t BTS_GM_MIN_DMA_BYTES = 16 * 1024;
constexpr uint32_t BTS_UB_FILL_NUMER = 7;
constexpr uint32_t BTS_UB_FILL_DENOM = 10;
constexpr uint32_t BTS_MIN_TASKS = 32;
constexpr uint32_t BTS_MIN_ACTIVE_CORES = 8;
constexpr uint64_t BTS_SMALL_OUTPUT_BYTES = 4 * 1024;
constexpr uint64_t BTS_MEDIUM_OUTPUT_BYTES = 32 * 1024;
constexpr uint32_t BTS_MAX_ROW_TILE = 8;
constexpr uint32_t BTS_MAX_EXT_BLOCK_LEN = 2097151;

// Fast-path hardware envelopes: one 512B burst, vector copy limits, the
// minimum DMA traffic that amortizes launch cost, and the gather index budget.
constexpr uint32_t BTS_FP_DTYPE_BYTES = 2;
constexpr uint64_t BTS_FP_MIN_TRAFFIC_BYTES = 4ULL * 1024ULL * 1024ULL;
constexpr uint32_t BTS_FP_DEPTH_BURST_BYTES = 512;
constexpr uint32_t BTS_FP_COPY_REPEAT_MAX = 255;
constexpr uint32_t BTS_FP_COPY_MASK_MAX_B16 = 128;
constexpr uint32_t BTS_FP_HFUSE_ASPECT = 64;
constexpr uint32_t BTS_FP_HFUSE_ROW_ALIGN = 8;
constexpr uint32_t BTS_FP_HFUSE_TARGET_ROWS = 24;
constexpr uint32_t BTS_FP_GI_IDX_BUDGET_BYTES = 184 * 1024;
constexpr uint32_t BTS_W512_MIN_TRAFFIC_BYTES = 4U * 1024U * 1024U;
constexpr uint32_t BTS_W512_COPY_MASK_ELEMS = 128;
constexpr uint32_t BTS_W512_MIN_OUT_WIDTH = 512;

uint32_t CeilDiv(uint32_t x, uint32_t y) {
  return y == 0 ? 0 : (x + y - 1) / y;
}

uint64_t AlignUp64(uint64_t x, uint64_t align) {
  return align == 0 ? x : ((x + align - 1) / align) * align;
}

uint32_t AlignUpU32(uint32_t x, uint32_t align) {
  return static_cast<uint32_t>(AlignUp64(x, align));
}

uint32_t GcdU32(uint32_t a, uint32_t b) {
  while (b != 0) {
    const uint32_t t = a % b;
    a = b;
    b = t;
  }
  return a;
}

uint64_t Gcd64(uint64_t a, uint64_t b) {
  while (b != 0) {
    const uint64_t t = a % b;
    a = b;
    b = t;
  }
  return a;
}

bool Is512BMultipleBytes(uint64_t nbytes) {
  return nbytes > 0 && (nbytes % BTS_GM_ALIGN_BYTES) == 0;
}

uint64_t FloorDiv64(uint64_t x, uint64_t y) {
  return y == 0 ? 0 : x / y;
}

uint64_t FloorAlign64(uint64_t x, uint64_t align) {
  return align == 0 ? x : (x / align) * align;
}

uint64_t CeilAlign64(uint64_t x, uint64_t align) {
  return align == 0 ? x : FloorAlign64(x + align - 1, align);
}

uint32_t CalcGmAlignOW(uint64_t columnBytes) {
  if (columnBytes == 0) {
    return 1;
  }
  const uint64_t g = Gcd64(columnBytes, BTS_GM_ALIGN_BYTES);
  return static_cast<uint32_t>(std::max<uint64_t>(1, BTS_GM_ALIGN_BYTES / g));
}

uint32_t SnapTileOWDown(uint32_t tileOW, uint32_t alignOW, uint32_t outWidth) {
  if (tileOW >= outWidth) {
    return outWidth;
  }
  if (alignOW <= 1 || tileOW < alignOW) {
    return std::max<uint32_t>(1, tileOW);
  }
  const uint32_t snapped = (tileOW / alignOW) * alignOW;
  if (snapped == 0) {
    return std::min(outWidth, alignOW);
  }
  return snapped;
}

uint32_t ChooseMaxTileOW(uint32_t outWidth, uint32_t maxTileOW, uint32_t alignOW) {
  if (outWidth == 0 || maxTileOW == 0) {
    return 1;
  }
  if (outWidth <= maxTileOW) {
    return outWidth;
  }
  return SnapTileOWDown(maxTileOW, alignOW, outWidth);
}

uint32_t SnapTileOWDivideOutWidth(uint32_t tileOW, uint32_t outWidth,
                                  uint32_t alignOW) {
  if (outWidth == 0 || tileOW == 0) {
    return 1;
  }
  if (tileOW >= outWidth) {
    return outWidth;
  }
  uint32_t snapped = SnapTileOWDown(tileOW, alignOW, outWidth);
  while (snapped >= alignOW && outWidth % snapped != 0) {
    snapped -= alignOW;
  }
  if (snapped >= alignOW && outWidth % snapped == 0) {
    return snapped;
  }
  return SnapTileOWDown(tileOW, alignOW, outWidth);
}

uint32_t Calc512BMaxTileOW(uint32_t outWidth, uint32_t depth, uint32_t tileDepth,
                           uint32_t dtypeSize, uint32_t blockSize,
                           uint64_t usableUb) {
  const uint64_t ubPixelBytes =
      AlignUp64(static_cast<uint64_t>(tileDepth) * dtypeSize, BTS_DATA_BLOCK_BYTES);
  uint32_t maxOW = static_cast<uint32_t>(
      std::max<uint64_t>(1, usableUb / std::max<uint64_t>(ubPixelBytes, 1)));
  maxOW = std::min(maxOW, outWidth);

  if (blockSize > 1) {
    const uint64_t maxBlockCountOW =
        static_cast<uint64_t>(BTS_MAX_BLOCK_COUNT) * blockSize;
    maxOW = std::min<uint32_t>(
        maxOW, static_cast<uint32_t>(
                   std::min<uint64_t>(maxBlockCountOW, BTS_MAX_UINT32)));
  }

  const uint64_t gmRowBytes = static_cast<uint64_t>(depth) * dtypeSize;
  const uint32_t alignOW = CalcGmAlignOW(gmRowBytes);
  return SnapTileOWDivideOutWidth(SnapTileOWDown(maxOW, alignOW, outWidth),
                                  outWidth, alignOW);
}

uint32_t Choose512BTileOW(uint32_t outWidth, uint32_t depth, uint32_t tileDepth,
                          uint32_t dtypeSize, uint32_t blockSize,
                          uint64_t usableUb) {
  const uint32_t maxTileOW = Calc512BMaxTileOW(outWidth, depth, tileDepth, dtypeSize,
                                               blockSize, usableUb);
  const uint64_t gmRowBytes = static_cast<uint64_t>(depth) * dtypeSize;
  const uint32_t alignOW = CalcGmAlignOW(gmRowBytes);
  const uint32_t tileOW = ChooseMaxTileOW(outWidth, maxTileOW, alignOW);
  return SnapTileOWDivideOutWidth(tileOW, outWidth, alignOW);
}

uint32_t ChooseAlignedDepthTileElems(uint32_t depth, uint32_t dtypeSize,
                                     uint64_t pixelBytes, uint64_t usableUb) {
  if (pixelBytes <= usableUb && pixelBytes <= BTS_MAX_EXT_BLOCK_LEN) {
    return depth;
  }
  uint64_t maxElems = usableUb / dtypeSize;
  maxElems = std::min<uint64_t>(maxElems, depth);
  maxElems = std::min<uint64_t>(maxElems, BTS_MAX_EXT_BLOCK_LEN / dtypeSize);
  const uint32_t elemsPerBlock = BTS_DATA_BLOCK_BYTES / dtypeSize;
  if (maxElems >= elemsPerBlock) {
    maxElems = (maxElems / elemsPerBlock) * elemsPerBlock;
  }
  return static_cast<uint32_t>(std::max<uint64_t>(1, maxElems));
}

uint32_t FloorTo32BDepthElems(uint32_t elems, uint32_t dtypeSize) {
  if (elems == 0 || dtypeSize == 0) {
    return 1;
  }
  const uint32_t elemsPerBlock = BTS_DATA_BLOCK_BYTES / dtypeSize;
  if (elemsPerBlock == 0) {
    return elems;
  }
  uint32_t snapped = elems;
  if (snapped >= elemsPerBlock) {
    snapped = (snapped / elemsPerBlock) * elemsPerBlock;
  }
  return std::max<uint32_t>(1, snapped);
}

uint32_t Largest32BAlignedDivisorDepth(uint32_t maxTile, uint32_t depth,
                                       uint32_t dtypeSize) {
  const uint32_t minTile =
      std::max<uint32_t>(1, BTS_DATA_BLOCK_BYTES / dtypeSize);
  if (maxTile >= depth) {
    return depth;
  }
  const uint32_t cap = std::min(maxTile, depth);
  if (cap < minTile) {
    return minTile;
  }
  for (uint32_t t = cap; t >= minTile; --t) {
    if (depth % t != 0) {
      continue;
    }
    if ((static_cast<uint64_t>(t) * dtypeSize) % BTS_DATA_BLOCK_BYTES != 0) {
      continue;
    }
    return t;
  }
  return std::max(minTile, FloorTo32BDepthElems(cap, dtypeSize));
}

uint32_t FloorTo512BDepthElems(uint32_t elems, uint32_t dtypeSize) {
  if (elems == 0 || dtypeSize == 0) {
    return 1;
  }
  const uint32_t elemsPerBlock = BTS_DATA_BLOCK_BYTES / dtypeSize;
  const uint32_t elemsPer512B = BTS_GM_ALIGN_BYTES / dtypeSize;
  uint32_t snapped = elems;
  if (elemsPer512B > 0 && elems >= elemsPer512B) {
    snapped = (elems / elemsPer512B) * elemsPer512B;
  }
  if (snapped == 0) {
    snapped = std::min(elems, std::max<uint32_t>(1, elemsPer512B));
  }
  if (elemsPerBlock > 0 && snapped >= elemsPerBlock) {
    snapped = (snapped / elemsPerBlock) * elemsPerBlock;
  }
  return std::max<uint32_t>(1, snapped);
}

uint32_t Largest512BAlignedDivisorDepth(uint32_t maxTile, uint32_t depth,
                                        uint32_t dtypeSize) {
  const uint32_t minTile = std::max<uint32_t>(1, BTS_DATA_BLOCK_BYTES / dtypeSize);
  if (maxTile >= depth) {
    return depth;
  }
  const uint32_t cap = std::min(maxTile, depth - 1);
  if (cap < minTile) {
    return minTile;
  }
  for (uint32_t t = cap; t >= minTile; --t) {
    if (depth % t != 0) {
      continue;
    }
    if (!Is512BMultipleBytes(static_cast<uint64_t>(t) * dtypeSize)) {
      continue;
    }
    return t;
  }
  return std::max(minTile, FloorTo512BDepthElems(cap, dtypeSize));
}

uint32_t Choose512BDepthTileElems(uint32_t depth, uint32_t dtypeSize,
                                  uint64_t pixelBytes, uint64_t usableUb) {
  uint32_t tileDepth =
      ChooseAlignedDepthTileElems(depth, dtypeSize, pixelBytes, usableUb);
  if (tileDepth >= depth) {
    return depth;
  }
  tileDepth = FloorTo512BDepthElems(tileDepth, dtypeSize);
  const uint32_t minDepth = std::max<uint32_t>(1, BTS_DATA_BLOCK_BYTES / dtypeSize);
  tileDepth = std::max(minDepth, tileDepth);
  while (tileDepth > minDepth &&
         static_cast<uint64_t>(tileDepth) * dtypeSize > usableUb) {
    const uint32_t half = std::max(minDepth, tileDepth / 2);
    tileDepth = FloorTo512BDepthElems(half, dtypeSize);
    if (tileDepth == half && tileDepth > minDepth) {
      tileDepth = half;
    }
  }
  if (tileDepth < depth) {
    tileDepth = Largest512BAlignedDivisorDepth(tileDepth, depth, dtypeSize);
  }
  return std::max<uint32_t>(1, std::min(tileDepth, depth));
}

bool AlignedTileOWParamOk(uint32_t blockSize, uint32_t tileOW,
                          uint64_t depthTileBytes) {
  if (depthTileBytes == 0 || depthTileBytes % BTS_DATA_BLOCK_BYTES != 0) {
    return false;
  }
  const uint64_t depthBlocks = depthTileBytes / BTS_DATA_BLOCK_BYTES;
  return depthBlocks > 0 && depthBlocks <= BTS_MAX_UINT16 &&
         CeilDiv(tileOW, blockSize) <= BTS_MAX_BLOCK_COUNT &&
         static_cast<uint64_t>(blockSize - 1) * depthBlocks <= BTS_MAX_UINT16;
}

bool CanUseAlignedAssemblyWithDepthTile(uint32_t depth, uint32_t dtypeSize,
                                        uint32_t blockSize, uint64_t usableUb) {
  const uint64_t pixelBytes = static_cast<uint64_t>(depth) * dtypeSize;
  const uint32_t tileDepth =
      ChooseAlignedDepthTileElems(depth, dtypeSize, pixelBytes, usableUb);
  const uint64_t depthTileBytes = static_cast<uint64_t>(tileDepth) * dtypeSize;
  return AlignedTileOWParamOk(blockSize, 1, depthTileBytes);
}

uint32_t ChooseTileOWDepth32BPadded(uint32_t outWidth, uint32_t depth,
                                    uint32_t dtypeSize, uint64_t usableUb) {
  if (outWidth == 0 || depth == 0 || dtypeSize == 0) {
    return 1;
  }

  const uint64_t pixelBytes = static_cast<uint64_t>(depth) * dtypeSize;
  const uint64_t ubPixelBytes = AlignUp64(pixelBytes, BTS_DATA_BLOCK_BYTES);
  const uint64_t rowTileBytes = AlignUp64(
      static_cast<uint64_t>(outWidth) * ubPixelBytes, BTS_DATA_BLOCK_BYTES);
  if (rowTileBytes <= usableUb) {
    return outWidth;
  }

  uint32_t maxTileOW =
      static_cast<uint32_t>(std::max<uint64_t>(1, usableUb / ubPixelBytes));
  maxTileOW = std::min(maxTileOW, outWidth);

  const uint32_t alignOW = CalcGmAlignOW(pixelBytes);
  return ChooseMaxTileOW(outWidth, std::max<uint32_t>(1, maxTileOW), alignOW);
}

struct RowPackEstimate {
  uint64_t totalBytes = 0;
  uint32_t paddedElems = 0;
  uint32_t compactElems = 0;
  uint32_t compactOffsetElems = 0;
  uint32_t tmpOffsetBytes = 0;
  uint32_t tmpBytes = 0;
};

uint32_t RowPackOWUnit(uint32_t blockSize, uint32_t depthBytes) {
  const uint32_t g = GcdU32(depthBytes, BTS_DATA_BLOCK_BYTES);
  const uint32_t pixelsFor32B = BTS_DATA_BLOCK_BYTES / std::max<uint32_t>(1, g);
  return std::max<uint32_t>(1, blockSize * pixelsFor32B);
}

uint64_t RowPackTotalBytes(uint64_t paddedBytes, uint64_t compactBytes,
                           uint32_t tmpBytes, uint32_t dtypeSize,
                           RowPackEstimate *est) {
  const uint64_t compactOffsetBytes =
      AlignUp64(paddedBytes, BTS_DATA_BLOCK_BYTES);
  const uint64_t tmpOffsetBytes =
      AlignUp64(compactOffsetBytes + compactBytes, BTS_DATA_BLOCK_BYTES);
  const uint64_t totalBytes =
      AlignUp64(tmpOffsetBytes + tmpBytes, BTS_DATA_BLOCK_BYTES);
  if (est != nullptr) {
    est->paddedElems = static_cast<uint32_t>(paddedBytes / dtypeSize);
    est->compactElems = static_cast<uint32_t>(compactBytes / dtypeSize);
    est->compactOffsetElems =
        static_cast<uint32_t>(compactOffsetBytes / dtypeSize);
    est->tmpOffsetBytes = static_cast<uint32_t>(tmpOffsetBytes);
    est->tmpBytes = tmpBytes;
    est->totalBytes = totalBytes;
  }
  return totalBytes;
}

bool EstimateRowPackBytes(uint32_t tileOW, uint32_t tileOH, uint32_t blockSize,
                          uint32_t depth, uint32_t dtypeSize,
                          uint64_t usableUb, RowPackEstimate &est) {
  if (tileOW == 0 || tileOH == 0 || blockSize == 0 || depth == 0 ||
      dtypeSize == 0) {
    return false;
  }

  const uint32_t depthBytes = depth * dtypeSize;
  if (depthBytes == 0 || depthBytes > BTS_MAX_EXT_BLOCK_LEN) {
    return false;
  }
  if (CeilDiv(tileOW, blockSize) > BTS_MAX_BLOCK_COUNT) {
    return false;
  }

  const uint32_t paddedDepthBytes =
      AlignUpU32(depthBytes, BTS_DATA_BLOCK_BYTES);
  const uint32_t paddedDepthElems = paddedDepthBytes / dtypeSize;
  if (paddedDepthElems <= depth) {
    return false;
  }

  const uint32_t rightPadElems = paddedDepthElems - depth;
  if (static_cast<uint64_t>(rightPadElems) * dtypeSize >= BTS_DATA_BLOCK_BYTES) {
    return false;
  }

  const uint64_t logicalRows = static_cast<uint64_t>(tileOH) * tileOW;
  if (logicalRows == 0 || logicalRows > BTS_MAX_UINT16) {
    return false;
  }

  const uint64_t paddedBytes = logicalRows * paddedDepthBytes;
  const uint64_t compactBytes = logicalRows * depthBytes;
  if (paddedBytes > BTS_MAX_UINT32 || compactBytes > BTS_MAX_UINT32) {
    return false;
  }

  RowPackEstimate candidate;
  const uint64_t total =
      RowPackTotalBytes(paddedBytes, compactBytes, 0, dtypeSize, &candidate);
  if (total > usableUb) {
    return false;
  }

  est = candidate;
  return true;
}

uint32_t ChooseRowPackTileOW(uint32_t outWidth, uint32_t blockSize,
                             uint32_t depth, uint32_t dtypeSize,
                             uint64_t usableUb, RowPackEstimate &bestEst) {
  if (outWidth == 0 || blockSize == 0 || depth == 0 || dtypeSize == 0) {
    return 0;
  }

  const uint32_t depthBytes = depth * dtypeSize;
  const uint32_t unit = RowPackOWUnit(blockSize, depthBytes);
  uint32_t best = 0;
  RowPackEstimate est;

  if (EstimateRowPackBytes(outWidth, 1, blockSize, depth, dtypeSize, usableUb,
                           est)) {
    bestEst = est;
    return outWidth;
  }

  if (unit <= outWidth) {
    for (uint32_t ow = (outWidth / unit) * unit; ow >= unit; ow -= unit) {
      if (outWidth % ow == 0 &&
          EstimateRowPackBytes(ow, 1, blockSize, depth, dtypeSize, usableUb,
                               est)) {
        bestEst = est;
        return ow;
      }
      if (ow == unit) {
        break;
      }
    }

    for (uint32_t ow = (outWidth / unit) * unit; ow >= unit; ow -= unit) {
      if (EstimateRowPackBytes(ow, 1, blockSize, depth, dtypeSize, usableUb,
                               est)) {
        bestEst = est;
        return ow;
      }
      if (ow == unit) {
        break;
      }
    }
  }

  uint32_t lo = 1;
  uint32_t hi = outWidth;
  while (lo <= hi) {
    const uint32_t mid = lo + (hi - lo) / 2;
    if (EstimateRowPackBytes(mid, 1, blockSize, depth, dtypeSize, usableUb,
                             est)) {
      best = mid;
      bestEst = est;
      lo = mid + 1;
    } else {
      if (mid == 0) {
        break;
      }
      hi = mid - 1;
    }
  }

  return best;
}

uint32_t ChooseRowPackTileOH(uint32_t outHeight, uint32_t tileOW,
                             uint32_t blockSize, uint32_t depth,
                             uint32_t dtypeSize, uint64_t usableUb,
                             RowPackEstimate &bestEst) {
  if (outHeight == 0 || tileOW == 0) {
    return 0;
  }

  uint32_t best = 1;
  RowPackEstimate est;
  const uint32_t cap =
      std::min<uint32_t>(outHeight, TunableU32("BTS_MAX_ROW_TILE",
                                               BTS_MAX_ROW_TILE));
  for (uint32_t rows = 1; rows <= cap; ++rows) {
    if (EstimateRowPackBytes(tileOW, rows, blockSize, depth, dtypeSize,
                             usableUb, est)) {
      best = rows;
      bestEst = est;
    } else {
      break;
    }
  }
  return best;
}

uint64_t EstimateTileBytesAligned(uint32_t tileOW, uint32_t tileOH,
                                  uint64_t pixelBytes) {
  return static_cast<uint64_t>(tileOH) *
         AlignUp64(static_cast<uint64_t>(tileOW) * pixelBytes,
                   BTS_DATA_BLOCK_BYTES);
}

uint64_t Estimate32BCopyOutDmaBytes(uint32_t tileOW, uint32_t tileOH,
                                    uint32_t tileDepth, uint32_t dtypeSize) {
  return static_cast<uint64_t>(tileOW) * tileOH * tileDepth * dtypeSize;
}

uint64_t Estimate32BCopyInLaneDmaBytes(uint32_t tileOW, uint32_t tileDepth,
                                       uint32_t dtypeSize, uint32_t blockSize) {
  if (tileOW == 0 || tileDepth == 0 || dtypeSize == 0 || blockSize == 0) {
    return 0;
  }
  const uint64_t depthTileBytes = static_cast<uint64_t>(tileDepth) * dtypeSize;
  const uint64_t countPerLane = CeilDiv(tileOW, blockSize);
  return countPerLane * depthTileBytes;
}

bool Meets32BMinDmaBytes(uint32_t tileOW, uint32_t tileOH, uint32_t tileDepth,
                         uint32_t dtypeSize, uint32_t blockSize) {
  const uint32_t minDmaBytes =
      TunableU32("BTS_GM_MIN_DMA_BYTES", BTS_GM_MIN_DMA_BYTES);
  return Estimate32BCopyOutDmaBytes(tileOW, tileOH, tileDepth, dtypeSize) >=
             minDmaBytes ||
         Estimate32BCopyInLaneDmaBytes(tileOW, tileDepth, dtypeSize,
                                       blockSize) >= minDmaBytes;
}

uint32_t MaxTileOHWithinUb(uint32_t tileOW, uint64_t depthTileBytes,
                           uint32_t outHeight, uint64_t usableUb) {
  if (tileOW == 0 || outHeight == 0) {
    return 1;
  }
  uint32_t maxRows = 1;
  const uint32_t cap =
      std::min<uint32_t>(outHeight, TunableU32("BTS_MAX_ROW_TILE",
                                               BTS_MAX_ROW_TILE));
  for (uint32_t rows = 1; rows <= cap; ++rows) {
    if (EstimateTileBytesAligned(tileOW, rows, depthTileBytes) <= usableUb) {
      maxRows = rows;
    } else {
      break;
    }
  }
  return maxRows;
}

uint32_t MinTileOHFor32BCopyOutDma(uint32_t tileOW, uint32_t tileDepth,
                                  uint32_t dtypeSize, uint32_t maxRows) {
  if (maxRows == 0) {
    return 1;
  }
  for (uint32_t rows = 1; rows <= maxRows; ++rows) {
    if (Estimate32BCopyOutDmaBytes(tileOW, rows, tileDepth, dtypeSize) >=
        TunableU32("BTS_GM_MIN_DMA_BYTES", BTS_GM_MIN_DMA_BYTES)) {
      return rows;
    }
  }
  return maxRows;
}

uint32_t ChooseTileOH32BAligned(uint32_t outHeight, uint32_t tileOW,
                                uint32_t tileDepth, uint32_t dtypeSize,
                                uint32_t blockSize, uint64_t depthTileBytes,
                                uint64_t usableUb) {
  if (outHeight == 0 || tileOW == 0) {
    return 1;
  }

  const uint32_t maxRows =
      MaxTileOHWithinUb(tileOW, depthTileBytes, outHeight, usableUb);
  const uint32_t minRowsForDma =
      MinTileOHFor32BCopyOutDma(tileOW, tileDepth, dtypeSize, maxRows);

  if (Meets32BMinDmaBytes(tileOW, minRowsForDma, tileDepth, dtypeSize,
                          blockSize)) {
    return std::max<uint32_t>(1, minRowsForDma);
  }

  // CopyOut 仍不足 16KB 时尽量用满 UB 行数；CopyIn lane 达标也可接受。
  for (uint32_t rows = maxRows; rows >= 1; --rows) {
    if (Meets32BMinDmaBytes(tileOW, rows, tileDepth, dtypeSize, blockSize)) {
      return rows;
    }
  }
  return std::max<uint32_t>(1, maxRows);
}

bool Fit32BTilesToUb(uint32_t &tileOW, uint32_t &tileOH, uint32_t &tileDepth,
                     uint32_t &tilesPerRow, uint32_t &depthTiles, uint32_t depth,
                     uint32_t outWidth, uint32_t outHeight, uint32_t dtypeSize,
                     uint32_t blockSize, uint64_t usableUb) {
  if (tileOW == 0 || tileOH == 0 || tileDepth == 0 || depth == 0 ||
      outWidth == 0 || outHeight == 0 || dtypeSize == 0 || blockSize == 0) {
    return false;
  }

  const uint32_t minDepthTile =
      std::max<uint32_t>(1, BTS_DATA_BLOCK_BYTES / dtypeSize);

  auto depthTileBytes64 = [&](uint32_t tileD) -> uint64_t {
    return static_cast<uint64_t>(tileD) * dtypeSize;
  };

  auto tileBytes64 = [&](uint32_t ow, uint32_t oh, uint32_t tileD) -> uint64_t {
    return EstimateTileBytesAligned(ow, oh, depthTileBytes64(tileD));
  };

  auto tileParamOk = [&](uint32_t ow, uint32_t tileD) -> bool {
    return AlignedTileOWParamOk(blockSize, ow, depthTileBytes64(tileD));
  };

  auto meetsMinDma = [&](uint32_t ow, uint32_t oh, uint32_t tileD) -> bool {
    return Meets32BMinDmaBytes(ow, oh, tileD, dtypeSize, blockSize);
  };

  auto copyOutDma64 = [&](uint32_t ow, uint32_t oh, uint32_t tileD) -> uint64_t {
    return Estimate32BCopyOutDmaBytes(ow, oh, tileD, dtypeSize);
  };

  auto snapTileOWBlock = [&](uint32_t ow) -> uint32_t {
    if (ow >= outWidth || blockSize <= 1 || ow < blockSize) {
      return ow;
    }
    const uint32_t rounded = (ow / blockSize) * blockSize;
    return rounded > 0 ? rounded : ow;
  };

  auto tryGrowTileOW = [&]() {
    uint32_t lo = tileOW + 1;
    uint32_t hi = outWidth;
    uint32_t best = tileOW;
    while (lo <= hi) {
      const uint32_t mid = lo + (hi - lo) / 2;
      if (tileBytes64(mid, tileOH, tileDepth) <= usableUb &&
          tileParamOk(mid, tileDepth)) {
        best = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    if (best > tileOW) {
      best = snapTileOWBlock(best);
      if (best > tileOW && best <= outWidth &&
          tileBytes64(best, tileOH, tileDepth) <= usableUb &&
          tileParamOk(best, tileDepth)) {
        tileOW = best;
        tilesPerRow = CeilDiv(outWidth, tileOW);
      }
    }
  };

  auto tryGrowTileOH = [&]() {
    const uint32_t cap =
        std::min<uint32_t>(outHeight, TunableU32("BTS_MAX_ROW_TILE",
                                                 BTS_MAX_ROW_TILE));
    const uint32_t targetRows =
        MinTileOHFor32BCopyOutDma(tileOW, tileDepth, dtypeSize, cap);
    uint32_t best = tileOH;
    if (targetRows > best && tileBytes64(tileOW, targetRows, tileDepth) <= usableUb) {
      best = targetRows;
    }
    for (uint32_t rows = best + 1; rows <= cap; ++rows) {
      if (tileBytes64(tileOW, rows, tileDepth) <= usableUb) {
        if (copyOutDma64(tileOW, rows, tileDepth) >= copyOutDma64(tileOW, best, tileDepth)) {
          best = rows;
        }
      } else {
        break;
      }
    }
    tileOH = best;
  };

  auto tryGrowTileOHForMinDma = [&]() {
    const uint32_t cap =
        MaxTileOHWithinUb(tileOW, depthTileBytes64(tileDepth), outHeight, usableUb);
    if (tileOH >= cap) {
      return;
    }
    const uint32_t targetRows =
        MinTileOHFor32BCopyOutDma(tileOW, tileDepth, dtypeSize, cap);
    if (targetRows > tileOH &&
        tileBytes64(tileOW, targetRows, tileDepth) <= usableUb) {
      tileOH = targetRows;
    }
  };

  auto tryCoalesceDepth = [&]() {
    const uint32_t fillDenom =
        std::max<uint32_t>(1, TunableU32("BTS_UB_FILL_DENOM",
                                         BTS_UB_FILL_DENOM));
    const uint64_t targetBytes = std::max<uint64_t>(
        TunableU32("BTS_GM_MIN_DMA_BYTES", BTS_GM_MIN_DMA_BYTES),
        usableUb * static_cast<uint64_t>(
                       TunableU32("BTS_UB_FILL_NUMER", BTS_UB_FILL_NUMER)) /
            fillDenom);
    while (depthTiles > 1 && tileDepth < depth &&
           copyOutDma64(tileOW, tileOH, tileDepth) < targetBytes &&
           !meetsMinDma(tileOW, tileOH, tileDepth)) {
      uint32_t merged = std::min(depth, tileDepth * 2);
      merged = Largest32BAlignedDivisorDepth(merged, depth, dtypeSize);
      merged = std::min(merged, depth);
      if (merged == tileDepth) {
        break;
      }
      if (tileBytes64(tileOW, tileOH, merged) > usableUb ||
          !tileParamOk(tileOW, merged)) {
        break;
      }
      tileDepth = merged;
      depthTiles = CeilDiv(depth, tileDepth);
      tryGrowTileOW();
      tryGrowTileOH();
    }
  };

  uint64_t bytes = tileBytes64(tileOW, tileOH, tileDepth);
  while (bytes > usableUb && tileDepth > minDepthTile) {
    const uint32_t nextDepth = FloorTo32BDepthElems(
        std::max(minDepthTile, tileDepth / 2), dtypeSize);
    tileDepth = std::max(minDepthTile, nextDepth);
    depthTiles = CeilDiv(depth, tileDepth);
    bytes = tileBytes64(tileOW, tileOH, tileDepth);
  }
  while (bytes > usableUb && tileOH > 1) {
    --tileOH;
    bytes = tileBytes64(tileOW, tileOH, tileDepth);
  }
  while (bytes > usableUb && tileOW > 1) {
    --tileOW;
    tilesPerRow = CeilDiv(outWidth, tileOW);
    bytes = tileBytes64(tileOW, tileOH, tileDepth);
  }
  if (bytes > usableUb || !tileParamOk(tileOW, tileDepth)) {
    return false;
  }

  tryGrowTileOW();
  tryCoalesceDepth();
  tryGrowTileOHForMinDma();
  tryGrowTileOW();
  tryGrowTileOHForMinDma();

  while (!meetsMinDma(tileOW, tileOH, tileDepth)) {
    bool grew = false;
    const uint32_t rowCap =
        MaxTileOHWithinUb(tileOW, depthTileBytes64(tileDepth), outHeight, usableUb);
    const uint32_t targetRows =
        MinTileOHFor32BCopyOutDma(tileOW, tileDepth, dtypeSize, rowCap);
    if (tileOH < targetRows &&
        tileBytes64(tileOW, targetRows, tileDepth) <= usableUb) {
      tileOH = targetRows;
      grew = true;
    } else if (tileOH < rowCap &&
               tileBytes64(tileOW, tileOH + 1, tileDepth) <= usableUb) {
      ++tileOH;
      grew = true;
    } else {
      uint32_t lo = tileOW + 1;
      uint32_t hi = outWidth;
      uint32_t best = tileOW;
      while (lo <= hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (tileBytes64(mid, tileOH, tileDepth) <= usableUb &&
            tileParamOk(mid, tileDepth)) {
          best = mid;
          lo = mid + 1;
        } else {
          hi = mid - 1;
        }
      }
      if (best > tileOW) {
        best = snapTileOWBlock(best);
        if (best > tileOW && tileBytes64(best, tileOH, tileDepth) <= usableUb &&
            tileParamOk(best, tileDepth)) {
          tileOW = best;
          tilesPerRow = CeilDiv(outWidth, tileOW);
          grew = true;
        }
      }
    }
    if (!grew) {
      break;
    }
  }

  bytes = tileBytes64(tileOW, tileOH, tileDepth);
  return bytes <= usableUb && bytes <= BTS_MAX_UINT32 &&
         tileParamOk(tileOW, tileDepth);
}

bool Fit512BTilesToUb(uint32_t &tileOW, uint32_t &tileDepth, uint32_t &tilesPerRow,
                      uint32_t &depthTiles, uint32_t depth, uint32_t outWidth,
                      uint32_t dtypeSize, uint32_t blockSize, uint64_t usableUb) {
  const uint32_t ubBlockElements = BTS_DATA_BLOCK_BYTES / dtypeSize;
  const uint32_t minDepthTile = std::max<uint32_t>(1, ubBlockElements);

  auto tileChunkBytes64 = [&](uint32_t ow, uint32_t tileD) -> uint64_t {
    return AlignUp64(static_cast<uint64_t>(ow) * tileD * dtypeSize,
                     BTS_DATA_BLOCK_BYTES);
  };

  auto tryGrowTileOW = [&]() {
    const uint32_t maxOW = Calc512BMaxTileOW(outWidth, depth, tileDepth, dtypeSize,
                                             blockSize, usableUb);
    if (tileOW >= maxOW || tileOW >= outWidth) {
      return;
    }
    const uint32_t target =
        Choose512BTileOW(outWidth, depth, tileDepth, dtypeSize, blockSize, usableUb);
    if (target > tileOW && tileChunkBytes64(target, tileDepth) <= usableUb) {
      tileOW = target;
      tilesPerRow = CeilDiv(outWidth, tileOW);
    }
  };

  auto tryCoalesceDepth = [&]() {
    const uint32_t fillDenom =
        std::max<uint32_t>(1, TunableU32("BTS_UB_FILL_DENOM",
                                         BTS_UB_FILL_DENOM));
    uint64_t targetBytes = std::max<uint64_t>(
        TunableU32("BTS_GM_MIN_DMA_BYTES", BTS_GM_MIN_DMA_BYTES),
        usableUb * static_cast<uint64_t>(
                       TunableU32("BTS_UB_FILL_NUMER", BTS_UB_FILL_NUMER)) /
            fillDenom);
    const uint64_t chunkBytes = tileChunkBytes64(tileOW, tileDepth);
    if (chunkBytes > 0 && chunkBytes < BTS_GM_ALIGN_BYTES) {
      const uint64_t chunks512 =
          (BTS_GM_ALIGN_BYTES + chunkBytes - 1) / chunkBytes;
      targetBytes = std::max(targetBytes, chunks512 * chunkBytes);
    }
    while (depthTiles > 1 && tileDepth < depth &&
           tileChunkBytes64(tileOW, tileDepth) < targetBytes) {
      uint32_t merged = std::min(depth, tileDepth * 2);
      merged = Largest512BAlignedDivisorDepth(merged, depth, dtypeSize);
      merged = std::min(merged, depth);
      if (merged == tileDepth) {
        break;
      }
      if (tileChunkBytes64(tileOW, merged) > usableUb) {
        break;
      }
      tileDepth = merged;
      depthTiles = CeilDiv(depth, tileDepth);
      tryGrowTileOW();
    }
  };

  const uint32_t maxTileOW =
      Calc512BMaxTileOW(outWidth, depth, tileDepth, dtypeSize, blockSize, usableUb);
  if (tileOW >= outWidth || tileChunkBytes64(tileOW, tileDepth) > usableUb) {
    tileOW = Choose512BTileOW(outWidth, depth, tileDepth, dtypeSize, blockSize, usableUb);
    tileOW = std::min(tileOW, maxTileOW);
    tilesPerRow = CeilDiv(outWidth, tileOW);
  }

  uint64_t tileBytes64 = tileChunkBytes64(tileOW, tileDepth);
  while (tileBytes64 > usableUb && tileDepth > minDepthTile) {
    const uint32_t nextDepth = FloorTo512BDepthElems(
        std::max(minDepthTile, tileDepth / 2), dtypeSize);
    tileDepth = std::max(minDepthTile, nextDepth);
    depthTiles = CeilDiv(depth, tileDepth);
    tileBytes64 = tileChunkBytes64(tileOW, tileDepth);
  }
  if (tileBytes64 > usableUb) {
    return false;
  }

  tryGrowTileOW();
  tryCoalesceDepth();
  tryGrowTileOW();
  tileBytes64 = tileChunkBytes64(tileOW, tileDepth);
  return tileBytes64 <= usableUb && tileBytes64 <= BTS_MAX_UINT32;
}

uint64_t EstimateSpatialTaskCount(uint32_t outBatch, uint32_t outHeight,
                                  uint32_t outWidth, uint32_t tileOW,
                                  uint32_t tileOH, uint32_t depthTiles) {
  return static_cast<uint64_t>(outBatch) * CeilDiv(outHeight, tileOH) *
         CeilDiv(outWidth, tileOW) * depthTiles;
}

bool ApplyFixedPathTuning(BatchToSpaceCopyPath path, uint32_t outBatch,
                          uint32_t outHeight, uint32_t outWidth,
                          uint32_t depth, uint32_t dtypeSize,
                          uint32_t blockSize, uint64_t usableUb,
                          uint32_t &tileOW, uint32_t &tileOH,
                          uint32_t &tileDepth, uint32_t &tileElemAligned,
                          uint32_t &totalTasks) {
  if (!TuningEnabled()) {
    return true;
  }

  uint32_t requested = 0;
  if (ReadTuningU32("BTS_TILE_OW", requested)) {
    tileOW = requested;
  }
  if (ReadTuningU32("BTS_TILE_OH", requested)) {
    tileOH = requested;
  }
  if (ReadTuningU32("BTS_TILE_DEPTH", requested)) {
    tileDepth = requested;
  }

  if (tileOW == 0 || tileOW > outWidth || tileOH == 0 ||
      tileOH > outHeight || tileDepth == 0 || tileDepth > depth) {
    return false;
  }

  if (path != BTS_PATH_DEPTH_32B_ALIGNED &&
      path != BTS_PATH_DEPTH_512B_ALIGNED &&
      path != BTS_PATH_DEPTH_TINY_SPATIAL &&
      path != BTS_PATH_ROW_PLANE_FP32 && path != BTS_PATH_DEPTH_SPLIT_TINY) {
    return true;
  }

  const uint64_t depthTileBytes =
      static_cast<uint64_t>(tileDepth) * dtypeSize;
  if ((depthTileBytes % BTS_DATA_BLOCK_BYTES) != 0 ||
      !AlignedTileOWParamOk(blockSize, tileOW, depthTileBytes)) {
    return false;
  }
  if ((path == BTS_PATH_DEPTH_512B_ALIGNED ||
       path == BTS_PATH_DEPTH_TINY_SPATIAL ||
       path == BTS_PATH_DEPTH_SPLIT_TINY) &&
      (depthTileBytes % BTS_GM_ALIGN_BYTES) != 0) {
    return false;
  }

  const uint64_t tileBytes =
      EstimateTileBytesAligned(tileOW, tileOH, depthTileBytes);
  if (tileBytes == 0 || tileBytes > usableUb || tileBytes > BTS_MAX_UINT32) {
    return false;
  }
  const uint64_t tasks64 =
      EstimateSpatialTaskCount(outBatch, outHeight, outWidth, tileOW, tileOH,
                               CeilDiv(depth, tileDepth));
  if (tasks64 == 0 || tasks64 > BTS_MAX_UINT32) {
    return false;
  }

  tileElemAligned = static_cast<uint32_t>(tileBytes / dtypeSize);
  totalTasks = static_cast<uint32_t>(tasks64);
  return true;
}

uint32_t ChooseTileOH(uint32_t outBatch, uint32_t outHeight, uint32_t outWidth,
                      uint32_t tileOW, uint32_t depthTiles, uint64_t pixelBytes,
                      uint64_t usableUb, bool depthFallback, uint32_t numCores) {
  if (outHeight == 0 || tileOW == 0) {
    return 1;
  }

  uint32_t maxRows = 1;
  const uint32_t cap =
      std::min<uint32_t>(outHeight, TunableU32("BTS_MAX_ROW_TILE",
                                               BTS_MAX_ROW_TILE));
  for (uint32_t rows = 1; rows <= cap; ++rows) {
    if (EstimateTileBytesAligned(tileOW, rows, pixelBytes) <= usableUb) {
      maxRows = rows;
    } else {
      break;
    }
  }

  if (depthFallback) {
    return maxRows >= 2 ? 2 : 1;
  }

  const uint32_t targetTasks = std::min<uint32_t>(
      std::max<uint32_t>(1, numCores),
      TunableU32("BTS_MIN_TASKS", BTS_MIN_TASKS));
  uint32_t rows = maxRows;
  while (rows > 1 && EstimateSpatialTaskCount(outBatch, outHeight, outWidth,
                                              tileOW, rows, depthTiles) <
                          targetTasks) {
    --rows;
  }
  return std::max<uint32_t>(1, rows);
}

uint32_t ChooseTileOWBlockMoveAlign(uint32_t outWidth, uint32_t blockSize,
                                    uint32_t cropLeft, uint64_t tileColumnBytes,
                                    uint64_t usableUb, uint32_t owAlignCols) {
  if (outWidth == 0 || blockSize == 0 || tileColumnBytes == 0) {
    return 1;
  }

  const uint64_t maxOWByUb =
      usableUb / std::max<uint64_t>(1, tileColumnBytes);
  uint64_t maxOW = std::min(maxOWByUb, static_cast<uint64_t>(outWidth));
  if (maxOW == 0) {
    return 1;
  }
  if (maxOW >= outWidth) {
    return outWidth;
  }

  const uint64_t ubFactorAlign = blockSize;
  const uint64_t leftAlign = cropLeft;
  const uint64_t dimVal = outWidth;

  const uint64_t lastLeftAlign = leftAlign % ubFactorAlign;
  uint64_t head = 0;
  uint64_t tail = 0;
  if (lastLeftAlign + dimVal > ubFactorAlign) {
    head = lastLeftAlign == 0 ? 0 : ubFactorAlign - lastLeftAlign;
    tail = (lastLeftAlign + dimVal) % ubFactorAlign;
  } else {
    head = dimVal;
  }
  const uint64_t middle =
      dimVal > (head + tail) ? dimVal - (head + tail) : 0;

  const uint64_t restFactor = std::min(maxOW, dimVal);
  uint64_t ubFactor = 1;
  if (middle > 0) {
    if (restFactor >= ubFactorAlign) {
      ubFactor = FloorAlign64(restFactor, ubFactorAlign);
    } else {
      ubFactor = std::max<uint64_t>(1, restFactor);
    }
    ubFactor = std::max<uint64_t>(1, std::min(ubFactor, maxOW));
  } else if (head > 0) {
    ubFactor = std::max<uint64_t>(1, std::min(head, maxOW));
  } else {
    ubFactor = std::max<uint64_t>(1, std::min(maxOW, dimVal));
  }

  uint32_t tileOW =
      static_cast<uint32_t>(std::max<uint64_t>(1, std::min(ubFactor, dimVal)));
  if (owAlignCols > 1 && tileOW < outWidth) {
    tileOW = SnapTileOWDown(tileOW, owAlignCols, outWidth);
  }

  while (tileOW > 1 &&
         !AlignedTileOWParamOk(blockSize, tileOW, tileColumnBytes)) {
    --tileOW;
  }
  if (!AlignedTileOWParamOk(blockSize, tileOW, tileColumnBytes)) {
    return 1;
  }
  return std::max<uint32_t>(1, std::min(tileOW, outWidth));
}

uint32_t ChooseTileOW512BMoveAlign(uint32_t outWidth, uint32_t blockSize,
                                   uint32_t cropLeft, uint64_t tileColumnBytes,
                                   uint64_t gmRowBytes, uint64_t usableUb) {
  const uint32_t alignOW = CalcGmAlignOW(gmRowBytes);
  uint32_t tileOW = ChooseTileOWBlockMoveAlign(outWidth, blockSize, cropLeft,
                                               tileColumnBytes, usableUb,
                                               alignOW);
  if (tileOW < outWidth) {
    tileOW = SnapTileOWDivideOutWidth(tileOW, outWidth, alignOW);
    while (tileOW > 1 &&
           !AlignedTileOWParamOk(blockSize, tileOW, tileColumnBytes)) {
      --tileOW;
    }
  }
  return std::max<uint32_t>(1, std::min(tileOW, outWidth));
}

uint32_t ChooseTileOW32BAligned(uint32_t outWidth, uint32_t blockSize,
                             uint64_t depthTileBytes, uint64_t usableUb,
                             bool depthFallback) {
  (void)depthFallback;
  if (outWidth == 0) {
    return 1;
  }

  uint32_t lo = 1;
  uint32_t hi = outWidth;
  uint32_t best = 1;
  while (lo <= hi) {
    const uint32_t mid = lo + (hi - lo) / 2;
    const uint64_t bytes = EstimateTileBytesAligned(mid, 1, depthTileBytes);
    if (bytes <= usableUb &&
        AlignedTileOWParamOk(blockSize, mid, depthTileBytes)) {
      best = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }

  if (best < outWidth && blockSize > 1) {
    const uint32_t alignOW = blockSize;
    if (best >= alignOW) {
      const uint32_t rounded = (best / alignOW) * alignOW;
      if (rounded > 0) {
        best = rounded;
      }
    }
  }
  return std::max<uint32_t>(1, std::min(best, outWidth));
}

uint32_t ChooseActiveCores(uint32_t totalTasks, uint32_t numCores,
                           uint64_t outputBytes, BatchToSpaceCopyPath path) {
  if (totalTasks == 0) {
    return 1;
  }

  if (path == BTS_PATH_DEPTH_SPLIT_TINY) {
    return std::min<uint32_t>(8, std::min(numCores, totalTasks));
  }

  uint32_t limit =
      std::min<uint32_t>(std::max<uint32_t>(1, numCores), totalTasks);

  if (outputBytes <=
      TunableU32("BTS_SMALL_OUTPUT_BYTES", BTS_SMALL_OUTPUT_BYTES)) {
    limit = std::min<uint32_t>(
        limit, totalTasks >= 64 ? 8 : (totalTasks >= 16 ? 4 : totalTasks));
  } else if (outputBytes <=
             TunableU32("BTS_MEDIUM_OUTPUT_BYTES",
                        BTS_MEDIUM_OUTPUT_BYTES)) {
    limit = std::min<uint32_t>(limit, 8);
  }

  if ((path == BTS_PATH_DEPTH_32B_ALIGNED ||
       path == BTS_PATH_DEPTH_512B_ALIGNED ||
       path == BTS_PATH_DEPTH_TINY_SPATIAL ||
       path == BTS_PATH_ROW_PLANE_FP32) &&
      totalTasks <= limit * 2) {
    limit = std::min<uint32_t>(limit, CeilDiv(totalTasks, 2));
  }

  const uint32_t minActiveCores = std::min(
      TunableU32("BTS_MIN_ACTIVE_CORES", BTS_MIN_ACTIVE_CORES),
      std::max<uint32_t>(1, numCores));
  return std::min(numCores, std::max(limit, minActiveCores));
}

struct BatchToSpaceTilePlan {
  uint32_t tileOW = 1;
  uint32_t tileOH = 1;
  uint32_t tileDepth = 1;
  uint32_t tilesPerRow = 1;
  uint32_t depthTiles = 1;
  uint32_t totalTasks = 0;
  uint32_t tileElemAligned = 1;
  uint64_t tileBytes = 0;
  BatchToSpaceCopyPath path = BTS_PATH_DEPTH_32B_ALIGNED;
  bool valid = false;
};

bool BuildAlignedOWTilePlan(BatchToSpaceTilePlan &plan, uint32_t outBatch, uint32_t outHeight,
                            uint32_t outWidth, uint32_t depth, uint32_t dtypeSize,
                            uint32_t blockSize, uint64_t usableUb, uint32_t numCores) {
  (void)numCores;
  const uint64_t pixelBytes = static_cast<uint64_t>(depth) * dtypeSize;
  plan.tileDepth =
      ChooseAlignedDepthTileElems(depth, dtypeSize, pixelBytes, usableUb);
  plan.depthTiles = CeilDiv(depth, plan.tileDepth);
  const bool depthFallback = plan.tileDepth < depth;
  uint64_t depthTileBytes = static_cast<uint64_t>(plan.tileDepth) * dtypeSize;
  if (plan.tileDepth == 0 || depthTileBytes == 0 ||
      depthTileBytes > BTS_MAX_UINT32 ||
      !AlignedTileOWParamOk(blockSize, 1, depthTileBytes)) {
    plan.valid = false;
    return false;
  }

  plan.tileOW = ChooseTileOW32BAligned(outWidth, blockSize, depthTileBytes, usableUb,
                                       depthFallback);
  plan.tilesPerRow = CeilDiv(outWidth, plan.tileOW);
  plan.tileOH = ChooseTileOH32BAligned(outHeight, plan.tileOW, plan.tileDepth,
                                       dtypeSize, blockSize, depthTileBytes,
                                       usableUb);
  if (!Fit32BTilesToUb(plan.tileOW, plan.tileOH, plan.tileDepth, plan.tilesPerRow,
                       plan.depthTiles, depth, outWidth, outHeight, dtypeSize,
                       blockSize, usableUb)) {
    plan.valid = false;
    return false;
  }
  depthTileBytes = static_cast<uint64_t>(plan.tileDepth) * dtypeSize;
  plan.tileBytes =
      EstimateTileBytesAligned(plan.tileOW, plan.tileOH, depthTileBytes);
  if (plan.tileBytes > usableUb || plan.tileBytes > BTS_MAX_UINT32) {
    plan.valid = false;
    return false;
  }
  const uint64_t totalTasks64 =
      EstimateSpatialTaskCount(outBatch, outHeight, outWidth, plan.tileOW,
                               plan.tileOH, plan.depthTiles);
  if (totalTasks64 == 0 || totalTasks64 > BTS_MAX_UINT32) {
    plan.valid = false;
    return false;
  }
  plan.tileElemAligned =
      std::max<uint32_t>(1, static_cast<uint32_t>(plan.tileBytes / dtypeSize));
  plan.totalTasks = static_cast<uint32_t>(totalTasks64);
  plan.path = BTS_PATH_DEPTH_32B_ALIGNED;
  plan.valid = true;
  return true;
}

bool Build512BTilePlan(BatchToSpaceTilePlan &plan, uint32_t outBatch, uint32_t outHeight,
                       uint32_t outWidth, uint32_t depth, uint32_t dtypeSize,
                       uint32_t blockSize, uint64_t usableUb, uint32_t numCores,
                       uint32_t cropLeft) {
  const uint64_t pixelBytes = static_cast<uint64_t>(depth) * dtypeSize;
  plan.tileDepth = Choose512BDepthTileElems(depth, dtypeSize, pixelBytes, usableUb);
  plan.depthTiles = CeilDiv(depth, plan.tileDepth);
  const bool depthFallback = plan.tileDepth < depth;
  uint64_t depthTileBytes = static_cast<uint64_t>(plan.tileDepth) * dtypeSize;
  if (plan.tileDepth == 0 || depthTileBytes == 0 ||
      depthTileBytes > BTS_MAX_UINT32 ||
      !AlignedTileOWParamOk(blockSize, 1, depthTileBytes)) {
    plan.valid = false;
    return false;
  }

  plan.tileOW = ChooseTileOW512BMoveAlign(outWidth, blockSize, cropLeft, depthTileBytes,
                                          pixelBytes, usableUb);
  plan.tilesPerRow = CeilDiv(outWidth, plan.tileOW);
  if (!Fit512BTilesToUb(plan.tileOW, plan.tileDepth, plan.tilesPerRow, plan.depthTiles,
                        depth, outWidth, dtypeSize, blockSize, usableUb) ||
      !AlignedTileOWParamOk(blockSize, plan.tileOW, depthTileBytes)) {
    plan.valid = false;
    return false;
  }

  if (plan.tileDepth < depth) {
    plan.tileDepth = Largest512BAlignedDivisorDepth(plan.tileDepth, depth, dtypeSize);
    plan.depthTiles = CeilDiv(depth, plan.tileDepth);
    depthTileBytes = static_cast<uint64_t>(plan.tileDepth) * dtypeSize;
    if (!AlignedTileOWParamOk(blockSize, plan.tileOW, depthTileBytes)) {
      plan.valid = false;
      return false;
    }
  }

  plan.tileOH = ChooseTileOH(outBatch, outHeight, outWidth, plan.tileOW,
                             plan.depthTiles, depthTileBytes, usableUb,
                             depthFallback, numCores);
  plan.tileBytes =
      EstimateTileBytesAligned(plan.tileOW, plan.tileOH, depthTileBytes);
  if (plan.tileBytes > usableUb || plan.tileBytes > BTS_MAX_UINT32) {
    plan.valid = false;
    return false;
  }
  const uint64_t totalTasks64 =
      EstimateSpatialTaskCount(outBatch, outHeight, outWidth, plan.tileOW,
                               plan.tileOH, plan.depthTiles);
  if (totalTasks64 == 0 || totalTasks64 > BTS_MAX_UINT32) {
    plan.valid = false;
    return false;
  }
  plan.tileElemAligned =
      std::max<uint32_t>(1, static_cast<uint32_t>(plan.tileBytes / dtypeSize));
  plan.totalTasks = static_cast<uint32_t>(totalTasks64);
  plan.path = BTS_PATH_DEPTH_512B_ALIGNED;
  plan.valid = true;
  return true;
}

bool ValidateStaticShape(int64_t batch, int64_t height, int64_t width,
                         int64_t depth, int64_t blockSize, uint32_t cropTop,
                         uint32_t cropBottom, uint32_t cropLeft,
                         uint32_t cropRight) {
  if (batch <= 0 || height <= 0 || width <= 0 || depth <= 0 ||
      blockSize <= 0) {
    return false;
  }

  const uint64_t blockElems = static_cast<uint64_t>(blockSize) * blockSize;
  const uint64_t fullHeight = static_cast<uint64_t>(height) * blockSize;
  const uint64_t fullWidth = static_cast<uint64_t>(width) * blockSize;
  const uint64_t cropHeight = static_cast<uint64_t>(cropTop) + cropBottom;
  const uint64_t cropWidth = static_cast<uint64_t>(cropLeft) + cropRight;
  return static_cast<uint64_t>(batch) <= BTS_MAX_UINT32 &&
         static_cast<uint64_t>(height) <= BTS_MAX_UINT32 &&
         static_cast<uint64_t>(width) <= BTS_MAX_UINT32 &&
         static_cast<uint64_t>(depth) <= BTS_MAX_UINT32 &&
         static_cast<uint64_t>(blockSize) <= BTS_MAX_UINT32 &&
         blockElems != 0 && blockElems <= BTS_MAX_UINT32 &&
         fullHeight <= BTS_MAX_UINT32 && fullWidth <= BTS_MAX_UINT32 &&
         (static_cast<uint64_t>(batch) % blockElems) == 0 &&
         cropHeight < fullHeight && cropWidth < fullWidth;
}

uint64_t GetUsableUbPerBuffer(uint64_t ubSize) {
  const uint64_t usableUb = ubSize == 0 ? BTS_UB_FALLBACK_BYTES : ubSize;
  return usableUb / BTS_DOUBLE_BUFFER_NUM;
}

uint64_t GetUsableUbSingle(uint64_t ubSize) {
  return ubSize == 0 ? BTS_UB_FALLBACK_BYTES : ubSize;
}

uint64_t CeilDiv64(uint64_t x, uint64_t y) {
  return y == 0 ? 0 : (x + y - 1) / y;
}

uint32_t LcmU32(uint32_t a, uint32_t b) {
  const uint32_t g = GcdU32(a, b);
  return g == 0 ? 0 : (a / g) * b;
}

uint64_t Lcm64(uint64_t a, uint64_t b) {
  const uint64_t g = Gcd64(a, b);
  return g == 0 ? 0 : (a / g) * b;
}

// ---------------------------------------------------------------------------
// Feature extraction + algorithm dispatch.
//
// Shapes are classified by feature quantities (alignment class, aspect class,
// spatial tier, crop class, traffic tier) measured against hardware envelopes
// (32B UB block, 512B GM burst, UB capacity, DMA-traffic floor, core count).
// Exact tile parameters come from closed-form fits plus the strategy LUT
// below; when the fitted parameters coincide with a static kernel's
// compile-time constants (kStaticArms rows), the static kernel is emitted.
// ---------------------------------------------------------------------------

struct DispatchInput {
  uint32_t batch = 0;
  uint32_t height = 0;
  uint32_t width = 0;
  uint32_t depth = 0;
  uint32_t dtypeSize = 0;
  uint32_t blockSize = 0;
  uint32_t cropTop = 0;
  uint32_t cropBottom = 0;
  uint32_t cropLeft = 0;
  uint32_t cropRight = 0;
  uint32_t numCores = 0;
  uint64_t ubBytes = 0;
};

enum DispatchAlignClass : uint8_t {
  BTS_ALIGN_UNALIGNED = 0,
  BTS_ALIGN_32B = 1,
  BTS_ALIGN_512B = 2,
};

enum DispatchAspectClass : uint8_t {
  BTS_ASPECT_SQUARE = 0,
  BTS_ASPECT_WIDE = 1,
  BTS_ASPECT_TALL = 2,
};

enum DispatchSpatialTier : uint8_t {
  BTS_SPACE_ONE = 0,
  BTS_SPACE_TINY = 1,
  BTS_SPACE_SMALL = 2,
  BTS_SPACE_BIG = 3,
};

enum DispatchCropClass : uint8_t {
  BTS_CROP_ZERO = 0,
  BTS_CROP_SYM = 1,
  BTS_CROP_ARB = 2,
};

enum DispatchPrototype : uint8_t {
  BTS_PROTO_NONE = 0,
  BTS_PROTO_DEPTH_SPLIT = 1,
  BTS_PROTO_ROW_GATHER = 2,
  BTS_PROTO_H_FUSE = 3,
  BTS_PROTO_HALF_ROW_BIG = 4,
  BTS_PROTO_MC_BIG = 5,
  BTS_PROTO_ROW_PLANE = 6,
  BTS_PROTO_GENERIC = 7,
};

struct DispatchFeatures {
  uint32_t outBatch = 0;
  uint32_t outHeight = 0;
  uint32_t outWidth = 0;
  uint32_t depthBytes = 0;
  uint8_t log2DepthBytes = 0;
  DispatchAlignClass align = BTS_ALIGN_UNALIGNED;
  DispatchAspectClass aspect = BTS_ASPECT_SQUARE;
  DispatchSpatialTier spatial = BTS_SPACE_BIG;
  DispatchCropClass crop = BTS_CROP_ZERO;
  uint64_t trafficBytes = 0;
  uint64_t ubPerBuf = 0;
  uint64_t ubSingle = 0;
};

inline uint8_t DispatchLog2Floor(uint64_t v) {
  uint8_t r = 0;
  while (v > 1) {
    v >>= 1;
    ++r;
  }
  return r;
}

DispatchFeatures ExtractFeatures(const DispatchInput &in) {
  DispatchFeatures f;
  const uint32_t blockElems = in.blockSize * in.blockSize;
  f.outBatch = blockElems == 0 ? 0 : in.batch / blockElems;
  f.outHeight = in.height * in.blockSize - in.cropTop - in.cropBottom;
  f.outWidth = in.width * in.blockSize - in.cropLeft - in.cropRight;
  f.depthBytes = in.depth * in.dtypeSize;
  f.log2DepthBytes = DispatchLog2Floor(std::max<uint32_t>(1, f.depthBytes));
  f.align = (f.depthBytes % BTS_GM_ALIGN_BYTES) == 0 ? BTS_ALIGN_512B
            : (f.depthBytes % BTS_DATA_BLOCK_BYTES) == 0 ? BTS_ALIGN_32B
                                                         : BTS_ALIGN_UNALIGNED;
  if (static_cast<uint64_t>(f.outHeight) >=
      static_cast<uint64_t>(BTS_FP_HFUSE_ASPECT) * f.outWidth) {
    f.aspect = BTS_ASPECT_TALL;
  } else if (static_cast<uint64_t>(f.outWidth) >=
             static_cast<uint64_t>(BTS_FP_HFUSE_ASPECT) * f.outHeight) {
    f.aspect = BTS_ASPECT_WIDE;
  } else {
    f.aspect = BTS_ASPECT_SQUARE;
  }
  // Tiers: collapsed plane, a few pixels, up to one aspect-square (64x64),
  // and everything larger.
  constexpr uint64_t kSmallPlane =
      static_cast<uint64_t>(BTS_FP_HFUSE_ASPECT) * BTS_FP_HFUSE_ASPECT;
  const uint64_t spatial = static_cast<uint64_t>(f.outHeight) * f.outWidth;
  f.spatial = spatial <= 1 ? BTS_SPACE_ONE
              : spatial <= 16 ? BTS_SPACE_TINY
              : spatial <= kSmallPlane ? BTS_SPACE_SMALL
                                       : BTS_SPACE_BIG;
  const bool zeroCrop = in.cropTop == 0 && in.cropBottom == 0 &&
                        in.cropLeft == 0 && in.cropRight == 0;
  const bool symCrop = in.cropTop == in.cropBottom && in.cropLeft == in.cropRight;
  f.crop = zeroCrop ? BTS_CROP_ZERO : (symCrop ? BTS_CROP_SYM : BTS_CROP_ARB);
  f.trafficBytes = static_cast<uint64_t>(f.outBatch) * f.outHeight * f.outWidth *
                   f.depthBytes;
  f.ubPerBuf = GetUsableUbPerBuffer(in.ubBytes);
  f.ubSingle = GetUsableUbSingle(in.ubBytes);
  return f;
}

// Benefit-ordered prototype selection. Predicates use feature quantities and
// hardware envelopes only; the chain mirrors the relative priority of the
// tiling builders (depth-dominant first, unaligned-depth gather second, then
// the wide/tall fp16 movers, then the aligned plane assembler).
DispatchPrototype SelectPath(const DispatchInput &in, const DispatchFeatures &f) {
  const bool fp16 = in.dtypeSize == BTS_FP_DTYPE_BYTES;
  if (f.depthBytes >= BTS_LARGE_DEPTH_MIN_BYTES &&
      f.align != BTS_ALIGN_UNALIGNED && f.spatial <= BTS_SPACE_TINY) {
    return BTS_PROTO_DEPTH_SPLIT;
  }
  if (f.align == BTS_ALIGN_UNALIGNED && in.blockSize >= 2) {
    return BTS_PROTO_ROW_GATHER;
  }
  if (fp16 && f.crop == BTS_CROP_ZERO && f.align == BTS_ALIGN_32B &&
      f.aspect == BTS_ASPECT_TALL && f.depthBytes < BTS_FP_DEPTH_BURST_BYTES &&
      static_cast<uint64_t>(in.width) * f.depthBytes <= BTS_FP_DEPTH_BURST_BYTES) {
    return BTS_PROTO_H_FUSE;
  }
  if (fp16 && f.crop == BTS_CROP_ZERO && in.blockSize == 2 &&
      f.depthBytes == BTS_FP_DEPTH_BURST_BYTES && f.outBatch == 1 &&
      f.outWidth >= f.outHeight && f.outWidth >= BTS_W512_MIN_OUT_WIDTH &&
      f.trafficBytes >= BTS_W512_MIN_TRAFFIC_BYTES) {
    return BTS_PROTO_HALF_ROW_BIG;
  }
  if (fp16 && f.align == BTS_ALIGN_32B && in.blockSize >= 2 &&
      f.depthBytes < BTS_FP_DEPTH_BURST_BYTES &&
      in.depth <= BTS_FP_COPY_MASK_MAX_B16 && f.outBatch == 1 &&
      f.outWidth >= f.outHeight && f.trafficBytes >= BTS_FP_MIN_TRAFFIC_BYTES) {
    return BTS_PROTO_MC_BIG;
  }
  if (f.align != BTS_ALIGN_UNALIGNED && f.spatial <= BTS_SPACE_SMALL) {
    return BTS_PROTO_ROW_PLANE;
  }
  return BTS_PROTO_GENERIC;
}

// One row per static kernel: required derived quantities (the kernel's
// compile-time constants) plus the emission triple. Shape literals are
// constexpr table data, never predicate logic.
struct DispatchStaticArm {
  DispatchPrototype proto;
  uint32_t batch, height, width, depth;
  uint32_t dtypeSize, blockSize;
  uint32_t cropTop, cropBottom, cropLeft, cropRight;
  uint32_t minCores;
  uint64_t reqUbBytes;
  uint8_t ubBudget; // 0 = per-buffer, 1 = single, 2 = gather-index budget
  uint32_t mode;
  uint32_t variant;
  uint32_t blockDim;
  uint32_t tileOW; // 0 = mode-only header
};

constexpr DispatchStaticArm kStaticArms[] = {
    {BTS_PROTO_ROW_PLANE, 8, 28, 28, 128, 4, 2, 0, 0, 0, 0, 28,
     2ULL * 56 * 128 * 4, 0, BTS_TILING_MODE_ROW_PLANE_STRIDED_STATIC,
     BTS_STATIC_VARIANT_ROW_PLANE_STRIDED, 28, 0},
    {BTS_PROTO_DEPTH_SPLIT, 4, 1, 1, 16384, 2, 2, 0, 0, 0, 0, 8,
     2ULL * 2 * 2048 * 2, 0, BTS_TILING_MODE_DEPTH_SPLIT_TINY_STATIC,
     BTS_STATIC_VARIANT_DEPTH_SPLIT_TINY, 8, 0},
    {BTS_PROTO_DEPTH_SPLIT, 4, 2, 2, 4096, 2, 2, 0, 0, 0, 0, 8,
     4ULL * 4 * 512 * 2, 0, BTS_TILING_MODE_DEPTH_SPLIT_LARGE_STATIC,
     BTS_STATIC_VARIANT_DEPTH_SPLIT_LARGE, 8, 0},
    {BTS_PROTO_ROW_PLANE, 20, 4, 6, 32, 4, 2, 0, 0, 0, 0, 10,
     8ULL * 12 * 32 * 4, 0, BTS_TILING_MODE_ROW_PLANE_SMALL_STATIC,
     BTS_STATIC_VARIANT_ROW_PLANE_SMALL, 10, 0},
    {BTS_PROTO_ROW_PLANE, 16, 14, 14, 64, 4, 2, 0, 0, 0, 0, 20,
     6ULL * 28 * 64 * 4, 1, BTS_TILING_MODE_ROW_PLANE_COMPACT_STATIC,
     BTS_STATIC_VARIANT_ROW_PLANE_COMPACT, 20, 0},
    {BTS_PROTO_ROW_GATHER, 4, 10, 15, 5, 4, 2, 2, 1, 3, 1, 17,
     280ULL * 4 + 576, 1, BTS_TILING_MODE_ROW_GATHER_FP32_STATIC,
     BTS_STATIC_VARIANT_ROW_GATHER_FP32, 17, 0},
    {BTS_PROTO_ROW_GATHER, 4, 128, 128, 65, 2, 2, 1, 1, 1, 1, 40,
     2ULL * 16576 * 2 + 8256ULL * 4, 2, BTS_TILING_MODE_ROW_GATHER_WIDE_STATIC,
     BTS_STATIC_VARIANT_ROW_GATHER_WIDE, 40, 0},
    {BTS_PROTO_HALF_ROW_BIG, 4, 10, 512, 256, 2, 2, 0, 0, 0, 0, 40,
     128ULL * 256 * 2 * 2, 1, BTS_TILING_MODE_HALF_ROW_BIG_STATIC,
     BTS_STATIC_VARIANT_HALF_ROW_BIG, 40, 0},
    {BTS_PROTO_MC_BIG, 16, 10, 512, 64, 2, 4, 0, 0, 513, 0, 40,
     2ULL * 768 * 64 * 2, 1, BTS_TILING_MODE_MC_BIGTILE,
     BTS_STATIC_VARIANT_NONE, 40, 768},
    {BTS_PROTO_H_FUSE, 16, 1024, 6, 32, 2, 2, 0, 0, 0, 0, 40,
     2ULL * 32 * 12 * 32 * 2, 0, BTS_TILING_MODE_H_FUSE_STATIC,
     BTS_STATIC_VARIANT_H_FUSE, 40, 0},
};

int MatchStaticArm(const DispatchInput &in, const DispatchFeatures &f,
                   DispatchPrototype proto) {
  for (size_t i = 0; i < sizeof(kStaticArms) / sizeof(kStaticArms[0]); ++i) {
    const DispatchStaticArm &a = kStaticArms[i];
    if (a.proto != proto || a.batch != in.batch || a.height != in.height ||
        a.width != in.width || a.depth != in.depth ||
        a.dtypeSize != in.dtypeSize || a.blockSize != in.blockSize ||
        a.cropTop != in.cropTop || a.cropBottom != in.cropBottom ||
        a.cropLeft != in.cropLeft || a.cropRight != in.cropRight) {
      continue;
    }
    if (in.numCores < a.minCores) {
      continue;
    }
    const uint64_t budget =
        a.ubBudget == 0 ? f.ubPerBuf
        : a.ubBudget == 1
            ? f.ubSingle
            : std::min<uint64_t>(2ULL * f.ubPerBuf, BTS_FP_GI_IDX_BUDGET_BYTES);
    if (a.reqUbBytes > budget) {
      continue;
    }
    return static_cast<int>(i);
  }
  return -1;
}

// Strategy LUT: bucketed by (dtype, blockSize, log2 depthBytes, alignment,
// aspect). Values are tile-shaping strategy knobs, not shapes; closed-form
// fits below consume them. Generated by docs/solution9/gen_dispatch_lut.py.
struct DispatchLutValue {
  uint8_t proto;
  uint8_t coreStrategy; // 0 task-count, 1 row-aligned, 2 cap-8 depth split
};

inline uint32_t DispatchLutKey(uint8_t dt, uint8_t bs, uint8_t l2db, uint8_t al,
                               uint8_t as) {
  return (static_cast<uint32_t>(dt) << 24) | (static_cast<uint32_t>(bs) << 19) |
         (static_cast<uint32_t>(l2db) << 8) | (static_cast<uint32_t>(al) << 4) |
         as;
}

// BTS_LUT_BEGIN
constexpr uint32_t kDispatchLutKeys[] = {
    0x00000100, 0x00000101, 0x00000102, 0x00000200, 0x00000201, 0x00000202,
    0x00000300, 0x00000301, 0x00000302, 0x00000400, 0x00000401, 0x00000402,
    0x00000500, 0x00000501, 0x00000502, 0x00000600, 0x00000601, 0x00000602,
    0x00000700, 0x00000701, 0x00000702, 0x00000800, 0x00000801, 0x00000802,
    0x00000900, 0x00000901, 0x00000902, 0x00000A00, 0x00000A01, 0x00000A02,
    0x00000B00, 0x00000B01, 0x00000B02, 0x00000C00, 0x00000C01, 0x00000C02,
    0x00000D00, 0x00000D01, 0x00000D02, 0x00000E00, 0x00000E01, 0x00000E02,
    0x00000F00, 0x00000F01, 0x00000F02, 0x00001000, 0x00001001, 0x00001002,
    0x00001100, 0x00001101, 0x00001102, 0x00001200, 0x00001201, 0x00001202,
    0x00001300, 0x00001301, 0x00001302, 0x00080100, 0x00080101, 0x00080102,
    0x00080200, 0x00080201, 0x00080202, 0x00080300, 0x00080301, 0x00080302,
    0x00080400, 0x00080401, 0x00080402, 0x00080500, 0x00080501, 0x00080502,
    0x00080600, 0x00080601, 0x00080602, 0x00080700, 0x00080701, 0x00080702,
    0x00080800, 0x00080801, 0x00080802, 0x00080900, 0x00080901, 0x00080902,
    0x00080A00, 0x00080A01, 0x00080A02, 0x00080B00, 0x00080B01, 0x00080B02,
    0x00080C00, 0x00080C01, 0x00080C02, 0x00080D00, 0x00080D01, 0x00080D02,
    0x00080E00, 0x00080E01, 0x00080E02, 0x00080F00, 0x00080F01, 0x00080F02,
    0x00081000, 0x00081001, 0x00081002, 0x00081100, 0x00081101, 0x00081102,
    0x00081200, 0x00081201, 0x00081202, 0x00081300, 0x00081301, 0x00081302,
    0x00100100, 0x00100101, 0x00100102, 0x00100200, 0x00100201, 0x00100202,
    0x00100300, 0x00100301, 0x00100302, 0x00100400, 0x00100401, 0x00100402,
    0x00100500, 0x00100501, 0x00100502, 0x00100600, 0x00100601, 0x00100602,
    0x00100700, 0x00100701, 0x00100702, 0x00100800, 0x00100801, 0x00100802,
    0x00100900, 0x00100901, 0x00100902, 0x00100A00, 0x00100A01, 0x00100A02,
    0x00100B00, 0x00100B01, 0x00100B02, 0x00100C00, 0x00100C01, 0x00100C02,
    0x00100D00, 0x00100D01, 0x00100D02, 0x00100E00, 0x00100E01, 0x00100E02,
    0x00100F00, 0x00100F01, 0x00100F02, 0x00101000, 0x00101001, 0x00101002,
    0x00101100, 0x00101101, 0x00101102, 0x00101200, 0x00101201, 0x00101202,
    0x00101300, 0x00101301, 0x00101302, 0x00180100, 0x00180101, 0x00180102,
    0x00180200, 0x00180201, 0x00180202, 0x00180300, 0x00180301, 0x00180302,
    0x00180400, 0x00180401, 0x00180402, 0x00180500, 0x00180501, 0x00180502,
    0x00180600, 0x00180601, 0x00180602, 0x00180700, 0x00180701, 0x00180702,
    0x00180800, 0x00180801, 0x00180802, 0x00180900, 0x00180901, 0x00180902,
    0x00180A00, 0x00180A01, 0x00180A02, 0x00180B00, 0x00180B01, 0x00180B02,
    0x00180C00, 0x00180C01, 0x00180C02, 0x00180D00, 0x00180D01, 0x00180D02,
    0x00180E00, 0x00180E01, 0x00180E02, 0x00180F00, 0x00180F01, 0x00180F02,
    0x00181000, 0x00181001, 0x00181002, 0x00181100, 0x00181101, 0x00181102,
    0x00181200, 0x00181201, 0x00181202, 0x00181300, 0x00181301, 0x00181302,
    0x01000200, 0x01000201, 0x01000202, 0x01000300, 0x01000301, 0x01000302,
    0x01000400, 0x01000401, 0x01000402, 0x01000500, 0x01000501, 0x01000502,
    0x01000600, 0x01000601, 0x01000602, 0x01000700, 0x01000701, 0x01000702,
    0x01000800, 0x01000801, 0x01000802, 0x01000900, 0x01000901, 0x01000902,
    0x01000A00, 0x01000A01, 0x01000A02, 0x01000B00, 0x01000B01, 0x01000B02,
    0x01000C00, 0x01000C01, 0x01000C02, 0x01000D00, 0x01000D01, 0x01000D02,
    0x01000E00, 0x01000E01, 0x01000E02, 0x01000F00, 0x01000F01, 0x01000F02,
    0x01001000, 0x01001001, 0x01001002, 0x01001100, 0x01001101, 0x01001102,
    0x01001200, 0x01001201, 0x01001202, 0x01001300, 0x01001301, 0x01001302,
    0x01080200, 0x01080201, 0x01080202, 0x01080300, 0x01080301, 0x01080302,
    0x01080400, 0x01080401, 0x01080402, 0x01080500, 0x01080501, 0x01080502,
    0x01080600, 0x01080601, 0x01080602, 0x01080700, 0x01080701, 0x01080702,
    0x01080800, 0x01080801, 0x01080802, 0x01080900, 0x01080901, 0x01080902,
    0x01080A00, 0x01080A01, 0x01080A02, 0x01080B00, 0x01080B01, 0x01080B02,
    0x01080C00, 0x01080C01, 0x01080C02, 0x01080D00, 0x01080D01, 0x01080D02,
    0x01080E00, 0x01080E01, 0x01080E02, 0x01080F00, 0x01080F01, 0x01080F02,
    0x01081000, 0x01081001, 0x01081002, 0x01081100, 0x01081101, 0x01081102,
    0x01081200, 0x01081201, 0x01081202, 0x01081300, 0x01081301, 0x01081302,
    0x01100200, 0x01100201, 0x01100202, 0x01100300, 0x01100301, 0x01100302,
    0x01100400, 0x01100401, 0x01100402, 0x01100500, 0x01100501, 0x01100502,
    0x01100600, 0x01100601, 0x01100602, 0x01100700, 0x01100701, 0x01100702,
    0x01100800, 0x01100801, 0x01100802, 0x01100900, 0x01100901, 0x01100902,
    0x01100A00, 0x01100A01, 0x01100A02, 0x01100B00, 0x01100B01, 0x01100B02,
    0x01100C00, 0x01100C01, 0x01100C02, 0x01100D00, 0x01100D01, 0x01100D02,
    0x01100E00, 0x01100E01, 0x01100E02, 0x01100F00, 0x01100F01, 0x01100F02,
    0x01101000, 0x01101001, 0x01101002, 0x01101100, 0x01101101, 0x01101102,
    0x01101200, 0x01101201, 0x01101202, 0x01101300, 0x01101301, 0x01101302,
    0x01180200, 0x01180201, 0x01180202, 0x01180300, 0x01180301, 0x01180302,
    0x01180400, 0x01180401, 0x01180402, 0x01180500, 0x01180501, 0x01180502,
    0x01180600, 0x01180601, 0x01180602, 0x01180700, 0x01180701, 0x01180702,
    0x01180800, 0x01180801, 0x01180802, 0x01180900, 0x01180901, 0x01180902,
    0x01180A00, 0x01180A01, 0x01180A02, 0x01180B00, 0x01180B01, 0x01180B02,
    0x01180C00, 0x01180C01, 0x01180C02, 0x01180D00, 0x01180D01, 0x01180D02,
    0x01180E00, 0x01180E01, 0x01180E02, 0x01180F00, 0x01180F01, 0x01180F02,
    0x01181000, 0x01181001, 0x01181002, 0x01181100, 0x01181101, 0x01181102,
    0x01181200, 0x01181201, 0x01181202, 0x01181300, 0x01181301, 0x01181302,
};
constexpr DispatchLutValue kDispatchLut[] = {
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
    {BTS_PROTO_ROW_GATHER, 0}, {BTS_PROTO_ROW_GATHER, 0},
};
// 444 buckets
// BTS_LUT_END

const DispatchLutValue *DispatchLutLookup(const DispatchInput &in,
                                          const DispatchFeatures &f) {
  const uint8_t dt = in.dtypeSize == BTS_FP_DTYPE_BYTES ? 0 : 1;
  const uint8_t bs = in.blockSize == 2 ? 0
                     : in.blockSize == 3 ? 1
                     : in.blockSize == 4 ? 2
                                         : 3;
  const uint32_t key =
      DispatchLutKey(dt, bs, f.log2DepthBytes, f.align, f.aspect);
  const uint32_t *begin = kDispatchLutKeys;
  const uint32_t *end =
      kDispatchLutKeys + sizeof(kDispatchLutKeys) / sizeof(kDispatchLutKeys[0]);
  const uint32_t *it = std::lower_bound(begin, end, key);
  if (it != end && *it == key) {
    return &kDispatchLut[it - begin];
  }
  return nullptr;
}

struct DispatchResult {
  bool valid = false;
  bool emitStatic = false;
  int staticArm = -1;
  DispatchPrototype proto = BTS_PROTO_NONE;
  // Static emission triple (when emitStatic).
  uint32_t mode = 0;
  uint32_t variant = 0;
  uint32_t blockDim = 0;
  uint32_t staticTileOW = 0;
  // Parameterized emission (full tiling fields, mode FULL).
  uint8_t path = BTS_PATH_DEPTH_32B_ALIGNED;
  uint32_t tileOW = 1;
  uint32_t tileOH = 1;
  uint32_t tileDepth = 1;
  uint32_t tileElemAligned = 1;
  uint32_t totalTasks = 0;
  uint32_t activeCores = 1;
  uint8_t disableL2 = 0;
};

void FillFastResult(DispatchResult &r, uint8_t path, uint32_t tileOW,
                    uint32_t tileOH, uint32_t tileDepth, uint32_t tileElems,
                    uint32_t totalTasks, uint32_t numCores) {
  const uint32_t activeCores =
      std::min<uint32_t>(std::max<uint32_t>(1, numCores), totalTasks);
  r.valid = true;
  r.path = path;
  r.tileOW = tileOW;
  r.tileOH = tileOH;
  r.tileDepth = tileDepth;
  r.tileElemAligned = tileElems;
  r.totalTasks = totalTasks;
  r.activeCores = activeCores;
  r.disableL2 = totalTasks > activeCores ? 1 : 0;
}

// Double-buffer compact interleave: per-pixel in+out slots in one UB buffer.
bool BuildMaskCompactPlan(const DispatchInput &in, const DispatchFeatures &f,
                          DispatchResult &r) {
  if (in.blockSize != 4 || in.cropTop != 0 || in.cropBottom != 0) {
    return false;
  }
  if (f.align == BTS_ALIGN_UNALIGNED || f.depthBytes >= BTS_FP_DEPTH_BURST_BYTES ||
      in.depth > BTS_FP_COPY_MASK_MAX_B16) {
    return false;
  }
  if (f.outBatch != 1 || f.trafficBytes < BTS_FP_MIN_TRAFFIC_BYTES) {
    return false;
  }
  const uint64_t perPxBytes = 2ULL * f.depthBytes;
  uint32_t tileOW = static_cast<uint32_t>(
      FloorAlign64(f.ubPerBuf / perPxBytes, in.blockSize));
  if (tileOW == 0 || tileOW / in.blockSize > BTS_FP_COPY_REPEAT_MAX) {
    return false;
  }
  tileOW = std::min(tileOW, f.outWidth);
  const uint32_t tilesPerRow = CeilDiv(f.outWidth, tileOW);
  if (tilesPerRow < 2) {
    return false;
  }
  const uint64_t totalTasks64 =
      static_cast<uint64_t>(f.outBatch) * f.outHeight * tilesPerRow;
  const uint64_t tileElem64 = 2ULL * tileOW * in.depth;
  if (totalTasks64 == 0 || totalTasks64 > BTS_MAX_UINT32 ||
      totalTasks64 < in.numCores || tileElem64 > BTS_MAX_UINT32) {
    return false;
  }
  FillFastResult(r, BTS_PATH_MASK_COMPACT, tileOW, 1, in.depth,
                 static_cast<uint32_t>(tileElem64),
                 static_cast<uint32_t>(totalTasks64), in.numCores);
  return true;
}

// Single full-UB buffer, half-row chunks: fewest/biggest DMA ops for wide
// interleave shapes that are MTE-op-latency bound.
bool BuildMcBigPlan(const DispatchInput &in, const DispatchFeatures &f,
                    DispatchResult &r) {
  if (in.dtypeSize != BTS_FP_DTYPE_BYTES || in.blockSize < 2) {
    return false;
  }
  if (f.align == BTS_ALIGN_UNALIGNED || f.depthBytes >= BTS_FP_DEPTH_BURST_BYTES ||
      in.depth > BTS_FP_COPY_MASK_MAX_B16) {
    return false;
  }
  if (f.outBatch != 1 || f.trafficBytes < BTS_FP_MIN_TRAFFIC_BYTES) {
    return false;
  }
  const uint64_t perPxBytes = 2ULL * f.depthBytes;
  uint32_t tileOW = static_cast<uint32_t>(
      FloorAlign64(f.ubSingle / perPxBytes, in.blockSize));
  const uint32_t halfRow = AlignUpU32(CeilDiv(f.outWidth, 2), in.blockSize);
  tileOW = std::min(tileOW, halfRow);
  tileOW = std::min(tileOW, f.outWidth);
  if (tileOW == 0 || CeilDiv(tileOW, in.blockSize) > BTS_FP_COPY_REPEAT_MAX) {
    return false;
  }
  const uint32_t tilesPerRow = CeilDiv(f.outWidth, tileOW);
  if (tilesPerRow < 2) {
    return false;
  }
  const uint64_t totalTasks64 =
      static_cast<uint64_t>(f.outHeight) * tilesPerRow;
  const uint64_t tileElem64 = 2ULL * tileOW * in.depth;
  if (totalTasks64 < in.numCores || totalTasks64 > BTS_MAX_UINT32 ||
      tileElem64 * in.dtypeSize > f.ubSingle) {
    return false;
  }
  FillFastResult(r, BTS_PATH_MC_BIG, tileOW, 1, in.depth,
                 static_cast<uint32_t>(tileElem64),
                 static_cast<uint32_t>(totalTasks64), in.numCores);
  return true;
}

// Tall outputs with sub-burst rows: fuse whole output rows per task.
bool BuildHFusePlan(const DispatchInput &in, const DispatchFeatures &f,
                    DispatchResult &r) {
  if (in.blockSize != 2 || f.crop != BTS_CROP_ZERO) {
    return false;
  }
  if (f.align == BTS_ALIGN_UNALIGNED || f.depthBytes >= BTS_FP_DEPTH_BURST_BYTES) {
    return false;
  }
  if (static_cast<uint64_t>(in.width) * f.depthBytes > BTS_FP_DEPTH_BURST_BYTES) {
    return false;
  }
  if (f.aspect != BTS_ASPECT_TALL || f.trafficBytes < BTS_FP_MIN_TRAFFIC_BYTES) {
    return false;
  }
  if (in.width > BTS_FP_COPY_REPEAT_MAX || (f.outHeight % in.blockSize) != 0) {
    return false;
  }
  const uint64_t bytesPerOutRow = 2ULL * in.blockSize *
                                  static_cast<uint64_t>(in.width) * in.depth *
                                  in.dtypeSize;
  uint32_t rowsPerTask = static_cast<uint32_t>(
      FloorAlign64(f.ubPerBuf / bytesPerOutRow, BTS_FP_HFUSE_ROW_ALIGN));
  if (rowsPerTask < BTS_FP_HFUSE_ROW_ALIGN ||
      rowsPerTask > 2 * BTS_FP_COPY_REPEAT_MAX) {
    return false;
  }
  rowsPerTask = std::min(rowsPerTask, BTS_FP_HFUSE_TARGET_ROWS);
  rowsPerTask = std::min(rowsPerTask, f.outHeight);
  const uint32_t rowPairsPerBatch = f.outHeight / in.blockSize;
  const uint64_t totalTasks64 =
      static_cast<uint64_t>(f.outBatch) * rowPairsPerBatch;
  const uint64_t tileElem64 = 2ULL * rowsPerTask * f.outWidth * in.depth;
  if (totalTasks64 == 0 || totalTasks64 > BTS_MAX_UINT32 ||
      totalTasks64 < in.numCores || tileElem64 > BTS_MAX_UINT32) {
    return false;
  }
  FillFastResult(r, BTS_PATH_H_FUSE, f.outWidth, rowsPerTask, in.depth,
                 static_cast<uint32_t>(tileElem64),
                 static_cast<uint32_t>(totalTasks64), in.numCores);
  return true;
}

// Unaligned-depth gather with symmetric crops: build ramp indices, interleave
// two half rows per task.
bool BuildGatherInterleavePlan(const DispatchInput &in, const DispatchFeatures &f,
                               DispatchResult &r) {
  if (in.blockSize != 2 || f.align != BTS_ALIGN_UNALIGNED) {
    return false;
  }
  if (in.cropTop != in.cropBottom || in.cropLeft != in.cropRight ||
      in.cropTop < 1 || in.cropLeft < 1) {
    return false;
  }
  if (f.outBatch != 1 || (f.outWidth % 2) != 0 ||
      f.trafficBytes < BTS_FP_MIN_TRAFFIC_BYTES) {
    return false;
  }
  const uint32_t halfPx = f.outWidth / 2;
  const uint32_t lanePx = (halfPx + 1) / 2;
  const uint64_t laneBytes = static_cast<uint64_t>(lanePx) * f.depthBytes;
  if ((laneBytes % BTS_DATA_BLOCK_BYTES) != 0) {
    return false;
  }
  if (static_cast<uint64_t>((in.cropLeft + halfPx + 1) / 2) + lanePx >
      static_cast<uint64_t>(in.width) * in.blockSize / 2) {
    return false;
  }
  const uint64_t outElems = static_cast<uint64_t>(halfPx) * in.depth;
  const uint64_t inElems = 2ULL * lanePx * in.depth;
  const uint64_t slotElems = AlignUp64(inElems + outElems, 16);
  const uint64_t pairElems = 2ULL * in.depth;
  const uint64_t rampElems = Lcm64(pairElems, 8);
  if (slotElems > BTS_MAX_UINT32 || pairElems == 0 || rampElems == 0 ||
      rampElems > BTS_MAX_UINT32) {
    return false;
  }
  const uint64_t nRamp = CeilDiv64(outElems, rampElems);
  if (nRamp == 0) {
    return false;
  }
  const uint64_t tail = AlignUp64(outElems - (nRamp - 1) * rampElems, 8);
  const uint64_t idxElems = (nRamp - 1) * rampElems + tail;
  const uint64_t giUsableUb =
      std::min<uint64_t>(2ULL * f.ubPerBuf, BTS_FP_GI_IDX_BUDGET_BYTES);
  const uint64_t idxBytes =
      2ULL * slotElems * in.dtypeSize + idxElems * sizeof(uint32_t);
  const uint64_t totalTasks64 = static_cast<uint64_t>(f.outHeight) * 2ULL;
  if (idxBytes > giUsableUb || idxBytes > BTS_MAX_UINT32 || totalTasks64 == 0 ||
      totalTasks64 > BTS_MAX_UINT32 || totalTasks64 < in.numCores) {
    return false;
  }
  FillFastResult(r, BTS_PATH_GATHER_INTERLEAVE, halfPx, 1, in.depth,
                 static_cast<uint32_t>(slotElems),
                 static_cast<uint32_t>(totalTasks64), in.numCores);
  r.disableL2 = 0;
  return true;
}

// Width-chunk movers prefer core counts that divide whole rows.
uint32_t ChooseWide512ActiveCores(uint32_t numCoresAiv, uint32_t totalTasks,
                                  uint32_t outHeight) {
  uint32_t activeCores =
      std::min<uint32_t>(std::max<uint32_t>(1, numCoresAiv), totalTasks);
  if (outHeight == 0 || activeCores < outHeight ||
      activeCores % outHeight == 0) {
    return activeCores;
  }
  const uint32_t rowAligned = (activeCores / outHeight) * outHeight;
  if (rowAligned >= outHeight && rowAligned <= totalTasks) {
    return rowAligned;
  }
  return activeCores;
}

// fp16 wide rows of exactly one GM burst per pixel: contiguous lane reads,
// vector-copy interleave, one contiguous write per chunk.
bool BuildWide512BPlan(const DispatchInput &in, const DispatchFeatures &f,
                       DispatchResult &r) {
  if (in.dtypeSize != BTS_FP_DTYPE_BYTES || in.blockSize != 2 ||
      f.crop != BTS_CROP_ZERO) {
    return false;
  }
  if (f.depthBytes != BTS_FP_DEPTH_BURST_BYTES ||
      in.depth < BTS_W512_COPY_MASK_ELEMS ||
      (in.depth % BTS_W512_COPY_MASK_ELEMS) != 0) {
    return false;
  }
  if (f.outBatch != 1 || f.outHeight == 0 ||
      f.outWidth < BTS_W512_MIN_OUT_WIDTH ||
      f.trafficBytes < BTS_W512_MIN_TRAFFIC_BYTES) {
    return false;
  }
  const uint64_t bytesPerChunkPx = 2ULL * f.depthBytes;
  uint32_t tileOW = static_cast<uint32_t>(
      FloorAlign64(f.ubPerBuf / bytesPerChunkPx, in.blockSize));
  if (tileOW == 0) {
    return false;
  }
  tileOW = std::min(tileOW, f.outWidth);
  tileOW = SnapTileOWDivideOutWidth(tileOW, f.outWidth,
                                    CalcGmAlignOW(f.depthBytes));
  if (tileOW == 0 || tileOW / in.blockSize > BTS_FP_COPY_REPEAT_MAX) {
    return false;
  }
  const uint32_t tilesPerRow = CeilDiv(f.outWidth, tileOW);
  if (tilesPerRow < 2) {
    return false;
  }
  const uint64_t totalTasks64 =
      static_cast<uint64_t>(f.outHeight) * tilesPerRow;
  const uint64_t tileElem64 = 2ULL * tileOW * in.depth;
  if (totalTasks64 == 0 || totalTasks64 > BTS_MAX_UINT32 ||
      tileElem64 > BTS_MAX_UINT32) {
    return false;
  }
  const uint32_t totalTasks = static_cast<uint32_t>(totalTasks64);
  r.valid = true;
  r.path = BTS_PATH_WIDE_512B;
  r.tileOW = tileOW;
  r.tileOH = 1;
  r.tileDepth = in.depth;
  r.tileElemAligned = static_cast<uint32_t>(tileElem64);
  r.totalTasks = totalTasks;
  r.activeCores = ChooseWide512ActiveCores(in.numCores, totalTasks, f.outHeight);
  r.disableL2 = 0; // rows are written sequentially; keep L2 for GM reuse
  return true;
}

// Aligned-depth fallbacks: depth-split tiles, plane assembly, or row-pack.
bool BuildGenericPlan(const DispatchInput &in, const DispatchFeatures &f,
                      DispatchResult &r) {
  const uint64_t pixelBytes64 =
      static_cast<uint64_t>(in.depth) * in.dtypeSize;
  const uint64_t usableUb = f.ubPerBuf;
  if (in.dtypeSize == 0 || pixelBytes64 == 0 || pixelBytes64 > BTS_MAX_UINT32) {
    return false;
  }
  const bool largeDepthPath = f.depthBytes >= BTS_LARGE_DEPTH_MIN_BYTES;
  if (!largeDepthPath && pixelBytes64 > usableUb) {
    return false;
  }
  const bool depth32BAligned = f.align != BTS_ALIGN_UNALIGNED;
  const bool canUseAlignedAssembly =
      depth32BAligned &&
      CanUseAlignedAssemblyWithDepthTile(in.depth, in.dtypeSize, in.blockSize,
                                         usableUb);
  const bool canUsePaddedDirect =
      !depth32BAligned &&
      static_cast<uint64_t>(in.blockSize - 1) * f.depthBytes <= BTS_MAX_UINT32;

  uint8_t path = BTS_PATH_DEPTH_32B_ALIGNED;
  uint32_t tileOW = 1, tileOH = 1, tileElemAligned = 1, totalTasks = 0;
  uint32_t tileDepth = in.depth;

  if (largeDepthPath) {
    if (!depth32BAligned || !canUseAlignedAssembly) {
      return false;
    }
    BatchToSpaceTilePlan plan;
    if (!Build512BTilePlan(plan, f.outBatch, f.outHeight, f.outWidth, in.depth,
                           in.dtypeSize, in.blockSize, usableUb, in.numCores,
                           in.cropLeft)) {
      return false;
    }
    path = plan.path;
    tileOW = plan.tileOW;
    tileOH = plan.tileOH;
    tileElemAligned = plan.tileElemAligned;
    totalTasks = plan.totalTasks;
    tileDepth = plan.tileDepth;
    if (f.spatial <= BTS_SPACE_TINY && f.outBatch == 1 && in.blockSize == 2 &&
        f.crop == BTS_CROP_ZERO && f.align == BTS_ALIGN_512B) {
      path = BTS_PATH_DEPTH_TINY_SPATIAL;
    }
  } else if (canUseAlignedAssembly) {
    BatchToSpaceTilePlan plan;
    if (!BuildAlignedOWTilePlan(plan, f.outBatch, f.outHeight, f.outWidth,
                                in.depth, in.dtypeSize, in.blockSize, usableUb,
                                in.numCores)) {
      return false;
    }
    path = plan.path;
    tileOW = plan.tileOW;
    tileOH = plan.tileOH;
    tileElemAligned = plan.tileElemAligned;
    totalTasks = plan.totalTasks;
    tileDepth = plan.tileDepth;
  } else if (canUsePaddedDirect) {
    RowPackEstimate rowPackEst;
    tileOW = ChooseRowPackTileOW(f.outWidth, in.blockSize, in.depth,
                                 in.dtypeSize, usableUb, rowPackEst);
    if (tileOW == 0) {
      return false;
    }
    tileOH = ChooseRowPackTileOH(f.outHeight, tileOW, in.blockSize, in.depth,
                                 in.dtypeSize, usableUb, rowPackEst);
    if (tileOH == 0 ||
        !EstimateRowPackBytes(tileOW, tileOH, in.blockSize, in.depth,
                              in.dtypeSize, usableUb, rowPackEst)) {
      return false;
    }
    const uint64_t tileBytes64 = rowPackEst.totalBytes;
    if (tileBytes64 > usableUb || tileBytes64 > BTS_MAX_UINT32) {
      return false;
    }
    tileElemAligned = std::max<uint32_t>(
        1, static_cast<uint32_t>(tileBytes64 / in.dtypeSize));
    const uint32_t tilesPerRow = CeilDiv(f.outWidth, tileOW);
    const uint32_t rowTilesPerBatch = CeilDiv(f.outHeight, tileOH);
    const uint64_t totalTasks64 =
        static_cast<uint64_t>(f.outBatch) * rowTilesPerBatch * tilesPerRow;
    if (totalTasks64 == 0 || totalTasks64 > BTS_MAX_UINT32) {
      return false;
    }
    totalTasks = static_cast<uint32_t>(totalTasks64);
    path = BTS_PATH_DEPTH_32B_PADDED;
    tileDepth = in.depth;
  } else {
    return false;
  }

  const uint64_t outputBytes = static_cast<uint64_t>(f.outBatch) * f.outHeight *
                               f.outWidth * in.depth * in.dtypeSize;
  const uint32_t activeCores =
      ChooseActiveCores(totalTasks, in.numCores, outputBytes,
                        static_cast<BatchToSpaceCopyPath>(path));
  r.valid = true;
  r.path = path;
  r.tileOW = tileOW;
  r.tileOH = tileOH;
  r.tileDepth = tileDepth;
  r.tileElemAligned = tileElemAligned;
  r.totalTasks = totalTasks;
  r.activeCores = activeCores;
  r.disableL2 = totalTasks > activeCores ? 1 : 0;
  return true;
}

// Parameterized fits per prototype; each falls through to the generic plan
// when its fit gate misses, exactly one prototype family per shape.
bool BuildPrototypePlan(const DispatchInput &in, const DispatchFeatures &f,
                        DispatchPrototype proto, DispatchResult &r) {
  switch (proto) {
  case BTS_PROTO_ROW_GATHER:
    // fp16 gather builds its indices at runtime; the fp32 gather kernel is a
    // static arm only (its index ramp is compile-time), so fp32 falls through
    // to the padded generic plan.
    if (in.dtypeSize == BTS_FP_DTYPE_BYTES &&
        BuildGatherInterleavePlan(in, f, r)) {
      return true;
    }
    return BuildGenericPlan(in, f, r);
  case BTS_PROTO_H_FUSE:
    if (in.dtypeSize == BTS_FP_DTYPE_BYTES && BuildHFusePlan(in, f, r)) {
      return true;
    }
    return BuildGenericPlan(in, f, r);
  case BTS_PROTO_HALF_ROW_BIG:
    if (BuildWide512BPlan(in, f, r)) {
      return true;
    }
    return BuildGenericPlan(in, f, r);
  case BTS_PROTO_MC_BIG:
    if (in.dtypeSize == BTS_FP_DTYPE_BYTES && BuildMaskCompactPlan(in, f, r)) {
      return true;
    }
    if (BuildMcBigPlan(in, f, r)) {
      return true;
    }
    return BuildGenericPlan(in, f, r);
  case BTS_PROTO_DEPTH_SPLIT:
  case BTS_PROTO_ROW_PLANE:
  case BTS_PROTO_GENERIC:
  default:
    return BuildGenericPlan(in, f, r);
  }
}

DispatchResult Dispatch(const DispatchInput &in) {
  DispatchResult r;
  const DispatchFeatures f = ExtractFeatures(in);
  if (f.outBatch == 0 || f.outHeight == 0 || f.outWidth == 0 || in.depth == 0 ||
      in.dtypeSize == 0) {
    return r;
  }
  r.proto = SelectPath(in, f);
  const DispatchLutValue *lut = DispatchLutLookup(in, f);
  if (lut != nullptr && lut->proto != BTS_PROTO_NONE) {
    r.proto = static_cast<DispatchPrototype>(lut->proto);
  }
  const int arm = MatchStaticArm(in, f, r.proto);
  if (arm >= 0) {
    const DispatchStaticArm &a = kStaticArms[arm];
    r.valid = true;
    r.emitStatic = true;
    r.staticArm = arm;
    r.mode = a.mode;
    r.variant = a.variant;
    r.blockDim = a.blockDim;
    r.staticTileOW = a.tileOW;
    return r;
  }
  if (!BuildPrototypePlan(in, f, r.proto, r)) {
    r.valid = false;
  }
  return r;
}
// BTS_DISPATCH_END

bool ReadCrops(const gert::TypedContinuousVector<int64_t> *crops,
               uint32_t &cropTop, uint32_t &cropBottom,
               uint32_t &cropLeft, uint32_t &cropRight) {
  if (crops == nullptr || crops->GetSize() < 4) {
    return false;
  }

  const int64_t *cropData = crops->GetData();
  if (cropData == nullptr) {
    return false;
  }

  const int64_t top = cropData[0];
  const int64_t bottom = cropData[1];
  const int64_t left = cropData[2];
  const int64_t right = cropData[3];

  if (top < 0 || bottom < 0 || left < 0 || right < 0) {
    return false;
  }

  cropTop = static_cast<uint32_t>(top);
  cropBottom = static_cast<uint32_t>(bottom);
  cropLeft = static_cast<uint32_t>(left);
  cropRight = static_cast<uint32_t>(right);
  return true;
}

ge::graphStatus EmitStaticArm(gert::TilingContext *context,
                              const DispatchResult &d) {
  if (d.mode == BTS_TILING_MODE_MC_BIGTILE) {
    BatchToSpaceMcWideTilingData *tiling =
        context->GetTilingData<BatchToSpaceMcWideTilingData>();
    tiling->mode = d.mode;
    tiling->tileOW = d.staticTileOW;
  } else {
    BatchToSpaceTilingHeader *tiling =
        context->GetTilingData<BatchToSpaceTilingHeader>();
    tiling->mode = d.mode;
  }
  context->SetBlockDim(d.blockDim);
  size_t *workspace = context->GetWorkspaceSizes(1);
  workspace[0] = 0;
  SetBatchToSpaceTilingKey(context, d.variant);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmitParameterized(gert::TilingContext *context,
                                  const DispatchInput &in,
                                  const DispatchFeatures &f,
                                  const DispatchResult &d) {
  BatchToSpaceTilingData *tiling =
      context->GetTilingData<BatchToSpaceTilingData>();
  *tiling = {};
  tiling->mode = BTS_TILING_MODE_FULL;
  tiling->height = in.height;
  tiling->width = in.width;
  tiling->depth = in.depth;
  tiling->outBatch = f.outBatch;
  tiling->outHeight = f.outHeight;
  tiling->outWidth = f.outWidth;
  tiling->blockSize = in.blockSize;
  tiling->cropTop = in.cropTop;
  tiling->cropLeft = in.cropLeft;
  tiling->tileOW = d.tileOW;
  tiling->tileOH = d.tileOH;
  tiling->tileDepth = d.tileDepth;
  tiling->tileElemAligned = d.tileElemAligned;
  tiling->totalTasks = d.totalTasks;
  tiling->path = d.path;
  tiling->activeCores = static_cast<uint16_t>(d.activeCores);
  tiling->disableL2 = d.disableL2;
  context->SetBlockDim(d.activeCores);
  size_t *workspace = context->GetWorkspaceSizes(1);
  workspace[0] = 0;
  SetBatchToSpaceTilingKey(context);
  return ge::GRAPH_SUCCESS;
}

// Tuning harness (BTS_TUNE_ENABLE=1): collapsed-spatial deep shapes get the
// parameterized depth-split tuning kernels; everything else reuses the
// parameterized plan with env overrides.
ge::graphStatus EmitTuned(gert::TilingContext *context, const DispatchInput &in,
                          const DispatchFeatures &f) {
  if (f.spatial <= BTS_SPACE_TINY && f.align == BTS_ALIGN_512B &&
      f.depthBytes >= BTS_LARGE_DEPTH_MIN_BYTES && f.outBatch == 1) {
    uint32_t tileOW = f.outWidth;
    uint32_t tileOH = f.outHeight;
    uint32_t tileDepth = f.spatial == BTS_SPACE_ONE ? 2048 : 512;
    uint32_t requested = 0;
    if (ReadTuningU32("BTS_TILE_OW", requested)) {
      tileOW = requested;
    }
    if (ReadTuningU32("BTS_TILE_OH", requested)) {
      tileOH = requested;
    }
    if (ReadTuningU32("BTS_TILE_DEPTH", requested)) {
      tileDepth = requested;
    }
    const uint64_t tileBytes =
        static_cast<uint64_t>(tileOW) * tileOH * tileDepth * in.dtypeSize;
    if (tileOW == 0 || tileOW > f.outWidth || tileOH == 0 ||
        tileOH > f.outHeight || tileDepth == 0 || tileDepth > in.depth ||
        (static_cast<uint64_t>(tileDepth) * in.dtypeSize %
         BTS_DATA_BLOCK_BYTES) != 0 ||
        tileBytes > f.ubPerBuf || tileBytes > BTS_MAX_UINT32) {
      return ge::GRAPH_FAILED;
    }
    const uint32_t totalTasks = CeilDiv(f.outHeight, tileOH) *
                                CeilDiv(f.outWidth, tileOW) *
                                CeilDiv(in.depth, tileDepth);
    uint32_t activeCores = 8;
    if (ReadTuningU32("BTS_ACTIVE_CORES", requested)) {
      activeCores = requested;
    }
    if (activeCores == 0 || activeCores > in.numCores) {
      return ge::GRAPH_FAILED;
    }
    BatchToSpaceTuningData *tiling =
        context->GetTilingData<BatchToSpaceTuningData>();
    tiling->mode = f.spatial == BTS_SPACE_ONE
                       ? BTS_TILING_MODE_DEPTH_SPLIT_TINY_TUNING
                       : BTS_TILING_MODE_DEPTH_SPLIT_LARGE_TUNING;
    tiling->tileOW = tileOW;
    tiling->tileOH = tileOH;
    tiling->tileDepth = tileDepth;
    tiling->tileElemAligned = static_cast<uint32_t>(tileBytes / in.dtypeSize);
    tiling->totalTasks = totalTasks;
    tiling->activeCores = activeCores;
    context->SetBlockDim(activeCores);
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    SetBatchToSpaceTilingKey(context);
    return ge::GRAPH_SUCCESS;
  }

  DispatchResult d;
  if (!BuildPrototypePlan(in, f, SelectPath(in, f), d) || !d.valid) {
    return ge::GRAPH_FAILED;
  }
  uint32_t tileOW = d.tileOW;
  uint32_t tileOH = d.tileOH;
  uint32_t tileDepth = d.tileDepth;
  uint32_t tileElemAligned = d.tileElemAligned;
  uint32_t totalTasks = d.totalTasks;
  if (!ApplyFixedPathTuning(static_cast<BatchToSpaceCopyPath>(d.path),
                            f.outBatch, f.outHeight, f.outWidth, in.depth,
                            in.dtypeSize, in.blockSize, f.ubPerBuf, tileOW,
                            tileOH, tileDepth, tileElemAligned, totalTasks)) {
    return ge::GRAPH_FAILED;
  }
  d.tileOW = tileOW;
  d.tileOH = tileOH;
  d.tileDepth = tileDepth;
  d.tileElemAligned = tileElemAligned;
  d.totalTasks = totalTasks;
  uint32_t requestedCores = 0;
  if (ReadTuningU32("BTS_ACTIVE_CORES", requestedCores)) {
    if (requestedCores == 0 || requestedCores > in.numCores) {
      return ge::GRAPH_FAILED;
    }
    d.activeCores = requestedCores;
  }
  d.disableL2 = d.totalTasks > d.activeCores ? 1 : 0;
  return EmitParameterized(context, in, f, d);
}
} // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t numCoresAiv = static_cast<uint32_t>(platform.GetCoreNumAiv());
  uint64_t ubSize = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

  const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
  ge::DataType dtype_x = tensor_x->GetDataType();
  uint32_t dtypeSize = static_cast<uint32_t>(ge::GetSizeByDataType(dtype_x));
  const gert::Shape &xShape = tensor_x->GetOriginShape();
  if (xShape.GetDimNum() != 4) {
    return ge::GRAPH_FAILED;
  }

  const gert::RuntimeAttrs *attrs = context->GetAttrs();
  if (attrs == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const gert::TypedContinuousVector<int64_t> *attr_crops = attrs->GetListInt(0);
  const int64_t *attr_block_size = attrs->GetInt(1);
  if (attr_crops == nullptr || attr_block_size == nullptr ||
      *attr_block_size <= 0) {
    return ge::GRAPH_FAILED;
  }

  uint32_t cropTop = 0;
  uint32_t cropBottom = 0;
  uint32_t cropLeft = 0;
  uint32_t cropRight = 0;
  if (!ReadCrops(attr_crops, cropTop, cropBottom, cropLeft, cropRight) ||
      !ValidateStaticShape(xShape.GetDim(0), xShape.GetDim(1),
                           xShape.GetDim(2), xShape.GetDim(3),
                           *attr_block_size, cropTop, cropBottom, cropLeft,
                           cropRight)) {
    return ge::GRAPH_FAILED;
  }

  DispatchInput in;
  in.batch = static_cast<uint32_t>(xShape.GetDim(0));
  in.height = static_cast<uint32_t>(xShape.GetDim(1));
  in.width = static_cast<uint32_t>(xShape.GetDim(2));
  in.depth = static_cast<uint32_t>(xShape.GetDim(3));
  in.dtypeSize = dtypeSize;
  in.blockSize = static_cast<uint32_t>(*attr_block_size);
  in.cropTop = cropTop;
  in.cropBottom = cropBottom;
  in.cropLeft = cropLeft;
  in.cropRight = cropRight;
  in.numCores = numCoresAiv;
  in.ubBytes = ubSize;

  const DispatchFeatures f = ExtractFeatures(in);
  if (TuningEnabled()) {
    return EmitTuned(context, in, f);
  }
  const DispatchResult d = Dispatch(in);
  if (!d.valid) {
    return ge::GRAPH_FAILED;
  }
  return d.emitStatic ? EmitStaticArm(context, d)
                      : EmitParameterized(context, in, f, d);
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
  const gert::Shape *xShape = context->GetInputShape(0);
  gert::Shape *yShape = context->GetOutputShape(0);
  const gert::RuntimeAttrs *attrs = context->GetAttrs();
  if (attrs == nullptr) {
    return GRAPH_FAILED;
  }
  const gert::TypedContinuousVector<int64_t> *crops = attrs->GetListInt(0);
  const int64_t *blockSizeAttr = attrs->GetInt(1);
  if (xShape == nullptr || yShape == nullptr || crops == nullptr ||
      blockSizeAttr == nullptr || xShape->GetDimNum() != 4 ||
      *blockSizeAttr <= 0) {
    return GRAPH_FAILED;
  }

  uint32_t cropTop = 0;
  uint32_t cropBottom = 0;
  uint32_t cropLeft = 0;
  uint32_t cropRight = 0;
  if (!optiling::ReadCrops(crops, cropTop, cropBottom, cropLeft, cropRight) ||
      !optiling::ValidateStaticShape(xShape->GetDim(0), xShape->GetDim(1),
                                     xShape->GetDim(2), xShape->GetDim(3),
                                     *blockSizeAttr, cropTop, cropBottom,
                                     cropLeft, cropRight)) {
    return GRAPH_FAILED;
  }

  const int64_t blockSize = *blockSizeAttr;
  const int64_t batch = xShape->GetDim(0);
  const int64_t height = xShape->GetDim(1);
  const int64_t width = xShape->GetDim(2);
  const int64_t depth = xShape->GetDim(3);
  const int64_t outBatch = batch / (blockSize * blockSize);
  const int64_t outHeight = height * blockSize - cropTop - cropBottom;
  const int64_t outWidth = width * blockSize - cropLeft - cropRight;
  *yShape = gert::Shape({outBatch, outHeight, outWidth, depth});
  return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
  context->SetOutputDataType(0, context->GetInputDataType(0));
  return ge::GRAPH_SUCCESS;
}
} // namespace ge

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
} // namespace ops
