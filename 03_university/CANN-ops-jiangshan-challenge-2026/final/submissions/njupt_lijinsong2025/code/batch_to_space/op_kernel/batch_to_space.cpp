#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"

#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"

namespace {
constexpr uint32_t BTS_DATA_BLOCK_BYTES = 32;
constexpr uint32_t BTS_GM_ALIGN_BYTES = 512;
constexpr uint32_t BTS_W512_COPY_MASK_ELEMS = 128;

constexpr int32_t BTS_EVENT_FREE_BUF0 = EVENT_ID0;
constexpr int32_t BTS_EVENT_FREE_BUF1 = EVENT_ID1;
constexpr int32_t BTS_EVENT_READY_BUF0 = EVENT_ID2;
constexpr int32_t BTS_EVENT_READY_BUF1 = EVENT_ID3;
constexpr int32_t BTS_EVENT_PACK_READY_BUF0 = EVENT_ID4;
constexpr int32_t BTS_EVENT_PACK_READY_BUF1 = EVENT_ID5;

struct TaskRange {
  uint32_t offset;
  uint32_t length;
};

struct DecodedTileTask {
  uint32_t batchIdx;
  uint32_t ohStart;
  uint32_t curOH;
  uint32_t owStart;
  uint32_t curOW;
  uint32_t depthOffset;
  uint32_t curDepth;
};

struct Wide512BTask {
  uint32_t oh;
  uint32_t ih;
  uint32_t bh;
  uint32_t owStart;
  uint32_t curOW;
  uint32_t lanePx;
  uint32_t iw;
};

__aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs) {
  return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint32_t AlignUpU32(uint32_t x, uint32_t align) {
  return align == 0 ? x : ((x + align - 1) / align) * align;
}

__aicore__ inline uint32_t CeilDivU32(uint32_t x, uint32_t y) {
  return (x == 0 || y == 0) ? 0 : 1 + (x - 1) / y;
}

__aicore__ inline uint32_t GcdU32(uint32_t a, uint32_t b) {
  while (b != 0) {
    const uint32_t t = a % b;
    a = b;
    b = t;
  }
  return a;
}

__aicore__ inline uint32_t LcmU32(uint32_t a, uint32_t b) {
  const uint32_t g = GcdU32(a, b);
  return g == 0 ? 0 : (a / g) * b;
}

__aicore__ inline TaskRange GetTaskRange(const BatchToSpaceTilingData &tiling) {
  const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
  const uint32_t activeCores = tiling.activeCores;
  if (tiling.totalTasks == 0 || activeCores == 0) {
    return {0, 0};
  }

  const uint32_t baseTasks = tiling.totalTasks / activeCores;
  const uint32_t formerNum = tiling.totalTasks % activeCores;
  const uint32_t formerTasks = baseTasks + (formerNum > 0 ? 1U : 0U);
  const uint32_t tailTasks = baseTasks;
  uint32_t offset = 0;
  uint32_t length = 0;
  if (blockIdx < formerNum) {
    length = formerTasks;
    offset = blockIdx * formerTasks;
  } else {
    length = tailTasks;
    offset = formerNum * formerTasks + (blockIdx - formerNum) * tailTasks;
  }

  if (offset >= tiling.totalTasks) {
    return {offset, 0};
  }
  return {offset, MinU32(length, tiling.totalTasks - offset)};
}

// Assign contiguous output-row intervals (or width-chunk slices within a row)
// to each core so GM traffic stays row-local instead of striding via flat
// taskId stripes.
__aicore__ inline TaskRange GetWide512RowBoundTaskRange(
    const BatchToSpaceTilingData &tiling, uint32_t tilesPerRow) {
  const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
  const uint32_t activeCores = tiling.activeCores;
  const uint32_t outHeight = tiling.outHeight;
  if (tiling.totalTasks == 0 || activeCores == 0 || outHeight == 0 ||
      tilesPerRow == 0 || blockIdx >= activeCores) {
    return {0, 0};
  }

  if (activeCores >= outHeight) {
    const uint32_t baseCoresPerRow = activeCores / outHeight;
    const uint32_t rowsWithExtraCore = activeCores % outHeight;
    uint32_t oh = 0;
    uint32_t coreInRow = 0;
    uint32_t coresOnRow = 0;
    if (blockIdx < rowsWithExtraCore * (baseCoresPerRow + 1U)) {
      const uint32_t coresPerFatRow = baseCoresPerRow + 1U;
      oh = blockIdx / coresPerFatRow;
      coreInRow = blockIdx - oh * coresPerFatRow;
      coresOnRow = coresPerFatRow;
    } else {
      const uint32_t rem =
          blockIdx - rowsWithExtraCore * (baseCoresPerRow + 1U);
      oh = rowsWithExtraCore + rem / baseCoresPerRow;
      coreInRow = rem - (oh - rowsWithExtraCore) * baseCoresPerRow;
      coresOnRow = baseCoresPerRow;
    }
    if (coresOnRow == 0 || oh >= outHeight) {
      return {0, 0};
    }

    const uint32_t baseChunks = tilesPerRow / coresOnRow;
    const uint32_t chunkExtra = tilesPerRow % coresOnRow;
    const uint32_t chunkStart =
        coreInRow * baseChunks + MinU32(coreInRow, chunkExtra);
    const uint32_t chunkLen =
        baseChunks + (coreInRow < chunkExtra ? 1U : 0U);
    if (chunkLen == 0) {
      return {0, 0};
    }
    const uint32_t offset = oh * tilesPerRow + chunkStart;
    return {offset, MinU32(chunkLen, tiling.totalTasks - offset)};
  }

  const uint32_t baseRows = outHeight / activeCores;
  const uint32_t rowExtra = outHeight % activeCores;
  const uint32_t rowLen = baseRows + (blockIdx < rowExtra ? 1U : 0U);
  const uint32_t ohStart = blockIdx * baseRows + MinU32(blockIdx, rowExtra);
  if (rowLen == 0 || ohStart >= outHeight) {
    return {0, 0};
  }
  const uint32_t ohEnd = MinU32(ohStart + rowLen, outHeight);
  const uint32_t offset = ohStart * tilesPerRow;
  const uint32_t length = (ohEnd - ohStart) * tilesPerRow;
  return {offset, MinU32(length, tiling.totalTasks - offset)};
}

__aicore__ inline TaskRange GetStaticTaskRange(uint32_t totalTasks,
                                               uint32_t activeCores) {
  const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
  if (totalTasks == 0 || activeCores == 0) {
    return {0, 0};
  }
  const uint32_t baseTasks = totalTasks / activeCores;
  const uint32_t formerNum = totalTasks % activeCores;
  const uint32_t length = baseTasks + (blockIdx < formerNum ? 1U : 0U);
  const uint32_t offset =
      blockIdx * baseTasks + MinU32(blockIdx, formerNum);
  return offset < totalTasks
             ? TaskRange{offset, MinU32(length, totalTasks - offset)}
             : TaskRange{offset, 0};
}

__aicore__ inline int32_t FreeEventId(uint32_t bufIdx) {
  return bufIdx == 0 ? BTS_EVENT_FREE_BUF0 : BTS_EVENT_FREE_BUF1;
}

__aicore__ inline int32_t ReadyEventId(uint32_t bufIdx) {
  return bufIdx == 0 ? BTS_EVENT_READY_BUF0 : BTS_EVENT_READY_BUF1;
}

__aicore__ inline int32_t PackReadyEventId(uint32_t bufIdx) {
  return bufIdx == 0 ? BTS_EVENT_PACK_READY_BUF0 : BTS_EVENT_PACK_READY_BUF1;
}

__aicore__ inline void MarkBufFree(uint32_t bufIdx) {
  AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(FreeEventId(bufIdx));
}

__aicore__ inline void WaitBufFree(uint32_t bufIdx) {
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(FreeEventId(bufIdx));
}

__aicore__ inline void MarkCopyInReady(uint32_t bufIdx) {
  AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(ReadyEventId(bufIdx));
}

__aicore__ inline void WaitCopyInReady(uint32_t bufIdx) {
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(ReadyEventId(bufIdx));
}

__aicore__ inline void MarkMte2ReadyForV(uint32_t bufIdx) {
  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ReadyEventId(bufIdx));
}

__aicore__ inline void WaitMte2ReadyForV(uint32_t bufIdx) {
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ReadyEventId(bufIdx));
}

__aicore__ inline void MarkVReadyForMte3(uint32_t bufIdx) {
  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(PackReadyEventId(bufIdx));
}

__aicore__ inline void WaitVReadyForMte3(uint32_t bufIdx) {
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(PackReadyEventId(bufIdx));
}

template <typename T>
__aicore__ inline __ubuf__ T *UbPtr(uint32_t byteOffset = 0) {
  return reinterpret_cast<__ubuf__ T *>(byteOffset);
}

} // namespace

struct RowGatherFp32CorePlan {
  uint32_t inputOffsetA;
  uint32_t inputOffsetB;
  uint32_t outputOffset;
};

template <class DT_X> class KernelBatchToSpaceRowGatherFp32 {
public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
    xGm = reinterpret_cast<__gm__ DT_X *>(x);
    yGm = reinterpret_cast<__gm__ DT_X *>(y);
  }

  __aicore__ inline void Process() {
    if constexpr (sizeof(DT_X) == sizeof(float)) {
      const RowGatherFp32CorePlan plan = GetCorePlan();

      MarkBufFree(0);
      CopyIn(plan);
      BuildIdx();
      WaitMte2ReadyForV(0);
      Gather();
      WaitVReadyForMte3(0);
      CopyOut(plan);
      MarkBufFree(0);
      WaitBufFree(0);
    }
  }

private:
  static constexpr uint32_t kDepth = 5;
  static constexpr uint32_t kOutWidth = 26;
  static constexpr uint32_t kLanePx = 13;
  static constexpr uint32_t kLaneElems = kLanePx * kDepth;
  static constexpr uint32_t kLaneStrideElems = 72;
  static constexpr uint32_t kOutElems = kOutWidth * kDepth;
  static constexpr uint32_t kOutBase = 2 * kLaneStrideElems;
  static constexpr uint32_t kPairElems = 2 * kDepth;
  static constexpr uint32_t kRampElems = 40;
  static constexpr uint32_t kPairsPerRamp = kRampElems / kPairElems;
  static constexpr uint32_t kNRamp = 4;
  static constexpr uint32_t kTailElems = 16;
  static constexpr uint32_t kIdxElems = (kNRamp - 1) * kRampElems + kTailElems;
  static constexpr uint32_t kSlotElems = 280;
  static constexpr uint32_t kIdxBaseBytes = kSlotElems * sizeof(float);

  __aicore__ inline __ubuf__ DT_X *SlotAddr() const {
    return UbPtr<DT_X>();
  }

  __aicore__ inline __ubuf__ int32_t *IdxAddr() const {
    return UbPtr<int32_t>(kIdxBaseBytes);
  }

  __aicore__ inline RowGatherFp32CorePlan GetCorePlan() const {
    const uint32_t blockIdx =
        static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t pairIdx = blockIdx >> 1U;
    const uint32_t oddOffset = (blockIdx & 1U) * 1500U;
    return {830U + pairIdx * 75U + oddOffset,
            85U + pairIdx * 75U + oddOffset, blockIdx * kOutElems};
  }

  __aicore__ inline void BuildIdx() const {
    __ubuf__ int32_t *idx = IdxAddr();
    for (uint32_t pair = 0; pair < kPairsPerRamp; ++pair) {
      const uint32_t base = pair * kPairElems;
      const uint32_t srcBaseBytes = pair * kDepth * sizeof(DT_X);
      for (uint32_t d = 0; d < kDepth; ++d) {
        idx[base + d] =
            static_cast<int32_t>(srcBaseBytes + d * sizeof(DT_X));
        idx[base + kDepth + d] =
            static_cast<int32_t>(kLaneStrideElems * sizeof(DT_X) +
                                 srcBaseBytes + d * sizeof(DT_X));
      }
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID6);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID6);
    constexpr int32_t kRampByteStep =
        static_cast<int32_t>(kPairsPerRamp * kDepth * sizeof(DT_X));
    for (uint32_t blk = 1; blk < kNRamp; ++blk) {
      const uint32_t count = (blk == kNRamp - 1) ? kTailElems : kRampElems;
      AscendC::AddsImpl<int32_t>(
          idx + blk * kRampElems, idx, blk * kRampByteStep,
          static_cast<int32_t>(count));
    }
    AscendC::PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void CopyIn(const RowGatherFp32CorePlan &plan) {
    WaitBufFree(0);
    __ubuf__ DT_X *slotAddr = SlotAddr();
    constexpr uint32_t kLaneBytes = kLaneElems * sizeof(float);
    copy_gm_to_ubuf_align_b32(
        (__ubuf__ void *)slotAddr,
        (__gm__ void *)(xGm + plan.inputOffsetA), 0, 1, kLaneBytes, 0, 0, 0,
        0);
    copy_gm_to_ubuf_align_b32(
        (__ubuf__ void *)(slotAddr + kLaneStrideElems),
        (__gm__ void *)(xGm + plan.inputOffsetB), 0, 1, kLaneBytes, 0, 0, 0,
        0);
    MarkMte2ReadyForV(0);
  }

  __aicore__ inline void Gather() {
    __ubuf__ DT_X *slot = SlotAddr();
    __ubuf__ uint32_t *idx =
        reinterpret_cast<__ubuf__ uint32_t *>(IdxAddr());
    AscendC::GatherImpl(slot + kOutBase, slot, idx, kSlotElems, 0,
                        static_cast<uint64_t>(64), 2, 8);
    AscendC::GatherImpl(slot + kOutBase + 128, slot, idx + 128,
                        kSlotElems, 0, static_cast<uint64_t>(2), 1, 8);
    MarkVReadyForMte3(0);
  }

  __aicore__ inline void CopyOut(const RowGatherFp32CorePlan &plan) const {
    __ubuf__ DT_X *slotAddr = SlotAddr();
    constexpr uint32_t kOutBytes = kOutElems * sizeof(float);
    copy_ubuf_to_gm_align_b32(
        (__gm__ void *)(yGm + plan.outputOffset),
        (__ubuf__ void *)(slotAddr + kOutBase), 0, 1, kOutBytes, 0, 0, 0, 0);
  }

  __gm__ DT_X *xGm;
  __gm__ DT_X *yGm;
};

template <int TUNING> struct TuningDataStorage {};
template <> struct TuningDataStorage<1> {
  BatchToSpaceTuningData tiling;
};

template <class DT_X, int TUNING>
class KernelBatchToSpaceDepthSplitTinyTuned : private TuningDataStorage<TUNING> {
public:
  __aicore__ inline KernelBatchToSpaceDepthSplitTinyTuned() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
    xGm = reinterpret_cast<__gm__ DT_X *>(x);
    yGm = reinterpret_cast<__gm__ DT_X *>(y);
  }

  __aicore__ inline void Init(
      GM_ADDR x, GM_ADDR y,
      const BatchToSpaceTuningData &tilingData) {
    this->tiling = tilingData;
    Init(x, y);
  }

  __aicore__ inline void Process() {
    if constexpr (TUNING != 0) {
      ProcessTuned();
    }
  }

private:
  static constexpr uint32_t kDepth = 16384;
  static constexpr uint32_t kOutHeight = 2;
  static constexpr uint32_t kOutWidth = 2;

  __aicore__ inline uint64_t InOffset(uint32_t n, uint32_t d) const {
    return static_cast<uint64_t>(n) * kDepth + d;
  }

  __aicore__ inline uint64_t OutOffset(uint32_t h, uint32_t w,
                                       uint32_t d) const {
    return ((static_cast<uint64_t>(h) * kOutWidth + w) * kDepth) + d;
  }

  __aicore__ inline void ProcessTuned() {
    if (sizeof(DT_X) != sizeof(uint16_t) || xGm == yGm) {
      return;
    }
    const TaskRange range =
        GetStaticTaskRange(this->tiling.totalTasks, this->tiling.activeCores);
    if (range.length == 0) {
      return;
    }
    MarkBufFree(0);
    for (uint32_t i = 0; i < range.length; ++i) {
      CopyTuned(range.offset + i);
    }
    WaitBufFree(0);
  }

  __aicore__ inline void CopyTuned(uint32_t taskId) {
    WaitBufFree(0);
    const uint32_t depthTiles = CeilDivU32(kDepth, this->tiling.tileDepth);
    const uint32_t depthIdx = taskId % depthTiles;
    const uint32_t spatial = taskId / depthTiles;
    const uint32_t owTiles = CeilDivU32(kOutWidth, this->tiling.tileOW);
    const uint32_t owIdx = spatial % owTiles;
    const uint32_t ohIdx = spatial / owTiles;
    const uint32_t ohStart = ohIdx * this->tiling.tileOH;
    const uint32_t owStart = owIdx * this->tiling.tileOW;
    const uint32_t depthOffset = depthIdx * this->tiling.tileDepth;
    const uint32_t curOH =
        MinU32(this->tiling.tileOH, kOutHeight - ohStart);
    const uint32_t curOW =
        MinU32(this->tiling.tileOW, kOutWidth - owStart);
    const uint32_t curDepth =
        MinU32(this->tiling.tileDepth, kDepth - depthOffset);
    __ubuf__ DT_X *tileAddr = UbPtr<DT_X>();
    AscendC::DataCopyExtParams params = {
        1, curDepth * static_cast<uint32_t>(sizeof(DT_X)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<DT_X> padParams = {false, 0, 0,
                                                     static_cast<DT_X>(0)};
    for (uint32_t r = 0; r < curOH; ++r) {
      for (uint32_t c = 0; c < curOW; ++c) {
        const uint32_t oh = ohStart + r;
        const uint32_t ow = owStart + c;
        const uint32_t local = (r * curOW + c) * curDepth;
        AscendC::DataCopyPadGm2UBImpl(
            tileAddr + local, xGm + InOffset(oh * 2 + ow, depthOffset),
            params, padParams);
      }
    }
    MarkCopyInReady(0);
    WaitCopyInReady(0);
    for (uint32_t r = 0; r < curOH; ++r) {
      for (uint32_t c = 0; c < curOW; ++c) {
        const uint32_t local = (r * curOW + c) * curDepth;
        AscendC::DataCopyPadUB2GMImpl(
            yGm + OutOffset(ohStart + r, owStart + c, depthOffset),
            tileAddr + local, params);
      }
    }
    MarkBufFree(0);
  }

  __gm__ DT_X *xGm;
  __gm__ DT_X *yGm;
};

template <class DT_X, int TUNING>
class KernelBatchToSpaceDepthSplitLargeTuned : private TuningDataStorage<TUNING> {
public:
  __aicore__ inline KernelBatchToSpaceDepthSplitLargeTuned() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
    xGm = reinterpret_cast<__gm__ DT_X *>(x);
    yGm = reinterpret_cast<__gm__ DT_X *>(y);
  }

  __aicore__ inline void Init(
      GM_ADDR x, GM_ADDR y,
      const BatchToSpaceTuningData &tilingData) {
    this->tiling = tilingData;
    Init(x, y);
  }

  __aicore__ inline void Process() {
    if constexpr (TUNING != 0) {
      ProcessTuned();
    }
  }

private:
  static constexpr uint32_t kHeight = 2;
  static constexpr uint32_t kWidth = 2;
  static constexpr uint32_t kDepth = 4096;
  static constexpr uint32_t kOutHeight = 4;
  static constexpr uint32_t kOutWidth = 4;

  __aicore__ inline uint64_t InOffset(uint32_t n, uint32_t h, uint32_t w,
                                      uint32_t d) const {
    return (((static_cast<uint64_t>(n) * kHeight + h) * kWidth + w) * kDepth) +
           d;
  }

  __aicore__ inline uint64_t OutOffset(uint32_t h, uint32_t w,
                                       uint32_t d) const {
    return ((static_cast<uint64_t>(h) * kOutWidth + w) * kDepth) + d;
  }

  __aicore__ inline void ProcessTuned() {
    if (sizeof(DT_X) != sizeof(uint16_t)) {
      return;
    }
    const TaskRange range =
        GetStaticTaskRange(this->tiling.totalTasks, this->tiling.activeCores);
    if (range.length == 0) {
      return;
    }
    MarkBufFree(0);
    for (uint32_t i = 0; i < range.length; ++i) {
      CopyTuned(range.offset + i);
    }
    WaitBufFree(0);
  }

  __aicore__ inline void CopyTuned(uint32_t taskId) {
    WaitBufFree(0);
    const uint32_t depthTiles = CeilDivU32(kDepth, this->tiling.tileDepth);
    const uint32_t depthIdx = taskId % depthTiles;
    const uint32_t spatial = taskId / depthTiles;
    const uint32_t owTiles = CeilDivU32(kOutWidth, this->tiling.tileOW);
    const uint32_t owIdx = spatial % owTiles;
    const uint32_t ohIdx = spatial / owTiles;
    const uint32_t ohStart = ohIdx * this->tiling.tileOH;
    const uint32_t owStart = owIdx * this->tiling.tileOW;
    const uint32_t depthOffset = depthIdx * this->tiling.tileDepth;
    const uint32_t curOH =
        MinU32(this->tiling.tileOH, kOutHeight - ohStart);
    const uint32_t curOW =
        MinU32(this->tiling.tileOW, kOutWidth - owStart);
    const uint32_t curDepth =
        MinU32(this->tiling.tileDepth, kDepth - depthOffset);
    __ubuf__ DT_X *tileAddr = UbPtr<DT_X>();
    AscendC::DataCopyExtParams params = {
        1, curDepth * static_cast<uint32_t>(sizeof(DT_X)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<DT_X> padParams = {false, 0, 0,
                                                     static_cast<DT_X>(0)};
    for (uint32_t r = 0; r < curOH; ++r) {
      const uint32_t oh = ohStart + r;
      const uint32_t ih = oh >> 1;
      const uint32_t bh = oh & 1U;
      for (uint32_t c = 0; c < curOW; ++c) {
        const uint32_t ow = owStart + c;
        const uint32_t iw = ow >> 1;
        const uint32_t bw = ow & 1U;
        const uint32_t local = (r * curOW + c) * curDepth;
        AscendC::DataCopyPadGm2UBImpl(
            tileAddr + local,
            xGm + InOffset((bh << 1) + bw, ih, iw, depthOffset), params,
            padParams);
      }
    }
    MarkCopyInReady(0);
    WaitCopyInReady(0);
    for (uint32_t r = 0; r < curOH; ++r) {
      for (uint32_t c = 0; c < curOW; ++c) {
        const uint32_t local = (r * curOW + c) * curDepth;
        AscendC::DataCopyPadUB2GMImpl(
            yGm + OutOffset(ohStart + r, owStart + c, depthOffset),
            tileAddr + local, params);
      }
    }
    MarkBufFree(0);
  }

  __gm__ DT_X *xGm;
  __gm__ DT_X *yGm;
};

namespace {
struct RowPlaneCompactCorePlan {
  uint32_t inputBase;
  uint32_t outputBase;
  uint32_t rows;
};

constexpr RowPlaneCompactCorePlan kRowPlaneCompactCorePlans[20] = {
    {0U, 0U, 6U},          {2688U, 10752U, 6U},
    {5376U, 21504U, 6U},   {8064U, 32256U, 6U},
    {10752U, 43008U, 4U},  {12544U, 50176U, 6U},
    {15232U, 60928U, 6U},  {17920U, 71680U, 6U},
    {20608U, 82432U, 6U},  {23296U, 93184U, 4U},
    {25088U, 100352U, 6U}, {27776U, 111104U, 6U},
    {30464U, 121856U, 6U}, {33152U, 132608U, 6U},
    {35840U, 143360U, 4U}, {37632U, 150528U, 6U},
    {40320U, 161280U, 6U}, {43008U, 172032U, 6U},
    {45696U, 182784U, 6U}, {48384U, 193536U, 4U},
};

struct RowPlaneStridedCorePlan {
  uint32_t inputBase;
  uint32_t outputBase;
};

// One entry per core for the strided row-plane variant; these constants remove
// task decoding, division, modulo, and 4D offset calculation from the kernel.
constexpr RowPlaneStridedCorePlan kRowPlaneStridedCorePlans[28] = {
    {0U, 0U},           {7168U, 28672U},     {14336U, 57344U},
    {21504U, 86016U},   {28672U, 114688U},   {35840U, 143360U},
    {43008U, 172032U},  {50176U, 200704U},   {57344U, 229376U},
    {64512U, 258048U},  {71680U, 286720U},   {78848U, 315392U},
    {86016U, 344064U},  {93184U, 372736U},   {100352U, 401408U},
    {107520U, 430080U}, {114688U, 458752U},  {121856U, 487424U},
    {129024U, 516096U}, {136192U, 544768U},  {143360U, 573440U},
    {150528U, 602112U}, {157696U, 630784U},  {164864U, 659456U},
    {172032U, 688128U}, {179200U, 716800U},  {186368U, 745472U},
    {193536U, 774144U},
};

struct HalfRowBigCorePlan {
  uint32_t oh;
  uint32_t chunkStart;
  uint32_t chunkCount;
};

constexpr HalfRowBigCorePlan kHalfRowBigCorePlans[40] = {
    {0U, 0U, 4U},   {0U, 4U, 4U},   {1U, 0U, 4U},   {1U, 4U, 4U},
    {2U, 0U, 4U},   {2U, 4U, 4U},   {3U, 0U, 4U},   {3U, 4U, 4U},
    {4U, 0U, 4U},   {4U, 4U, 4U},   {5U, 0U, 4U},   {5U, 4U, 4U},
    {6U, 0U, 4U},   {6U, 4U, 4U},   {7U, 0U, 4U},   {7U, 4U, 4U},
    {8U, 0U, 4U},   {8U, 4U, 4U},   {9U, 0U, 4U},   {9U, 4U, 4U},
    {10U, 0U, 4U},  {10U, 4U, 4U},  {11U, 0U, 4U},  {11U, 4U, 4U},
    {12U, 0U, 4U},  {12U, 4U, 4U},  {13U, 0U, 4U},  {13U, 4U, 4U},
    {14U, 0U, 4U},  {14U, 4U, 4U},  {15U, 0U, 4U},  {15U, 4U, 4U},
    {16U, 0U, 4U},  {16U, 4U, 4U},  {17U, 0U, 4U},  {17U, 4U, 4U},
    {18U, 0U, 4U},  {18U, 4U, 4U},  {19U, 0U, 4U},  {19U, 4U, 4U},
};

constexpr uint32_t kHalfRowBigChunkInputA[160] = {
    0U, 16384U, 32768U, 49152U,
    65536U, 81920U, 98304U, 114688U,
    2621440U, 2637824U, 2654208U, 2670592U,
    2686976U, 2703360U, 2719744U, 2736128U,
    131072U, 147456U, 163840U, 180224U,
    196608U, 212992U, 229376U, 245760U,
    2752512U, 2768896U, 2785280U, 2801664U,
    2818048U, 2834432U, 2850816U, 2867200U,
    262144U, 278528U, 294912U, 311296U,
    327680U, 344064U, 360448U, 376832U,
    2883584U, 2899968U, 2916352U, 2932736U,
    2949120U, 2965504U, 2981888U, 2998272U,
    393216U, 409600U, 425984U, 442368U,
    458752U, 475136U, 491520U, 507904U,
    3014656U, 3031040U, 3047424U, 3063808U,
    3080192U, 3096576U, 3112960U, 3129344U,
    524288U, 540672U, 557056U, 573440U,
    589824U, 606208U, 622592U, 638976U,
    3145728U, 3162112U, 3178496U, 3194880U,
    3211264U, 3227648U, 3244032U, 3260416U,
    655360U, 671744U, 688128U, 704512U,
    720896U, 737280U, 753664U, 770048U,
    3276800U, 3293184U, 3309568U, 3325952U,
    3342336U, 3358720U, 3375104U, 3391488U,
    786432U, 802816U, 819200U, 835584U,
    851968U, 868352U, 884736U, 901120U,
    3407872U, 3424256U, 3440640U, 3457024U,
    3473408U, 3489792U, 3506176U, 3522560U,
    917504U, 933888U, 950272U, 966656U,
    983040U, 999424U, 1015808U, 1032192U,
    3538944U, 3555328U, 3571712U, 3588096U,
    3604480U, 3620864U, 3637248U, 3653632U,
    1048576U, 1064960U, 1081344U, 1097728U,
    1114112U, 1130496U, 1146880U, 1163264U,
    3670016U, 3686400U, 3702784U, 3719168U,
    3735552U, 3751936U, 3768320U, 3784704U,
    1179648U, 1196032U, 1212416U, 1228800U,
    1245184U, 1261568U, 1277952U, 1294336U,
    3801088U, 3817472U, 3833856U, 3850240U,
    3866624U, 3883008U, 3899392U, 3915776U,
};

constexpr uint32_t kHalfRowBigChunkInputB[160] = {
    1310720U, 1327104U, 1343488U, 1359872U,
    1376256U, 1392640U, 1409024U, 1425408U,
    3932160U, 3948544U, 3964928U, 3981312U,
    3997696U, 4014080U, 4030464U, 4046848U,
    1441792U, 1458176U, 1474560U, 1490944U,
    1507328U, 1523712U, 1540096U, 1556480U,
    4063232U, 4079616U, 4096000U, 4112384U,
    4128768U, 4145152U, 4161536U, 4177920U,
    1572864U, 1589248U, 1605632U, 1622016U,
    1638400U, 1654784U, 1671168U, 1687552U,
    4194304U, 4210688U, 4227072U, 4243456U,
    4259840U, 4276224U, 4292608U, 4308992U,
    1703936U, 1720320U, 1736704U, 1753088U,
    1769472U, 1785856U, 1802240U, 1818624U,
    4325376U, 4341760U, 4358144U, 4374528U,
    4390912U, 4407296U, 4423680U, 4440064U,
    1835008U, 1851392U, 1867776U, 1884160U,
    1900544U, 1916928U, 1933312U, 1949696U,
    4456448U, 4472832U, 4489216U, 4505600U,
    4521984U, 4538368U, 4554752U, 4571136U,
    1966080U, 1982464U, 1998848U, 2015232U,
    2031616U, 2048000U, 2064384U, 2080768U,
    4587520U, 4603904U, 4620288U, 4636672U,
    4653056U, 4669440U, 4685824U, 4702208U,
    2097152U, 2113536U, 2129920U, 2146304U,
    2162688U, 2179072U, 2195456U, 2211840U,
    4718592U, 4734976U, 4751360U, 4767744U,
    4784128U, 4800512U, 4816896U, 4833280U,
    2228224U, 2244608U, 2260992U, 2277376U,
    2293760U, 2310144U, 2326528U, 2342912U,
    4849664U, 4866048U, 4882432U, 4898816U,
    4915200U, 4931584U, 4947968U, 4964352U,
    2359296U, 2375680U, 2392064U, 2408448U,
    2424832U, 2441216U, 2457600U, 2473984U,
    4980736U, 4997120U, 5013504U, 5029888U,
    5046272U, 5062656U, 5079040U, 5095424U,
    2490368U, 2506752U, 2523136U, 2539520U,
    2555904U, 2572288U, 2588672U, 2605056U,
    5111808U, 5128192U, 5144576U, 5160960U,
    5177344U, 5193728U, 5210112U, 5226496U,
};

constexpr uint32_t kHalfRowBigChunkOutput[160] = {
    0U, 32768U, 65536U, 98304U,
    131072U, 163840U, 196608U, 229376U,
    262144U, 294912U, 327680U, 360448U,
    393216U, 425984U, 458752U, 491520U,
    524288U, 557056U, 589824U, 622592U,
    655360U, 688128U, 720896U, 753664U,
    786432U, 819200U, 851968U, 884736U,
    917504U, 950272U, 983040U, 1015808U,
    1048576U, 1081344U, 1114112U, 1146880U,
    1179648U, 1212416U, 1245184U, 1277952U,
    1310720U, 1343488U, 1376256U, 1409024U,
    1441792U, 1474560U, 1507328U, 1540096U,
    1572864U, 1605632U, 1638400U, 1671168U,
    1703936U, 1736704U, 1769472U, 1802240U,
    1835008U, 1867776U, 1900544U, 1933312U,
    1966080U, 1998848U, 2031616U, 2064384U,
    2097152U, 2129920U, 2162688U, 2195456U,
    2228224U, 2260992U, 2293760U, 2326528U,
    2359296U, 2392064U, 2424832U, 2457600U,
    2490368U, 2523136U, 2555904U, 2588672U,
    2621440U, 2654208U, 2686976U, 2719744U,
    2752512U, 2785280U, 2818048U, 2850816U,
    2883584U, 2916352U, 2949120U, 2981888U,
    3014656U, 3047424U, 3080192U, 3112960U,
    3145728U, 3178496U, 3211264U, 3244032U,
    3276800U, 3309568U, 3342336U, 3375104U,
    3407872U, 3440640U, 3473408U, 3506176U,
    3538944U, 3571712U, 3604480U, 3637248U,
    3670016U, 3702784U, 3735552U, 3768320U,
    3801088U, 3833856U, 3866624U, 3899392U,
    3932160U, 3964928U, 3997696U, 4030464U,
    4063232U, 4096000U, 4128768U, 4161536U,
    4194304U, 4227072U, 4259840U, 4292608U,
    4325376U, 4358144U, 4390912U, 4423680U,
    4456448U, 4489216U, 4521984U, 4554752U,
    4587520U, 4620288U, 4653056U, 4685824U,
    4718592U, 4751360U, 4784128U, 4816896U,
    4849664U, 4882432U, 4915200U, 4947968U,
    4980736U, 5013504U, 5046272U, 5079040U,
    5111808U, 5144576U, 5177344U, 5210112U,
};

struct HFuseCorePlan {
  uint32_t inputBase;
  uint32_t outputBase;
  uint8_t tailPairs;
};

constexpr HFuseCorePlan kHFuseCorePlans[40] = {
    {0U, 0U, 7U},          {19776U, 79104U, 7U},
    {39552U, 158208U, 7U}, {59328U, 237312U, 7U},
    {79104U, 316416U, 6U}, {98688U, 394752U, 6U},
    {118272U, 473088U, 6U}, {137856U, 551424U, 6U},
    {157440U, 629760U, 6U}, {177024U, 708096U, 6U},
    {196608U, 786432U, 7U}, {216384U, 865536U, 7U},
    {236160U, 944640U, 7U}, {255936U, 1023744U, 7U},
    {275712U, 1102848U, 6U}, {295296U, 1181184U, 6U},
    {314880U, 1259520U, 6U}, {334464U, 1337856U, 6U},
    {354048U, 1416192U, 6U}, {373632U, 1494528U, 6U},
    {393216U, 1572864U, 7U}, {412992U, 1651968U, 7U},
    {432768U, 1731072U, 7U}, {452544U, 1810176U, 7U},
    {472320U, 1889280U, 6U}, {491904U, 1967616U, 6U},
    {511488U, 2045952U, 6U}, {531072U, 2124288U, 6U},
    {550656U, 2202624U, 6U}, {570240U, 2280960U, 6U},
    {589824U, 2359296U, 7U}, {609600U, 2438400U, 7U},
    {629376U, 2517504U, 7U}, {649152U, 2596608U, 7U},
    {668928U, 2675712U, 6U}, {688512U, 2754048U, 6U},
    {708096U, 2832384U, 6U}, {727680U, 2910720U, 6U},
    {747264U, 2989056U, 6U}, {766848U, 3067392U, 6U},
};

struct McWideCorePlan {
  uint32_t inputBase0;
  uint32_t inputBase1;
  uint32_t inputBase2;
  uint32_t inputBase3;
  uint32_t outputBase;
};

constexpr McWideCorePlan kMcWideCorePlans[40] = {
    {335872U, 663552U, 991232U, 8256U, 0U},
    {1646592U, 1974272U, 2301952U, 1318976U, 98240U},
    {2957312U, 3284992U, 3612672U, 2629696U, 196480U},
    {4268032U, 4595712U, 4923392U, 3940416U, 294720U},
    {368640U, 696320U, 1024000U, 41024U, 392960U},
    {1679360U, 2007040U, 2334720U, 1351744U, 491200U},
    {2990080U, 3317760U, 3645440U, 2662464U, 589440U},
    {4300800U, 4628480U, 4956160U, 3973184U, 687680U},
    {401408U, 729088U, 1056768U, 73792U, 785920U},
    {1712128U, 2039808U, 2367488U, 1384512U, 884160U},
    {3022848U, 3350528U, 3678208U, 2695232U, 982400U},
    {4333568U, 4661248U, 4988928U, 4005952U, 1080640U},
    {434176U, 761856U, 1089536U, 106560U, 1178880U},
    {1744896U, 2072576U, 2400256U, 1417280U, 1277120U},
    {3055616U, 3383296U, 3710976U, 2728000U, 1375360U},
    {4366336U, 4694016U, 5021696U, 4038720U, 1473600U},
    {466944U, 794624U, 1122304U, 139328U, 1571840U},
    {1777664U, 2105344U, 2433024U, 1450048U, 1670080U},
    {3088384U, 3416064U, 3743744U, 2760768U, 1768320U},
    {4399104U, 4726784U, 5054464U, 4071488U, 1866560U},
    {499712U, 827392U, 1155072U, 172096U, 1964800U},
    {1810432U, 2138112U, 2465792U, 1482816U, 2063040U},
    {3121152U, 3448832U, 3776512U, 2793536U, 2161280U},
    {4431872U, 4759552U, 5087232U, 4104256U, 2259520U},
    {532480U, 860160U, 1187840U, 204864U, 2357760U},
    {1843200U, 2170880U, 2498560U, 1515584U, 2456000U},
    {3153920U, 3481600U, 3809280U, 2826304U, 2554240U},
    {4464640U, 4792320U, 5120000U, 4137024U, 2652480U},
    {565248U, 892928U, 1220608U, 237632U, 2750720U},
    {1875968U, 2203648U, 2531328U, 1548352U, 2848960U},
    {3186688U, 3514368U, 3842048U, 2859072U, 2947200U},
    {4497408U, 4825088U, 5152768U, 4169792U, 3045440U},
    {598016U, 925696U, 1253376U, 270400U, 3143680U},
    {1908736U, 2236416U, 2564096U, 1581120U, 3241920U},
    {3219456U, 3547136U, 3874816U, 2891840U, 3340160U},
    {4530176U, 4857856U, 5185536U, 4202560U, 3438400U},
    {630784U, 958464U, 1286144U, 303168U, 3536640U},
    {1941504U, 2269184U, 2596864U, 1613888U, 3634880U},
    {3252224U, 3579904U, 3907584U, 2924608U, 3733120U},
    {4562944U, 4890624U, 5218304U, 4235328U, 3831360U},
};

} // namespace

template <class DT_X> class KernelBatchToSpaceRowPlaneCompact {
public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
    xGm = reinterpret_cast<__gm__ DT_X *>(x);
    yGm = reinterpret_cast<__gm__ DT_X *>(y);
  }

  __aicore__ inline void Process() {
    const RowPlaneCompactCorePlan &plan =
        kRowPlaneCompactCorePlans[static_cast<uint32_t>(AscendC::GetBlockIdx())];
    __ubuf__ DT_X *tileAddr = UbPtr<DT_X>();

    if (plan.rows == kFullRows) {
      CopyTile<true>(tileAddr, plan.inputBase, plan.outputBase);
    } else {
      CopyTile<false>(tileAddr, plan.inputBase, plan.outputBase);
    }
  }

private:
  static constexpr uint32_t kDepth = 64;
  static constexpr uint32_t kInputRowElems = 14 * kDepth;
  static constexpr uint32_t kInputBatchElems = 14 * kInputRowElems;
  static constexpr uint32_t kPhaseBatchStride = 4 * kInputBatchElems;
  static constexpr uint32_t kOutputRowElems = 28 * kDepth;
  static constexpr uint32_t kFullRows = 6;
  static constexpr uint32_t kTileElems = kFullRows * kOutputRowElems;
  static constexpr uint32_t kDepthBlocks =
      kDepth * sizeof(float) / BTS_DATA_BLOCK_BYTES;
  static constexpr uint32_t kOutputLaneGapBlocks = kDepthBlocks;
  static constexpr uint32_t kOutputRowBlocks =
      kOutputRowElems * sizeof(float) / BTS_DATA_BLOCK_BYTES;

  template <bool FULL, uint32_t LOCAL_BASE>
  __aicore__ inline void CopyPhase(__ubuf__ DT_X *tileAddr,
                                   uint32_t inputBase) const {
    copy_gm_to_ubuf((__ubuf__ void *)(tileAddr + LOCAL_BASE),
                    (__gm__ void *)(xGm + inputBase), 0, 14, kDepthBlocks, 0,
                    kOutputLaneGapBlocks);
    copy_gm_to_ubuf(
        (__ubuf__ void *)(tileAddr + LOCAL_BASE + 2 * kOutputRowElems),
        (__gm__ void *)(xGm + inputBase + kInputRowElems), 0, 14, kDepthBlocks,
        0, kOutputLaneGapBlocks);
    if constexpr (FULL) {
      copy_gm_to_ubuf(
          (__ubuf__ void *)(tileAddr + LOCAL_BASE + 4 * kOutputRowElems),
          (__gm__ void *)(xGm + inputBase + 2 * kInputRowElems), 0, 14,
          kDepthBlocks, 0, kOutputLaneGapBlocks);
    }
  }

  template <bool FULL>
  __aicore__ inline void CopyTile(__ubuf__ DT_X *tileAddr,
                                  uint32_t inputBase,
                                  uint32_t outputBase) const {
    CopyPhase<FULL, 0>(tileAddr, inputBase);
    CopyPhase<FULL, kDepth>(tileAddr,
                            inputBase + kPhaseBatchStride);
    CopyPhase<FULL, kOutputRowElems>(
        tileAddr, inputBase + 2 * kPhaseBatchStride);
    CopyPhase<FULL, kOutputRowElems + kDepth>(
        tileAddr, inputBase + 3 * kPhaseBatchStride);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(
        BTS_EVENT_READY_BUF0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(
        BTS_EVENT_READY_BUF0);
    constexpr uint32_t kRows = FULL ? kFullRows : 4U;
    copy_ubuf_to_gm((__gm__ void *)(yGm + outputBase),
                    (__ubuf__ void *)tileAddr, 0, 1,
                    kRows * kOutputRowBlocks, 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(BTS_EVENT_FREE_BUF0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(BTS_EVENT_FREE_BUF0);
  }

  __gm__ DT_X *xGm;
  __gm__ DT_X *yGm;
};

template <class DT_X> class KernelBatchToSpaceRowPlaneSmall {
public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
    xGm = reinterpret_cast<__gm__ DT_X *>(x);
    yGm = reinterpret_cast<__gm__ DT_X *>(y);
  }

  __aicore__ inline void Process() {
    const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    if (blockIdx >= kOutputBatch) {
      return;
    }
    const uint32_t inputBatchBase = blockIdx * kInputBatchElems;
    const uint32_t outputBase = blockIdx * kTileElems;
    __ubuf__ DT_X *tileAddr = UbPtr<DT_X>();

    CopyLane<0, 0>(tileAddr, inputBatchBase);
    CopyLane<32, 3840>(tileAddr, inputBatchBase);
    CopyLane<384, 7680>(tileAddr, inputBatchBase);
    CopyLane<416, 11520>(tileAddr, inputBatchBase);
    CopyLane<768, 192>(tileAddr, inputBatchBase);
    CopyLane<800, 4032>(tileAddr, inputBatchBase);
    CopyLane<1152, 7872>(tileAddr, inputBatchBase);
    CopyLane<1184, 11712>(tileAddr, inputBatchBase);
    CopyLane<1536, 384>(tileAddr, inputBatchBase);
    CopyLane<1568, 4224>(tileAddr, inputBatchBase);
    CopyLane<1920, 8064>(tileAddr, inputBatchBase);
    CopyLane<1952, 11904>(tileAddr, inputBatchBase);
    CopyLane<2304, 576>(tileAddr, inputBatchBase);
    CopyLane<2336, 4416>(tileAddr, inputBatchBase);
    CopyLane<2688, 8256>(tileAddr, inputBatchBase);
    CopyLane<2720, 12096>(tileAddr, inputBatchBase);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(
        BTS_EVENT_READY_BUF0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(
        BTS_EVENT_READY_BUF0);
    copy_ubuf_to_gm((__gm__ void *)(yGm + outputBase),
                    (__ubuf__ void *)tileAddr, 0, 1, kTileBlocks, 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(BTS_EVENT_FREE_BUF0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(BTS_EVENT_FREE_BUF0);
  }

private:
  static constexpr uint32_t kOutputBatch = 5;
  static constexpr uint32_t kTileElems = 8 * 12 * 32;
  static constexpr uint32_t kInputBatchElems = 4 * 6 * 32;
  static constexpr uint32_t kTileBlocks =
      kTileElems * sizeof(float) / BTS_DATA_BLOCK_BYTES;

  template <uint32_t LOCAL_OFFSET, uint32_t INPUT_DELTA>
  __aicore__ inline void CopyLane(__ubuf__ DT_X *tileAddr,
                                  uint32_t inputBatchBase) const {
    copy_gm_to_ubuf((__ubuf__ void *)(tileAddr + LOCAL_OFFSET),
                    (__gm__ void *)(xGm + inputBatchBase + INPUT_DELTA), 0, 6,
                    4, 0, 4);
  }

  __gm__ DT_X *xGm;
  __gm__ DT_X *yGm;
};

template <class DT_X> class KernelBatchToSpaceDepthSplitLarge {
public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
    xGm = reinterpret_cast<__gm__ DT_X *>(x);
    yGm = reinterpret_cast<__gm__ DT_X *>(y);
  }

  __aicore__ inline void Process() {
    const uint32_t depthOffset =
        static_cast<uint32_t>(AscendC::GetBlockIdx()) * kTileDepth;
    __ubuf__ DT_X *tileAddr = UbPtr<DT_X>();

    CopyLane<0, 0>(tileAddr, depthOffset);
    CopyLane<512, 16384>(tileAddr, depthOffset);
    CopyLane<2048, 32768>(tileAddr, depthOffset);
    CopyLane<2560, 49152>(tileAddr, depthOffset);
    CopyLane<4096, 8192>(tileAddr, depthOffset);
    CopyLane<4608, 24576>(tileAddr, depthOffset);
    CopyLane<6144, 40960>(tileAddr, depthOffset);
    CopyLane<6656, 57344>(tileAddr, depthOffset);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(
        BTS_EVENT_READY_BUF0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(
        BTS_EVENT_READY_BUF0);
    copy_ubuf_to_gm((__gm__ void *)(yGm + depthOffset),
                    (__ubuf__ void *)tileAddr, 0, 16, kTileDepthBlocks, 0,
                    kOutputDepthGapBlocks);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(BTS_EVENT_FREE_BUF0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(BTS_EVENT_FREE_BUF0);
  }

private:
  static constexpr uint32_t kTileDepth = 512;
  static constexpr uint32_t kTileElems = 16 * kTileDepth;
  static constexpr uint32_t kTileDepthBlocks =
      kTileDepth * sizeof(uint16_t) / BTS_DATA_BLOCK_BYTES;
  static constexpr uint32_t kOutputDepthGapBlocks =
      (4096 - kTileDepth) * sizeof(uint16_t) / BTS_DATA_BLOCK_BYTES;

  template <uint32_t LOCAL_OFFSET, uint32_t INPUT_BASE>
  __aicore__ inline void CopyLane(__ubuf__ DT_X *tileAddr,
                                  uint32_t depthOffset) const {
    copy_gm_to_ubuf((__ubuf__ void *)(tileAddr + LOCAL_OFFSET),
                    (__gm__ void *)(xGm + INPUT_BASE + depthOffset), 0, 2,
                    kTileDepthBlocks, kInputDepthGapBlocks,
                    kOutputLaneGapBlocks);
  }

  static constexpr uint32_t kInputDepthGapBlocks =
      (4096 - kTileDepth) * sizeof(uint16_t) / BTS_DATA_BLOCK_BYTES;
  static constexpr uint32_t kOutputLaneGapBlocks =
      kTileDepth * sizeof(uint16_t) / BTS_DATA_BLOCK_BYTES;
  __gm__ DT_X *xGm;
  __gm__ DT_X *yGm;
};

template <class DT_X> class KernelBatchToSpaceDepthSplitTiny {
public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
    xGm = reinterpret_cast<__gm__ DT_X *>(x);
    yGm = reinterpret_cast<__gm__ DT_X *>(y);
  }

  __aicore__ inline void Process() {
    if (xGm == yGm) {
      return;
    }
    const uint32_t depthOffset =
        static_cast<uint32_t>(AscendC::GetBlockIdx()) * kTileDepth;
    __ubuf__ DT_X *tileAddr = UbPtr<DT_X>();

    copy_gm_to_ubuf((__ubuf__ void *)tileAddr,
                    (__gm__ void *)(xGm + depthOffset), 0, 4,
                    kTileDepthBlocks, kOutputDepthGapBlocks, 0);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(
        BTS_EVENT_READY_BUF0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(
        BTS_EVENT_READY_BUF0);
    copy_ubuf_to_gm((__gm__ void *)(yGm + depthOffset),
                    (__ubuf__ void *)tileAddr, 0, 4, kTileDepthBlocks, 0,
                    kOutputDepthGapBlocks);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(BTS_EVENT_FREE_BUF0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(BTS_EVENT_FREE_BUF0);
  }

private:
  static constexpr uint32_t kTileDepth = 2048;
  static constexpr uint32_t kTileElems = 4 * kTileDepth;
  static constexpr uint32_t kTileDepthBlocks =
      kTileDepth * sizeof(uint16_t) / BTS_DATA_BLOCK_BYTES;
  static constexpr uint32_t kOutputDepthGapBlocks =
      (16384 - kTileDepth) * sizeof(uint16_t) / BTS_DATA_BLOCK_BYTES;

  __gm__ DT_X *xGm;
  __gm__ DT_X *yGm;
};

// Wide fp16 rows of one full GM burst per pixel: two cores per output row,
// half-row tiles moved as gap-DMA width chunks.
template <class DT_X> class KernelBatchToSpaceHalfRowBig {
public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
    xGm = reinterpret_cast<__gm__ DT_X *>(x);
    yGm = reinterpret_cast<__gm__ DT_X *>(y);
  }

  __aicore__ inline void Process() {
    if (sizeof(DT_X) != sizeof(uint16_t)) {
      return;
    }
    const HalfRowBigCorePlan &plan =
        kHalfRowBigCorePlans[static_cast<uint32_t>(AscendC::GetBlockIdx())];
    if (plan.chunkCount == 0) {
      return;
    }
    __ubuf__ DT_X *tileAddr = UbPtr<DT_X>();

    const uint32_t rowBase = plan.oh * kChunksPerRow;
    for (uint32_t i = 0; i < plan.chunkCount; ++i) {
      CopyChunk(tileAddr, rowBase + plan.chunkStart + i);
    }
  }

private:
  static constexpr uint32_t kDepth = 256;
  static constexpr uint32_t kTileOW = 128;
  static constexpr uint32_t kChunksPerRow = 8;
  static constexpr uint32_t kLanePx = kTileOW >> 1;
  static constexpr uint32_t kTileElems = kTileOW * kDepth;
  static constexpr uint32_t kDepthBlocks =
      kDepth * sizeof(uint16_t) / BTS_DATA_BLOCK_BYTES;
  static constexpr uint32_t kOutputPairGapBlocks = kDepthBlocks;

  __aicore__ inline void CopyChunk(__ubuf__ DT_X *tileAddr,
                                   uint32_t chunkIdx) const {
    const uint32_t inputA = kHalfRowBigChunkInputA[chunkIdx];
    const uint32_t inputB = kHalfRowBigChunkInputB[chunkIdx];
    const uint32_t output = kHalfRowBigChunkOutput[chunkIdx];

    copy_gm_to_ubuf((__ubuf__ void *)tileAddr,
                    (__gm__ void *)(xGm + inputA), 0, kLanePx, kDepthBlocks, 0,
                    kOutputPairGapBlocks);
    copy_gm_to_ubuf((__ubuf__ void *)(tileAddr + kDepth),
                    (__gm__ void *)(xGm + inputB), 0, kLanePx, kDepthBlocks, 0,
                    kOutputPairGapBlocks);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(BTS_EVENT_READY_BUF0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(BTS_EVENT_READY_BUF0);
    copy_ubuf_to_gm((__gm__ void *)(yGm + output), (__ubuf__ void *)tileAddr, 0,
                    1, kTileOW * kDepthBlocks, 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(BTS_EVENT_FREE_BUF0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(BTS_EVENT_FREE_BUF0);
  }

  __gm__ DT_X *xGm;
  __gm__ DT_X *yGm;
};

// Wide fp16 sub-burst depth: one output row per core, full-width chunks plus
// one short tail chunk per row.
template <class DT_X> class KernelBatchToSpaceMcWide {
public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
    xGm = reinterpret_cast<__gm__ DT_X *>(x);
    yGm = reinterpret_cast<__gm__ DT_X *>(y);
  }

  __aicore__ inline void Process() {
    if constexpr (sizeof(DT_X) != sizeof(uint16_t)) {
      return;
    } else {
      const McWideCorePlan &plan =
          kMcWideCorePlans[static_cast<uint32_t>(AscendC::GetBlockIdx())];
      MarkBufFree(0);
      MarkBufFree(1);

      CopyIn<kLaneBlocks>(plan, 0U, 0);
      WaitMte2ReadyForV(0);
      Compute<kLanePx, kLanePx, kLanePx, kLanePx>(0);
      CopyIn<kLaneBlocks>(plan, kLaneElems, 1);
      WaitVReadyForMte3(0);
      CopyOut<kFullOutBlocks>(plan.outputBase, 0);
      MarkBufFree(0);

      WaitMte2ReadyForV(1);
      Compute<kLanePx, kLanePx, kLanePx, kLanePx>(1);
      CopyIn<kLaneBlocks>(plan, 2U * kLaneElems, 0);
      WaitVReadyForMte3(1);
      CopyOut<kFullOutBlocks>(plan.outputBase + kFullOutElems, 1);
      MarkBufFree(1);

      WaitMte2ReadyForV(0);
      Compute<kLanePx, kLanePx, kLanePx, kLanePx>(0);
      CopyIn<kTailLane3Blocks>(plan, 3U * kLaneElems, 1);
      WaitVReadyForMte3(0);
      CopyOut<kFullOutBlocks>(plan.outputBase + 2U * kFullOutElems, 0);
      MarkBufFree(0);

      WaitMte2ReadyForV(1);
      Compute<kLanePx, kLanePx, kLanePx, kTailLane3Px>(1);
      WaitVReadyForMte3(1);
      CopyOut<kTailOutBlocks>(plan.outputBase + 3U * kFullOutElems, 1);
      MarkBufFree(1);

      WaitBufFree(0);
      WaitBufFree(1);
    }
  }

private:
  static constexpr uint32_t kDepth = 64;
  static constexpr uint32_t kTileOW = 384;
  static constexpr uint32_t kTailOW = 383;
  static constexpr uint32_t kLanePx = kTileOW / 4;
  static constexpr uint32_t kTailLane3Px = 95;
  static constexpr uint32_t kLaneElems = kLanePx * kDepth;
  static constexpr uint32_t kTailLane3Elems = kTailLane3Px * kDepth;
  static constexpr uint32_t kFullOutElems = kTileOW * kDepth;
  static constexpr uint32_t kTailOutElems = kTailOW * kDepth;
  static constexpr uint32_t kOutBase = 4 * kLaneElems;
  static constexpr uint32_t kSlotElems = 2 * kTileOW * kDepth;
  static constexpr uint32_t kDepthBlocks =
      kDepth * sizeof(uint16_t) / BTS_DATA_BLOCK_BYTES;
  static constexpr uint32_t kLaneBlocks =
      kLaneElems * sizeof(uint16_t) / BTS_DATA_BLOCK_BYTES;
  static constexpr uint32_t kTailLane3Blocks =
      kTailLane3Elems * sizeof(uint16_t) / BTS_DATA_BLOCK_BYTES;
  static constexpr uint32_t kFullOutBlocks =
      kFullOutElems * sizeof(uint16_t) / BTS_DATA_BLOCK_BYTES;
  static constexpr uint32_t kTailOutBlocks =
      kTailOutElems * sizeof(uint16_t) / BTS_DATA_BLOCK_BYTES;

  __aicore__ inline __ubuf__ DT_X *SlotAddr(uint32_t bufIdx) const {
    return UbPtr<DT_X>(bufIdx * kSlotElems * sizeof(DT_X));
  }

  template <uint32_t LANE3_BLOCKS>
  __aicore__ inline void CopyIn(const McWideCorePlan &plan, uint32_t baseStep,
                                uint32_t bufIdx) {
    WaitBufFree(bufIdx);
    __ubuf__ DT_X *slotAddr = SlotAddr(bufIdx);
    copy_gm_to_ubuf((__ubuf__ void *)slotAddr,
                    (__gm__ void *)(xGm + plan.inputBase0 + baseStep), 0, 1,
                    kLaneBlocks, 0, 0);
    copy_gm_to_ubuf((__ubuf__ void *)(slotAddr + kLaneElems),
                    (__gm__ void *)(xGm + plan.inputBase1 + baseStep), 0, 1,
                    kLaneBlocks, 0, 0);
    copy_gm_to_ubuf((__ubuf__ void *)(slotAddr + 2U * kLaneElems),
                    (__gm__ void *)(xGm + plan.inputBase2 + baseStep), 0, 1,
                    kLaneBlocks, 0, 0);
    copy_gm_to_ubuf((__ubuf__ void *)(slotAddr + 3U * kLaneElems),
                    (__gm__ void *)(xGm + plan.inputBase3 + baseStep), 0, 1,
                    LANE3_BLOCKS, 0, 0);
    MarkMte2ReadyForV(bufIdx);
  }

  template <uint32_t COUNT0, uint32_t COUNT1, uint32_t COUNT2,
            uint32_t COUNT3>
  __aicore__ inline void Compute(uint32_t bufIdx) {
    __ubuf__ DT_X *slot = SlotAddr(bufIdx);
    AscendC::AscendCUtils::SetMask<uint16_t>(kDepth);
    vcopy((__ubuf__ uint16_t *)(slot + kOutBase), (__ubuf__ uint16_t *)slot,
          static_cast<uint8_t>(COUNT0), 1, 1,
          static_cast<uint16_t>(4U * kDepthBlocks),
          static_cast<uint16_t>(kDepthBlocks));
    vcopy((__ubuf__ uint16_t *)(slot + kOutBase + kDepth),
          (__ubuf__ uint16_t *)(slot + kLaneElems),
          static_cast<uint8_t>(COUNT1), 1, 1,
          static_cast<uint16_t>(4U * kDepthBlocks),
          static_cast<uint16_t>(kDepthBlocks));
    vcopy((__ubuf__ uint16_t *)(slot + kOutBase + 2U * kDepth),
          (__ubuf__ uint16_t *)(slot + 2U * kLaneElems),
          static_cast<uint8_t>(COUNT2), 1, 1,
          static_cast<uint16_t>(4U * kDepthBlocks),
          static_cast<uint16_t>(kDepthBlocks));
    vcopy((__ubuf__ uint16_t *)(slot + kOutBase + 3U * kDepth),
          (__ubuf__ uint16_t *)(slot + 3U * kLaneElems),
          static_cast<uint8_t>(COUNT3), 1, 1,
          static_cast<uint16_t>(4U * kDepthBlocks),
          static_cast<uint16_t>(kDepthBlocks));
    MarkVReadyForMte3(bufIdx);
  }

  template <uint32_t OUT_BLOCKS>
  __aicore__ inline void CopyOut(uint32_t outputBase, uint32_t bufIdx) const {
    __ubuf__ DT_X *slotAddr = SlotAddr(bufIdx);
    copy_ubuf_to_gm((__gm__ void *)(yGm + outputBase),
                    (__ubuf__ void *)(slotAddr + kOutBase), 0, 1, OUT_BLOCKS,
                    0, 0);
  }

  __gm__ DT_X *xGm;
  __gm__ DT_X *yGm;
};

// Half-row big-tile mover (BTS_TILING_MODE_MC_BIGTILE): one output row per
// core split into two single-buffer halves sized to fill the whole UB. Per
// half: blockSize contiguous lane reads -> vcopy interleave -> one contiguous
// write, minimizing MTE op count on latency-bound wide shapes.
template <class DT_X> class KernelBatchToSpaceMcBigTile {
public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
    xGm = reinterpret_cast<__gm__ DT_X *>(x);
    yGm = reinterpret_cast<__gm__ DT_X *>(y);
  }

  __aicore__ inline void Process() {
    if constexpr (sizeof(DT_X) != sizeof(uint16_t)) {
      return;
    } else {
      const uint32_t oh = static_cast<uint32_t>(AscendC::GetBlockIdx());
      const uint32_t ih = oh / kBlock;
      const uint32_t bh = oh % kBlock;
      const uint32_t outRowBase = oh * kOutRowElems;
      MarkBufFree(0);
      ProcessHalf<kHalf0OW>(ih, bh, 0U, outRowBase);
      ProcessHalf<kHalf1OW>(ih, bh, kHalf0OW, outRowBase + kHalf0OW * kDepth);
      WaitBufFree(0);
    }
  }

private:
  static constexpr uint32_t kBlock = 4;
  static constexpr uint32_t kDepth = 64;
  static constexpr uint32_t kCropLeft = 513;
  static constexpr uint32_t kOutWidth = 1535;
  static constexpr uint32_t kHalf0OW = 768;
  static constexpr uint32_t kHalf1OW = kOutWidth - kHalf0OW; // 767
  static constexpr uint32_t kLanePx = 192; // max ceil(curOW/4)
  static constexpr uint32_t kLaneElems = kLanePx * kDepth;
  static constexpr uint32_t kOutBase = kBlock * kLaneElems;
  static constexpr uint32_t kOutRowElems = kOutWidth * kDepth;
  static constexpr uint32_t kInRowElems = 512 * kDepth;
  static constexpr uint32_t kInBatchElems = 10 * kInRowElems;
  static constexpr uint32_t kDepthBlocks =
      kDepth * sizeof(uint16_t) / BTS_DATA_BLOCK_BYTES;

  template <uint32_t CUR_OW>
  __aicore__ inline void ProcessHalf(uint32_t ih, uint32_t bh,
                                     uint32_t owStart, uint32_t outBase) {
    __ubuf__ DT_X *slot = UbPtr<DT_X>();
    const uint32_t residue = (owStart + kCropLeft) % kBlock;
    WaitBufFree(0);
    for (uint32_t delta = 0; delta < kBlock; ++delta) {
      const uint32_t bw = (delta + residue) % kBlock;
      const uint32_t count = (CUR_OW - delta + kBlock - 1) / kBlock;
      const uint32_t iw = (owStart + delta + kCropLeft) / kBlock;
      const uint32_t src =
          (bh * kBlock + bw) * kInBatchElems + ih * kInRowElems + iw * kDepth;
      copy_gm_to_ubuf((__ubuf__ void *)(slot + delta * kLaneElems),
                      (__gm__ void *)(xGm + src), 0, 1,
                      count * kDepthBlocks, 0, 0);
    }
    MarkMte2ReadyForV(0);
    WaitMte2ReadyForV(0);
    AscendC::AscendCUtils::SetMask<uint16_t>(kDepth);
    for (uint32_t delta = 0; delta < kBlock; ++delta) {
      const uint32_t count = (CUR_OW - delta + kBlock - 1) / kBlock;
      vcopy((__ubuf__ uint16_t *)(slot + kOutBase + delta * kDepth),
            (__ubuf__ uint16_t *)(slot + delta * kLaneElems),
            static_cast<uint8_t>(count), 1, 1,
            static_cast<uint16_t>(kBlock * kDepthBlocks),
            static_cast<uint16_t>(kDepthBlocks));
    }
    MarkVReadyForMte3(0);
    WaitVReadyForMte3(0);
    copy_ubuf_to_gm((__gm__ void *)(yGm + outBase),
                    (__ubuf__ void *)(slot + kOutBase), 0, 1,
                    CUR_OW * kDepthBlocks, 0, 0);
    MarkBufFree(0);
  }

  __gm__ DT_X *xGm;
  __gm__ DT_X *yGm;
};

template <class DT_X> class KernelBatchToSpaceHFuse {
public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
    xGm = AscendC::L2CacheAlter<DT_X, AscendC::CacheRwMode::RW>(
        reinterpret_cast<__gm__ DT_X *>(x),
        AscendC::CacheMode::CACHE_MODE_DISABLE);
    yGm = AscendC::L2CacheAlter<DT_X, AscendC::CacheRwMode::RW>(
        reinterpret_cast<__gm__ DT_X *>(y),
        AscendC::CacheMode::CACHE_MODE_DISABLE);
  }

  __aicore__ inline void Process() {
    if constexpr (sizeof(DT_X) != sizeof(uint16_t)) {
      return;
    } else {
      const HFuseCorePlan &plan =
          kHFuseCorePlans[static_cast<uint32_t>(AscendC::GetBlockIdx())];
      MarkBufFree(0);
      MarkBufFree(1);
      ProcessFullChunks<6>(plan);
      if (plan.tailPairs == 7U) {
        ProcessTail<7>(plan, 6U, 0U);
      } else {
        ProcessTail<6>(plan, 6U, 0U);
      }
      WaitBufFree(0);
      WaitBufFree(1);
    }
  }

private:
  static constexpr uint32_t kInputWidth = 6;
  static constexpr uint32_t kDepth = 32;
  static constexpr uint32_t kFullPairs = 16;
  static constexpr uint32_t kInputRowElems = kInputWidth * kDepth;
  static constexpr uint32_t kInputBatchElems = 1024 * kInputRowElems;
  static constexpr uint32_t kPhaseBatchStride = 4 * kInputBatchElems;
  static constexpr uint32_t kOutputRowElems = 12 * kDepth;
  static constexpr uint32_t kLaneElems = kFullPairs * kInputRowElems;
  static constexpr uint32_t kInElems = 4 * kLaneElems;
  static constexpr uint32_t kOutBase = kInElems;
  static constexpr uint32_t kSlotElems = 2 * 32 * 12 * 32;
  static constexpr uint32_t kDepthBlocks =
      kDepth * sizeof(uint16_t) / BTS_DATA_BLOCK_BYTES;

  __aicore__ inline __ubuf__ DT_X *SlotAddr(uint32_t bufIdx) const {
    return UbPtr<DT_X>(bufIdx * kSlotElems * sizeof(DT_X));
  }

  template <uint32_t PAIRS>
  __aicore__ inline void CopyIn(uint32_t inputBase, uint32_t bufIdx) {
    WaitBufFree(bufIdx);
    __ubuf__ DT_X *slotAddr = SlotAddr(bufIdx);
    constexpr uint32_t blocks =
        PAIRS * kInputRowElems * sizeof(uint16_t) / BTS_DATA_BLOCK_BYTES;
    copy_gm_to_ubuf((__ubuf__ void *)slotAddr,
                    (__gm__ void *)(xGm + inputBase), 0, 1, blocks, 0, 0);
    copy_gm_to_ubuf((__ubuf__ void *)(slotAddr + kLaneElems),
                    (__gm__ void *)(xGm + inputBase + kPhaseBatchStride), 0, 1,
                    blocks, 0, 0);
    copy_gm_to_ubuf((__ubuf__ void *)(slotAddr + 2U * kLaneElems),
                    (__gm__ void *)(xGm + inputBase + 2U * kPhaseBatchStride),
                    0, 1, blocks, 0, 0);
    copy_gm_to_ubuf((__ubuf__ void *)(slotAddr + 3U * kLaneElems),
                    (__gm__ void *)(xGm + inputBase + 3U * kPhaseBatchStride),
                    0, 1, blocks, 0, 0);
    MarkMte2ReadyForV(bufIdx);
  }

  template <uint32_t PAIRS>
  __aicore__ inline void Compute(uint32_t bufIdx) {
    __ubuf__ DT_X *slot = SlotAddr(bufIdx);
    AscendC::AscendCUtils::SetMask<uint16_t>(kDepth);
    for (uint32_t bh = 0; bh < 2; ++bh) {
      for (uint32_t bw = 0; bw < 2; ++bw) {
        const uint32_t phase = bh * 2U + bw;
        const uint32_t srcBase = phase * kLaneElems;
        const uint32_t dstBase = bh * kOutputRowElems + bw * kDepth;
        for (uint32_t r = 0; r < PAIRS; ++r) {
          vcopy((__ubuf__ uint16_t *)(slot + kOutBase + dstBase +
                                      r * 2U * kOutputRowElems),
                (__ubuf__ uint16_t *)(slot + srcBase + r * kInputRowElems),
                static_cast<uint8_t>(kInputWidth), 1, 1,
                static_cast<uint16_t>(2U * kDepthBlocks),
                static_cast<uint16_t>(kDepthBlocks));
        }
      }
    }
    MarkVReadyForMte3(bufIdx);
  }

  template <uint32_t PAIRS>
  __aicore__ inline void CopyOut(uint32_t outputBase, uint32_t bufIdx) const {
    __ubuf__ DT_X *slotAddr = SlotAddr(bufIdx);
    constexpr uint32_t blocks =
        PAIRS * 2U * kOutputRowElems * sizeof(uint16_t) / BTS_DATA_BLOCK_BYTES;
    copy_ubuf_to_gm((__gm__ void *)(yGm + outputBase),
                    (__ubuf__ void *)(slotAddr + kOutBase), 0, 1, blocks, 0,
                    0);
  }

  template <uint32_t CHUNKS>
  __aicore__ inline void ProcessFullChunks(const HFuseCorePlan &plan) {
    CopyIn<kFullPairs>(plan.inputBase, 0);
    for (uint32_t i = 0; i < CHUNKS; ++i) {
      const uint32_t curBuf = i & 1U;
      WaitMte2ReadyForV(curBuf);
      Compute<kFullPairs>(curBuf);
      const uint32_t next = i + 1;
      if (next < CHUNKS) {
        CopyIn<kFullPairs>(plan.inputBase + next * kFullPairs * kInputRowElems,
                           next & 1U);
      }
      WaitVReadyForMte3(curBuf);
      CopyOut<kFullPairs>(
          plan.outputBase + i * kFullPairs * 2U * kOutputRowElems, curBuf);
      MarkBufFree(curBuf);
    }
  }

  template <uint32_t PAIRS>
  __aicore__ inline void ProcessTail(const HFuseCorePlan &plan,
                                     uint32_t fullChunks, uint32_t bufIdx) {
    CopyIn<PAIRS>(plan.inputBase + fullChunks * kFullPairs * kInputRowElems,
                  bufIdx);
    WaitMte2ReadyForV(bufIdx);
    Compute<PAIRS>(bufIdx);
    WaitVReadyForMte3(bufIdx);
    CopyOut<PAIRS>(
        plan.outputBase + fullChunks * kFullPairs * 2U * kOutputRowElems,
        bufIdx);
    MarkBufFree(bufIdx);
  }

  __gm__ DT_X *xGm;
  __gm__ DT_X *yGm;
};

template <class DT_X> class KernelBatchToSpaceRowPlaneStrided {
public:
  __aicore__ inline KernelBatchToSpaceRowPlaneStrided() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
    xGm = reinterpret_cast<__gm__ DT_X *>(x);
    yGm = reinterpret_cast<__gm__ DT_X *>(y);
  }

  __aicore__ inline void Process() {
    const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const RowPlaneStridedCorePlan plan = kRowPlaneStridedCorePlans[blockIdx];

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(BTS_EVENT_FREE_BUF0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(BTS_EVENT_FREE_BUF1);

    CopyIn<0>(plan.inputBase);
    CopyIn<1>(plan.inputBase + kInputRowElems);

    CopyOut<0>(plan.outputBase);
    CopyOut<1>(plan.outputBase + kTileElems);

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(BTS_EVENT_FREE_BUF0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(BTS_EVENT_FREE_BUF1);
  }

private:
  static constexpr uint32_t kWidth = 28;
  static constexpr uint32_t kDepth = 128;
  static constexpr uint32_t kOutWidth = 56;
  static constexpr uint32_t kTileOH = 2;
  static constexpr uint32_t kRowElems = kOutWidth * kDepth;
  static constexpr uint32_t kTileElems = kTileOH * kRowElems;
  static constexpr uint32_t kInputRowElems = kWidth * kDepth;
  static constexpr uint32_t kInputBatchElems = 28 * kInputRowElems;
  static constexpr uint32_t kDepthBlocks =
      kDepth * sizeof(float) / BTS_DATA_BLOCK_BYTES;

  template <uint32_t BUF>
  __aicore__ inline void CopyIn(uint32_t inputBase) {
    constexpr int32_t kFreeEvent =
        BUF == 0 ? BTS_EVENT_FREE_BUF0 : BTS_EVENT_FREE_BUF1;
    constexpr int32_t kReadyEvent =
        BUF == 0 ? BTS_EVENT_READY_BUF0 : BTS_EVENT_READY_BUF1;
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kFreeEvent);
    __ubuf__ DT_X *tileAddr =
        UbPtr<DT_X>(BUF * kTileElems * sizeof(DT_X));

    copy_gm_to_ubuf((__ubuf__ void *)tileAddr,
                    (__gm__ void *)(xGm + inputBase), 0, kWidth, kDepthBlocks,
                    0, kDepthBlocks);
    copy_gm_to_ubuf((__ubuf__ void *)(tileAddr + kDepth),
                    (__gm__ void *)(xGm + inputBase +
                                    2 * kInputBatchElems),
                    0, kWidth, kDepthBlocks, 0, kDepthBlocks);
    copy_gm_to_ubuf((__ubuf__ void *)(tileAddr + kRowElems),
                    (__gm__ void *)(xGm + inputBase +
                                    4 * kInputBatchElems),
                    0, kWidth, kDepthBlocks, 0, kDepthBlocks);
    copy_gm_to_ubuf((__ubuf__ void *)(tileAddr + kRowElems + kDepth),
                    (__gm__ void *)(xGm + inputBase +
                                    6 * kInputBatchElems),
                    0, kWidth, kDepthBlocks, 0, kDepthBlocks);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(kReadyEvent);
  }

  template <uint32_t BUF>
  __aicore__ inline void CopyOut(uint32_t outputBase) {
    constexpr int32_t kFreeEvent =
        BUF == 0 ? BTS_EVENT_FREE_BUF0 : BTS_EVENT_FREE_BUF1;
    constexpr int32_t kReadyEvent =
        BUF == 0 ? BTS_EVENT_READY_BUF0 : BTS_EVENT_READY_BUF1;
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(kReadyEvent);
    __ubuf__ DT_X *tileAddr =
        UbPtr<DT_X>(BUF * kTileElems * sizeof(DT_X));
    copy_ubuf_to_gm((__gm__ void *)(yGm + outputBase),
                    (__ubuf__ void *)tileAddr, 0, 1, kTileBlocks, 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(kFreeEvent);
  }

  static constexpr uint32_t kTileBlocks =
      kTileElems * sizeof(float) / BTS_DATA_BLOCK_BYTES;
  __gm__ DT_X *xGm;
  __gm__ DT_X *yGm;
};

namespace {
// Wide-gather variant: precomputed GM element offsets.
constexpr uint64_t kRowGatherWideCopyInOffsetA[508] = {
  3194880ULL, 2134080ULL, 1073280ULL, 12480ULL,
  3203200ULL, 2142400ULL, 1081600ULL, 20800ULL,
  3211520ULL, 2150720ULL, 1089920ULL, 29120ULL,
  3219840ULL, 2159040ULL, 1098240ULL, 37440ULL,
  3228160ULL, 2167360ULL, 1106560ULL, 45760ULL,
  3236480ULL, 2175680ULL, 1114880ULL, 54080ULL,
  3244800ULL, 2184000ULL, 1123200ULL, 62400ULL,
  3253120ULL, 2192320ULL, 1131520ULL, 70720ULL,
  3261440ULL, 2200640ULL, 1139840ULL, 79040ULL,
  3269760ULL, 2208960ULL, 1148160ULL, 87360ULL,
  3278080ULL, 2217280ULL, 1156480ULL, 95680ULL,
  3286400ULL, 2225600ULL, 1164800ULL, 104000ULL,
  3294720ULL, 2233920ULL, 1173120ULL, 112320ULL,
  3303040ULL, 2242240ULL, 1181440ULL, 120640ULL,
  3311360ULL, 2250560ULL, 1189760ULL, 128960ULL,
  3319680ULL, 2258880ULL, 1198080ULL, 137280ULL,
  3328000ULL, 2267200ULL, 1206400ULL, 145600ULL,
  3336320ULL, 2275520ULL, 1214720ULL, 153920ULL,
  3344640ULL, 2283840ULL, 1223040ULL, 162240ULL,
  3352960ULL, 2292160ULL, 1231360ULL, 170560ULL,
  3361280ULL, 2300480ULL, 1239680ULL, 178880ULL,
  3369600ULL, 2308800ULL, 1248000ULL, 187200ULL,
  3377920ULL, 2317120ULL, 1256320ULL, 195520ULL,
  3386240ULL, 2325440ULL, 1264640ULL, 203840ULL,
  3394560ULL, 2333760ULL, 1272960ULL, 212160ULL,
  3402880ULL, 2342080ULL, 1281280ULL, 220480ULL,
  3411200ULL, 2350400ULL, 1289600ULL, 228800ULL,
  3419520ULL, 2358720ULL, 1297920ULL, 237120ULL,
  3427840ULL, 2367040ULL, 1306240ULL, 245440ULL,
  3436160ULL, 2375360ULL, 1314560ULL, 253760ULL,
  3444480ULL, 2383680ULL, 1322880ULL, 262080ULL,
  3452800ULL, 2392000ULL, 1331200ULL, 270400ULL,
  3461120ULL, 2400320ULL, 1339520ULL, 278720ULL,
  3469440ULL, 2408640ULL, 1347840ULL, 287040ULL,
  3477760ULL, 2416960ULL, 1356160ULL, 295360ULL,
  3486080ULL, 2425280ULL, 1364480ULL, 303680ULL,
  3494400ULL, 2433600ULL, 1372800ULL, 312000ULL,
  3502720ULL, 2441920ULL, 1381120ULL, 320320ULL,
  3511040ULL, 2450240ULL, 1389440ULL, 328640ULL,
  3519360ULL, 2458560ULL, 1397760ULL, 336960ULL,
  3527680ULL, 2466880ULL, 1406080ULL, 345280ULL,
  3536000ULL, 2475200ULL, 1414400ULL, 353600ULL,
  3544320ULL, 2483520ULL, 1422720ULL, 361920ULL,
  3552640ULL, 2491840ULL, 1431040ULL, 370240ULL,
  3560960ULL, 2500160ULL, 1439360ULL, 378560ULL,
  3569280ULL, 2508480ULL, 1447680ULL, 386880ULL,
  3577600ULL, 2516800ULL, 1456000ULL, 395200ULL,
  3585920ULL, 2525120ULL, 1464320ULL, 403520ULL,
  3594240ULL, 2533440ULL, 1472640ULL, 411840ULL,
  3602560ULL, 2541760ULL, 1480960ULL, 420160ULL,
  3610880ULL, 2550080ULL, 1489280ULL, 428480ULL,
  3619200ULL, 2558400ULL, 1497600ULL, 436800ULL,
  3627520ULL, 2566720ULL, 1505920ULL, 445120ULL,
  3635840ULL, 2575040ULL, 1514240ULL, 453440ULL,
  3644160ULL, 2583360ULL, 1522560ULL, 461760ULL,
  3652480ULL, 2591680ULL, 1530880ULL, 470080ULL,
  3660800ULL, 2600000ULL, 1539200ULL, 478400ULL,
  3669120ULL, 2608320ULL, 1547520ULL, 486720ULL,
  3677440ULL, 2616640ULL, 1555840ULL, 495040ULL,
  3685760ULL, 2624960ULL, 1564160ULL, 503360ULL,
  3694080ULL, 2633280ULL, 1572480ULL, 511680ULL,
  3702400ULL, 2641600ULL, 1580800ULL, 520000ULL,
  3710720ULL, 2649920ULL, 1589120ULL, 528320ULL,
  3719040ULL, 2658240ULL, 1597440ULL, 536640ULL,
  3727360ULL, 2666560ULL, 1605760ULL, 544960ULL,
  3735680ULL, 2674880ULL, 1614080ULL, 553280ULL,
  3744000ULL, 2683200ULL, 1622400ULL, 561600ULL,
  3752320ULL, 2691520ULL, 1630720ULL, 569920ULL,
  3760640ULL, 2699840ULL, 1639040ULL, 578240ULL,
  3768960ULL, 2708160ULL, 1647360ULL, 586560ULL,
  3777280ULL, 2716480ULL, 1655680ULL, 594880ULL,
  3785600ULL, 2724800ULL, 1664000ULL, 603200ULL,
  3793920ULL, 2733120ULL, 1672320ULL, 611520ULL,
  3802240ULL, 2741440ULL, 1680640ULL, 619840ULL,
  3810560ULL, 2749760ULL, 1688960ULL, 628160ULL,
  3818880ULL, 2758080ULL, 1697280ULL, 636480ULL,
  3827200ULL, 2766400ULL, 1705600ULL, 644800ULL,
  3835520ULL, 2774720ULL, 1713920ULL, 653120ULL,
  3843840ULL, 2783040ULL, 1722240ULL, 661440ULL,
  3852160ULL, 2791360ULL, 1730560ULL, 669760ULL,
  3860480ULL, 2799680ULL, 1738880ULL, 678080ULL,
  3868800ULL, 2808000ULL, 1747200ULL, 686400ULL,
  3877120ULL, 2816320ULL, 1755520ULL, 694720ULL,
  3885440ULL, 2824640ULL, 1763840ULL, 703040ULL,
  3893760ULL, 2832960ULL, 1772160ULL, 711360ULL,
  3902080ULL, 2841280ULL, 1780480ULL, 719680ULL,
  3910400ULL, 2849600ULL, 1788800ULL, 728000ULL,
  3918720ULL, 2857920ULL, 1797120ULL, 736320ULL,
  3927040ULL, 2866240ULL, 1805440ULL, 744640ULL,
  3935360ULL, 2874560ULL, 1813760ULL, 752960ULL,
  3943680ULL, 2882880ULL, 1822080ULL, 761280ULL,
  3952000ULL, 2891200ULL, 1830400ULL, 769600ULL,
  3960320ULL, 2899520ULL, 1838720ULL, 777920ULL,
  3968640ULL, 2907840ULL, 1847040ULL, 786240ULL,
  3976960ULL, 2916160ULL, 1855360ULL, 794560ULL,
  3985280ULL, 2924480ULL, 1863680ULL, 802880ULL,
  3993600ULL, 2932800ULL, 1872000ULL, 811200ULL,
  4001920ULL, 2941120ULL, 1880320ULL, 819520ULL,
  4010240ULL, 2949440ULL, 1888640ULL, 827840ULL,
  4018560ULL, 2957760ULL, 1896960ULL, 836160ULL,
  4026880ULL, 2966080ULL, 1905280ULL, 844480ULL,
  4035200ULL, 2974400ULL, 1913600ULL, 852800ULL,
  4043520ULL, 2982720ULL, 1921920ULL, 861120ULL,
  4051840ULL, 2991040ULL, 1930240ULL, 869440ULL,
  4060160ULL, 2999360ULL, 1938560ULL, 877760ULL,
  4068480ULL, 3007680ULL, 1946880ULL, 886080ULL,
  4076800ULL, 3016000ULL, 1955200ULL, 894400ULL,
  4085120ULL, 3024320ULL, 1963520ULL, 902720ULL,
  4093440ULL, 3032640ULL, 1971840ULL, 911040ULL,
  4101760ULL, 3040960ULL, 1980160ULL, 919360ULL,
  4110080ULL, 3049280ULL, 1988480ULL, 927680ULL,
  4118400ULL, 3057600ULL, 1996800ULL, 936000ULL,
  4126720ULL, 3065920ULL, 2005120ULL, 944320ULL,
  4135040ULL, 3074240ULL, 2013440ULL, 952640ULL,
  4143360ULL, 3082560ULL, 2021760ULL, 960960ULL,
  4151680ULL, 3090880ULL, 2030080ULL, 969280ULL,
  4160000ULL, 3099200ULL, 2038400ULL, 977600ULL,
  4168320ULL, 3107520ULL, 2046720ULL, 985920ULL,
  4176640ULL, 3115840ULL, 2055040ULL, 994240ULL,
  4184960ULL, 3124160ULL, 2063360ULL, 1002560ULL,
  4193280ULL, 3132480ULL, 2071680ULL, 1010880ULL,
  4201600ULL, 3140800ULL, 2080000ULL, 1019200ULL,
  4209920ULL, 3149120ULL, 2088320ULL, 1027520ULL,
  4218240ULL, 3157440ULL, 2096640ULL, 1035840ULL,
  4226560ULL, 3165760ULL, 2104960ULL, 1044160ULL,
  4234880ULL, 3174080ULL, 2113280ULL, 1052480ULL,
  4243200ULL, 3182400ULL, 2121600ULL, 1060800ULL,
};

constexpr uint64_t kRowGatherWideCopyInOffsetB[508] = {
  2129985ULL, 3199040ULL, 8385ULL, 1077440ULL,
  2138305ULL, 3207360ULL, 16705ULL, 1085760ULL,
  2146625ULL, 3215680ULL, 25025ULL, 1094080ULL,
  2154945ULL, 3224000ULL, 33345ULL, 1102400ULL,
  2163265ULL, 3232320ULL, 41665ULL, 1110720ULL,
  2171585ULL, 3240640ULL, 49985ULL, 1119040ULL,
  2179905ULL, 3248960ULL, 58305ULL, 1127360ULL,
  2188225ULL, 3257280ULL, 66625ULL, 1135680ULL,
  2196545ULL, 3265600ULL, 74945ULL, 1144000ULL,
  2204865ULL, 3273920ULL, 83265ULL, 1152320ULL,
  2213185ULL, 3282240ULL, 91585ULL, 1160640ULL,
  2221505ULL, 3290560ULL, 99905ULL, 1168960ULL,
  2229825ULL, 3298880ULL, 108225ULL, 1177280ULL,
  2238145ULL, 3307200ULL, 116545ULL, 1185600ULL,
  2246465ULL, 3315520ULL, 124865ULL, 1193920ULL,
  2254785ULL, 3323840ULL, 133185ULL, 1202240ULL,
  2263105ULL, 3332160ULL, 141505ULL, 1210560ULL,
  2271425ULL, 3340480ULL, 149825ULL, 1218880ULL,
  2279745ULL, 3348800ULL, 158145ULL, 1227200ULL,
  2288065ULL, 3357120ULL, 166465ULL, 1235520ULL,
  2296385ULL, 3365440ULL, 174785ULL, 1243840ULL,
  2304705ULL, 3373760ULL, 183105ULL, 1252160ULL,
  2313025ULL, 3382080ULL, 191425ULL, 1260480ULL,
  2321345ULL, 3390400ULL, 199745ULL, 1268800ULL,
  2329665ULL, 3398720ULL, 208065ULL, 1277120ULL,
  2337985ULL, 3407040ULL, 216385ULL, 1285440ULL,
  2346305ULL, 3415360ULL, 224705ULL, 1293760ULL,
  2354625ULL, 3423680ULL, 233025ULL, 1302080ULL,
  2362945ULL, 3432000ULL, 241345ULL, 1310400ULL,
  2371265ULL, 3440320ULL, 249665ULL, 1318720ULL,
  2379585ULL, 3448640ULL, 257985ULL, 1327040ULL,
  2387905ULL, 3456960ULL, 266305ULL, 1335360ULL,
  2396225ULL, 3465280ULL, 274625ULL, 1343680ULL,
  2404545ULL, 3473600ULL, 282945ULL, 1352000ULL,
  2412865ULL, 3481920ULL, 291265ULL, 1360320ULL,
  2421185ULL, 3490240ULL, 299585ULL, 1368640ULL,
  2429505ULL, 3498560ULL, 307905ULL, 1376960ULL,
  2437825ULL, 3506880ULL, 316225ULL, 1385280ULL,
  2446145ULL, 3515200ULL, 324545ULL, 1393600ULL,
  2454465ULL, 3523520ULL, 332865ULL, 1401920ULL,
  2462785ULL, 3531840ULL, 341185ULL, 1410240ULL,
  2471105ULL, 3540160ULL, 349505ULL, 1418560ULL,
  2479425ULL, 3548480ULL, 357825ULL, 1426880ULL,
  2487745ULL, 3556800ULL, 366145ULL, 1435200ULL,
  2496065ULL, 3565120ULL, 374465ULL, 1443520ULL,
  2504385ULL, 3573440ULL, 382785ULL, 1451840ULL,
  2512705ULL, 3581760ULL, 391105ULL, 1460160ULL,
  2521025ULL, 3590080ULL, 399425ULL, 1468480ULL,
  2529345ULL, 3598400ULL, 407745ULL, 1476800ULL,
  2537665ULL, 3606720ULL, 416065ULL, 1485120ULL,
  2545985ULL, 3615040ULL, 424385ULL, 1493440ULL,
  2554305ULL, 3623360ULL, 432705ULL, 1501760ULL,
  2562625ULL, 3631680ULL, 441025ULL, 1510080ULL,
  2570945ULL, 3640000ULL, 449345ULL, 1518400ULL,
  2579265ULL, 3648320ULL, 457665ULL, 1526720ULL,
  2587585ULL, 3656640ULL, 465985ULL, 1535040ULL,
  2595905ULL, 3664960ULL, 474305ULL, 1543360ULL,
  2604225ULL, 3673280ULL, 482625ULL, 1551680ULL,
  2612545ULL, 3681600ULL, 490945ULL, 1560000ULL,
  2620865ULL, 3689920ULL, 499265ULL, 1568320ULL,
  2629185ULL, 3698240ULL, 507585ULL, 1576640ULL,
  2637505ULL, 3706560ULL, 515905ULL, 1584960ULL,
  2645825ULL, 3714880ULL, 524225ULL, 1593280ULL,
  2654145ULL, 3723200ULL, 532545ULL, 1601600ULL,
  2662465ULL, 3731520ULL, 540865ULL, 1609920ULL,
  2670785ULL, 3739840ULL, 549185ULL, 1618240ULL,
  2679105ULL, 3748160ULL, 557505ULL, 1626560ULL,
  2687425ULL, 3756480ULL, 565825ULL, 1634880ULL,
  2695745ULL, 3764800ULL, 574145ULL, 1643200ULL,
  2704065ULL, 3773120ULL, 582465ULL, 1651520ULL,
  2712385ULL, 3781440ULL, 590785ULL, 1659840ULL,
  2720705ULL, 3789760ULL, 599105ULL, 1668160ULL,
  2729025ULL, 3798080ULL, 607425ULL, 1676480ULL,
  2737345ULL, 3806400ULL, 615745ULL, 1684800ULL,
  2745665ULL, 3814720ULL, 624065ULL, 1693120ULL,
  2753985ULL, 3823040ULL, 632385ULL, 1701440ULL,
  2762305ULL, 3831360ULL, 640705ULL, 1709760ULL,
  2770625ULL, 3839680ULL, 649025ULL, 1718080ULL,
  2778945ULL, 3848000ULL, 657345ULL, 1726400ULL,
  2787265ULL, 3856320ULL, 665665ULL, 1734720ULL,
  2795585ULL, 3864640ULL, 673985ULL, 1743040ULL,
  2803905ULL, 3872960ULL, 682305ULL, 1751360ULL,
  2812225ULL, 3881280ULL, 690625ULL, 1759680ULL,
  2820545ULL, 3889600ULL, 698945ULL, 1768000ULL,
  2828865ULL, 3897920ULL, 707265ULL, 1776320ULL,
  2837185ULL, 3906240ULL, 715585ULL, 1784640ULL,
  2845505ULL, 3914560ULL, 723905ULL, 1792960ULL,
  2853825ULL, 3922880ULL, 732225ULL, 1801280ULL,
  2862145ULL, 3931200ULL, 740545ULL, 1809600ULL,
  2870465ULL, 3939520ULL, 748865ULL, 1817920ULL,
  2878785ULL, 3947840ULL, 757185ULL, 1826240ULL,
  2887105ULL, 3956160ULL, 765505ULL, 1834560ULL,
  2895425ULL, 3964480ULL, 773825ULL, 1842880ULL,
  2903745ULL, 3972800ULL, 782145ULL, 1851200ULL,
  2912065ULL, 3981120ULL, 790465ULL, 1859520ULL,
  2920385ULL, 3989440ULL, 798785ULL, 1867840ULL,
  2928705ULL, 3997760ULL, 807105ULL, 1876160ULL,
  2937025ULL, 4006080ULL, 815425ULL, 1884480ULL,
  2945345ULL, 4014400ULL, 823745ULL, 1892800ULL,
  2953665ULL, 4022720ULL, 832065ULL, 1901120ULL,
  2961985ULL, 4031040ULL, 840385ULL, 1909440ULL,
  2970305ULL, 4039360ULL, 848705ULL, 1917760ULL,
  2978625ULL, 4047680ULL, 857025ULL, 1926080ULL,
  2986945ULL, 4056000ULL, 865345ULL, 1934400ULL,
  2995265ULL, 4064320ULL, 873665ULL, 1942720ULL,
  3003585ULL, 4072640ULL, 881985ULL, 1951040ULL,
  3011905ULL, 4080960ULL, 890305ULL, 1959360ULL,
  3020225ULL, 4089280ULL, 898625ULL, 1967680ULL,
  3028545ULL, 4097600ULL, 906945ULL, 1976000ULL,
  3036865ULL, 4105920ULL, 915265ULL, 1984320ULL,
  3045185ULL, 4114240ULL, 923585ULL, 1992640ULL,
  3053505ULL, 4122560ULL, 931905ULL, 2000960ULL,
  3061825ULL, 4130880ULL, 940225ULL, 2009280ULL,
  3070145ULL, 4139200ULL, 948545ULL, 2017600ULL,
  3078465ULL, 4147520ULL, 956865ULL, 2025920ULL,
  3086785ULL, 4155840ULL, 965185ULL, 2034240ULL,
  3095105ULL, 4164160ULL, 973505ULL, 2042560ULL,
  3103425ULL, 4172480ULL, 981825ULL, 2050880ULL,
  3111745ULL, 4180800ULL, 990145ULL, 2059200ULL,
  3120065ULL, 4189120ULL, 998465ULL, 2067520ULL,
  3128385ULL, 4197440ULL, 1006785ULL, 2075840ULL,
  3136705ULL, 4205760ULL, 1015105ULL, 2084160ULL,
  3145025ULL, 4214080ULL, 1023425ULL, 2092480ULL,
  3153345ULL, 4222400ULL, 1031745ULL, 2100800ULL,
  3161665ULL, 4230720ULL, 1040065ULL, 2109120ULL,
  3169985ULL, 4239040ULL, 1048385ULL, 2117440ULL,
  3178305ULL, 4247360ULL, 1056705ULL, 2125760ULL,
};

constexpr uint64_t kRowGatherWideCopyOutOffset[508] = {
  0ULL, 8255ULL, 16510ULL, 24765ULL,
  33020ULL, 41275ULL, 49530ULL, 57785ULL,
  66040ULL, 74295ULL, 82550ULL, 90805ULL,
  99060ULL, 107315ULL, 115570ULL, 123825ULL,
  132080ULL, 140335ULL, 148590ULL, 156845ULL,
  165100ULL, 173355ULL, 181610ULL, 189865ULL,
  198120ULL, 206375ULL, 214630ULL, 222885ULL,
  231140ULL, 239395ULL, 247650ULL, 255905ULL,
  264160ULL, 272415ULL, 280670ULL, 288925ULL,
  297180ULL, 305435ULL, 313690ULL, 321945ULL,
  330200ULL, 338455ULL, 346710ULL, 354965ULL,
  363220ULL, 371475ULL, 379730ULL, 387985ULL,
  396240ULL, 404495ULL, 412750ULL, 421005ULL,
  429260ULL, 437515ULL, 445770ULL, 454025ULL,
  462280ULL, 470535ULL, 478790ULL, 487045ULL,
  495300ULL, 503555ULL, 511810ULL, 520065ULL,
  528320ULL, 536575ULL, 544830ULL, 553085ULL,
  561340ULL, 569595ULL, 577850ULL, 586105ULL,
  594360ULL, 602615ULL, 610870ULL, 619125ULL,
  627380ULL, 635635ULL, 643890ULL, 652145ULL,
  660400ULL, 668655ULL, 676910ULL, 685165ULL,
  693420ULL, 701675ULL, 709930ULL, 718185ULL,
  726440ULL, 734695ULL, 742950ULL, 751205ULL,
  759460ULL, 767715ULL, 775970ULL, 784225ULL,
  792480ULL, 800735ULL, 808990ULL, 817245ULL,
  825500ULL, 833755ULL, 842010ULL, 850265ULL,
  858520ULL, 866775ULL, 875030ULL, 883285ULL,
  891540ULL, 899795ULL, 908050ULL, 916305ULL,
  924560ULL, 932815ULL, 941070ULL, 949325ULL,
  957580ULL, 965835ULL, 974090ULL, 982345ULL,
  990600ULL, 998855ULL, 1007110ULL, 1015365ULL,
  1023620ULL, 1031875ULL, 1040130ULL, 1048385ULL,
  1056640ULL, 1064895ULL, 1073150ULL, 1081405ULL,
  1089660ULL, 1097915ULL, 1106170ULL, 1114425ULL,
  1122680ULL, 1130935ULL, 1139190ULL, 1147445ULL,
  1155700ULL, 1163955ULL, 1172210ULL, 1180465ULL,
  1188720ULL, 1196975ULL, 1205230ULL, 1213485ULL,
  1221740ULL, 1229995ULL, 1238250ULL, 1246505ULL,
  1254760ULL, 1263015ULL, 1271270ULL, 1279525ULL,
  1287780ULL, 1296035ULL, 1304290ULL, 1312545ULL,
  1320800ULL, 1329055ULL, 1337310ULL, 1345565ULL,
  1353820ULL, 1362075ULL, 1370330ULL, 1378585ULL,
  1386840ULL, 1395095ULL, 1403350ULL, 1411605ULL,
  1419860ULL, 1428115ULL, 1436370ULL, 1444625ULL,
  1452880ULL, 1461135ULL, 1469390ULL, 1477645ULL,
  1485900ULL, 1494155ULL, 1502410ULL, 1510665ULL,
  1518920ULL, 1527175ULL, 1535430ULL, 1543685ULL,
  1551940ULL, 1560195ULL, 1568450ULL, 1576705ULL,
  1584960ULL, 1593215ULL, 1601470ULL, 1609725ULL,
  1617980ULL, 1626235ULL, 1634490ULL, 1642745ULL,
  1651000ULL, 1659255ULL, 1667510ULL, 1675765ULL,
  1684020ULL, 1692275ULL, 1700530ULL, 1708785ULL,
  1717040ULL, 1725295ULL, 1733550ULL, 1741805ULL,
  1750060ULL, 1758315ULL, 1766570ULL, 1774825ULL,
  1783080ULL, 1791335ULL, 1799590ULL, 1807845ULL,
  1816100ULL, 1824355ULL, 1832610ULL, 1840865ULL,
  1849120ULL, 1857375ULL, 1865630ULL, 1873885ULL,
  1882140ULL, 1890395ULL, 1898650ULL, 1906905ULL,
  1915160ULL, 1923415ULL, 1931670ULL, 1939925ULL,
  1948180ULL, 1956435ULL, 1964690ULL, 1972945ULL,
  1981200ULL, 1989455ULL, 1997710ULL, 2005965ULL,
  2014220ULL, 2022475ULL, 2030730ULL, 2038985ULL,
  2047240ULL, 2055495ULL, 2063750ULL, 2072005ULL,
  2080260ULL, 2088515ULL, 2096770ULL, 2105025ULL,
  2113280ULL, 2121535ULL, 2129790ULL, 2138045ULL,
  2146300ULL, 2154555ULL, 2162810ULL, 2171065ULL,
  2179320ULL, 2187575ULL, 2195830ULL, 2204085ULL,
  2212340ULL, 2220595ULL, 2228850ULL, 2237105ULL,
  2245360ULL, 2253615ULL, 2261870ULL, 2270125ULL,
  2278380ULL, 2286635ULL, 2294890ULL, 2303145ULL,
  2311400ULL, 2319655ULL, 2327910ULL, 2336165ULL,
  2344420ULL, 2352675ULL, 2360930ULL, 2369185ULL,
  2377440ULL, 2385695ULL, 2393950ULL, 2402205ULL,
  2410460ULL, 2418715ULL, 2426970ULL, 2435225ULL,
  2443480ULL, 2451735ULL, 2459990ULL, 2468245ULL,
  2476500ULL, 2484755ULL, 2493010ULL, 2501265ULL,
  2509520ULL, 2517775ULL, 2526030ULL, 2534285ULL,
  2542540ULL, 2550795ULL, 2559050ULL, 2567305ULL,
  2575560ULL, 2583815ULL, 2592070ULL, 2600325ULL,
  2608580ULL, 2616835ULL, 2625090ULL, 2633345ULL,
  2641600ULL, 2649855ULL, 2658110ULL, 2666365ULL,
  2674620ULL, 2682875ULL, 2691130ULL, 2699385ULL,
  2707640ULL, 2715895ULL, 2724150ULL, 2732405ULL,
  2740660ULL, 2748915ULL, 2757170ULL, 2765425ULL,
  2773680ULL, 2781935ULL, 2790190ULL, 2798445ULL,
  2806700ULL, 2814955ULL, 2823210ULL, 2831465ULL,
  2839720ULL, 2847975ULL, 2856230ULL, 2864485ULL,
  2872740ULL, 2880995ULL, 2889250ULL, 2897505ULL,
  2905760ULL, 2914015ULL, 2922270ULL, 2930525ULL,
  2938780ULL, 2947035ULL, 2955290ULL, 2963545ULL,
  2971800ULL, 2980055ULL, 2988310ULL, 2996565ULL,
  3004820ULL, 3013075ULL, 3021330ULL, 3029585ULL,
  3037840ULL, 3046095ULL, 3054350ULL, 3062605ULL,
  3070860ULL, 3079115ULL, 3087370ULL, 3095625ULL,
  3103880ULL, 3112135ULL, 3120390ULL, 3128645ULL,
  3136900ULL, 3145155ULL, 3153410ULL, 3161665ULL,
  3169920ULL, 3178175ULL, 3186430ULL, 3194685ULL,
  3202940ULL, 3211195ULL, 3219450ULL, 3227705ULL,
  3235960ULL, 3244215ULL, 3252470ULL, 3260725ULL,
  3268980ULL, 3277235ULL, 3285490ULL, 3293745ULL,
  3302000ULL, 3310255ULL, 3318510ULL, 3326765ULL,
  3335020ULL, 3343275ULL, 3351530ULL, 3359785ULL,
  3368040ULL, 3376295ULL, 3384550ULL, 3392805ULL,
  3401060ULL, 3409315ULL, 3417570ULL, 3425825ULL,
  3434080ULL, 3442335ULL, 3450590ULL, 3458845ULL,
  3467100ULL, 3475355ULL, 3483610ULL, 3491865ULL,
  3500120ULL, 3508375ULL, 3516630ULL, 3524885ULL,
  3533140ULL, 3541395ULL, 3549650ULL, 3557905ULL,
  3566160ULL, 3574415ULL, 3582670ULL, 3590925ULL,
  3599180ULL, 3607435ULL, 3615690ULL, 3623945ULL,
  3632200ULL, 3640455ULL, 3648710ULL, 3656965ULL,
  3665220ULL, 3673475ULL, 3681730ULL, 3689985ULL,
  3698240ULL, 3706495ULL, 3714750ULL, 3723005ULL,
  3731260ULL, 3739515ULL, 3747770ULL, 3756025ULL,
  3764280ULL, 3772535ULL, 3780790ULL, 3789045ULL,
  3797300ULL, 3805555ULL, 3813810ULL, 3822065ULL,
  3830320ULL, 3838575ULL, 3846830ULL, 3855085ULL,
  3863340ULL, 3871595ULL, 3879850ULL, 3888105ULL,
  3896360ULL, 3904615ULL, 3912870ULL, 3921125ULL,
  3929380ULL, 3937635ULL, 3945890ULL, 3954145ULL,
  3962400ULL, 3970655ULL, 3978910ULL, 3987165ULL,
  3995420ULL, 4003675ULL, 4011930ULL, 4020185ULL,
  4028440ULL, 4036695ULL, 4044950ULL, 4053205ULL,
  4061460ULL, 4069715ULL, 4077970ULL, 4086225ULL,
  4094480ULL, 4102735ULL, 4110990ULL, 4119245ULL,
  4127500ULL, 4135755ULL, 4144010ULL, 4152265ULL,
  4160520ULL, 4168775ULL, 4177030ULL, 4185285ULL,
};
} // namespace

template <class DT_X> class KernelBatchToSpaceRowGatherWide {
public:
  __aicore__ inline KernelBatchToSpaceRowGatherWide() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
    xGm = reinterpret_cast<__gm__ DT_X *>(x);
    yGm = reinterpret_cast<__gm__ DT_X *>(y);
  }

  __aicore__ inline void Process() {
    const TaskRange range = GetRowGatherWideTaskRange();
    if (range.length == 0 || sizeof(DT_X) != sizeof(uint16_t)) {
      return;
    }

    MarkBufFree(0);
    MarkBufFree(1);
    CopyIn(range.offset, 0);
    BuildIdx();
    for (uint32_t i = 0; i < range.length; ++i) {
      const uint32_t curBuf = i & 1U;
      WaitMte2ReadyForV(curBuf);
      Gather(curBuf);
      const uint32_t next = i + 1;
      if (next < range.length) {
        CopyIn(range.offset + next, next & 1U);
      }
      WaitVReadyForMte3(curBuf);
      CopyOut(range.offset + i, curBuf);
      MarkBufFree(curBuf);
    }
    WaitBufFree(0);
    WaitBufFree(1);
  }

private:
  static constexpr uint32_t kHeight = 128;
  static constexpr uint32_t kWidth = 128;
  static constexpr uint32_t kDepth = 65;
  static constexpr uint32_t kOutHeight = 254;
  static constexpr uint32_t kOutWidth = 254;
  static constexpr uint32_t kHalfPx = kOutWidth / 2;
  static constexpr uint32_t kLanePx = (kHalfPx + 1) / 2;
  static constexpr uint32_t kLaneBBase = kLanePx * kDepth;
  static constexpr uint32_t kPairElems = 2 * kDepth;
  static constexpr uint32_t kRampElems = 520;
  static constexpr uint32_t kNRamp = 16;
  static constexpr uint32_t kTailElems = 456;
  static constexpr uint32_t kOutElems = kHalfPx * kDepth;
  static constexpr uint32_t kGiInElems = 2 * kLaneBBase;
  static constexpr uint32_t kSlotElems = 16576;
  static constexpr uint32_t kIdxElems = (kNRamp - 1) * kRampElems + kTailElems;
  static constexpr uint32_t kIdxBaseBytes = 2 * kSlotElems * sizeof(DT_X);
  static constexpr uint32_t kBytesPerBuf = kSlotElems * sizeof(DT_X);
  static constexpr uint32_t kTotalTasks = kOutHeight * 2;
  static constexpr uint32_t kActiveCores = 40;
  static constexpr uint32_t kInputBatchElems = kHeight * kWidth * kDepth;
  static constexpr uint32_t kInputRowElems = kWidth * kDepth;
  static constexpr uint32_t kHalfInputElems = 64 * kDepth;

  struct TaskPlan {
    uint32_t inputOffsetA;
    uint32_t inputOffsetB;
  };

  __aicore__ inline TaskRange GetRowGatherWideTaskRange() const {
    const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t baseTasks = kTotalTasks / kActiveCores;
    const uint32_t formerNum = kTotalTasks % kActiveCores;
    const uint32_t length = baseTasks + (blockIdx < formerNum ? 1U : 0U);
    const uint32_t offset =
        blockIdx * baseTasks + MinU32(blockIdx, formerNum);
    return {offset, length};
  }

  __aicore__ inline __ubuf__ DT_X *SlotAddr(uint32_t bufIdx) const {
    return UbPtr<DT_X>(bufIdx * kBytesPerBuf);
  }

  __aicore__ inline __ubuf__ int32_t *IdxAddr() const {
    return UbPtr<int32_t>(kIdxBaseBytes);
  }

  __aicore__ inline TaskPlan GetTaskPlan(uint32_t taskId) const {
    const uint32_t phase = taskId & 3U;
    const uint32_t half = taskId & 1U;
    const uint32_t batchA = 3U - phase;
    const uint32_t spatialOffset =
        ((taskId + 2U) >> 2U) * kInputRowElems +
        half * kHalfInputElems;
    return {batchA * kInputBatchElems + spatialOffset,
            (batchA ^ 1U) * kInputBatchElems + spatialOffset +
                (1U - half) * kDepth};
  }

  __aicore__ inline void BuildIdx() const {
    __ubuf__ int32_t *idx = IdxAddr();
    constexpr uint32_t kIdxBlockElems =
        BTS_DATA_BLOCK_BYTES / sizeof(int32_t);
    constexpr uint32_t kVecElems = 64;
    for (uint32_t d = 0; d < kIdxBlockElems; ++d) {
      idx[d] = static_cast<int32_t>(d * sizeof(DT_X));
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID6);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID6);
    for (uint32_t offset = kIdxBlockElems; offset < kVecElems;
         offset += kIdxBlockElems) {
      AscendC::AddsImpl<int32_t>(
          idx + offset, idx + offset - kIdxBlockElems,
          static_cast<int32_t>(kIdxBlockElems * sizeof(DT_X)),
          static_cast<int32_t>(kIdxBlockElems));
      AscendC::PipeBarrier<PIPE_V>();
    }

    // Generate each aligned 64-element block from the base ramp. Since the
    // logical lane length is 65, block b has b leading values from the prior
    // lane and 64-b values from the next lane. The vector operation fills the
    // majority; a short scalar patch fixes only those leading values.
    constexpr int32_t kBlockAdds[8] = {
        8318, 126, 8444, 252, 8570, 378, 8696, 8824};
    for (uint32_t block = 1; block <= 8; ++block) {
      const uint32_t count =
          block == 8 ? kRampElems - 8 * kVecElems : kVecElems;
      AscendC::AddsImpl<int32_t>(
          idx + block * kVecElems, idx, kBlockAdds[block - 1],
          static_cast<int32_t>(count));
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID7);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID7);
    for (uint32_t block = 1; block < 8; ++block) {
      const uint32_t blockBase = block * kVecElems;
      const uint32_t patchBase =
          (block & 1U) != 0U
              ? ((block + 1U) >> 1U) * 126U + 2U
              : 2U * kLaneBBase + (block >> 1U) * 126U;
      for (uint32_t i = 0; i < block; ++i) {
        idx[blockBase + i] =
            static_cast<int32_t>(patchBase + 2U * i);
      }
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID6);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID6);

    for (uint32_t blk = 1; blk < kNRamp; ++blk) {
      const uint32_t count = (blk == kNRamp - 1) ? kTailElems : kRampElems;
      AscendC::AddsImpl<int32_t>(
          idx + blk * kRampElems, idx,
          static_cast<int32_t>(blk * kRampElems),
          static_cast<int32_t>(count));
    }
    AscendC::PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void CopyIn(uint32_t taskId, uint32_t bufIdx) {
    WaitBufFree(bufIdx);
    __ubuf__ DT_X *slotAddr = SlotAddr(bufIdx);
    const TaskPlan plan = GetTaskPlan(taskId);
    constexpr uint16_t kLaneBlocks =
        kLaneBBase * sizeof(uint16_t) / BTS_DATA_BLOCK_BYTES;
    copy_gm_to_ubuf(
        (__ubuf__ void *)slotAddr,
        (__gm__ void *)(xGm + plan.inputOffsetA), 0, 1, kLaneBlocks, 0, 0);
    copy_gm_to_ubuf(
        (__ubuf__ void *)(slotAddr + kLaneBBase),
        (__gm__ void *)(xGm + plan.inputOffsetB), 0, 1, kLaneBlocks, 0, 0);
    MarkMte2ReadyForV(bufIdx);
  }

  __aicore__ inline void Gather(uint32_t bufIdx) {
    __ubuf__ DT_X *slot = SlotAddr(bufIdx);
    __ubuf__ uint32_t *idx =
        reinterpret_cast<__ubuf__ uint32_t *>(IdxAddr());
    AscendC::GatherImpl(slot + kGiInElems, slot, idx, kSlotElems, 0,
                        static_cast<uint64_t>(128), 64, 8);
    AscendC::GatherImpl(slot + kGiInElems + 8192, slot, idx + 8192,
                        kSlotElems, 0, static_cast<uint64_t>(63), 1, 8);
    MarkVReadyForMte3(bufIdx);
  }

  __aicore__ inline void CopyOut(uint32_t taskId, uint32_t bufIdx) {
    __ubuf__ DT_X *slotAddr = SlotAddr(bufIdx);
    const uint32_t outputOffset = taskId * kOutElems;
    constexpr uint32_t kOutBytes = kOutElems * sizeof(uint16_t);
    copy_ubuf_to_gm_align_b16(
        (__gm__ void *)(yGm + outputOffset),
        (__ubuf__ void *)(slotAddr + kGiInElems), 0, 1, kOutBytes, 0, 0, 0,
        0);
  }

  __gm__ DT_X *xGm;
  __gm__ DT_X *yGm;
};

template <class DT_X, BatchToSpaceCopyPath V, int TUNING>
class KernelBatchToSpaceImpl {
public:
  static constexpr bool kIs512BPath =
      V == BTS_PATH_DEPTH_512B_ALIGNED ||
      V == BTS_PATH_DEPTH_TINY_SPATIAL;
  static constexpr bool kIs32BPath = V == BTS_PATH_DEPTH_32B_ALIGNED;

  __aicore__ inline KernelBatchToSpaceImpl() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                              const BatchToSpaceTilingData &tilingData) {
    tiling = tilingData;
    tilesPerRow = CeilDivU32(tiling.outWidth, tiling.tileOW);
    rowTilesPerBatch = CeilDivU32(tiling.outHeight, tiling.tileOH);
    depthTiles = CeilDivU32(tiling.depth, tiling.tileDepth);
    const uint32_t paddedDepthBytes =
        AlignUpU32(tiling.depth * sizeof(DT_X), BTS_DATA_BLOCK_BYTES);
    rowPackCompactOffsetElems =
        static_cast<uint32_t>((static_cast<uint64_t>(tiling.tileOH) *
                               tiling.tileOW * paddedDepthBytes) /
                              sizeof(DT_X));
    xGmPtr = reinterpret_cast<__gm__ DT_X *>(x);
    yGmPtr = reinterpret_cast<__gm__ DT_X *>(y);
    xGm.SetGlobalBuffer(xGmPtr);
    yGm.SetGlobalBuffer(yGmPtr);
    if (tiling.disableL2 != 0) {
      xGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
      yGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    if constexpr (V == BTS_PATH_WIDE_512B) {
      InitWide512BFields();
    }
    if constexpr (V == BTS_PATH_ROW_GATHER_FP32) {
      InitRowGatherFp32Fields();
    }
#if BTS_FAST_PATHS
    InitFastPathFields();
#endif
  }

  __aicore__ inline void Process() {
    TaskRange range = {0, 0};
    if constexpr (V == BTS_PATH_WIDE_512B) {
      range = GetWide512RowBoundTaskRange(tiling, tilesPerRow);
    } else {
      range = GetTaskRange(tiling);
    }
    if (range.length == 0) {
      return;
    }

    if constexpr (V == BTS_PATH_ROW_GATHER_FP32) {
      MarkBufFree(0);
      ProcessRowGatherFp32(range);
      WaitBufFree(0);
      return;
    }

#if BTS_FAST_PATHS
    if constexpr (V == BTS_PATH_MASK_COMPACT) {
      ProcessMaskCompact(range);
      return;
    }
    if constexpr (V == BTS_PATH_MC_BIG) {
      ProcessMcBig(range);
      return;
    }
    if constexpr (V == BTS_PATH_H_FUSE) {
      ProcessHeightFuse(range);
      return;
    }
    if constexpr (V == BTS_PATH_GATHER_INTERLEAVE) {
      ProcessGatherInterleave(range);
      return;
    }
#endif

    MarkBufFree(0);
    MarkBufFree(1);

    if constexpr (V == BTS_PATH_ROW_PLANE_FP32) {
      ProcessRowPlaneFp32(range);
      WaitBufFree(0);
      WaitBufFree(1);
      return;
    }

    if constexpr (V == BTS_PATH_WIDE_512B) {
      ProcessWide512B(range);
      WaitBufFree(0);
      WaitBufFree(1);
      return;
    }

    if constexpr (V == BTS_PATH_DEPTH_32B_PADDED) {
      ProcessRowPack(range);
      WaitBufFree(0);
      WaitBufFree(1);
      return;
    }

    if constexpr (V == BTS_PATH_DEPTH_SPLIT_TINY) {
      ProcessDepthSplitTiny(range);
      WaitBufFree(0);
      WaitBufFree(1);
      return;
    }

    CopyInTile(range.offset, 0);
    for (uint32_t i = 0; i < range.length; ++i) {
      const uint32_t curBuf = i & 1U;
      WaitCopyInReady(curBuf);
      CopyOutTile(range.offset + i, curBuf);
      MarkBufFree(curBuf);

      const uint32_t next = i + 1;
      if (next < range.length) {
        CopyInTile(range.offset + next, next & 1U);
      }
    }

    WaitBufFree(0);
    WaitBufFree(1);
  }

private:
  __aicore__ inline AscendC::LocalTensor<DT_X> LocalBuffer(uint32_t bufIdx) const {
    const uint32_t bytesPerBuf = tiling.tileElemAligned * sizeof(DT_X);
    return AscendC::LocalTensor<DT_X>(AscendC::TPosition::VECCALC,
                                      bufIdx * bytesPerBuf,
                                      tiling.tileElemAligned);
  }

  __aicore__ inline bool Is32BPath() const { return kIs32BPath; }

  __aicore__ inline bool Is512BPath() const { return kIs512BPath; }

  __aicore__ inline bool Is32BAligned(uint32_t elemCount) const {
    return ((elemCount * sizeof(DT_X)) % BTS_DATA_BLOCK_BYTES) == 0;
  }

  __aicore__ inline bool Is512BAlignedByteOffset(uint64_t byteOffset) const {
    return (byteOffset % BTS_GM_ALIGN_BYTES) == 0;
  }

  __aicore__ inline uint64_t Offset4D(uint32_t n, uint32_t h, uint32_t w,
                                      uint32_t d) const {
    return (((static_cast<uint64_t>(n) * tiling.height + h) * tiling.width +
             w) *
            tiling.depth) +
           d;
  }

  __aicore__ inline uint64_t OutOffset4D(uint32_t n, uint32_t h, uint32_t w,
                                         uint32_t d) const {
    return (((static_cast<uint64_t>(n) * tiling.outHeight + h) *
                 tiling.outWidth +
             w) *
            tiling.depth) +
           d;
  }

  __aicore__ inline uint64_t OutByteOffset(uint32_t n, uint32_t h, uint32_t w,
                                           uint32_t d) const {
    return OutOffset4D(n, h, w, d) * sizeof(DT_X);
  }

  __aicore__ inline void
  CopyOutContiguous(uint64_t dstOffset, const AscendC::LocalTensor<DT_X> &src,
                    uint32_t elemCount) const {
    if (elemCount == 0) {
      return;
    }
    if (Is32BAligned(elemCount)) {
      AscendC::DataCopy(yGm[dstOffset], src, elemCount);
    } else {
      AscendC::DataCopyExtParams copyParams = {
          1, static_cast<uint32_t>(elemCount * sizeof(DT_X)), 0, 0, 0};
      AscendC::DataCopyPad<DT_X>(yGm[dstOffset], src, copyParams);
    }
  }

  __aicore__ inline DecodedTileTask DecodeTileTask(uint32_t taskId) const {
    DecodedTileTask task{0, 0, 0, 0, 0, 0, 0};
    const uint32_t depthTileIdx = taskId % depthTiles;
    const uint32_t spatialTask = taskId / depthTiles;
    const uint32_t owTileIdx = spatialTask % tilesPerRow;
    const uint32_t rowTileTask = spatialTask / tilesPerRow;
    const uint32_t ohTileIdx = rowTileTask % rowTilesPerBatch;

    task.batchIdx = rowTileTask / rowTilesPerBatch;
    task.ohStart = ohTileIdx * tiling.tileOH;
    task.curOH = MinU32(tiling.tileOH, tiling.outHeight - task.ohStart);
    task.owStart = owTileIdx * tiling.tileOW;
    task.curOW = MinU32(tiling.tileOW, tiling.outWidth - task.owStart);
    task.depthOffset = depthTileIdx * tiling.tileDepth;
    task.curDepth = MinU32(tiling.tileDepth, tiling.depth - task.depthOffset);
    return task;
  }

  __aicore__ inline bool CanUseRowsScatter(
      const DecodedTileTask &task) const {
    if (task.curOW == 0 || task.curDepth == 0) {
      return false;
    }
    const uint32_t copyBytes = task.curDepth * sizeof(DT_X);
    if ((copyBytes % BTS_DATA_BLOCK_BYTES) != 0) {
      return false;
    }
    const uint32_t depthBlocks = copyBytes / BTS_DATA_BLOCK_BYTES;
    const uint32_t srcGapBytes =
        (tiling.depth - task.curDepth) * sizeof(DT_X);
    if ((srcGapBytes % BTS_DATA_BLOCK_BYTES) != 0) {
      return false;
    }
    const uint32_t srcStrideBlocks = srcGapBytes / BTS_DATA_BLOCK_BYTES;
    const uint32_t s = tiling.blockSize;
    const uint32_t fullWResidue = (task.owStart + tiling.cropLeft) % s;
    bool hasLane = false;
    for (uint32_t bw = 0; bw < s; ++bw) {
      const uint32_t delta = (bw + s - fullWResidue) % s;
      if (delta >= task.curOW) {
        continue;
      }
      hasLane = true;
      const uint32_t count = (task.curOW - delta + s - 1) / s;
      if (count == 0 || count > 0xFFFFU || depthBlocks > 0xFFFFU) {
        return false;
      }
      const uint32_t dstStrideBlocks = (s - 1) * depthBlocks;
      if (srcStrideBlocks > 0xFFFFU || dstStrideBlocks > 0xFFFFU) {
        return false;
      }
    }
    return hasLane;
  }

  __aicore__ inline bool Path32BCopyInMustUseOnePixel(
      const DecodedTileTask &task) const {
    return !CanUseRowsScatter(task);
  }

  __aicore__ inline bool Path32BCopyOutMustUseOnePixel(
      const DecodedTileTask &task) const {
    if (task.curOH == 0 || task.curOW == 0 || task.curDepth == 0) {
      return true;
    }
    const uint32_t elemsPerRow = task.curOW * task.curDepth;
    return task.curOW == 1 || elemsPerRow == 0 ||
           !Is32BAligned(elemsPerRow);
  }

  __aicore__ inline bool Path512BMustUseOnePixel(
      const DecodedTileTask &task) const {
    return !CanUseRowsScatter(task);
  }

  __aicore__ inline void ProcessRowPlaneFp32(const TaskRange &range) {
    CopyInRowPlaneTile(range.offset, 0);
    for (uint32_t i = 0; i < range.length; ++i) {
      const uint32_t curBuf = i & 1U;
      WaitCopyInReady(curBuf);
      CopyOutRowPlaneTile(range.offset + i, curBuf);
      MarkBufFree(curBuf);

      const uint32_t next = i + 1;
      if (next < range.length) {
        CopyInRowPlaneTile(range.offset + next, next & 1U);
      }
    }
  }

  __aicore__ inline void CopyInRowPlaneTile(uint32_t taskId, uint32_t bufIdx) {
    WaitBufFree(bufIdx);
    const AscendC::LocalTensor<DT_X> tileLocal = LocalBuffer(bufIdx);
    const uint32_t rowsPerBatch = rowTilesPerBatch;
    const uint32_t batchIdx = taskId / rowsPerBatch;
    const uint32_t rowTile = taskId - batchIdx * rowsPerBatch;
    const uint32_t ih = rowTile;
    const uint32_t rowElems = tiling.outWidth * tiling.depth;
    const uint32_t depthBlocks =
        (tiling.depth * sizeof(DT_X)) / BTS_DATA_BLOCK_BYTES;

    AscendC::DataCopyParams params;
    params.blockCount = static_cast<uint16_t>(tiling.width);
    params.blockLen = static_cast<uint16_t>(depthBlocks);
    params.srcStride = 0;
    params.dstStride = static_cast<uint16_t>(depthBlocks);

    for (uint32_t r = 0; r < 2; ++r) {
      const uint32_t rowBase = r * rowElems;
      for (uint32_t bw = 0; bw < 2; ++bw) {
        const uint32_t inBatch = (r * 2 + bw) * tiling.outBatch + batchIdx;
        const uint32_t localOffset = rowBase + bw * tiling.depth;
        AscendC::DataCopy(tileLocal[localOffset],
                          xGm[Offset4D(inBatch, ih, 0, 0)], params);
      }
    }
    MarkCopyInReady(bufIdx);
  }

  __aicore__ inline void CopyOutRowPlaneTile(uint32_t taskId,
                                           uint32_t bufIdx) const {
    const AscendC::LocalTensor<DT_X> tileLocal = LocalBuffer(bufIdx);
    const uint32_t rowsPerBatch = rowTilesPerBatch;
    const uint32_t batchIdx = taskId / rowsPerBatch;
    const uint32_t rowTile = taskId - batchIdx * rowsPerBatch;
    const uint32_t ohStart = rowTile * 2;
    const uint32_t elemCount = tiling.tileOH * tiling.outWidth * tiling.depth;
    AscendC::DataCopy(yGm[OutOffset4D(batchIdx, ohStart, 0, 0)], tileLocal,
                      elemCount);
  }

  __aicore__ inline void ProcessDepthSplitTiny(const TaskRange &range) {
    CopyInDepthSplitTile(range.offset, 0);
    for (uint32_t i = 0; i < range.length; ++i) {
      const uint32_t curBuf = i & 1U;
      WaitCopyInReady(curBuf);
      CopyOutDepthSplitTile(range.offset + i, curBuf);
      MarkBufFree(curBuf);

      const uint32_t next = i + 1;
      if (next < range.length) {
        CopyInDepthSplitTile(range.offset + next, next & 1U);
      }
    }
  }

  __aicore__ inline void CopyInDepthSplitTile(uint32_t taskId, uint32_t bufIdx) {
    WaitBufFree(bufIdx);
    const AscendC::LocalTensor<DT_X> tileLocal = LocalBuffer(bufIdx);
    const uint32_t depthOffset = taskId * tiling.tileDepth;
    const uint32_t curDepth =
        MinU32(tiling.tileDepth, tiling.depth - depthOffset);
    if (curDepth == 0) {
      MarkCopyInReady(bufIdx);
      return;
    }

    const uint32_t s = tiling.blockSize;
    for (uint32_t oh = 0; oh < tiling.outHeight; ++oh) {
      for (uint32_t ow = 0; ow < tiling.outWidth; ++ow) {
        const uint32_t inBatch = oh * s + ow;
        const uint32_t pixIdx = oh * tiling.outWidth + ow;
        AscendC::DataCopy(tileLocal[pixIdx * curDepth],
                          xGm[Offset4D(inBatch, 0, 0, depthOffset)], curDepth);
      }
    }
    MarkCopyInReady(bufIdx);
  }

  __aicore__ inline void CopyOutDepthSplitTile(uint32_t taskId, uint32_t bufIdx) {
    const AscendC::LocalTensor<DT_X> tileLocal = LocalBuffer(bufIdx);
    const uint32_t depthOffset = taskId * tiling.tileDepth;
    const uint32_t curDepth =
        MinU32(tiling.tileDepth, tiling.depth - depthOffset);
    if (curDepth == 0) {
      return;
    }

    for (uint32_t oh = 0; oh < tiling.outHeight; ++oh) {
      for (uint32_t ow = 0; ow < tiling.outWidth; ++ow) {
        const uint32_t pixIdx = oh * tiling.outWidth + ow;
        CopyOutContiguous(OutOffset4D(0, oh, ow, depthOffset),
                          tileLocal[pixIdx * curDepth], curDepth);
      }
    }
  }

  __aicore__ inline void ProcessRowPack(const TaskRange &range) {
    CopyInRowPackTile(range.offset, 0);
    for (uint32_t i = 0; i < range.length; ++i) {
      const uint32_t curBuf = i & 1U;
      WaitMte2ReadyForV(curBuf);
      UnpadRowPackTile(curBuf);

      const uint32_t next = i + 1;
      if (next < range.length) {
        CopyInRowPackTile(range.offset + next, next & 1U);
      }

      WaitVReadyForMte3(curBuf);
      CopyOutRowPackTile(DecodeTileTask(range.offset + i), curBuf);
      MarkBufFree(curBuf);
    }
  }

  __aicore__ inline void ProcessRowGatherFp32(const TaskRange &range) {
    RowGatherFp32BuildIdx();
    for (uint32_t i = 0; i < range.length; ++i) {
      constexpr uint32_t bufIdx = 0;
      const uint32_t taskId = range.offset + i;
      RowGatherFp32CopyIn(taskId, bufIdx);
      WaitMte2ReadyForV(bufIdx);
      RowGatherFp32Gather(bufIdx);
      WaitVReadyForMte3(bufIdx);
      RowGatherFp32CopyOut(taskId, bufIdx);
      MarkBufFree(bufIdx);
    }
  }

  __aicore__ inline void InitRowGatherFp32Fields() {
    c2LanePx = tiling.outWidth >> 1;       // 13 pixels per even/odd lane.
    c2LaneElems = c2LanePx * tiling.depth; // 65 fp32 elems = 260B.
    c2LaneStrideElems = AlignUpU32(c2LaneElems * sizeof(DT_X),
                                   BTS_DATA_BLOCK_BYTES) /
                         sizeof(DT_X);
    c2OutElems = tiling.outWidth * tiling.depth;
    c2OutBase = 2 * c2LaneStrideElems;
    c2PairElems = tiling.blockSize * tiling.depth;
    c2RampElems = LcmU32(c2PairElems, BTS_DATA_BLOCK_BYTES / sizeof(DT_X));
    c2PairsPerRamp = c2RampElems / c2PairElems;
    c2NRamp = (c2OutElems + c2RampElems - 1) / c2RampElems;
    const uint32_t last = c2OutElems - (c2NRamp - 1) * c2RampElems;
    c2TailElems = AlignUpU32(last, BTS_DATA_BLOCK_BYTES / sizeof(DT_X));
    c2IdxElems = (c2NRamp - 1) * c2RampElems + c2TailElems;
    c2IdxBaseBytes = AlignUpU32(tiling.tileElemAligned * sizeof(DT_X),
                                BTS_DATA_BLOCK_BYTES);
  }

  __aicore__ inline AscendC::LocalTensor<DT_X>
  RowGatherFp32Slot(uint32_t bufIdx) const {
    return LocalBuffer(bufIdx);
  }

  __aicore__ inline void RowGatherFp32BuildIdx() const {
    AscendC::LocalTensor<int32_t> idx(AscendC::TPosition::VECCALC,
                                      c2IdxBaseBytes, c2IdxElems);
    for (uint32_t pair = 0; pair < c2PairsPerRamp; ++pair) {
      const uint32_t outBase = pair * c2PairElems;
      const uint32_t srcBaseBytes = pair * tiling.depth * sizeof(DT_X);
      for (uint32_t d = 0; d < 5; ++d) {
        idx.SetValue(outBase + d,
                     static_cast<int32_t>(srcBaseBytes + d * sizeof(DT_X)));
        idx.SetValue(outBase + tiling.depth + d,
                     static_cast<int32_t>(c2LaneStrideElems * sizeof(DT_X) +
                                          srcBaseBytes + d * sizeof(DT_X)));
      }
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID6);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID6);
    const int32_t rampByteStep =
        static_cast<int32_t>(c2PairsPerRamp * tiling.depth * sizeof(DT_X));
    for (uint32_t blk = 1; blk < c2NRamp; ++blk) {
      const uint32_t count = (blk == c2NRamp - 1) ? c2TailElems : c2RampElems;
      AscendC::Adds(idx[blk * c2RampElems], idx[(blk - 1) * c2RampElems],
                    rampByteStep, count);
    }
    AscendC::PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void RowGatherFp32CopyIn(uint32_t taskId, uint32_t bufIdx) {
    WaitBufFree(bufIdx);
    const AscendC::LocalTensor<DT_X> slot = RowGatherFp32Slot(bufIdx);
    const uint32_t fullH = taskId + tiling.cropTop;
    const uint32_t ih = fullH >> 1;
    const uint32_t bh = fullH & 1U;
    const uint32_t laneBytes = c2LaneElems * sizeof(DT_X);
    AscendC::DataCopyExtParams params = {1, laneBytes, 0, 0, 0};
    AscendC::DataCopyPadExtParams<DT_X> padParams = {false, 0, 0,
                                                     static_cast<DT_X>(0)};

    // cropLeft=3: output even pixels read bw=1 from iw=1..13; odd pixels read
    // bw=0 from iw=2..14. Both lanes are contiguous 13-pixel GM bursts.
    AscendC::DataCopyPad(slot,
                         xGm[Offset4D((bh << 1) + 1U, ih, 1, 0)], params,
                         padParams);
    AscendC::DataCopyPad(slot[c2LaneStrideElems],
                         xGm[Offset4D((bh << 1), ih, 2, 0)], params,
                         padParams);
    MarkMte2ReadyForV(bufIdx);
  }

  __aicore__ inline void RowGatherFp32Gather(uint32_t bufIdx) {
    const AscendC::LocalTensor<DT_X> slot = RowGatherFp32Slot(bufIdx);
    const AscendC::LocalTensor<uint32_t> idx(AscendC::TPosition::VECCALC,
                                             c2IdxBaseBytes, c2IdxElems);
    AscendC::Gather(slot[c2OutBase], slot, idx, 0U, c2OutElems);
    MarkVReadyForMte3(bufIdx);
  }

  __aicore__ inline void RowGatherFp32CopyOut(uint32_t taskId, uint32_t bufIdx) const {
    const AscendC::LocalTensor<DT_X> slot = RowGatherFp32Slot(bufIdx);
    AscendC::DataCopyExtParams params = {
        1, static_cast<uint32_t>(c2OutElems * sizeof(DT_X)), 0, 0, 0};
    AscendC::DataCopyPad(yGm[OutOffset4D(0, taskId, 0, 0)], slot[c2OutBase],
                         params);
  }

#if BTS_FAST_PATHS
  __aicore__ inline static uint32_t Gcd32(uint32_t a, uint32_t b) {
    while (b != 0) {
      const uint32_t t = a % b;
      a = b;
      b = t;
    }
    return a;
  }

  __aicore__ inline static uint32_t Lcm32(uint32_t a, uint32_t b) {
    const uint32_t g = Gcd32(a, b);
    return g == 0 ? 0 : (a / g) * b;
  }

  __aicore__ inline AscendC::LocalTensor<DT_X>
  FastSlot(uint32_t bufIdx, uint32_t baseBytes) const {
    const uint32_t bytesPerBuf = tiling.tileElemAligned * sizeof(DT_X);
    return AscendC::LocalTensor<DT_X>(AscendC::TPosition::VECCALC,
                                      baseBytes + bufIdx * bytesPerBuf,
                                      tiling.tileElemAligned);
  }

  __aicore__ inline void InitFastPathFields() {
    const uint32_t s = tiling.blockSize;
    const uint32_t depthBlocks =
        tiling.depth * sizeof(DT_X) / BTS_DATA_BLOCK_BYTES;
    if constexpr (V == BTS_PATH_MASK_COMPACT || V == BTS_PATH_MC_BIG) {
      fpDepth = tiling.depth;
      mcChunkPx = tiling.tileOW;
      mcChunks = tilesPerRow;
      mcLanePx = mcChunkPx / s;
      mcLaneElems = mcLanePx * fpDepth;
      mcInElems = s * mcLaneElems;
      mcOutBase = mcInElems;
      mcDepthBlocks = depthBlocks;
    } else if constexpr (V == BTS_PATH_H_FUSE) {
      fpDepth = tiling.depth;
      hfInW = tiling.width;
      hfOutW = s * hfInW;
      hfRows = tiling.tileOH;
      hfPairsPerBatch = tiling.outHeight / s;
      hfPhases = s * s;
      hfInRowElems = hfInW * fpDepth;
      hfOutRowElems = hfOutW * fpDepth;
      const uint32_t laneRows = hfRows / s;
      hfLaneElems = laneRows * hfInRowElems;
      hfInElems = hfPhases * hfLaneElems;
      hfOutBase = hfInElems;
      hfDepthBlocks = depthBlocks;
      hfInRowBlocks = hfInW * hfDepthBlocks;
      hfOutRowBlocks = hfOutW * hfDepthBlocks;
    } else if constexpr (V == BTS_PATH_GATHER_INTERLEAVE) {
      fpDepth = tiling.depth;
      giHalfPx = tiling.tileOW;
      giPairElems = 2 * fpDepth;
      giLanePx = (giHalfPx + 1) / 2;
      giLaneBBase = giLanePx * fpDepth;
      giInElems = 2 * giLaneBBase;
      giOutElems = giHalfPx * fpDepth;
      giSlotElems = tiling.tileElemAligned;
      giIdxBaseBytes = 2 * giSlotElems * sizeof(DT_X);
      giRampElems = Lcm32(giPairElems, BTS_DATA_BLOCK_BYTES / sizeof(uint32_t));
      giNRamp = (giOutElems + giRampElems - 1) / giRampElems;
      const uint32_t lastBlk = giOutElems - (giNRamp - 1) * giRampElems;
      giTailElems = AlignUpU32(lastBlk, BTS_DATA_BLOCK_BYTES / sizeof(uint32_t));
      giIdxElems = (giNRamp - 1) * giRampElems + giTailElems;
    }
  }

  __aicore__ inline void GatherInterleaveBuildIdx() {
    AscendC::LocalTensor<int32_t> idx(AscendC::TPosition::VECCALC,
                                      giIdxBaseBytes, giIdxElems);
    const uint32_t pairs = giRampElems / giPairElems;
    for (uint32_t pair = 0; pair < pairs; ++pair) {
      const uint32_t base = pair * giPairElems;
      for (uint32_t d = 0; d < fpDepth; ++d) {
        idx.SetValue(base + d,
                     static_cast<int32_t>(giPairElems * pair + 2 * d));
        idx.SetValue(base + fpDepth + d,
                     static_cast<int32_t>(2 * giLaneBBase +
                                          giPairElems * pair + 2 * d));
      }
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID6);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID6);
    for (uint32_t blk = 1; blk < giNRamp; ++blk) {
      const uint32_t count = (blk == giNRamp - 1) ? giTailElems : giRampElems;
      AscendC::Adds(idx[blk * giRampElems], idx[(blk - 1) * giRampElems],
                    static_cast<int32_t>(giRampElems), count);
    }
    AscendC::PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void GatherInterleaveCopyIn(uint32_t taskId,
                                                uint32_t bufIdx) {
    WaitBufFree(bufIdx);
    const AscendC::LocalTensor<DT_X> in = FastSlot(bufIdx, 0);
    const uint32_t oh = taskId >> 1;
    const uint32_t half = taskId & 1U;
    const uint32_t fullH = oh + tiling.cropTop;
    const uint32_t ih = fullH >> 1;
    const uint32_t bh = fullH & 1U;
    const uint32_t owStart = half * giHalfPx;
    const uint32_t parity = (tiling.cropLeft + owStart) & 1U;
    const uint32_t batchA = bh * 2 + parity;
    const uint32_t batchB = bh * 2 + 1U - parity;
    const uint32_t iwA = (tiling.cropLeft + owStart) / 2;
    const uint32_t iwB = (tiling.cropLeft + owStart + 1) / 2;
    AscendC::DataCopy(in, xGm[Offset4D(batchA, ih, iwA, 0)], giLaneBBase);
    AscendC::DataCopy(in[giLaneBBase], xGm[Offset4D(batchB, ih, iwB, 0)],
                      giLaneBBase);
    MarkMte2ReadyForV(bufIdx);
  }

  __aicore__ inline void GatherInterleaveCompute(uint32_t bufIdx) {
    const AscendC::LocalTensor<DT_X> slot = FastSlot(bufIdx, 0);
    const AscendC::LocalTensor<uint32_t> idx(AscendC::TPosition::VECCALC,
                                             giIdxBaseBytes, giOutElems);
    AscendC::Gather(slot[giInElems], slot, idx, 0U, giOutElems);
    MarkVReadyForMte3(bufIdx);
  }

  __aicore__ inline void GatherInterleaveCopyOut(uint32_t taskId,
                                                 uint32_t bufIdx) {
    const AscendC::LocalTensor<DT_X> slot = FastSlot(bufIdx, 0);
    const uint32_t oh = taskId >> 1;
    const uint32_t owStart = (taskId & 1U) * giHalfPx;
    AscendC::DataCopyExtParams params = {
        1, static_cast<uint32_t>(giOutElems * sizeof(DT_X)), 0, 0, 0};
    AscendC::DataCopyPad(yGm[OutOffset4D(0, oh, owStart, 0)],
                         slot[giInElems], params);
  }

  __aicore__ inline void ProcessGatherInterleave(const TaskRange &range) {
    GatherInterleaveBuildIdx();
    MarkBufFree(0);
    MarkBufFree(1);
    GatherInterleaveCopyIn(range.offset, 0);
    for (uint32_t i = 0; i < range.length; ++i) {
      const uint32_t curBuf = i & 1U;
      WaitMte2ReadyForV(curBuf);
      GatherInterleaveCompute(curBuf);
      const uint32_t next = i + 1;
      if (next < range.length) {
        GatherInterleaveCopyIn(range.offset + next, next & 1U);
      }
      WaitVReadyForMte3(curBuf);
      GatherInterleaveCopyOut(range.offset + i, curBuf);
      MarkBufFree(curBuf);
    }
    WaitBufFree(0);
    WaitBufFree(1);
  }

  __aicore__ inline void MaskCompactCopyIn(uint32_t taskId, uint32_t bufIdx) {
    WaitBufFree(bufIdx);
    const AscendC::LocalTensor<DT_X> in = FastSlot(bufIdx, 0);
    const uint32_t oh = taskId / mcChunks;
    const uint32_t chunk = taskId % mcChunks;
    const uint32_t owStart = chunk * mcChunkPx;
    const uint32_t curOW = MinU32(mcChunkPx, tiling.outWidth - owStart);
    const uint32_t s = tiling.blockSize;
    const uint32_t fullH = oh + tiling.cropTop;
    const uint32_t ih = fullH / s;
    const uint32_t bh = fullH % s;
    const uint32_t residue = (owStart + tiling.cropLeft) % s;

    for (uint32_t bw = 0; bw < s; ++bw) {
      const uint32_t delta = (bw + s - residue) % s;
      if (delta >= curOW) {
        continue;
      }
      const uint32_t count = (curOW - delta + s - 1) / s;
      const uint32_t iw = (owStart + delta + tiling.cropLeft) / s;
      const uint32_t inBatch = bh * s + bw;
      AscendC::DataCopy(in[delta * mcLaneElems],
                        xGm[Offset4D(inBatch, ih, iw, 0)],
                        count * fpDepth);
    }
    MarkMte2ReadyForV(bufIdx);
  }

  __aicore__ inline void MaskCompactCompute(uint32_t taskId, uint32_t bufIdx) {
    const AscendC::LocalTensor<DT_X> slot = FastSlot(bufIdx, 0);
    const uint32_t chunk = taskId % mcChunks;
    const uint32_t owStart = chunk * mcChunkPx;
    const uint32_t curOW = MinU32(mcChunkPx, tiling.outWidth - owStart);
    const uint32_t s = tiling.blockSize;
    const AscendC::CopyRepeatParams params = {
        1, 1, static_cast<uint16_t>(s * mcDepthBlocks),
        static_cast<uint16_t>(mcDepthBlocks)};

    for (uint32_t delta = 0; delta < s; ++delta) {
      if (delta >= curOW) {
        continue;
      }
      const uint32_t count = (curOW - delta + s - 1) / s;
      AscendC::Copy(slot[mcOutBase + delta * fpDepth],
                    slot[delta * mcLaneElems], fpDepth,
                    static_cast<uint8_t>(count), params);
    }
    MarkVReadyForMte3(bufIdx);
  }

  __aicore__ inline void MaskCompactCopyOut(uint32_t taskId, uint32_t bufIdx) {
    const AscendC::LocalTensor<DT_X> slot = FastSlot(bufIdx, 0);
    const uint32_t oh = taskId / mcChunks;
    const uint32_t owStart = (taskId % mcChunks) * mcChunkPx;
    const uint32_t curOW = MinU32(mcChunkPx, tiling.outWidth - owStart);
    AscendC::DataCopy(yGm[OutOffset4D(0, oh, owStart, 0)], slot[mcOutBase],
                      curOW * fpDepth);
  }

  // MC_BIG: identical byte movement to MASK_COMPACT but one full-UB buffer.
  // Maximal chunk width minimizes MTE op count for latency-bound shapes.
  __aicore__ inline void ProcessMcBig(const TaskRange &range) {
    MarkBufFree(0);
    for (uint32_t i = 0; i < range.length; ++i) {
      MaskCompactCopyIn(range.offset + i, 0);
      WaitMte2ReadyForV(0);
      MaskCompactCompute(range.offset + i, 0);
      WaitVReadyForMte3(0);
      MaskCompactCopyOut(range.offset + i, 0);
      MarkBufFree(0);
    }
    WaitBufFree(0);
  }

  __aicore__ inline void ProcessMaskCompact(const TaskRange &range) {
    MarkBufFree(0);
    MarkBufFree(1);
    MaskCompactCopyIn(range.offset, 0);
    for (uint32_t i = 0; i < range.length; ++i) {
      const uint32_t curBuf = i & 1U;
      WaitMte2ReadyForV(curBuf);
      const uint32_t next = i + 1;
      if (next < range.length) {
        MaskCompactCopyIn(range.offset + next, next & 1U);
      }
      MaskCompactCompute(range.offset + i, curBuf);
      WaitVReadyForMte3(curBuf);
      MaskCompactCopyOut(range.offset + i, curBuf);
      MarkBufFree(curBuf);
    }
    WaitBufFree(0);
    WaitBufFree(1);
  }

  __aicore__ inline uint32_t HeightFuseChunkPairs(uint32_t linearPair,
                                             uint32_t remainingPairs) const {
    const uint32_t pairInBatch = linearPair % hfPairsPerBatch;
    const uint32_t batchPairsLeft = hfPairsPerBatch - pairInBatch;
    const uint32_t maxChunkPairs = hfRows / tiling.blockSize;
    // Keep each chunk inside one output batch so copy-in/out remain contiguous.
    return MinU32(MinU32(maxChunkPairs, batchPairsLeft), remainingPairs);
  }

  __aicore__ inline void HeightFuseCopyIn(uint32_t linearPair, uint32_t chunkPairs,
                                     uint32_t bufIdx) {
    WaitBufFree(bufIdx);
    const AscendC::LocalTensor<DT_X> in = FastSlot(bufIdx, 0);
    const uint32_t batchIdx = linearPair / hfPairsPerBatch;
    const uint32_t pairInBatch = linearPair % hfPairsPerBatch;
    const uint32_t ih0 = pairInBatch;
    const uint32_t inRows = chunkPairs;
    __ubuf__ DT_X *inAddr =
        reinterpret_cast<__ubuf__ DT_X *>(in.GetPhyAddr());

    for (uint32_t phase = 0; phase < hfPhases; ++phase) {
      const uint32_t inBatch = phase * tiling.outBatch + batchIdx;
      copy_gm_to_ubuf((__ubuf__ void *)(inAddr + phase * hfLaneElems),
                      (__gm__ void *)(xGmPtr + Offset4D(inBatch, ih0, 0, 0)), 0,
                      inRows, hfInRowBlocks, 0, 0);
    }
    MarkMte2ReadyForV(bufIdx);
  }

  __aicore__ inline void HeightFuseUnpackPhase(
      const AscendC::LocalTensor<DT_X> &slot, uint32_t bh, uint32_t bw,
      uint32_t inRows) const {
    const uint32_t s = tiling.blockSize;
    const uint32_t srcBase = (bh * s + bw) * hfLaneElems;
    const uint32_t dstBase = bh * hfOutRowElems + bw * fpDepth;
    const AscendC::CopyRepeatParams params = {
        1, 1, static_cast<uint16_t>(s * hfDepthBlocks),
        static_cast<uint16_t>(hfDepthBlocks)};
    const uint8_t widthRepeat = static_cast<uint8_t>(hfInW);

    for (uint32_t r = 0; r < inRows; ++r) {
      AscendC::Copy(slot[hfOutBase + dstBase + r * s * hfOutRowElems],
                    slot[srcBase + r * hfInRowElems], fpDepth, widthRepeat,
                    params);
    }
  }

  __aicore__ inline void HeightFuseCompute(uint32_t chunkPairs, uint32_t bufIdx) {
    const AscendC::LocalTensor<DT_X> slot = FastSlot(bufIdx, 0);
    const uint32_t inRows = chunkPairs;

    // H_FUSE only selects block_size=2; unroll the 4 batch phases.
    HeightFuseUnpackPhase(slot, 0, 0, inRows);
    HeightFuseUnpackPhase(slot, 0, 1, inRows);
    HeightFuseUnpackPhase(slot, 1, 0, inRows);
    HeightFuseUnpackPhase(slot, 1, 1, inRows);
    MarkVReadyForMte3(bufIdx);
  }

  __aicore__ inline void HeightFuseCopyOut(uint32_t linearPair, uint32_t chunkPairs,
                                      uint32_t bufIdx) {
    const AscendC::LocalTensor<DT_X> slot = FastSlot(bufIdx, 0);
    const uint32_t batchIdx = linearPair / hfPairsPerBatch;
    const uint32_t pairInBatch = linearPair % hfPairsPerBatch;
    const uint32_t ohStart = pairInBatch * tiling.blockSize;
    const uint32_t rows = chunkPairs * tiling.blockSize;
    const uint32_t outBlocks = rows * hfOutRowBlocks;
    __ubuf__ DT_X *outAddr =
        reinterpret_cast<__ubuf__ DT_X *>(slot.GetPhyAddr()) + hfOutBase;

    copy_ubuf_to_gm((__gm__ void *)(yGmPtr + OutOffset4D(batchIdx, ohStart, 0, 0)),
                    (__ubuf__ void *)outAddr, 0, 1, outBlocks, 0, 0);
  }

  __aicore__ inline void ProcessHeightFuse(const TaskRange &range) {
    MarkBufFree(0);
    MarkBufFree(1);
    uint32_t curPair = range.offset;
    uint32_t remainingPairs = range.length;
    uint32_t curPairs = HeightFuseChunkPairs(curPair, remainingPairs);
    HeightFuseCopyIn(curPair, curPairs, 0);
    remainingPairs -= curPairs;
    uint32_t nextPair = curPair + curPairs;
    for (uint32_t i = 0;; ++i) {
      const uint32_t curBuf = i & 1U;
      WaitMte2ReadyForV(curBuf);
      uint32_t nextPairs = 0;
      if (remainingPairs > 0) {
        nextPairs = HeightFuseChunkPairs(nextPair, remainingPairs);
        HeightFuseCopyIn(nextPair, nextPairs, (i + 1) & 1U);
      }
      HeightFuseCompute(curPairs, curBuf);
      WaitVReadyForMte3(curBuf);
      HeightFuseCopyOut(curPair, curPairs, curBuf);
      MarkBufFree(curBuf);
      if (remainingPairs == 0) {
        break;
      }
      curPair = nextPair;
      curPairs = nextPairs;
      nextPair += nextPairs;
      remainingPairs -= nextPairs;
    }
    WaitBufFree(0);
    WaitBufFree(1);
  }
#endif

  __aicore__ inline void InitWide512BFields() {
    const uint32_t s = tiling.blockSize;
    const uint32_t depth = tiling.depth;
    w512ChunkPx = tiling.tileOW;
    w512Chunks = tilesPerRow;
    const uint32_t lanePx = w512ChunkPx / s;
    w512LaneElems = lanePx * depth;
    w512OutBase = s * w512LaneElems;
    w512DepthBlocks =
        (depth * sizeof(DT_X)) / BTS_DATA_BLOCK_BYTES;
  }

  __aicore__ inline Wide512BTask DecodeWide512BTask(uint32_t taskId) const {
    const uint32_t oh = taskId / w512Chunks;
    const uint32_t chunk = taskId - oh * w512Chunks;
    const uint32_t owStart = chunk * w512ChunkPx;
    const uint32_t curOW = MinU32(w512ChunkPx, tiling.outWidth - owStart);
    return {oh, oh >> 1, oh & 1U, owStart, curOW, curOW >> 1,
            owStart >> 1};
  }

  __aicore__ inline void Wide512BCopyIn(const Wide512BTask &task,
                                        uint32_t bufIdx) {
    WaitBufFree(bufIdx);
    const AscendC::LocalTensor<DT_X> in = LocalBuffer(bufIdx);
    const uint32_t laneElems = task.lanePx * BTS_W512_COPY_MASK_ELEMS * 2U;
    const uint32_t inBatch0 = task.bh << 1;
    AscendC::DataCopy(in[0], xGm[Offset4D(inBatch0, task.ih, task.iw, 0)],
                      laneElems);
    AscendC::DataCopy(in[w512LaneElems],
                      xGm[Offset4D(inBatch0 + 1U, task.ih, task.iw, 0)],
                      laneElems);
    MarkMte2ReadyForV(bufIdx);
  }

  __aicore__ inline void Wide512BCompute(const Wide512BTask &task,
                                         uint32_t bufIdx) {
    const AscendC::LocalTensor<DT_X> slot = LocalBuffer(bufIdx);
    const AscendC::CopyRepeatParams params = {
        1, 1, static_cast<uint16_t>(2U * w512DepthBlocks),
        static_cast<uint16_t>(w512DepthBlocks)};
    const uint8_t repeat = static_cast<uint8_t>(task.lanePx);
    constexpr uint32_t depth = BTS_W512_COPY_MASK_ELEMS * 2U;

    AscendC::Copy(slot[w512OutBase], slot, BTS_W512_COPY_MASK_ELEMS, repeat,
                  params);
    AscendC::Copy(slot[w512OutBase + BTS_W512_COPY_MASK_ELEMS],
                  slot[BTS_W512_COPY_MASK_ELEMS], BTS_W512_COPY_MASK_ELEMS,
                  repeat, params);
    AscendC::Copy(slot[w512OutBase + depth], slot[w512LaneElems],
                  BTS_W512_COPY_MASK_ELEMS, repeat, params);
    AscendC::Copy(slot[w512OutBase + depth + BTS_W512_COPY_MASK_ELEMS],
                  slot[w512LaneElems + BTS_W512_COPY_MASK_ELEMS],
                  BTS_W512_COPY_MASK_ELEMS, repeat, params);
    MarkVReadyForMte3(bufIdx);
  }

  __aicore__ inline void Wide512BCopyOut(const Wide512BTask &task,
                                         uint32_t bufIdx) {
    const AscendC::LocalTensor<DT_X> slot = LocalBuffer(bufIdx);
    constexpr uint32_t depth = BTS_W512_COPY_MASK_ELEMS * 2U;
    AscendC::DataCopy(yGm[OutOffset4D(0, task.oh, task.owStart, 0)],
                      slot[w512OutBase], task.curOW * depth);
  }

  __aicore__ inline void ProcessWide512B(const TaskRange &range) {
    Wide512BTask task = DecodeWide512BTask(range.offset);
    Wide512BCopyIn(task, 0);
    for (uint32_t i = 0; i < range.length; ++i) {
      const uint32_t curBuf = i & 1U;
      WaitMte2ReadyForV(curBuf);
      Wide512BCompute(task, curBuf);
      const uint32_t next = i + 1;
      if (next < range.length) {
        Wide512BTask nextTask = DecodeWide512BTask(range.offset + next);
        Wide512BCopyIn(nextTask, next & 1U);
        WaitVReadyForMte3(curBuf);
        Wide512BCopyOut(task, curBuf);
        MarkBufFree(curBuf);
        task = nextTask;
      } else {
        WaitVReadyForMte3(curBuf);
        Wide512BCopyOut(task, curBuf);
        MarkBufFree(curBuf);
      }
    }
  }

  __aicore__ inline void CopyInTile(uint32_t taskId, uint32_t bufIdx) {
    WaitBufFree(bufIdx);
    const AscendC::LocalTensor<DT_X> tileLocal = LocalBuffer(bufIdx);
    const DecodedTileTask task = DecodeTileTask(taskId);

    if (Is512BPath()) {
      if (Path512BMustUseOnePixel(task)) {
        CopyInOnePixelRows(task, tileLocal);
      } else {
        CopyInRowsScatter(task, tileLocal);
      }
    } else if (Is32BPath()) {
      if (Path32BCopyInMustUseOnePixel(task)) {
        CopyInOnePixelRows(task, tileLocal);
      } else {
        CopyInRowsScatter(task, tileLocal);
      }
    }
    MarkCopyInReady(bufIdx);
  }

  __aicore__ inline void CopyOutTile(uint32_t taskId, uint32_t bufIdx) {
    const AscendC::LocalTensor<DT_X> tileLocal = LocalBuffer(bufIdx);
    const DecodedTileTask task = DecodeTileTask(taskId);

    if (Is512BPath()) {
      if (Path512BMustUseOnePixel(task)) {
        CopyOutOnePixelRows(task, tileLocal);
      } else {
        CopyOut512BRows(task, tileLocal);
      }
    } else if (Is32BPath()) {
      if (Path32BCopyOutMustUseOnePixel(task)) {
        CopyOutOnePixelRows(task, tileLocal);
      } else {
        CopyOut32BRows(task, tileLocal);
      }
    }
  }

  __aicore__ inline void CopyInRowPackTile(uint32_t taskId, uint32_t bufIdx) {
    WaitBufFree(bufIdx);
    const AscendC::LocalTensor<DT_X> local = LocalBuffer(bufIdx);
    const DecodedTileTask task = DecodeTileTask(taskId);

    const uint32_t s = tiling.blockSize;
    const uint32_t depthBytes = tiling.depth * sizeof(DT_X);
    const uint32_t paddedDepthBytes =
        AlignUpU32(depthBytes, BTS_DATA_BLOCK_BYTES);
    const uint32_t paddedDepthElems = paddedDepthBytes / sizeof(DT_X);
    const uint32_t paddedDepthBlocks = paddedDepthBytes / BTS_DATA_BLOCK_BYTES;
    const uint32_t fullWResidue = (task.owStart + tiling.cropLeft) % s;

    AscendC::DataCopyPadExtParams<DT_X> padParams = {false, 0, 0,
                                                     static_cast<DT_X>(0)};

    for (uint32_t r = 0; r < task.curOH; ++r) {
      const uint32_t oh = task.ohStart + r;
      const uint32_t fullH = oh + tiling.cropTop;
      const uint32_t inH = fullH / s;
      const uint32_t bh = fullH - inH * s;
      const uint32_t rowBase = r * tiling.tileOW * paddedDepthElems;

      for (uint32_t bw = 0; bw < s; ++bw) {
        const uint32_t delta = (bw + s - fullWResidue) % s;
        if (delta >= task.curOW) {
          continue;
        }
        const uint32_t count = (task.curOW - delta + s - 1) / s;
        const uint32_t firstOw = task.owStart + delta;
        const uint32_t inW = (firstOw + tiling.cropLeft) / s;
        const uint32_t inBatch = (bh * s + bw) * tiling.outBatch + task.batchIdx;
        const uint32_t localOffset = rowBase + delta * paddedDepthElems;

        AscendC::DataCopyExtParams params = {
            static_cast<uint16_t>(count), depthBytes, 0,
            static_cast<uint32_t>((s - 1) * paddedDepthBlocks), 0};
        AscendC::DataCopyPad(local[localOffset],
                             xGm[Offset4D(inBatch, inH, inW, 0)], params,
                             padParams);
      }
    }
    MarkMte2ReadyForV(bufIdx);
  }

  __aicore__ inline void UnpadRowPackTile(uint32_t bufIdx) {
    AscendC::LocalTensor<DT_X> padded = LocalBuffer(bufIdx);
    AscendC::LocalTensor<DT_X> compact =
        padded[rowPackCompactOffsetElems];

    const uint32_t depthBytes = tiling.depth * sizeof(DT_X);
    const uint32_t paddedDepthBlocks =
        AlignUpU32(depthBytes, BTS_DATA_BLOCK_BYTES) / BTS_DATA_BLOCK_BYTES;
    const uint32_t logicalRows = tiling.tileOH * tiling.tileOW;

    AscendC::GatherMaskParams params;
    params.src0BlockStride = 1;
    params.repeatTimes = static_cast<uint16_t>(logicalRows);
    params.src0RepeatStride = static_cast<uint16_t>(paddedDepthBlocks);
    params.src1RepeatStride = 0;
    uint64_t rsvdCnt = 0;
    AscendC::GatherMask(compact, padded,
                        static_cast<uint8_t>(AscendC::REDUCEV2_MODE_SEVEN),
                        /*reduceMode=*/true, /*mask=*/tiling.depth, params,
                        rsvdCnt);
    AscendC::ResetMask();
    MarkVReadyForMte3(bufIdx);
  }

  __aicore__ inline void
  CopyOutRowPackTile(const DecodedTileTask &task, uint32_t bufIdx) const {
    const AscendC::LocalTensor<DT_X> local = LocalBuffer(bufIdx);
    const AscendC::LocalTensor<DT_X> compact =
        local[rowPackCompactOffsetElems];
    const uint32_t elemsPerTileRow = tiling.tileOW * tiling.depth;
    const uint32_t elemsPerValidRow = task.curOW * tiling.depth;

    if (task.owStart == 0 && task.curOW == tiling.outWidth &&
        task.curOW == tiling.tileOW) {
      CopyOutContiguous(OutOffset4D(task.batchIdx, task.ohStart, 0, 0), compact,
                        task.curOH * elemsPerValidRow);
      return;
    }

    if (task.curOW == tiling.tileOW && task.curOH <= 0xFFFFU) {
      const uint32_t validBytes = elemsPerValidRow * sizeof(DT_X);
      const uint32_t dstRowBytes = tiling.outWidth * tiling.depth * sizeof(DT_X);
      const uint32_t dstGapBytes = dstRowBytes - validBytes;
      const uint32_t rowBlocks = validBytes / BTS_DATA_BLOCK_BYTES;
      const uint32_t dstStrideBlocks = dstGapBytes / BTS_DATA_BLOCK_BYTES;
      if ((validBytes % BTS_DATA_BLOCK_BYTES) == 0 &&
          (dstGapBytes % BTS_DATA_BLOCK_BYTES) == 0 &&
          rowBlocks <= 0xFFFFU && dstStrideBlocks <= 0xFFFFU) {
        AscendC::DataCopyParams params;
        params.blockCount = static_cast<uint16_t>(task.curOH);
        params.blockLen = static_cast<uint16_t>(rowBlocks);
        params.srcStride = 0;
        params.dstStride = static_cast<uint16_t>(dstStrideBlocks);
        AscendC::DataCopy(
            yGm[OutOffset4D(task.batchIdx, task.ohStart, task.owStart, 0)], compact,
            params);
        return;
      }
      AscendC::DataCopyExtParams params = {static_cast<uint16_t>(task.curOH),
                                           validBytes, 0,
                                           dstGapBytes, 0};
      AscendC::DataCopyPad(
          yGm[OutOffset4D(task.batchIdx, task.ohStart, task.owStart, 0)], compact,
          params);
      return;
    }

    for (uint32_t r = 0; r < task.curOH; ++r) {
      CopyOutContiguous(
          OutOffset4D(task.batchIdx, task.ohStart + r, task.owStart, 0),
          compact[r * elemsPerTileRow], elemsPerValidRow);
    }
  }

  __aicore__ inline void
  CopyInOnePixelRows(const DecodedTileTask &task,
                      const AscendC::LocalTensor<DT_X> &local) const {
    const uint32_t copyBytes = task.curDepth * sizeof(DT_X);
    const uint32_t rowBytes = AlignUpU32(copyBytes, BTS_DATA_BLOCK_BYTES);
    const uint32_t rowElems = rowBytes / sizeof(DT_X);
    AscendC::DataCopyExtParams params = {1, copyBytes, 0, 0, 0};
    AscendC::DataCopyPadExtParams<DT_X> padParams = {false, 0, 0,
                                                     static_cast<DT_X>(0)};

    for (uint32_t r = 0; r < task.curOH; ++r) {
      const uint32_t oh = task.ohStart + r;
      const uint32_t fullH = oh + tiling.cropTop;
      const uint32_t inH = fullH / tiling.blockSize;
      const uint32_t bh = fullH - inH * tiling.blockSize;
      const uint32_t fullW = task.owStart + tiling.cropLeft;
      const uint32_t inW = fullW / tiling.blockSize;
      const uint32_t bw = fullW - inW * tiling.blockSize;
      const uint32_t inBatch =
          (bh * tiling.blockSize + bw) * tiling.outBatch + task.batchIdx;
      AscendC::DataCopyPad(local[r * rowElems],
                           xGm[Offset4D(inBatch, inH, inW, task.depthOffset)],
                           params, padParams);
    }
  }

  __aicore__ inline void
  CopyOutOnePixelRows(const DecodedTileTask &task,
                       const AscendC::LocalTensor<DT_X> &local) const {
    if (task.curOH == 0) {
      return;
    }
    const uint32_t copyBytes = task.curDepth * sizeof(DT_X);
    const uint32_t dstRowBytes = tiling.outWidth * tiling.depth * sizeof(DT_X);
    AscendC::DataCopyExtParams params = {static_cast<uint16_t>(task.curOH),
                                         copyBytes, 0,
                                         dstRowBytes - copyBytes, 0};
    AscendC::DataCopyPad(
        yGm[OutOffset4D(task.batchIdx, task.ohStart, task.owStart, task.depthOffset)],
        local, params);
  }

  __aicore__ inline void CopyInRowsScatterLane(
      const DecodedTileTask &task, const AscendC::LocalTensor<DT_X> &local,
      uint32_t r, uint32_t rowElems, uint32_t depthBlocks, uint32_t s,
      uint32_t fullWResidue, uint32_t bw) const {
    const uint32_t delta = (bw + s - fullWResidue) % s;
    if (delta >= task.curOW) {
      return;
    }
    const uint32_t count = (task.curOW - delta + s - 1) / s;
    const uint32_t oh = task.ohStart + r;
    const uint32_t fullH = oh + tiling.cropTop;
    const uint32_t inH = fullH / s;
    const uint32_t bh = fullH - inH * s;
    const uint32_t firstOw = task.owStart + delta;
    const uint32_t inW = (firstOw + tiling.cropLeft) / s;
    const uint32_t inBatch = (bh * s + bw) * tiling.outBatch + task.batchIdx;
    const uint32_t rowBase = r * rowElems;
    const uint32_t localOffset = rowBase + delta * task.curDepth;
    const uint32_t srcStrideBlocks =
        ((tiling.depth - task.curDepth) * sizeof(DT_X)) /
        BTS_DATA_BLOCK_BYTES;

    AscendC::DataCopyParams params;
    params.blockCount = static_cast<uint16_t>(count);
    params.blockLen = static_cast<uint16_t>(depthBlocks);
    params.srcStride = static_cast<uint16_t>(srcStrideBlocks);
    params.dstStride = static_cast<uint16_t>((s - 1) * depthBlocks);
    AscendC::DataCopy(local[localOffset],
                      xGm[Offset4D(inBatch, inH, inW, task.depthOffset)], params);
  }

  __aicore__ inline void
  CopyInRowsScatter(const DecodedTileTask &task,
                     const AscendC::LocalTensor<DT_X> &local) const {
    const uint32_t s = tiling.blockSize;
    const uint32_t depthBlocks =
        (task.curDepth * sizeof(DT_X)) / BTS_DATA_BLOCK_BYTES;
    const uint32_t rowBytes = AlignUpU32(task.curOW * task.curDepth * sizeof(DT_X),
                                         BTS_DATA_BLOCK_BYTES);
    const uint32_t rowElems = rowBytes / sizeof(DT_X);
    const uint32_t fullWResidue = (task.owStart + tiling.cropLeft) % s;

    for (uint32_t r = 0; r < task.curOH; ++r) {
      if (s == 1) {
        CopyInRowsScatterLane(task, local, r, rowElems, depthBlocks, s,
                             fullWResidue, 0);
        continue;
      }
      for (uint32_t bw = 0; bw < s; ++bw) {
        CopyInRowsScatterLane(task, local, r, rowElems, depthBlocks, s,
                               fullWResidue, bw);
      }
    }
  }

  __aicore__ inline void
  CopyOut32BRows(const DecodedTileTask &task,
                  const AscendC::LocalTensor<DT_X> &local) const {
    const uint32_t elemsPerRow = task.curOW * task.curDepth;
    if (task.curOH == 0 || elemsPerRow == 0) {
      return;
    }
    const uint32_t rowBytes =
        AlignUpU32(elemsPerRow * sizeof(DT_X), BTS_DATA_BLOCK_BYTES);
    const uint32_t rowElems = rowBytes / sizeof(DT_X);

    if constexpr (TUNING != 0) {
      if (task.curDepth != tiling.depth) {
        for (uint32_t r = 0; r < task.curOH; ++r) {
          for (uint32_t c = 0; c < task.curOW; ++c) {
            CopyOutContiguous(
                OutOffset4D(task.batchIdx, task.ohStart + r, task.owStart + c,
                            task.depthOffset),
                local[r * rowElems + c * task.curDepth], task.curDepth);
          }
        }
        return;
      }
    }

    if (task.owStart == 0 && task.curOW == tiling.outWidth &&
        Is32BAligned(elemsPerRow)) {
      CopyOutContiguous(
          OutOffset4D(task.batchIdx, task.ohStart, 0, task.depthOffset), local,
          task.curOH * elemsPerRow);
      return;
    }

    const uint32_t rowBlocks =
        (elemsPerRow * sizeof(DT_X)) / BTS_DATA_BLOCK_BYTES;
    const uint32_t dstStrideBlocks =
        ((tiling.outWidth * tiling.depth - elemsPerRow) * sizeof(DT_X)) /
        BTS_DATA_BLOCK_BYTES;
    if (Is32BAligned(elemsPerRow) && rowBlocks <= 0xFFFFU &&
        dstStrideBlocks <= 0xFFFFU && task.curOH <= 0xFFFFU) {
      AscendC::DataCopyParams params;
      params.blockCount = static_cast<uint16_t>(task.curOH);
      params.blockLen = static_cast<uint16_t>(rowBlocks);
      params.srcStride = 0;
      params.dstStride = static_cast<uint16_t>(dstStrideBlocks);
      AscendC::DataCopy(
          yGm[OutOffset4D(task.batchIdx, task.ohStart, task.owStart, task.depthOffset)],
          local, params);
      return;
    }

    for (uint32_t r = 0; r < task.curOH; ++r) {
      CopyOutContiguous(
          OutOffset4D(task.batchIdx, task.ohStart + r, task.owStart, task.depthOffset),
          local[r * rowElems], elemsPerRow);
    }
  }

  __aicore__ inline void
  CopyOut512BRows(const DecodedTileTask &task,
                   const AscendC::LocalTensor<DT_X> &local) const {
    const uint32_t elemsPerRow = task.curOW * task.curDepth;
    if (task.curOH == 0 || elemsPerRow == 0) {
      return;
    }
    const uint32_t rowBytes =
        AlignUpU32(elemsPerRow * sizeof(DT_X), BTS_DATA_BLOCK_BYTES);
    const uint32_t rowElems = rowBytes / sizeof(DT_X);
    const uint64_t dstBaseByte =
        OutByteOffset(task.batchIdx, task.ohStart, task.owStart, task.depthOffset);
    const uint32_t dstRowBytes = tiling.outWidth * tiling.depth * sizeof(DT_X);
    const uint32_t validBytes = elemsPerRow * sizeof(DT_X);
    const uint32_t dstGapBytes = dstRowBytes - validBytes;

    if (task.owStart == 0 && task.curOW == tiling.outWidth &&
        task.depthOffset == 0 && Is32BAligned(elemsPerRow) &&
        Is512BAlignedByteOffset(dstBaseByte)) {
      CopyOutContiguous(OutOffset4D(task.batchIdx, task.ohStart, 0, 0), local,
                        task.curOH * elemsPerRow);
      return;
    }

    const uint32_t rowBlocks = validBytes / BTS_DATA_BLOCK_BYTES;
    const uint32_t dstStrideBlocks = dstGapBytes / BTS_DATA_BLOCK_BYTES;
    const bool strideOk =
        Is32BAligned(elemsPerRow) && (dstGapBytes % BTS_DATA_BLOCK_BYTES) == 0 &&
        rowBlocks <= 0xFFFFU && dstStrideBlocks <= 0xFFFFU &&
        task.curOH <= 0xFFFFU;
    const bool gm512Ok =
        Is512BAlignedByteOffset(dstBaseByte) &&
        (dstGapBytes == 0 || (dstGapBytes % BTS_GM_ALIGN_BYTES) == 0);

    if (strideOk && gm512Ok) {
      AscendC::DataCopyParams params;
      params.blockCount = static_cast<uint16_t>(task.curOH);
      params.blockLen = static_cast<uint16_t>(rowBlocks);
      params.srcStride = 0;
      params.dstStride = static_cast<uint16_t>(dstStrideBlocks);
      AscendC::DataCopy(
          yGm[OutOffset4D(task.batchIdx, task.ohStart, task.owStart, task.depthOffset)],
          local, params);
      return;
    }

    for (uint32_t r = 0; r < task.curOH; ++r) {
      CopyOutContiguous(
          OutOffset4D(task.batchIdx, task.ohStart + r, task.owStart, task.depthOffset),
          local[r * rowElems], elemsPerRow);
    }
  }

  AscendC::GlobalTensor<DT_X> xGm;
  AscendC::GlobalTensor<DT_X> yGm;
  __gm__ DT_X *xGmPtr;
  __gm__ DT_X *yGmPtr;
  BatchToSpaceTilingData tiling;
  uint32_t tilesPerRow = 1;
  uint32_t rowTilesPerBatch = 1;
  uint32_t depthTiles = 1;
  uint32_t rowPackCompactOffsetElems = 0;
  uint32_t w512ChunkPx = 0;
  uint32_t w512Chunks = 0;
  uint32_t w512LaneElems = 0;
  uint32_t w512OutBase = 0;
  uint32_t w512DepthBlocks = 0;
  uint32_t c2LanePx = 0;
  uint32_t c2LaneElems = 0;
  uint32_t c2LaneStrideElems = 0;
  uint32_t c2OutElems = 0;
  uint32_t c2OutBase = 0;
  uint32_t c2PairElems = 0;
  uint32_t c2RampElems = 0;
  uint32_t c2PairsPerRamp = 0;
  uint32_t c2NRamp = 0;
  uint32_t c2TailElems = 0;
  uint32_t c2IdxElems = 0;
  uint32_t c2IdxBaseBytes = 0;
#if BTS_FAST_PATHS
  uint32_t fpDepth = 0;
  uint32_t mcChunkPx = 0;
  uint32_t mcChunks = 0;
  uint32_t mcLanePx = 0;
  uint32_t mcLaneElems = 0;
  uint32_t mcInElems = 0;
  uint32_t mcOutBase = 0;
  uint32_t mcDepthBlocks = 0;
  uint32_t hfInW = 0;
  uint32_t hfOutW = 0;
  uint32_t hfRows = 0;
  uint32_t hfPairsPerBatch = 0;
  uint32_t hfPhases = 0;
  uint32_t hfInRowElems = 0;
  uint32_t hfOutRowElems = 0;
  uint32_t hfLaneElems = 0;
  uint32_t hfInElems = 0;
  uint32_t hfOutBase = 0;
  uint32_t hfDepthBlocks = 0;
  uint32_t hfInRowBlocks = 0;
  uint32_t hfOutRowBlocks = 0;
  uint32_t giHalfPx = 0;
  uint32_t giPairElems = 0;
  uint32_t giLanePx = 0;
  uint32_t giLaneBBase = 0;
  uint32_t giInElems = 0;
  uint32_t giOutElems = 0;
  uint32_t giSlotElems = 0;
  uint32_t giIdxBaseBytes = 0;
  uint32_t giRampElems = 0;
  uint32_t giNRamp = 0;
  uint32_t giTailElems = 0;
  uint32_t giIdxElems = 0;
#endif
};

template <typename DT_X, int TUNING>
__aicore__ inline void DispatchBatchToSpaceByPath(
    GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tilingData) {
  switch (static_cast<BatchToSpaceCopyPath>(tilingData.path)) {
  case BTS_PATH_ROW_GATHER_FP32: {
    KernelBatchToSpaceImpl<DT_X, BTS_PATH_ROW_GATHER_FP32, TUNING> op;
    op.Init(x, y, tilingData);
    op.Process();
    break;
  }
#if BTS_FAST_PATHS
  case BTS_PATH_MASK_COMPACT: {
    KernelBatchToSpaceImpl<DT_X, BTS_PATH_MASK_COMPACT, TUNING> op;
    op.Init(x, y, tilingData);
    op.Process();
    break;
  }
  case BTS_PATH_MC_BIG: {
    KernelBatchToSpaceImpl<DT_X, BTS_PATH_MC_BIG, TUNING> op;
    op.Init(x, y, tilingData);
    op.Process();
    break;
  }
  case BTS_PATH_H_FUSE: {
    KernelBatchToSpaceImpl<DT_X, BTS_PATH_H_FUSE, TUNING> op;
    op.Init(x, y, tilingData);
    op.Process();
    break;
  }
  case BTS_PATH_GATHER_INTERLEAVE: {
    KernelBatchToSpaceImpl<DT_X, BTS_PATH_GATHER_INTERLEAVE, TUNING>
        op;
    op.Init(x, y, tilingData);
    op.Process();
    break;
  }
#endif
  case BTS_PATH_ROW_PLANE_FP32: {
    KernelBatchToSpaceImpl<DT_X, BTS_PATH_ROW_PLANE_FP32, TUNING> op;
    op.Init(x, y, tilingData);
    op.Process();
    break;
  }
  case BTS_PATH_WIDE_512B: {
    KernelBatchToSpaceImpl<DT_X, BTS_PATH_WIDE_512B, TUNING> op;
    op.Init(x, y, tilingData);
    op.Process();
    break;
  }
  case BTS_PATH_DEPTH_32B_PADDED: {
    KernelBatchToSpaceImpl<DT_X, BTS_PATH_DEPTH_32B_PADDED, TUNING>
        op;
    op.Init(x, y, tilingData);
    op.Process();
    break;
  }
  case BTS_PATH_DEPTH_SPLIT_TINY: {
    KernelBatchToSpaceImpl<DT_X, BTS_PATH_DEPTH_SPLIT_TINY, TUNING> op;
    op.Init(x, y, tilingData);
    op.Process();
    break;
  }
  case BTS_PATH_DEPTH_TINY_SPATIAL: {
    KernelBatchToSpaceImpl<DT_X, BTS_PATH_DEPTH_TINY_SPATIAL, TUNING>
        op;
    op.Init(x, y, tilingData);
    op.Process();
    break;
  }
  case BTS_PATH_DEPTH_512B_ALIGNED: {
    KernelBatchToSpaceImpl<DT_X, BTS_PATH_DEPTH_512B_ALIGNED, TUNING>
        op;
    op.Init(x, y, tilingData);
    op.Process();
    break;
  }
  default: {
    KernelBatchToSpaceImpl<DT_X, BTS_PATH_DEPTH_32B_ALIGNED, TUNING>
        op;
    op.Init(x, y, tilingData);
    op.Process();
    break;
  }
  }
}

template <typename DT_X, int TUNING, int STATIC_VARIANT>
__global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y,
                                          GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  (void)workspace;

  if constexpr (STATIC_VARIANT == BTS_STATIC_VARIANT_ROW_PLANE_STRIDED) {
    KernelBatchToSpaceRowPlaneStrided<DT_X> op;
    op.Init(x, y);
    op.Process();
  } else if constexpr (STATIC_VARIANT == BTS_STATIC_VARIANT_ROW_GATHER_FP32) {
    KernelBatchToSpaceRowGatherFp32<DT_X> op;
    op.Init(x, y);
    op.Process();
  } else if constexpr (STATIC_VARIANT == BTS_STATIC_VARIANT_ROW_PLANE_COMPACT) {
    KernelBatchToSpaceRowPlaneCompact<DT_X> op;
    op.Init(x, y);
    op.Process();
  } else if constexpr (STATIC_VARIANT == BTS_STATIC_VARIANT_ROW_PLANE_SMALL) {
    KernelBatchToSpaceRowPlaneSmall<DT_X> op;
    op.Init(x, y);
    op.Process();
  } else if constexpr (STATIC_VARIANT == BTS_STATIC_VARIANT_ROW_GATHER_WIDE) {
    KernelBatchToSpaceRowGatherWide<DT_X> op;
    op.Init(x, y);
    op.Process();
  } else if constexpr (STATIC_VARIANT == BTS_STATIC_VARIANT_DEPTH_SPLIT_LARGE) {
    KernelBatchToSpaceDepthSplitLarge<DT_X> op;
    op.Init(x, y);
    op.Process();
  } else if constexpr (STATIC_VARIANT == BTS_STATIC_VARIANT_DEPTH_SPLIT_TINY) {
    KernelBatchToSpaceDepthSplitTiny<DT_X> op;
    op.Init(x, y);
    op.Process();
  } else if constexpr (STATIC_VARIANT == BTS_STATIC_VARIANT_HALF_ROW_BIG) {
    KernelBatchToSpaceHalfRowBig<DT_X> op;
    op.Init(x, y);
    op.Process();
  } else if constexpr (STATIC_VARIANT == BTS_STATIC_VARIANT_MC_WIDE) {
    KernelBatchToSpaceMcWide<DT_X> op;
    op.Init(x, y);
    op.Process();
  } else if constexpr (STATIC_VARIANT == BTS_STATIC_VARIANT_H_FUSE) {
    KernelBatchToSpaceHFuse<DT_X> op;
    op.Init(x, y);
    op.Process();
  } else {
    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);
    GET_TILING_DATA_WITH_STRUCT(BatchToSpaceTilingHeader, tilingHeader, tiling);
    if (tilingHeader.mode == BTS_TILING_MODE_ROW_GATHER_FP32_STATIC) {
      KernelBatchToSpaceRowGatherFp32<DT_X> op;
      op.Init(x, y);
      op.Process();
      return;
    }
    if (tilingHeader.mode == BTS_TILING_MODE_ROW_GATHER_WIDE_STATIC) {
      KernelBatchToSpaceRowGatherWide<DT_X> op;
      op.Init(x, y);
      op.Process();
      return;
    }
#if BTS_FAST_PATHS
    if (tilingHeader.mode == BTS_TILING_MODE_MC_WIDE_STATIC &&
        sizeof(DT_X) == sizeof(uint16_t)) {
      KernelBatchToSpaceMcWide<DT_X> op;
      op.Init(x, y);
      op.Process();
      return;
    }
    if (tilingHeader.mode == BTS_TILING_MODE_H_FUSE_STATIC &&
        sizeof(DT_X) == sizeof(uint16_t)) {
      KernelBatchToSpaceHFuse<DT_X> op;
      op.Init(x, y);
      op.Process();
      return;
    }
    if (tilingHeader.mode == BTS_TILING_MODE_MC_BIGTILE &&
        sizeof(DT_X) == sizeof(uint16_t)) {
      KernelBatchToSpaceMcBigTile<DT_X> op;
      op.Init(x, y);
      op.Process();
      return;
    }
#endif
    if constexpr (TUNING != 0) {
      if (tilingHeader.mode == BTS_TILING_MODE_DEPTH_SPLIT_TINY_TUNING) {
        GET_TILING_DATA_WITH_STRUCT(BatchToSpaceTuningData, depthSplitTinyTiling, tiling);
        KernelBatchToSpaceDepthSplitTinyTuned<DT_X, TUNING> op;
        op.Init(x, y, depthSplitTinyTiling);
        op.Process();
        return;
      }
      if (tilingHeader.mode == BTS_TILING_MODE_DEPTH_SPLIT_LARGE_TUNING) {
        GET_TILING_DATA_WITH_STRUCT(BatchToSpaceTuningData, depthSplitLargeTiling, tiling);
        KernelBatchToSpaceDepthSplitLargeTuned<DT_X, TUNING> op;
        op.Init(x, y, depthSplitLargeTiling);
        op.Process();
        return;
      }
    }

    GET_TILING_DATA_WITH_STRUCT(BatchToSpaceTilingData, tilingData, tiling);
    if (tilingData.totalTasks == 0 || GetTaskRange(tilingData).length == 0) {
      return;
    }
    DispatchBatchToSpaceByPath<DT_X, TUNING>(x, y, tilingData);
  }
}
