// Host侧Tiling实现
#define K_MAX_SHAPE_DIM 0 // ← 必须在 include 之前
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
constexpr uint32_t VECTOR_FLOAT_ELEMENTS = 64;
constexpr uint32_t UB_RESERVE_SIZE = 8 * 1024;
constexpr uint32_t MAX_INPUT_QUEUE_PADDING = 256;
constexpr uint32_t ONE_TILE_CORE_LENGTH_LIMIT = 256 * 1024;
constexpr uint32_t FULL_CORE_TARGET_ELEMS_PER_CORE = VECTOR_FLOAT_ELEMENTS;
constexpr uint32_t BASELINE_TARGET_QUEUE_NUM = 2;
// ver103 改成 1-scratch 后，baseline target 也跟随 erf_tiling.h。
// 否则默认 activeCore 估算仍会按旧 scratch 数的 tile 上限计算。
// 如果后续想故意多激活核，可以单独用 ERF_TARGET_ELEMS_PER_CORE / ERF_ACTIVE_CORE 扫。
constexpr uint32_t BASELINE_TARGET_SCRATCH_TILES =
    ERF_CONFIG_FORCE_SCRATCH_TILES <= 1
        ? 1U
        : (ERF_CONFIG_FORCE_SCRATCH_TILES == 2 ? 2U : 3U);
// 与 kernel 中 ERF_V103_BANK_STAGGER_BYTES 保持严格一致。
// 1-scratch 时 CalcStaggerCount(1)=0，因此不会额外吃 256B 错位 padding。
constexpr uint32_t ERF_VECCALC_BANK_STAGGER_BYTES = 256;

inline uint32_t CalcStaggerCount(uint32_t scratchTiles) {
  return scratchTiles >= 2 ? scratchTiles - 1U : 0U;
}

// ============================================================================
// Compile-time tuning control panel.
//
// Set a value to -1 to keep auto/env behavior. Set a valid value to force that
// policy in host tiling. For queue/pipe/axpy this also selects the
// corresponding kernel template through the tiling key, so the AICore code uses
// TQue<..., 1> or TQue<..., 2> directly instead of a runtime queue depth.
//
// Useful sweeps:
//   ERF_FORCE_QUEUE_NUM = 1 or 2
//   ERF_FORCE_INPUT_QUEUE_PADDING = 0 or 256
//   ERF_FORCE_PIPE_IMPL = 0(simple) or 1(prefetch), stored in tiling data
//   ERF_FORCE_AXPY_H1 = kept in the tiling key for compatibility; ver17 does
//                       not currently use Axpy.
//   ERF_FORCE_SCRATCH_TILES = follows erf_tiling.h; current ver103 uses 1
//   scratch tile. ERF_FORCE_TILE_LENGTH = 1/2/... for tiny-shape and UB cliff
//   checks
// ============================================================================
// Current ver103 policy: Q2 + 256B input queue padding, one scratch tile,
// and prefetch only when pipelineMode chooses it.
constexpr int32_t ERF_FORCE_QUEUE_NUM = 2;             // 恢复：Q2+预取
constexpr int32_t ERF_FORCE_INPUT_QUEUE_PADDING = 256; // valid: 0..256 bytes
constexpr int32_t ERF_FORCE_AXPY_H1 = 1;               // valid: 0, 1
constexpr int32_t ERF_FORCE_PIPE_IMPL = 1;             // valid: 0, 1
constexpr int32_t ERF_FORCE_SCRATCH_TILES = ERF_CONFIG_FORCE_SCRATCH_TILES;
constexpr int32_t ERF_FORCE_ACTIVE_CORE = -1;           // valid: >= 1
constexpr int32_t ERF_FORCE_TILE_LENGTH = -1;           // valid: >= 1
constexpr int32_t ERF_FORCE_TARGET_ELEMS_PER_CORE = -1; // valid: >= 1
constexpr int32_t ERF_FORCE_UB_RESERVE_SIZE = -1;       // valid: >= 0 bytes

enum QueuePolicy : uint32_t {
  QUEUE_POLICY_AUTO = 0,
  QUEUE_POLICY_Q1 = 1,
  QUEUE_POLICY_Q2 = 2,
};

inline uint32_t CeilDiv(uint32_t x, uint32_t y) {
  return y == 0 ? 0 : (x + y - 1) / y;
}

inline uint32_t AlignDown(uint32_t x, uint32_t align) {
  return align == 0 ? x : (x / align) * align;
}

inline uint32_t AlignUp(uint32_t x, uint32_t align) {
  return align == 0 ? x : ((x + align - 1) / align) * align;
}

inline uint32_t ReadEnvU32(const char *name, uint32_t fallback) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  char *end = nullptr;
  unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value) {
    return fallback;
  }
  return parsed > std::numeric_limits<uint32_t>::max()
             ? std::numeric_limits<uint32_t>::max()
             : static_cast<uint32_t>(parsed);
}

inline uint32_t ReadEnvBool(const char *name, uint32_t fallback) {
  uint32_t value = ReadEnvU32(name, fallback);
  return value == 0 ? 0U : 1U;
}

inline uint32_t ForcedBoolOrEnv(int32_t forced, const char *name,
                                uint32_t fallback) {
  if (forced >= 0) {
    return forced == 0 ? 0U : 1U;
  }
  return ReadEnvBool(name, fallback);
}

inline uint32_t ForcedU32OrEnv(int32_t forced, const char *name,
                               uint32_t fallback) {
  if (forced >= 0) {
    return static_cast<uint32_t>(forced);
  }
  return ReadEnvU32(name, fallback);
}

inline uint32_t ReadPaddingEnv() {
  if (ERF_FORCE_INPUT_QUEUE_PADDING >= 0) {
    return std::min(static_cast<uint32_t>(ERF_FORCE_INPUT_QUEUE_PADDING),
                    MAX_INPUT_QUEUE_PADDING);
  }
  uint32_t padding = ReadEnvU32("ERF_INPUT_QUEUE_PADDING", 0);
  return std::min(padding, MAX_INPUT_QUEUE_PADDING);
}

inline uint32_t ReadPipeImplEnv() {
  if (ERF_FORCE_PIPE_IMPL >= 0) {
    return ERF_FORCE_PIPE_IMPL == 0 ? 0U : 1U;
  }
  const char *value = std::getenv("ERF_PIPE_IMPL");
  if (value == nullptr || value[0] == '\0') {
    return 1U; // prefetch
  }
  if (std::strcmp(value, "simple") == 0 || std::strcmp(value, "0") == 0) {
    return 0U;
  }
  if (std::strcmp(value, "prefetch") == 0 || std::strcmp(value, "1") == 0) {
    return 1U;
  }
  return 1U;
}

inline QueuePolicy ReadQueuePolicyEnv() {
  if (ERF_FORCE_QUEUE_NUM == 1) {
    return QUEUE_POLICY_Q1;
  }
  if (ERF_FORCE_QUEUE_NUM == 2) {
    return QUEUE_POLICY_Q2;
  }
  const char *value = std::getenv("ERF_QUEUE_POLICY");
  if (value == nullptr || value[0] == '\0' || std::strcmp(value, "auto") == 0) {
    return QUEUE_POLICY_AUTO;
  }
  if (std::strcmp(value, "q1") == 0 || std::strcmp(value, "1") == 0) {
    return QUEUE_POLICY_Q1;
  }
  if (std::strcmp(value, "q2") == 0 || std::strcmp(value, "2") == 0) {
    return QUEUE_POLICY_Q2;
  }
  return QUEUE_POLICY_AUTO;
}

inline const char *QueuePolicyName(QueuePolicy policy) {
  if (policy == QUEUE_POLICY_Q1) {
    return "q1";
  }
  if (policy == QUEUE_POLICY_Q2) {
    return "q2";
  }
  return "auto";
}

inline const char *PipeImplName(uint32_t pipeImplMode) {
  return pipeImplMode == 0 ? "simple" : "prefetch";
}

inline uint32_t GetScratchTiles() {
  // 跟随 erf_tiling.h 中的 ERF_CONFIG_FORCE_SCRATCH_TILES（ver103 = 1）
  if (ERF_FORCE_SCRATCH_TILES <= 1)
    return 1U;
  if (ERF_FORCE_SCRATCH_TILES == 2)
    return 2U;
  return 3U;
}

inline uint32_t ChooseTargetElemsPerCore(uint32_t length,
                                         uint32_t baselineTargetTileLength) {
  // ★ 优化：不再区分大小数据量。始终以基于UB计算出的理想Tile长度为目标，
  // 这有助于在激活核心数和单核工作量之间取得更好的平衡，
  // 从而让整核与尾核的工作量差异百分比降低。
  return std::max(baselineTargetTileLength, VECTOR_FLOAT_ELEMENTS);
}

// UB 占用：
//   inQueueX  : queueNum × (tile*4 + inputQueuePadding)
//   outQueueY : queueNum × tile*4
//   calc      : scratchTiles × tile*4  + (scratchTiles - 1) × 256B stagger
inline uint64_t GetTileBytes(uint32_t tileLength, uint32_t queueNum,
                             uint32_t inputQueuePadding,
                             uint32_t scratchTiles) {
  uint64_t queueBytes = static_cast<uint64_t>(queueNum) *
                        (tileLength * sizeof(float) + inputQueuePadding);
  queueBytes += static_cast<uint64_t>(queueNum) * tileLength * sizeof(float);
  uint64_t calcBytes =
      static_cast<uint64_t>(tileLength) * sizeof(float) * scratchTiles;
  calcBytes += static_cast<uint64_t>(CalcStaggerCount(scratchTiles)) *
               ERF_VECCALC_BANK_STAGGER_BYTES;
  return queueBytes + calcBytes;
}

inline uint32_t ComputeMaxTileLength(uint64_t usableUbSize, uint32_t queueNum,
                                     uint32_t inputQueuePadding,
                                     uint32_t scratchTiles) {
  uint64_t fixedBytes = static_cast<uint64_t>(queueNum) * inputQueuePadding;
  fixedBytes += static_cast<uint64_t>(CalcStaggerCount(scratchTiles)) *
                ERF_VECCALC_BANK_STAGGER_BYTES;
  uint64_t bytesPerElement =
      static_cast<uint64_t>(queueNum) * 2 * sizeof(float) +
      scratchTiles * sizeof(float);
  uint64_t rawTileLength = usableUbSize > fixedBytes
                               ? (usableUbSize - fixedBytes) / bytesPerElement
                               : 0;
  uint32_t tileLength = static_cast<uint32_t>(
      AlignDown(static_cast<uint32_t>(rawTileLength), VECTOR_FLOAT_ELEMENTS));
  tileLength = std::max(tileLength, VECTOR_FLOAT_ELEMENTS);
  while (tileLength > VECTOR_FLOAT_ELEMENTS &&
         GetTileBytes(tileLength, queueNum, inputQueuePadding, scratchTiles) >
             usableUbSize) {
    tileLength -= VECTOR_FLOAT_ELEMENTS;
  }
  return tileLength;
}

struct ErfTunePoint {
  uint32_t maxLen;
  uint32_t activeCore;
  uint32_t tileLength; // 0 = AUTO_TILE，继续使用原 host tile 公式
};

constexpr uint32_t AUTO_TILE = 0;

static constexpr ErfTunePoint kTuneTable[] = {
    // tiny
    {1, 8, 128},
    {2, 8, 192},
    {4, 8, 128},
    {8, 8, 192},
    {12, 8, 192},
    {16, 8, 128},

    // small
    {32, 8, 256},
    {64, 10, 384},
    {128, 8, 2048},

    // small-middle
    // 旧的 256/512/768/1024/2048 core4 分段会让 OJ 7/10 变差，所以先不恢复。
    {4096, 8, 512},

    // 4096 ~ 8192
    {7168, 8, 2048},
    {8192, 8, 12288},

    // mid
    {9216, 8, 65536},
    {9472, 8, 12288},
    {16384, 8, 2048},
    {32768, 8, 12288},

    // large
    {65536, 16, 8192},

    // sampled points around (130048, 131072]
    {130049, 20, 12288},
    {130304, 20, 8192},
    {130560, 24, 12288},
    {130816, 20, 8192},
    {131072, 20, 12288},

    {262144, 28, 98304},

    // sampled points around (287744, 288768]
    {287745, 26, 12288},
    {288000, 26, 12288},
    {288256, 26, 12288},
    {288512, 28, 12288},
    {288768, 24, 3072},

    // large transition
    {524288, 32, 3072},
    {655360, 36, 3072},
    {786432, 34, 3072},
    {917504, 32, 4096},

    {1048576, 40, 4096},
};

inline ErfTunePoint SelectTune(uint32_t length) {
  for (const auto &point : kTuneTable) {
    if (length <= point.maxLen) {
      return point;
    }
  }
  return {length, 40, 4096};
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t numCoresAiv = static_cast<uint32_t>(platform.GetCoreNumAiv());
  uint64_t ubSize = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

  const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
  ge::DataType dtypeX = tensorX->GetDataType();
  uint32_t length = static_cast<uint32_t>(tensorX->GetShapeSize());

  uint32_t dtX = static_cast<uint32_t>(dtypeX);
  uint32_t axpyMode = ForcedBoolOrEnv(ERF_FORCE_AXPY_H1, "ERF_AXPY_H1", 1);
  uint32_t pipeImplMode = ReadPipeImplEnv();
  uint32_t inputQueuePadding = ReadPaddingEnv();
  QueuePolicy queuePolicy = ReadQueuePolicyEnv();
  uint32_t scratchTiles = GetScratchTiles();

  uint32_t ubReserveSize = ForcedU32OrEnv(
      ERF_FORCE_UB_RESERVE_SIZE, "ERF_UB_RESERVE_SIZE", UB_RESERVE_SIZE);
  uint64_t usableUbSize =
      ubSize > ubReserveSize ? ubSize - ubReserveSize : ubSize;
  uint32_t maxTileLengthQ1 =
      ComputeMaxTileLength(usableUbSize, 1, inputQueuePadding, scratchTiles);
  uint32_t maxTileLengthQ2 =
      ComputeMaxTileLength(usableUbSize, 2, inputQueuePadding, scratchTiles);
  uint32_t baselineTargetTileLength =
      ComputeMaxTileLength(usableUbSize, BASELINE_TARGET_QUEUE_NUM,
                           inputQueuePadding, BASELINE_TARGET_SCRATCH_TILES);

  uint32_t coreLimit = numCoresAiv == 0 ? 1 : numCoresAiv;
  uint32_t targetElemsPerCore =
      ChooseTargetElemsPerCore(length, baselineTargetTileLength);
  targetElemsPerCore =
      ForcedU32OrEnv(ERF_FORCE_TARGET_ELEMS_PER_CORE,
                     "ERF_TARGET_ELEMS_PER_CORE", targetElemsPerCore);
  targetElemsPerCore = std::max(targetElemsPerCore, VECTOR_FLOAT_ELEMENTS);

  uint32_t activeCoreNum =
      length == 0 ? 1
                  : std::min(coreLimit,
                             std::max(1U, CeilDiv(length, targetElemsPerCore)));

  ErfTunePoint tune = SelectTune(length);
  if (tune.activeCore != 0) {
    activeCoreNum = std::min(coreLimit, std::max(1U, tune.activeCore));
  }

  // Priority: env override, compile-time override, then kTuneTable.
  // kTuneTable uses the same semantics as ERF_ACTIVE_CORE: request this
  // blockDim directly, clipped only by the physical core limit.
  uint32_t activeCoreOverride = ReadEnvU32("ERF_ACTIVE_CORE", 0);
  if (activeCoreOverride != 0) {
    activeCoreNum = std::min(coreLimit, std::max(1U, activeCoreOverride));
  } else if (ERF_FORCE_ACTIVE_CORE >= 0) {
    activeCoreNum = std::min(
        coreLimit, std::max(1U, static_cast<uint32_t>(ERF_FORCE_ACTIVE_CORE)));
  }

  // ★ 优化：引入基于 32B 对齐的尾核 Tiling 策略
  uint32_t formerNum = 0;
  uint32_t formerLength = 0;
  uint32_t tailLength = 0;
  // 保留 blockLength 概念，用于评估 queue / tile
  uint32_t blockLength = 0;

  if (length > 0 && activeCoreNum > 0) {
    // 强制按 VECTOR_FLOAT_ELEMENTS (即 256 Bytes) 粒度进行整体对齐
    // 这比严格的 32B (8个 float) 对齐更保守，以兼容旧版设计和 DataCopyPadExt
    uint32_t alignedLength = AlignUp(length, VECTOR_FLOAT_ELEMENTS);

    // 按粒度块的总数分配给各个核
    uint32_t totalBlocksCount = alignedLength / VECTOR_FLOAT_ELEMENTS;

    formerNum = totalBlocksCount % activeCoreNum;

    uint32_t baseCoreElems =
        (totalBlocksCount / activeCoreNum) * VECTOR_FLOAT_ELEMENTS;

    tailLength = baseCoreElems;
    formerLength = tailLength + (formerNum > 0 ? VECTOR_FLOAT_ELEMENTS : 0);

    // 取较大的处理量用于评估 UB/Tile 参数
    blockLength = formerNum > 0 ? formerLength : tailLength;
  }

  uint32_t queueNum = 2;
  if (queuePolicy == QUEUE_POLICY_Q1) {
    queueNum = 1;
  } else if (queuePolicy == QUEUE_POLICY_Q2) {
    queueNum = 2;
  } else {
    queueNum = blockLength <= maxTileLengthQ1 ? 1U : 2U;
  }
  uint32_t maxTileLength = queueNum == 1 ? maxTileLengthQ1 : maxTileLengthQ2;

  uint32_t requestedTileLength =
      blockLength == 0 ? VECTOR_FLOAT_ELEMENTS
                       : AlignUp(std::min(blockLength, maxTileLength),
                                 VECTOR_FLOAT_ELEMENTS);
  uint32_t tileOverride = tune.tileLength;
  // Priority: env override, compile-time override, then kTuneTable.
  // kTuneTable uses the same semantics as ERF_TILE_LENGTH.
  uint32_t envTileOverride = ReadEnvU32("ERF_TILE_LENGTH", 0);
  if (envTileOverride != 0) {
    tileOverride = envTileOverride;
  } else if (ERF_FORCE_TILE_LENGTH >= 0) {
    tileOverride = static_cast<uint32_t>(ERF_FORCE_TILE_LENGTH);
  }
  if (tileOverride != 0) {
    requestedTileLength = std::max(1U, std::min(tileOverride, maxTileLength));
  }
  uint32_t tileLength = std::max(requestedTileLength, 1U);
  while (tileLength > VECTOR_FLOAT_ELEMENTS &&
         GetTileBytes(tileLength, queueNum, inputQueuePadding, scratchTiles) >
             usableUbSize) {
    tileLength -= VECTOR_FLOAT_ELEMENTS;
  }
  uint32_t tileBufferLength = std::max(tileLength, VECTOR_FLOAT_ELEMENTS);

  uint32_t tileCount = tileLength == 0 ? 0 : CeilDiv(blockLength, tileLength);
  uint32_t tailTileLength =
      tileCount == 0 ? 0 : blockLength - (tileCount - 1) * tileLength;
  if (tileCount != 0 && tailTileLength == 0) {
    tailTileLength = tileLength;
  }
  // Queue depth and pipeline scheduling are intentionally separate.
  // ProcessPipeline prefetches the next tile before computing the current tile,
  // so it is only legal when the queue has depth 2.
  uint32_t queue2Mode = queueNum == 2 ? 1U : 0U;
  uint32_t pipelineMode =
      (pipeImplMode != 0 && queueNum == 2 && tileCount > 1) ? 1U : 0U;
  uint32_t tailAligned32B = ((tailTileLength * sizeof(float)) % 32) == 0;

  // ★ 写入新的 Tiling 结构
  ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
  tiling->totalLength = length;
  tiling->formerNum = formerNum;
  tiling->formerLength = formerLength;
  tiling->tailLength = tailLength;
  tiling->tileLength = tileLength;
  tiling->inputQueuePadding = static_cast<uint16_t>(inputQueuePadding);
  tiling->pipelineMode = static_cast<uint8_t>(pipelineMode); // 合并 pipeImpl
  tiling->reserved = 0;
  context->GetRawTilingData()->SetDataSize(sizeof(ErfTilingData));

  ASCENDC_TPL_SEL_PARAM(context, dtX, queue2Mode, pipelineMode, axpyMode);

  if (std::getenv("ERF_TILING_DEBUG") != nullptr) {
    std::printf(
        "[ERF_TILING] length=%u ub=%lu ub_reserve=%u usable_ub=%lu "
        "active_core=%u former_num=%u former_len=%u tail_len=%u "
        "block=%u tile=%u queue_num=%u pipeline=%u tile_count=%u "
        "tail=%u max_tile_q1=%u max_tile_q2=%u "
        "target_elems_per_core=%u tile_bytes=%lu scratch_tiles=%u "
        "tail_aligned_32b=%u\n",
        length, static_cast<unsigned long>(ubSize), ubReserveSize,
        static_cast<unsigned long>(usableUbSize), activeCoreNum, formerNum,
        formerLength, tailLength, blockLength, tileLength, queueNum,
        pipelineMode, tileCount, tailTileLength, maxTileLengthQ1,
        maxTileLengthQ2, targetElemsPerCore,
        static_cast<unsigned long>(GetTileBytes(
            tileBufferLength, queueNum, inputQueuePadding, scratchTiles)),
        scratchTiles, tailAligned32B);
  }
  // std::printf("[ERF_DBG] HOST Tiling: length=%u ub_size=%lu usable_ub=%lu "
  //             "active_core=%u former_num=%u former_len=%u tail_len=%u "
  //             "tile_len=%u queue_num=%u pipe_mode=%u\n",
  //             length, static_cast<unsigned long>(ubSize),
  //             static_cast<unsigned long>(usableUbSize), activeCoreNum,
  //             formerNum, formerLength, tailLength, tileLength, queueNum,
  //             pipelineMode);

  context->SetBlockDim(activeCoreNum);
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;
  return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
  const gert::Shape *xShape = context->GetInputShape(0);
  gert::Shape *yShape = context->GetOutputShape(0);
  *yShape = *xShape;
  return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
  context->SetOutputDataType(0, context->GetInputDataType(0));
  return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Erf : public OpDef {
public:
  explicit Erf(const char *name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND});
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND});
    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
  }
};
OP_ADD(Erf);
} // namespace ops
