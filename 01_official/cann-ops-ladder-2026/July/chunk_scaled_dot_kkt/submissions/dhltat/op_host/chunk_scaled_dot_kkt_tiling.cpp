/*!
 * \file chunk_scaled_dot_kkt_tiling.cpp
 * \brief ChunkScaledDotKkt operator tiling
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/chunk_scaled_dot_kkt_tiling_data.h"
#include "../op_kernel/chunk_scaled_dot_kkt_tiling_key.h"

#include <algorithm>
#include <cstdint>

namespace optiling {

constexpr int64_t DEFAULT_CHUNK_SIZE = 64;
constexpr uint64_t WORKSPACE_ALIGN = 32;
constexpr uint64_t BF16_BYTES = 2;
constexpr uint64_t WORKSPACE_SLOT_COUNT = 4;
constexpr uint32_t MANUAL_MIX_CORE_COUNT = 20;
constexpr uint32_t GROUP_ALIGNED_CORE_COUNT = 16;
constexpr int64_t DEEP_PIPELINE_TASKS_PER_CORE = 20;
constexpr int64_t SMALL_TASK_MAX = 32;
constexpr int64_t SHORT_MANUAL_TASK_MAX = 256;
constexpr int64_t MEDIUM_TASK_MAX = 512;
constexpr int64_t LARGE_ALIGNED_TASK_MIN = 2048;
constexpr int64_t VERY_LONG_TASK_MIN = 4096;
constexpr uint32_t SMALL_TASK_BLOCK_DIM = 8;

static uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return (value + align - 1) / align * align;
}

static ge::graphStatus SetWorkspaceSize(gert::TilingContext* context, size_t workspaceSize)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAic();
    if (coreNum == 0) {
        coreNum = ascendcPlatform.GetCoreNumAiv();
    }
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static inline bool HasTensorShape(const gert::StorageShape* shape)
{
    if (shape == nullptr || shape->GetStorageShape().GetDimNum() == 0) {
        return false;
    }
    const gert::Shape& storageShape = shape->GetStorageShape();
    for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
        if (storageShape.GetDim(i) <= 0) {
            return false;
        }
    }
    return true;
}

static inline int64_t GetDim(const gert::StorageShape* shape, size_t index)
{
    return shape->GetStorageShape().GetDim(index);
}

static int64_t GetChunkSize(gert::TilingContext* context)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return DEFAULT_CHUNK_SIZE;
    }
    const int64_t* chunkSize = attrs->GetAttrPointer<int64_t>(0);
    if (chunkSize == nullptr || *chunkSize <= 0) {
        return DEFAULT_CHUNK_SIZE;
    }
    return *chunkSize;
}

static uint32_t SelectScheduleMode(
    int64_t groupHeads,
    int64_t keyDim,
    int64_t headPerGroup,
    int64_t taskNum,
    int64_t chunkNum,
    uint32_t blockDim,
    bool allChunksFull)
{
    const int64_t pipelineWindow =
        static_cast<int64_t>(blockDim) * WORKSPACE_SLOT_COUNT;
    const bool fitsSinglePipelineWindow = taskNum <= pipelineWindow;
    const bool isMediumTask =
        taskNum > pipelineWindow && taskNum <= MEDIUM_TASK_MAX;
    const bool isDeepPipeline =
        taskNum > static_cast<int64_t>(blockDim) * DEEP_PIPELINE_TASKS_PER_CORE;
    const bool fitsShallowManualPipeline =
        taskNum > static_cast<int64_t>(MANUAL_MIX_CORE_COUNT) * WORKSPACE_SLOT_COUNT &&
        taskNum <= static_cast<int64_t>(MANUAL_MIX_CORE_COUNT) * DEEP_PIPELINE_TASKS_PER_CORE;
    const bool supportsLongGh2Manual =
        groupHeads == 2 &&
        taskNum >= LARGE_ALIGNED_TASK_MIN && allChunksFull;
    const bool supportsVeryLongGh2Hpg3SharedL1 =
        groupHeads == 2 &&
        keyDim == 128 &&
        headPerGroup == 3 &&
        taskNum >= VERY_LONG_TASK_MIN &&
        allChunksFull;
    const bool supportsFullAlignedGh2K256Hpg4 =
        groupHeads == 2 &&
        keyDim == 256 &&
        headPerGroup == 4 &&
        isMediumTask &&
        allChunksFull;
    const bool supportsGroupAlignedGh16K128Hpg4 =
        groupHeads == 16 &&
        keyDim == 128 &&
        headPerGroup == 4 &&
        isMediumTask && fitsShallowManualPipeline;
    const bool supportsGroupAlignedGh16K128Hpg3 =
        groupHeads == 16 &&
        keyDim == 128 &&
        headPerGroup == 3 &&
        isMediumTask && fitsShallowManualPipeline;
    const bool supportsGroupAlignedGh8 =
        groupHeads == 8 &&
        keyDim == 128 &&
        headPerGroup == 4 &&
        taskNum <= SHORT_MANUAL_TASK_MAX &&
        fitsShallowManualPipeline;
    const bool supportsGroupAlignedGh2Hpg4Tail =
        groupHeads == 2 &&
        keyDim == 128 &&
        headPerGroup == 4 &&
        isMediumTask && isDeepPipeline && !allChunksFull;
    const bool supportsManualCube =
        (blockDim == MANUAL_MIX_CORE_COUNT ||
            (groupHeads == GROUP_ALIGNED_CORE_COUNT &&
                blockDim == GROUP_ALIGNED_CORE_COUNT) ||
            ((groupHeads == 2 || groupHeads == 8) &&
                blockDim == GROUP_ALIGNED_CORE_COUNT)) &&
        (keyDim == 128 || keyDim == 256) &&
        (headPerGroup == 3 || headPerGroup == 4) &&
        ((groupHeads == 8 &&
            taskNum > SMALL_TASK_MAX && taskNum <= SHORT_MANUAL_TASK_MAX) ||
            ((groupHeads == 2 || groupHeads == 16) &&
                taskNum > SHORT_MANUAL_TASK_MAX && taskNum <= MEDIUM_TASK_MAX) ||
            supportsLongGh2Manual);

    if (supportsFullAlignedGh2K256Hpg4) {
        return CHUNKSCALEDDOTKKT_TPL_SCH_MODE_28;
    }
    if (supportsManualCube) {
        if (supportsGroupAlignedGh8) {
            return CHUNKSCALEDDOTKKT_TPL_SCH_MODE_22;
        }
        if (supportsGroupAlignedGh16K128Hpg3) {
            return CHUNKSCALEDDOTKKT_TPL_SCH_MODE_20;
        }
        if (headPerGroup == 4) {
            if (supportsGroupAlignedGh2Hpg4Tail) {
                return CHUNKSCALEDDOTKKT_TPL_SCH_MODE_24;
            }
            if (supportsGroupAlignedGh16K128Hpg4) {
                return CHUNKSCALEDDOTKKT_TPL_SCH_MODE_19;
            }
            if (isDeepPipeline) {
                return keyDim == 128 && groupHeads == 2 ?
                    CHUNKSCALEDDOTKKT_TPL_SCH_MODE_10 :
                    CHUNKSCALEDDOTKKT_TPL_SCH_MODE_9;
            }
            if (groupHeads == 2 && keyDim == 128) {
                return CHUNKSCALEDDOTKKT_TPL_SCH_MODE_12;
            }
            if ((groupHeads == 2 && keyDim == 256) || groupHeads == 16) {
                return CHUNKSCALEDDOTKKT_TPL_SCH_MODE_15;
            }
            return keyDim == 128 ?
                CHUNKSCALEDDOTKKT_TPL_SCH_MODE_16 :
                CHUNKSCALEDDOTKKT_TPL_SCH_MODE_8;
        }

        if (isDeepPipeline) {
            if (supportsVeryLongGh2Hpg3SharedL1) {
                return CHUNKSCALEDDOTKKT_TPL_SCH_MODE_18;
            }
            return keyDim == 256 && groupHeads == 2 ?
                CHUNKSCALEDDOTKKT_TPL_SCH_MODE_11 :
                CHUNKSCALEDDOTKKT_TPL_SCH_MODE_7;
        }
        if (groupHeads == 2) {
            return keyDim == 128 ?
                CHUNKSCALEDDOTKKT_TPL_SCH_MODE_14 :
                CHUNKSCALEDDOTKKT_TPL_SCH_MODE_6;
        }
        if (keyDim == 128 && groupHeads == 16) {
            return CHUNKSCALEDDOTKKT_TPL_SCH_MODE_6;
        }
        return CHUNKSCALEDDOTKKT_TPL_SCH_MODE_13;
    }

    if (groupHeads == 2 && keyDim == 128 &&
        (headPerGroup == 3 || headPerGroup == 4) &&
        fitsSinglePipelineWindow && allChunksFull) {
        return CHUNKSCALEDDOTKKT_TPL_SCH_MODE_17;
    }
    if (keyDim == 128 && headPerGroup >= 2 &&
        taskNum >= LARGE_ALIGNED_TASK_MIN && allChunksFull) {
        return CHUNKSCALEDDOTKKT_TPL_SCH_MODE_5;
    }
    if (groupHeads == 8 && headPerGroup == 4 &&
        isMediumTask && allChunksFull) {
        return CHUNKSCALEDDOTKKT_TPL_SCH_MODE_4;
    }
    if (fitsSinglePipelineWindow && groupHeads == blockDim &&
        chunkNum <= static_cast<int64_t>(WORKSPACE_SLOT_COUNT) && allChunksFull) {
        return CHUNKSCALEDDOTKKT_TPL_SCH_MODE_3;
    }
    if (groupHeads == 2 && keyDim == 128 &&
        (headPerGroup == 3 || headPerGroup == 4) && isMediumTask) {
        return CHUNKSCALEDDOTKKT_TPL_SCH_MODE_1;
    }
    if (headPerGroup == 4 && isMediumTask) {
        return CHUNKSCALEDDOTKKT_TPL_SCH_MODE_2;
    }
    return CHUNKSCALEDDOTKKT_TPL_SCH_MODE_0;
}

static ge::graphStatus GetMatmulTiling(
    gert::TilingContext* context,
    const platform_ascendc::PlatformAscendC& ascendcPlatform,
    int64_t chunkSize,
    int64_t groupHeads,
    int64_t keyDim,
    int32_t fixedBaseM,
    int32_t fixedBaseN,
    int32_t fixedBaseK,
    AscendC::tiling::TCubeTiling& matmulTiling)
{
    const int64_t kRowStride = groupHeads * keyDim;
    OP_CHECK_IF(chunkSize > INT32_MAX || keyDim > INT32_MAX || kRowStride > INT32_MAX,
        OP_LOGE(context, "chunkSize/keyDim/kRowStride exceed int32 range"),
        return ge::GRAPH_FAILED);

    matmul_tiling::MatmulApiTiling mmTiling(ascendcPlatform);
    mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_BF16, false);
    mmTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_BF16, true);
    mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_FLOAT);
    mmTiling.SetBias(false);
    mmTiling.SetMadType(matmul_tiling::MatrixMadType::NORMAL);
    mmTiling.SetShape(static_cast<int32_t>(chunkSize), static_cast<int32_t>(chunkSize), static_cast<int32_t>(keyDim));
    mmTiling.SetOrgShape(static_cast<int32_t>(chunkSize), static_cast<int32_t>(chunkSize),
        static_cast<int32_t>(kRowStride), static_cast<int32_t>(kRowStride));
    mmTiling.SetBufferSpace(-1, -1, -1);
    if (fixedBaseK > 0) {
        OP_CHECK_IF(
            mmTiling.SetFixSplit(fixedBaseM, fixedBaseN, fixedBaseK) != 0,
            OP_LOGE(context, "Set matmul fixed split failed"),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(
        mmTiling.GetTiling(matmulTiling) == -1,
        OP_LOGE(context, "Get matmul tiling failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        matmulTiling.singleCoreM <= 0 || matmulTiling.singleCoreN <= 0 ||
            matmulTiling.baseM <= 0 || matmulTiling.baseN <= 0,
        OP_LOGE(context, "matmul tiling result is invalid"),
        return ge::GRAPH_FAILED);
    matmulTiling.singleCoreM = static_cast<int32_t>(chunkSize);
    matmulTiling.singleCoreN = static_cast<int32_t>(chunkSize);
    matmulTiling.singleCoreK = static_cast<int32_t>(keyDim);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ChunkScaledDotKktTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    (void)ubSize;
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    const gert::StorageShape* kStorageShape = context->GetInputShape(0);
    const gert::StorageShape* betaStorageShape = context->GetInputShape(1);
    const gert::StorageShape* gStorageShape = context->GetInputShape(2);
    const gert::StorageShape* offsetsStorageShape = context->GetInputShape(3);
    OP_CHECK_IF(!HasTensorShape(kStorageShape), OP_LOGE(context, "k shape is invalid"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!HasTensorShape(betaStorageShape), OP_LOGE(context, "beta shape is invalid"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!HasTensorShape(gStorageShape), OP_LOGE(context, "g_cumsum shape is invalid"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!HasTensorShape(offsetsStorageShape), OP_LOGE(context, "chunk_offsets shape is invalid"),
        return ge::GRAPH_FAILED);

    const gert::Shape& kShape = kStorageShape->GetStorageShape();
    const gert::Shape& betaShape = betaStorageShape->GetStorageShape();
    const gert::Shape& gShape = gStorageShape->GetStorageShape();
    const gert::Shape& offsetsShape = offsetsStorageShape->GetStorageShape();
    OP_CHECK_IF(kShape.GetDimNum() != 4, OP_LOGE(context, "k must be 4D"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(betaShape.GetDimNum() != 3, OP_LOGE(context, "beta must be 3D"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(gShape.GetDimNum() != 3, OP_LOGE(context, "g_cumsum must be 3D"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(offsetsShape.GetDimNum() != 1, OP_LOGE(context, "chunk_offsets must be 1D"), return ge::GRAPH_FAILED);

    const int64_t batch = GetDim(kStorageShape, 0);
    const int64_t seqLen = GetDim(kStorageShape, 1);
    const int64_t groupHeads = GetDim(kStorageShape, 2);
    const int64_t keyDim = GetDim(kStorageShape, 3);
    const int64_t betaBatch = GetDim(betaStorageShape, 0);
    const int64_t heads = GetDim(betaStorageShape, 1);
    const int64_t betaSeqLen = GetDim(betaStorageShape, 2);
    const int64_t chunkNum = GetDim(offsetsStorageShape, 0) - 1;
    const int64_t chunkSize = GetChunkSize(context);

    OP_CHECK_IF(batch != 1 || betaBatch != batch, OP_LOGE(context, "only batch=1 is supported"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(betaSeqLen != seqLen || GetDim(gStorageShape, 0) != batch || GetDim(gStorageShape, 1) != heads ||
                    GetDim(gStorageShape, 2) != seqLen,
        OP_LOGE(context, "input shape mismatch"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(groupHeads <= 0 || heads <= 0 || heads % groupHeads != 0,
        OP_LOGE(context, "heads must be divisible by groupHeads"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyDim != 128 && keyDim != 256, OP_LOGE(context, "keyDim must be 128 or 256"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(chunkNum <= 0 || chunkSize <= 0 || chunkSize > DEFAULT_CHUNK_SIZE,
        OP_LOGE(context, "invalid chunk settings"),
        return ge::GRAPH_FAILED);

    const int64_t taskNum = chunkNum * groupHeads;
    const int64_t headPerGroup = heads / groupHeads;
    const bool useK256Hpg3FullK = keyDim == 256 && headPerGroup == 3 &&
        groupHeads != 2 &&
        taskNum > SMALL_TASK_MAX && taskNum <= MEDIUM_TASK_MAX;
    const bool useK256Gh2Hpg4FullK = keyDim == 256 && groupHeads == 2 &&
        headPerGroup == 4 &&
        taskNum > SMALL_TASK_MAX && taskNum <= MEDIUM_TASK_MAX;
    const int32_t fixedBaseM = static_cast<int32_t>(chunkSize);
    const int32_t fixedBaseN = static_cast<int32_t>(chunkSize);
    const int32_t fixedBaseK = (useK256Hpg3FullK || useK256Gh2Hpg4FullK) ? 256 : 0;
    ChunkScaledDotKktTilingData* tiling = context->GetTilingData<ChunkScaledDotKktTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        GetMatmulTiling(context, ascendcPlatform, chunkSize, groupHeads, keyDim,
            fixedBaseM, fixedBaseN, fixedBaseK, tiling->matmulTiling) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetMatmulTiling error"),
        return ge::GRAPH_FAILED);

    const uint32_t defaultBlockDim =
        static_cast<uint32_t>(std::max<int64_t>(1, std::min(coreNum, taskNum)));
    const bool useFullAlignedGh2K256Hpg4BlockDim =
        groupHeads == 2 && keyDim == 256 && headPerGroup == 4 &&
        taskNum > static_cast<int64_t>(MANUAL_MIX_CORE_COUNT) * WORKSPACE_SLOT_COUNT &&
        taskNum <= static_cast<int64_t>(MANUAL_MIX_CORE_COUNT) * DEEP_PIPELINE_TASKS_PER_CORE &&
        chunkSize == DEFAULT_CHUNK_SIZE && seqLen % chunkSize == 0 &&
        chunkNum == seqLen / chunkSize;
    const bool useGroupAlignedBlockDim =
        (groupHeads == GROUP_ALIGNED_CORE_COUNT && keyDim == 128 &&
            (headPerGroup == 3 || headPerGroup == 4) &&
            taskNum > static_cast<int64_t>(MANUAL_MIX_CORE_COUNT) * WORKSPACE_SLOT_COUNT &&
            taskNum <= static_cast<int64_t>(MANUAL_MIX_CORE_COUNT) * DEEP_PIPELINE_TASKS_PER_CORE) ||
        (groupHeads == 8 && keyDim == 128 && headPerGroup == 4 &&
            taskNum > static_cast<int64_t>(MANUAL_MIX_CORE_COUNT) * WORKSPACE_SLOT_COUNT &&
            taskNum <= SHORT_MANUAL_TASK_MAX) ||
        (groupHeads == 2 && keyDim == 128 &&
            headPerGroup == 4 &&
            taskNum > static_cast<int64_t>(MANUAL_MIX_CORE_COUNT) * DEEP_PIPELINE_TASKS_PER_CORE &&
            taskNum <= MEDIUM_TASK_MAX &&
            !(chunkSize == DEFAULT_CHUNK_SIZE && seqLen % chunkSize == 0 &&
                chunkNum == seqLen / chunkSize));
    const uint32_t blockDim =
        taskNum <= SMALL_TASK_MAX ? SMALL_TASK_BLOCK_DIM :
        (useFullAlignedGh2K256Hpg4BlockDim ? GROUP_ALIGNED_CORE_COUNT :
            (useGroupAlignedBlockDim ? GROUP_ALIGNED_CORE_COUNT : defaultBlockDim));
    tiling->matmulTiling.usedCoreNum = 1;

    const uint64_t resultWorkspacePerCore =
        AlignUp(static_cast<uint64_t>(chunkSize) * static_cast<uint64_t>(chunkSize) * sizeof(float), WORKSPACE_ALIGN);
    const uint64_t compactWorkspacePerCore =
        AlignUp(static_cast<uint64_t>(chunkSize) * static_cast<uint64_t>(keyDim) * BF16_BYTES, WORKSPACE_ALIGN);
    const uint64_t userWorkspacePerCore =
        (compactWorkspacePerCore + resultWorkspacePerCore) * WORKSPACE_SLOT_COUNT;
    const uint64_t userWorkspace = userWorkspacePerCore * static_cast<uint64_t>(blockDim);
    const size_t workspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize() + userWorkspace);

    tiling->seqLen = static_cast<uint32_t>(seqLen);
    tiling->heads = static_cast<uint32_t>(heads);
    tiling->groupHeads = static_cast<uint32_t>(groupHeads);
    tiling->keyDim = static_cast<uint32_t>(keyDim);
    tiling->chunkNum = static_cast<uint32_t>(chunkNum);
    tiling->chunkSize = static_cast<uint32_t>(chunkSize);
    tiling->headPerGroup = static_cast<uint32_t>(heads / groupHeads);
    tiling->taskNum = static_cast<uint32_t>(taskNum);
    tiling->taskCoreNum = blockDim;
    tiling->blockFactor = (taskNum + blockDim - 1) / blockDim;
    tiling->workspacePerCore = static_cast<uint32_t>(userWorkspacePerCore);

    OP_CHECK_IF(
        SetWorkspaceSize(context, workspaceSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "SetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    context->SetBlockDim(blockDim);
    const bool allChunksFull =
        chunkSize == DEFAULT_CHUNK_SIZE &&
        seqLen % chunkSize == 0 && chunkNum == seqLen / chunkSize;
    const uint32_t scheduleMode = SelectScheduleMode(
        groupHeads, keyDim, headPerGroup, taskNum, chunkNum, blockDim, allChunksFull);
    context->SetTilingKey(GET_TPL_TILING_KEY(scheduleMode));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForChunkScaledDotKkt([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ChunkScaledDotKktCompileInfo {};

IMPL_OP_OPTILING(ChunkScaledDotKkt)
    .Tiling(ChunkScaledDotKktTilingFunc)
    .TilingParse<ChunkScaledDotKktCompileInfo>(TilingParseForChunkScaledDotKkt);

} // namespace optiling
