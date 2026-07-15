/*!
 * \file hard_swish_tiling.cpp
 * \brief HardSwish tiling implementation
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/hard_swish_tiling_data.h"
#include "../op_kernel/hard_swish_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t INPUT_BUFFER_NUM = 2;
constexpr int64_t OUTPUT_BUFFER_NUM = 2;
constexpr int64_t SCALAR_TENSOR_NUM = 1;
constexpr int64_t TINY_QUEUE_TENSOR_NUM = 15;
constexpr int64_t SMALL_TENSOR_THRESHOLD = 1024;
constexpr int64_t MEDIUM_TENSOR_THRESHOLD = 2048;
constexpr int64_t FULL_CORE_TENSOR_THRESHOLD = 8192;
constexpr int64_t MID_CORE_TARGET = 1024;
constexpr int64_t MEDIUM_CORE_TARGET = 512;
constexpr int64_t LARGE_CORE_TARGET = 192;
constexpr int64_t HUGE_TENSOR_THRESHOLD = 49152;
constexpr int64_t HUGE_CORE_TARGET = 128;
constexpr int64_t UB_RESERVED_BYTES = 4096;
constexpr int64_t BLOCK_BYTES = 32;

static inline int64_t CeilDivInt(int64_t value, int64_t factor)
{
    return (value + factor - 1) / factor;
}

static inline int64_t AlignDownInt(int64_t value, int64_t align)
{
    return value / align * align;
}

static inline int64_t AlignUpInt(int64_t value, int64_t align)
{
    return CeilDivInt(value, align) * align;
}

static inline int64_t SelectCoreTarget(int64_t totalNum)
{
    if (totalNum >= HUGE_TENSOR_THRESHOLD) {
        return HUGE_CORE_TARGET;
    }
    if (totalNum >= MEDIUM_TENSOR_THRESHOLD && totalNum < FULL_CORE_TENSOR_THRESHOLD) {
        return MEDIUM_CORE_TARGET;
    }
    return totalNum >= FULL_CORE_TENSOR_THRESHOLD ? LARGE_CORE_TARGET : MID_CORE_TARGET;
}

static inline int64_t SelectBlockDim(int64_t totalNum, int64_t coreNum)
{
    if (totalNum <= SMALL_TENSOR_THRESHOLD) {
        return 1;
    }
    return std::min(coreNum, CeilDivInt(totalNum, SelectCoreTarget(totalNum)));
}

static inline int64_t AlignBlockFactor(int64_t totalNum, int64_t blockDim, int64_t alignElems)
{
    const int64_t blockFactorBase = CeilDivInt(totalNum, blockDim);
    return blockDim > 1 ? AlignUpInt(blockFactorBase, alignElems) : blockFactorBase;
}

static inline int64_t CalcUbFactor(uint64_t ubSize, int64_t blockFactor, int64_t alignElems)
{
    const int64_t usableUbSize = static_cast<int64_t>(ubSize) > UB_RESERVED_BYTES
        ? static_cast<int64_t>(ubSize) - UB_RESERVED_BYTES
        : static_cast<int64_t>(ubSize);
    int64_t maxUbFactor = usableUbSize / (TYPE_SIZE * (INPUT_BUFFER_NUM + OUTPUT_BUFFER_NUM));
    maxUbFactor = std::max<int64_t>(alignElems, AlignDownInt(maxUbFactor, alignElems));
    return std::min(maxUbFactor, AlignUpInt(blockFactor, alignElems));
}

static inline uint64_t SelectTilingKey(int64_t totalNum)
{
    if (totalNum == SCALAR_TENSOR_NUM) {
        return GET_TPL_TILING_KEY(HARDSWISH_TPL_SCH_MODE_1);
    }
    if (totalNum == TINY_QUEUE_TENSOR_NUM) {
        return GET_TPL_TILING_KEY(HARDSWISH_TPL_SCH_MODE_2);
    }
    return GET_TPL_TILING_KEY(HARDSWISH_TPL_SCH_MODE_0);
}

static ge::graphStatus CommitTilingResult(gert::TilingContext* context, size_t dataSize = sizeof(HardSwishTilingData))
{
    auto rawTilingData = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);
    rawTilingData->SetDataSize(dataSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static int64_t GetTotalNum(const gert::StorageShape* storageShape)
{
    const gert::Shape& shape = storageShape->GetStorageShape();
    if (shape.GetDimNum() == 0) {
        return 1;
    }

    int64_t totalNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        totalNum *= shape.GetDim(i);
    }
    return totalNum;
}

static ge::graphStatus HardSwishTilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    const int64_t totalNum = GetTotalNum(inputShape);
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum must be positive"), return ge::GRAPH_FAILED);

    const int64_t alignElems = BLOCK_BYTES / TYPE_SIZE;
    if (totalNum == SCALAR_TENSOR_NUM) {
        context->SetBlockDim(1);
        context->SetTilingKey(GET_TPL_TILING_KEY(HARDSWISH_TPL_SCH_MODE_1));
        return ge::GRAPH_SUCCESS;
    }
    if (totalNum == TINY_QUEUE_TENSOR_NUM) {
        context->SetBlockDim(1);
        context->SetTilingKey(GET_TPL_TILING_KEY(HARDSWISH_TPL_SCH_MODE_2));
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    int64_t blockDim = SelectBlockDim(totalNum, coreNum);
    const int64_t blockFactor = AlignBlockFactor(totalNum, blockDim, alignElems);
    blockDim = CeilDivInt(totalNum, blockFactor);
    const int64_t ubFactor = CalcUbFactor(ubSize, blockFactor, alignElems);

    HardSwishTilingData* tiling = context->GetTilingData<HardSwishTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(SelectTilingKey(totalNum));
    return CommitTilingResult(context);
}

static ge::graphStatus TilingParseForHardSwish([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct HardSwishCompileInfo {};

IMPL_OP_OPTILING(HardSwish).Tiling(HardSwishTilingFunc).TilingParse<HardSwishCompileInfo>(TilingParseForHardSwish);

} // namespace optiling
