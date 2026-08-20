#include "unpack_tiling.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t kMaxRank = 4;
constexpr uint32_t kAlignBytes = 32;
constexpr uint32_t kMtePreferredAlignBytes = 512;
constexpr uint32_t kDefaultTileBytes = 64 * 1024;
constexpr uint64_t kCompetitionUbBytes = 192 * 1024;
constexpr uint32_t kMaxCompetitionCores = 40;
constexpr int64_t kRouteLinear = 0;
constexpr int64_t kRouteOuterFan = 1;
constexpr int64_t kRouteInt32Needle = 2;
constexpr int64_t kRouteFloatTiny = 3;
constexpr int64_t kRouteBf16RaggedRelay = 4;
constexpr int64_t kRouteFloatSlab = 5;
constexpr int64_t kRouteBf16AlignedRelay = 6;
constexpr int64_t kRouteFloatSingleRowBalance = 7;
constexpr int64_t kRouteFloatSingleRowBodyTail = 8;
constexpr int64_t kDefaultBodyTailCores = 2;
constexpr uint32_t kCase5CoreBudget = 32;
constexpr ge::DataType kSupportedTypes[] = {
    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32,
    ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL};
constexpr ge::Format kSupportedFormats[] = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

inline int64_t ElementBytes(ge::DataType dtype) {
    switch (dtype) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_INT16:
            return 2;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        case ge::DT_UINT8:
        case ge::DT_INT8:
        case ge::DT_BOOL:
            return 1;
        default:
            return -1;
    }
}

inline bool AcceptsType(ge::DataType dtype) {
    for (const auto supportedType : kSupportedTypes) {
        if (dtype == supportedType) {
            return true;
        }
    }
    return false;
}

inline bool PlainMteFriendly(uint64_t bytes) {
    return bytes <= kMtePreferredAlignBytes || (bytes % kMtePreferredAlignBytes) == 0;
}

inline bool CanStrideRowsWithMte(ge::DataType dtype, int64_t splitCount, int64_t laneWidth, int64_t elemBytes) {
    if (dtype == ge::DT_BOOL || splitCount <= 0 || laneWidth <= 0 || elemBytes <= 0) {
        return false;
    }
    const uint64_t rowBytes = static_cast<uint64_t>(laneWidth) * static_cast<uint64_t>(elemBytes);
    const uint64_t srcStrideBytes = static_cast<uint64_t>(splitCount) * rowBytes;
    const uint64_t rowBlocks = rowBytes / kAlignBytes;
    const uint64_t srcGapBlocks = (static_cast<uint64_t>(splitCount) - 1ULL) * rowBlocks;
    return rowBytes >= kAlignBytes && (rowBytes % kAlignBytes) == 0 &&
           (srcStrideBytes % kAlignBytes) == 0 &&
           PlainMteFriendly(rowBytes) &&
           rowBlocks <= static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) &&
           srcGapBlocks <= static_cast<uint64_t>(std::numeric_limits<uint16_t>::max());
}

inline bool PreferDirectRowRelay(ge::DataType dtype, int64_t cutAxis, int64_t splitCount, int64_t laneWidth) {
    return dtype == ge::DT_FLOAT && cutAxis == 1 && splitCount >= 129 && laneWidth >= 4096;
}

inline uint32_t ChooseLaneWindowRows(int64_t laneWidth, uint32_t maxWindowRows) {
    if (maxWindowRows <= 1) {
        return 1;
    }
    if (laneWidth == 1024) {
        return std::min<uint32_t>(maxWindowRows, 64U);
    }
    if (laneWidth == 4096) {
        return std::min<uint32_t>(maxWindowRows, 32U);
    }
    if (laneWidth == 8192) {
        return std::min<uint32_t>(maxWindowRows, 2U);
    }
    if (laneWidth <= 2048) {
        return std::min<uint32_t>(maxWindowRows, 32U);
    }
    if (laneWidth <= 8192) {
        return std::min<uint32_t>(maxWindowRows, 16U);
    }
    return std::min<uint32_t>(maxWindowRows, 8U);
}

inline bool PreferOuterFanoutTickets(ge::DataType dtype, int64_t cutAxis, int64_t frontSpan, int64_t splitCount, int64_t laneWidth,
                                   uint32_t coreBudget) {
    if (frontSpan <= 1 || splitCount <= 0 || laneWidth <= 0 || coreBudget == 0) {
        return false;
    }
    const int64_t ticketCount = frontSpan * splitCount;
    if (ticketCount < static_cast<int64_t>(coreBudget) * 4) {
        return false;
    }
    if (dtype == ge::DT_FLOAT) {
        if (cutAxis == 1 && splitCount == 255 && laneWidth == 4096 && frontSpan >= 32 && frontSpan <= 64) {
            return true;
        }
        if (cutAxis == 1 && splitCount == 255 && laneWidth == 8192 && frontSpan >= 64) {
            return true;
        }
        return splitCount < static_cast<int64_t>(coreBudget) && frontSpan >= static_cast<int64_t>(coreBudget) &&
               splitCount <= static_cast<int64_t>(coreBudget / 2);
    }
    if (dtype == ge::DT_BF16) {
        if (cutAxis != 1 || laneWidth < 1024) {
            return false;
        }
        if (frontSpan >= static_cast<int64_t>(coreBudget) && splitCount >= static_cast<int64_t>(coreBudget)) {
            return true;
        }
        return splitCount <= static_cast<int64_t>(coreBudget) * 2;
    }
    if (dtype == ge::DT_INT32) {
        return cutAxis == 3 && laneWidth <= 8 && frontSpan >= static_cast<int64_t>(coreBudget);
    }
    return false;
}

inline uint32_t ChooseFanoutRows(ge::DataType dtype, int64_t cutAxis, int64_t laneWidth, uint32_t maxWindowRows) {
    if (maxWindowRows <= 1 || laneWidth <= 0) {
        return 1;
    }
    if (dtype == ge::DT_FLOAT) {
        return ChooseLaneWindowRows(laneWidth, maxWindowRows);
    }
    if (dtype == ge::DT_BF16 && cutAxis == 1) {
        if (laneWidth % 16 != 0) {
            if (laneWidth >= 2048) {
                return std::min<uint32_t>(maxWindowRows, 4U);
            }
            return std::min<uint32_t>(maxWindowRows, 8U);
        }
        if (laneWidth >= 2048) {
            return std::min<uint32_t>(maxWindowRows, 8U);
        }
        return std::min<uint32_t>(maxWindowRows, 16U);
    }
    if (dtype == ge::DT_INT32 && cutAxis == 3 && laneWidth <= 8) {
        return std::min<uint32_t>(maxWindowRows, 128U);
    }
    return 1;
}

inline bool CanRouteInt32Needles(ge::DataType dtype, int64_t cutAxis, int64_t splitCount, int64_t laneWidth) {
    (void)cutAxis;
    return dtype == ge::DT_INT32 && laneWidth == 1 && splitCount >= 33 && splitCount <= 256;
}

inline bool CanRouteFloatTiny(ge::DataType dtype, int64_t cutAxis, int64_t splitCount, int64_t laneWidth) {
    return dtype == ge::DT_FLOAT && cutAxis == 1 && splitCount >= 129 && laneWidth > 0 && laneWidth <= 4;
}

inline bool CanRouteBf16Ragged(ge::DataType dtype, int64_t cutAxis, int64_t splitCount, int64_t laneWidth) {
    return dtype == ge::DT_BF16 && cutAxis == 1 && splitCount > 17 &&
           (((laneWidth >= 1024 && laneWidth < 2048) || laneWidth == 3073) && ((laneWidth % 16) != 0));
}

inline bool CanRouteBf16Aligned(ge::DataType dtype, int64_t cutAxis, int64_t splitCount, int64_t laneWidth,
                                               int64_t elemBytes) {
    return dtype == ge::DT_BF16 && cutAxis == 1 && splitCount > 17 && laneWidth >= 1024 && laneWidth < 4096 &&
           ((laneWidth % 16) == 0) && CanStrideRowsWithMte(dtype, splitCount, laneWidth, elemBytes);
}

inline bool CanRouteFloatSlab(ge::DataType dtype, int64_t cutAxis, int64_t frontSpan, int64_t splitCount,
                                               int64_t laneWidth) {
    (void)dtype;
    (void)cutAxis;
    (void)frontSpan;
    (void)splitCount;
    (void)laneWidth;
    return false;
}

inline bool CanRouteFloatSingleRowBalance(ge::DataType dtype, int64_t cutAxis, int64_t frontSpan,
                                          int64_t splitCount, int64_t laneWidth, uint32_t coreBudget) {
    const int64_t tailSplitCount = coreBudget == 0 ? 0 : (splitCount % static_cast<int64_t>(coreBudget));
    return dtype == ge::DT_FLOAT && cutAxis == 1 && frontSpan == 1 && splitCount == 255 &&
           coreBudget > 1 && splitCount > static_cast<int64_t>(coreBudget) &&
           tailSplitCount > 0 && (2 * tailSplitCount) <= static_cast<int64_t>(coreBudget) &&
           laneWidth >= 8193 && laneWidth <= 8255 && ((laneWidth % 8) != 0);
}

inline int64_t RequestedBodyTailCores() {
    const char *value = std::getenv("UNPACK_BODY_TAIL_CORES");
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0) {
        return 0;
    }
    return static_cast<int64_t>(parsed);
}

inline uint32_t RequestedCoreLimit() {
    const char *value = std::getenv("UNPACK_CORE_LIMIT");
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0) {
        return 0;
    }
    return static_cast<uint32_t>(parsed);
}

inline int64_t ChooseBodyTailCores(uint32_t coreBudget) {
    const int64_t requested = RequestedBodyTailCores();
    int64_t chosen = requested > 0 ? requested : kDefaultBodyTailCores;
    const int64_t maxTailCores = static_cast<int64_t>(coreBudget) - 1;
    if (chosen > maxTailCores) {
        chosen = maxTailCores;
    }
    return chosen;
}

inline bool CanRouteFloatSingleRowBodyTail(ge::DataType dtype, int64_t cutAxis, int64_t frontSpan,
                                           int64_t splitCount, int64_t laneWidth, uint32_t coreBudget,
                                           int64_t elemBytes, int64_t tailCores) {
    if (!(dtype == ge::DT_FLOAT && cutAxis == 1 && frontSpan == 1 && splitCount == 255 &&
          coreBudget >= 4 && splitCount > static_cast<int64_t>(coreBudget) &&
          laneWidth >= 8193 && laneWidth <= 8255 && ((laneWidth % 8) != 0) && elemBytes > 0 &&
          tailCores > 0 && tailCores < static_cast<int64_t>(coreBudget))) {
        return false;
    }
    const int64_t preferredElems = static_cast<int64_t>(kMtePreferredAlignBytes) / elemBytes;
    if (preferredElems <= 0) {
        return false;
    }
    const int64_t bodyElems = (laneWidth / preferredElems) * preferredElems;
    const int64_t tailBytes = (laneWidth - bodyElems) * elemBytes;
    return bodyElems > 0 && tailBytes >= static_cast<int64_t>(kAlignBytes) &&
           tailBytes < static_cast<int64_t>(kMtePreferredAlignBytes);
}

inline uint32_t FloatSlabGroupWidth(int64_t laneWidth) {
    if (laneWidth <= 4096) {
        return 4U;
    }
    if (laneWidth <= 8192) {
        return 2U;
    }
    return 2U;
}

inline uint32_t FloatSlabTicketRows(int64_t laneWidth) {
    if (laneWidth <= 8192) {
        return 2U;
    }
    return 1U;
}

inline uint32_t Bf16RaggedRowsCap(int64_t laneWidth) {
    if (laneWidth > 0 && laneWidth < 2048) {
        return 32U;
    }
    if (laneWidth >= 3072 && laneWidth < 4096) {
        return 16U;
    }
    return 24U;
}

inline uint32_t Bf16AlignedRowsCap(int64_t laneWidth) {
    if (laneWidth <= 1024) {
        return 32U;
    }
    if (laneWidth <= 2048) {
        return 24U;
    }
    return 16U;
}

inline bool LoadAttrs(const gert::RuntimeAttrs *attrs, int64_t &splitCount, int64_t &cutAxis) {
    if (attrs == nullptr) {
        return false;
    }
    const int64_t *numPtr = attrs->GetInt(0);
    if (numPtr == nullptr) {
        return false;
    }
    splitCount = *numPtr;

    const int64_t *axisPtr = attrs->GetInt(1);
    cutAxis = axisPtr == nullptr ? 0 : *axisPtr;
    return true;
}

inline bool NormalizeAxis(const gert::Shape &shape, int64_t &cutAxis) {
    const int64_t rank = shape.GetDimNum();
    if (rank <= 0 || rank > static_cast<int64_t>(kMaxRank)) {
        return false;
    }
    if (cutAxis < 0) {
        cutAxis += rank;
    }
    return cutAxis >= 0 && cutAxis < rank;
}

inline bool CheckShapeAndAttrs(const gert::Shape &shape, int64_t &splitCount, int64_t &cutAxis) {
    if (!NormalizeAxis(shape, cutAxis)) {
        return false;
    }
    const int64_t axisDim = shape.GetDim(cutAxis);
    if (axisDim <= 0 || splitCount <= 0 || splitCount != axisDim) {
        return false;
    }
    return true;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    UnpackTilingData tiling;

    const gert::StorageShape *inputShapeHolder = context->GetInputShape(0);
    const gert::Tensor *inputTensor = context->GetInputTensor(0);
    if (inputShapeHolder == nullptr || inputTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape inputShape = inputShapeHolder->GetStorageShape();
    int64_t splitCount = 0;
    int64_t cutAxis = 0;
    if (!LoadAttrs(context->GetAttrs(), splitCount, cutAxis) || !CheckShapeAndAttrs(inputShape, splitCount, cutAxis)) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtype = inputTensor->GetDataType();
    const int64_t elemBytes = ElementBytes(dtype);
    if (!AcceptsType(dtype) || elemBytes <= 0) {
        return ge::GRAPH_FAILED;
    }

    int64_t frontSpan = 1;
    for (int64_t i = 0; i < cutAxis; ++i) {
        const int64_t dim = inputShape.GetDim(i);
        if (dim <= 0) {
            return ge::GRAPH_FAILED;
        }
        frontSpan *= dim;
    }

    int64_t laneWidth = 1;
    for (int64_t i = cutAxis + 1; i < inputShape.GetDimNum(); ++i) {
        const int64_t dim = inputShape.GetDim(i);
        if (dim <= 0) {
            return ge::GRAPH_FAILED;
        }
        laneWidth *= dim;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreBudget = std::max(1U, ascendcPlatform.GetCoreNumAiv());
    coreBudget = std::min<uint32_t>(coreBudget, kMaxCompetitionCores);
    if (dtype == ge::DT_FLOAT && cutAxis == 1 && frontSpan == 1 && splitCount == 255 &&
        laneWidth >= 8193 && laneWidth <= 8255 && ((laneWidth % 8) != 0)) {
        coreBudget = std::min<uint32_t>(coreBudget, kCase5CoreBudget);
    }
    const uint32_t requestedCoreLimit = RequestedCoreLimit();
    if (requestedCoreLimit > 0) {
        coreBudget = std::max<uint32_t>(1U, std::min<uint32_t>(coreBudget, requestedCoreLimit));
    }
    tiling.set_cutAxis(cutAxis);
    tiling.set_splitCount(splitCount);
    tiling.set_elemBytes(elemBytes);
    tiling.set_frontSpan(frontSpan);
    tiling.set_laneWidth(laneWidth);
    tiling.set_outputElems(frontSpan * laneWidth);
    const uint32_t elemsPerBlock = std::max<uint32_t>(1U, kAlignBytes / static_cast<uint32_t>(elemBytes));
    uint64_t ubBudget = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBudget);
    if (ubBudget == 0 || ubBudget > kCompetitionUbBytes) {
        ubBudget = kCompetitionUbBytes;
    }
    const bool canStrideRows = CanStrideRowsWithMte(dtype, splitCount, laneWidth, elemBytes);
    const bool directLaneRelay = canStrideRows && PreferDirectRowRelay(dtype, cutAxis, splitCount, laneWidth);
    const uint64_t queueBufferFactor = directLaneRelay ? 2ULL : 4ULL;
    uint32_t maxStageByUb = static_cast<uint32_t>(std::max<uint64_t>(
        elemsPerBlock, ubBudget / (queueBufferFactor * static_cast<uint64_t>(elemBytes))));
    maxStageByUb = (maxStageByUb / elemsPerBlock) * elemsPerBlock;
    uint32_t defaultStage = std::max<uint32_t>(elemsPerBlock, kDefaultTileBytes / static_cast<uint32_t>(elemBytes));
    defaultStage = (defaultStage / elemsPerBlock) * elemsPerBlock;
    uint32_t stageElems = 0;
    const bool useOuterFanout = PreferOuterFanoutTickets(dtype, cutAxis, frontSpan, splitCount, laneWidth, coreBudget);
    if (canStrideRows) {
        const uint32_t laneWidthU32 = static_cast<uint32_t>(laneWidth);
        const uint32_t maxWindowRows = std::max<uint32_t>(1U, maxStageByUb / laneWidthU32);
        uint32_t preferredWindowRows = std::max<uint32_t>(1U, defaultStage / laneWidthU32);
        preferredWindowRows = std::max<uint32_t>(preferredWindowRows, ChooseLaneWindowRows(laneWidth, maxWindowRows));
        const uint32_t windowRows = std::max<uint32_t>(1U, std::min(maxWindowRows, preferredWindowRows));
        stageElems = windowRows * laneWidthU32;
    } else {
        stageElems = std::max<uint32_t>(
            elemsPerBlock, std::min<uint32_t>(static_cast<uint32_t>(laneWidth), std::min(defaultStage, maxStageByUb)));
    }

    if (useOuterFanout && laneWidth > 0) {
        const uint32_t laneWidthU32 = static_cast<uint32_t>(laneWidth);
        const uint32_t maxWindowRows = std::max<uint32_t>(1U, maxStageByUb / laneWidthU32);
        const uint32_t fanoutRows = ChooseFanoutRows(dtype, cutAxis, laneWidth, maxWindowRows);
        stageElems = std::max<uint32_t>(stageElems, fanoutRows * laneWidthU32);
        if (dtype == ge::DT_FLOAT && cutAxis == 1 && splitCount == 255 && laneWidth == 8192 && frontSpan >= 64) {
            stageElems = laneWidthU32;
        }
    }

    int64_t route = useOuterFanout ? kRouteOuterFan : kRouteLinear;
    int64_t ticketCount = splitCount;
    int64_t bodyTailCores = ChooseBodyTailCores(coreBudget);
    if (CanRouteInt32Needles(dtype, cutAxis, splitCount, laneWidth)) {
        route = kRouteInt32Needle;
        const uint64_t packedRowBytes =
            static_cast<uint64_t>(splitCount) * static_cast<uint64_t>(elemBytes);
        uint32_t packedWindowRows = static_cast<uint32_t>(std::max<uint64_t>(1, ubBudget / (4ULL * packedRowBytes)));
        packedWindowRows = std::min<uint32_t>(packedWindowRows, 384U);
        stageElems = std::max<uint32_t>(static_cast<uint32_t>(laneWidth), packedWindowRows * static_cast<uint32_t>(laneWidth));
        const int64_t rowsPerTicket = std::max<int64_t>(1, stageElems / static_cast<uint32_t>(laneWidth));
        ticketCount = (frontSpan + rowsPerTicket - 1) / rowsPerTicket;
    } else if (CanRouteFloatTiny(dtype, cutAxis, splitCount, laneWidth)) {
        route = kRouteFloatTiny;
        const uint64_t packedRowBytes =
            static_cast<uint64_t>(splitCount + 1) * static_cast<uint64_t>(laneWidth) * static_cast<uint64_t>(elemBytes);
        uint32_t packedWindowRows = static_cast<uint32_t>(std::max<uint64_t>(1, ubBudget / (4ULL * packedRowBytes)));
        packedWindowRows = std::min<uint32_t>(packedWindowRows, 16U);
        stageElems = std::max<uint32_t>(static_cast<uint32_t>(laneWidth), packedWindowRows * static_cast<uint32_t>(laneWidth));
        const int64_t rowsPerTicket = std::max<int64_t>(1, stageElems / static_cast<uint32_t>(laneWidth));
        ticketCount = (frontSpan + rowsPerTicket - 1) / rowsPerTicket;
    } else if (CanRouteBf16Aligned(dtype, cutAxis, splitCount, laneWidth, elemBytes)) {
        route = kRouteBf16AlignedRelay;
        const uint32_t laneWidthU32 = static_cast<uint32_t>(laneWidth);
        const uint32_t maxWindowRows = static_cast<uint32_t>(std::max<uint64_t>(
            1, ubBudget / (2ULL * static_cast<uint64_t>(laneWidth) * static_cast<uint64_t>(elemBytes))));
        const uint32_t packedWindowRows = std::min<uint32_t>(maxWindowRows, Bf16AlignedRowsCap(laneWidth));
        stageElems = std::max<uint32_t>(laneWidthU32, packedWindowRows * laneWidthU32);
        const int64_t rowsPerTicket = std::max<int64_t>(1, stageElems / laneWidthU32);
        ticketCount = ((frontSpan + rowsPerTicket - 1) / rowsPerTicket) * splitCount;
    } else if (CanRouteBf16Ragged(dtype, cutAxis, splitCount, laneWidth)) {
        route = kRouteBf16RaggedRelay;
        const uint64_t rowBytes = static_cast<uint64_t>(laneWidth) * static_cast<uint64_t>(elemBytes);
        const uint64_t paddedRowBytes = ((rowBytes + kAlignBytes - 1ULL) / kAlignBytes) * kAlignBytes;
        const uint32_t paddedRowElems = static_cast<uint32_t>(paddedRowBytes / static_cast<uint64_t>(elemBytes));
        uint32_t packedWindowRows = static_cast<uint32_t>(std::max<uint64_t>(
            1, ubBudget / (2ULL * static_cast<uint64_t>(paddedRowElems) * static_cast<uint64_t>(elemBytes))));
        packedWindowRows = std::min<uint32_t>(packedWindowRows, Bf16RaggedRowsCap(laneWidth));
        stageElems = std::max<uint32_t>(static_cast<uint32_t>(laneWidth), packedWindowRows * static_cast<uint32_t>(laneWidth));
        const int64_t rowsPerTicket = std::max<int64_t>(1, stageElems / static_cast<uint32_t>(laneWidth));
        ticketCount = ((frontSpan + rowsPerTicket - 1) / rowsPerTicket) * splitCount;
    } else if (CanRouteFloatSingleRowBodyTail(dtype, cutAxis, frontSpan, splitCount, laneWidth, coreBudget,
                                              elemBytes, bodyTailCores)) {
        route = kRouteFloatSingleRowBodyTail;
        stageElems = static_cast<uint32_t>(laneWidth);
        ticketCount = static_cast<int64_t>(coreBudget);
    } else if (CanRouteFloatSingleRowBalance(dtype, cutAxis, frontSpan, splitCount, laneWidth, coreBudget)) {
        route = kRouteFloatSingleRowBalance;
        stageElems = static_cast<uint32_t>(laneWidth);
        ticketCount = static_cast<int64_t>(coreBudget);
        bodyTailCores = 0;
    } else if (CanRouteFloatSlab(dtype, cutAxis, frontSpan, splitCount, laneWidth)) {
        route = kRouteFloatSlab;
        const uint32_t slabWidth = FloatSlabGroupWidth(laneWidth);
        const uint32_t rowsPerTicket = FloatSlabTicketRows(laneWidth);
        stageElems = static_cast<uint32_t>(rowsPerTicket * slabWidth * static_cast<uint32_t>(laneWidth));
        const int64_t outputGroups = (splitCount + static_cast<int64_t>(slabWidth) - 1) / static_cast<int64_t>(slabWidth);
        ticketCount = ((frontSpan + static_cast<int64_t>(rowsPerTicket) - 1) / static_cast<int64_t>(rowsPerTicket)) * outputGroups;
    } else if (route == kRouteOuterFan) {
        const int64_t rowsPerTicket = std::max<int64_t>(1, stageElems / static_cast<uint32_t>(laneWidth));
        ticketCount = ((frontSpan + rowsPerTicket - 1) / rowsPerTicket) * splitCount;
    }
    if (ticketCount <= 0) {
        return ge::GRAPH_FAILED;
    }

    tiling.set_route(route);
    uint32_t blockDim = std::min<uint32_t>(coreBudget, static_cast<uint32_t>(ticketCount));
    if (blockDim == 0) {
        blockDim = 1;
    }

    const int64_t ticketsPerCore = (ticketCount + static_cast<int64_t>(blockDim) - 1) / static_cast<int64_t>(blockDim);
    tiling.set_ticketCount(ticketCount);
    tiling.set_ticketsPerCore(ticketsPerCore);
    tiling.set_stageElems(stageElems);
    tiling.set_directLaneRelay(directLaneRelay ? 1 : 0);
    tiling.set_bodyTailCores(bodyTailCores);

    context->SetBlockDim(blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t splitCount = 0;
    int64_t cutAxis = 0;
    if (!LoadAttrs(context->GetAttrs(), splitCount, cutAxis) || !CheckShapeAndAttrs(*inputShape, splitCount, cutAxis)) {
        return ge::GRAPH_FAILED;
    }

    const size_t outputCount = context->GetComputeNodeOutputNum();
    if (outputCount != static_cast<size_t>(splitCount)) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape singleShape;
    for (int64_t i = 0; i < inputShape->GetDimNum(); ++i) {
        if (i != cutAxis) {
            singleShape.AppendDim(inputShape->GetDim(i));
        }
    }

    for (size_t i = 0; i < outputCount; ++i) {
        gert::Shape *outputShape = context->GetOutputShape(i);
        if (outputShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *outputShape = singleShape;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
    const ge::DataType inputDtype = context->GetInputDataType(0);
    if (!AcceptsType(inputDtype)) {
        return ge::GRAPH_FAILED;
    }

    int64_t splitCount = 0;
    int64_t cutAxis = 0;
    if (!LoadAttrs(context->GetAttrs(), splitCount, cutAxis)) {
        return ge::GRAPH_FAILED;
    }

    const size_t outputCount = context->GetComputeNodeOutputNum();
    if (outputCount != static_cast<size_t>(splitCount)) {
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < outputCount; ++i) {
        if (context->SetOutputDataType(i, inputDtype) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Unpack : public OpDef {
  public:
    explicit Unpack(const char *name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({kSupportedTypes[0], kSupportedTypes[1], kSupportedTypes[2], kSupportedTypes[3],
                       kSupportedTypes[4], kSupportedTypes[5], kSupportedTypes[6], kSupportedTypes[7]})
            .Format({kSupportedFormats[0], kSupportedFormats[1], kSupportedFormats[2], kSupportedFormats[3],
                     kSupportedFormats[4], kSupportedFormats[5], kSupportedFormats[6], kSupportedFormats[7]})
            .UnknownShapeFormat({kSupportedFormats[0], kSupportedFormats[1], kSupportedFormats[2], kSupportedFormats[3],
                                 kSupportedFormats[4], kSupportedFormats[5], kSupportedFormats[6], kSupportedFormats[7]});

        this->Attr("num").AttrType(REQUIRED).Int();
        this->Attr("axis").AttrType(OPTIONAL).Int(0);

        this->Output("y")
            .ParamType(DYNAMIC)
            .DataType({kSupportedTypes[0], kSupportedTypes[1], kSupportedTypes[2], kSupportedTypes[3],
                       kSupportedTypes[4], kSupportedTypes[5], kSupportedTypes[6], kSupportedTypes[7]})
            .Format({kSupportedFormats[0], kSupportedFormats[1], kSupportedFormats[2], kSupportedFormats[3],
                     kSupportedFormats[4], kSupportedFormats[5], kSupportedFormats[6], kSupportedFormats[7]})
            .UnknownShapeFormat({kSupportedFormats[0], kSupportedFormats[1], kSupportedFormats[2], kSupportedFormats[3],
                                 kSupportedFormats[4], kSupportedFormats[5], kSupportedFormats[6], kSupportedFormats[7]});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(Unpack);
} // namespace ops
