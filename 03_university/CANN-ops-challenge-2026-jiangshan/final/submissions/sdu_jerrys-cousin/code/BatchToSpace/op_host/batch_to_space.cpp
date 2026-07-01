#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>

#include "../op_kernel/batch_to_space_tiling.h"
#include "../op_kernel/tiling_key_batch_to_space.h"

namespace optiling {
namespace {
constexpr uint32_t CORE_CAP_NONE = 0;
constexpr uint32_t POINT_COUNT = 10;
constexpr uint32_t P6_TILE_D = 4096;
// 数据点 7 固定策略：H=1/W=1 走连续 copy，其余 P7 保留大 D 兜底。
constexpr uint32_t P7_TILE_D = 4096;
constexpr uint32_t P7_COPY_TILE_ELEMS = 8192;

enum class PointId : uint32_t {
    GENERIC = 0,
    P1 = 1,
    P2 = 2,
    P3 = 3,
    P4 = 4,
    P5 = 5,
    P6 = 6,
    P7 = 7,
    P8 = 8,
    P9 = 9,
    P10 = 10,
};

enum class ShapeClass : uint32_t {
    GENERIC = 0,
    ALIGNED_ROW,
    UNALIGNED_ROW,
    LARGE_D,
    BLOCK4_WIDE,
    TALL_NARROW,
};

struct ShapeDesc {
    uint32_t B;
    uint32_t H;
    uint32_t W;
    uint32_t D;
    uint32_t block;
    uint32_t ct;
    uint32_t cb;
    uint32_t cl;
    uint32_t cr;
    ge::DataType dtype;
    bool d32BAligned;
};

struct ModePlan {
    uint32_t mode;
    uint32_t tileW;
    uint32_t dTileLen;
    uint32_t gatherDepth;
    uint32_t idxLen;
};

// 10 个数据点的默认核心数调参入口。0 表示不额外限核。
constexpr uint32_t POINT_CORE_CAP[POINT_COUNT + 1] = {
    CORE_CAP_NONE,  // 通用
    20,             // 数据点 1
    8,              // 数据点 2: 小非对齐 Gather，少核减少重复 idx 构建
    20,             // 数据点 3: 与 P1 同类 B2 direct，小任务固定 20 核
    10,             // 数据点 4: P4 exact 单元很少，减少核启动/尾块开销
    CORE_CAP_NONE,  // 数据点 5
    8,              // 数据点 6: 旧 D-direct 为 8 个任务，pair-pack 实测无收益
    CORE_CAP_NONE,  // 数据点 7: copy 模式不限核
    CORE_CAP_NONE,  // 数据点 8
    16,             // 数据点 9: block4 crop 中等任务量，先做保守限核 A/B
    20,             // 数据点 10: tall narrow 特化路径也避免过多核抢 MTE
};

constexpr bool POINT_USE_SMALL_TASK_CAP[POINT_COUNT + 1] = {
    false,  // 通用
    false,  // 数据点 1
    false,  // 数据点 2
    false,  // 数据点 3
    false,  // 数据点 4
    false,  // 数据点 5
    false,  // 数据点 6
    false,  // 数据点 7
    false,  // 数据点 8
    false,  // 数据点 9
    false,  // 数据点 10
};

static uint32_t AlignUpTo32B(uint32_t elems, uint32_t bytesPerElem)
{
    const uint32_t blockElems = 32 / bytesPerElem;
    return (elems + blockElems - 1) / blockElems * blockElems;
}

static const std::array<uint32_t, BTS_MAX_GATHER_IDX_ELEMS> &GetPoint5GatherIndex()
{
    static const std::array<uint32_t, BTS_MAX_GATHER_IDX_ELEMS> idx = []() {
        std::array<uint32_t, BTS_MAX_GATHER_IDX_ELEMS> value{};
        constexpr uint32_t block = 2;
        constexpr uint32_t W = 128;
        constexpr uint32_t D = 65;
        constexpr uint32_t OW = 254;
        constexpr uint32_t cl = 1;
        constexpr uint32_t bytesPerElem = 2;
        constexpr uint32_t regionStride = 8320;

        uint32_t k = 0;
        for (uint32_t ow = 0; ow < OW; ++ow) {
            const uint32_t widx = ow + cl;
            const uint32_t j = widx % block;
            const uint32_t w = widx / block;
            const uint32_t base = j * regionStride + w * D;
            for (uint32_t d = 0; d < D; ++d) {
                value[k++] = (base + d) * bytesPerElem;
            }
        }
        return value;
    }();
    return idx;
}

static bool NoCrop(const ShapeDesc &s)
{
    return s.ct == 0 && s.cb == 0 && s.cl == 0 && s.cr == 0;
}

static PointId DetectPoint(const ShapeDesc &s)
{
    const bool noCrop = NoCrop(s);
    if (s.B == 8 && s.block == 2 && s.H >= 16 && s.H <= 32 &&
        s.W == 28 && s.D == 128 && s.d32BAligned && noCrop) {
        return PointId::P1;
    }
    if (s.B == 4 && s.block == 2 && s.H >= 9 && s.H <= 32 &&
        s.W == 15 && s.D == 5 && !s.d32BAligned && (s.ct + s.cl) != 0) {
        return PointId::P2;
    }
    if (s.B == 16 && s.block == 2 && s.H >= 9 && s.H <= 32 &&
        s.W == 14 && s.D == 64 && s.d32BAligned && noCrop) {
        return PointId::P3;
    }
    if (s.B == 20 && s.block == 2 && s.H == 4 && s.W == 6 &&
        s.D == 32 && s.d32BAligned && noCrop) {
        return PointId::P4;
    }
    if (s.B == 4 && s.block == 2 && s.H == 128 && s.W == 128 &&
        s.D == 65 && s.dtype == ge::DT_FLOAT16 &&
        s.ct == 1 && s.cb == 1 && s.cl == 1 && s.cr == 1) {
        return PointId::P5;
    }
    if (s.B == 4 && s.block == 2 && s.H <= 32 && s.W == 2 &&
        s.D == 4096 && s.d32BAligned && noCrop) {
        return PointId::P6;
    }
    if (s.B == 4 && s.block == 2 && s.H <= 32 && s.W == 1 && s.D == 16384 && noCrop) {
        return PointId::P7;
    }
    if (s.B == 4 && s.block == 2 && s.H <= 32 && s.W >= 257 && s.W <= 512 &&
        s.D == 256 && s.d32BAligned) {
        return PointId::P8;
    }
    if (s.B == 16 && s.block == 4 && s.H <= 16 && s.W == 512 &&
        s.D == 64 && s.dtype == ge::DT_FLOAT16 && s.d32BAligned) {
        return PointId::P9;
    }
    if (s.B == 16 && s.block == 2 && s.H == 1024 && s.W == 6 &&
        s.D == 32 && s.dtype == ge::DT_FLOAT16) {
        return PointId::P10;
    }
    return PointId::GENERIC;
}

static ShapeClass ClassifyShape(PointId point, const ShapeDesc &s)
{
    if (point == PointId::P7) {
        return ShapeClass::LARGE_D;
    }
    if (point == PointId::P9) {
        return ShapeClass::BLOCK4_WIDE;
    }
    if (point == PointId::P10) {
        return ShapeClass::TALL_NARROW;
    }
    return s.d32BAligned ? ShapeClass::ALIGNED_ROW : ShapeClass::UNALIGNED_ROW;
}

static bool IsPoint2Candidate(const ShapeDesc &s)
{
    return s.B == 4 && s.block == 2 && s.H >= 9 && s.H <= 32 &&
           s.W >= 3 && s.W <= 32 && s.D >= 2 && s.D <= 64 &&
           !s.d32BAligned && (s.ct + s.cl) != 0;
}

static bool IsPoint4FastShape(const ShapeDesc &s)
{
    return s.B == 20 && s.block == 2 && s.H == 4 && s.W == 6 &&
           s.D == 32 && s.d32BAligned && NoCrop(s);
}

static bool IsPoint5FastShape(const ShapeDesc &s)
{
    return s.B == 4 && s.block == 2 && s.H == 128 && s.W == 128 &&
           s.D == 65 && s.dtype == ge::DT_FLOAT16 &&
           s.ct == 1 && s.cb == 1 && s.cl == 1 && s.cr == 1;
}

static uint32_t SelectBaseMode(PointId point)
{
    switch (point) {
        case PointId::P1:
            return BTS_MODE_P1_B2_DIRECT;
        case PointId::P3:
            return BTS_MODE_P3_B2_DIRECT;
        case PointId::P4:
            return BTS_MODE_B4_W_PACK_TILE;
        case PointId::P6:
            return BTS_MODE_P6_B2_DIRECT;
        case PointId::P7:
            return BTS_MODE_P7_D_DIRECT;
        case PointId::P10:
            return BTS_MODE_FLAT_TILE;
        default:
            return BTS_MODE_ROW_TILE;
    }
}

static bool IsP7FlatCopyShape(const ShapeDesc &s)
{
    return s.B == 4 && s.block == 2 && s.H == 1 && s.W == 1 &&
           s.D == 16384 && s.d32BAligned && NoCrop(s);
}

static uint32_t SelectSmallTaskCoreCap(uint32_t totalUnits, uint32_t coreNum)
{
    uint32_t maxCore = coreNum;
    if (totalUnits <= 64) {
        maxCore = 8;
    } else if (totalUnits <= 128) {
        maxCore = 16;
    } else if (totalUnits <= 256) {
        maxCore = 32;
    }
    return maxCore;
}

static uint32_t LimitCore(uint32_t totalUnits, uint32_t coreNum, uint32_t cap)
{
    uint32_t usedCoreNum = std::min(totalUnits, coreNum);
    if (cap != CORE_CAP_NONE) {
        usedCoreNum = std::min(usedCoreNum, cap);
    }
    return usedCoreNum;
}

static bool IsP10SpecialMode(uint32_t mode)
{
    return mode == BTS_MODE_P10_OH_GATHER_TILE || mode == BTS_MODE_P10_UB_PACK_TILE;
}

static uint32_t SelectCoreNum(PointId point, ShapeClass shapeClass, uint32_t mode,
                              uint32_t totalUnits, uint32_t coreNum)
{
    const uint32_t pointIdx = static_cast<uint32_t>(point);
    uint32_t cap = POINT_CORE_CAP[pointIdx];

    uint32_t usedCoreNum = LimitCore(totalUnits, coreNum, cap);
    const bool usePointSmallCap = POINT_USE_SMALL_TASK_CAP[pointIdx];
    const bool useGenericAlignedCap =
        point == PointId::GENERIC && shapeClass == ShapeClass::ALIGNED_ROW &&
        mode != BTS_MODE_FLAT_TILE && mode != BTS_MODE_P10_OH_GATHER_TILE &&
        mode != BTS_MODE_P10_UB_PACK_TILE && mode != BTS_MODE_P2_GATHER_FAST &&
        mode != BTS_MODE_B4_W_PACK_TILE;
    if (usePointSmallCap || useGenericAlignedCap) {
        usedCoreNum = std::min(usedCoreNum, SelectSmallTaskCoreCap(totalUnits, coreNum));
    }
    return usedCoreNum;
}

static void SelectLargeDTile(const ShapeDesc &s, uint32_t bytesPerElem, uint32_t coreNum,
                             uint64_t perBufElems, uint32_t OH, uint32_t OW,
                             uint32_t N, ModePlan &plan)
{
    const uint32_t blockElems = 32 / bytesPerElem;
    uint32_t maxChunk = static_cast<uint32_t>(perBufElems / blockElems) * blockElems;
    plan.dTileLen = std::min(s.D, maxChunk);
    plan.tileW = 1;

    const uint64_t outPixels = static_cast<uint64_t>(N) * OH * OW;
    const uint32_t numDTilesNow = (s.D + plan.dTileLen - 1) / plan.dTileLen;
    const uint64_t tasksNow = outPixels * numDTilesNow;
    const uint64_t want = static_cast<uint64_t>(coreNum) * 4;
    if (tasksNow >= want) {
        return;
    }

    uint32_t wantD = (outPixels > 0 && outPixels < want) ?
        static_cast<uint32_t>((want + outPixels - 1) / outPixels) : 1u;
    uint32_t minChunkElems = 4096u / bytesPerElem;
    if (minChunkElems < blockElems) {
        minChunkElems = blockElems;
    }
    uint32_t maxSplit = s.D / minChunkElems;
    if (maxSplit < 1) {
        maxSplit = 1;
    }
    if (wantD > maxSplit) {
        wantD = maxSplit;
    }
    if (wantD > numDTilesNow) {
        uint32_t chunk = (s.D + wantD - 1) / wantD;
        chunk = (chunk + blockElems - 1) / blockElems * blockElems;
        if (chunk < plan.dTileLen) {
            plan.dTileLen = chunk;
        }
    }
}

static void SelectP7Tile(const ShapeDesc &s, ModePlan &plan)
{
    if (IsP7FlatCopyShape(s)) {
        plan = ModePlan{BTS_MODE_P7_FLAT_COPY, P7_COPY_TILE_ELEMS, s.D, 0, 0};
    } else {
        plan.tileW = 1;
        plan.dTileLen = std::min(s.D, P7_TILE_D);
    }
}

static void SelectBaseTile(const ShapeDesc &s, uint64_t perBufElems, uint32_t colAligned, ModePlan &plan)
{
    const uint64_t slotElems = s.d32BAligned ? static_cast<uint64_t>(s.D) : static_cast<uint64_t>(colAligned);
    uint64_t tileW64 = perBufElems / slotElems;
    if (plan.mode == BTS_MODE_FLAT_TILE) {
        tileW64 = 384;
    } else if (tileW64 > s.W) {
        tileW64 = s.W;
    }
    if (tileW64 > 32768ULL) {
        tileW64 = 32768ULL;
    }
    plan.tileW = static_cast<uint32_t>(tileW64);
}

static void TrySelectP10UbPack(const ShapeDesc &s, uint32_t bytesPerElem,
                               uint64_t ubBytes, uint32_t OW, ModePlan &plan)
{
    if (s.block != 2 || !NoCrop(s)) {
        return;
    }

    constexpr uint32_t P10_PACK_TILE_CANDIDATES[] = {62, 60, 56, 48, 40, 32};
    uint32_t selectedTileOH = 0;
    uint32_t selectedDepth = 0;
    for (uint32_t candTileOH : P10_PACK_TILE_CANDIDATES) {
        const uint64_t p10HRows = (static_cast<uint64_t>(candTileOH) + s.block - 1) / s.block;
        const uint64_t srcElems = static_cast<uint64_t>(s.block) * s.block * p10HRows * s.W * s.D;
        const uint64_t outElems = static_cast<uint64_t>(candTileOH) * OW * s.D;
        const uint64_t needBytes1 = (srcElems + outElems) * bytesPerElem;
        const uint64_t needBytes2 = 2ull * needBytes1;
        if (needBytes2 <= ubBytes) {
            selectedTileOH = candTileOH;
            selectedDepth = 2u;
            break;
        }
        if (selectedTileOH == 0 && needBytes1 <= ubBytes) {
            selectedTileOH = candTileOH;
            selectedDepth = 1u;
        }
    }

    if (selectedTileOH != 0) {
        plan = ModePlan{BTS_MODE_P10_UB_PACK_TILE, selectedTileOH, s.D, selectedDepth, 0};
    }
}

static void TrySelectP2Gather(const ShapeDesc &s, uint32_t bytesPerElem, uint64_t ubBytes,
                              uint32_t OW, ModePlan &plan)
{
    const uint32_t regionStride = AlignUpTo32B(s.W * s.D, bytesPerElem);
    const uint64_t idxElems64 = static_cast<uint64_t>(OW) * s.D;
    const uint64_t srcBytes = static_cast<uint64_t>(s.block) * regionStride * bytesPerElem;
    const uint64_t outBytes = idxElems64 * bytesPerElem;
    const uint64_t idxBytes = idxElems64 * sizeof(uint32_t);
    const uint64_t needBytes2 = 2ull * (srcBytes + outBytes) + idxBytes;
    plan = ModePlan{BTS_MODE_P2_GATHER_FAST, 15, s.D, (needBytes2 <= ubBytes) ? 2u : 1u, 0};
}

static void TrySelectP10OhGather(const ShapeDesc &s, uint32_t bytesPerElem, uint64_t ubBytes,
                                 uint32_t OW, ModePlan &plan)
{
    if (plan.mode == BTS_MODE_P10_UB_PACK_TILE) {
        return;
    }

    constexpr uint32_t P10_TILE_CANDIDATES[] = {42, 40, 32, 20, 16};
    for (uint32_t candTileOH : P10_TILE_CANDIDATES) {
        const uint64_t p10HRows = (static_cast<uint64_t>(candTileOH) + s.block - 1) / s.block;
        const uint64_t srcElems = static_cast<uint64_t>(s.block) * s.block * p10HRows * s.W * s.D;
        const uint64_t outElems = static_cast<uint64_t>(candTileOH) * OW * s.D;
        const uint64_t idxElems64 = outElems;
        const uint64_t srcBytes = srcElems * bytesPerElem;
        const uint64_t outBytes = outElems * bytesPerElem;
        const uint64_t idxBytes = idxElems64 * sizeof(uint32_t);
        const uint64_t needBytes1 = srcBytes + outBytes + idxBytes;
        const uint64_t needBytes2 = 2ull * (srcBytes + outBytes) + idxBytes;
        if (idxElems64 <= BTS_MAX_GATHER_IDX_ELEMS && needBytes1 <= ubBytes) {
            plan = ModePlan{BTS_MODE_P10_OH_GATHER_TILE, candTileOH, s.D,
                            (needBytes2 <= ubBytes) ? 2u : 1u,
                            static_cast<uint32_t>(idxElems64)};
            break;
        }
    }
}

static void TrySelectP9B4WPack(const ShapeDesc &s, uint32_t bytesPerElem, uint64_t ubBytes,
                               uint32_t OW, ModePlan &plan)
{
    if (IsP10SpecialMode(plan.mode)) {
        return;
    }

    constexpr uint32_t P9_OW_TILE_CANDIDATES[] = {1024, 768, 512, 384, 256, 128, 64};
    uint32_t selectedTileOW = 0;
    uint32_t selectedDepth = 0;
    uint32_t lastCandOW = 0;
    for (uint32_t candTileOW : P9_OW_TILE_CANDIDATES) {
        if (candTileOW > OW) {
            candTileOW = OW;
        }
        if (candTileOW == lastCandOW) {
            continue;
        }
        lastCandOW = candTileOW;
        const uint64_t srcCols = (static_cast<uint64_t>(candTileOW) + s.block - 1) / s.block;
        const uint64_t srcElems = static_cast<uint64_t>(s.block) * srcCols * s.D;
        const uint64_t outElems = static_cast<uint64_t>(candTileOW) * s.D;
        const uint64_t needBytes1 = (srcElems + outElems) * bytesPerElem;
        const uint64_t needBytes2 = 2ull * needBytes1;
        if (needBytes2 <= ubBytes) {
            selectedTileOW = candTileOW;
            selectedDepth = 2u;
            break;
        }
        if (selectedTileOW == 0 && needBytes1 <= ubBytes) {
            selectedTileOW = candTileOW;
            selectedDepth = 1u;
        }
    }

    if (selectedTileOW == 0) {
        // P9 固定形状下 64 列 OW tile 可放入 UB，避免 UB 查询异常时回落通用 ROW。
        selectedTileOW = 64;
        selectedDepth = 1u;
    }
    plan = ModePlan{BTS_MODE_B4_W_PACK_TILE, selectedTileOW, s.D, selectedDepth, 0};
}

static void TrySelectGatherRow(const ShapeDesc &s, PointId point, uint32_t bytesPerElem,
                               uint64_t ubBytes, uint32_t OW, ModePlan &plan)
{
    if (plan.mode == BTS_MODE_GATHER_ROW || plan.mode == BTS_MODE_P2_GATHER_FAST ||
        IsP10SpecialMode(plan.mode)) {
        return;
    }

    const bool gatherCandidate = IsPoint2Candidate(s) || point == PointId::P5;
    if (!gatherCandidate) {
        return;
    }

    const uint32_t regionStride = AlignUpTo32B(s.W * s.D, bytesPerElem);
    const uint64_t idxElems64 = static_cast<uint64_t>(OW) * s.D;
    const uint64_t srcBytes = static_cast<uint64_t>(s.block) * regionStride * bytesPerElem;
    const uint64_t outBytes = idxElems64 * bytesPerElem;
    const uint64_t idxBytes = idxElems64 * sizeof(uint32_t);
    const uint64_t needBytes1 = srcBytes + outBytes + idxBytes;
    const uint64_t needBytes2 = 2ull * (srcBytes + outBytes) + idxBytes;
    if (needBytes1 <= ubBytes) {
        plan = ModePlan{BTS_MODE_GATHER_ROW, plan.tileW, plan.dTileLen,
                        (needBytes2 <= ubBytes) ? 2u : 1u,
                        static_cast<uint32_t>(idxElems64)};
    }
}

static void SelectModeAndTile(const ShapeDesc &s, PointId point, ShapeClass shapeClass,
                              uint32_t bytesPerElem, uint32_t coreNum, uint64_t ubBytes,
                              uint64_t perBufElems, uint32_t colAligned,
                              uint32_t N, uint32_t OH, uint32_t OW, ModePlan &plan)
{
    plan = ModePlan{SelectBaseMode(point), 1, s.D, 0, 0};

    if (point == PointId::P7) {
        SelectP7Tile(s, plan);
        return;
    }
    if (shapeClass == ShapeClass::LARGE_D) {
        SelectLargeDTile(s, bytesPerElem, coreNum, perBufElems, OH, OW, N, plan);
        return;
    }

    SelectBaseTile(s, perBufElems, colAligned, plan);

    if (point == PointId::P6) {
        plan.tileW = s.W;
        plan.dTileLen = P6_TILE_D;
        return;
    }
    if (point == PointId::P4) {
        plan = ModePlan{BTS_MODE_B4_W_PACK_TILE, OW, s.D, 1, 0};
        return;
    }
    if (point == PointId::P10) {
        TrySelectP10UbPack(s, bytesPerElem, ubBytes, OW, plan);
        TrySelectP10OhGather(s, bytesPerElem, ubBytes, OW, plan);
    }
    if (point == PointId::P2) {
        TrySelectP2Gather(s, bytesPerElem, ubBytes, OW, plan);
    }
    if (point == PointId::P9) {
        TrySelectP9B4WPack(s, bytesPerElem, ubBytes, OW, plan);
    }
    TrySelectGatherRow(s, point, bytesPerElem, ubBytes, OW, plan);
}

static uint64_t CalcTotalUnits(const ShapeDesc &s, const ModePlan &plan, uint32_t N, uint32_t OH,
                               uint32_t OW, uint64_t outerUnits64, uint64_t flatElems64)
{
    const uint64_t rowOuterUnits64 =
        (plan.mode == BTS_MODE_P1_ROW_FAST || plan.mode == BTS_MODE_P1_B2_DIRECT) ?
        static_cast<uint64_t>(N) * s.block : outerUnits64;
    const uint64_t numTilesPerRow =
        ((plan.mode == BTS_MODE_B4_W_PACK_TILE ? static_cast<uint64_t>(OW) : static_cast<uint64_t>(s.W)) +
         plan.tileW - 1) / plan.tileW;
    const uint64_t numDTiles = (static_cast<uint64_t>(s.D) + plan.dTileLen - 1) / plan.dTileLen;

    if (plan.mode == BTS_MODE_GATHER_ROW || plan.mode == BTS_MODE_P2_GATHER_FAST) {
        return static_cast<uint64_t>(N) * OH;
    }
    if (plan.mode == BTS_MODE_P10_OH_GATHER_TILE || plan.mode == BTS_MODE_P10_UB_PACK_TILE) {
        return static_cast<uint64_t>(N) * ((static_cast<uint64_t>(OH) + plan.tileW - 1) / plan.tileW);
    }
    if (plan.mode == BTS_MODE_B4_W_PACK_TILE) {
        if (IsPoint4FastShape(s)) {
            return static_cast<uint64_t>(N) * s.H;
        }
        return static_cast<uint64_t>(N) * OH * numTilesPerRow;
    }
    if (plan.mode == BTS_MODE_P6_B2_DIRECT) {
        return outerUnits64 * s.H * numDTiles;
    }
    if (plan.mode == BTS_MODE_P1_B2_DIRECT || plan.mode == BTS_MODE_P3_B2_DIRECT ||
        plan.mode == BTS_MODE_P4_B2_DIRECT) {
        return static_cast<uint64_t>(N) * s.block * s.H;
    }
    if (plan.mode == BTS_MODE_P7_FLAT_COPY) {
        const uint64_t elems = static_cast<uint64_t>(s.B) * s.H * s.W * s.D;
        return (elems + plan.tileW - 1) / plan.tileW;
    }
    if (plan.mode == BTS_MODE_LARGE_D || plan.mode == BTS_MODE_P7_D_DIRECT) {
        return static_cast<uint64_t>(N) * OH * OW * numDTiles;
    }
    if (plan.mode == BTS_MODE_FLAT_TILE) {
        const uint64_t flatTiles64 = (flatElems64 + plan.tileW - 1) / plan.tileW;
        return outerUnits64 * flatTiles64;
    }
    return rowOuterUnits64 * s.H * numTilesPerRow;
}

static void FillGatherRowIndex(uint8_t *raw, const ShapeDesc &s, uint32_t bytesPerElem,
                               uint32_t OW, size_t baseSize)
{
    uint32_t *idx = reinterpret_cast<uint32_t *>(raw + baseSize);
    if (IsPoint5FastShape(s)) {
        std::memcpy(idx, GetPoint5GatherIndex().data(), static_cast<size_t>(BTS_MAX_GATHER_IDX_ELEMS) * sizeof(uint32_t));
        return;
    }
    uint32_t k = 0;
    const uint32_t regionStride = AlignUpTo32B(s.W * s.D, bytesPerElem);
    for (uint32_t ow = 0; ow < OW; ++ow) {
        const uint32_t widx = ow + s.cl;
        const uint32_t j = widx % s.block;
        const uint32_t w = widx / s.block;
        const uint32_t base = j * regionStride + w * s.D;
        for (uint32_t d = 0; d < s.D; ++d) {
            idx[k++] = (base + d) * bytesPerElem;
        }
    }
}

static void FillP10GatherIndex(uint8_t *raw, const ShapeDesc &s, uint32_t bytesPerElem,
                               uint32_t tileOH, uint32_t OW, size_t baseSize)
{
    uint32_t *idx = reinterpret_cast<uint32_t *>(raw + baseSize);
    uint32_t k = 0;
    const uint32_t p10HRows = (tileOH + s.block - 1) / s.block;
    const uint32_t srcRowElems = s.W * s.D;
    const uint32_t slabStride = p10HRows * srcRowElems;
    const uint32_t ctMod = s.ct % s.block;
    for (uint32_t localOh = 0; localOh < tileOH; ++localOh) {
        const uint32_t hidx = localOh + s.ct;
        const uint32_t i = hidx % s.block;
        const uint32_t firstLocal = (i + s.block - ctMod) % s.block;
        const uint32_t hRel = (localOh - firstLocal) / s.block;
        for (uint32_t ow = 0; ow < OW; ++ow) {
            const uint32_t widx = ow + s.cl;
            const uint32_t j = widx % s.block;
            const uint32_t w = widx / s.block;
            const uint32_t base = (i * s.block + j) * slabStride + hRel * srcRowElems + w * s.D;
            for (uint32_t d = 0; d < s.D; ++d) {
                idx[k++] = (base + d) * bytesPerElem;
            }
        }
    }
}

static void FillGatherIndex(gert::TilingContext *context, const ShapeDesc &s,
                            const ModePlan &plan, uint32_t bytesPerElem, uint32_t OW)
{
    auto *rawTiling = context->GetRawTilingData();
    const size_t baseSize = sizeof(BatchToSpaceTilingData);
    if (plan.mode != BTS_MODE_GATHER_ROW && plan.mode != BTS_MODE_P10_OH_GATHER_TILE) {
        rawTiling->SetDataSize(baseSize);
        return;
    }

    const size_t idxBytes = static_cast<size_t>(plan.idxLen) * sizeof(uint32_t);
    uint8_t *raw = reinterpret_cast<uint8_t *>(rawTiling->GetData());
    if (plan.mode == BTS_MODE_GATHER_ROW) {
        FillGatherRowIndex(raw, s, bytesPerElem, OW, baseSize);
    } else {
        FillP10GatherIndex(raw, s, bytesPerElem, plan.tileW, OW, baseSize);
    }
    rawTiling->SetDataSize(baseSize + idxBytes);
}
}  // 匿名命名空间

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (coreNum == 0) {
        coreNum = 1;
    }

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    const ge::DataType dtypeX = tensorX->GetDataType();
    const gert::Shape &xShape = tensorX->GetStorageShape();
    const uint32_t B = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t H = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t W = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t D = static_cast<uint32_t>(xShape.GetDim(3));

    const uint32_t bytesPerElem = static_cast<uint32_t>(ge::GetSizeByDataType(dtypeX));
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const gert::TypedContinuousVector<int64_t> *attrCrops = attrs->GetListInt(0);
    const int64_t *blockPtr = attrs->GetInt(1);
    const int64_t *crops = attrCrops->GetData();
    const ShapeDesc shape = {
        B, H, W, D, static_cast<uint32_t>(*blockPtr),
        static_cast<uint32_t>(crops[0]), static_cast<uint32_t>(crops[1]),
        static_cast<uint32_t>(crops[2]), static_cast<uint32_t>(crops[3]),
        dtypeX, ((static_cast<uint64_t>(D) * bytesPerElem) % 32ULL) == 0ULL
    };

    const uint32_t N = shape.B / (shape.block * shape.block);
    const uint32_t OH = shape.H * shape.block - shape.ct - shape.cb;
    const uint32_t OW = shape.W * shape.block - shape.cl - shape.cr;

    const uint64_t outerUnits64 = static_cast<uint64_t>(N) * shape.block * shape.block;
    const uint64_t flatElems64 = static_cast<uint64_t>(shape.H) * shape.W;
    const uint64_t ubBytes = (ubSize != 0) ? ubSize : (192u * 1024u);
    const uint64_t perBufElems = ubBytes / 4 / bytesPerElem;
    const uint32_t colAligned = AlignUpTo32B(shape.D, bytesPerElem);
    const PointId point = DetectPoint(shape);
    const ShapeClass shapeClass = ClassifyShape(point, shape);

    ModePlan plan = {};
    SelectModeAndTile(shape, point, shapeClass, bytesPerElem, coreNum,
                      ubBytes, perBufElems, colAligned, N, OH, OW, plan);

    const uint64_t totalUnits64 = CalcTotalUnits(shape, plan, N, OH, OW, outerUnits64, flatElems64);
    const uint32_t totalUnits = static_cast<uint32_t>(totalUnits64);
    const uint32_t usedCoreNum = SelectCoreNum(point, shapeClass, plan.mode, totalUnits, coreNum);

    const uint32_t DT_X = static_cast<uint32_t>(dtypeX);
    ASCENDC_TPL_SEL_PARAM(context, DT_X, plan.mode);

    const uint32_t alignedFlag =
        (point == PointId::P4 && plan.mode == BTS_MODE_B4_W_PACK_TILE) ? 2u : (shape.d32BAligned ? 1u : 0u);
    BatchToSpaceTilingData *tiling = context->GetTilingData<BatchToSpaceTilingData>();
    *tiling = BatchToSpaceTilingData{
        N, shape.H, shape.W, shape.D, shape.block, shape.ct, shape.cl, OH, OW,
        totalUnits, usedCoreNum, plan.tileW, alignedFlag,
        plan.dTileLen, plan.gatherDepth
    };

    FillGatherIndex(context, shape, plan, bytesPerElem, OW);

    context->SetBlockDim(usedCoreNum);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
}  // 命名空间 optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);

    const int64_t B = xShape->GetDim(0);
    const int64_t H = xShape->GetDim(1);
    const int64_t W = xShape->GetDim(2);
    const int64_t D = xShape->GetDim(3);

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const gert::TypedContinuousVector<int64_t> *attrCrops = attrs->GetListInt(0);
    const int64_t *blockPtr = attrs->GetInt(1);
    const int64_t *crops = attrCrops->GetData();
    const int64_t ct = crops[0];
    const int64_t cb = crops[1];
    const int64_t cl = crops[2];
    const int64_t cr = crops[3];
    const int64_t block = *blockPtr;

    const int64_t N = B / (block * block);
    const int64_t OH = H * block - ct - cb;
    const int64_t OW = W * block - cl - cr;

    yShape->SetDimNum(4);
    yShape->SetDim(0, N);
    yShape->SetDim(1, OH);
    yShape->SetDim(2, OW);
    yShape->SetDim(3, D);

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // 命名空间 ge

namespace ops {
class BatchToSpace : public OpDef {
public:
    explicit BatchToSpace(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("crops").AttrType(REQUIRED).ListInt();
        this->Attr("block_size").AttrType(REQUIRED).Int();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(BatchToSpace);
}  // 命名空间 ops
