#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ascendc/host_api/tiling/template_argument.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace {
constexpr uint32_t FLOAT_SIZE = 4;
constexpr uint64_t UB_BYTES_DEFAULT = 192ULL * 1024ULL;
constexpr uint64_t UB_BYTES_RESERVED = 16ULL * 1024ULL;
constexpr uint32_t DDR_BLOCK = 128;
constexpr uint32_t UB_CACHE_MAX = 2048;
constexpr uint32_t TILE_MIN = 4096;
constexpr uint32_t DB_INBUF = 2;
constexpr uint32_t DB_OUTBUF = 2;
constexpr uint32_t DB_TMPBUF = 1;
constexpr uint32_t DB_RATIO = DB_INBUF + DB_OUTBUF + DB_TMPBUF;

static uint64_t FloorTo(uint64_t v, uint64_t a) {
    return (v / a) * a;
}

static uint64_t CeilTo(uint64_t v, uint64_t a) {
    uint64_t r = v % a;
    return r == 0 ? v : v + (a - r);
}

static uint32_t Bound(uint32_t v, uint32_t lo, uint32_t hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static uint32_t CalcCores(uint64_t len, uint32_t maxCores) {
    if (len == 0) return 1;

    uint64_t blocks = len / DDR_BLOCK;
    if (blocks == 0) return 1;

    uint32_t cores = static_cast<uint32_t>(blocks);
    if (cores > maxCores) cores = maxCores;

    // Small inputs: 1 core per block avoids unnecessary multi-core overhead
    if (blocks <= 4) return Bound(static_cast<uint32_t>(blocks), 1, maxCores);

    // Ensure at least 2 blocks per core for effective load balance
    uint64_t perCore = blocks / cores;
    if (perCore < 2) {
        cores = static_cast<uint32_t>(blocks / 2);
        if (cores < 1) cores = 1;
    }

    // Piecewise core caps based on input size (empirically tuned for ascend910b)
    if (len <= 4096)  return Bound(cores, 1, 8);
    if (len <= 16384) return Bound(cores, 1, 16);
    if (len <= 65536) return Bound(cores, 2, 24);
    return Bound(cores, 4, maxCores);
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    uint32_t aivNum = platform.GetCoreNumAiv();
    if (aivNum == 0) aivNum = 1;

    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    if (ubBytes == 0) ubBytes = UB_BYTES_DEFAULT;

    const gert::Tensor *xTensor = context->GetRequiredInputTensor(0);
    if (xTensor == nullptr || xTensor->GetDataType() != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    int64_t shapeSize = xTensor->GetShapeSize();
    uint64_t len = shapeSize > 0 ? static_cast<uint64_t>(shapeSize) : 0ULL;

    uint64_t mode = SCH_CACHE;
    uint32_t cores = 1;
    uint32_t tile = DDR_BLOCK;

    if (len >= 1 && len <= 8) {
        mode = SCH_BLK8;
        cores = 1;
    } else if (len >= 9 && len <= 32) {
        mode = SCH_BLK32;
        cores = 1;
    } else if (len >= 33 && len <= 64) {
        mode = SCH_BLK64;
        cores = 1;
    } else if (len >= 65 && len <= 128) {
        mode = SCH_BLK128;
        cores = 1;
    } else if (len >= 129 && len <= 256) {
        mode = SCH_BLK256;
        cores = 1;
    } else if (len >= 257 && len <= 512) {
        mode = SCH_BLK512;
        cores = 1;
    } else if (len > 512 && len <= UB_CACHE_MAX) {
        mode = SCH_CACHE;
        cores = 1;  // fits in UB, single-core direct path
    } else if (len > UB_CACHE_MAX) {
        mode = SCH_PIPELINE;
        cores = CalcCores(len, aivNum);

        uint64_t blocks = len / DDR_BLOCK;
        uint64_t tail = len % DDR_BLOCK;
        uint64_t perCore = 0;

        if (cores > 1 && blocks > 0) {
            // Branchless ceil division: (blocks + cores - 1) / cores
            uint64_t maxBlocks = (blocks + cores - 1) / cores;
            perCore = maxBlocks * DDR_BLOCK;
            if (tail > 0 && (blocks % cores) == 0) {
                perCore += tail;
            }
        } else {
            perCore = len;
        }

        uint64_t usable = ubBytes > UB_BYTES_RESERVED ? (ubBytes - UB_BYTES_RESERVED) : ubBytes;
        uint64_t maxTileRaw = usable / (DB_RATIO * FLOAT_SIZE);
        uint32_t maxTile = static_cast<uint32_t>(FloorTo(maxTileRaw, DDR_BLOCK));

        uint32_t tileLimit;
        if (maxTile >= TILE_MIN) {
            tileLimit = maxTile;
        } else {
            tileLimit = maxTile > 0 ? maxTile : DDR_BLOCK;
        }

        if (perCore == 0) {
            tile = DDR_BLOCK;
        } else if (perCore <= tileLimit) {
            tile = static_cast<uint32_t>(CeilTo(perCore, DDR_BLOCK));
            if (tile > tileLimit) tile = tileLimit;
        } else {
            tile = tileLimit;
        }
    }

    uint32_t dtype = static_cast<uint32_t>(xTensor->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, dtype, static_cast<uint32_t>(mode));

    ErfTilingDesc *desc = context->GetTilingData<ErfTilingDesc>();
    if (desc == nullptr) return ge::GRAPH_FAILED;

    desc->total = len;
    desc->nCore = cores;
    desc->chunk = tile;
    desc->nBlk = static_cast<uint32_t>(len / DDR_BLOCK);
    desc->rem = static_cast<uint32_t>(len % DDR_BLOCK);

    context->SetBlockDim(cores);

    size_t *ws = context->GetWorkspaceSizes(1);
    if (ws != nullptr) ws[0] = 0;

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *ctx) {
    const gert::Shape *in = ctx->GetInputShape(0);
    gert::Shape *out = ctx->GetOutputShape(0);
    if (in == nullptr || out == nullptr) return GRAPH_FAILED;
    *out = *in;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *ctx) {
    ctx->SetOutputDataType(0, ctx->GetInputDataType(0));
    return ge::SUCCESS;
}
}

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
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(Erf);
}