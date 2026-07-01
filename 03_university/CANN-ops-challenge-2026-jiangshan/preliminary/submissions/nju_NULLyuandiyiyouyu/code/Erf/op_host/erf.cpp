#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

constexpr uint32_t ALIGN_NUM  = 8;
constexpr uint32_t ALIGN_MASK = ALIGN_NUM - 1;

constexpr uint32_t MIN_ELEMS_PER_CORE  = 256;
constexpr uint32_t MIN_CORES_SMALL     = 8;
constexpr uint32_t MAX_WAVE_CORES      = 32;
constexpr uint32_t BALANCED_THRESHOLD  = 23;

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint64_t ubSize;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

        const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
        uint32_t totalLength = tensorX->GetShapeSize();

        uint32_t maxTileLength =
            (uint32_t)(ubSize / ((2 * 2 + 1) * sizeof(float))) & ~ALIGN_MASK;
        uint32_t maxTileLength_buf1 =
            (uint32_t)(ubSize / ((2 * 1 + 1) * sizeof(float))) & ~ALIGN_MASK;

        ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
        tiling->totalLength = totalLength;

        if (totalLength <= BALANCED_THRESHOLD) {
            uint32_t cores = (totalLength < MIN_CORES_SMALL) ? totalLength : MIN_CORES_SMALL;
            if (!cores) cores = 1;
            uint32_t base = totalLength / cores;
            uint32_t rem  = totalLength % cores;
            uint32_t tileLength = ((base + 1) + ALIGN_MASK) & ~ALIGN_MASK;
            if (!tileLength) tileLength = ALIGN_NUM;
            if (tileLength > maxTileLength_buf1) tileLength = maxTileLength_buf1 & ~ALIGN_MASK;
            if (!tileLength) tileLength = ALIGN_NUM;

            tiling->blockLength = base;
            tiling->tileLength  = tileLength;
            tiling->rem         = rem;
            context->SetBlockDim(cores);

            ASCENDC_TPL_SEL_PARAM(context,
                static_cast<uint32_t>(tensorX->GetDataType()),
                1u,  // USE_SMALL=true
                0u); // USE_MID=false
        } else {
            uint32_t targetCores = (totalLength + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE;
            if (targetCores < MIN_CORES_SMALL) targetCores = MIN_CORES_SMALL;
            if (targetCores > MAX_WAVE_CORES)  targetCores = MAX_WAVE_CORES;

            uint32_t blockLength =
                ((totalLength + targetCores - 1) / targetCores + ALIGN_MASK) & ~ALIGN_MASK;
            if (!blockLength) blockLength = ALIGN_NUM;

            uint32_t usableCores = targetCores;
            if (usableCores > 1) {
                uint32_t realCores = (totalLength + blockLength - 1) / blockLength;
                uint32_t tail = totalLength - (realCores - 1) * blockLength;
                if (tail * 3 < blockLength * 2 && blockLength <= 2 * ALIGN_NUM) {
                    usableCores = realCores - 1;
                } else {
                    usableCores = realCores;
                }
            }

            uint32_t lastCoreLen     = totalLength - (usableCores - 1) * blockLength;
            uint32_t lastCoreAligned = (lastCoreLen + ALIGN_MASK) & ~ALIGN_MASK;
            uint32_t effectiveBlock  = (blockLength > lastCoreAligned) ? blockLength : lastCoreAligned;

            bool useMid = (effectiveBlock <= maxTileLength_buf1);
            uint32_t cap = useMid ? maxTileLength_buf1 : maxTileLength;
            uint32_t tileLength = (effectiveBlock < cap) ? effectiveBlock : cap;
            if (!tileLength) tileLength = ALIGN_NUM;

            tiling->blockLength = blockLength;
            tiling->tileLength  = tileLength;
            tiling->rem         = 0;
            context->SetBlockDim(usableCores);

            ASCENDC_TPL_SEL_PARAM(context,
                static_cast<uint32_t>(tensorX->GetDataType()),
                0u,                          // USE_SMALL=false
                static_cast<uint32_t>(useMid)); // USE_MID
        }

        context->GetWorkspaceSizes(1)[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

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
}  // namespace ops
