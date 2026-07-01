#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t coreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());
        if (coreNum == 0U) {
            coreNum = 1U;
        }

        const gert::Tensor *x = context->GetRequiredInputTensor(0);
        ge::DataType dtype = x->GetDataType();
        uint32_t length = static_cast<uint32_t>(x->GetShapeSize());

        uint32_t usePoly = (length >= 16384U) ? 1U : 0U;

        uint32_t unitSize = (length >= 65536U) ? 128U : 8U;
        uint32_t totalUnits = (length + unitSize - 1U) / unitSize;

        uint32_t blockDim = 1U;
        if (totalUnits > 0U) {
            uint32_t minUnitsPerCore = 32U;

            if (unitSize == 8U && length >= 16384U) {
                minUnitsPerCore = 16U;
            }
            if (unitSize == 128U) {
                minUnitsPerCore = 8U;
            }

            uint32_t needCore = (totalUnits + minUnitsPerCore - 1U) / minUnitsPerCore;
            blockDim = needCore < coreNum ? needCore : coreNum;
            blockDim = blockDim < totalUnits ? blockDim : totalUnits;

            if (blockDim == 0U) {
                blockDim = 1U;
            }
        }

        uint32_t unitBase = 0U;
        uint32_t unitRemainder = 0U;
        if (blockDim > 0U) {
            unitBase = totalUnits / blockDim;
            unitRemainder = totalUnits % blockDim;
        }

        uint32_t maxUnitsPerBlock = unitBase;
        if (unitRemainder > 0U) {
            maxUnitsPerBlock += 1U;
        }

        uint32_t maxTileLength = (usePoly == 0U) ? 8192U : 6144U;
        uint32_t tileLength = maxUnitsPerBlock * unitSize;

        if (tileLength > maxTileLength) {
            tileLength = maxTileLength;
        }
        if (tileLength < 4096U && length >= 4096U) {
            tileLength = 4096U;
        }
        if (tileLength < 8U) {
            tileLength = 8U;
        }

        tileLength = ((tileLength + 127U) / 128U) * 128U;
        if (tileLength > maxTileLength) {
            tileLength = maxTileLength;
        }

        auto *tiling = context->GetTilingData<ErfTilingData>();
        tiling->length = length;
        tiling->blockDim = blockDim;
        tiling->unitBase = unitBase;
        tiling->unitRemainder = unitRemainder;
        tiling->unitSize = unitSize;
        tiling->tileLength = tileLength;
        tiling->mode = usePoly;

        context->SetBlockDim(blockDim);
        context->GetWorkspaceSizes(1)[0] = 0;

        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dtype));
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        *context->GetOutputShape(0) = *context->GetInputShape(0);
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class Erf : public OpDef {
    public:
        explicit Erf(const char *name) : OpDef(name) {
            this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
            this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
        }
    };

    OP_ADD(Erf);
}  // namespace ops