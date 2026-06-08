// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

#define AlignUp(value, align) (((value) + (align) - 1U) & ~((align) - 1U))

namespace optiling {
    constexpr uint32_t SMALL_TIER_THRESHOLD = 1024U * 8U;
    constexpr uint32_t SMALL_TIER_CORE_NUM = 8U;
    constexpr uint32_t SMALL_TIER_ALIGN_ELEMS = 8U;

    constexpr uint32_t MID_TIER_THRESHOLD = 4096U * 32U;
    constexpr uint32_t MID_TIER_CORE_NUM = 32U;
    constexpr uint32_t MID_TIER_ALIGN_ELEMS = 128U;

    constexpr uint32_t LARGE_CORE_ALIGN_ELEMS = 128U;
    constexpr uint32_t LARGE_MIN_CORE_COPY_ELEMS = 8192U;

    static uint32_t CeilDiv(uint32_t value, uint32_t divisor) {
        return (value + divisor - 1U) / divisor;
    }

    static uint32_t NeedsPad(uint32_t value) {
        return (value & 7U) != 0U;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        uint32_t totalLen = static_cast<uint32_t>(context->GetRequiredInputTensor(0)->GetShapeSize());

        uint32_t scheduleMode, coreNum;

        ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();

        if (totalLen <= 1U) {
            coreNum = 1U;
            scheduleMode = ERF_SCH_MODE_SINGLE;
        } else if (totalLen <= SMALL_TIER_THRESHOLD) {
            uint32_t coreLenAligned = AlignUp(CeilDiv(totalLen, SMALL_TIER_CORE_NUM),
                                              SMALL_TIER_ALIGN_ELEMS);
            uint32_t fullCoreNum = totalLen / coreLenAligned;
            uint32_t tailLen = totalLen - fullCoreNum * coreLenAligned;
            tiling->coreLenAligned = coreLenAligned;
            tiling->fullCoreNum = fullCoreNum;
            tiling->tailLen = tailLen;
            tiling->tailNeedPad = NeedsPad(tailLen);
            coreNum = fullCoreNum + ((tailLen > 0U) ? 1U : 0U);
            scheduleMode = ERF_SCH_MODE_GENERAL;
        } else if (totalLen <= MID_TIER_THRESHOLD) {
            uint32_t coreLenAligned = AlignUp(CeilDiv(totalLen, MID_TIER_CORE_NUM),
                                              MID_TIER_ALIGN_ELEMS);
            uint32_t fullCoreNum = totalLen / coreLenAligned;
            uint32_t tailLen = totalLen - fullCoreNum * coreLenAligned;
            tiling->coreLenAligned = coreLenAligned;
            tiling->fullCoreNum = fullCoreNum;
            tiling->tailLen = tailLen;
            tiling->tailNeedPad = NeedsPad(tailLen);
            coreNum = fullCoreNum + ((tailLen > 0U) ? 1U : 0U);
            scheduleMode = ERF_SCH_MODE_GENERAL;
        } else {
            auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
            uint32_t maxCoreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());
            uint32_t copyEfficientCoreNum = CeilDiv(totalLen, LARGE_MIN_CORE_COPY_ELEMS);
            coreNum = copyEfficientCoreNum < maxCoreNum ? copyEfficientCoreNum : maxCoreNum;
            uint32_t coreLenAligned = AlignUp(CeilDiv(totalLen, coreNum), LARGE_CORE_ALIGN_ELEMS);
            uint32_t fullCoreNum = totalLen / coreLenAligned;
            uint32_t tailLen = totalLen - fullCoreNum * coreLenAligned;
            tiling->coreLenAligned = coreLenAligned;
            tiling->fullCoreNum = fullCoreNum;
            tiling->tailLen = tailLen;
            tiling->tailNeedPad = NeedsPad(tailLen);
            scheduleMode = ERF_SCH_MODE_GENERAL;
        }

        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(ge::DT_FLOAT), scheduleMode);
        context->SetBlockDim(coreNum);

        context->GetWorkspaceSizes(1)[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

#undef AlignUp

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *xShape = context->GetInputShape(0);
        gert::Shape *yShape = context->GetOutputShape(0);
        auto dimNum = xShape->GetDimNum();
        yShape->SetDimNum(dimNum);
        for (size_t i = 0; i < dimNum; i++) {
            yShape->SetDim(i, xShape->GetDim(i));
        }
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, ge::DT_FLOAT);
        return GRAPH_SUCCESS;
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
