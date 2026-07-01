// Host-side op definition, shape/type inference, and tiling for Erf.
#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace
{
    constexpr uint32_t BUFFER_NUM = 4;          // Multi: inQueue×2 + outQueue×1 + tmp1Buf×1
    constexpr uint32_t BLOCK_BYTES = 32;
    constexpr uint32_t WIDE_ALIGN_BYTES = 512;   // 512B GM alignment for max DMA burst efficiency
    constexpr uint32_t MIN_TILE_LENGTH = 8;

    uint32_t AlignDown(uint32_t value, uint32_t align)
    {
        if (align == 0)
        {
            return value;
        }
        return value / align * align;
    }

    uint32_t AlignUp(uint32_t value, uint32_t align)
    {
        if (align == 0)
        {
            return value;
        }
        return (value + align - 1) / align * align;
    }

} // namespace

namespace optiling
{
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t coreNum = platform.GetCoreNumAiv();

        uint64_t ubSize = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

        const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
        ge::DataType dtypeX = tensorX->GetDataType();

        uint32_t totalLength = static_cast<uint32_t>(tensorX->GetShapeSize());
        const uint32_t typeSize = static_cast<uint32_t>(ge::GetSizeByDataType(dtypeX));
        const uint32_t alignNum = std::max<uint32_t>(BLOCK_BYTES / typeSize, 1);
        const uint32_t wideAlignNum = std::max<uint32_t>(WIDE_ALIGN_BYTES / typeSize, 1);  // 128 elems for float

        uint32_t maxTileLength = static_cast<uint32_t>((ubSize - 128) / typeSize / BUFFER_NUM);
        // tileLength 对齐到 512B：每个 tile 在 GM 中的偏移都是 512B 对齐，DMA 突发最优
        uint32_t tileAlignNum = (maxTileLength >= wideAlignNum) ? wideAlignNum : alignNum;
        maxTileLength = std::max<uint32_t>(AlignDown(maxTileLength, tileAlignNum), MIN_TILE_LENGTH);

        uint32_t targetCoreNum = 1;
        if (totalLength > alignNum)
        {
            constexpr uint32_t MIN_CORES = 2;
            uint32_t c = static_cast<uint32_t>(coreNum);

            if (totalLength < 5000U)
            {
                uint32_t elemsPerCore;
                if      (totalLength <=  32) elemsPerCore = 32;
                else if (totalLength <= 128) elemsPerCore = 16;
                else if (totalLength <= 256) elemsPerCore = 16;
                else if (totalLength <= 512) elemsPerCore = 32;
                else if (totalLength <= 850U) elemsPerCore = 40;
                else if (totalLength <= 1024U) elemsPerCore = 40;
                else                           elemsPerCore = 64;
                targetCoreNum = std::min<uint32_t>(
                    (totalLength + elemsPerCore - 1) / elemsPerCore, c);
            }
            else
            {
                if      (totalLength <  8000U) targetCoreNum = c * 29 / 32;
                else if (totalLength < 12000U) targetCoreNum = c * 3 / 4;
                else if (totalLength < 16000U) targetCoreNum = c * 5 / 8;
                else if (totalLength < 20000U) targetCoreNum = c * 9 / 16;
                else if (totalLength < 24000U) targetCoreNum = c * 21 / 32;
                else if (totalLength < 28000U) targetCoreNum = c * 25 / 32;
                else if (totalLength < 32000U) targetCoreNum = c * 29 / 32;
                else if (totalLength < 40000U) targetCoreNum = c * 31 / 32;
                else                           targetCoreNum = c;
            }
            targetCoreNum = std::max<uint32_t>(targetCoreNum, MIN_CORES);
        }
        uint32_t blockLength_raw = (totalLength + targetCoreNum - 1) / targetCoreNum;
        // blockLength_raw > 2560: padding 占比 < 5%，用 512B 对齐；否则用 32B 对齐
        uint32_t blockLength = (blockLength_raw > 2560)
            ? AlignUp(blockLength_raw, wideAlignNum)
            : AlignUp(blockLength_raw, alignNum);
        blockLength = std::max<uint32_t>(blockLength, alignNum);
        uint32_t usedCoreNum = std::max<uint32_t>((totalLength + blockLength - 1) / blockLength, 1);

        uint32_t tileLength = maxTileLength;

        ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
        uint32_t tailLength = AlignUp(totalLength - (usedCoreNum - 1) * blockLength, alignNum);

        tiling->tileLength = tileLength;
        tiling->blockLength = blockLength;
        tiling->tailLength = tailLength;
        tiling->formerNum = usedCoreNum;
        // isSingle: 每核数据量 ≤ 一次 MTE 搬运（tileLength），分发到 KernelErfSingle 类
        tiling->isSingle = (blockLength <= tileLength) ? 1 : 0;
        tiling->fullTileCount   = blockLength / tileLength;        // Multi 用
        tiling->tailFullTileCnt = tailLength  / tileLength;        // Multi 尾核用

        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dtypeX));
        context->SetBlockDim(usedCoreNum);

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
} // namespace optiling

namespace ge
{
    static graphStatus InferShape(gert::InferShapeContext *context)
    {
        const gert::Shape *xShape = context->GetInputShape(0);
        if (xShape == nullptr)
        {
            return GRAPH_FAILED;
        }
        gert::Shape *yShape = context->GetOutputShape(0);
        if (yShape == nullptr)
        {
            return GRAPH_FAILED;
        }
        yShape->SetDimNum(xShape->GetDimNum());
        for (size_t i = 0; i < xShape->GetDimNum(); ++i)
        {
            yShape->SetDim(i, xShape->GetDim(i));
        }
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
    }
} // namespace ge

namespace ops
{
    class Erf : public OpDef
    {
    public:
        explicit Erf(const char *name) : OpDef(name)
        {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
                .AutoContiguous();
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
                .AutoContiguous();
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Erf);
} // namespace ops