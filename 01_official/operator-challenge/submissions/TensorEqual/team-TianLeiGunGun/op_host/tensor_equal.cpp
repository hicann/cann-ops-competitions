#include "tensor_equal_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstdint>
#include <tuple>

namespace optiling
{
    const uint32_t BLOCK_SIZE = 32;
    const uint32_t BUFFER_NUM = 2;

    static std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>
    singleCoreTiling(uint64_t ubSize, uint32_t inputNum, uint32_t inputBytes, uint32_t UB_PATITION_NUM, uint32_t INPUT_PARTITION_WEIGHT)
    {
        uint32_t ubPartition_Size_AF256 = ubSize / UB_PATITION_NUM;
        ubPartition_Size_AF256 = ubPartition_Size_AF256 / 256 * 256;
        uint32_t tmp = (inputNum * inputBytes + INPUT_PARTITION_WEIGHT - 1) / INPUT_PARTITION_WEIGHT;
        tmp = (tmp + 31) / 32 * 32;
        uint32_t ubPartition_Size_AF32 = std::min(ubPartition_Size_AF256, tmp);

        uint32_t ubPartition_Length_AF32 = ubPartition_Size_AF32 / inputBytes;
        uint32_t formerLength = ubPartition_Length_AF32 * INPUT_PARTITION_WEIGHT;
        uint32_t formerNum = inputNum / formerLength;
        uint32_t tailLength = inputNum - formerNum * formerLength;
        return std::make_tuple(formerLength, formerNum, tailLength, ubPartition_Length_AF32);
    }
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {

        TensorEqualTilingData tiling;

        uint32_t x1Size = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
        uint32_t x2Size = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
        uint32_t ySize = context->GetOutputShape(0)->GetStorageShape().GetShapeSize();
        uint32_t blockDim = 1;

        if (ySize != x1Size || ySize != x2Size)
        {
            auto shape_y = context->GetOutputShape(0)->GetOriginShape();
            auto shape_x1 = context->GetInputTensor(0)->GetOriginShape();
            auto shape_x2 = context->GetInputTensor(1)->GetOriginShape();
            int32_t y_dimensional = shape_y.GetDimNum();
            int32_t x1_dimensional = shape_x1.GetDimNum();
            int32_t x2_dimensional = shape_x2.GetDimNum();
            ge::DataType dataType = context->GetInputTensor(0)->GetDataType();
            bool vectorBroadcastType = dataType == ge::DT_FLOAT16 || dataType == ge::DT_FLOAT || dataType == ge::DT_INT32 || dataType == ge::DT_INT8;
            bool x1LastDimBroadcast = false;
            bool x2LastDimBroadcast = false;
            bool x1InnerVectorBroadcast = false;
            bool x2InnerVectorBroadcast = false;
            bool x1LeadingBroadcast = false;
            bool x2LeadingBroadcast = false;
            uint32_t lastDim = 1;
            uint32_t outerDim = 1;
            if (vectorBroadcastType && y_dimensional == x1_dimensional && y_dimensional == x2_dimensional && y_dimensional >= 2)
            {
                int32_t lastIndex = y_dimensional - 1;
                bool prefixSame = true;
                for (int32_t i = 0; i < lastIndex; ++i)
                {
                    if (shape_x1.GetDim(i) != shape_y.GetDim(i) || shape_x2.GetDim(i) != shape_y.GetDim(i))
                    {
                        prefixSame = false;
                    }
                    outerDim *= shape_y.GetDim(i);
                }
                lastDim = shape_y.GetDim(lastIndex);
                x1LastDimBroadcast = prefixSame && shape_x1.GetDim(lastIndex) == 1 && shape_x2.GetDim(lastIndex) == shape_y.GetDim(lastIndex);
                x2LastDimBroadcast = prefixSame && shape_x2.GetDim(lastIndex) == 1 && shape_x1.GetDim(lastIndex) == shape_y.GetDim(lastIndex);
                x1InnerVectorBroadcast = shape_x1.GetDim(lastIndex) == 1 && shape_x2.GetDim(lastIndex) == shape_y.GetDim(lastIndex);
                x2InnerVectorBroadcast = shape_x2.GetDim(lastIndex) == 1 && shape_x1.GetDim(lastIndex) == shape_y.GetDim(lastIndex);
            }
            if (vectorBroadcastType && y_dimensional == x1_dimensional && y_dimensional == x2_dimensional && y_dimensional >= 2)
            {
                bool x1PrefixOnly = x1Size < ySize && x2Size == ySize;
                bool x2PrefixOnly = x2Size < ySize && x1Size == ySize;
                bool x1SeenSuffix = false;
                bool x2SeenSuffix = false;
                for (int32_t i = 0; i < y_dimensional; ++i)
                {
                    if (x1PrefixOnly)
                    {
                        if (!x1SeenSuffix && shape_x1.GetDim(i) == 1 && shape_y.GetDim(i) != 1)
                        {
                        }
                        else if (shape_x1.GetDim(i) == shape_y.GetDim(i))
                        {
                            x1SeenSuffix = true;
                        }
                        else
                        {
                            x1PrefixOnly = false;
                        }
                    }
                    if (x2PrefixOnly)
                    {
                        if (!x2SeenSuffix && shape_x2.GetDim(i) == 1 && shape_y.GetDim(i) != 1)
                        {
                        }
                        else if (shape_x2.GetDim(i) == shape_y.GetDim(i))
                        {
                            x2SeenSuffix = true;
                        }
                        else
                        {
                            x2PrefixOnly = false;
                        }
                    }
                }
                x1LeadingBroadcast = x1PrefixOnly && x1SeenSuffix;
                x2LeadingBroadcast = x2PrefixOnly && x2SeenSuffix;
            }

            if ((x1Size == 1 && x2Size == ySize) || (x2Size == 1 && x1Size == ySize))
            {
                context->SetTilingKey(9);
                tiling.set_y_dimensional(x1Size == 1 ? 1 : 2);
                uint32_t formerLength = 1024;
                uint32_t formerNum = ySize / formerLength;
                uint32_t tailLength = ySize - formerNum * formerLength;
                tiling.set_formerLength(formerLength);
                tiling.set_formerNum(formerNum);
                tiling.set_tailLength(tailLength);
                tiling.set_totalLength(ySize);
                tiling.set_partitionLength(formerLength);
            }
            else if (x1LastDimBroadcast || x2LastDimBroadcast)
            {
                context->SetTilingKey(10);
                tiling.set_y_dimensional(x1LastDimBroadcast ? 1 : 2);
                tiling.set_formerLength(lastDim);
                tiling.set_formerNum(outerDim);
                tiling.set_tailLength(0);
                tiling.set_totalLength(ySize);
                tiling.set_partitionLength(1024);
            }
            else if (x1InnerVectorBroadcast || x2InnerVectorBroadcast)
            {
                context->SetTilingKey(12);
                int32_t outerRank = y_dimensional - 1;
                int32_t y_ndarray[20], x1_ndarray[20], x2_ndarray[20];
                int32_t y_sumndarray[20], x1_sumndarray[20], x2_sumndarray[20];
                uint32_t outerTotal = 1;
                int32_t x1Stride = shape_x1.GetDim(y_dimensional - 1);
                int32_t x2Stride = shape_x2.GetDim(y_dimensional - 1);
                for (int32_t i = outerRank - 1; i >= 0; --i)
                {
                    y_ndarray[i] = shape_y.GetDim(i);
                    x1_ndarray[i] = shape_x1.GetDim(i);
                    x2_ndarray[i] = shape_x2.GetDim(i);
                    x1_sumndarray[i] = x1Stride;
                    x2_sumndarray[i] = x2Stride;
                    x1Stride *= shape_x1.GetDim(i);
                    x2Stride *= shape_x2.GetDim(i);
                }
                for (int32_t i = 0; i < outerRank; ++i)
                {
                    outerTotal *= y_ndarray[i];
                }
                y_sumndarray[0] = outerTotal;
                tiling.set_y_dimensional(outerRank);
                tiling.set_y_ndarray(y_ndarray);
                tiling.set_x1_ndarray(x1_ndarray);
                tiling.set_x2_ndarray(x2_ndarray);
                tiling.set_y_sumndarray(y_sumndarray);
                tiling.set_x1_sumndarray(x1_sumndarray);
                tiling.set_x2_sumndarray(x2_sumndarray);
                tiling.set_tailLength(x1InnerVectorBroadcast ? 1 : 2);
                tiling.set_formerLength(lastDim);
                tiling.set_formerNum(outerTotal);
                tiling.set_totalLength(ySize);
                tiling.set_partitionLength(1024);
            }
            else if (x1LeadingBroadcast || x2LeadingBroadcast)
            {
                context->SetTilingKey(11);
                uint32_t blockLength = x1LeadingBroadcast ? x1Size : x2Size;
                tiling.set_y_dimensional(x1LeadingBroadcast ? 1 : 2);
                tiling.set_formerLength(blockLength);
                tiling.set_formerNum(ySize / blockLength);
                tiling.set_tailLength(0);
                tiling.set_totalLength(ySize);
                tiling.set_partitionLength(1024);
            }
            else
            {
                context->SetTilingKey(5);

                int32_t y_ndarray[20], x1_ndarray[20], x2_ndarray[20];

                int32_t max_y_dimensional;
                max_y_dimensional = y_dimensional;
                if (x1_dimensional > max_y_dimensional)
                    max_y_dimensional = x1_dimensional;
                if (x2_dimensional > max_y_dimensional)
                    max_y_dimensional = x2_dimensional;

                for (int i = 0; i < max_y_dimensional; i++)
                {
                    if (i < y_dimensional)
                        y_ndarray[y_dimensional - i - 1] = shape_y.GetDim(i);
                    else
                        y_ndarray[i] = 1;
                    if (i < x1_dimensional)
                        x1_ndarray[x1_dimensional - i - 1] = shape_x1.GetDim(i);
                    else
                        x1_ndarray[i] = 1;
                    if (i < x2_dimensional)
                        x2_ndarray[x2_dimensional - i - 1] = shape_x2.GetDim(i);
                    else
                        x2_ndarray[i] = 1;
                }
                tiling.set_y_dimensional(max_y_dimensional);
                tiling.set_y_ndarray(y_ndarray);
                tiling.set_x1_ndarray(x1_ndarray);
                tiling.set_x2_ndarray(x2_ndarray);

                int32_t y_sumndarray[20], x1_sumndarray[20], x2_sumndarray[20];
                y_sumndarray[0] = 1;
                x1_sumndarray[0] = 1;
                x2_sumndarray[0] = 1;
                for (int i = 1; i <= max_y_dimensional; i++)
                {
                    y_sumndarray[i] = y_sumndarray[i - 1] * y_ndarray[i - 1];
                    x1_sumndarray[i] = x1_sumndarray[i - 1] * x1_ndarray[i - 1];
                    x2_sumndarray[i] = x2_sumndarray[i - 1] * x2_ndarray[i - 1];
                }
                tiling.set_y_sumndarray(y_sumndarray);
                tiling.set_x1_sumndarray(x1_sumndarray);
                tiling.set_x2_sumndarray(x2_sumndarray);
            }
        }
        else
        {

            uint64_t ubSize;
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
            ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

            // 获取输入数据的shapesize    shape(2,3,4),shapesize=24, 660
            uint32_t x1Num = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
            // 获取输入数据的类型 如 half 2
            uint32_t x1Bytes = GetSizeByDataType(context->GetInputDesc(0)->GetDataType());
            // 数据的大小 1320
            uint32_t x1Length = x1Num * x1Bytes;

            // 使用的变量数
            ge::DataType dataType = context->GetInputTensor(0)->GetDataType();
            uint32_t INPUT_PARTITION_WEIGHT;  // 输入变量 的分区比重
            uint32_t INPUT_PARTITION_NUM;     // 输入变量 的分区数目
            uint32_t CALC_PARTITION_WEIGHT;   // 计算用临时变量 的分区比重
            uint32_t CALC_PARTITION_NUM;      // 计算用临时变量 的分区数目
            uint32_t OUTPUT_PARTITION_WEIGHT; // 输出变量 的分区比重
            uint32_t OUTPUT_PARTITION_NUM;    // 输出变量 的分区数目
            switch (dataType)
            {
            case ge::DT_BF16:
                context->SetTilingKey(6);
                INPUT_PARTITION_WEIGHT = 2;
                INPUT_PARTITION_NUM = 2;

                CALC_PARTITION_WEIGHT = 9;
                CALC_PARTITION_NUM = 1;

                OUTPUT_PARTITION_WEIGHT = 1;
                OUTPUT_PARTITION_NUM = 1;
                break;
            case ge::DT_INT32:
                context->SetTilingKey(3);
                INPUT_PARTITION_WEIGHT = 4;
                INPUT_PARTITION_NUM = 2;

                CALC_PARTITION_WEIGHT = 6;
                CALC_PARTITION_NUM = 1;

                OUTPUT_PARTITION_WEIGHT = 1;
                OUTPUT_PARTITION_NUM = 1;
                break;
            case ge::DT_FLOAT:
                context->SetTilingKey(2);
                INPUT_PARTITION_WEIGHT = 4;
                INPUT_PARTITION_NUM = 2;

                CALC_PARTITION_WEIGHT = 3;
                CALC_PARTITION_NUM = 1;

                OUTPUT_PARTITION_WEIGHT = 1;
                OUTPUT_PARTITION_NUM = 1;
                break;
            case ge::DT_FLOAT16:
                context->SetTilingKey(1);
                INPUT_PARTITION_WEIGHT = 2;
                INPUT_PARTITION_NUM = 2;

                CALC_PARTITION_WEIGHT = 1;
                CALC_PARTITION_NUM = 1;

                OUTPUT_PARTITION_WEIGHT = 1;
                OUTPUT_PARTITION_NUM = 1;
                break;
            case ge::DT_INT16:
                context->SetTilingKey(7);
                INPUT_PARTITION_WEIGHT = 2;
                INPUT_PARTITION_NUM = 2;

                CALC_PARTITION_WEIGHT = 9;
                CALC_PARTITION_NUM = 1;

                OUTPUT_PARTITION_WEIGHT = 1;
                OUTPUT_PARTITION_NUM = 1;
                break;
            case ge::DT_UINT8:
                context->SetTilingKey(8);
                INPUT_PARTITION_WEIGHT = 1;
                INPUT_PARTITION_NUM = 2;

                CALC_PARTITION_WEIGHT = 9;
                CALC_PARTITION_NUM = 1;

                OUTPUT_PARTITION_WEIGHT = 1;
                OUTPUT_PARTITION_NUM = 1;
                break;
            case ge::DT_INT8:
                context->SetTilingKey(4);
                INPUT_PARTITION_WEIGHT = 1;
                INPUT_PARTITION_NUM = 2;

                CALC_PARTITION_WEIGHT = 5;
                CALC_PARTITION_NUM = 1;

                OUTPUT_PARTITION_WEIGHT = 1;
                OUTPUT_PARTITION_NUM = 1;
                break;
            default:
                break;
            }
            uint32_t UB_PATITION_NUM = // 总计分区块数
                INPUT_PARTITION_WEIGHT * INPUT_PARTITION_NUM * BUFFER_NUM +
                CALC_PARTITION_WEIGHT * CALC_PARTITION_NUM +
                OUTPUT_PARTITION_WEIGHT * OUTPUT_PARTITION_NUM * BUFFER_NUM;

            auto tilingInfo = singleCoreTiling(ubSize, x1Num, x1Bytes, UB_PATITION_NUM, INPUT_PARTITION_WEIGHT);
            uint32_t formerLength = std::get<0>(tilingInfo);
            uint32_t formerNum = std::get<1>(tilingInfo);
            uint32_t tailLength = std::get<2>(tilingInfo);
            uint32_t partitionLength = std::get<3>(tilingInfo);
            tiling.set_formerLength(formerLength);
            tiling.set_formerNum(formerNum);
            tiling.set_tailLength(tailLength);
            tiling.set_totalLength(x1Num);
            tiling.set_partitionLength(partitionLength);
        }

        context->SetBlockDim(blockDim);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}

namespace ge
{
    static ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        const gert::Shape *x1_shape = context->GetInputShape(0);
        gert::Shape *y_shape = context->GetOutputShape(0);
        *y_shape = *x1_shape;
        return GRAPH_SUCCESS;
    }
}

namespace ops
{
    class TensorEqual : public OpDef
    {
    public:
        explicit TensorEqual(const char *name) : OpDef(name)
        {
            this->Input("x1")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("x2")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

            this->SetInferShape(ge::InferShape);

            this->AICore()
                .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b").AddConfig("ascend310b");
        }
    };

    OP_ADD(TensorEqual);
}
