#include<vector>
#include "gelu_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

using namespace std;

// --------------------[set tiling]--------------------
#define SetTilingData(obj, field, value) obj.set_##field(value)

// --------------------[print function]--------------------
#define ENABLE_PRINT 0  // 改成 0 就可以禁止打印

#if ENABLE_PRINT
    #define MY_PRINTF(...) printf(__VA_ARGS__)
    #define PRINT_SHAPE(...) printShape(__VA_ARGS__)
    #define PRINT_ARRAY(...) printArray(__VA_ARGS__)
#else
    #define MY_PRINTF(...)  // 空宏
    #define PRINT_SHAPE(...)
    #define PRINT_ARRAY(...)
#endif

void printShape(vector<int32_t> &x_shape){
    for(int i = 0; i < x_shape.size(); ++i){
        printf("%d ", x_shape[i]);
    }
    printf("\n");
}

void printArray(float *arr, int32_t len){
    for(int32_t i = 0; i < len; ++i){
        printf("%f ", arr[i]);
    }
    printf("\n");
}

// --------------------[tiling fuction]--------------------
// 单核切分参数
const int32_t BLOCK_SIZE = 32;
struct SingleTilingArg {
    int32_t tileNum;
    int32_t tileLength;
    int32_t lastTileLength;
    int32_t invalidLength;
    int32_t resTileNum;
    int32_t invalidLenPerResTile;
    float ub_util;
};

// 单核切分函数
SingleTilingArg computeSingleTilingArgs(int32_t ubSize, int32_t totalLength, int32_t sizeOfDataType, int32_t BUFFER_NUM, int32_t dataNum){
    /*
        这个函数按照 UB 大小（ubSize），数据个数（dataNum），数据字节数（sizeOfDataType），数据总数（totalLength），buffer个数（BUFFER_NUM），计算数据切分参数。

        关键是数据长度不仅需要和 BLOCK_SIZE 对齐还要和 BUFFER_NUM 对齐。
    */
    int32_t blockDataNum = BLOCK_SIZE / sizeOfDataType;     // 1 * BLOCK_SIZE 有多少个数据
    int32_t bufferBlockDataNum = blockDataNum * BUFFER_NUM; // BUFFER_NUM * BLOCK_SIZE 有多少个数据
    int32_t tatalLengthAligned = (totalLength + bufferBlockDataNum - 1) / bufferBlockDataNum * bufferBlockDataNum;  // 按 BUFFER_NUM 和 BLOCK_SIZE 进行对齐的数据长度
    int32_t max_tile_block_num = ubSize / dataNum / BUFFER_NUM / BLOCK_SIZE;    // 每个单元数据在一个 buffer 内所能够使用的最大 BLOCK 数量，所能使用的最大 UB 的容量，按照 BLOCK 大小计算的

    // 由于要使用CompareScalar函数，保证256字节对齐，为了 512字节对齐搬运效果最佳
    int32_t repeatBytes = 256;
    int32_t blockNumPerRepeat = repeatBytes / sizeOfDataType / blockDataNum;
    max_tile_block_num = max_tile_block_num / blockNumPerRepeat * blockNumPerRepeat;

    int32_t tile_block_num = max_tile_block_num / BUFFER_NUM * BUFFER_NUM;  // 再次 buffer 对齐，单个数据在 UB 上所能使用的 BLOCK 数量
    int32_t alignedDataBlockNum = tatalLengthAligned / blockDataNum;    // 已对齐数据的 BLOCK 数量
    SingleTilingArg arg;
    arg.tileNum = alignedDataBlockNum / tile_block_num; // 按照 tile_block_num 切分，得到分块个数
    arg.tileNum = arg.tileNum / BUFFER_NUM * BUFFER_NUM;    // 保证 tileNum 是 BUFFER_NUM 的倍数
    arg.tileLength = tile_block_num * blockDataNum; // 一个分块的长度
    arg.lastTileLength = (tatalLengthAligned - arg.tileNum * arg.tileLength) / BUFFER_NUM;
    if(arg.lastTileLength == 0){
        arg.lastTileLength = arg.tileLength;
        arg.tileNum = arg.tileNum - BUFFER_NUM;
    }
    arg.invalidLength = arg.tileNum * arg.tileLength + arg.lastTileLength * BUFFER_NUM - totalLength;   // 无效长度
    arg.resTileNum = BUFFER_NUM - arg.invalidLength / arg.lastTileLength;   // BUFFER_NUM 个尾块中的有效块个数
    arg.invalidLenPerResTile = arg.tileNum * arg.tileLength + arg.lastTileLength * arg.resTileNum - totalLength;    // 既包含有效数据和无效数据的那个尾块中的无效数据个数

    arg.ub_util = (arg.tileNum * arg.tileLength + BUFFER_NUM * arg.lastTileLength) * dataNum * sizeOfDataType / ubSize;

    return arg;
}

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext* context) {
        // 1. 获取平台信息
        uint64_t ub_size;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        auto socVersion = ascendcPlatform.GetSocVersion();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        int32_t aivNum = ascendcPlatform.GetCoreNumAiv();
        int32_t aicNum = ascendcPlatform.GetCoreNumAic();
        MY_PRINTF("ub_size = %ldB, aivNum = %d, aicNum = %d\n", ub_size, aivNum, aicNum);
        aivNum = 1;
        MY_PRINTF("Auto set aivNum to %d\n", aivNum);

        // 2. 获取数据信息
        auto dt = context->GetInputTensor(0)->GetDataType();
        const int64_t *papproximate = context->GetAttrs()->GetInt(0);
        int32_t approximate = *papproximate;
        int32_t xTotalLength = context->GetInputTensor(0)->GetShapeSize();

        MY_PRINTF("approximate: %d\n", approximate);

        GeluV2TilingData1 tiling;
        if(approximate == 0){
            SingleTilingArg tilingArgs;
            int32_t sizeOfDataType, bufferNum, dataNum;
            if(dt == ge::DT_FLOAT){
                MY_PRINTF("DT_FLOAT\n");
                context->SetTilingKey(0);
                sizeOfDataType = sizeof(float);
                bufferNum = 2;
                dataNum = 3;
                tilingArgs = computeSingleTilingArgs(ub_size, xTotalLength, sizeOfDataType, bufferNum, dataNum);
            }else if(dt == ge::DT_BF16){
                MY_PRINTF("DT_BF16\n");
                context->SetTilingKey(1);
                sizeOfDataType = 2;
                bufferNum = 2;
                dataNum = 5;
                tilingArgs = computeSingleTilingArgs(ub_size, xTotalLength, sizeOfDataType, bufferNum, dataNum);
            }else if(dt == ge::DT_FLOAT16){
                MY_PRINTF("DT_FLOAT16\n");
                context->SetTilingKey(2);
                sizeOfDataType = 2;
                bufferNum = 2;
                dataNum = 2;
                tilingArgs = computeSingleTilingArgs(ub_size, xTotalLength, sizeOfDataType, bufferNum, dataNum);
            }

            MY_PRINTF("ub_util: %f\n", tilingArgs.ub_util);
            MY_PRINTF("tileNum: %d,  tileLength: %d\n", tilingArgs.tileNum, tilingArgs.tileLength);
            MY_PRINTF("lastTileLength: %d, invalidLength: %d\n", tilingArgs.lastTileLength, tilingArgs.invalidLength);
            MY_PRINTF("resTileNum: %d, invalidLenPerResTile: %d\n", tilingArgs.resTileNum, tilingArgs.invalidLenPerResTile);

            SetTilingData(tiling, xTotalLength, xTotalLength);
            SetTilingData(tiling, tileNum, tilingArgs.tileNum);
            SetTilingData(tiling, tileLength, tilingArgs.tileLength);
            SetTilingData(tiling, lastTileLength, tilingArgs.lastTileLength);
            SetTilingData(tiling, invalidLength, tilingArgs.invalidLength);
            SetTilingData(tiling, resTileNum, tilingArgs.resTileNum);
            SetTilingData(tiling, invalidLenPerResTile, tilingArgs.invalidLenPerResTile);
        }else{
            SingleTilingArg tilingArgs;
            int32_t sizeOfDataType, bufferNum, dataNum;
            if(dt == ge::DT_FLOAT){
                MY_PRINTF("DT_FLOAT\n");
                context->SetTilingKey(3);
                sizeOfDataType = sizeof(float);
                bufferNum = 2;
                dataNum = 3;
                tilingArgs = computeSingleTilingArgs(ub_size, xTotalLength, sizeOfDataType, bufferNum, dataNum);
            }else if(dt == ge::DT_BF16){
                MY_PRINTF("DT_BF16\n");
                context->SetTilingKey(4);
                sizeOfDataType = 2;
                bufferNum = 2;
                dataNum = 5;
                tilingArgs = computeSingleTilingArgs(ub_size, xTotalLength, sizeOfDataType, bufferNum, dataNum);
            }else if(dt == ge::DT_FLOAT16){
                MY_PRINTF("DT_FLOAT16\n");
                context->SetTilingKey(5);
                sizeOfDataType = 2;
                bufferNum = 2;
                dataNum = 5;
                tilingArgs = computeSingleTilingArgs(ub_size, xTotalLength, sizeOfDataType, bufferNum, dataNum);
            }

            MY_PRINTF("ub_util: %f\n", tilingArgs.ub_util);
            MY_PRINTF("tileNum: %d,  tileLength: %d\n", tilingArgs.tileNum, tilingArgs.tileLength);
            MY_PRINTF("lastTileLength: %d, invalidLength: %d\n", tilingArgs.lastTileLength, tilingArgs.invalidLength);
            MY_PRINTF("resTileNum: %d, invalidLenPerResTile: %d\n", tilingArgs.resTileNum, tilingArgs.invalidLenPerResTile);

            SetTilingData(tiling, xTotalLength, xTotalLength);
            SetTilingData(tiling, tileNum, tilingArgs.tileNum);
            SetTilingData(tiling, tileLength, tilingArgs.tileLength);
            SetTilingData(tiling, lastTileLength, tilingArgs.lastTileLength);
            SetTilingData(tiling, invalidLength, tilingArgs.invalidLength);
            SetTilingData(tiling, resTileNum, tilingArgs.resTileNum);
            SetTilingData(tiling, invalidLenPerResTile, tilingArgs.invalidLenPerResTile);
        }
        

        context->SetBlockDim(aivNum);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        return ge::GRAPH_SUCCESS;
    }
}


namespace ge {
    static ge::graphStatus InferShape(gert::InferShapeContext* context) {
        const gert::Shape* x1_shape = context->GetInputShape(0);
        gert::Shape* y_shape = context->GetOutputShape(0);
        *y_shape = *x1_shape;
        return GRAPH_SUCCESS;
    }

    static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const auto inputDataType = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDataType);
        return ge::GRAPH_SUCCESS;
    }
}


namespace ops {
class GeluV2 : public OpDef {
public:
    explicit GeluV2(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("approximate").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(GeluV2);
}
