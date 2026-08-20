
#include "assign_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
const uint32_t BLOCK_SIZE = 512;
const uint32_t TARGET_TILE_BYTES = 16 * 1024;
const uint32_t BUFFER_NUM = 2;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  
   uint64_t ubSize;
    uint32_t NUM =0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto coreNum = ascendcPlatform.GetCoreNum();
    
    // Based on the input length and the number of inputs, the number of bytes of the input data type is obtained
  

    uint32_t datatype = 0;

    auto dt = context->GetInputTensor(0)->GetDataType();
    if(dt == ge::DT_INT8 || dt == ge::DT_UINT8 || dt == ge::DT_BOOL){
      
        NUM =4;
        datatype =1;
    }
    else if(dt == ge::DT_FLOAT16 ){
       
        NUM = 4;
        datatype =2;
    }

    else if(dt == ge::DT_BF16 ){
       
        NUM = 4;
         datatype =2;
    }
 
    else if (dt == ge::DT_INT16) {
        NUM = 4;
         datatype =2;
    }
    else if (dt == ge::DT_INT32){

        NUM = 4;
         datatype =4;

        
    }

    else if (dt == ge::DT_FLOAT) {
        NUM = 4;
         datatype =4;
    }

    uint32_t x1Size = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t x2Size = context->GetInputShape(1)->GetStorageShape().GetShapeSize();

    //获取输入shape信息

    if (dt == ge::DT_FLOAT16 ){

        AssignTilingDataHalf tiling;

        context->SetTilingKey(0);

        uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
        uint32_t typeLength = 0;
        ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
        uint32_t inputLength = inputNum * typeLength;
        uint32_t inputBytes = inputLength / inputNum;
    
        uint32_t totalDataNum =
            context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    
        uint32_t typeLen = 0;
        ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLen);

        uint32_t alignUnit = 32 / typeLen;

        uint32_t coreDataNum =  (totalDataNum + 1 - 1) / 1;
    
        coreDataNum = ((coreDataNum + alignUnit - 1) / alignUnit) * alignUnit;
    
        uint32_t tileDataNumALL = TARGET_TILE_BYTES / typeLen;
        tileDataNumALL = (tileDataNumALL / alignUnit) * alignUnit;
    
        uint32_t tileNum = (coreDataNum + tileDataNumALL - 1) / tileDataNumALL;
    
        uint32_t lastTileDataNum =coreDataNum - (tileNum - 1) * tileDataNumALL;

        context->SetBlockDim(1);
        tiling.set_coreDataNum(coreDataNum);
        tiling.set_tileDataNum(tileDataNumALL);
        tiling.set_tileNum(tileNum);
        tiling.set_lastTileDataNum(lastTileDataNum);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
        

    }
    else {
        AssignTilingData tiling;

        if( x1Size != x2Size)
    {
        
        context->SetTilingKey(2);

        int32_t y_ndarray[20], x1_ndarray[20], x2_ndarray[20];
        int32_t y_dimensional, x1_dimensional, x2_dimensional;
        //auto shape_y = context->GetOutputShape(0)->GetOriginShape();
        auto shape_x1 = context->GetInputTensor(0)->GetOriginShape();
        auto shape_x2 = context->GetInputTensor(1)->GetOriginShape();

        y_dimensional =  shape_x1.GetDimNum();
        x1_dimensional =  shape_x1.GetDimNum();
        x2_dimensional =  shape_x2.GetDimNum();

        for(int i = 0; i < y_dimensional; i++)
        {
            y_ndarray[y_dimensional-i-1] = shape_x1.GetDim(i);
            if(i<x1_dimensional) x1_ndarray[x1_dimensional-i-1] = shape_x1.GetDim(i);
            else                    x1_ndarray[i] = 1;
            if(i<x2_dimensional) x2_ndarray[x2_dimensional-i-1] = shape_x2.GetDim(i);
            else                  x2_ndarray[i] = 1;
        }
        
        tiling.set_y_dimensional(y_dimensional);
        tiling.set_y_ndarray(y_ndarray);
        tiling.set_x1_ndarray(x1_ndarray);
        tiling.set_x2_ndarray(x2_ndarray);

        int32_t y_sumndarray[20], x1_sumndarray[20], x2_sumndarray[20];
        y_sumndarray[0] = 1;
        x1_sumndarray[0] = 1;
        x2_sumndarray[0] = 1;
        for(int i = 1; i <= y_dimensional; i++)
        {
            y_sumndarray[i] = y_sumndarray[i-1]*y_ndarray[i-1];
            x1_sumndarray[i] = x1_sumndarray[i-1]*x1_ndarray[i-1];
            x2_sumndarray[i] = x2_sumndarray[i-1]*x2_ndarray[i-1];
        }
        tiling.set_y_sumndarray(y_sumndarray);
        tiling.set_x1_sumndarray(x1_sumndarray);
        tiling.set_x2_sumndarray(x2_sumndarray);
        context->SetBlockDim(1);
    }
    else
    {
        context->SetTilingKey(1);
        context->SetBlockDim(coreNum);
    }

    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    uint32_t inputLength = inputNum * typeLength;
    uint32_t inputBytes = inputLength / inputNum;

    uint32_t tileBlockNum = (ubSize / BLOCK_SIZE / BUFFER_NUM) / NUM;
    int32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes;


    // Input data for 32B alignment
    uint32_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number of audits is the actual number of audits
    coreNum = (coreNum <  inputLengthAlgin32 / BLOCK_SIZE) ? coreNum : inputLengthAlgin32 / BLOCK_SIZE;
    coreNum = (coreNum >= 1) ? coreNum : 1;
    uint32_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
    uint32_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;
    
    // Small chunks are calculated and sliced several times using the number of data on each core
    uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    // Tail block calculation for small chunks of data
    uint32_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;
    
    // The total length of a large block of data is 32B larger than that of a small block of data
    everyCoreInputBlockNum += 1;
    uint32_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint32_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;
    
    tiling.set_smallCoreDataNum(smallCoreDataNum);
    tiling.set_bigCoreDataNum(bigCoreDataNum);
    tiling.set_tileDataNum(tileDataNum);
    tiling.set_smallTailDataNum(smallTailDataNum);
    tiling.set_bigTailDataNum(bigTailDataNum);
    tiling.set_finalSmallTileNum(finalSmallTileNum);
    tiling.set_finalBigTileNum(finalBigTileNum);
    tiling.set_tailBlockNum(tailBlockNum);
    
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
    }

    
    

    

    // There are a total of 3 shared UB spaces in the input and output. If it's int8, there are 2 more TBUFs
    //uint32_t ubDataNumber = (inputBytes == 1) ? 5 : 3;
    // The number of 32B data blocks that can be used for each data. DOUBLE BUFFER is already counted here
   
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class Assign : public OpDef {
public:
    explicit Assign(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("other")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("use_locking").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Assign);
}
