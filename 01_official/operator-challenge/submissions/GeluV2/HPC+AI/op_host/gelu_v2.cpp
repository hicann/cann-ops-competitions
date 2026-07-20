
#include "gelu_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"   

namespace optiling {
const uint32_t BLOCK_SIZE = 256;   
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    GeluV2TilingData tiling;
    //获取硬件平台存储空间UB的内存大小
    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize); 
    //获取输入shape信息(没对齐)
    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize(); //输入数量
    uint32_t inputBytes = GetSizeByDataType(context->GetInputDesc(0)->GetDataType()); //输入类型
    //printf("inputBytes is %d",inputBytes);
    uint32_t inputLength = inputBytes * inputNum; //输入长度
    //每一次处理量，可使用的ub空间 输入3输出1，手动考虑双缓存
    //假设UB为256KB,每次提供2K个block
    uint32_t ubDataNumber  =  0;
    auto dt = context->GetInputTensor(0)->GetDataType();
    uint32_t sizeofdatatype;  
    if (dt == ge::DT_BF16 || dt == ge::DT_FLOAT16) {      
        ubDataNumber = 12;
    }         
    else if ((dt == ge::DT_FLOAT) ){  
        ubDataNumber = 6;
    }  
    //printf("ubDataNumber is %d",ubDataNumber);
    uint32_t tileBlockNum = (ubSize / BLOCK_SIZE) / ubDataNumber; //每个ub段可用的空间块数，这时候还没有对齐，空间可以有一些冗余
    uint32_t tileDataNum = tileBlockNum * (BLOCK_SIZE / inputBytes) ; //每次处理的数据量,保证数据都可以在UB里  
    //对齐后总量
    //假设4096个float16, 128*2=256个block需要，一次循环就结束了
    uint32_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE); //输入长度 对齐处理
    uint32_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE;// 输入数据需要多少空间块
    uint32_t CoreDataNum = everyCoreInputBlockNum * (BLOCK_SIZE / inputBytes); //对齐空间后的输入数量
    //求循环次数和尾数
    uint32_t TileNum = everyCoreInputBlockNum / tileBlockNum;  //输入数据需要的空间块/每次UB可以提供的空间块
    uint32_t finalTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? TileNum : TileNum + 1; //需要循环处理几次
    uint32_t TailDataNum = CoreDataNum - (tileDataNum * TileNum);   //如果TileNum次处理的数据和对齐后处理的数据一样，也就是没有尾巴
    TailDataNum = TailDataNum == 0 ? tileDataNum : TailDataNum; //最后一次需要处理的数据量
    tiling.set_CoreDataNum(CoreDataNum); //对齐空间后的输入数量
    tiling.set_finalTileNum(finalTileNum);//需要循环处理几次
    tiling.set_tileDataNum(tileDataNum); //每次处理的数据量
    tiling.set_TailDataNum(TailDataNum); //最后一次需要处理的数据量
    context->SetBlockDim(1);

    int32_t approximate = *context->GetAttrs()->GetInt(0);
    tiling.set_Approximate(approximate);
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
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
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(GeluV2);
}
