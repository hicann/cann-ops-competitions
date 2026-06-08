#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"
#include "../op_kernel/erf_tiling.h"


#include <algorithm>


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNumAiv();

    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    uint32_t inputLength = inputNum * typeLength;

    const uint32_t BLOCK_SIZE = 32;
    uint32_t BufferNum = 2;
    uint32_t ubPartNum = 5;
    uint32_t mode;
    uint32_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    uint32_t totalBlockNum = inputLengthAlgin32 / BLOCK_SIZE;
    uint32_t realBlockNum;
    uint32_t realBlockNum_1;
    if(totalBlockNum <= 8){
         coreNum = totalBlockNum ;//注上面注释回来这里要改
         ubPartNum = 3;
         mode = 1;
    }
    else if (totalBlockNum <= 64) {
       coreNum = std::min(static_cast<uint32_t>(8), coreNum);
       ubPartNum = 3;
       mode = 1;
  }
    else if (totalBlockNum <= 219) {
       uint32_t newcorenum =  (totalBlockNum-56)/9+9; 
       coreNum = std::min(newcorenum, coreNum);
       ubPartNum = 3;
       mode = 1;
  }
    else if (totalBlockNum <= coreNum * 8) {  // ← 边界用 maxCoreNum 动态算
         coreNum  =  totalBlockNum / 7;            // ← 直接用全核，不限制30
         ubPartNum = 3;
         mode      = 1;
   }
    else{
            constexpr uint32_t minBlocksPerCore = 1024;
            uint32_t maxUsable = totalBlockNum / minBlocksPerCore;
            coreNum  = std::min(maxUsable, coreNum); 
            mode      = 2;
    }
    coreNum = std::max(coreNum, static_cast<uint32_t>(1));

    uint32_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
    uint32_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;
    context->SetBlockDim(coreNum);

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    //uint32_t mode = BufferNum;
    if(everyCoreInputBlockNum* BLOCK_SIZE <= (ubSize/3)&& mode == 2)
    {
         mode =1;
         ubPartNum = 3;
    }
    uint32_t tileBlockNum = (ubSize / BLOCK_SIZE) / ubPartNum;
    uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / typeLength;

    uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalSmallTileNum,smallTailDataNum;
    finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    everyCoreInputBlockNum += 1;
    uint32_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint32_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalBigTileNum,bigTailDataNum;
    finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;


    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->tailBlockNum = tailBlockNum;
    tiling->mode = mode;
    context->GetRawTilingData()->SetDataSize(sizeof(ErfTilingData));
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
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
