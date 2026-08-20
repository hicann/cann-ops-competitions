#include "fills_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include <cmath>
#include <cstdint>
#include <algorithm>

inline uint16_t float_to_bfloat16_bits_rne(float f) {
    uint32_t raw;
    std::memcpy(&raw, &f, sizeof(raw));

    uint32_t sign = raw >> 31;
    uint32_t exp  = (raw >> 23) & 0xFF;
    uint32_t mant = raw & 0x7FFFFF;

    // NaN / Inf
    if (exp == 0xFF) {
        uint16_t bf_mant = mant >> 16;
        if (mant != 0 && bf_mant == 0) bf_mant = 1;
        return (sign << 15) | (0xFF << 7) | bf_mant;
    }
    // 零 / 次正规
    if (exp == 0) {
        return sign << 15;
    }

    // 舍入到最近偶数
    uint32_t round_bits = mant & 0xFFFF;
    uint16_t bf_mant = mant >> 16;

    bool round_up = false;
    if (round_bits > 0x8000) {
        round_up = true;
    } else if (round_bits == 0x8000) {
        if (bf_mant & 1) round_up = true;   // ties to even
    }

    if (round_up) {
        bf_mant += 1;
        if (bf_mant == 0x80) {   // 尾数溢出，指数进位
            bf_mant = 0;
            exp += 1;
            if (exp == 0xFF) {
                return (sign << 15) | (0xFF << 7);
            }
        }
    }
    return (sign << 15) | (exp << 7) | bf_mant;
}

// 将 float 舍入到最近的 bfloat16，再转回 float
inline float float_to_bf16_float(float f) {
    uint16_t bits = float_to_bfloat16_bits_rne(f);
    // 将 16 位位模式扩展到 32 位，位模式左对齐
    uint32_t expanded = uint32_t(bits) << 16;
    float result;
    std::memcpy(&result, &expanded, sizeof(result));
    return result;
}
namespace optiling {

const uint32_t TARGET_TILE_BYTES = 16 * 1024;
// const uint32_t BUFFER_NUM = 1;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  //FillsTilingData tiling;
  uint32_t BUFFER_NUM = 2;
  uint32_t BLOCK_SIZE = 512;
  uint64_t ubSize;
  uint32_t NUM =0;

   auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto coreNum = ascendcPlatform.GetCoreNum();
    
    // Based on the input length and the number of inputs, the number of bytes of the input data type is obtained
    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    uint32_t inputLength = inputNum * typeLength;
    uint32_t inputBytes = inputLength / inputNum;

     uint32_t totalDataNum =
        context->GetInputShape(0)->GetStorageShape().GetShapeSize();

    auto dt = context->GetInputTensor(0)->GetDataType();

    float value = *context->GetAttrs()->GetFloat(0);
   // auto dt = context->GetInputTensor(0)->GetDataType();

    if(dt == ge::DT_INT8){
      
        NUM =3;

        float truncated = std::truncf(value);
        float mod = std::fmod(truncated, 256.0f);
        if (mod < 0.0f) mod += 256.0f;
        // 将 [0, 256) 映射到 int8 范围 [-128, 127]，结果仍为 float
        float mapped;
        if (mod >= 128.0f)
            mapped = mod - 256.0f;   // 128~255 -> -128~-1
        else
            mapped = mod;            // 0~127 -> 0~127
        value =mapped;

         BLOCK_SIZE = 512;

        
    }
    else if (dt == ge::DT_UINT8){

        NUM =3;

        float truncated = std::truncf(value);
        // 2. 对 256.0f 取模，得到 [0, 256) 范围内的浮点数
        float mod = std::fmod(truncated, 256.0f);
        // 修正负数模结果
        if (mod < 0.0f) mod += 256.0f;

        value =mod;

        //BLOCK_SIZE = 512;
    }

    else if(dt == ge::DT_FLOAT16 ){
       
        NUM = 1;
        if (inputLength <=4096){
            BLOCK_SIZE =32;

        }
        
        BUFFER_NUM =1;
    }

    else if(dt == ge::DT_BF16 ){
       
        NUM = 3;
        BLOCK_SIZE =32;
        value = float_to_bf16_float(value);
    }
 
    else if (dt == ge::DT_INT16) {
        NUM = 1;
    }

    else if (dt == ge::DT_INT32) {
        NUM = 1;
    }

    else if (dt == ge::DT_FLOAT) {
        NUM = 1;
    }

   uint32_t tileBlockNum = (ubSize / BLOCK_SIZE )  / NUM;
    uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes;

    // Input data for 32B alignment
    uint32_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number of audits is the actual number of audits
    coreNum = (coreNum <  inputLengthAlgin32 / BLOCK_SIZE) ? coreNum : inputLengthAlgin32 / BLOCK_SIZE;
    coreNum = (coreNum >= 1) ? coreNum : 1;
    if(dt == ge::DT_FLOAT16 && inputLength <=4096){

        coreNum =2;
        
    }
    uint32_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
    uint32_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum; //大核的数量
    
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
    
   
    if (dt == ge::DT_BF16){

        
        FillsTilingDataBF16 tiling;

        //std::cout<<"走了这个分支：:"<<std::endl;

        uint32_t alignUnit = BLOCK_SIZE / typeLength;

        uint32_t coreDataNum =  (totalDataNum + coreNum - 1) / coreNum;
    
        coreDataNum = ((coreDataNum + alignUnit - 1) / alignUnit) * alignUnit;
    
        uint32_t tileDataNumALL = TARGET_TILE_BYTES / typeLength;
        tileDataNumALL = (tileDataNumALL / alignUnit) * alignUnit;
    
        uint32_t tileNum = (coreDataNum + tileDataNumALL - 1) / tileDataNumALL;
    
        uint32_t lastTileDataNum =coreDataNum - (tileNum - 1) * tileDataNumALL;
        context->SetTilingKey(1);
        tiling.set_coreDataNum(coreDataNum);
        tiling.set_tileDataNum(tileDataNumALL);
        tiling.set_tileNum(tileNum);
        tiling.set_lastTileDataNum(lastTileDataNum);
        std::cout<<"value:"<<value<<std::endl;
        tiling.set_value(value);
        context->SetBlockDim(coreNum);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
      
    }
    else{

       
        if (dt == ge::DT_FLOAT16 && inputLength <=4096 ){

            FillsTilingDataHalf tiling;
    
            context->SetTilingKey(2);
         
            tiling.set_smallCoreDataNum(smallCoreDataNum);
            tiling.set_bigCoreDataNum(bigCoreDataNum);
            tiling.set_tileDataNum(tileDataNum);
            tiling.set_smallTailDataNum(smallTailDataNum);
            tiling.set_bigTailDataNum(bigTailDataNum);
            tiling.set_finalSmallTileNum(finalSmallTileNum);
            tiling.set_finalBigTileNum(finalBigTileNum);
            tiling.set_tailBlockNum(tailBlockNum);
            tiling.set_value(value);
            context->SetBlockDim(coreNum);
    
            tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
            context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
            size_t *currentWorkspace = context->GetWorkspaceSizes(1);
            currentWorkspace[0] = 0;
            return ge::GRAPH_SUCCESS;
    }
        else {

            FillsTilingData tiling;
           
            context->SetTilingKey(0);
    
            
            tiling.set_smallCoreDataNum(smallCoreDataNum);
            tiling.set_bigCoreDataNum(bigCoreDataNum);
            tiling.set_tileDataNum(tileDataNum);
            tiling.set_smallTailDataNum(smallTailDataNum);
            tiling.set_bigTailDataNum(bigTailDataNum);
            tiling.set_finalSmallTileNum(finalSmallTileNum);
            tiling.set_finalBigTileNum(finalBigTileNum);
            tiling.set_tailBlockNum(tailBlockNum);
            tiling.set_value(value);
            context->SetBlockDim(coreNum);
    
            tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
            context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
            size_t *currentWorkspace = context->GetWorkspaceSizes(1);
            currentWorkspace[0] = 0;
            return ge::GRAPH_SUCCESS;

    }
    }
         
   
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
class Fills : public OpDef {
public:
    explicit Fills(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("value").AttrType(OPTIONAL).Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Fills);
}
