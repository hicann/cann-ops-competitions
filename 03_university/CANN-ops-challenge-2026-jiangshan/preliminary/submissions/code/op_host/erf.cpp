#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstring>
#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t DB_TILING_MIN_NUM = 262144;
constexpr uint32_t GM_ALIGN_ELEMS = 128;       //512 / sizeof(float)  
constexpr uint32_t REPEAT_ALIGN_ELEMS = 64;     //256 / sizeof(float)
constexpr uint32_t GM_ALIGN_ELEMS32 = 32;      //32 / sizeof(float)

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ErfTilingData* tiling = context->GetTilingData<ErfTilingData>();
    uint64_t inputNum = context->GetRequiredInputTensor(0)->GetShapeSize();
    if (inputNum <= 1024) {
        context->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0));
        
        // Ascend 底层最小内存对齐是 32 Bytes，即 8 个 float
        uint32_t align8 = (inputNum + 7) / 8 * 8;

        tiling->smallCoreDataNum = 0;
        tiling->bigCoreDataNum = align8;
        tiling->finalBigTileNum = 1;
        tiling->finalSmallTileNum = 0;
        tiling->tileDataNum = align8;
        tiling->smallTailDataNum = 0;
        tiling->bigTailDataNum = align8;

        size_t* currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        context->SetBlockDim(1); // 强制单核
        return ge::GRAPH_SUCCESS; // 直接返回，跳过后续复杂运算
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t aivCoreNum = platform.GetCoreNumAiv();
    


    uint64_t ubSize;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t ubBudget = ubSize * 0.9;  

    // DB: 5份缓冲(2in+2out+1calc), NDB: 3份缓冲(1in+1out+1calc)
    uint32_t dbTileMax  = static_cast<uint32_t>(ubBudget / (5 * 4));
    dbTileMax  = (dbTileMax / GM_ALIGN_ELEMS) * GM_ALIGN_ELEMS;
    uint32_t ndbTileMax = static_cast<uint32_t>(ubBudget / (3 * 4));
    ndbTileMax = (ndbTileMax / GM_ALIGN_ELEMS32 ) * GM_ALIGN_ELEMS32 ;

    uint64_t inputNumAlign64 = (inputNum + REPEAT_ALIGN_ELEMS - 1) / REPEAT_ALIGN_ELEMS * REPEAT_ALIGN_ELEMS;

    uint64_t tileDataNum = 0, finalBigTileNum = 0, finalSmallTileNum = 0;
    uint64_t bigTailDataNum = 0, smallTailDataNum = 0;
    uint64_t bigCoreDataNum = 0, smallCoreDataNum = 0;
    int64_t coreNum = aivCoreNum;

    if (inputNum <= DB_TILING_MIN_NUM) {
        // NDB：无双缓冲
        context->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0));

        uint32_t suggested;
        if (inputNum <= 256)          suggested = 256;
        else if (inputNum <= 512)     suggested = 512;
        else if (inputNum <= 2048)    suggested = 512;
        else if (inputNum <= 20480)   suggested = 1024;
        else if (inputNum <= 61440)   suggested = 2048;
        else if (inputNum <= 122880)  suggested = 4096;
        else                          suggested = 8192;

        tileDataNum = (suggested + GM_ALIGN_ELEMS32 - 1) / GM_ALIGN_ELEMS32 * GM_ALIGN_ELEMS32 ;
        if (tileDataNum > ndbTileMax) tileDataNum = ndbTileMax;

        coreNum = (inputNum + tileDataNum - 1) / tileDataNum;

        // 单核时用对齐后的输入大小，但不超UB上限
        if (coreNum == 1) {
            tileDataNum = inputNumAlign64;
            if (tileDataNum > ndbTileMax) {
                tileDataNum = ndbTileMax;
                coreNum = (inputNum + tileDataNum - 1) / tileDataNum;
            }
        }

        finalBigTileNum = 1;
        bigCoreDataNum = tileDataNum;
        bigTailDataNum = inputNumAlign64 - (coreNum - 1) * tileDataNum;
    } else {
        // DB：双缓冲
        context->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1));

        tileDataNum = dbTileMax;
        uint64_t tileDataNum_db = tileDataNum * BUFFER_NUM;
        coreNum = (inputNum + tileDataNum_db - 1) / tileDataNum_db;
        coreNum = (coreNum > aivCoreNum) ? aivCoreNum : coreNum;

        bigCoreDataNum = (inputNum + coreNum - 1) / coreNum;
        bigCoreDataNum = (bigCoreDataNum + GM_ALIGN_ELEMS - 1) / GM_ALIGN_ELEMS * GM_ALIGN_ELEMS;
        finalBigTileNum = (bigCoreDataNum + tileDataNum - 1) / tileDataNum;
        if (((finalBigTileNum - 1) * tileDataNum) > bigCoreDataNum) finalBigTileNum -= 1;
        bigTailDataNum = bigCoreDataNum - (finalBigTileNum - 1) * tileDataNum;

        smallCoreDataNum = inputNumAlign64 - (coreNum - 1) * bigCoreDataNum;
        finalSmallTileNum = (smallCoreDataNum + tileDataNum - 1) / tileDataNum;
        if (((finalSmallTileNum - 1) * tileDataNum) > smallCoreDataNum) finalSmallTileNum -= 1;
        smallTailDataNum = smallCoreDataNum - (finalSmallTileNum - 1) * tileDataNum;
    }

    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->tileDataNum = static_cast<uint32_t>(tileDataNum);
    tiling->smallTailDataNum = static_cast<uint32_t>(smallTailDataNum);
    tiling->bigTailDataNum = static_cast<uint32_t>(bigTailDataNum);
    tiling->finalSmallTileNum = static_cast<uint32_t>(finalSmallTileNum);
    tiling->finalBigTileNum = static_cast<uint32_t>(finalBigTileNum);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = platform.GetLibApiWorkSpaceSize();
    context->SetBlockDim(coreNum);
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext* context) {
        *context->GetOutputShape(0) = *context->GetInputShape(0);
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext*) {
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class Erf : public OpDef {
    public:
        explicit Erf(const char* name) : OpDef(name) {
            this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
            this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
        }
    };
    OP_ADD(Erf);
}  // namespace ops