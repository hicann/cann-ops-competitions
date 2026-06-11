// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/lerp_tiling.h"
#include "../op_kernel/tiling_key_lerp.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t num_cores_aiv = platform.GetCoreNumAiv();
        
        uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
        uint32_t typeLength = 0;
        ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
        uint32_t inputLength = inputNum * typeLength;
        const uint32_t BLOCK_SIZE = 32;

        uint32_t inputLengthAlign32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
        num_cores_aiv = std::min(num_cores_aiv, inputLengthAlign32 / BLOCK_SIZE);
        num_cores_aiv = std::max(num_cores_aiv, static_cast<uint32_t>(1));
        uint32_t perCoreBlockNum = inputLengthAlign32 / BLOCK_SIZE /num_cores_aiv;
        uint32_t tailBlockNum = (inputLengthAlign32 / BLOCK_SIZE) % num_cores_aiv;
        context->SetBlockDim(num_cores_aiv);

        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        uint32_t ubDataNumber = 3;
        const uint32_t BUFFER_NUM = 2;
        uint32_t tileBlockNum = (ub_size / BLOCK_SIZE / BUFFER_NUM) / ubDataNumber;
        uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / typeLength;
        
        uint32_t smallCoreDataNum = perCoreBlockNum * BLOCK_SIZE / typeLength;
        uint32_t smallTileNum = perCoreBlockNum / tileBlockNum;
        uint32_t finalSmallTileNum = (perCoreBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
        uint32_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
        smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;
        
        perCoreBlockNum += 1;
        uint32_t bigCoreDataNum = perCoreBlockNum * BLOCK_SIZE / typeLength;
        uint32_t bigTileNum = perCoreBlockNum / tileBlockNum;
        uint32_t finalBigTileNum = (perCoreBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
        uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
        bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const float *attr_weight = attrs->GetFloat(0);
       
        LerpTilingData *tiling = context->GetTilingData<LerpTilingData>();
        tiling->smallCoreDataNum = smallCoreDataNum;
        tiling->bigCoreDataNum = bigCoreDataNum;
        tiling->tileDataNum = tileDataNum;
        tiling->smallTailDataNum = smallTailDataNum;
        tiling->bigTailDataNum = bigTailDataNum;
        tiling->finalSmallTileNum = finalSmallTileNum;
        tiling->finalBigTileNum = finalBigTileNum;
        tiling->tailBlockNum = tailBlockNum;
        tiling->weight = *attr_weight;

        auto dtype_start = context->GetInputDesc(0)->GetDataType();
        uint32_t DT_START = static_cast<uint32_t>(dtype_start);
        ASCENDC_TPL_SEL_PARAM(context, DT_START);

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape* s_shape = context->GetInputShape(0);
        gert::Shape* y_shape = context->GetOutputShape(0);
        *y_shape = *s_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const auto inputDataType = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDataType);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class Lerp : public OpDef {
    public:
        explicit Lerp(const char *name) : OpDef(name) {
            this->Input("start")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("end")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("weight").AttrType(REQUIRED).Float();
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Lerp);
}  // namespace ops

