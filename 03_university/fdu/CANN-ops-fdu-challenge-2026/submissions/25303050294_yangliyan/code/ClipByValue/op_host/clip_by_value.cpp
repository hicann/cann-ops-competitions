// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/clip_by_value_tiling.h"
#include "../op_kernel/tiling_key_clip_by_value.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        
        // 示例: 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        // 示例: 获取算子输入数组信息
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType(); // 获取数据类型
        int dtype_size_x = ge::GetSizeByDataType(dtype_x); // 获取数据类型的字长
        uint32_t length_x = tensor_x->GetShapeSize(); // 获取元素个数
        uint32_t size_x = tensor_x->GetSize(); // 获取内存大小

        // 示例: 获取算子输入属性
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const float *attr_min = attrs->GetFloat(0);
        const float *attr_max = attrs->GetFloat(1);

        const uint32_t BLOCK_SIZE = 32;
        uint32_t blockElemNum = BLOCK_SIZE / dtype_size_x;
        uint32_t totalBlockNum = (length_x + blockElemNum - 1) / blockElemNum;
        num_cores_aiv = std::min(num_cores_aiv, static_cast<int32_t>(totalBlockNum));
        num_cores_aiv = std::max(num_cores_aiv, 1);
        uint32_t everyCoreInputBlockNum = totalBlockNum / num_cores_aiv;
        uint32_t tailBlockNum = totalBlockNum % num_cores_aiv;

        uint32_t ubDataNumber = 2;
        uint32_t tileBlockNum = static_cast<uint32_t>(ub_size / BLOCK_SIZE / ubDataNumber);
        uint32_t tileDataNum = tileBlockNum * blockElemNum;
        if (tileDataNum == 0) {
            tileDataNum = blockElemNum;
        }

        uint32_t smallCoreDataNum = everyCoreInputBlockNum * blockElemNum;
        uint32_t smallTileNum = smallCoreDataNum / tileDataNum;
        uint32_t finalSmallTileNum = (smallCoreDataNum % tileDataNum) == 0 ? smallTileNum : (smallTileNum + 1);
        uint32_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
        smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

        uint32_t bigCoreDataNum = (everyCoreInputBlockNum + 1) * blockElemNum;
        uint32_t bigTileNum = bigCoreDataNum / tileDataNum;
        uint32_t finalBigTileNum = (bigCoreDataNum % tileDataNum) == 0 ? bigTileNum : (bigTileNum + 1);
        uint32_t bigTailDataNum = bigCoreDataNum - (tileDataNum * bigTileNum);
        bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

        // 示例: 计算tiling方案并填充tiling结构体
        ClipByValueTilingData *tiling = context->GetTilingData<ClipByValueTilingData>();
        tiling->smallCoreDataNum = smallCoreDataNum;
        tiling->bigCoreDataNum = bigCoreDataNum;
        tiling->finalSmallTileNum = finalSmallTileNum;
        tiling->tileDataNum = tileDataNum;
        tiling->smallTailDataNum = smallTailDataNum;
        tiling->bigTailDataNum = bigTailDataNum;
        tiling->tailBlockNum = tailBlockNum;
        tiling->finalBigTileNum = length_x;
        tiling->min = *attr_min;
        tiling->max = *attr_max;

        // 配置启动核数
        context->SetBlockDim(num_cores_aiv);

        // 示例: 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *inputShape = context->GetInputShape(0);
        gert::Shape *outputShape = context->GetOutputShape(0);
        *outputShape = *inputShape; // 输出形状与输入形状相同
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0)); // 输出数据类型与输入数据类型相同
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class ClipByValue : public OpDef {
    public:
        explicit ClipByValue(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("min").AttrType(REQUIRED).Float();
            this->Attr("max").AttrType(REQUIRED).Float();
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(ClipByValue);
}  // namespace ops

