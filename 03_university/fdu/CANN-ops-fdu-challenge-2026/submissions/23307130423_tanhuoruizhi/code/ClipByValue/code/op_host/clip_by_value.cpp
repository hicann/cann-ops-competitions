// Host侧Tiling实现
#include <algorithm>

#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/clip_by_value_tiling.h"
#include "../op_kernel/tiling_key_clip_by_value.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 示例: 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t core_num = platform.GetCoreNum();
        // 示例: 获取算子输入数组信息
        uint32_t input_num = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
        uint32_t type_length = 0;
        ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), type_length);
        uint32_t input_length = input_num * type_length;
        // 示例: 获取算子输入属性
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const float *attr_min = attrs->GetFloat(0);
        const float *attr_max = attrs->GetFloat(1);
        // 示例: 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_X = static_cast<uint32_t>(context->GetInputDesc(0)->GetDataType()); // 根据x的数据类型配置tiling key
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        // 示例: 计算tiling方案并填充tiling结构体
        constexpr uint32_t BLOCK_SIZE = 32;
        constexpr uint32_t MIN_BYTES_PER_CORE = 12 * 1024;
        uint32_t input_length_align32 = (((input_length + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
        uint32_t input_block_num = input_length_align32 / BLOCK_SIZE;
        uint32_t core_num_by_size = (input_length_align32 + MIN_BYTES_PER_CORE - 1) / MIN_BYTES_PER_CORE;
        core_num = std::min(core_num, core_num_by_size);
        core_num = std::min(core_num, input_block_num);
        core_num = std::max(core_num, static_cast<uint32_t>(1));
        uint32_t every_core_input_block_num = input_block_num / core_num;
        uint32_t tail_block_num = input_block_num % core_num;
        context->SetBlockDim(core_num);

        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        constexpr uint32_t BUFFER_NUM = 2;
        constexpr uint32_t UB_DATA_NUMBER = 2; // xLocal + yLocal
        uint32_t tile_block_num = (ub_size / BLOCK_SIZE / BUFFER_NUM) / UB_DATA_NUMBER;
        uint32_t tile_data_num = (tile_block_num * BLOCK_SIZE) / type_length;

        uint32_t small_core_data_num = every_core_input_block_num * BLOCK_SIZE / type_length;
        uint32_t small_tile_num = every_core_input_block_num / tile_block_num;
        uint32_t final_small_tile_num = (every_core_input_block_num % tile_block_num) == 0 ? small_tile_num : small_tile_num + 1;
        uint32_t small_tail_data_num = small_core_data_num - (tile_data_num * small_tile_num);
        small_tail_data_num = small_tail_data_num == 0 ? tile_data_num : small_tail_data_num;

        // 大核比小核多处理一个32B block。
        every_core_input_block_num += 1;
        uint32_t big_core_data_num = every_core_input_block_num * BLOCK_SIZE / type_length;
        uint32_t big_tile_num = every_core_input_block_num / tile_block_num;
        uint32_t final_big_tile_num = (every_core_input_block_num % tile_block_num) == 0 ? big_tile_num : big_tile_num + 1;
        uint32_t big_tail_data_num = big_core_data_num - tile_data_num * big_tile_num;
        big_tail_data_num = big_tail_data_num == 0 ? tile_data_num : big_tail_data_num;

        ClipByValueTilingData *tiling = context->GetTilingData<ClipByValueTilingData>();
        tiling->smallCoreDataNum = small_core_data_num;
        tiling->bigCoreDataNum = big_core_data_num;
        tiling->finalBigTileNum = final_big_tile_num;
        tiling->finalSmallTileNum = final_small_tile_num;
        tiling->tileDataNum = tile_data_num;
        tiling->smallTailDataNum = small_tail_data_num;
        tiling->bigTailDataNum = big_tail_data_num;
        tiling->tailBlockNum = tail_block_num;
        tiling->minValue = *attr_min;
        tiling->maxValue = *attr_max;
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
        *outputShape = *inputShape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
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
