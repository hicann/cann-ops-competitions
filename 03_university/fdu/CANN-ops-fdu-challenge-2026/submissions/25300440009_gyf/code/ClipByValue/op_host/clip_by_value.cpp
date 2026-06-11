// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/clip_by_value_tiling.h"
#include "../op_kernel/tiling_key_clip_by_value.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext* context) {
        //每个字节块大小为32字节
        const uint32_t BLOCK_SIZE = 32;
        //申请的buffer数
        constexpr int32_t NUM = 4;
        uint32_t sizeofdatatype;
        //从平台获取信息
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        auto socVersion = ascendcPlatform.GetSocVersion();
        //获取到的unified buffer大小，缓存区
        uint64_t ub_size;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        //ai core数量
        auto aivNum = ascendcPlatform.GetCoreNum();
        //从算子第一个参数形状获取总元素个数
        uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
        //张量的数据类型
        auto dt = context->GetInputDesc(0)->GetDataType();
        switch(dt) {
            case ge::DT_FLOAT16:
            case ge::DT_BF16:
                sizeofdatatype = 2;
                break;
            case ge::DT_FLOAT:
            case ge::DT_INT32:
                sizeofdatatype = 4;
                break;
            default:
                return ge::GRAPH_FAILED;
        }
        //对齐参数计算，一个字节块能存的元素的个数    
        uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
        /*
            ub_size/BLOCK_SIZE UB能容纳的32字节块的数量
            上式/2，分为计算和缓存两部分
            上式/NUM，为每一块申请的buffer分配内存空间
        */
       //每一次处理的块的数量
        uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 2) / NUM;
        //向下对齐到8的倍数
        if(tiling_size>8){
            tiling_size = tiling_size / 8 * 8;
        }
        //块数*每块元素数，表明每次处理的元素数量
        uint32_t block_size = tiling_size * ALIGN_NUM;
        //实际使用的核心数量
        if(aivNum > totalLength / block_size) {
            aivNum = totalLength / block_size;
        }
        if(aivNum < 1) {
            aivNum = 1;
        }
        //每个核心处理的元素数量，向下对齐到ALIGN_NUM的8倍,保证每个核的起始地址按 ALIGN_NUM 对齐，处理的数据量是 8 的倍数（向量指令要求）
        uint32_t core_size = (totalLength / aivNum) / (ALIGN_NUM * 8) * (ALIGN_NUM * 8);
        //未分配的元素数量
        uint32_t core_remain = totalLength - aivNum * core_size;
        //序列化Tiling数据
        ClipByValueTilingData tiling;
        tiling.set_totalLength(totalLength);
        tiling.set_ALIGN_NUM(ALIGN_NUM);
        tiling.set_tiling_size(tiling_size);
        tiling.set_block_size(block_size);
        tiling.set_aivNum(aivNum);
        tiling.set_core_size(core_size);
        tiling.set_core_remain(core_remain);
        //从context中获取算子属性
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        if (attrs != nullptr) {
            //获取第一个float属性
            const float *minVal = attrs->GetFloat(0);
            //获取第二个
            const float *maxVal = attrs->GetFloat(1);
            if (minVal != nullptr && maxVal != nullptr) {
                tiling.set_minVal(*minVal);
                tiling.set_maxVal(*maxVal);
            }
        }
        //设置启动的AI core的数量
        context->SetBlockDim(aivNum);
        //将Tilling结构体数据保存在buffer内
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        //获取工作空间
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        //设置为0，不需要额外空间
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape* x1_shape = context->GetInputShape(0);
        gert::Shape* y_shape = context->GetOutputShape(0);
        *y_shape = *x1_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        ge::DataType input_dtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, input_dtype);
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

