// Host侧Tiling实现

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/addcmul_tiling.h"
#include "../op_kernel/tiling_key_addcmul.h"
#include <algorithm>

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext* context) {
        //每个字节块大小为32字节
        const uint32_t BLOCK_SIZE = 32;
        //申请的内存块的个数
        int32_t NUM = 10;
        uint32_t sizeofdatatype;
        //从平台获取信息
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        auto socVersion = ascendcPlatform.GetSocVersion();
        //获取到的unified buffer大小，缓存区
        uint64_t ub_size;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        //ai core数量
        auto aivNum = ascendcPlatform.GetCoreNum();

        uint32_t totalLength = 0;
        uint32_t minLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
        //找出输入的三个tensor中元素的最大和最小，后续会根据这个数量来决定分块的大小
        for (int i = 0; i < 3; ++i) {
            totalLength = std::max<uint32_t>(totalLength, context->GetInputShape(i)->GetStorageShape().GetShapeSize());
            minLength = std::min<uint32_t>(minLength, context->GetInputShape(i)->GetStorageShape().GetShapeSize());
        }
        //输入数据的数量
        uint32_t input_data_length = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
        uint32_t x1_length = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
        uint32_t x2_length = context->GetInputShape(2)->GetStorageShape().GetShapeSize();
        auto dt = context->GetInputDesc(0)->GetDataType();
        switch(dt) {
                case ge::DT_FLOAT16:
                case ge::DT_BF16:
                    sizeofdatatype = 2;
                    break;
                case ge::DT_INT8:
                    sizeofdatatype = 1;
                    // INT8计算支持有限，先转为half，再转回INT8，多申请了两块buffer
                    NUM = 12;
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
        //每一次处理的块的数量
        uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 2) / NUM;
        //向下对齐到8的倍数,让blocksize对齐到8*ALIGN_NUM，满足向量指令的要求
        if(tiling_size>8){
            tiling_size = tiling_size / 8 * 8;
        }
        //块数*每块元素数，表明每次处理的元素数量
        uint32_t block_size = tiling_size * ALIGN_NUM;
        //判断是否存在广播
        if (totalLength != minLength) {
            //如果存在广播，每次处理的元素数量不能超过最小长度
            block_size = std::min<uint32_t>(block_size, minLength);
            //保证每次处理的元素个数是最小长度的约数的前提下找到最大的一个
            uint32_t best = 1;
            for (uint32_t i = 1; i * i <= minLength; i++) {
                if (minLength % i == 0) {
                    if (i <= block_size && i > best) best = i;
                    uint32_t j = minLength / i;
                    if (j <= block_size && j > best) best = j;
                }
            }
            block_size = best;
        }
        //输出元素个数/每个tile处理元素个数
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
        AddcmulTilingData tiling;
        tiling.set_ALIGN_NUM(ALIGN_NUM);
        tiling.set_block_size(block_size);
        tiling.set_aivNum(aivNum);
        tiling.set_core_size(core_size);
        tiling.set_core_remain(core_remain);
        tiling.set_total_length(totalLength);
        tiling.set_input_data_length(input_data_length);
        tiling.set_x1_length(x1_length);
        tiling.set_x2_length(x2_length);
        //设置启动的AI core的数量
        context->SetBlockDim(aivNum);
        //将Tilling结构体数据保存在buffer内
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
}


namespace ops {
class Addcmul : public OpDef {
public:
    explicit Addcmul(const char* name) : OpDef(name)
    {
        this->Input("input_data")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Addcmul);
}