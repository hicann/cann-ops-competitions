#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t num_cores = platform.GetCoreNumAiv();

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        uint32_t totalLength = tensor_x->GetShapeSize();

        ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
        
        if (totalLength == 0) {
            context->SetBlockDim(1);
            tiling->totalLength = 0;
            return ge::GRAPH_SUCCESS;
        }

        // --- 优化点：256字节(64个float)对齐，这是910B的DMA黄金比例 ---
        const uint32_t ALIGN_VAL = 64; 
        uint32_t totalBlocks = (totalLength + ALIGN_VAL - 1) / ALIGN_VAL;
        
        // --- 优化点：动态核数策略，小数据量强行单核，防止调度比计算慢 ---
        uint32_t usedCoreNum = num_cores;
        if (totalLength < 16384) { // 阈值：小于16K元素不建议分多核
            usedCoreNum = 1;
        } else if (totalBlocks < num_cores) {
            usedCoreNum = totalBlocks;
        }

        uint32_t blocksPerCore = totalBlocks / usedCoreNum;
        uint32_t tailBlocks = totalBlocks % usedCoreNum;

        // --- 优化点：极致的大 Tile。在UB不溢出的情况下，尽量装满数据以减少循环开销 ---
        // 910B UB通常有 192KB~256KB。分配给 X 和 Y 两个 Queue，每个 Queue 设双缓冲。
        // 一个 float 4字节。设每个 Tile 处理 4096 个元素(16KB)，4块共64KB，非常安全且高效。
        uint32_t tileLength = 4096; 

        tiling->totalLength = totalLength;
        tiling->blocksPerCore = blocksPerCore;
        tiling->tailBlocks = tailBlocks;
        tiling->tileLength = tileLength; 

        context->SetBlockDim(usedCoreNum);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        return ge::GRAPH_SUCCESS;
    }
} // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) { return GRAPH_SUCCESS; }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) { return ge::GRAPH_SUCCESS; }
}

namespace ops {
    class Erf : public OpDef {
    public:
        explicit Erf(const char *name) : OpDef(name) {
            this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
            this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore().SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b");
        }
    };
    OP_ADD(Erf);
}