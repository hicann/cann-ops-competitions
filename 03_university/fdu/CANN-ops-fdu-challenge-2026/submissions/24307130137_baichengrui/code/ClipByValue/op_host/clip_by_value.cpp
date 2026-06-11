// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/clip_by_value_tiling.h"
#include "../op_kernel/tiling_key_clip_by_value.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 1. 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t numCores = platform.GetCoreNumAiv();
        uint64_t ubSize;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

        // 2. 获取输入信息
        const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
        ge::DataType dtype = tensorX->GetDataType();
        uint32_t totalLength = tensorX->GetShapeSize();
        int32_t dtypeSize = ge::GetSizeByDataType(dtype);

        // 3. 获取属性（min/max 为 float 标量）
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const float *attrMin = attrs->GetFloat(0);
        const float *attrMax = attrs->GetFloat(1);
        if (attrMin == nullptr || attrMax == nullptr) {
            return ge::GRAPH_FAILED;
        }

        // 4. 配置 TilingKey（区分 dtype）
        uint32_t DT_X = static_cast<uint32_t>(dtype);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // 5. 计算 blockDim：每核至少处理 4KB = 32768 bits
        constexpr int64_t MIN_DATA_BITS = 32768;
        int64_t neededCores = (static_cast<int64_t>(totalLength) * dtypeSize * 8 + MIN_DATA_BITS - 1) / MIN_DATA_BITS;
        uint32_t blockDim = static_cast<uint32_t>(neededCores > int64_t(numCores) ? numCores : neededCores);
        if (blockDim == 0) blockDim = 1;

        // 6. 计算 blockFormer：256 字节对齐
        constexpr int64_t ALIGN_BYTES = 256;
        uint32_t elemAlign = static_cast<uint32_t>(ALIGN_BYTES / dtypeSize);
        uint32_t blockFormer = (totalLength + blockDim - 1) / blockDim;
        blockFormer = ((blockFormer + elemAlign - 1) / elemAlign) * elemAlign;

        // 7. 计算 ubFormer：256B 对齐，受 UB 容量和 blockLen uint16_t 上限约束
        constexpr int32_t BUFFER_NUM = 4;
        int64_t bufferDivisor = static_cast<int64_t>(BUFFER_NUM) * dtypeSize;
        int64_t maxElemNum = static_cast<int64_t>(ubSize) / bufferDivisor;
        constexpr int64_t ALIGN_256 = 256;
        int64_t alignFactor = ALIGN_256 / dtypeSize;
        uint32_t ubFormer = static_cast<uint32_t>((maxElemNum / alignFactor) * alignFactor);

        // 受 uint16_t blockLen 上限 (65535 字节) 约束：
        // ubFormer * dtypeSize <= 65535
        uint32_t maxUbFormerByBlockLen = 65535u / dtypeSize;
        if (ubFormer > maxUbFormerByBlockLen) {
            ubFormer = (maxUbFormerByBlockLen / static_cast<uint32_t>(alignFactor)) * static_cast<uint32_t>(alignFactor);
        }
        // 确保 ubFormer 至少为 alignFactor（防止对齐后为 0）
        if (ubFormer < static_cast<uint32_t>(alignFactor)) {
            ubFormer = static_cast<uint32_t>(alignFactor);
        }

        // 8. 填充 TilingData
        ClipByValueTilingData *tiling = context->GetTilingData<ClipByValueTilingData>();
        tiling->length = totalLength;
        tiling->blockDim = blockDim;
        tiling->blockFormer = blockFormer;
        tiling->ubFormer = ubFormer;
        tiling->min_val = *attrMin;
        tiling->max_val = *attrMax;

        // 9. 配置启动核数和 workspace 大小
        context->SetBlockDim(blockDim);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *x_shape = context->GetInputShape(0);
        gert::Shape *y_shape = context->GetOutputShape(0);
        *y_shape = *x_shape;
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

