// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 示例: 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        // 示例: 获取算子输入数组信息
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        const gert::StorageShape *shape_x = context->GetRequiredInputShape(0);
        auto storage_shape = shape_x->GetStorageShape();
        uint32_t dim_num = storage_shape.GetDimNum();
        uint32_t last_dim = dim_num > 0 ? static_cast<uint32_t>(storage_shape.GetDim(dim_num - 1)) : 1;
        ge::DataType dtype_x = tensor_x->GetDataType(); // 获取数据类型
        int dtype_size_x = ge::GetSizeByDataType(dtype_x); // 获取数据类型的字长
        uint32_t length_x = tensor_x->GetShapeSize(); // 获取元素个数
        uint32_t size_x = tensor_x->GetSize(); // 获取内存大小
        // 示例: 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        // 示例: 计算tiling方案并填充tiling结构体
        ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
        uint32_t block_dim = 0;
        if (length_x > 65536 && length_x <= 262144) {
            block_dim = dim_num > 2 ? 27 : 24;
        } else if (length_x > 2048 && length_x <= 65536) {
            block_dim = 1;
        } else if (length_x <= 64 && dim_num > 2 && (last_dim & 31) == 0) {
            block_dim = 4;
        } else if (length_x <= 64 && dim_num <= 2) {
            block_dim = (length_x + 23) / 24;
        } else if (length_x <= 128 && dim_num <= 2) {
            block_dim = (length_x + 15) / 16;
        } else if (length_x <= 2048 && dim_num <= 2) {
            block_dim = (length_x + 23) / 24;
        } else if (length_x <= 8192 && dim_num <= 2) {
            block_dim = (length_x + 19) / 20;
        } else if (length_x > 8192 && length_x <= 65536) {
            block_dim = (length_x + 95) / 96;
        } else {
            block_dim = (length_x + 31) / 32;
        }
        if (block_dim == 0) {
            block_dim = 1;
        }
        if (block_dim > static_cast<uint32_t>(num_cores_aiv)) {
            block_dim = static_cast<uint32_t>(num_cores_aiv);
        }
        tiling->length = length_x;
        tiling->blockDim = block_dim;
        tiling->dimNum = dim_num;
        // 配置启动核数
        context->SetBlockDim(block_dim);
        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

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
}  // namespace ops
