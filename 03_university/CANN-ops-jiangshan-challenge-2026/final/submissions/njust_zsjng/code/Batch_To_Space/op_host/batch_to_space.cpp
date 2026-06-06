// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/batch_to_space_tiling.h"
#include "../op_kernel/tiling_key_batch_to_space.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();

        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        // 获取输入 Tensor 信息
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();

        // 输入形状：[batch, height, width, depth]
        const gert::Shape &x_shape = tensor_x->GetOriginShape();

        uint32_t batch = static_cast<uint32_t>(x_shape.GetDim(0));
        uint32_t height = static_cast<uint32_t>(x_shape.GetDim(1));
        uint32_t width = static_cast<uint32_t>(x_shape.GetDim(2));
        uint32_t depth = static_cast<uint32_t>(x_shape.GetDim(3));

        uint32_t input_length = static_cast<uint32_t>(tensor_x->GetShapeSize());

        // 获取属性 crops 和 block_size
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const gert::TypedContinuousVector<int64_t> *attr_crops = attrs->GetListInt(0);
        const int64_t *attr_block_size = attrs->GetInt(1);

        // crops 不能用 (*attr_crops)[i] 访问，需要先取 GetData()
        const int64_t *crops_data = attr_crops->GetData();

        uint32_t crop_top = static_cast<uint32_t>(crops_data[0]);
        uint32_t crop_bottom = static_cast<uint32_t>(crops_data[1]);
        uint32_t crop_left = static_cast<uint32_t>(crops_data[2]);
        uint32_t crop_right = static_cast<uint32_t>(crops_data[3]);
        uint32_t block_size = static_cast<uint32_t>(*attr_block_size);

        // BatchToSpace 输出形状
        uint32_t block_area = block_size * block_size;
        uint32_t out_batch = batch / block_area;
        uint32_t out_height = height * block_size - crop_top - crop_bottom;
        uint32_t out_width = width * block_size - crop_left - crop_right;
        uint32_t output_length = out_batch * out_height * out_width * depth;

        // 配置 tiling key，用于 kernel 侧区分 fp16 / fp32
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        uint32_t block_dim = static_cast<uint32_t>(num_cores_aiv);
        if (output_length <= 65536 && block_dim > 8) {
            block_dim = 8;
        } else if (output_length <= 262144 && block_dim > 16) {
            block_dim = 16;
        }
        
        if (output_length < block_dim) {
            block_dim = output_length > 0 ? output_length : 1;
        }

        // 填充 tiling 数据
        BatchToSpaceTilingData *tiling = context->GetTilingData<BatchToSpaceTilingData>();

        tiling->inputLength = input_length;
        tiling->outputLength = output_length;

        tiling->batch = batch;
        tiling->height = height;
        tiling->width = width;
        tiling->depth = depth;

        tiling->outBatch = out_batch;
        tiling->outHeight = out_height;
        tiling->outWidth = out_width;

        tiling->blockSize = block_size;
        tiling->cropTop = crop_top;
        tiling->cropBottom = crop_bottom;
        tiling->cropLeft = crop_left;
        tiling->cropRight = crop_right;

        tiling->blockDim = block_dim;

        // 配置启动核数
        context->SetBlockDim(block_dim);

        // 配置 workspace 大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *x_shape = context->GetInputShape(0);
        gert::Shape *y_shape = context->GetOutputShape(0);

        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const gert::TypedContinuousVector<int64_t> *attr_crops = attrs->GetListInt(0);
        const int64_t *attr_block_size = attrs->GetInt(1);

        // crops 不能用 (*attr_crops)[i] 访问，需要先取 GetData()
        const int64_t *crops_data = attr_crops->GetData();

        int64_t batch = x_shape->GetDim(0);
        int64_t height = x_shape->GetDim(1);
        int64_t width = x_shape->GetDim(2);
        int64_t depth = x_shape->GetDim(3);

        int64_t crop_top = crops_data[0];
        int64_t crop_bottom = crops_data[1];
        int64_t crop_left = crops_data[2];
        int64_t crop_right = crops_data[3];
        int64_t block_size = *attr_block_size;

        int64_t out_batch = batch / (block_size * block_size);
        int64_t out_height = height * block_size - crop_top - crop_bottom;
        int64_t out_width = width * block_size - crop_left - crop_right;

        *y_shape = {out_batch, out_height, out_width, depth};

        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class BatchToSpace : public OpDef {
    public:
        explicit BatchToSpace(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});

            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});

            this->Attr("crops").AttrType(REQUIRED).ListInt();
            this->Attr("block_size").AttrType(REQUIRED).Int();

            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };

    OP_ADD(BatchToSpace);
}  // namespace ops