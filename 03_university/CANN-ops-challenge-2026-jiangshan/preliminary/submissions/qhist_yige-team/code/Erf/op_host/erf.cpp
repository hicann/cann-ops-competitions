// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        // 获取算子输入信息
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t totalLength = tensor_x->GetShapeSize();

        // 配置tiling key
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // 多核切分: 每核基础处理元素个数
        // 小数据量优化: 数据量较小时减少核数，避免多核启动开销
        // v12: MIN_ELEMENTS_PER_CORE 64 → 256
        // v14: 256 → 512
        // v16: 两阶段 MIN，按 totalLength 分流——
        //   小/中等小数据走 MIN=512（少核降 launch 开销，针对测试点 5/8）
        //   中等大数据走 MIN=256（保留更多核，针对测试点 11/2）
        uint32_t blockDim = static_cast<uint32_t>(num_cores_aiv);
        if (blockDim == 0) {
            blockDim = 1;
        }
        constexpr uint32_t MIN_SMALL = 512;
        constexpr uint32_t MIN_LARGE = 256;
        constexpr uint32_t MIN_SWITCH_THRESHOLD = 3000;
        uint32_t min_per_core = (totalLength <= MIN_SWITCH_THRESHOLD) ? MIN_SMALL : MIN_LARGE;
        if (totalLength < blockDim * min_per_core) {
            blockDim = (totalLength + min_per_core - 1) / min_per_core;
            if (blockDim == 0) {
                blockDim = 1;
            }
        }
        uint32_t blockLength = totalLength / blockDim;
        if (blockLength == 0) {
            blockLength = totalLength;
            blockDim = 1;
        }

        // UB切分: 根据UB容量确定tile大小
        // Buffer需求: inQueueX(2) + outQueueY(2) = 4个buffer (双缓冲)
        // 预留20%余量，确保不溢出
        uint32_t bufferCount = 4;
        uint32_t dtypeSize = sizeof(float);
        uint32_t maxTileBytes = static_cast<uint32_t>((ub_size * 0.8) / bufferCount);
        uint32_t maxTileLen = maxTileBytes / dtypeSize;
        // 对齐到32字节(8个float32)
        uint32_t tileLength = (maxTileLen / 8) * 8;
        // 限制范围: 最小8个元素(32字节对齐), 最大8192个元素
        if (tileLength > 8192) {
            tileLength = 8192;
        }
        if (tileLength < 8) {
            tileLength = 8;
        }
        // 小数据量优化: 如果总元素数小于tileLength，将tileLength缩减为totalLength（对齐）
        // 减少InitBuffer分配的内存，降低小数据量场景的overhead
        if (totalLength < tileLength) {
            tileLength = ((totalLength + 7) / 8) * 8;
            if (tileLength < 8) {
                tileLength = 8;
            }
        }

        // 填充tiling结构体（精简为2字段）
        ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
        tiling->totalLength = totalLength;
        tiling->tileLength = tileLength;

        // 配置启动核数
        context->SetBlockDim(blockDim);
        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *xShape = context->GetInputShape(0);
        gert::Shape *yShape = context->GetOutputShape(0);
        *yShape = *xShape;
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
