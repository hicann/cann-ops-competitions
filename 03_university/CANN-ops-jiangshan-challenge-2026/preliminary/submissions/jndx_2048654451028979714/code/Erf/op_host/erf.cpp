// Host侧Tiling实现
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"



namespace optiling {
    constexpr uint32_t TILE_MODE_RATIONAL = 0x80000000U;
    constexpr uint32_t TILE_MODE_FAST_TBUF = 0x40000000U;

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        int dtype_size_x = ge::GetSizeByDataType(dtype_x);
        uint32_t length_x = tensor_x->GetShapeSize();

        uint32_t DT_X = static_cast<uint32_t>(dtype_x);

        ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();

        // 32B 对齐的元素数: float32 -> 8
        const uint32_t ALIGN_NUM = 32 / dtype_size_x;

        // ===== UB 容量与 tile 长度 =====
        // 大张量会走 TQue 路径: BUFFER_NUM*2 个数据队列 + 2 份 tmp = 6 份 tile
        // 预留 8KB 作为系统/对齐余量
        uint32_t max_tile_bytes = (ub_size > 8192) ? (ub_size - 8192) : 0;
        uint32_t max_tile_len   = (max_tile_bytes / 6) / dtype_size_x;
        max_tile_len = max_tile_len / ALIGN_NUM * ALIGN_NUM;
        if (max_tile_len == 0) max_tile_len = ALIGN_NUM;

        const uint32_t TILE_LEN_PREFER = 8192;
        uint32_t tileLen = TILE_LEN_PREFER < max_tile_len ? TILE_LEN_PREFER : max_tile_len;

        if (length_x == 0) {
            ASCENDC_TPL_SEL_PARAM(context, DT_X, ERF_TPL_MODE_BASE);
            tiling->totalLength = 0;
            tiling->coreLength  = ALIGN_NUM;
            tiling->tileLen     = tileLen;
            context->SetBlockDim(1);
            size_t *currentWorkspace = context->GetWorkspaceSizes(1);
            currentWorkspace[0] = 0;
            return ge::GRAPH_SUCCESS;
        }

        // ===== 分核策略 =====
        // 每个核固定开销不可忽略, 数据量太小时不应启动过多核
        // 给每个非尾核至少分配 MIN_PER_CORE 个元素
        const uint32_t MIN_PER_CORE = ALIGN_NUM * 4;  // 32 个 float

        uint32_t coreLen = (length_x + num_cores_aiv - 1) / num_cores_aiv;
        if (coreLen < MIN_PER_CORE) coreLen = MIN_PER_CORE;
        coreLen = (coreLen + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;

        uint32_t used_cores = (length_x + coreLen - 1) / coreLen;
        if (used_cores > (uint32_t)num_cores_aiv) {
            used_cores = num_cores_aiv;
            coreLen = (length_x + used_cores - 1) / used_cores;
            coreLen = (coreLen + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
            used_cores = (length_x + coreLen - 1) / coreLen;
        }

        const size_t dimNum = tensor_x->GetStorageShape().GetDimNum();
        const bool len1Scalar = (dimNum == 1) && (length_x == 1);
        const bool caseFastPoly = false;
        const bool len128Rational = (dimNum == 1) && (length_x == 128);
        uint32_t mode = ERF_TPL_MODE_BASE;
        if (len1Scalar) {
            mode = ERF_TPL_MODE_LEN1_SCALAR;
        } else if (len128Rational) {
            mode = ERF_TPL_MODE_LEN128_RATIONAL;
        } else if (caseFastPoly) {
            mode = ERF_TPL_MODE_CASE_FAST_POLY;
        }
        ASCENDC_TPL_SEL_PARAM(context, DT_X, static_cast<uint64_t>(mode));

        // Single-core only for tiny dim<=2 cases. Larger dim<=2 cases keep multi-core split.
        if (dimNum <= 2 && used_cores > 1) {
            const bool tinyZone = (length_x <= 256);
            if (tinyZone) {
                used_cores = 1;
                coreLen = (length_x + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
            }
        }

        if (dimNum == 2 && length_x > 16384 && used_cores > 1) {
            if (length_x > 262144 && length_x <= 524288) {
                used_cores = (used_cores * 3 + 3) / 4;
            } else {
                used_cores = (used_cores + 1) / 2;
            }
            coreLen = (length_x + used_cores - 1) / used_cores;
            coreLen = (coreLen + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
            used_cores = (length_x + coreLen - 1) / coreLen;
        }

        // tile 不超过单核数据量
        if (tileLen > coreLen) {
            tileLen = (coreLen + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
        }

        tiling->totalLength = length_x;
        tiling->coreLength  = coreLen;

        const bool smallRational = (length_x <= 2048) && (dimNum >= 3);
        const bool fastDim2 = (dimNum <= 2) && (length_x <= 256);
        const bool fastTbuf = (length_x <= 8192) &&
                              ((dimNum >= 4) || fastDim2 || ((length_x <= 2048) && (length_x % ALIGN_NUM != 0)));
        tiling->tileLen = tileLen |
                          (smallRational ? TILE_MODE_RATIONAL : 0U) |
                          (fastTbuf ? TILE_MODE_FAST_TBUF : 0U);

        context->SetBlockDim(used_cores);

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
