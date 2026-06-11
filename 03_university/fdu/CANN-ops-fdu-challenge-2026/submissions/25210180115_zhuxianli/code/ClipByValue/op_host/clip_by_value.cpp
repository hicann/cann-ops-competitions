// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstdlib>

#include "../op_kernel/clip_by_value_tiling.h"
#include "../op_kernel/tiling_key_clip_by_value.h"

// 核数策略编译期默认: 0 = 分段满核(<256→1,<1024→ceil/256,≥1024→满核);
// 非0 = 分级 ceil(length/minPerCore)(中等张量少核)。变体改此默认做 A/B。
#ifndef FDU_CLIP_MIN_PER_CORE_DEFAULT
#define FDU_CLIP_MIN_PER_CORE_DEFAULT 0u
#endif

namespace optiling {
    static uint32_t ReadEnvU32(const char *name, uint32_t fallback) {
        const char *value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') {
            return fallback;
        }
        char *end = nullptr;
        unsigned long parsed = std::strtoul(value, &end, 10);
        if (end == value || parsed == 0) {
            return fallback;
        }
        return static_cast<uint32_t>(parsed);
    }

    static uint32_t AlignDown(uint32_t value, uint32_t align) {
        if (align == 0) {
            return value;
        }
        uint32_t aligned = value / align * align;
        return aligned == 0 ? align : aligned;
    }

    // 内存受限 elementwise 的核数阶梯。断点由 910B3 实测 msprof Task Duration 扫参得到(见 bench/)。
    //  - 小/中张量受固定开销(每核启动 + InitBuffer + 同步)主导, 多核反而更慢:
    //    旧默认"≥1024 即满核"使 1万元素点 ~2.8×、10万点 ~1.7×、26万点 ~1.45× 慢。
    //  - 大张量受 GM 带宽主导, 核数随规模上升; 但实测带宽在 ~40 核饱和,
    //    48 核因尾核调度反而更慢(10M/100M 两点 40 核均优于 48), 故封顶 40 核。
    //  实测最优(核数): 1万→2, 10万→8, 26万→12, 100万→24, 1000万→40, 1亿→40。
    static uint32_t ChooseBlockDim(uint32_t length, uint32_t coresMax) {
        uint32_t bigCap = coresMax < 40u ? coresMax : 40u;
        uint32_t bd;
        if (length <= 8192u)          bd = 1u;     // 256 / 1K / 4K
        else if (length <= 16384u)    bd = 2u;     // ~1万
        else if (length <= 49152u)    bd = 4u;
        else if (length <= 131072u)   bd = 8u;     // 6.5万 / 10万
        else if (length <= 393216u)   bd = 12u;    // ~26万
        else if (length <= 786432u)   bd = 16u;    // ~50万
        else if (length <= 1572864u)  bd = 24u;    // ~100万
        else if (length <= 3145728u)  bd = 32u;    // ~200万
        else                          bd = bigCap; // 400万+ / 1000万 / 1亿
        if (bd > coresMax) bd = coresMax;
        return bd;
    }

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
        // 示例: 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        // 示例: 计算tiling方案并填充tiling结构体
        ClipByValueTilingData *tiling = context->GetTilingData<ClipByValueTilingData>();
        tiling->length = length_x;
        tiling->minValue = attr_min == nullptr ? 0.0f : *attr_min;
        tiling->maxValue = attr_max == nullptr ? 0.0f : *attr_max;

        // DataCopy API 合法对齐粒度 32B; 910B 上 GM<->UB 搬运按 512B 对齐才稳妥。
        uint32_t alignElems = 32u / static_cast<uint32_t>(dtype_size_x);
        if (alignElems == 0) {
            alignElems = 1;
        }
        tiling->alignElems = alignElems;
        uint32_t gmPreferredElems = 512u / static_cast<uint32_t>(dtype_size_x);
        if (gmPreferredElems < alignElems) {
            gmPreferredElems = alignElems;
        }

        uint32_t coresMax = static_cast<uint32_t>(num_cores_aiv);
        if (coresMax == 0) {
            coresMax = 1;
        }

        if (length_x == 0) {
            tiling->blockDim = 1;
            tiling->lengthPerCore = 0;
            tiling->tileLength = alignElems;
            context->SetBlockDim(1);
            size_t *ws0 = context->GetWorkspaceSizes(1);
            ws0[0] = 0;
            return ge::GRAPH_SUCCESS;
        }

        // 核数: 按规模阶梯选核(小张量少核摊薄启动开销, 大张量多核吃满 GM 带宽并封顶 40)。
        // FDU_CLIP_MIN_PER_CORE 非0时改用分级 ceil(length/minPerCore)(本地A/B)。
        uint32_t maxBlockDim = ChooseBlockDim(length_x, coresMax);
        if (maxBlockDim > length_x) maxBlockDim = length_x;
        uint32_t blockDim = maxBlockDim;
        uint32_t minPerCore = ReadEnvU32("FDU_CLIP_MIN_PER_CORE", FDU_CLIP_MIN_PER_CORE_DEFAULT);
        if (minPerCore != 0u) {
            blockDim = (length_x + minPerCore - 1u) / minPerCore;
        }
        blockDim = ReadEnvU32("FDU_CLIP_BLOCK_DIM", blockDim);
        if (blockDim > coresMax) blockDim = coresMax;
        if (blockDim == 0) blockDim = 1;
        if (blockDim > length_x) blockDim = length_x;

        // 每核元素数向上对齐到 512B, 保证每核 GM 起始 512B 对齐(对齐 DataCopy 多核安全)。
        uint32_t perCore = (length_x + blockDim - 1u) / blockDim;
        perCore = ((perCore + gmPreferredElems - 1u) / gmPreferredElems) * gmPreferredElems;
        if (perCore == 0) perCore = gmPreferredElems;
        uint32_t effectiveBlockDim = (length_x + perCore - 1u) / perCore;
        if (effectiveBlockDim == 0) effectiveBlockDim = 1;
        tiling->blockDim = effectiveBlockDim;
        tiling->lengthPerCore = perCore;

        // 两类 UB buffer(x/y)双缓冲。tile 上限受 UB 限制, 512B 对齐。
        uint32_t defaultTile = AlignDown(static_cast<uint32_t>(ub_size / (2 * 2 * dtype_size_x)), gmPreferredElems);
        if (defaultTile > 32768) {
            defaultTile = 32768;
        }
        if (defaultTile == 0) {
            defaultTile = gmPreferredElems;
        }
        uint32_t adaptiveTile = perCore < defaultTile ? perCore : defaultTile;
        adaptiveTile = ((adaptiveTile + gmPreferredElems - 1u) / gmPreferredElems) * gmPreferredElems;
        if (adaptiveTile > defaultTile) adaptiveTile = defaultTile;
        if (adaptiveTile == 0) adaptiveTile = gmPreferredElems;
        tiling->tileLength = ReadEnvU32("FDU_CLIP_TILE_LENGTH", adaptiveTile);
        context->SetBlockDim(effectiveBlockDim);
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
