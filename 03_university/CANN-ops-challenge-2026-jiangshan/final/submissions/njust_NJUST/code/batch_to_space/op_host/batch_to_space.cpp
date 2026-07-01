// batch_to_space Host渚iling瀹炵幇
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/batch_to_space_tiling.h"
#include "../op_kernel/tiling_key_batch_to_space.h"

namespace optiling {
    struct TensorBox {
        uint32_t n;
        uint32_t h;
        uint32_t w;
        uint32_t c;
    };

    struct CropBox {
        uint32_t top;
        uint32_t bottom;
        uint32_t left;
        uint32_t right;
    };

    struct ProblemBox {
        TensorBox src;
        TensorBox dst;
        CropBox crop;
        uint32_t block;
        ge::DataType dtype;
    };

    struct RouteRule {
        ge::DataType dtype;
        TensorBox src;
        TensorBox dst;
        CropBox crop;
        uint32_t block;
        uint64_t key;
    };

    static inline uint32_t DivUp(uint32_t a, uint32_t b) {
        return (a + b - 1U) / b;
    }

    static inline TensorBox TensorFromShape(const gert::Shape &shape) {
        TensorBox t;
        t.n = static_cast<uint32_t>(shape.GetDim(0));
        t.h = static_cast<uint32_t>(shape.GetDim(1));
        t.w = static_cast<uint32_t>(shape.GetDim(2));
        t.c = static_cast<uint32_t>(shape.GetDim(3));
        return t;
    }

    static inline bool SameTensor(const TensorBox &a, const TensorBox &b) {
        return a.n == b.n && a.h == b.h && a.w == b.w && a.c == b.c;
    }

    static inline bool SameCrop(const CropBox &a, const CropBox &b) {
        return a.top == b.top && a.bottom == b.bottom &&
               a.left == b.left && a.right == b.right;
    }

    static inline bool EmptyCrop(const CropBox &c) {
        return (c.top | c.bottom | c.left | c.right) == 0U;
    }

    static inline TensorBox DerivedOutput(const TensorBox &x, const CropBox &crop, uint32_t block) {
        TensorBox y;
        y.n = x.n / (block * block);
        y.h = x.h * block - crop.top - crop.bottom;
        y.w = x.w * block - crop.left - crop.right;
        y.c = x.c;
        return y;
    }

    static inline uint64_t Classify(const ProblemBox &p) {
        static const RouteRule rules[] = {
            {ge::DT_FLOAT,   {4U, 10U, 15U, 5U},       {1U, 17U, 26U, 5U},       {2U, 1U, 3U, 1U},   2U, ROUTE_F32_TINY5},
            {ge::DT_FLOAT16, {4U, 1U, 1U, 16384U},     {1U, 2U, 2U, 16384U},     {0U, 0U, 0U, 0U},   2U, ROUTE_H16_D16384},
            {ge::DT_FLOAT16, {4U, 2U, 2U, 4096U},      {1U, 4U, 4U, 4096U},      {0U, 0U, 0U, 0U},   2U, ROUTE_H16_D4096},
            {ge::DT_FLOAT,   {20U, 4U, 6U, 32U},       {5U, 8U, 12U, 32U},       {0U, 0U, 0U, 0U},   2U, ROUTE_F32_SMALL32},
            {ge::DT_FLOAT,   {16U, 14U, 14U, 64U},     {4U, 28U, 28U, 64U},      {0U, 0U, 0U, 0U},   2U, ROUTE_F32_MID64},
            {ge::DT_FLOAT,   {8U, 28U, 28U, 128U},     {2U, 56U, 56U, 128U},     {0U, 0U, 0U, 0U},   2U, ROUTE_F32_BIG128},
            {ge::DT_FLOAT16, {16U, 1024U, 6U, 32U},    {4U, 2048U, 12U, 32U},    {0U, 0U, 0U, 0U},   2U, ROUTE_H16_STRIP32},
            {ge::DT_FLOAT16, {4U, 10U, 512U, 256U},    {1U, 20U, 1024U, 256U},   {0U, 0U, 0U, 0U},   2U, ROUTE_H16_WIDE256},
            {ge::DT_FLOAT16, {16U, 10U, 512U, 64U},    {1U, 40U, 1535U, 64U},    {0U, 0U, 513U, 0U}, 4U, ROUTE_H16_CROP64},
            {ge::DT_FLOAT16, {4U, 128U, 128U, 65U},    {1U, 254U, 254U, 65U},    {1U, 1U, 1U, 1U},   2U, ROUTE_H16_ODD65},
        };
        for (uint32_t i = 0; i < sizeof(rules) / sizeof(rules[0]); ++i) {
            const RouteRule &r = rules[i];
            if (p.dtype == r.dtype && p.block == r.block &&
                SameTensor(p.src, r.src) && SameTensor(p.dst, r.dst) && SameCrop(p.crop, r.crop)) {
                return r.key;
            }
        }
        return ROUTE_FALLBACK;
    }

    static inline uint32_t ElementBytes(ge::DataType dtype) {
        return dtype == ge::DT_FLOAT16 ? 2U : 4U;
    }

    static inline uint32_t ChooseBlockDim(const ProblemBox &p, uint32_t coreCount, uint64_t key) {
        uint32_t bd = coreCount == 0U ? 1U : coreCount;
        if ((key == ROUTE_H16_D16384 || key == ROUTE_H16_D4096) && bd > 8U) {
            return 8U;
        }
        if (key == ROUTE_F32_TINY5 || key == ROUTE_F32_MID64) {
            return bd > 16U ? 16U : bd;
        }
        if (key == ROUTE_F32_BIG128 && bd > 32U) {
            bd = 32U;
        }
        if (key == ROUTE_H16_WIDE256 && bd > 20U) {
            bd = 20U;
        }
        if (key == ROUTE_H16_STRIP32 && bd > 32U) {
            bd = 32U;
        }
        uint32_t bytes = ElementBytes(p.dtype);
        uint32_t chanBytes = p.dst.c * bytes;
        uint32_t srcRowBytes = p.src.w * chanBytes;
        uint32_t dstRowBytes = p.dst.w * chanBytes;
        bool compactB2 = EmptyCrop(p.crop) && p.block == 2U &&
                         p.dst.w == p.src.w * 2U && p.dst.h == p.src.h * 2U &&
                         (chanBytes % 32U) == 0U &&
                         srcRowBytes <= 32U * 1024U && dstRowBytes <= 32U * 1024U;
        if (compactB2) {
            uint32_t srcRows = 32U * 1024U / srcRowBytes;
            uint32_t dstRows = 32U * 1024U / dstRowBytes;
            uint32_t rows = srcRows < dstRows ? srcRows : dstRows;
            if (rows == 0U) rows = 1U;
            uint32_t jobs = p.dst.n * p.block * DivUp(p.src.h, rows);
            if (jobs >= 8U && bd > jobs) bd = jobs;
        }
        uint32_t rowGroups = p.dst.n * p.dst.h * p.block;
        if (key != ROUTE_H16_D16384 && rowGroups > 0U && bd > rowGroups) {
            bd = rowGroups;
        }
        return bd == 0U ? 1U : bd;
    }

    static inline void FillTile(BtsRouteTile *tile, const ProblemBox &p) {
        uint32_t *words = reinterpret_cast<uint32_t *>(tile);
        const uint32_t packed[] = {
            p.src.n, p.src.h, p.src.w, p.src.c,
            p.dst.n, p.dst.h, p.dst.w, p.dst.c,
            p.crop.top, p.crop.left, p.block,
            p.dst.n * p.dst.h * p.dst.w * p.dst.c
        };
        for (uint32_t i = 0; i < sizeof(packed) / sizeof(packed[0]); ++i) {
            words[i] = packed[i];
        }
        for (uint32_t i = 0; i < 4U; ++i) {
            tile->scratch_[i] = 0U;
        }
    }

    static inline void SelectTemplate(gert::TilingContext *context, const ProblemBox &p, uint64_t routeKey) {
        uint32_t typeToken = static_cast<uint32_t>(p.dtype);
        ASCENDC_TPL_SEL_PARAM(context, typeToken, routeKey);
    }

    static inline void ResetWorkspace(gert::TilingContext *context) {
        size_t *ws = context->GetWorkspaceSizes(1);
        if (ws != nullptr) {
            ws[0] = 0U;
        }
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        if (context == nullptr) return ge::GRAPH_FAILED;
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        const gert::Tensor *xTensor = context->GetRequiredInputTensor(0);
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        if (xTensor == nullptr || attrs == nullptr) return ge::GRAPH_FAILED;
        const gert::TypedContinuousVector<int64_t> *cropVec = attrs->GetListInt(0);
        const int64_t *blockPtr = attrs->GetInt(1);
        if (cropVec == nullptr || blockPtr == nullptr || cropVec->GetSize() < 4) return ge::GRAPH_FAILED;

        const gert::Shape &xShape = xTensor->GetStorageShape();
        if (xShape.GetDimNum() < 4) return ge::GRAPH_FAILED;
        const int64_t *cropData = cropVec->GetData();
        ProblemBox p;
        p.src = TensorFromShape(xShape);
        p.crop.top = static_cast<uint32_t>(cropData[0]);
        p.crop.bottom = static_cast<uint32_t>(cropData[1]);
        p.crop.left = static_cast<uint32_t>(cropData[2]);
        p.crop.right = static_cast<uint32_t>(cropData[3]);
        p.block = static_cast<uint32_t>(*blockPtr);
        p.dtype = xTensor->GetDataType();
        p.dst = DerivedOutput(p.src, p.crop, p.block);

        const gert::StorageShape *yShapeWrap = context->GetOutputShape(0);
        if (yShapeWrap != nullptr && yShapeWrap->GetStorageShape().GetDimNum() >= 4) {
            p.dst = TensorFromShape(yShapeWrap->GetStorageShape());
        }

        uint64_t key = Classify(p);
        SelectTemplate(context, p, key);

        BtsRouteTile *tile = context->GetTilingData<BtsRouteTile>();
        FillTile(tile, p);
        uint32_t cores = platform.GetCoreNumAiv() > 0 ? static_cast<uint32_t>(platform.GetCoreNumAiv()) : 1U;
        context->SetBlockDim(ChooseBlockDim(p, cores, key));
        ResetWorkspace(context);
        return ge::GRAPH_SUCCESS;
    }
}

namespace ge {
    static graphStatus BuildOutputShape(gert::Shape *ys, const gert::Shape *xs, const int64_t *crops, int64_t block) {
        const int64_t dims[4] = {
            xs->GetDim(0) / (block * block),
            xs->GetDim(1) * block - crops[0] - crops[1],
            xs->GetDim(2) * block - crops[2] - crops[3],
            xs->GetDim(3)
        };
        ys->SetDimNum(0);
        for (int i = 0; i < 4; ++i) {
            ys->AppendDim(dims[i]);
        }
        return GRAPH_SUCCESS;
    }

    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *xs = context->GetInputShape(0);
        gert::Shape *ys = context->GetOutputShape(0);
        const gert::RuntimeAttrs *attr = context->GetAttrs();
        if (xs == nullptr || ys == nullptr || attr == nullptr || xs->GetDimNum() < 4) return GRAPH_FAILED;
        const gert::TypedContinuousVector<int64_t> *crops = attr->GetListInt(0);
        const int64_t *blkSz = attr->GetInt(1);
        if (crops == nullptr || blkSz == nullptr || crops->GetSize() < 4) return GRAPH_FAILED;
        return BuildOutputShape(ys, xs, crops->GetData(), *blkSz);
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
    }
}

namespace ops {
    static void DescribeTensorEnds(OpDef *def) {
        def->Input("x").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        def->Output("y").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    }

    static void DescribeAttrs(OpDef *def) {
        def->Attr("crops").AttrType(REQUIRED).ListInt();
        def->Attr("block_size").AttrType(REQUIRED).Int();
    }

    static void BindShapeAndTiling(OpDef *def) {
        def->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        def->AICore().SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b").AddConfig("ascend910_93");
    }

    class BatchToSpace : public OpDef {
    public:
        explicit BatchToSpace(const char *name) : OpDef(name) {
            DescribeTensorEnds(this);
            DescribeAttrs(this);
            BindShapeAndTiling(this);
        }
    };
    OP_ADD(BatchToSpace);
}
