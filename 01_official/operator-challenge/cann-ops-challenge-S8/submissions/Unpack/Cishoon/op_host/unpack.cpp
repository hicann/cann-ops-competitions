
#include "unpack_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t MAX_TILE_ELEMENTS = 8192;

#define CEIL(x, y)  (((x) + (y) - 1) / (y))

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    UnpackTilingData tiling;

    const gert::Shape input_shape = context->GetInputShape(0)->GetStorageShape();
    int dim_num = input_shape.GetDimNum();

    auto    ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t core_num        = ascendcPlatform.GetCoreNum();
    if (core_num < 1) core_num = 1;

    uint64_t ub_size = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    const ge::DataType data_type      = context->GetInputTensor(0)->GetDataType();
    int32_t            data_type_size = ge::GetSizeByDataType(data_type);
    if (data_type_size < 1) data_type_size = 1;

    int64_t num_attr  = *(context->GetAttrs()->GetInt(0));
    int64_t axis_attr = *(context->GetAttrs()->GetInt(1));
    if (axis_attr < 0) axis_attr += dim_num;

    uint32_t M = 1;
    for (int i = 0; i < axis_attr; i++) M *= static_cast<uint32_t>(input_shape.GetDim(i));
    uint32_t K = static_cast<uint32_t>(input_shape.GetDim(axis_attr));
    uint32_t L = 1;
    for (int i = axis_attr + 1; i < dim_num; i++) L *= static_cast<uint32_t>(input_shape.GetDim(i));

    if (K == 0) K = static_cast<uint32_t>(num_attr);

    uint32_t block_n = static_cast<uint32_t>(BLOCK_SIZE) / static_cast<uint32_t>(data_type_size);
    if (block_n == 0) block_n = 1;

    uint32_t tile_length;
    uint32_t iter_per_row;
    uint32_t real_residue;
    uint32_t pad_residue;

    if (L <= static_cast<uint32_t>(MAX_TILE_ELEMENTS)) {
        tile_length  = CEIL(L, block_n) * block_n;
        if (tile_length == 0) tile_length = block_n;
        iter_per_row = 1;
        real_residue = L;
        pad_residue  = tile_length;
    } else {
        tile_length  = static_cast<uint32_t>(MAX_TILE_ELEMENTS);
        iter_per_row = CEIL(L, tile_length);
        uint32_t last = L - (iter_per_row - 1) * tile_length;
        real_residue  = last;
        pad_residue   = CEIL(last, block_n) * block_n;
    }

    uint64_t total_iter64 = static_cast<uint64_t>(M) * static_cast<uint64_t>(K) *
                            static_cast<uint64_t>(iter_per_row);
    uint32_t iterations   = static_cast<uint32_t>(total_iter64);

    // Pack paths: scatter has O(M*K) DMAs with per-DMA overhead; pack collapses
    // that to O(M/tile_m * K). Two flavors depending on L*ts (cache line util):
    //
    //   mode=1 strided pack (l_bytes ≥ 32):
    //     DMA-in reads column k as [tile_m × L] strided block; DMA-out writes
    //     contiguous tile row. blockLen ≥ 32 B = full cache line, 100% HBM util.
    //
    //   mode=2 gather pack (l_bytes < 32):
    //     strided reads with blockLen < 32 B waste cache lines (6-50% util)
    //     and HBM controller chokes on thousands of tiny strided requests.
    //     Switch to contiguous DMA-in of [tile_m × K × L], UB Gather to extract
    //     column k into a contiguous buffer, contiguous DMA-out. Pays extra UB
    //     (K× the packed data) but gets 100% HBM util and clean pipeline.
    uint32_t l_bytes     = L * static_cast<uint32_t>(data_type_size);
    bool     use_pack    = (L <= 2048);
    bool     use_gather  = use_pack && (l_bytes < 32);
    uint32_t pack_mode   = use_gather ? 2 : (use_pack ? 1 : 0);
    uint32_t pack_tile_m = 1;
    if (use_pack) {
        uint32_t bn = static_cast<uint32_t>(32) / static_cast<uint32_t>(data_type_size);
        if (bn == 0) bn = 1;
        // 3 regions × up to 31 B of 32-B alignment rounding = 96 B; -128 covers it.
        int64_t ub_budget = static_cast<int64_t>(ub_size) - 128;
        if (ub_budget < 0) ub_budget = 0;

        int64_t max_tile_m;
        if (use_gather) {
            // UB slots per m row:
            //   packed:  K * L * ts              (source for Gather)
            //   col:     L * ts                  (Gather output, contiguous across rows)
            //   offset:  L * 4                   (one uint32 per gathered element)
            int64_t  packed_per_m   = static_cast<int64_t>(K) * L * data_type_size;
            int64_t  col_per_m      = static_cast<int64_t>(L) * data_type_size;
            int64_t  offset_per_m   = static_cast<int64_t>(L) * 4;
            int64_t  per_m          = packed_per_m + col_per_m + offset_per_m;
            if (per_m < 1) per_m = 1;
            max_tile_m = ub_budget / per_m;
        } else {
            // Strided pack: UB row pitch = padded_L * ts bytes.
            uint32_t padded_l       = ((L + bn - 1) / bn) * bn;
            int64_t  padded_l_bytes = static_cast<int64_t>(padded_l) * data_type_size;
            if (padded_l_bytes < 32) padded_l_bytes = 32;
            max_tile_m = ub_budget / padded_l_bytes;
        }

        if (max_tile_m > 4095) max_tile_m = 4095;  // DataCopyExtParams blockCount limit
        // Strided pack with blockLen<32B hits HBM bad path if blockCount huge;
        // gather pack uses 1-block contiguous DMAs so this clamp doesn't apply.
        if (!use_gather && l_bytes < 32 && max_tile_m > 256) max_tile_m = 256;
        if (max_tile_m < 1) max_tile_m = 1;
        pack_tile_m = static_cast<uint32_t>(std::min(static_cast<int64_t>(M), max_tile_m));
    }

    uint32_t used_core_num = static_cast<uint32_t>(core_num);
    if (use_pack) {
        // Dispatch by M: each core owns a disjoint m-slice, no L2 fan-out.
        used_core_num = std::min(static_cast<uint32_t>(core_num), M);
    } else {
        if (used_core_num > iterations) used_core_num = iterations > 0 ? iterations : 1;
    }
    if (used_core_num == 0) used_core_num = 1;

    tiling.set_M(M);
    tiling.set_K(K);
    tiling.set_L(L);
    tiling.set_tile_length(tile_length);
    tiling.set_iter_per_row(iter_per_row);
    tiling.set_real_residue(real_residue);
    tiling.set_pad_residue(pad_residue);
    tiling.set_iterations(iterations);
    tiling.set_pack_mode(pack_mode);
    tiling.set_pack_tile_m(pack_tile_m);

    context->SetBlockDim(used_core_num);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x1_shape = context->GetInputShape(0);
    int dim_num = x1_shape->GetDimNum();

    const int64_t* num_ptr  = context->GetAttrs()->GetInt(0);
    const int64_t* axis_ptr = context->GetAttrs()->GetInt(1);
    int64_t num  = *num_ptr;
    int64_t axis = *axis_ptr;
    if (axis < 0) axis += dim_num;

    for (int64_t i = 0; i < num; i++) {
        gert::Shape* y_shape = context->GetOutputShape(i);
        if (y_shape == nullptr) continue;
        y_shape->SetDimNum(dim_num - 1);
        int k = 0;
        for (int j = 0; j < dim_num; j++) {
            if (j == axis) continue;
            y_shape->SetDim(k++, x1_shape->GetDim(j));
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    const auto inputDataType = context->GetInputDataType(0);
    const int64_t* num_ptr = context->GetAttrs()->GetInt(0);
    int64_t num = *num_ptr;
    for (int64_t i = 0; i < num; i++) {
        context->SetOutputDataType(i, inputDataType);
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace ge


namespace ops {
class Unpack : public OpDef {
public:
    explicit Unpack(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("num").Int();
        this->Attr("axis").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Unpack);
}
