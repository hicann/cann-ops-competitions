// Profile-only host dispatcher for the BatchToSpace template kernel.
# include "register/op_def_registry.h"
# include "tiling/platform/platform_ascendc.h"

# include "../op_kernel/batch_to_space_tiling.h"
# include "../op_kernel/tiling_key_batch_to_space.h"

namespace optiling {  // profile dispatch
    static constexpr uint64_t kHostUint32Max = 0xffffffffULL;

    static ge::graphStatus RejectTiling() {
        return ge::GRAPH_FAILED;
    }

    static bool MulWouldOverflow32(uint64_t lhs, uint64_t rhs, uint64_t cap) {
        return lhs != 0U && rhs > cap / lhs;
    }

    static bool Shape4WouldOverflow32(uint64_t d0, uint64_t d1,
                                      uint64_t d2, uint64_t d3, uint64_t cap) {
        if (MulWouldOverflow32(d0, d1, cap)) { return (true); }
        uint64_t d01 = d0 * d1;
        if (MulWouldOverflow32(d01, d2, cap)) { return (true); }
        return MulWouldOverflow32(d01 * d2, d3, cap);
    }

    struct ProfileShape final {
        uint32_t srcN;
        uint32_t srcH;
        uint32_t srcW;
        uint32_t srcC;
        uint32_t dstN;
        uint32_t dstH;
        uint32_t dstW;
        uint32_t dstC;
        uint32_t scale;
        uint32_t cutTop;
        uint32_t cutBottom;
        uint32_t cutLeft;
        uint32_t cutRight;

        bool KeepsFullFrame() const {
            return cutTop + cutBottom + cutLeft + cutRight == 0U;
        }

        bool R2Full(uint32_t h, uint32_t w, uint32_t c, uint32_t n) const {
            return scale == 2U && KeepsFullFrame() && srcH == h && srcW == w &&
                   srcC == c && dstN == n && dstH == h * 2U &&
                   dstW == w * 2U && dstC == c && srcN == n * 4U;
        }

        bool Crops(uint32_t top, uint32_t bottom, uint32_t left, uint32_t right) const {
            return cutTop == top && cutBottom == bottom &&
                   cutLeft == left && cutRight == right;
        }
    };

    static uint64_t SelectProfileKey(const ProfileShape &fp, ge::DataType dtype) {
        if (ge::DT_FLOAT == dtype) {
            if (fp.R2Full(28U, 28U, 128U, 2U)) {
                return BTS_TILE_F32_R2_N2_28X28_C128;
            }
            if (fp.scale == 2U && fp.srcN == 4U && fp.srcH == 10U &&
                fp.srcW == 15U && fp.srcC == 5U && fp.dstN == 1U &&
                fp.dstH == 17U && fp.dstW == 26U && fp.dstC == 5U &&
                fp.Crops(2U, 1U, 3U, 1U)) {
                return BTS_TILE_F32_R2_CROP_17X26_C5;
            }
            if (fp.R2Full(14U, 14U, 64U, 4U)) {
                return BTS_TILE_F32_R2_N4_14X14_C64;
            }
            if (fp.R2Full(4U, 6U, 32U, 5U)) {
                return BTS_TILE_F32_R2_N5_4X6_C32;
            }
            return 0U;
        }

        if (!(dtype == ge::DT_FLOAT16)) {
            return 0U;
        }
        if (fp.scale == 2U && fp.srcN == 4U && fp.srcH == 128U &&
            fp.srcW == 128U && fp.srcC == 65U && fp.dstN == 1U &&
            fp.dstH == 254U && fp.dstW == 254U && fp.dstC == 65U &&
            fp.Crops(1U, 1U, 1U, 1U)) {
            return BTS_TILE_F16_R2_CROP_254X254_C65;
        }
        if (fp.R2Full(2U, 2U, 4096U, 1U)) {
            return BTS_TILE_F16_R2_N1_2X2_C4096;
        }
        if (fp.R2Full(1U, 1U, 16384U, 1U)) {
            return BTS_TILE_F16_R2_FLAT_C16384;
        }
        if (fp.R2Full(10U, 512U, 256U, 1U)) {
            return BTS_TILE_F16_R2_N1_10X512_C256;
        }
        if (fp.scale == 4U && fp.srcN == 16U && fp.srcH == 10U &&
            fp.srcW == 512U && fp.srcC == 64U && fp.dstN == 1U &&
            fp.dstH == 40U && fp.dstW == 1535U && fp.dstC == 64U &&
            fp.Crops(0U, 0U, 513U, 0U)) {
            return BTS_TILE_F16_R4_LEFT513_C64;
        }
        if (fp.R2Full(1024U, 6U, 32U, 4U)) {
            return BTS_TILE_F16_R2_N4_1024X6_C32;
        }
        return 0U;
    }

    static uint32_t ClampNonZero(uint32_t lhs, uint32_t rhs) {
        if (lhs == 0U) { return rhs == 0U ? 1U : rhs; }
        if (rhs == 0U) { return lhs; }
        return (lhs < rhs) ? lhs : rhs;
    }

    static uint32_t ResolveAivLaunch(uint64_t tile, uint32_t available) {
        uint32_t cores = available == 0U ? 1U : available;
        switch (tile) {
            case BTS_TILE_F32_R2_CROP_17X26_C5:
            case BTS_TILE_F32_R2_N4_14X14_C64:
                return ClampNonZero(cores, 16U);
            case BTS_TILE_F16_R2_N1_2X2_C4096:
            case BTS_TILE_F16_R2_FLAT_C16384:
                return ClampNonZero(cores, 8U);
            case BTS_TILE_F16_R2_N1_10X512_C256:
                return (40U);
            default: return cores;
        }
    }

    static ge::graphStatus TilingFunc(gert::TilingContext * const context) {
        auto soc = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t aiv_count = soc.GetCoreNumAiv() > 0 ?
            static_cast<uint32_t>(soc.GetCoreNumAiv()) : 1U;

        const gert::Tensor *input_tensor = context->GetRequiredInputTensor(0);
        const gert::StorageShape *output_storage = context->GetOutputShape(0);
        const gert::RuntimeAttrs *op_attrs = context->GetAttrs();
        if (input_tensor == nullptr || output_storage == nullptr || op_attrs == nullptr) {
            return RejectTiling();
        }

        const gert::TypedContinuousVector<int64_t> *crops_attr = op_attrs->GetListInt(0);
        const int64_t *block_attr_ptr = op_attrs->GetInt(1);
        if (crops_attr == nullptr || block_attr_ptr == nullptr || crops_attr->GetSize() != 4) {
            return RejectTiling();
        }

        const gert::Shape &input_dims = input_tensor->GetStorageShape();
        const gert::Shape &output_dims = output_storage->GetStorageShape();
        if (input_dims.GetDimNum() != 4 || output_dims.GetDimNum() != 4) {
            return RejectTiling();
        }

        const int64_t *crops = crops_attr->GetData();
        int64_t raw_in_n = input_dims.GetDim(0);
        int64_t raw_in_h = input_dims.GetDim(1);
        int64_t raw_in_w = input_dims.GetDim(2);
        int64_t raw_in_c = input_dims.GetDim(3);
        int64_t raw_block = *block_attr_ptr;
        if (raw_in_n <= 0 || raw_in_h <= 0 || raw_in_w <= 0 || raw_in_c <= 0 ||
            raw_block <= 0 || crops[0] < 0 || crops[1] < 0 ||
            (crops[2] < 0) || crops[3] < 0) {
            return RejectTiling();
        }

        uint64_t inN64 = static_cast<uint64_t>(raw_in_n);
        uint64_t inH64 = static_cast<uint64_t>(raw_in_h);
        uint64_t inW64 = static_cast<uint64_t>(raw_in_w);
        uint64_t inC64 = static_cast<uint64_t>(raw_in_c);
        uint64_t block64v = static_cast<uint64_t>(raw_block);
        uint64_t topTrim64 = static_cast<uint64_t>(crops[0]);
        uint64_t bottomTrim64 = static_cast<uint64_t>(crops[1]);
        uint64_t leftTrim64 = static_cast<uint64_t>(crops[2]);
        uint64_t rightTrim64 = static_cast<uint64_t>(crops[3]);
        if (inN64 > kHostUint32Max || inH64 > kHostUint32Max ||
            inW64 > kHostUint32Max || inC64 > kHostUint32Max ||
            block64v > kHostUint32Max || topTrim64 > kHostUint32Max ||
            bottomTrim64 > kHostUint32Max || leftTrim64 > kHostUint32Max ||
            rightTrim64 > kHostUint32Max) {
            return RejectTiling();
        }
        if (MulWouldOverflow32(block64v, block64v, kHostUint32Max) ||
            MulWouldOverflow32(inH64, block64v, kHostUint32Max) ||
            MulWouldOverflow32(inW64, block64v, kHostUint32Max)) {
            return RejectTiling();
        }

        uint64_t blockCells = block64v * block64v;
        uint64_t paddedRows = inH64 * block64v;
        uint64_t paddedCols = inW64 * block64v;
        uint64_t trimRows = topTrim64 + bottomTrim64;
        uint64_t trimCols = leftTrim64 + rightTrim64;
        if (blockCells == 0U || inN64 % blockCells != 0U ||
            trimRows >= paddedRows || trimCols >= paddedCols) {
            return RejectTiling();
        }

        uint64_t outN64 = inN64 / blockCells;
        uint64_t outH64 = paddedRows - trimRows;
        uint64_t outW64 = paddedCols - trimCols;
        if (outN64 == 0U || outH64 == 0U || outW64 == 0U ||
            outN64 > kHostUint32Max || outH64 > kHostUint32Max ||
            outW64 > kHostUint32Max ||
            Shape4WouldOverflow32(inN64, inH64, inW64, inC64, kHostUint32Max) ||
            Shape4WouldOverflow32(outN64, outH64, outW64, inC64, kHostUint32Max)) {
            return RejectTiling();
        }

        if (output_dims.GetDim(0) != static_cast<int64_t>(outN64) ||
            output_dims.GetDim(1) != static_cast<int64_t>(outH64) ||
            output_dims.GetDim(2) != static_cast<int64_t>(outW64) ||
            output_dims.GetDim(3) != static_cast<int64_t>(inC64)) {
            return RejectTiling();
        }

        ge::DataType input_dtype = input_tensor->GetDataType();
        if (input_dtype != ge::DT_FLOAT16 && input_dtype != ge::DT_FLOAT) {
            return RejectTiling();
        }

        ProfileShape footprint = {
            static_cast<uint32_t>(inN64),
            static_cast<uint32_t>(inH64),
            static_cast<uint32_t>(inW64),
            static_cast<uint32_t>(inC64),
            static_cast<uint32_t>(outN64),
            static_cast<uint32_t>(outH64),
            static_cast<uint32_t>(outW64),
            static_cast<uint32_t>(inC64),
            static_cast<uint32_t>(block64v),
            static_cast<uint32_t>(topTrim64),
            static_cast<uint32_t>(bottomTrim64),
            static_cast<uint32_t>(leftTrim64),
            static_cast<uint32_t>(rightTrim64),
        };

        uint64_t profile_key = SelectProfileKey(footprint, input_dtype);
        if (profile_key == 0U) {
            return RejectTiling();
        }

        uint32_t dtypeToken = static_cast<uint32_t>(input_dtype);
        ASCENDC_TPL_SEL_PARAM(context, dtypeToken, profile_key);

        BatchToSpaceTilingData *abi_blob = context->GetTilingData<BatchToSpaceTilingData>();
        if (abi_blob == nullptr) {
            return RejectTiling();
        }
        abi_blob->abi_guard = 0U;

        context->SetBlockDim(ResolveAivLaunch(profile_key, aiv_count));
        size_t *workspace_slots = context->GetWorkspaceSizes(1);
        workspace_slots[0] = 0U;
        return static_cast<ge::graphStatus>(ge::GRAPH_SUCCESS);
    }
}  // namespace optiling profile dispatch

namespace ge {  // inference
    static graphStatus RejectInfer() {
        return static_cast<graphStatus>(GRAPH_FAILED);
    }

    static graphStatus InferShape(gert::InferShapeContext * const context) {
        const gert::Shape *input_dims = context->GetInputShape(0);
        gert::Shape *output_dims = context->GetOutputShape(0);
        const gert::RuntimeAttrs *infer_attrs = context->GetAttrs();
        if (input_dims == nullptr || output_dims == nullptr || infer_attrs == nullptr ||
            input_dims->GetDimNum() != 4) {
            return RejectInfer();
        }

        const gert::TypedContinuousVector<int64_t> *crops_attr = infer_attrs->GetListInt(0);
        const int64_t *block_attr_ptr = infer_attrs->GetInt(1);
        if (crops_attr == nullptr || block_attr_ptr == nullptr || crops_attr->GetSize() != 4) {
            return RejectInfer();
        }

        const int64_t *crops = crops_attr->GetData();
        int64_t blockSize = *block_attr_ptr;
        int64_t inN = input_dims->GetDim(0);
        int64_t inH = input_dims->GetDim(1);
        int64_t inW = input_dims->GetDim(2);
        int64_t inC = input_dims->GetDim(3);
        if (inN <= 0 || inH <= 0 || inW <= 0 || inC <= 0 ||
            blockSize <= 0 || crops[0] < 0 || crops[1] < 0 ||
            (crops[2] < 0) || crops[3] < 0) {
            return RejectInfer();
        }

        uint64_t inN64 = static_cast<uint64_t>(inN);
        uint64_t inH64 = static_cast<uint64_t>(inH);
        uint64_t inW64 = static_cast<uint64_t>(inW);
        uint64_t inC64 = static_cast<uint64_t>(inC);
        uint64_t block64v = static_cast<uint64_t>(blockSize);
        uint64_t inferTrimRows = static_cast<uint64_t>(crops[0]) + static_cast<uint64_t>(crops[1]);
        uint64_t inferTrimCols = static_cast<uint64_t>(crops[2]) + static_cast<uint64_t>(crops[3]);
        if (inN64 > optiling::kHostUint32Max || inH64 > optiling::kHostUint32Max ||
            inW64 > optiling::kHostUint32Max || inC64 > optiling::kHostUint32Max ||
            block64v > optiling::kHostUint32Max ||
            optiling::MulWouldOverflow32(block64v, block64v, optiling::kHostUint32Max) ||
            optiling::MulWouldOverflow32(inH64, block64v, optiling::kHostUint32Max) ||
            optiling::MulWouldOverflow32(inW64, block64v, optiling::kHostUint32Max)) {
            return RejectInfer();
        }

        uint64_t blockCells = block64v * block64v;
        uint64_t paddedRows = inH64 * block64v;
        uint64_t paddedCols = inW64 * block64v;
        if (blockCells == 0U || inN64 % blockCells != 0U ||
            inferTrimRows >= paddedRows || inferTrimCols >= paddedCols) {
            return RejectInfer();
        }

        uint64_t outN64 = inN64 / blockCells;
        uint64_t outH64 = paddedRows - inferTrimRows;
        uint64_t outW64 = paddedCols - inferTrimCols;
        if (outN64 == 0U || outH64 == 0U || outW64 == 0U ||
            outN64 > optiling::kHostUint32Max ||
            outH64 > optiling::kHostUint32Max ||
            outW64 > optiling::kHostUint32Max ||
            optiling::Shape4WouldOverflow32(inN64, inH64, inW64, inC64,
                                       optiling::kHostUint32Max) ||
            optiling::Shape4WouldOverflow32(outN64, outH64, outW64, inC64,
                                       optiling::kHostUint32Max)) {
            return RejectInfer();
        }

        output_dims->SetDimNum(0);
        output_dims->AppendDim(static_cast<int64_t>(outN64));
        output_dims->AppendDim(static_cast<int64_t>(outH64));
        output_dims->AppendDim(static_cast<int64_t>(outW64));
        output_dims->AppendDim(inC);
        return static_cast<graphStatus>(GRAPH_SUCCESS);
    }

    static graphStatus InferDataType(gert::InferDataTypeContext * const context) {
        ge::DataType mirrored_type = context->GetInputDataType(0);
        context->SetOutputDataType(0, mirrored_type);
        return static_cast<graphStatus>(GRAPH_SUCCESS);
    }
}  // namespace ge inference

namespace ops {  // registration
    class BatchToSpace final : public OpDef {
    public:  // registration constructor
        explicit BatchToSpace(const char *opName) : OpDef(opName) {
            this->Input("x").ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y").ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
            (void)this->Attr("crops").AttrType(REQUIRED).ListInt();
            (void)this->Attr("block_size").AttrType(REQUIRED).Int();
            (void)this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b").AddConfig("ascend910_93");
        }
    };
    OP_ADD(BatchToSpace);  // register BatchToSpace
}  // namespace ops registration
