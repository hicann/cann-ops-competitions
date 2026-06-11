// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

#include "../op_kernel/addcmul_tiling.h"
#include "../op_kernel/tiling_key_addcmul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // Get platform info
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t numCoresAiv = static_cast<uint32_t>(platform.GetCoreNumAiv());
        uint64_t ubSize;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

        // Get input tensors
        const gert::Tensor *tensorInputData = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensorX1 = context->GetRequiredInputTensor(1);
        const gert::Tensor *tensorX2 = context->GetRequiredInputTensor(2);
        const gert::Tensor *tensorValue = context->GetRequiredInputTensor(3);

        // Get dtype and element size
        ge::DataType dtype = tensorInputData->GetDataType();
        int32_t dtypeSize = static_cast<int32_t>(ge::GetSizeByDataType(dtype));

        // Get origin shapes for broadcast detection
        const auto &shape0 = tensorInputData->GetOriginShape();
        const auto &shape1 = tensorX1->GetOriginShape();
        const auto &shape2 = tensorX2->GetOriginShape();

        size_t rank0 = shape0.GetDimNum();
        size_t rank1 = shape1.GetDimNum();
        size_t rank2 = shape2.GetDimNum();

        // Check if broadcast is needed (shapes must all match for linear mode)
        bool needBroadcast = false;
        if (rank0 != rank1 || rank1 != rank2) {
            needBroadcast = true;
        } else {
            for (size_t i = 0; i < rank0; ++i) {
                if (shape0.GetDim(i) != shape1.GetDim(i) ||
                    shape1.GetDim(i) != shape2.GetDim(i)) {
                    needBroadcast = true;
                    break;
                }
            }
        }

        // Compute total length (broadcast output shape)
        uint32_t totalLength;
        if (!needBroadcast) {
            // Linear mode: all inputs have same shape
            totalLength = static_cast<uint32_t>(tensorInputData->GetShapeSize());
        } else {
            // Broadcast mode: compute broadcast output shape dimensions
            size_t maxRank = std::max({rank0, rank1, rank2});
            uint32_t len = 1;
            for (size_t i = 0; i < maxRank; ++i) {
                int64_t d0 = (i < rank0) ? shape0.GetDim(rank0 - 1 - i) : 1;
                int64_t d1 = (i < rank1) ? shape1.GetDim(rank1 - 1 - i) : 1;
                int64_t d2 = (i < rank2) ? shape2.GetDim(rank2 - 1 - i) : 1;
                int64_t dim = std::max({d0, d1, d2});
                len *= static_cast<uint32_t>(dim);
            }
            totalLength = len;
        }

        // Compute per-input total elements (pre-broadcast)
        uint32_t x1Total = static_cast<uint32_t>(tensorX1->GetShapeSize());
        uint32_t x2Total = static_cast<uint32_t>(tensorX2->GetShapeSize());
        uint32_t inTotal = static_cast<uint32_t>(tensorInputData->GetShapeSize());

        // Re-evaluate broadcast: check if any input has fewer elements than output
        bool inputBroadcast = (x1Total < totalLength) || (x2Total < totalLength) || (inTotal < totalLength);
        needBroadcast = needBroadcast || inputBroadcast;

        // Configure TilingKey for kernel template selection
        uint32_t DT_INPUT_DATA = static_cast<uint32_t>(dtype);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_DATA);

        // --- Multi-core split ---
        // Ensure at least 4KB data per core
        constexpr uint32_t MIN_BYTES_PER_CORE = 4096;
        uint32_t coreNum = (totalLength * dtypeSize + MIN_BYTES_PER_CORE - 1) / MIN_BYTES_PER_CORE;
        if (coreNum < 1) coreNum = 1;
        if (coreNum > numCoresAiv) coreNum = numCoresAiv;

        // For broadcast, find the minimum input size to constrain block alignment
        uint32_t minInputSize = totalLength;
        if (needBroadcast) {
            minInputSize = std::min({x1Total, x2Total, inTotal});
        }

        // Calculate blockFormer (256-byte aligned)
        uint32_t blockFormerRaw = (totalLength + coreNum - 1) / coreNum;
        uint32_t blockFormerBytes = ((blockFormerRaw * dtypeSize + 255) / 256) * 256;
        uint32_t blockFormer = blockFormerBytes / dtypeSize;

        // For broadcast, align blockFormer to minInputSize to avoid wrap in middle of tile
        if (needBroadcast && minInputSize > 0) {
            uint32_t alignedElements = 256 / dtypeSize;
            if (alignedElements == 0) alignedElements = 1;
            blockFormer = (blockFormer / minInputSize) * minInputSize;
            if (blockFormer < minInputSize) blockFormer = minInputSize;
        }

        uint32_t blockNum = (totalLength + blockFormer - 1) / blockFormer;
        uint32_t blockTail = totalLength - (blockNum - 1) * blockFormer;

        // Handle case where blockNum < coreNum (adjust core count)
        if (blockNum < coreNum) {
            coreNum = (blockNum > 0) ? blockNum : 1;
            blockFormerRaw = (totalLength + coreNum - 1) / coreNum;
            blockFormerBytes = ((blockFormerRaw * dtypeSize + 255) / 256) * 256;
            blockFormer = blockFormerBytes / dtypeSize;
            blockNum = (totalLength + blockFormer - 1) / blockFormer;
            blockTail = totalLength - (blockNum - 1) * blockFormer;
        }

        // --- UB tile split ---
        constexpr uint32_t EXTRA_SIZE = 1024;
        constexpr uint32_t VALUE_BUF = 32;
        uint32_t ubAvail = static_cast<uint32_t>(ubSize - EXTRA_SIZE - VALUE_BUF);
        uint32_t ubFormer = 0;

        // Calculate ubFormer based on dtype peak bytes/element from DESIGN.md
        if (dtype == ge::DT_FLOAT16) {
            // float16 upcast to float32: peak 10B/elem (x1Fp32 4B + x2Half 2B + x2Fp32 4B)
            ubFormer = ubAvail / 10;
        } else if (dtype == ge::DT_INT8) {
            // int8 cast to half: peak 5B/elem (x1Half 2B + x2Local 1B + x2Half 2B)
            ubFormer = ubAvail / 5;
        } else {
            // float32/int32 direct: peak 8B/elem (2 * 4 = 2 * dtypeSize)
            ubFormer = ubAvail / static_cast<uint32_t>(2 * dtypeSize);
        }
        // Align to 256 bytes
        uint32_t alignElements = 256 / static_cast<uint32_t>(dtypeSize);
        if (alignElements == 0) alignElements = 1;
        ubFormer = (ubFormer / alignElements) * alignElements;

        // Clamp ubFormer to blockFormer (no need to tile further if within one tile)
        if (ubFormer > blockFormer) {
            ubFormer = blockFormer;
        }

        // For broadcast, cap ubFormer to minInputSize to prevent wrap within a single tile
        if (needBroadcast && minInputSize > 0 && ubFormer > minInputSize) {
            ubFormer = (minInputSize / alignElements) * alignElements;
            if (ubFormer == 0) ubFormer = alignElements;
            if (ubFormer > blockFormer) ubFormer = blockFormer;
        }

        // Calculate loop counts and tile tails
        uint32_t loopNum = 0;
        uint32_t ubTail = 0;
        if (blockFormer > 0 && ubFormer > 0) {
            loopNum = (blockFormer + ubFormer - 1) / ubFormer;
            ubTail = blockFormer - (loopNum - 1) * ubFormer;
            if (ubTail == 0) {
                ubTail = ubFormer;
                loopNum = (loopNum > 1) ? loopNum - 1 : 1;
            }
        }

        uint32_t tailLoopNum = 0;
        uint32_t tailUbTail = 0;
        if (blockTail > 0 && blockNum > 1 && ubFormer > 0) {
            tailLoopNum = (blockTail + ubFormer - 1) / ubFormer;
            tailUbTail = blockTail - (tailLoopNum - 1) * ubFormer;
            if (tailUbTail == 0) {
                tailUbTail = ubFormer;
                tailLoopNum = (tailLoopNum > 1) ? tailLoopNum - 1 : 1;
            }
        } else if (blockNum == 1) {
            // Single block: tail params same as main block
            tailLoopNum = loopNum;
            tailUbTail = ubTail;
        }

        // Handle empty tensor
        if (totalLength == 0) {
            blockFormer = 0;
            blockTail = 0;
            ubFormer = 0;
            loopNum = 0;
            ubTail = 0;
            tailLoopNum = 0;
            tailUbTail = 0;
        }

        // Fill TilingData structure
        AddcmulTilingData *tiling = context->GetTilingData<AddcmulTilingData>();
        tiling->dtypeSize = dtypeSize;
        tiling->totalLength = totalLength;
        tiling->blockFormer = blockFormer;
        tiling->blockTail = blockTail;
        tiling->ubFormer = ubFormer;
        tiling->ubTail = ubTail;
        tiling->loopNum = loopNum;
        tiling->tailLoopNum = tailLoopNum;
        tiling->tailUbTail = tailUbTail;
        tiling->needBroadcast = needBroadcast ? 1 : 0;
        tiling->x1Total = x1Total;
        tiling->x2Total = x2Total;
        tiling->inTotal = inTotal;

        // Configure kernel launch parameters
        context->SetBlockDim(coreNum);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // Compute broadcast output shape from input shapes
        const gert::Tensor *t0 = context->GetRequiredInputTensor(0);
        const gert::Tensor *t1 = context->GetRequiredInputTensor(1);
        const gert::Tensor *t2 = context->GetRequiredInputTensor(2);

        const auto &s0 = t0->GetOriginShape();
        const auto &s1 = t1->GetOriginShape();
        const auto &s2 = t2->GetOriginShape();

        size_t r0 = s0.GetDimNum();
        size_t r1 = s1.GetDimNum();
        size_t r2 = s2.GetDimNum();
        size_t maxRank = std::max({r0, r1, r2});

        // Compute broadcast output dims (NumPy-style, from rightmost)
        int64_t outDims[8];  // max 8 dims
        for (size_t i = 0; i < maxRank; ++i) {
            int64_t d0 = (i < r0) ? s0.GetDim(r0 - 1 - i) : 1;
            int64_t d1 = (i < r1) ? s1.GetDim(r1 - 1 - i) : 1;
            int64_t d2 = (i < r2) ? s2.GetDim(r2 - 1 - i) : 1;
            outDims[maxRank - 1 - i] = std::max({d0, d1, d2});
        }

        // Set output shape
        gert::Shape *outShape = context->GetOutputShape(0);
        outShape->SetDimNum(maxRank);
        for (size_t i = 0; i < maxRank; ++i) {
            outShape->SetDim(i, outDims[i]);
        }
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class Addcmul : public OpDef {
    public:
        explicit Addcmul(const char *name) : OpDef(name) {
            this->Input("input_data")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("x1")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("x2")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("value")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Addcmul);
}  // namespace ops

