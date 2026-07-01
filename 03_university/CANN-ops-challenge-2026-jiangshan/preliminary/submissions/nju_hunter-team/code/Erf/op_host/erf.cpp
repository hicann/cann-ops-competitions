// Host侧Tiling实现
#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
    namespace {
        constexpr uint32_t FP32_BYTES = 4;
        constexpr uint32_t ALIGN_NUM = 64;
        constexpr uint32_t SMALL_TILE_LENGTH = 16384;
        constexpr uint64_t UB_RESERVED_BYTES = 4096;
        constexpr uint32_t BUFFER_NUM = 1;
        constexpr uint32_t FLOAT_TMP_BUFFER_NUM = 1;
        constexpr uint32_t MASK_BUFFER_NUM = 0;
        constexpr uint64_t SMALL_SHAPE_LIMIT = 1024;
        constexpr uint64_t GENERAL_SHAPE_LIMIT = 262144;
        constexpr uint32_t GENERAL_ELEMS_PER_CORE = 4096;
        constexpr uint64_t FAST_SMALL_CORE_LIMIT = 1024;
        constexpr uint64_t FAST_SMALL_SINGLE_MODE = 1;

        uint32_t AlignDown(uint32_t value, uint32_t align) {
            return value & ~(align - 1);
        }

        uint32_t AlignUp(uint32_t value, uint32_t align) {
            return (value + align - 1) & ~(align - 1);
        }

        uint64_t AlignDown(uint64_t value, uint64_t align) {
            return value & ~(align - 1);
        }

        uint64_t AlignUp(uint64_t value, uint64_t align) {
            return (value + align - 1) & ~(align - 1);
        }

        uint64_t CeilDiv(uint64_t value, uint64_t divisor) {
            return (value + divisor - 1) / divisor;
        }

        uint32_t CeilDivPowerOfTwo(uint64_t value, uint32_t divisor) {
            return static_cast<uint32_t>((value + divisor - 1) >> __builtin_ctz(divisor));
        }

        uint32_t CalcMaxTileLength(uint64_t ubSize) {
            uint64_t availableUb = (ubSize > UB_RESERVED_BYTES) ? (ubSize - UB_RESERVED_BYTES) : ubSize;
            uint64_t bytesPerElement = (BUFFER_NUM * FP32_BYTES) + (BUFFER_NUM * FP32_BYTES) +
                (FLOAT_TMP_BUFFER_NUM * FP32_BYTES) + MASK_BUFFER_NUM;
            uint32_t maxTileLength = static_cast<uint32_t>(availableUb / bytesPerElement);
            maxTileLength = AlignDown(maxTileLength, ALIGN_NUM);
            if (maxTileLength == 0) {
                return ALIGN_NUM;
            }
            return maxTileLength;
        }

        uint32_t CapTileLength(uint32_t requestTileLength, uint32_t maxTileLength) {
            return std::max(std::min(requestTileLength, maxTileLength), ALIGN_NUM);
        }

        uint64_t CalcMaxCoreLength(uint64_t totalLength, uint32_t coreNum, uint64_t blockLength) {
            if (totalLength == 0 || coreNum <= 1) {
                return totalLength;
            }
            uint64_t lastCoreLength = totalLength - blockLength * (coreNum - 1);
            return std::max(blockLength, lastCoreLength);
        }

        uint64_t CalcDownBlockLength(uint64_t totalLength, uint32_t coreNum) {
            if (totalLength == 0 || coreNum <= 1) {
                return totalLength;
            }
            uint64_t blockLength = AlignDown(CeilDiv(totalLength, coreNum), static_cast<uint64_t>(ALIGN_NUM));
            return std::max(blockLength, static_cast<uint64_t>(ALIGN_NUM));
        }

        uint64_t CalcBlockLength(uint64_t totalLength, uint32_t coreNum) {
            if (totalLength == 0 || coreNum <= 1) {
                return totalLength;
            }
            uint64_t avgLength = CeilDiv(totalLength, coreNum);
            uint64_t downBlock = CalcDownBlockLength(totalLength, coreNum);
            uint64_t upBlock = AlignUp(avgLength, static_cast<uint64_t>(ALIGN_NUM));
            uint64_t downMaxCore = CalcMaxCoreLength(totalLength, coreNum, downBlock);

            if (upBlock * (coreNum - 1) >= totalLength) {
                return downBlock;
            }

            uint64_t upMaxCore = CalcMaxCoreLength(totalLength, coreNum, upBlock);
            return (upMaxCore < downMaxCore) ? upBlock : downBlock;
        }

        uint64_t MakeFastHeader(uint64_t totalLength, uint32_t smallMode, uint32_t coreNum,
                                uint64_t blockLength) {
            uint64_t fastMode = 0;
            if (smallMode != 0 && totalLength != 0) {
                uint64_t maxCoreLength = CalcMaxCoreLength(totalLength, coreNum, blockLength);
                if (maxCoreLength <= FAST_SMALL_CORE_LIMIT) {
                    if (coreNum <= 1) {
                        fastMode = FAST_SMALL_SINGLE_MODE;
                    }
                }
            }
            if (fastMode == 0) {
                return 0;
            }
            return (totalLength << 32) | fastMode;
        }

        void SelectTiling(uint64_t totalLength, uint32_t maxTileLength, int32_t aivCoreNum,
            uint32_t &tileLength, uint32_t &coreNum, uint32_t &smallMode, uint64_t &blockLength) {
            uint32_t maxCoreNum = static_cast<uint32_t>(std::max(aivCoreNum, 1));
            smallMode = 0;

            if (totalLength == 0) {
                tileLength = ALIGN_NUM;
                coreNum = 1;
                blockLength = 0;
                return;
            }

            if (totalLength <= SMALL_SHAPE_LIMIT) {
                smallMode = 1;
                coreNum = 1;
                blockLength = CalcDownBlockLength(totalLength, coreNum);
                uint64_t maxCoreLength = CalcMaxCoreLength(totalLength, coreNum, blockLength);
                uint32_t smallTile = AlignUp(static_cast<uint32_t>(maxCoreLength), ALIGN_NUM);
                tileLength = CapTileLength(std::min(smallTile, SMALL_TILE_LENGTH), maxTileLength);
                return;
            }

            if (totalLength <= GENERAL_SHAPE_LIMIT) {
                tileLength = maxTileLength;
                coreNum = std::max(CeilDivPowerOfTwo(totalLength, GENERAL_ELEMS_PER_CORE), 1U);
                coreNum = std::min(coreNum, maxCoreNum);
                blockLength = CalcBlockLength(totalLength, coreNum);
                return;
            }

            tileLength = maxTileLength;
            coreNum = maxCoreNum;
            blockLength = CalcBlockLength(totalLength, coreNum);
        }
    } // namespace

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t aivCoreNum = platform.GetCoreNumAiv();
        uint64_t ubSize = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

        auto inputShape = context->GetInputShape(0);
        if (inputShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const auto &storageShape = inputShape->GetStorageShape();
        auto inputDesc = context->GetInputDesc(0);
        if (inputDesc == nullptr) {
            return ge::GRAPH_FAILED;
        }
        ge::DataType dtype_x = inputDesc->GetDataType();
        if (dtype_x != ge::DT_FLOAT) {
            return ge::GRAPH_FAILED;
        }

        uint64_t totalLength = static_cast<uint64_t>(storageShape.GetShapeSize());
        uint32_t maxTileLength = CalcMaxTileLength(ubSize);
        uint32_t tileLength = ALIGN_NUM;
        uint32_t coreNum = 1;
        uint32_t smallMode = 0;
        uint64_t blockLength = 0;
        SelectTiling(totalLength, maxTileLength, aivCoreNum, tileLength, coreNum, smallMode, blockLength);

        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
        if (tiling == nullptr) {
            return ge::GRAPH_FAILED;
        }
        tiling->fastHeader = MakeFastHeader(totalLength, smallMode, coreNum, blockLength);
        tiling->totalLength = totalLength;
        tiling->blockLength = blockLength;
        tiling->tileLength = tileLength;
        tiling->coreNum = coreNum;
        tiling->smallMode = smallMode;

        context->SetBlockDim(coreNum);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *xShape = context->GetInputShape(0);
        if (xShape == nullptr) {
            return GRAPH_FAILED;
        }
        gert::Shape *yShape = context->GetOutputShape(0);
        if (yShape == nullptr) {
            return GRAPH_FAILED;
        }
        *yShape = *xShape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
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
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Erf);
}  // namespace ops
