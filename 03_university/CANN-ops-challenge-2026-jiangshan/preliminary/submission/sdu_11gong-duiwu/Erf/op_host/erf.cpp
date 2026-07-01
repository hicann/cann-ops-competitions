// Host-side Tiling — auto-selects tile size & core count per data point
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {

    // Auto-select tile size based on total data length.
    // Small data → single tile (zero DMA split overhead), min 64 bytes for alignment
    // Large data → ERF_TILE_LENGTH (max safe tile for buffer memory)
    static uint32_t SelectTileSize(uint64_t totalLength) {
        constexpr uint32_t MIN_TILE = 256;
        if (totalLength <= ERF_TILE_LENGTH) {
            uint32_t n = static_cast<uint32_t>(totalLength);
            if (n < MIN_TILE) n = MIN_TILE;
            // Round up to 8 for vector instruction alignment
            return (n + 7) / 8 * 8;
        }
        return ERF_TILE_LENGTH;
    }

    // Auto-select core count: target ~1 tile per core.
    static uint32_t SelectCoreCount(uint64_t totalLength, uint32_t tileLength, int32_t maxCores) {
        uint32_t tiles = static_cast<uint32_t>((totalLength + tileLength - 1) / tileLength);
        if (tiles <= 1) return 1;
        if (static_cast<int32_t>(tiles) > maxCores) return static_cast<uint32_t>(maxCores);
        return tiles;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        uint64_t totalLength = tensor_x->GetShapeSize();

        if (totalLength == 0) {
            context->SetBlockDim(1);
            return ge::GRAPH_SUCCESS;
        }

        uint32_t tileLength;
        uint32_t coreNum;

        // Probe modes 5-9 override the auto-selector for testing
        if (kErfProbeMode == ERF_PROBE_FORCE_SINGLE_CORE) {
            tileLength = SelectTileSize(totalLength);
            coreNum = 1;
        } else if (kErfProbeMode == ERF_PROBE_FORCE_MAX_CORES) {
            tileLength = SelectTileSize(totalLength);
            coreNum = static_cast<uint32_t>(num_cores_aiv);
        } else if (kErfProbeMode == ERF_PROBE_FORCE_EXACT_CORES) {
            tileLength = SelectTileSize(totalLength);
            coreNum = kErfProbeSpin == 0 ? 1 : kErfProbeSpin;
        } else if (kErfProbeMode == ERF_PROBE_SUBTILE_EIGHT_CORES) {
            tileLength = SelectTileSize(totalLength);
            coreNum = totalLength < static_cast<uint64_t>(ERF_TILE_LENGTH) ? 8 : SelectCoreCount(totalLength, tileLength, num_cores_aiv);
        } else if (kErfProbeMode == ERF_PROBE_SUBTILE_SIXTEEN_CORES) {
            tileLength = SelectTileSize(totalLength);
            coreNum = totalLength < static_cast<uint64_t>(ERF_TILE_LENGTH) ? 16 : SelectCoreCount(totalLength, tileLength, num_cores_aiv);
        } else {
            // Auto-selector: modes 0-4 use dynamic tile+core
            tileLength = SelectTileSize(totalLength);
            coreNum = SelectCoreCount(totalLength, tileLength, num_cores_aiv);
        }

        // Clamp core count to available hardware cores
        if (coreNum > static_cast<uint32_t>(num_cores_aiv))
            coreNum = static_cast<uint32_t>(num_cores_aiv);

        uint64_t coreDataLength = (totalLength + coreNum - 1) / coreNum;
        coreDataLength = (coreDataLength + ERF_ALIGN_NUM - 1) / ERF_ALIGN_NUM * ERF_ALIGN_NUM;

        if (coreDataLength * (coreNum - 1) >= totalLength) {
            coreNum = (totalLength + coreDataLength - 1) / coreDataLength;
            if (coreNum == 0) coreNum = 1;
        }

        uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
        tiling->totalLength = totalLength;
        tiling->coreDataLength = coreDataLength;
        tiling->coreNum = coreNum;
        tiling->tileLength = tileLength;
        tiling->fusionMode = 0;  // 0 = Erf, 1 = GELU

        context->SetBlockDim(coreNum);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *inputShape = context->GetInputShape(0);
        gert::Shape *outputShape = context->GetOutputShape(0);
        *outputShape = *inputShape;
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
