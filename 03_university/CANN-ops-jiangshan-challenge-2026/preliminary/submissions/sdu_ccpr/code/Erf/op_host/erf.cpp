// Host侧Tiling实现
#include <algorithm>
#include <cstdint>
#include <cstring>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace {
    constexpr uint32_t ALIGN_BYTES = 32;
    constexpr uint32_t REPEAT_BYTES = 256;
    constexpr uint32_t GM_ALIGN_BYTES = 512;
    constexpr uint64_t SMALL_INPUT_THRESHOLD = 262144;
    constexpr uint64_t DIRECT_SMALL_THRESHOLD = 2048*30;
    // Tuning knobs (defaults keep current baseline behavior).
    constexpr bool ENABLE_DIRECT_ONE_TILE_SPLIT = true;
    constexpr uint64_t DIRECT_ONE_TILE_MIN = 1;
    constexpr uint64_t DIRECT_ONE_TILE_MAX = 8192;
    constexpr uint64_t DIRECT_TINY_MAX = 256;
    constexpr bool DIRECT_SMALL_LIMIT_BY_CORE_NUM = true;
    constexpr uint64_t DIRECT_SMALL_BATCH = 6;
    constexpr bool DIRECT_SMALL_USE_BATCH = true;
    constexpr uint32_t DOUBLE_BUFFER_TILE = 4096;
    constexpr uint32_t SINGLE_TILE_MIN = 512;
    constexpr uint32_t SINGLE_TILE_LOW = 1024;
    constexpr uint32_t SINGLE_TILE_MEDIUM = 2048;
    constexpr uint32_t SINGLE_TILE_HIGH = 4096;
    constexpr uint32_t SINGLE_TILE_MAX = 6656;
    constexpr uint64_t SINGLE_THRESHOLD_ONE = 2048;
    constexpr uint64_t SINGLE_THRESHOLD_TWO = 1024 * 20;
    constexpr uint64_t SINGLE_THRESHOLD_THREE = 2048 * 30;
    constexpr uint64_t SINGLE_THRESHOLD_FOUR = 122880;
    inline uint64_t CeilDiv(uint64_t a, uint64_t b) {
        return (a + b - 1) / b;
    }

    inline uint64_t AlignUp(uint64_t value, uint64_t alignment) {
        return CeilDiv(value, alignment) * alignment;
    }

    inline uint32_t SelectSingleTileLength(uint64_t total_length, uint32_t align_unit) {
        uint32_t tile_length = SINGLE_TILE_MAX;
        if (total_length <= SINGLE_THRESHOLD_ONE) {
            tile_length = SINGLE_TILE_MIN;
        } else if (total_length <= SINGLE_THRESHOLD_TWO) {
            tile_length = SINGLE_TILE_LOW;
        } else if (total_length <= SINGLE_THRESHOLD_THREE) {
            tile_length = SINGLE_TILE_MEDIUM;
        } else if (total_length <= SINGLE_THRESHOLD_FOUR) {
            tile_length = SINGLE_TILE_HIGH;
        }
        tile_length = (tile_length / align_unit) * align_unit;
        return std::max(tile_length, align_unit);
    }

    inline uint32_t SelectDirectTileLength(uint64_t total_length, uint32_t align_unit, uint32_t repeat_alignment) {
        constexpr uint64_t DIRECT_ONE_TILE_LIMIT = 6440;
        constexpr uint32_t DIRECT_TILE_LOW = 6144;
        constexpr uint32_t DIRECT_TILE_MEDIUM = 8192;
        constexpr uint32_t DIRECT_TILE_HIGH = 12288;

        uint64_t tile_length = 0;
        if (total_length <= DIRECT_ONE_TILE_LIMIT) {
            tile_length = AlignUp(total_length, static_cast<uint64_t>(repeat_alignment));
        } else if (total_length <= 14336) {
            tile_length = DIRECT_TILE_LOW;
        } else if (total_length <= 28672) {
            tile_length = DIRECT_TILE_MEDIUM;
        } else {
            tile_length = DIRECT_TILE_HIGH;
        }

        tile_length = AlignUp(tile_length, static_cast<uint64_t>(repeat_alignment));
        tile_length = std::max<uint64_t>(tile_length, static_cast<uint64_t>(align_unit));
        return static_cast<uint32_t>(tile_length);
    }

    inline uint32_t SelectDirectSmallUsedCores(uint64_t tile_count, uint64_t num_cores) {
        uint64_t used = std::max<uint64_t>(tile_count, 1);
        if (DIRECT_SMALL_USE_BATCH) {
            used = CeilDiv(tile_count, DIRECT_SMALL_BATCH);
            used = std::max<uint64_t>(used, 1);
        }
        if (DIRECT_SMALL_LIMIT_BY_CORE_NUM) {
            used = std::min<uint64_t>(used, num_cores);
        }
        return static_cast<uint32_t>(used);
    }

}  // namespace

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t dtype_size = static_cast<uint32_t>(ge::GetSizeByDataType(dtype_x));
        uint64_t total_length = static_cast<uint64_t>(tensor_x->GetShapeSize());

        bool use_double_buffer = total_length > SMALL_INPUT_THRESHOLD;
        bool use_direct_tiny = ENABLE_DIRECT_ONE_TILE_SPLIT &&
                               total_length >= DIRECT_ONE_TILE_MIN &&
                               total_length <= DIRECT_TINY_MAX;
        bool use_direct_one_tile = !use_direct_tiny &&
                                   ENABLE_DIRECT_ONE_TILE_SPLIT &&
                                   total_length >= DIRECT_ONE_TILE_MIN &&
                                   total_length <= DIRECT_ONE_TILE_MAX;
        bool use_direct_small = total_length > 0 &&
                                total_length <= DIRECT_SMALL_THRESHOLD &&
                                !use_direct_one_tile &&
                                !use_direct_tiny;
        uint32_t alignment = std::max<uint32_t>(1, ALIGN_BYTES / dtype_size);
        uint32_t repeat_alignment = std::max<uint32_t>(1, REPEAT_BYTES / dtype_size);
        uint32_t gm_alignment = std::max<uint32_t>(1, GM_ALIGN_BYTES / dtype_size);
        uint32_t base_alignment = std::max(alignment, gm_alignment);
        (void)ub_size;

        uint32_t tile_length = base_alignment;
        uint64_t big_core_data_num = 0;
        uint64_t small_core_data_num = 0;
        uint32_t big_core_count = 0;
        uint32_t used_cores = 1;

        if (total_length > 0) {
            if (use_direct_tiny) {
                tile_length = AlignUp(total_length, static_cast<uint64_t>(alignment));
                tile_length = std::max<uint64_t>(tile_length, static_cast<uint64_t>(alignment));
                used_cores = 1;
                big_core_data_num = tile_length;
                small_core_data_num = 0;
                big_core_count = 1;
            } else if (use_double_buffer) {
                tile_length = (DOUBLE_BUFFER_TILE / base_alignment) * base_alignment;
                tile_length = std::max(tile_length, base_alignment);

                used_cores = static_cast<uint32_t>(std::min<uint64_t>(
                    static_cast<uint64_t>(num_cores), CeilDiv(total_length, static_cast<uint64_t>(tile_length) * 2)));
                used_cores = std::max<uint32_t>(used_cores, 1U);

                big_core_data_num = AlignUp(CeilDiv(total_length, static_cast<uint64_t>(used_cores)), base_alignment);
                if (big_core_data_num == 0) {
                    big_core_data_num = base_alignment;
                }

                uint64_t tail = total_length > static_cast<uint64_t>(used_cores - 1) * big_core_data_num
                                    ? total_length - static_cast<uint64_t>(used_cores - 1) * big_core_data_num
                                    : 0;
                if (used_cores == 1 || tail == big_core_data_num) {
                    big_core_count = used_cores;
                    small_core_data_num = 0;
                } else {
                    big_core_count = used_cores - 1;
                    small_core_data_num = tail;
                }
            } else if (use_direct_one_tile) {
                tile_length = AlignUp(total_length, static_cast<uint64_t>(repeat_alignment));
                tile_length = std::max<uint64_t>(tile_length, static_cast<uint64_t>(base_alignment));
                used_cores = 1;
                big_core_data_num = tile_length;
                small_core_data_num = 0;
                big_core_count = 1;
            } else if (use_direct_small) {
                tile_length = SelectDirectTileLength(total_length, base_alignment, repeat_alignment);
                uint64_t tile_count = CeilDiv(total_length, static_cast<uint64_t>(tile_length));
                used_cores = SelectDirectSmallUsedCores(tile_count, static_cast<uint64_t>(num_cores));
                big_core_data_num = 0;
                small_core_data_num = 0;
                big_core_count = used_cores;
            } else {
                tile_length = SelectSingleTileLength(total_length, base_alignment);
                uint64_t tile_count = CeilDiv(total_length, static_cast<uint64_t>(tile_length));
                used_cores = static_cast<uint32_t>(
                    std::min<uint64_t>(static_cast<uint64_t>(num_cores), tile_count));
                used_cores = std::max<uint32_t>(used_cores, 1U);
                big_core_data_num = tile_length;
                big_core_count = used_cores;
                small_core_data_num = 0;
            }
        }

        uint64_t sch_mode = static_cast<uint64_t>(ERF_SCH_MODE_SINGLE_BUFFER);
        if (use_direct_tiny) {
            sch_mode = static_cast<uint64_t>(ERF_SCH_MODE_DIRECT_TINY);
        } else if (use_double_buffer) {
            sch_mode = static_cast<uint64_t>(ERF_SCH_MODE_DOUBLE_BUFFER);
        } else if (use_direct_one_tile) {
            sch_mode = static_cast<uint64_t>(ERF_SCH_MODE_DIRECT_ONE_TILE);
        } else if (use_direct_small) {
            sch_mode = static_cast<uint64_t>(ERF_SCH_MODE_DIRECT_SMALL);
        }
        uint32_t tiling_key = GET_TPL_TILING_KEY(sch_mode);
        context->SetTilingKey(tiling_key);

        ErfOpTilingData tiling = {};
        tiling.totalLength = total_length;
        tiling.bigCoreDataNum = static_cast<uint32_t>(big_core_data_num);
        tiling.smallCoreDataNum = static_cast<uint32_t>(small_core_data_num);
        tiling.bigCoreCount = big_core_count;
        tiling.usedCoreCount = used_cores;
        tiling.tileLength = tile_length;

        auto raw_tiling = context->GetRawTilingData();
        if (raw_tiling == nullptr || raw_tiling->GetCapacity() < sizeof(tiling)) {
            return ge::GRAPH_FAILED;
        }
        std::memcpy(raw_tiling->GetData(), &tiling, sizeof(tiling));
        raw_tiling->SetDataSize(sizeof(tiling));

        context->SetBlockDim(static_cast<int32_t>(used_cores));
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *x_shape = context->GetInputShape(0);
        gert::Shape *y_shape = context->GetOutputShape(0);
        *y_shape = *x_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const auto dtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, dtype);
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
