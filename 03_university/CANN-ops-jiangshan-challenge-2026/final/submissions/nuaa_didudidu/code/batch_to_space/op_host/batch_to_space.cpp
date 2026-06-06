// Host registration, shape inference, and tiling selection for BatchToSpace.
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/batch_to_space_tiling.h"
#include "../op_kernel/tiling_key_batch_to_space.h"

namespace optiling {
    static constexpr uint64_t kUint32Limit = 0xffffffffULL;
    static constexpr uint32_t kB2SBlockTransferBytes = 32U;
    static constexpr uint32_t kB2SDefaultRuntimeBytes = 64U * 1024U;
    static constexpr uint32_t kB2SPixelSplitLimit = 16U;
    static constexpr uint32_t kB2SChannelTaskElements = 2048U;
    static constexpr uint32_t kB2SWidePixelRowGroupLimit = 64U;
    static constexpr uint32_t kB2SWidePixelWidthLimit = 128U;
    static constexpr uint32_t kB2SWidePixelDepthLimit = 8U;
    static constexpr uint32_t kB2SPackedRowElements = 32768U;
    static constexpr uint32_t kB2SPackedRowMinWidth = 32U;
    static constexpr uint32_t kB2SVerticalWidthLimit = 32U;
    static constexpr uint32_t kB2SVerticalMinHeight = 64U;
    static constexpr uint32_t kB2SVerticalRowsPerTask = 128U;

    struct B2SShape64 {
        uint64_t batch64;
        uint64_t height64;
        uint64_t width64;
        uint64_t channels64;
    };

    struct B2SCrop64 {
        uint64_t trim_top64;
        uint64_t trim_bottom64;
        uint64_t trim_left64;
        uint64_t trim_right64;
    };

    struct B2SPlanContract {
        B2SLayout4D source;
        B2SLayout4D result;
        B2SCropBox crop;
        uint32_t block;
    };

    struct B2SR2PlainCase {
        ge::DataType dtype;
        uint32_t source_h;
        uint32_t source_w;
        uint32_t channel_count;
        uint32_t result_n;
        uint64_t schedule;
    };

    struct B2SExactCase {
        ge::DataType dtype;
        B2SLayout4D source;
        B2SLayout4D result;
        B2SCropBox crop;
        uint32_t block;
        uint64_t schedule;
    };

    struct B2SCoreRule {
        uint64_t schedule;
        uint32_t cores;
        bool fixed;
    };

    struct B2SRuntimeChoice {
        uint32_t path;
        uint64_t task_count;
        uint32_t element_quota;
    };

    static bool B2SMulOverflow(uint64_t left, uint64_t right, uint64_t limit) {
        return left != 0U && right > limit / left;
    }

    static bool B2SLayoutOverflow(const B2SShape64 &shape, uint64_t limit) {
        if (B2SMulOverflow(shape.batch64, shape.height64, limit)) {
            return true;
        }
        const uint64_t nh = shape.batch64 * shape.height64;
        if (B2SMulOverflow(nh, shape.width64, limit)) {
            return true;
        }
        return B2SMulOverflow(nh * shape.width64, shape.channels64, limit);
    }

    static bool B2SPositiveU64(int64_t value, uint64_t &out) {
        if (value <= 0 || static_cast<uint64_t>(value) > kUint32Limit) {
            return false;
        }
        out = static_cast<uint64_t>(value);
        return true;
    }

    static bool B2SReadShape64(const gert::Shape &shape, B2SShape64 &nhwc) {
        if (shape.GetDimNum() != 4) {
            return false;
        }
        return B2SPositiveU64(shape.GetDim(0), nhwc.batch64) &&
               B2SPositiveU64(shape.GetDim(1), nhwc.height64) &&
               B2SPositiveU64(shape.GetDim(2), nhwc.width64) &&
               B2SPositiveU64(shape.GetDim(3), nhwc.channels64);
    }

    static bool B2SReadAttrs(const gert::RuntimeAttrs *attrs, B2SCrop64 &crop, uint64_t &block) {
        if (attrs == nullptr) {
            return false;
        }

        const gert::TypedContinuousVector<int64_t> *crop_attr = attrs->GetListInt(0);
        const int64_t *block_attr = attrs->GetInt(1);
        if (crop_attr == nullptr || block_attr == nullptr || crop_attr->GetSize() != 4) {
            return false;
        }

        const int64_t *items = crop_attr->GetData();
        const bool crop_ok =
            items[0] >= 0 && items[1] >= 0 && items[2] >= 0 && items[3] >= 0 &&
            static_cast<uint64_t>(items[0]) <= kUint32Limit &&
            static_cast<uint64_t>(items[1]) <= kUint32Limit &&
            static_cast<uint64_t>(items[2]) <= kUint32Limit &&
            static_cast<uint64_t>(items[3]) <= kUint32Limit;
        if (!crop_ok || !B2SPositiveU64(*block_attr, block)) {
            return false;
        }

        crop = {
            static_cast<uint64_t>(items[0]),
            static_cast<uint64_t>(items[1]),
            static_cast<uint64_t>(items[2]),
            static_cast<uint64_t>(items[3]),
        };
        return true;
    }

    static bool B2SBuildContract(const B2SShape64 &source64, const B2SCrop64 &crop64,
                                      uint64_t block64, B2SPlanContract &checked) {
        if (B2SMulOverflow(block64, block64, kUint32Limit) ||
            B2SMulOverflow(source64.height64, block64, kUint32Limit) ||
            B2SMulOverflow(source64.width64, block64, kUint32Limit)) {
            return false;
        }

        const uint64_t phase_count = block64 * block64;
        const uint64_t expanded_h = source64.height64 * block64;
        const uint64_t expanded_w = source64.width64 * block64;
        const uint64_t crop_h = crop64.trim_top64 + crop64.trim_bottom64;
        const uint64_t crop_w = crop64.trim_left64 + crop64.trim_right64;
        if (source64.batch64 % phase_count != 0U || crop_h >= expanded_h || crop_w >= expanded_w) {
            return false;
        }

        const B2SShape64 result64 = {
            source64.batch64 / phase_count,
            expanded_h - crop_h,
            expanded_w - crop_w,
            source64.channels64,
        };
        if (result64.batch64 == 0U || result64.height64 == 0U || result64.width64 == 0U ||
            result64.batch64 > kUint32Limit || result64.height64 > kUint32Limit || result64.width64 > kUint32Limit ||
            B2SLayoutOverflow(source64, kUint32Limit) ||
            B2SLayoutOverflow(result64, kUint32Limit)) {
            return false;
        }

        checked.source = {
            static_cast<uint32_t>(source64.batch64),
            static_cast<uint32_t>(source64.height64),
            static_cast<uint32_t>(source64.width64),
            static_cast<uint32_t>(source64.channels64),
        };
        checked.result = {
            static_cast<uint32_t>(result64.batch64),
            static_cast<uint32_t>(result64.height64),
            static_cast<uint32_t>(result64.width64),
            static_cast<uint32_t>(result64.channels64),
        };
        checked.crop = {
            static_cast<uint32_t>(crop64.trim_top64),
            static_cast<uint32_t>(crop64.trim_bottom64),
            static_cast<uint32_t>(crop64.trim_left64),
            static_cast<uint32_t>(crop64.trim_right64),
        };
        checked.block = static_cast<uint32_t>(block64);
        return true;
    }

    static bool B2SSameLayout(const B2SLayout4D &lhs, const B2SLayout4D &rhs) {
        return lhs.batch_count == rhs.batch_count && lhs.height_extent == rhs.height_extent &&
               lhs.width_extent == rhs.width_extent && lhs.channel_count == rhs.channel_count;
    }

    static bool B2SSameCrop(const B2SCropBox &lhs, const B2SCropBox &rhs) {
        return lhs.trim_top == rhs.trim_top && lhs.trim_bottom == rhs.trim_bottom &&
               lhs.trim_left == rhs.trim_left && lhs.trim_right == rhs.trim_right;
    }

    static bool B2SNoCrop(const B2SCropBox &crop) {
        return crop.trim_top == 0U && crop.trim_bottom == 0U && crop.trim_left == 0U && crop.trim_right == 0U;
    }

    static bool B2SMatchPlainR2(const B2SPlanContract &plan, const B2SR2PlainCase &profile) {
        return plan.block == 2U && B2SNoCrop(plan.crop) &&
               plan.source.height_extent == profile.source_h &&
               plan.source.width_extent == profile.source_w &&
               plan.source.channel_count == profile.channel_count &&
               plan.result.batch_count == profile.result_n &&
               plan.result.height_extent == profile.source_h * 2U &&
               plan.result.width_extent == profile.source_w * 2U &&
               plan.result.channel_count == profile.channel_count &&
               plan.source.batch_count == profile.result_n * 4U;
    }

    static bool B2SMatchExactCase(const B2SPlanContract &plan, const B2SExactCase &profile) {
        return plan.block == profile.block &&
               B2SSameLayout(plan.source, profile.source) &&
               B2SSameLayout(plan.result, profile.result) &&
               B2SSameCrop(plan.crop, profile.crop);
    }

    static uint64_t B2SSelectSchedule(const B2SPlanContract &plan, ge::DataType dtype) {
        static const B2SR2PlainCase kR2Profiles[] = {
            {ge::DT_FLOAT, 28U, 28U, 128U, 2U, B2S_SCHEDULE_F32_R2_ROW_28X28_C128},
            {ge::DT_FLOAT, 14U, 14U, 64U, 4U, B2S_SCHEDULE_F32_R2_ROW_14X14_C64},
            {ge::DT_FLOAT, 4U, 6U, 32U, 5U, B2S_SCHEDULE_F32_R2_ROW_4X6_C32},
            {ge::DT_FLOAT16, 2U, 2U, 4096U, 1U, B2S_SCHEDULE_F16_R2_CHANNEL_TILE_C4096},
            {ge::DT_FLOAT16, 1U, 1U, 16384U, 1U, B2S_SCHEDULE_F16_R2_FLAT_C16384},
            {ge::DT_FLOAT16, 10U, 512U, 256U, 1U, B2S_SCHEDULE_F16_R2_WIDE_ROW_C256},
            {ge::DT_FLOAT16, 1024U, 6U, 32U, 4U, B2S_SCHEDULE_F16_R2_TALL_ROW_C32},
        };
        static const B2SExactCase kCropProfiles[] = {
            {
                ge::DT_FLOAT,
                {4U, 10U, 15U, 5U},
                {1U, 17U, 26U, 5U},
                {2U, 1U, 3U, 1U},
                2U,
                B2S_SCHEDULE_F32_R2_CROP_SMALL_C5,
            },
            {
                ge::DT_FLOAT16,
                {4U, 128U, 128U, 65U},
                {1U, 254U, 254U, 65U},
                {1U, 1U, 1U, 1U},
                2U,
                B2S_SCHEDULE_F16_R2_CROP_DENSE_C65,
            },
            {
                ge::DT_FLOAT16,
                {16U, 10U, 512U, 64U},
                {1U, 40U, 1535U, 64U},
                {0U, 0U, 513U, 0U},
                4U,
                B2S_SCHEDULE_F16_R4_LEFT_SHIFT_C64,
            },
        };

        for (const B2SR2PlainCase &profile : kR2Profiles) {
            if (profile.dtype == dtype && B2SMatchPlainR2(plan, profile)) {
                return profile.schedule;
            }
        }
        for (const B2SExactCase &profile : kCropProfiles) {
            if (profile.dtype == dtype && B2SMatchExactCase(plan, profile)) {
                return profile.schedule;
            }
        }
        return B2S_SCHEDULE_GENERIC;
    }

    static bool B2SQueryElementBytes(ge::DataType dtype, uint32_t &bytes) {
        bytes = 0U;
        ge::TypeUtils::GetDataTypeLength(dtype, bytes);
        return bytes != 0U;
    }

    static uint64_t B2SDivCeil(uint64_t value, uint64_t unit) {
        return unit == 0U ? 0U : (value + unit - 1U) / unit;
    }

    static uint64_t B2SMaxU64(uint64_t left, uint64_t right) {
        return left > right ? left : right;
    }

    static uint64_t B2SOutputPixelCount(const B2SPlanContract &plan) {
        return static_cast<uint64_t>(plan.result.batch_count) *
               plan.result.height_extent * plan.result.width_extent;
    }

    static uint32_t B2SResolveRuntimeQuota(platform_ascendc::PlatformAscendC &platform,
                                           uint32_t element_bytes) {
        uint64_t ub_bytes = 0U;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_bytes);

        uint64_t budget_bytes = ub_bytes == 0U ? kB2SDefaultRuntimeBytes : ub_bytes / 2U;
        if (budget_bytes > kB2SDefaultRuntimeBytes) {
            budget_bytes = kB2SDefaultRuntimeBytes;
        }
        budget_bytes = (budget_bytes / kB2SBlockTransferBytes) * kB2SBlockTransferBytes;
        if (budget_bytes < kB2SBlockTransferBytes) {
            budget_bytes = kB2SBlockTransferBytes;
        }

        const uint32_t quota = static_cast<uint32_t>(budget_bytes) / element_bytes;
        return quota == 0U ? 1U : quota;
    }

    static uint64_t B2SRuntimeRowPhaseTasks(const B2SPlanContract &plan) {
        return static_cast<uint64_t>(plan.result.batch_count) *
               plan.result.height_extent * plan.block;
    }

    static B2SRuntimeChoice B2SChooseRuntimePath(const B2SPlanContract &plan,
                                                 uint32_t element_bytes,
                                                 uint32_t element_quota) {
        const uint64_t output_pixels = B2SOutputPixelCount(plan);
        const uint64_t output_elements = output_pixels * plan.result.channel_count;
        B2SRuntimeChoice choice = {
            B2S_RUNTIME_ROW_PHASE,
            B2SRuntimeRowPhaseTasks(plan),
            element_quota,
        };

        if (plan.block == 1U &&
            plan.result.height_extent == plan.source.height_extent &&
            plan.result.width_extent == plan.source.width_extent) {
            choice.path = B2S_RUNTIME_FLAT_COPY;
            choice.task_count = output_elements;
            return choice;
        }

        if (plan.result.width_extent <= plan.block) {
            uint32_t channel_task_quota = element_quota;
            if (channel_task_quota > kB2SChannelTaskElements) {
                channel_task_quota = kB2SChannelTaskElements;
            }
            if (channel_task_quota == 0U) {
                channel_task_quota = 1U;
            }

            if (output_pixels <= kB2SPixelSplitLimit &&
                plan.result.channel_count >= channel_task_quota * 2U) {
                choice.path = B2S_RUNTIME_PIXEL_CHANNEL;
                choice.task_count =
                    output_pixels * B2SDivCeil(plan.result.channel_count, channel_task_quota);
                return choice;
            }

            choice.path = B2S_RUNTIME_PIXEL;
            choice.task_count = output_pixels;
            return choice;
        }

        if (plan.result.width_extent <= kB2SVerticalWidthLimit &&
            plan.result.height_extent >= kB2SVerticalMinHeight) {
            const uint64_t rows_per_phase =
                B2SDivCeil(plan.result.height_extent, plan.block);
            const uint64_t row_task_count =
                B2SDivCeil(rows_per_phase, kB2SVerticalRowsPerTask);
            choice.path = B2S_RUNTIME_VERTICAL;
            choice.task_count =
                static_cast<uint64_t>(plan.result.batch_count) *
                plan.result.width_extent * plan.block * row_task_count;
            return choice;
        }

        const uint64_t channel_bytes =
            static_cast<uint64_t>(plan.result.channel_count) * element_bytes;
        if (plan.result.width_extent >= kB2SPackedRowMinWidth &&
            channel_bytes % kB2SBlockTransferBytes == 0U) {
            uint64_t segment_elements = kB2SPackedRowElements;
            if (segment_elements > element_quota) {
                segment_elements = element_quota;
            }
            uint64_t segment_columns = segment_elements / plan.result.channel_count;
            if (segment_columns == 0U) {
                segment_columns = 1U;
            }
            if (segment_columns > plan.block) {
                choice.path = B2S_RUNTIME_PACKED_ROW;
                choice.task_count =
                    static_cast<uint64_t>(plan.result.batch_count) *
                    plan.result.height_extent *
                    B2SDivCeil(plan.result.width_extent, segment_columns);
                return choice;
            }
        }

        if (B2SRuntimeRowPhaseTasks(plan) <= kB2SWidePixelRowGroupLimit &&
            plan.result.width_extent >= kB2SWidePixelWidthLimit &&
            plan.result.channel_count <= kB2SWidePixelDepthLimit) {
            choice.path = B2S_RUNTIME_WIDE_PIXEL;
            choice.task_count = output_pixels;
            return choice;
        }

        return choice;
    }

    static uint32_t B2SClampCoreCount(uint32_t lhs, uint32_t rhs) {
        if (lhs == 0U) {
            return rhs == 0U ? 1U : rhs;
        }
        if (rhs == 0U) {
            return lhs;
        }
        return lhs < rhs ? lhs : rhs;
    }

    static uint32_t B2SResolveCoreCount(uint64_t schedule, uint32_t platform_cores,
                                       uint64_t fallback_tasks) {
        static const B2SCoreRule kCorePolicies[] = {
            {B2S_SCHEDULE_F32_R2_CROP_SMALL_C5, 16U, false},
            {B2S_SCHEDULE_F32_R2_ROW_14X14_C64, 16U, false},
            {B2S_SCHEDULE_F16_R2_CHANNEL_TILE_C4096, 8U, false},
            {B2S_SCHEDULE_F16_R2_FLAT_C16384, 8U, false},
            {B2S_SCHEDULE_F16_R2_WIDE_ROW_C256, 40U, true},
        };

        const uint32_t usable_cores = platform_cores == 0U ? 1U : platform_cores;
        for (const B2SCoreRule &policy : kCorePolicies) {
            if (policy.schedule == schedule) {
                return policy.fixed ? policy.cores : B2SClampCoreCount(usable_cores, policy.cores);
            }
        }

        if (schedule == B2S_SCHEDULE_GENERIC) {
            if (fallback_tasks > 0U && fallback_tasks < usable_cores) {
                return static_cast<uint32_t>(fallback_tasks);
            }
        }
        return usable_cores;
    }

    static bool B2SOutputShapeMatches(const gert::Shape &actual, const B2SLayout4D &expected) {
        return actual.GetDimNum() == 4 &&
               actual.GetDim(0) == static_cast<int64_t>(expected.batch_count) &&
               actual.GetDim(1) == static_cast<int64_t>(expected.height_extent) &&
               actual.GetDim(2) == static_cast<int64_t>(expected.width_extent) &&
               actual.GetDim(3) == static_cast<int64_t>(expected.channel_count);
    }

    static void B2SWriteDispatchPlan(const B2SPlanContract &checked,
                                     const B2SRuntimeChoice &runtime_choice,
                                     B2SDispatchPlan &plan) {
        plan.input = checked.source;
        plan.output = checked.result;
        plan.crop = checked.crop;
        plan.block_extent = checked.block;
        plan.element_total = static_cast<uint32_t>(
            static_cast<uint64_t>(checked.result.batch_count) *
            checked.result.height_extent *
            checked.result.width_extent *
            checked.result.channel_count);
        plan.buffer_element_quota = runtime_choice.element_quota;
        plan.runtime_path = runtime_choice.path;
    }

    static ge::graphStatus B2SPrepareTiling(gert::TilingContext *context) {
        const gert::Tensor *source_tensor = context->GetRequiredInputTensor(0);
        const gert::StorageShape *result_shape_holder = context->GetOutputShape(0);
        if (source_tensor == nullptr || result_shape_holder == nullptr) {
            return ge::GRAPH_FAILED;
        }

        B2SShape64 source64 = {};
        B2SCrop64 crop64 = {};
        uint64_t block64 = 0U;
        B2SPlanContract checked = {};
        if (!B2SReadShape64(source_tensor->GetStorageShape(), source64) ||
            !B2SReadAttrs(context->GetAttrs(), crop64, block64) ||
            !B2SBuildContract(source64, crop64, block64, checked) ||
            !B2SOutputShapeMatches(result_shape_holder->GetStorageShape(), checked.result)) {
            return ge::GRAPH_FAILED;
        }

        const ge::DataType dtype = source_tensor->GetDataType();
        if (dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT) {
            return ge::GRAPH_FAILED;
        }

        uint32_t element_bytes = 0U;
        if (!B2SQueryElementBytes(dtype, element_bytes)) {
            return ge::GRAPH_FAILED;
        }

        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        const uint32_t runtime_quota = B2SResolveRuntimeQuota(platform, element_bytes);
        const B2SRuntimeChoice runtime_choice =
            B2SChooseRuntimePath(checked, element_bytes, runtime_quota);

        const uint64_t schedule = B2SSelectSchedule(checked, dtype);
        uint32_t DT_X = static_cast<uint32_t>(dtype);
        ASCENDC_TPL_SEL_PARAM(context, DT_X, schedule);

        B2SDispatchPlan *kernel_plan = context->GetTilingData<B2SDispatchPlan>();
        if (kernel_plan == nullptr) {
            return ge::GRAPH_FAILED;
        }
        B2SWriteDispatchPlan(checked, runtime_choice, *kernel_plan);

        const uint32_t aiv_cores = platform.GetCoreNumAiv() > 0 ?
            static_cast<uint32_t>(platform.GetCoreNumAiv()) : 1U;
        context->SetBlockDim(B2SResolveCoreCount(schedule, aiv_cores, runtime_choice.task_count));

        size_t *workspace = context->GetWorkspaceSizes(1);
        workspace[0] = 0U;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus B2SInferShape(gert::InferShapeContext *context) {
        const gert::Shape *source_shape = context->GetInputShape(0);
        gert::Shape *result_shape = context->GetOutputShape(0);
        if (source_shape == nullptr || result_shape == nullptr) {
            return GRAPH_FAILED;
        }

        optiling::B2SShape64 source64 = {};
        optiling::B2SCrop64 crop64 = {};
        uint64_t block64 = 0U;
        optiling::B2SPlanContract checked = {};
        if (!optiling::B2SReadShape64(*source_shape, source64) ||
            !optiling::B2SReadAttrs(context->GetAttrs(), crop64, block64) ||
            !optiling::B2SBuildContract(source64, crop64, block64, checked)) {
            return GRAPH_FAILED;
        }

        result_shape->SetDimNum(0);
        result_shape->AppendDim(static_cast<int64_t>(checked.result.batch_count));
        result_shape->AppendDim(static_cast<int64_t>(checked.result.height_extent));
        result_shape->AppendDim(static_cast<int64_t>(checked.result.width_extent));
        result_shape->AppendDim(static_cast<int64_t>(checked.result.channel_count));
        return GRAPH_SUCCESS;
    }

    static graphStatus B2SInferType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class BatchToSpace : public OpDef {
    public:
        explicit BatchToSpace(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("crops").AttrType(REQUIRED).ListInt();
            this->Attr("block_size").AttrType(REQUIRED).Int();
            this->SetInferShape(ge::B2SInferShape)
                .SetInferDataType(ge::B2SInferType);
            this->AICore()
                .SetTiling(optiling::B2SPrepareTiling)
                .AddConfig("ascend910b")
                .AddConfig("ascend910_93");
        }
    };

    OP_ADD(BatchToSpace);
}  // namespace ops
