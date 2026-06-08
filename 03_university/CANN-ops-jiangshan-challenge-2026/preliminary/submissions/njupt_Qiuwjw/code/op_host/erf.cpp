// Host侧Tiling实现
#include <algorithm>
#include <cstdint>
#include <vector>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
    constexpr uint32_t BLOCK_TARGET = 2048;
    constexpr uint64_t UB_RESERVE_BYTES = 16 * 1024;
    constexpr uint32_t SINGLE_BUFFER_COUNT = 4;
    constexpr uint32_t DOUBLE_BUFFER_COUNT = 6;
    constexpr uint32_t DOUBLE_BUFFER_MIN_LENGTH = 65536;
    constexpr uint32_t TINY_DIRECT_N = 64;
    constexpr uint32_t SMALL_FAST_LENGTH = 4096;
    constexpr uint32_t MID_NATIVE_MIN_LENGTH = 1024;
    constexpr uint32_t PATH_V13C_POLY_SMALL = 0;
    constexpr uint32_t PATH_TINY64_FAST = 1;
    constexpr uint32_t PATH_MID_NATIVE_REVERT = 2;
    constexpr uint32_t PATH_V13C_LARGE = 3;

    static uint64_t GetShapeSize(const gert::StorageShape *shape) {
        uint64_t length = 1;
        const gert::Shape &storageShape = shape->GetStorageShape();
        size_t dimNum = storageShape.GetDimNum();
        for (size_t i = 0; i < dimNum; ++i) {
            length *= static_cast<uint64_t>(storageShape.GetDim(i));
        }
        return length;
    }

    static uint32_t SelectTileLength(uint64_t ubSize, uint64_t totalLength, uint32_t dtypeSize) {
        if (totalLength == 0) {
            return 1;
        }
        if (totalLength <= SMALL_FAST_LENGTH) {
            uint64_t alignedLength = (totalLength + 7UL) / 8UL * 8UL;
            return static_cast<uint32_t>(std::max<uint64_t>(alignedLength, 8));
        }
        uint64_t usableUb = ubSize > UB_RESERVE_BYTES ? ubSize - UB_RESERVE_BYTES : ubSize;
        uint32_t bufferCount = totalLength >= DOUBLE_BUFFER_MIN_LENGTH ? DOUBLE_BUFFER_COUNT : SINGLE_BUFFER_COUNT;
        constexpr uint32_t candidateTiles[] = {8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 1};

        for (uint32_t candidate : candidateTiles) {
            uint32_t curTile = static_cast<uint32_t>(std::min<uint64_t>(candidate, totalLength));
            if (curTile >= 8) {
                curTile = curTile / 8 * 8;
            }
            if (curTile == 0) {
                continue;
            }
            uint64_t needUb = static_cast<uint64_t>(bufferCount) * curTile * dtypeSize;
            if (needUb <= usableUb) {
                return curTile;
            }
        }
        return 1;
    }

    static uint32_t Align32(uint32_t size) {
        return (size + 31U) / 32U * 32U;
    }

    static uint32_t CalcErfTmpSize(uint32_t tileLength, uint32_t dtypeSize) {
        uint32_t maxTmpSize = 0;
        uint32_t minTmpSize = 0;
        std::vector<int64_t> shapeVec{static_cast<int64_t>(tileLength)};
        ge::Shape shape(shapeVec);
        AscendC::GetErfMaxMinTmpSize(shape, dtypeSize, false, maxTmpSize, minTmpSize);
        uint32_t tmpSize = Align32(minTmpSize);
        return tmpSize == 0 ? 32U : tmpSize;
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
        uint32_t dtypeSize = dtype_size_x > 0 ? static_cast<uint32_t>(dtype_size_x) : sizeof(float);
        uint64_t length_x = GetShapeSize(context->GetInputShape(0)); // 获取元素个数
        uint32_t size_x = tensor_x->GetSize(); // 获取内存大小
        (void)size_x;
        // 示例: 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        // 示例: 计算tiling方案并填充tiling结构体
        uint32_t tileLength = SelectTileLength(ub_size, length_x, dtypeSize);
        uint32_t tmpSize = 0;
        uint32_t pathType = PATH_V13C_LARGE;
        uint32_t blockDim = 1;
        if (length_x > 0 && length_x <= TINY_DIRECT_N) {
            blockDim = 1;
            pathType = PATH_TINY64_FAST;
        } else if (length_x > MID_NATIVE_MIN_LENGTH && length_x <= SMALL_FAST_LENGTH) {
            blockDim = 1;
            tmpSize = CalcErfTmpSize(tileLength, dtypeSize);
            pathType = PATH_MID_NATIVE_REVERT;
        } else if (length_x > 0 && length_x <= SMALL_FAST_LENGTH) {
            blockDim = 1;
            pathType = PATH_V13C_POLY_SMALL;
        } else if (length_x > 0 && num_cores_aiv > 0) {
            pathType = PATH_V13C_LARGE;
            uint64_t blockNumByLoad = (length_x + BLOCK_TARGET - 1) / BLOCK_TARGET;
            uint64_t maxUsefulBlockNum = std::min<uint64_t>(static_cast<uint64_t>(num_cores_aiv), length_x);
            blockDim = static_cast<uint32_t>(
                std::max<uint64_t>(1, std::min<uint64_t>(blockNumByLoad, maxUsefulBlockNum)));
        }
        uint64_t blockLength = length_x == 0 ? 1 : (length_x + blockDim - 1) / blockDim;

        ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
        tiling->totalLength = length_x;
        tiling->blockLength = blockLength;
        tiling->tileLength = tileLength;
        tiling->tmpSize = tmpSize;
        tiling->blockDim = blockDim;
        tiling->pathType = pathType;
        // 配置启动核数
        context->SetBlockDim(blockDim);
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
                .AddConfig("ascend910b")
                .AddConfig("ascend910_93");
        }
    };
    OP_ADD(Erf);
}  // namespace ops
