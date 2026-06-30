#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <vector>
#include <tuple>
#include <utility>
#include <type_traits>
using std::max;
using std::min;
using namespace ge;

// simple algorithms
#define ALIGN32(mem) ((mem) / 32u * 32u)
#define CEIL(x, align_num) (((x) + (align_num) - 1) / (align_num) * (align_num))
#define FLOOR(x, align_num) ((x) / (align_num) * (align_num))

// tiling struct operations
#define SET(param) tiling.set_##param(param)
#define _SAVE_TI(context, tiling)                                                                               \
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());    \
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize())
#define SAVE_TI(tiling) _SAVE_TI(context, tiling)
#define SAVE SAVE_TI(tiling)

// debug output ctrls
#ifndef DEBUG_OUTPUT
#define DEBUG_OUTPUT 0
#endif
#if DEBUG_OUTPUT
#define PRINTF(fmt, ...) printf(fmt, ##__VA_ARGS__)
#define TI_PRINT(fmt, ...) printf("tilingFunc: " fmt, ##__VA_ARGS__)
#define TI_PRINTLN(fmt, ...) printf("tilingFunc: " fmt "\n", ##__VA_ARGS__)
#else 
#define PRINTF(fmt, ...)
#define TI_PRINT(fmt, ...)
#define TI_PRINTLN(fmt, ...)
#endif


namespace optiling {
    template<class Tp>
    inline void printShape(const char *prefixString, Tp arr, const size_t size) {
        TI_PRINT("%s", prefixString);
        for(int i = 0; i < size; i++) 
            PRINTF("%u%s", *(arr + i), ", \0]\n" + (i + 1 == size ? 3 : 0));
    }
    template<class Tp>
    inline void printShape(Tp arr, const size_t size) {
        TI_PRINT("[");
        for(int i = 0; i < size; i++) 
            PRINTF("%u%s", *(arr + i), ", \0]\n" + (i + 1 == size ? 3 : 0));
    }
    template<class Tp>
    inline void printShapes(const size_t shapeSize, const Tp &shape) {
        printShape(shape, shapeSize);
    }
    template<class Tp, class... Tps>
    inline void printShapes(const size_t shapeSize, const Tp &shape, const Tps&... shapes) {
        printShape(shape, shapeSize);
        printShapes(shapeSize, shapes...);
    }

    enum class TensorType {IN, OUT, ERR};
    /**
     * @brief get sizeof dataType with given index
     * @tparam tt enumerator of TensorType {IN, OUT, ...}
     * @param context tiling context pointer
     * @param index Tensor index
     * @return sizeof(Tensor[index].DataType)
     */
    template<TensorType tt = TensorType::IN>
    inline int getDTSize(gert::TilingContext* context, const size_t index) {
        if constexpr(tt == TensorType::IN) 
            return GetSizeByDataType(context->GetInputTensor(index)->GetDataType());
        else if constexpr(tt == TensorType::OUT) 
            return GetSizeByDataType(context->GetOutputDesc(index)->GetDataType());
        else return 0;
    }
    /**
     * @brief get input shape ptr and length
     * @param context tiling context pointer
     * @param index InputTensor index
     * @return a `std::pair` contains shape ptr and length
     * @author LinuxKiller
     * @version 1.0
     */
    inline std::pair<const gert::StorageShape *, uint32_t> getInputShape(gert::TilingContext* context, const size_t index) {
        const auto shape = context->GetInputShape(index);
        uint32_t length = 1;
        for (int i = 0; i < shape->GetStorageShape().GetDimNum(); i++) length *= shape->GetStorageShape().GetDim(i);
        return std::make_pair(shape, length);
    }
    /**
     * @brief get output shape ptr and length
     * @param context tiling context pointer
     * @param index OutputTensor index
     * @return a `std::pair` contains shape ptr and length
     * @author LinuxKiller
     * @version 1.0
     */
    inline std::pair<const gert::StorageShape *, uint32_t> getOutputShape(gert::TilingContext* context, const size_t index) {
        const auto shape = context->GetOutputShape(index);
        uint32_t length = 1;
        for (int i = 0; i < shape->GetStorageShape().GetDimNum(); i++) length *= shape->GetStorageShape().GetDim(i);
        return std::make_pair(shape, length);
    }
    /**
     * @brief get Unified Buffer size
     * @param context tiling context pointer
     * @return Unified Buffer Size(in bytes)
     * @author LinuxKiller
     * @version 1.0
     */
    inline auto getUBSize(gert::TilingContext* context) -> uint64_t {
        uint64_t ubSize{};
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo()).GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        return ubSize;
    }

    /**
     * @brief Print Datatype to STDOUT
     * @param dt datatype enumerator `ge::DataType`
     * @author LinuxKiller
     * @version 1.0
     */
    inline void printDataType(DataType dt) {
        switch(dt) {
            case DT_UINT64:
                TI_PRINTLN("DT = uint64");
                break;
            case DT_INT64:
                TI_PRINTLN("DT = int64");
                break;
            case DT_DOUBLE:
                TI_PRINTLN("DT = float64");
                break;
            case DT_INT32:
                TI_PRINTLN("DT = int32");
                break;
            case DT_UINT32:
                TI_PRINTLN("DT = uint32");
                break;
            case DT_FLOAT:
                TI_PRINTLN("DT = float32");
                break;
            case DT_UINT16:
                TI_PRINTLN("DT = uint16");
                break;
            case DT_INT16:
                TI_PRINTLN("DT = int16");
                break;
            case DT_FLOAT16:
                TI_PRINTLN("DT = float16");
                break;
            case DT_BF16:
                TI_PRINTLN("DT = bf16");
                break;
            case DT_UINT8:
                TI_PRINTLN("DT = uint8");
                break;
            case DT_INT8:
                TI_PRINTLN("DT = int8");
                break;
            default:
                TI_PRINTLN("DT = UNKNOWN: %d", dt);
        }
    }
    /**
     * @brief print InputTensor[a, b) Shape Info to STDOUT
     * @tparam InputCount `start index`
     * @tparam InputMax `end index` + 1
     * @param context tiling context pointer
     * @author LinuxKiller
     * @version 1.0
     */
    template<size_t InputCount, size_t InputMax>
    constexpr inline void printInputShapeIt(gert::TilingContext* context) {
        if constexpr (InputCount < InputMax) {
            const auto shape = context->GetInputShape(InputCount)->GetStorageShape();
            TI_PRINT("Input[%lu] Shape: [", InputCount);
            for(int i = 0; i < shape.GetDimNum(); i++) 
                PRINTF("%ld%s", shape.GetDim(i), ", \0]\n" + (i == shape.GetDimNum() - 1 ? 3 : 0));
            printDataType(context->GetInputDesc(InputCount)->GetDataType());
            printInputShapeIt<InputCount + 1, InputMax>(context);
        }
    }
    /**
     * @brief print OutputTensor[a, b) Shape Info to STDOUT
     * @tparam OutputCount `start index`
     * @tparam OutputMax `end index` + 1
     * @param context tiling context pointer
     * @author LinuxKiller
     * @version 1.0
     */
    template<size_t OutputCount, size_t OutputMax>
    constexpr inline void printOutputShapeIt(gert::TilingContext* context) {
        if constexpr (OutputCount < OutputMax) {
            const auto shape = context->GetOutputShape(OutputCount)->GetStorageShape();
            TI_PRINT("Output[%lu] Shape: [", OutputCount);
            for(int i = 0; i < shape.GetDimNum(); i++) 
                PRINTF("%ld%s", shape.GetDim(i), ", \0]\n" + (i == shape.GetDimNum() - 1 ? 3 : 0));
            printDataType(context->GetOutputDesc(OutputCount)->GetDataType());
            printOutputShapeIt<OutputCount + 1, OutputMax>(context);
        }
    }
    /**
     * @brief print DEBUG information to STDOUT, 
     * @brief including SOC Version, SOC attributes, Tensor Shapes
     * @tparam InputCount total input tensor number
     * @tparam OuputCount total output tensor number
     * @param context tiling context pointer
     * @author LinuxKiller
     * @version 1.0
     */
    template<int InputCount, int OutputCount>
    void printDebugInfo(gert::TilingContext* context) {
        using namespace platform_ascendc;
        auto dev_info = PlatformAscendC(context->GetPlatformInfo());
        switch (dev_info.GetSocVersion()) {
            case SocVersion::ASCEND910:
                TI_PRINTLN("SOC Version: Ascend910");
                break;
            case SocVersion::ASCEND910B:
                TI_PRINTLN("SOC Version: Ascend910B");
                break;
            case SocVersion::ASCEND310P:
                TI_PRINTLN("SOC Version: Ascend310P");
                break;
            case SocVersion::ASCEND310B:
                TI_PRINTLN("SOC Version: Ascend310B");
                break;
            default:
                TI_PRINTLN("Unknown SOC Version!");
        }
        TI_PRINTLN("Core#: %u, Aic#: %u, Aiv#: %u, Vec#: %u", dev_info.GetCoreNum(), dev_info.GetCoreNumAic(), dev_info.GetCoreNumAiv(), dev_info.GetCoreNumVector());
        uint64_t ubSize; dev_info.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        TI_PRINTLN("UB Size: %lu Bytes", ubSize);
        printInputShapeIt<0, InputCount>(context);
        printOutputShapeIt<0, OutputCount>(context);
    }
    /**
     * @brief calculate `tileLength`, `tileNumber`, and `reminder` with align 
     * @tparam Tp output type(same as typeof `batchLength`)
     * @tparam TypeSizeTp typeof `typeSize`
     * @tparam AlignTp typeof `batchAlignSize` and `reminderAlignSize`
     * @tparam bufferTp typeof `bufferLength`,usually same as typeof(`ubSize`)
     * @param bufferLength same as `tileLength` without align, 
     * @param batchLength usually `yLength` or `totalLength`, number of elements
     * @param typeSize size of `dataType` usually get by `ge::GetSizeByDataType(dt)`
     * @param batchAlignSize alignNumber of buffer(in `bytes`), common values: `32`(1*DB), `256`(8*DB), `0`(disabled)
     * @param reminderAlignSize alignNumber of reminder(in `bytes`), common values: `32`(1*DB), `0`(disabled)
     * @return a `tuple<Tp, Tp, Tp>` of tilingInfo, order: {`tileLength`, `tileNumber`, `reminder`}
     * @author LinuxKiller
     * @version 1.0
     */
    template<typename Tp, typename TypeSizeTp, typename AlignTp, typename BufferTp>
    inline std::tuple<Tp, Tp, Tp> getTilingInfo( const BufferTp bufferLength,
            const Tp batchLength, const TypeSizeTp typeSize, 
            const AlignTp batchAlignSize, const AlignTp reminderAlignSize) {
        Tp tileLength = bufferLength;
        if(batchAlignSize != 0) tileLength = FLOOR(tileLength, batchAlignSize / typeSize);
        Tp tileNumber = batchLength / tileLength;
        Tp reminder = batchLength % tileLength;
        if(reminderAlignSize != 0) reminder = CEIL(batchLength % tileLength, reminderAlignSize / typeSize);
        TI_PRINTLN("tileLength: %u, tileNum: %u, reminder: %u", tileLength, tileNumber, reminder);
        return {tileLength, tileNumber, reminder};
    }

    template<typename Tp>
    Tp max(const Tp &a, const Tp &b) {
        return std::max(a, b);
    }
    
    template<typename Tp, typename... Tps>
    Tp max(const Tp &a, const Tps& ...bs) {
        return max(a, max(bs...));
    }

    template<typename Tp>
    Tp min(const Tp &a, const Tp &b) {
        return std::min(a, b);
    }
    
    template<typename Tp, typename... Tps>
    Tp min(const Tp &a, const Tps& ...bs) {
        return min(a, min(bs...));
    }
    
    template<typename Tp>
    uint32_t set_remove(const vector<Tp*> &inputs, Tp* y, uint32_t size, uint32_t idx, uint32_t count, const Tp &val) {
        if (idx >= size) return size;
        for (Tp* arr : inputs) arr[idx] = val;
        y[idx] = val;
        idx += count;
        while (idx < size) {
            for (Tp* arr : inputs) {
                arr[idx - count + 1] = arr[idx];
            }
            y[idx - count + 1] = y[idx];
            idx++;
        }
        return size - count + 1;
    }
    
    template<class Tp>
    std::tuple<bool, uint32_t, uint32_t, Tp> check_arrs(const std::vector<Tp*> &inputs, const uint32_t size) {
        const size_t n = inputs.size();
        for (uint32_t i = 0; i < size - 1; ++i) {
            bool all_equal_i   = true;
            bool all_equal_ip1 = true;
            for (size_t j = 0; j < n - 1; ++j) {
                if (inputs[j][i] != inputs[j + 1][i]) all_equal_i = false;
                if (inputs[j][i + 1] != inputs[j + 1][i + 1]) all_equal_ip1 = false;
            }
            if (all_equal_i && all_equal_ip1) {
                return {true, i, 2, inputs[0][i] * inputs[0][i + 1]};
            }
        }
        return {false, 0, 0, Tp{}};
    }

    template<typename... Inputs, typename Output>
    uint32_t shrink_shapes(uint32_t shapeSize, Output &output, Inputs& ...inputs) {
        std::vector<Output> input_v = {inputs...};
        while(true) {
            const auto [flag, idx, count, val] = check_arrs(input_v, shapeSize);
            if(!flag) break;
            shapeSize = set_remove(input_v, output, shapeSize, idx, count, val);
        }
        return shapeSize;
    }

    template<typename TP, size_t idx>
    constexpr inline const TP *GetAttr(const gert::TilingContext* context) {
        return context->GetAttrs()->GetAttrPointer<TP>(idx);
    }
}