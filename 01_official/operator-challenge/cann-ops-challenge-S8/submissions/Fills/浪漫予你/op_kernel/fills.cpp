#include "kernel_operator.h"
using namespace AscendC;

#define TILE_BYTES 36*1024
#define BUFFER_NUM 1

template <typename T>
class KernelFills {
public:
    __aicore__ inline KernelFills() {}

    __aicore__ inline void Init(
        GM_ADDR z,
        uint16_t tilingIncSize,
        uint32_t smallSize,
        uint16_t formerNum,
        float value,
        uint32_t valuePack,
        AscendC::LocalTensor<T> zLocalO)
    {
        this->value = value;
        this->valuePack = valuePack;

        uint32_t srcBeginIndex = 0;
        uint32_t incSize = tilingIncSize;  // 修正：不要用 uint8_t
        uint32_t size;

        if (GetBlockIdx() < formerNum) {
            size = smallSize + incSize;
            srcBeginIndex = size * GetBlockIdx();
        } else {
            size = smallSize;
            srcBeginIndex = size * GetBlockIdx() + formerNum * incSize;
        }

        zGm.SetGlobalBuffer((__gm__ T *)z + srcBeginIndex, size);

        tileLength = min(uint32_t(size), uint32_t(TILE_BYTES / sizeof(T)));

        tileNum = (size + tileLength - 1) / tileLength;
        lastTileLength = size - (tileNum - 1) * tileLength;
        zLocal = zLocalO;
        if constexpr (std::is_same_v<T, half> ||
                      std::is_same_v<T, float> ||
                      std::is_same_v<T, int16_t> ||
                      std::is_same_v<T, int32_t>) {
            T valueT = T(value);
            Duplicate(zLocal, valueT, tileLength);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            LocalTensor<int32_t> z32 = zLocal.template ReinterpretCast<int32_t>();
            uint32_t wordCount = (tileLength + 1) / 2;
            Duplicate(z32, static_cast<int32_t>(valuePack), wordCount);
        } else {
            LocalTensor<int32_t> z32 = zLocal.template ReinterpretCast<int32_t>();
            uint32_t wordCount = (tileLength + 3) / 4;
            Duplicate(z32, static_cast<int32_t>(valuePack), wordCount);
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < tileNum - 1; ++i) {
            DataCopy(zGm[offset], zLocal, tileLength);
            offset += tileLength;
        }

        DataCopy(zGm[offset], zLocal, lastTileLength);
    }

private:
    float value;
    uint32_t valuePack;

    GlobalTensor<T> zGm;

    uint32_t tileLength;
    uint32_t lastTileLength;
    uint32_t tileNum;

    AscendC::LocalTensor<T> zLocal;
};

extern "C" __global__ __aicore__ void fills(
    GM_ADDR input,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    AscendC::LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
    GET_TILING_DATA(tiling_data, tiling);
    static constexpr int TILE_LENGTH =
        (std::is_same_v<DTYPE_X, float> ||
         std::is_same_v<DTYPE_X, int32_t>)
            ? (TILE_BYTES / 4)
            : (std::is_same_v<DTYPE_X, half> ||
               std::is_same_v<DTYPE_X, bfloat16_t> ||
               std::is_same_v<DTYPE_X, int16_t>)
                  ? (TILE_BYTES / 2)
                  : TILE_BYTES;

    AscendC::LocalTensor<DTYPE_X> zLocal =
        ubAllocator.Alloc<DTYPE_X, TILE_LENGTH>();
    KernelFills<DTYPE_X> op;
    op.Init(
        output,
        tiling_data.incSize,
        tiling_data.smallSize,
        tiling_data.formerNum,
        tiling_data.value,
        tiling_data.valuePack,
        zLocal);
    op.Process();
}