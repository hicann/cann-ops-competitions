#include "kernel_operator.h"
#include <type_traits>

class KernelFills {
public:
    __aicore__ inline KernelFills() {}

    __aicore__ inline void Init(GM_ADDR output, uint32_t totalLength, uint32_t tileLength, uint32_t smallCoreLength,
                                uint32_t incCoreLength, uint32_t formerNum, float value)
    {
        this->tileLength = tileLength;
        this->value = value;

        const uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx < formerNum) {
            this->coreLength = smallCoreLength + incCoreLength;
            this->coreOffset = this->coreLength * blockIdx;
        } else {
            this->coreLength = smallCoreLength;
            this->coreOffset = (smallCoreLength + incCoreLength) * formerNum + smallCoreLength * (blockIdx - formerNum);
        }

        if (this->coreOffset >= totalLength || this->coreLength == 0U) {
            this->coreLength = 0U;
            this->tileNum = 0U;
            this->tailLength = 0U;
            return;
        }
        if (this->coreOffset + this->coreLength > totalLength) {
            this->coreLength = totalLength - this->coreOffset;
        }

        this->tileNum = (this->coreLength + tileLength - 1U) / tileLength;
        this->tailLength = this->coreLength - (this->tileNum - 1U) * tileLength;

        outputGm.SetGlobalBuffer((__gm__ DTYPE_OUTPUT *)output + this->coreOffset, this->coreLength);
    }

    __aicore__ inline void Process()
    {
        if (this->coreLength == 0U) {
            return;
        }

        AscendC::LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
        static constexpr uint32_t STATIC_TILE_LENGTH =
            (std::is_same_v<DTYPE_OUTPUT, float> || std::is_same_v<DTYPE_OUTPUT, int32_t>) ? 16384U :
            (std::is_same_v<DTYPE_OUTPUT, half> || std::is_same_v<DTYPE_OUTPUT, bfloat16_t>) ? 24576U :
            std::is_same_v<DTYPE_OUTPUT, int16_t> ? 32768U :
                                                    49152U; // int8/uint8
        AscendC::LocalTensor<DTYPE_OUTPUT> outLocal = ubAllocator.Alloc<DTYPE_OUTPUT, STATIC_TILE_LENGTH>();

        AscendC::LocalTensor<float> tmpFloatLocal;
        AscendC::LocalTensor<half> tmpHalfLocal;
        AscendC::LocalTensor<float> *tmpFloatPtr = nullptr;
        AscendC::LocalTensor<half> *tmpHalfPtr = nullptr;

        if constexpr (std::is_same_v<DTYPE_OUTPUT, bfloat16_t>) {
            tmpFloatLocal = ubAllocator.Alloc<float, STATIC_TILE_LENGTH>();
            tmpFloatPtr = &tmpFloatLocal;
        }
        if constexpr (std::is_same_v<DTYPE_OUTPUT, int8_t> || std::is_same_v<DTYPE_OUTPUT, uint8_t>) {
            tmpHalfLocal = ubAllocator.Alloc<half, STATIC_TILE_LENGTH>();
            tmpHalfPtr = &tmpHalfLocal;
        }

        // Static Tensor programming style: fill once and reuse for all tiles in one core.
        FillLocal<DTYPE_OUTPUT>(outLocal, this->tileLength, tmpFloatPtr, tmpHalfPtr);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        for (uint32_t i = 0; i < this->tileNum; ++i) {
            const uint32_t length = (i + 1U == this->tileNum) ? this->tailLength : this->tileLength;
            CopyOut(i, length, outLocal);
        }
    }

private:
    template <typename T>
    __aicore__ inline void FillLocal(AscendC::LocalTensor<T> &outLocal, uint32_t length,
                                     AscendC::LocalTensor<float> *tmpFloatLocal,
                                     AscendC::LocalTensor<half> *tmpHalfLocal)
    {
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            // Match torch.full_like integer conversion: truncate toward zero.
            const int32_t truncI32 = static_cast<int32_t>(value);
            const T fillValue = static_cast<T>(truncI32);
            const int32_t fillValueI32 = static_cast<int32_t>(fillValue);
            const half fillValueHalf = static_cast<half>(static_cast<float>(fillValueI32));
            AscendC::Duplicate(*tmpHalfLocal, fillValueHalf, length);
            AscendC::Cast(outLocal, *tmpHalfLocal, AscendC::RoundMode::CAST_NONE, length);
        } else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t>) {
            const int32_t truncI32 = static_cast<int32_t>(value);
            const T fillValue = static_cast<T>(truncI32);
            AscendC::Duplicate(outLocal, fillValue, length);
        } else if constexpr (std::is_same_v<T, float>) {
            AscendC::Duplicate(outLocal, static_cast<float>(value), length);
        } else if constexpr (std::is_same_v<T, half>) {
            AscendC::Duplicate(outLocal, static_cast<half>(value), length);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            AscendC::Duplicate(*tmpFloatLocal, value, length);
            AscendC::Cast(outLocal, *tmpFloatLocal, AscendC::RoundMode::CAST_RINT, length);
        }
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx, uint32_t length, AscendC::LocalTensor<DTYPE_OUTPUT> &outLocal)
    {
        if (length == tileLength) {
            AscendC::DataCopy(outputGm[tileIdx * tileLength], outLocal, tileLength);
            return;
        }
        AscendC::DataCopyExtParams copyParams = {1, static_cast<uint32_t>(length * sizeof(DTYPE_OUTPUT)), 0, 0, 0};
        AscendC::DataCopyPad(outputGm[tileIdx * tileLength], outLocal, copyParams);
    }

private:
    AscendC::GlobalTensor<DTYPE_OUTPUT> outputGm;
    uint32_t tileLength = 0;
    uint32_t coreOffset = 0;
    uint32_t coreLength = 0;
    uint32_t tileNum = 0;
    uint32_t tailLength = 0;
    float value = 0.0F;
};

extern "C" __global__ __aicore__ void fills(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelFills op;
    op.Init(output, tilingData.totalLength, tilingData.tileLength, tilingData.smallCoreLength, tilingData.incCoreLength,
            tilingData.formerNum, tilingData.value);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void fills_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *input, uint8_t *output, uint8_t *workspace,
              uint8_t *tiling)
{
    fills<<<blockDim, l2ctrl, stream>>>(input, output, workspace, tiling);
}
#endif
