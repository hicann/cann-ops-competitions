#include "kernel_operator.h"
#include <type_traits> 

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T> struct AtanhTraits { static constexpr bool isNative = false; };

template <> struct AtanhTraits<float> {
    using CalcT = float;
    static constexpr bool isNative = true;
};
template <> struct AtanhTraits<half> {
    using CalcT = half;
    static constexpr bool isNative = true;
};

template <> struct AtanhTraits<bfloat16_t> {
    using CalcT = float;
    static constexpr bool isNative = false;
    static constexpr RoundMode modeIn = RoundMode::CAST_NONE;
    static constexpr RoundMode modeOut = RoundMode::CAST_ROUND;
};

template <> struct AtanhTraits<int16_t> {
    using CalcT = float;
    static constexpr bool isNative = false;
    static constexpr RoundMode modeIn = RoundMode::CAST_NONE;
    static constexpr RoundMode modeOut = RoundMode::CAST_ROUND;
};
template <> struct AtanhTraits<int32_t> {
    using CalcT = float;
    static constexpr bool isNative = false;
    static constexpr RoundMode modeIn = RoundMode::CAST_NONE;
    static constexpr RoundMode modeOut = RoundMode::CAST_ROUND;
};

template <> struct AtanhTraits<int8_t> {
    using CalcT = half;
    static constexpr bool isNative = false;
    static constexpr RoundMode modeIn = RoundMode::CAST_NONE;
    static constexpr RoundMode modeOut = RoundMode::CAST_NONE;
};
template <> struct AtanhTraits<uint8_t> {
    using CalcT = half;
    static constexpr bool isNative = false;
    static constexpr RoundMode modeIn = RoundMode::CAST_NONE;
    static constexpr RoundMode modeOut = RoundMode::CAST_NONE;
};

template<typename T>
class KernelAtanh {
public:
    __aicore__ inline KernelAtanh() {}
    
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t size,
                                uint32_t smallCoreDataNum, uint32_t finalSmallTileNum, 
                                uint32_t tileDataNum, uint32_t smallTailDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t bigTailDataNum, uint32_t tailBlockNum) {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
        uint32_t coreId = GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * coreId;
        this->tileDataNum = tileDataNum;

        if(coreId < tailBlockNum){
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        }else{
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (coreId - tailBlockNum);
        }
        inputGm.SetGlobalBuffer((__gm__ T *)input + globalBufferIndex, this->coreDataNum);
        outGm.SetGlobalBuffer((__gm__ T *)output + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueueInput, BUFFER_NUM, ((this->tileDataNum * sizeof(T))));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, ((this->tileDataNum * sizeof(T))));

        if constexpr (!AtanhTraits<T>::isNative) {
            using CalcT = typename AtanhTraits<T>::CalcT;
            pipe.InitBuffer(calcBufX, this->tileDataNum * sizeof(CalcT));
            pipe.InitBuffer(calcBufY, this->tileDataNum * sizeof(CalcT));
        }
    }

    __aicore__ inline void process() {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++) {
            if (i == this->tileNum - 1) {
                this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<T> xLocal = inQueueInput.AllocTensor<T>();
        DataCopy(xLocal, inputGm[progress * this->tileDataNum], this->processDataNum);
        inQueueInput.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<T> xLocal = inQueueInput.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        
        if constexpr (AtanhTraits<T>::isNative) {
            // 修正：彻底替换为 Adds 和 Muls
            Adds(yLocal, xLocal, (T)1.0, this->processDataNum);
            Muls(xLocal, xLocal, (T)-1.0, this->processDataNum);
            Adds(xLocal, xLocal, (T)1.0, this->processDataNum);
            Div(yLocal, yLocal, xLocal, this->processDataNum);
            Ln(yLocal, yLocal, this->processDataNum);
            Muls(yLocal, yLocal, (T)0.5, this->processDataNum);
        } else {
            using CalcT = typename AtanhTraits<T>::CalcT;
            LocalTensor<CalcT> xCalc = calcBufX.Get<CalcT>();
            LocalTensor<CalcT> yCalc = calcBufY.Get<CalcT>();

            Cast(xCalc, xLocal, AtanhTraits<T>::modeIn, this->processDataNum);

            // 修正：彻底替换为 Adds 和 Muls
            Adds(yCalc, xCalc, (CalcT)1.0, this->processDataNum);
            Muls(xCalc, xCalc, (CalcT)-1.0, this->processDataNum);
            Adds(xCalc, xCalc, (CalcT)1.0, this->processDataNum);
            Div(yCalc, yCalc, xCalc, this->processDataNum);
            Ln(yCalc, yCalc, this->processDataNum);
            Muls(yCalc, yCalc, (CalcT)0.5, this->processDataNum);

            Cast(yLocal, yCalc, AtanhTraits<T>::modeOut, this->processDataNum);
        }

        outQueueY.EnQue(yLocal);
        inQueueInput.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopy(outGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueInput;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueY;

    TBuf<TPosition::VECCALC> calcBufX;
    TBuf<TPosition::VECCALC> calcBufY;

    GlobalTensor<T> inputGm;
    GlobalTensor<T> outGm;

    uint32_t tileDataNum;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tailDataNum;
    uint32_t smallCoreDataNum;
    uint32_t finalSmallTileNum;
    uint32_t smallTailDataNum;
    uint32_t bigCoreDataNum;
    uint32_t finalBigTileNum;
    uint32_t bigTailDataNum;
    uint32_t processDataNum;

};

#define ATANH_LAUNCH(DTYPE) \
    KernelAtanh<DTYPE> op; \
    op.Init(input, output, tiling_data.size, \
        tiling_data.smallCoreDataNum, tiling_data.finalSmallTileNum, \
        tiling_data.tileDataNum, tiling_data.smallTailDataNum, \
        tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum, \
        tiling_data.bigTailDataNum, tiling_data.tailBlockNum); \
    op.process();

extern "C" __global__ __aicore__ void atanh(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    
    if (TILING_KEY_IS(1)) {
        ATANH_LAUNCH(float);
    } else if (TILING_KEY_IS(2)) {
        ATANH_LAUNCH(half);
    } else if (TILING_KEY_IS(3)) {
        ATANH_LAUNCH(bfloat16_t);
    } else if (TILING_KEY_IS(4)) {
        ATANH_LAUNCH(int32_t);
    } else if (TILING_KEY_IS(5)) {
        ATANH_LAUNCH(int16_t);
    } else if (TILING_KEY_IS(6)) {
        ATANH_LAUNCH(int8_t);
    } else if (TILING_KEY_IS(7)) {
        ATANH_LAUNCH(uint8_t);
    }
}