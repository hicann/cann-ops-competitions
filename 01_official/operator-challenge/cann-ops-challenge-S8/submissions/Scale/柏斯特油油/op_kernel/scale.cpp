#define K_MAX_SHAPE_DIM 0

#include "kernel_operator.h"

using namespace AscendC;

#define ll uint64_t

constexpr int32_t TQUE_DEPTH = 1;
constexpr uint64_t BUFFER_NUM = 1;

constexpr uint32_t REPEAT_SIZE = 256;
constexpr uint32_t MAX_REPEAT_TIME = 255;
constexpr uint32_t MAX_ROW = 96;

template<typename T> class Scale {
private:
    TPipe* pipe;
    TQue<TPosition::VECIN, TQUE_DEPTH> inQueueInput, inQueueScale, inQueueBias;
    TQue<TPosition::VECOUT, TQUE_DEPTH> outQueueOutput;
    TQue<TPosition::VECCALC, TQUE_DEPTH> calcQueue1, calcQueue2, calcQueue3;
    GlobalTensor<T> inputGm, scaleGm, biasGm, outputGm;
    ll preLength;
    ll midLength;
    ll postLength;
    ll totalLength;
    int blockIdx;
    int blockNum;
    int CALC_TYPE_SIZE;
    int NUM_PER_BLOCK;
    int CALC_NUM_PER_BLOCK;
    ll tileLength;
    ll postStep;
    ll midStep;

public:
    __aicore__ inline Scale() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, TPipe* pipeIn, ll preLength, ll midLength, ll postLength)
    {
        pipe = pipeIn;
        this->preLength = preLength;
        this->midLength = midLength;
        this->postLength = postLength;
        totalLength = preLength * midLength * postLength;

        blockIdx = GetBlockIdx();
        blockNum = GetBlockNum();
        
        CALC_TYPE_SIZE = sizeof(T);
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            CALC_TYPE_SIZE = sizeof(float);
        }
        NUM_PER_BLOCK = 32 / sizeof(T);
        CALC_NUM_PER_BLOCK = 32 / CALC_TYPE_SIZE;
        tileLength = 8192 * 2 / BUFFER_NUM;
        postStep = postLength > tileLength ? tileLength : postLength;
        midStep = tileLength / postStep;

        inputGm.SetGlobalBuffer((__gm__ T *)input, totalLength);
        scaleGm.SetGlobalBuffer((__gm__ T *)scale, midLength);
        biasGm.SetGlobalBuffer((__gm__ T *)bias, midLength);
        outputGm.SetGlobalBuffer((__gm__ T *)output, totalLength);
        inputGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        outputGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

        pipe->InitBuffer(inQueueInput, BUFFER_NUM, tileLength * sizeof(T));
        pipe->InitBuffer(outQueueOutput, BUFFER_NUM, tileLength * sizeof(T));
        pipe->InitBuffer(inQueueScale, BUFFER_NUM, midStep * 32);
        pipe->InitBuffer(inQueueBias, BUFFER_NUM, midStep * 32);
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            pipe->InitBuffer(calcQueue1, BUFFER_NUM, tileLength * sizeof(float));
            pipe->InitBuffer(calcQueue2, BUFFER_NUM, midStep * 32 * 2);
            pipe->InitBuffer(calcQueue3, BUFFER_NUM, midStep * 32 * 2);
        }
    }
    __aicore__ inline void Process()
    {
        #define min(x, y) ((x) < (y) ? (x) : (y))
        ll tasksPerPre = (midLength + midStep - 1) / midStep;
        ll totalTasks = preLength * tasksPerPre;
    
            for (ll taskId = blockIdx; taskId < totalTasks; taskId += blockNum) {
                ll preIdx = taskId / tasksPerPre;
                ll midChunkIdx = taskId % tasksPerPre;
        
                ll midBase = midChunkIdx * midStep;
                ll midRemain = midLength - midBase;
                ll midCalcLength = min(midStep, midRemain);
                ll midVecLength = midCalcLength * NUM_PER_BLOCK;
                DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                DataCopyExtParams midCopyParams = {
                    (uint16_t)(midCalcLength),
                    (uint32_t)(1 * sizeof(T)),
                    0,
                    0,
                    0
                };
                LocalTensor<T> scaleLocal, biasLocal;
                LocalTensor<float> scaleFloat, biasFloat;
                scaleLocal = inQueueScale.AllocTensor<T>();
                biasLocal = inQueueBias.AllocTensor<T>();
                DataCopyPad(scaleLocal, scaleGm[midBase], midCopyParams, padParams);
                DataCopyPad(biasLocal, biasGm[midBase], midCopyParams, padParams);
                inQueueScale.EnQue(scaleLocal);
                inQueueBias.EnQue(biasLocal);
                inQueueScale.DeQue<T>();
                inQueueBias.DeQue<T>();
                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    scaleFloat = calcQueue2.AllocTensor<float>();
                    biasFloat = calcQueue3.AllocTensor<float>();
                    Cast(scaleFloat, scaleLocal, RoundMode::CAST_NONE, midVecLength);
                    Cast(biasFloat, biasLocal, RoundMode::CAST_NONE, midVecLength);
                    inQueueScale.FreeTensor(scaleLocal);
                    inQueueBias.FreeTensor(biasLocal);
                }
                for (ll postBase = 0; postBase < postLength; postBase += postStep) {
                    ll postRemain = postLength - postBase;
                    ll postCalcLength = min(postStep, postRemain);
                    ll postCalcLengthAlign = (postCalcLength + NUM_PER_BLOCK - 1) / NUM_PER_BLOCK * NUM_PER_BLOCK;
                    ll totalCalcLengthAlign = midCalcLength * postCalcLengthAlign;
                    ll gmOffset = preIdx * midLength * postLength + midBase * postLength + postBase;
                    DataCopyExtParams copyParams = {
                        (uint16_t)(midCalcLength),
                        (uint32_t)(postCalcLength * sizeof(T)),
                        (uint32_t)((postLength - postCalcLength) * sizeof(T)),
                        0,
                        0
                    };
                    LocalTensor<T> inputLocal, outputLocal;
                    inputLocal = inQueueInput.AllocTensor<T>();
                    DataCopyPad(inputLocal, inputGm[gmOffset], copyParams, padParams);
                    inQueueInput.EnQue(inputLocal);
                    inQueueInput.DeQue<T>();
                    if constexpr (std::is_same_v<T, bfloat16_t>) {
                        LocalTensor<float> inputFloat;
                        inputFloat = calcQueue1.AllocTensor<float>();
                        outputLocal = outQueueOutput.AllocTensor<T>();
                        for (int i = 0; i < midCalcLength; i++) {
                            ll ubOffset = i * postCalcLengthAlign;
                            Cast(inputFloat[ubOffset], outputLocal[ubOffset], RoundMode::CAST_NONE, postCalcLength);
                            float scaleVal = scaleFloat.GetValue(i * NUM_PER_BLOCK);
                            Muls(inputFloat[ubOffset], inputFloat[ubOffset], scaleVal, postCalcLength);
                            Cast(outputLocal[ubOffset], inputFloat[ubOffset], RoundMode::CAST_RINT, postCalcLength);
                            Cast(inputFloat[ubOffset], outputLocal[ubOffset], RoundMode::CAST_NONE, postCalcLength);
                            float biasVal = biasFloat.GetValue(i * NUM_PER_BLOCK);
                            Adds(inputFloat[ubOffset], inputFloat[ubOffset], biasVal, postCalcLength);
                            Cast(outputLocal[ubOffset], inputFloat[ubOffset], RoundMode::CAST_RINT, postCalcLength);
                        }
                        inQueueInput.FreeTensor(inputLocal);
                        calcQueue1.FreeTensor(inputFloat);
                    } else {
                        outputLocal = outQueueOutput.AllocTensor<T>();
                        for (int i = 0; i < midCalcLength; i++) {
                            ll ubOffset = i * postCalcLengthAlign;
                            T scaleVal = scaleLocal.GetValue(i * NUM_PER_BLOCK);
                            Muls(outputLocal[ubOffset], inputLocal[ubOffset], scaleVal, postCalcLength);
                            T biasVal = biasLocal.GetValue(i * NUM_PER_BLOCK);
                            Adds(outputLocal[ubOffset], outputLocal[ubOffset], biasVal, postCalcLength);
                        }
                        inQueueInput.FreeTensor(inputLocal);
                    }
                    outQueueOutput.EnQue(outputLocal);
                    outQueueOutput.DeQue<T>();
                    copyParams = {
                        (uint16_t)(midCalcLength),
                        (uint32_t)(postCalcLength * sizeof(T)),
                        0,
                        (uint32_t)((postLength - postCalcLength) * sizeof(T)),
                        0
                    };
                    DataCopyPad(outputGm[gmOffset], outputLocal, copyParams);
                    outQueueOutput.FreeTensor(outputLocal);
                }
                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    calcQueue2.FreeTensor(scaleFloat);
                    calcQueue3.FreeTensor(biasFloat);
                } else {
                    inQueueScale.FreeTensor(scaleLocal);
                    inQueueBias.FreeTensor(biasLocal);
                }
            }
    }
};


template<typename T> class ScaleSmallPost {
private:
    TPipe* pipe;
    TQue<TPosition::VECIN, TQUE_DEPTH> inQueueInput, inQueueScale, inQueueBias;
    TQue<TPosition::VECOUT, TQUE_DEPTH> outQueueOutput;
    TQue<TPosition::VECCALC, TQUE_DEPTH> calcQueue1, calcQueue2, calcQueue3;
    TQue<TPosition::VECCALC, TQUE_DEPTH> calcQueueScale, calcQueueBias;
    GlobalTensor<T> inputGm, scaleGm, biasGm, outputGm;
    ll preLength;
    ll midLength;
    ll postLength;
    ll totalLength;
    int blockIdx;
    int blockNum;
    int CALC_TYPE_SIZE;
    int NUM_PER_BLOCK;
    int CALC_NUM_PER_BLOCK;
    ll tileLength;
    ll postStep;
    ll midStep;

public:
    __aicore__ inline ScaleSmallPost() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, TPipe* pipeIn, ll preLength, ll midLength, ll postLength)
    {
        pipe = pipeIn;
        this->preLength = preLength;
        this->midLength = midLength;
        this->postLength = postLength;
        totalLength = preLength * midLength * postLength;

        blockIdx = GetBlockIdx();
        blockNum = GetBlockNum();

        
        CALC_TYPE_SIZE = sizeof(T);
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            CALC_TYPE_SIZE = sizeof(float);
        }
        NUM_PER_BLOCK = 32 / sizeof(T);
        CALC_NUM_PER_BLOCK = 32 / CALC_TYPE_SIZE;
        tileLength = 4096 * (2 / BUFFER_NUM) * (4 / CALC_TYPE_SIZE);
        postStep = postLength > tileLength ? tileLength : postLength;
        midStep = tileLength / postStep;

        inputGm.SetGlobalBuffer((__gm__ T *)input, totalLength);
        scaleGm.SetGlobalBuffer((__gm__ T *)scale, midLength);
        biasGm.SetGlobalBuffer((__gm__ T *)bias, midLength);
        outputGm.SetGlobalBuffer((__gm__ T *)output, totalLength);
        inputGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        outputGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

        pipe->InitBuffer(inQueueInput, BUFFER_NUM, tileLength * sizeof(T));
        pipe->InitBuffer(outQueueOutput, BUFFER_NUM, tileLength * sizeof(T));
        pipe->InitBuffer(inQueueScale, BUFFER_NUM, midStep * 32);
        pipe->InitBuffer(inQueueBias, BUFFER_NUM, midStep * 32);
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            pipe->InitBuffer(calcQueue1, BUFFER_NUM, tileLength * sizeof(float));
            pipe->InitBuffer(calcQueue2, BUFFER_NUM, midStep * 32 * 2);
            pipe->InitBuffer(calcQueue3, BUFFER_NUM, midStep * 32 * 2);
        }
        pipe->InitBuffer(calcQueueScale, BUFFER_NUM, tileLength * CALC_TYPE_SIZE);
        pipe->InitBuffer(calcQueueBias, BUFFER_NUM, tileLength * CALC_TYPE_SIZE);
    }
    __aicore__ inline void Process()
    {
        #define min(x, y) ((x) < (y) ? (x) : (y))
        ll tasksPerPre = (midLength + midStep - 1) / midStep;
        ll totalTasks = preLength * tasksPerPre;
    
            for (ll taskId = blockIdx; taskId < totalTasks; taskId += blockNum) {
                ll preIdx = taskId / tasksPerPre;
                ll midChunkIdx = taskId % tasksPerPre;
        
                ll midBase = midChunkIdx * midStep;
                ll midRemain = midLength - midBase;
                ll midCalcLength = min(midStep, midRemain);
                ll midCalcLengthAlign = (midCalcLength + NUM_PER_BLOCK - 1) / NUM_PER_BLOCK * NUM_PER_BLOCK;
                DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                DataCopyExtParams midCopyParams = {
                    (uint16_t)1,
                    (uint32_t)(midCalcLengthAlign * sizeof(T)),
                    0,
                    0,
                    0
                };
                LocalTensor<T> scaleLocal, biasLocal;
                LocalTensor<float> scaleFloat, biasFloat;
                scaleLocal = inQueueScale.AllocTensor<T>();
                biasLocal = inQueueBias.AllocTensor<T>();
                DataCopyPad(scaleLocal, scaleGm[midBase], midCopyParams, padParams);
                DataCopyPad(biasLocal, biasGm[midBase], midCopyParams, padParams);
                inQueueScale.EnQue(scaleLocal);
                inQueueBias.EnQue(biasLocal);
                inQueueScale.DeQue<T>();
                inQueueBias.DeQue<T>();
                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    scaleFloat = calcQueue2.AllocTensor<float>();
                    biasFloat = calcQueue3.AllocTensor<float>();
                    Cast(scaleFloat, scaleLocal, RoundMode::CAST_NONE, midCalcLength);
                    Cast(biasFloat, biasLocal, RoundMode::CAST_NONE, midCalcLength);
                    inQueueScale.FreeTensor(scaleLocal);
                    inQueueBias.FreeTensor(biasLocal);
                }
                const uint32_t dstShape[2] = {(uint32_t)midCalcLength, (uint32_t)postLength};
                const uint32_t srcShape[2] = {(uint32_t)midCalcLength, 1};
                const int32_t dim = 2;
                const int32_t axis = 1;
                LocalTensor<T> scaleBroad, biasBroad;
                LocalTensor<float> scaleFloatBroad, biasFloatBroad;
                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    scaleFloatBroad = calcQueueScale.AllocTensor<float>();
                    biasFloatBroad = calcQueueBias.AllocTensor<float>();
                    BroadCast<float, dim, axis>(scaleFloatBroad, scaleFloat, dstShape, srcShape);
                    BroadCast<float, dim, axis>(biasFloatBroad, biasFloat, dstShape, srcShape);
                    calcQueue2.FreeTensor(scaleFloat);
                    calcQueue3.FreeTensor(biasFloat);
                } else {
                    scaleBroad = calcQueueScale.AllocTensor<T>();
                    biasBroad = calcQueueBias.AllocTensor<T>();
                    BroadCast<T, dim, axis>(scaleBroad, scaleLocal, dstShape, srcShape);
                    BroadCast<T, dim, axis>(biasBroad, biasLocal, dstShape, srcShape);
                    inQueueScale.FreeTensor(scaleLocal);
                    inQueueBias.FreeTensor(biasLocal);
                }
                    ll postCalcLength = postLength;
                    ll postCalcLengthAlign = (postCalcLength + NUM_PER_BLOCK - 1) / NUM_PER_BLOCK * NUM_PER_BLOCK;
                    ll totalCalcLength = midCalcLength * postCalcLength;
                    ll totalCalcLengthAlign = midCalcLength * postCalcLengthAlign;
                    ll gmOffset = preIdx * midLength * postLength + midBase * postLength;
                    DataCopyExtParams copyParams = {
                        (uint16_t)1,
                        (uint32_t)(totalCalcLength * sizeof(T)),
                        0,
                        0,
                        0
                    };
                    LocalTensor<T> inputLocal, outputLocal;
                    inputLocal = inQueueInput.AllocTensor<T>();
                    DataCopyPad(inputLocal, inputGm[gmOffset], copyParams, padParams);
                    inQueueInput.EnQue(inputLocal);
                    inQueueInput.DeQue<T>();
                    if constexpr (std::is_same_v<T, bfloat16_t>) {
                        LocalTensor<float> inputFloat;
                        inputFloat = calcQueue1.AllocTensor<float>();
                        outputLocal = outQueueOutput.AllocTensor<T>();
                        Cast(inputFloat, inputLocal, RoundMode::CAST_NONE, totalCalcLength);
                        Mul(inputFloat, inputFloat, scaleFloatBroad, totalCalcLength);
                        Cast(outputLocal, inputFloat, RoundMode::CAST_RINT, totalCalcLength);
                        Cast(inputFloat, outputLocal, RoundMode::CAST_NONE, totalCalcLength);
                        Add(inputFloat, inputFloat, biasFloatBroad, totalCalcLength);
                        Cast(outputLocal, inputFloat, RoundMode::CAST_RINT, totalCalcLength);
                        inQueueInput.FreeTensor(inputLocal);
                        calcQueue1.FreeTensor(inputFloat);
                    } else {
                        outputLocal = outQueueOutput.AllocTensor<T>();
                        Mul(outputLocal, inputLocal, scaleBroad, totalCalcLength);
                        Add(outputLocal, outputLocal, biasBroad, totalCalcLength);
                        inQueueInput.FreeTensor(inputLocal);
                    }
                    outQueueOutput.EnQue(outputLocal);
                    outQueueOutput.DeQue<T>();
                    copyParams = {
                        (uint16_t)1,
                        (uint32_t)(totalCalcLength * sizeof(T)),
                        0,
                        0,
                        0
                    };
                    DataCopyPad(outputGm[gmOffset], outputLocal, copyParams);
                    outQueueOutput.FreeTensor(outputLocal);
                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    calcQueueScale.FreeTensor(scaleFloatBroad);
                    calcQueueBias.FreeTensor(biasFloatBroad);
                } else {
                    calcQueueScale.FreeTensor(scaleBroad);
                    calcQueueBias.FreeTensor(biasBroad);
                }
            }
    }
};


template<typename T> class ScalePreBroadcast {
private:
    TPipe* pipe;
    TQue<TPosition::VECIN, TQUE_DEPTH> inQueueInput, inQueueScale, inQueueBias;
    TQue<TPosition::VECOUT, TQUE_DEPTH> outQueueOutput;
    TQue<TPosition::VECCALC, TQUE_DEPTH> calcQueue1, calcQueue2, calcQueue3;
    GlobalTensor<T> inputGm, scaleGm, biasGm, outputGm;
    ll rowLength;
    ll colLength;
    ll totalLength;
    int blockIdx;
    int blockNum;
    int tileLength;
    int CALC_TYPE_SIZE;
    int NUM_PER_BLOCK;
    int CALC_NUM_PER_BLOCK;
    uint32_t REPEAT_LENGTH;

public:
    __aicore__ inline ScalePreBroadcast() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, TPipe* pipeIn, ll rowLength, ll colLength, ll postLength)
    {
        pipe = pipeIn;
        this->rowLength = rowLength;
        this->colLength = colLength;
        totalLength = rowLength * colLength * postLength;

        blockIdx = GetBlockIdx();
        blockNum = GetBlockNum();


        CALC_TYPE_SIZE = sizeof(T);
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            CALC_TYPE_SIZE = sizeof(float);
        }
        NUM_PER_BLOCK = 32 / sizeof(T);
        CALC_NUM_PER_BLOCK = 32 / CALC_TYPE_SIZE;
        REPEAT_LENGTH = REPEAT_SIZE / CALC_TYPE_SIZE;
        tileLength = 4096 * 2 / BUFFER_NUM;

        inputGm.SetGlobalBuffer((__gm__ T *)input, totalLength);
        scaleGm.SetGlobalBuffer((__gm__ T *)scale, colLength);
        biasGm.SetGlobalBuffer((__gm__ T *)bias, colLength);
        outputGm.SetGlobalBuffer((__gm__ T *)output, totalLength);
        inputGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        outputGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

        pipe->InitBuffer(inQueueInput, BUFFER_NUM, tileLength * sizeof(T));
        pipe->InitBuffer(inQueueScale, BUFFER_NUM, tileLength * sizeof(T));
        pipe->InitBuffer(inQueueBias, BUFFER_NUM, tileLength * sizeof(T));
        pipe->InitBuffer(outQueueOutput, BUFFER_NUM, tileLength * sizeof(T));
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            pipe->InitBuffer(calcQueue1, BUFFER_NUM, tileLength * sizeof(float));
            pipe->InitBuffer(calcQueue2, BUFFER_NUM, tileLength * sizeof(float));
            pipe->InitBuffer(calcQueue3, BUFFER_NUM, tileLength * sizeof(float));
        }
    }
    __aicore__ inline void Process()
    {
        for (ll colBase = 0; colBase < colLength; colBase += tileLength) {
            ll colRemain = colLength - colBase;
            ll colCalcLength = min(tileLength, colRemain);
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyExtParams copyParams = {
                (uint16_t)1,
                (uint32_t)(colCalcLength * sizeof(T)),
                0,
                0,
                0
            };
            LocalTensor<T> scaleLocal, biasLocal;
            LocalTensor<float> scaleFloat, biasFloat;
            scaleLocal = inQueueScale.AllocTensor<T>();
            biasLocal = inQueueBias.AllocTensor<T>();
            DataCopyPad(scaleLocal, scaleGm[colBase], copyParams, padParams);
            DataCopyPad(biasLocal, biasGm[colBase], copyParams, padParams);
            inQueueScale.EnQue(scaleLocal);
            inQueueBias.EnQue(biasLocal);
            inQueueScale.DeQue<T>();
            inQueueBias.DeQue<T>();
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                scaleFloat = calcQueue2.AllocTensor<float>();
                biasFloat = calcQueue3.AllocTensor<float>();
                Cast(scaleFloat, scaleLocal, RoundMode::CAST_NONE, colCalcLength);
                Cast(biasFloat, biasLocal, RoundMode::CAST_NONE, colCalcLength);
                inQueueScale.FreeTensor(scaleLocal);
                inQueueBias.FreeTensor(biasLocal);
            }
            for (ll rowIdx = blockIdx; rowIdx < rowLength; rowIdx += blockNum) {
                LocalTensor<T> inputLocal, outputLocal;
                LocalTensor<float> inputFloat;
                ll gmOffset = rowIdx * colLength + colBase;
                inputLocal = inQueueInput.AllocTensor<T>();
                DataCopyPad(inputLocal, inputGm[gmOffset], copyParams, padParams);
                inQueueInput.EnQue(inputLocal);
                inQueueInput.DeQue<T>();
                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    inputFloat = calcQueue1.AllocTensor<float>();
                    Cast(inputFloat, inputLocal, RoundMode::CAST_NONE, colCalcLength);
                    inQueueInput.FreeTensor(inputLocal);
                    Mul(inputFloat, inputFloat, scaleFloat, colCalcLength);
                    outputLocal = outQueueOutput.AllocTensor<T>();
                    Cast(outputLocal, inputFloat, RoundMode::CAST_RINT, colCalcLength);
                    Cast(inputFloat, outputLocal, RoundMode::CAST_NONE, colCalcLength);
                    Add(inputFloat, inputFloat, biasFloat, colCalcLength);
                    Cast(outputLocal, inputFloat, RoundMode::CAST_RINT, colCalcLength);
                    calcQueue1.FreeTensor(inputFloat);
                } else {
                    outputLocal = outQueueOutput.AllocTensor<T>();
                    Mul(outputLocal, inputLocal, scaleLocal, colCalcLength);
                    Add(outputLocal, outputLocal, biasLocal, colCalcLength);
                    inQueueInput.FreeTensor(inputLocal);
                }
                outQueueOutput.EnQue(outputLocal);
                outQueueOutput.DeQue<T>();
                DataCopyPad(outputGm[gmOffset], outputLocal, copyParams);
                outQueueOutput.FreeTensor(outputLocal);
            }
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                calcQueue2.FreeTensor(scaleFloat);
                calcQueue3.FreeTensor(biasFloat);
            } else {
                inQueueScale.FreeTensor(scaleLocal);
                inQueueBias.FreeTensor(biasLocal);
            }
        }
    }
};

extern "C" __global__ __aicore__ void scale(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        ScalePreBroadcast<DTYPE_INPUT> op;
        op.Init(input, scale, bias, output, &pipe, tiling_data.preLength, tiling_data.midLength, tiling_data.postLength);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ScaleSmallPost<DTYPE_INPUT> op;
        op.Init(input, scale, bias, output, &pipe, tiling_data.preLength, tiling_data.midLength, tiling_data.postLength);
        op.Process();
    } else if (TILING_KEY_IS(0)) {
        Scale<DTYPE_INPUT> op;
        op.Init(input, scale, bias, output, &pipe, tiling_data.preLength, tiling_data.midLength, tiling_data.postLength);
        op.Process();
    }
}