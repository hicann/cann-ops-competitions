#include "kernel_operator.h"

using namespace AscendC;

template <typename T>
class ScaleKernel_0
{
public:
    __aicore__ inline ScaleKernel_0() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, int tileDataNum, int bigCoreNum, int bigCoreProcessNum, int smallCoreProcessNum, TPipe *pipe)
    {
        int coreNum = GetBlockIdx();
        int globalBufferIndex = coreNum * bigCoreProcessNum;
        this->tileDataNum = tileDataNum;
        if (coreNum < bigCoreNum)
            this->processNum = bigCoreProcessNum;
        else
        {
            this->processNum = smallCoreProcessNum;
            globalBufferIndex -= (bigCoreProcessNum - smallCoreProcessNum) * (coreNum - bigCoreNum);
        }
        inputGm.SetGlobalBuffer((__gm__ T *)input + globalBufferIndex, processNum);
        scaleGm.SetGlobalBuffer((__gm__ T *)scale + globalBufferIndex, processNum);
        biasGm.SetGlobalBuffer((__gm__ T *)bias + globalBufferIndex, processNum);
        outputGm.SetGlobalBuffer((__gm__ T *)output + globalBufferIndex, processNum);
        pipe->InitBuffer(Qin_input, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qin_scale, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qin_bias, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qout, 1, tileDataNum * sizeof(T));
        if constexpr (std::is_same_v<T, bfloat16_t>)
        {
            pipe->InitBuffer(B_input, tileDataNum * 4);
            pipe->InitBuffer(B_scale, tileDataNum * 4);
            pipe->InitBuffer(B_bias, tileDataNum * 4);
        }
    }
    __aicore__ inline void Process()
    {
        int count = (processNum + tileDataNum - 1) / tileDataNum;
        int loop_processNum = tileDataNum;
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>)
        {
            for (int i = 0; i < count; i ++)
            {
                if (i == count - 1) loop_processNum = processNum % tileDataNum;
                DataCopyExtParams copyParams{1, (uint32_t)(loop_processNum * sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                LocalTensor<T> input = Qin_input.AllocTensor<T>();
                LocalTensor<T> scale = Qin_scale.AllocTensor<T>();
                LocalTensor<T> bias = Qin_bias.AllocTensor<T>();
                DataCopyPad(input, inputGm[i * tileDataNum], copyParams, padParams);
                DataCopyPad(scale, scaleGm[i * tileDataNum], copyParams, padParams);
                DataCopyPad(bias, biasGm[i * tileDataNum], copyParams, padParams);
                Qin_input.EnQue(input);
                Qin_scale.EnQue(scale);
                Qin_bias.EnQue(bias);
                input = Qin_input.DeQue<T>();
                scale = Qin_scale.DeQue<T>();
                bias = Qin_bias.DeQue<T>();
                LocalTensor<T> out = Qout.AllocTensor<T>();
                Mul(out, input, scale, loop_processNum);
                Add(out, out, bias, loop_processNum);
                Qout.EnQue(out);
                out = Qout.DeQue<T>();
                DataCopyPad(outputGm[i * tileDataNum], out, copyParams);
                Qin_input.FreeTensor(input);
                Qin_scale.FreeTensor(scale);
                Qin_bias.FreeTensor(bias);
                Qout.FreeTensor(out);
            }
        }
        else
        {
            for (int i = 0; i < count; i ++)
            {
                if (i == count - 1) loop_processNum = processNum % tileDataNum;
                DataCopyExtParams copyParams{1, (uint32_t)(loop_processNum * sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                LocalTensor<T> input = Qin_input.AllocTensor<T>();
                LocalTensor<T> scale = Qin_scale.AllocTensor<T>();
                LocalTensor<T> bias = Qin_bias.AllocTensor<T>();
                DataCopyPad(input, inputGm[i * tileDataNum], copyParams, padParams);
                DataCopyPad(scale, scaleGm[i * tileDataNum], copyParams, padParams);
                DataCopyPad(bias, biasGm[i * tileDataNum], copyParams, padParams);
                Qin_input.EnQue(input);
                Qin_scale.EnQue(scale);
                Qin_bias.EnQue(bias);
                input = Qin_input.DeQue<T>();
                scale = Qin_scale.DeQue<T>();
                bias = Qin_bias.DeQue<T>();
                LocalTensor<T> out = Qout.AllocTensor<T>();
                LocalTensor<float> input_f32 = B_input.Get<float>();
                LocalTensor<float> scale_f32 = B_scale.Get<float>();
                LocalTensor<float> bias_f32 = B_bias.Get<float>();
                Cast(input_f32, input, RoundMode::CAST_NONE, loop_processNum);
                Cast(scale_f32, scale, RoundMode::CAST_NONE, loop_processNum);
                Cast(bias_f32, bias, RoundMode::CAST_NONE, loop_processNum);
                Mul(input_f32, input_f32, scale_f32, loop_processNum);
                Cast(input, input_f32, RoundMode::CAST_RINT, loop_processNum);
                Cast(input_f32, input, RoundMode::CAST_NONE, loop_processNum);
                Add(input_f32, input_f32, bias_f32, loop_processNum);
                Cast(out, input_f32, RoundMode::CAST_RINT, loop_processNum);
                Qout.EnQue(out);
                out = Qout.DeQue<T>();
                DataCopyPad(outputGm[i * tileDataNum], out, copyParams);
                Qin_input.FreeTensor(input);
                Qin_scale.FreeTensor(scale);
                Qin_bias.FreeTensor(bias);
                Qout.FreeTensor(out);
            }
        }
    }
private:
    GlobalTensor<T> inputGm, scaleGm, biasGm, outputGm;
    TQue<QuePosition::VECIN, 1> Qin_input, Qin_scale, Qin_bias;
    TQue<QuePosition::VECOUT, 1> Qout;
    TBuf<QuePosition::VECCALC> B_input, B_scale, B_bias;
    int processNum, tileDataNum;
};

template <typename T>
class ScaleKernel_1
{
public:
    __aicore__ inline ScaleKernel_1() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, int tileDataNum, int innerStride, int bigCoreNum, int bigCoreProcessNum, int smallCoreProcessNum, TPipe *pipe)
    {
        int coreNum = GetBlockIdx();
        int globalBufferIndex = coreNum * bigCoreProcessNum;
        this->tileDataNum = tileDataNum;
        this->innerStride = innerStride;
        if (coreNum < bigCoreNum)
            this->processNum = bigCoreProcessNum;
        else
        {
            this->processNum = smallCoreProcessNum;
            globalBufferIndex -= (bigCoreProcessNum - smallCoreProcessNum) * (coreNum - bigCoreNum);
        }
        inputGm.SetGlobalBuffer((__gm__ T *)input + globalBufferIndex * innerStride, processNum * innerStride);
        scaleGm.SetGlobalBuffer((__gm__ T *)scale);
        biasGm.SetGlobalBuffer((__gm__ T *)bias);
        outputGm.SetGlobalBuffer((__gm__ T *)output + globalBufferIndex * innerStride, processNum * innerStride);
        pipe->InitBuffer(Qin_input, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qin_scale, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qin_bias, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qout, 1, tileDataNum * sizeof(T));
        if constexpr (std::is_same_v<T, bfloat16_t>)
        {
            pipe->InitBuffer(B_input, tileDataNum * 4);
            pipe->InitBuffer(B_scale, tileDataNum * 4);
            pipe->InitBuffer(B_bias, tileDataNum * 4);
        }
    }
    __aicore__ inline void Process()
    {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>)
        {
            int count = (innerStride + tileDataNum - 1) / tileDataNum;
            int loop_processNum = tileDataNum;
            for (int i = 0; i < count; i ++)
            {
                if (i == count - 1) loop_processNum = innerStride % tileDataNum;
                DataCopyExtParams copyParams{1, (uint32_t)(loop_processNum * sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                LocalTensor<T> scale = Qin_scale.AllocTensor<T>();
                LocalTensor<T> bias = Qin_bias.AllocTensor<T>();
                DataCopyPad(scale, scaleGm[i * tileDataNum], copyParams, padParams);
                DataCopyPad(bias, biasGm[i * tileDataNum], copyParams, padParams);
                Qin_scale.EnQue(scale);
                Qin_bias.EnQue(bias);
                scale = Qin_scale.DeQue<T>();
                bias = Qin_bias.DeQue<T>();
                for (int j = 0; j < processNum; j ++)
                {
                    LocalTensor<T> input = Qin_input.AllocTensor<T>();
                    DataCopyPad(input, inputGm[j * innerStride + i * tileDataNum], copyParams, padParams);
                    Qin_input.EnQue(input);
                    input = Qin_input.DeQue<T>();
                    LocalTensor<T> out = Qout.AllocTensor<T>();
                    Mul(out, input, scale, loop_processNum);
                    Add(out, out, bias, loop_processNum);
                    Qout.EnQue(out);
                    out = Qout.DeQue<T>();
                    DataCopyPad(outputGm[j * innerStride + i * tileDataNum], out, copyParams);
                    Qin_input.FreeTensor(input);
                    Qout.FreeTensor(out);
                }
                Qin_scale.FreeTensor(scale);
                Qin_bias.FreeTensor(bias);
            }
        }
        else
        {
            int count = (innerStride + tileDataNum - 1) / tileDataNum;
            int loop_processNum = tileDataNum;
            for (int i = 0; i < count; i ++)
            {
                if (i == count - 1) loop_processNum = innerStride % tileDataNum;
                DataCopyExtParams copyParams{1, (uint32_t)(loop_processNum * sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                LocalTensor<T> scale = Qin_scale.AllocTensor<T>();
                LocalTensor<T> bias = Qin_bias.AllocTensor<T>();
                DataCopyPad(scale, scaleGm[i * tileDataNum], copyParams, padParams);
                DataCopyPad(bias, biasGm[i * tileDataNum], copyParams, padParams);
                Qin_scale.EnQue(scale);
                Qin_bias.EnQue(bias);
                scale = Qin_scale.DeQue<T>();
                bias = Qin_bias.DeQue<T>();
                LocalTensor<float> scale_f32 = B_scale.Get<float>();
                LocalTensor<float> bias_f32 = B_bias.Get<float>();
                Cast(scale_f32, scale, RoundMode::CAST_NONE, loop_processNum);
                Cast(bias_f32, bias, RoundMode::CAST_NONE, loop_processNum);
                for (int j = 0; j < processNum; j ++)
                {
                    LocalTensor<T> input = Qin_input.AllocTensor<T>();
                    DataCopyPad(input, inputGm[j * innerStride + i * tileDataNum], copyParams, padParams);
                    Qin_input.EnQue(input);
                    input = Qin_input.DeQue<T>();
                    LocalTensor<float> input_f32 = B_input.Get<float>();
                    Cast(input_f32, input, RoundMode::CAST_NONE, loop_processNum);
                    Mul(input_f32, input_f32, scale_f32, loop_processNum);
                    Cast(input, input_f32, RoundMode::CAST_RINT, loop_processNum);
                    Cast(input_f32, input, RoundMode::CAST_NONE, loop_processNum);
                    Add(input_f32, input_f32, bias_f32, loop_processNum);
                    LocalTensor<T> out = Qout.AllocTensor<T>();
                    Cast(out, input_f32, RoundMode::CAST_RINT, loop_processNum);
                    Qout.EnQue(out);
                    out = Qout.DeQue<T>();
                    DataCopyPad(outputGm[j * innerStride + i * tileDataNum], out, copyParams);
                    Qin_input.FreeTensor(input);
                    Qout.FreeTensor(out);
                }
                Qin_scale.FreeTensor(scale);
                Qin_bias.FreeTensor(bias);
            }
        }
    }
private:
    GlobalTensor<T> inputGm, scaleGm, biasGm, outputGm;
    TQue<QuePosition::VECIN, 1> Qin_input, Qin_scale, Qin_bias;
    TQue<QuePosition::VECOUT, 1> Qout;
    TBuf<QuePosition::VECCALC> B_input, B_scale, B_bias;
    int processNum, tileDataNum, innerStride;
};

template <typename T>
class ScaleKernel_2
{
public:
    __aicore__ inline ScaleKernel_2() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, int tileDataNum, int innerStride, int outerStride, int bigCoreNum, int bigCoreProcessNum, int smallCoreProcessNum, TPipe *pipe)
    {
        int coreNum = GetBlockIdx();
        this->globalBufferIndex = coreNum * bigCoreProcessNum;
        this->tileDataNum = tileDataNum;
        this->innerStride = innerStride;
        this->outerStride = outerStride;
        if (coreNum < bigCoreNum)
            this->processNum = bigCoreProcessNum;
        else
        {
            this->processNum = smallCoreProcessNum;
            this->globalBufferIndex -= (bigCoreProcessNum - smallCoreProcessNum) * (coreNum - bigCoreNum);
        }
        inputGm.SetGlobalBuffer((__gm__ T *)input + globalBufferIndex * innerStride, processNum * innerStride);
        scaleGm.SetGlobalBuffer((__gm__ T *)scale, outerStride);
        biasGm.SetGlobalBuffer((__gm__ T *)bias, outerStride);
        outputGm.SetGlobalBuffer((__gm__ T *)output + globalBufferIndex * innerStride, processNum * innerStride);
        pipe->InitBuffer(Qin_input, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qout, 1, tileDataNum * sizeof(T));
        if constexpr (std::is_same_v<T, bfloat16_t>)
            pipe->InitBuffer(B_input, tileDataNum * 4);
        else
            pipe->InitBuffer(Qin_broad, 1, (tileDataNum / innerStride) * sizeof(T));
    }
    __aicore__ inline void Process()
    {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>)
        {
            if (tileDataNum < innerStride)
            {
                for (int i = 0; i < processNum; i ++)
                {
                    int idx = (i + globalBufferIndex) % outerStride;
                    T scale = scaleGm.GetValue(idx), bias = biasGm.GetValue(idx);
                    int count = (innerStride + tileDataNum - 1) / tileDataNum;
                    int loop_processNum = tileDataNum;
                    for (int j = 0; j < count; j ++)
                    {
                        if (j == count - 1) loop_processNum = innerStride % tileDataNum;
                        DataCopyExtParams copyParams{1, (uint32_t)(loop_processNum * sizeof(T)), 0, 0, 0};
                        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                        LocalTensor<T> input = Qin_input.AllocTensor<T>();
                        DataCopyPad(input, inputGm[i * innerStride + j * tileDataNum], copyParams, padParams);
                        Qin_input.EnQue(input);
                        input = Qin_input.DeQue<T>();
                        LocalTensor<T> out = Qout.AllocTensor<T>();
                        Muls(out, input, scale, loop_processNum);
                        Adds(out, out, bias, loop_processNum);
                        Qout.EnQue(out);
                        out = Qout.DeQue<T>();
                        DataCopyPad(outputGm[i * innerStride + j * tileDataNum], out, copyParams);
                        Qin_input.FreeTensor(input);
                        Qout.FreeTensor(out);
                    }
                }
            }
            else
            {
                int blockCount = tileDataNum / innerStride;
                int loopCount = (processNum + blockCount - 1) / blockCount;
                int count = blockCount;
                for (int i = 0; i < loopCount; i ++)
                {
                    if (i == loopCount - 1 && processNum % blockCount != 0) count = processNum % blockCount;
                    LocalTensor<T> broad = Qin_broad.AllocTensor<T>();
                    uint32_t dstShape[2] = {(uint32_t)count, (uint32_t)innerStride}, srcShape[2] = {(uint32_t)count, 1};
                    LocalTensor<T> input = Qin_input.AllocTensor<T>();
                    LocalTensor<T> out = Qout.AllocTensor<T>();
                    DataCopyExtParams copyParams{1, (uint32_t)(count * innerStride * sizeof(T)), 0, 0, 0};
                    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                    DataCopyPad(input, inputGm[i * blockCount * innerStride], copyParams, padParams);
                    Qin_input.EnQue(input);
                    input = Qin_input.DeQue<T>();
                    if (count <= outerStride && (i * blockCount + globalBufferIndex) % outerStride < (i * blockCount + count - 1 + globalBufferIndex) % outerStride)
                    {
                        DataCopyExtParams copyParams{1, (uint32_t)(count * sizeof(T)), 0, 0, 0};
                        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                        DataCopyPad(broad, scaleGm[(i * blockCount + globalBufferIndex) % outerStride], copyParams, padParams);
                        Qin_broad.EnQue(broad);
                        broad = Qin_broad.DeQue<T>();
                    }
                    else
                    {
                        for (int j = 0; j < count; j ++)
                        {
                            int idx = (i * blockCount + j + globalBufferIndex) % outerStride;
                            broad.SetValue(j, scaleGm.GetValue(idx));
                        }
                    }
                    Broadcast<T, 2, 1>(out, broad, dstShape, srcShape);
                    Mul(input, input, out, count * innerStride);
                    if (count <= outerStride && (i * blockCount + globalBufferIndex) % outerStride < (i * blockCount + count - 1 + globalBufferIndex) % outerStride)
                    {
                        DataCopyExtParams copyParams{1, (uint32_t)(count * sizeof(T)), 0, 0, 0};
                        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                        DataCopyPad(broad, biasGm[(i * blockCount + globalBufferIndex) % outerStride], copyParams, padParams);
                        Qin_broad.EnQue(broad);
                        broad = Qin_broad.DeQue<T>();
                    }
                    else
                    {
                        for (int j = 0; j < count; j ++)
                        {
                            int idx = (i * blockCount + j + globalBufferIndex) % outerStride;
                            broad.SetValue(j, biasGm.GetValue(idx));
                        }
                    }
                    Broadcast<T, 2, 1>(out, broad, dstShape, srcShape);
                    Add(out, input, out, count * innerStride);
                    Qout.EnQue(out);
                    out = Qout.DeQue<T>();
                    DataCopyPad(outputGm[i * blockCount * innerStride], out, copyParams);
                    Qin_input.FreeTensor(input);
                    Qin_broad.FreeTensor(broad);
                    Qout.FreeTensor(out);
                }
            }
        }
        else
        {
            for (int i = 0; i < processNum; i ++)
            {
                int idx = (i + globalBufferIndex) % outerStride;
                T scale = scaleGm.GetValue(idx), bias = biasGm.GetValue(idx);
                float scale_f32 = ToFloat(scale), bias_f32 = ToFloat(bias);
                int count = (innerStride + tileDataNum - 1) / tileDataNum;
                int loop_processNum = tileDataNum;
                for (int j = 0; j < count; j ++)
                {
                    if (j == count - 1) loop_processNum = innerStride % tileDataNum;
                    DataCopyExtParams copyParams{1, (uint32_t)(loop_processNum * sizeof(T)), 0, 0, 0};
                    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                    LocalTensor<T> input = Qin_input.AllocTensor<T>();
                    DataCopyPad(input, inputGm[i * innerStride + j * tileDataNum], copyParams, padParams);
                    Qin_input.EnQue(input);
                    input = Qin_input.DeQue<T>();
                    LocalTensor<T> out = Qout.AllocTensor<T>();
                    LocalTensor<float> input_f32 = B_input.Get<float>();
                    Cast(input_f32, input, RoundMode::CAST_NONE, loop_processNum);
                    Muls(input_f32, input_f32, scale_f32, loop_processNum);
                    Cast(input, input_f32, RoundMode::CAST_RINT, loop_processNum);
                    Cast(input_f32, input, RoundMode::CAST_NONE, loop_processNum);
                    Adds(input_f32, input_f32, bias_f32, loop_processNum);
                    Cast(out, input_f32, RoundMode::CAST_RINT, loop_processNum);
                    Qout.EnQue(out);
                    out = Qout.DeQue<T>();
                    DataCopyPad(outputGm[i * innerStride + j * tileDataNum], out, copyParams);
                    Qin_input.FreeTensor(input);
                    Qout.FreeTensor(out);
                }
            }
        }
    }
private:
    GlobalTensor<T> inputGm, scaleGm, biasGm, outputGm;
    TQue<QuePosition::VECIN, 1> Qin_input, Qin_broad;
    TQue<QuePosition::VECOUT, 1> Qout;
    TBuf<QuePosition::VECCALC> B_input;
    int globalBufferIndex, processNum, tileDataNum, innerStride, outerStride;
};

extern "C" __global__ __aicore__ void scale(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    TPipe pipe;
    if (TILING_KEY_IS(0))
    {
        ScaleKernel_0<DTYPE_INPUT> op;
        op.Init(input, scale, bias, output, tiling_data.tileDataNum, tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreProcessNum, &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(1))
    {
        ScaleKernel_1<DTYPE_INPUT> op;
        op.Init(input, scale, bias, output, tiling_data.tileDataNum, tiling_data.innerStride, tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreProcessNum, &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(2))
    {
        ScaleKernel_2<DTYPE_INPUT> op;
        op.Init(input, scale, bias, output, tiling_data.tileDataNum, tiling_data.innerStride, tiling_data.outerStride, tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreProcessNum, &pipe);
        op.Process();
    }
}