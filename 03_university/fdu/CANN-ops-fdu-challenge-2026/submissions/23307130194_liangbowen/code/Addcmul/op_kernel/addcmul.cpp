// Kernel侧核函数实现
#include "kernel_operator.h"

#include "addcmul_tiling.h"
#include "tiling_key_addcmul.h"

constexpr int32_t BUFFER_NUM = 1;

template <class DT_INPUT_DATA>
class KernelAddcmul {
public:
    __aicore__ inline KernelAddcmul() {}
    __aicore__ inline void Init(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
                                AddcmulTilingData &td)
    {
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t basePerCore = td.totalLength / blockNum;
        uint32_t tailCores = td.tailCoreNum;

        this->coreDataNum = basePerCore;
        if (coreIdx < tailCores) {
            this->coreDataNum = basePerCore + 1;
        }

        uint32_t offset = coreIdx * basePerCore + (coreIdx < tailCores ? coreIdx : tailCores);

        inputDataGm.SetGlobalBuffer((__gm__ DT_INPUT_DATA *)input_data + offset, this->coreDataNum);
        x1Gm.SetGlobalBuffer((__gm__ DT_INPUT_DATA *)x1 + offset, this->coreDataNum);
        x2Gm.SetGlobalBuffer((__gm__ DT_INPUT_DATA *)x2 + offset, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_INPUT_DATA *)y + offset, this->coreDataNum);
        valuePtr = (__gm__ DT_INPUT_DATA *)value;
        this->valueScalar = valuePtr[0];

        pipe.InitBuffer(inQueueInputData, BUFFER_NUM, this->coreDataNum * sizeof(DT_INPUT_DATA));
        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->coreDataNum * sizeof(DT_INPUT_DATA));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->coreDataNum * sizeof(DT_INPUT_DATA));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->coreDataNum * sizeof(DT_INPUT_DATA));
        pipe.InitBuffer(tempBuf, this->coreDataNum * sizeof(DT_INPUT_DATA));
    }

    __aicore__ inline void Process()
    {
        if (this->coreDataNum == 0) { return; }
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<DT_INPUT_DATA> d1 = inQueueInputData.AllocTensor<DT_INPUT_DATA>();
        AscendC::LocalTensor<DT_INPUT_DATA> d2 = inQueueX1.AllocTensor<DT_INPUT_DATA>();
        AscendC::LocalTensor<DT_INPUT_DATA> d3 = inQueueX2.AllocTensor<DT_INPUT_DATA>();
        AscendC::DataCopy(d1, inputDataGm, this->coreDataNum);
        AscendC::DataCopy(d2, x1Gm, this->coreDataNum);
        AscendC::DataCopy(d3, x2Gm, this->coreDataNum);
        inQueueInputData.EnQue(d1);
        inQueueX1.EnQue(d2);
        inQueueX2.EnQue(d3);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<DT_INPUT_DATA> d1 = inQueueInputData.DeQue<DT_INPUT_DATA>();
        AscendC::LocalTensor<DT_INPUT_DATA> d2 = inQueueX1.DeQue<DT_INPUT_DATA>();
        AscendC::LocalTensor<DT_INPUT_DATA> d3 = inQueueX2.DeQue<DT_INPUT_DATA>();
        AscendC::LocalTensor<DT_INPUT_DATA> yLocal = outQueueY.AllocTensor<DT_INPUT_DATA>();

        auto t = tempBuf.Get<DT_INPUT_DATA>();
        AscendC::Mul(t, d2, d3, this->coreDataNum);
        AscendC::Muls(t, t, this->valueScalar, this->coreDataNum);
        AscendC::Add(yLocal, d1, t, this->coreDataNum);

        outQueueY.EnQue<DT_INPUT_DATA>(yLocal);
        inQueueInputData.FreeTensor(d1);
        inQueueX1.FreeTensor(d2);
        inQueueX2.FreeTensor(d3);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<DT_INPUT_DATA> yLocal = outQueueY.DeQue<DT_INPUT_DATA>();
        AscendC::DataCopy(yGm, yLocal, this->coreDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueInputData;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX1;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX2;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tempBuf;
    AscendC::GlobalTensor<DT_INPUT_DATA> inputDataGm;
    AscendC::GlobalTensor<DT_INPUT_DATA> x1Gm;
    AscendC::GlobalTensor<DT_INPUT_DATA> x2Gm;
    AscendC::GlobalTensor<DT_INPUT_DATA> yGm;
    __gm__ DT_INPUT_DATA *valuePtr;
    DT_INPUT_DATA valueScalar;
    uint32_t coreDataNum;
};

template <>
__aicore__ inline void KernelAddcmul<int8_t>::Compute()
{
    AscendC::LocalTensor<int8_t> d1 = inQueueInputData.DeQue<int8_t>();
    AscendC::LocalTensor<int8_t> d2 = inQueueX1.DeQue<int8_t>();
    AscendC::LocalTensor<int8_t> d3 = inQueueX2.DeQue<int8_t>();
    AscendC::LocalTensor<int8_t> yLocal = outQueueY.AllocTensor<int8_t>();

    int8_t sv = this->valueScalar;
    for (uint32_t i = 0; i < this->coreDataNum; i++) {
        int32_t prod = static_cast<int32_t>(d2.GetValue(i)) * static_cast<int32_t>(d3.GetValue(i));
        prod = prod * static_cast<int32_t>(sv);
        prod = prod + static_cast<int32_t>(d1.GetValue(i));
        yLocal.SetValue(i, static_cast<int8_t>(prod));
    }

    outQueueY.EnQue<int8_t>(yLocal);
    inQueueInputData.FreeTensor(d1);
    inQueueX1.FreeTensor(d2);
    inQueueX2.FreeTensor(d3);
}

template <typename DT_INPUT_DATA>
 __global__ __aicore__ void addcmul(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(AddcmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddcmulTilingData, tiling_data, tiling);
    KernelAddcmul<DT_INPUT_DATA> op;
    op.Init(input_data, x1, x2, value, y, tiling_data);
    op.Process();
}