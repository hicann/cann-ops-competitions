#include "kernel_operator.h"
#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

constexpr uint8_t BUFFER_NUM = 2;
constexpr float A = 1.12861325f;
constexpr float B = 0.10098584f;

__aicore__ inline void ErfCalc(
    LocalTensor<float>& dst, LocalTensor<float>& src,
    LocalTensor<float>& tmp, uint32_t len)
{
    Mul(tmp, src, src, len);             // x²
    Muls(tmp, tmp, B, len);              // bx²
    Adds(tmp, tmp, A, len);              // a+bx²
    Mul(tmp, tmp, src, len);             // x(a+bx²)
    Tanh(dst, tmp, len);                 // erf ≈ tanh(tmp)
}

// ======================== 大输入: 多核+双缓冲 ========================
class KernelErfDB {
public:
    __aicore__ inline KernelErfDB() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
        uint64_t bigCoreDataNum, uint32_t finalBigTileNum, uint32_t tileDataNum, uint32_t bigTailDataNum,
        uint64_t smallCoreDataNum, uint32_t finalSmallTileNum, uint32_t smallTailDataNum, TPipe* pipe)
    {
        uint64_t coreId = GetBlockIdx();
        uint64_t coreNum = GetBlockNum();
        this->tileDataNum = tileDataNum;

        if (coreId != coreNum - 1) {
            tileNum = finalBigTileNum;
            tailDataNum = bigTailDataNum;
            xGm.SetGlobalBuffer((__gm__ float*)x + bigCoreDataNum * coreId, bigCoreDataNum);
            yGm.SetGlobalBuffer((__gm__ float*)y + bigCoreDataNum * coreId, bigCoreDataNum);
        } else {
            tileNum = finalSmallTileNum;
            tailDataNum = smallTailDataNum;
            xGm.SetGlobalBuffer((__gm__ float*)x + bigCoreDataNum * coreId, smallCoreDataNum);
            yGm.SetGlobalBuffer((__gm__ float*)y + bigCoreDataNum * coreId, smallCoreDataNum);
        }

        pipe->InitBuffer(inQueueX, BUFFER_NUM, tileDataNum * sizeof(float));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, tileDataNum * sizeof(float));
        pipe->InitBuffer(calcBuf, tileDataNum * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        processDataNum = tileDataNum;
        for (uint32_t i = 0; i < tileNum - 1; i++) {
            CopyIn(i); Compute(i); CopyOut(i);
        }
        processDataNum = tailDataNum;
        CopyIn(tileNum - 1); Compute(tileNum - 1); CopyOut(tileNum - 1);
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress) {
        LocalTensor<float> xL = inQueueX.AllocTensor<float>();
        DataCopy(xL, xGm[progress * tileDataNum], processDataNum);
        inQueueX.EnQue(xL);
    }
    __aicore__ inline void Compute(uint32_t) {
        LocalTensor<float> xL = inQueueX.DeQue<float>();
        LocalTensor<float> yL = outQueueY.AllocTensor<float>();
        LocalTensor<float> tmp = calcBuf.Get<float>();
        ErfCalc(yL, xL, tmp, processDataNum);
        outQueueY.EnQue<float>(yL);
        inQueueX.FreeTensor(xL);
    }
    __aicore__ inline void CopyOut(uint32_t progress) {
        LocalTensor<float> yL = outQueueY.DeQue<float>();
        DataCopy(yGm[progress * tileDataNum], yL, processDataNum);
        outQueueY.FreeTensor(yL);
    }

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> calcBuf;
    GlobalTensor<float> xGm, yGm;
    uint32_t tileNum, tileDataNum, tailDataNum, processDataNum;
};

// ======================== 小输入: 单缓冲 ========================
__aicore__ inline void KernelErfNDB(GM_ADDR x, GM_ADDR y,
    uint32_t tileDataNum, uint32_t bigTailDataNum, TPipe* pipe)
{
    uint32_t coreId = GetBlockIdx();
    uint32_t length = (coreId == GetBlockNum() - 1) ? bigTailDataNum : tileDataNum;

    TBuf<QuePosition::VECCALC> buf;                                    
    pipe->InitBuffer(buf, 3 * tileDataNum * sizeof(float));            
    LocalTensor<float> all = buf.Get<float>();                         
    LocalTensor<float> xL  = all;                                      
    LocalTensor<float> yL  = all[tileDataNum];                         
    LocalTensor<float> tmp = all[2 * tileDataNum];                     

    GlobalTensor<float> xGm;
    xGm.SetGlobalBuffer((__gm__ float*)x + coreId * tileDataNum);
    DataCopy(xL, xGm, length);
    event_t evtM2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(evtM2V);

    GlobalTensor<float> yGm;
    yGm.SetGlobalBuffer((__gm__ float*)y + coreId * tileDataNum);

    WaitFlag<HardEvent::MTE2_V>(evtM2V);
    ErfCalc(yL, xL, tmp, length);

    event_t evtV2M = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(evtV2M);
    WaitFlag<HardEvent::V_MTE3>(evtV2M);
    DataCopy(yGm, yL, length);
}

// ======================== Kernel入口 ========================
template <uint32_t schMode>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, td, tiling);
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_1) {
        KernelErfDB op;
        op.Init(x, y, td.bigCoreDataNum, td.finalBigTileNum, td.tileDataNum, td.bigTailDataNum,
                     td.smallCoreDataNum, td.finalSmallTileNum, td.smallTailDataNum, &pipe);
        op.Process();
    }
    if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_0) {
        KernelErfNDB(x, y, td.tileDataNum, td.bigTailDataNum, &pipe);
    }
}