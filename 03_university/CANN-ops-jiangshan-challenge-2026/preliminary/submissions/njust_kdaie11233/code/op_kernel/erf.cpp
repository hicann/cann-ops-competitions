// Kernel侧核函数实现
#include "kernel_operator.h"
#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

template <class DT_X>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t wsSize) {
        xGm.SetGlobalBuffer((__gm__ DT_X*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y, totalLength);
        workspaceSize = wsSize;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        uint32_t rawLen = (totalLength + blockNum - 1) / blockNum;
        rawLen = (rawLen + 7) / 8 * 8;

        startIdx = blockIdx * rawLen;
        endIdx   = startIdx + rawLen;
        if (startIdx > totalLength) startIdx = totalLength;
        if (endIdx   > totalLength) endIdx   = totalLength;

        coreLen = endIdx - startIdx;
        uint32_t tsize = (coreLen <= TILE_LEN) ? coreLen : TILE_LEN;
        uint32_t bufB = tsize * sizeof(DT_X);
        uint32_t d = (coreLen <= TILE_LEN) ? 1 : 2;
        pipe.InitBuffer(inQ, d, bufB);
        pipe.InitBuffer(outQ, d, bufB);
    }

    __aicore__ inline bool HasWork() { return startIdx < endIdx; }

    __aicore__ inline void Process() {
        if (coreLen == 0) return;

        if (coreLen <= TILE_LEN) {
            Tile(0, coreLen);
            return;
        }

        uint32_t o = 0;
        for (; o + TILE_LEN <= coreLen; o += TILE_LEN) Tile(o, TILE_LEN);
        if (o < coreLen) Tile(o, coreLen - o);
    }
private:
    static constexpr uint32_t TILE_LEN = 8192;
    uint32_t startIdx = 0, endIdx = 0, coreLen = 0;
    uint32_t workspaceSize = 0;

    __aicore__ inline void Tile(uint32_t off, uint32_t sz) {
        uint32_t N = (sz + 7) / 8 * 8;

        LocalTensor<DT_X> x = inQ.AllocTensor<DT_X>();
        DataCopy(x, xGm[startIdx + off], N);
        inQ.EnQue<DT_X>(x);

        LocalTensor<DT_X> y = outQ.AllocTensor<DT_X>();
        x = inQ.DeQue<DT_X>();

        AscendC::Erf<DT_X, false>(y, x);

        outQ.EnQue<DT_X>(y);
        inQ.FreeTensor(x);

        y = outQ.DeQue<DT_X>();
        DataCopy(yGm[startIdx + off], y, N);
        outQ.FreeTensor(y);
    }

    GlobalTensor<DT_X> xGm, yGm;
    TPipe pipe;
    TQue<QuePosition::VECIN,   2> inQ;
    TQue<QuePosition::VECOUT,  2> outQ;
};

template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    KernelErf<DT_X> op;
    op.Init(x, y, tiling_data.length, tiling_data.workspaceSize);
    if (op.HasWork()) op.Process();
}
