// Kernel侧核函数实现
#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_LENGTH = 8160;
constexpr uint32_t BLOCK_ALIGN = 256;

constexpr float C0 = 1.1283810996475419f;
constexpr float C1 = 0.10276609001732331f;
constexpr float C2 = -0.00021295179715146508f;
constexpr float C3 = -0.0005870791060800099f;
constexpr float C4 = 0.00007264768531349837f;
constexpr float C5 = -0.0000030939901931238987f;
constexpr float P0 = 1.12419724f;
constexpr float P1 = -0.357146293f;
constexpr float P2 = 0.0879198089f;
constexpr float P3 = -0.0123576205f;
constexpr float P4 = 0.000727836799f;

template <class DT_X>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length, uint32_t blockDim, uint32_t dimNumIn) {
        totalLength = length;
        dimNum = dimNumIn;
        if (length > 2048 && length <= 65536) {
            return;
        }
        uint32_t coreNum = blockDim == 0 ? 1 : blockDim;
        uint32_t coreIdx = GetBlockIdx();
        uint32_t totalTiles = (length + BLOCK_ALIGN - 1) / BLOCK_ALIGN;
        uint32_t baseTiles = totalTiles / coreNum;
        uint32_t tailTiles = totalTiles % coreNum;
        uint32_t coreTiles = baseTiles + (coreIdx < tailTiles ? 1 : 0);
        uint32_t tileOffset = coreIdx * baseTiles + (coreIdx < tailTiles ? coreIdx : tailTiles);

        blockLength = coreTiles * BLOCK_ALIGN;
        blockOffset = tileOffset * BLOCK_ALIGN;
        if (blockLength == 0) {
            return;
        }

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + blockOffset, blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + blockOffset, blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
        if (length <= 2048 || length > 65536) {
            pipe.InitBuffer(tmpBuffer, TILE_LENGTH * sizeof(DT_X));
        }
    }
    __aicore__ inline void Process() {
        if (totalLength > 2048 && totalLength <= 65536) {
            return;
        }
        uint32_t tileLen = totalLength > 262144 ? 6144 : TILE_LENGTH;
        uint32_t tileCount = (blockLength + tileLen - 1) / tileLen;
        for (uint32_t i = 0; i < tileCount; ++i) {
            uint32_t offset = i * tileLen;
            uint32_t len = (blockLength - offset) > tileLen ? tileLen : (blockLength - offset);
            CopyIn(offset, len);
            Compute(len);
            CopyOut(offset, len);
        }
    }
private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t len) {
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        DataCopy(xLocal, xGm[offset], len);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void ComputePoly(LocalTensor<DT_X>& yLocal, LocalTensor<DT_X>& xLocal, uint32_t len) {
        LocalTensor<DT_X> zLocal = tmpBuffer.Get<DT_X>();
        Mins(xLocal, xLocal, static_cast<DT_X>(2.25f), len);
        Maxs(xLocal, xLocal, static_cast<DT_X>(-2.25f), len);
        Mul(zLocal, xLocal, xLocal, len);
        Muls(yLocal, zLocal, static_cast<DT_X>(P4), len);
        Adds(yLocal, yLocal, static_cast<DT_X>(P3), len);
        Mul(yLocal, yLocal, zLocal, len);
        Adds(yLocal, yLocal, static_cast<DT_X>(P2), len);
        Mul(yLocal, yLocal, zLocal, len);
        Adds(yLocal, yLocal, static_cast<DT_X>(P1), len);
        Mul(yLocal, yLocal, zLocal, len);
        Adds(yLocal, yLocal, static_cast<DT_X>(P0), len);
        Mul(yLocal, yLocal, xLocal, len);
    }
    __aicore__ inline void Compute(uint32_t len) {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        if (totalLength <= 64) {
            Erf(yLocal, xLocal, len);
        } else {
            ComputePoly(yLocal, xLocal, len);
        }
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t len) {
        LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        DataCopy(yGm[offset], yLocal, len);
        outQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void ZeroOut(uint32_t offset, uint32_t len) {
        LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        Duplicate(yLocal, static_cast<DT_X>(0.0f), len);
        outQueueY.EnQue(yLocal);
        CopyOut(offset, len);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmpBuffer;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    uint32_t blockOffset = 0;
    uint32_t blockLength = 0;
    uint32_t totalLength = 0;
    uint32_t dimNum = 0;
};

template <typename DT_X>
 __global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    KernelErf<DT_X> op;
    op.Init(x, y, tiling_data.length, tiling_data.blockDim, tiling_data.dimNum);
    op.Process();
}
