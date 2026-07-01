#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t ALIGN_NUM = 8;

template <class DT_X>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length, uint32_t blockLength, uint32_t tileLength) {
        this->length = length;
        this->blockLength = blockLength;
        this->tileLength = tileLength;

        uint32_t block_idx = AscendC::GetBlockIdx();
        uint32_t start = block_idx * blockLength;
        if (start >= length) {
            this->blockOffset = length;
            this->blockActualLength = 0;
            return;
        }

        this->blockOffset = start;
        uint32_t remain = length - start;
        this->blockActualLength = remain < blockLength ? remain : blockLength;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + blockOffset, blockActualLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + blockOffset, blockActualLength);

        pipe.InitBuffer(xQueue, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(yQueue, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(calcBuf, tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (blockActualLength == 0) {
            return;
        }

        CopyIn(0, GetTileCount(0));
        for (uint32_t offset = 0; offset < blockActualLength; offset += tileLength) {
            uint32_t nextOffset = offset + tileLength;
            if (nextOffset < blockActualLength) {
                CopyIn(nextOffset, GetTileCount(nextOffset));
            }

            uint32_t calc_count = GetTileCount(offset);
            Compute(calc_count);
            CopyOut(offset, calc_count);
        }
    }

private:
    __aicore__ inline uint32_t GetTileCount(uint32_t offset) const {
        uint32_t remain = blockActualLength - offset;
        return remain < tileLength ? remain : tileLength;
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t calcCount) {
        AscendC::LocalTensor<DT_X> x_local = xQueue.AllocTensor<DT_X>();
        if ((calcCount & (ALIGN_NUM - 1)) == 0) {
            AscendC::DataCopy(x_local, xGm[offset], calcCount);
        } else {
            AscendC::DataCopyExtParams copy_params{1, static_cast<uint32_t>(calcCount * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_X> pad_params{false, 0, 0, 0};
            AscendC::DataCopyPad(x_local, xGm[offset], copy_params, pad_params);
        }
        xQueue.EnQue(x_local);
    }

    __aicore__ inline void Compute(uint32_t calcCount) {
        AscendC::LocalTensor<DT_X> x_local = xQueue.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> y_local = yQueue.AllocTensor<DT_X>();
#if defined(ERF_USE_ASCENDC_BUILTIN)
        AscendC::Erf(y_local, x_local, calcCount);
#else
        FastErf(y_local, x_local, calcCount);
#endif
        yQueue.EnQue(y_local);
        xQueue.FreeTensor(x_local);
    }

    __aicore__ inline void FastErf(AscendC::LocalTensor<DT_X> y, AscendC::LocalTensor<DT_X> x, uint32_t count) {
        AscendC::LocalTensor<DT_X> x2 = calcBuf.Get<DT_X>();

        AscendC::Mul(x2, x, x, count);
        AscendC::Muls(y, x2, static_cast<DT_X>(4.817597158808e-08f), count);
        AscendC::Adds(y, y, static_cast<DT_X>(-2.219404783531e-06f), count);
        AscendC::Mul(y, y, x2, count);
        AscendC::Adds(y, y, static_cast<DT_X>(4.523259585903e-05f), count);
        AscendC::Mul(y, y, x2, count);
        AscendC::Adds(y, y, static_cast<DT_X>(-5.446524481595e-04f), count);
        AscendC::Mul(y, y, x2, count);
        AscendC::Adds(y, y, static_cast<DT_X>(4.389702257440e-03f), count);
        AscendC::Mul(y, y, x2, count);
        AscendC::Adds(y, y, static_cast<DT_X>(-2.550514823992e-02f), count);
        AscendC::Mul(y, y, x2, count);
        AscendC::Adds(y, y, static_cast<DT_X>(1.116382595732e-01f), count);
        AscendC::Mul(y, y, x2, count);
        AscendC::Adds(y, y, static_cast<DT_X>(-3.756677716195e-01f), count);
        AscendC::Mul(y, y, x2, count);
        AscendC::Adds(y, y, static_cast<DT_X>(1.128335662832e+00f), count);
        AscendC::Mul(y, y, x, count);
        AscendC::Mins(y, y, static_cast<DT_X>(1.0f), count);
        AscendC::Maxs(y, y, static_cast<DT_X>(-1.0f), count);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t calcCount) {
        AscendC::LocalTensor<DT_X> y_local = yQueue.DeQue<DT_X>();
        if ((calcCount & (ALIGN_NUM - 1)) == 0) {
            AscendC::DataCopy(yGm[offset], y_local, calcCount);
        } else {
            AscendC::DataCopyExtParams copy_params{1, static_cast<uint32_t>(calcCount * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPad(yGm[offset], y_local, copy_params);
        }
        yQueue.FreeTensor(y_local);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> xQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> yQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t length = 0;
    uint32_t blockLength = 0;
    uint32_t blockOffset = 0;
    uint32_t blockActualLength = 0;
    uint32_t tileLength = 0;
};

template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    (void)workspace;
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    KernelErf<DT_X> op;
    op.Init(x, y, tiling_data.length, tiling_data.blockLength, tiling_data.tileLength);
    op.Process();
}
