#include "kernel_operator.h"
#include "erf_tiling.h"
#include "tiling_key_erf.h"

__aicore__ inline float erf_tiny_scalar(float x) {
    return x * (1.128379167f - 0.376126389f * x * x);
}

__aicore__ inline float erf_scalar(float x) {
    if (x >= 2.8f) return 1.0f;
    if (x <= -2.8f) return -1.0f;
    if (x >= -0.2f && x <= 0.2f) {
        return erf_tiny_scalar(x);
    }
    float x2 = x * x;
    float n = 0.0000022166f;
    n = n * x2 + 0.0002930812f; n = n * x2 + 0.0036894527f;
    n = n * x2 + 0.0525886436f; n = n * x2 + 0.1873858920f;
    n = n * x2 + 1.1283791545f; n = n * x;
    float d = 0.0000402078f;
    d = d * x2 + 0.0011613911f; d = d * x2 + 0.0148240933f;
    d = d * x2 + 0.1130755186f; d = d * x2 + 0.4993990490f;
    d = d * x2 + 1.0f;
    return n / d;
}

__aicore__ inline void ProcessScalar(__gm__ float* xp, __gm__ float* yp, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        yp[i] = erf_scalar(xp[i]);
    }
}

__aicore__ inline void RunTiles(AscendC::GlobalTensor<float>& xGm,
                                AscendC::GlobalTensor<float>& yGm,
                                AscendC::TPipe& pipe,
                                uint32_t total, uint32_t tile) {
    if (total <= tile) {
        if (total <= 96) {
            AscendC::TBuf<AscendC::TPosition::VECCALC> xBuf, yBuf;
            pipe.InitBuffer(xBuf, total * sizeof(float));
            pipe.InitBuffer(yBuf, total * sizeof(float));
            auto xL = xBuf.Get<float>();
            auto yL = yBuf.Get<float>();

            AscendC::DataCopy(xL, xGm, total);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::Erf<float>(yL, xL, total);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::DataCopy(yGm, yL, total);
            return;
        }

        AscendC::TQue<AscendC::QuePosition::VECIN, 1> iQ;
        AscendC::TQue<AscendC::QuePosition::VECOUT, 1> oQ;
        pipe.InitBuffer(iQ, 1, total * sizeof(float));
        pipe.InitBuffer(oQ, 1, total * sizeof(float));

        auto xL = iQ.AllocTensor<float>();
        AscendC::DataCopy(xL, xGm, total);
        iQ.EnQue(xL);

        auto xC = iQ.DeQue<float>();
        auto yL = oQ.AllocTensor<float>();
        AscendC::Erf<float>(yL, xC, total);
        iQ.FreeTensor(xC);
        oQ.EnQue(yL);

        auto yO = oQ.DeQue<float>();
        AscendC::DataCopy(yGm, yO, total);
        oQ.FreeTensor(yO);
        return;
    }

    uint32_t nFull = total / tile;
    uint32_t tailLen = total - nFull * tile;

    AscendC::TQue<AscendC::QuePosition::VECIN, 2> iQ;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> oQ;
    pipe.InitBuffer(iQ, 2, tile * sizeof(float));
    pipe.InitBuffer(oQ, 2, tile * sizeof(float));

    {
        auto xL = iQ.AllocTensor<float>();
        AscendC::DataCopy(xL, xGm, tile);
        iQ.EnQue(xL);
    }

    for (uint32_t t = 0; t < nFull; ++t) {
        if (t + 1 < nFull) {
            auto xN = iQ.AllocTensor<float>();
            AscendC::DataCopy(xN, xGm[(t + 1) * tile], tile);
            iQ.EnQue(xN);
        }
        auto xC = iQ.DeQue<float>();
        auto yL = oQ.AllocTensor<float>();
        AscendC::Erf<float>(yL, xC, tile);
        iQ.FreeTensor(xC);
        oQ.EnQue(yL);
        auto yO = oQ.DeQue<float>();
        AscendC::DataCopy(yGm[t * tile], yO, tile);
        oQ.FreeTensor(yO);
    }

    if (tailLen > 0) {
        uint32_t off = nFull * tile;
        auto xL = iQ.AllocTensor<float>();
        AscendC::DataCopy(xL, xGm[off], tailLen);
        iQ.EnQue(xL);
        auto xC = iQ.DeQue<float>();
        auto yL = oQ.AllocTensor<float>();
        AscendC::Erf<float>(yL, xC, tailLen);
        iQ.FreeTensor(xC);
        oQ.EnQue(yL);
        auto yO = oQ.DeQue<float>();
        AscendC::DataCopy(yGm[off], yO, tailLen);
        oQ.FreeTensor(yO);
    }
}

template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, td, tiling);

    uint32_t bi = static_cast<uint32_t>(AscendC::GetBlockIdx());
    uint32_t coreLen; uint64_t off;
    if (bi < td.formerNum) {
        coreLen = td.formerLength;
        off = static_cast<uint64_t>(td.formerLength) * bi;
    } else {
        coreLen = td.tailLength;
        if (coreLen == 0) return;
        off = static_cast<uint64_t>(td.formerLength) * td.formerNum
            + static_cast<uint64_t>(td.tailLength) * (bi - td.formerNum);
    }
    if (off >= td.totalLength) return;

    uint32_t valid = coreLen;
    uint64_t rem = static_cast<uint64_t>(td.totalLength) - off;
    if (rem < valid) valid = static_cast<uint32_t>(rem);
    if (valid == 0) return;

    __gm__ float* xp = reinterpret_cast<__gm__ float*>(x) + off;
    __gm__ float* yp = reinterpret_cast<__gm__ float*>(y) + off;

    if (valid < 8) {
        ProcessScalar(xp, yp, valid);
        return;
    }

    constexpr uint32_t kAlign = 8;
    uint32_t vecLen = (valid / kAlign) * kAlign;
    uint32_t sclTail = valid - vecLen;

    if (vecLen > 0) {
        AscendC::GlobalTensor<float> xGm, yGm;
        xGm.SetGlobalBuffer(xp, vecLen);
        yGm.SetGlobalBuffer(yp, vecLen);
        AscendC::TPipe pipe;

        uint32_t tile = (vecLen <= 2048) ? 2048
                       : ((vecLen <= 16384) ? 4096 : 8192);
        RunTiles(xGm, yGm, pipe, vecLen, tile);
    }

    if (sclTail > 0) {
        ProcessScalar(xp + vecLen, yp + vecLen, sclTail);
    }
}
