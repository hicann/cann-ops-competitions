#include "kernel_operator.h"
#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

// 完全继承 GPT 的高优 4 阶公式
constexpr float ERF_CLIP = 2.19f;
constexpr float C0 = 1.12727035e+00f;
constexpr float C1 = -3.65317645e-01f;
constexpr float C2 = 9.40435818e-02f;
constexpr float C3 = -1.40861894e-02f;
constexpr float C4 = 8.91647912e-04f;

__aicore__ inline void ErfPoly(LocalTensor<float> &x, LocalTensor<float> &y, LocalTensor<float> &z, uint32_t len) {
    Mins(x, x, ERF_CLIP, len);
    Maxs(x, x, -ERF_CLIP, len);
    Mul(z, x, x, len);
    Muls(y, z, C4, len);
    Adds(y, y, C3, len);
    Mul(y, y, z, len);
    Adds(y, y, C2, len);
    Mul(y, y, z, len);
    Adds(y, y, C1, len);
    Mul(y, y, z, len);
    Adds(y, y, C0, len);
    Mul(y, x, y, len);
}

// ---------------------------------------------------------
// 小张量静态 Tensor 处理：去掉 TPipe/TBuf/InitBuffer 管理开销
// ---------------------------------------------------------
template <typename T>
__aicore__ inline void ProcessSmall(GM_ADDR x, GM_ADDR y, uint32_t start, uint32_t len) {
    GlobalTensor<T> xGm, yGm;
    xGm.SetGlobalBuffer((__gm__ T*)x + start, len);
    yGm.SetGlobalBuffer((__gm__ T*)y + start, len);

    const uint32_t alignLen = (len + 7U) & ~7U;

    // 静态 Tensor：不创建 TPipe，不调用 InitBuffer。
    // LocalMemAllocator 是线性 UB 分配器，适合 small latency path。
    LocalMemAllocator<Hardware::UB> ubAlloc;
    LocalTensor<float> xL = ubAlloc.Alloc<float>(alignLen);
    LocalTensor<float> yL = ubAlloc.Alloc<float>(alignLen);
    LocalTensor<float> zL = ubAlloc.Alloc<float>(alignLen);

    if ((len & 7U) == 0U) {
        DataCopy(xL, xGm, len);
        SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
        ErfPoly(xL, yL, zL, len);
        SetFlag<HardEvent::V_MTE3>(1); WaitFlag<HardEvent::V_MTE3>(1);
        DataCopy(yGm, yL, len);
    } else {
        DataCopyParams cp{1, static_cast<uint16_t>(len << 2), 0, 0};
        DataCopyPadParams pp{false, 0, 0, 0};
        DataCopyPad(xL, xGm, cp, pp);
        SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
        ErfPoly(xL, yL, zL, alignLen);
        SetFlag<HardEvent::V_MTE3>(1); WaitFlag<HardEvent::V_MTE3>(1);
        DataCopyPad(yGm, yL, cp);
    }
}

// ---------------------------------------------------------
// 🚀 大张量流水线
// ---------------------------------------------------------
template <typename T>
__aicore__ inline void ProcessPipeline(GM_ADDR x, GM_ADDR y, uint32_t start, uint32_t len, uint32_t tileLen, TPipe* pipe) {
    GlobalTensor<T> xGm, yGm;
    xGm.SetGlobalBuffer((__gm__ T*)x + start, len);
    yGm.SetGlobalBuffer((__gm__ T*)y + start, len);

    TQue<QuePosition::VECIN, 2> inQ;
    TQue<QuePosition::VECOUT, 2> outQ;
    TBuf<TPosition::VECCALC> zB;

    uint32_t bytes = tileLen << 2;
    pipe->InitBuffer(inQ, 2, bytes);
    pipe->InitBuffer(outQ, 2, bytes);
    pipe->InitBuffer(zB, bytes);

    LocalTensor<float> zL = zB.Get<float>();

    uint32_t offset = 0;
    uint32_t alignLen = len & ~7;
    uint32_t tail = len - alignLen;

    while (offset < alignLen) {
        uint32_t cur = alignLen - offset;
        if (cur > tileLen) cur = tileLen;

        LocalTensor<float> xL = inQ.AllocTensor<float>();
        DataCopy(xL, xGm[offset], cur);
        inQ.EnQue(xL);

        LocalTensor<float> xD = inQ.DeQue<float>();
        LocalTensor<float> yL = outQ.AllocTensor<float>();
        ErfPoly(xD, yL, zL, cur);
        outQ.EnQue(yL);
        inQ.FreeTensor(xD);

        LocalTensor<float> yD = outQ.DeQue<float>();
        DataCopy(yGm[offset], yD, cur);
        outQ.FreeTensor(yD);

        offset += cur;
    }

    if (tail > 0) {
        LocalTensor<float> xL = inQ.AllocTensor<float>();
        DataCopyParams cp{1, (uint16_t)(tail << 2), 0, 0};
        DataCopyPadParams pp{false, 0, 0, 0};
        DataCopyPad(xL, xGm[offset], cp, pp);
        inQ.EnQue(xL);

        LocalTensor<float> xD = inQ.DeQue<float>();
        LocalTensor<float> yL = outQ.AllocTensor<float>();
        ErfPoly(xD, yL, zL, 8);
        outQ.EnQue(yL);
        inQ.FreeTensor(xD);

        LocalTensor<float> yD = outQ.DeQue<float>();
        DataCopyPad(yGm[offset], yD, cp);
        outQ.FreeTensor(yD);
    }
}

// ---------------------------------------------------------
// 究极入口：严格遵循 GPT 模板规范，彻底规避宏冲突
// ---------------------------------------------------------
template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);

    __gm__ uint32_t* tp = (__gm__ uint32_t*)tiling;
    uint32_t totalLength = tp[0];
    if (totalLength == 0) return;

    uint32_t tileLen = tp[1];
    uint32_t baseLen = tp[2];
    uint32_t mode = tp[3];

    // 多核切割：确保 40 个核各司其职，处理属于自己的那块内存
    uint32_t start = baseLen * GetBlockIdx();
    if (start >= totalLength) return;
    
    uint32_t len = totalLength - start;
    if (len > baseLen) len = baseLen;

    if (mode == 1) {
        ProcessSmall<DT_X>(x, y, start, len);
    } else {
        TPipe pipe;
        ProcessPipeline<DT_X>(x, y, start, len, tileLen, &pipe);
    }
}