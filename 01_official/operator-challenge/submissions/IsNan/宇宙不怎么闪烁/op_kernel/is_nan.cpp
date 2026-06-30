#include "kernel_operator.h"
using namespace AscendC;

#define DUMP(tensor)     DumpTensor(tensor, __LINE__, 16)
#define DUMPN(tensor, n) DumpTensor(tensor, __LINE__, n)
#define PRINTF(fmt, ...)                                                                           \
    printf("================LINE %d================\n" fmt, __LINE__, ##__VA_ARGS__);
#define min(a, b) ((a) < (b) ? (a) : (b))

template <typename T> class KernelIsNan {
  public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, int64_t totalLength, int32_t tileLength,
                                int32_t perCoreBlock, int32_t lastCoreBlock);
    __aicore__ inline void Process();

  private:
    static constexpr int32_t BUF_NUM        = 1;
    static constexpr int32_t ALIGN_BYTES    = 1024;
    static constexpr int32_t ELEM_PER_BLOCK = ALIGN_BYTES / sizeof(T);

    GlobalTensor<T>       xGm;
    GlobalTensor<uint8_t> yGm;

    int64_t totalLen = 0;
    int32_t localLen = 0;
    int32_t tileLen  = 0;
    int32_t tileNum  = 0;
    int32_t lastLen  = 0;
    int32_t start    = 0;

    __aicore__ inline void InitCoreRange(int32_t perCoreBlock, int32_t lastCoreBlock);

    __aicore__ inline LocalTensor<T>       GetBufX(int32_t bufIdx);
    __aicore__ inline LocalTensor<uint8_t> GetBufY(int32_t bufIdx);
    __aicore__ inline LocalTensor<uint8_t> GetMskU8(int32_t bufIdx);
    __aicore__ inline LocalTensor<half>    GetTmpHalf(int32_t bufIdx);
    __aicore__ inline LocalTensor<float>   GetTmpFlt(int32_t bufIdx);

    __aicore__ inline void Pipe_CopyIn(int32_t bufIdx, int32_t tile, int32_t curLen);
    __aicore__ inline void Pipe_Compute(int32_t bufIdx, int32_t tile, int32_t curLen, bool waitPrev,
                                        bool setNext);
    __aicore__ inline void Pipe_CopyOut(int32_t bufIdx, int32_t tile, int32_t curLen, bool setNext);

    LocalTensor<half> zeros;

    __aicore__ inline void FillConstants();
};

// ============================================================================
//  Init
// ============================================================================

template <typename T>
__aicore__ inline void KernelIsNan<T>::Init(GM_ADDR x, GM_ADDR y, int64_t totalLength,
                                            int32_t tileLength, int32_t perCoreBlock,
                                            int32_t lastCoreBlock) {
    xGm.SetGlobalBuffer((__gm__ T *)x, totalLength);
    yGm.SetGlobalBuffer((__gm__ uint8_t *)y, totalLength);

    totalLen = totalLength;
    tileLen  = tileLength;
    InitCoreRange(perCoreBlock, lastCoreBlock);
}

template <typename T>
__aicore__ inline void KernelIsNan<T>::InitCoreRange(int32_t perCoreBlock, int32_t lastCoreBlock) {
    int32_t coreNum = static_cast<int32_t>(GetBlockNum());
    int32_t coreIdx = static_cast<int32_t>(GetBlockIdx());

    if (coreNum == 40 && totalLen > 10000000) { // memory-bound
        if (coreIdx & 1) return;
        coreIdx = (coreIdx >> 1) + ((coreIdx & 1) ? 20 : 0);
        coreNum = 20;
    }

    int32_t myBlock = perCoreBlock + ((coreIdx < lastCoreBlock) ? 1 : 0);
    start           = (coreIdx * perCoreBlock + min(coreIdx, lastCoreBlock)) * ELEM_PER_BLOCK;
    localLen        = myBlock * ELEM_PER_BLOCK;

    tileNum = (localLen + tileLen - 1) / tileLen;
    lastLen = localLen - tileLen * (tileNum - 1);
}

// ============================================================================
//  Buf getters
// ============================================================================

template <typename T> __aicore__ inline LocalTensor<T> KernelIsNan<T>::GetBufX(int32_t bufIdx) {
    return LocalTensor<T>(TPosition::VECCALC, bufIdx * tileLen * sizeof(T), tileLen);
}

template <typename T>
__aicore__ inline LocalTensor<uint8_t> KernelIsNan<T>::GetBufY(int32_t bufIdx) {
    int32_t off = BUF_NUM * tileLen * sizeof(T) + bufIdx * tileLen * sizeof(uint8_t);
    return LocalTensor<uint8_t>(TPosition::VECCALC, off, tileLen);
}

// ============================================================================
//  Constant fill
// ============================================================================

template <typename T> __aicore__ inline void KernelIsNan<T>::FillConstants() {
    int32_t base = BUF_NUM * tileLen * sizeof(T) + BUF_NUM * tileLen * sizeof(uint8_t);
    zeros        = LocalTensor<half>(TPosition::VECCALC, base, tileLen);
    Duplicate(zeros, static_cast<half>(0), tileLen);
}

// ============================================================================
//  Tmp getters
// ============================================================================

template <typename T>
__aicore__ inline LocalTensor<uint8_t> KernelIsNan<T>::GetMskU8(int32_t bufIdx) {
    (void)bufIdx;
    int32_t off = BUF_NUM * tileLen * sizeof(T) + BUF_NUM * tileLen * sizeof(uint8_t) +
                  tileLen * sizeof(half);
    return LocalTensor<uint8_t>(TPosition::VECCALC, off, tileLen);
}

template <typename T>
__aicore__ inline LocalTensor<half> KernelIsNan<T>::GetTmpHalf(int32_t bufIdx) {
    (void)bufIdx;
    int32_t off = BUF_NUM * tileLen * sizeof(T) + BUF_NUM * tileLen * sizeof(uint8_t) +
                  tileLen * sizeof(half) + tileLen * sizeof(uint8_t);
    return LocalTensor<half>(TPosition::VECCALC, off, tileLen);
}

template <typename T>
__aicore__ inline LocalTensor<float> KernelIsNan<T>::GetTmpFlt(int32_t bufIdx) {
    (void)bufIdx;
    int32_t off = BUF_NUM * tileLen * sizeof(T) + BUF_NUM * tileLen * sizeof(uint8_t) +
                  tileLen * sizeof(half) + tileLen * sizeof(uint8_t);
    return LocalTensor<float>(TPosition::VECCALC, off, tileLen);
}

// ============================================================================
//  Pipeline stages
// ============================================================================

template <typename T>
__aicore__ inline void KernelIsNan<T>::Pipe_CopyIn(int32_t bufIdx, int32_t tile, int32_t curLen) {
    (void)tile;
    DataCopy(GetBufX(bufIdx), xGm[start + tile * tileLen], curLen);
    SetFlag<HardEvent::MTE2_V>(bufIdx);
}

template <typename T>
__aicore__ inline void KernelIsNan<T>::Pipe_Compute(int32_t bufIdx, int32_t tile, int32_t curLen,
                                                    bool waitPrev, bool setNext) {
    (void)tile;
    auto xBuf = GetBufX(bufIdx);
    auto yBuf = GetBufY(bufIdx);

    if constexpr (std::is_same_v<T, float>) {
        auto tmpHalf = GetTmpHalf(bufIdx);
        auto mskU8   = GetMskU8(bufIdx);

        Compare(mskU8, xBuf, xBuf, CMPMODE::EQ, curLen);
        if (setNext) SetFlag<HardEvent::V_MTE2>(bufIdx);

        Select(tmpHalf, mskU8, zeros, static_cast<half>(1), SELMODE::VSEL_TENSOR_SCALAR_MODE,
               curLen);

        if (waitPrev) WaitFlag<HardEvent::MTE3_V>(bufIdx);
        Cast(yBuf, tmpHalf, RoundMode::CAST_NONE, curLen);

    } else if constexpr (std::is_same_v<T, half>) {
        auto tmpHalf = GetTmpHalf(bufIdx);
        auto mskU8   = GetMskU8(bufIdx);

        Compare(mskU8, xBuf, xBuf, CMPMODE::EQ, curLen);
        if (setNext) SetFlag<HardEvent::V_MTE2>(bufIdx);

        Select(tmpHalf, mskU8, zeros, static_cast<half>(1), SELMODE::VSEL_TENSOR_SCALAR_MODE,
               curLen);

        if (waitPrev) WaitFlag<HardEvent::MTE3_V>(bufIdx);
        Cast(yBuf, tmpHalf, RoundMode::CAST_NONE, curLen);

    } else {
        auto tmpHalf = GetTmpHalf(bufIdx);
        auto mskU8   = GetMskU8(bufIdx);
        auto tmpFlt  = GetTmpFlt(bufIdx);

        Cast(tmpFlt, xBuf, RoundMode::CAST_NONE, curLen);
        if (setNext) SetFlag<HardEvent::V_MTE2>(bufIdx);

        Compare(mskU8, tmpFlt, tmpFlt, CMPMODE::EQ, curLen);
        Select(tmpHalf, mskU8, zeros, static_cast<half>(1), SELMODE::VSEL_TENSOR_SCALAR_MODE,
               curLen);

        if (waitPrev) WaitFlag<HardEvent::MTE3_V>(bufIdx);
        Cast(yBuf, tmpHalf, RoundMode::CAST_NONE, curLen);
    }

    SetFlag<HardEvent::V_MTE3>(bufIdx);
}

template <typename T>
__aicore__ inline void KernelIsNan<T>::Pipe_CopyOut(int32_t bufIdx, int32_t tile, int32_t curLen,
                                                    bool setNext) {
    DataCopy(yGm[start + tile * tileLen], GetBufY(bufIdx), curLen);
    if (setNext) SetFlag<HardEvent::MTE3_V>(bufIdx);
}

// ============================================================================
//  Process
// ============================================================================

template <typename T> __aicore__ inline void KernelIsNan<T>::Process() {
    if (tileNum <= 0) return;

    FillConstants();

    for (int32_t i = 0; i < tileNum; i += BUF_NUM) {
        for (int32_t j = 0; j < BUF_NUM; j++) {
            int32_t tile = i + j;
            if (tile >= tileNum) break;

            int32_t curLen   = (tile == tileNum - 1) ? lastLen : tileLen;
            bool    waitPrev = (tile >= BUF_NUM);
            bool    setNext  = (tile + BUF_NUM < tileNum);

            if (waitPrev) WaitFlag<HardEvent::V_MTE2>(j);
            Pipe_CopyIn(j, tile, curLen);

            WaitFlag<HardEvent::MTE2_V>(j);
            Pipe_Compute(j, tile, curLen, waitPrev, setNext);

            WaitFlag<HardEvent::V_MTE3>(j);
            Pipe_CopyOut(j, tile, curLen, setNext);
        }
    }
}

// ============================================================================
//  Entry
// ============================================================================

extern "C" __global__ __aicore__ void is_nan(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                             GM_ADDR tiling) {
    InitSocState();
    GET_TILING_DATA(tiling_data, tiling);
    KernelIsNan<DTYPE_X> op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileLength, tiling_data.perCoreBlock,
            tiling_data.lastCoreBlock);
    op.Process();
}
