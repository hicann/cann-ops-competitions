#define K_MAX_SHAPE_DIM 0

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t TQUE_NUM = 1;

template <typename T> class ErfinvKernel {
private:
  TPipe *pipe;
  TQue<QuePosition::VECIN, TQUE_NUM> inQueueX;
  TQue<QuePosition::VECOUT, TQUE_NUM> outQueueY;
  TQue<QuePosition::VECCALC, TQUE_NUM> calcQueueA;
  TQue<QuePosition::VECCALC, TQUE_NUM> calcQueueB;
  TQue<QuePosition::VECCALC, TQUE_NUM> calcQueueC;
  TQue<QuePosition::VECCALC, TQUE_NUM> calcQueueMask;
  GlobalTensor<T> xGm;
  GlobalTensor<T> yGm;
  uint32_t blockLength;
  uint32_t tileLength;

public:
  __aicore__ inline ErfinvKernel() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, TPipe *pipeIn,
                              uint32_t coreSize, uint32_t lastCoreSize,
                              uint32_t tileLengthIn) {
    pipe = pipeIn;
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockNum = GetBlockNum();
    blockLength = coreSize;
    if (blockIdx + 1 == blockNum) {
      blockLength = lastCoreSize;
    }
    tileLength = tileLengthIn;
    if (tileLength > blockLength) {
      tileLength = blockLength;
    }
    uint32_t start = blockIdx * coreSize;
    xGm.SetGlobalBuffer((__gm__ T *)x + start, blockLength);
    yGm.SetGlobalBuffer((__gm__ T *)y + start, blockLength);

    uint32_t alignedT = AlignElements<T>(tileLength);
    uint32_t alignedF = AlignCompareElements(tileLength);
    pipe->InitBuffer(inQueueX, BUFFER_NUM, alignedT * sizeof(T));
    pipe->InitBuffer(outQueueY, BUFFER_NUM, alignedT * sizeof(T));
    pipe->InitBuffer(calcQueueA, BUFFER_NUM, alignedF * sizeof(float));
    pipe->InitBuffer(calcQueueB, BUFFER_NUM, alignedF * sizeof(float));
    pipe->InitBuffer(calcQueueC, BUFFER_NUM, alignedF * sizeof(float));
    pipe->InitBuffer(calcQueueMask, BUFFER_NUM, AlignMaskBytes(alignedF));
  }

  __aicore__ inline void Process() {
    for (uint32_t offset = 0; offset < blockLength; offset += tileLength) {
      uint32_t calcLength = tileLength;
      if (offset + calcLength > blockLength) {
        calcLength = blockLength - offset;
      }
      Compute(offset, calcLength);
    }
  }

private:
  template <typename U> __aicore__ inline uint32_t AlignElements(uint32_t n) {
    return ((n * sizeof(U) + 31) / 32) * 32 / sizeof(U);
  }

  __aicore__ inline uint32_t AlignCompareElements(uint32_t n) {
    return ((n + 63) / 64) * 64;
  }

  __aicore__ inline uint32_t AlignMaskBytes(uint32_t n) {
    return ((n / 8 + 31) / 32) * 32;
  }

  __aicore__ inline void Horner(LocalTensor<float> &dst,
                                 const LocalTensor<float> &w, float c,
                                 uint32_t count) {
    Mul(dst, dst, w, count);
    Adds(dst, dst, c, count);
  }

  __aicore__ inline void Compute(uint32_t offset, uint32_t calcLength) {
    uint32_t alignedT = AlignElements<T>(calcLength);
    uint32_t alignedF = AlignElements<float>(calcLength);
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(calcLength * sizeof(T)), 0,
                                 0, 0};

    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    DataCopyPadExtParams<T> pad{alignedT > calcLength, 0,
                                static_cast<uint8_t>(alignedT - calcLength), 0};
    DataCopyPad(xLocal, xGm[offset], copyParams, pad);
    inQueueX.EnQue(xLocal);
    xLocal = inQueueX.DeQue<T>();

    LocalTensor<float> xFloat = calcQueueA.AllocTensor<float>();
    if constexpr (std::is_same_v<T, float>) {
      DataCopy(xFloat, xLocal, alignedF);
    } else {
      Cast(xFloat, xLocal, RoundMode::CAST_NONE, calcLength);
    }
    inQueueX.FreeTensor(xLocal);

    LocalTensor<float> w = calcQueueB.AllocTensor<float>();
    LocalTensor<float> yFloat = calcQueueC.AllocTensor<float>();

    // w = -log(1 - x*x). The polynomial regions below cover center and tails.
    Mul(w, xFloat, xFloat, calcLength);
    Muls(w, w, -1.0f, calcLength);
    Adds(w, w, 1.0f, calcLength);
    Ln(w, w, calcLength);
    Muls(w, w, -1.0f, calcLength);

    if constexpr (std::is_same_v<T, float>) {
      Muls(w, w, 0.1428571429f, calcLength);
      Adds(w, w, -1.0f, calcLength);
      Duplicate(yFloat, -0.01243596835f, calcLength);
      Horner(yFloat, w, -0.006070801957f, calcLength);
      Horner(yFloat, w, 0.06394989803f, calcLength);
      Horner(yFloat, w, -0.05800587708f, calcLength);
      Horner(yFloat, w, -0.01292312380f, calcLength);
      Horner(yFloat, w, 0.1002698567f, calcLength);
      Horner(yFloat, w, -0.2857103648f, calcLength);
      Horner(yFloat, w, 1.310153916f, calcLength);
      Horner(yFloat, w, 2.479775486f, calcLength);
      Mul(yFloat, yFloat, xFloat, calcLength);
    } else {
      Duplicate(yFloat, 0.00016694325f, calcLength);
      Horner(yFloat, w, -0.0029748488f, calcLength);
      Horner(yFloat, w, 0.012057142f, calcLength);
      Horner(yFloat, w, 0.23188046f, calcLength);
      Horner(yFloat, w, 0.88625031f, calcLength);
      Mul(yFloat, yFloat, xFloat, calcLength);
    }

    calcQueueA.FreeTensor(xFloat);
    calcQueueB.FreeTensor(w);

    LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
    if constexpr (std::is_same_v<T, float>) {
      DataCopy(yLocal, yFloat, alignedF);
    } else {
      Cast(yLocal, yFloat, RoundMode::CAST_RINT, calcLength);
    }
    calcQueueC.FreeTensor(yFloat);

    outQueueY.EnQue(yLocal);
    yLocal = outQueueY.DeQue<T>();
    DataCopyPad(yGm[offset], yLocal, copyParams);
    outQueueY.FreeTensor(yLocal);
  }
};

extern "C" __global__ __aicore__ void erfinv(GM_ADDR x, GM_ADDR y,
                                               GM_ADDR workspace,
                                               GM_ADDR tiling) {
  (void)workspace;
  GET_TILING_DATA(tilingData, tiling);
  TPipe pipe;
  ErfinvKernel<DTYPE_X> op;
  op.Init(x, y, &pipe, tilingData.coreSize, tilingData.lastCoreSize,
          tilingData.tileLength);
  op.Process();
}
