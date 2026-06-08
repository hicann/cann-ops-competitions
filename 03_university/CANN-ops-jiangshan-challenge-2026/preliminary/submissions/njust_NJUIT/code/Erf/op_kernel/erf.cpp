 #include "kernel_operator.h"

  #include "erf_tiling.h"
  #include "tiling_key_erf.h"

  using namespace AscendC;

  constexpr int32_t BUFFER_NUM = 2;
  constexpr uint32_t ALIGN_NUM = 64;
  constexpr uint32_t DATA_COPY_ALIGN_NUM = 8;
  constexpr uint32_t MAX_TILE_LENGTH = 4096;
  constexpr float ERF_P = 0.3275911f;
  constexpr float ERF_A1 = 0.254829592f;
  constexpr float ERF_A2 = -0.284496736f;
  constexpr float ERF_A3 = 1.421413741f;
  constexpr float ERF_A4 = -1.453152027f;
  constexpr float ERF_A5 = 1.061405429f;
  constexpr float SIGN_EPS = 1.0e-12f;

  template <class DT_X>
  class KernelErf {
  public:
      __aicore__ inline KernelErf() {}

      __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, ErfTilingData *tilingData)
      {
          uint32_t coreIdx = GetBlockIdx();
          uint32_t blockNum = GetBlockNum();

          this->tileLength = tilingData->tileLength;
          if (blockNum == 0 ||
              tilingData->usedCoreNum == 0 ||
              blockNum != tilingData->usedCoreNum ||
              tilingData->largeCoreCount > tilingData->usedCoreNum) {
              this->valid = 0;
              return;
          }

          InitCoreRange(coreIdx, tilingData);

          if (this->coreDataLength == 0 ||
              this->tileLength < ALIGN_NUM ||
              this->tileLength > MAX_TILE_LENGTH ||
              this->tileLength % ALIGN_NUM != 0) {
              this->valid = 0;
              return;
          }

          RecomputeTileLayout();

          if (this->tileNum == 0 || this->tileTailLength > this->tileLength) {
              this->valid = 0;
              return;
          }

          if (static_cast<uint64_t>(this->coreOffset) + this->coreDataLength > tilingData->totalLength) {
              this->valid = 0;
              return;
          }

          xGm.SetGlobalBuffer((__gm__ float *)x + this->coreOffset, this->coreDataLength);
          yGm.SetGlobalBuffer((__gm__ float *)y + this->coreOffset, this->coreDataLength);

          pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
          pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));

          pipe.InitBuffer(bufAbs, this->tileLength * sizeof(float));
          pipe.InitBuffer(bufX2, this->tileLength * sizeof(float));
          pipe.InitBuffer(bufT, this->tileLength * sizeof(float));
          pipe.InitBuffer(bufPoly, this->tileLength * sizeof(float));

          this->valid = 1;
      }

      __aicore__ inline void Process()
      {
          if (this->valid == 0) {
              return;
          }

          for (uint32_t i = 0; i < this->tileNum; ++i) {
              uint32_t currentLength = (i == this->tileNum - 1) ? this->tileTailLength : this->tileLength;
              CopyIn(i, currentLength);
              Compute(currentLength);
              CopyOut(i, currentLength);
          }
      }

  private:
      __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align)
      {
          return ((value + align - 1) / align) * align;
      }

      __aicore__ inline bool IsAlignedLength(uint32_t value, uint32_t align)
      {
          return value % align == 0;
      }

      __aicore__ inline void RecomputeTileLayout()
      {
          if (this->coreDataLength == 0) {
              this->tileNum = 0;
              this->tileTailLength = 0;
              return;
          }

          this->tileNum = (this->coreDataLength + this->tileLength - 1) / this->tileLength;
          this->tileTailLength = this->coreDataLength - (this->tileNum - 1) * this->tileLength;
      }

      __aicore__ inline void InitCoreRange(uint32_t coreIdx, const ErfTilingData *tilingData)
      {
          if (coreIdx < tilingData->largeCoreCount) {
              this->coreOffset = coreIdx * tilingData->largeCoreLength;
              this->coreDataLength = tilingData->largeCoreLength;
          } else {
              this->coreOffset = tilingData->largeCoreCount * tilingData->largeCoreLength +
                                 (coreIdx - tilingData->largeCoreCount) * tilingData->smallCoreLength;
              this->coreDataLength = tilingData->smallCoreLength;
          }

          if (coreIdx == tilingData->usedCoreNum - 1) {
              this->coreDataLength += tilingData->tailLength;
          }
      }

      __aicore__ inline void CopyIn(uint32_t progress, uint32_t length)
      {
          LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();

          if (length == this->tileLength || IsAlignedLength(length, ALIGN_NUM)) {
              DataCopy(xLocal, xGm[progress * this->tileLength], length);
          } else if (IsAlignedLength(length, DATA_COPY_ALIGN_NUM)) {
              Duplicate(xLocal, 0.0f, this->tileLength);
              PipeBarrier<PIPE_ALL>();
              DataCopy(xLocal, xGm[progress * this->tileLength], length);
          } else {
              Duplicate(xLocal, 0.0f, this->tileLength);
              PipeBarrier<PIPE_ALL>();

              uint32_t copyBytes = length * static_cast<uint32_t>(sizeof(float));
              DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};

              uint32_t copyAlignLength = AlignUp(length, DATA_COPY_ALIGN_NUM);
              uint8_t rightPadding = static_cast<uint8_t>(copyAlignLength - length);
              DataCopyPadExtParams<float> padParams{true, 0, rightPadding, 0.0f};

              DataCopyPad(xLocal, xGm[progress * this->tileLength], copyParams, padParams);
          }
          inQueueX.EnQue(xLocal);
      }

      __aicore__ inline void Compute(uint32_t length)
      {
          LocalTensor<float> xLocal = inQueueX.DeQue<float>();
          LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

          uint32_t alignedLength = AlignUp(length, ALIGN_NUM);

          LocalTensor<float> xAbs = bufAbs.Get<float>();
          LocalTensor<float> x2 = bufX2.Get<float>();
          LocalTensor<float> t = bufT.Get<float>();
          LocalTensor<float> poly = bufPoly.Get<float>();
          Abs(xAbs, xLocal, alignedLength);
          Muls(t, xAbs, ERF_P, alignedLength);
          Adds(t, t, 1.0f, alignedLength);
          Duplicate(x2, 1.0f, alignedLength);
          Div(t, x2, t, alignedLength);

          Muls(poly, t, ERF_A5, alignedLength);
          Adds(poly, poly, ERF_A4, alignedLength);
          Mul(poly, poly, t, alignedLength);
          Adds(poly, poly, ERF_A3, alignedLength);
          Mul(poly, poly, t, alignedLength);
          Adds(poly, poly, ERF_A2, alignedLength);
          Mul(poly, poly, t, alignedLength);
          Adds(poly, poly, ERF_A1, alignedLength);
          Mul(poly, poly, t, alignedLength);

          Mul(x2, xLocal, xLocal, alignedLength);
          Muls(x2, x2, -1.0f, alignedLength);
          Exp(x2, x2, alignedLength);

          Mul(poly, poly, x2, alignedLength);
          Muls(poly, poly, -1.0f, alignedLength);
          Adds(poly, poly, 1.0f, alignedLength);

          Adds(x2, xAbs, SIGN_EPS, alignedLength);
          Div(x2, xLocal, x2, alignedLength);
          Mul(yLocal, poly, x2, alignedLength);

          outQueueY.EnQue<float>(yLocal);
          inQueueX.FreeTensor(xLocal);
      }

      __aicore__ inline void CopyOut(uint32_t progress, uint32_t length)
      {
          LocalTensor<float> yLocal = outQueueY.DeQue<float>();

          if (length == this->tileLength || IsAlignedLength(length, DATA_COPY_ALIGN_NUM)) {
              DataCopy(yGm[progress * this->tileLength], yLocal, length);
          } else {
              uint32_t copyBytes = length * static_cast<uint32_t>(sizeof(float));
              DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
              DataCopyPad(yGm[progress * this->tileLength], yLocal, copyParams);
          }
          outQueueY.FreeTensor(yLocal);
      }

  private:
      TPipe pipe;

      TQue<QuePosition::VECIN, 2> inQueueX;
      TQue<QuePosition::VECOUT, 2> outQueueY;

      TBuf<TPosition::VECCALC> bufAbs;
      TBuf<TPosition::VECCALC> bufX2;
      TBuf<TPosition::VECCALC> bufT;
      TBuf<TPosition::VECCALC> bufPoly;
      GlobalTensor<float> xGm;
      GlobalTensor<float> yGm;

      uint32_t valid = 0;
      uint32_t tileNum = 0;
      uint32_t tileLength = 0;
      uint32_t tileTailLength = 0;
      uint32_t coreOffset = 0;
      uint32_t coreDataLength = 0;
  };

  template <typename DT_X>
  __global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
  {
      REGISTER_TILING_DEFAULT(ErfTilingData);
      GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tilingData, tiling);

      KernelErf<DT_X> op;
      op.Init(x, y, &tilingData);
      op.Process();
  }
