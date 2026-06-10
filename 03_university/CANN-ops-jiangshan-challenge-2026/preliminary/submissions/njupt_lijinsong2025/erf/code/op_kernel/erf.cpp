// Kernel侧核函数实现
#define K_MAX_SHAPE_DIM 0 // ← 必须在 include 之前
#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

namespace {
constexpr uint32_t INPUT_QUEUE_MAX_PADDING = 256;

// ver102: 2-scratch VECCALC layout: tLocal / pLocal.
//
// Bank-conflict 缓解：两段 scratch 之间插 64 个 float (= 256B = 8 banks)
// 错位，破坏 tile*4 与 bank-group 跨度的对齐。错位区永不访问，
// 只用来推开第二段起址。
constexpr uint32_t ERF_V102_BANK_STAGGER_ELEMS = 64; // 256 字节 = 8 banks
constexpr uint32_t ERF_V102_BANK_STAGGER_BYTES =
    ERF_V102_BANK_STAGGER_ELEMS * sizeof(float);

// 与 host UB 模型 ERF_CONFIG_FORCE_SCRATCH_TILES=2 保持一致：
// 2 段 tile + 1 段错位。
constexpr uint32_t ERF_V102_CALC_TILES = 2U;
constexpr uint32_t ERF_V102_CALC_STAGGERS = 1U;

constexpr float ERF_V102_DEN_SHIFT = 9.990964f;
constexpr float ERF_V102_DEN_BIAS  = 23.329983f;
constexpr float ERF_V102_NUM       = 20.884548f;

// ver102 使用 clamp(out_raw, -1, 1) 替代区间外 Compare+Select，
// 因此 base 略大于 ver100 minimax base，让 |x|≈2.327 附近进入饱和区。
constexpr float ERF_V102_BASE      = 0.23410137f;

__aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs) {
  return lhs < rhs ? lhs : rhs;
}
__aicore__ inline uint32_t AlignUp(uint32_t x, uint32_t align) {
  return align == 0 ? x : ((x + align - 1) / align) * align;
}
__aicore__ inline uint32_t GetCalcBufferBytes(uint32_t tileLength) {
  // 单块 VECCALC：t + stagger + p
  return ERF_V102_CALC_TILES * tileLength * sizeof(float) +
         ERF_V102_CALC_STAGGERS * ERF_V102_BANK_STAGGER_BYTES;
}


__aicore__ inline void
ErfFastVer102Compute(const AscendC::LocalTensor<float> &yLocal,
                     const AscendC::LocalTensor<float> &xLocal,
                     const AscendC::LocalTensor<float> &pLocal,
                     const AscendC::LocalTensor<float> &tLocal,
                     uint32_t count) {
  const uint32_t calcCount = AlignUp(count, 64);

  // t = z = x * x
  AscendC::Mul(tLocal, xLocal, xLocal, calcCount);

  // p = z + c
  AscendC::Adds(pLocal, tLocal, ERF_V102_DEN_SHIFT, calcCount);

  // 2-scratch denominator:
  //   t = z * (z + c)
  //   p = t + d = z * (z + c) + d
  //
  // 这里避免了 ver100/ver101 的 workLocal 和 Duplicate(DEN_BIAS)。
  // 注意：Mul 的 dst 与 src0 同址，这是 2-scratch 方案不可避免的
  // in-place 写法；如果 msprof 显示这里冲突变大，可尝试把下一行改成
  // AscendC::Mul(tLocal, pLocal, tLocal, calcCount);
  AscendC::Mul(tLocal, tLocal, pLocal, calcCount);
  AscendC::Adds(pLocal, tLocal, ERF_V102_DEN_BIAS, calcCount);

  // out_raw = base * x + num * (x / den)
  AscendC::Div(tLocal, xLocal, pLocal, calcCount);        // t = x / den
  AscendC::Muls(yLocal, xLocal, ERF_V102_BASE, calcCount);// y = base * x
  AscendC::Axpy(yLocal, tLocal, ERF_V102_NUM, calcCount); // y += num * t

  // 区间外直接返回 ±1：用 clamp 替代 CompareScalar + Select。
  AscendC::Maxs(yLocal, yLocal, -1.0f, calcCount);
  AscendC::Mins(yLocal, yLocal,  1.0f, calcCount);
}
} // namespace

template <class DT_X, int ERF_QUEUE2, int ERF_PIPELINE, int ERF_AXPY_H1>
class KernelErf {
public:
  static constexpr uint32_t QUEUE_NUM = ERF_QUEUE2 == 1 ? 2U : 1U;

  __aicore__ inline KernelErf() {}

  // ★ 改动 1：新增 TPipe* 形参；不再拷贝 tilingData 全部字段进类成员
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                              const ErfTilingData &tilingData,
                              AscendC::TPipe *pipeIn) {
    pipe = pipeIn;
    tileLength = tilingData.tileLength;
    const uint32_t totalLength = tilingData.totalLength;
    const uint32_t formerNum = tilingData.formerNum;
    const uint32_t formerLength = tilingData.formerLength;
    const uint32_t tailLength = tilingData.tailLength;
    const uint32_t inputQueuePadding =
        MinU32(static_cast<uint32_t>(tilingData.inputQueuePadding),
               INPUT_QUEUE_MAX_PADDING);

    const uint32_t blockIdx = AscendC::GetBlockIdx();
    const uint32_t numCores = AscendC::GetBlockNum();

    if (blockIdx >= numCores) {
      coreLength = 0;
      return;
    }

    // ★ 优化：根据整核和尾核策略计算偏移和长度
    uint32_t coreOffset = 0;
    uint32_t currentBlockLength = 0;

    if (blockIdx < formerNum) {
      // 当前核是整核
      currentBlockLength = formerLength;
      coreOffset = formerLength * blockIdx;
    } else {
      // 当前核是尾核
      currentBlockLength = tailLength;
      coreOffset =
          formerLength * formerNum + tailLength * (blockIdx - formerNum);
    }

    if (coreOffset >= totalLength) {
      coreLength = 0;
      return;
    }

    // 边界保护：确保不越过 totalLength
    coreLength = MinU32(currentBlockLength, totalLength - coreOffset);

    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + coreOffset,
                        coreLength);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + coreOffset,
                        coreLength);

    pipe->InitBuffer(inQueueX, QUEUE_NUM,
                     tileLength * sizeof(DT_X) + inputQueuePadding);
    pipe->InitBuffer(outQueueY, QUEUE_NUM, tileLength * sizeof(DT_X));

    // ★ 解决方案：在 outQueueY 和 calcBuf 之间插入一个 256 字节的 Padding Buffer
    // 这个大小（8个Bank）通常足以打破 Bank 冲突的对齐关系。
    // 我们可以为此定义一个新的 TBuf。
    pipe->InitBuffer(paddingBuf, 256);

    pipe->InitBuffer(calcBuf, GetCalcBufferBytes(tileLength));

    // ★ 优化：在 Init 阶段一次性打印所有 Buffer 的地址
    // AscendC::printf("[ERF_DBG] Core %u Init: coreLength=%u, tileLength=%u\n", blockIdx, coreLength, tileLength);

    // 打印 inQueueX 地址
    // AscendC::LocalTensor<DT_X> tempInTensor = inQueueX.template AllocTensor<DT_X>();
    // AscendC::printf("[ERF_DBG] Core %u Addr: inQueueX=0x%llx, size=%u\n", blockIdx,
    //                 (uint64_t)tempInTensor.GetPhyAddr(),
    //                 QUEUE_NUM * (tileLength * sizeof(DT_X) + inputQueuePadding));
    // inQueueX.FreeTensor(tempInTensor);

    // 打印 outQueueY 地址
    // AscendC::LocalTensor<DT_X> tempOutTensor = outQueueY.template AllocTensor<DT_X>();
    // AscendC::printf("[ERF_DBG] Core %u Addr: outQueueY=0x%llx, size=%u\n", blockIdx,
    //                 (uint64_t)tempOutTensor.GetPhyAddr(),
    //                 QUEUE_NUM * tileLength * sizeof(DT_X));
    // outQueueY.FreeTensor(tempOutTensor);

    // 打印 calcBuf 地址
    // AscendC::LocalTensor<float> calcTensor = calcBuf.Get<float>();
    // AscendC::printf("[ERF_DBG] Core %u Addr: calcBuf=0x%llx, size=%u\n", blockIdx,
    //                 (uint64_t)calcTensor.GetPhyAddr(),
    //                 GetCalcBufferBytes(tileLength));
  }

  __aicore__ inline void Process() {
    if (coreLength == 0) {
      return;
    }
    for (uint32_t offset = 0; offset < coreLength; offset += tileLength) {
      uint32_t curCount = MinU32(tileLength, coreLength - offset);
      // ★ 新增：打印 Process 流程日志
      // AscendC::printf("[ERF_DBG] Core %u Process: offset=%u, curCount=%u\n",
      //                 AscendC::GetBlockIdx(), offset, curCount);
      CopyIn(offset, curCount);
      Compute(curCount);
      CopyOut(offset, curCount);
    }
  }

  __aicore__ inline void ProcessPipeline() {
    if (coreLength == 0 || tileLength == 0) {
      return;
    }
    const uint32_t tileCount = AlignUp(coreLength, tileLength) / tileLength;
    // ★ 新增：打印 Pipeline 启动日志
    // AscendC::printf("[ERF_DBG] Core %u Pipeline Start: tileCount=%u\n",
    //                 AscendC::GetBlockIdx(), tileCount);
    CopyIn(0, MinU32(tileLength, coreLength));
    for (uint32_t tileIdx = 0; tileIdx < tileCount; ++tileIdx) {
      const uint32_t offset = tileIdx * tileLength;
      const uint32_t curCount = MinU32(tileLength, coreLength - offset);
      const uint32_t nextTileIdx = tileIdx + 1;
      // AscendC::printf("[ERF_DBG] Core %u Pipeline Loop: tileIdx=%u, offset=%u, "
      //                 "curCount=%u\n",
      //                 AscendC::GetBlockIdx(), tileIdx, offset, curCount);
      if (nextTileIdx < tileCount) {
        const uint32_t nextOffset = nextTileIdx * tileLength;
        const uint32_t nextCount = MinU32(tileLength, coreLength - nextOffset);
        // AscendC::printf("[ERF_DBG] Core %u Pipeline CopyInNext: nextOffset=%u, "
        //                 "nextCount=%u\n",
        //                 AscendC::GetBlockIdx(), nextOffset, nextCount);
        CopyIn(nextOffset, nextCount);
      }
      // AscendC::printf("[ERF_DBG] Core %u Pipeline Compute\n",
      //                 AscendC::GetBlockIdx());
      Compute(curCount);
      // AscendC::printf("[ERF_DBG] Core %u Pipeline CopyOut\n",
      //                 AscendC::GetBlockIdx());
      CopyOut(offset, curCount);
    }
  }

private:
  __aicore__ inline bool Is32BAligned(uint32_t count) const {
    return ((count * sizeof(DT_X)) % 32) == 0;
  }

  __aicore__ inline void CopyIn(uint32_t offset, uint32_t curCount) {
    AscendC::LocalTensor<DT_X> xLocal = inQueueX.template AllocTensor<DT_X>();
    if (Is32BAligned(curCount)) {
      AscendC::DataCopy(xLocal, xGm[offset], curCount);
    } else {
      AscendC::DataCopyExtParams copyParams = {
          1, static_cast<uint32_t>(curCount * sizeof(DT_X)), 0, 0, 0};
      AscendC::DataCopyPadExtParams<DT_X> padParams = {false, 0, 0,
                                                       static_cast<DT_X>(0)};
      AscendC::DataCopyPad<DT_X>(xLocal, xGm[offset], copyParams, padParams);
    }
    inQueueX.template EnQue<DT_X>(xLocal);
  }

  __aicore__ inline void Compute(uint32_t curCount) {
    
    AscendC::LocalTensor<DT_X> xLocal = inQueueX.template DeQue<DT_X>();
    AscendC::LocalTensor<DT_X> yLocal = outQueueY.template AllocTensor<DT_X>();
    AscendC::LocalTensor<float> calcAll = calcBuf.Get<float>();
    // calcBuf 布局：
    //   [0,                          tile)               -> tLocal
    //   [tile + STAG,                2*tile + STAG)      -> pLocal
    AscendC::LocalTensor<float> tLocal = calcAll;
    AscendC::LocalTensor<float> pLocal =
        calcAll[tileLength + ERF_V102_BANK_STAGGER_ELEMS];
    ErfFastVer102Compute(yLocal, xLocal, /*pLocal=*/pLocal, tLocal, curCount);
    outQueueY.template EnQue<DT_X>(yLocal);
    inQueueX.FreeTensor(xLocal);
  }

  __aicore__ inline void CopyOut(uint32_t offset, uint32_t curCount) {
    AscendC::LocalTensor<DT_X> yLocal = outQueueY.template DeQue<DT_X>();
    if (Is32BAligned(curCount)) {
      AscendC::DataCopy(yGm[offset], yLocal, curCount);
    } else {
      AscendC::DataCopyExtParams copyParams = {
          1, static_cast<uint32_t>(curCount * sizeof(DT_X)), 0, 0, 0};
      AscendC::DataCopyPad<DT_X>(yGm[offset], yLocal, copyParams);
    }
    outQueueY.FreeTensor(yLocal);
  }

  AscendC::TPipe *pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_NUM> inQueueX;
  AscendC::TQue<AscendC::TPosition::VECOUT, QUEUE_NUM> outQueueY;
  AscendC::TBuf<AscendC::TPosition::VECCALC> paddingBuf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
  AscendC::GlobalTensor<DT_X> xGm;
  AscendC::GlobalTensor<DT_X> yGm;
  uint32_t tileLength = 0;
  uint32_t coreLength = 0;
};

template <typename DT_X, int ERF_QUEUE2, int ERF_PIPELINE, int ERF_AXPY_H1>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                               GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  (void)workspace;
  REGISTER_TILING_DEFAULT(ErfTilingData);
  GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);

  // ★ 改动 4：TPipe 在 kernel 入口创建
  AscendC::TPipe pipe;
  KernelErf<DT_X, ERF_QUEUE2, ERF_PIPELINE, ERF_AXPY_H1> op;
  op.Init(x, y, tiling_data, &pipe);

  // ★ 改动 5：pipeImplMode → pipelineMode（字段已重命名）
  if (ERF_PIPELINE == 1 && tiling_data.pipelineMode == 1) {
    op.ProcessPipeline();
  } else {
    op.Process();
  }
}
