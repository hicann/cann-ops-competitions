#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include <type_traits>
#include <cstdint>

using namespace AscendC;

// Kernel 数据模型
// ---------------
// Host 将任意 rank<=4 的输入展开为紧凑的 [outer, axis, inner]：
//   input[o, a, i] -> output[a][o, i]
// input 偏移为 (o * axis_size + a) * inner_size + i，单个输出偏移为 o * inner_size + i。
// output 是 ListTensor 描述符而非普通 GM 首地址，访问前必须绑定其中第 a 个真实输出地址。
//
// 本算子只重排位模式，不进行数值计算。入口按 sizeof(DTYPE_INPUT) 将同宽类型统一映射为
// uint8/uint16/uint32，从而让浮点、有符号整数和 bool 共用搬运代码。
//
// TILING_KEY：1=首轴单输出分核，2=首轴大输出多核，7=中轴 line stream，
// 5=末轴 TransData 转置，8=末轴 Gather 转置。
namespace matmul {
// 构建模板要求的兼容符号；Unpack 不使用 workspace。
__aicore__ inline void clearWorkspace(GM_ADDR) {}
}  // namespace matmul

// 事件 helper：A->B 表示 B 流等待 A 流此前提交的任务完成。
// MTE2=GM->UB，V=向量计算/转置，MTE3=UB->GM。
__aicore__ inline void WaitMte3ToMte2(const int32_t evtId) {
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
}

__aicore__ inline void SignalMte3ToMte2(const int32_t evtId) {
  AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
}

__aicore__ inline void SyncMte2ToMte3(const int32_t evtId) {
  // 输入 DMA 完成后，才允许输出 DMA 消费同一 UB。
  AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);
}

__aicore__ inline void SyncMte2ToV(const int32_t evtId) {
  // Gather/TransData 必须等待源 tile 完整进入 UB。
  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtId);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtId);
}

__aicore__ inline void SyncMte3ToV(const int32_t evtId) {
  // V 流覆写 dst stage 前，等待上一轮 MTE3 写回结束。
  AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(evtId);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(evtId);
}

__aicore__ inline void SyncVToMte3(const int32_t evtId) {
  // 转置或 Gather 写完 dst UB 后，才允许 MTE3 读取结果。
  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evtId);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evtId);
}

__aicore__ inline void InitDoubleBufferEvents(int32_t &evtPing, int32_t &evtPong, const int32_t pingId,
                                              const int32_t pongId) {
  // 初始两份缓冲均空闲，预置信号使第一次 Wait 可直接通过。
  evtPing = pingId;
  evtPong = pongId;
  SignalMte3ToMte2(evtPing);
  SignalMte3ToMte2(evtPong);
}

__aicore__ inline void FinishDoubleBufferEvents(const int32_t evtPing, const int32_t evtPong) {
  // 返回前等待 ping/pong 的最后一批 GM 写回完成。
  WaitMte3ToMte2(evtPing);
  WaitMte3ToMte2(evtPong);
}

template <typename T>
// 非对齐一维 GM->UB；elemCnt/offset 以元素计，blockLen 以字节计。
__aicore__ inline void CopyGmToUbPad(const LocalTensor<T> &dstUb, const uint32_t dstUbOffset,
                                     const GlobalTensor<T> &srcGm, const uint32_t srcGmOffset,
                                     const uint32_t elemCnt) {
  AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(elemCnt * sizeof(T)), 0, 0, 0};
  AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
  AscendC::DataCopyPad<T>(dstUb[dstUbOffset], srcGm[srcGmOffset], copyParams, padParams);
}

template <typename T>
// 非对齐一维 UB->GM；写出侧不需要 DataCopyPadExtParams。
__aicore__ inline void CopyUbToGmPad(const GlobalTensor<T> &dstGm, const uint32_t dstGmOffset,
                                     const LocalTensor<T> &srcUb, const uint32_t srcUbOffset,
                                     const uint32_t elemCnt) {
  AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(elemCnt * sizeof(T)), 0, 0, 0};
  AscendC::DataCopyPad<T>(dstGm[dstGmOffset], srcUb[srcUbOffset], copyParams);
}

template <typename T>
// 32B 对齐的连续搬运快路径，DataCopy 长度以元素计。
__aicore__ inline void CopyUbToGmContinuous(const GlobalTensor<T> &dstGm, const uint32_t dstGmOffset,
                                            const LocalTensor<T> &srcUb, const uint32_t srcUbOffset,
                                            const uint32_t elemCnt) {
  AscendC::DataCopy(dstGm[dstGmOffset], srcUb[srcUbOffset], elemCnt);
}

template <typename T>
__aicore__ inline void CopyGmToUbContinuous(const LocalTensor<T> &dstUb, const uint32_t dstUbOffset,
                                            const GlobalTensor<T> &srcGm, const uint32_t srcGmOffset,
                                            const uint32_t elemCnt) {
  AscendC::DataCopy(dstUb[dstUbOffset], srcGm[srcGmOffset], elemCnt);
}

template <typename T>
// 二维 GM->UB：blockLen/srcStride 单位为字节，dstStride 单位为 32B datablock。
// DataCopyPad 会将非对齐 block 在 UB 中补齐至 32B。
__aicore__ inline void CopyGmToUbPad2D(const LocalTensor<T> &dstUb, const uint32_t dstUbOffsetElem,
                                       const GlobalTensor<T> &srcGm, const uint32_t srcGmOffsetElem,
                                       const uint16_t blockCount, const uint32_t blockLenBytes,
                                       const uint32_t srcStrideBytes, const uint32_t dstStrideDataBlock) {
  AscendC::DataCopyExtParams params{blockCount, blockLenBytes, srcStrideBytes, dstStrideDataBlock, 0};
  AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
  AscendC::DataCopyPad<T>(dstUb[dstUbOffsetElem], srcGm[srcGmOffsetElem], params, padParams);
}

template <typename T>
// 二维 UB->GM：srcStride 单位为 32B datablock，dstStride 单位为字节。
__aicore__ inline void CopyUbToGmPad2D(GlobalTensor<T> &dstGm, const uint32_t dstGmOffsetElem,
                                       const LocalTensor<T> &srcUb, const uint32_t srcUbOffsetElem,
                                       const uint16_t blockCount, const uint32_t blockLenBytes,
                                       const uint32_t srcStrideDataBlock, const uint32_t dstStrideBytes) {
  AscendC::DataCopyExtParams params{blockCount, blockLenBytes, srcStrideDataBlock, dstStrideBytes, 0};
  AscendC::DataCopyPad<T>(dstGm[dstGmOffsetElem], srcUb[srcUbOffsetElem], params);
}



template <typename T>
// 从动态输出 ListTensor 中绑定第 out_idx 个真实 GM 地址。
__aicore__ inline void BindOutputTensor(GlobalTensor<T> &outGm, GM_ADDR outputList, const uint32_t out_idx) {
  AscendC::ListTensorDesc outList((__gm__ void *)outputList);
  outGm.SetGlobalBuffer(outList.GetDataPtr<T>(out_idx));
}

template <typename T>
// 缓存 ListTensorDesc 的重载，供热点循环复用。
__aicore__ inline void BindOutputTensor(GlobalTensor<T> &outGm, AscendC::ListTensorDesc &outList,
                                        const uint32_t out_idx) {
  outGm.SetGlobalBuffer(outList.GetDataPtr<T>(out_idx));
}

template <typename T>
// 首轴 mode0：输入已按 [output_num, inner] 连续排列，一个核负责一个或多个完整输出。
// 核内按 96KB 切 tile，并用 ping/pong UB 覆盖当前 MTE2 与上一 tile 的 MTE3。
class KernelUnpackAxis0Mode0 {
  public:
  static constexpr uint32_t GATHER_COL_CHUNK = 4U;
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    total_ = tilingData.total_elem;
    output_num_ = tilingData.output_num;
    inner_size_ = tilingData.inner_size;
    inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input), total_);
    //inputGm_.template SetL2CacheHint<CacheRwMode::RW>(AscendC::CacheMode::CACHE_MODE_DISABLE);
    outputList_ = output;
    outList_ = AscendC::ListTensorDesc((__gm__ void *)outputList_);
  }

  __aicore__ inline void Process() {
    const uint32_t one_out_elem = inner_size_;
    const uint32_t ub_cap = UB_BYTES / static_cast<uint32_t>(sizeof(T));
    LocalTensor<T> ub0(AscendC::TPosition::VECCALC, STATIC_BUF0_ADDR, ub_cap);
    LocalTensor<T> ub1(AscendC::TPosition::VECCALC, STATIC_BUF1_ADDR, ub_cap);
    int32_t evtPing = TRANS_EVT_PING;
    int32_t evtPong = TRANS_EVT_PONG;
    SignalMte3ToMte2(evtPing);
    SignalMte3ToMte2(evtPong);

    const uint32_t core_idx = AscendC::GetBlockIdx();
    const uint32_t block_dim = AscendC::GetBlockNum();
    // 连续均分输出，前 rem 个核多领取一个，保证覆盖完整且无写地址重叠。
    uint32_t out_begin = 0U;
    uint32_t out_end = output_num_;
    const uint32_t base = output_num_ / block_dim;
    const uint32_t rem = output_num_ - base * block_dim;
    const uint32_t extra = (core_idx < rem) ? 1U : 0U;
    out_begin = core_idx * base + ((core_idx < rem) ? core_idx : rem);
    out_end = out_begin + base + extra;
    for (uint32_t out_idx = out_begin; out_idx < out_end; ++out_idx) {
      GlobalTensor<T> outGm;
      ::BindOutputTensor(outGm, outList_, out_idx);
      const uint32_t src_base = out_idx * one_out_elem;
      const uint32_t span = one_out_elem;
      // 完整 tile 走连续 DataCopy；尾块按字节对齐情况选择 DataCopy 或 DataCopyPad。
      const uint32_t full_tiles = span / ub_cap;
      const uint32_t tail = span - full_tiles * ub_cap;
      uint32_t done = 0U;
      uint32_t tileCounter = 0U;
      for (uint32_t t = 0U; t < full_tiles; ++t) {
        const uint32_t buf_idx = tileCounter & 1U;
        const int32_t evtId = (buf_idx == 0U) ? evtPing : evtPong;
        LocalTensor<T> &ub = (buf_idx == 0U) ? ub0 : ub1;
        // 只等待即将复用的缓冲，另一缓冲的写回仍可并行推进。
        WaitMte3ToMte2(evtId);
        ::CopyGmToUbContinuous(ub, 0U, inputGm_, src_base + done, ub_cap);
        ::SyncMte2ToMte3(evtId);
        ::CopyUbToGmContinuous(outGm, done, ub, 0U, ub_cap);
        SignalMte3ToMte2(evtId);
        done += ub_cap;
        ++tileCounter;
      }
      if (tail != 0U) {
        const uint32_t buf_idx = tileCounter & 1U;
        const int32_t evtId = (buf_idx == 0U) ? evtPing : evtPong;
        LocalTensor<T> &ub = (buf_idx == 0U) ? ub0 : ub1;
        WaitMte3ToMte2(evtId);
        if (((tail * static_cast<uint32_t>(sizeof(T))) & 31U) == 0U) {
          ::CopyGmToUbContinuous(ub, 0U, inputGm_, src_base + done, tail);
          ::SyncMte2ToMte3(evtId);
          ::CopyUbToGmContinuous(outGm, done, ub, 0U, tail);
        } else {
          ::CopyGmToUbPad(ub, 0U, inputGm_, src_base + done, tail);
          ::SyncMte2ToMte3(evtId);
          ::CopyUbToGmPad(outGm, done, ub, 0U, tail);
        }
        SignalMte3ToMte2(evtId);
      }
    }

    WaitMte3ToMte2(evtPing);
    WaitMte3ToMte2(evtPong);
  }

 private:
  static constexpr uint32_t UB_BYTES = 96U * 1024U;
  static constexpr uint32_t STATIC_BUF0_ADDR = 0U;
  static constexpr uint32_t STATIC_BUF1_ADDR = UB_BYTES;
  static constexpr int32_t TRANS_EVT_PING = 0;
  static constexpr int32_t TRANS_EVT_PONG = 1;

  GlobalTensor<T> inputGm_;
  GM_ADDR outputList_ = nullptr;
  AscendC::ListTensorDesc outList_;
  uint32_t total_ = 0;
  uint32_t output_num_ = 0;
  uint32_t inner_size_ = 1;
};

template <typename T>
// 首轴 mode1：输出数少但单输出很大时，多个核协作处理一个输出。
// core_idx 映射为 [tensor_idx, lane_idx]，lane 负责连续 inner 分片。
class KernelUnpackAxis0Mode1 {
 public:
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    total_ = tilingData.total_elem;
    output_num_ = tilingData.output_num;
    inner_size_ = tilingData.inner_size;
    axis0_inner_tile_elems_ = tilingData.axis0_inner_tile_elems;
    axis0_cores_per_tensor_ = tilingData.axis0_cores_per_tensor;
    axis0_tensor_group_count_ = tilingData.axis0_tensor_group_count;
    inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input), total_);
    //inputGm_.template SetL2CacheHint<CacheRwMode::RW>(AscendC::CacheMode::CACHE_MODE_DISABLE);
    outputList_ = output;
    outList_ = AscendC::ListTensorDesc((__gm__ void *)outputList_);
  }

  __aicore__ inline void Process() {
    const uint32_t one_out_elem = inner_size_;
    const uint32_t ub_cap = UB_BYTES / static_cast<uint32_t>(sizeof(T));
    LocalTensor<T> ub0(AscendC::TPosition::VECCALC, STATIC_BUF0_ADDR, ub_cap);
    LocalTensor<T> ub1(AscendC::TPosition::VECCALC, STATIC_BUF1_ADDR, ub_cap);
    int32_t evtPing = TRANS_EVT_PING;
    int32_t evtPong = TRANS_EVT_PONG;
    SignalMte3ToMte2(evtPing);
    SignalMte3ToMte2(evtPong);
    const uint32_t core_idx = AscendC::GetBlockIdx();
    // tensor_idx 选择并行输出组，lane_idx 选择组内的 inner 分片。
    const uint32_t tensor_idx = core_idx / axis0_cores_per_tensor_;
    const uint32_t lane_idx = core_idx - tensor_idx * axis0_cores_per_tensor_;
    if (tensor_idx >= axis0_tensor_group_count_) {
      WaitMte3ToMte2(evtPing);
      WaitMte3ToMte2(evtPong);
      return;
    }
    // 最后一 lane 接收整除余数，确保 [0, inner_size) 无空洞。
    const uint32_t inner_begin = lane_idx * axis0_inner_tile_elems_;
    uint32_t inner_end = (lane_idx + 1U == axis0_cores_per_tensor_) ? one_out_elem : (inner_begin + axis0_inner_tile_elems_);
    inner_end = Std::min(inner_end, one_out_elem);
    // 同一物理核组处理后续输出时按 tensor_group_count 递增。
    for (uint32_t out_idx = tensor_idx; out_idx < output_num_; out_idx += axis0_tensor_group_count_) {
      GlobalTensor<T> outGm;
      ::BindOutputTensor(outGm, outList_, out_idx);
      const uint32_t src_base = out_idx * one_out_elem;
      const uint32_t span = inner_end - inner_begin;
      const uint32_t full_tiles = span / ub_cap;
      const uint32_t tail = span - full_tiles * ub_cap;
      uint32_t done = inner_begin;
      uint32_t tileCounter = 0U;
      for (uint32_t t = 0U; t < full_tiles; ++t) {
        const uint32_t buf_idx = tileCounter & 1U;
        const int32_t evtId = (buf_idx == 0U) ? evtPing : evtPong;
        LocalTensor<T> &ub = (buf_idx == 0U) ? ub0 : ub1;
        WaitMte3ToMte2(evtId);
        ::CopyGmToUbContinuous(ub, 0U, inputGm_, src_base + done, ub_cap);
        ::SyncMte2ToMte3(evtId);
        ::CopyUbToGmContinuous(outGm, done, ub, 0U, ub_cap);
        SignalMte3ToMte2(evtId);
        done += ub_cap;
        ++tileCounter;
      }
      if (tail != 0U) {
        const uint32_t buf_idx = tileCounter & 1U;
        const int32_t evtId = (buf_idx == 0U) ? evtPing : evtPong;
        LocalTensor<T> &ub = (buf_idx == 0U) ? ub0 : ub1;
        WaitMte3ToMte2(evtId);
        if (((tail * static_cast<uint32_t>(sizeof(T))) & 31U) == 0U) {
          ::CopyGmToUbContinuous(ub, 0U, inputGm_, src_base + done, tail);
          ::SyncMte2ToMte3(evtId);
          ::CopyUbToGmContinuous(outGm, done, ub, 0U, tail);
        } else {
          ::CopyGmToUbPad(ub, 0U, inputGm_, src_base + done, tail);
          ::SyncMte2ToMte3(evtId);
          ::CopyUbToGmPad(outGm, done, ub, 0U, tail);
        }
        SignalMte3ToMte2(evtId);
      }
    }
    WaitMte3ToMte2(evtPing);
    WaitMte3ToMte2(evtPong);
  }

 private:
  static constexpr uint32_t UB_BYTES = 96U * 1024U;
  static constexpr uint32_t STATIC_BUF0_ADDR = 0U;
  static constexpr uint32_t STATIC_BUF1_ADDR = UB_BYTES;
  static constexpr int32_t TRANS_EVT_PING = 0;
  static constexpr int32_t TRANS_EVT_PONG = 1;

  GlobalTensor<T> inputGm_;
  GM_ADDR outputList_ = nullptr;
  AscendC::ListTensorDesc outList_;
  uint32_t total_ = 0;
  uint32_t output_num_ = 0;
  uint32_t inner_size_ = 1;
  uint32_t axis0_inner_tile_elems_ = 0;
  uint32_t axis0_cores_per_tensor_ = 1;
  uint32_t axis0_tensor_group_count_ = 1;
};

template <typename T>
// 中轴 line-stream 路径
// 将 [outer, axis] 展平成 total_lines，每条 line 含 inner 个连续元素。读入连续 line 后，
// 通过二维 UB->GM 将间隔 axis_size 条 line 的同一 axis_idx 数据写入对应输出。
// 对齐行在 UB 中紧凑存放；非对齐行使用 Align32(row_bytes) 的物理 pitch。
class KernelUnpackMidAxisStream {
 public:
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    total_ = tilingData.total_elem;
    output_num_ = tilingData.output_num;
    outer_size_ = tilingData.outer_size;
    axis_size_ = tilingData.axis_size;
    inner_size_ = tilingData.inner_size;
    tile_lines_ = tilingData.tile_o;
    row_bytes_ = tilingData.row_bytes;
    row_aligned_ = (tilingData.row_aligned != 0U);
    ub_row_pitch_bytes_ = tilingData.ub_row_pitch_bytes;
    inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input), total_);
    //inputGm_.template SetL2CacheHint<CacheRwMode::RW>(AscendC::CacheMode::CACHE_MODE_DISABLE);
    outputList_ = output;
    outList_ = AscendC::ListTensorDesc((__gm__ void *)outputList_);
    ub_row_pitch_elem_ = ub_row_pitch_bytes_ / static_cast<uint32_t>(sizeof(T));
    total_lines_ = outer_size_ * axis_size_;
    // 下列 stride 以 32B datablock 为单位，与二维 DataCopy 接口匹配。
    aligned_row_block_len_db_ = row_bytes_ / 32U;
    aligned_src_stride_db_ = ((axis_size_ - 1U) * row_bytes_) / 32U;
    unaligned_copyin_dst_stride_db_ = (ub_row_pitch_bytes_ - row_bytes_) / 32U;
    unaligned_src_stride_db_ = ((axis_size_ - 1U) * ub_row_pitch_bytes_) / 32U;
  }

  __aicore__ inline void Process() {
    // 固定地址构造两份 96KB LocalTensor，避免队列元数据开销。
    const uint32_t ub_cap = MID_STREAM_BUF_BYTES / static_cast<uint32_t>(sizeof(T));
    LocalTensor<T> ub0(AscendC::TPosition::VECCALC, 0U, ub_cap);
    LocalTensor<T> ub1(AscendC::TPosition::VECCALC, MID_STREAM_BUF_BYTES, ub_cap);
    constexpr int32_t evt0 = 0;
    constexpr int32_t evt1 = 1;
    SignalMte3ToMte2(evt0);
    SignalMte3ToMte2(evt1);
    const uint32_t core_idx = AscendC::GetBlockIdx();
    const uint32_t block_dim = AscendC::GetBlockNum();
    const uint32_t line_tile_count = (total_lines_ + tile_lines_ - 1U) / tile_lines_;
    if (row_aligned_) {
      ProcessAligned(ub0, ub1, evt0, evt1, core_idx, block_dim, line_tile_count);
    } else {
      ProcessUnaligned(ub0, ub1, evt0, evt1, core_idx, block_dim, line_tile_count);
    }
    WaitMte3ToMte2(evt0);
    WaitMte3ToMte2(evt1);
  }

 private:
  __aicore__ inline void ProcessAligned(LocalTensor<T> &ub0, LocalTensor<T> &ub1,
                                        const int32_t evt0, const int32_t evt1,
                                        const uint32_t core_idx, const uint32_t block_dim,
                                        const uint32_t line_tile_count) {
    LocalTensor<T> *ubBufs[2] = {&ub0, &ub1};
    const int32_t evtIds[2] = {evt0, evt1};
    uint32_t buf_idx = 0U;
    // row tile 按 core_idx + n*block_dim 轮转分配，核间工作量最多相差一个 tile。
    for (uint32_t tile_idx = core_idx; tile_idx < line_tile_count; tile_idx += block_dim) {
      const uint32_t line_base = tile_idx * tile_lines_;
      uint32_t valid_lines = total_lines_ - line_base;
      valid_lines = Std::min(valid_lines, tile_lines_);
      const int32_t evtId = evtIds[buf_idx];
      LocalTensor<T> &ub = *ubBufs[buf_idx];
      WaitMte3ToMte2(evtId);
      CopyInLinesAligned(ub, line_base, valid_lines);
      // 无需 V/S 重排，MTE2 完成后 MTE3 可直接按 stride 写出。
      SyncMte2ToMte3(evtId);
      CopyOutLinesAligned(ub, line_base, valid_lines);
      SignalMte3ToMte2(evtId);
      buf_idx ^= 1U;
    }
  }

  __aicore__ inline void ProcessUnaligned(LocalTensor<T> &ub0, LocalTensor<T> &ub1,
                                          const int32_t evt0, const int32_t evt1,
                                          const uint32_t core_idx, const uint32_t block_dim,
                                          const uint32_t line_tile_count) {
    LocalTensor<T> *ubBufs[2] = {&ub0, &ub1};
    const int32_t evtIds[2] = {evt0, evt1};
    uint32_t buf_idx = 0U;
    for (uint32_t tile_idx = core_idx; tile_idx < line_tile_count; tile_idx += block_dim) {
      const uint32_t line_base = tile_idx * tile_lines_;
      uint32_t valid_lines = total_lines_ - line_base;
      valid_lines = Std::min(valid_lines, tile_lines_);
      const int32_t evtId = evtIds[buf_idx];
      LocalTensor<T> &ub = *ubBufs[buf_idx];
      WaitMte3ToMte2(evtId);
      // GM 行紧邻，进入 UB 后每行补齐为 ub_row_pitch_bytes_。
      CopyInLinesUnaligned(ub, line_base, valid_lines);
      SyncMte2ToMte3(evtId);
      CopyOutLinesUnaligned(ub, line_base, valid_lines);
      SignalMte3ToMte2(evtId);
      buf_idx ^= 1U;
    }
  }

  __aicore__ inline void CopyInLinesAligned(LocalTensor<T> &ub, const uint32_t line_base,
                                            const uint32_t valid_lines) {
    ::CopyGmToUbContinuous(ub, 0U, inputGm_, line_base * inner_size_, valid_lines * inner_size_);
  }

  __aicore__ inline void CopyInLinesUnaligned(LocalTensor<T> &ub, const uint32_t line_base,
                                              const uint32_t valid_lines) {
    const uint32_t gm_src = line_base * inner_size_;
    ::CopyGmToUbPad2D(ub, 0U, inputGm_, gm_src, static_cast<uint16_t>(valid_lines), row_bytes_, 0U,
                      unaligned_copyin_dst_stride_db_);
  }

  __aicore__ inline void CopyOutLinesAligned(LocalTensor<T> &ub, const uint32_t line_base,
                                             const uint32_t valid_lines) {
    // tile 可从任意 axis_idx 开始并跨越多个 axis 周期，需计算各输出在 tile 中出现次数。
    const uint32_t group_count = (valid_lines < axis_size_) ? valid_lines : axis_size_;
    uint32_t out_idx = line_base;
    uint32_t outer_begin = 0U;
    if (out_idx >= axis_size_) {
      outer_begin = out_idx / axis_size_;
      out_idx -= outer_begin * axis_size_;
    }
    // 前 extra_groups 个输出比其余输出多出现一次。
    const uint32_t base_line_count = valid_lines / axis_size_;
    const uint32_t extra_groups = valid_lines - base_line_count * axis_size_;
    for (uint32_t i = 0U; i < group_count; ++i) {
      const uint32_t line_count = base_line_count + ((i < extra_groups) ? 1U : 0U);
      CopyOutLineGroupAligned(ub[i * inner_size_], out_idx, outer_begin, line_count);
      const uint32_t next_out_idx = out_idx + 1U;
      outer_begin += next_out_idx / axis_size_;
      out_idx = next_out_idx % axis_size_;
    }
  }

  __aicore__ inline void CopyOutLineGroupAligned(LocalTensor<T> ub, const uint32_t out_idx,
                                                 const uint32_t outer_begin, const uint32_t line_count) {
    GlobalTensor<T> outGm;
    ::BindOutputTensor(outGm, outList_, out_idx);
    // 同一输出在 UB 中相隔 axis_size_ 行，在 GM 输出中连续，因此 dstStride=0。
    const AscendC::DataCopyParams params{static_cast<uint16_t>(line_count), static_cast<uint16_t>(aligned_row_block_len_db_),
                                         static_cast<uint16_t>(aligned_src_stride_db_), 0U};
    AscendC::DataCopy(outGm[outer_begin * inner_size_], ub, params);
  }

  __aicore__ inline void CopyOutLineGroupUnaligned(LocalTensor<T> ub, const uint32_t out_idx,
                                                   const uint32_t outer_begin, const uint32_t line_count) {
    GlobalTensor<T> outGm;
    ::BindOutputTensor(outGm, outList_, out_idx);
    // 非对齐版本使用补齐后的 UB pitch，只有有效 row_bytes 写回 GM。
    ::CopyUbToGmPad2D(outGm, outer_begin * inner_size_, ub, 0U, static_cast<uint16_t>(line_count),
                      row_bytes_, unaligned_src_stride_db_, 0U);
  }

  __aicore__ inline void CopyOutLinesUnaligned(LocalTensor<T> &ub, const uint32_t line_base,
                                               const uint32_t valid_lines) {
    // 输出编号推进与 aligned 版本相同，仅 UB 行 pitch 不同。
    const uint32_t group_count = (valid_lines < axis_size_) ? valid_lines : axis_size_;
    uint32_t out_idx = line_base;
    uint32_t outer_begin = 0U;
    if (out_idx >= axis_size_) {
      outer_begin = out_idx / axis_size_;
      out_idx -= outer_begin * axis_size_;
    }
    const uint32_t base_line_count = valid_lines / axis_size_;
    const uint32_t extra_groups = valid_lines - base_line_count * axis_size_;
    for (uint32_t i = 0U; i < group_count; ++i) {
      const uint32_t line_count = base_line_count + ((i < extra_groups) ? 1U : 0U);
      CopyOutLineGroupUnaligned(ub[i * ub_row_pitch_elem_], out_idx, outer_begin, line_count);
      const uint32_t next_out_idx = out_idx + 1U;
      outer_begin += next_out_idx / axis_size_;
      out_idx = next_out_idx % axis_size_;
    }
  }

  GlobalTensor<T> inputGm_;
  GM_ADDR outputList_ = nullptr;
  AscendC::ListTensorDesc outList_;
  uint32_t total_ = 0U;
  uint32_t output_num_ = 0U;
  uint32_t outer_size_ = 1U;
  uint32_t axis_size_ = 1U;
  uint32_t inner_size_ = 1U;
  uint32_t tile_lines_ = 1U;
  uint32_t row_bytes_ = 0U;
  uint32_t ub_row_pitch_bytes_ = 0U;
  uint32_t ub_row_pitch_elem_ = 0U;
  uint32_t total_lines_ = 0U;
  uint32_t aligned_row_block_len_db_ = 0U;
  uint32_t aligned_src_stride_db_ = 0U;
  uint32_t unaligned_copyin_dst_stride_db_ = 0U;
  uint32_t unaligned_src_stride_db_ = 0U;
  bool row_aligned_ = false;
  static constexpr uint32_t MID_STREAM_BUF_BYTES = 96U * 1024U;
};

template <typename T>
// 末轴 TransData 路径（key 5）
// 输入等价于紧凑矩阵 [outer, output_num]。按 [rows_per_tile, c_tile] 读入 UB 后，
// TransDataTo5HD 将行主序块转为按列连续布局，再逐列写入 output list。
//
// UB 固定布局：32B phase bias | src ping | src pong | dst ping(双 stage) | dst pong(双 stage)。
// src ping/pong 做 row tile 流水；dst stage0/1 交替覆盖向量转置与上一列块的 MTE3 写回。
class KernelUnpackAxisLastT {
 public:
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    total_ = tilingData.total_elem;
    output_num_ = tilingData.output_num;
    outer_size_ = tilingData.outer_size;
    axis_last_c_tile_ = tilingData.axis_last_c_tile;
    axis_last_groups_ = tilingData.axis_last_groups;
    axis_last_rows_per_tile_ = tilingData.axis_last_rows_per_tile;
    axis_last_col_tile_count_ = tilingData.axis_last_col_tile_count;
    axis_last_full_repeats_ = tilingData.axis_last_full_repeats;
    axis_last_tail_repeats_ = tilingData.axis_last_tail_repeats;
    inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input), total_);
    //inputGm_.template SetL2CacheHint<CacheRwMode::RW>(AscendC::CacheMode::CACHE_MODE_DISABLE);
    outputList_ = output;
    outList_ = AscendC::ListTensorDesc((__gm__ void *)outputList_);
    // 预先拆出完整/尾部 row、column tile，热点循环只需比较索引，不再重复做整除判断。
    axis_last_full_row_tile_count_ = outer_size_ / axis_last_rows_per_tile_;
    axis_last_tail_rows_ = outer_size_ - axis_last_full_row_tile_count_ * axis_last_rows_per_tile_;
    axis_last_tail_cols_ = output_num_ - (output_num_ / axis_last_c_tile_) * axis_last_c_tile_;
    axis_last_full_col_tile_count_ = axis_last_col_tile_count_ - ((axis_last_tail_cols_ != 0U) ? 1U : 0U);
  }

  __aicore__ inline void Process() {
    // UB 地址表仅依赖 tile 参数，每核初始化一次后复用全部 row/column tile。
    InitFixedTransAddrLists();
    AxisLastTransposePath();
  }

 private:
  // 总 UB 192KB，按两份 96KB 逻辑缓冲规划；phase bias/guard 由 host 预算保证。
  static constexpr uint32_t AXIS_LAST_BUFFER_BYTES = 96U * 1024U;
  static constexpr uint32_t AXIS_LAST_TOTAL_BYTES = 2U * AXIS_LAST_BUFFER_BYTES;
  static constexpr uint32_t AXIS_LAST_DST_PHASE_BIAS = 32U;
  static constexpr uint32_t AXIS_LAST_DST_COL_BLOCK_BYTES = 32U;
  static constexpr uint32_t TRANS_DST0_ADDR = AXIS_LAST_DST_PHASE_BIAS;
  static constexpr int32_t TRANS_EVT_PING = 0;
  static constexpr int32_t TRANS_EVT_PONG = 1;
  static constexpr int32_t TRANS_STAGE_EVT_PING = 2;
  static constexpr int32_t TRANS_STAGE_EVT_PONG = 3;
  // 每个 column repeat 需要一组源地址；int8 还要为上下两个 16 行半块各保留一组。
  // dst 地址表按两个 stage 连续存放，stage stride 与单次 TransData 所需地址数一致。
  static constexpr uint32_t kSrcRepeatCap = (sizeof(T) == 1U) ? 8U : ((sizeof(T) == 2U) ? 16U : 32U);
  static constexpr uint32_t kSrcAddrCacheCap =
      (sizeof(T) == 1U) ? (kSrcRepeatCap * 32U) : (kSrcRepeatCap * 16U);
  static constexpr uint32_t kDstAddrListStride = (sizeof(T) == 1U) ? 32U : 16U;
  static constexpr uint32_t kDstAddrCacheCap = 2U * kDstAddrListStride;
  uint64_t srcAddrListPing[kSrcAddrCacheCap];
  uint64_t dstAddrListPing[kDstAddrCacheCap];
  uint64_t srcAddrListPong[kSrcAddrCacheCap];
  uint64_t dstAddrListPong[kDstAddrCacheCap];
  __aicore__ inline uint32_t Align32(const uint32_t bytes) const {
    return (bytes + 31U) & (~31U);
  }

  __aicore__ inline uint32_t Align32Down(const uint32_t bytes) const {
    return bytes & (~31U);
  }

  __aicore__ inline uint32_t GetAxisLastDstStageBytes() const {
    // 一个 stage 容纳一个 32B 列块转置后的全部输出列。
    return GetAxisLastDstRowPitchBytes() * (32U / static_cast<uint32_t>(sizeof(T)));
  }

  __aicore__ inline uint32_t GetAxisLastDstScratchBytes() const {
    // 每份 dst 含两个 stage，用于列块级 ping/pong。
    return 2U * GetAxisLastDstStageBytes();
  }

  __aicore__ inline uint32_t GetAxisLastDstRowPitchElems() const {
    const uint32_t pitch_bytes = GetAxisLastDstRowPitchBytes();
    return pitch_bytes / static_cast<uint32_t>(sizeof(T));
  }

  __aicore__ inline uint32_t GetAxisLastDstRowPitchBytes() const {
    // 有效列长度向上对齐，并强制奇数 datablock pitch 以减轻 UB bank 冲突。
    uint32_t pitch_bytes = Align32(axis_last_rows_per_tile_ * static_cast<uint32_t>(sizeof(T)));
    uint32_t datablocks = pitch_bytes / 32U;
    if ((datablocks & 1U) == 0U) {
      pitch_bytes += 32U;
    }
    return pitch_bytes;
  }

  __aicore__ inline uint32_t GetTransDstAddr(const uint32_t buf_idx) const {
    // 两份 src 之后依次放 dst ping/pong，每份 dst 内部再分两个 stage。
    const uint32_t src_base = AXIS_LAST_DST_PHASE_BIAS;
    const uint32_t src_span = GetAxisLastSrcBytes();
    return src_base + 2U * src_span + buf_idx * GetAxisLastDstScratchBytes();
  }

  __aicore__ inline uint32_t GetTransSrcAddr(const uint32_t buf_idx) const {
    return AXIS_LAST_DST_PHASE_BIAS + buf_idx * GetAxisLastSrcBytes();
  }

  __aicore__ inline uint32_t GetAxisLastSrcBytes() const {
    // 扣除 phase bias 和两份 dst scratch 后，剩余空间平分给 src ping/pong。
    const uint32_t dst_total_bytes = 2U * GetAxisLastDstScratchBytes();
    return Align32Down((AXIS_LAST_TOTAL_BYTES - AXIS_LAST_DST_PHASE_BIAS - dst_total_bytes) / 2U);
  }

  __aicore__ inline uint64_t *GetSrcAddrList(const uint32_t buf_idx) {
    return (buf_idx == 0U) ? srcAddrListPing : srcAddrListPong;
  }

  __aicore__ inline uint64_t *GetDstAddrList(const uint32_t buf_idx) {
    return (buf_idx == 0U) ? dstAddrListPing : dstAddrListPong;
  }

  __aicore__ inline int32_t GetTransEvtId(const uint32_t buf_idx) const {
    return (buf_idx == 0U) ? TRANS_EVT_PING : TRANS_EVT_PONG;
  }

  __aicore__ inline int32_t GetStageEvtId(const uint32_t stage_idx) const {
    return (stage_idx == 0U) ? TRANS_STAGE_EVT_PING : TRANS_STAGE_EVT_PONG;
  }

  template <uint32_t BYTES>
  // 读入 [valid_rows, cur_cols] 子矩阵。覆盖整行时退化为连续 DMA，否则使用二维 Pad DMA，
  // 并以 c_tile 作为 UB 行 pitch，为尾列保留固定布局。
  __aicore__ inline void CopyInAxisLastTile(LocalTensor<T> &srcUb, const uint32_t valid_rows,
                                            const uint32_t c_tile,
                                            const uint32_t gm_src, const uint32_t cols_total,
                                            const uint32_t cur_cols) {
    if (cur_cols == cols_total) {
      ::CopyGmToUbContinuous(srcUb, 0U, inputGm_, gm_src, valid_rows * cur_cols);
      return;
    }
    const uint32_t cur_bytes = cur_cols * BYTES;
    const uint32_t src_stride_bytes = cols_total * BYTES - cur_bytes;
    const uint32_t dst_row_bytes = c_tile * BYTES;
    const uint32_t dst_stride_db = (dst_row_bytes - Align32(cur_bytes)) / 32U;
    ::CopyGmToUbPad2D(srcUb, 0U, inputGm_, gm_src, static_cast<uint16_t>(valid_rows), cur_bytes, src_stride_bytes,
                      dst_stride_db);
  }

  __aicore__ inline void CopyOutAxisLastTile(const LocalTensor<T> &dstCache,
                                             const uint32_t row_base, const uint32_t col_base,
                                             const uint32_t cur_cols, const uint32_t dst_row_pitch,
                                             const uint32_t valid_rows, const bool row_contiguous) {
    // dstCache=[cur_cols, dst_row_pitch]，每个 UB 行对应一个最终输出的连续 outer 段。
    GlobalTensor<T> outGm;
    uint32_t out_idx = col_base;
    uint32_t ub_row_base = 0U;
    for (uint32_t local_c = 0U; local_c < cur_cols; ++local_c) {
      ::BindOutputTensor(outGm, outList_, out_idx);
      if (row_contiguous) {
        ::CopyUbToGmContinuous(outGm, row_base, dstCache, ub_row_base, valid_rows);
      } else {
        ::CopyUbToGmPad(outGm, row_base, dstCache, ub_row_base, valid_rows);
      }
      ++out_idx;
      ub_row_base += dst_row_pitch;
    }
  }

  __aicore__ inline void CopyOutAxisLastColBlock(const LocalTensor<T> &dstCache,
                                                 const uint32_t row_base, const uint32_t col_base,
                                                 const uint32_t local_col_base, const uint32_t block_cols,
                                                 const uint32_t dst_row_pitch, const uint32_t valid_rows,
                                                 const bool row_contiguous) {
    // 只写当前 32B 列块；local_col_base 将 stage 内列号映射回整个 column tile。
    GlobalTensor<T> outGm;
    uint32_t out_idx = col_base + local_col_base;
    uint32_t ub_row_base = 0U;
    for (uint32_t local_c = 0U; local_c < block_cols; ++local_c) {
      ::BindOutputTensor(outGm, outList_, out_idx);
      if (row_contiguous) {
        ::CopyUbToGmContinuous(outGm, row_base, dstCache, ub_row_base, valid_rows);
      } else {
        ::CopyUbToGmPad(outGm, row_base, dstCache, ub_row_base, valid_rows);
      }
      ++out_idx;
      ub_row_base += dst_row_pitch;
    }
  }

  __aicore__ inline uint32_t GetAxisLastRowTileCount() const {
    return axis_last_full_row_tile_count_ + ((axis_last_tail_rows_ != 0U) ? 1U : 0U);
  }

  __aicore__ inline void GetAxisLastTileShape(const uint32_t row_tile_idx, const uint32_t col_tile_idx,
                                              uint32_t &row_base, uint32_t &col_base, uint32_t &valid_rows,
                                              uint32_t &cur_cols, bool &row_contiguous) const {
    row_base = row_tile_idx * axis_last_rows_per_tile_;
    col_base = col_tile_idx * axis_last_c_tile_;
    valid_rows = outer_size_ - row_base;
    valid_rows = Std::min(valid_rows, axis_last_rows_per_tile_);
    cur_cols = output_num_ - col_base;
    cur_cols = Std::min(cur_cols, axis_last_c_tile_);
    // 尾 row tile 不满足 32B 时写出走 DataCopyPad。
    row_contiguous = (((valid_rows * static_cast<uint32_t>(sizeof(T))) & 31U) == 0U);
  }

  template <uint32_t BYTES>
  // 执行一个列块的 V 转置与 MTE3 写出。buf_idx 选择 tile 级 ping/pong，stage_idx 选择
  // dst 内列块级 ping/pong；复用 stage 前由调用方等待上一轮 MTE3 完成。
  __aicore__ inline void ExecuteAxisLastTileBlock(const LocalTensor<T> &dstCache, const uint32_t buf_idx,
                                                  const uint8_t repeat_times, const uint16_t src_rep_stride,
                                                  const uint16_t dst_rep_stride,
                                                  const uint32_t dst_stage_elems, const uint32_t row_base,
                                                  const uint32_t col_base, const uint32_t col_off,
                                                  const uint32_t col_idx, const uint32_t block_cols,
                                                  const uint32_t dst_row_pitch, const uint32_t valid_rows,
                                                  const bool row_contiguous, const uint32_t stage_idx) {
    uint64_t *dstAddrList = GetDstAddrList(buf_idx) + stage_idx * kDstAddrListStride;
    if constexpr (BYTES == 1U) {
      // int8 的 32x32 块由四次 16x16 half 组合为上下/左右四个象限。
      constexpr uint32_t kSrcColStride = 16U;
      constexpr uint32_t kSrcRepeatCap = 8U;
      constexpr uint32_t kSrcHalfBase = kSrcRepeatCap * kSrcColStride;
      constexpr uint32_t kDstHalfBase = 16U;
      constexpr bool kLowHalf = false;
      constexpr bool kHighHalf = true;
      uint64_t *const srcAddrListBase = GetSrcAddrList(buf_idx);
      uint64_t *srcTopAddrList = srcAddrListBase + col_idx * kSrcColStride;
      uint64_t *srcBotAddrList = srcAddrListBase + kSrcHalfBase + col_idx * kSrcColStride;
      AscendC::TransDataTo5HD<T>(dstAddrList, srcTopAddrList,
                                 {kLowHalf, kLowHalf, repeat_times, dst_rep_stride, src_rep_stride});
      AscendC::TransDataTo5HD<T>(dstAddrList, srcBotAddrList,
                                 {kHighHalf, kLowHalf, repeat_times, dst_rep_stride, src_rep_stride});
      AscendC::TransDataTo5HD<T>(dstAddrList + kDstHalfBase, srcTopAddrList,
                                 {kLowHalf, kHighHalf, repeat_times, dst_rep_stride, src_rep_stride});
      AscendC::TransDataTo5HD<T>(dstAddrList + kDstHalfBase, srcBotAddrList,
                                 {kHighHalf, kHighHalf, repeat_times, dst_rep_stride, src_rep_stride});
    } else {
      // 2B/4B 一个 TransData 指令生成一个 32B 宽列块。
      uint64_t *srcAddrList = GetSrcAddrList(buf_idx) + col_idx * 16U;
      AscendC::TransDataTo5HD<T>(dstAddrList, srcAddrList,
                                 {false, false, repeat_times, dst_rep_stride, src_rep_stride});
    }
    // 当前 stage 的向量结果就绪后立即提交该列块写回。
    SyncVToMte3(GetStageEvtId(stage_idx));
    CopyOutAxisLastColBlock(dstCache[stage_idx * dst_stage_elems], row_base, col_base, col_off, block_cols,
                            dst_row_pitch, valid_rows, row_contiguous);
  }

  __aicore__ inline void InitFixedTransAddrLists() {
    // TransData 接收 UB 字节地址表。UB 内 pitch 固定，因此为 src/dst ping-pong 一次性建表。
    const uint64_t src0Base = static_cast<uint64_t>(GetTransSrcAddr(0U));
    const uint64_t src1Base = static_cast<uint64_t>(GetTransSrcAddr(1U));
    const uint64_t dst0Base = static_cast<uint64_t>(GetTransDstAddr(0U));
    const uint64_t dst1Base = static_cast<uint64_t>(GetTransDstAddr(1U));
    const uint64_t elemBytes = static_cast<uint64_t>(sizeof(T));
    if constexpr (sizeof(T) == 1U) {
      // int8 将 32 行拆成上下两个 16 行源表，dst 也分两个 16 列半区。
      constexpr uint32_t kBlockRows = 32U;
      constexpr uint32_t kSrcColStride = 16U;
      constexpr uint32_t kSrcRepeatCap = 8U;
      constexpr uint32_t kSrcHalfBase = kSrcRepeatCap * kSrcColStride;
      constexpr uint32_t kDstHalfBase = 16U;
      const uint32_t cTile = axis_last_c_tile_;
      const uint32_t dst_row_pitch = GetAxisLastDstRowPitchElems();
      for (uint32_t col_idx = 0U; col_idx < axis_last_full_repeats_; ++col_idx) {
        const uint32_t col_off = col_idx * kBlockRows;
        for (uint32_t i = 0U; i < 16U; ++i) {
          const uint32_t idx = col_idx * kSrcColStride + i;
          srcAddrListPing[idx] = src0Base + static_cast<uint64_t>(i * cTile + col_off);
          srcAddrListPing[kSrcHalfBase + idx] = src0Base + static_cast<uint64_t>((16U + i) * cTile + col_off);
          srcAddrListPong[idx] = src1Base + static_cast<uint64_t>(i * cTile + col_off);
          srcAddrListPong[kSrcHalfBase + idx] = src1Base + static_cast<uint64_t>((16U + i) * cTile + col_off);
        }
      }
      const uint64_t dstStageBytes = static_cast<uint64_t>(GetAxisLastDstStageBytes());
      for (uint32_t stage_idx = 0U; stage_idx < 2U; ++stage_idx) {
        const uint32_t stage_base = stage_idx * kDstAddrListStride;
        const uint64_t dst0StageBase = dst0Base + stage_idx * dstStageBytes;
        const uint64_t dst1StageBase = dst1Base + stage_idx * dstStageBytes;
        for (uint32_t i = 0U; i < 16U; ++i) {
          dstAddrListPing[stage_base + i] = dst0StageBase + static_cast<uint64_t>(i * dst_row_pitch);
          dstAddrListPing[stage_base + kDstHalfBase + i] =
              dst0StageBase + static_cast<uint64_t>((16U + i) * dst_row_pitch);
          dstAddrListPong[stage_base + i] = dst1StageBase + static_cast<uint64_t>(i * dst_row_pitch);
          dstAddrListPong[stage_base + kDstHalfBase + i] =
              dst1StageBase + static_cast<uint64_t>((16U + i) * dst_row_pitch);
        }
      }
      return;
    }

    // 2B/4B 使用 16 项源地址表，每项指向输入 tile 中对应行的当前列块。
    const uint32_t cTile = axis_last_c_tile_;
    for (uint32_t col_idx = 0U; col_idx < axis_last_full_repeats_; ++col_idx) {
      const uint32_t col_off = col_idx * (32U / static_cast<uint32_t>(sizeof(T)));
      for (uint32_t i = 0U; i < 16U; ++i) {
        const uint32_t idx = col_idx * 16U + i;
        const uint64_t src_off_bytes = static_cast<uint64_t>(i * cTile + col_off) * elemBytes;
        srcAddrListPing[idx] = src0Base + src_off_bytes;
        srcAddrListPong[idx] = src1Base + src_off_bytes;
      }
    }
    const uint32_t dst_row_pitch = GetAxisLastDstRowPitchElems();
    if constexpr (sizeof(T) == 2U) {
      // 2B 的 32B 列块包含 16 列，dst 地址逐列按 row pitch 排列。
      const uint64_t dstStageBytes = static_cast<uint64_t>(GetAxisLastDstStageBytes());
      for (uint32_t stage_idx = 0U; stage_idx < 2U; ++stage_idx) {
        const uint32_t stage_base = stage_idx * kDstAddrListStride;
        const uint64_t dst0StageBase = dst0Base + stage_idx * dstStageBytes;
        const uint64_t dst1StageBase = dst1Base + stage_idx * dstStageBytes;
        for (uint32_t i = 0U; i < 16U; ++i) {
          const uint64_t dst_off_bytes = static_cast<uint64_t>(i * dst_row_pitch) * elemBytes;
          dstAddrListPing[stage_base + i] = dst0StageBase + dst_off_bytes;
          dstAddrListPong[stage_base + i] = dst1StageBase + dst_off_bytes;
        }
      }
    } else {
      // 4B 的 32B 列块包含 8 列，地址表按指令要求交错两个 8 元素半区。
      const uint64_t dstStageBytes = static_cast<uint64_t>(GetAxisLastDstStageBytes());
      for (uint32_t stage_idx = 0U; stage_idx < 2U; ++stage_idx) {
        const uint32_t stage_base = stage_idx * kDstAddrListStride;
        const uint64_t dst0StageBase = dst0Base + stage_idx * dstStageBytes;
        const uint64_t dst1StageBase = dst1Base + stage_idx * dstStageBytes;
        for (uint32_t i = 0U; i < 8U; ++i) {
          const uint32_t row_base_elems = i * dst_row_pitch;
          dstAddrListPing[stage_base + 2U * i] =
              dst0StageBase + static_cast<uint64_t>(row_base_elems) * elemBytes;
          dstAddrListPing[stage_base + 2U * i + 1U] =
              dst0StageBase + static_cast<uint64_t>(row_base_elems + 8U) * elemBytes;
          dstAddrListPong[stage_base + 2U * i] =
              dst1StageBase + static_cast<uint64_t>(row_base_elems) * elemBytes;
          dstAddrListPong[stage_base + 2U * i + 1U] =
              dst1StageBase + static_cast<uint64_t>(row_base_elems + 8U) * elemBytes;
        }
      }
    }
  }

  __aicore__ inline void AxisLastTransposePath() {
    if constexpr (sizeof(T) == 1U) {
      // int8 基本转置单元为 32 行 x 32 列。
      constexpr uint32_t kBytes = 1U;
      constexpr uint32_t kColsUnit = 32U;
      LocalTensor<T> srcCache0(AscendC::TPosition::VECCALC, GetTransSrcAddr(0U),
                               GetAxisLastSrcBytes() / static_cast<uint32_t>(sizeof(T)));
      LocalTensor<T> srcCache1(AscendC::TPosition::VECCALC, GetTransSrcAddr(1U),
                               GetAxisLastSrcBytes() / static_cast<uint32_t>(sizeof(T)));
      LocalTensor<T> dstCache0(AscendC::TPosition::VECCALC, GetTransDstAddr(0U),
                               GetAxisLastDstScratchBytes() / static_cast<uint32_t>(sizeof(T)));
      LocalTensor<T> dstCache1(AscendC::TPosition::VECCALC, GetTransDstAddr(1U),
                               GetAxisLastDstScratchBytes() / static_cast<uint32_t>(sizeof(T)));
      int32_t evtPing = 0;
      int32_t evtPong = 0;
      ::InitDoubleBufferEvents(evtPing, evtPong, TRANS_EVT_PING, TRANS_EVT_PONG);
      LocalTensor<T> *srcCaches[2] = {&srcCache0, &srcCache1};
      LocalTensor<T> *dstCaches[2] = {&dstCache0, &dstCache1};
      const int32_t evtIds[2] = {evtPing, evtPong};
      uint32_t tileCounter = 0U;
      const uint32_t core_idx = AscendC::GetBlockIdx();
      const uint32_t block_dim = AscendC::GetBlockNum();
      const uint32_t row_tile_count = GetAxisLastRowTileCount();
      const uint16_t src_rep_stride = static_cast<uint16_t>(axis_last_c_tile_);
      const uint32_t dst_stage_elems = GetAxisLastDstStageBytes() / kBytes;
      const uint32_t dst_row_pitch = GetAxisLastDstRowPitchElems();
      // 外层遍历列 tile；row tile 按 core_idx+n*block_dim 分核，写出区间互不重叠。
      for (uint32_t col_tile_idx = 0U; col_tile_idx < axis_last_col_tile_count_; ++col_tile_idx) {
        for (uint32_t row_tile_idx = core_idx; row_tile_idx < row_tile_count; row_tile_idx += block_dim) {
          uint32_t row_base = 0U;
          uint32_t col_base = 0U;
          uint32_t valid_rows = 0U;
          uint32_t cur_cols = 0U;
          bool row_contiguous = false;
          GetAxisLastTileShape(row_tile_idx, col_tile_idx, row_base, col_base, valid_rows, cur_cols,
                               row_contiguous);
          const uint32_t buf_idx = tileCounter & 1U;
          WaitMte3ToMte2(evtIds[buf_idx]);
          CopyInAxisLastTile<kBytes>(*srcCaches[buf_idx], valid_rows, axis_last_c_tile_,
                                     row_base * output_num_ + col_base, output_num_, cur_cols);
          SyncMte2ToV(evtIds[buf_idx]);
          const uint8_t repeat_times =
              static_cast<uint8_t>(Std::max((valid_rows + kColsUnit - 1U) / kColsUnit, 2U));
          uint32_t col_off = 0U;
          uint32_t col_idx = 0U;
          // 先填充 stage0/1，后续每次覆写 stage 前等待其上一批写回完成。
          const uint32_t first_cols = Std::min(cur_cols, kColsUnit);
          ExecuteAxisLastTileBlock<kBytes>(*dstCaches[buf_idx], buf_idx, repeat_times, src_rep_stride, 1U,
                                           dst_stage_elems, row_base, col_base, col_off, col_idx, first_cols,
                                           dst_row_pitch, valid_rows, row_contiguous, 0U);
          col_off += first_cols;
          ++col_idx;
          if (col_off < cur_cols) {
            const uint32_t second_cols = Std::min(cur_cols - col_off, kColsUnit);
            ExecuteAxisLastTileBlock<kBytes>(*dstCaches[buf_idx], buf_idx, repeat_times, src_rep_stride, 1U,
                                             dst_stage_elems, row_base, col_base, col_off, col_idx, second_cols,
                                             dst_row_pitch, valid_rows, row_contiguous, 1U);
            col_off += second_cols;
            ++col_idx;
          }
          for (; (col_off + 2U * kColsUnit) <= cur_cols; col_off += 2U * kColsUnit, col_idx += 2U) {
            SyncMte3ToV(GetStageEvtId(0U));
            ExecuteAxisLastTileBlock<kBytes>(*dstCaches[buf_idx], buf_idx, repeat_times, src_rep_stride, 1U,
                                             dst_stage_elems, row_base, col_base, col_off, col_idx, kColsUnit,
                                             dst_row_pitch, valid_rows, row_contiguous, 0U);
            SyncMte3ToV(GetStageEvtId(1U));
            ExecuteAxisLastTileBlock<kBytes>(*dstCaches[buf_idx], buf_idx, repeat_times, src_rep_stride, 1U,
                                             dst_stage_elems, row_base, col_base, col_off + kColsUnit, col_idx + 1U,
                                             kColsUnit, dst_row_pitch, valid_rows, row_contiguous, 1U);
          }
          if (col_off < cur_cols) {
            SyncMte3ToV(GetStageEvtId(0U));
            const uint32_t tail_cols = Std::min(cur_cols - col_off, kColsUnit);
            ExecuteAxisLastTileBlock<kBytes>(*dstCaches[buf_idx], buf_idx, repeat_times, src_rep_stride, 1U,
                                             dst_stage_elems, row_base, col_base, col_off, col_idx, tail_cols,
                                             dst_row_pitch, valid_rows, row_contiguous, 0U);
            col_off += tail_cols;
            ++col_idx;
          }
          if (col_off < cur_cols) {
            SyncMte3ToV(GetStageEvtId(1U));
            ExecuteAxisLastTileBlock<kBytes>(*dstCaches[buf_idx], buf_idx, repeat_times, src_rep_stride, 1U,
                                             dst_stage_elems, row_base, col_base, col_off, col_idx,
                                             cur_cols - col_off, dst_row_pitch, valid_rows, row_contiguous, 1U);
          }
          SyncMte3ToV(GetStageEvtId(0U));
          if (cur_cols > kColsUnit) {
            SyncMte3ToV(GetStageEvtId(1U));
          }
          SignalMte3ToMte2(evtIds[buf_idx]);
          ++tileCounter;
        }
      }
      ::FinishDoubleBufferEvents(evtPing, evtPong);
      return;
    }
    if constexpr (sizeof(T) == 2U) {
      // 2B 基本列块为 16 列，基本 row group 为 16 行。
      constexpr uint32_t kColsUnit = 16U;
      LocalTensor<T> srcSlab0(AscendC::TPosition::VECCALC, GetTransSrcAddr(0U),
                              GetAxisLastSrcBytes() / static_cast<uint32_t>(sizeof(T)));
      LocalTensor<T> srcSlab1(AscendC::TPosition::VECCALC, GetTransSrcAddr(1U),
                              GetAxisLastSrcBytes() / static_cast<uint32_t>(sizeof(T)));
      LocalTensor<T> dstCache0(AscendC::TPosition::VECCALC, GetTransDstAddr(0U),
                               GetAxisLastDstScratchBytes() / static_cast<uint32_t>(sizeof(T)));
      LocalTensor<T> dstCache1(AscendC::TPosition::VECCALC, GetTransDstAddr(1U),
                               GetAxisLastDstScratchBytes() / static_cast<uint32_t>(sizeof(T)));
      int32_t evtPing = 0;
      int32_t evtPong = 0;
      ::InitDoubleBufferEvents(evtPing, evtPong, TRANS_EVT_PING, TRANS_EVT_PONG);
      LocalTensor<T> *srcCaches[2] = {&srcSlab0, &srcSlab1};
      LocalTensor<T> *dstCaches[2] = {&dstCache0, &dstCache1};
      const int32_t evtIds[2] = {evtPing, evtPong};
      uint32_t tileCounter = 0U;
      const uint32_t core_idx = AscendC::GetBlockIdx();
      const uint32_t block_dim = AscendC::GetBlockNum();
      const uint32_t row_tile_count = GetAxisLastRowTileCount();
      constexpr uint32_t kBlockRows = 16U;
      constexpr uint32_t kBytes = 2U;
      // TransData 的 repeat stride 以 32B datablock 为单位；一次 repeat 跨过 16 行源数据。
      const uint16_t src_rep_stride = static_cast<uint16_t>((kBlockRows * axis_last_c_tile_ * kBytes) / 32U);
      const uint32_t dst_stage_elems = GetAxisLastDstStageBytes() / kBytes;
      const uint32_t dst_row_pitch = GetAxisLastDstRowPitchElems();
      // column tile 逐个扫描；每核以 block_dim 为步长领取互不重叠的 row tile。
      for (uint32_t col_tile_idx = 0U; col_tile_idx < axis_last_col_tile_count_; ++col_tile_idx) {
        for (uint32_t row_tile_idx = core_idx; row_tile_idx < row_tile_count; row_tile_idx += block_dim) {
          uint32_t row_base = 0U;
          uint32_t col_base = 0U;
          uint32_t valid_rows = 0U;
          uint32_t cur_cols = 0U;
          bool row_contiguous = false;
          GetAxisLastTileShape(row_tile_idx, col_tile_idx, row_base, col_base, valid_rows, cur_cols,
                               row_contiguous);
          const uint32_t buf_idx = tileCounter & 1U;
          WaitMte3ToMte2(evtIds[buf_idx]);
          CopyInAxisLastTile<kBytes>(*srcCaches[buf_idx], valid_rows, axis_last_c_tile_,
                                     row_base * output_num_ + col_base, output_num_, cur_cols);
          SyncMte2ToV(evtIds[buf_idx]);
          // 保持 TransData 至少两个 repeat；尾部无效行不会被 CopyOut 写回 GM。
          const uint8_t repeat_times =
              static_cast<uint8_t>(Std::max((valid_rows + kBlockRows - 1U) / kBlockRows, 2U));
          uint32_t col_off = 0U;
          uint32_t col_idx = 0U;
          // 先占用 stage0/1；之后每复用一个 stage，先等待该 stage 上一列块写回完成。
          const uint32_t first_cols = Std::min(cur_cols, kColsUnit);
          ExecuteAxisLastTileBlock<kBytes>(*dstCaches[buf_idx], buf_idx, repeat_times, src_rep_stride, 1U,
                                           dst_stage_elems, row_base, col_base, col_off, col_idx, first_cols,
                                           dst_row_pitch, valid_rows, row_contiguous, 0U);
          col_off += first_cols;
          ++col_idx;
          if (col_off < cur_cols) {
            const uint32_t second_cols = Std::min(cur_cols - col_off, kColsUnit);
            ExecuteAxisLastTileBlock<kBytes>(*dstCaches[buf_idx], buf_idx, repeat_times, src_rep_stride, 1U,
                                             dst_stage_elems, row_base, col_base, col_off, col_idx, second_cols,
                                             dst_row_pitch, valid_rows, row_contiguous, 1U);
            col_off += second_cols;
            ++col_idx;
          }
          for (; (col_off + 2U * kColsUnit) <= cur_cols; col_off += 2U * kColsUnit, col_idx += 2U) {
            SyncMte3ToV(GetStageEvtId(0U));
            ExecuteAxisLastTileBlock<kBytes>(*dstCaches[buf_idx], buf_idx, repeat_times, src_rep_stride, 1U,
                                             dst_stage_elems, row_base, col_base, col_off, col_idx, kColsUnit,
                                             dst_row_pitch, valid_rows, row_contiguous, 0U);
            SyncMte3ToV(GetStageEvtId(1U));
            ExecuteAxisLastTileBlock<kBytes>(*dstCaches[buf_idx], buf_idx, repeat_times, src_rep_stride, 1U,
                                             dst_stage_elems, row_base, col_base, col_off + kColsUnit, col_idx + 1U,
                                             kColsUnit, dst_row_pitch, valid_rows, row_contiguous, 1U);
          }
          if (col_off < cur_cols) {
            SyncMte3ToV(GetStageEvtId(0U));
            const uint32_t tail_cols = Std::min(cur_cols - col_off, kColsUnit);
            ExecuteAxisLastTileBlock<kBytes>(*dstCaches[buf_idx], buf_idx, repeat_times, src_rep_stride, 1U,
                                             dst_stage_elems, row_base, col_base, col_off, col_idx, tail_cols,
                                             dst_row_pitch, valid_rows, row_contiguous, 0U);
            col_off += tail_cols;
            ++col_idx;
          }
          if (col_off < cur_cols) {
            SyncMte3ToV(GetStageEvtId(1U));
            ExecuteAxisLastTileBlock<kBytes>(*dstCaches[buf_idx], buf_idx, repeat_times, src_rep_stride, 1U,
                                             dst_stage_elems, row_base, col_base, col_off, col_idx,
                                             cur_cols - col_off, dst_row_pitch, valid_rows, row_contiguous, 1U);
          }
          SyncMte3ToV(GetStageEvtId(0U));
          if (cur_cols > kColsUnit) {
            SyncMte3ToV(GetStageEvtId(1U));
          }
          // tile 级事件覆盖 src 与其两份 dst stage，全部释放后下一轮 MTE2 才能复用该组地址。
          SignalMte3ToMte2(evtIds[buf_idx]);
          ++tileCounter;
        }
      }
      ::FinishDoubleBufferEvents(evtPing, evtPong);
      return;
    }
    if constexpr (sizeof(T) == 4U) {
      // 4B 基本列块为 8 列，dst repeat stride=2 对应相邻结果的 64B 跨度。
      constexpr uint32_t kColsUnit = 8U;
      LocalTensor<T> srcSlab0(AscendC::TPosition::VECCALC, GetTransSrcAddr(0U),
                              GetAxisLastSrcBytes() / static_cast<uint32_t>(sizeof(T)));
      LocalTensor<T> srcSlab1(AscendC::TPosition::VECCALC, GetTransSrcAddr(1U),
                              GetAxisLastSrcBytes() / static_cast<uint32_t>(sizeof(T)));
      LocalTensor<T> dstCache0(AscendC::TPosition::VECCALC, GetTransDstAddr(0U),
                               GetAxisLastDstScratchBytes() / static_cast<uint32_t>(sizeof(T)));
      LocalTensor<T> dstCache1(AscendC::TPosition::VECCALC, GetTransDstAddr(1U),
                               GetAxisLastDstScratchBytes() / static_cast<uint32_t>(sizeof(T)));
      int32_t evtPing = 0;
      int32_t evtPong = 0;
      ::InitDoubleBufferEvents(evtPing, evtPong, TRANS_EVT_PING, TRANS_EVT_PONG);
      LocalTensor<T> *srcCaches[2] = {&srcSlab0, &srcSlab1};
      LocalTensor<T> *dstCaches[2] = {&dstCache0, &dstCache1};
      const int32_t evtIds[2] = {evtPing, evtPong};
      uint32_t tileCounter = 0U;
      const uint32_t core_idx = AscendC::GetBlockIdx();
      const uint32_t block_dim = AscendC::GetBlockNum();
      const uint32_t row_tile_count = GetAxisLastRowTileCount();
      constexpr uint32_t kBlockRows = 16U;
      constexpr uint32_t kBytes = 4U;
      // 4B 路径仍按 16 行构造源地址表，repeat stride 同样换算为 32B datablock 数。
      const uint16_t src_rep_stride = static_cast<uint16_t>((kBlockRows * axis_last_c_tile_ * kBytes) / 32U);
      const uint32_t dst_stage_elems = GetAxisLastDstStageBytes() / kBytes;
      const uint32_t dst_row_pitch = GetAxisLastDstRowPitchElems();
      // 调度与 2B 路径一致，但一个 32B 列块只包含 8 个 4B 输出列。
      for (uint32_t col_tile_idx = 0U; col_tile_idx < axis_last_col_tile_count_; ++col_tile_idx) {
        for (uint32_t row_tile_idx = core_idx; row_tile_idx < row_tile_count; row_tile_idx += block_dim) {
          uint32_t row_base = 0U;
          uint32_t col_base = 0U;
          uint32_t valid_rows = 0U;
          uint32_t cur_cols = 0U;
          bool row_contiguous = false;
          GetAxisLastTileShape(row_tile_idx, col_tile_idx, row_base, col_base, valid_rows, cur_cols,
                               row_contiguous);
          const uint32_t buf_idx = tileCounter & 1U;
          WaitMte3ToMte2(evtIds[buf_idx]);
          CopyInAxisLastTile<kBytes>(*srcCaches[buf_idx], valid_rows, axis_last_c_tile_,
                                     row_base * output_num_ + col_base, output_num_, cur_cols);
          SyncMte2ToV(evtIds[buf_idx]);
          // repeat 向上覆盖有效行，实际写回长度仍由 valid_rows 精确裁剪。
          const uint8_t repeat_times =
              static_cast<uint8_t>(Std::max((valid_rows + kBlockRows - 1U) / kBlockRows, 2U));
          uint32_t col_off = 0U;
          uint32_t col_idx = 0U;
          // 两个 stage 交替承载相邻列块，使当前 V 转置可与上一 stage 的 MTE3 写回重叠。
          const uint32_t first_cols = Std::min(cur_cols, kColsUnit);
          ExecuteAxisLastTileBlock<kBytes>(*dstCaches[buf_idx], buf_idx, repeat_times, src_rep_stride, 2U,
                                           dst_stage_elems, row_base, col_base, col_off, col_idx, first_cols,
                                           dst_row_pitch, valid_rows, row_contiguous, 0U);
          col_off += first_cols;
          ++col_idx;
          if (col_off < cur_cols) {
            const uint32_t second_cols = Std::min(cur_cols - col_off, kColsUnit);
            ExecuteAxisLastTileBlock<kBytes>(*dstCaches[buf_idx], buf_idx, repeat_times, src_rep_stride, 2U,
                                             dst_stage_elems, row_base, col_base, col_off, col_idx, second_cols,
                                             dst_row_pitch, valid_rows, row_contiguous, 1U);
            col_off += second_cols;
            ++col_idx;
          }
          for (; (col_off + 2U * kColsUnit) <= cur_cols; col_off += 2U * kColsUnit, col_idx += 2U) {
            SyncMte3ToV(GetStageEvtId(0U));
            ExecuteAxisLastTileBlock<kBytes>(*dstCaches[buf_idx], buf_idx, repeat_times, src_rep_stride, 2U,
                                             dst_stage_elems, row_base, col_base, col_off, col_idx, kColsUnit,
                                             dst_row_pitch, valid_rows, row_contiguous, 0U);
            SyncMte3ToV(GetStageEvtId(1U));
            ExecuteAxisLastTileBlock<kBytes>(*dstCaches[buf_idx], buf_idx, repeat_times, src_rep_stride, 2U,
                                             dst_stage_elems, row_base, col_base, col_off + kColsUnit, col_idx + 1U,
                                             kColsUnit, dst_row_pitch, valid_rows, row_contiguous, 1U);
          }
          if (col_off < cur_cols) {
            SyncMte3ToV(GetStageEvtId(0U));
            const uint32_t tail_cols = Std::min(cur_cols - col_off, kColsUnit);
            ExecuteAxisLastTileBlock<kBytes>(*dstCaches[buf_idx], buf_idx, repeat_times, src_rep_stride, 2U,
                                             dst_stage_elems, row_base, col_base, col_off, col_idx, tail_cols,
                                             dst_row_pitch, valid_rows, row_contiguous, 0U);
            col_off += tail_cols;
            ++col_idx;
          }
          if (col_off < cur_cols) {
            SyncMte3ToV(GetStageEvtId(1U));
            ExecuteAxisLastTileBlock<kBytes>(*dstCaches[buf_idx], buf_idx, repeat_times, src_rep_stride, 2U,
                                             dst_stage_elems, row_base, col_base, col_off, col_idx,
                                             cur_cols - col_off, dst_row_pitch, valid_rows, row_contiguous, 1U);
          }
          SyncMte3ToV(GetStageEvtId(0U));
          if (cur_cols > kColsUnit) {
            SyncMte3ToV(GetStageEvtId(1U));
          }
          // 发布 tile 组空闲信号，允许同一 ping/pong 地址被下一 row tile 重新读入。
          SignalMte3ToMte2(evtIds[buf_idx]);
          ++tileCounter;
        }
      }
      ::FinishDoubleBufferEvents(evtPing, evtPong);
      return;
    }
  }

  GlobalTensor<T> inputGm_;
  GM_ADDR outputList_ = nullptr;
  AscendC::ListTensorDesc outList_;
  uint32_t total_ = 0;
  uint32_t output_num_ = 0;
  uint32_t outer_size_ = 1;
  uint32_t axis_last_c_tile_ = 0;
  uint32_t axis_last_groups_ = 0;
  uint32_t axis_last_rows_per_tile_ = 0;
  uint32_t axis_last_col_tile_count_ = 0;
  uint32_t axis_last_full_repeats_ = 0;
  uint32_t axis_last_tail_repeats_ = 0;
  uint32_t axis_last_full_row_tile_count_ = 0;
  uint32_t axis_last_full_col_tile_count_ = 0;
  uint32_t axis_last_tail_rows_ = 0;
  uint32_t axis_last_tail_cols_ = 0;
};

template <typename TKernel>
// 统一路径入口；每个静态 TILING_KEY 仅实例化对应实现类。
__aicore__ inline void RunUnpackKernel(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
  TKernel op;
  op.Init(input, output, workspace, tiling);
  op.Process();
}

template <typename T>
using KernelUnpackAxisLastTranspose = KernelUnpackAxisLastT<T>;

template <typename T>
// 末轴 Gather 路径（key 8）
// 用于 input row_bytes 非 32B 对齐的矩阵 [outer, output_num]，本版本覆盖 1B/2B/4B。
// src tile 紧凑读入多行，index[row]=row*row_bytes；以 col*sizeof(T) 为 Gather 基址即可得到
// 第 col 列连续的 valid_rows 个元素。
//
// UB：src0|src1|四份 dst stage|phase pad|index0|phase pad|index1。
// src/index 做 row tile ping/pong，dst stage 做 column chunk ping/pong。
class KernelUnpackAxisLastGather {
 public:
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    total_ = tilingData.total_elem;
    output_num_ = tilingData.output_num;
    outer_size_ = tilingData.outer_size;
    axis_last_rows_per_tile_ = tilingData.axis_last_rows_per_tile;
    gather_chunk_cols_ = tilingData.axis_last_c_tile;
    inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input), total_);
    //inputGm_.template SetL2CacheHint<CacheRwMode::RW>(AscendC::CacheMode::CACHE_MODE_DISABLE);
    outputList_ = output;
    outList_ = AscendC::ListTensorDesc((__gm__ void *)outputList_);
    // index pitch 存 uint32 字节偏移；dst pitch 存单列 Gather 结果。
    row_bytes_ = output_num_ * static_cast<uint32_t>(sizeof(T));
    row_template_pitch_bytes_ = (axis_last_rows_per_tile_ * static_cast<uint32_t>(sizeof(uint32_t)) + 31U) & (~31U);
    row_template_pitch_elem_ = row_template_pitch_bytes_ / static_cast<uint32_t>(sizeof(uint32_t));
    dst_row_pitch_bytes_ = (axis_last_rows_per_tile_ * static_cast<uint32_t>(sizeof(T)) + 31U) & (~31U);
    // 奇数 datablock pitch 用于降低相邻列的 UB bank 冲突。
    if (((dst_row_pitch_bytes_ / 32U) & 1U) == 0U) {
      dst_row_pitch_bytes_ += 32U;
    }
    dst_row_pitch_elem_ = dst_row_pitch_bytes_ / static_cast<uint32_t>(sizeof(T));
    src_tile_bytes_ = (axis_last_rows_per_tile_ * row_bytes_ + 31U) & (~31U);
    index_tile_bytes_ = row_template_pitch_bytes_;
    dst_tile_bytes_ = gather_chunk_cols_ * dst_row_pitch_bytes_;
  }

  __aicore__ inline void Process() {
    // host 按同一布局核算容量；kernel 用固定偏移构造 LocalTensor，不做动态分配。
    const uint32_t index_total_bytes = index_tile_bytes_;
    const uint32_t src0_addr = 0U;
    const uint32_t src1_addr = src0_addr + src_tile_bytes_;
    const uint32_t dst00_addr = src1_addr + src_tile_bytes_;
    const uint32_t dst01_addr = dst00_addr + dst_tile_bytes_;
    const uint32_t dst10_addr = dst01_addr + dst_tile_bytes_;
    const uint32_t dst11_addr = dst10_addr + dst_tile_bytes_;
    const uint32_t index0_addr = dst11_addr + dst_tile_bytes_ + INDEX_PHASE_PAD_BYTES;
    const uint32_t index1_addr = index0_addr + index_total_bytes + INDEX_PHASE_PAD_BYTES;
    LocalTensor<T> srcUb0(AscendC::TPosition::VECCALC, src0_addr, src_tile_bytes_ / static_cast<uint32_t>(sizeof(T)));
    LocalTensor<T> srcUb1(AscendC::TPosition::VECCALC, src1_addr, src_tile_bytes_ / static_cast<uint32_t>(sizeof(T)));
    LocalTensor<T> dstUb00(AscendC::TPosition::VECCALC, dst00_addr, dst_tile_bytes_ / static_cast<uint32_t>(sizeof(T)));
    LocalTensor<T> dstUb01(AscendC::TPosition::VECCALC, dst01_addr, dst_tile_bytes_ / static_cast<uint32_t>(sizeof(T)));
    LocalTensor<T> dstUb10(AscendC::TPosition::VECCALC, dst10_addr, dst_tile_bytes_ / static_cast<uint32_t>(sizeof(T)));
    LocalTensor<T> dstUb11(AscendC::TPosition::VECCALC, dst11_addr, dst_tile_bytes_ / static_cast<uint32_t>(sizeof(T)));
    LocalTensor<uint32_t> indexUb0(AscendC::TPosition::VECCALC, index0_addr, index_total_bytes / sizeof(uint32_t));
    LocalTensor<uint32_t> indexUb1(AscendC::TPosition::VECCALC, index1_addr, index_total_bytes / sizeof(uint32_t));
    // 两份偏移模板只依赖 row_bytes，在主循环前各构造一次。
    BuildRowOffsets(indexUb0, axis_last_rows_per_tile_);
    BuildRowOffsets(indexUb1, axis_last_rows_per_tile_);
    int32_t evtPing = TRANS_EVT_PING;
    int32_t evtPong = TRANS_EVT_PONG;
    ::InitDoubleBufferEvents(evtPing, evtPong, TRANS_EVT_PING, TRANS_EVT_PONG);
    LocalTensor<T> *srcBufs[2] = {&srcUb0, &srcUb1};
    LocalTensor<uint32_t> *indexBufs[2] = {&indexUb0, &indexUb1};
    LocalTensor<T> *dstPingBufs[2] = {&dstUb00, &dstUb10};
    LocalTensor<T> *dstPongBufs[2] = {&dstUb01, &dstUb11};
    const int32_t evtIds[2] = {evtPing, evtPong};
    int32_t rowEvtPing = TRANS_ROW_EVT_PING;
    int32_t rowEvtPong = TRANS_ROW_EVT_PONG;
    const int32_t rowEvtIds[2] = {rowEvtPing, rowEvtPong};
    const uint32_t core_idx = AscendC::GetBlockIdx();
    const uint32_t block_dim = AscendC::GetBlockNum();
    // 每核负责 core_idx、core_idx+block_dim... 的 row tile。
    const uint32_t row_tile_stride = block_dim * axis_last_rows_per_tile_;
    uint32_t tileCounter = 0U;
    uint32_t row_base = core_idx * axis_last_rows_per_tile_;
    for (; row_base < outer_size_; row_base += row_tile_stride) {
      uint32_t valid_rows = outer_size_ - row_base;
      valid_rows =Std::min(valid_rows,axis_last_rows_per_tile_);
      const uint32_t buf_idx = tileCounter & 1U;
      ProcessTile(*srcBufs[buf_idx], *indexBufs[buf_idx], *dstPingBufs[buf_idx], *dstPongBufs[buf_idx],
                  row_base, valid_rows, evtIds[buf_idx], rowEvtIds);
      ++tileCounter;
    }
    ::FinishDoubleBufferEvents(evtPing, evtPong);
  }

 private:
  // 输入多行在 GM 中整体连续，因此一次读入紧凑 src UB，不在行间插 padding。
  __aicore__ inline void CopyInPackedTile(LocalTensor<T> &srcUb, const uint32_t row_base, const uint32_t valid_rows) {
    const uint32_t gm_src = row_base * output_num_;
    const uint32_t elem_cnt = valid_rows * output_num_;
    ::CopyGmToUbPad(srcUb, 0U, inputGm_, gm_src, elem_cnt);
  }

  __aicore__ inline void BuildRowOffsets(LocalTensor<uint32_t> &indexUb, const uint32_t valid_rows) {
    // 标量生成首 64 项，后续用 Adds 复制前一块并增加 64*row_bytes。
    constexpr uint32_t kVecIterElems = 64U;
    const uint32_t seed_rows = (valid_rows < kVecIterElems) ? valid_rows : kVecIterElems;
    for (uint32_t row = 0U; row < seed_rows; ++row) {
      indexUb.SetValue(row, row * row_bytes_);
    }

    const int32_t row_block_delta = static_cast<int32_t>(kVecIterElems * row_bytes_);
    for (uint32_t row_base = kVecIterElems; row_base < valid_rows; row_base += kVecIterElems) {
      const uint32_t cur_block =
          ((valid_rows - row_base) < kVecIterElems) ? (valid_rows - row_base) : kVecIterElems;
      AscendC::Adds<int32_t>(indexUb[row_base].ReinterpretCast<int32_t>(),
                              indexUb[row_base - kVecIterElems].ReinterpretCast<int32_t>(), row_block_delta,
                              static_cast<int32_t>(cur_block));
    }
    
    // 对齐 padding 清零，避免尾向量接触未定义 UB 内容。
    for (uint32_t row = valid_rows; row < row_template_pitch_elem_; ++row) {
      indexUb.SetValue(row, 0U);
    }
  }

  __aicore__ inline void CopyOutGatherRow(const LocalTensor<T> &dstUb,
                                          const uint32_t row_base, const uint32_t col, const uint32_t valid_rows) {
    // 一个 Gather row 对应一个动态输出的连续 outer 区间。
    GlobalTensor<T> outGm;
    ::BindOutputTensor(outGm, outList_, col);
    ::CopyUbToGmPad(outGm, row_base, dstUb, 0U, valid_rows);
  }

  __aicore__ inline void CopyOutChunk(const LocalTensor<T> &dstUb,
                                      const uint32_t row_base, const uint32_t col_base, const uint32_t chunk_cols,
                                      const uint32_t valid_rows) {
    // 不同列在 dstUb 中相隔 dst_row_pitch_elem_。
    for (uint32_t i = 0U; i < chunk_cols; ++i) {
      CopyOutGatherRow(dstUb[i * dst_row_pitch_elem_], row_base, col_base + i, valid_rows);
    }
  }

  __aicore__ inline void GatherChunk(LocalTensor<uint32_t> &indexUb, const LocalTensor<T> &srcUb,
                                     LocalTensor<T> &dstUb, const uint32_t valid_rows, const uint32_t col_base,
                                     const uint32_t chunk_cols) {
    // offset 按字节选行，src_base_addr 按字节选列。
    LocalTensor<uint32_t> rowOffset = indexUb;
    for (uint32_t c = 0U; c < chunk_cols; ++c) {
      const uint32_t src_base_addr = (col_base + c) * static_cast<uint32_t>(sizeof(T));
      AscendC::Gather(dstUb[c * dst_row_pitch_elem_], srcUb, rowOffset, src_base_addr, valid_rows);
    }
  }

  __aicore__ inline void CopyOutGatherChunk(const LocalTensor<T> &dstUb,
                                            const uint32_t row_base, const uint32_t col_base,
                                            const uint32_t chunk_cols, const uint32_t valid_rows,
                                            const int32_t rowEvtId) {
    // 当前 column chunk Gather 完成后提交其全部输出 DMA。
    SyncVToMte3(rowEvtId);
    CopyOutChunk(dstUb, row_base, col_base, chunk_cols, valid_rows);
  }

  __aicore__ inline void FinishCopyOutGatherChunk(const uint32_t cur_slice_rows, const int32_t rowEvtId) {
    // 当前实现固定插入 MTE3->V 同步；cur_slice_rows 保留 stage 使用状态，便于后续裁剪首次等待。
    SyncMte3ToV(rowEvtId);
  }

  __aicore__ inline void ProcessTile(LocalTensor<T> &srcUb, LocalTensor<uint32_t> &indexUb,
                                     LocalTensor<T> &dstUb0, LocalTensor<T> &dstUb1,
                                     const uint32_t row_base, const uint32_t valid_rows, const int32_t evtId,
                                     const int32_t rowEvtIds[2]) {
    // 复用该 tile 组前，等待上一轮对其 dst/src 的 MTE3 消费结束。
    WaitMte3ToMte2(evtId);
    CopyInPackedTile(srcUb, row_base, valid_rows);
    SyncMte2ToV(evtId);
    const uint32_t chunk_cols_cap = (output_num_ < gather_chunk_cols_) ? output_num_ : gather_chunk_cols_;
    LocalTensor<T> *dstBufs[2] = {&dstUb0, &dstUb1};
    uint32_t rows_done[2] = {0U, 0U};
    uint32_t buf_idx = 0U;
    // 列分 chunk，两个 dst stage 交替覆盖 Gather 与上一 chunk 写回。
    for (uint32_t col_base = 0U; col_base < output_num_; col_base += chunk_cols_cap) {
      const uint32_t chunk_cols =
          ((output_num_ - col_base) < chunk_cols_cap) ? (output_num_ - col_base) : chunk_cols_cap;
      LocalTensor<T> &dstUb = *dstBufs[buf_idx];
      const int32_t rowEvtId = rowEvtIds[buf_idx];
      FinishCopyOutGatherChunk(rows_done[buf_idx], rowEvtId);
      GatherChunk(indexUb, srcUb, dstUb, valid_rows, col_base, chunk_cols);
      CopyOutGatherChunk(dstUb, row_base, col_base, chunk_cols, valid_rows, rowEvtId);
      rows_done[buf_idx] = valid_rows;
      buf_idx ^= 1U;
    }
    FinishCopyOutGatherChunk(rows_done[0], rowEvtIds[0]);
    FinishCopyOutGatherChunk(rows_done[1], rowEvtIds[1]);
    SignalMte3ToMte2(evtId);
  }

  GlobalTensor<T> inputGm_;
  GM_ADDR outputList_ = nullptr;
  AscendC::ListTensorDesc outList_;
  uint32_t total_ = 0U;
  uint32_t output_num_ = 0U;
  uint32_t outer_size_ = 0U;
  uint32_t axis_last_rows_per_tile_ = 1U;
  uint32_t gather_chunk_cols_ = 1U;
  uint32_t row_bytes_ = 0U;
  uint32_t row_template_pitch_bytes_ = 0U;
  uint32_t row_template_pitch_elem_ = 0U;
  uint32_t dst_row_pitch_bytes_ = 0U;
  uint32_t dst_row_pitch_elem_ = 0U;
  uint32_t src_tile_bytes_ = 0U;
  uint32_t index_tile_bytes_ = 0U;
  uint32_t dst_tile_bytes_ = 0U;
  static constexpr uint32_t INDEX_PHASE_PAD_BYTES = 32U;
  static constexpr int32_t TRANS_EVT_PING = 0;
  static constexpr int32_t TRANS_EVT_PONG = 1;
  static constexpr int32_t TRANS_ROW_EVT_PING = 2;
  static constexpr int32_t TRANS_ROW_EVT_PONG = 3;
};

#if defined(DTYPE_INPUT)
// 只按字宽选择无符号搬运类型，保持输入位模式不变。
using UnpackUIntByWidth =
    std::conditional_t<sizeof(DTYPE_INPUT) == 1U, uint8_t,
    std::conditional_t<sizeof(DTYPE_INPUT) == 2U, uint16_t, uint32_t>>;
#else
using UnpackUIntByWidth = uint32_t;
#endif

extern "C" __global__ __aicore__ void unpack(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
  // 所有路径只使用 AIV/MTE；key 必须与 host TilingFunc 同步维护。
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::InitSocState();
  if (TILING_KEY_IS(1)) {
    RunUnpackKernel<KernelUnpackAxis0Mode0<UnpackUIntByWidth>>(input, output, workspace, tiling);
  } else if (TILING_KEY_IS(2)) {
    RunUnpackKernel<KernelUnpackAxis0Mode1<UnpackUIntByWidth>>(input, output, workspace, tiling);
  } else if (TILING_KEY_IS(7)) {
    RunUnpackKernel<KernelUnpackMidAxisStream<UnpackUIntByWidth>>(input, output, workspace, tiling);
  } else if (TILING_KEY_IS(5)) {
    RunUnpackKernel<KernelUnpackAxisLastTranspose<UnpackUIntByWidth>>(input, output, workspace, tiling);
  } else if (TILING_KEY_IS(8)) {
    RunUnpackKernel<KernelUnpackAxisLastGather<UnpackUIntByWidth>>(input, output, workspace, tiling);
  }
}
