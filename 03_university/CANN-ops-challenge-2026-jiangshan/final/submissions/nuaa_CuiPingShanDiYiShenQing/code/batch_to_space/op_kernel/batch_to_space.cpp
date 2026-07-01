# include "kernel_operator.h"

# include "batch_to_space_tiling.h"
# include "tiling_key_batch_to_space.h"

namespace ac = AscendC;

namespace {
constexpr uint32_t kBtsDataBlockBytes = 32U;
constexpr int32_t kBtsEventFreeBuf0 = EVENT_ID0;
constexpr int32_t kBtsEventFreeBuf1 = EVENT_ID1;
constexpr int32_t kBtsEventReadyBuf0 = EVENT_ID2;
constexpr int32_t kBtsEventReadyBuf1 = EVENT_ID3;
constexpr int32_t kBtsEventPackReadyBuf0 = EVENT_ID4;
constexpr int32_t kBtsEventPackReadyBuf1 = EVENT_ID5;

struct BtsTaskRange {
    uint32_t offset;
    uint32_t length;
};

__aicore__ inline uint32_t BtsMin(uint32_t lhs, uint32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline BtsTaskRange BtsGetStaticTaskRange(uint32_t total_tasks, uint32_t active_cores)
{
    const uint32_t block_idx = static_cast<uint32_t>(ac::GetBlockIdx());
    if (total_tasks == 0U || active_cores == 0U || block_idx >= active_cores) {
        return {0U, 0U};
    }

    const uint32_t base_tasks = total_tasks / active_cores;
    const uint32_t former_num = total_tasks % active_cores;
    const uint32_t length = base_tasks + (block_idx < former_num ? 1U : 0U);
    const uint32_t offset = block_idx * base_tasks + BtsMin(block_idx, former_num);
    return {offset, length};
}

__aicore__ inline int32_t BtsFreeEventId(uint32_t buf_idx)
{
    return buf_idx == 0U ? kBtsEventFreeBuf0 : kBtsEventFreeBuf1;
}

__aicore__ inline int32_t BtsReadyEventId(uint32_t buf_idx)
{
    return buf_idx == 0U ? kBtsEventReadyBuf0 : kBtsEventReadyBuf1;
}

__aicore__ inline int32_t BtsPackReadyEventId(uint32_t buf_idx)
{
    return buf_idx == 0U ? kBtsEventPackReadyBuf0 : kBtsEventPackReadyBuf1;
}

__aicore__ inline void BtsMarkBufFree(uint32_t buf_idx)
{
    ac::SetFlag<ac::HardEvent::MTE3_MTE2>(BtsFreeEventId(buf_idx));
}

__aicore__ inline void BtsWaitBufFree(uint32_t buf_idx)
{
    ac::WaitFlag<ac::HardEvent::MTE3_MTE2>(BtsFreeEventId(buf_idx));
}

__aicore__ inline void BtsMarkMte2ReadyForV(uint32_t buf_idx)
{
    ac::SetFlag<ac::HardEvent::MTE2_V>(BtsReadyEventId(buf_idx));
}

__aicore__ inline void BtsWaitMte2ReadyForV(uint32_t buf_idx)
{
    ac::WaitFlag<ac::HardEvent::MTE2_V>(BtsReadyEventId(buf_idx));
}

__aicore__ inline void BtsMarkVReadyForMte3(uint32_t buf_idx)
{
    ac::SetFlag<ac::HardEvent::V_MTE3>(BtsPackReadyEventId(buf_idx));
}

__aicore__ inline void BtsWaitVReadyForMte3(uint32_t buf_idx)
{
    ac::WaitFlag<ac::HardEvent::V_MTE3>(BtsPackReadyEventId(buf_idx));
}

template <typename T>
__aicore__ inline __ubuf__ T *BtsUbPtr(uint32_t byte_offset = 0U)
{
    return reinterpret_cast<__ubuf__ T *>(byte_offset);
}
} // namespace

template <typename T>
__aicore__ inline void BindGm(GM_ADDR src_addr, GM_ADDR dst_addr,
                              ac::GlobalTensor<T> &src, ac::GlobalTensor<T> &dst)
{
    src.SetGlobalBuffer((__gm__ T *)src_addr);
    dst.SetGlobalBuffer((__gm__ T *)dst_addr);
}

template <typename T>
__aicore__ inline void CopyFlatWindow(GM_ADDR src_addr, GM_ADDR dst_addr)
{
    constexpr uint32_t kTotalBlocks = 4096U;
    constexpr uint32_t kBlocksPerSlice = 2048U;
    constexpr uint32_t kElemsInBlock = 32U / sizeof(T);
    constexpr uint32_t kScratchElems = kBlocksPerSlice * kElemsInBlock;

    ac::GlobalTensor<T> src;
    ac::GlobalTensor<T> dst;
    BindGm(src_addr, dst_addr, src, dst);
    ac::LocalTensor<T> window(ac::TPosition::VECCALC, 0, kScratchElems);

    const uint32_t core = static_cast<uint32_t>(ac::GetBlockIdx());
    const uint32_t cores = static_cast<uint32_t>(ac::GetBlockNum());
    const uint32_t quota = (kTotalBlocks + cores - 1U) / cores;
    uint32_t begin = core * quota;
    if (begin >= kTotalBlocks) {
        return;
    }

    uint32_t todo = kTotalBlocks - begin;
    if (todo > quota) {
        todo = quota;
    }
    uint32_t elem_pos = begin * kElemsInBlock;

    ac::DataCopyParams io;
    io.blockCount = 1;
    io.srcStride = 0;
    io.dstStride = 0;
    while (todo != 0U) {
        const uint32_t step = todo > kBlocksPerSlice ? kBlocksPerSlice : todo;
        io.blockLen = static_cast<uint16_t>(step);
        ac::DataCopy(window, src[elem_pos], io);
        ac::PipeBarrier<PIPE_MTE2>();
        ac::DataCopy(dst[elem_pos], window, io);
        ac::PipeBarrier<PIPE_MTE3>();
        elem_pos += step * kElemsInBlock;
        todo -= step;
    }
}

template <typename T, uint32_t H, uint32_t W, uint32_t C, uint32_t N, uint32_t SliceC>
__aicore__ inline void SplitChannelsR2(GM_ADDR src_addr, GM_ADDR dst_addr)
{
    constexpr uint32_t kOutH = H * 2U;
    constexpr uint32_t kOutW = W * 2U;
    constexpr uint32_t kSlices = C / SliceC;
    constexpr uint32_t kSliceBlocks = (SliceC * sizeof(T)) / 32U;
    constexpr uint32_t kStoreGap = ((C - SliceC) * sizeof(T)) / 32U;
    constexpr uint32_t kLineElems = kOutW * SliceC;

    ac::GlobalTensor<T> src;
    ac::GlobalTensor<T> dst;
    BindGm(src_addr, dst_addr, src, dst);
    ac::LocalTensor<T> line(ac::TPosition::VECCALC, 0, kLineElems);

    ac::DataCopyParams take;
    take.blockCount = 1;
    take.blockLen = static_cast<uint16_t>(kSliceBlocks);
    take.srcStride = 0;
    take.dstStride = 0;

    ac::DataCopyParams put;
    put.blockCount = kOutW;
    put.blockLen = static_cast<uint16_t>(kSliceBlocks);
    put.srcStride = 0;
    put.dstStride = static_cast<uint16_t>(kStoreGap);

    const uint32_t core = static_cast<uint32_t>(ac::GetBlockIdx());
    const uint32_t cores = static_cast<uint32_t>(ac::GetBlockNum());
    for (uint32_t job = core; job < N * kOutH * kSlices; job += cores) {
        const uint32_t slice = job % kSlices;
        const uint32_t row_job = job / kSlices;
        const uint32_t oh = row_job % kOutH;
        const uint32_t n = row_job / kOutH;
        const uint32_t c0 = slice * SliceC;
        const uint32_t ih = oh >> 1U;
        const uint32_t phase_h = oh & 1U;
        const uint32_t in0 = phase_h * 2U * N + n;
        const uint32_t in1 = in0 + N;

#pragma unroll
        for (uint32_t iw = 0; iw < W; ++iw) {
            const uint32_t even_src = ((in0 * H + ih) * W + iw) * C + c0;
            const uint32_t odd_src = ((in1 * H + ih) * W + iw) * C + c0;
            ac::DataCopy(line[(iw << 1U) * SliceC], src[even_src], take);
            ac::DataCopy(line[((iw << 1U) + 1U) * SliceC], src[odd_src], take);
        }
        ac::PipeBarrier<PIPE_MTE2>();

        const uint32_t out_base = ((n * kOutH + oh) * kOutW) * C + c0;
        ac::DataCopy(dst[out_base], line, put);
        ac::PipeBarrier<PIPE_MTE3>();
    }
}

template <typename T, uint32_t H, uint32_t W, uint32_t C, uint32_t N,
          uint32_t TileRows, uint32_t AlignBytes, bool FullLoadFence, bool FullShuffleFence>
__aicore__ inline void RowsPairR2(GM_ADDR src_addr, GM_ADDR dst_addr)
{
    constexpr uint32_t kOutH = H * 2U;
    constexpr uint32_t kOutW = W * 2U;
    constexpr uint32_t kRowTiles = (H + TileRows - 1U) / TileRows;
    constexpr uint32_t kCBlocks = (C * sizeof(T)) / 32U;
    constexpr uint32_t kInRowBlocks = (W * C * sizeof(T)) / 32U;
    constexpr uint32_t kOutRowBlocks = (kOutW * C * sizeof(T)) / 32U;
    constexpr uint32_t kInElems = TileRows * W * C;
    constexpr uint32_t kOutElems = TileRows * kOutW * C;
    constexpr uint32_t kInBytes =
        ((kInElems * sizeof(T) + AlignBytes - 1U) / AlignBytes) * AlignBytes;

    ac::GlobalTensor<T> src;
    ac::GlobalTensor<T> dst;
    BindGm(src_addr, dst_addr, src, dst);
    ac::LocalTensor<T> first(ac::TPosition::VECCALC, 0, kInElems);
    ac::LocalTensor<T> second(ac::TPosition::VECCALC, kInBytes, kInElems);
    ac::LocalTensor<T> merged(ac::TPosition::VECCALC, 2U * kInBytes, kOutElems);

    ac::DataCopyParams gm_in;
    gm_in.blockCount = 1;
    gm_in.blockLen = 0;
    gm_in.srcStride = 0;
    gm_in.dstStride = 0;

    ac::DataCopyParams ub_move;
    ub_move.blockLen = static_cast<uint16_t>(kCBlocks);
    ub_move.srcStride = 0;
    ub_move.dstStride = static_cast<uint16_t>(kCBlocks);

    ac::DataCopyParams gm_out;
    gm_out.blockLen = static_cast<uint16_t>(kOutRowBlocks);
    gm_out.srcStride = 0;
    gm_out.dstStride = static_cast<uint16_t>(kOutRowBlocks);

    const uint32_t core = static_cast<uint32_t>(ac::GetBlockIdx());
    const uint32_t cores = static_cast<uint32_t>(ac::GetBlockNum());
    for (uint32_t group = core; group < N * 2U * kRowTiles; group += cores) {
        const uint32_t tile = group % kRowTiles;
        const uint32_t hn = group / kRowTiles;
        const uint32_t phase_h = hn & 1U;
        const uint32_t n = hn >> 1U;
        const uint32_t ih0 = tile * TileRows;
        uint32_t rows = H - ih0;
        if (rows > TileRows) {
            rows = TileRows;
        }

        const uint32_t batch0 = phase_h * 2U * N + n;
        const uint32_t batch1 = batch0 + N;
        const uint32_t src0 = ((batch0 * H + ih0) * W) * C;
        const uint32_t src1 = ((batch1 * H + ih0) * W) * C;

        gm_in.blockLen = static_cast<uint16_t>(rows * kInRowBlocks);
        ac::DataCopy(first, src[src0], gm_in);
        ac::DataCopy(second, src[src1], gm_in);
        if constexpr (FullLoadFence) {
            ac::PipeBarrier<PIPE_ALL>();
        } else {
            ac::PipeBarrier<PIPE_MTE2>();
        }

        ub_move.blockCount = static_cast<uint16_t>(rows * W);
        ac::DataCopy(merged, first, ub_move);
        ac::DataCopy(merged[C], second, ub_move);
        if constexpr (FullShuffleFence) {
            ac::PipeBarrier<PIPE_ALL>();
        } else {
            ac::PipeBarrier<PIPE_MTE2>();
        }

        gm_out.blockCount = static_cast<uint16_t>(rows);
        const uint32_t out_pos = (n * kOutH + phase_h + ih0 * 2U) * kOutW * C;
        ac::DataCopy(dst[out_pos], merged, gm_out);
        ac::PipeBarrier<PIPE_MTE3>();
    }
}

template <typename T>
__aicore__ inline void Rows14FloatPath(GM_ADDR src_addr, GM_ADDR dst_addr)
{
    constexpr uint32_t H = 14U;
    constexpr uint32_t W = 14U;
    constexpr uint32_t C = 64U;
    constexpr uint32_t N = 4U;
    constexpr uint32_t TileRows = 7U;
    constexpr uint32_t kOutH = 28U;
    constexpr uint32_t kOutW = 28U;
    constexpr uint32_t kCBlocks = (C * sizeof(T)) / 32U;
    constexpr uint32_t kInRowBlocks = (W * C * sizeof(T)) / 32U;
    constexpr uint32_t kOutRowBlocks = (kOutW * C * sizeof(T)) / 32U;
    constexpr uint32_t kInElems = TileRows * W * C;
    constexpr uint32_t kOutElems = TileRows * kOutW * C;
    constexpr uint32_t kInBytes = ((kInElems * sizeof(T) + 511U) / 512U) * 512U;

    ac::GlobalTensor<T> src;
    ac::GlobalTensor<T> dst;
    BindGm(src_addr, dst_addr, src, dst);
    ac::LocalTensor<T> even_rows(ac::TPosition::VECCALC, 0, kInElems);
    ac::LocalTensor<T> odd_rows(ac::TPosition::VECCALC, kInBytes, kInElems);
    ac::LocalTensor<T> out_rows(ac::TPosition::VECCALC, 2U * kInBytes, kOutElems);

    ac::DataCopyParams load;
    load.blockCount = 1;
    load.blockLen = 0;
    load.srcStride = 0;
    load.dstStride = 0;

    ac::DataCopyParams reshuffle;
    reshuffle.blockLen = static_cast<uint16_t>(kCBlocks);
    reshuffle.srcStride = 0;
    reshuffle.dstStride = static_cast<uint16_t>(kCBlocks);

    ac::DataCopyParams store;
    store.blockLen = static_cast<uint16_t>(kOutRowBlocks);
    store.srcStride = 0;
    store.dstStride = static_cast<uint16_t>(kOutRowBlocks);

    const uint32_t core = static_cast<uint32_t>(ac::GetBlockIdx());
    const uint32_t cores = static_cast<uint32_t>(ac::GetBlockNum());
    for (uint32_t group = core; group < N * 4U; group += cores) {
        const uint32_t upper = group >> 1U;
        const uint32_t tile = group & 1U;
        const uint32_t phase_h = upper & 1U;
        const uint32_t n = upper >> 1U;
        const uint32_t ih = tile * TileRows;
        const uint32_t in0 = phase_h * 2U * N + n;
        const uint32_t in1 = in0 + N;

        load.blockLen = static_cast<uint16_t>(TileRows * kInRowBlocks);
        ac::DataCopy(even_rows, src[((in0 * H + ih) * W) * C], load);
        ac::DataCopy(odd_rows, src[((in1 * H + ih) * W) * C], load);
        ac::PipeBarrier<PIPE_ALL>();

        reshuffle.blockCount = static_cast<uint16_t>(TileRows * W);
        ac::DataCopy(out_rows, even_rows, reshuffle);
        ac::DataCopy(out_rows[C], odd_rows, reshuffle);
        ac::PipeBarrier<PIPE_MTE2>();

        store.blockCount = static_cast<uint16_t>(TileRows);
        const uint32_t out_pos = (n * kOutH + phase_h + ih * 2U) * kOutW * C;
        ac::DataCopy(dst[out_pos], out_rows, store);
        ac::PipeBarrier<PIPE_MTE3>();
    }
}

template <typename T, uint32_t H, uint32_t W, uint32_t C, uint32_t N,
          uint32_t OutRows, uint32_t AlignBytes>
__aicore__ inline void PhasePackedRowsR2(GM_ADDR src_addr, GM_ADDR dst_addr)
{
    constexpr uint32_t kOutH = H * 2U;
    constexpr uint32_t kOutW = W * 2U;
    constexpr uint32_t kTiles = (kOutH + OutRows - 1U) / OutRows;
    constexpr uint32_t kInRows = (OutRows + 1U) / 2U;
    constexpr uint32_t kCBlocks = (C * sizeof(T)) / 32U;
    constexpr uint32_t kInRowBlocks = (W * C * sizeof(T)) / 32U;
    constexpr uint32_t kOutRowBlocks = (kOutW * C * sizeof(T)) / 32U;
    constexpr uint32_t kInElems = kInRows * W * C;
    constexpr uint32_t kOutElems = OutRows * kOutW * C;
    constexpr uint32_t kInBytes =
        ((kInElems * sizeof(T) + AlignBytes - 1U) / AlignBytes) * AlignBytes;

    ac::GlobalTensor<T> src;
    ac::GlobalTensor<T> dst;
    BindGm(src_addr, dst_addr, src, dst);
    ac::LocalTensor<T> q0(ac::TPosition::VECCALC, 0, kInElems);
    ac::LocalTensor<T> q1(ac::TPosition::VECCALC, kInBytes, kInElems);
    ac::LocalTensor<T> q2(ac::TPosition::VECCALC, 2U * kInBytes, kInElems);
    ac::LocalTensor<T> q3(ac::TPosition::VECCALC, 3U * kInBytes, kInElems);
    ac::LocalTensor<T> out(ac::TPosition::VECCALC, 4U * kInBytes, kOutElems);

    ac::DataCopyParams load;
    load.blockCount = 1;
    load.blockLen = 0;
    load.srcStride = 0;
    load.dstStride = 0;

    ac::DataCopyParams stripe;
    stripe.blockLen = static_cast<uint16_t>(kCBlocks);
    stripe.srcStride = static_cast<uint16_t>((W - 1U) * kCBlocks);
    stripe.dstStride = static_cast<uint16_t>(2U * kOutRowBlocks - kCBlocks);

    ac::DataCopyParams store;
    store.blockCount = 1;
    store.blockLen = 0;
    store.srcStride = 0;
    store.dstStride = 0;

    const uint32_t core = static_cast<uint32_t>(ac::GetBlockIdx());
    const uint32_t cores = static_cast<uint32_t>(ac::GetBlockNum());
    for (uint32_t job = core; job < N * kTiles; job += cores) {
        const uint32_t tile = job % kTiles;
        const uint32_t n = job / kTiles;
        const uint32_t oh0 = tile * OutRows;
        uint32_t rows = kOutH - oh0;
        if (rows > OutRows) {
            rows = OutRows;
        }

        const uint32_t ih0 = oh0 >> 1U;
        const uint32_t rows_even = (rows + 1U) >> 1U;
        const uint32_t rows_odd = rows >> 1U;
        const uint32_t n00 = n;
        const uint32_t n01 = n00 + N;
        const uint32_t n10 = n00 + 2U * N;
        const uint32_t n11 = n10 + N;

        load.blockLen = static_cast<uint16_t>(rows_even * kInRowBlocks);
        ac::DataCopy(q0, src[((n00 * H + ih0) * W) * C], load);
        ac::DataCopy(q1, src[((n01 * H + ih0) * W) * C], load);
        if (rows_odd != 0U) {
            load.blockLen = static_cast<uint16_t>(rows_odd * kInRowBlocks);
            ac::DataCopy(q2, src[((n10 * H + ih0) * W) * C], load);
            ac::DataCopy(q3, src[((n11 * H + ih0) * W) * C], load);
        }
        ac::PipeBarrier<PIPE_ALL>();

        stripe.blockCount = static_cast<uint16_t>(rows_even);
        for (uint32_t iw = 0; iw < W; ++iw) {
            const uint32_t in_off = iw * C;
            const uint32_t out_off = (iw * 2U) * C;
            ac::DataCopy(out[out_off], q0[in_off], stripe);
            ac::DataCopy(out[out_off + C], q1[in_off], stripe);
        }
        if (rows_odd != 0U) {
            stripe.blockCount = static_cast<uint16_t>(rows_odd);
            for (uint32_t iw = 0; iw < W; ++iw) {
                const uint32_t in_off = iw * C;
                const uint32_t out_off = kOutW * C + (iw * 2U) * C;
                ac::DataCopy(out[out_off], q2[in_off], stripe);
                ac::DataCopy(out[out_off + C], q3[in_off], stripe);
            }
        }
        ac::PipeBarrier<PIPE_ALL>();

        store.blockLen = static_cast<uint16_t>(rows * kOutRowBlocks);
        ac::DataCopy(dst[(n * kOutH + oh0) * kOutW * C], out, store);
        ac::PipeBarrier<PIPE_MTE3>();
    }
}

template <typename T, uint32_t H, uint32_t W, uint32_t C, uint32_t N, uint32_t FixedBlocks>
__aicore__ inline void DirectRowsR2(GM_ADDR src_addr, GM_ADDR dst_addr)
{
    constexpr uint32_t kOutH = H * 2U;
    constexpr uint32_t kOutW = W * 2U;
    constexpr uint32_t kRowElems = kOutW * C;
    constexpr uint32_t kBufElems = ((kRowElems + 1023U) / 1024U) * 1024U;
    constexpr uint32_t kCBytes = C * sizeof(T);
    constexpr uint32_t kCGap = kCBytes >> 5U;

    ac::GlobalTensor<T> src;
    ac::GlobalTensor<T> dst;
    BindGm(src_addr, dst_addr, src, dst);
    ac::LocalMemAllocator<ac::Hardware::UB> ub;
    ac::LocalTensor<T> ping = ub.Alloc<ac::TPosition::VECCALC, T>(kBufElems);
    ac::LocalTensor<T> pong = ub.Alloc<ac::TPosition::VECCALC, T>(kBufElems);

    ac::DataCopyExtParams row_in{static_cast<uint16_t>(W), kCBytes, 0, kCGap, 0};
    ac::DataCopyPadExtParams<T> no_pad{false, 0, 0, 0};

    constexpr uint32_t kTotalRows = N * kOutH;
    constexpr uint32_t kRowsPerCore = (kTotalRows + FixedBlocks - 1U) / FixedBlocks;
    const uint32_t core = static_cast<uint32_t>(ac::GetBlockIdx());
    uint32_t first_row = core * kRowsPerCore;
    if (first_row >= kTotalRows) {
        return;
    }
    uint32_t rows = kTotalRows - first_row;
    if (rows > kRowsPerCore) {
        rows = kRowsPerCore;
    }

    uint32_t n = first_row / kOutH;
    uint32_t oh = first_row - n * kOutH;
    const uint32_t ih_start = oh >> 1U;
    uint32_t phase_h = oh & 1U;
    uint32_t in0 = phase_h * 2U * N + n;
    uint32_t in1 = in0 + N;
    uint64_t src0 = (static_cast<uint64_t>(in0) * H + ih_start) * W * C;
    uint64_t src1 = (static_cast<uint64_t>(in1) * H + ih_start) * W * C;
    ac::DataCopyPad(ping, src[src0], row_in, no_pad);
    ac::DataCopyPad(ping[C], src[src1], row_in, no_pad);
    ac::SetFlag<ac::HardEvent::MTE2_MTE3>(static_cast<event_t>(0));

    uint32_t active = 0U;
    for (uint32_t r = 0; r < rows; ++r) {
        const event_t cur = active == 0U ? static_cast<event_t>(0) : static_cast<event_t>(1);
        const event_t nxt = active == 0U ? static_cast<event_t>(1) : static_cast<event_t>(0);
        ac::LocalTensor<T> current = active == 0U ? ping : pong;
        ac::LocalTensor<T> spare = active == 0U ? pong : ping;

        ac::WaitFlag<ac::HardEvent::MTE2_MTE3>(cur);

        uint32_t next_n = n;
        uint32_t next_oh = oh + 1U;
        if (next_oh == kOutH) {
            next_oh = 0U;
            ++next_n;
        }
        if (r + 1U < rows) {
            const uint32_t next_ih = next_oh >> 1U;
            const uint32_t next_phase = next_oh & 1U;
            const uint32_t next_in0 = next_phase * 2U * N + next_n;
            const uint32_t next_in1 = next_in0 + N;
            const uint64_t next_src0 = (static_cast<uint64_t>(next_in0) * H + next_ih) * W * C;
            const uint64_t next_src1 = (static_cast<uint64_t>(next_in1) * H + next_ih) * W * C;
            ac::DataCopyPad(spare, src[next_src0], row_in, no_pad);
            ac::DataCopyPad(spare[C], src[next_src1], row_in, no_pad);
            ac::SetFlag<ac::HardEvent::MTE2_MTE3>(nxt);
        }

        const uint64_t out_pos = static_cast<uint64_t>(first_row + r) * kRowElems;
        ac::DataCopy(dst[out_pos], current, kRowElems);
        ac::SetFlag<ac::HardEvent::MTE3_MTE2>(cur);
        ac::WaitFlag<ac::HardEvent::MTE3_MTE2>(cur);

        n = next_n;
        oh = next_oh;
        active ^= 1U;
    }
}

template <typename T>
__aicore__ inline void GatherSmallCropR2(GM_ADDR src_addr, GM_ADDR dst_addr)
{
    constexpr uint32_t H = 10U;
    constexpr uint32_t W = 15U;
    constexpr uint32_t C = 5U;
    constexpr uint32_t OutH = 17U;
    constexpr uint32_t OutW = 26U;
    constexpr uint32_t EvenCols = 13U;
    constexpr uint32_t OddCols = 13U;
    constexpr uint32_t CBytes = C * sizeof(T);
    constexpr uint32_t GroupBytes = ((EvenCols * CBytes + 31U) / 32U) * 32U;
    constexpr uint32_t GroupElems = GroupBytes / sizeof(T);
    constexpr uint32_t InputElems = 2U * GroupElems;
    constexpr uint32_t OutputElems = OutW * C;
    constexpr uint32_t PairElems = 2U * C;
    constexpr uint32_t PatternPairs = 4U;
    constexpr uint32_t PatternElems = PatternPairs * PairElems;
    constexpr uint32_t FullPatterns = OutputElems / PatternElems;
    constexpr uint32_t TailElems =
        ((OutputElems - FullPatterns * PatternElems + 7U) / 8U) * 8U;
    constexpr uint32_t IndexElems = FullPatterns * PatternElems + TailElems;
    constexpr event_t ScalarReady = static_cast<event_t>(6);

    ac::GlobalTensor<T> src;
    ac::GlobalTensor<T> dst;
    BindGm(src_addr, dst_addr, src, dst);
    ac::LocalTensor<T> input(ac::TPosition::VECCALC, 0, InputElems);
    ac::LocalTensor<T> output(ac::TPosition::VECCALC, 1024, OutputElems);
    ac::LocalTensor<int32_t> index_s32(ac::TPosition::VECCALC, 2048, IndexElems);
    ac::LocalTensor<uint32_t> index_u32(ac::TPosition::VECCALC, 2048, OutputElems);

    ac::DataCopyPadExtParams<T> no_pad{false, 0, 0, 0};
    ac::DataCopyExtParams load0{1, EvenCols * CBytes, 0, 0, 0};
    ac::DataCopyExtParams load1{1, OddCols * CBytes, 0, 0, 0};
    ac::DataCopyExtParams store{1, OutW * CBytes, 0, 0, 0};

    for (uint32_t pair = 0; pair < PatternPairs; ++pair) {
        const uint32_t elem = pair * PairElems;
        const uint32_t bytes = pair * CBytes;
        for (uint32_t c = 0; c < C; ++c) {
            index_s32.SetValue(elem + c, static_cast<int32_t>(bytes + c * sizeof(T)));
            index_s32.SetValue(elem + C + c,
                static_cast<int32_t>(GroupElems * sizeof(T) + bytes + c * sizeof(T)));
        }
    }
    ac::SetFlag<ac::HardEvent::S_V>(ScalarReady);
    ac::WaitFlag<ac::HardEvent::S_V>(ScalarReady);
    for (uint32_t part = 1; part <= FullPatterns; ++part) {
        const uint32_t count = part == FullPatterns ? TailElems : PatternElems;
        ac::Adds(index_s32[part * PatternElems], index_s32[(part - 1U) * PatternElems],
                 static_cast<int32_t>(PatternPairs * CBytes), count);
    }
    ac::PipeBarrier<PIPE_V>();

    const uint32_t core = static_cast<uint32_t>(ac::GetBlockIdx());
    const uint32_t cores = static_cast<uint32_t>(ac::GetBlockNum());
    for (uint32_t oh = core; oh < OutH; oh += cores) {
        const uint32_t padded_h = oh + 2U;
        const uint32_t ih = padded_h >> 1U;
        const uint32_t phase_h = padded_h & 1U;
        const uint32_t first_n = phase_h * 2U + 1U;
        const uint32_t second_n = phase_h * 2U;
        const uint32_t first_src = ((first_n * H + ih) * W + 1U) * C;
        const uint32_t second_src = ((second_n * H + ih) * W + 2U) * C;

        ac::DataCopyPad(input, src[first_src], load0, no_pad);
        ac::DataCopyPad(input[GroupElems], src[second_src], load1, no_pad);
        ac::PipeBarrier<PIPE_ALL>();
        ac::Gather(output, input, index_u32, 0, OutputElems);
        ac::PipeBarrier<PIPE_ALL>();
        ac::DataCopyPad(dst[oh * OutW * C], output, store);
        ac::PipeBarrier<PIPE_MTE3>();
    }
}

template <typename T>
__aicore__ inline void DenseCropC65HalfTiled(GM_ADDR src_addr, GM_ADDR dst_addr)
{
    if constexpr (sizeof(T) != 2U) {
        return;
    } else {
        constexpr uint64_t kH = 128U;
        constexpr uint64_t kW = 128U;
        constexpr uint64_t kC = 65U;
        constexpr uint64_t kOutH = 254U;
        constexpr uint64_t kOutW = 254U;
        constexpr uint64_t kLaneElems = kW * kC;
        constexpr uint64_t kRowElems = kOutW * kC;
        constexpr uint64_t kTileW = 128U;
        constexpr uint64_t kTileLanePixels = (kTileW + 1U) >> 1U;
        constexpr uint64_t kLaneTileElems = kTileLanePixels * kC;
        constexpr uint64_t kTilesPerRow = (kOutW + kTileW - 1U) / kTileW;
        constexpr uint64_t kTotalTiles = kOutH * kTilesPerRow;
        constexpr uint64_t kPairs = 4U;
        constexpr uint32_t kGroupOffsets = static_cast<uint32_t>(kPairs * (kC << 1U));
        constexpr uint64_t kOffsetGroups = (kTileLanePixels + kPairs - 1U) / kPairs;
        constexpr uint32_t kSrcBytes = static_cast<uint32_t>(2U * kLaneTileElems * sizeof(T));
        constexpr uint32_t kDstBytes = static_cast<uint32_t>(kTileW * kC * sizeof(T));
        constexpr uint32_t kOffsetBytes = static_cast<uint32_t>(kTileW * kC * sizeof(uint32_t));

        ac::GlobalTensor<T> src;
        ac::GlobalTensor<T> dst;
        BindGm(src_addr, dst_addr, src, dst);

        ac::TPipe pipe;
        ac::TQue<ac::TPosition::VECIN, 2> src_queue;
        ac::TQue<ac::TPosition::VECOUT, 2> dst_queue;
        ac::TBuf<ac::TPosition::VECCALC> offset_buf;
        pipe.InitBuffer(src_queue, 2, kSrcBytes);
        pipe.InitBuffer(dst_queue, 2, kDstBytes);
        pipe.InitBuffer(offset_buf, kOffsetBytes);

        ac::LocalTensor<uint32_t> offset = offset_buf.Get<uint32_t>();
        ac::LocalTensor<int32_t> offset_i32 = offset_buf.Get<int32_t>();
        ac::DataCopyPadExtParams<T> no_pad{false, 0, 0, 0};

        for (uint64_t pair = 0U; pair < kPairs; ++pair) {
            const uint64_t even_base = (pair << 1U) * kC;
            const uint64_t odd_base = ((pair << 1U) + 1U) * kC;
            for (uint64_t c = 0U; c < kC; ++c) {
                offset_i32.SetValue(
                    static_cast<uint32_t>(even_base + c),
                    static_cast<int32_t>((pair * kC + c) * sizeof(T)));
                offset_i32.SetValue(
                    static_cast<uint32_t>(odd_base + c),
                    static_cast<int32_t>((kLaneTileElems + pair * kC + c) * sizeof(T)));
            }
        }
        for (uint64_t group = 1U; group < kOffsetGroups; ++group) {
            ac::Adds(
                offset_i32[static_cast<uint32_t>(group * kGroupOffsets)],
                offset_i32,
                static_cast<int32_t>(group * kPairs * kC * sizeof(T)),
                kGroupOffsets);
        }

        const uint32_t core = static_cast<uint32_t>(ac::GetBlockIdx());
        const uint32_t cores = static_cast<uint32_t>(ac::GetBlockNum());
        for (uint64_t work = core; work < kTotalTiles; work += cores) {
            const uint64_t out_h = work >> 1U;
            const uint64_t tile = work & 1U;
            const uint64_t start_w = tile * kTileW;
            const uint64_t tile_w = tile == 0U ? kTileW : kOutW - kTileW;
            const uint64_t even_pixels = (tile_w + 1U) >> 1U;
            const uint64_t odd_pixels = tile_w >> 1U;

            const uint64_t padded_h = out_h + 1U;
            const uint64_t in_h = padded_h >> 1U;
            const uint64_t phase_h = padded_h & 1U;
            const uint64_t row_offset = in_h * kLaneElems;
            const uint64_t batch_base = phase_h << 1U;
            const uint64_t in_w = start_w >> 1U;
            const uint64_t even_src =
                (batch_base + 1U) * kH * kLaneElems + row_offset + in_w * kC;
            const uint64_t odd_src =
                batch_base * kH * kLaneElems + row_offset + (in_w + 1U) * kC;

            ac::LocalTensor<T> packed_src = src_queue.template AllocTensor<T>();
            ac::DataCopyExtParams even_load{
                1U, static_cast<uint32_t>(even_pixels * kC * sizeof(T)), 0U, 0U, 0U};
            ac::DataCopyExtParams odd_load{
                1U, static_cast<uint32_t>(odd_pixels * kC * sizeof(T)), 0U, 0U, 0U};
            ac::DataCopyPad(packed_src, src[even_src], even_load, no_pad);
            ac::DataCopyPad(packed_src[static_cast<uint32_t>(kLaneTileElems)],
                            src[odd_src], odd_load, no_pad);
            src_queue.template EnQue<T>(packed_src);

            packed_src = src_queue.template DeQue<T>();
            ac::LocalTensor<T> packed_dst = dst_queue.template AllocTensor<T>();
            ac::Gather(packed_dst, packed_src, offset, 0U, static_cast<uint32_t>(tile_w * kC));
            src_queue.template FreeTensor<T>(packed_src);
            dst_queue.template EnQue<T>(packed_dst);

            packed_dst = dst_queue.template DeQue<T>();
            ac::DataCopyExtParams store{
                1U, static_cast<uint32_t>(tile_w * kC * sizeof(T)), 0U, 0U, 0U};
            ac::DataCopyPad(dst[out_h * kRowElems + start_w * kC], packed_dst, store);
            dst_queue.template FreeTensor<T>(packed_dst);
        }
    }
}

constexpr uint32_t kDenseCropC65Height = 128U;
constexpr uint32_t kDenseCropC65Width = 128U;
constexpr uint32_t kDenseCropC65Depth = 65U;
constexpr uint32_t kDenseCropC65OutHeight = 254U;
constexpr uint32_t kDenseCropC65OutWidth = 254U;
constexpr uint32_t kDenseCropC65HalfPx = kDenseCropC65OutWidth / 2U;
constexpr uint32_t kDenseCropC65LanePx = (kDenseCropC65HalfPx + 1U) / 2U;
constexpr uint32_t kDenseCropC65LaneBaseElems = kDenseCropC65LanePx * kDenseCropC65Depth;
constexpr uint32_t kDenseCropC65RampElems = 520U;
constexpr uint32_t kDenseCropC65NRamp = 16U;
constexpr uint32_t kDenseCropC65TailElems = 456U;
constexpr uint32_t kDenseCropC65OutElems = kDenseCropC65HalfPx * kDenseCropC65Depth;
constexpr uint32_t kDenseCropC65GatherInputElems = 2U * kDenseCropC65LaneBaseElems;
constexpr uint32_t kDenseCropC65SlotElems = 16576U;
constexpr uint32_t kDenseCropC65IdxBaseBytes =
    2U * kDenseCropC65SlotElems * sizeof(uint16_t);
constexpr uint32_t kDenseCropC65BytesPerBuf =
    kDenseCropC65SlotElems * sizeof(uint16_t);
constexpr uint32_t kDenseCropC65TotalTasks = kDenseCropC65OutHeight * 2U;
constexpr uint32_t kDenseCropC65ActiveCores = 40U;
constexpr uint32_t kDenseCropC65InputBatchElems =
    kDenseCropC65Height * kDenseCropC65Width * kDenseCropC65Depth;
constexpr uint32_t kDenseCropC65InputRowElems = kDenseCropC65Width * kDenseCropC65Depth;
constexpr uint32_t kDenseCropC65HalfInputElems = 64U * kDenseCropC65Depth;

struct DenseCropC65TaskPlan {
    uint32_t input_offset_a;
    uint32_t input_offset_b;
};

__aicore__ inline BtsTaskRange DenseCropC65GetTaskRange()
{
    return BtsGetStaticTaskRange(kDenseCropC65TotalTasks, kDenseCropC65ActiveCores);
}

template <typename T>
__aicore__ inline __ubuf__ T *DenseCropC65SlotAddr(uint32_t buf_idx)
{
    return BtsUbPtr<T>(buf_idx * kDenseCropC65BytesPerBuf);
}

__aicore__ inline __ubuf__ int32_t *DenseCropC65IdxAddr()
{
    return BtsUbPtr<int32_t>(kDenseCropC65IdxBaseBytes);
}

__aicore__ inline DenseCropC65TaskPlan DenseCropC65GetTaskPlan(uint32_t task_id)
{
    const uint32_t phase = task_id & 3U;
    const uint32_t half = task_id & 1U;
    const uint32_t batch_a = 3U - phase;
    const uint32_t spatial_offset =
        ((task_id + 2U) >> 2U) * kDenseCropC65InputRowElems +
        half * kDenseCropC65HalfInputElems;
    return {
        batch_a * kDenseCropC65InputBatchElems + spatial_offset,
        (batch_a ^ 1U) * kDenseCropC65InputBatchElems + spatial_offset +
            (1U - half) * kDenseCropC65Depth};
}

template <typename T>
__aicore__ inline void DenseCropC65BuildIndex()
{
    __ubuf__ int32_t *idx = DenseCropC65IdxAddr();
    constexpr uint32_t kIdxBlockElems = kBtsDataBlockBytes / sizeof(int32_t);
    constexpr uint32_t kVecElems = 64U;
    for (uint32_t d = 0U; d < kIdxBlockElems; ++d) {
        idx[d] = static_cast<int32_t>(d * sizeof(T));
    }
    ac::SetFlag<ac::HardEvent::S_V>(EVENT_ID6);
    ac::WaitFlag<ac::HardEvent::S_V>(EVENT_ID6);
    for (uint32_t offset = kIdxBlockElems; offset < kVecElems; offset += kIdxBlockElems) {
        ac::AddsImpl<int32_t>(
            idx + offset, idx + offset - kIdxBlockElems,
            static_cast<int32_t>(kIdxBlockElems * sizeof(T)),
            static_cast<int32_t>(kIdxBlockElems));
        ac::PipeBarrier<PIPE_V>();
    }

    constexpr int32_t kBlockAdds[8] = {8318, 126, 8444, 252, 8570, 378, 8696, 8824};
    for (uint32_t block = 1U; block <= 8U; ++block) {
        const uint32_t count =
            block == 8U ? kDenseCropC65RampElems - 8U * kVecElems : kVecElems;
        ac::AddsImpl<int32_t>(
            idx + block * kVecElems, idx, kBlockAdds[block - 1U],
            static_cast<int32_t>(count));
    }
    ac::PipeBarrier<PIPE_V>();
    ac::SetFlag<ac::HardEvent::V_S>(EVENT_ID7);
    ac::WaitFlag<ac::HardEvent::V_S>(EVENT_ID7);
    for (uint32_t block = 1U; block < 8U; ++block) {
        const uint32_t block_base = block * kVecElems;
        const uint32_t patch_base = (block & 1U) != 0U
            ? ((block + 1U) >> 1U) * 126U + 2U
            : 2U * kDenseCropC65LaneBaseElems + (block >> 1U) * 126U;
        for (uint32_t i = 0U; i < block; ++i) {
            idx[block_base + i] = static_cast<int32_t>(patch_base + 2U * i);
        }
    }
    ac::SetFlag<ac::HardEvent::S_V>(EVENT_ID6);
    ac::WaitFlag<ac::HardEvent::S_V>(EVENT_ID6);

    for (uint32_t block = 1U; block < kDenseCropC65NRamp; ++block) {
        const uint32_t count =
            block == kDenseCropC65NRamp - 1U ? kDenseCropC65TailElems :
            kDenseCropC65RampElems;
        ac::AddsImpl<int32_t>(
            idx + block * kDenseCropC65RampElems, idx,
            static_cast<int32_t>(block * kDenseCropC65RampElems),
            static_cast<int32_t>(count));
    }
    ac::PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void DenseCropC65CopyIn(__gm__ T *src, uint32_t task_id, uint32_t buf_idx)
{
    BtsWaitBufFree(buf_idx);
    __ubuf__ T *slot_addr = DenseCropC65SlotAddr<T>(buf_idx);
    const DenseCropC65TaskPlan plan = DenseCropC65GetTaskPlan(task_id);
    constexpr uint16_t kLaneBlocks =
        kDenseCropC65LaneBaseElems * sizeof(uint16_t) / kBtsDataBlockBytes;
    copy_gm_to_ubuf(
        (__ubuf__ void *)slot_addr, (__gm__ void *)(src + plan.input_offset_a),
        0, 1, kLaneBlocks, 0, 0);
    copy_gm_to_ubuf(
        (__ubuf__ void *)(slot_addr + kDenseCropC65LaneBaseElems),
        (__gm__ void *)(src + plan.input_offset_b), 0, 1, kLaneBlocks, 0, 0);
    BtsMarkMte2ReadyForV(buf_idx);
}

template <typename T>
__aicore__ inline void DenseCropC65Gather(uint32_t buf_idx)
{
    __ubuf__ T *slot = DenseCropC65SlotAddr<T>(buf_idx);
    __ubuf__ uint32_t *idx = reinterpret_cast<__ubuf__ uint32_t *>(DenseCropC65IdxAddr());
    ac::GatherImpl(slot + kDenseCropC65GatherInputElems, slot, idx, kDenseCropC65SlotElems, 0,
                   static_cast<uint64_t>(128), 64, 8);
    ac::GatherImpl(
        slot + kDenseCropC65GatherInputElems + 8192U, slot, idx + 8192U,
        kDenseCropC65SlotElems, 0, static_cast<uint64_t>(63), 1, 8);
    BtsMarkVReadyForMte3(buf_idx);
}

template <typename T>
__aicore__ inline void DenseCropC65CopyOut(__gm__ T *dst, uint32_t task_id, uint32_t buf_idx)
{
    __ubuf__ T *slot_addr = DenseCropC65SlotAddr<T>(buf_idx);
    constexpr uint32_t kOutBytes = kDenseCropC65OutElems * sizeof(uint16_t);
    copy_ubuf_to_gm_align_b16(
        (__gm__ void *)(dst + task_id * kDenseCropC65OutElems),
        (__ubuf__ void *)(slot_addr + kDenseCropC65GatherInputElems), 0, 1, kOutBytes,
        0, 0, 0, 0);
}

template <typename T>
__aicore__ inline void DenseCropC65HalfStatic(GM_ADDR src_addr, GM_ADDR dst_addr)
{
    if constexpr (sizeof(T) != sizeof(uint16_t)) {
        return;
    } else {
        __gm__ T *src = reinterpret_cast<__gm__ T *>(src_addr);
        __gm__ T *dst = reinterpret_cast<__gm__ T *>(dst_addr);
        const BtsTaskRange range = DenseCropC65GetTaskRange();
        if (range.length == 0U) {
            return;
        }

        BtsMarkBufFree(0U);
        BtsMarkBufFree(1U);
        DenseCropC65CopyIn(src, range.offset, 0U);
        DenseCropC65BuildIndex<T>();
        for (uint32_t i = 0U; i < range.length; ++i) {
            const uint32_t buf_idx = i & 1U;
            BtsWaitMte2ReadyForV(buf_idx);
            DenseCropC65Gather<T>(buf_idx);
            const uint32_t next = i + 1U;
            if (next < range.length) {
                DenseCropC65CopyIn(src, range.offset + next, next & 1U);
            }
            BtsWaitVReadyForMte3(buf_idx);
            DenseCropC65CopyOut(dst, range.offset + i, buf_idx);
            BtsMarkBufFree(buf_idx);
        }
        BtsWaitBufFree(0U);
        BtsWaitBufFree(1U);
    }
}

template <typename T>
__aicore__ inline void WideRowsHalf(GM_ADDR src_addr, GM_ADDR dst_addr)
{
    constexpr uint32_t kH = 10U;
    constexpr uint32_t kW = 512U;
    constexpr uint32_t kD = 256U;
    constexpr uint32_t kOutB = 1U;
    constexpr uint32_t kOutW = 1024U;
    constexpr uint32_t kActiveCores = 40U;
    constexpr uint32_t kTileElemsWideRow = 16384U;
    constexpr uint32_t kMaxCols = kTileElemsWideRow / kD;
    constexpr event_t kEvent0 = static_cast<event_t>(0);
    constexpr event_t kEvent1 = static_cast<event_t>(1);

    ac::GlobalTensor<T> x_gm_direct;
    ac::GlobalTensor<T> y_gm_direct;
    x_gm_direct.SetGlobalBuffer((__gm__ T *)src_addr);
    y_gm_direct.SetGlobalBuffer((__gm__ T *)dst_addr);
    ac::LocalMemAllocator<ac::Hardware::UB> ub;
    ac::LocalTensor<T> buffer0 =
        ub.Alloc<ac::TPosition::VECCALC, T>(kTileElemsWideRow);
    ac::LocalTensor<T> buffer1 =
        ub.Alloc<ac::TPosition::VECCALC, T>(kTileElemsWideRow);

    constexpr uint32_t kInputBatch = kOutB * 4U;
    constexpr uint32_t kTotalRows = kInputBatch * kH;
    const BtsTaskRange range = BtsGetStaticTaskRange(kTotalRows, kActiveCores);
    if (range.length == 0U || range.offset >= kTotalRows) {
        return;
    }

    for (uint32_t r = 0; r < range.length; ++r) {
        uint32_t row = range.offset + r;
        uint32_t inB = row / kH;
        uint32_t inH = row - inB * kH;
        uint32_t blockIdxInBatch = inB / kOutB;
        uint32_t blockH = blockIdxInBatch >> 1U;
        uint32_t blockW = blockIdxInBatch & 1U;
        uint32_t oh = (inH << 1U) + blockH;
        uint64_t srcBase = static_cast<uint64_t>(row) * kW * kD;
        uint64_t dstBase = (static_cast<uint64_t>(oh) * kOutW + blockW) * kD;

        uint32_t doneCols = 0U;
        uint32_t curCols = kW > kMaxCols ? kMaxCols : kW;
        uint32_t curCount = curCols * kD;
        ac::DataCopy(buffer0, x_gm_direct[srcBase], curCount);
        ac::SetFlag<ac::HardEvent::MTE2_MTE3>(kEvent0);
        uint32_t bufIdx = 0U;
        while (doneCols < kW) {
            event_t curEvent = bufIdx == 0U ? kEvent0 : kEvent1;
            event_t nextEvent = bufIdx == 0U ? kEvent1 : kEvent0;
            ac::LocalTensor<T> curBuffer = bufIdx == 0U ? buffer0 : buffer1;
            ac::LocalTensor<T> nextBuffer = bufIdx == 0U ? buffer1 : buffer0;
            ac::WaitFlag<ac::HardEvent::MTE2_MTE3>(curEvent);

            uint32_t nextDone = doneCols + curCols;
            uint32_t nextCols = 0U;
            if (nextDone < kW) {
                nextCols = kW - nextDone;
                if (nextCols > kMaxCols) {
                    nextCols = kMaxCols;
                }
                uint32_t nextCount = nextCols * kD;
                uint64_t nextSrc = srcBase + static_cast<uint64_t>(nextDone) * kD;
                ac::DataCopy(nextBuffer, x_gm_direct[nextSrc], nextCount);
                ac::SetFlag<ac::HardEvent::MTE2_MTE3>(nextEvent);
            }

            uint64_t dst = dstBase + static_cast<uint64_t>(doneCols) * 2U * kD;
            ac::DataCopyExtParams outParams{
                static_cast<uint16_t>(curCols),
                static_cast<uint32_t>(kD * sizeof(T)),
                0U,
                static_cast<uint32_t>(kD * sizeof(T)),
                0U};
            ac::DataCopyPad(y_gm_direct[dst], curBuffer, outParams);
            ac::SetFlag<ac::HardEvent::MTE3_MTE2>(curEvent);
            ac::WaitFlag<ac::HardEvent::MTE3_MTE2>(curEvent);

            doneCols = nextDone;
            curCols = nextCols;
            bufIdx ^= 1U;
        }
    }
}

template <typename T>
__aicore__ inline void Block4PhaseRowsHalf(GM_ADDR src_addr, GM_ADDR dst_addr)
{
    if constexpr (sizeof(T) != 2U) {
        return;
    } else {
        constexpr uint32_t kPhases = 4U;
        constexpr uint32_t kBlockBytes = 32U;
        constexpr uint32_t kH = 10U;
        constexpr uint32_t kW = 512U;
        constexpr uint32_t kC = 64U;
        constexpr uint32_t kOutH = 40U;
        constexpr uint32_t kOutW = 1535U;
        constexpr uint32_t kCropLeft = 513U;
        constexpr uint32_t kTileCols = 512U;
        constexpr uint32_t kTilesPerRow = 3U;
        constexpr uint32_t kTotalJobs = kOutH * kTilesPerRow;
        constexpr uint32_t kCBlocks = (kC * sizeof(T)) / kBlockBytes;
        constexpr uint32_t kPhaseTileCols = (kTileCols + kPhases - 1U) >> 2U;
        constexpr uint32_t kPhaseTileElems = kPhaseTileCols * kC;
        constexpr uint32_t kOutputTileElems = kTileCols * kC;
        constexpr uint32_t kOutputTileBytes = kOutputTileElems * sizeof(T);
        constexpr uint32_t kPhaseTileBytes = kPhaseTileElems * sizeof(T);

        ac::GlobalTensor<T> src;
        ac::GlobalTensor<T> dst;
        BindGm(src_addr, dst_addr, src, dst);

        ac::LocalTensor<T> output(ac::TPosition::VECCALC, 0, kOutputTileElems);
        ac::LocalTensor<T> input01(ac::TPosition::VECCALC, kOutputTileBytes, kPhaseTileElems * 2U);
        ac::LocalTensor<T> input23(
            ac::TPosition::VECCALC, kOutputTileBytes + kPhaseTileBytes * 2U,
            kPhaseTileElems * 2U);

        ac::DataCopyParams load{1U, 0U, 0U, 0U};
        ac::DataCopyParams scatter{
            0U, static_cast<uint16_t>(kCBlocks), 0U,
            static_cast<uint16_t>((kPhases - 1U) * kCBlocks)};
        ac::DataCopyParams store{1U, 0U, 0U, 0U};

        const uint32_t core = static_cast<uint32_t>(ac::GetBlockIdx());
        const uint32_t cores = static_cast<uint32_t>(ac::GetBlockNum());
        for (uint32_t task = core; task < kTotalJobs; task += cores) {
            const uint32_t out_row = task / kTilesPerRow;
            const uint32_t tile_id = task - out_row * kTilesPerRow;
            const uint32_t tile_ow = tile_id * kTileCols;
            uint32_t cols = kOutW - tile_ow;
            if (cols > kTileCols) {
                cols = kTileCols;
            }

            const uint32_t in_h = out_row >> 2U;
            const uint32_t phase_h = out_row & 3U;
            const uint32_t input_base = phase_h << 2U;
            const uint32_t crop_phase = kCropLeft & 3U;
            for (uint32_t phase = 0U; phase < kPhases; ++phase) {
                const uint32_t first_col =
                    phase >= crop_phase ? phase - crop_phase : phase + kPhases - crop_phase;
                if (first_col >= cols) {
                    continue;
                }
                const uint32_t count = ((cols - 1U - first_col) >> 2U) + 1U;
                const uint32_t src_col = (tile_ow + first_col + kCropLeft) >> 2U;
                const uint32_t input_n = input_base + phase;
                const uint32_t src_pos = ((input_n * kH + in_h) * kW + src_col) * kC;
                ac::LocalTensor<T> phase_local =
                    phase < 2U ? input01[phase * kPhaseTileElems] :
                                  input23[(phase - 2U) * kPhaseTileElems];
                load.blockLen = static_cast<uint16_t>(count * kCBlocks);
                ac::DataCopy(phase_local, src[src_pos], load);
            }

            ac::PipeBarrier<PIPE_ALL>();
            for (uint32_t phase = 0U; phase < kPhases; ++phase) {
                const uint32_t first_col =
                    phase >= crop_phase ? phase - crop_phase : phase + kPhases - crop_phase;
                if (first_col >= cols) {
                    continue;
                }
                const uint32_t count = ((cols - 1U - first_col) >> 2U) + 1U;
                ac::LocalTensor<T> phase_local =
                    phase < 2U ? input01[phase * kPhaseTileElems] :
                                  input23[(phase - 2U) * kPhaseTileElems];
                scatter.blockCount = static_cast<uint16_t>(count);
                ac::DataCopy(output[first_col * kC], phase_local, scatter);
            }

            ac::PipeBarrier<PIPE_ALL>();
            const uint32_t out_pos = (out_row * kOutW + tile_ow) * kC;
            store.blockLen = static_cast<uint16_t>(cols * kCBlocks);
            ac::DataCopy(dst[out_pos], output, store);
            ac::PipeBarrier<PIPE_MTE3>();
        }
    }
}

template <typename T>
__aicore__ inline void HugeHSmallWHalf(GM_ADDR src_addr, GM_ADDR dst_addr)
{
    if constexpr (sizeof(T) != 2U) {
        return;
    } else {
        constexpr uint32_t kBlockBytes = 32U;
        constexpr uint64_t kW = 6U;
        constexpr uint64_t kH = 1024U;
        constexpr uint64_t kC = 32U;
        constexpr uint64_t kOutH = 2048U;
        constexpr uint64_t kOutN = 4U;
        constexpr uint64_t kLaneElems = kW * kC;
        constexpr uint64_t kInputBatchElems = kH * kLaneElems;
        constexpr uint64_t kOutRowElems = 2U * kW * kC;
        constexpr uint64_t kBatchUnit = kOutN * kInputBatchElems;
        constexpr uint64_t kRowsPerChunk = 16U;
        constexpr uint64_t kChunksPerOutN = kH / kRowsPerChunk;
        constexpr uint64_t kTotalChunks = kOutN * kChunksPerOutN;
        constexpr uint16_t kDepthBlocks = static_cast<uint16_t>(kC / (kBlockBytes / sizeof(T)));
        constexpr uint16_t kOutRowBlocks = static_cast<uint16_t>(kOutRowElems / (kBlockBytes / sizeof(T)));
        constexpr uint32_t kParityElems = static_cast<uint32_t>(kRowsPerChunk * kOutRowElems);
        constexpr uint32_t kQueueBytes =
            static_cast<uint32_t>(2U * kRowsPerChunk * 2U * kW * kC * sizeof(T));

        ac::GlobalTensor<T> src;
        ac::GlobalTensor<T> dst;
        BindGm(src_addr, dst_addr, src, dst);

        ac::TPipe pipe;
        ac::TQue<ac::TPosition::VECIN, 2> src_queue;
        ac::TQue<ac::TPosition::VECOUT, 2> dst_queue;
        pipe.InitBuffer(src_queue, 2, kQueueBytes);
        pipe.InitBuffer(dst_queue, 2, kQueueBytes);

        const uint32_t lane_bytes = static_cast<uint32_t>(kC * sizeof(T));
        ac::DataCopyPadExtParams<T> no_pad{false, 0, 0, 0};
        ac::DataCopyExtParams load{
            static_cast<uint16_t>(kRowsPerChunk * kW), lane_bytes, 0U, kDepthBlocks, 0U};

        const uint32_t core = static_cast<uint32_t>(ac::GetBlockIdx());
        const uint32_t cores = static_cast<uint32_t>(ac::GetBlockNum());
        for (uint64_t chunk = core; chunk < kTotalChunks; chunk += cores) {
            const uint64_t out_n = chunk / kChunksPerOutN;
            const uint64_t in_h0 = (chunk - out_n * kChunksPerOutN) * kRowsPerChunk;
            const uint64_t base = out_n * kInputBatchElems + in_h0 * kLaneElems;

            ac::LocalTensor<T> packed = src_queue.template AllocTensor<T>();
            ac::DataCopyPad(packed, src[base], load, no_pad);
            ac::DataCopyPad(packed[kC], src[base + kBatchUnit], load, no_pad);
            ac::DataCopyPad(packed[kParityElems], src[base + 2U * kBatchUnit], load, no_pad);
            ac::DataCopyPad(
                packed[kParityElems + static_cast<uint32_t>(kC)],
                src[base + 3U * kBatchUnit], load, no_pad);
            src_queue.template EnQue<T>(packed);

            packed = src_queue.template DeQue<T>();
            ac::LocalTensor<T> output = dst_queue.template AllocTensor<T>();
            ac::DataCopy(
                output, packed,
                ac::DataCopyParams(static_cast<uint16_t>(kRowsPerChunk), kOutRowBlocks, 0, kOutRowBlocks));
            ac::DataCopy(
                output[static_cast<uint32_t>(kOutRowElems)], packed[kParityElems],
                ac::DataCopyParams(static_cast<uint16_t>(kRowsPerChunk), kOutRowBlocks, 0, kOutRowBlocks));
            src_queue.template FreeTensor<T>(packed);
            dst_queue.template EnQue<T>(output);

            output = dst_queue.template DeQue<T>();
            const uint64_t out_row = out_n * kOutH + (in_h0 << 1U);
            ac::DataCopy(dst[out_row * kOutRowElems], output,
                         static_cast<uint32_t>(2U * kRowsPerChunk * kOutRowElems));
            dst_queue.template FreeTensor<T>(output);
        }
    }
}

template <typename T>
class FallbackMover {
public:
    __aicore__ inline void Init(GM_ADDR src_addr, GM_ADDR dst_addr, const BtsShapePacket &packet)
    {
        BindGm(src_addr, dst_addr, src_, dst_);
        pipe_.InitBuffer(tile_, kBufferBytes);
        local_ = tile_.Get<T>();
        in_h_ = packet.src_h;
        in_w_ = packet.src_w;
        in_c_ = packet.src_c;
        out_n_ = packet.dst_n;
        out_h_ = packet.dst_h;
        out_w_ = packet.dst_w;
        out_c_ = packet.dst_c;
        top_ = packet.crop_t;
        bottom_ = packet.crop_b;
        left_ = packet.crop_l;
        right_ = packet.crop_r;
        block_ = packet.block;
        out_elems_ = packet.dst_elems;
        c_bytes_ = out_c_ * sizeof(T);
        aligned_c_bytes_ = ((c_bytes_ + kBlockBytes - 1U) / kBlockBytes) * kBlockBytes;
    }

    __aicore__ inline void Run()
    {
        const uint32_t core = static_cast<uint32_t>(ac::GetBlockIdx());
        const uint32_t cores = static_cast<uint32_t>(ac::GetBlockNum());
        if (TryFlatCopy(core, cores)) {
            return;
        }

        ac::DataCopyPadExtParams<T> no_pad{false, 0, 0, 0};
        uint32_t cols_per_tile = aligned_c_bytes_ == 0U ? 1U : kBufferBytes / aligned_c_bytes_;
        if (cols_per_tile == 0U) {
            cols_per_tile = 1U;
        }

        const uint32_t row_groups = out_n_ * out_h_ * block_;
        for (uint32_t group = core; group < row_groups; group += cores) {
            const uint32_t phase_w = group % block_;
            const uint32_t row_key = group / block_;
            const uint32_t oh = row_key % out_h_;
            const uint32_t n = row_key / out_h_;
            const uint32_t left_phase = left_ - (left_ / block_) * block_;
            const uint32_t first_ow = phase_w >= left_phase ?
                phase_w - left_phase : phase_w + block_ - left_phase;
            if (first_ow >= out_w_) {
                continue;
            }

            const uint32_t padded_h = oh + top_;
            const uint32_t ih = padded_h / block_;
            const uint32_t phase_h = padded_h - ih * block_;
            const uint32_t in_n = (phase_h * block_ + phase_w) * out_n_ + n;
            const uint32_t run_cols = (out_w_ - 1U - first_ow) / block_ + 1U;

            if (c_bytes_ > kBufferBytes) {
                CopyWideChannels(no_pad, in_n, ih, first_ow, run_cols);
                continue;
            }
            CopyRegularChannels(no_pad, cols_per_tile, in_n, ih, first_ow, run_cols);
        }
    }

private:
    __aicore__ inline bool TryFlatCopy(uint32_t core, uint32_t cores)
    {
        const bool layout_is_flat = top_ == 0U && bottom_ == 0U && left_ == 0U && right_ == 0U &&
            in_h_ == 1U && in_w_ == 1U && out_n_ == 1U && out_h_ == block_ && out_w_ == block_;
        if (!layout_is_flat || (c_bytes_ & (kBlockBytes - 1U)) != 0U) {
            return false;
        }

        const uint32_t total_blocks = (out_elems_ * sizeof(T)) / kBlockBytes;
        const uint32_t share = (total_blocks + cores - 1U) / cores;
        uint32_t block0 = core * share;
        if (block0 >= total_blocks) {
            return true;
        }
        uint32_t count = total_blocks - block0;
        if (count > share) {
            count = share;
        }

        ac::DataCopyParams copy;
        copy.blockCount = 1;
        copy.srcStride = 0;
        copy.dstStride = 0;
        uint32_t elem = block0 * (kBlockBytes / sizeof(T));
        constexpr uint32_t max_blocks = kBufferBytes / kBlockBytes;
        while (count != 0U) {
            const uint32_t step = count > max_blocks ? max_blocks : count;
            copy.blockLen = static_cast<uint16_t>(step);
            ac::DataCopy(local_, src_[elem], copy);
            ac::PipeBarrier<PIPE_ALL>();
            ac::DataCopy(dst_[elem], local_, copy);
            ac::PipeBarrier<PIPE_ALL>();
            elem += step * (kBlockBytes / sizeof(T));
            count -= step;
        }
        return true;
    }

    __aicore__ inline void CopyWideChannels(ac::DataCopyPadExtParams<T> &no_pad, uint32_t in_n,
                                            uint32_t ih, uint32_t first_ow, uint32_t run_cols)
    {
        constexpr uint32_t max_chunk_bytes = 32U * 1024U;
        constexpr uint32_t max_chunk_elems = max_chunk_bytes / sizeof(T);
        for (uint32_t c0 = 0; c0 < out_c_; c0 += max_chunk_elems) {
            uint32_t chunk = out_c_ - c0;
            if (chunk > max_chunk_elems) {
                chunk = max_chunk_elems;
            }
            const uint32_t chunk_bytes = chunk * sizeof(T);
            const uint32_t aligned = ((chunk_bytes + kBlockBytes - 1U) / kBlockBytes) * kBlockBytes;
            uint32_t cols_per_tile = kBufferBytes / aligned;
            if (cols_per_tile == 0U) {
                cols_per_tile = 1U;
            }

            ac::DataCopyExtParams load;
            load.blockLen = chunk_bytes;
            load.srcStride = c_bytes_ - chunk_bytes;
            load.dstStride = 0;
            load.rsv = 0;

            ac::DataCopyExtParams store;
            store.blockLen = chunk_bytes;
            store.srcStride = 0;
            store.dstStride = block_ * c_bytes_ - chunk_bytes;
            store.rsv = 0;

            for (uint32_t col0 = 0; col0 < run_cols; col0 += cols_per_tile) {
                uint32_t cols = run_cols - col0;
                if (cols > cols_per_tile) {
                    cols = cols_per_tile;
                }
                const uint32_t ow = first_ow + col0 * block_;
                const uint32_t iw = (ow + left_) / block_;
                const uint32_t src_pos = ((in_n * in_h_ + ih) * in_w_ + iw) * in_c_ + c0;
                const uint32_t dst_index = ((CurrentN(in_n) * out_h_ + CurrentOh(ih, in_n)) * out_w_ + ow) * out_c_ + c0;
                load.blockCount = static_cast<uint16_t>(cols);
                store.blockCount = static_cast<uint16_t>(cols);
                ac::DataCopyPad(local_, src_[src_pos], load, no_pad);
                ac::PipeBarrier<PIPE_ALL>();
                ac::DataCopyPad(dst_[dst_index], local_, store);
                ac::PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void CopyRegularChannels(ac::DataCopyPadExtParams<T> &no_pad, uint32_t cols_per_tile,
                                               uint32_t in_n, uint32_t ih, uint32_t first_ow,
                                               uint32_t run_cols)
    {
        const uint32_t pad_bytes = aligned_c_bytes_ - c_bytes_;
        for (uint32_t col0 = 0; col0 < run_cols; col0 += cols_per_tile) {
            uint32_t cols = run_cols - col0;
            if (cols > cols_per_tile) {
                cols = cols_per_tile;
            }
            const uint32_t ow = first_ow + col0 * block_;
            const uint32_t iw = (ow + left_) / block_;
            const uint32_t src_pos = ((in_n * in_h_ + ih) * in_w_ + iw) * in_c_;
            const uint32_t dst_pos = ((CurrentN(in_n) * out_h_ + CurrentOh(ih, in_n)) * out_w_ + ow) * out_c_;

            if (pad_bytes == 0U) {
                const uint32_t c_blocks = c_bytes_ / kBlockBytes;
                ac::DataCopyParams load;
                load.blockCount = 1;
                load.blockLen = static_cast<uint16_t>((cols * c_bytes_) / kBlockBytes);
                load.srcStride = 0;
                load.dstStride = 0;

                ac::DataCopyParams store;
                store.blockCount = static_cast<uint16_t>(cols);
                store.blockLen = static_cast<uint16_t>(c_blocks);
                store.srcStride = 0;
                store.dstStride = static_cast<uint16_t>((block_ - 1U) * c_blocks);

                ac::DataCopy(local_, src_[src_pos], load);
                ac::PipeBarrier<PIPE_ALL>();
                ac::DataCopy(dst_[dst_pos], local_, store);
                ac::PipeBarrier<PIPE_ALL>();
                continue;
            }

            ac::DataCopyExtParams load;
            load.blockCount = static_cast<uint16_t>(cols);
            load.blockLen = c_bytes_;
            load.srcStride = 0;
            load.dstStride = 0;
            load.rsv = 0;

            ac::DataCopyExtParams store;
            store.blockCount = static_cast<uint16_t>(cols);
            store.blockLen = c_bytes_;
            store.srcStride = 0;
            store.dstStride = block_ * c_bytes_ - c_bytes_;
            store.rsv = 0;

            ac::DataCopyPad(local_, src_[src_pos], load, no_pad);
            ac::PipeBarrier<PIPE_ALL>();
            ac::DataCopyPad(dst_[dst_pos], local_, store);
            ac::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline uint32_t CurrentN(uint32_t input_n) const
    {
        return input_n % out_n_;
    }

    __aicore__ inline uint32_t CurrentOh(uint32_t ih, uint32_t input_n) const
    {
        const uint32_t phase_h = (input_n / out_n_) / block_;
        return ih * block_ + phase_h - top_;
    }

    static constexpr uint32_t kBlockBytes = 32U;
    static constexpr uint32_t kBufferBytes = 64U * 1024U;
    ac::TPipe pipe_;
    ac::TBuf<ac::TPosition::VECIN> tile_;
    ac::LocalTensor<T> local_;
    ac::GlobalTensor<T> src_;
    ac::GlobalTensor<T> dst_;
    uint32_t in_h_;
    uint32_t in_w_;
    uint32_t in_c_;
    uint32_t out_n_;
    uint32_t out_h_;
    uint32_t out_w_;
    uint32_t out_c_;
    uint32_t top_;
    uint32_t bottom_;
    uint32_t left_;
    uint32_t right_;
    uint32_t block_;
    uint32_t out_elems_;
    uint32_t c_bytes_;
    uint32_t aligned_c_bytes_;
};

template <typename T>
__aicore__ inline void RunLane0(GM_ADDR src, GM_ADDR dst)
{
    DirectRowsR2<T, 28U, 28U, 128U, 2U, 32U>(src, dst);
}

template <typename T>
__aicore__ inline void RunLane1(GM_ADDR src, GM_ADDR dst)
{
    GatherSmallCropR2<T>(src, dst);
}

template <typename T>
__aicore__ inline void RunLane2(GM_ADDR src, GM_ADDR dst)
{
    Rows14FloatPath<T>(src, dst);
}

template <typename T>
__aicore__ inline void RunLane3(GM_ADDR src, GM_ADDR dst)
{
    RowsPairR2<T, 4U, 6U, 32U, 5U, 4U, 4096U, false, false>(src, dst);
}

template <typename T>
__aicore__ inline void RunLane4(GM_ADDR src, GM_ADDR dst)
{
    DenseCropC65HalfStatic<T>(src, dst);
}

template <typename T>
__aicore__ inline void RunLane5(GM_ADDR src, GM_ADDR dst)
{
    SplitChannelsR2<T, 2U, 2U, 4096U, 1U, 2048U>(src, dst);
}

template <typename T>
__aicore__ inline void RunLane6(GM_ADDR src, GM_ADDR dst)
{
    CopyFlatWindow<T>(src, dst);
}

template <typename T>
__aicore__ inline void RunLane7(GM_ADDR src, GM_ADDR dst)
{
    WideRowsHalf<T>(src, dst);
}

template <typename T>
__aicore__ inline void RunLane8(GM_ADDR src, GM_ADDR dst)
{
    Block4PhaseRowsHalf<T>(src, dst);
}

template <typename T>
__aicore__ inline void RunLane9(GM_ADDR src, GM_ADDR dst)
{
    HugeHSmallWHalf<T>(src, dst);
}

template <typename T, uint64_t Lane>
__aicore__ inline void RunFixedLane(GM_ADDR src, GM_ADDR dst)
{
    if constexpr (Lane == BTS_LANE_0) {
        RunLane0<T>(src, dst);
    } else if constexpr (Lane == BTS_LANE_1) {
        RunLane1<T>(src, dst);
    } else if constexpr (Lane == BTS_LANE_2) {
        RunLane2<T>(src, dst);
    } else if constexpr (Lane == BTS_LANE_3) {
        RunLane3<T>(src, dst);
    } else if constexpr (Lane == BTS_LANE_4) {
        RunLane4<T>(src, dst);
    } else if constexpr (Lane == BTS_LANE_5) {
        RunLane5<T>(src, dst);
    } else if constexpr (Lane == BTS_LANE_6) {
        RunLane6<T>(src, dst);
    } else if constexpr (Lane == BTS_LANE_7) {
        RunLane7<T>(src, dst);
    } else if constexpr (Lane == BTS_LANE_8) {
        RunLane8<T>(src, dst);
    } else if constexpr (Lane == BTS_LANE_9) {
        RunLane9<T>(src, dst);
    }
}

template <class DT_X, uint64_t SCH_MODE>
__global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    if constexpr (SCH_MODE < BTS_LANE_SAFE) {
        RunFixedLane<DT_X, SCH_MODE>(x, y);
        return;
    }

    REGISTER_TILING_DEFAULT(BtsShapePacket);
    GET_TILING_DATA_WITH_STRUCT(BtsShapePacket, runtime_shape, tiling);
    FallbackMover<DT_X> fallback;
    fallback.Init(x, y, runtime_shape);
    fallback.Run();
}
