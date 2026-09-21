#pragma once
#include "kernel_operator.h"
#include "pool_tiling.h"

namespace MaxPoolBackward {
constexpr int64_t TILE = 8192;
constexpr int64_t CHUNK = 1024;
constexpr int64_t CHUNK_ALIGN_BUF = CHUNK + 32;

template<class T> __aicore__ inline float LoadFloat(T value) { return static_cast<float>(value); }
#if defined(POOL_SUPPORT_BF16) || __NPU_ARCH__ == 2201
template<> __aicore__ inline float LoadFloat<bfloat16_t>(bfloat16_t value) { return AscendC::ToFloat(value); }
#endif

template<class T>
__aicore__ inline void LoadChunk(
    int64_t chunk_idx, int64_t glob_stream_start, int64_t total_stream_elems,
    AscendC::GlobalTensor<T>& g, AscendC::GlobalTensor<int32_t>& idx,
    AscendC::TQue<AscendC::QuePosition::VECIN, 2>& gradQueue,
    AscendC::TQue<AscendC::QuePosition::VECIN, 2>& idxQueue)
{
    const int64_t offset = chunk_idx * CHUNK;
    const int64_t cur_glob_start = glob_stream_start + offset;
    const int64_t chunk_len = (total_stream_elems - offset < CHUNK) ? (total_stream_elems - offset) : CHUNK;

    const int64_t g_elem_size = sizeof(T);
    const int64_t g_byte_start = cur_glob_start * g_elem_size;
    const int64_t g_aligned_byte = (g_byte_start / 32) * 32;
    const int64_t g_byte_end = g_byte_start + chunk_len * g_elem_size;
    const int64_t g_aligned_end = ((g_byte_end + 31) / 32) * 32;
    const uint32_t g_copy_elems = static_cast<uint32_t>((g_aligned_end - g_aligned_byte) / g_elem_size);
    const int64_t g_gm_offset = g_aligned_byte / g_elem_size;

    const int64_t idx_byte_start = cur_glob_start * 4;
    const int64_t idx_aligned_byte = (idx_byte_start / 32) * 32;
    const int64_t idx_byte_end = idx_byte_start + chunk_len * 4;
    const int64_t idx_aligned_end = ((idx_byte_end + 31) / 32) * 32;
    const uint32_t idx_copy_elems = static_cast<uint32_t>((idx_aligned_end - idx_aligned_byte) / 4);
    const int64_t idx_gm_offset = idx_aligned_byte / 4;

    auto gradLocal = gradQueue.AllocTensor<T>();
    auto idxLocal = idxQueue.AllocTensor<int32_t>();

    AscendC::DataCopy(gradLocal, g[g_gm_offset], g_copy_elems);
    AscendC::DataCopy(idxLocal, idx[idx_gm_offset], idx_copy_elems);

    gradQueue.EnQue(gradLocal);
    idxQueue.EnQue(idxLocal);
}

template<class T, bool PaddedOutput = true>
__aicore__ inline void Compute(GM_ADDR grad, GM_ADDR indices, GM_ADDR output, PoolTiling t)
{
    const int64_t spatial = t.height * t.width;
    const int64_t outputSpatial = t.outHeight * t.outWidth;
    const int64_t total = t.planes * spatial;
    const int64_t totalGrad = t.planes * outputSpatial;
    const int64_t padded = (total + TILE - 1) / TILE * TILE;
    const int64_t paddedGrad = ((totalGrad * static_cast<int64_t>(sizeof(T)) + 63) / 32) * 32 / static_cast<int64_t>(sizeof(T)) + 64;
    const int64_t paddedIdx = ((totalGrad * 4 + 63) / 32) * 32 / 4 + 64;

    AscendC::GlobalTensor<T> g, result;
    AscendC::GlobalTensor<int32_t> idx;
    g.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad), paddedGrad);
    idx.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices), paddedIdx);
    result.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output), PaddedOutput ? padded : total);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumBuffer;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outputQueue;
    // Ping-Pong Double Buffering (depth 2) for DMA latency hiding
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> gradQueue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> idxQueue;

    pipe.InitBuffer(accumBuffer, TILE * sizeof(float));
    pipe.InitBuffer(outputQueue, 1, TILE * sizeof(T));
    pipe.InitBuffer(gradQueue, 2, CHUNK_ALIGN_BUF * sizeof(T));
    pipe.InitBuffer(idxQueue, 2, CHUNK_ALIGN_BUF * sizeof(int32_t));

    auto accum = accumBuffer.Get<float>();
    auto accumData = reinterpret_cast<__ubuf__ float*>(accum.GetPhyAddr());

    // Tile ownership is aligned to 32 bytes: no cross-core partial-line writes.
    for (int64_t start = static_cast<int64_t>(AscendC::GetBlockIdx()) * TILE;
         start < total; start += static_cast<int64_t>(AscendC::GetBlockNum()) * TILE) {
        const int64_t end = (start + TILE < total) ? (start + TILE) : total;

        // Zero-initialize accumulator in Unified Buffer
        AscendC::Duplicate(accum, 0.0f, TILE);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

        const int64_t p_min = start / spatial;
        const int64_t p_max = (end - 1) / spatial;

        if (spatial <= TILE) {
            // Path 1: Batched-Plane streaming across all planes in [p_min, p_max]
            const int64_t glob_stream_start = p_min * outputSpatial;
            const int64_t glob_stream_end = (p_max + 1) * outputSpatial;
            const int64_t total_stream_elems = glob_stream_end - glob_stream_start;
            const int64_t num_chunks = (total_stream_elems + CHUNK - 1) / CHUNK;

            int64_t cur_p = p_min;
            int64_t cur_plane_spatial_base = cur_p * spatial;
            bool cur_plane_fully_in = (cur_plane_spatial_base >= start && cur_plane_spatial_base + spatial <= end);
            int64_t in_plane_counter = 0;

            if (num_chunks > 0) {
                LoadChunk<T>(0, glob_stream_start, total_stream_elems, g, idx, gradQueue, idxQueue);

                for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
                    if (chunk_idx + 1 < num_chunks) {
                        LoadChunk<T>(chunk_idx + 1, glob_stream_start, total_stream_elems, g, idx, gradQueue, idxQueue);
                    }

                    const int64_t offset = chunk_idx * CHUNK;
                    const int64_t cur_glob_start = glob_stream_start + offset;
                    const int64_t chunk_len = (total_stream_elems - offset < CHUNK) ? (total_stream_elems - offset) : CHUNK;

                    const int64_t g_elem_size = sizeof(T);
                    const int64_t g_skip_elems = (cur_glob_start * g_elem_size - ((cur_glob_start * g_elem_size) / 32) * 32) / g_elem_size;
                    const int64_t idx_skip_elems = (cur_glob_start * 4 - ((cur_glob_start * 4) / 32) * 32) / 4;

                    auto gradLocal = gradQueue.DeQue<T>();
                    auto idxLocal = idxQueue.DeQue<int32_t>();

                    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

                    auto gradPtr = reinterpret_cast<__ubuf__ T*>(gradLocal.GetPhyAddr());
                    auto idxPtr = reinterpret_cast<__ubuf__ int32_t*>(idxLocal.GetPhyAddr());

                    #pragma unroll 4
                    for (int64_t i = 0; i < chunk_len; ++i) {
                        const int32_t pos = idxPtr[idx_skip_elems + i];
                        const int64_t flat = cur_plane_spatial_base + pos;
                        if (cur_plane_fully_in) {
                            accumData[flat - start] += LoadFloat<T>(gradPtr[g_skip_elems + i]);
                        } else {
                            if (flat >= start && flat < end) {
                                accumData[flat - start] += LoadFloat<T>(gradPtr[g_skip_elems + i]);
                            }
                        }

                        if (++in_plane_counter == outputSpatial) {
                            in_plane_counter = 0;
                            cur_p++;
                            cur_plane_spatial_base += spatial;
                            cur_plane_fully_in = (cur_plane_spatial_base >= start && cur_plane_spatial_base + spatial <= end);
                        }
                    }

                    gradQueue.FreeTensor(gradLocal);
                    idxQueue.FreeTensor(idxLocal);
                }
            }
        } else {
            // Path 2: Large image (spatial > TILE). Bounded row streaming for each plane in [p_min, p_max]
            for (int64_t p = p_min; p <= p_max; ++p) {
                const int64_t plane_spatial_base = p * spatial;
                const int64_t in_start = (start > plane_spatial_base) ? (start - plane_spatial_base) : 0;
                const int64_t in_end = (end < plane_spatial_base + spatial) ? (end - plane_spatial_base) : spatial;

                const int64_t y_min = in_start / t.width;
                const int64_t y_max = (in_end - 1) / t.width;

                const int64_t y_begin_denom = y_min + t.padH - t.kernelH + 1;
                const int64_t oy_min = (y_begin_denom > 0) ? (y_begin_denom / t.strideH) : 0;
                const int64_t oy_candidate = (y_max + t.padH) / t.strideH;
                const int64_t oy_max = (oy_candidate < t.outHeight) ? oy_candidate : (t.outHeight - 1);

                if (oy_min > oy_max) {
                    continue;
                }

                const int64_t out_elem_start = oy_min * t.outWidth;
                const int64_t out_elem_end = (oy_max + 1) * t.outWidth;
                const int64_t out_elem_count = out_elem_end - out_elem_start;
                const int64_t glob_stream_start = p * outputSpatial + out_elem_start;
                const int64_t num_chunks = (out_elem_count + CHUNK - 1) / CHUNK;

                if (num_chunks > 0) {
                    LoadChunk<T>(0, glob_stream_start, out_elem_count, g, idx, gradQueue, idxQueue);

                    for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
                        if (chunk_idx + 1 < num_chunks) {
                            LoadChunk<T>(chunk_idx + 1, glob_stream_start, out_elem_count, g, idx, gradQueue, idxQueue);
                        }

                        const int64_t offset = chunk_idx * CHUNK;
                        const int64_t cur_glob_start = glob_stream_start + offset;
                        const int64_t chunk_len = (out_elem_count - offset < CHUNK) ? (out_elem_count - offset) : CHUNK;

                        const int64_t g_elem_size = sizeof(T);
                        const int64_t g_skip_elems = (cur_glob_start * g_elem_size - ((cur_glob_start * g_elem_size) / 32) * 32) / g_elem_size;
                        const int64_t idx_skip_elems = (cur_glob_start * 4 - ((cur_glob_start * 4) / 32) * 32) / 4;

                        auto gradLocal = gradQueue.DeQue<T>();
                        auto idxLocal = idxQueue.DeQue<int32_t>();

                        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

                        auto gradPtr = reinterpret_cast<__ubuf__ T*>(gradLocal.GetPhyAddr());
                        auto idxPtr = reinterpret_cast<__ubuf__ int32_t*>(idxLocal.GetPhyAddr());

                        #pragma unroll 4
                        for (int64_t i = 0; i < chunk_len; ++i) {
                            const int32_t pos = idxPtr[idx_skip_elems + i];
                            const int64_t flat = plane_spatial_base + pos;
                            if (flat >= start && flat < end) {
                                accumData[flat - start] += LoadFloat<T>(gradPtr[g_skip_elems + i]);
                            }
                        }

                        gradQueue.FreeTensor(gradLocal);
                        idxQueue.FreeTensor(idxLocal);
                    }
                }
            }
        }

        auto out = outputQueue.AllocTensor<T>();
        // Scalar writes must be visible to the vector pipeline.
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        if constexpr (sizeof(T) == sizeof(float)) {
            AscendC::Adds(out, accum, 0.0f, TILE);
        } else {
#if defined(POOL_SUPPORT_BF16) || __NPU_ARCH__ == 2201
            AscendC::Cast(out, accum, AscendC::RoundMode::CAST_RINT, TILE);
#else
            AscendC::Cast(out, accum, AscendC::RoundMode::CAST_NONE, TILE);
#endif
        }
        outputQueue.EnQue(out);
        out = outputQueue.DeQue<T>();
        if constexpr (PaddedOutput) {
            AscendC::DataCopy(result[start], out, TILE);
        } else {
            if (start + TILE <= total) {
                AscendC::DataCopy(result[start], out, TILE);
            } else {
#if defined(POOL_SUPPORT_BF16) || __NPU_ARCH__ == 2201
                AscendC::DataCopyParams copy{1, static_cast<uint16_t>((total - start) * sizeof(T)), 0, 0};
                AscendC::DataCopyPad(result[start], out, copy);
#else
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
                auto source = reinterpret_cast<__ubuf__ T*>(out.GetPhyAddr());
                for (int64_t tail = 0; tail < total - start; ++tail) result.SetValue(start + tail, source[tail]);
                AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::ENTIRE_DATA_CACHE>(result);
#endif
            }
        }
        outputQueue.FreeTensor(out);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
    }
}
} // namespace MaxPoolBackward
