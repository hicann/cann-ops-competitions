#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t UB_BLOCK_BYTES = 32768; 

template <typename T>
class KernelScale {
public:
    __aicore__ inline KernelScale() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, 
                               uint32_t m, uint32_t n, uint32_t k, uint32_t total, int flag) {
        this->m = m;
        this->n = n;
        this->k = k;
        this->total_size = total;
        
        this->tile_elements = UB_BLOCK_BYTES / sizeof(T);
        this->elements_per_32b = 32 / sizeof(T);

        this->blocknum = GetBlockNum();
        this->blockidx = GetBlockIdx();

        if(k == 1) {
            if(n <= tile_elements) {
                this->mode = 2;
            } else if(n >= 40 * 4096) {
                this->mode = 3;
            } else {
                this->mode = 4;
            }
        } else {
            this->mode = 5;
        }
        this->flag = flag;

        inputGm.SetGlobalBuffer((__gm__ T*)input);
        outputGm.SetGlobalBuffer((__gm__ T*)output);
        scaleGm.SetGlobalBuffer((__gm__ T*)scale);
        biasGm.SetGlobalBuffer((__gm__ T*)bias);
        outputGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

        pipe.InitBuffer(inQueueX, 2, UB_BLOCK_BYTES);
        pipe.InitBuffer(inQueueS, 1, UB_BLOCK_BYTES); 
        pipe.InitBuffer(inQueueB, 1, UB_BLOCK_BYTES);
        pipe.InitBuffer(outQueueY, 2, UB_BLOCK_BYTES);
    }

    __aicore__ inline void Process() {
        if(mode == 2) {
            int start = (m * blockidx) / blocknum;
            int end = (m * (blockidx + 1)) / blocknum;
            Compute2(start, end);
        } else if(mode == 3) {
            int cache_line = 128;
            int lines = (n + cache_line - 1) / cache_line;
            int start = (lines * blockidx) / blocknum;
            start *= cache_line;
            int end = (lines * (blockidx + 1)) / blocknum;
            end = min(int(end * cache_line), int(n));
            Compute3(start, end);
        } else if(mode == 4) {
            int start = (m * blockidx) / blocknum;
            int end = (m * (blockidx + 1)) / blocknum;
            Compute4(start, end);
        } else {
            int cache_line = 256;
            int lines = (total_size + cache_line - 1) / cache_line;
            int start = (lines * blockidx) / blocknum;
            start *= cache_line;
            int end = (lines * (blockidx + 1)) / blocknum;
            end = min(int(end * cache_line), int(total_size));
            Compute5(start, end);
        }
    }

private:
    __aicore__ inline uint32_t AlignTo(uint32_t val, uint32_t align) {
        return (val + align - 1) & ~(align - 1);
    }

    __aicore__ inline void Compute2(uint32_t start, uint32_t end) {
        if(start == end) return;
        LocalTensor<T> sLocal = inQueueS.AllocTensor<T>();
        LocalTensor<T> bLocal = inQueueB.AllocTensor<T>();
        uint32_t align_n = AlignTo(n, 32);
        DataCopy(sLocal, scaleGm, align_n);
        DataCopy(bLocal, biasGm, align_n);

        inQueueS.EnQue(sLocal); 
        inQueueS.DeQue<T>();
        inQueueB.EnQue(bLocal); 
        inQueueB.DeQue<T>();

        LocalTensor<T> xLocal;
        LocalTensor<T> yLocal;

        int i, iter, cur_end;
        if(flag) {
            i = start;
            iter = 1;
            cur_end = end;
        } else {
            i = end - 1;
            iter = -1;
            cur_end = start - 1;
        }

        for(; i != cur_end; i+=iter) {
            xLocal = inQueueX.AllocTensor<T>();
            yLocal = outQueueY.AllocTensor<T>();

            DataCopy(xLocal, inputGm[i*n], align_n);

            inQueueX.EnQue(xLocal); 
            xLocal = inQueueX.DeQue<T>();

            Mul(yLocal, xLocal, sLocal, align_n);
            inQueueX.FreeTensor(xLocal);
            Add(yLocal, yLocal, bLocal, align_n);

            outQueueY.EnQue(yLocal); 
            yLocal = outQueueY.DeQue<T>();

            if(n % 32) {
                DataCopyExtParams copyParams{1, (uint32_t)(n * sizeof(T)), 0, 0, 0};
                DataCopyPad(outputGm[i*n], yLocal, copyParams);
            } else {
                DataCopy(outputGm[i*n], yLocal, n);
            }

            outQueueY.FreeTensor(yLocal);
        }
        inQueueS.FreeTensor(sLocal);
        inQueueB.FreeTensor(bLocal);
    }

    __aicore__ inline void Compute3(uint32_t start, uint32_t end) {
        if(start == end) return;
        int loop = (end - start + tile_elements - 1) / tile_elements;
        int base_offset = start, cur_n;
        for(int j = 0; j != loop; j++) {
            cur_n = min(end - base_offset, tile_elements);

            LocalTensor<T> sLocal = inQueueS.AllocTensor<T>();
            LocalTensor<T> bLocal = inQueueB.AllocTensor<T>();
            uint32_t align_n = AlignTo(cur_n, 32);
            DataCopy(sLocal, scaleGm[base_offset], align_n);
            DataCopy(bLocal, biasGm[base_offset], align_n);

            inQueueS.EnQue(sLocal); 
            inQueueS.DeQue<T>();
            inQueueB.EnQue(bLocal); 
            inQueueB.DeQue<T>();

            LocalTensor<T> xLocal;
            LocalTensor<T> yLocal;

            int i, iter, cur_end;
            if(flag) {
                i = 0;
                iter = 1;
                cur_end = m;
            } else {
                i = m - 1;
                iter = -1;
                cur_end = -1;
            }

            for(; i != cur_end; i+=iter) {
                xLocal = inQueueX.AllocTensor<T>();
                yLocal = outQueueY.AllocTensor<T>();

                DataCopy(xLocal, inputGm[i*n+base_offset], align_n);

                inQueueX.EnQue(xLocal); 
                xLocal = inQueueX.DeQue<T>();

                Mul(yLocal, xLocal, sLocal, align_n);
                inQueueX.FreeTensor(xLocal);
                Add(yLocal, yLocal, bLocal, align_n);

                outQueueY.EnQue(yLocal); 
                yLocal = outQueueY.DeQue<T>();

                if(cur_n % 32) {
                    DataCopyExtParams copyParams{1, (uint32_t)(cur_n * sizeof(T)), 0, 0, 0};
                    DataCopyPad(outputGm[i*n+base_offset], yLocal, copyParams);
                } else {
                    DataCopy(outputGm[i*n+base_offset], yLocal, cur_n);
                }

                outQueueY.FreeTensor(yLocal);
            }
            inQueueS.FreeTensor(sLocal);
            inQueueB.FreeTensor(bLocal);

            base_offset += cur_n;
        }
    }

    __aicore__ inline void Compute4(uint32_t start, uint32_t end) {
        if(start == end) return;
        int loop = (n + tile_elements - 1) / tile_elements;
        int base_offset = 0, cur_n;
        for(int j = 0; j != loop; j++) {
            cur_n = min(n - base_offset, tile_elements);

            LocalTensor<T> sLocal = inQueueS.AllocTensor<T>();
            LocalTensor<T> bLocal = inQueueB.AllocTensor<T>();
            uint32_t align_n = AlignTo(cur_n, 32);
            DataCopy(sLocal, scaleGm[base_offset], align_n);
            DataCopy(bLocal, biasGm[base_offset], align_n);

            inQueueS.EnQue(sLocal); 
            inQueueS.DeQue<T>();
            inQueueB.EnQue(bLocal); 
            inQueueB.DeQue<T>();

            LocalTensor<T> xLocal;
            LocalTensor<T> yLocal;

            int i, iter, cur_end;
            if(flag) {
                i = start;
                iter = 1;
                cur_end = end;
            } else {
                i = end - 1;
                iter = -1;
                cur_end = start - 1;
            }

            for(; i != cur_end; i+=iter) {
                xLocal = inQueueX.AllocTensor<T>();
                yLocal = outQueueY.AllocTensor<T>();

                DataCopy(xLocal, inputGm[i*n+base_offset], align_n);

                inQueueX.EnQue(xLocal); 
                xLocal = inQueueX.DeQue<T>();

                Mul(yLocal, xLocal, sLocal, align_n);
                inQueueX.FreeTensor(xLocal);
                Add(yLocal, yLocal, bLocal, align_n);

                outQueueY.EnQue(yLocal); 
                yLocal = outQueueY.DeQue<T>();

                if(cur_n % 32) {
                    DataCopyExtParams copyParams{1, (uint32_t)(cur_n * sizeof(T)), 0, 0, 0};
                    DataCopyPad(outputGm[i*n+base_offset], yLocal, copyParams);
                } else {
                    DataCopy(outputGm[i*n+base_offset], yLocal, cur_n);
                }

                outQueueY.FreeTensor(yLocal);
            }
            inQueueS.FreeTensor(sLocal);
            inQueueB.FreeTensor(bLocal);

            base_offset += cur_n;
        }
    }

//     __aicore__ inline void Compute5(uint32_t start, uint32_t end) {
//         if(start == end) return;

//         int loop = (end - start + tile_elements - 1) / tile_elements;
//         int base_offset = start;
//         if(flag == 0) base_offset = start + (loop - 1) * tile_elements;

//         int i, iter, cur_end;

//         if(flag) {
//             i = 0;
//             iter = 1;
//             cur_end = loop;
//         } else {
//             i = loop - 1;
//             iter = -1;
//             cur_end = -1;
//         }

//         LocalTensor<T> xLocal;
//         LocalTensor<T> yLocal;

//         for(; i != cur_end; i+=iter) {
//             int cur = min(tile_elements, end - base_offset);
//             int align_cur = AlignTo(cur, 32);

//             xLocal = inQueueX.AllocTensor<T>();
//             yLocal = outQueueY.AllocTensor<T>();

//             DataCopy(xLocal, inputGm[base_offset], align_cur);

//             inQueueX.EnQue(xLocal); 
//             xLocal = inQueueX.DeQue<T>();

//             int curr_end = base_offset + cur;
//             while(curr_end > base_offset) {
//                 uint32_t last_elem_idx = curr_end - 1;
//                 uint32_t k_block_idx = last_elem_idx / k;
//                 uint32_t k_block_start = k_block_idx * k;
//                 uint32_t left_bound = (k_block_start > base_offset) ? k_block_start : base_offset;
//                 uint32_t step = curr_end - left_bound;
//                 curr_end = left_bound;
//                 left_bound -= base_offset;
//                 uint32_t n_idx = k_block_idx % n;

//                 T s_val = scaleGm.GetValue(n_idx); 
//                 T b_val = biasGm.GetValue(n_idx);

//                 step += left_bound % elements_per_32b;
//                 left_bound = int(left_bound / elements_per_32b) * elements_per_32b;

//                 Muls(yLocal[left_bound], xLocal[left_bound], s_val, step);
//                 Adds(yLocal[left_bound], yLocal[left_bound], b_val, step);
//             }
//             inQueueX.FreeTensor(xLocal);
//             outQueueY.EnQue(yLocal); 
//             yLocal = outQueueY.DeQue<T>();

//             if(cur % 32) {
//                 DataCopyExtParams copyParams{1, (uint32_t)(cur * sizeof(T)), 0, 0, 0};
//                 DataCopyPad(outputGm[base_offset], yLocal, copyParams);
//             } else {
//                 DataCopy(outputGm[base_offset], yLocal, cur);
//             }

//             outQueueY.FreeTensor(yLocal);
//             base_offset += cur*iter;
//         }
//     }
//     __aicore__ inline void Compute5(uint32_t start, uint32_t end) {
//         if(start == end) return;

//         int loop = (end - start + tile_elements - 1) / tile_elements;

//         int i, iter, cur_end;
//         if(flag) {
//             i = 0;
//             iter = 1;
//             cur_end = loop;
//         } else {
//             i = loop - 1;
//             iter = -1;
//             cur_end = -1;
//         }

//         LocalTensor<T> xLocal;
//         LocalTensor<T> yLocal;

//         for(; i != cur_end; i+=iter) {
//             int base_offset = start + i * (int)tile_elements;   // 由 tile 索引直接算，正反向都正确
//             int cur = min(tile_elements, end - base_offset);
//             int align_cur = AlignTo(cur, 32);

//             xLocal = inQueueX.AllocTensor<T>();
//             yLocal = outQueueY.AllocTensor<T>();

//             DataCopy(xLocal, inputGm[base_offset], align_cur);

//             inQueueX.EnQue(xLocal); 
//             xLocal = inQueueX.DeQue<T>();

//             int curr_end = base_offset + cur;
//             while(curr_end > base_offset) {
//                 uint32_t last_elem_idx = curr_end - 1;
//                 uint32_t k_block_idx = last_elem_idx / k;
//                 uint32_t k_block_start = k_block_idx * k;
//                 uint32_t left_bound = (k_block_start > base_offset) ? k_block_start : base_offset;
//                 uint32_t step = curr_end - left_bound;
//                 curr_end = left_bound;
//                 left_bound -= base_offset;
//                 uint32_t n_idx = k_block_idx % n;

//                 T s_val = scaleGm.GetValue(n_idx); 
//                 T b_val = biasGm.GetValue(n_idx);

//                 step += left_bound % elements_per_32b;
//                 left_bound = int(left_bound / elements_per_32b) * elements_per_32b;

//                 Muls(yLocal[left_bound], xLocal[left_bound], s_val, step);
//                 Adds(yLocal[left_bound], yLocal[left_bound], b_val, step);
//             }
//             inQueueX.FreeTensor(xLocal);
//             outQueueY.EnQue(yLocal); 
//             yLocal = outQueueY.DeQue<T>();

//             if(cur % 32) {
//                 DataCopyExtParams copyParams{1, (uint32_t)(cur * sizeof(T)), 0, 0, 0};
//                 DataCopyPad(outputGm[base_offset], yLocal, copyParams);
//             } else {
//                 DataCopy(outputGm[base_offset], yLocal, cur);
//             }

//             outQueueY.FreeTensor(yLocal);
//         }
//     }
    __aicore__ inline void Compute5(uint32_t start, uint32_t end) {
    if(start == end) return;

    int loop = (end - start + tile_elements - 1) / tile_elements;

    int i, iter, cur_end_idx;
    if(flag) { i = 0;        iter = 1;  cur_end_idx = loop; }
    else     { i = loop - 1; iter = -1; cur_end_idx = -1;   }

    LocalTensor<T> xLocal;
    LocalTensor<T> yLocal;

    for(; i != cur_end_idx; i += iter) {
        int base_offset = start + i * (int)tile_elements;
        int cur = min(tile_elements, end - base_offset);
        int align_cur = AlignTo(cur, 32);

        xLocal = inQueueX.AllocTensor<T>();
        yLocal = outQueueY.AllocTensor<T>();
        DataCopy(xLocal, inputGm[base_offset], align_cur);
        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.DeQue<T>();

        uint32_t curr_end  = base_offset + cur;
        uint32_t last_blk  = (curr_end - 1) / k;   // 每 tile 仅 1 次 div/mul/mod
        uint32_t blk_start = last_blk * k;
        uint32_t nid       = last_blk % n;

        while(curr_end > (uint32_t)base_offset) {
            // 1) 先算最多 4 个 block 的几何（只有减法/回绕，无 div/mod）
            uint32_t lb[4], step[4], id[4];
            int cnt = 0;
            for(; cnt < 4 && curr_end > (uint32_t)base_offset; ++cnt) {
                uint32_t left_bound = (blk_start > (uint32_t)base_offset) ? blk_start : (uint32_t)base_offset;
                uint32_t s = curr_end - left_bound;
                uint32_t l = left_bound - base_offset;
                s += l % elements_per_32b;
                l  = (l / elements_per_32b) * elements_per_32b;
                lb[cnt] = l; step[cnt] = s; id[cnt] = nid;
                curr_end  = left_bound;
                blk_start -= k;
                if(nid == 0) nid = n - 1; else --nid;
            }

            if(cnt == 4) {
                // 2) 一次性批量取数（4 个同数组、地址连续 → 流水隐藏延迟）
                T s0 = scaleGm.GetValue(id[0]); T s1 = scaleGm.GetValue(id[1]);
                T s2 = scaleGm.GetValue(id[2]); T s3 = scaleGm.GetValue(id[3]);
                T b0 = biasGm.GetValue(id[0]);  T b1 = biasGm.GetValue(id[1]);
                T b2 = biasGm.GetValue(id[2]);  T b3 = biasGm.GetValue(id[3]);
                // 3) Muls/Adds 成对、右→左（不可拆成全 Muls 再全 Adds）
                Muls(yLocal[lb[0]], xLocal[lb[0]], s0, (int)step[0]); Adds(yLocal[lb[0]], yLocal[lb[0]], b0, (int)step[0]);
                Muls(yLocal[lb[1]], xLocal[lb[1]], s1, (int)step[1]); Adds(yLocal[lb[1]], yLocal[lb[1]], b1, (int)step[1]);
                Muls(yLocal[lb[2]], xLocal[lb[2]], s2, (int)step[2]); Adds(yLocal[lb[2]], yLocal[lb[2]], b2, (int)step[2]);
                Muls(yLocal[lb[3]], xLocal[lb[3]], s3, (int)step[3]); Adds(yLocal[lb[3]], yLocal[lb[3]], b3, (int)step[3]);
            } else {
                // 尾部 1~3 块（每 tile 最多一次）
                for(int j = 0; j < cnt; ++j) {
                    T sv = scaleGm.GetValue(id[j]);
                    T bv = biasGm.GetValue(id[j]);
                    Muls(yLocal[lb[j]], xLocal[lb[j]], sv, (int)step[j]);
                    Adds(yLocal[lb[j]], yLocal[lb[j]], bv, (int)step[j]);
                }
            }
        }

        inQueueX.FreeTensor(xLocal);
        outQueueY.EnQue(yLocal);
        yLocal = outQueueY.DeQue<T>();

        if(cur % 32) {
            DataCopyExtParams copyParams{1, (uint32_t)(cur * sizeof(T)), 0, 0, 0};
            DataCopyPad(outputGm[base_offset], yLocal, copyParams);
        } else {
            DataCopy(outputGm[base_offset], yLocal, cur);
        }
        outQueueY.FreeTensor(yLocal);
    }
}

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueS; 
    TQue<QuePosition::VECIN, 1> inQueueB;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    
    GlobalTensor<T> inputGm, outputGm, scaleGm, biasGm;
    uint32_t m, n, k, total_size;
    uint32_t blockidx, blocknum, mode;
    uint32_t tile_elements, elements_per_32b;
    int flag;
};


class KernelScaleBf16 {
public:
    __aicore__ inline KernelScaleBf16() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, 
                               uint32_t m, uint32_t n, uint32_t k, uint32_t total, int flag) {
        this->m = m;
        this->n = n;
        this->k = k;
        this->total_size = total;
        
        this->tile_elements = 16384 / sizeof(bfloat16_t);
        this->elements_per_32b = 32 / sizeof(bfloat16_t);

        this->blocknum = GetBlockNum();
        this->blockidx = GetBlockIdx();

        if(k == 1) {
            if(n <= tile_elements) {
                this->mode = 2;
            } else if(n >= 40 * 4096) {
                this->mode = 3;
            } else {
                this->mode = 4;
            }
        } else {
            this->mode = 5;
        }
        this->flag = flag;

        inputGm.SetGlobalBuffer((__gm__ bfloat16_t*)input);
        outputGm.SetGlobalBuffer((__gm__ bfloat16_t*)output);
        scaleGm.SetGlobalBuffer((__gm__ bfloat16_t*)scale);
        biasGm.SetGlobalBuffer((__gm__ bfloat16_t*)bias);
        outputGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

        pipe.InitBuffer(inQueueX, 2, 16384);
        pipe.InitBuffer(inQueueS, 1, 16384); 
        pipe.InitBuffer(inQueueB, 1, 16384);
        pipe.InitBuffer(outQueueY, 2, 16384);
        pipe.InitBuffer(workBuf, 98304);
    }

    __aicore__ inline void Process() {
        if(mode == 2) {
            int start = (m * blockidx) / blocknum;
            int end = (m * (blockidx + 1)) / blocknum;
            Compute2(start, end);
        } else if(mode == 3) {
            int cache_line = 128;
            int lines = (n + cache_line - 1) / cache_line;
            int start = (lines * blockidx) / blocknum;
            start *= cache_line;
            int end = (lines * (blockidx + 1)) / blocknum;
            end = min(int(end * cache_line), int(n));
            Compute3(start, end);
        } else if(mode == 4) {
            int start = (m * blockidx) / blocknum;
            int end = (m * (blockidx + 1)) / blocknum;
            Compute4(start, end);
        } else {
            int cache_line = 256;
            int lines = (total_size + cache_line - 1) / cache_line;
            int start = (lines * blockidx) / blocknum;
            start *= cache_line;
            int end = (lines * (blockidx + 1)) / blocknum;
            end = min(int(end * cache_line), int(total_size));
            Compute5(start, end);
        }
    }

private:
    __aicore__ inline uint32_t AlignTo(uint32_t val, uint32_t align) {
        return (val + align - 1) & ~(align - 1);
    }

    __aicore__ inline void Compute2(uint32_t start, uint32_t end) { 
        if(start == end) return;

        LocalTensor<float> tempTensor = workBuf.Get<float>();
        LocalTensor<float> sLocal_32 = tempTensor;
        LocalTensor<float> bLocal_32 = tempTensor[tile_elements];
        LocalTensor<float> xyLocal_32 = tempTensor[2*tile_elements];

        LocalTensor<bfloat16_t> sLocal = inQueueS.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> bLocal = inQueueB.AllocTensor<bfloat16_t>();
        uint32_t align_n = AlignTo(n, 32);
        DataCopy(sLocal, scaleGm, align_n);
        DataCopy(bLocal, biasGm, align_n);

        inQueueS.EnQue(sLocal); 
        inQueueS.DeQue<bfloat16_t>();
        inQueueB.EnQue(bLocal); 
        inQueueB.DeQue<bfloat16_t>();

        Cast(sLocal_32, sLocal, RoundMode::CAST_NONE, align_n);
        Cast(bLocal_32, bLocal, RoundMode::CAST_NONE, align_n);

        inQueueS.FreeTensor(sLocal);
        inQueueB.FreeTensor(bLocal);

        LocalTensor<bfloat16_t> xLocal;
        LocalTensor<bfloat16_t> yLocal;

        int i, iter, cur_end;
        if(flag) {
            i = start;
            iter = 1;
            cur_end = end;
        } else {
            i = end - 1;
            iter = -1;
            cur_end = start - 1;
        }

        for(; i != cur_end; i+=iter) {
            xLocal = inQueueX.AllocTensor<bfloat16_t>();
            yLocal = outQueueY.AllocTensor<bfloat16_t>();

            DataCopy(xLocal, inputGm[i*n], align_n);

            inQueueX.EnQue(xLocal); 
            xLocal = inQueueX.DeQue<bfloat16_t>();

            Cast(xyLocal_32, xLocal, RoundMode::CAST_NONE, align_n);

            inQueueX.FreeTensor(xLocal);

            Mul(xyLocal_32, xyLocal_32, sLocal_32, align_n);
            Cast(yLocal, xyLocal_32, RoundMode::CAST_RINT, align_n);
            Cast(xyLocal_32, yLocal, RoundMode::CAST_NONE, align_n);
            Add(xyLocal_32, xyLocal_32, bLocal_32, align_n);
            Cast(yLocal, xyLocal_32, RoundMode::CAST_RINT, align_n);

            outQueueY.EnQue(yLocal); 
            yLocal = outQueueY.DeQue<bfloat16_t>();

            if(n % 32) {
                DataCopyExtParams copyParams{1, (uint32_t)(n * sizeof(bfloat16_t)), 0, 0, 0};
                DataCopyPad(outputGm[i*n], yLocal, copyParams);
            } else {
                DataCopy(outputGm[i*n], yLocal, n);
            }

            outQueueY.FreeTensor(yLocal);
        }
    }

    __aicore__ inline void Compute3(uint32_t start, uint32_t end) {
        if(start == end) return;

        LocalTensor<float> tempTensor = workBuf.Get<float>();
        LocalTensor<float> sLocal_32 = tempTensor;
        LocalTensor<float> bLocal_32 = tempTensor[tile_elements];
        LocalTensor<float> xyLocal_32 = tempTensor[2*tile_elements];

        int loop = (end - start + tile_elements - 1) / tile_elements;
        int base_offset = start, cur_n;
        for(int j = 0; j != loop; j++) {
            cur_n = min(end - base_offset, tile_elements);

            LocalTensor<bfloat16_t> sLocal = inQueueS.AllocTensor<bfloat16_t>();
            LocalTensor<bfloat16_t> bLocal = inQueueB.AllocTensor<bfloat16_t>();
            uint32_t align_n = AlignTo(cur_n, 32);
            DataCopy(sLocal, scaleGm[base_offset], align_n);
            DataCopy(bLocal, biasGm[base_offset], align_n);

            inQueueS.EnQue(sLocal); 
            inQueueS.DeQue<bfloat16_t>();
            inQueueB.EnQue(bLocal); 
            inQueueB.DeQue<bfloat16_t>();

            Cast(sLocal_32, sLocal, RoundMode::CAST_NONE, align_n);
            Cast(bLocal_32, bLocal, RoundMode::CAST_NONE, align_n);

            inQueueS.FreeTensor(sLocal);
            inQueueB.FreeTensor(bLocal);

            LocalTensor<bfloat16_t> xLocal;
            LocalTensor<bfloat16_t> yLocal;

            int i, iter, cur_end;
            if(flag) {
                i = 0;
                iter = 1;
                cur_end = m;
            } else {
                i = m - 1;
                iter = -1;
                cur_end = -1;
            }

            for(; i != cur_end; i+=iter) {
                xLocal = inQueueX.AllocTensor<bfloat16_t>();
                yLocal = outQueueY.AllocTensor<bfloat16_t>();

                DataCopy(xLocal, inputGm[i*n+base_offset], align_n);

                inQueueX.EnQue(xLocal); 
                xLocal = inQueueX.DeQue<bfloat16_t>();

                Cast(xyLocal_32, xLocal, RoundMode::CAST_NONE, align_n);

                inQueueX.FreeTensor(xLocal);

                Mul(xyLocal_32, xyLocal_32, sLocal_32, align_n);
                Cast(yLocal, xyLocal_32, RoundMode::CAST_RINT, align_n);
                Cast(xyLocal_32, yLocal, RoundMode::CAST_NONE, align_n);
                Add(xyLocal_32, xyLocal_32, bLocal_32, align_n);
                Cast(yLocal, xyLocal_32, RoundMode::CAST_RINT, align_n);

                outQueueY.EnQue(yLocal); 
                yLocal = outQueueY.DeQue<bfloat16_t>();

                if(cur_n % 32) {
                    DataCopyExtParams copyParams{1, (uint32_t)(cur_n * sizeof(bfloat16_t)), 0, 0, 0};
                    DataCopyPad(outputGm[i*n+base_offset], yLocal, copyParams);
                } else {
                    DataCopy(outputGm[i*n+base_offset], yLocal, cur_n);
                }

                outQueueY.FreeTensor(yLocal);
            }
            base_offset += cur_n;
        }
    }

    __aicore__ inline void Compute4(uint32_t start, uint32_t end) {
        if(start == end) return;

        LocalTensor<float> tempTensor = workBuf.Get<float>();
        LocalTensor<float> sLocal_32 = tempTensor;
        LocalTensor<float> bLocal_32 = tempTensor[tile_elements];
        LocalTensor<float> xyLocal_32 = tempTensor[2*tile_elements];

        int loop = (n + tile_elements - 1) / tile_elements;
        int base_offset = 0, cur_n;
        for(int j = 0; j != loop; j++) {
            cur_n = min(n - base_offset, tile_elements);

            LocalTensor<bfloat16_t> sLocal = inQueueS.AllocTensor<bfloat16_t>();
            LocalTensor<bfloat16_t> bLocal = inQueueB.AllocTensor<bfloat16_t>();
            uint32_t align_n = AlignTo(cur_n, 32);
            DataCopy(sLocal, scaleGm[base_offset], align_n);
            DataCopy(bLocal, biasGm[base_offset], align_n);

            inQueueS.EnQue(sLocal); 
            inQueueS.DeQue<bfloat16_t>();
            inQueueB.EnQue(bLocal); 
            inQueueB.DeQue<bfloat16_t>();

            Cast(sLocal_32, sLocal, RoundMode::CAST_NONE, align_n);
            Cast(bLocal_32, bLocal, RoundMode::CAST_NONE, align_n);

            inQueueS.FreeTensor(sLocal);
            inQueueB.FreeTensor(bLocal);

            LocalTensor<bfloat16_t> xLocal;
            LocalTensor<bfloat16_t> yLocal;

            int i, iter, cur_end;
            if(flag) {
                i = start;
                iter = 1;
                cur_end = end;
            } else {
                i = end - 1;
                iter = -1;
                cur_end = start - 1;
            }

            for(; i != cur_end; i+=iter) {
                xLocal = inQueueX.AllocTensor<bfloat16_t>();
                yLocal = outQueueY.AllocTensor<bfloat16_t>();

                DataCopy(xLocal, inputGm[i*n+base_offset], align_n);

                inQueueX.EnQue(xLocal); 
                xLocal = inQueueX.DeQue<bfloat16_t>();

                Cast(xyLocal_32, xLocal, RoundMode::CAST_NONE, align_n);

                inQueueX.FreeTensor(xLocal);

                Mul(xyLocal_32, xyLocal_32, sLocal_32, align_n);
                Cast(yLocal, xyLocal_32, RoundMode::CAST_RINT, align_n);
                Cast(xyLocal_32, yLocal, RoundMode::CAST_NONE, align_n);
                Add(xyLocal_32, xyLocal_32, bLocal_32, align_n);
                Cast(yLocal, xyLocal_32, RoundMode::CAST_RINT, align_n);

                outQueueY.EnQue(yLocal); 
                yLocal = outQueueY.DeQue<bfloat16_t>();

                if(cur_n % 32) {
                    DataCopyExtParams copyParams{1, (uint32_t)(cur_n * sizeof(bfloat16_t)), 0, 0, 0};
                    DataCopyPad(outputGm[i*n+base_offset], yLocal, copyParams);
                } else {
                    DataCopy(outputGm[i*n+base_offset], yLocal, cur_n);
                }

                outQueueY.FreeTensor(yLocal);
            }
            base_offset += cur_n;
        }
    }

    __aicore__ inline void Compute5(uint32_t start, uint32_t end) {
        if(start == end) return;

        LocalTensor<float> tempTensor = workBuf.Get<float>();
        LocalTensor<float> xLocal_32 = tempTensor;
        LocalTensor<float> yLocal_32 = tempTensor[tile_elements];

        int loop = (end - start + tile_elements - 1) / tile_elements;
        int base_offset = start;

        LocalTensor<bfloat16_t> xLocal;
        LocalTensor<bfloat16_t> yLocal;

        for(int i = 0; i != loop; i++) {
            int cur = min(tile_elements, end - base_offset);
            int align_cur = AlignTo(cur, 32);

            xLocal = inQueueX.AllocTensor<bfloat16_t>();
            yLocal = outQueueY.AllocTensor<bfloat16_t>();

            DataCopy(xLocal, inputGm[base_offset], align_cur);

            inQueueX.EnQue(xLocal); 
            xLocal = inQueueX.DeQue<bfloat16_t>();

            Cast(xLocal_32, xLocal, RoundMode::CAST_NONE, align_cur);

            inQueueX.FreeTensor(xLocal);

            int curr_end = base_offset + cur;
            while(curr_end > base_offset) {
                uint32_t last_elem_idx = curr_end - 1;
                uint32_t k_block_idx = last_elem_idx / k;
                uint32_t k_block_start = k_block_idx * k;
                uint32_t left_bound = (k_block_start > base_offset) ? k_block_start : base_offset;
                uint32_t step = curr_end - left_bound;
                curr_end = left_bound;
                left_bound -= base_offset;
                uint32_t n_idx = k_block_idx % n;

                bfloat16_t s_val = scaleGm.GetValue(n_idx); 
                bfloat16_t b_val = biasGm.GetValue(n_idx);

                step += left_bound % 8;
                left_bound = int(left_bound / 8) * 8;

                Muls(yLocal_32[left_bound], xLocal_32[left_bound], ToFloat(s_val), step);
                Cast(yLocal, yLocal_32[left_bound], RoundMode::CAST_RINT, step);
                Cast(yLocal_32[left_bound], yLocal, RoundMode::CAST_NONE, step);
                Adds(yLocal_32[left_bound], yLocal_32[left_bound], ToFloat(b_val), step);
            }
            Cast(yLocal, yLocal_32, RoundMode::CAST_RINT, align_cur);

            outQueueY.EnQue(yLocal); 
            yLocal = outQueueY.DeQue<bfloat16_t>();

            if(cur % 32) {
                DataCopyExtParams copyParams{1, (uint32_t)(cur * sizeof(bfloat16_t)), 0, 0, 0};
                DataCopyPad(outputGm[base_offset], yLocal, copyParams);
            } else {
                DataCopy(outputGm[base_offset], yLocal, cur);
            }

            outQueueY.FreeTensor(yLocal);
            base_offset += cur;
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueS; 
    TQue<QuePosition::VECIN, 1> inQueueB;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<TPosition::VECCALC> workBuf;
    
    GlobalTensor<bfloat16_t> inputGm, outputGm, scaleGm, biasGm;
    uint32_t m, n, k, total_size;
    uint32_t blockidx, blocknum, mode;
    uint32_t tile_elements, elements_per_32b;
    int flag;
};

extern "C" __global__ __aicore__ void scale(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    GlobalTensor<int32_t> workGm;
    workGm.SetGlobalBuffer((__gm__ int32_t *)workspace);

    if (TILING_KEY_IS(1)) { 
        KernelScale<float> op;
        op.Init(input, scale, bias, output, tiling_data.m, tiling_data.n, tiling_data.k, tiling_data.total_size, workGm(0)%2);
        op.Process();
    } else if (TILING_KEY_IS(2)) { 
        KernelScale<half> op;
        op.Init(input, scale, bias, output, tiling_data.m, tiling_data.n, tiling_data.k, tiling_data.total_size, (workGm(0)+1)%2);
        op.Process();
    } else if (TILING_KEY_IS(3)) { 
        KernelScaleBf16 op;
        op.Init(input, scale, bias, output, tiling_data.m, tiling_data.n, tiling_data.k, tiling_data.total_size, workGm(0)%2);
        op.Process();
    }
    if(GetBlockIdx() == 0) workGm(0) += 1;
}