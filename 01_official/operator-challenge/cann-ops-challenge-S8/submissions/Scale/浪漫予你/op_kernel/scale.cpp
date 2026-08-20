#include "kernel_operator.h"

using namespace AscendC;


#define MIN(a, b) ((a) < (b) ? (a) : (b))

template<typename T>
class KernelScale {
public:
    static constexpr uint32_t BUFFER_NUM = 2;
    static constexpr uint32_t TILE_BYTES = 1024;
    static constexpr bool IS_BF16 = std::is_same_v<T, bfloat16_t>;
    static constexpr uint32_t CALC_BUF_NUM = IS_BF16 ? 4 : 1;

    static constexpr uint32_t TYPE_BYTES =
        std::is_same_v<T, float> ? 4 :
        (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) ? 2 : 1;

    static constexpr uint32_t VEC_TILE_SIZE = TILE_BYTES / TYPE_BYTES;

    __aicore__ inline KernelScale() {}

    __aicore__ inline void Init(
        GM_ADDR input,
        GM_ADDR scale,
        GM_ADDR bias,
        GM_ADDR output,
        TPipe* pipeIn,
        uint32_t batch_size,
        uint32_t length,
        uint32_t batch_size_vec,
        uint32_t has_bias)
    {
        this->pipe = pipeIn;
        this->batch_size = batch_size;
        this->length = length;
        this->batch_size_vec = batch_size_vec;
        this->has_bias = has_bias;

        input_gm.SetGlobalBuffer((__gm__ T*)input);
        scale_gm.SetGlobalBuffer((__gm__ T*)scale);
        output_gm.SetGlobalBuffer((__gm__ T*)output);

        if (this->has_bias) {
            bias_gm.SetGlobalBuffer((__gm__ T*)bias);
        }

        vec_tile_size = MIN(VEC_TILE_SIZE, batch_size_vec);
        if (vec_tile_size == 0) {
            vec_tile_size = 1;
        }

        pipe->InitBuffer(in_queue, BUFFER_NUM, vec_tile_size * sizeof(T));
        pipe->InitBuffer(out_queue, BUFFER_NUM, vec_tile_size * sizeof(T));
        pipe->InitBuffer(scale_queue, BUFFER_NUM, sizeof(T));

        if (this->has_bias) {
            pipe->InitBuffer(bias_queue, BUFFER_NUM, sizeof(T));
        }

        InitCalcBuffer();
    }

    __aicore__ inline void Process()
    {
        uint32_t block_idx = GetBlockIdx();
        uint32_t block_num = GetBlockNum();

        uint32_t len_per_core = (length + block_num - 1) / block_num;
        uint32_t len_start = block_idx * len_per_core;

        if (len_start >= length) {
            return;
        }

        uint32_t len_end = MIN(len_start + len_per_core, length);

        for (uint32_t l_idx = len_start; l_idx < len_end; ++l_idx) {
            if (has_bias) {
                ProcessLengthWithBias(l_idx);
            } else {
                ProcessLengthNoBias(l_idx);
            }
        }
    }

private:
    __aicore__ inline void InitCalcBuffer()
    {
        if constexpr (IS_BF16) {
            pipe->InitBuffer(calc_buf[0], vec_tile_size * sizeof(float));
            pipe->InitBuffer(calc_buf[1], vec_tile_size * sizeof(float));
            pipe->InitBuffer(calc_buf[2], sizeof(float));

            if (has_bias) {
                pipe->InitBuffer(calc_buf[3], sizeof(float));
            }
        }
    }

    __aicore__ inline void ProcessLengthWithBias(uint32_t l_idx)
    {
        CopyScaleBias(l_idx);

        LocalTensor<T> scale_local = scale_queue.DeQue<T>();
        LocalTensor<T> bias_local = bias_queue.DeQue<T>();

        if constexpr (IS_BF16) {
            LocalTensor<float> scale_fp32 = calc_buf[2].template Get<float>();
            LocalTensor<float> bias_fp32 = calc_buf[3].template Get<float>();

            Cast(scale_fp32, scale_local, RoundMode::CAST_NONE, 1);
            Cast(bias_fp32, bias_local, RoundMode::CAST_NONE, 1);

            float scale_val = scale_fp32.GetValue(0);
            float bias_val = bias_fp32.GetValue(0);

            for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                for (uint32_t v_base = 0; v_base < batch_size_vec; v_base += vec_tile_size) {
                    uint32_t cur_vec = MIN(vec_tile_size, batch_size_vec - v_base);
                    uint32_t gm_offset = GetBaseOffset(batch_idx, l_idx, v_base);

                    CopyIn(gm_offset, cur_vec);

                    LocalTensor<T> in_local = in_queue.DeQue<T>();
                    LocalTensor<T> out_local = out_queue.AllocTensor<T>();

                    LocalTensor<float> in_fp32 = calc_buf[0].template Get<float>();
                    LocalTensor<float> result_fp32 = calc_buf[1].template Get<float>();

                    Cast(in_fp32, in_local, RoundMode::CAST_NONE, cur_vec);
                    Muls(result_fp32, in_fp32, scale_val, cur_vec);
                    Adds(result_fp32, result_fp32, bias_val, cur_vec);
                    Cast(out_local, result_fp32, RoundMode::CAST_RINT, cur_vec);

                    in_queue.FreeTensor(in_local);
                    out_queue.EnQue(out_local);
                    CopyOut(gm_offset, cur_vec);
                }
            }
        } else {
            T scale_val = scale_local.GetValue(0);
            T bias_val = bias_local.GetValue(0);

            for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                for (uint32_t v_base = 0; v_base < batch_size_vec; v_base += vec_tile_size) {
                    uint32_t cur_vec = MIN(vec_tile_size, batch_size_vec - v_base);
                    uint32_t gm_offset = GetBaseOffset(batch_idx, l_idx, v_base);

                    CopyIn(gm_offset, cur_vec);

                    LocalTensor<T> in_local = in_queue.DeQue<T>();
                    LocalTensor<T> out_local = out_queue.AllocTensor<T>();

                    Muls(out_local, in_local, scale_val, cur_vec);
                    Adds(out_local, out_local, bias_val, cur_vec);

                    in_queue.FreeTensor(in_local);
                    out_queue.EnQue(out_local);
                    CopyOut(gm_offset, cur_vec);
                }
            }
        }

        scale_queue.FreeTensor(scale_local);
        bias_queue.FreeTensor(bias_local);
    }

    __aicore__ inline void ProcessLengthNoBias(uint32_t l_idx)
    {
        CopyScale(l_idx);

        LocalTensor<T> scale_local = scale_queue.DeQue<T>();

        if constexpr (IS_BF16) {
            LocalTensor<float> scale_fp32 = calc_buf[2].template Get<float>();
            Cast(scale_fp32, scale_local, RoundMode::CAST_NONE, 1);

            float scale_val = scale_fp32.GetValue(0);

            for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                for (uint32_t v_base = 0; v_base < batch_size_vec; v_base += vec_tile_size) {
                    uint32_t cur_vec = MIN(vec_tile_size, batch_size_vec - v_base);
                    uint32_t gm_offset = GetBaseOffset(batch_idx, l_idx, v_base);

                    CopyIn(gm_offset, cur_vec);

                    LocalTensor<T> in_local = in_queue.DeQue<T>();
                    LocalTensor<T> out_local = out_queue.AllocTensor<T>();

                    LocalTensor<float> in_fp32 = calc_buf[0].template Get<float>();
                    LocalTensor<float> result_fp32 = calc_buf[1].template Get<float>();

                    Cast(in_fp32, in_local, RoundMode::CAST_NONE, cur_vec);
                    Muls(result_fp32, in_fp32, scale_val, cur_vec);
                    Cast(out_local, result_fp32, RoundMode::CAST_RINT, cur_vec);

                    in_queue.FreeTensor(in_local);
                    out_queue.EnQue(out_local);
                    CopyOut(gm_offset, cur_vec);
                }
            }
        } else {
            T scale_val = scale_local.GetValue(0);

            for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                for (uint32_t v_base = 0; v_base < batch_size_vec; v_base += vec_tile_size) {
                    uint32_t cur_vec = MIN(vec_tile_size, batch_size_vec - v_base);
                    uint32_t gm_offset = GetBaseOffset(batch_idx, l_idx, v_base);

                    CopyIn(gm_offset, cur_vec);

                    LocalTensor<T> in_local = in_queue.DeQue<T>();
                    LocalTensor<T> out_local = out_queue.AllocTensor<T>();

                    Muls(out_local, in_local, scale_val, cur_vec);

                    in_queue.FreeTensor(in_local);
                    out_queue.EnQue(out_local);
                    CopyOut(gm_offset, cur_vec);
                }
            }
        }

        scale_queue.FreeTensor(scale_local);
    }

    __aicore__ inline uint32_t GetBaseOffset(
        uint32_t batch_idx,
        uint32_t l_idx,
        uint32_t v_base)
    {
        return batch_idx * length * batch_size_vec
             + l_idx * batch_size_vec
             + v_base;
    }

    __aicore__ inline void CopyIn(uint32_t gm_offset, uint32_t size)
    {
        LocalTensor<T> in_local = in_queue.AllocTensor<T>();

        DataCopyExtParams copy_params {
            1,
            static_cast<uint32_t>(size * sizeof(T)),
            0,
            0,
            0
        };

        DataCopyPadExtParams<T> pad_params {false, 0, 0, (T)0};
        DataCopyPad(in_local, input_gm[gm_offset], copy_params, pad_params);
        in_queue.EnQue(in_local);
    }

    __aicore__ inline void CopyOut(uint32_t gm_offset, uint32_t size)
    {
        LocalTensor<T> out_local = out_queue.DeQue<T>();

        DataCopyExtParams copy_params {
            1,
            static_cast<uint32_t>(size * sizeof(T)),
            0,
            0,
            0
        };

        DataCopyPad(output_gm[gm_offset], out_local, copy_params);
        out_queue.FreeTensor(out_local);
    }

    __aicore__ inline void CopyScale(uint32_t offset)
    {
        LocalTensor<T> scale_local = scale_queue.AllocTensor<T>();

        DataCopyExtParams copy_params {
            1,
            static_cast<uint32_t>(sizeof(T)),
            0,
            0,
            0
        };

        DataCopyPadExtParams<T> pad_params {false, 0, 0, (T)0};
        DataCopyPad(scale_local, scale_gm[offset], copy_params, pad_params);
        scale_queue.EnQue(scale_local);
    }

    __aicore__ inline void CopyScaleBias(uint32_t offset)
    {
        LocalTensor<T> scale_local = scale_queue.AllocTensor<T>();
        LocalTensor<T> bias_local = bias_queue.AllocTensor<T>();

        DataCopyExtParams copy_params {
            1,
            static_cast<uint32_t>(sizeof(T)),
            0,
            0,
            0
        };

        DataCopyPadExtParams<T> pad_params {false, 0, 0, (T)0};

        DataCopyPad(scale_local, scale_gm[offset], copy_params, pad_params);
        DataCopyPad(bias_local, bias_gm[offset], copy_params, pad_params);

        scale_queue.EnQue(scale_local);
        bias_queue.EnQue(bias_local);
    }

private:
    TPipe* pipe;

    TQue<TPosition::VECIN, BUFFER_NUM> in_queue;
    TQue<TPosition::VECIN, BUFFER_NUM> scale_queue;
    TQue<TPosition::VECIN, BUFFER_NUM> bias_queue;
    TQue<TPosition::VECOUT, BUFFER_NUM> out_queue;

    TBuf<TPosition::VECCALC> calc_buf[CALC_BUF_NUM];

    GlobalTensor<T> input_gm;
    GlobalTensor<T> scale_gm;
    GlobalTensor<T> bias_gm;
    GlobalTensor<T> output_gm;

    uint32_t batch_size;
    uint32_t length;
    uint32_t batch_size_vec;
    uint32_t has_bias;

    uint32_t vec_tile_size;
};
template<typename T>
class KernelScaleLengthTile {
public:
    static constexpr bool IS_BF16 = std::is_same_v<T, bfloat16_t>;
    static constexpr uint32_t BUFFER_NUM = 1;
    static constexpr uint32_t LENGTH_TILE_SIZE = 
        IS_BF16 ? 8192 : 
        (sizeof(T) == 4 ? 12288 : 24576);

    static constexpr uint32_t CALC_BUF_NUM = IS_BF16 ? 4 : 1;

    __aicore__ inline KernelScaleLengthTile() {}

    __aicore__ inline void Init(
        GM_ADDR input,
        GM_ADDR scale,
        GM_ADDR bias,
        GM_ADDR output,
        TPipe* pipeIn,
        uint32_t batch_size,
        uint32_t length,
        uint32_t has_bias)
    {
        this->pipe = pipeIn;
        this->batch_size = batch_size;
        this->length = length;
        this->has_bias = has_bias;

        input_gm.SetGlobalBuffer((__gm__ T*)input);
        scale_gm.SetGlobalBuffer((__gm__ T*)scale);
        output_gm.SetGlobalBuffer((__gm__ T*)output);

        if (this->has_bias) {
            bias_gm.SetGlobalBuffer((__gm__ T*)bias);
        }

        pipe->InitBuffer(in_queue, BUFFER_NUM, LENGTH_TILE_SIZE * sizeof(T));
        pipe->InitBuffer(out_queue, BUFFER_NUM, LENGTH_TILE_SIZE * sizeof(T));
        pipe->InitBuffer(scale_queue, BUFFER_NUM, LENGTH_TILE_SIZE * sizeof(T));

        if (this->has_bias) {
            pipe->InitBuffer(bias_queue, BUFFER_NUM, LENGTH_TILE_SIZE * sizeof(T));
        }

        InitCalcBuffer();
    }

    __aicore__ inline void Process()
    {
        uint32_t length_tiles = (length + LENGTH_TILE_SIZE - 1) / LENGTH_TILE_SIZE;
        uint32_t total_tasks = batch_size * length_tiles;

        if (has_bias) {
            for (uint32_t task_id = GetBlockIdx();
                 task_id < total_tasks;
                 task_id += GetBlockNum()) {
                uint32_t batch_idx = task_id / length_tiles;
                uint32_t tile_idx = task_id % length_tiles;

                uint32_t l_base = tile_idx * LENGTH_TILE_SIZE;
                uint32_t cur_len = MIN(LENGTH_TILE_SIZE, length - l_base);

                uint32_t gm_offset = batch_idx * length + l_base;

                ProcessSingleTileWithBias(gm_offset, l_base, cur_len);
            }
        } else {
            for (uint32_t task_id = GetBlockIdx();
                 task_id < total_tasks;
                 task_id += GetBlockNum()) {
                uint32_t batch_idx = task_id / length_tiles;
                uint32_t tile_idx = task_id % length_tiles;

                uint32_t l_base = tile_idx * LENGTH_TILE_SIZE;
                uint32_t cur_len = MIN(LENGTH_TILE_SIZE, length - l_base);

                uint32_t gm_offset = batch_idx * length + l_base;

                ProcessSingleTileNoBias(gm_offset, l_base, cur_len);
            }
        }
    }

private:
    __aicore__ inline void InitCalcBuffer()
    {
        if constexpr (IS_BF16) {
            pipe->InitBuffer(calc_buf[0], LENGTH_TILE_SIZE * sizeof(float));
            pipe->InitBuffer(calc_buf[1], LENGTH_TILE_SIZE * sizeof(float));
            pipe->InitBuffer(calc_buf[2], LENGTH_TILE_SIZE * sizeof(float));

            if (has_bias) {
                pipe->InitBuffer(calc_buf[3], LENGTH_TILE_SIZE * sizeof(float));
            }
        }
    }

    __aicore__ inline void ProcessSingleTileWithBias(
        uint32_t gm_offset,
        uint32_t scale_offset,
        uint32_t tile_size)
    {
        CopyIn(gm_offset, tile_size);
        CopyScaleBias(scale_offset, tile_size);

        LocalTensor<T> in_local = in_queue.DeQue<T>();
        LocalTensor<T> scale_local = scale_queue.DeQue<T>();
        LocalTensor<T> bias_local = bias_queue.DeQue<T>();
        LocalTensor<T> out_local = out_queue.AllocTensor<T>();

        if constexpr (IS_BF16) {
            LocalTensor<float> scale_fp32 = calc_buf[2].template Get<float>();
            LocalTensor<float> bias_fp32 = calc_buf[3].template Get<float>();

            Cast(scale_fp32, scale_local, RoundMode::CAST_NONE, tile_size);
            Cast(bias_fp32, bias_local, RoundMode::CAST_NONE, tile_size);

            ComputeWithBiasBf16Vector(
                in_local,
                scale_fp32,
                bias_fp32,
                out_local,
                tile_size);
        } else {
            ComputeWithBiasVectorDirect(
                in_local,
                scale_local,
                bias_local,
                out_local,
                tile_size);
        }

        in_queue.FreeTensor(in_local);
        scale_queue.FreeTensor(scale_local);
        bias_queue.FreeTensor(bias_local);

        out_queue.EnQue(out_local);
        CopyOut(gm_offset, tile_size);
    }

    __aicore__ inline void ProcessSingleTileNoBias(
        uint32_t gm_offset,
        uint32_t scale_offset,
        uint32_t tile_size)
    {
        CopyIn(gm_offset, tile_size);
        CopyScale(scale_offset, tile_size);

        LocalTensor<T> in_local = in_queue.DeQue<T>();
        LocalTensor<T> scale_local = scale_queue.DeQue<T>();
        LocalTensor<T> out_local = out_queue.AllocTensor<T>();

        if constexpr (IS_BF16) {
            LocalTensor<float> scale_fp32 = calc_buf[2].template Get<float>();

            Cast(scale_fp32, scale_local, RoundMode::CAST_NONE, tile_size);

            ComputeNoBiasBf16Vector(
                in_local,
                scale_fp32,
                out_local,
                tile_size);
        } else {
            ComputeNoBiasVectorDirect(
                in_local,
                scale_local,
                out_local,
                tile_size);
        }

        in_queue.FreeTensor(in_local);
        scale_queue.FreeTensor(scale_local);

        out_queue.EnQue(out_local);
        CopyOut(gm_offset, tile_size);
    }

    __aicore__ inline void ComputeWithBiasBf16Vector(
        LocalTensor<T>& in_local,
        LocalTensor<float>& scale_fp32,
        LocalTensor<float>& bias_fp32,
        LocalTensor<T>& out_local,
        uint32_t tile_size)
    {
        LocalTensor<float> in_fp32 = calc_buf[0].template Get<float>();
        LocalTensor<float> result_fp32 = calc_buf[1].template Get<float>();

        Cast(in_fp32, in_local, RoundMode::CAST_NONE, tile_size);

        DataCopy(result_fp32, bias_fp32, tile_size);
        MulAddDst(result_fp32, in_fp32, scale_fp32, tile_size);

        Cast(out_local, result_fp32, RoundMode::CAST_RINT, tile_size);
    }

    __aicore__ inline void ComputeNoBiasBf16Vector(
        LocalTensor<T>& in_local,
        LocalTensor<float>& scale_fp32,
        LocalTensor<T>& out_local,
        uint32_t tile_size)
    {
        LocalTensor<float> in_fp32 = calc_buf[0].template Get<float>();
        LocalTensor<float> result_fp32 = calc_buf[1].template Get<float>();

        Cast(in_fp32, in_local, RoundMode::CAST_NONE, tile_size);
        Mul(result_fp32, in_fp32, scale_fp32, tile_size);
        Cast(out_local, result_fp32, RoundMode::CAST_RINT, tile_size);
    }

    __aicore__ inline void ComputeWithBiasVectorDirect(
        LocalTensor<T>& in_local,
        LocalTensor<T>& scale_local,
        LocalTensor<T>& bias_local,
        LocalTensor<T>& out_local,
        uint32_t tile_size)
    {
        Mul(out_local, in_local, scale_local, tile_size);
        Add(out_local, out_local, bias_local, tile_size);
    }

    __aicore__ inline void ComputeNoBiasVectorDirect(
        LocalTensor<T>& in_local,
        LocalTensor<T>& scale_local,
        LocalTensor<T>& out_local,
        uint32_t tile_size)
    {
        Mul(out_local, in_local, scale_local, tile_size);
    }

    __aicore__ inline void CopyIn(uint32_t gm_offset, uint32_t size)
    {
        LocalTensor<T> in_local = in_queue.AllocTensor<T>();

        DataCopyExtParams copy_params {
            1,
            static_cast<uint32_t>(size * sizeof(T)),
            0,
            0,
            0
        };

        DataCopyPadExtParams<T> pad_params {false, 0, 0, (T)0};
        DataCopyPad(in_local, input_gm[gm_offset], copy_params, pad_params);
        in_queue.EnQue(in_local);
    }

    __aicore__ inline void CopyScale(uint32_t offset, uint32_t size)
    {
        LocalTensor<T> scale_local = scale_queue.AllocTensor<T>();

        DataCopyExtParams copy_params {
            1,
            static_cast<uint32_t>(size * sizeof(T)),
            0,
            0,
            0
        };

        DataCopyPadExtParams<T> pad_params {false, 0, 0, (T)0};
        DataCopyPad(scale_local, scale_gm[offset], copy_params, pad_params);
        scale_queue.EnQue(scale_local);
    }

    __aicore__ inline void CopyScaleBias(uint32_t offset, uint32_t size)
    {
        LocalTensor<T> scale_local = scale_queue.AllocTensor<T>();
        LocalTensor<T> bias_local = bias_queue.AllocTensor<T>();

        DataCopyExtParams copy_params {
            1,
            static_cast<uint32_t>(size * sizeof(T)),
            0,
            0,
            0
        };

        DataCopyPadExtParams<T> pad_params {false, 0, 0, (T)0};

        DataCopyPad(scale_local, scale_gm[offset], copy_params, pad_params);
        DataCopyPad(bias_local, bias_gm[offset], copy_params, pad_params);

        scale_queue.EnQue(scale_local);
        bias_queue.EnQue(bias_local);
    }

    __aicore__ inline void CopyOut(uint32_t gm_offset, uint32_t size)
    {
        LocalTensor<T> out_local = out_queue.DeQue<T>();

        DataCopyExtParams copy_params {
            1,
            static_cast<uint32_t>(size * sizeof(T)),
            0,
            0,
            0
        };

        DataCopyPad(output_gm[gm_offset], out_local, copy_params);
        out_queue.FreeTensor(out_local);
    }

private:
    TPipe* pipe;

    TQue<TPosition::VECIN, BUFFER_NUM> in_queue;
    TQue<TPosition::VECIN, BUFFER_NUM> scale_queue;
    TQue<TPosition::VECIN, BUFFER_NUM> bias_queue;
    TQue<TPosition::VECOUT, BUFFER_NUM> out_queue;

    TBuf<TPosition::VECCALC> calc_buf[CALC_BUF_NUM];

    GlobalTensor<T> input_gm;
    GlobalTensor<T> scale_gm;
    GlobalTensor<T> bias_gm;
    GlobalTensor<T> output_gm;

    uint32_t batch_size;
    uint32_t length;
    uint32_t has_bias;
};


template<typename T>
class KernelScaleBatchLengthVecTile {
public:
    static constexpr uint32_t BUFFER_NUM = 1;
    static constexpr uint32_t TILE_BYTES =
            std::is_same_v<T, bfloat16_t> ? 10 * 1024 :
            std::is_same_v<T, half>       ? 24 * 1024 :
            std::is_same_v<T, float>      ? 24 * 1024 :
                                            20 * 1024;

    static constexpr bool IS_BF16 = std::is_same_v<T, bfloat16_t>;
    static constexpr uint32_t CALC_BUF_NUM = IS_BF16 ? 4 : 1;

    static constexpr uint32_t TYPE_BYTES =
        std::is_same_v<T, float> ? 4 :
        (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) ? 2 : 1;

    static constexpr uint32_t TILE_SIZE = TILE_BYTES / TYPE_BYTES;

    static constexpr uint32_t VEC_TILE_ALIGN =
        std::is_same_v<T, float> ? 8 :
        (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) ? 16 : 32;

    __aicore__ inline KernelScaleBatchLengthVecTile() {}

    __aicore__ inline void Init(
        GM_ADDR input,
        GM_ADDR scale,
        GM_ADDR bias,
        GM_ADDR output,
        TPipe* pipeIn,
        uint32_t batch_size,
        uint32_t length,
        uint32_t batch_size_vec,
        uint32_t has_bias)
    {
        this->pipe = pipeIn;
        this->batch_size = batch_size;
        this->length = length;
        this->batch_size_vec = batch_size_vec;
        this->has_bias = has_bias;

        input_gm.SetGlobalBuffer((__gm__ T*)input);
        scale_gm.SetGlobalBuffer((__gm__ T*)scale);
        output_gm.SetGlobalBuffer((__gm__ T*)output);

        if (this->has_bias) {
            bias_gm.SetGlobalBuffer((__gm__ T*)bias);
        }

        const uint32_t vec_bytes = batch_size_vec * sizeof(T);
        uint32_t raw_vec_tile_num = TILE_BYTES / vec_bytes;

        vec_tile_num = (raw_vec_tile_num / VEC_TILE_ALIGN) * VEC_TILE_ALIGN;
        if (vec_tile_num == 0) {
            vec_tile_num = VEC_TILE_ALIGN;
        }

        pipe->InitBuffer(in_queue, BUFFER_NUM, TILE_SIZE * sizeof(T));
        pipe->InitBuffer(out_queue, BUFFER_NUM, TILE_SIZE * sizeof(T));

        pipe->InitBuffer(scale_queue, BUFFER_NUM, vec_tile_num * sizeof(T));

        if (this->has_bias) {
            pipe->InitBuffer(bias_queue, BUFFER_NUM, vec_tile_num * sizeof(T));
        }

        pipe->InitBuffer(offset_buf, TILE_SIZE * sizeof(uint32_t));

        if constexpr (IS_BF16) {
            pipe->InitBuffer(calc_buf[0], TILE_SIZE * sizeof(float));
            pipe->InitBuffer(calc_buf[1], TILE_SIZE * sizeof(float));
            pipe->InitBuffer(calc_buf[2], vec_tile_num * sizeof(float));

            if (has_bias) {
                pipe->InitBuffer(calc_buf[3], vec_tile_num * sizeof(float));
            }

            pipe->InitBuffer(scale_expand_buf, TILE_SIZE * sizeof(float));

            if (this->has_bias) {
                pipe->InitBuffer(bias_expand_buf, TILE_SIZE * sizeof(float));
            }

            BuildGatherOffset(sizeof(float));
        } else {
            pipe->InitBuffer(scale_expand_buf, TILE_SIZE * sizeof(T));

            if (this->has_bias) {
                pipe->InitBuffer(bias_expand_buf, TILE_SIZE * sizeof(T));
            }

            BuildGatherOffset(sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t block_idx = GetBlockIdx();
        const uint32_t block_num = GetBlockNum();
        const uint32_t l_step = block_num * vec_tile_num;

        for (uint32_t l_base = block_idx * vec_tile_num;
             l_base < length;
             l_base += l_step) {
            const uint32_t remain_len = length - l_base;
            const uint32_t cur_vec_num = MIN(vec_tile_num, remain_len);

            if (has_bias) {
                ProcessLengthTileWithBias(l_base, cur_vec_num);
            } else {
                ProcessLengthTileNoBias(l_base, cur_vec_num);
            }
        }
    }

private:
    __aicore__ inline void BuildGatherOffset(uint32_t src_type_bytes)
    {
        LocalTensor<uint32_t> offset_local = offset_buf.Get<uint32_t>();

        uint32_t src_offset = 0;
        uint32_t dst_base = 0;
        for (uint32_t i = 0; i < vec_tile_num; ++i) {
            uint32_t dst_idx = dst_base;
            for (uint32_t j = 0; j < batch_size_vec; ++j) {
                offset_local.SetValue(dst_idx, src_offset);
                ++dst_idx;
            }

            src_offset += src_type_bytes;
            dst_base += batch_size_vec;
        }
    }

    __aicore__ inline void ProcessLengthTileWithBias(
        uint32_t l_base,
        uint32_t cur_vec_num)
    {
        const uint32_t total_size = cur_vec_num * batch_size_vec;
        const uint32_t base_offset = l_base * batch_size_vec;
        const uint32_t batch_stride = length * batch_size_vec;

        CopyScaleBiasTile(l_base, cur_vec_num);

        LocalTensor<T> scale_local = scale_queue.DeQue<T>();
        LocalTensor<T> bias_local = bias_queue.DeQue<T>();
        LocalTensor<uint32_t> offset_local = offset_buf.Get<uint32_t>();

        if constexpr (IS_BF16) {
            LocalTensor<float> scale_fp32 = calc_buf[2].template Get<float>();
            LocalTensor<float> bias_fp32 = calc_buf[3].template Get<float>();
            LocalTensor<float> scale_expand = scale_expand_buf.template Get<float>();
            LocalTensor<float> bias_expand = bias_expand_buf.template Get<float>();

            Cast(scale_fp32, scale_local, RoundMode::CAST_NONE, cur_vec_num);
            Cast(bias_fp32, bias_local, RoundMode::CAST_NONE, cur_vec_num);

            Gather(scale_expand, scale_fp32, offset_local, (uint32_t)0, total_size);
            Gather(bias_expand, bias_fp32, offset_local, (uint32_t)0, total_size);

            for (uint32_t b = 0, gm_offset = base_offset;
                 b < batch_size;
                 ++b, gm_offset += batch_stride) {
                CopyVecTileIn(gm_offset, total_size);

                LocalTensor<T> in_local = in_queue.DeQue<T>();
                LocalTensor<T> out_local = out_queue.AllocTensor<T>();

                LocalTensor<float> in_fp32 = calc_buf[0].template Get<float>();
                LocalTensor<float> result_fp32 = calc_buf[1].template Get<float>();

                Cast(in_fp32, in_local, RoundMode::CAST_NONE, total_size);

                Mul(result_fp32, in_fp32, scale_expand, total_size);
                Add(result_fp32, result_fp32, bias_expand, total_size);

                Cast(out_local, result_fp32, RoundMode::CAST_RINT, total_size);

                in_queue.FreeTensor(in_local);
                out_queue.EnQue(out_local);
                CopyVecTileOut(gm_offset, total_size);
            }
        } else {
            LocalTensor<T> scale_expand = scale_expand_buf.template Get<T>();
            LocalTensor<T> bias_expand = bias_expand_buf.template Get<T>();

            Gather(scale_expand, scale_local, offset_local, (uint32_t)0, total_size);
            Gather(bias_expand, bias_local, offset_local, (uint32_t)0, total_size);

            for (uint32_t b = 0, gm_offset = base_offset;
                 b < batch_size;
                 ++b, gm_offset += batch_stride) {
                CopyVecTileIn(gm_offset, total_size);

                LocalTensor<T> in_local = in_queue.DeQue<T>();
                LocalTensor<T> out_local = out_queue.AllocTensor<T>();

                Mul(out_local, in_local, scale_expand, total_size);
                Add(out_local, out_local, bias_expand, total_size);

                in_queue.FreeTensor(in_local);
                out_queue.EnQue(out_local);
                CopyVecTileOut(gm_offset, total_size);
            }
        }

        scale_queue.FreeTensor(scale_local);
        bias_queue.FreeTensor(bias_local);
    }

    __aicore__ inline void ProcessLengthTileNoBias(
        uint32_t l_base,
        uint32_t cur_vec_num)
    {
        const uint32_t total_size = cur_vec_num * batch_size_vec;
        const uint32_t base_offset = l_base * batch_size_vec;
        const uint32_t batch_stride = length * batch_size_vec;

        CopyScaleTile(l_base, cur_vec_num);

        LocalTensor<T> scale_local = scale_queue.DeQue<T>();
        LocalTensor<uint32_t> offset_local = offset_buf.Get<uint32_t>();

        if constexpr (IS_BF16) {
            LocalTensor<float> scale_fp32 = calc_buf[2].template Get<float>();
            LocalTensor<float> scale_expand = scale_expand_buf.template Get<float>();

            Cast(scale_fp32, scale_local, RoundMode::CAST_NONE, cur_vec_num);

            Gather(scale_expand, scale_fp32, offset_local, (uint32_t)0, total_size);

            for (uint32_t b = 0, gm_offset = base_offset;
                 b < batch_size;
                 ++b, gm_offset += batch_stride) {
                CopyVecTileIn(gm_offset, total_size);

                LocalTensor<T> in_local = in_queue.DeQue<T>();
                LocalTensor<T> out_local = out_queue.AllocTensor<T>();

                LocalTensor<float> in_fp32 = calc_buf[0].template Get<float>();
                LocalTensor<float> result_fp32 = calc_buf[1].template Get<float>();

                Cast(in_fp32, in_local, RoundMode::CAST_NONE, total_size);

                Mul(result_fp32, in_fp32, scale_expand, total_size);

                Cast(out_local, result_fp32, RoundMode::CAST_RINT, total_size);

                in_queue.FreeTensor(in_local);
                out_queue.EnQue(out_local);
                CopyVecTileOut(gm_offset, total_size);
            }
        } else {
            LocalTensor<T> scale_expand = scale_expand_buf.template Get<T>();

            Gather(scale_expand, scale_local, offset_local, (uint32_t)0, total_size);

            for (uint32_t b = 0, gm_offset = base_offset;
                 b < batch_size;
                 ++b, gm_offset += batch_stride) {
                CopyVecTileIn(gm_offset, total_size);

                LocalTensor<T> in_local = in_queue.DeQue<T>();
                LocalTensor<T> out_local = out_queue.AllocTensor<T>();

                Mul(out_local, in_local, scale_expand, total_size);

                in_queue.FreeTensor(in_local);
                out_queue.EnQue(out_local);
                CopyVecTileOut(gm_offset, total_size);
            }
        }

        scale_queue.FreeTensor(scale_local);
    }

    __aicore__ inline void CopyVecTileIn(
        uint32_t gm_offset,
        uint32_t copy_size)
    {
        LocalTensor<T> in_local = in_queue.AllocTensor<T>();

        DataCopyExtParams copy_params {
            1,
            static_cast<uint32_t>(copy_size * sizeof(T)),
            0,
            0,
            0
        };

        DataCopyPadExtParams<T> pad_params {false, 0, 0, (T)0};

        DataCopyPad(in_local, input_gm[gm_offset], copy_params, pad_params);
        in_queue.EnQue(in_local);
    }

    __aicore__ inline void CopyVecTileOut(
        uint32_t gm_offset,
        uint32_t copy_size)
    {
        LocalTensor<T> out_local = out_queue.DeQue<T>();

        DataCopyExtParams copy_params {
            1,
            static_cast<uint32_t>(copy_size * sizeof(T)),
            0,
            0,
            0
        };

        DataCopyPad(output_gm[gm_offset], out_local, copy_params);
        out_queue.FreeTensor(out_local);
    }

    __aicore__ inline void CopyScaleTile(
        uint32_t scale_offset,
        uint32_t cur_vec_num)
    {
        LocalTensor<T> scale_local = scale_queue.AllocTensor<T>();

        DataCopyExtParams copy_params {
            1,
            static_cast<uint32_t>(cur_vec_num * sizeof(T)),
            0,
            0,
            0
        };

        DataCopyPadExtParams<T> pad_params {false, 0, 0, (T)0};

        DataCopyPad(scale_local, scale_gm[scale_offset], copy_params, pad_params);
        scale_queue.EnQue(scale_local);
    }

    __aicore__ inline void CopyScaleBiasTile(
        uint32_t scale_offset,
        uint32_t cur_vec_num)
    {
        LocalTensor<T> scale_local = scale_queue.AllocTensor<T>();
        LocalTensor<T> bias_local = bias_queue.AllocTensor<T>();

        DataCopyExtParams copy_params {
            1,
            static_cast<uint32_t>(cur_vec_num * sizeof(T)),
            0,
            0,
            0
        };

        DataCopyPadExtParams<T> pad_params {false, 0, 0, (T)0};

        DataCopyPad(scale_local, scale_gm[scale_offset], copy_params, pad_params);
        DataCopyPad(bias_local, bias_gm[scale_offset], copy_params, pad_params);

        scale_queue.EnQue(scale_local);
        bias_queue.EnQue(bias_local);
    }

private:
    TPipe* pipe;

    TQue<TPosition::VECIN, BUFFER_NUM> in_queue;
    TQue<TPosition::VECIN, BUFFER_NUM> scale_queue;
    TQue<TPosition::VECIN, BUFFER_NUM> bias_queue;
    TQue<TPosition::VECOUT, BUFFER_NUM> out_queue;

    TBuf<TPosition::VECCALC> calc_buf[CALC_BUF_NUM];

    TBuf<TPosition::VECCALC> scale_expand_buf;
    TBuf<TPosition::VECCALC> bias_expand_buf;
    TBuf<TPosition::VECCALC> offset_buf;

    GlobalTensor<T> input_gm;
    GlobalTensor<T> scale_gm;
    GlobalTensor<T> bias_gm;
    GlobalTensor<T> output_gm;

    uint32_t batch_size;
    uint32_t length;
    uint32_t batch_size_vec;
    uint32_t has_bias;

    uint32_t vec_tile_num;
};

template<typename T>
class KernelScaleBatchLengthVecAlignedTile {
public:
    static constexpr uint32_t BUFFER_NUM = 2;
    static constexpr uint32_t LENGTH_TILE_SIZE = 512;
    static constexpr bool IS_BF16 = std::is_same_v<T, bfloat16_t>;

    static constexpr uint32_t CALC_BUF_NUM = IS_BF16 ? 4 : 1;

    static constexpr uint32_t TYPE_BYTES =
        std::is_same_v<T, float> ? 4 :
        (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) ? 2 : 1;

    static constexpr uint32_t TILE_BYTES =
        std::is_same_v<T, bfloat16_t> ? (21 * 1024) :
        std::is_same_v<T, half>       ? (42 * 1024) :
        std::is_same_v<T, float>      ? (44 * 1024) :
                                        (45 * 1024);

    static constexpr uint32_t TILE_SIZE = TILE_BYTES / TYPE_BYTES;

    __aicore__ inline KernelScaleBatchLengthVecAlignedTile() {}

    __aicore__ inline void Init(
        GM_ADDR input,
        GM_ADDR scale,
        GM_ADDR bias,
        GM_ADDR output,
        TPipe* pipeIn,
        uint32_t batch_size,
        uint32_t length,
        uint32_t batch_size_vec,
        uint32_t has_bias)
    {
        this->pipe = pipeIn;
        this->batch_size = batch_size;
        this->length = length;
        this->batch_size_vec = batch_size_vec;
        this->has_bias = has_bias;

        input_gm.SetGlobalBuffer((__gm__ T*)input);
        scale_gm.SetGlobalBuffer((__gm__ T*)scale);
        output_gm.SetGlobalBuffer((__gm__ T*)output);

        if (this->has_bias) {
            bias_gm.SetGlobalBuffer((__gm__ T*)bias);
        }

        pipe->InitBuffer(in_queue, BUFFER_NUM, TILE_SIZE * sizeof(T));
        pipe->InitBuffer(out_queue, BUFFER_NUM, TILE_SIZE * sizeof(T));
        pipe->InitBuffer(scale_queue, BUFFER_NUM, LENGTH_TILE_SIZE * sizeof(T));

        if (this->has_bias) {
            pipe->InitBuffer(bias_queue, BUFFER_NUM, LENGTH_TILE_SIZE * sizeof(T));
        }

        InitCalcBuffer();
    }

    __aicore__ inline void Process()
    {
        uint32_t block_idx = GetBlockIdx();
        uint32_t block_num = GetBlockNum();

        uint32_t len_per_core = (length + block_num - 1) / block_num;
        uint32_t len_start = block_idx * len_per_core;

        if (len_start >= length) {
            return;
        }

        uint32_t len_end = MIN(len_start + len_per_core, length);

        if (has_bias) {
            for (uint32_t l_base = len_start; l_base < len_end; l_base += LENGTH_TILE_SIZE) {
                uint32_t cur_len = MIN(LENGTH_TILE_SIZE, len_end - l_base);
                ProcessLengthChunkWithBias(l_base, cur_len);
            }
        } else {
            for (uint32_t l_base = len_start; l_base < len_end; l_base += LENGTH_TILE_SIZE) {
                uint32_t cur_len = MIN(LENGTH_TILE_SIZE, len_end - l_base);
                ProcessLengthChunkNoBias(l_base, cur_len);
            }
        }
    }

private:
    __aicore__ inline void InitCalcBuffer()
    {
        if constexpr (IS_BF16) {
            pipe->InitBuffer(calc_buf[0], TILE_SIZE * sizeof(float));
            pipe->InitBuffer(calc_buf[1], TILE_SIZE * sizeof(float));
            pipe->InitBuffer(calc_buf[2], LENGTH_TILE_SIZE * sizeof(float));

            if (has_bias) {
                pipe->InitBuffer(calc_buf[3], LENGTH_TILE_SIZE * sizeof(float));
            }
        }
    }

    __aicore__ inline void ProcessLengthChunkWithBias(
        uint32_t l_base,
        uint32_t l_count)
    {
        CopyScaleBiasChunk(l_base, l_count);

        LocalTensor<T> scale_local = scale_queue.DeQue<T>();
        LocalTensor<T> bias_local = bias_queue.DeQue<T>();

        uint32_t cur_length_small = TILE_SIZE / batch_size_vec;
        if (cur_length_small == 0) {
            cur_length_small = 1;
        }

        const uint32_t batch_stride = length * batch_size_vec;

        if constexpr (IS_BF16) {
            LocalTensor<float> scale_fp32 = calc_buf[2].template Get<float>();
            LocalTensor<float> bias_fp32 = calc_buf[3].template Get<float>();

            Cast(scale_fp32, scale_local, RoundMode::CAST_NONE, l_count);
            Cast(bias_fp32, bias_local, RoundMode::CAST_NONE, l_count);

            // vec/l_off 外层，batch 内层：cur_len、total_size、scale/bias offset 等只计算一次。
            for (uint32_t l_off = 0; l_off < l_count; l_off += cur_length_small) {
                uint32_t cur_len = MIN(cur_length_small, l_count - l_off);
                uint32_t total_size = cur_len * batch_size_vec;
                uint32_t gm_offset = (l_base + l_off) * batch_size_vec;

                for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                    ProcessLengthTileWithBiasBf16(
                        gm_offset,
                        l_off,
                        cur_len,
                        total_size,
                        scale_fp32,
                        bias_fp32);
                    gm_offset += batch_stride;
                }
            }
        } else {
            // vec/l_off 外层，batch 内层：复用 l_off/cur_len/total_size，减少 batch 循环内重复 scalar 控制。
            for (uint32_t l_off = 0; l_off < l_count; l_off += cur_length_small) {
                uint32_t cur_len = MIN(cur_length_small, l_count - l_off);
                uint32_t total_size = cur_len * batch_size_vec;
                uint32_t gm_offset = (l_base + l_off) * batch_size_vec;

                for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                    ProcessLengthTileWithBiasDirect(
                        gm_offset,
                        l_off,
                        cur_len,
                        total_size,
                        scale_local,
                        bias_local);
                    gm_offset += batch_stride;
                }
            }
        }

        scale_queue.FreeTensor(scale_local);
        bias_queue.FreeTensor(bias_local);
    }

    __aicore__ inline void ProcessLengthChunkNoBias(
        uint32_t l_base,
        uint32_t l_count)
    {
        CopyScaleChunk(l_base, l_count);

        LocalTensor<T> scale_local = scale_queue.DeQue<T>();

        uint32_t cur_length_small = TILE_SIZE / batch_size_vec;
        if (cur_length_small == 0) {
            cur_length_small = 1;
        }

        const uint32_t batch_stride = length * batch_size_vec;

        if constexpr (IS_BF16) {
            LocalTensor<float> scale_fp32 = calc_buf[2].template Get<float>();
            Cast(scale_fp32, scale_local, RoundMode::CAST_NONE, l_count);

            for (uint32_t l_off = 0; l_off < l_count; l_off += cur_length_small) {
                uint32_t cur_len = MIN(cur_length_small, l_count - l_off);
                uint32_t total_size = cur_len * batch_size_vec;
                uint32_t gm_offset = (l_base + l_off) * batch_size_vec;

                for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                    ProcessLengthTileNoBiasBf16(
                        gm_offset,
                        l_off,
                        cur_len,
                        total_size,
                        scale_fp32);
                    gm_offset += batch_stride;
                }
            }
        } else {
            for (uint32_t l_off = 0; l_off < l_count; l_off += cur_length_small) {
                uint32_t cur_len = MIN(cur_length_small, l_count - l_off);
                uint32_t total_size = cur_len * batch_size_vec;
                uint32_t gm_offset = (l_base + l_off) * batch_size_vec;

                for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                    ProcessLengthTileNoBiasDirect(
                        gm_offset,
                        l_off,
                        cur_len,
                        total_size,
                        scale_local);
                    gm_offset += batch_stride;
                }
            }
        }

        scale_queue.FreeTensor(scale_local);
    }

    __aicore__ inline void ProcessLengthTileWithBiasDirect(
        uint32_t gm_offset,
        uint32_t l_off,
        uint32_t cur_len,
        uint32_t total_size,
        LocalTensor<T>& scale_local,
        LocalTensor<T>& bias_local)
    {
        CopyIn(gm_offset, total_size);

        LocalTensor<T> in_local = in_queue.DeQue<T>();
        LocalTensor<T> out_local = out_queue.AllocTensor<T>();

        uint32_t local_offset = 0;
        uint32_t scale_idx = l_off;
        for (uint32_t i = 0; i < cur_len; ++i) {
            T scale_val = scale_local.GetValue(scale_idx);
            T bias_val = bias_local.GetValue(scale_idx);

            Muls(out_local[local_offset], in_local[local_offset], scale_val, batch_size_vec);
            Adds(out_local[local_offset], out_local[local_offset], bias_val, batch_size_vec);

            local_offset += batch_size_vec;
            ++scale_idx;
        }

        in_queue.FreeTensor(in_local);
        out_queue.EnQue(out_local);
        CopyOut(gm_offset, total_size);
    }

    __aicore__ inline void ProcessLengthTileNoBiasDirect(
        uint32_t gm_offset,
        uint32_t l_off,
        uint32_t cur_len,
        uint32_t total_size,
        LocalTensor<T>& scale_local)
    {
        CopyIn(gm_offset, total_size);

        LocalTensor<T> in_local = in_queue.DeQue<T>();
        LocalTensor<T> out_local = out_queue.AllocTensor<T>();

        uint32_t local_offset = 0;
        uint32_t scale_idx = l_off;
        for (uint32_t i = 0; i < cur_len; ++i) {
            T scale_val = scale_local.GetValue(scale_idx);

            Muls(out_local[local_offset], in_local[local_offset], scale_val, batch_size_vec);

            local_offset += batch_size_vec;
            ++scale_idx;
        }

        in_queue.FreeTensor(in_local);
        out_queue.EnQue(out_local);
        CopyOut(gm_offset, total_size);
    }

    __aicore__ inline void ProcessLengthTileWithBiasBf16(
        uint32_t gm_offset,
        uint32_t l_off,
        uint32_t cur_len,
        uint32_t total_size,
        LocalTensor<float>& scale_fp32,
        LocalTensor<float>& bias_fp32)
    {
        CopyIn(gm_offset, total_size);

        LocalTensor<T> in_local = in_queue.DeQue<T>();
        LocalTensor<T> out_local = out_queue.AllocTensor<T>();
        LocalTensor<float> in_fp32 = calc_buf[0].template Get<float>();
        LocalTensor<float> result_fp32 = calc_buf[1].template Get<float>();

        Cast(in_fp32, in_local, RoundMode::CAST_NONE, total_size);

        uint32_t local_offset = 0;
        uint32_t scale_idx = l_off;
        for (uint32_t i = 0; i < cur_len; ++i) {
            float scale_val = scale_fp32.GetValue(scale_idx);
            float bias_val = bias_fp32.GetValue(scale_idx);

            Muls(result_fp32[local_offset], in_fp32[local_offset], scale_val, batch_size_vec);
            Adds(result_fp32[local_offset], result_fp32[local_offset], bias_val, batch_size_vec);

            local_offset += batch_size_vec;
            ++scale_idx;
        }

        Cast(out_local, result_fp32, RoundMode::CAST_RINT, total_size);

        in_queue.FreeTensor(in_local);
        out_queue.EnQue(out_local);
        CopyOut(gm_offset, total_size);
    }

    __aicore__ inline void ProcessLengthTileNoBiasBf16(
        uint32_t gm_offset,
        uint32_t l_off,
        uint32_t cur_len,
        uint32_t total_size,
        LocalTensor<float>& scale_fp32)
    {
        CopyIn(gm_offset, total_size);

        LocalTensor<T> in_local = in_queue.DeQue<T>();
        LocalTensor<T> out_local = out_queue.AllocTensor<T>();
        LocalTensor<float> in_fp32 = calc_buf[0].template Get<float>();
        LocalTensor<float> result_fp32 = calc_buf[1].template Get<float>();

        Cast(in_fp32, in_local, RoundMode::CAST_NONE, total_size);

        uint32_t local_offset = 0;
        uint32_t scale_idx = l_off;
        for (uint32_t i = 0; i < cur_len; ++i) {
            float scale_val = scale_fp32.GetValue(scale_idx);

            Muls(result_fp32[local_offset], in_fp32[local_offset], scale_val, batch_size_vec);

            local_offset += batch_size_vec;
            ++scale_idx;
        }

        Cast(out_local, result_fp32, RoundMode::CAST_RINT, total_size);

        in_queue.FreeTensor(in_local);
        out_queue.EnQue(out_local);
        CopyOut(gm_offset, total_size);
    }

    __aicore__ inline void CopyScaleChunk(
        uint32_t offset,
        uint32_t size)
    {
        LocalTensor<T> scale_local = scale_queue.AllocTensor<T>();

        DataCopyExtParams copy_params {
            1,
            static_cast<uint32_t>(size * sizeof(T)),
            0,
            0,
            0
        };

        DataCopyPadExtParams<T> pad_params {
            false,
            0,
            0,
            (T)0
        };

        DataCopyPad(scale_local, scale_gm[offset], copy_params, pad_params);
        scale_queue.EnQue(scale_local);
    }

    __aicore__ inline void CopyScaleBiasChunk(
        uint32_t offset,
        uint32_t size)
    {
        LocalTensor<T> scale_local = scale_queue.AllocTensor<T>();
        LocalTensor<T> bias_local = bias_queue.AllocTensor<T>();

        DataCopyExtParams copy_params {
            1,
            static_cast<uint32_t>(size * sizeof(T)),
            0,
            0,
            0
        };

        DataCopyPadExtParams<T> pad_params {
            false,
            0,
            0,
            (T)0
        };

        DataCopyPad(scale_local, scale_gm[offset], copy_params, pad_params);
        DataCopyPad(bias_local, bias_gm[offset], copy_params, pad_params);

        scale_queue.EnQue(scale_local);
        bias_queue.EnQue(bias_local);
    }

    __aicore__ inline void CopyIn(
        uint32_t gm_offset,
        uint32_t size)
    {
        LocalTensor<T> in_local = in_queue.AllocTensor<T>();

        DataCopyExtParams copy_params {
            1,
            static_cast<uint32_t>(size * sizeof(T)),
            0,
            0,
            0
        };

        DataCopyPadExtParams<T> pad_params {
            false,
            0,
            0,
            (T)0
        };

        DataCopyPad(in_local, input_gm[gm_offset], copy_params, pad_params);
        in_queue.EnQue(in_local);
    }

    __aicore__ inline void CopyOut(
        uint32_t gm_offset,
        uint32_t size)
    {
        LocalTensor<T> out_local = out_queue.DeQue<T>();

        DataCopyExtParams copy_params {
            1,
            static_cast<uint32_t>(size * sizeof(T)),
            0,
            0,
            0
        };

        DataCopyPad(output_gm[gm_offset], out_local, copy_params);
        out_queue.FreeTensor(out_local);
    }

private:
    TPipe* pipe;

    TQue<TPosition::VECIN, BUFFER_NUM> in_queue;
    TQue<TPosition::VECIN, BUFFER_NUM> scale_queue;
    TQue<TPosition::VECIN, BUFFER_NUM> bias_queue;
    TQue<TPosition::VECOUT, BUFFER_NUM> out_queue;

    TBuf<TPosition::VECCALC> calc_buf[CALC_BUF_NUM];

    GlobalTensor<T> input_gm;
    GlobalTensor<T> scale_gm;
    GlobalTensor<T> bias_gm;
    GlobalTensor<T> output_gm;

    uint32_t batch_size;
    uint32_t length;
    uint32_t batch_size_vec;
    uint32_t has_bias;
};

extern "C" __global__ __aicore__ void scale(
    GM_ADDR input,
    GM_ADDR scale,
    GM_ADDR bias,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(3)) {
        // batch_size_vec * sizeof(T) 是 32 字节对齐时走 aligned tiling3
        KernelScaleBatchLengthVecAlignedTile<DTYPE_INPUT> op;
        op.Init(
            input,
            scale,
            bias,
            output,
            &pipe,
            tiling_data.batch_size,
            tiling_data.length,
            tiling_data.batch_size_vec,
            tiling_data.has_bias);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        // 非 32 字节对齐，且 batch_size_vec < 1024 时走 tiling2
        KernelScaleBatchLengthVecTile<DTYPE_INPUT> op;
        op.Init(
            input,
            scale,
            bias,
            output,
            &pipe,
            tiling_data.batch_size,
            tiling_data.length,
            tiling_data.batch_size_vec,
            tiling_data.has_bias);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        // 保留原来的 length-only 实现，方便兼容旧 key；新的 tiling 逻辑默认不会发 key=1
        KernelScaleLengthTile<DTYPE_INPUT> op;
        op.Init(
            input,
            scale,
            bias,
            output,
            &pipe,
            tiling_data.batch_size,
            tiling_data.length,
            tiling_data.has_bias);
        op.Process();
    } else if (TILING_KEY_IS(0)) {
        // 通用 tiling0
        KernelScale<DTYPE_INPUT> op;
        op.Init(
            input,
            scale,
            bias,
            output,
            &pipe,
            tiling_data.batch_size,
            tiling_data.length,
            tiling_data.batch_size_vec,
            tiling_data.has_bias);
        op.Process();
    }
}