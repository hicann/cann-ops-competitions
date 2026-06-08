// Kernel side implementation for torch.erf / torch.special.erf.
#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t MAX_TILE_LENGTH = 8192;
constexpr uint32_t ALIGN_NUM = 32 / sizeof(float);

template <class DT_X>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length, uint32_t split_align, uint32_t use_direct) {
        this->use_direct_path = use_direct != 0;

        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), length);
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), length);

        if (this->use_direct_path) {
            this->offset = 0;
            this->block_length = length;
            this->tile_length = (length + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
            if (this->tile_length == 0) {
                this->tile_length = ALIGN_NUM;
            }
            pipe.InitBuffer(x_buf, this->tile_length * sizeof(DT_X));
            pipe.InitBuffer(y_buf, this->tile_length * sizeof(DT_X));
            pipe.InitBuffer(poly_buf, this->tile_length * sizeof(float));
            return;
        }

        uint32_t block_num = GetBlockNum();
        uint32_t block_idx = GetBlockIdx();
        uint32_t aligned_blocks = (length + split_align - 1) / split_align;
        uint32_t start_block = aligned_blocks * block_idx / block_num;
        uint32_t end_block = aligned_blocks * (block_idx + 1) / block_num;
        this->offset = start_block * split_align;
        uint32_t end = end_block * split_align;
        if (end > length) {
            end = length;
        }
        this->block_length = end > this->offset ? end - this->offset : 0;

        this->tile_length = this->block_length < MAX_TILE_LENGTH
            ? ((this->block_length + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM
            : MAX_TILE_LENGTH;
        if (this->tile_length == 0) {
            this->tile_length = ALIGN_NUM;
        }

        pipe.InitBuffer(x_queue, BUFFER_NUM, this->tile_length * sizeof(DT_X));
        pipe.InitBuffer(y_queue, BUFFER_NUM, this->tile_length * sizeof(DT_X));
        pipe.InitBuffer(poly_buf, this->tile_length * sizeof(float));
    }

    __aicore__ inline void Process() {
        if (this->use_direct_path) {
            if (this->block_length > 0) {
                ComputeDirect(this->offset, this->block_length);
            }
            return;
        }

        uint32_t loop_count = this->block_length / this->tile_length;
        uint32_t tail_count = this->block_length % this->tile_length;
        for (uint32_t i = 0; i < loop_count; ++i) {
            Compute(this->offset + i * this->tile_length, this->tile_length);
        }
        if (tail_count > 0) {
            Compute(this->offset + loop_count * this->tile_length, tail_count);
        }
    }

private:
    __aicore__ inline uint32_t AlignUpCount(uint32_t count) {
        return (count + ALIGN_NUM - 1) & ~(ALIGN_NUM - 1);
    }

    __aicore__ inline void EvalErf(LocalTensor<DT_X> y_local, LocalTensor<DT_X> x_local, uint32_t count) {
        LocalTensor<float> poly = poly_buf.Get<float>();

        Mins(y_local, x_local, static_cast<DT_X>(2.58f), count);
        Maxs(y_local, y_local, static_cast<DT_X>(-2.58f), count);

        Mul(x_local, y_local, y_local, count);
        Muls(poly, x_local, 0.0000083365511798f, count);
        Adds(poly, poly, -0.00024177864235f, count);
        Mul(poly, poly, x_local, count);
        Adds(poly, poly, 0.0030436666415f, count);
        Mul(poly, poly, x_local, count);
        Adds(poly, poly, -0.022274390069f, count);
        Mul(poly, poly, x_local, count);
        Adds(poly, poly, 0.10781285329f, count);
        Mul(poly, poly, x_local, count);
        Adds(poly, poly, -0.37394373302f, count);
        Mul(poly, poly, x_local, count);
        Adds(poly, poly, 1.1282205742f, count);
        Mul(y_local, y_local, poly, count);
    }

    __aicore__ inline void ComputeDirect(uint32_t gm_offset, uint32_t count) {
        DataCopyExtParams copy_params{
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(count * sizeof(DT_X)),
            0,
            0,
            0
        };
        DataCopyPadExtParams<DT_X> pad_params{false, 0, 0, static_cast<DT_X>(0)};

        LocalTensor<DT_X> x_local = x_buf.Get<DT_X>();
        LocalTensor<DT_X> y_local = y_buf.Get<DT_X>();
        bool aligned_copy = (count % ALIGN_NUM) == 0;
        if (aligned_copy) {
            DataCopy(x_local, x_gm[gm_offset], count);
        } else {
            DataCopyPad(x_local, x_gm[gm_offset], copy_params, pad_params);
        }

        int32_t event_id_mte2_to_v = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(event_id_mte2_to_v);
        WaitFlag<HardEvent::MTE2_V>(event_id_mte2_to_v);

        uint32_t compute_count = aligned_copy ? count : AlignUpCount(count);
        EvalErf(y_local, x_local, compute_count);

        int32_t event_id_v_to_mte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event_id_v_to_mte3);
        WaitFlag<HardEvent::V_MTE3>(event_id_v_to_mte3);

        if (aligned_copy) {
            DataCopy(y_gm[gm_offset], y_local, count);
        } else {
            DataCopyPad(y_gm[gm_offset], y_local, copy_params);
        }
    }

    __aicore__ inline void Compute(uint32_t gm_offset, uint32_t count) {
        DataCopyExtParams copy_params{
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(count * sizeof(DT_X)),
            0,
            0,
            0
        };
        DataCopyPadExtParams<DT_X> pad_params{false, 0, 0, static_cast<DT_X>(0)};

        LocalTensor<DT_X> x_local = x_queue.AllocTensor<DT_X>();
        bool aligned_copy = (count % ALIGN_NUM) == 0;
        if (aligned_copy) {
            DataCopy(x_local, x_gm[gm_offset], count);
        } else {
            DataCopyPad(x_local, x_gm[gm_offset], copy_params, pad_params);
        }
        x_queue.EnQue(x_local);

        x_local = x_queue.DeQue<DT_X>();
        LocalTensor<DT_X> y_local = y_queue.AllocTensor<DT_X>();

        uint32_t compute_count = aligned_copy ? count : AlignUpCount(count);
        EvalErf(y_local, x_local, compute_count);

        x_queue.FreeTensor(x_local);
        y_queue.EnQue(y_local);

        y_local = y_queue.DeQue<DT_X>();
        if (aligned_copy) {
            DataCopy(y_gm[gm_offset], y_local, count);
        } else {
            DataCopyPad(y_gm[gm_offset], y_local, copy_params);
        }
        y_queue.FreeTensor(y_local);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> x_queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> y_queue;
    TBuf<QuePosition::VECCALC> x_buf;
    TBuf<QuePosition::VECCALC> y_buf;
    TBuf<QuePosition::VECCALC> poly_buf;

    GlobalTensor<DT_X> x_gm;
    GlobalTensor<DT_X> y_gm;
    uint32_t offset;
    uint32_t block_length;
    uint32_t tile_length;
    bool use_direct_path;
};

extern "C" __global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    KernelErf<float> op;
    op.Init(x, y, tiling_data.length, tiling_data.split_align, tiling_data.use_direct);
    op.Process();
}
