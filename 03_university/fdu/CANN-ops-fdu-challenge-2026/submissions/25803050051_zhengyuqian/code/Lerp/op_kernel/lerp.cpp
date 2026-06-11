// Kernel侧核函数实现
#include "kernel_operator.h"

#include "lerp_tiling.h"
#include "tiling_key_lerp.h"

constexpr uint32_t ALIGN_BYTES = 32;

template <class DT_START>
class KernelLerp {
public:
    __aicore__ inline KernelLerp() {}

    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR y, uint32_t length,
        uint32_t tileLength, uint32_t perCoreLength, float weight) {
        uint32_t per_core_length = perCoreLength == 0 ? 1 : perCoreLength;
        this->tileLength = tileLength == 0 ? 1 : tileLength;
        this->weight = static_cast<DT_START>(weight);

        uint32_t start_offset = AscendC::GetBlockIdx() * per_core_length;
        uint32_t remain = start_offset >= length ? 0 : length - start_offset;
        this->coreLength = remain < per_core_length ? remain : per_core_length;
        if (this->coreLength == 0) {
            return;
        }

        startGm.SetGlobalBuffer((__gm__ DT_START *)start + start_offset, this->coreLength);
        endGm.SetGlobalBuffer((__gm__ DT_START *)end + start_offset, this->coreLength);
        yGm.SetGlobalBuffer((__gm__ DT_START *)y + start_offset, this->coreLength);
        pipe.InitBuffer(startBuf, this->tileLength * sizeof(DT_START));
        pipe.InitBuffer(endBuf, this->tileLength * sizeof(DT_START));
    }

    __aicore__ inline void Process() {
        if (this->coreLength == 0) {
            return;
        }
        uint32_t offset = 0;
        while (offset < this->coreLength) {
            uint32_t remain = this->coreLength - offset;
            uint32_t count = remain < this->tileLength ? remain : this->tileLength;
            ProcessTile(offset, count);
            offset += count;
            if (offset < this->coreLength) {
                Sync<AscendC::HardEvent::MTE3_MTE2>();
            }
        }
    }

private:
    template <AscendC::HardEvent event>
    __aicore__ inline void Sync() {
        event_t eventId = static_cast<event_t>(pipe.FetchEventID(event));
        AscendC::SetFlag<event>(eventId);
        AscendC::WaitFlag<event>(eventId);
        pipe.ReleaseEventID<event>(eventId);
    }

    __aicore__ inline void ProcessTile(uint32_t offset, uint32_t count) {
        AscendC::LocalTensor<DT_START> startLocal = startBuf.Get<DT_START>();
        AscendC::LocalTensor<DT_START> endLocal = endBuf.Get<DT_START>();
        uint32_t copyBytes = count * sizeof(DT_START);
        bool aligned = (copyBytes & (ALIGN_BYTES - 1)) == 0;

        if (aligned) {
            AscendC::DataCopy(startLocal, startGm[offset], count);
            AscendC::DataCopy(endLocal, endGm[offset], count);
        } else {
            AscendC::DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_START> padParams{false, 0, 0, static_cast<DT_START>(0)};
            AscendC::DataCopyPad(startLocal, startGm[offset], copyParams, padParams);
            AscendC::DataCopyPad(endLocal, endGm[offset], copyParams, padParams);
        }

        Sync<AscendC::HardEvent::MTE2_V>();
        AscendC::Sub(endLocal, endLocal, startLocal, static_cast<int32_t>(count));
        AscendC::Axpy(startLocal, endLocal, this->weight, static_cast<int32_t>(count));
        Sync<AscendC::HardEvent::V_MTE3>();

        if (aligned) {
            AscendC::DataCopy(yGm[offset], startLocal, count);
        } else {
            AscendC::DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
            AscendC::DataCopyPad(yGm[offset], startLocal, copyParams);
        }
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> startBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> endBuf;
    AscendC::GlobalTensor<DT_START> startGm;
    AscendC::GlobalTensor<DT_START> endGm;
    AscendC::GlobalTensor<DT_START> yGm;
    uint32_t tileLength = 1;
    uint32_t coreLength = 0;
    DT_START weight;
};

template <typename DT_START>
__global__ __aicore__ void lerp(GM_ADDR start, GM_ADDR end, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LerpTilingData);
    GET_TILING_DATA_WITH_STRUCT(LerpTilingData, tiling_data, tiling);
    KernelLerp<DT_START> op;
    op.Init(start, end, y, tiling_data.length, tiling_data.tileLength, tiling_data.perCoreLength, tiling_data.weight);
    op.Process();
}
