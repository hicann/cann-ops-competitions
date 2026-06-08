// Kernel侧核函数实现
// v37: 回退到 v29/v34 稳定版本（AscendC::Erf API）
//      手动多项式方向彻底放弃（v30~v36 全部失败，72.66% 固定错误率）
//      失败根因：judge CANN 环境中 Div/Reciprocal 行为异常，无法用于手动实现
#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

template <class DT_X>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, ErfTilingData tilingData, AscendC::TPipe* pipeIn) {
        this->totalLength = tilingData.totalLength;
        this->tileLength = tilingData.tileLength;

        uint32_t blockDim = AscendC::GetBlockNum();
        uint32_t blockLength = this->totalLength / blockDim;
        uint32_t offset = blockLength * AscendC::GetBlockIdx();

        this->myBlockLength = blockLength;
        if (AscendC::GetBlockIdx() == blockDim - 1) {
            this->myBlockLength = this->totalLength - offset;
        }

        this->tileNum = (this->myBlockLength + this->tileLength - 1) / this->tileLength;

        xGm.SetGlobalBuffer((__gm__ DT_X*)x + offset, this->myBlockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + offset, this->myBlockLength);

        pipe = pipeIn;
        if (this->tileNum == 1) {
            // 单 tile 路径（v21）：不分配 xTBuf/yTBuf，改为 LocalTensor 直构
        } else {
            // 多 tile 路径：4 个 TBuf 实现 ping/pong 双缓冲
            pipe->InitBuffer(xPingBuf, this->tileLength * sizeof(DT_X));
            pipe->InitBuffer(yPingBuf, this->tileLength * sizeof(DT_X));
            pipe->InitBuffer(xPongBuf, this->tileLength * sizeof(DT_X));
            pipe->InitBuffer(yPongBuf, this->tileLength * sizeof(DT_X));
        }
    }

    __aicore__ inline void Process() {
        if (this->tileNum == 1) {
            ProcessSingleTile();
        } else {
            ProcessMultiTile();
        }
    }

private:
    __aicore__ inline void ProcessSingleTile() {
        AscendC::LocalTensor<DT_X> xLocal(
            AscendC::TPosition::VECIN,
            0x0000,
            this->myBlockLength
        );
        AscendC::LocalTensor<DT_X> yLocal(
            AscendC::TPosition::VECOUT,
            0x8000,
            this->myBlockLength
        );

        uint32_t actualBytes = this->myBlockLength * sizeof(DT_X);
        if ((actualBytes & 31) == 0) {
            AscendC::DataCopy(xLocal, xGm, this->myBlockLength);
        } else {
            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = actualBytes;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;
            AscendC::DataCopyPadExtParams<DT_X> padParams;
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = 0;
            padParams.paddingValue = (DT_X)0;
            AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        AscendC::Erf<DT_X, false>(yLocal, xLocal, this->myBlockLength);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        if ((actualBytes & 31) == 0) {
            AscendC::DataCopy(yGm, yLocal, this->myBlockLength);
        } else {
            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = actualBytes;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;
            AscendC::DataCopyPad(yGm, yLocal, copyParams);
        }
    }

    __aicore__ inline void ProcessMultiTile() {
        AscendC::LocalTensor<DT_X> xPing = xPingBuf.Get<DT_X>();
        AscendC::LocalTensor<DT_X> yPing = yPingBuf.Get<DT_X>();
        AscendC::LocalTensor<DT_X> xPong = xPongBuf.Get<DT_X>();
        AscendC::LocalTensor<DT_X> yPong = yPongBuf.Get<DT_X>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);

        for (uint32_t loopIdx = 0; loopIdx < this->tileNum; loopIdx++) {
            int32_t eid = (loopIdx & 1);
            AscendC::LocalTensor<DT_X>& xCur = (eid == 0) ? xPing : xPong;
            AscendC::LocalTensor<DT_X>& yCur = (eid == 0) ? yPing : yPong;

            uint32_t offset = loopIdx * this->tileLength;
            uint32_t curLen = this->tileLength;
            if (offset + curLen > this->myBlockLength) {
                curLen = this->myBlockLength - offset;
            }
            uint32_t actualBytes = curLen * sizeof(DT_X);
            bool aligned = ((actualBytes & 31) == 0);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eid);
            if (aligned) {
                AscendC::DataCopy(xCur, xGm[offset], curLen);
            } else {
                AscendC::DataCopyExtParams cp;
                cp.blockCount = 1;
                cp.blockLen = actualBytes;
                cp.srcStride = 0;
                cp.dstStride = 0;
                cp.rsv = 0;
                AscendC::DataCopyPadExtParams<DT_X> pp;
                pp.isPad = true;
                pp.leftPadding = 0;
                pp.rightPadding = 0;
                pp.paddingValue = (DT_X)0;
                AscendC::DataCopyPad(xCur, xGm[offset], cp, pp);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eid);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eid);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eid);
            AscendC::Erf<DT_X, false>(yCur, xCur, curLen);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eid);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eid);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eid);
            if (aligned) {
                AscendC::DataCopy(yGm[offset], yCur, curLen);
            } else {
                AscendC::DataCopyExtParams cp;
                cp.blockCount = 1;
                cp.blockLen = actualBytes;
                cp.srcStride = 0;
                cp.dstStride = 0;
                cp.rsv = 0;
                AscendC::DataCopyPad(yGm[offset], yCur, cp);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eid);
        }

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
    }

private:
    AscendC::TPipe* pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xTBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yTBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN>   xPingBuf;
    AscendC::TBuf<AscendC::TPosition::VECOUT>  yPingBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN>   xPongBuf;
    AscendC::TBuf<AscendC::TPosition::VECOUT>  yPongBuf;

    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;

    uint32_t totalLength = 0;
    uint32_t tileLength = 0;
    uint32_t myBlockLength = 0;
    uint32_t tileNum = 0;
};

template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    AscendC::TPipe pipe;
    KernelErf<DT_X> op;
    op.Init(x, y, tiling_data, &pipe);
    op.Process();
}
