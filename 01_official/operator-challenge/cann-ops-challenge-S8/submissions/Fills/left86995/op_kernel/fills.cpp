#include "kernel_operator.h"

// ============================================================================
// Static Tensor Programming — Fills operator
// TILING_KEY(0): Small data (totalBytes <= 512), single core, one DataCopyPad
// TILING_KEY(1): Large data, multi-core, Duplicate once + loop DataCopy
// ============================================================================

class KernelFills {
public:
    __aicore__ inline KernelFills() {}

    // ---- TILING_KEY(0): small data (single core, no loop) ----
    __aicore__ inline void InitSmall(GM_ADDR output, const FillsTilingData &td)
    {
        outGm.SetGlobalBuffer((__gm__ uint32_t *)(output), (td.totalBytes + 3) >> 2);
        fillVal = td.fillBits32;
        smallBytes = td.totalBytes;
    }

    __aicore__ inline void ProcessSmall()
    {
        // Static tensor at UB address 0, 128 uint32 = 512B
        AscendC::LocalTensor<uint32_t> localBuf(AscendC::TPosition::VECCALC, 0, 128);
        AscendC::Duplicate(localBuf, fillVal, 128);

        // V→MTE3 sync (static tensor mode uses EVENT_ID constants)
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        // Unified DataCopyPad for tail
        AscendC::DataCopyExtParams cp{1, smallBytes, 0, 0, 0};
        AscendC::DataCopyPad(outGm, localBuf, cp);
    }

    // ---- TILING_KEY(1): large data (multi-core) ----
    __aicore__ inline void InitLarge(GM_ADDR output, const FillsTilingData &td)
    {
        uint32_t coreIdx = AscendC::GetBlockIdx();
        bool lastCore = (coreIdx == td.usedCoreNum - 1);

        uint32_t myElems;
        uint32_t gmElemOff;
        if (coreIdx < td.formerNum) {
            myElems = td.formerElems;
            gmElemOff = coreIdx * td.formerElems;
        } else {
            myElems = td.tailElems;
            gmElemOff = td.formerNum * td.formerElems + (coreIdx - td.formerNum) * td.tailElems;
        }
        if (lastCore) {
            myElems = (td.lastCoreBytes + 3) >> 2;
        }

        outGm.SetGlobalBuffer((__gm__ uint32_t *)(output) + gmElemOff, myElems);

        bufElems = td.bufferElems;
        fillVal = td.fillBits32;
        loopCount = myElems / bufElems;
        uint32_t remElems = myElems - loopCount * bufElems;

        if (remElems == 0) {
            tailBytes = 0;
        } else if (!lastCore) {
            // Non-last core: remElems is 512B-aligned
            tailBytes = remElems << 2;
        } else {
            // Last core: actual remaining bytes
            tailBytes = td.lastCoreBytes - loopCount * td.bufferLen;
        }
    }

    __aicore__ inline void ProcessLarge()
    {
        if (loopCount == 0 && tailBytes == 0) {
            return;
        }

        // Static tensor at UB address 0
        AscendC::LocalTensor<uint32_t> localBuf(AscendC::TPosition::VECCALC, 0, bufElems);

        // Duplicate fill value into entire buffer
        AscendC::Duplicate(localBuf, fillVal, bufElems);

        // V→MTE3 sync (static tensor mode uses EVENT_ID constants)
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        // Stream out full tiles
        uint32_t gmOff = 0;
        for (uint32_t i = 0; i < loopCount; i++) {
            AscendC::DataCopy(outGm[gmOff], localBuf, bufElems);
            gmOff += bufElems;
        }

        // Tail — unified DataCopyPad
        if (tailBytes > 0) {
            AscendC::DataCopyExtParams cp{1, tailBytes, 0, 0, 0};
            AscendC::DataCopyPad(outGm[gmOff], localBuf, cp);
        }
    }

private:
    AscendC::GlobalTensor<uint32_t> outGm;
    uint32_t bufElems;
    uint32_t fillVal;
    uint32_t loopCount;
    uint32_t tailBytes;
    uint32_t smallBytes;
};

extern "C" __global__ __aicore__ void fills(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::InitSocState();
    GET_TILING_DATA(tiling_data, tiling);

    KernelFills op;
    if (TILING_KEY_IS(0)) {
        op.InitSmall(output, tiling_data);
        op.ProcessSmall();
    } else if (TILING_KEY_IS(1)) {
        op.InitLarge(output, tiling_data);
        op.ProcessLarge();
    }
}
