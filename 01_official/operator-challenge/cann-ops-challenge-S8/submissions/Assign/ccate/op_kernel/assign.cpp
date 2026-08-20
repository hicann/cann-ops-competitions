#include "kernel_operator.h"

constexpr int BUFFER_NUM = 2;
constexpr uint32_t UB_BLOCK_BYTES = 48 * 1024;

class KernelAssign {
public:
    __aicore__ inline KernelAssign() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR other,
                                 uint32_t bytesPerCore, uint32_t tailBytes,
                                 AscendC::TPipe* pipeIn)
    {
        this->pipe = pipeIn;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t usedCores = AscendC::GetBlockNum();
        uint64_t byteOffset = static_cast<uint64_t>(blockIdx) * bytesPerCore;
        this->myBytes = (blockIdx == usedCores - 1) ? tailBytes : bytesPerCore;

        srcGm.SetGlobalBuffer((__gm__ uint8_t*)(other + byteOffset), myBytes);
        dstGm.SetGlobalBuffer((__gm__ uint8_t*)(input + byteOffset), myBytes);

        // v9: 单次 InitBuffer 初始化绑定队列
        pipe->InitBuffer(queBind, BUFFER_NUM, UB_BLOCK_BYTES);
    }

    __aicore__ inline void Process()
    {
        if (myBytes == 0) { return; }
        uint32_t fullLoops = myBytes / UB_BLOCK_BYTES;
        uint32_t residual = myBytes % UB_BLOCK_BYTES;
        for (uint32_t i = 0; i < fullLoops; ++i) {
            CopyOnce(static_cast<uint64_t>(i) * UB_BLOCK_BYTES, UB_BLOCK_BYTES);
        }
        if (residual > 0) {
            CopyOnce(static_cast<uint64_t>(fullLoops) * UB_BLOCK_BYTES, residual);
        }
    }

private:
    __aicore__ inline void CopyOnce(uint64_t byteOff, uint32_t bytes)
    {
        // === MTE2: GM → UB (绑定队列) ===
        AscendC::LocalTensor<uint8_t> bindT = queBind.AllocTensor<uint8_t>();
        AscendC::DataCopyExtParams inParams;
        inParams.blockCount = 1;
        inParams.blockLen = bytes;
        inParams.srcStride = 0;
        inParams.dstStride = 0;
        inParams.rsv = 0;
        AscendC::DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(bindT, srcGm[byteOff], inParams, padParams);
        queBind.EnQue(bindT);
        bindT = queBind.DeQue<uint8_t>();
        AscendC::DataCopyExtParams outParams;
        outParams.blockCount = 1;
        outParams.blockLen = bytes;
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        outParams.rsv = 0;
        AscendC::DataCopyPad(dstGm[byteOff], bindT, outParams);
        queBind.FreeTensor(bindT);
    }

    AscendC::TPipe* pipe;
    // v9: 用 TQueBind 替换 inQ + outQ
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, BUFFER_NUM> queBind;
    AscendC::GlobalTensor<uint8_t> srcGm;
    AscendC::GlobalTensor<uint8_t> dstGm;
    uint32_t myBytes{0};
};

extern "C" __global__ __aicore__ void assign(GM_ADDR input, GM_ADDR other,
                                              GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(td, tiling);
    AscendC::TPipe pipe;
    KernelAssign op;
    op.Init(input, other, td.bytesPerCore, td.tailBytes, &pipe);
    op.Process();
}
