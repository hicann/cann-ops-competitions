#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BUFFER_NUM = 1;

constexpr uint32_t MODE_DIRECT = 0;
constexpr uint32_t MODE_STRIDED_ALIGNED = 1;
constexpr uint32_t MODE_STRIDED_PAD = 2;
constexpr uint32_t MODE_READ_ALL_ALIGNED = 3;
constexpr uint32_t MODE_COPY_ALL = 4;
constexpr uint32_t MODE_GATHER_INNER1 = 5;
constexpr uint32_t MODE_READ_ALL_PAD = 6;
}  // namespace

template<typename T, uint32_t MODE>
class KernelUnpack {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                                uint64_t outer, uint64_t num, uint64_t inner,
                                uint32_t coreNum, uint32_t tileOuter, uint32_t tileInner,
                                uint32_t tileOuterNum, uint32_t tileInnerNum,
                                uint32_t ubBufferSize, uint32_t ubOutBufferSize)
    {
        this->coreId = GetBlockIdx();
        this->outer = outer;
        this->num = num;
        this->inner = inner;
        this->coreNum = coreNum;
        this->tileOuter = tileOuter;
        this->tileInner = tileInner;
        this->tileOuterNum = tileOuterNum;
        this->tileInnerNum = tileInnerNum;
        this->totalElements = outer * num * inner;

        inGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input), totalElements);

        __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(output);
        uint64_t dataPtrOffset = *dataAddr;
        this->tensorPtrs = dataAddr + (dataPtrOffset >> 3);

        pipe.InitBuffer(inQueue, BUFFER_NUM, ubBufferSize);
        pipe.InitBuffer(outQueue, BUFFER_NUM, ubOutBufferSize);
        if (MODE == MODE_GATHER_INNER1) {
            uint32_t offsetBytes = static_cast<uint32_t>(
                ((static_cast<uint64_t>(tileOuter) * sizeof(uint32_t) + BLOCK_SIZE - 1) / BLOCK_SIZE) *
                BLOCK_SIZE);
            pipe.InitBuffer(offsetBuf, offsetBytes);
        }
    }

    __aicore__ inline void Process()
    {
        if (MODE == MODE_DIRECT) {
            ProcessDirect();
        } else if (MODE == MODE_STRIDED_ALIGNED) {
            ProcessStridedAligned();
        } else if (MODE == MODE_STRIDED_PAD) {
            ProcessStridedPad();
        } else if (MODE == MODE_READ_ALL_ALIGNED) {
            ProcessReadAllAligned();
        } else if (MODE == MODE_COPY_ALL) {
            ProcessCopyAll();
        } else if (MODE == MODE_GATHER_INNER1) {
            ProcessGatherInner1();
        } else {
            ProcessReadAllPad();
        }
    }

private:
    __aicore__ inline uint64_t Min(uint64_t lhs, uint64_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline __gm__ T* GetOutputPtr(uint64_t outputIdx) const
    {
        return reinterpret_cast<__gm__ T*>(*(this->tensorPtrs + outputIdx));
    }

    __aicore__ inline bool CanUseAlignedCopy(uint64_t inOffset, uint64_t outOffset,
                                             uint64_t copyElems) const
    {
        constexpr uint64_t elemsPerBlock = BLOCK_SIZE / sizeof(T);
        return (copyElems % elemsPerBlock == 0) &&
               (inOffset % elemsPerBlock == 0) &&
               (outOffset % elemsPerBlock == 0);
    }

    __aicore__ inline uint8_t GetRightPadding(uint32_t validBytes) const
    {
        uint32_t tailBytes = validBytes % BLOCK_SIZE;
        if (tailBytes == 0) {
            return 0;
        }
        return static_cast<uint8_t>((BLOCK_SIZE - tailBytes) / sizeof(T));
    }

    __aicore__ inline uint32_t AlignElements(uint64_t elems) const
    {
        constexpr uint64_t elemsPerBlock = BLOCK_SIZE / sizeof(T);
        return static_cast<uint32_t>(((elems + elemsPerBlock - 1) / elemsPerBlock) *
                                     elemsPerBlock);
    }

    __aicore__ inline void CopyInput(LocalTensor<T>& ub, uint64_t inOffset, uint64_t copyElems)
    {
        constexpr uint64_t elemsPerBlock = BLOCK_SIZE / sizeof(T);
        bool aligned = (copyElems % elemsPerBlock == 0) && (inOffset % elemsPerBlock == 0);
        if (aligned) {
            DataCopy(ub, inGm[inOffset], static_cast<uint32_t>(copyElems));
        } else {
            uint32_t copyBytes = static_cast<uint32_t>(copyElems * sizeof(T));
            DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, GetRightPadding(copyBytes), 0};
            DataCopyPad(ub, inGm[inOffset], copyParams, padParams);
        }
    }

    __aicore__ inline void CopyOutput(uint64_t outputIdx, LocalTensor<T>& ub,
                                      uint64_t outOffset, uint64_t copyElems)
    {
        outGm.SetGlobalBuffer(GetOutputPtr(outputIdx), outer * inner);

        constexpr uint64_t elemsPerBlock = BLOCK_SIZE / sizeof(T);
        bool aligned = (copyElems % elemsPerBlock == 0) && (outOffset % elemsPerBlock == 0);
        if (aligned) {
            DataCopy(outGm[outOffset], ub, static_cast<uint32_t>(copyElems));
        } else {
            uint32_t copyBytes = static_cast<uint32_t>(copyElems * sizeof(T));
            DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
            DataCopyPad(outGm[outOffset], ub, copyParams);
        }
    }

    __aicore__ inline void CopyContiguous(uint64_t outputIdx, uint64_t inOffset,
                                          uint64_t outOffset, uint64_t copyElems)
    {
        if (copyElems == 0) {
            return;
        }

        outGm.SetGlobalBuffer(GetOutputPtr(outputIdx), outer * inner);
        CopyContiguousCurrentOutput(inOffset, outOffset, copyElems);
    }

    __aicore__ inline void CopyContiguousCurrentOutput(uint64_t inOffset, uint64_t outOffset,
                                                       uint64_t copyElems)
    {
        if (copyElems == 0) {
            return;
        }
        LocalTensor<T> inUb = inQueue.AllocTensor<T>();
        bool aligned = CanUseAlignedCopy(inOffset, outOffset, copyElems);
        if (aligned) {
            DataCopy(inUb, inGm[inOffset], static_cast<uint32_t>(copyElems));
        } else {
            uint32_t copyBytes = static_cast<uint32_t>(copyElems * sizeof(T));
            DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, GetRightPadding(copyBytes), 0};
            DataCopyPad(inUb, inGm[inOffset], copyParams, padParams);
        }
        inQueue.EnQue(inUb);

        LocalTensor<T> srcUb = inQueue.DeQue<T>();
        LocalTensor<T> outUb = outQueue.AllocTensor<T>();
        DataCopy(outUb, srcUb, AlignElements(copyElems));
        inQueue.FreeTensor(srcUb);

        outQueue.EnQue(outUb);
        LocalTensor<T> dstUb = outQueue.DeQue<T>();
        if (aligned) {
            DataCopy(outGm[outOffset], dstUb, static_cast<uint32_t>(copyElems));
        } else {
            uint32_t copyBytes = static_cast<uint32_t>(copyElems * sizeof(T));
            DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
            DataCopyPad(outGm[outOffset], dstUb, copyParams);
        }
        outQueue.FreeTensor(dstUb);
    }

    __aicore__ inline void ProcessDirect()
    {
        uint64_t taskNum = outer * num * tileInnerNum;
        for (uint64_t taskId = coreId; taskId < taskNum; taskId += coreNum) {
            uint64_t innerTileIdx = taskId % tileInnerNum;
            uint64_t rowTask = taskId / tileInnerNum;
            uint64_t outputIdx = rowTask % num;
            uint64_t outerIdx = rowTask / num;
            uint64_t innerStart = innerTileIdx * static_cast<uint64_t>(tileInner);
            uint64_t copyElems = Min(static_cast<uint64_t>(tileInner), inner - innerStart);
            uint64_t inOffset = (outerIdx * num + outputIdx) * inner + innerStart;
            uint64_t outOffset = outerIdx * inner + innerStart;
            CopyContiguous(outputIdx, inOffset, outOffset, copyElems);
        }
    }

    __aicore__ inline void ProcessStridedAligned()
    {
        uint64_t taskNum = num * tileOuterNum;
        constexpr uint64_t elemsPerBlock = BLOCK_SIZE / sizeof(T);
        uint64_t innerBlocks = inner / elemsPerBlock;
        uint64_t srcStrideBlocks = (num - 1) * innerBlocks;

        for (uint64_t taskId = coreId; taskId < taskNum; taskId += coreNum) {
            uint64_t outputIdx = taskId / tileOuterNum;
            uint64_t outerTileIdx = taskId % tileOuterNum;
            uint64_t startOuter = outerTileIdx * static_cast<uint64_t>(tileOuter);
            uint64_t rows = Min(static_cast<uint64_t>(tileOuter), outer - startOuter);
            uint64_t inOffset = (startOuter * num + outputIdx) * inner;
            uint64_t outOffset = startOuter * inner;

            LocalTensor<T> ub = inQueue.AllocTensor<T>();
            DataCopyParams inParams{static_cast<uint16_t>(rows),
                                    static_cast<uint16_t>(innerBlocks),
                                    static_cast<uint16_t>(srcStrideBlocks),
                                    0};
            DataCopy(ub, inGm[inOffset], inParams);
            inQueue.EnQue(ub);

            outGm.SetGlobalBuffer(GetOutputPtr(outputIdx), outer * inner);
            LocalTensor<T> inUb = inQueue.DeQue<T>();
            LocalTensor<T> outUb = outQueue.AllocTensor<T>();
            uint64_t copyElems = rows * inner;
            DataCopy(outUb, inUb, static_cast<uint32_t>(copyElems));
            inQueue.FreeTensor(inUb);
            outQueue.EnQue(outUb);
            LocalTensor<T> dstUb = outQueue.DeQue<T>();
            DataCopy(outGm[outOffset], dstUb, static_cast<uint32_t>(copyElems));
            outQueue.FreeTensor(dstUb);
        }
    }

    __aicore__ inline void ProcessStridedPad()
    {
        uint64_t taskNum = num * tileOuterNum;
        uint32_t innerBytes = static_cast<uint32_t>(inner * sizeof(T));
        uint32_t packedInnerBytes = ((innerBytes + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
        uint32_t packedInnerElems = packedInnerBytes / sizeof(T);
        uint32_t srcStrideBytes = static_cast<uint32_t>((num - 1) * inner * sizeof(T));
        DataCopyPadExtParams<T> padParams{true, 0, GetRightPadding(innerBytes), 0};

        for (uint64_t taskId = coreId; taskId < taskNum; taskId += coreNum) {
            uint64_t outputIdx = taskId / tileOuterNum;
            uint64_t outerTileIdx = taskId % tileOuterNum;
            uint64_t startOuter = outerTileIdx * static_cast<uint64_t>(tileOuter);
            uint64_t rows = Min(static_cast<uint64_t>(tileOuter), outer - startOuter);
            uint64_t inOffset = (startOuter * num + outputIdx) * inner;
            uint64_t outOffset = startOuter * inner;

            LocalTensor<T> inUb = inQueue.AllocTensor<T>();
            DataCopyExtParams inParams{static_cast<uint16_t>(rows), innerBytes,
                                       srcStrideBytes, 0, 0};
            DataCopyPad(inUb, inGm[inOffset], inParams, padParams);
            inQueue.EnQue(inUb);

            LocalTensor<T> srcUb = inQueue.DeQue<T>();
            LocalTensor<T> outUb = outQueue.AllocTensor<T>();
            DataCopy(outUb, srcUb, static_cast<uint32_t>(rows * packedInnerElems));
            inQueue.FreeTensor(srcUb);

            outQueue.EnQue(outUb);
            outGm.SetGlobalBuffer(GetOutputPtr(outputIdx), outer * inner);
            LocalTensor<T> dstUb = outQueue.DeQue<T>();
            DataCopyExtParams outParams{static_cast<uint16_t>(rows), innerBytes, 0, 0, 0};
            DataCopyPad(outGm[outOffset], dstUb, outParams);
            outQueue.FreeTensor(dstUb);
        }
    }

    __aicore__ inline void ProcessReadAllAligned()
    {
        constexpr uint64_t elemsPerBlock = BLOCK_SIZE / sizeof(T);
        uint64_t innerBlocks = inner / elemsPerBlock;
        uint64_t srcStrideBlocks = (num - 1) * innerBlocks;

        for (uint64_t outerTileIdx = coreId; outerTileIdx < tileOuterNum; outerTileIdx += coreNum) {
            uint64_t startOuter = outerTileIdx * static_cast<uint64_t>(tileOuter);
            uint64_t rows = Min(static_cast<uint64_t>(tileOuter), outer - startOuter);
            uint64_t inOffset = startOuter * num * inner;
            uint64_t readElems = rows * num * inner;

            LocalTensor<T> ub = inQueue.AllocTensor<T>();
            DataCopy(ub, inGm[inOffset], static_cast<uint32_t>(readElems));
            inQueue.EnQue(ub);
            LocalTensor<T> inUb = inQueue.DeQue<T>();
            LocalTensor<T> outUb = outQueue.AllocTensor<T>();
            DataCopy(outUb, inUb, static_cast<uint32_t>(readElems));
            inQueue.FreeTensor(inUb);
            outQueue.EnQue(outUb);
            LocalTensor<T> dstUb = outQueue.DeQue<T>();

            DataCopyParams outParams{static_cast<uint16_t>(rows),
                                     static_cast<uint16_t>(innerBlocks),
                                     static_cast<uint16_t>(srcStrideBlocks),
                                     0};
            uint64_t outOffset = startOuter * inner;
            for (uint64_t outputIdx = 0; outputIdx < num; ++outputIdx) {
                outGm.SetGlobalBuffer(GetOutputPtr(outputIdx), outer * inner);
                DataCopy(outGm[outOffset], dstUb[outputIdx * inner], outParams);
            }
            outQueue.FreeTensor(dstUb);
        }
    }

    __aicore__ inline void ProcessReadAllPad()
    {
        uint32_t innerBytes = static_cast<uint32_t>(inner * sizeof(T));
        uint32_t rowBytes = static_cast<uint32_t>(num * inner * sizeof(T));
        uint32_t packedInnerBytes = ((innerBytes + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
        uint32_t srcStrideBytes = rowBytes - packedInnerBytes;

        for (uint64_t outerTileIdx = coreId; outerTileIdx < tileOuterNum; outerTileIdx += coreNum) {
            uint64_t startOuter = outerTileIdx * static_cast<uint64_t>(tileOuter);
            uint64_t rows = Min(static_cast<uint64_t>(tileOuter), outer - startOuter);
            uint64_t inOffset = startOuter * num * inner;
            uint64_t readElems = rows * num * inner;

            LocalTensor<T> inUb = inQueue.AllocTensor<T>();
            CopyInput(inUb, inOffset, readElems);
            inQueue.EnQue(inUb);

            LocalTensor<T> srcUb = inQueue.DeQue<T>();
            LocalTensor<T> outUb = outQueue.AllocTensor<T>();
            DataCopy(outUb, srcUb, AlignElements(readElems));
            inQueue.FreeTensor(srcUb);
            outQueue.EnQue(outUb);

            LocalTensor<T> dstUb = outQueue.DeQue<T>();
            DataCopyExtParams outParams{static_cast<uint16_t>(rows), innerBytes,
                                        srcStrideBytes, 0, 0};
            uint64_t outOffset = startOuter * inner;
            for (uint64_t outputIdx = 0; outputIdx < num; ++outputIdx) {
                outGm.SetGlobalBuffer(GetOutputPtr(outputIdx), outer * inner);
                DataCopyPad(outGm[outOffset], dstUb[outputIdx * inner], outParams);
            }
            outQueue.FreeTensor(dstUb);
        }
    }

    __aicore__ inline void ProcessCopyAll()
    {
        uint64_t total = outer * inner;
        outGm.SetGlobalBuffer(GetOutputPtr(0), total);

        for (uint64_t tileIdx = coreId; tileIdx < tileInnerNum; tileIdx += coreNum) {
            uint64_t offset = tileIdx * static_cast<uint64_t>(tileInner);
            uint64_t copyElems = Min(static_cast<uint64_t>(tileInner), total - offset);
            CopyContiguousCurrentOutput(offset, offset, copyElems);
        }
    }

    __aicore__ inline void ProcessGatherInner1()
    {
        LocalTensor<int32_t> offsetI32 = offsetBuf.Get<int32_t>();
        LocalTensor<uint32_t> offsetU32 = offsetI32.ReinterpretCast<uint32_t>();
        int32_t strideBytes = static_cast<int32_t>(num * sizeof(T));
        ArithProgression<int32_t>(offsetI32, 0, strideBytes, static_cast<int32_t>(tileOuter));

        for (uint64_t outerTileIdx = coreId; outerTileIdx < tileOuterNum; outerTileIdx += coreNum) {
            uint64_t startOuter = outerTileIdx * static_cast<uint64_t>(tileOuter);
            uint64_t rows = Min(static_cast<uint64_t>(tileOuter), outer - startOuter);
            uint64_t inOffset = startOuter * num;
            uint64_t readElems = rows * num;

            LocalTensor<T> inUb = inQueue.AllocTensor<T>();
            CopyInput(inUb, inOffset, readElems);
            inQueue.EnQue(inUb);
            LocalTensor<T> srcUb = inQueue.DeQue<T>();

            for (uint64_t outputIdx = 0; outputIdx < num; ++outputIdx) {
                LocalTensor<T> outUb = outQueue.AllocTensor<T>();
                Gather(outUb, srcUb[static_cast<uint32_t>(outputIdx)], offsetU32, 0,
                       static_cast<uint32_t>(rows));
                outQueue.EnQue(outUb);
                LocalTensor<T> dstUb = outQueue.DeQue<T>();
                CopyOutput(outputIdx, dstUb, startOuter, rows);
                outQueue.FreeTensor(dstUb);
            }

            inQueue.FreeTensor(srcUb);
        }
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueue;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<TPosition::VECCALC> offsetBuf;

    GlobalTensor<T> inGm;
    GlobalTensor<T> outGm;
    __gm__ uint64_t* tensorPtrs;

    uint64_t outer;
    uint64_t num;
    uint64_t inner;
    uint64_t totalElements;
    uint32_t coreNum;
    uint32_t coreId;
    uint32_t tileOuter;
    uint32_t tileInner;
    uint32_t tileOuterNum;
    uint32_t tileInnerNum;
};

#define UNPACK_LAUNCH(DTYPE, MODE_VALUE)                                             \
    do {                                                                             \
        KernelUnpack<DTYPE, MODE_VALUE> op;                                          \
        op.Init(input, output, tiling_data.outer, tiling_data.num, tiling_data.inner, \
                tiling_data.coreNum, tiling_data.tile_outer, tiling_data.tile_inner,  \
                tiling_data.tile_outer_num, tiling_data.tile_inner_num,               \
                tiling_data.ub_buffer_size, tiling_data.ub_out_buffer_size);          \
        op.Process();                                                                \
    } while (0)

extern "C" __global__ __aicore__ void unpack(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    if (TILING_KEY_IS(1)) { UNPACK_LAUNCH(float, MODE_DIRECT); return; }
    if (TILING_KEY_IS(2)) { UNPACK_LAUNCH(half, MODE_DIRECT); return; }
    if (TILING_KEY_IS(3)) { UNPACK_LAUNCH(uint16_t, MODE_DIRECT); return; }
    if (TILING_KEY_IS(4)) { UNPACK_LAUNCH(int32_t, MODE_DIRECT); return; }
    if (TILING_KEY_IS(5)) { UNPACK_LAUNCH(int16_t, MODE_DIRECT); return; }
    if (TILING_KEY_IS(6)) { UNPACK_LAUNCH(int8_t, MODE_DIRECT); return; }
    if (TILING_KEY_IS(7)) { UNPACK_LAUNCH(uint8_t, MODE_DIRECT); return; }
    if (TILING_KEY_IS(8)) { UNPACK_LAUNCH(uint8_t, MODE_DIRECT); return; }

    if (TILING_KEY_IS(17)) { UNPACK_LAUNCH(float, MODE_STRIDED_ALIGNED); return; }
    if (TILING_KEY_IS(18)) { UNPACK_LAUNCH(half, MODE_STRIDED_ALIGNED); return; }
    if (TILING_KEY_IS(19)) { UNPACK_LAUNCH(uint16_t, MODE_STRIDED_ALIGNED); return; }
    if (TILING_KEY_IS(20)) { UNPACK_LAUNCH(int32_t, MODE_STRIDED_ALIGNED); return; }
    if (TILING_KEY_IS(21)) { UNPACK_LAUNCH(int16_t, MODE_STRIDED_ALIGNED); return; }
    if (TILING_KEY_IS(22)) { UNPACK_LAUNCH(int8_t, MODE_STRIDED_ALIGNED); return; }
    if (TILING_KEY_IS(23)) { UNPACK_LAUNCH(uint8_t, MODE_STRIDED_ALIGNED); return; }
    if (TILING_KEY_IS(24)) { UNPACK_LAUNCH(uint8_t, MODE_STRIDED_ALIGNED); return; }

    if (TILING_KEY_IS(33)) { UNPACK_LAUNCH(float, MODE_STRIDED_PAD); return; }
    if (TILING_KEY_IS(34)) { UNPACK_LAUNCH(half, MODE_STRIDED_PAD); return; }
    if (TILING_KEY_IS(35)) { UNPACK_LAUNCH(uint16_t, MODE_STRIDED_PAD); return; }
    if (TILING_KEY_IS(36)) { UNPACK_LAUNCH(int32_t, MODE_STRIDED_PAD); return; }
    if (TILING_KEY_IS(37)) { UNPACK_LAUNCH(int16_t, MODE_STRIDED_PAD); return; }
    if (TILING_KEY_IS(38)) { UNPACK_LAUNCH(int8_t, MODE_STRIDED_PAD); return; }
    if (TILING_KEY_IS(39)) { UNPACK_LAUNCH(uint8_t, MODE_STRIDED_PAD); return; }
    if (TILING_KEY_IS(40)) { UNPACK_LAUNCH(uint8_t, MODE_STRIDED_PAD); return; }

    if (TILING_KEY_IS(49)) { UNPACK_LAUNCH(float, MODE_READ_ALL_ALIGNED); return; }
    if (TILING_KEY_IS(50)) { UNPACK_LAUNCH(half, MODE_READ_ALL_ALIGNED); return; }
    if (TILING_KEY_IS(51)) { UNPACK_LAUNCH(uint16_t, MODE_READ_ALL_ALIGNED); return; }
    if (TILING_KEY_IS(52)) { UNPACK_LAUNCH(int32_t, MODE_READ_ALL_ALIGNED); return; }
    if (TILING_KEY_IS(53)) { UNPACK_LAUNCH(int16_t, MODE_READ_ALL_ALIGNED); return; }
    if (TILING_KEY_IS(54)) { UNPACK_LAUNCH(int8_t, MODE_READ_ALL_ALIGNED); return; }
    if (TILING_KEY_IS(55)) { UNPACK_LAUNCH(uint8_t, MODE_READ_ALL_ALIGNED); return; }
    if (TILING_KEY_IS(56)) { UNPACK_LAUNCH(uint8_t, MODE_READ_ALL_ALIGNED); return; }

    if (TILING_KEY_IS(65)) { UNPACK_LAUNCH(float, MODE_COPY_ALL); return; }
    if (TILING_KEY_IS(66)) { UNPACK_LAUNCH(half, MODE_COPY_ALL); return; }
    if (TILING_KEY_IS(67)) { UNPACK_LAUNCH(uint16_t, MODE_COPY_ALL); return; }
    if (TILING_KEY_IS(68)) { UNPACK_LAUNCH(int32_t, MODE_COPY_ALL); return; }
    if (TILING_KEY_IS(69)) { UNPACK_LAUNCH(int16_t, MODE_COPY_ALL); return; }
    if (TILING_KEY_IS(70)) { UNPACK_LAUNCH(int8_t, MODE_COPY_ALL); return; }
    if (TILING_KEY_IS(71)) { UNPACK_LAUNCH(uint8_t, MODE_COPY_ALL); return; }
    if (TILING_KEY_IS(72)) { UNPACK_LAUNCH(uint8_t, MODE_COPY_ALL); return; }

    if (TILING_KEY_IS(81)) { UNPACK_LAUNCH(float, MODE_GATHER_INNER1); return; }
    if (TILING_KEY_IS(82)) { UNPACK_LAUNCH(half, MODE_GATHER_INNER1); return; }
    if (TILING_KEY_IS(83)) { UNPACK_LAUNCH(uint16_t, MODE_GATHER_INNER1); return; }
    if (TILING_KEY_IS(84)) { UNPACK_LAUNCH(uint32_t, MODE_GATHER_INNER1); return; }
    if (TILING_KEY_IS(85)) { UNPACK_LAUNCH(uint16_t, MODE_GATHER_INNER1); return; }
    if (TILING_KEY_IS(86)) { UNPACK_LAUNCH(int8_t, MODE_GATHER_INNER1); return; }
    if (TILING_KEY_IS(87)) { UNPACK_LAUNCH(uint8_t, MODE_GATHER_INNER1); return; }
    if (TILING_KEY_IS(88)) { UNPACK_LAUNCH(uint8_t, MODE_GATHER_INNER1); return; }

    if (TILING_KEY_IS(97)) { UNPACK_LAUNCH(float, MODE_READ_ALL_PAD); return; }
    if (TILING_KEY_IS(98)) { UNPACK_LAUNCH(half, MODE_READ_ALL_PAD); return; }
    if (TILING_KEY_IS(99)) { UNPACK_LAUNCH(uint16_t, MODE_READ_ALL_PAD); return; }
    if (TILING_KEY_IS(100)) { UNPACK_LAUNCH(int32_t, MODE_READ_ALL_PAD); return; }
    if (TILING_KEY_IS(101)) { UNPACK_LAUNCH(int16_t, MODE_READ_ALL_PAD); return; }
    if (TILING_KEY_IS(102)) { UNPACK_LAUNCH(int8_t, MODE_READ_ALL_PAD); return; }
    if (TILING_KEY_IS(103)) { UNPACK_LAUNCH(uint8_t, MODE_READ_ALL_PAD); return; }
    if (TILING_KEY_IS(104)) { UNPACK_LAUNCH(uint8_t, MODE_READ_ALL_PAD); return; }
}
