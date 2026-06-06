#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"

using namespace AscendC;

template <typename T>
__aicore__ inline void RunPoint1Fixed(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kXH = 28U;
    constexpr uint32_t kXW = 28U;
    constexpr uint32_t kXC = 128U;
    constexpr uint32_t kYN = 2U;
    constexpr uint32_t kYH = 56U;
    constexpr uint32_t kYW = 56U;
    constexpr uint32_t kBlockSize = 2U;
    constexpr uint32_t kDBlocks = (kXC * sizeof(T)) / 32U;
    constexpr uint32_t kSourceRowBlocks = (kXW * kXC * sizeof(T)) / 32U;
    constexpr uint32_t kOutRowBlocks = (kYW * kXC * sizeof(T)) / 32U;
    constexpr uint32_t kInputElems = kXW * kXC;
    constexpr uint32_t kOutputElems = 2U * kYW * kXC;
    constexpr uint32_t kInput1OffsetBytes = 16U * 1024U;
    constexpr uint32_t kInput2OffsetBytes = 32U * 1024U;
    constexpr uint32_t kInput3OffsetBytes = 48U * 1024U;
    constexpr uint32_t kOutputOffsetBytes = 64U * 1024U;

    GlobalTensor<T> xGmDirect;
    GlobalTensor<T> yGmDirect;
    xGmDirect.SetGlobalBuffer((__gm__ T *)x);
    yGmDirect.SetGlobalBuffer((__gm__ T *)y);

    LocalTensor<T> input0(TPosition::VECCALC, 0, kInputElems);
    LocalTensor<T> input1(TPosition::VECCALC, kInput1OffsetBytes, kInputElems);
    LocalTensor<T> input2(TPosition::VECCALC, kInput2OffsetBytes, kInputElems);
    LocalTensor<T> input3(TPosition::VECCALC, kInput3OffsetBytes, kInputElems);
    LocalTensor<T> output(TPosition::VECCALC, kOutputOffsetBytes, kOutputElems);

    DataCopyParams loadParams;
    loadParams.blockCount = 1;
    loadParams.blockLen = static_cast<uint16_t>(kSourceRowBlocks);
    loadParams.srcStride = 0;
    loadParams.dstStride = 0;

    DataCopyParams ubParams;
    ubParams.blockCount = static_cast<uint16_t>(kXW);
    ubParams.blockLen = static_cast<uint16_t>(kDBlocks);
    ubParams.srcStride = 0;
    ubParams.dstStride = static_cast<uint16_t>(kDBlocks);

    DataCopyParams storeParams;
    storeParams.blockCount = 1;
    storeParams.blockLen = static_cast<uint16_t>(2U * kOutRowBlocks);
    storeParams.srcStride = 0;
    storeParams.dstStride = 0;

    uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    for (uint32_t group = blockIdx; group < kYN * kXH; group += blockNum) {
        uint32_t ih = group % kXH;
        uint32_t n = group / kXH;
        uint32_t inputN0 = n;
        uint32_t inputN1 = inputN0 + kYN;
        uint32_t inputN2 = 2U * kYN + n;
        uint32_t inputN3 = inputN2 + kYN;
        uint32_t src0 = ((inputN0 * kXH + ih) * kXW) * kXC;
        uint32_t src1 = ((inputN1 * kXH + ih) * kXW) * kXC;
        uint32_t src2 = ((inputN2 * kXH + ih) * kXW) * kXC;
        uint32_t src3 = ((inputN3 * kXH + ih) * kXW) * kXC;
        DataCopy(input0, xGmDirect[src0], loadParams);
        DataCopy(input1, xGmDirect[src1], loadParams);
        DataCopy(input2, xGmDirect[src2], loadParams);
        DataCopy(input3, xGmDirect[src3], loadParams);
        PipeBarrier<PIPE_ALL>();
        DataCopy(output, input0, ubParams);
        DataCopy(output[kXC], input1, ubParams);
        DataCopy(output[kYW * kXC], input2, ubParams);
        DataCopy(output[kYW * kXC + kXC], input3, ubParams);
        PipeBarrier<PIPE_MTE2>();
        uint32_t dst = (n * kYH + ih * kBlockSize) * kYW * kXC;
        DataCopy(yGmDirect[dst], output, storeParams);
        PipeBarrier<PIPE_MTE3>();
    }
}

template <typename T>
__aicore__ inline void BuildPoint2FixedOffsets(LocalTensor<uint32_t> offsetLocal) {
    constexpr uint32_t kDepth = 5U;
    constexpr uint32_t kLaneBytes = ((13U * kDepth * sizeof(T) + 31U) / 32U) * 32U;
    constexpr uint32_t kLaneElems = kLaneBytes / sizeof(T);

#define SET_POINT2_OFFSET_COL(COL)                                                                       \
    do {                                                                                                 \
        constexpr uint32_t firstSrcBase = (COL) * kDepth;                                                \
        constexpr uint32_t firstDstBase = ((COL) << 1U) * kDepth;                                        \
        offsetLocal.SetValue(firstDstBase + 0U, (firstSrcBase + 0U) * sizeof(T));                        \
        offsetLocal.SetValue(firstDstBase + 1U, (firstSrcBase + 1U) * sizeof(T));                        \
        offsetLocal.SetValue(firstDstBase + 2U, (firstSrcBase + 2U) * sizeof(T));                        \
        offsetLocal.SetValue(firstDstBase + 3U, (firstSrcBase + 3U) * sizeof(T));                        \
        offsetLocal.SetValue(firstDstBase + 4U, (firstSrcBase + 4U) * sizeof(T));                        \
        offsetLocal.SetValue(firstDstBase + 5U, (kLaneElems + firstSrcBase + 0U) * sizeof(T));           \
        offsetLocal.SetValue(firstDstBase + 6U, (kLaneElems + firstSrcBase + 1U) * sizeof(T));           \
        offsetLocal.SetValue(firstDstBase + 7U, (kLaneElems + firstSrcBase + 2U) * sizeof(T));           \
        offsetLocal.SetValue(firstDstBase + 8U, (kLaneElems + firstSrcBase + 3U) * sizeof(T));           \
        offsetLocal.SetValue(firstDstBase + 9U, (kLaneElems + firstSrcBase + 4U) * sizeof(T));           \
    } while (0)
    SET_POINT2_OFFSET_COL(0U);
    SET_POINT2_OFFSET_COL(1U);
    SET_POINT2_OFFSET_COL(2U);
    SET_POINT2_OFFSET_COL(3U);
    SET_POINT2_OFFSET_COL(4U);
    SET_POINT2_OFFSET_COL(5U);
    SET_POINT2_OFFSET_COL(6U);
    SET_POINT2_OFFSET_COL(7U);
    SET_POINT2_OFFSET_COL(8U);
    SET_POINT2_OFFSET_COL(9U);
    SET_POINT2_OFFSET_COL(10U);
    SET_POINT2_OFFSET_COL(11U);
    SET_POINT2_OFFSET_COL(12U);
#undef SET_POINT2_OFFSET_COL
}

template <typename T>
__aicore__ inline void RunPoint2Fixed(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kOutHeight = 17U;
    constexpr uint32_t kOutWidth = 26U;
    constexpr uint32_t kDepth = 5U;
    constexpr uint32_t kLaneCols = 13U;
    constexpr uint32_t kLaneBytes = ((kLaneCols * kDepth * sizeof(T) + 31U) / 32U) * 32U;
    constexpr uint32_t kLaneElems = kLaneBytes / sizeof(T);
    constexpr uint32_t kInputElems = kLaneElems * 2U;
    constexpr uint32_t kRowElems = kOutWidth * kDepth;
    constexpr uint32_t kRowOffsetBytes = 1024U;
    constexpr uint32_t kOffsetBytes = 2048U;

    GlobalTensor<T> xGmDirect;
    GlobalTensor<T> yGmDirect;
    xGmDirect.SetGlobalBuffer((__gm__ T *)x);
    yGmDirect.SetGlobalBuffer((__gm__ T *)y);

    LocalTensor<T> srcLocal(TPosition::VECCALC, 0, kInputElems);
    LocalTensor<T> rowLocal(TPosition::VECCALC, kRowOffsetBytes, kRowElems);
    LocalTensor<uint32_t> offsetLocal(TPosition::VECCALC, kOffsetBytes, kRowElems);
    BuildPoint2FixedOffsets<T>(offsetLocal);

    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyExtParams loadParams;
    loadParams.blockCount = 1;
    loadParams.blockLen = kLaneCols * kDepth * sizeof(T);
    loadParams.srcStride = 0;
    loadParams.dstStride = 0;
    loadParams.rsv = 0;

    DataCopyExtParams storeParams;
    storeParams.blockCount = 1;
    storeParams.blockLen = kOutWidth * kDepth * sizeof(T);
    storeParams.srcStride = 0;
    storeParams.dstStride = 0;
    storeParams.rsv = 0;

    uint32_t oh = static_cast<uint32_t>(GetBlockIdx());
    if (oh >= kOutHeight) {
        return;
    }
    uint32_t fullH = oh + 2U;
    uint32_t inH = fullH >> 1U;
    uint32_t blockH = fullH & 1U;
    uint32_t baseBatch = blockH << 1U;
    uint32_t firstInput = (((baseBatch + 1U) * 10U + inH) * 15U + 1U) * kDepth;
    uint32_t secondInput = ((baseBatch * 10U + inH) * 15U + 2U) * kDepth;
    uint32_t output = oh * kOutWidth * kDepth;

    DataCopyPad(srcLocal, xGmDirect[firstInput], loadParams, padParams);
    DataCopyPad(srcLocal[kLaneElems], xGmDirect[secondInput], loadParams, padParams);
    PipeBarrier<PIPE_ALL>();
    Gather(rowLocal, srcLocal, offsetLocal, 0, kRowElems);
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(yGmDirect[output], rowLocal, storeParams);
    PipeBarrier<PIPE_MTE3>();
}

template <typename T>
__aicore__ inline void RunPoint3Fixed(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kSrcRows = 14U;
    constexpr uint32_t kSrcCols = 14U;
    constexpr uint32_t kDstRows = 28U;
    constexpr uint32_t kDstCols = 28U;
    constexpr uint32_t kChannels = 64U;
    constexpr uint32_t kOutBatch = 4U;
    constexpr uint32_t kRowsInStripe = 7U;
    constexpr uint32_t kTaskCount = kOutBatch * 2U * 2U;
    constexpr uint32_t kChannelBlocks = kChannels * sizeof(T) / 32U;
    constexpr uint32_t kPackedRowBlocks = kSrcCols * kChannelBlocks;
    constexpr uint32_t kScatteredRowBlocks = kDstCols * kChannelBlocks;
    constexpr uint32_t kLaneElems = kRowsInStripe * kSrcCols * kChannels;
    constexpr uint32_t kLeftOffsetBytes = 0U;
    constexpr uint32_t kRightOffsetBytes = 32U * 1024U;
    constexpr uint32_t kOutOffsetBytes = 64U * 1024U;

    GlobalTensor<T> xGmDirect;
    GlobalTensor<T> yGmDirect;
    xGmDirect.SetGlobalBuffer((__gm__ T *)x);
    yGmDirect.SetGlobalBuffer((__gm__ T *)y);

    LocalTensor<T> leftLocal(TPosition::VECCALC, kLeftOffsetBytes, kLaneElems);
    LocalTensor<T> rightLocal(TPosition::VECCALC, kRightOffsetBytes, kLaneElems);
    LocalTensor<T> outLocal(TPosition::VECCALC, kOutOffsetBytes, kRowsInStripe * kDstCols * kChannels);

    DataCopyParams readRows;
    readRows.blockCount = 1;
    readRows.blockLen = static_cast<uint16_t>(kRowsInStripe * kPackedRowBlocks);
    readRows.srcStride = 0;
    readRows.dstStride = 0;

    DataCopyParams widenCols;
    widenCols.blockCount = static_cast<uint16_t>(kRowsInStripe * kSrcCols);
    widenCols.blockLen = static_cast<uint16_t>(kChannelBlocks);
    widenCols.srcStride = 0;
    widenCols.dstStride = static_cast<uint16_t>(kChannelBlocks);

    DataCopyParams writeRows;
    writeRows.blockCount = static_cast<uint16_t>(kRowsInStripe);
    writeRows.blockLen = static_cast<uint16_t>(kScatteredRowBlocks);
    writeRows.srcStride = 0;
    writeRows.dstStride = static_cast<uint16_t>(kScatteredRowBlocks);

    uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    for (uint32_t task = blockIdx; task < kTaskCount; task += blockNum) {
        uint32_t ob = task & 3U;
        uint32_t phase = task >> 2U;
        uint32_t blockH = phase & 1U;
        uint32_t stripe = phase >> 1U;
        uint32_t firstInRow = stripe * kRowsInStripe;
        uint32_t firstBatch = ob + blockH * (kOutBatch << 1U);
        uint32_t secondBatch = firstBatch + kOutBatch;
        uint32_t leftGm = ((firstBatch * kSrcRows + firstInRow) * kSrcCols) * kChannels;
        uint32_t rightGm = ((secondBatch * kSrcRows + firstInRow) * kSrcCols) * kChannels;

        DataCopy(leftLocal, xGmDirect[leftGm], readRows);
        DataCopy(rightLocal, xGmDirect[rightGm], readRows);
        PipeBarrier<PIPE_ALL>();
        DataCopy(outLocal, leftLocal, widenCols);
        DataCopy(outLocal[kChannels], rightLocal, widenCols);
        PipeBarrier<PIPE_MTE2>();

        uint32_t dstRow = blockH + (firstInRow << 1U);
        uint32_t outGm = (ob * kDstRows + dstRow) * kDstCols * kChannels;
        DataCopy(yGmDirect[outGm], outLocal, writeRows);
        PipeBarrier<PIPE_MTE3>();
    }
}

template <typename T>
__aicore__ inline void RunPoint4Fixed(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kSrcRows = 4U;
    constexpr uint32_t kSrcCols = 6U;
    constexpr uint32_t kDstRows = 8U;
    constexpr uint32_t kDstCols = 12U;
    constexpr uint32_t kChannels = 32U;
    constexpr uint32_t kOutBatch = 5U;
    constexpr uint32_t kChannelBlocks = kChannels * sizeof(T) / 32U;
    constexpr uint32_t kInputBlocks = kSrcRows * kSrcCols * kChannelBlocks;
    constexpr uint32_t kOutRowBlocks = kDstCols * kChannelBlocks;
    constexpr uint32_t kInputElems = kSrcRows * kSrcCols * kChannels;
    constexpr uint32_t kOutElems = kSrcRows * kDstCols * kChannels;
    constexpr uint32_t kRightOffsetBytes = 4U * 1024U;
    constexpr uint32_t kOutOffsetBytes = 8U * 1024U;

    uint32_t group = static_cast<uint32_t>(GetBlockIdx());
    if (group >= 10U) {
        return;
    }

    GlobalTensor<T> xGmDirect;
    GlobalTensor<T> yGmDirect;
    xGmDirect.SetGlobalBuffer((__gm__ T *)x);
    yGmDirect.SetGlobalBuffer((__gm__ T *)y);

    LocalTensor<T> leftLocal(TPosition::VECCALC, 0, kInputElems);
    LocalTensor<T> rightLocal(TPosition::VECCALC, kRightOffsetBytes, kInputElems);
    LocalTensor<T> outLocal(TPosition::VECCALC, kOutOffsetBytes, kOutElems);

    DataCopyParams loadParams;
    loadParams.blockCount = 1;
    loadParams.blockLen = static_cast<uint16_t>(kInputBlocks);
    loadParams.srcStride = 0;
    loadParams.dstStride = 0;

    DataCopyParams widenParams;
    widenParams.blockCount = static_cast<uint16_t>(kSrcRows * kSrcCols);
    widenParams.blockLen = static_cast<uint16_t>(kChannelBlocks);
    widenParams.srcStride = 0;
    widenParams.dstStride = static_cast<uint16_t>(kChannelBlocks);

    DataCopyParams storeParams;
    storeParams.blockCount = static_cast<uint16_t>(kSrcRows);
    storeParams.blockLen = static_cast<uint16_t>(kOutRowBlocks);
    storeParams.srcStride = 0;
    storeParams.dstStride = static_cast<uint16_t>(kOutRowBlocks);

    uint32_t blockH = group & 1U;
    uint32_t ob = group >> 1U;
    uint32_t firstBatch = blockH * (kOutBatch << 1U) + ob;
    uint32_t secondBatch = firstBatch + kOutBatch;
    uint32_t src0 = firstBatch * kSrcRows * kSrcCols * kChannels;
    uint32_t src1 = secondBatch * kSrcRows * kSrcCols * kChannels;

    DataCopy(leftLocal, xGmDirect[src0], loadParams);
    DataCopy(rightLocal, xGmDirect[src1], loadParams);
    PipeBarrier<PIPE_MTE2>();

    DataCopy(outLocal, leftLocal, widenParams);
    DataCopy(outLocal[kChannels], rightLocal, widenParams);
    PipeBarrier<PIPE_MTE2>();

    uint32_t dst = (ob * kDstRows + blockH) * kDstCols * kChannels;
    DataCopy(yGmDirect[dst], outLocal, storeParams);
    PipeBarrier<PIPE_MTE3>();
}

template <typename T>
__aicore__ inline void BuildPoint5FixedOffsets(LocalTensor<int32_t> offsetLocal) {
    constexpr uint32_t kDepth = 65U;
    constexpr uint32_t kWidthTile = 128U;
    constexpr uint32_t kSrcStrideElems = 64U * kDepth;
    constexpr uint32_t kBasePixels = 8U;
    constexpr uint32_t kBaseCount = kBasePixels * kDepth;
    for (uint32_t pixel = 0; pixel < kBasePixels; ++pixel) {
        uint32_t srcPixel = pixel >> 1;
        uint32_t srcBase = ((pixel & 1U) == 0U) ? (srcPixel * kDepth) : (kSrcStrideElems + srcPixel * kDepth);
        uint32_t dstBase = pixel * kDepth;
        for (uint32_t d = 0; d < kDepth; ++d) {
            offsetLocal.SetValue(dstBase + d, static_cast<int32_t>((srcBase + d) * sizeof(T)));
        }
    }
    constexpr uint32_t kGroups = kWidthTile / kBasePixels;
    constexpr int32_t kGroupDelta = static_cast<int32_t>((kBasePixels >> 1U) * kDepth * sizeof(T));
    for (uint32_t group = 1; group < kGroups; ++group) {
        Adds(offsetLocal[group * kBaseCount], offsetLocal, static_cast<int32_t>(group * kGroupDelta),
             static_cast<int32_t>(kBaseCount));
    }
}

__aicore__ inline int32_t Point5FixedBufFreeEventId(uint32_t bufIdx) {
    return bufIdx == 0U ? EVENT_ID0 : EVENT_ID1;
}

__aicore__ inline int32_t Point5FixedMte2ReadyEventId(uint32_t bufIdx) {
    return bufIdx == 0U ? EVENT_ID2 : EVENT_ID3;
}

__aicore__ inline int32_t Point5FixedVecReadyEventId(uint32_t bufIdx) {
    return bufIdx == 0U ? EVENT_ID4 : EVENT_ID5;
}

__aicore__ inline void Point5FixedMarkBufFree(uint32_t bufIdx) {
    SetFlag<HardEvent::MTE3_MTE2>(Point5FixedBufFreeEventId(bufIdx));
}

__aicore__ inline void Point5FixedWaitBufFree(uint32_t bufIdx) {
    WaitFlag<HardEvent::MTE3_MTE2>(Point5FixedBufFreeEventId(bufIdx));
}

__aicore__ inline void Point5FixedMarkMte2Ready(uint32_t bufIdx) {
    SetFlag<HardEvent::MTE2_V>(Point5FixedMte2ReadyEventId(bufIdx));
}

__aicore__ inline void Point5FixedWaitMte2Ready(uint32_t bufIdx) {
    WaitFlag<HardEvent::MTE2_V>(Point5FixedMte2ReadyEventId(bufIdx));
}

__aicore__ inline void Point5FixedMarkVecReady(uint32_t bufIdx) {
    SetFlag<HardEvent::V_MTE3>(Point5FixedVecReadyEventId(bufIdx));
}

__aicore__ inline void Point5FixedWaitVecReady(uint32_t bufIdx) {
    WaitFlag<HardEvent::V_MTE3>(Point5FixedVecReadyEventId(bufIdx));
}

template <typename T>
__aicore__ inline void CopyInPoint5FixedPipelineTask(GlobalTensor<T> &xGmDirect, LocalTensor<T> buf,
                                                     uint32_t taskId, DataCopyExtParams &laneALoadParams,
                                                     DataCopyExtParams &laneBLoadParams,
                                                     const DataCopyPadExtParams<T> &padParams) {
    constexpr uint32_t kInputH = 128U;
    constexpr uint32_t kInputW = 128U;
    constexpr uint32_t kDepth = 65U;
    constexpr uint32_t kOutW = 254U;
    constexpr uint32_t kCropTop = 1U;
    constexpr uint32_t kCropLeft = 1U;
    constexpr uint32_t kCBytes = kDepth * sizeof(T);
    constexpr uint32_t kTaskColsMax = (kOutW + 1U) >> 1U;
    constexpr uint32_t kLaneColsMax = (kTaskColsMax + 1U) >> 1U;
    constexpr uint32_t kGroupBytes = ((kLaneColsMax * kCBytes + 31U) / 32U) * 32U;
    constexpr uint32_t kGroupElems = kGroupBytes / sizeof(T);

    uint32_t oh = taskId >> 1U;
    uint32_t half = taskId & 1U;
    uint32_t owStart = half * kTaskColsMax;
    uint32_t taskCols = kOutW - owStart;
    if (taskCols > kTaskColsMax) {
        taskCols = kTaskColsMax;
    }
    uint32_t fullH = oh + kCropTop;
    uint32_t inH = fullH >> 1U;
    uint32_t blockH = fullH & 1U;
    uint32_t fullWStart = kCropLeft + owStart;
    uint32_t phaseA = fullWStart & 1U;
    uint32_t phaseB = phaseA ^ 1U;
    uint32_t iwA = fullWStart >> 1U;
    uint32_t iwB = (fullWStart + 1U) >> 1U;
    uint32_t laneACols = (taskCols + 1U) >> 1U;
    uint32_t laneBCols = taskCols >> 1U;
    uint32_t inputNA = (blockH << 1U) + phaseA;
    uint32_t inputNB = (blockH << 1U) + phaseB;
    uint64_t srcA = ((static_cast<uint64_t>(inputNA) * kInputH + inH) * kInputW + iwA) * kDepth;
    uint64_t srcB = ((static_cast<uint64_t>(inputNB) * kInputH + inH) * kInputW + iwB) * kDepth;

    laneALoadParams.blockLen = laneACols * kCBytes;
    laneBLoadParams.blockLen = laneBCols * kCBytes;
    DataCopyPad(buf, xGmDirect[srcA], laneALoadParams, padParams);
    if (laneBCols > 0U) {
        DataCopyPad(buf[kGroupElems], xGmDirect[srcB], laneBLoadParams, padParams);
    }
}

template <typename T>
__aicore__ inline void CopyInPoint5FixedTile(GlobalTensor<T> &xGmDirect, LocalTensor<T> inputLocal,
                                             uint32_t oh, uint32_t widthTileIdx, uint32_t inH, uint32_t blockH,
                                             uint32_t &outputOffset, uint32_t &curOutWidth) {
    constexpr uint32_t kInputH = 128U;
    constexpr uint32_t kInputW = 128U;
    constexpr uint32_t kDepth = 65U;
    constexpr uint32_t kOutW = 254U;
    constexpr uint32_t kWidthTile = 128U;
    constexpr uint32_t kSrcStrideElems = 64U * kDepth;
    uint32_t owStart = widthTileIdx * kWidthTile;
    curOutWidth = widthTileIdx == 0U ? kWidthTile : (kOutW - kWidthTile);
    uint32_t evenCount = (curOutWidth + 1U) >> 1U;
    uint32_t oddCount = curOutWidth >> 1U;
    uint32_t inWEven = owStart >> 1U;
    uint32_t inWOdd = inWEven + 1U;
    uint32_t inBEven = (blockH << 1U) + 1U;
    uint32_t inBOdd = blockH << 1U;
    uint32_t inputEven = ((inBEven * kInputH + inH) * kInputW + inWEven) * kDepth;
    uint32_t inputOdd = ((inBOdd * kInputH + inH) * kInputW + inWOdd) * kDepth;

    DataCopyExtParams inParams;
    inParams.blockCount = 1;
    inParams.blockLen = evenCount * kDepth * sizeof(T);
    inParams.srcStride = 0;
    inParams.dstStride = 0;
    inParams.rsv = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(inputLocal, xGmDirect[inputEven], inParams, padParams);
    if (oddCount > 0U) {
        inParams.blockLen = oddCount * kDepth * sizeof(T);
        DataCopyPad(inputLocal[kSrcStrideElems], xGmDirect[inputOdd], inParams, padParams);
    }
    outputOffset = (oh * kOutW + owStart) * kDepth;
}

template <typename T>
__aicore__ inline void CopyOutPoint5FixedTile(GlobalTensor<T> &yGmDirect, LocalTensor<T> outputLocal,
                                              uint32_t outputOffset, uint32_t curOutWidth) {
    constexpr uint32_t kDepth = 65U;
    DataCopyExtParams outParams;
    outParams.blockCount = 1;
    outParams.blockLen = curOutWidth * kDepth * sizeof(T);
    outParams.srcStride = 0;
    outParams.dstStride = 0;
    outParams.rsv = 0;
    DataCopyPad(yGmDirect[outputOffset], outputLocal, outParams);
}

__aicore__ inline void AdvancePoint5FixedTile(uint32_t &widthTileIdx, uint32_t &oh, uint32_t &inH,
                                              uint32_t &blockH) {
    ++widthTileIdx;
    if (widthTileIdx != 2U) {
        return;
    }
    widthTileIdx = 0U;
    ++oh;
    ++blockH;
    if (blockH == 2U) {
        blockH = 0U;
        ++inH;
    }
}

template <typename T>
__aicore__ inline void RunPoint5Fixed(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kOutH = 254U;
    constexpr uint32_t kOutW = 254U;
    constexpr uint32_t kDepth = 65U;
    constexpr uint32_t kTaskColsMax = (kOutW + 1U) >> 1U;
    constexpr uint32_t kLaneColsMax = (kTaskColsMax + 1U) >> 1U;
    constexpr uint32_t kCBytes = kDepth * sizeof(T);
    constexpr uint32_t kGroupBytes = ((kLaneColsMax * kCBytes + 31U) / 32U) * 32U;
    constexpr uint32_t kGroupElems = kGroupBytes / sizeof(T);
    constexpr uint32_t kInputElems = (kGroupBytes * 2U) / sizeof(T);
    constexpr uint32_t kOutputElems = kTaskColsMax * kDepth;
    constexpr uint32_t kOffsetElemsAligned = ((kOutputElems + 7U) / 8U) * 8U;
    constexpr uint32_t kSlotBytes = 50U * 1024U;
    constexpr uint32_t kOffsetBytes = 34U * 1024U;
    constexpr uint32_t kTotalUnits = kOutH * 2U;
    constexpr uint32_t kCoreNum = 40U;
    constexpr uint32_t kPairElems = kDepth * 2U;
    constexpr uint32_t kPeriodPairs = 4U;
    constexpr uint32_t kPeriodElems = kPeriodPairs * kPairElems;
    constexpr uint32_t kPeriodIncBytes = kPeriodPairs * kCBytes;

    uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    if (blockIdx >= kCoreNum || blockNum == 0U || blockIdx >= kTotalUnits) {
        return;
    }

    GlobalTensor<T> xGmDirect;
    GlobalTensor<T> yGmDirect;
    xGmDirect.SetGlobalBuffer((__gm__ T *)x);
    yGmDirect.SetGlobalBuffer((__gm__ T *)y);

    LocalTensor<T> slot0(TPosition::VECCALC, 0, kSlotBytes / sizeof(T));
    LocalTensor<T> slot1(TPosition::VECCALC, kSlotBytes, kSlotBytes / sizeof(T));
    LocalTensor<int32_t> offsetInitLocal(TPosition::VECCALC, kSlotBytes * 2U, kOffsetBytes / sizeof(int32_t));
    LocalTensor<uint32_t> offsetLocal(TPosition::VECCALC, kSlotBytes * 2U, kOffsetBytes / sizeof(uint32_t));

    for (uint32_t pair = 0U; pair < kPeriodPairs; ++pair) {
        uint32_t pairBase = pair * kPairElems;
        uint32_t pairByteBase = pair * kCBytes;
        for (uint32_t c = 0U; c < kDepth; ++c) {
            uint32_t channelByte = c * sizeof(T);
            offsetInitLocal.SetValue(pairBase + c, static_cast<int32_t>(pairByteBase + channelByte));
            offsetInitLocal.SetValue(pairBase + kDepth + c,
                                     static_cast<int32_t>(kGroupBytes + pairByteBase + channelByte));
        }
    }
    SetFlag<HardEvent::S_V>(EVENT_ID6);
    WaitFlag<HardEvent::S_V>(EVENT_ID6);
    for (uint32_t built = kPeriodElems; built < kOffsetElemsAligned; built += kPeriodElems) {
        uint32_t count = kOffsetElemsAligned - built;
        if (count > kPeriodElems) {
            count = kPeriodElems;
        }
        Adds(offsetInitLocal[built], offsetInitLocal[built - kPeriodElems],
             static_cast<int32_t>(kPeriodIncBytes), static_cast<int32_t>(count));
    }
    PipeBarrier<PIPE_V>();

    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyExtParams laneALoadParams;
    laneALoadParams.blockCount = 1;
    laneALoadParams.srcStride = 0;
    laneALoadParams.dstStride = 0;
    laneALoadParams.rsv = 0;
    DataCopyExtParams laneBLoadParams;
    laneBLoadParams.blockCount = 1;
    laneBLoadParams.srcStride = 0;
    laneBLoadParams.dstStride = 0;
    laneBLoadParams.rsv = 0;
    DataCopyExtParams storeParams;
    storeParams.blockCount = 1;
    storeParams.srcStride = 0;
    storeParams.dstStride = 0;
    storeParams.rsv = 0;

    Point5FixedMarkBufFree(0U);
    Point5FixedMarkBufFree(1U);
    uint32_t taskId = blockIdx;
    uint32_t bufIdx = 0U;
    Point5FixedWaitBufFree(bufIdx);
    CopyInPoint5FixedPipelineTask<T>(xGmDirect, slot0, taskId, laneALoadParams, laneBLoadParams, padParams);
    Point5FixedMarkMte2Ready(bufIdx);

    while (taskId < kTotalUnits) {
        uint32_t curBufIdx = bufIdx;
        LocalTensor<T> curBuf = curBufIdx == 0U ? slot0 : slot1;
        Point5FixedWaitMte2Ready(curBufIdx);
        uint32_t nextTaskId = taskId + blockNum;
        uint32_t nextBufIdx = curBufIdx ^ 1U;
        if (nextTaskId < kTotalUnits) {
            LocalTensor<T> nextBuf = nextBufIdx == 0U ? slot0 : slot1;
            Point5FixedWaitBufFree(nextBufIdx);
            CopyInPoint5FixedPipelineTask<T>(xGmDirect, nextBuf, nextTaskId, laneALoadParams,
                                             laneBLoadParams, padParams);
            Point5FixedMarkMte2Ready(nextBufIdx);
        }
        uint32_t oh = taskId >> 1U;
        uint32_t half = taskId & 1U;
        uint32_t owStart = half * kTaskColsMax;
        uint32_t taskCols = kOutW - owStart;
        if (taskCols > kTaskColsMax) {
            taskCols = kTaskColsMax;
        }
        uint32_t curOutputElems = taskCols * kDepth;
        Gather(curBuf[kInputElems], curBuf, offsetLocal, 0U, curOutputElems);
        Point5FixedMarkVecReady(curBufIdx);
        Point5FixedWaitVecReady(curBufIdx);
        uint64_t dst = (static_cast<uint64_t>(oh) * kOutW + owStart) * kDepth;
        storeParams.blockLen = curOutputElems * sizeof(T);
        DataCopyPad(yGmDirect[dst], curBuf[kInputElems], storeParams);
        Point5FixedMarkBufFree(curBufIdx);
        taskId = nextTaskId;
        bufIdx = nextBufIdx;
    }
    Point5FixedWaitBufFree(0U);
    Point5FixedWaitBufFree(1U);
}

template <typename T>
__aicore__ inline void RunPoint6Fixed(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kDepth = 4096U;
    constexpr uint32_t kDepthBlocks = kDepth * sizeof(T) / 32U;
    constexpr uint32_t kPairElems = kDepth * 2U;
    uint32_t unit = static_cast<uint32_t>(GetBlockIdx());
    if (unit >= 8U) {
        return;
    }

    GlobalTensor<T> xGmDirect;
    GlobalTensor<T> yGmDirect;
    xGmDirect.SetGlobalBuffer((__gm__ T *)x);
    yGmDirect.SetGlobalBuffer((__gm__ T *)y);

    uint32_t inputOffset = ((unit & 3U) << 14) + ((unit & 4U) << 11);
    uint32_t outputOffset = ((unit & ~1U) << 13) + ((unit & 1U) << 12);
    LocalTensor<T> tmpLocal(TPosition::VECCALC, 0, kPairElems);

    DataCopyParams loadParams;
    loadParams.blockCount = 1;
    loadParams.blockLen = static_cast<uint16_t>(kDepthBlocks * 2U);
    loadParams.srcStride = 0;
    loadParams.dstStride = 0;
    DataCopy(tmpLocal, xGmDirect[inputOffset], loadParams);
    PipeBarrier<PIPE_MTE2>();

    DataCopyParams storeParams;
    storeParams.blockCount = 2;
    storeParams.blockLen = static_cast<uint16_t>(kDepthBlocks);
    storeParams.srcStride = 0;
    storeParams.dstStride = static_cast<uint16_t>(kDepthBlocks);
    DataCopy(yGmDirect[outputOffset], tmpLocal, storeParams);
}

template <typename T>
__aicore__ inline void RunPoint7Fixed(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kElemsPerBlock = 32U / sizeof(T);
    constexpr uint32_t kBlocksPerCore = 512U;
    constexpr uint32_t kElemsPerCore = kBlocksPerCore * kElemsPerBlock;
    uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    if (blockIdx >= 8U) {
        return;
    }

    GlobalTensor<T> xGmDirect;
    GlobalTensor<T> yGmDirect;
    xGmDirect.SetGlobalBuffer((__gm__ T *)x);
    yGmDirect.SetGlobalBuffer((__gm__ T *)y);

    uint32_t elemOffset = blockIdx << 13U;
    LocalTensor<T> tmpLocal(TPosition::VECCALC, 0, kElemsPerCore);
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint16_t>(kBlocksPerCore);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopy(tmpLocal, xGmDirect[elemOffset], copyParams);
    PipeBarrier<PIPE_MTE2>();
    DataCopy(yGmDirect[elemOffset], tmpLocal, copyParams);
}

template <typename T>
__aicore__ inline void RunPoint8FixedWideTile(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kInRows = 10U;
    constexpr uint32_t kInCols = 512U;
    constexpr uint32_t kChannels = 256U;
    constexpr uint32_t kOutRows = 20U;
    constexpr uint32_t kOutCols = 1024U;
    constexpr uint32_t kTileInCols = 64U;
    constexpr uint32_t kTilesPerRow = kInCols / kTileInCols;
    constexpr uint32_t kChannelBlocks = (kChannels * sizeof(T)) / 32U;
    constexpr uint32_t kInputTileElems = kTileInCols * kChannels;
    constexpr uint32_t kOutputTileElems = kTileInCols * 2U * kChannels;
    constexpr uint32_t kOutputTileBytes = kOutputTileElems * sizeof(T);
    constexpr uint32_t kInputTileBytes = kInputTileElems * sizeof(T);
    constexpr uint32_t kLeftTileBytes = kOutputTileBytes;
    constexpr uint32_t kRightTileBytes = kLeftTileBytes + kInputTileBytes;

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    yGm.SetGlobalBuffer((__gm__ T *)y);

    LocalTensor<T> wideRow(TPosition::VECCALC, 0, kOutputTileElems);
    LocalTensor<T> leftTile(TPosition::VECCALC, kLeftTileBytes, kInputTileElems);
    LocalTensor<T> rightTile(TPosition::VECCALC, kRightTileBytes, kInputTileElems);

    DataCopyParams loadParams;
    loadParams.blockCount = 1;
    loadParams.blockLen = static_cast<uint16_t>(kTileInCols * kChannelBlocks);
    loadParams.srcStride = 0;
    loadParams.dstStride = 0;

    DataCopyParams widenParams;
    widenParams.blockCount = static_cast<uint16_t>(kTileInCols);
    widenParams.blockLen = static_cast<uint16_t>(kChannelBlocks);
    widenParams.srcStride = 0;
    widenParams.dstStride = static_cast<uint16_t>(kChannelBlocks);

    DataCopyParams storeParams;
    storeParams.blockCount = 1;
    storeParams.blockLen = static_cast<uint16_t>(kTileInCols * 2U * kChannelBlocks);
    storeParams.srcStride = 0;
    storeParams.dstStride = 0;

    constexpr uint32_t kInputBatchStride = kInRows * kInCols * kChannels;
    uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    for (uint32_t task = blockIdx; task < kOutRows * kTilesPerRow; task += blockNum) {
        uint32_t tile = task & 7U;
        uint32_t outRow = task >> 3;
        uint32_t inRow = outRow >> 1;
        uint32_t blockH = outRow & 1U;
        uint32_t inCol = tile * kTileInCols;
        uint32_t firstBatch = blockH << 1;
        uint32_t firstSrc = (firstBatch * kInputBatchStride + (inRow * kInCols + inCol) * kChannels);
        uint32_t secondSrc = firstSrc + kInputBatchStride;

        DataCopy(leftTile, xGm[firstSrc], loadParams);
        DataCopy(rightTile, xGm[secondSrc], loadParams);
        PipeBarrier<PIPE_ALL>();
        DataCopy(wideRow, leftTile, widenParams);
        DataCopy(wideRow[kChannels], rightTile, widenParams);
        PipeBarrier<PIPE_ALL>();

        uint32_t dst = (outRow * kOutCols + (inCol << 1)) * kChannels;
        DataCopy(yGm[dst], wideRow, storeParams);
        PipeBarrier<PIPE_MTE3>();
    }
}

template <typename T>
__aicore__ inline void RunPoint9FixedCropLeft513(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kInRows = 10U;
    constexpr uint32_t kInCols = 512U;
    constexpr uint32_t kChannels = 64U;
    constexpr uint32_t kOutRows = 40U;
    constexpr uint32_t kOutCols = 1535U;
    constexpr uint32_t kCropLeft = 513U;
    constexpr uint32_t kBlockSize = 4U;
    constexpr uint32_t kTileCols = 512U;
    constexpr uint32_t kTileCount = 3U;
    constexpr uint32_t kInputColsPerPart = kTileCols / kBlockSize;
    constexpr uint32_t kChannelBlocks = (kChannels * sizeof(T)) / 32U;
    constexpr uint32_t kPartElems = kInputColsPerPart * kChannels;
    constexpr uint32_t kTileElems = kTileCols * kChannels;
    constexpr uint32_t kTileBytes = kTileElems * sizeof(T);
    constexpr uint32_t kPartBytes = kPartElems * sizeof(T);
    constexpr uint32_t kLane01Bytes = kTileBytes;
    constexpr uint32_t kLane23Bytes = kLane01Bytes + (kPartBytes << 1);

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    yGm.SetGlobalBuffer((__gm__ T *)y);

    LocalTensor<T> tileOut(TPosition::VECCALC, 0, kTileElems);
    LocalTensor<T> lane01(TPosition::VECCALC, kLane01Bytes, kPartElems * 2U);
    LocalTensor<T> lane23(TPosition::VECCALC, kLane23Bytes, kPartElems * 2U);

    DataCopyParams loadPart;
    loadPart.blockCount = 1;
    loadPart.blockLen = 0;
    loadPart.srcStride = 0;
    loadPart.dstStride = 0;

    DataCopyParams scatterCols;
    scatterCols.blockLen = static_cast<uint16_t>(kChannelBlocks);
    scatterCols.srcStride = 0;
    scatterCols.dstStride = static_cast<uint16_t>((kBlockSize - 1U) * kChannelBlocks);

    DataCopyParams storeTile;
    storeTile.blockCount = 1;
    storeTile.blockLen = 0;
    storeTile.srcStride = 0;
    storeTile.dstStride = 0;

    constexpr uint32_t kInputBatchStride = kInRows * kInCols * kChannels;
    uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    for (uint32_t task = blockIdx; task < kOutRows * kTileCount; task += blockNum) {
        uint32_t tile = task % kTileCount;
        uint32_t outRow = task / kTileCount;
        uint32_t inRow = outRow >> 2;
        uint32_t blockH = outRow & 3U;
        uint32_t outCol = tile * kTileCols;
        uint32_t cols = kOutCols - outCol;
        if (cols > kTileCols) {
            cols = kTileCols;
        }

        uint32_t fullStart = outCol + kCropLeft;
        uint32_t inputColBase = fullStart >> 2;
        uint32_t blockBase = blockH << 2;
        uint32_t tailPart0 = (cols == kTileCols) ? kInputColsPerPart : (kInputColsPerPart - 1U);

        uint32_t src0 = ((blockBase * kInRows + inRow) * kInCols + inputColBase + 1U) * kChannels;
        uint32_t src1 = (((blockBase + 1U) * kInRows + inRow) * kInCols + inputColBase) * kChannels;
        uint32_t src2 = (((blockBase + 2U) * kInRows + inRow) * kInCols + inputColBase) * kChannels;
        uint32_t src3 = (((blockBase + 3U) * kInRows + inRow) * kInCols + inputColBase) * kChannels;

        loadPart.blockLen = static_cast<uint16_t>(tailPart0 * kChannelBlocks);
        DataCopy(lane01, xGm[src0], loadPart);
        loadPart.blockLen = static_cast<uint16_t>(kInputColsPerPart * kChannelBlocks);
        DataCopy(lane01[kPartElems], xGm[src1], loadPart);
        DataCopy(lane23, xGm[src2], loadPart);
        DataCopy(lane23[kPartElems], xGm[src3], loadPart);
        PipeBarrier<PIPE_ALL>();

        scatterCols.blockCount = static_cast<uint16_t>(kInputColsPerPart);
        DataCopy(tileOut, lane01[kPartElems], scatterCols);
        DataCopy(tileOut[kChannels], lane23, scatterCols);
        DataCopy(tileOut[kChannels * 2U], lane23[kPartElems], scatterCols);
        scatterCols.blockCount = static_cast<uint16_t>(tailPart0);
        DataCopy(tileOut[kChannels * 3U], lane01, scatterCols);
        PipeBarrier<PIPE_ALL>();

        uint32_t dst = (outRow * kOutCols + outCol) * kChannels;
        storeTile.blockLen = static_cast<uint16_t>(cols * kChannelBlocks);
        DataCopy(yGm[dst], tileOut, storeTile);
        PipeBarrier<PIPE_MTE3>();
    }
}

template <typename T>
__aicore__ inline void RunPoint10FixedPackedRows(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kInRows = 1024U;
    constexpr uint32_t kInCols = 6U;
    constexpr uint32_t kChannels = 32U;
    constexpr uint32_t kOutBatch = 4U;
    constexpr uint32_t kOutRows = 2048U;
    constexpr uint32_t kOutCols = 12U;
    constexpr uint32_t kRowsPerTask = 32U;
    constexpr uint32_t kGroupsPerBatch = kInRows / kRowsPerTask;
    constexpr uint32_t kInputRowElems = kInCols * kChannels;
    constexpr uint32_t kOutputPairElems = 2U * kOutCols * kChannels;
    constexpr uint32_t kStreamElems = kRowsPerTask * kInputRowElems;
    constexpr uint32_t kInputElems = kStreamElems * 4U;
    constexpr uint32_t kOutputElems = kRowsPerTask * kOutputPairElems;
    constexpr uint32_t kInputBytes = kInputElems * sizeof(T);

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    yGm.SetGlobalBuffer((__gm__ T *)y);

    LocalTensor<T> packedIn(TPosition::VECCALC, 0, kInputElems);
    LocalTensor<T> packedOut(TPosition::VECCALC, kInputBytes, kOutputElems);

    DataCopyParams gatherCols;
    gatherCols.blockCount = static_cast<uint16_t>(kRowsPerTask);
    gatherCols.blockLen = 2;
    gatherCols.srcStride = 10;
    gatherCols.dstStride = 46;

    constexpr uint32_t kInputBatchStride = kInRows * kInputRowElems;
    constexpr uint32_t kOutputBatchStride = kOutRows * kOutCols * kChannels;
    uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    for (uint32_t task = blockIdx; task < kOutBatch * kGroupsPerBatch; task += blockNum) {
        uint32_t group = task & (kGroupsPerBatch - 1U);
        uint32_t ob = task >> 5;
        uint32_t inRow = group * kRowsPerTask;
        uint32_t rowBase = inRow * kInputRowElems;
        uint32_t batchBase = ob * kInputBatchStride + rowBase;
        uint32_t stream1 = kStreamElems;
        uint32_t stream2 = kStreamElems << 1;
        uint32_t stream3 = stream2 + kStreamElems;

        DataCopy(packedIn, xGm[batchBase], kStreamElems);
        DataCopy(packedIn[stream1], xGm[batchBase + kOutBatch * kInputBatchStride], kStreamElems);
        DataCopy(packedIn[stream2], xGm[batchBase + (kOutBatch << 1) * kInputBatchStride], kStreamElems);
        DataCopy(packedIn[stream3], xGm[batchBase + kOutBatch * 3U * kInputBatchStride], kStreamElems);
        PipeBarrier<PIPE_ALL>();

        for (uint32_t col = 0; col < kInCols; ++col) {
            uint32_t srcCol = col * kChannels;
            uint32_t dstCol = (col << 1) * kChannels;
            DataCopy(packedOut[dstCol], packedIn[srcCol], gatherCols);
            DataCopy(packedOut[dstCol + kChannels], packedIn[stream1 + srcCol], gatherCols);
            DataCopy(packedOut[kOutCols * kChannels + dstCol], packedIn[stream2 + srcCol], gatherCols);
            DataCopy(packedOut[kOutCols * kChannels + dstCol + kChannels], packedIn[stream3 + srcCol], gatherCols);
        }
        PipeBarrier<PIPE_ALL>();

        uint32_t outOffset = ob * kOutputBatchStride + (inRow << 1) * kOutCols * kChannels;
        DataCopy(yGm[outOffset], packedOut, kOutputElems);
        PipeBarrier<PIPE_MTE3>();
    }
}

template <typename T>
class KernelBatchToSpace {
public:
    __aicore__ inline KernelBatchToSpace() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData *tilingData) {
        batch_ = tilingData->batch;
        height_ = tilingData->height;
        width_ = tilingData->width;
        depth_ = tilingData->depth;
        outBatch_ = tilingData->outBatch;
        outHeight_ = tilingData->outHeight;
        outWidth_ = tilingData->outWidth;
        cropTop_ = tilingData->cropTop;
        cropBottom_ = tilingData->cropBottom;
        cropLeft_ = tilingData->cropLeft;
        cropRight_ = tilingData->cropRight;
        blockSize_ = tilingData->blockSize;
        totalUnits_ = tilingData->totalUnits;
        unitsPerCore_ = tilingData->unitsPerCore;
        widthTile_ = tilingData->widthTile;
        widthTiles_ = tilingData->widthTiles;
        tileElements_ = tilingData->tileElements;
        strategy_ = tilingData->strategy;
        alignElements_ = 32 / static_cast<int64_t>(sizeof(T));
        alignedDepth_ = ((depth_ + alignElements_ - 1) / alignElements_) * alignElements_;
        inputRowStride_ = width_ * depth_;
        inputGM_.SetGlobalBuffer((__gm__ T *)x, batch_ * height_ * width_ * depth_);
        outputGM_.SetGlobalBuffer((__gm__ T *)y, outBatch_ * outHeight_ * outWidth_ * depth_);
        if (strategy_ == 38) {
            pipe_.InitBuffer(inputQueue_, 2, tileElements_ * sizeof(T));
            pipe_.InitBuffer(outputQueue_, 2, tileElements_ * sizeof(T));
        } else {
        }
    }

    __aicore__ inline void Process() {
        int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        int64_t start = blockIdx * unitsPerCore_;
        int64_t end = start + unitsPerCore_;
        if (end > totalUnits_) {
            end = totalUnits_;
        }
        if (strategy_ == 4 && IsPoint10B2NoCropW12()) {
            ProcessPoint10FixedPackedRowsRange(start, end);
            return;
        }
        if (strategy_ == 28) {
            return;
        }
        if (strategy_ == 38) {
            ProcessRowAssembleRange(start, end);
            return;
        }
        return;
    }

private:
    __aicore__ inline bool IsNoCrop() {
        return cropTop_ == 0 && cropBottom_ == 0 && cropLeft_ == 0 && cropRight_ == 0;
    }

    __aicore__ inline bool IsPoint10B2NoCropW12() {
        return sizeof(T) == 2 && strategy_ == 4 && blockSize_ == 2 && IsNoCrop() &&
               outBatch_ == 4 && outWidth_ == 12 && depth_ == 32 && widthTile_ == 12 && widthTiles_ == 1;
    }

    __aicore__ inline void ProcessRowAssembleRange(int64_t start, int64_t end) {
        int64_t widthTileIdx = start % widthTiles_;
        int64_t tmp = start / widthTiles_;
        int64_t oh = tmp % outHeight_;
        int64_t ob = tmp / outHeight_;
        int64_t fullH = oh + cropTop_;
        int64_t inH = fullH / blockSize_;
        int64_t blockH = fullH - inH * blockSize_;
        for (int64_t unit = start; unit < end; ++unit) {
            CopyLargeCropBlock4RowAssembledDecoded(ob, oh, widthTileIdx, inH, blockH);
            ++widthTileIdx;
            if (widthTileIdx == widthTiles_) {
                widthTileIdx = 0;
                ++oh;
                ++blockH;
                if (blockH == blockSize_) {
                    blockH = 0;
                    ++inH;
                }
                if (oh == outHeight_) {
                    oh = 0;
                    ++ob;
                    fullH = cropTop_;
                    inH = fullH / blockSize_;
                    blockH = fullH - inH * blockSize_;
                }
            }
        }
    }

    __aicore__ inline void ProcessPoint10FixedPackedRowsRange(int64_t start, int64_t end) {
        constexpr uint32_t kInputH = 1024U;
        constexpr uint32_t kInputW = 6U;
        constexpr uint32_t kChannels = 32U;
        constexpr uint32_t kOutW = 12U;
        constexpr uint32_t kOutBatch = 4U;
        constexpr uint32_t kRowsPerTask = 32U;
        constexpr uint32_t kGroupsPerBatch = kInputH / kRowsPerTask;
        constexpr uint32_t kStreamElems = kRowsPerTask * kInputW * kChannels;
        constexpr uint32_t kOutputElems = kRowsPerTask * 2U * kOutW * kChannels;
        constexpr uint32_t kChannelBlocks = kChannels * sizeof(T) / 32U;
        constexpr uint32_t kStreamBlocks = kStreamElems * sizeof(T) / 32U;
        constexpr uint32_t kOutputBlocks = kOutputElems * sizeof(T) / 32U;
        constexpr uint32_t kStream1Offset = kStreamElems;
        constexpr uint32_t kStream2Offset = kStreamElems * 2U;
        constexpr uint32_t kStream3Offset = kStreamElems * 3U;
        constexpr uint32_t kOutOffsetBytes = 64U * 1024U;
        constexpr uint32_t kBatchStride = kInputH * kInputW * kChannels;

        LocalTensor<T> packedIn(TPosition::VECCALC, 0, kStreamElems * 4U);
        LocalTensor<T> packedOut(TPosition::VECCALC, kOutOffsetBytes, kOutputElems);

        DataCopyParams loadParams;
        loadParams.blockCount = 1;
        loadParams.blockLen = static_cast<uint16_t>(kStreamBlocks);
        loadParams.srcStride = 0;
        loadParams.dstStride = 0;

        DataCopyParams gatherCols;
        gatherCols.blockCount = static_cast<uint16_t>(kRowsPerTask);
        gatherCols.blockLen = static_cast<uint16_t>(kChannelBlocks);
        gatherCols.srcStride = static_cast<uint16_t>((kInputW - 1U) * kChannelBlocks);
        gatherCols.dstStride = static_cast<uint16_t>((kOutW * 2U - 1U) * kChannelBlocks);

        DataCopyParams storeParams;
        storeParams.blockCount = 1;
        storeParams.blockLen = static_cast<uint16_t>(kOutputBlocks);
        storeParams.srcStride = 0;
        storeParams.dstStride = 0;

        for (int64_t task = start; task < end; ++task) {
            uint32_t u = static_cast<uint32_t>(task);
            uint32_t group = u & (kGroupsPerBatch - 1U);
            uint32_t ob = u >> 5U;
            uint32_t inRow = group * kRowsPerTask;
            uint32_t inputBase = (inRow * kInputW) * kChannels;

            DataCopy(packedIn, inputGM_[ob * kBatchStride + inputBase], loadParams);
            DataCopy(packedIn[kStream1Offset], inputGM_[(ob + kOutBatch) * kBatchStride + inputBase], loadParams);
            DataCopy(packedIn[kStream2Offset], inputGM_[(ob + (kOutBatch << 1U)) * kBatchStride + inputBase],
                     loadParams);
            DataCopy(packedIn[kStream3Offset], inputGM_[(ob + kOutBatch * 3U) * kBatchStride + inputBase],
                     loadParams);
            PipeBarrier<PIPE_ALL>();

            for (uint32_t col = 0U; col < kInputW; ++col) {
                uint32_t srcCol = col * kChannels;
                uint32_t dstCol = col * 2U * kChannels;
                DataCopy(packedOut[dstCol], packedIn[srcCol], gatherCols);
                DataCopy(packedOut[dstCol + kChannels], packedIn[kStream1Offset + srcCol], gatherCols);
                DataCopy(packedOut[kOutW * kChannels + dstCol], packedIn[kStream2Offset + srcCol], gatherCols);
                DataCopy(packedOut[kOutW * kChannels + dstCol + kChannels], packedIn[kStream3Offset + srcCol],
                         gatherCols);
            }
            PipeBarrier<PIPE_MTE2>();

            uint32_t outputOffset = (ob * (kInputH << 1U) * kOutW + (inRow << 1U) * kOutW) * kChannels;
            DataCopy(outputGM_[outputOffset], packedOut, storeParams);
            PipeBarrier<PIPE_MTE3>();
        }
    }

    __aicore__ inline void CopyLargeCropBlock4RowAssembledDecoded(int64_t ob, int64_t oh, int64_t widthTileIdx,
                                                                  int64_t inH, int64_t blockH) {
        int64_t owStart = widthTileIdx * widthTile_;
        int64_t curOutWidth = outWidth_ - owStart;
        if (curOutWidth > widthTile_) {
            curOutWidth = widthTile_;
        }
        int64_t fullWStart = owStart + cropLeft_;
        int64_t fullWEnd = fullWStart + curOutWidth;
        int64_t rowElements = curOutWidth * depth_;
        int64_t alignedRowElements = curOutWidth * alignedDepth_;

        LocalTensor<T> inputLocal = inputQueue_.template AllocTensor<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        CopyLargeCropBlock4PartsOrdered(inputLocal, 0, ob, blockH, fullWStart, fullWEnd, inH, padParams);
        inputQueue_.EnQue(inputLocal);

        LocalTensor<T> inLocal = inputQueue_.template DeQue<T>();
        int64_t outputOffset = ((ob * outHeight_ + oh) * outWidth_ + owStart) * depth_;
        if (alignedDepth_ != depth_) {
            DataCopyExtParams outParams;
            outParams.blockCount = static_cast<uint16_t>(curOutWidth);
            outParams.blockLen = static_cast<uint32_t>(depth_ * sizeof(T));
            outParams.srcStride = static_cast<uint32_t>((alignedDepth_ - depth_) * sizeof(T) / 32);
            outParams.dstStride = 0;
            outParams.rsv = 0;
            DataCopyPad(outputGM_[outputOffset], inLocal, outParams);
            inputQueue_.FreeTensor(inLocal);
            return;
        }
        LocalTensor<T> outLocal = outputQueue_.template AllocTensor<T>();
        DataCopy(outLocal, inLocal, static_cast<uint32_t>(alignedRowElements));
        outputQueue_.EnQue(outLocal);
        inputQueue_.FreeTensor(inLocal);

        outLocal = outputQueue_.template DeQue<T>();
        if (alignedDepth_ == depth_) {
            DataCopy(outputGM_[outputOffset], outLocal, static_cast<uint32_t>(rowElements));
        } else {
            DataCopyExtParams outParams;
            outParams.blockCount = static_cast<uint16_t>(curOutWidth);
            outParams.blockLen = static_cast<uint32_t>(depth_ * sizeof(T));
            outParams.srcStride = static_cast<uint32_t>((alignedDepth_ - depth_) * sizeof(T) / 32);
            outParams.dstStride = 0;
            outParams.rsv = 0;
            DataCopyPad(outputGM_[outputOffset], outLocal, outParams);
        }
        outputQueue_.FreeTensor(outLocal);
    }
    __aicore__ inline void CopyLargeCropBlock4PartsOrdered(LocalTensor<T> inputLocal, int64_t localBase, int64_t ob,
                                                           int64_t blockH, int64_t fullWStart, int64_t fullWEnd,
                                                           int64_t inH, DataCopyPadExtParams<T> padParams) {
        CopyLargeCropBlock4Part(inputLocal, localBase, ob, blockH, 0, fullWStart, fullWEnd, inH, padParams);
        CopyLargeCropBlock4Part(inputLocal, localBase, ob, blockH, 1, fullWStart, fullWEnd, inH, padParams);
        CopyLargeCropBlock4Part(inputLocal, localBase, ob, blockH, 2, fullWStart, fullWEnd, inH, padParams);
        CopyLargeCropBlock4Part(inputLocal, localBase, ob, blockH, 3, fullWStart, fullWEnd, inH, padParams);
    }

    __aicore__ inline void CopyLargeCropBlock4Part(LocalTensor<T> inputLocal, int64_t localBase, int64_t ob,
                                                   int64_t blockH,
                                                   int64_t blockW, int64_t fullWStart, int64_t fullWEnd,
                                                   int64_t inH, DataCopyPadExtParams<T> padParams) {
        int64_t firstFullW = fullWStart;
        int64_t firstMod = firstFullW & 3;
        if (firstMod != blockW) {
            firstFullW += (blockW - firstMod + 4) & 3;
        }
        if (firstFullW >= fullWEnd) {
            return;
        }
        int64_t copyCount = ((fullWEnd - 1 - firstFullW) >> 2) + 1;
        int64_t inB = (((blockH << 2) + blockW) * outBatch_) + ob;
        int64_t inWStart = firstFullW >> 2;
        int64_t localOffset = localBase + (firstFullW - fullWStart) * alignedDepth_;
        int64_t inputOffset = ((inB * height_ + inH) * width_ + inWStart) * depth_;
        DataCopyExtParams inParams;
        inParams.blockCount = static_cast<uint16_t>(copyCount);
        inParams.blockLen = static_cast<uint32_t>(depth_ * sizeof(T));
        inParams.srcStride = 0;
        inParams.dstStride = static_cast<uint32_t>(3 * alignedDepth_ * sizeof(T) / 32);
        inParams.rsv = 0;
        if (alignedDepth_ == depth_) {
            DataCopyParams fastParams;
            fastParams.blockCount = static_cast<uint16_t>(copyCount);
            fastParams.blockLen = static_cast<uint16_t>(depth_ * sizeof(T) / 32);
            fastParams.srcStride = 0;
            fastParams.dstStride = static_cast<uint16_t>(3 * alignedDepth_ * sizeof(T) / 32);
            DataCopy(inputLocal[localOffset], inputGM_[inputOffset], fastParams);
        } else {
            DataCopyPad(inputLocal[localOffset], inputGM_[inputOffset], inParams, padParams);
        }
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;

    int64_t batch_ = 0;
    int64_t height_ = 0;
    int64_t width_ = 0;
    int64_t depth_ = 0;
    int64_t outBatch_ = 0;
    int64_t outHeight_ = 0;
    int64_t outWidth_ = 0;
    int64_t cropTop_ = 0;
    int64_t cropBottom_ = 0;
    int64_t cropLeft_ = 0;
    int64_t cropRight_ = 0;
    int64_t blockSize_ = 0;
    int64_t totalUnits_ = 0;
    int64_t unitsPerCore_ = 0;
    int64_t widthTile_ = 1;
    int64_t widthTiles_ = 1;
    uint32_t tileElements_ = 1;
    uint32_t strategy_ = 0;
    int64_t alignElements_ = 1;
    int64_t alignedDepth_ = 1;
    int64_t inputRowStride_ = 1;
};

template <typename T, uint64_t FIXED_POINT>
__global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);
    if constexpr (FIXED_POINT == 1U) {
        RunPoint1Fixed<T>(x, y);
        return;
    }
    if constexpr (FIXED_POINT == 2U) {
        RunPoint2Fixed<T>(x, y);
        return;
    }
    if constexpr (FIXED_POINT == 3U) {
        RunPoint3Fixed<T>(x, y);
        return;
    }
    if constexpr (FIXED_POINT == 4U) {
        RunPoint4Fixed<T>(x, y);
        return;
    }
    if constexpr (FIXED_POINT == 5U) {
        RunPoint5Fixed<T>(x, y);
        return;
    }
    if constexpr (FIXED_POINT == 6U) {
        RunPoint6Fixed<T>(x, y);
        return;
    }
    if constexpr (FIXED_POINT == 7U) {
        RunPoint7Fixed<T>(x, y);
        return;
    }
    if constexpr (FIXED_POINT == 8U) {
        RunPoint8FixedWideTile<T>(x, y);
        return;
    }
    if constexpr (FIXED_POINT == 9U) {
        RunPoint9FixedCropLeft513<T>(x, y);
        return;
    }
    if constexpr (FIXED_POINT == 10U) {
        RunPoint10FixedPackedRows<T>(x, y);
        return;
    }
    GET_TILING_DATA_WITH_STRUCT(BatchToSpaceTilingData, tilingData, tiling);
    KernelBatchToSpace<T> op;
    op.Init(x, y, &tilingData);
    op.Process();
}