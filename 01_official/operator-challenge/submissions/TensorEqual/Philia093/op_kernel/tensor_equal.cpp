#include "kernel_operator.h"

#include <type_traits>

using namespace AscendC;

namespace {

constexpr uint32_t kBufferNum = 2;
constexpr uint32_t kAlignBytes = 32;
constexpr uint32_t kVectorAlignElems = 128;
constexpr uint32_t kScalarBlockElems = 32;
constexpr int64_t kMaxRank = 8;

__aicore__ inline uint32_t AlignUpBytes(uint32_t bytes)
{
    return ((bytes + kAlignBytes - 1) / kAlignBytes) * kAlignBytes;
}

__aicore__ inline uint32_t AlignUpElems(uint32_t elems)
{
    return ((elems + kVectorAlignElems - 1) / kVectorAlignElems) * kVectorAlignElems;
}

__aicore__ inline uint32_t CmpMaskBytes(uint32_t elems)
{
    return AlignUpBytes(AlignUpElems(elems) / 8);
}

template <typename T>
__aicore__ inline void CopyInAuto(GlobalTensor<T>& inputGm, int64_t offset, LocalTensor<T>& inputLocal, uint32_t len)
{
    const uint32_t copyBytes = len * sizeof(T);
    const uint64_t offsetBytes = static_cast<uint64_t>(offset) * sizeof(T);
    if ((copyBytes % kAlignBytes) == 0 && (offsetBytes % kAlignBytes) == 0) {
        DataCopy(inputLocal, inputGm[offset], len);
    } else {
        DataCopyExtParams copyParams = {1, copyBytes, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(inputLocal, inputGm[offset], copyParams, padParams);
    }
}

__aicore__ inline void CopyOutBoolAuto(GlobalTensor<uint8_t>& outputGm, int64_t offset,
                                       LocalTensor<uint8_t>& outputLocal, uint32_t len)
{
    if ((len % kAlignBytes) == 0 && (offset % kAlignBytes) == 0) {
        DataCopy(outputGm[offset], outputLocal, len);
    } else {
        DataCopyExtParams copyParams = {1, len, 0, 0, 0};
        DataCopyPad(outputGm[offset], outputLocal, copyParams);
    }
}

template <typename T>
struct TensorEqualStorage {
    using Type = T;
};

template <>
struct TensorEqualStorage<bool> {
    using Type = uint8_t;
};

template <>
struct TensorEqualStorage<half> {
    using Type = uint16_t;
};

template <>
struct TensorEqualStorage<bfloat16_t> {
    using Type = uint16_t;
};

template <typename T>
using TensorEqualStorageT = typename TensorEqualStorage<T>::Type;

template <typename T>
class KernelTensorEqual {
public:
    using StorageT = TensorEqualStorageT<T>;

    __aicore__ inline KernelTensorEqual() {}

    template <typename TilingData>
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR output, const TilingData& tiling)
    {
        const int64_t tileElems = tiling.tile_elems;
        tileElems_ = tileElems > 0 ? AlignUpElems(static_cast<uint32_t>(tileElems)) : kVectorAlignElems;
        x1Gm_.SetGlobalBuffer((__gm__ StorageT*)x1);
        x2Gm_.SetGlobalBuffer((__gm__ StorageT*)x2);
        outputGm_.SetGlobalBuffer((__gm__ uint8_t*)output);

        bool x1ScalarBuffer = false;
        bool x2ScalarBuffer = false;
        GetScalarBufferFlags(tiling, x1ScalarBuffer, x2ScalarBuffer);

        const uint32_t x1InputBufferBytes = InputBufferBytes(x1ScalarBuffer);
        const uint32_t x2InputBufferBytes = InputBufferBytes(x2ScalarBuffer);
        const uint32_t outputBufferBytes = AlignUpBytes(tileElems_ * sizeof(uint8_t));
        pipe_.InitBuffer(x1Queue_, kBufferNum, x1InputBufferBytes);
        pipe_.InitBuffer(x2Queue_, kBufferNum, x2InputBufferBytes);
        pipe_.InitBuffer(outputQueue_, 1, outputBufferBytes);
        pipe_.InitBuffer(maskBuf_, CmpMaskBytes(tileElems_));
        if constexpr (NeedsFloatCompareBuffer()) {
            pipe_.InitBuffer(calc1Buf_, CompareBufferBytes<float>(x1ScalarBuffer));
            pipe_.InitBuffer(calc2Buf_, CompareBufferBytes<float>(x2ScalarBuffer));
        } else if constexpr (NeedsHalfCompareBuffer()) {
            pipe_.InitBuffer(calc1Buf_, CompareBufferBytes<half>(x1ScalarBuffer));
            pipe_.InitBuffer(calc2Buf_, CompareBufferBytes<half>(x2ScalarBuffer));
        }
    }

    template <typename TilingData>
    __aicore__ inline void Process(const TilingData& tiling)
    {
        if (tiling.total_elems <= 0) {
            return;
        }

        if (tiling.same_shape != 0) {
            ProcessSameShape(tiling.total_elems);
            return;
        }
        bool x1AllBroadcast = false;
        bool x2AllBroadcast = false;
        GetAllBroadcastFlags(tiling, x1AllBroadcast, x2AllBroadcast);
        if (x1AllBroadcast && !x2AllBroadcast) {
            ProcessX1GlobalScalar(tiling.total_elems);
            return;
        }
        if (!x1AllBroadcast && x2AllBroadcast) {
            ProcessX2GlobalScalar(tiling.total_elems);
            return;
        }
        ProcessBroadcast(tiling);
    }

private:
    __aicore__ inline void ProcessSameShape(int64_t totalElems)
    {
        const int64_t tileCount = (totalElems + tileElems_ - 1) / tileElems_;
        const int64_t preload = tileCount < kBufferNum ? tileCount : kBufferNum;
        for (int64_t tile = 0; tile < preload; ++tile) {
            CopyIn(TileOffset(tile), TileLen(tile, totalElems));
        }

        for (int64_t tile = 0; tile < tileCount; ++tile) {
            const uint32_t len = TileLen(tile, totalElems);
            Compute(len);
            const int64_t nextTile = tile + kBufferNum;
            if (nextTile < tileCount) {
                CopyIn(TileOffset(nextTile), TileLen(nextTile, totalElems));
            }
            CopyOut(TileOffset(tile), len);
        }
    }

    __aicore__ inline int64_t TileOffset(int64_t tile)
    {
        return tile * static_cast<int64_t>(tileElems_);
    }

    __aicore__ inline uint32_t TileLen(int64_t tile, int64_t totalElems)
    {
        const int64_t offset = TileOffset(tile);
        uint32_t len = static_cast<uint32_t>(totalElems - offset);
        if (len > tileElems_) {
            len = tileElems_;
        }
        return len;
    }

    __aicore__ inline void CopyIn(int64_t offset, uint32_t len)
    {
        LocalTensor<StorageT> x1Local = x1Queue_.AllocTensor<StorageT>();
        LocalTensor<StorageT> x2Local = x2Queue_.AllocTensor<StorageT>();
        CopyInAuto(x1Gm_, offset, x1Local, len);
        CopyInAuto(x2Gm_, offset, x2Local, len);
        x1Queue_.EnQue(x1Local);
        x2Queue_.EnQue(x2Local);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        LocalTensor<StorageT> x1Local = x1Queue_.DeQue<StorageT>();
        LocalTensor<StorageT> x2Local = x2Queue_.DeQue<StorageT>();
        LocalTensor<uint8_t> outputLocal = outputQueue_.AllocTensor<uint8_t>();
        VectorEqualToBool(outputLocal, x1Local, x2Local, len, AlignUpElems(len), false, false);
        outputQueue_.EnQue(outputLocal);
        x1Queue_.FreeTensor(x1Local);
        x2Queue_.FreeTensor(x2Local);
    }

    __aicore__ inline void CopyOut(int64_t offset, uint32_t len)
    {
        LocalTensor<uint8_t> outputLocal = outputQueue_.DeQue<uint8_t>();
        CopyOutBoolAuto(outputGm_, offset, outputLocal, len);
        outputQueue_.FreeTensor(outputLocal);
    }

    template <typename TilingData>
    __aicore__ inline void ProcessBroadcast(const TilingData& tiling)
    {
        const int64_t outDims[kMaxRank] = {tiling.out_dim0, tiling.out_dim1, tiling.out_dim2, tiling.out_dim3,
                                           tiling.out_dim4, tiling.out_dim5, tiling.out_dim6, tiling.out_dim7};
        const int64_t x1Dims[kMaxRank] = {tiling.x1_dim0, tiling.x1_dim1, tiling.x1_dim2, tiling.x1_dim3,
                                          tiling.x1_dim4, tiling.x1_dim5, tiling.x1_dim6, tiling.x1_dim7};
        const int64_t x2Dims[kMaxRank] = {tiling.x2_dim0, tiling.x2_dim1, tiling.x2_dim2, tiling.x2_dim3,
                                          tiling.x2_dim4, tiling.x2_dim5, tiling.x2_dim6, tiling.x2_dim7};
        const int64_t x1Strides[kMaxRank] = {tiling.x1_stride0, tiling.x1_stride1, tiling.x1_stride2,
                                             tiling.x1_stride3, tiling.x1_stride4, tiling.x1_stride5,
                                             tiling.x1_stride6, tiling.x1_stride7};
        const int64_t x2Strides[kMaxRank] = {tiling.x2_stride0, tiling.x2_stride1, tiling.x2_stride2,
                                             tiling.x2_stride3, tiling.x2_stride4, tiling.x2_stride5,
                                             tiling.x2_stride6, tiling.x2_stride7};

        const int64_t rank = tiling.rank;
        const int64_t lastDim = outDims[rank - 1];
        const bool x1LastBroadcast = (x1Dims[rank - 1] == 1);
        const bool x2LastBroadcast = (x2Dims[rank - 1] == 1);
        const int64_t outerRows = tiling.total_elems / lastDim;
        if (OuterDimsSame(rank, x1Dims, x2Dims)) {
            if (x1LastBroadcast && !x2LastBroadcast) {
                ProcessBroadcastX1ScalarContiguous(outerRows, lastDim);
                return;
            }
            if (!x1LastBroadcast && x2LastBroadcast) {
                ProcessBroadcastX2ScalarContiguous(outerRows, lastDim);
                return;
            }
        }

        int64_t coords[kMaxRank] = {0, 0, 0, 0, 0, 0, 0, 0};
        int64_t x1Base = 0;
        int64_t x2Base = 0;
        const int64_t rowTileCount = (lastDim + tileElems_ - 1) / tileElems_;
        const int64_t preload = rowTileCount < kBufferNum ? rowTileCount : kBufferNum;

        for (int64_t row = 0; row < outerRows; ++row) {
            const int64_t outBase = row * lastDim;
            if (x1LastBroadcast && x2LastBroadcast) {
                ProcessBroadcastBothScalarRow(x1Base, x2Base, outBase, lastDim, rowTileCount);
            } else if (x1LastBroadcast && !x2LastBroadcast) {
                ProcessBroadcastX1ScalarRow(x1Base, x2Base, outBase, lastDim, rowTileCount, preload);
            } else if (!x1LastBroadcast && x2LastBroadcast) {
                ProcessBroadcastX2ScalarRow(x1Base, x2Base, outBase, lastDim, rowTileCount, preload);
            } else {
                for (int64_t tile = 0; tile < preload; ++tile) {
                    const int64_t inner = tile * static_cast<int64_t>(tileElems_);
                    CopyInBroadcastTile(x1Base, x2Base, inner, x1LastBroadcast, x2LastBroadcast,
                                        BroadcastTileLen(inner, lastDim));
                }

                for (int64_t tile = 0; tile < rowTileCount; ++tile) {
                    const int64_t inner = tile * static_cast<int64_t>(tileElems_);
                    const uint32_t len = BroadcastTileLen(inner, lastDim);
                    ComputeBroadcast(len, AlignUpElems(len), x1LastBroadcast, x2LastBroadcast);
                    const int64_t nextTile = tile + kBufferNum;
                    if (nextTile < rowTileCount) {
                        const int64_t nextInner = nextTile * static_cast<int64_t>(tileElems_);
                        CopyInBroadcastTile(x1Base, x2Base, nextInner, x1LastBroadcast, x2LastBroadcast,
                                            BroadcastTileLen(nextInner, lastDim));
                    }
                    CopyOut(outBase + inner, len);
                }
            }
            if (row + 1 < outerRows) {
                AdvanceOuter(outDims, x1Dims, x2Dims, x1Strides, x2Strides, rank, coords, x1Base, x2Base);
            }
        }
    }

    __aicore__ inline uint32_t BroadcastTileLen(int64_t inner, int64_t lastDim)
    {
        uint32_t len = static_cast<uint32_t>(lastDim - inner);
        if (len > tileElems_) {
            len = tileElems_;
        }
        return len;
    }

    __aicore__ inline void CopyInBroadcastTile(int64_t x1Base, int64_t x2Base,
                                               int64_t inner, bool x1LastBroadcast,
                                               bool x2LastBroadcast, uint32_t len)
    {
        LocalTensor<StorageT> x1Local = x1Queue_.AllocTensor<StorageT>();
        LocalTensor<StorageT> x2Local = x2Queue_.AllocTensor<StorageT>();
        CopyBroadcastInput(x1Gm_, x1Base, inner, x1LastBroadcast, x1Local, len);
        CopyBroadcastInput(x2Gm_, x2Base, inner, x2LastBroadcast, x2Local, len);
        x1Queue_.EnQue(x1Local);
        x2Queue_.EnQue(x2Local);
    }

    __aicore__ inline void CopyBroadcastInput(GlobalTensor<StorageT>& inputGm, int64_t baseOffset,
                                              int64_t inner, bool lastDimBroadcast,
                                              LocalTensor<StorageT>& inputLocal, uint32_t len)
    {
        if (lastDimBroadcast) {
            CopyInAuto(inputGm, baseOffset, inputLocal, 1);
            return;
        }
        CopyInAuto(inputGm, baseOffset + inner, inputLocal, len);
    }

    __aicore__ inline void CopyInX1Tile(int64_t offset, uint32_t len)
    {
        LocalTensor<StorageT> x1Local = x1Queue_.AllocTensor<StorageT>();
        CopyInAuto(x1Gm_, offset, x1Local, len);
        x1Queue_.EnQue(x1Local);
    }

    __aicore__ inline void CopyInX2Tile(int64_t offset, uint32_t len)
    {
        LocalTensor<StorageT> x2Local = x2Queue_.AllocTensor<StorageT>();
        CopyInAuto(x2Gm_, offset, x2Local, len);
        x2Queue_.EnQue(x2Local);
    }

    __aicore__ inline LocalTensor<StorageT> LoadX1Scalar(int64_t offset)
    {
        return LoadX1ScalarBlock(offset, 1);
    }

    __aicore__ inline LocalTensor<StorageT> LoadX2Scalar(int64_t offset)
    {
        return LoadX2ScalarBlock(offset, 1);
    }

    __aicore__ inline LocalTensor<StorageT> LoadX1ScalarBlock(int64_t offset, uint32_t len)
    {
        LocalTensor<StorageT> x1Local = x1Queue_.AllocTensor<StorageT>();
        CopyInAuto(x1Gm_, offset, x1Local, len);
        x1Queue_.EnQue(x1Local);
        return x1Queue_.DeQue<StorageT>();
    }

    __aicore__ inline LocalTensor<StorageT> LoadX2ScalarBlock(int64_t offset, uint32_t len)
    {
        LocalTensor<StorageT> x2Local = x2Queue_.AllocTensor<StorageT>();
        CopyInAuto(x2Gm_, offset, x2Local, len);
        x2Queue_.EnQue(x2Local);
        return x2Queue_.DeQue<StorageT>();
    }

    __aicore__ inline void ProcessBroadcastBothScalarRow(int64_t x1Base, int64_t x2Base, int64_t outBase,
                                                         int64_t lastDim, int64_t rowTileCount)
    {
        LocalTensor<StorageT> x1ScalarLocal = LoadX1Scalar(x1Base);
        LocalTensor<StorageT> x2ScalarLocal = LoadX2Scalar(x2Base);
        for (int64_t tile = 0; tile < rowTileCount; ++tile) {
            const int64_t inner = tile * static_cast<int64_t>(tileElems_);
            const uint32_t len = BroadcastTileLen(inner, lastDim);
            ComputeBroadcastBothScalar(x1ScalarLocal, x2ScalarLocal, len, AlignUpElems(len));
            CopyOut(outBase + inner, len);
        }
        x1Queue_.FreeTensor(x1ScalarLocal);
        x2Queue_.FreeTensor(x2ScalarLocal);
    }

    __aicore__ inline void ProcessBroadcastX1ScalarRow(int64_t x1Base, int64_t x2Base, int64_t outBase,
                                                       int64_t lastDim, int64_t rowTileCount, int64_t preload)
    {
        LocalTensor<StorageT> x1ScalarLocal = LoadX1Scalar(x1Base);
        ProcessBroadcastX1LoadedScalarRow(x1ScalarLocal, x2Base, outBase, lastDim, rowTileCount, preload);
        x1Queue_.FreeTensor(x1ScalarLocal);
    }

    __aicore__ inline void ProcessBroadcastX1LoadedScalarRow(LocalTensor<StorageT>& x1ScalarLocal,
                                                             int64_t x2Base, int64_t outBase,
                                                             int64_t lastDim, int64_t rowTileCount,
                                                             int64_t preload)
    {
        for (int64_t tile = 0; tile < preload; ++tile) {
            const int64_t inner = tile * static_cast<int64_t>(tileElems_);
            CopyInX2Tile(x2Base + inner, BroadcastTileLen(inner, lastDim));
        }

        for (int64_t tile = 0; tile < rowTileCount; ++tile) {
            const int64_t inner = tile * static_cast<int64_t>(tileElems_);
            const uint32_t len = BroadcastTileLen(inner, lastDim);
            ComputeBroadcastX1Scalar(x1ScalarLocal, len, AlignUpElems(len));
            const int64_t nextTile = tile + kBufferNum;
            if (nextTile < rowTileCount) {
                const int64_t nextInner = nextTile * static_cast<int64_t>(tileElems_);
                CopyInX2Tile(x2Base + nextInner, BroadcastTileLen(nextInner, lastDim));
            }
            CopyOut(outBase + inner, len);
        }
    }

    __aicore__ inline void ProcessBroadcastX2ScalarRow(int64_t x1Base, int64_t x2Base, int64_t outBase,
                                                       int64_t lastDim, int64_t rowTileCount, int64_t preload)
    {
        LocalTensor<StorageT> x2ScalarLocal = LoadX2Scalar(x2Base);
        ProcessBroadcastX2LoadedScalarRow(x2ScalarLocal, x1Base, outBase, lastDim, rowTileCount, preload);
        x2Queue_.FreeTensor(x2ScalarLocal);
    }

    __aicore__ inline void ProcessBroadcastX2LoadedScalarRow(LocalTensor<StorageT>& x2ScalarLocal,
                                                             int64_t x1Base, int64_t outBase,
                                                             int64_t lastDim, int64_t rowTileCount,
                                                             int64_t preload)
    {
        for (int64_t tile = 0; tile < preload; ++tile) {
            const int64_t inner = tile * static_cast<int64_t>(tileElems_);
            CopyInX1Tile(x1Base + inner, BroadcastTileLen(inner, lastDim));
        }

        for (int64_t tile = 0; tile < rowTileCount; ++tile) {
            const int64_t inner = tile * static_cast<int64_t>(tileElems_);
            const uint32_t len = BroadcastTileLen(inner, lastDim);
            ComputeBroadcastX2Scalar(x2ScalarLocal, len, AlignUpElems(len));
            const int64_t nextTile = tile + kBufferNum;
            if (nextTile < rowTileCount) {
                const int64_t nextInner = nextTile * static_cast<int64_t>(tileElems_);
                CopyInX1Tile(x1Base + nextInner, BroadcastTileLen(nextInner, lastDim));
            }
            CopyOut(outBase + inner, len);
        }
    }

    __aicore__ inline void ProcessBroadcastX1ScalarContiguous(int64_t outerRows, int64_t lastDim)
    {
        const int64_t rowTileCount = (lastDim + tileElems_ - 1) / tileElems_;
        const int64_t preload = rowTileCount < kBufferNum ? rowTileCount : kBufferNum;
        for (int64_t rowBase = 0; rowBase < outerRows; rowBase += kScalarBlockElems) {
            uint32_t rows = kScalarBlockElems;
            const int64_t remain = outerRows - rowBase;
            if (remain < static_cast<int64_t>(rows)) {
                rows = static_cast<uint32_t>(remain);
            }
            LocalTensor<StorageT> x1ScalarLocal = LoadX1ScalarBlock(rowBase, rows);
            for (uint32_t r = 0; r < rows; ++r) {
                if (r != 0) {
                    x1ScalarLocal.SetValue(0, x1ScalarLocal.GetValue(r));
                }
                const int64_t row = rowBase + static_cast<int64_t>(r);
                const int64_t outBase = row * lastDim;
                ProcessBroadcastX1LoadedScalarRow(x1ScalarLocal, outBase, outBase, lastDim, rowTileCount, preload);
            }
            x1Queue_.FreeTensor(x1ScalarLocal);
        }
    }

    __aicore__ inline void ProcessBroadcastX2ScalarContiguous(int64_t outerRows, int64_t lastDim)
    {
        const int64_t rowTileCount = (lastDim + tileElems_ - 1) / tileElems_;
        const int64_t preload = rowTileCount < kBufferNum ? rowTileCount : kBufferNum;
        for (int64_t rowBase = 0; rowBase < outerRows; rowBase += kScalarBlockElems) {
            uint32_t rows = kScalarBlockElems;
            const int64_t remain = outerRows - rowBase;
            if (remain < static_cast<int64_t>(rows)) {
                rows = static_cast<uint32_t>(remain);
            }
            LocalTensor<StorageT> x2ScalarLocal = LoadX2ScalarBlock(rowBase, rows);
            for (uint32_t r = 0; r < rows; ++r) {
                if (r != 0) {
                    x2ScalarLocal.SetValue(0, x2ScalarLocal.GetValue(r));
                }
                const int64_t row = rowBase + static_cast<int64_t>(r);
                const int64_t outBase = row * lastDim;
                ProcessBroadcastX2LoadedScalarRow(x2ScalarLocal, outBase, outBase, lastDim, rowTileCount, preload);
            }
            x2Queue_.FreeTensor(x2ScalarLocal);
        }
    }

    __aicore__ inline void ProcessX1GlobalScalar(int64_t totalElems)
    {
        LocalTensor<StorageT> x1ScalarLocal = LoadX1Scalar(0);
        const int64_t tileCount = (totalElems + tileElems_ - 1) / tileElems_;
        const int64_t preload = tileCount < kBufferNum ? tileCount : kBufferNum;
        for (int64_t tile = 0; tile < preload; ++tile) {
            CopyInX2Tile(TileOffset(tile), TileLen(tile, totalElems));
        }

        for (int64_t tile = 0; tile < tileCount; ++tile) {
            const uint32_t len = TileLen(tile, totalElems);
            ComputeBroadcastX1Scalar(x1ScalarLocal, len, AlignUpElems(len));
            const int64_t nextTile = tile + kBufferNum;
            if (nextTile < tileCount) {
                CopyInX2Tile(TileOffset(nextTile), TileLen(nextTile, totalElems));
            }
            CopyOut(TileOffset(tile), len);
        }
        x1Queue_.FreeTensor(x1ScalarLocal);
    }

    __aicore__ inline void ProcessX2GlobalScalar(int64_t totalElems)
    {
        LocalTensor<StorageT> x2ScalarLocal = LoadX2Scalar(0);
        const int64_t tileCount = (totalElems + tileElems_ - 1) / tileElems_;
        const int64_t preload = tileCount < kBufferNum ? tileCount : kBufferNum;
        for (int64_t tile = 0; tile < preload; ++tile) {
            CopyInX1Tile(TileOffset(tile), TileLen(tile, totalElems));
        }

        for (int64_t tile = 0; tile < tileCount; ++tile) {
            const uint32_t len = TileLen(tile, totalElems);
            ComputeBroadcastX2Scalar(x2ScalarLocal, len, AlignUpElems(len));
            const int64_t nextTile = tile + kBufferNum;
            if (nextTile < tileCount) {
                CopyInX1Tile(TileOffset(nextTile), TileLen(nextTile, totalElems));
            }
            CopyOut(TileOffset(tile), len);
        }
        x2Queue_.FreeTensor(x2ScalarLocal);
    }

    __aicore__ inline void AdvanceOuter(const int64_t (&outDims)[kMaxRank], const int64_t (&x1Dims)[kMaxRank],
                                        const int64_t (&x2Dims)[kMaxRank], const int64_t (&x1Strides)[kMaxRank],
                                        const int64_t (&x2Strides)[kMaxRank], int64_t rank,
                                        int64_t (&coords)[kMaxRank], int64_t& x1Index, int64_t& x2Index)
    {
        for (int64_t dim = rank - 2; dim >= 0; --dim) {
            coords[dim] += 1;
            if (x1Dims[dim] != 1) {
                x1Index += x1Strides[dim];
            }
            if (x2Dims[dim] != 1) {
                x2Index += x2Strides[dim];
            }
            if (coords[dim] < outDims[dim]) {
                break;
            }

            coords[dim] = 0;
            if (x1Dims[dim] != 1) {
                x1Index -= outDims[dim] * x1Strides[dim];
            }
            if (x2Dims[dim] != 1) {
                x2Index -= outDims[dim] * x2Strides[dim];
            }
        }
    }

    __aicore__ inline void VectorEqualToBool(LocalTensor<uint8_t>& outputLocal,
                                             LocalTensor<StorageT>& x1Local,
                                             LocalTensor<StorageT>& x2Local, uint32_t len,
                                             uint32_t calcLen,
                                             bool x1Scalar, bool x2Scalar)
    {
        LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
            LocalTensor<half> x1Compare = x1Local.template ReinterpretCast<half>();
            LocalTensor<half> x2Compare = x2Local.template ReinterpretCast<half>();
            CompareWithOptionalScalar(maskLocal, x1Compare, x2Compare, calcLen, x1Scalar, x2Scalar);
            LocalTensor<half> resultLocal = ResultScratchFromInputs(x1Local, x2Local, x1Scalar, x2Scalar);
            ExpandMaskToBool(outputLocal, maskLocal, resultLocal, calcLen);
        } else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t>) {
            LocalTensor<float> x1Compare = calc1Buf_.Get<float>();
            LocalTensor<float> x2Compare = calc2Buf_.Get<float>();
            Cast(x1Compare, x1Local, RoundMode::CAST_NONE, ScalarCastLen(calcLen, x1Scalar));
            Cast(x2Compare, x2Local, RoundMode::CAST_NONE, ScalarCastLen(calcLen, x2Scalar));
            PipeBarrier<PIPE_V>();
            CompareWithOptionalScalar(maskLocal, x1Compare, x2Compare, calcLen, x1Scalar, x2Scalar);
            LocalTensor<half> resultLocal = (x1Scalar && !x2Scalar) ? calc2Buf_.Get<half>() : calc1Buf_.Get<half>();
            ExpandMaskToBool(outputLocal, maskLocal, resultLocal, calcLen);
        } else if constexpr (NeedsHalfCompareBuffer()) {
            LocalTensor<half> x1Compare = calc1Buf_.Get<half>();
            LocalTensor<half> x2Compare = calc2Buf_.Get<half>();
            Cast(x1Compare, x1Local, RoundMode::CAST_NONE, ScalarCastLen(calcLen, x1Scalar));
            Cast(x2Compare, x2Local, RoundMode::CAST_NONE, ScalarCastLen(calcLen, x2Scalar));
            PipeBarrier<PIPE_V>();
            CompareWithOptionalScalar(maskLocal, x1Compare, x2Compare, calcLen, x1Scalar, x2Scalar);
            LocalTensor<half> resultLocal = (x1Scalar && !x2Scalar) ? x2Compare : x1Compare;
            ExpandMaskToBool(outputLocal, maskLocal, resultLocal, calcLen);
        } else {
            CompareWithOptionalScalar(maskLocal, x1Local, x2Local, calcLen, x1Scalar, x2Scalar);
            LocalTensor<half> resultLocal = ResultScratchFromInputs(x1Local, x2Local, x1Scalar, x2Scalar);
            ExpandMaskToBool(outputLocal, maskLocal, resultLocal, calcLen);
        }
    }

    __aicore__ inline LocalTensor<half> ResultScratchFromInputs(LocalTensor<StorageT>& x1Local,
                                                               LocalTensor<StorageT>& x2Local,
                                                               bool x1Scalar, bool x2Scalar)
    {
        if (x1Scalar && !x2Scalar) {
            return x2Local.template ReinterpretCast<half>();
        }
        return x1Local.template ReinterpretCast<half>();
    }

    __aicore__ inline void ExpandMaskToBool(LocalTensor<uint8_t>& outputLocal,
                                            LocalTensor<uint8_t>& maskLocal,
                                            LocalTensor<half>& resultLocal,
                                            uint32_t calcLen)
    {
        PipeBarrier<PIPE_V>();
        Duplicate(resultLocal, static_cast<half>(1.0f), calcLen);
        PipeBarrier<PIPE_V>();
        Select<half, uint8_t>(resultLocal, maskLocal, resultLocal, static_cast<half>(0.0f),
                              SELMODE::VSEL_TENSOR_SCALAR_MODE, calcLen);
        PipeBarrier<PIPE_V>();
        Cast(outputLocal, resultLocal, RoundMode::CAST_NONE, calcLen);
    }

    __aicore__ inline void ComputeBroadcast(uint32_t len, uint32_t calcLen, bool x1Scalar, bool x2Scalar)
    {
        LocalTensor<StorageT> x1Local = x1Queue_.DeQue<StorageT>();
        LocalTensor<StorageT> x2Local = x2Queue_.DeQue<StorageT>();
        LocalTensor<uint8_t> outputLocal = outputQueue_.AllocTensor<uint8_t>();
        VectorEqualToBool(outputLocal, x1Local, x2Local, len, calcLen, x1Scalar, x2Scalar);
        outputQueue_.EnQue(outputLocal);
        x1Queue_.FreeTensor(x1Local);
        x2Queue_.FreeTensor(x2Local);
    }

    __aicore__ inline void ComputeBroadcastBothScalar(LocalTensor<StorageT>& x1ScalarLocal,
                                                      LocalTensor<StorageT>& x2ScalarLocal,
                                                      uint32_t len, uint32_t calcLen)
    {
        LocalTensor<uint8_t> outputLocal = outputQueue_.AllocTensor<uint8_t>();
        VectorEqualToBool(outputLocal, x1ScalarLocal, x2ScalarLocal, len, calcLen, true, true);
        outputQueue_.EnQue(outputLocal);
    }

    __aicore__ inline void ComputeBroadcastX1Scalar(LocalTensor<StorageT>& x1ScalarLocal,
                                                    uint32_t len, uint32_t calcLen)
    {
        LocalTensor<StorageT> x2Local = x2Queue_.DeQue<StorageT>();
        LocalTensor<uint8_t> outputLocal = outputQueue_.AllocTensor<uint8_t>();
        VectorEqualToBool(outputLocal, x1ScalarLocal, x2Local, len, calcLen, true, false);
        outputQueue_.EnQue(outputLocal);
        x2Queue_.FreeTensor(x2Local);
    }

    __aicore__ inline void ComputeBroadcastX2Scalar(LocalTensor<StorageT>& x2ScalarLocal,
                                                    uint32_t len, uint32_t calcLen)
    {
        LocalTensor<StorageT> x1Local = x1Queue_.DeQue<StorageT>();
        LocalTensor<uint8_t> outputLocal = outputQueue_.AllocTensor<uint8_t>();
        VectorEqualToBool(outputLocal, x1Local, x2ScalarLocal, len, calcLen, false, true);
        outputQueue_.EnQue(outputLocal);
        x1Queue_.FreeTensor(x1Local);
    }

    template <typename CompareT>
    __aicore__ inline void CompareWithOptionalScalar(LocalTensor<uint8_t>& maskLocal,
                                                     LocalTensor<CompareT>& x1Compare,
                                                     LocalTensor<CompareT>& x2Compare,
                                                     uint32_t calcLen,
                                                     bool x1Scalar, bool x2Scalar)
    {
        if (x1Scalar && !x2Scalar) {
            const CompareT scalar = x1Compare.GetValue(0);
            CompareScalar<CompareT, uint8_t>(maskLocal, x2Compare, scalar, CMPMODE::EQ, calcLen);
            return;
        }
        if (!x1Scalar && x2Scalar) {
            const CompareT scalar = x2Compare.GetValue(0);
            CompareScalar<CompareT, uint8_t>(maskLocal, x1Compare, scalar, CMPMODE::EQ, calcLen);
            return;
        }
        if (x1Scalar && x2Scalar) {
            const CompareT x1Value = x1Compare.GetValue(0);
            const CompareT x2Value = x2Compare.GetValue(0);
            Duplicate(x1Compare, x1Value, calcLen);
            PipeBarrier<PIPE_V>();
            CompareScalar<CompareT, uint8_t>(maskLocal, x1Compare, x2Value, CMPMODE::EQ, calcLen);
            return;
        }
        Compare<CompareT, uint8_t>(maskLocal, x1Compare, x2Compare, CMPMODE::EQ, calcLen);
    }

    __aicore__ inline uint32_t ScalarCastLen(uint32_t len, bool isScalar)
    {
        return isScalar ? kVectorAlignElems : len;
    }

    template <typename TilingData>
    __aicore__ inline void GetScalarBufferFlags(const TilingData& tiling, bool& x1ScalarBuffer,
                                                bool& x2ScalarBuffer)
    {
        x1ScalarBuffer = false;
        x2ScalarBuffer = false;
        if (tiling.same_shape != 0 || tiling.rank <= 0) {
            return;
        }

        const int64_t x1Dims[kMaxRank] = {tiling.x1_dim0, tiling.x1_dim1, tiling.x1_dim2, tiling.x1_dim3,
                                          tiling.x1_dim4, tiling.x1_dim5, tiling.x1_dim6, tiling.x1_dim7};
        const int64_t x2Dims[kMaxRank] = {tiling.x2_dim0, tiling.x2_dim1, tiling.x2_dim2, tiling.x2_dim3,
                                          tiling.x2_dim4, tiling.x2_dim5, tiling.x2_dim6, tiling.x2_dim7};
        bool x1AllBroadcast = false;
        bool x2AllBroadcast = false;
        GetAllBroadcastFlagsFromDims(tiling.rank, x1Dims, x2Dims, x1AllBroadcast, x2AllBroadcast);
        if (x1AllBroadcast && !x2AllBroadcast) {
            x1ScalarBuffer = true;
            return;
        }
        if (!x1AllBroadcast && x2AllBroadcast) {
            x2ScalarBuffer = true;
            return;
        }
        const int64_t last = tiling.rank - 1;
        const bool x1LastBroadcast = (x1Dims[last] == 1);
        const bool x2LastBroadcast = (x2Dims[last] == 1);
        x1ScalarBuffer = x1LastBroadcast && !x2LastBroadcast;
        x2ScalarBuffer = x2LastBroadcast;
    }

    template <typename TilingData>
    __aicore__ inline void GetAllBroadcastFlags(const TilingData& tiling, bool& x1AllBroadcast,
                                                bool& x2AllBroadcast)
    {
        if (tiling.same_shape != 0 || tiling.rank <= 0) {
            x1AllBroadcast = false;
            x2AllBroadcast = false;
            return;
        }
        const int64_t x1Dims[kMaxRank] = {tiling.x1_dim0, tiling.x1_dim1, tiling.x1_dim2, tiling.x1_dim3,
                                          tiling.x1_dim4, tiling.x1_dim5, tiling.x1_dim6, tiling.x1_dim7};
        const int64_t x2Dims[kMaxRank] = {tiling.x2_dim0, tiling.x2_dim1, tiling.x2_dim2, tiling.x2_dim3,
                                          tiling.x2_dim4, tiling.x2_dim5, tiling.x2_dim6, tiling.x2_dim7};
        GetAllBroadcastFlagsFromDims(tiling.rank, x1Dims, x2Dims, x1AllBroadcast, x2AllBroadcast);
    }

    __aicore__ inline void GetAllBroadcastFlagsFromDims(int64_t rank, const int64_t (&x1Dims)[kMaxRank],
                                                        const int64_t (&x2Dims)[kMaxRank],
                                                        bool& x1AllBroadcast, bool& x2AllBroadcast)
    {
        x1AllBroadcast = true;
        x2AllBroadcast = true;
        for (int64_t i = 0; i < rank; ++i) {
            if (x1Dims[i] != 1) {
                x1AllBroadcast = false;
            }
            if (x2Dims[i] != 1) {
                x2AllBroadcast = false;
            }
        }
    }

    __aicore__ inline bool OuterDimsSame(int64_t rank, const int64_t (&x1Dims)[kMaxRank],
                                         const int64_t (&x2Dims)[kMaxRank])
    {
        for (int64_t i = 0; i + 1 < rank; ++i) {
            if (x1Dims[i] != x2Dims[i]) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline uint32_t ScalarInputElems()
    {
        if constexpr (NeedsFloatCompareBuffer() || NeedsHalfCompareBuffer()) {
            return kVectorAlignElems;
        }
        return kScalarBlockElems;
    }

    __aicore__ inline uint32_t InputBufferBytes(bool scalarBuffer)
    {
        const uint32_t elems = scalarBuffer ? ScalarInputElems() : tileElems_;
        return AlignUpBytes(elems * sizeof(StorageT));
    }

    template <typename CompareT>
    __aicore__ inline uint32_t CompareBufferBytes(bool scalarBuffer)
    {
        const uint32_t elems = scalarBuffer ? kVectorAlignElems : tileElems_;
        return AlignUpBytes(elems * sizeof(CompareT));
    }

    __aicore__ static constexpr bool NeedsFloatCompareBuffer()
    {
        return std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t>;
    }

    __aicore__ static constexpr bool NeedsHalfCompareBuffer()
    {
        return std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, bool>;
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, kBufferNum> x1Queue_;
    TQue<QuePosition::VECIN, kBufferNum> x2Queue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TBuf<QuePosition::VECCALC> maskBuf_;
    TBuf<QuePosition::VECCALC> calc1Buf_;
    TBuf<QuePosition::VECCALC> calc2Buf_;
    GlobalTensor<StorageT> x1Gm_;
    GlobalTensor<StorageT> x2Gm_;
    GlobalTensor<uint8_t> outputGm_;
    uint32_t tileElems_ = 1;
};

template <typename T, typename TilingData>
__aicore__ inline void RunTensorEqual(GM_ADDR x1, GM_ADDR x2, GM_ADDR output, const TilingData& tiling)
{
    KernelTensorEqual<T> kernel;
    kernel.Init(x1, x2, output, tiling);
    kernel.Process(tiling);
}

}  // namespace

extern "C" __global__ __aicore__ void tensor_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR output, GM_ADDR workspace,
                                                   GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    RunTensorEqual<DTYPE_X1>(x1, x2, output, tilingData);
}
