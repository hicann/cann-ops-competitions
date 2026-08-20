#include "kernel_inc.h"

#include <type_traits>

namespace {
constexpr uint32_t kBufferNum = 2;
constexpr uint32_t kSameShapeBindBufferNum = 2;
constexpr uint32_t kFastPathTileBytes = 8192;
constexpr uint32_t kSameShapeTileBytes = 32768;
constexpr uint32_t kFastPathTileBytes1B = 65536;
constexpr uint32_t kSameShapeBindTileBytes = 98304;
constexpr uint32_t kSameShapeBindTileBytes2B = 196608;
constexpr uint32_t kSameShapeBindTileBytes4B = 196608;
constexpr uint32_t kSameShapeBindTileBytes4BVeryHuge = 196608;
constexpr uint32_t kSameShapeBind2BThresholdElems = 8192;
constexpr int64_t kSameShapeBind4BVeryHugeThresholdElems = 262144;
constexpr uint32_t kSameShapeTinyDirectBytes = 1024;
constexpr uint32_t kAlignBytes = 32;
constexpr uint32_t kSameShapeDoubleBufferBudgetBytes = 98304;
constexpr int64_t kMaxRank = 8;
constexpr int64_t kBroadcastModeGeneric = 0;
constexpr int64_t kBroadcastModeScalar = 1;
constexpr int64_t kBroadcastModeLastDimContiguous = 2;

template <typename T>
__aicore__ inline void CopyInPad(GlobalTensor<T> &inputGm, int64_t offset, LocalTensor<T> &inputLocal, uint32_t len) {
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(inputLocal, inputGm[offset], copyParams, padParams);
}

template <typename T>
__aicore__ inline void CopyOutPad(GlobalTensor<T> &outputGm, int64_t offset, LocalTensor<T> &outputLocal, uint32_t len) {
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGm[offset], outputLocal, copyParams);
}

template <typename T>
__aicore__ inline void CopyInAuto(GlobalTensor<T> &inputGm, int64_t offset, LocalTensor<T> &inputLocal, uint32_t len) {
    const uint32_t copyBytes = len * sizeof(T);
    if ((copyBytes % kAlignBytes) == 0) {
        DataCopy(inputLocal, inputGm[offset], len);
    } else {
        CopyInPad(inputGm, offset, inputLocal, len);
    }
}

template <typename T>
__aicore__ inline void CopyOutAuto(GlobalTensor<T> &outputGm, int64_t offset, LocalTensor<T> &outputLocal, uint32_t len) {
    const uint32_t copyBytes = len * sizeof(T);
    if ((copyBytes % kAlignBytes) == 0) {
        DataCopy(outputGm[offset], outputLocal, len);
    } else {
        CopyOutPad(outputGm, offset, outputLocal, len);
    }
}

template <typename T>
struct AssignStorage {
    using Type = T;
};

template <>
struct AssignStorage<bool> {
    using Type = uint8_t;
};

template <typename T>
using AssignStorageT = typename AssignStorage<T>::Type;

template <typename StorageT>
__aicore__ inline uint32_t GetDirectCopyTileBytes(int64_t spanElems) {
    auto clampForBufferNum = [](uint32_t tileBytes) -> uint32_t {
        if constexpr (kSameShapeBindBufferNum > 1) {
            return tileBytes > kSameShapeDoubleBufferBudgetBytes ? kSameShapeDoubleBufferBudgetBytes : tileBytes;
        }
        return tileBytes;
    };
    if constexpr (sizeof(StorageT) == 1) {
        return clampForBufferNum(kFastPathTileBytes1B);
    } else if constexpr (sizeof(StorageT) == 2) {
        const uint32_t tileBytes =
            (spanElems >= static_cast<int64_t>(kSameShapeBind2BThresholdElems)) ? kSameShapeBindTileBytes2B
                                                                                 : kSameShapeBindTileBytes;
        return clampForBufferNum(tileBytes);
    } else if constexpr (sizeof(StorageT) == 4) {
        const uint32_t tileBytes =
            (spanElems >= kSameShapeBind4BVeryHugeThresholdElems) ? kSameShapeBindTileBytes4BVeryHuge
                                                                  : kSameShapeBindTileBytes4B;
        return clampForBufferNum(tileBytes);
    } else if constexpr (std::is_same_v<StorageT, bfloat16_t>) {
        return clampForBufferNum(kSameShapeBindTileBytes);
    } else {
        return clampForBufferNum(kSameShapeBindTileBytes2B);
    }
}

template <typename T>
class KernelAssignSameShape {
  public:
    using StorageT = AssignStorageT<T>;

    __aicore__ inline void Init(TPipe *pipe, GM_ADDR other, GM_ADDR output, const AssignTilingData &tiling) {
        InitSocState();
        blockIdx_ = GetBlockIdx();
        usedCoreNum_ = tiling.used_core_num;
        blockFactor_ = tiling.block_factor;
        tailBlockFactor_ = tiling.tail_block_factor;
        ubFactor_ = tiling.ub_factor;
        tailBlockTailUbFactor_ = tiling.tail_block_tail_ub_factor;
        isTailCore_ = (blockIdx_ + 1 == usedCoreNum_);
        blockLength_ = isTailCore_ ? tailBlockFactor_ : blockFactor_;
        blockOffset_ = static_cast<int64_t>(blockIdx_) * blockFactor_ * ubFactor_;

        otherGm_.SetGlobalBuffer((__gm__ StorageT *)other);
        outputGm_.SetGlobalBuffer((__gm__ StorageT *)output);
        if (ubFactor_ <= 0) {
            ubFactor_ = 1;
        }
        pipe->InitBuffer(bindQueue_, kSameShapeBindBufferNum,
                         static_cast<uint32_t>(ubFactor_ * sizeof(StorageT)));
    }

    __aicore__ inline void Process() {
        if (blockIdx_ >= usedCoreNum_ || blockLength_ <= 0) {
            return;
        }
        for (int64_t loopIdx = 0; loopIdx < blockLength_; ++loopIdx) {
            uint32_t currentUbFactor = static_cast<uint32_t>(ubFactor_);
            if (isTailCore_ && (loopIdx + 1 == blockLength_)) {
                currentUbFactor = static_cast<uint32_t>(tailBlockTailUbFactor_);
            }
            const int64_t offset = blockOffset_ + loopIdx * ubFactor_;
            CopyInOutBind(offset, currentUbFactor);
        }
    }

  private:
    __aicore__ inline void CopyInOutBind(int64_t offset, uint32_t len) {
        LocalTensor<StorageT> bindLocal = bindQueue_.AllocTensor<StorageT>();
        CopyInAuto(otherGm_, offset, bindLocal, len);
        bindQueue_.EnQue(bindLocal);
        LocalTensor<StorageT> readyLocal = bindQueue_.DeQue<StorageT>();
        CopyOutAuto(outputGm_, offset, readyLocal, len);
        bindQueue_.FreeTensor(readyLocal);
    }

  private:
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, kSameShapeBindBufferNum> bindQueue_;
    GlobalTensor<StorageT> otherGm_;
    GlobalTensor<StorageT> outputGm_;
    uint32_t blockIdx_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t ubFactor_ = 1;
    int64_t blockFactor_ = 1;
    int64_t tailBlockFactor_ = 1;
    int64_t tailBlockTailUbFactor_ = 1;
    int64_t blockOffset_ = 0;
    int64_t blockLength_ = 0;
    bool isTailCore_ = false;
};

class KernelAssignBroadcast {
  public:
    __aicore__ inline void Init(GM_ADDR other, GM_ADDR output) {
        otherGm.SetGlobalBuffer((__gm__ uint8_t *)other);
        outputGm.SetGlobalBuffer((__gm__ uint8_t *)output);
    }

    template <typename Tiling>
    __aicore__ inline void Process(const Tiling &tiling) {
        if (tiling.total_elems <= 0) {
            return;
        }
        CopyBroadcast(tiling);
    }

  private:
    template <typename Tiling>
    __aicore__ inline void CopyBroadcast(const Tiling &tiling) {
        const uint32_t blockIdx = GetBlockIdx();
        const int64_t start = blockIdx * tiling.elems_per_core;
        if (start >= tiling.total_elems) {
            return;
        }

        const int64_t end = start + tiling.elems_per_core > tiling.total_elems ? tiling.total_elems : start + tiling.elems_per_core;
        const int64_t otherDims[4] = {tiling.other_dim0, tiling.other_dim1, tiling.other_dim2, tiling.other_dim3};
        const int64_t outStrides[4] = {tiling.out_stride0, tiling.out_stride1, tiling.out_stride2, tiling.out_stride3};
        const int64_t otherStrides[4] = {tiling.other_stride0, tiling.other_stride1, tiling.other_stride2, tiling.other_stride3};

        for (int64_t linear = start; linear < end; ++linear) {
            int64_t remain = linear;
            int64_t otherIndex = 0;
            for (int64_t dim = 0; dim < tiling.rank; ++dim) {
                const int64_t coord = remain / outStrides[dim];
                remain %= outStrides[dim];
                if (otherDims[dim] != 1) {
                    otherIndex += coord * otherStrides[dim];
                }
            }

            const int64_t dstByte = linear * tiling.elem_size;
            const int64_t srcByte = otherIndex * tiling.elem_size;
            for (int64_t i = 0; i < tiling.elem_size; ++i) {
                outputGm.SetValue(dstByte + i, otherGm.GetValue(srcByte + i));
            }
        }
    }

    GlobalTensor<uint8_t> otherGm;
    GlobalTensor<uint8_t> outputGm;
};

template <typename T>
class KernelAssignGeneric {
  public:
    using StorageT = AssignStorageT<T>;

    __aicore__ inline void Init(GM_ADDR other, GM_ADDR output) {
        otherGm_.SetGlobalBuffer((__gm__ StorageT *)other);
        outputGm_.SetGlobalBuffer((__gm__ StorageT *)output);
    }

    __aicore__ inline void Process(const AssignTilingData &tiling) {
        if (tiling.total_elems <= 0) {
            return;
        }
        const uint32_t blockIdx = GetBlockIdx();
        const int64_t start = static_cast<int64_t>(blockIdx) * tiling.elems_per_core;
        if (start >= tiling.total_elems) {
            return;
        }
        int64_t end = start + tiling.elems_per_core;
        if (end > tiling.total_elems) {
            end = tiling.total_elems;
        }

        const int64_t outDims[kMaxRank] = {tiling.dim0, tiling.dim1, tiling.dim2, tiling.dim3,
                                           tiling.dim4, tiling.dim5, tiling.dim6, tiling.dim7};
        const int64_t outStrides[kMaxRank] = {tiling.out_stride0, tiling.out_stride1, tiling.out_stride2,
                                              tiling.out_stride3, tiling.out_stride4, tiling.out_stride5,
                                              tiling.out_stride6, tiling.out_stride7};
        const int64_t otherDims[kMaxRank] = {tiling.other_dim0, tiling.other_dim1, tiling.other_dim2,
                                             tiling.other_dim3, tiling.other_dim4, tiling.other_dim5,
                                             tiling.other_dim6, tiling.other_dim7};
        const int64_t otherStrides[kMaxRank] = {tiling.other_stride0, tiling.other_stride1, tiling.other_stride2,
                                                tiling.other_stride3, tiling.other_stride4, tiling.other_stride5,
                                                tiling.other_stride6, tiling.other_stride7};
        int64_t coords[kMaxRank];
        InitGenericCoords(start, outStrides, tiling.rank, coords);
        int64_t otherIndex = InitGenericOtherIndex(coords, otherDims, otherStrides, tiling.rank);

        for (int64_t linear = start; linear < end; ++linear) {
            outputGm_.SetValue(linear, otherGm_.GetValue(otherIndex));
            if (linear + 1 < end) {
                AdvanceGenericState(outDims, otherDims, otherStrides, tiling.rank, coords, otherIndex);
            }
        }
    }

  private:
    __aicore__ inline void InitGenericCoords(int64_t linear, const int64_t (&outStrides)[kMaxRank], int64_t rank,
                                             int64_t (&coords)[kMaxRank]) {
        for (int64_t i = 0; i < kMaxRank; ++i) {
            coords[i] = 0;
        }
        int64_t remain = linear;
        for (int64_t dim = 0; dim < rank; ++dim) {
            const int64_t coord = remain / outStrides[dim];
            remain -= coord * outStrides[dim];
            coords[dim] = coord;
        }
    }

    __aicore__ inline int64_t InitGenericOtherIndex(const int64_t (&coords)[kMaxRank],
                                                    const int64_t (&otherDims)[kMaxRank],
                                                    const int64_t (&otherStrides)[kMaxRank], int64_t rank) {
        int64_t otherIndex = 0;
        for (int64_t dim = 0; dim < rank; ++dim) {
            if (otherDims[dim] != 1) {
                otherIndex += coords[dim] * otherStrides[dim];
            }
        }
        return otherIndex;
    }

    __aicore__ inline void AdvanceGenericState(const int64_t (&outDims)[kMaxRank],
                                               const int64_t (&otherDims)[kMaxRank],
                                               const int64_t (&otherStrides)[kMaxRank], int64_t rank,
                                               int64_t (&coords)[kMaxRank], int64_t &otherIndex) {
        for (int64_t dim = rank - 1; dim >= 0; --dim) {
            ++coords[dim];
            if (coords[dim] < outDims[dim]) {
                if (otherDims[dim] != 1) {
                    otherIndex += otherStrides[dim];
                }
                return;
            }
            coords[dim] = 0;
            if (otherDims[dim] != 1) {
                otherIndex -= (outDims[dim] - 1) * otherStrides[dim];
            }
        }
    }

  private:
    GlobalTensor<StorageT> otherGm_;
    GlobalTensor<StorageT> outputGm_;
};

template <typename T>
class KernelAssignScalar {
  public:
    using StorageT = AssignStorageT<T>;

    __aicore__ inline void Init(TPipe *pipe, GM_ADDR other, GM_ADDR output, int64_t blockOffset, int64_t blockLength) {
        InitSocState();
        blockLength_ = blockLength;
        tileLength_ = kFastPathTileBytes / sizeof(StorageT);
        if (tileLength_ == 0) {
            tileLength_ = 1;
        }
        if (static_cast<int64_t>(tileLength_) > blockLength_) {
            tileLength_ = static_cast<uint32_t>(blockLength_);
        }
        if (tileLength_ == 0) {
            tileLength_ = 1;
        }

        const uint32_t alignElems = 32 / sizeof(StorageT);
        allocTileLength_ = tileLength_ < alignElems ? alignElems : tileLength_;
        scalarValue_ = ((__gm__ StorageT *)other)[0];
        outputGm_.SetGlobalBuffer((__gm__ StorageT *)output + blockOffset, blockLength_);
        pipe->InitBuffer(outQueue_, kBufferNum, allocTileLength_ * sizeof(StorageT));
    }

    __aicore__ inline void Process() {
        for (int64_t offset = 0; offset < blockLength_; offset += tileLength_) {
            uint32_t currentTileLength = static_cast<uint32_t>(blockLength_ - offset);
            if (currentTileLength > tileLength_) {
                currentTileLength = tileLength_;
            }
            Fill(currentTileLength);
            CopyOut(offset, currentTileLength);
        }
    }

  private:
    __aicore__ inline void Fill(uint32_t len) {
        LocalTensor<StorageT> outputLocal = outQueue_.AllocTensor<StorageT>();
        if constexpr (std::is_same_v<StorageT, half> || std::is_same_v<StorageT, bfloat16_t> ||
                      std::is_same_v<StorageT, int16_t> || std::is_same_v<StorageT, int32_t> ||
                      std::is_same_v<StorageT, float>) {
            const uint32_t alignElems = 32 / sizeof(StorageT);
            const uint32_t aligned = (len / alignElems) * alignElems;
            if (aligned > 0) {
                Duplicate(outputLocal, scalarValue_, aligned);
            }
            for (uint32_t i = aligned; i < len; ++i) {
                outputLocal.SetValue(i, scalarValue_);
            }
        } else {
            for (uint32_t i = 0; i < len; ++i) {
                outputLocal.SetValue(i, scalarValue_);
            }
        }
        outQueue_.EnQue(outputLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, uint32_t len) {
        LocalTensor<StorageT> outputLocal = outQueue_.DeQue<StorageT>();
        CopyOutAuto(outputGm_, offset, outputLocal, len);
        outQueue_.FreeTensor(outputLocal);
    }

  private:
    TQue<QuePosition::VECOUT, kBufferNum> outQueue_;
    GlobalTensor<StorageT> outputGm_;
    StorageT scalarValue_ = 0;
    int64_t blockLength_ = 0;
    uint32_t tileLength_ = 1;
    uint32_t allocTileLength_ = 1;
};

template <typename T>
class KernelAssignLastDimContiguous {
  public:
    using StorageT = AssignStorageT<T>;

    __aicore__ inline void Init(TPipe *pipe, GM_ADDR other, GM_ADDR output, const AssignTilingData &tiling) {
        InitSocState();
        otherGm_.SetGlobalBuffer((__gm__ StorageT *)other);
        outputGm_.SetGlobalBuffer((__gm__ StorageT *)output);
        tileLength_ = GetDirectCopyTileBytes<StorageT>(tiling.out_last_dim) / sizeof(StorageT);
        if (tileLength_ == 0) {
            tileLength_ = 1;
        }
        if (static_cast<int64_t>(tileLength_) > tiling.out_last_dim) {
            tileLength_ = static_cast<uint32_t>(tiling.out_last_dim);
        }
        if (tileLength_ == 0) {
            tileLength_ = 1;
        }
        const uint32_t alignElems = 32 / sizeof(StorageT);
        allocTileLength_ = tileLength_ < alignElems ? alignElems : tileLength_;
        pipe->InitBuffer(bindQueue_, 1, allocTileLength_ * sizeof(StorageT));

        for (int64_t i = 0; i < kMaxRank; ++i) {
            outDims_[i] = 1;
            otherDims_[i] = 1;
            outStrides_[i] = 1;
            otherStrides_[i] = 0;
        }
        outDims_[0] = tiling.dim0; outDims_[1] = tiling.dim1; outDims_[2] = tiling.dim2; outDims_[3] = tiling.dim3;
        outDims_[4] = tiling.dim4; outDims_[5] = tiling.dim5; outDims_[6] = tiling.dim6; outDims_[7] = tiling.dim7;
        otherDims_[0] = tiling.other_dim0; otherDims_[1] = tiling.other_dim1; otherDims_[2] = tiling.other_dim2;
        otherDims_[3] = tiling.other_dim3; otherDims_[4] = tiling.other_dim4; otherDims_[5] = tiling.other_dim5;
        otherDims_[6] = tiling.other_dim6; otherDims_[7] = tiling.other_dim7;
        outStrides_[0] = tiling.out_stride0; outStrides_[1] = tiling.out_stride1; outStrides_[2] = tiling.out_stride2;
        outStrides_[3] = tiling.out_stride3; outStrides_[4] = tiling.out_stride4; outStrides_[5] = tiling.out_stride5;
        outStrides_[6] = tiling.out_stride6; outStrides_[7] = tiling.out_stride7;
        otherStrides_[0] = tiling.other_stride0; otherStrides_[1] = tiling.other_stride1; otherStrides_[2] = tiling.other_stride2;
        otherStrides_[3] = tiling.other_stride3; otherStrides_[4] = tiling.other_stride4; otherStrides_[5] = tiling.other_stride5;
        otherStrides_[6] = tiling.other_stride6; otherStrides_[7] = tiling.other_stride7;
    }

    __aicore__ inline void Process(const AssignTilingData &tiling) {
        const int64_t startOuter = static_cast<int64_t>(GetBlockIdx()) * tiling.outer_elems_per_core;
        if (startOuter >= tiling.outer_elems) {
            return;
        }
        int64_t endOuter = startOuter + tiling.outer_elems_per_core;
        if (endOuter > tiling.outer_elems) {
            endOuter = tiling.outer_elems;
        }

        int64_t coords[kMaxRank];
        InitOuterCoords(startOuter, tiling.rank, coords, tiling.out_last_dim);
        int64_t otherIndex = InitOuterOtherIndex(coords, tiling.rank);

        for (int64_t outer = startOuter; outer < endOuter; ++outer) {
            const int64_t outBase = outer * tiling.out_last_dim;
            CopySegment(otherIndex, outBase, tiling.out_last_dim);
            if (outer + 1 < endOuter) {
                AdvanceOuterState(tiling.rank, coords, otherIndex);
            }
        }
    }

  private:
    __aicore__ inline void InitOuterCoords(int64_t outerIndex, int64_t rank, int64_t (&coords)[kMaxRank], int64_t lastDim) {
        for (int64_t i = 0; i < kMaxRank; ++i) {
            coords[i] = 0;
        }
        for (int64_t dim = 0; dim + 1 < rank; ++dim) {
            const int64_t outerStride = outStrides_[dim] / lastDim;
            const int64_t coord = outerStride > 0 ? (outerIndex / outerStride) : 0;
            outerIndex -= coord * outerStride;
            coords[dim] = coord;
        }
    }

    __aicore__ inline int64_t InitOuterOtherIndex(const int64_t (&coords)[kMaxRank], int64_t rank) {
        int64_t otherIndex = 0;
        for (int64_t dim = 0; dim + 1 < rank; ++dim) {
            if (otherDims_[dim] != 1) {
                otherIndex += coords[dim] * otherStrides_[dim];
            }
        }
        return otherIndex;
    }

    __aicore__ inline void AdvanceOuterState(int64_t rank, int64_t (&coords)[kMaxRank], int64_t &otherIndex) {
        for (int64_t dim = rank - 2; dim >= 0; --dim) {
            ++coords[dim];
            if (coords[dim] < outDims_[dim]) {
                if (otherDims_[dim] != 1) {
                    otherIndex += otherStrides_[dim];
                }
                return;
            }
            coords[dim] = 0;
            if (otherDims_[dim] != 1) {
                otherIndex -= (outDims_[dim] - 1) * otherStrides_[dim];
            }
        }
    }

    __aicore__ inline void CopySegment(int64_t srcBase, int64_t dstBase, int64_t len) {
        const int64_t fullTiles = len / tileLength_;
        for (int64_t tileIdx = 0; tileIdx < fullTiles; ++tileIdx) {
            const int64_t tileOffset = tileIdx * tileLength_;
            CopyInOut(srcBase + tileOffset, dstBase + tileOffset, tileLength_);
        }
        const uint32_t tail = static_cast<uint32_t>(len - fullTiles * tileLength_);
        if (tail > 0) {
            const int64_t tailOffset = fullTiles * tileLength_;
            CopyInOut(srcBase + tailOffset, dstBase + tailOffset, tail);
        }
    }

    __aicore__ inline void CopyInOut(int64_t srcOffset, int64_t dstOffset, uint32_t len) {
        LocalTensor<StorageT> local = bindQueue_.AllocTensor<StorageT>();
        CopyInAuto(otherGm_, srcOffset, local, len);
        bindQueue_.EnQue(local);
        LocalTensor<StorageT> readyLocal = bindQueue_.DeQue<StorageT>();
        CopyOutAuto(outputGm_, dstOffset, readyLocal, len);
        bindQueue_.FreeTensor(readyLocal);
    }

  private:
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> bindQueue_;
    GlobalTensor<StorageT> otherGm_;
    GlobalTensor<StorageT> outputGm_;
    int64_t outDims_[kMaxRank];
    int64_t otherDims_[kMaxRank];
    int64_t outStrides_[kMaxRank];
    int64_t otherStrides_[kMaxRank];
    uint32_t tileLength_ = 1;
    uint32_t allocTileLength_ = 1;
};

} // namespace

extern "C" __global__ __aicore__ void assign(GM_ADDR input, GM_ADDR other, GM_ADDR output,
                                             GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    (void)input;
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    if (tiling_data.same_shape != 0) {
        if (GetBlockIdx() >= tiling_data.used_core_num) {
            return;
        }
        if constexpr (std::is_same_v<DTYPE_OTHER, bool>) {
            KernelAssignSameShape<uint8_t> op;
            op.Init(&pipe, other, output, tiling_data);
            op.Process();
        } else {
            KernelAssignSameShape<DTYPE_OTHER> op;
            op.Init(&pipe, other, output, tiling_data);
            op.Process();
        }
        return;
    }

    if (tiling_data.broadcast_mode == kBroadcastModeScalar) {
        const int64_t start = static_cast<int64_t>(GetBlockIdx()) * tiling_data.elems_per_core;
        if (start >= tiling_data.total_elems) {
            return;
        }
        const int64_t blockLength =
            (start + tiling_data.elems_per_core > tiling_data.total_elems)
                ? (tiling_data.total_elems - start)
                : tiling_data.elems_per_core;

        if constexpr (std::is_same_v<DTYPE_OTHER, bool>) {
            KernelAssignScalar<uint8_t> op;
            op.Init(&pipe, other, output, start, blockLength);
            op.Process();
        } else {
            KernelAssignScalar<DTYPE_OTHER> op;
            op.Init(&pipe, other, output, start, blockLength);
            op.Process();
        }
        return;
    }

    if (tiling_data.broadcast_mode == kBroadcastModeLastDimContiguous) {
        if constexpr (std::is_same_v<DTYPE_OTHER, bool>) {
            KernelAssignLastDimContiguous<uint8_t> op;
            op.Init(&pipe, other, output, tiling_data);
            op.Process(tiling_data);
        } else {
            KernelAssignLastDimContiguous<DTYPE_OTHER> op;
            op.Init(&pipe, other, output, tiling_data);
            op.Process(tiling_data);
        }
        return;
    }

    if constexpr (std::is_same_v<DTYPE_OTHER, bool>) {
        KernelAssignGeneric<uint8_t> op;
        op.Init(other, output);
        op.Process(tiling_data);
    } else {
        KernelAssignGeneric<DTYPE_OTHER> op;
        op.Init(other, output);
        op.Process(tiling_data);
    }
}
