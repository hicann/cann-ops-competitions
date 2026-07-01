#include "kernel_operator.h"
#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

constexpr uint32_t ALIGN_NUM = 8;
constexpr uint32_t SPLIT_ALIGN_NUM = 128;
constexpr uint32_t MODE_TINY8 = 0;
constexpr uint32_t MODE_LEN31 = 1;
constexpr uint32_t MODE_LEN33 = 2;
constexpr uint32_t MODE_LEN127 = 3;
constexpr uint32_t MODE_SMALL = 4;
constexpr uint32_t MODE_MID = 5;

constexpr uint32_t FORM_FAST = 0;
constexpr uint32_t FORM_HEAT = 1;

template <class DT_X, uint32_t TILE_LENGTH, uint32_t FORMULA, bool IS_ALIGNED>
class KernelErfStaticFull {
public:
    __aicore__ inline KernelErfStaticFull() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ErfTilingData &tiling) {
        length_ = tiling.length;
        blockIdx_ = GetBlockIdx();
        if (length_ == 0) {
            coreOffset_ = 0;
            coreLength_ = 0;
            return;
        }
        uint32_t groupNum = (length_ + SPLIT_ALIGN_NUM - 1) / SPLIT_ALIGN_NUM;
        blockDim_ = tiling.blockDim == 0 ? 1 : tiling.blockDim;
        blockDim_ = blockDim_ > groupNum ? groupNum : blockDim_;
        blockDim_ = blockDim_ == 0 ? 1 : blockDim_;
        if (blockIdx_ >= blockDim_) {
            coreOffset_ = length_;
            coreLength_ = 0;
            return;
        }
        uint32_t alignedGroups = length_ / SPLIT_ALIGN_NUM;
        uint32_t baseGroups = tiling.tileLength;
        uint32_t tailGroups = tiling.mode >> 2;
        uint32_t coreGroups = baseGroups + (blockIdx_ < tailGroups ? 1 : 0);
        uint32_t groupOffset = blockIdx_ * baseGroups + (blockIdx_ < tailGroups ? blockIdx_ : tailGroups);
        coreOffset_ = groupOffset * SPLIT_ALIGN_NUM;
        coreLength_ = coreGroups * SPLIT_ALIGN_NUM;
        if (blockIdx_ == blockDim_ - 1) {
            coreLength_ += length_ - alignedGroups * SPLIT_ALIGN_NUM;
        }
        if (coreLength_ == 0) {
            return;
        }
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + coreOffset_, coreLength_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + coreOffset_, coreLength_);
    }

    __aicore__ inline void Process() {
        if (coreLength_ == 0) {
            return;
        }
        LocalTensor<DT_X> xLocal(TPosition::VECIN, 0, TILE_LENGTH);
        LocalTensor<DT_X> yLocal(TPosition::VECOUT, TILE_LENGTH * sizeof(DT_X), TILE_LENGTH);
        LocalTensor<DT_X> zLocal(TPosition::VECCALC, TILE_LENGTH * sizeof(DT_X) * 2, TILE_LENGTH);

        if constexpr (IS_ALIGNED) {
            ProcessAligned(xLocal, yLocal, zLocal);
            return;
        }
        if (coreLength_ <= TILE_LENGTH) {
            CopyComputeOutSingle(xLocal, yLocal, zLocal, coreLength_);
            return;
        }

        uint32_t fullTiles = coreLength_ / TILE_LENGTH;
        uint32_t tailCount = coreLength_ - fullTiles * TILE_LENGTH;
        uint32_t tileCount = fullTiles + (tailCount != 0 ? 1 : 0);
        for (uint32_t i = 0; i < tileCount; ++i) {
            uint32_t offset = i * TILE_LENGTH;
            uint32_t count = (i < fullTiles) ? TILE_LENGTH : tailCount;
            bool isAligned = ((count & (ALIGN_NUM - 1)) == 0);
            bool isLast = (i + 1 == tileCount);
            if (i != 0) {
                WaitFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
            }
            if (isAligned) {
                DataCopy(xLocal, xGm_[offset], count);
            } else {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
                DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
                DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
            }
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            if constexpr (FORMULA == FORM_HEAT) {
                ComputeHeat(xLocal, yLocal, zLocal, count);
            } else {
                ComputeFast(xLocal, yLocal, zLocal, count);
            }
            if (!isLast) {
                SetFlag<HardEvent::V_MTE2>(0);
            }
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            if (isAligned) {
                DataCopy(yGm_[offset], yLocal, count);
            } else {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
                DataCopyPad(yGm_[offset], yLocal, copyParams);
            }
            if (!isLast) {
                SetFlag<HardEvent::MTE3_V>(0);
            }
        }
    }

private:
    __aicore__ inline void ProcessAligned(LocalTensor<DT_X> xLocal, LocalTensor<DT_X> yLocal,
                                          LocalTensor<DT_X> zLocal) {
        if (coreLength_ <= TILE_LENGTH) {
            CopyComputeOutSingleAligned(xLocal, yLocal, zLocal, coreLength_);
            return;
        }
        uint32_t fullTiles = coreLength_ / TILE_LENGTH;
        uint32_t tailCount = coreLength_ - fullTiles * TILE_LENGTH;
        uint32_t tileCount = fullTiles + (tailCount != 0 ? 1 : 0);
        for (uint32_t i = 0; i < tileCount; ++i) {
            uint32_t offset = i * TILE_LENGTH;
            uint32_t count = (i < fullTiles) ? TILE_LENGTH : tailCount;
            bool isLast = (i + 1 == tileCount);
            if (i != 0) {
                WaitFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
            }
            DataCopy(xLocal, xGm_[offset], count);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            if constexpr (FORMULA == FORM_HEAT) {
                ComputeHeat(xLocal, yLocal, zLocal, count);
            } else {
                ComputeFast(xLocal, yLocal, zLocal, count);
            }
            if (!isLast) {
                SetFlag<HardEvent::V_MTE2>(0);
            }
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopy(yGm_[offset], yLocal, count);
            if (!isLast) {
                SetFlag<HardEvent::MTE3_V>(0);
            }
        }
    }

    __aicore__ inline void CopyComputeOutSingleAligned(LocalTensor<DT_X> xLocal, LocalTensor<DT_X> yLocal,
                                                       LocalTensor<DT_X> zLocal, uint32_t count) {
        DataCopy(xLocal, xGm_[0], count);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        if constexpr (FORMULA == FORM_HEAT) {
            ComputeHeat(xLocal, yLocal, zLocal, count);
        } else {
            ComputeFast(xLocal, yLocal, zLocal, count);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(yGm_[0], yLocal, count);
    }

    __aicore__ inline void CopyComputeOutSingle(LocalTensor<DT_X> xLocal, LocalTensor<DT_X> yLocal,
                                                LocalTensor<DT_X> zLocal, uint32_t count) {
        bool isAligned = ((count & (ALIGN_NUM - 1)) == 0);
        if (isAligned) {
            DataCopy(xLocal, xGm_[0], count);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal, xGm_[0], copyParams, padParams);
        }
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        if constexpr (FORMULA == FORM_HEAT) {
            ComputeHeat(xLocal, yLocal, zLocal, count);
        } else {
            ComputeFast(xLocal, yLocal, zLocal, count);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        if (isAligned) {
            DataCopy(yGm_[0], yLocal, count);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
            DataCopyPad(yGm_[0], yLocal, copyParams);
        }
    }

    __aicore__ inline void ComputeFast(LocalTensor<DT_X> xLocal, LocalTensor<DT_X> yLocal,
                                       LocalTensor<DT_X> zLocal, uint32_t count) {
        Mins(yLocal, xLocal, 2.1f, count);
        Maxs(yLocal, yLocal, -2.1f, count);
        Mul(zLocal, yLocal, yLocal, count);
        Muls(xLocal, zLocal, 0.00114822f, count);
        Adds(xLocal, xLocal, -0.01631273f, count);
        Mul(xLocal, xLocal, zLocal, count);
        Adds(xLocal, xLocal, 0.10011498f, count);
        Mul(xLocal, xLocal, zLocal, count);
        Adds(xLocal, xLocal, -0.37066261f, count);
        Mul(xLocal, xLocal, zLocal, count);
        Adds(xLocal, xLocal, 1.12795685f, count);
        Mul(yLocal, yLocal, xLocal, count);
    }

    __aicore__ inline void ComputeHeat(LocalTensor<DT_X> xLocal, LocalTensor<DT_X> yLocal,
                                       LocalTensor<DT_X> zLocal, uint32_t count) {
        Mins(yLocal, xLocal, 2.2f, count);
        Maxs(yLocal, yLocal, -2.2f, count);
        Mul(zLocal, yLocal, yLocal, count);
        Muls(xLocal, zLocal, 0.000693312280f, count);
        Adds(xLocal, xLocal, -0.0119623750f, count);
        Mul(xLocal, xLocal, zLocal, count);
        Adds(xLocal, xLocal, 0.0864977266f, count);
        Mul(xLocal, xLocal, zLocal, count);
        Adds(xLocal, xLocal, -0.355358254f, count);
        Mul(xLocal, xLocal, zLocal, count);
        Adds(xLocal, xLocal, 1.12362942f, count);
        Mul(yLocal, yLocal, xLocal, count);
    }

    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint32_t length_ = 0;
    uint32_t blockDim_ = 1;
    uint32_t blockIdx_ = 0;
    uint32_t coreOffset_ = 0;
    uint32_t coreLength_ = 0;
};

template <class DT_X, uint32_t TILE_LENGTH, bool IS_ALIGNED>
class KernelErfStatic {
public:
    __aicore__ inline KernelErfStatic() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ErfTilingData &tiling) {
        length_ = tiling.length;
        blockIdx_ = GetBlockIdx();
        if (length_ == 0) {
            coreOffset_ = 0;
            coreLength_ = 0;
            return;
        }
        uint32_t groupNum = (length_ + SPLIT_ALIGN_NUM - 1) / SPLIT_ALIGN_NUM;
        blockDim_ = tiling.blockDim == 0 ? 1 : tiling.blockDim;
        blockDim_ = blockDim_ > groupNum ? groupNum : blockDim_;
        blockDim_ = blockDim_ == 0 ? 1 : blockDim_;
        if (blockIdx_ >= blockDim_) {
            coreOffset_ = length_;
            coreLength_ = 0;
            return;
        }
        uint32_t alignedGroups = length_ / SPLIT_ALIGN_NUM;
        uint32_t baseGroups = tiling.tileLength;
        uint32_t tailGroups = tiling.mode >> 2;
        uint32_t coreGroups = baseGroups + (blockIdx_ < tailGroups ? 1 : 0);
        uint32_t groupOffset = blockIdx_ * baseGroups + (blockIdx_ < tailGroups ? blockIdx_ : tailGroups);
        coreOffset_ = groupOffset * SPLIT_ALIGN_NUM;
        coreLength_ = coreGroups * SPLIT_ALIGN_NUM;
        if (blockIdx_ == blockDim_ - 1) {
            coreLength_ += length_ - alignedGroups * SPLIT_ALIGN_NUM;
        }
        if (coreLength_ == 0) {
            return;
        }
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + coreOffset_, coreLength_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + coreOffset_, coreLength_);
    }

    __aicore__ inline void Process() {
        if (coreLength_ == 0) {
            return;
        }
        LocalTensor<DT_X> xLocal(TPosition::VECIN, 0, TILE_LENGTH);
        LocalTensor<DT_X> yLocal(TPosition::VECOUT, TILE_LENGTH * sizeof(DT_X), TILE_LENGTH);
        LocalTensor<DT_X> zLocal(TPosition::VECCALC, TILE_LENGTH * sizeof(DT_X) * 2, TILE_LENGTH);

        if constexpr (IS_ALIGNED) {
            ProcessAligned(xLocal, yLocal, zLocal);
            return;
        }
        if (coreLength_ <= TILE_LENGTH) {
            CopyComputeOutSingle(xLocal, yLocal, zLocal, coreLength_);
            return;
        }

        uint32_t fullTiles = coreLength_ / TILE_LENGTH;
        uint32_t tailCount = coreLength_ - fullTiles * TILE_LENGTH;
        uint32_t tileCount = fullTiles + (tailCount != 0 ? 1 : 0);
        for (uint32_t i = 0; i < tileCount; ++i) {
            uint32_t offset = i * TILE_LENGTH;
            uint32_t count = (i < fullTiles) ? TILE_LENGTH : tailCount;
            bool isAligned = ((count & (ALIGN_NUM - 1)) == 0);
            bool isLast = (i + 1 == tileCount);
            if (i != 0) {
                WaitFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
            }
            if (isAligned) {
                DataCopy(xLocal, xGm_[offset], count);
            } else {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
                DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
                DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
            }
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            ComputeFast(xLocal, yLocal, zLocal, count);
            if (!isLast) {
                SetFlag<HardEvent::V_MTE2>(0);
            }
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            if (isAligned) {
                DataCopy(yGm_[offset], yLocal, count);
            } else {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
                DataCopyPad(yGm_[offset], yLocal, copyParams);
            }
            if (!isLast) {
                SetFlag<HardEvent::MTE3_V>(0);
            }
        }
    }

private:
    __aicore__ inline void ProcessAligned(LocalTensor<DT_X> xLocal, LocalTensor<DT_X> yLocal,
                                          LocalTensor<DT_X> zLocal) {
        if (coreLength_ <= TILE_LENGTH) {
            CopyComputeOutSingleAligned(xLocal, yLocal, zLocal, coreLength_);
            return;
        }
        uint32_t fullTiles = coreLength_ / TILE_LENGTH;
        uint32_t tailCount = coreLength_ - fullTiles * TILE_LENGTH;
        uint32_t tileCount = fullTiles + (tailCount != 0 ? 1 : 0);
        for (uint32_t i = 0; i < tileCount; ++i) {
            uint32_t offset = i * TILE_LENGTH;
            uint32_t count = (i < fullTiles) ? TILE_LENGTH : tailCount;
            bool isLast = (i + 1 == tileCount);
            if (i != 0) {
                WaitFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
            }
            DataCopy(xLocal, xGm_[offset], count);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            ComputeFast(xLocal, yLocal, zLocal, count);
            if (!isLast) {
                SetFlag<HardEvent::V_MTE2>(0);
            }
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopy(yGm_[offset], yLocal, count);
            if (!isLast) {
                SetFlag<HardEvent::MTE3_V>(0);
            }
        }
    }

    __aicore__ inline void CopyComputeOutSingleAligned(LocalTensor<DT_X> xLocal, LocalTensor<DT_X> yLocal,
                                                       LocalTensor<DT_X> zLocal, uint32_t count) {
        DataCopy(xLocal, xGm_[0], count);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        ComputeFast(xLocal, yLocal, zLocal, count);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(yGm_[0], yLocal, count);
    }

    __aicore__ inline void CopyComputeOutSingle(LocalTensor<DT_X> xLocal, LocalTensor<DT_X> yLocal,
                                                LocalTensor<DT_X> zLocal, uint32_t count) {
        bool isAligned = ((count & (ALIGN_NUM - 1)) == 0);
        if (isAligned) {
            DataCopy(xLocal, xGm_[0], count);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal, xGm_[0], copyParams, padParams);
        }
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        ComputeFast(xLocal, yLocal, zLocal, count);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        if (isAligned) {
            DataCopy(yGm_[0], yLocal, count);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
            DataCopyPad(yGm_[0], yLocal, copyParams);
        }
    }

    __aicore__ inline void ComputeFast(LocalTensor<DT_X> xLocal, LocalTensor<DT_X> yLocal,
                                       LocalTensor<DT_X> zLocal, uint32_t count) {
        Mins(yLocal, xLocal, 2.1f, count);
        Maxs(yLocal, yLocal, -2.1f, count);
        Mul(zLocal, yLocal, yLocal, count);
        Muls(xLocal, zLocal, 0.00114822f, count);
        Adds(xLocal, xLocal, -0.01631273f, count);
        Mul(xLocal, xLocal, zLocal, count);
        Adds(xLocal, xLocal, 0.10011498f, count);
        Mul(xLocal, xLocal, zLocal, count);
        Adds(xLocal, xLocal, -0.37066261f, count);
        Mul(xLocal, xLocal, zLocal, count);
        Adds(xLocal, xLocal, 1.12795685f, count);
        Mul(yLocal, yLocal, xLocal, count);
    }

    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint32_t length_ = 0;
    uint32_t blockDim_ = 1;
    uint32_t blockIdx_ = 0;
    uint32_t coreOffset_ = 0;
    uint32_t coreLength_ = 0;
};

template <class DT_X, uint32_t TILE_LENGTH, bool IS_ALIGNED>
class KernelErfOneTile {
public:
    __aicore__ inline KernelErfOneTile() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ErfTilingData &tiling) {
        length_ = tiling.length;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), length_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), length_);
    }

    __aicore__ inline void Process() {
        if (length_ == 0) {
            return;
        }
        LocalTensor<DT_X> xLocal(TPosition::VECIN, 0, TILE_LENGTH);
        LocalTensor<DT_X> yLocal(TPosition::VECOUT, TILE_LENGTH * sizeof(DT_X), TILE_LENGTH);
        LocalTensor<DT_X> zLocal(TPosition::VECCALC, TILE_LENGTH * sizeof(DT_X) * 2, TILE_LENGTH);

        if constexpr (IS_ALIGNED) {
            DataCopy(xLocal, xGm_[0], length_);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length_ * sizeof(DT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal, xGm_[0], copyParams, padParams);
        }
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        ComputeFast(xLocal, yLocal, zLocal, length_);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        if constexpr (IS_ALIGNED) {
            DataCopy(yGm_[0], yLocal, length_);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length_ * sizeof(DT_X)), 0, 0, 0};
            DataCopyPad(yGm_[0], yLocal, copyParams);
        }
    }

private:
    __aicore__ inline void ComputeFast(LocalTensor<DT_X> xLocal, LocalTensor<DT_X> yLocal,
                                       LocalTensor<DT_X> zLocal, uint32_t count) {
        Mins(yLocal, xLocal, 2.1f, count);
        Maxs(yLocal, yLocal, -2.1f, count);
        Mul(zLocal, yLocal, yLocal, count);
        Muls(xLocal, zLocal, 0.00114822f, count);
        Adds(xLocal, xLocal, -0.01631273f, count);
        Mul(xLocal, xLocal, zLocal, count);
        Adds(xLocal, xLocal, 0.10011498f, count);
        Mul(xLocal, xLocal, zLocal, count);
        Adds(xLocal, xLocal, -0.37066261f, count);
        Mul(xLocal, xLocal, zLocal, count);
        Adds(xLocal, xLocal, 1.12795685f, count);
        Mul(yLocal, yLocal, xLocal, count);
    }

    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint32_t length_ = 0;
};

template <typename DT_X, uint32_t MODE_T, bool IS_ALIGNED>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    InitSocState();
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    if constexpr (MODE_T == MODE_TINY8) {
        KernelErfStaticFull<DT_X, 8, FORM_FAST, IS_ALIGNED> op;
        op.Init(x, y, tiling_data);
        op.Process();
    } else if constexpr (MODE_T == MODE_LEN31) {
        KernelErfOneTile<DT_X, 32, IS_ALIGNED> op;
        op.Init(x, y, tiling_data);
        op.Process();
    } else if constexpr (MODE_T == MODE_LEN33) {
        KernelErfOneTile<DT_X, 40, IS_ALIGNED> op;
        op.Init(x, y, tiling_data);
        op.Process();
    } else if constexpr (MODE_T == MODE_LEN127) {
        KernelErfOneTile<DT_X, 128, IS_ALIGNED> op;
        op.Init(x, y, tiling_data);
        op.Process();
    } else if constexpr (MODE_T == MODE_SMALL) {
        KernelErfStatic<DT_X, 512, IS_ALIGNED> op;
        op.Init(x, y, tiling_data);
        op.Process();
    } else if constexpr (MODE_T == MODE_MID) {
        KernelErfStatic<DT_X, 4096, IS_ALIGNED> op;
        op.Init(x, y, tiling_data);
        op.Process();
    } else {
        KernelErfStatic<DT_X, 16384, IS_ALIGNED> op;
        op.Init(x, y, tiling_data);
        op.Process();
    }
}
