#include "kernel_operator.h"

using namespace AscendC;

// 统一宏定义
constexpr int32_t BUFFER_NUM = 2;
constexpr float ONE_F = 1.0f;
constexpr float ZERO_F = 0.0f;

// 模板类：T为输入数据类型（half/float/bfloat16_t），输出固定为uint8_t (bool)
template <typename T>
class KernelTensorEqual {
public:
    __aicore__ inline KernelTensorEqual() {}

    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                GM_ADDR workspace, GM_ADDR tiling) {
        GET_TILING_DATA(tilingData, tiling);
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->pipe = pipeIn;
        uint32_t coreIdx = GetBlockIdx();
        this->typeLength = tilingData.typeLength;

        // 分核逻辑
        bool isBigCore = (coreIdx < tilingData.tailBlockNum);
        this->coreDataNum = isBigCore ? tilingData.bigCoreDataNum : tilingData.smallCoreDataNum;
        this->tileNum = isBigCore ? tilingData.finalBigTileNum : tilingData.finalSmallTileNum;
        this->tailDataNum = isBigCore ? tilingData.bigTailDataNum : tilingData.smallTailDataNum;
        this->processDataNum_computes = isBigCore ? tilingData.bigprocessDataNum_computes : tilingData.smallprocessDataNum_computes;
        this->tailprocessDataNum_computes = isBigCore ? tilingData.tailbigprocessDataNum_computes : tilingData.tailsmallprocessDataNum_computes;
        this->tileDataNum = tilingData.tileDataNum;

        // 全局偏移计算
        uint32_t globalOffset = isBigCore ? (coreIdx * tilingData.bigCoreDataNum) :
                               (tilingData.tailBlockNum * tilingData.bigCoreDataNum + (coreIdx - tilingData.tailBlockNum) * tilingData.smallCoreDataNum);

        auto selfPtr = reinterpret_cast<__gm__ T*>(x1);
        auto otherPtr = reinterpret_cast<__gm__ T*>(x2);
        auto outPtr = reinterpret_cast<__gm__ uint8_t*>(y);
        this->selfGm.SetGlobalBuffer(selfPtr + globalOffset, this->coreDataNum);
        this->otherGm.SetGlobalBuffer(otherPtr + globalOffset, this->coreDataNum);
        this->outGm.SetGlobalBuffer(outPtr + globalOffset, this->coreDataNum);

        // UB缓冲区初始化
        uint32_t bufferSizeCompute = this->processDataNum_computes * sizeof(float);
        uint32_t bufferSizeCopy = this->tileDataNum * this->typeLength;
        uint32_t bufferSizeOut = this->tileDataNum * sizeof(uint8_t);
        uint32_t maskBufBytes = ((this->processDataNum_computes + 7) / 8 + 31) / 32 * 32;

        pipe->InitBuffer(inQueueX, BUFFER_NUM, bufferSizeCopy);
        pipe->InitBuffer(inQueueY, BUFFER_NUM, bufferSizeCopy);
        pipe->InitBuffer(outQueueZ, BUFFER_NUM, bufferSizeOut);

        pipe->InitBuffer(mainComputeBuf, bufferSizeCompute);
        pipe->InitBuffer(maskBuf, maskBufBytes);
        pipe->InitBuffer(onesBuf, bufferSizeCompute);
        pipe->InitBuffer(zerosBuf, bufferSizeCompute);
        pipe->InitBuffer(resultFloatBuf, bufferSizeCompute);
        pipe->InitBuffer(tempResultBuf, bufferSizeCompute);
        if constexpr (!std::is_same_v<T, int32_t> && std::is_integral_v<T>) {
            pipe->InitBuffer(narrowCastBuf, this->processDataNum_computes * sizeof(half));
        }
    }

    __aicore__ inline void UpdateTileParams(int32_t progress) {
        this->processDataNum = (progress == static_cast<int32_t>(this->tileNum) - 1) ?
            this->tailDataNum : this->tileDataNum;
        this->currentProcessDataNumComputes = (progress == static_cast<int32_t>(this->tileNum) - 1) ?
            this->tailprocessDataNum_computes : this->processDataNum_computes;
    }

    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        if (loopCount == 0) return;

        // 预先填充常量缓冲区（仅一次）
        LocalTensor<float> ones = onesBuf.Get<float>();
        LocalTensor<float> zeros = zerosBuf.Get<float>();
        Duplicate<float>(ones, ONE_F, this->processDataNum_computes);
        Duplicate<float>(zeros, ZERO_F, this->processDataNum_computes);

        // 预取前两个Tile
        int32_t ping = 0;
        UpdateTileParams(0);
        CopyIn(ping, 0);
        if (loopCount > 1) {
            UpdateTileParams(1);
            CopyIn(1 - ping, 1);
        }
        if constexpr (std::is_same_v<T, float>) {
            for (int32_t i = 0; i < loopCount; ++i) {
                int32_t pong = 1 - ping;

                UpdateTileParams(i);
                Compute_float(i, ping);
                CopyOut(i, ping);

                if (i + 2 < loopCount) {
                    UpdateTileParams(i + 2);
                    CopyIn(pong, i + 2);
                }

                ping = pong;
            }
        } else if constexpr (std::is_integral_v<T>) {
            for (int32_t i = 0; i < loopCount; ++i) {
                int32_t pong = 1 - ping;

                UpdateTileParams(i);
                Compute_int(i, ping);
                CopyOut(i, ping);

                if (i + 2 < loopCount) {
                    UpdateTileParams(i + 2);
                    CopyIn(pong, i + 2);
                }

                ping = pong;
            }
        } else {
            for (int32_t i = 0; i < loopCount; ++i) {
                int32_t pong = 1 - ping;

                UpdateTileParams(i);
                Compute_half(i, ping);
                CopyOut(i, ping);

                if (i + 2 < loopCount) {
                    UpdateTileParams(i + 2);
                    CopyIn(pong, i + 2);
                }

                ping = pong;
            }
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t bufIndex, int32_t progress) {
        uint32_t offset = progress * this->tileDataNum;
        LocalTensor<T> selfLocal = inQueueX.AllocTensor<T>();
        LocalTensor<T> otherLocal = inQueueY.AllocTensor<T>();
        DataCopy(selfLocal, this->selfGm[offset], this->processDataNum);
        DataCopy(otherLocal, this->otherGm[offset], this->processDataNum);
        inQueueX.EnQue(selfLocal);
        inQueueY.EnQue(otherLocal);
    }

    __aicore__ inline void Compute_float(int32_t progress, int32_t bufIndex)
    {
        LocalTensor<float> inputData = mainComputeBuf.Get<float>();
        LocalTensor<uint8_t> conditionMask = maskBuf.Get<uint8_t>();
        LocalTensor<float> ones = onesBuf.Get<float>();
        LocalTensor<float> zeros = zerosBuf.Get<float>();
        LocalTensor<float> resultFloat = resultFloatBuf.Get<float>();

        LocalTensor<T> selfLocal = inQueueX.DeQue<T>();
        LocalTensor<T> otherLocal = inQueueY.DeQue<T>();
        LocalTensor<uint8_t> outLocal = outQueueZ.AllocTensor<uint8_t>();

        // 浮点直接 Compare（避免 inf-inf=NaN 导致误判）
        Compare(conditionMask, selfLocal, otherLocal, CMPMODE::EQ, this->processDataNum);
        // step3: 根据mask选择1.0f或0.0f
        Select(resultFloat, conditionMask, ones, zeros, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        // step4: 将float(1.0/0.0)转为uint8_t(1/0)即bool
        // 310B不支持float直接转uint8_t，需两步：float→half→uint8_t
        {
            LocalTensor<half> halfTemp = tempResultBuf.Get<half>();
            Cast(halfTemp, resultFloat, RoundMode::CAST_NONE, this->processDataNum);
            Cast(outLocal, halfTemp, RoundMode::CAST_RINT, this->processDataNum);
        }

        inQueueX.FreeTensor(selfLocal);
        inQueueY.FreeTensor(otherLocal);
        outQueueZ.EnQue(outLocal);
    }

    __aicore__ inline void Compute_int(int32_t progress, int32_t bufIndex)
    {
        LocalTensor<uint8_t> conditionMask = maskBuf.Get<uint8_t>();
        LocalTensor<float> ones = onesBuf.Get<float>();
        LocalTensor<float> zeros = zerosBuf.Get<float>();
        LocalTensor<float> resultFloat = resultFloatBuf.Get<float>();

        LocalTensor<T> selfLocal = inQueueX.DeQue<T>();
        LocalTensor<T> otherLocal = inQueueY.DeQue<T>();
        LocalTensor<uint8_t> outLocal = outQueueZ.AllocTensor<uint8_t>();

        // int32: Sub + CompareScalar(0)；窄整型: T→half→float
        if constexpr (std::is_same_v<T, int32_t>) {
            LocalTensor<int32_t> diffLocal = mainComputeBuf.Get<int32_t>();
            Sub(diffLocal, selfLocal, otherLocal, this->processDataNum);
            CompareScalar(conditionMask, diffLocal, static_cast<int32_t>(0), CMPMODE::EQ, this->processDataNum);
        } else {
            LocalTensor<half> halfTmp = narrowCastBuf.Get<half>();
            LocalTensor<float> inputData = mainComputeBuf.Get<float>();
            LocalTensor<float> tempBuf = tempResultBuf.Get<float>();
            Cast(halfTmp, selfLocal, RoundMode::CAST_NONE, this->processDataNum);
            Cast(inputData, halfTmp, RoundMode::CAST_NONE, this->processDataNum);
            Cast(halfTmp, otherLocal, RoundMode::CAST_NONE, this->processDataNum);
            Cast(tempBuf, halfTmp, RoundMode::CAST_NONE, this->processDataNum);
            Sub(inputData, inputData, tempBuf, this->processDataNum);
            CompareScalar(conditionMask, inputData, ZERO_F, CMPMODE::EQ, this->processDataNum);
        }
        Select(resultFloat, conditionMask, ones, zeros, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        {
            LocalTensor<half> halfTemp = tempResultBuf.Get<half>();
            Cast(halfTemp, resultFloat, RoundMode::CAST_NONE, this->processDataNum);
            Cast(outLocal, halfTemp, RoundMode::CAST_RINT, this->processDataNum);
        }

        inQueueX.FreeTensor(selfLocal);
        inQueueY.FreeTensor(otherLocal);
        outQueueZ.EnQue(outLocal);
    }

    __aicore__ inline void Compute_half(int32_t progress, int32_t bufIndex)
    {
        LocalTensor<float> inputData = mainComputeBuf.Get<float>();
        LocalTensor<float> tempBuf = tempResultBuf.Get<float>();
        LocalTensor<uint8_t> conditionMask = maskBuf.Get<uint8_t>();
        LocalTensor<float> ones = onesBuf.Get<float>();
        LocalTensor<float> zeros = zerosBuf.Get<float>();
        LocalTensor<float> resultFloat = resultFloatBuf.Get<float>();

        LocalTensor<T> selfLocal = inQueueX.DeQue<T>();
        LocalTensor<T> otherLocal = inQueueY.DeQue<T>();
        LocalTensor<uint8_t> outLocal = outQueueZ.AllocTensor<uint8_t>();

        // half/bfloat16 先转换到 float 域再比较
        Cast(inputData, selfLocal, RoundMode::CAST_NONE, this->processDataNum);
        Cast(tempBuf, otherLocal, RoundMode::CAST_NONE, this->processDataNum);
        Sub(inputData, inputData, tempBuf, this->processDataNum);
        CompareScalar(conditionMask, inputData, ZERO_F, CMPMODE::EQ, this->processDataNum);
        Select(resultFloat, conditionMask, ones, zeros, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        {
            LocalTensor<half> halfTemp = tempResultBuf.Get<half>();
            Cast(halfTemp, resultFloat, RoundMode::CAST_NONE, this->processDataNum);
            Cast(outLocal, halfTemp, RoundMode::CAST_RINT, this->processDataNum);
        }

        inQueueX.FreeTensor(selfLocal);
        inQueueY.FreeTensor(otherLocal);
        outQueueZ.EnQue(outLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, int32_t bufIndex) {
        uint32_t offset = progress * this->tileDataNum;
        LocalTensor<uint8_t> yLocal = outQueueZ.DeQue<uint8_t>();
        DataCopy(this->outGm[offset], yLocal, this->processDataNum);
        outQueueZ.FreeTensor(yLocal);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    TBuf<QuePosition::VECCALC> mainComputeBuf;
    TBuf<QuePosition::VECCALC> maskBuf;
    TBuf<QuePosition::VECCALC> onesBuf;
    TBuf<QuePosition::VECCALC> zerosBuf;
    TBuf<QuePosition::VECCALC> resultFloatBuf;
    TBuf<QuePosition::VECCALC> tempResultBuf;
    TBuf<QuePosition::VECCALC> narrowCastBuf;

    GlobalTensor<T> selfGm;
    GlobalTensor<T> otherGm;
    GlobalTensor<uint8_t> outGm;

    uint32_t typeLength;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum; // 实际有效计算元素数
    uint32_t processDataNum_computes; // 对齐后的缓冲区元素数
    uint32_t tailprocessDataNum_computes;
    uint32_t currentProcessDataNumComputes;
};

// 通用 N 维广播实现：按 broadcast strides 逐元素从 GM 取数到 UB 后做 element-wise 比较。
template <typename T>
class KernelTensorEqualBroadcast {
public:
    __aicore__ inline KernelTensorEqualBroadcast() {}

    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                GM_ADDR workspace, GM_ADDR tiling) {
        GET_TILING_DATA(tilingData, tiling);
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->pipe = pipeIn;
        uint32_t coreIdx = GetBlockIdx();
        this->typeLength = tilingData.typeLength;
        this->rank = tilingData.rank;
        for (uint32_t i = 0; i < 8; ++i) {
            this->outDims[i] = tilingData.outDims[i];
            this->selfStrides[i] = tilingData.selfStrides[i];
            this->otherStrides[i] = tilingData.otherStrides[i];
        }

        bool isBigCore = (coreIdx < tilingData.tailBlockNum);
        this->coreDataNum = isBigCore ? tilingData.bigCoreDataNum : tilingData.smallCoreDataNum;
        this->tileNum = isBigCore ? tilingData.finalBigTileNum : tilingData.finalSmallTileNum;
        this->tailDataNum = isBigCore ? tilingData.bigTailDataNum : tilingData.smallTailDataNum;
        this->processDataNum_computes = isBigCore ? tilingData.bigprocessDataNum_computes : tilingData.smallprocessDataNum_computes;
        this->tailprocessDataNum_computes = isBigCore ? tilingData.tailbigprocessDataNum_computes : tilingData.tailsmallprocessDataNum_computes;
        this->tileDataNum = tilingData.tileDataNum;

        uint32_t outOffset = isBigCore ? (coreIdx * tilingData.bigCoreDataNum) :
                             (tilingData.tailBlockNum * tilingData.bigCoreDataNum +
                              (coreIdx - tilingData.tailBlockNum) * tilingData.smallCoreDataNum);
        this->coreBaseIdx = outOffset;

        uint32_t selfNumel = 1;
        uint32_t otherNumel = 1;
        for (uint32_t i = 0; i < this->rank; ++i) {
            uint32_t selfDim = (this->selfStrides[i] == 0) ? 1 : this->outDims[i];
            uint32_t otherDim = (this->otherStrides[i] == 0) ? 1 : this->outDims[i];
            selfNumel *= selfDim;
            otherNumel *= otherDim;
        }
        if (selfNumel == 0) selfNumel = 1;
        if (otherNumel == 0) otherNumel = 1;
        this->selfNumel = selfNumel;
        this->otherNumel = otherNumel;

        auto selfPtr = reinterpret_cast<__gm__ T*>(x1);
        auto otherPtr = reinterpret_cast<__gm__ T*>(x2);
        auto outPtr = reinterpret_cast<__gm__ uint8_t*>(y);
        this->selfGm.SetGlobalBuffer(selfPtr, selfNumel);
        this->otherGm.SetGlobalBuffer(otherPtr, otherNumel);
        this->outGm.SetGlobalBuffer(outPtr + outOffset, this->coreDataNum);

        uint32_t bufferSizeCompute = this->processDataNum_computes * sizeof(float);
        uint32_t bufferSizeCopy = this->tileDataNum * this->typeLength;
        uint32_t bufferSizeOut = this->tileDataNum * sizeof(uint8_t);
        uint32_t maskBufBytes = ((this->processDataNum_computes + 7) / 8 + 31) / 32 * 32;

        pipe->InitBuffer(inQueueX, 1, bufferSizeCopy);
        pipe->InitBuffer(inQueueY, 1, bufferSizeCopy);
        pipe->InitBuffer(outQueueZ, 1, bufferSizeOut);
        pipe->InitBuffer(mainComputeBuf, bufferSizeCompute);
        pipe->InitBuffer(maskBuf, maskBufBytes);
        pipe->InitBuffer(onesBuf, bufferSizeCompute);
        pipe->InitBuffer(zerosBuf, bufferSizeCompute);
        pipe->InitBuffer(resultFloatBuf, bufferSizeCompute);
        pipe->InitBuffer(tempResultBuf, bufferSizeCompute);
        if constexpr (!std::is_same_v<T, int32_t> && std::is_integral_v<T>) {
            pipe->InitBuffer(narrowCastBuf, this->processDataNum_computes * sizeof(half));
        }
    }

    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        if (loopCount == 0) return;

        LocalTensor<float> ones = onesBuf.Get<float>();
        LocalTensor<float> zeros = zerosBuf.Get<float>();
        Duplicate<float>(ones, ONE_F, this->processDataNum_computes);
        Duplicate<float>(zeros, ZERO_F, this->processDataNum_computes);

        for (int32_t i = 0; i < loopCount; ++i) {
            this->processDataNum = (i == loopCount - 1) ? this->tailDataNum : this->tileDataNum;
            this->currentProcessDataNumComputes = (i == loopCount - 1) ? this->tailprocessDataNum_computes : this->processDataNum_computes;

            CopyInBroadcast(i);
            if constexpr (std::is_same_v<T, float>) {
                Compute_float();
            } else if constexpr (std::is_integral_v<T>) {
                Compute_int();
            } else {
                Compute_half();
            }
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyInBroadcast(int32_t progress) {
        uint32_t baseIdx = this->coreBaseIdx + static_cast<uint32_t>(progress) * this->tileDataNum;
        LocalTensor<T> selfLocal = inQueueX.AllocTensor<T>();
        LocalTensor<T> otherLocal = inQueueY.AllocTensor<T>();
        for (uint32_t i = 0; i < this->processDataNum; ++i) {
            uint32_t idx = baseIdx + i;
            uint32_t selfOff = 0;
            uint32_t otherOff = 0;
            uint32_t tmp = idx;
            for (int32_t d = static_cast<int32_t>(this->rank) - 1; d >= 0; --d) {
                uint32_t coord = tmp % this->outDims[d];
                tmp /= this->outDims[d];
                selfOff += coord * this->selfStrides[d];
                otherOff += coord * this->otherStrides[d];
            }
            selfLocal.SetValue(i, this->selfGm.GetValue(selfOff < this->selfNumel ? selfOff : 0));
            otherLocal.SetValue(i, this->otherGm.GetValue(otherOff < this->otherNumel ? otherOff : 0));
        }
        inQueueX.EnQue(selfLocal);
        inQueueY.EnQue(otherLocal);
    }

    __aicore__ inline void Compute_float() {
        LocalTensor<float> inputData = mainComputeBuf.Get<float>();
        LocalTensor<uint8_t> conditionMask = maskBuf.Get<uint8_t>();
        LocalTensor<float> ones = onesBuf.Get<float>();
        LocalTensor<float> zeros = zerosBuf.Get<float>();
        LocalTensor<float> resultFloat = resultFloatBuf.Get<float>();

        LocalTensor<T> selfLocal = inQueueX.DeQue<T>();
        LocalTensor<T> otherLocal = inQueueY.DeQue<T>();
        LocalTensor<uint8_t> outLocal = outQueueZ.AllocTensor<uint8_t>();

        Compare(conditionMask, selfLocal, otherLocal, CMPMODE::EQ, this->processDataNum);
        Select(resultFloat, conditionMask, ones, zeros, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        {
            LocalTensor<half> halfTemp = tempResultBuf.Get<half>();
            Cast(halfTemp, resultFloat, RoundMode::CAST_NONE, this->processDataNum);
            Cast(outLocal, halfTemp, RoundMode::CAST_RINT, this->processDataNum);
        }

        inQueueX.FreeTensor(selfLocal);
        inQueueY.FreeTensor(otherLocal);
        outQueueZ.EnQue(outLocal);
    }

    __aicore__ inline void Compute_int() {
        LocalTensor<uint8_t> conditionMask = maskBuf.Get<uint8_t>();
        LocalTensor<float> ones = onesBuf.Get<float>();
        LocalTensor<float> zeros = zerosBuf.Get<float>();
        LocalTensor<float> resultFloat = resultFloatBuf.Get<float>();

        LocalTensor<T> selfLocal = inQueueX.DeQue<T>();
        LocalTensor<T> otherLocal = inQueueY.DeQue<T>();
        LocalTensor<uint8_t> outLocal = outQueueZ.AllocTensor<uint8_t>();

        if constexpr (std::is_same_v<T, int32_t>) {
            LocalTensor<int32_t> diffLocal = mainComputeBuf.Get<int32_t>();
            Sub(diffLocal, selfLocal, otherLocal, this->processDataNum);
            CompareScalar(conditionMask, diffLocal, static_cast<int32_t>(0), CMPMODE::EQ, this->processDataNum);
        } else {
            LocalTensor<half> halfTmp = narrowCastBuf.Get<half>();
            LocalTensor<float> inputData = mainComputeBuf.Get<float>();
            LocalTensor<float> tempBuf = tempResultBuf.Get<float>();
            Cast(halfTmp, selfLocal, RoundMode::CAST_NONE, this->processDataNum);
            Cast(inputData, halfTmp, RoundMode::CAST_NONE, this->processDataNum);
            Cast(halfTmp, otherLocal, RoundMode::CAST_NONE, this->processDataNum);
            Cast(tempBuf, halfTmp, RoundMode::CAST_NONE, this->processDataNum);
            Sub(inputData, inputData, tempBuf, this->processDataNum);
            CompareScalar(conditionMask, inputData, ZERO_F, CMPMODE::EQ, this->processDataNum);
        }
        Select(resultFloat, conditionMask, ones, zeros, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        {
            LocalTensor<half> halfTemp = tempResultBuf.Get<half>();
            Cast(halfTemp, resultFloat, RoundMode::CAST_NONE, this->processDataNum);
            Cast(outLocal, halfTemp, RoundMode::CAST_RINT, this->processDataNum);
        }

        inQueueX.FreeTensor(selfLocal);
        inQueueY.FreeTensor(otherLocal);
        outQueueZ.EnQue(outLocal);
    }

    __aicore__ inline void Compute_half() {
        LocalTensor<float> inputData = mainComputeBuf.Get<float>();
        LocalTensor<float> tempBuf = tempResultBuf.Get<float>();
        LocalTensor<uint8_t> conditionMask = maskBuf.Get<uint8_t>();
        LocalTensor<float> ones = onesBuf.Get<float>();
        LocalTensor<float> zeros = zerosBuf.Get<float>();
        LocalTensor<float> resultFloat = resultFloatBuf.Get<float>();

        LocalTensor<T> selfLocal = inQueueX.DeQue<T>();
        LocalTensor<T> otherLocal = inQueueY.DeQue<T>();
        LocalTensor<uint8_t> outLocal = outQueueZ.AllocTensor<uint8_t>();

        Cast(inputData, selfLocal, RoundMode::CAST_NONE, this->processDataNum);
        Cast(tempBuf, otherLocal, RoundMode::CAST_NONE, this->processDataNum);
        Sub(inputData, inputData, tempBuf, this->processDataNum);
        CompareScalar(conditionMask, inputData, ZERO_F, CMPMODE::EQ, this->processDataNum);
        Select(resultFloat, conditionMask, ones, zeros, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        {
            LocalTensor<half> halfTemp = tempResultBuf.Get<half>();
            Cast(halfTemp, resultFloat, RoundMode::CAST_NONE, this->processDataNum);
            Cast(outLocal, halfTemp, RoundMode::CAST_RINT, this->processDataNum);
        }

        inQueueX.FreeTensor(selfLocal);
        inQueueY.FreeTensor(otherLocal);
        outQueueZ.EnQue(outLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        uint32_t offset = static_cast<uint32_t>(progress) * this->tileDataNum;
        LocalTensor<uint8_t> yLocal = outQueueZ.DeQue<uint8_t>();
        DataCopy(this->outGm[offset], yLocal, this->processDataNum);
        outQueueZ.FreeTensor(yLocal);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueY;
    TQue<QuePosition::VECOUT, 1> outQueueZ;

    TBuf<QuePosition::VECCALC> mainComputeBuf;
    TBuf<QuePosition::VECCALC> maskBuf;
    TBuf<QuePosition::VECCALC> onesBuf;
    TBuf<QuePosition::VECCALC> zerosBuf;
    TBuf<QuePosition::VECCALC> resultFloatBuf;
    TBuf<QuePosition::VECCALC> tempResultBuf;
    TBuf<QuePosition::VECCALC> narrowCastBuf;

    GlobalTensor<T> selfGm;
    GlobalTensor<T> otherGm;
    GlobalTensor<uint8_t> outGm;

    uint32_t typeLength;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    uint32_t processDataNum_computes;
    uint32_t tailprocessDataNum_computes;
    uint32_t currentProcessDataNumComputes;
    uint32_t coreBaseIdx;
    uint32_t rank;
    uint32_t outDims[8];
    uint32_t selfStrides[8];
    uint32_t otherStrides[8];
    uint32_t selfNumel;
    uint32_t otherNumel;
};


extern "C" __global__ __aicore__ void tensor_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                                GM_ADDR workspace, GM_ADDR tiling) {
    TPipe pipe;
    if (TILING_KEY_IS(0)) { // DT_FLOAT16 element-wise
        KernelTensorEqual<half> op;
        op.Init(&pipe, x1, x2, y, workspace, tiling);
        op.Process();
    } else if (TILING_KEY_IS(1)) { // DT_FLOAT element-wise
        KernelTensorEqual<float> op;
        op.Init(&pipe, x1, x2, y, workspace, tiling);
        op.Process();
    } else if (TILING_KEY_IS(2)) { // DT_BF16 element-wise
        KernelTensorEqual<bfloat16_t> op;
        op.Init(&pipe, x1, x2, y, workspace, tiling);
        op.Process();
    } else if (TILING_KEY_IS(3)) { // DT_INT32 element-wise
        KernelTensorEqual<int32_t> op;
        op.Init(&pipe, x1, x2, y, workspace, tiling);
        op.Process();
    } else if (TILING_KEY_IS(4)) { // DT_INT16 element-wise
        KernelTensorEqual<int16_t> op;
        op.Init(&pipe, x1, x2, y, workspace, tiling);
        op.Process();
    } else if (TILING_KEY_IS(5)) { // DT_INT8 element-wise
        KernelTensorEqual<int8_t> op;
        op.Init(&pipe, x1, x2, y, workspace, tiling);
        op.Process();
    } else if (TILING_KEY_IS(6)) { // DT_UINT8 element-wise
        KernelTensorEqual<uint8_t> op;
        op.Init(&pipe, x1, x2, y, workspace, tiling);
        op.Process();
    } else if (TILING_KEY_IS(7)) { // DT_FLOAT16 broadcast
        KernelTensorEqualBroadcast<half> op;
        op.Init(&pipe, x1, x2, y, workspace, tiling);
        op.Process();
    } else if (TILING_KEY_IS(8)) { // DT_FLOAT broadcast
        KernelTensorEqualBroadcast<float> op;
        op.Init(&pipe, x1, x2, y, workspace, tiling);
        op.Process();
    } else if (TILING_KEY_IS(9)) { // DT_BF16 broadcast
        KernelTensorEqualBroadcast<bfloat16_t> op;
        op.Init(&pipe, x1, x2, y, workspace, tiling);
        op.Process();
    } else if (TILING_KEY_IS(10)) { // DT_INT32 broadcast
        KernelTensorEqualBroadcast<int32_t> op;
        op.Init(&pipe, x1, x2, y, workspace, tiling);
        op.Process();
    } else if (TILING_KEY_IS(11)) { // DT_INT16 broadcast
        KernelTensorEqualBroadcast<int16_t> op;
        op.Init(&pipe, x1, x2, y, workspace, tiling);
        op.Process();
    } else if (TILING_KEY_IS(12)) { // DT_INT8 broadcast
        KernelTensorEqualBroadcast<int8_t> op;
        op.Init(&pipe, x1, x2, y, workspace, tiling);
        op.Process();
    } else if (TILING_KEY_IS(13)) { // DT_UINT8 broadcast
        KernelTensorEqualBroadcast<uint8_t> op;
        op.Init(&pipe, x1, x2, y, workspace, tiling);
        op.Process();
    }
}
