// kernel侧Tiling实现
#include "kernel_operator.h"
#include <type_traits>

constexpr int32_t BUFFER_NUM = 2;
template<typename TYPE_X> class KernelClipByValue {
public:
    __aicore__ inline KernelClipByValue() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain,TYPE_X clip_min, TYPE_X clip_max ) {
        // 获取当前核心信息
        uint32_t blockNum = AscendC::GetBlockNum();   // 总核心数
        uint32_t blockIdx = AscendC::GetBlockIdx();   // 当前核心索引
        // 计算当前核心处理的数据量
        uint32_t blockLength = core_size;
        bool isLastBlock = (blockIdx == blockNum - 1);  // 是否是最后一个核心
        if (isLastBlock) {
            blockLength += core_remain;  // 最后一个核心加上剩余数据
        }
        this->blockLength = blockLength;
        //每个tile的元素数
        this->tileLength = block_size;
        //对齐处理，确保每个核心处理的数据量是ALIGN_NUM的倍数
        uint32_t remainder = this->blockLength % ALIGN_NUM;
        if (remainder != 0) {
            this->blockLength += ALIGN_NUM - remainder;
        }
        if(this->blockLength == 0){
            this->tileNum = 0;
            return;
        }
        auto startPointer = core_size * AscendC::GetBlockIdx();
        auto bufferlength = this->blockLength;

        Gm_x.SetGlobalBuffer((__gm__ TYPE_X*)x + startPointer, bufferlength);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_X*)y + startPointer, bufferlength);
        //向上取整计算需要处理的tile数量
        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);
        pipe.InitBuffer(Q_x, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));
        this->clip_value_min = clip_min;
        this->clip_value_max = clip_max;
    }
    __aicore__ inline void Process() {
        if (this->tileNum == 0) return;
        // 预加载第一个tile
        CopyIn(0, this->tileLength);
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount - 1; i++) {
            // 预加载下一个tile
            CopyIn(i + 1, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        // 处理最后一个tile
        uint32_t length = this->blockLength - this->tileLength * (loopCount - 1);
        Compute(loopCount - 1, length);
        CopyOut(loopCount - 1, length);
    }
private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        AscendC::LocalTensor<TYPE_X> x = Q_x.AllocTensor<TYPE_X>();
        AscendC::DataCopy(x, Gm_x[progress * this->tileLength], length);
        Q_x.EnQue(x);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        AscendC::LocalTensor<TYPE_X> x = Q_x.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> y = Q_y.AllocTensor<TYPE_X>();
        AscendC::Mins(x, x, this->clip_value_max, length);
        AscendC::Maxs(y, x, this->clip_value_min, length);
        Q_x.FreeTensor(x);
        Q_y.EnQue<TYPE_X>(y);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        AscendC::LocalTensor<TYPE_X> y = Q_y.DeQue<TYPE_X>();
        AscendC::DataCopy(Gm_y[progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> Q_x;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> Q_y;
    AscendC::GlobalTensor<TYPE_X> Gm_x;
    AscendC::GlobalTensor<TYPE_X> Gm_y;
    TYPE_X clip_value_min;  
    TYPE_X clip_value_max;  
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void clip_by_value(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    //反序列化Tiling数据
    GET_TILING_DATA(tiling_data, tiling);
    // 从tiling_data中获取min和max值
    DTYPE_X clip_min = static_cast<DTYPE_X>(tiling_data.minVal);
    DTYPE_X clip_max = static_cast<DTYPE_X>(tiling_data.maxVal);
    KernelClipByValue<DTYPE_X> op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain, clip_min, clip_max);
    op.Process();
}