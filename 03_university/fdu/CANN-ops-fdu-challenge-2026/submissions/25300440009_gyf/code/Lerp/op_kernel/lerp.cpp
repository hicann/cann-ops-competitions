// Kernel侧核函数实现
#include "kernel_operator.h"
#include <type_traits>

constexpr int32_t BUFFER_NUM = 2;

template <class DT_START>
class KernelLerp {
public:
    __aicore__ inline KernelLerp() {}
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR y, uint32_t totalLength, 
        uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain, DT_START weight) {
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
        
        Gm_start.SetGlobalBuffer((__gm__ DT_START*)start + startPointer, bufferlength);
        Gm_end.SetGlobalBuffer((__gm__ DT_START*)end + startPointer, bufferlength);
        Gm_y.SetGlobalBuffer((__gm__ DT_START*)y + startPointer, bufferlength);
        
        //向上取整计算需要处理的tile数量
        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);
        pipe.InitBuffer(Q_start, BUFFER_NUM, this->tileLength * sizeof(DT_START));
        pipe.InitBuffer(Q_end, BUFFER_NUM, this->tileLength * sizeof(DT_START));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(DT_START));
        this->weight = weight;
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
        AscendC::LocalTensor<DT_START> start = Q_start.AllocTensor<DT_START>();
        AscendC::LocalTensor<DT_START> end = Q_end.AllocTensor<DT_START>();
        AscendC::DataCopy(start, Gm_start[progress * this->tileLength], length);
        AscendC::DataCopy(end, Gm_end[progress * this->tileLength], length);
        Q_start.EnQue(start);
        Q_end.EnQue(end);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        AscendC::LocalTensor<DT_START> start = Q_start.DeQue<DT_START>();
        AscendC::LocalTensor<DT_START> end = Q_end.DeQue<DT_START>();
        AscendC::LocalTensor<DT_START> y = Q_y.AllocTensor<DT_START>();

        // 将 weight 转换为 float 进行比较
        float weight_f = static_cast<float>(this->weight);
    
        if (weight_f == 0.0f) {
            AscendC::DataCopy(y, start, length);
        } else if (weight_f == 1.0f) {
            AscendC::DataCopy(y, end, length);
        } else {
            AscendC::Sub(y, end, start, length);
            AscendC::Muls(y, y, this->weight, length);
            AscendC::Add(y, start, y, length);
        }
        Q_start.FreeTensor(start);
        Q_end.FreeTensor(end);
        Q_y.EnQue<DT_START>(y);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        AscendC::LocalTensor<DT_START> y = Q_y.DeQue<DT_START>();
        AscendC::DataCopy(Gm_y[progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> Q_start;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> Q_end;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> Q_y;
    AscendC::GlobalTensor<DT_START> Gm_start;
    AscendC::GlobalTensor<DT_START> Gm_end;
    AscendC::GlobalTensor<DT_START> Gm_y;
    DT_START weight;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

__global__ __aicore__ void lerp(GM_ADDR start, GM_ADDR end, GM_ADDR y, 
                                 GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    DTYPE_START weight = static_cast<DTYPE_START>(tiling_data.weight);
    KernelLerp<DTYPE_START> op;
    op.Init(start, end, y, tiling_data.totalLength, 
            tiling_data.ALIGN_NUM, tiling_data.block_size, 
            tiling_data.core_size, tiling_data.core_remain, weight);
    op.Process();
}