// Kernel侧核函数实现
#include "kernel_operator.h"
#include <type_traits>

constexpr int32_t BUFFER_NUM = 2;                                     
template<typename T> struct Map {using type = T;};
template<> struct Map<int8_t> {using type = half;};
//非广播场景的核函数实现
template<class TYPE_INPUT_DATA> class KernelAddcmul {
    using T = TYPE_INPUT_DATA;
public:
    __aicore__ inline KernelAddcmul() {}
    __aicore__ inline void Init(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y, uint32_t totalLength, 
        uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {
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
        auto startPointer = core_size * AscendC::GetBlockIdx();
        auto bufferlength = this->blockLength;

        Gm_input_data.SetGlobalBuffer((__gm__ TYPE_INPUT_DATA*)input_data + startPointer, bufferlength);
        Gm_x1.SetGlobalBuffer((__gm__ TYPE_INPUT_DATA*)x1 + startPointer, bufferlength);
        Gm_x2.SetGlobalBuffer((__gm__ TYPE_INPUT_DATA*)x2 + startPointer, bufferlength);
        Gm_value.SetGlobalBuffer((__gm__ TYPE_INPUT_DATA*)value, 1);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_INPUT_DATA*)y + startPointer, bufferlength);

        //向上取整计算需要处理的tile数量
        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        pipe.InitBuffer(Q_input_data, BUFFER_NUM, this->tileLength * sizeof(TYPE_INPUT_DATA));
        pipe.InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(TYPE_INPUT_DATA));
        pipe.InitBuffer(Q_x2, BUFFER_NUM, this->tileLength * sizeof(TYPE_INPUT_DATA));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_INPUT_DATA));
        pipe.InitBuffer(temp1, this->tileLength * sizeof(half));
        pipe.InitBuffer(temp2, this->tileLength * sizeof(half));
        this->value = Gm_value.GetValue(0);
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
        AscendC::LocalTensor<TYPE_INPUT_DATA> input_data = Q_input_data.AllocTensor<TYPE_INPUT_DATA>();
        AscendC::LocalTensor<TYPE_INPUT_DATA> x1 = Q_x1.AllocTensor<TYPE_INPUT_DATA>();
        AscendC::LocalTensor<TYPE_INPUT_DATA> x2 = Q_x2.AllocTensor<TYPE_INPUT_DATA>();
        AscendC::DataCopy(input_data, Gm_input_data[progress * this->tileLength], length);
        AscendC::DataCopy(x1, Gm_x1[progress * this->tileLength], length);
        AscendC::DataCopy(x2, Gm_x2[progress * this->tileLength], length);
        Q_input_data.EnQue(input_data);
        Q_x1.EnQue(x1);
        Q_x2.EnQue(x2);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        AscendC::LocalTensor<TYPE_INPUT_DATA> input_data = Q_input_data.DeQue<TYPE_INPUT_DATA>();
        AscendC::LocalTensor<TYPE_INPUT_DATA> x1 = Q_x1.DeQue<TYPE_INPUT_DATA>();
        AscendC::LocalTensor<TYPE_INPUT_DATA> x2 = Q_x2.DeQue<TYPE_INPUT_DATA>();
        AscendC::LocalTensor<TYPE_INPUT_DATA> y = Q_y.AllocTensor<TYPE_INPUT_DATA>();
        //如果是INT8类型的数据，由于ASCENDC不支持INT8的乘法和加法，所以需要先转为half类型计算，计算完成后再转回INT8
        if constexpr (std::is_same_v<TYPE_INPUT_DATA, int8_t>) {
            AscendC::LocalTensor<half> x1_half = temp1.Get<half>();
            AscendC::LocalTensor<half> x2_half = temp2.Get<half>();
            // 转换 x1, x2 为 half
            AscendC::Cast(x1_half, x1, AscendC::RoundMode::CAST_NONE, length);
            AscendC::Cast(x2_half, x2, AscendC::RoundMode::CAST_NONE, length);
            // 计算 x1 * x2 * value，存储在 x1_half 中
            AscendC::Mul(x1_half, x1_half, x2_half, length);
            AscendC::Muls(x1_half, x1_half, value, length);
            // 复用 x2_half 存 input_data (half)
            AscendC::Cast(x2_half, input_data, AscendC::RoundMode::CAST_NONE, length);
            // x1_half = x1_half + input_data
            AscendC::Add(x1_half, x1_half, x2_half, length);

            AscendC::Cast(x2_half.ReinterpretCast<int16_t>(), x1_half, AscendC::RoundMode::CAST_RINT, length);
            AscendC::ShiftLeft(x2_half.ReinterpretCast<int16_t>(), x2_half.ReinterpretCast<int16_t>(), int16_t(8), length);
            AscendC::ShiftRight(x2_half.ReinterpretCast<int16_t>(), x2_half.ReinterpretCast<int16_t>(), int16_t(8), length);
            //转回half类型
            AscendC::Cast(x1_half, x2_half.ReinterpretCast<int16_t>(), AscendC::RoundMode::CAST_NONE, length);
            //最后转换回 int8 输出
            AscendC::Cast(y, x1_half, AscendC::RoundMode::CAST_NONE, length);
        }
        else {
            AscendC::Mul(x1, x1, x2, length);
            AscendC::Muls(x1, x1, value, length);
            AscendC::Add(y, x1, input_data, length);
        }
        Q_input_data.FreeTensor(input_data);
        Q_x1.FreeTensor(x1);
        Q_x2.FreeTensor(x2);
        Q_y.EnQue<TYPE_INPUT_DATA>(y);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        AscendC::LocalTensor<TYPE_INPUT_DATA> y = Q_y.DeQue<TYPE_INPUT_DATA>();
        AscendC::DataCopy(Gm_y[progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> Q_input_data, Q_x1, Q_x2;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> Q_y;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> temp1, temp2;
    AscendC::GlobalTensor<TYPE_INPUT_DATA> Gm_input_data;
    AscendC::GlobalTensor<TYPE_INPUT_DATA> Gm_x1;
    AscendC::GlobalTensor<TYPE_INPUT_DATA> Gm_x2;
    AscendC::GlobalTensor<TYPE_INPUT_DATA> Gm_value;
    AscendC::GlobalTensor<TYPE_INPUT_DATA> Gm_y;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    //将INT_8类型的value转为half类型
    typename Map<TYPE_INPUT_DATA>::type value;
};
//广播场景的核函数实现
template<class TYPE_INPUT_DATA> class KernelAddcmul_Broadcast {
public:
    __aicore__ inline KernelAddcmul_Broadcast() {}
    __aicore__ inline void Init(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y, uint32_t input_data_length, 
        uint32_t x1_length, uint32_t x2_length, uint32_t total_length, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {
        //初始化输入输出数据的长度和每个核心处理的数据量
        this->inputdataLength = input_data_length;
        this->x1Length = x1_length;
        this->x2Length = x2_length;
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
        this->startPointer = core_size * AscendC::GetBlockIdx();
        //广播版本使用全长，不是偏移
        Gm_input_data.SetGlobalBuffer((__gm__ TYPE_INPUT_DATA*)input_data, total_length);
        Gm_x1.SetGlobalBuffer((__gm__ TYPE_INPUT_DATA*)x1, total_length);
        Gm_x2.SetGlobalBuffer((__gm__ TYPE_INPUT_DATA*)x2, total_length);
        Gm_value.SetGlobalBuffer((__gm__ TYPE_INPUT_DATA*)value, 1);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_INPUT_DATA*)y, total_length);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);
        pipe.InitBuffer(Q_input_data, BUFFER_NUM, this->tileLength * sizeof(TYPE_INPUT_DATA));
        pipe.InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(TYPE_INPUT_DATA));
        pipe.InitBuffer(Q_x2, BUFFER_NUM, this->tileLength * sizeof(TYPE_INPUT_DATA));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_INPUT_DATA));
        pipe.InitBuffer(temp1, this->tileLength * sizeof(half));
        pipe.InitBuffer(temp2, this->tileLength * sizeof(half));
        this->value = Gm_value.GetValue(0);
    }
    __aicore__ inline void Process() {
        if (this->tileNum == 0) return;
        // 预加载第一个tile
        uint32_t pos0 = startPointer + 0 * this->tileLength;
        CopyIn(pos0, this->tileLength);
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount - 1; i++) {
            // 预加载下一个tile，tileLength表示每个tile的元素数量
            uint32_t nextPos = startPointer + (i + 1) * this->tileLength;
            CopyIn(nextPos, this->tileLength);
            // 计算当前tile
            uint32_t currPos = startPointer + i * this->tileLength;
            Compute(this->tileLength);
            CopyOut(currPos, this->tileLength);
        }
        // 处理最后一个tile
        uint32_t lastPos = startPointer + (loopCount - 1) * this->tileLength;
        uint32_t length = this->blockLength - this->tileLength * (loopCount - 1);
        Compute(length);
        CopyOut(lastPos, length);
    }

private:
    //position表示当前tile的起始位置
    __aicore__ inline void CopyIn(int32_t position, uint32_t length) {
        AscendC::LocalTensor<TYPE_INPUT_DATA> input_data = Q_input_data.AllocTensor<TYPE_INPUT_DATA>();
        AscendC::LocalTensor<TYPE_INPUT_DATA> x1 = Q_x1.AllocTensor<TYPE_INPUT_DATA>();
        AscendC::LocalTensor<TYPE_INPUT_DATA> x2 = Q_x2.AllocTensor<TYPE_INPUT_DATA>();
        AscendC::DataCopy(input_data, Gm_input_data[position % inputdataLength], length);
        //输入数据的长度小于输出数据长度时，对position取模，确保访问的是正确的数据的同时保证数据的循环访问
        AscendC::DataCopy(x1, Gm_x1[position % x1Length], length);
        AscendC::DataCopy(x2, Gm_x2[position % x2Length], length);
        Q_input_data.EnQue(input_data);
        Q_x1.EnQue(x1);
        Q_x2.EnQue(x2);
    }
    __aicore__ inline void Compute(uint32_t length) {
        AscendC::LocalTensor<TYPE_INPUT_DATA> input_data = Q_input_data.DeQue<TYPE_INPUT_DATA>();
        AscendC::LocalTensor<TYPE_INPUT_DATA> x1 = Q_x1.DeQue<TYPE_INPUT_DATA>();
        AscendC::LocalTensor<TYPE_INPUT_DATA> x2 = Q_x2.DeQue<TYPE_INPUT_DATA>();
        AscendC::LocalTensor<TYPE_INPUT_DATA> y = Q_y.AllocTensor<TYPE_INPUT_DATA>();
        if constexpr (std::is_same_v<TYPE_INPUT_DATA, int8_t>) {
            AscendC::LocalTensor<half> x1_half = temp1.Get<half>();
            AscendC::LocalTensor<half> x2_half = temp2.Get<half>();
            // 转换 x1, x2 为 half
            AscendC::Cast(x1_half, x1, AscendC::RoundMode::CAST_NONE, length);
            AscendC::Cast(x2_half, x2, AscendC::RoundMode::CAST_NONE, length);
            // 计算 x1 * x2 * value，存储在 x1_half 中
            AscendC::Mul(x1_half, x1_half, x2_half, length);
            AscendC::Muls(x1_half, x1_half, value, length);
            // 复用 x2_half 存 input_data (half)
            AscendC::Cast(x2_half, input_data, AscendC::RoundMode::CAST_NONE, length);
            // x1_half = x1_half + input_data
            AscendC::Add(x1_half, x1_half, x2_half, length);

            AscendC::Cast(x2_half.ReinterpretCast<int16_t>(), x1_half, AscendC::RoundMode::CAST_RINT, length);
            AscendC::ShiftLeft(x2_half.ReinterpretCast<int16_t>(), x2_half.ReinterpretCast<int16_t>(), int16_t(8), length);
            AscendC::ShiftRight(x2_half.ReinterpretCast<int16_t>(), x2_half.ReinterpretCast<int16_t>(), int16_t(8), length);
            //转回half类型
            AscendC::Cast(x1_half, x2_half.ReinterpretCast<int16_t>(), AscendC::RoundMode::CAST_NONE, length);
            //最后转换回 int8 输出
            AscendC::Cast(y, x1_half, AscendC::RoundMode::CAST_NONE, length);
        }
        else {
            AscendC::Mul(x1, x1, x2, length);
            AscendC::Muls(x1, x1, value, length);
            AscendC::Add(y, x1, input_data, length);
        }
        Q_input_data.FreeTensor(input_data);
        Q_x1.FreeTensor(x1);
        Q_x2.FreeTensor(x2);
        Q_y.EnQue<TYPE_INPUT_DATA>(y);
    }
    __aicore__ inline void CopyOut(int32_t position, uint32_t length) {
        AscendC::LocalTensor<TYPE_INPUT_DATA> y = Q_y.DeQue<TYPE_INPUT_DATA>();
        AscendC::DataCopy(Gm_y[position], y, length);
        Q_y.FreeTensor(y);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> Q_input_data, Q_x1, Q_x2;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> Q_y;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> temp1, temp2;
    AscendC::GlobalTensor<TYPE_INPUT_DATA> Gm_input_data;
    AscendC::GlobalTensor<TYPE_INPUT_DATA> Gm_x1;
    AscendC::GlobalTensor<TYPE_INPUT_DATA> Gm_x2;
    AscendC::GlobalTensor<TYPE_INPUT_DATA> Gm_value;
    AscendC::GlobalTensor<TYPE_INPUT_DATA> Gm_y;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t tileNum;
    uint32_t startPointer;
    uint32_t inputdataLength;
    uint32_t x1Length;
    uint32_t x2Length;
    typename Map<TYPE_INPUT_DATA>::type value;
};
extern "C" __global__ __aicore__ void addcmul(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    //非广播场景，输入数据的长度、x1的长度、x2的长度都等于total_length
    if (tiling_data.input_data_length == tiling_data.total_length && tiling_data.x1_length == tiling_data.total_length && tiling_data.x2_length == tiling_data.total_length) {
        KernelAddcmul<DTYPE_INPUT_DATA> op;
        op.Init(input_data, x1, x2, value, y, tiling_data.total_length, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
        op.Process();
    }
    //广播场景
    else {
        KernelAddcmul_Broadcast<DTYPE_INPUT_DATA> op;
        op.Init(input_data, x1, x2, value, y, tiling_data.input_data_length, tiling_data.x1_length, tiling_data.x2_length, tiling_data.total_length, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
        op.Process();
    }
}


