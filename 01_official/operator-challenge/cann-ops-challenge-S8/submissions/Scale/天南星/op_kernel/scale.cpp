#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
template<typename T, typename... Ts>
struct is_one_of : std::false_type {};
template<typename T, typename U, typename... Ts>
struct is_one_of<T, U, Ts...>
    : std::conditional_t<std::is_same_v<T, U>,
                         std::true_type,
                         is_one_of<T, Ts...>> {};
template<typename T, typename... Ts>
constexpr bool is_one_of_v = is_one_of<T, Ts...>::value;

template<typename T>
__aicore__ inline constexpr T ceil_div(T x, T y)
{ return (x - 1) / y + 1; }

template<typename T>
__aicore__ inline constexpr T ceil_round(T x, T y)
{ return ceil_div(x, y) * y; }


struct TilingParam {
    uint32_t total_length;
    uint32_t start_length;
    uint32_t end_length;
    uint32_t weight_length;
    uint32_t ALIGN_NUM;
    uint32_t tiling_size;
    uint32_t block_size;
    uint32_t core_size;
    uint32_t core_remain;
    uint32_t has_bias;
    uint32_t shape[20];
    uint32_t reduce1[20];
    uint32_t reduce2[20];
    uint32_t reduce3[20];
    uint32_t dim;
};
template<typename TYPE_S, typename TYPE_E, typename TYPE_W, typename TYPE_Y>
class Scale {
public:
    __aicore__ inline Scale() {}
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR weight,
                                GM_ADDR y, TilingParam& paramList, TPipe* pipeIn) {
        this->blockLength = paramList.core_size +
            (GetBlockNum() == GetBlockIdx() + 1 ? paramList.core_remain : 0);
        this->tileLength = paramList.block_size;
        this->ALIGN_NUM  = paramList.ALIGN_NUM;
        //this->mode       = paramList.mode;
        this->has_bias   = paramList.has_bias;

        ASSERT(this->ALIGN_NUM != 0 && "ALIGN_NUM can not be zero!");

        // blockLength 向上对齐
        this->blockLength = this->blockLength +
            (this->blockLength % this->ALIGN_NUM ?
             this->ALIGN_NUM - this->blockLength % this->ALIGN_NUM : 0);

        auto startPointer = paramList.core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        startGm.SetGlobalBuffer((__gm__ TYPE_S*)start + startPointer, bufferlength);
        endGm.SetGlobalBuffer((__gm__ TYPE_E*)end + startPointer, bufferlength);

        // 非float需要cast buffer
        if constexpr (!std::is_same_v<TYPE_S, float>) {
            pipeIn->InitBuffer(calbuf1, this->tileLength * sizeof(float));
            pipeIn->InitBuffer(calbuf2, this->tileLength * sizeof(float));
            pipeIn->InitBuffer(calbuf3, this->tileLength * sizeof(float));
            //pipeIn->InitBuffer(calbuf4, this->tileLength * sizeof(float));
        } else {
            if ( this->has_bias) {
                pipeIn->InitBuffer(calbuf3, this->tileLength * sizeof(float));
            }
        }

        //if (this->mode == 1) {
            // bias是逐元素tensor
        weightGm.SetGlobalBuffer((__gm__ TYPE_W*)weight + startPointer, bufferlength);
        pipeIn->InitBuffer(inQueueWEIGHT, BUFFER_NUM, this->tileLength * sizeof(TYPE_W));
        //} 
        // else {
        //     // bias是scalar（length=1）
        //     if (this->has_bias) {
        //         weightGm.SetGlobalBuffer((__gm__ TYPE_W*)weight, 1);
        //         this->weightTmp = calbuf3.Get<float>();

        //         if constexpr (std::is_same_v<TYPE_S, bfloat16_t>) {
        //             pipeIn->InitBuffer(tmp1, 1 * sizeof(TYPE_S));
        //             auto p1 = tmp1.Get<TYPE_S>();
        //             Duplicate(p1, (TYPE_S)weightGm.GetValue(0), 1);
        //             Cast(this->weightTmp, p1, RoundMode::CAST_NONE, 1);
        //             Duplicate(this->weightTmp, this->weightTmp.GetValue(0), this->tileLength);
        //         } else if constexpr (std::is_same_v<TYPE_S, half>) {
        //             pipeIn->InitBuffer(tmp1, 1 * sizeof(TYPE_S));
        //             auto p1 = tmp1.Get<TYPE_S>();
        //             Duplicate(p1, (TYPE_S)weightGm.GetValue(0), 1);
        //             Cast(this->weightTmp, p1, RoundMode::CAST_NONE, 1);
        //             Duplicate(this->weightTmp, this->weightTmp.GetValue(0), this->tileLength);
        //         } else {
        //             TYPE_W w = weightGm.GetValue(0);
        //             Duplicate(this->weightTmp, (float)w, this->tileLength);
        //         }
        //     }
        // }

        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength +
            (this->blockLength % this->tileLength > 0);

        pipeIn->InitBuffer(inQueueSTART, BUFFER_NUM, this->tileLength * sizeof(TYPE_S));
        pipeIn->InitBuffer(inQueueEND,   BUFFER_NUM, this->tileLength * sizeof(TYPE_E));
        pipeIn->InitBuffer(outQueueY,    BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
    }

    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;

      {
            // ---- mode1: bias是逐元素tensor ----
            if constexpr (std::is_same_v<TYPE_S, float>) {
                for (int32_t i = 0; i < loopCount - 1; i++) {
                    CopyIn_mode1(i, this->tileLength);
                    Computefp32_mode1(i, this->tileLength);
                    CopyOut(i, this->tileLength);
                }
                uint32_t tail = this->blockLength - this->tileLength * (loopCount - 1);
                uint32_t tailAligned = (tail + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM;
                CopyIn_mode1(loopCount - 1, tailAligned);
                Computefp32_mode1(loopCount - 1, tail);
                CopyOut(loopCount - 1, tailAligned);

            } else if constexpr (std::is_same_v<TYPE_S, half>) {
                for (int32_t i = 0; i < loopCount - 1; i++) {
                    CopyIn_mode1(i, this->tileLength);
                    Computefp16_mode1(i, this->tileLength);
                    CopyOut(i, this->tileLength);
                }
                uint32_t tail = this->blockLength - this->tileLength * (loopCount - 1);
                uint32_t tailAligned = (tail + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM;
                CopyIn_mode1(loopCount - 1, tailAligned);
                Computefp16_mode1(loopCount - 1, tail);
                CopyOut(loopCount - 1, tailAligned);

            } else if constexpr (std::is_same_v<TYPE_S, bfloat16_t>) {
                for (int32_t i = 0; i < loopCount - 1; i++) {
                    CopyIn_mode1(i, this->tileLength);
                    Computebf16_mode1(i, this->tileLength);
                    CopyOut(i, this->tileLength);
                }
                uint32_t tail = this->blockLength - this->tileLength * (loopCount - 1);
                uint32_t tailAligned = (tail + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM;
                CopyIn_mode1(loopCount - 1, tailAligned);
                Computebf16_mode1(loopCount - 1, tail);
                CopyOut(loopCount - 1, tailAligned);
            }
        }
    }

private:
    __aicore__ inline void CopyIn_mode0(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_S> startLocal = inQueueSTART.AllocTensor<TYPE_S>();
        LocalTensor<TYPE_E> endLocal   = inQueueEND.AllocTensor<TYPE_E>();
        DataCopy(startLocal, startGm[progress * this->tileLength], length);
        DataCopy(endLocal,   endGm[progress * this->tileLength],   length);
        inQueueSTART.EnQue(startLocal);
        inQueueEND.EnQue(endLocal);
    }

    __aicore__ inline void CopyIn_mode1(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_S> startLocal  = inQueueSTART.AllocTensor<TYPE_S>();
        LocalTensor<TYPE_E> endLocal    = inQueueEND.AllocTensor<TYPE_E>();
        LocalTensor<TYPE_W> weightLocal = inQueueWEIGHT.AllocTensor<TYPE_W>();
        DataCopy(startLocal,  startGm[progress * this->tileLength],  length);
        DataCopy(endLocal,    endGm[progress * this->tileLength],    length);
        DataCopy(weightLocal, weightGm[progress * this->tileLength], length);
        inQueueSTART.EnQue(startLocal);
        inQueueEND.EnQue(endLocal);
        inQueueWEIGHT.EnQue(weightLocal);
    }

    // ---- fp32 mode0 ----
    __aicore__ inline void Computefp32_mode0(int32_t progress, uint32_t length) {
        LocalTensor<float> startLocal = inQueueSTART.DeQue<float>();
        LocalTensor<float> endLocal   = inQueueEND.DeQue<float>();
        LocalTensor<float> yLocal     = outQueueY.AllocTensor<float>();
        Mul(yLocal, startLocal, endLocal, length);
        if (this->has_bias) {
            Add(yLocal, yLocal, this->weightTmp, length);
        }
        outQueueY.EnQue<float>(yLocal);
        inQueueSTART.FreeTensor(startLocal);
        inQueueEND.FreeTensor(endLocal);
    }

    // ---- fp16 mode0 ----
    __aicore__ inline void Computefp16_mode0(int32_t progress, uint32_t length) {
        LocalTensor<half> startLocal = inQueueSTART.DeQue<half>();
        LocalTensor<half> endLocal   = inQueueEND.DeQue<half>();
        LocalTensor<half> yLocal     = outQueueY.AllocTensor<half>();
        auto fstart = calbuf1.Get<float>();
        auto fend   = calbuf2.Get<float>();
        //auto fy     = calbuf4.Get<float>();
        Cast(fstart, startLocal, RoundMode::CAST_NONE, length);
        Cast(fend,   endLocal,   RoundMode::CAST_NONE, length);
        Mul(fstart, fstart, fend, length);
        if (this->has_bias) {
            Add(fstart, fstart, this->weightTmp, length);
        }
        Cast(yLocal, fstart, RoundMode::CAST_NONE, length);
        outQueueY.EnQue<half>(yLocal);
        inQueueSTART.FreeTensor(startLocal);
        inQueueEND.FreeTensor(endLocal);
    }

    // ---- bf16 mode0 ----
    __aicore__ inline void Computebf16_mode0(int32_t progress, uint32_t length) {
        LocalTensor<bfloat16_t> startLocal = inQueueSTART.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> endLocal   = inQueueEND.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> yLocal     = outQueueY.AllocTensor<bfloat16_t>();
        auto fstart = calbuf1.Get<float>();
        auto fend   = calbuf2.Get<float>();
        //auto fy     = calbuf4.Get<float>();
        Cast(fstart, startLocal, RoundMode::CAST_NONE, length);
        Cast(fend,   endLocal,   RoundMode::CAST_NONE, length);
        Mul(fstart, fstart, fend, length);
        if (this->has_bias) {
            Add(fstart, fstart, this->weightTmp, length);
        }
        Cast(yLocal, fstart, RoundMode::CAST_RINT, length);
        outQueueY.EnQue<bfloat16_t>(yLocal);
        inQueueSTART.FreeTensor(startLocal);
        inQueueEND.FreeTensor(endLocal);
    }

    // ---- fp32 mode1 ----
    __aicore__ inline void Computefp32_mode1(int32_t progress, uint32_t length) {
        LocalTensor<float> startLocal  = inQueueSTART.DeQue<float>();
        LocalTensor<float> endLocal    = inQueueEND.DeQue<float>();
        LocalTensor<float> weightLocal = inQueueWEIGHT.DeQue<float>();
        LocalTensor<float> yLocal      = outQueueY.AllocTensor<float>();
        Mul(yLocal, startLocal, endLocal, length);
        if (this->has_bias) {
            Add(yLocal, yLocal, weightLocal, length);
        }
        outQueueY.EnQue<float>(yLocal);
        inQueueSTART.FreeTensor(startLocal);
        inQueueEND.FreeTensor(endLocal);
        inQueueWEIGHT.FreeTensor(weightLocal);
    }

    // ---- fp16 mode1 ----
    __aicore__ inline void Computefp16_mode1(int32_t progress, uint32_t length) {
        LocalTensor<half> startLocal  = inQueueSTART.DeQue<half>();
        LocalTensor<half> endLocal    = inQueueEND.DeQue<half>();
        LocalTensor<half> weightLocal = inQueueWEIGHT.DeQue<half>();
        LocalTensor<half> yLocal      = outQueueY.AllocTensor<half>();
        auto fstart  = calbuf1.Get<float>();
        auto fend    = calbuf2.Get<float>();
        auto fweight = calbuf3.Get<float>();
        //auto fy      = calbuf4.Get<float>();
        Cast(fstart, startLocal,  RoundMode::CAST_NONE, length);
        Cast(fend,   endLocal,    RoundMode::CAST_NONE, length);
        Mul(fstart, fstart, fend, length);
        if (this->has_bias) {
            Cast(fweight, weightLocal, RoundMode::CAST_NONE, length);
            Add(fstart, fstart, fweight, length);
        }
        Cast(yLocal, fstart, RoundMode::CAST_NONE, length);
        outQueueY.EnQue<half>(yLocal);
        inQueueSTART.FreeTensor(startLocal);
        inQueueEND.FreeTensor(endLocal);
        inQueueWEIGHT.FreeTensor(weightLocal);
    }

    // ---- bf16 mode1 ----
    __aicore__ inline void Computebf16_mode1(int32_t progress, uint32_t length) {
        LocalTensor<bfloat16_t> startLocal  = inQueueSTART.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> endLocal    = inQueueEND.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> weightLocal = inQueueWEIGHT.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> yLocal      = outQueueY.AllocTensor<bfloat16_t>();
        auto fstart  = calbuf1.Get<float>();
        auto fend    = calbuf2.Get<float>();
        auto fweight = calbuf3.Get<float>();
        //auto fy      = calbuf4.Get<float>();
        Cast(fstart, startLocal,  RoundMode::CAST_NONE, length);
        Cast(fend,   endLocal,    RoundMode::CAST_NONE, length);
        Mul(fstart, fstart, fend, length);
        if (this->has_bias) {
            Cast(fweight, weightLocal, RoundMode::CAST_NONE, length);
            Add(fstart, fstart, fweight, length);
        }
        Cast(yLocal, fstart, RoundMode::CAST_RINT, length);
        outQueueY.EnQue<bfloat16_t>(yLocal);
        inQueueSTART.FreeTensor(startLocal);
        inQueueEND.FreeTensor(endLocal);
        inQueueWEIGHT.FreeTensor(weightLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
        DataCopy(yGm[progress * this->tileLength], yLocal, length);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueSTART, inQueueEND, inQueueWEIGHT;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> calbuf1, calbuf2, calbuf3, calbuf4, tmp1;

    GlobalTensor<TYPE_S> startGm;
    GlobalTensor<TYPE_E> endGm;
    GlobalTensor<TYPE_W> weightGm;
    GlobalTensor<TYPE_Y> yGm;

    LocalTensor<float> weightTmp;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t ALIGN_NUM;
    //uint32_t mode;
    uint32_t has_bias;
};



template<typename TYPE_S, typename TYPE_E, typename TYPE_W, typename TYPE_Y>
class Scale_Broadcast {
public:
    __aicore__ inline Scale_Broadcast() {}

    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR weight,
                                GM_ADDR y, TilingParam& paramList, TPipe* pipeIn) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        this->tileLength = paramList.block_size;
        this->ALIGN_NUM = paramList.ALIGN_NUM;
        this->has_bias = paramList.has_bias;
        ASSERT(this->ALIGN_NUM != 0 && "ALIGN_NUM can not be zero!");

        this->reduce1 = paramList.reduce1;
        this->reduce2 = paramList.reduce2;
        this->reduce3 = paramList.reduce3;
        this->shape = paramList.shape;
        this->dim = paramList.dim;

        this->totalLength = paramList.total_length;
        this->startLength = paramList.start_length;
        this->endLength = paramList.end_length;
        this->weightLength = paramList.weight_length;

        this->innerLength = paramList.shape[paramList.dim - 1];
        this->outerLength = paramList.total_length / this->innerLength;

        this->scaleLastBroadcast = (this->reduce2[this->dim - 1] == 1);
        this->biasLastBroadcast = (this->has_bias && this->reduce3[this->dim - 1] == 1);
        this->scaleOuterBroadcast = IsOuterBroadcast(this->reduce2);
        this->biasOuterBroadcast = this->has_bias ? IsOuterBroadcast(this->reduce3) : 1;
        this->useInnerReuseFloat = 0;

        if constexpr (std::is_same_v<TYPE_S, float>) {
            if ((!this->scaleLastBroadcast) && this->scaleOuterBroadcast &&
                ((!this->has_bias) || this->biasOuterBroadcast)) {
                this->useInnerReuseFloat = 1;
            }
        }

        if (paramList.dim >= 2) {
            this->row_d[paramList.dim - 2] = 1;
            for (int32_t k = (int32_t)(paramList.dim - 2) - 1; k >= 0; k--) {
                this->row_d[k] = this->row_d[k + 1] * paramList.shape[k + 1];
            }
        }

        startGmB.SetGlobalBuffer((__gm__ TYPE_S*)start, startLength);
        endGmB.SetGlobalBuffer((__gm__ TYPE_E*)end, endLength);
        if (this->has_bias) {
            weightGmB.SetGlobalBuffer((__gm__ TYPE_W*)weight, weightLength);
        }
        yGmB.SetGlobalBuffer((__gm__ TYPE_Y*)y, totalLength);

        pipeIn->InitBuffer(inQueueSTARTB, BUFFER_NUM, this->tileLength * sizeof(TYPE_S));
        pipeIn->InitBuffer(inQueueENDB, BUFFER_NUM, this->tileLength * sizeof(TYPE_E));
        if (this->has_bias) {
            pipeIn->InitBuffer(inQueueWEIGHTB, BUFFER_NUM, this->tileLength * sizeof(TYPE_W));
        }
        pipeIn->InitBuffer(outQueueYB, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));

        if constexpr (!std::is_same_v<TYPE_S, float>) {
            pipeIn->InitBuffer(calbufB1, this->tileLength * sizeof(float));
            pipeIn->InitBuffer(calbufB2, this->tileLength * sizeof(float));
            if (this->has_bias) {
                pipeIn->InitBuffer(calbufB3, this->tileLength * sizeof(float));
            }
        }
    }

    __aicore__ inline void Process() {
        this->tileNum = this->innerLength / this->tileLength +
                        (this->innerLength % this->tileLength > 0);

        int32_t loopCount = (int32_t)this->tileNum;
        uint32_t tail = this->innerLength - this->tileLength * (uint32_t)(loopCount - 1);
        uint32_t tailAligned = (tail + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM;

        int32_t blkNum = (int32_t)GetBlockNum();
        int32_t blkIdx = (int32_t)GetBlockIdx();
        int32_t per = this->outerLength / blkNum;
        int32_t rem = this->outerLength % blkNum;
        int32_t j_start = blkIdx * per + (blkIdx < rem ? blkIdx : rem);
        int32_t j_end = j_start + per + (blkIdx < rem ? 1 : 0);

        if constexpr (std::is_same_v<TYPE_S, float>) {
            if (this->useInnerReuseFloat) {
                ProcessInnerReuseFloat(j_start, j_end, loopCount, tail, tailAligned);
                return;
            }
        }

        for (int32_t j = j_start; j < j_end; j++) {
            uint32_t offset1 = (uint32_t)j * this->innerLength;
            uint32_t offset2 = CalcOffsetOptimized(j, this->reduce2);
            uint32_t offset3 = 0;
            if (this->has_bias) {
                offset3 = CalcOffsetOptimized(j, this->reduce3);
            }

            for (int32_t i = 0; i < loopCount - 1; i++) {
                CopyInOptimized(offset1, offset2, offset3, i, this->tileLength);
                ComputeOptimized(i, this->tileLength);
                CopyOutOptimized(offset1, i, this->tileLength);
            }

            CopyInOptimized(offset1, offset2, offset3, loopCount - 1, tailAligned);
            ComputeOptimized(loopCount - 1, tail);
            CopyOutOptimized(offset1, loopCount - 1, tailAligned);
        }
    }

private:
    __aicore__ inline uint32_t IsOuterBroadcast(uint32_t* reduce) {
        for (int32_t k = 0; k < (int32_t)this->dim - 1; k++) {
            if (reduce[k] == 0) {
                return 0;
            }
        }
        return 1;
    }

    __aicore__ inline uint32_t CalcOffsetOptimized(int32_t j, uint32_t* reduce) {
        uint32_t offset = 0;
        uint32_t tensorStride = (reduce[this->dim - 1] == 0) ? this->innerLength : 1;

        for (int32_t k = (int32_t)this->dim - 2; k >= 0; k--) {
            uint32_t dimIndex = ((uint32_t)j / this->row_d[k]) % this->shape[k];
            if (reduce[k] == 0) {
                offset += tensorStride * dimIndex;
                tensorStride *= this->shape[k];
            }
        }
        return offset;
    }

    __aicore__ inline void ProcessInnerReuseFloat(int32_t j_start, int32_t j_end,
                                                 int32_t loopCount,
                                                 uint32_t tail,
                                                 uint32_t tailAligned) {
        if (j_start >= j_end) {
            return;
        }

        for (int32_t i = 0; i < loopCount; i++) {
            uint32_t copyLength = (i == loopCount - 1) ? tailAligned : this->tileLength;
            uint32_t computeLength = (i == loopCount - 1) ? tail : this->tileLength;
            uint32_t tileOffset = (uint32_t)i * this->tileLength;

            LocalTensor<float> scaleLocal = inQueueENDB.AllocTensor<float>();
            DataCopy(scaleLocal, endGmB[tileOffset], copyLength);
            inQueueENDB.EnQue(scaleLocal);
            scaleLocal = inQueueENDB.DeQue<float>();

            if (this->has_bias) {
                LocalTensor<float> biasLocal = inQueueWEIGHTB.AllocTensor<float>();
                if (this->biasLastBroadcast) {
                    DataCopy(biasLocal, weightGmB[0], this->ALIGN_NUM);
                } else {
                    DataCopy(biasLocal, weightGmB[tileOffset], copyLength);
                }
                inQueueWEIGHTB.EnQue(biasLocal);
                biasLocal = inQueueWEIGHTB.DeQue<float>();

                for (int32_t j = j_start; j < j_end; j++) {
                    ProcessInnerReuseRowBiasFloat((uint32_t)j * this->innerLength,
                                                  i, copyLength, computeLength,
                                                  scaleLocal, biasLocal);
                }
                inQueueWEIGHTB.FreeTensor(biasLocal);
            } else {
                for (int32_t j = j_start; j < j_end; j++) {
                    ProcessInnerReuseRowNoBiasFloat((uint32_t)j * this->innerLength,
                                                    i, copyLength, computeLength,
                                                    scaleLocal);
                }
            }

            inQueueENDB.FreeTensor(scaleLocal);
        }
    }

    __aicore__ inline void ProcessInnerReuseRowNoBiasFloat(uint32_t rowOffset,
                                                          int32_t progress,
                                                          uint32_t copyLength,
                                                          uint32_t computeLength,
                                                          LocalTensor<float> scaleLocal) {
        LocalTensor<float> startLocal = inQueueSTARTB.AllocTensor<float>();
        DataCopy(startLocal, startGmB[rowOffset + (uint32_t)progress * this->tileLength], copyLength);
        inQueueSTARTB.EnQue(startLocal);

        startLocal = inQueueSTARTB.DeQue<float>();
        LocalTensor<float> yLocal = outQueueYB.AllocTensor<float>();
        Mul(yLocal, startLocal, scaleLocal, computeLength);
        outQueueYB.EnQue<float>(yLocal);
        inQueueSTARTB.FreeTensor(startLocal);

        yLocal = outQueueYB.DeQue<float>();
        DataCopy(yGmB[rowOffset + (uint32_t)progress * this->tileLength], yLocal, copyLength);
        outQueueYB.FreeTensor(yLocal);
    }

    __aicore__ inline void ProcessInnerReuseRowBiasFloat(uint32_t rowOffset,
                                                        int32_t progress,
                                                        uint32_t copyLength,
                                                        uint32_t computeLength,
                                                        LocalTensor<float> scaleLocal,
                                                        LocalTensor<float> biasLocal) {
        LocalTensor<float> startLocal = inQueueSTARTB.AllocTensor<float>();
        DataCopy(startLocal, startGmB[rowOffset + (uint32_t)progress * this->tileLength], copyLength);
        inQueueSTARTB.EnQue(startLocal);

        startLocal = inQueueSTARTB.DeQue<float>();
        LocalTensor<float> yLocal = outQueueYB.AllocTensor<float>();
        Mul(yLocal, startLocal, scaleLocal, computeLength);
        if (this->biasLastBroadcast) {
            Adds(yLocal, yLocal, biasLocal.GetValue(0), computeLength);
        } else {
            Add(yLocal, yLocal, biasLocal, computeLength);
        }
        outQueueYB.EnQue<float>(yLocal);
        inQueueSTARTB.FreeTensor(startLocal);

        yLocal = outQueueYB.DeQue<float>();
        DataCopy(yGmB[rowOffset + (uint32_t)progress * this->tileLength], yLocal, copyLength);
        outQueueYB.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyInOptimized(uint32_t offset1, uint32_t offset2, uint32_t offset3,
                                          int32_t progress, uint32_t length) {
        LocalTensor<TYPE_S> startLocal = inQueueSTARTB.AllocTensor<TYPE_S>();
        LocalTensor<TYPE_E> endLocal = inQueueENDB.AllocTensor<TYPE_E>();

        DataCopy(startLocal, startGmB[offset1 + (uint32_t)progress * this->tileLength], length);
        inQueueSTARTB.EnQue(startLocal);

        if (this->scaleLastBroadcast) {
            DataCopy(endLocal, endGmB[offset2], this->ALIGN_NUM);
        } else {
            DataCopy(endLocal, endGmB[offset2 + (uint32_t)progress * this->tileLength], length);
        }
        inQueueENDB.EnQue(endLocal);

        if (this->has_bias) {
            LocalTensor<TYPE_W> weightLocal = inQueueWEIGHTB.AllocTensor<TYPE_W>();
            if (this->biasLastBroadcast) {
                DataCopy(weightLocal, weightGmB[offset3], this->ALIGN_NUM);
            } else {
                DataCopy(weightLocal, weightGmB[offset3 + (uint32_t)progress * this->tileLength], length);
            }
            inQueueWEIGHTB.EnQue(weightLocal);
        }
    }

    __aicore__ inline void ComputeOptimized(uint32_t progress, uint32_t length) {
        if constexpr (std::is_same_v<TYPE_S, float>) {
            ComputeFloatOptimized(progress, length);
        } else if constexpr (std::is_same_v<TYPE_S, half>) {
            ComputeHalfOptimized(progress, length);
        } else if constexpr (std::is_same_v<TYPE_S, bfloat16_t>) {
            ComputeBf16Optimized(progress, length);
        }
    }

    __aicore__ inline void ComputeFloatOptimized(uint32_t progress, uint32_t length) {
        LocalTensor<float> startLocal = inQueueSTARTB.DeQue<float>();
        LocalTensor<float> endLocal = inQueueENDB.DeQue<float>();
        LocalTensor<float> yLocal = outQueueYB.AllocTensor<float>();

        if (this->scaleLastBroadcast) {
            float val = endLocal.GetValue(0);
            Muls(yLocal, startLocal, val, length);
        } else {
            Mul(yLocal, startLocal, endLocal, length);
        }

        if (this->has_bias) {
            LocalTensor<float> weightLocal = inQueueWEIGHTB.DeQue<float>();
            if (this->biasLastBroadcast) {
                float val = weightLocal.GetValue(0);
                Adds(yLocal, yLocal, val, length);
            } else {
                Add(yLocal, yLocal, weightLocal, length);
            }
            inQueueWEIGHTB.FreeTensor(weightLocal);
        }

        outQueueYB.EnQue<float>(yLocal);
        inQueueSTARTB.FreeTensor(startLocal);
        inQueueENDB.FreeTensor(endLocal);
    }

    __aicore__ inline void ComputeHalfOptimized(uint32_t progress, uint32_t length) {
        LocalTensor<half> startLocal = inQueueSTARTB.DeQue<half>();
        LocalTensor<half> endLocal = inQueueENDB.DeQue<half>();
        LocalTensor<half> yLocal = outQueueYB.AllocTensor<half>();

        auto fstartB = calbufB1.Get<float>();
        auto fendB = calbufB2.Get<float>();

        Cast(fstartB, startLocal, RoundMode::CAST_NONE, length);

        if (this->scaleLastBroadcast) {
            half val = endLocal.GetValue(0);
            Duplicate(endLocal, val, length);
        }
        Cast(fendB, endLocal, RoundMode::CAST_NONE, length);

        Mul(fstartB, fstartB, fendB, length);

        if (this->has_bias) {
            LocalTensor<half> weightLocal = inQueueWEIGHTB.DeQue<half>();
            auto fweightB = calbufB3.Get<float>();
            if (this->biasLastBroadcast) {
                half val = weightLocal.GetValue(0);
                Duplicate(weightLocal, val, length);
            }
            Cast(fweightB, weightLocal, RoundMode::CAST_NONE, length);
            Add(fstartB, fstartB, fweightB, length);
            inQueueWEIGHTB.FreeTensor(weightLocal);
        }

        Cast(yLocal, fstartB, RoundMode::CAST_NONE, length);
        outQueueYB.EnQue<half>(yLocal);
        inQueueSTARTB.FreeTensor(startLocal);
        inQueueENDB.FreeTensor(endLocal);
    }

    __aicore__ inline void ComputeBf16Optimized(uint32_t progress, uint32_t length) {
        LocalTensor<bfloat16_t> startLocal = inQueueSTARTB.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> endLocal = inQueueENDB.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> yLocal = outQueueYB.AllocTensor<bfloat16_t>();

        auto fstartB = calbufB1.Get<float>();
        auto fendB = calbufB2.Get<float>();

        Cast(fstartB, startLocal, RoundMode::CAST_NONE, length);

        if (this->scaleLastBroadcast) {
            bfloat16_t val = endLocal.GetValue(0);
            Duplicate(endLocal, val, length);
        }
        Cast(fendB, endLocal, RoundMode::CAST_NONE, length);

        Mul(fstartB, fstartB, fendB, length);

        if (this->has_bias) {
            LocalTensor<bfloat16_t> weightLocal = inQueueWEIGHTB.DeQue<bfloat16_t>();
            auto fweightB = calbufB3.Get<float>();
            if (this->biasLastBroadcast) {
                bfloat16_t val = weightLocal.GetValue(0);
                Duplicate(weightLocal, val, length);
            }
            Cast(fweightB, weightLocal, RoundMode::CAST_NONE, length);
            Add(fstartB, fstartB, fweightB, length);
            inQueueWEIGHTB.FreeTensor(weightLocal);
        }

        Cast(yLocal, fstartB, RoundMode::CAST_RINT, length);
        outQueueYB.EnQue<bfloat16_t>(yLocal);
        inQueueSTARTB.FreeTensor(startLocal);
        inQueueENDB.FreeTensor(endLocal);
    }

    __aicore__ inline void CopyOutOptimized(uint32_t start, uint32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> yLocal = outQueueYB.DeQue<TYPE_Y>();
        DataCopy(yGmB[start + (uint32_t)progress * this->tileLength], yLocal, length);
        outQueueYB.FreeTensor(yLocal);
    }

private:
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueSTARTB;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueENDB;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueWEIGHTB;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueYB;
    TBuf<QuePosition::VECCALC> calbufB1, calbufB2, calbufB3;

    GlobalTensor<TYPE_S> startGmB;
    GlobalTensor<TYPE_E> endGmB;
    GlobalTensor<TYPE_W> weightGmB;
    GlobalTensor<TYPE_Y> yGmB;

    uint32_t innerLength;
    uint32_t outerLength;
    uint32_t row_d[21];

    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t ALIGN_NUM;
    uint32_t totalLength;
    uint32_t startLength;
    uint32_t weightLength;
    uint32_t endLength;
    uint32_t has_bias;
    uint32_t scaleLastBroadcast;
    uint32_t biasLastBroadcast;
    uint32_t scaleOuterBroadcast;
    uint32_t biasOuterBroadcast;
    uint32_t useInnerReuseFloat;

    uint32_t* reduce1;
    uint32_t* reduce2;
    uint32_t* reduce3;
    uint32_t* shape;
    uint32_t dim;
};




__aicore__ inline void scale_flat_float_fast(
    GM_ADDR input, GM_ADDR scale, GM_ADDR bias,
    GM_ADDR out, ScaleTilingData &tiling)
{
    int block_index = GetBlockIdx();
    int block_dim   = GetBlockNum();
    constexpr int DATA_BLOCK_SIZE = 32 / sizeof(float);

    int compute_blocks = ceil_div(tiling.size, DATA_BLOCK_SIZE);
    int compute_start  = (int)((long)compute_blocks * block_index
                               / block_dim * DATA_BLOCK_SIZE);
    int compute_end    = (int)min(
        (long)compute_blocks * (block_index + 1)
        / block_dim * DATA_BLOCK_SIZE,
        (long)tiling.size);
    if (compute_start >= compute_end) return;

    GlobalTensor<float> input_gm, scale_gm, bias_gm, out_gm;
    input_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
    scale_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale));
    out_gm  .SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out));
    if (tiling.has_bias)
        bias_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias));

    constexpr int BUF_BYTES_NOBIAS = 56 << 10;
    constexpr int BUF_BYTES_BIAS   = 40 << 10;
    int BUF_BYTES = tiling.has_bias ? BUF_BYTES_BIAS : BUF_BYTES_NOBIAS;
    int MAX_TILE_SIZE = (BUF_BYTES / (int)sizeof(float)
                         / DATA_BLOCK_SIZE) * DATA_BLOCK_SIZE;
    if (MAX_TILE_SIZE < DATA_BLOCK_SIZE)
        MAX_TILE_SIZE = DATA_BLOCK_SIZE;

    TPipe pipe;
    TQue<TPosition::VECIN,  1> input_q, scale_q, bias_q;
    TQue<TPosition::VECOUT, 1> out_q;

    pipe.InitBuffer(input_q, 1, BUF_BYTES);
    pipe.InitBuffer(scale_q, 1, BUF_BYTES);
    pipe.InitBuffer(out_q,   1, BUF_BYTES);
    if (tiling.has_bias)
        pipe.InitBuffer(bias_q, 1, BUF_BYTES);

    for (int i = compute_start; i < compute_end; i += MAX_TILE_SIZE) {
        int len = min(compute_end - i, MAX_TILE_SIZE);
        bool aligned = (len % DATA_BLOCK_SIZE == 0);

        LocalTensor<float> in = input_q.AllocTensor<float>();
        LocalTensor<float> sc = scale_q.AllocTensor<float>();
        if (aligned) {
            DataCopy(in, input_gm[i], len);
            DataCopy(sc, scale_gm[i], len);
        } else {
            DataCopyExtParams cp{1, (uint32_t)(len * sizeof(float)), 0, 0, 0};
            DataCopyPad(in, input_gm[i], cp, {});
            DataCopyPad(sc, scale_gm[i], cp, {});
        }
        input_q.EnQue(in);
        scale_q.EnQue(sc);

        if (tiling.has_bias) {
            LocalTensor<float> bi = bias_q.AllocTensor<float>();
            if (aligned) {
                DataCopy(bi, bias_gm[i], len);
            } else {
                DataCopyExtParams cp{1, (uint32_t)(len * sizeof(float)), 0, 0, 0};
                DataCopyPad(bi, bias_gm[i], cp, {});
            }
            bias_q.EnQue(bi);
        }

        in = input_q.DeQue<float>();
        sc = scale_q.DeQue<float>();
        LocalTensor<float> res = out_q.AllocTensor<float>();
        Mul(res, in, sc, len);
        input_q.FreeTensor(in);
        scale_q.FreeTensor(sc);
        if (tiling.has_bias) {
            LocalTensor<float> bi = bias_q.DeQue<float>();
            Add(res, res, bi, len);
            bias_q.FreeTensor(bi);
        }
        out_q.EnQue(res);

        res = out_q.DeQue<float>();
        if (aligned) {
            DataCopy(out_gm[i], res, len);
        } else {
            DataCopyExtParams cp{1, (uint32_t)(len * sizeof(float)), 0, 0, 0};
            DataCopyPad(out_gm[i], res, cp);
        }
        out_q.FreeTensor(res);
    }
}

// ═══════════════════════════════════════════════════════
//  scale_flat：float / half 版本（stype==0 / stype==1）
// ═══════════════════════════════════════════════════════
template<typename T>
__aicore__ inline std::enable_if_t<is_one_of_v<T, float, half>>
scale_flat(GM_ADDR input, GM_ADDR scale, GM_ADDR bias,
           GM_ADDR out, ScaleTilingData &tiling)
{
    int block_index = GetBlockIdx();
    int block_dim   = GetBlockNum();
    constexpr int DATA_BLOCK_SIZE = 32 / sizeof(T);

    int compute_blocks = ceil_div(tiling.size, DATA_BLOCK_SIZE);
    int compute_start  = (int)((long)compute_blocks * block_index
                               / block_dim * DATA_BLOCK_SIZE);
    int compute_end    = (int)min(
        (long)compute_blocks * (block_index + 1)
        / block_dim * DATA_BLOCK_SIZE,
        (long)tiling.size);

    GlobalTensor<T> input_gm, scale_gm, bias_gm, out_gm;
    input_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
    scale_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scale));
    out_gm  .SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
    if (tiling.has_bias)
        bias_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias));

    TPipe pipe;
    TQue<TPosition::VECIN,  1> input_q, scale_q, bias_q;
    TQue<TPosition::VECOUT, 1> out_q;

    int BUF_BYTES     = tiling.has_bias ? (24 << 10) : (32 << 10);
    int MAX_TILE_SIZE = BUF_BYTES / (int)sizeof(T);

    pipe.InitBuffer(input_q, 2, BUF_BYTES);
    pipe.InitBuffer(scale_q, 2, BUF_BYTES);
    pipe.InitBuffer(out_q,   2, BUF_BYTES);
    if (tiling.has_bias)
        pipe.InitBuffer(bias_q, 2, BUF_BYTES);

    uint16_t atomic_type, atomic_op;
    GetStoreAtomicConfig(atomic_type, atomic_op);

    int loop_start, loop_end, loop_step;
    if (atomic_type) {
        loop_start = compute_start
                     + ceil_round(compute_end - compute_start,
                                  MAX_TILE_SIZE)
                     - MAX_TILE_SIZE;
        loop_end   = compute_start - MAX_TILE_SIZE;
        loop_step  = -MAX_TILE_SIZE;
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_NONE,
                             AtomicOp::ATOMIC_SUM>();
    } else {
        loop_start = compute_start;
        loop_end   = compute_start
                     + ceil_round(compute_end - compute_start,
                                  MAX_TILE_SIZE);
        loop_step  = MAX_TILE_SIZE;
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_S32,
                             AtomicOp::ATOMIC_SUM>();
    }

    for (int i = loop_start; i != loop_end; i += loop_step) {
        int len = min(compute_end - i, MAX_TILE_SIZE);
        if (len <= 0) break;

        {
            DataCopyExtParams cp;
            cp.blockLen = (uint32_t)(len * sizeof(T));

            LocalTensor<T> in = input_q.AllocTensor<T>();
            DataCopyPad(in, input_gm[i], cp, {});
            input_q.EnQue(in);

            LocalTensor<T> sc = scale_q.AllocTensor<T>();
            DataCopyPad(sc, scale_gm[i], cp, {});
            scale_q.EnQue(sc);

            if (tiling.has_bias) {
                LocalTensor<T> bi = bias_q.AllocTensor<T>();
                DataCopyPad(bi, bias_gm[i], cp, {});
                bias_q.EnQue(bi);
            }
        }
        {
            LocalTensor<T> in  = input_q.DeQue<T>();
            LocalTensor<T> sc  = scale_q.DeQue<T>();
            LocalTensor<T> res = out_q.AllocTensor<T>();
            Mul(res, in, sc, len);
            if (tiling.has_bias) {
                LocalTensor<T> bi = bias_q.DeQue<T>();
                Add(res, res, bi, len);
                bias_q.FreeTensor(bi);
            }
            input_q.FreeTensor(in);
            scale_q.FreeTensor(sc);
            out_q.EnQue(res);
        }
        {
            LocalTensor<T> res = out_q.DeQue<T>();
            DataCopyExtParams cp;
            cp.blockLen = (uint32_t)(len * sizeof(T));
            DataCopyPad(out_gm[i], res, cp);
            out_q.FreeTensor(res);
        }
    }
}

// ═══════════════════════════════════════════════════════
//  scale_flat：bfloat16 版本
// ═══════════════════════════════════════════════════════
__aicore__ inline void scale_broadcast_1dim_half_fast(
    GM_ADDR input, GM_ADDR scale, GM_ADDR bias,
    GM_ADDR out, ScaleTilingData &tiling)
{
    using T = half;
    int block_index = GetBlockIdx();
    int block_dim   = GetBlockNum();

    constexpr int DATA_BLOCK_SIZE = 32 / sizeof(T);
    constexpr int AUX_BUF_BYTES = 8 << 10;
    constexpr int DATA_BUF_BYTES_BIAS = 80 << 10;
    constexpr int DATA_BUF_BYTES_NOBIAS = 88 << 10;

    GlobalTensor<T> input_gm, scale_gm, bias_gm, out_gm;
    input_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
    scale_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scale));
    out_gm  .SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
    if (tiling.has_bias)
        bias_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias));

    int mid_size = tiling.size;
    int high_size = tiling.high_size;
    int low_size = tiling.low_size;
    int low_bytes = low_size * (int)sizeof(T);
    int low_padded = (int)ceil_round(low_bytes, 32) / (int)sizeof(T);
    bool low_aligned = (low_bytes % 32 == 0);

    int data_buf_bytes = tiling.has_bias ?
        DATA_BUF_BYTES_BIAS : DATA_BUF_BYTES_NOBIAS;
    int data_buf_size = data_buf_bytes / (int)sizeof(T);
    int aux_buf_size = AUX_BUF_BYTES / (int)sizeof(T);

    TPipe pipe;
    TQue<TPosition::VECIN,  1> input_q, scale_q, bias_q;
    TQue<TPosition::VECOUT, 1> out_q;

    pipe.InitBuffer(input_q, 1, data_buf_bytes);
    pipe.InitBuffer(scale_q, 1, AUX_BUF_BYTES);
    pipe.InitBuffer(out_q,   1, data_buf_bytes);
    if (tiling.has_bias)
        pipe.InitBuffer(bias_q, 1, AUX_BUF_BYTES);

    int split_low_tile = 16 << 10;
    if (split_low_tile > data_buf_size)
        split_low_tile = data_buf_size;
    split_low_tile = (split_low_tile / DATA_BLOCK_SIZE) * DATA_BLOCK_SIZE;
    if (split_low_tile < DATA_BLOCK_SIZE)
        split_low_tile = DATA_BLOCK_SIZE;

    bool need_low_split = (low_padded > data_buf_size) ||
        (low_size > split_low_tile &&
         (long)high_size * mid_size < (long)block_dim * 8);

    if (need_low_split) {
        int low_tile = (low_padded > data_buf_size) ?
            (data_buf_size / DATA_BLOCK_SIZE) * DATA_BLOCK_SIZE :
            split_low_tile;
        if (low_tile < DATA_BLOCK_SIZE) low_tile = data_buf_size;
        int low_blocks = ceil_div(low_size, low_tile);
        long high_low = (long)high_size * low_blocks;
        long total_work = (long)mid_size * high_low;
        long work_start = total_work * block_index / block_dim;
        long work_end = total_work * (block_index + 1) / block_dim;
        if (work_start >= work_end) return;

        int prev_m = -1;
        LocalTensor<T> scale_local, bias_local;

        for (long w = work_start; w < work_end; w++) {
            int m = (int)(w / high_low);
            long rem = w - (long)m * high_low;
            int hi = (int)(rem / low_blocks);
            int lb = (int)(rem - (long)hi * low_blocks);
            int low_off = lb * low_tile;
            int len = min(low_size - low_off, low_tile);
            long base_off = ((long)hi * mid_size + m) * (long)low_size
                          + low_off;

            if (m != prev_m) {
                if (prev_m >= 0) {
                    scale_q.FreeTensor(scale_local);
                    if (tiling.has_bias) bias_q.FreeTensor(bias_local);
                }
                prev_m = m;

                LocalTensor<T> sc = scale_q.AllocTensor<T>();
                DataCopyExtParams cp_scale;
                cp_scale.blockLen = (uint32_t)sizeof(T);
                DataCopyPad(sc, scale_gm[m], cp_scale, {});
                scale_q.EnQue(sc);
                scale_local = scale_q.DeQue<T>();

                if (tiling.has_bias) {
                    LocalTensor<T> bi = bias_q.AllocTensor<T>();
                    DataCopyExtParams cp_bias;
                    cp_bias.blockLen = (uint32_t)sizeof(T);
                    DataCopyPad(bi, bias_gm[m], cp_bias, {});
                    bias_q.EnQue(bi);
                    bias_local = bias_q.DeQue<T>();
                }
            }

            DataCopyExtParams cp;
            cp.blockLen = (uint32_t)(len * sizeof(T));

            LocalTensor<T> in = input_q.AllocTensor<T>();
            DataCopyPad(in, input_gm[base_off], cp, {});
            input_q.EnQue(in);

            in = input_q.DeQue<T>();
            LocalTensor<T> res = out_q.AllocTensor<T>();
            Muls(res, in, scale_local.GetValue(0), len);
            if (tiling.has_bias)
                Adds(res, res, bias_local.GetValue(0), len);
            input_q.FreeTensor(in);
            out_q.EnQue(res);

            res = out_q.DeQue<T>();
            DataCopyPad(out_gm[base_off], res, cp);
            out_q.FreeTensor(res);
        }

        if (prev_m >= 0) {
            scale_q.FreeTensor(scale_local);
            if (tiling.has_bias) bias_q.FreeTensor(bias_local);
        }
        return;
    }

    int mid_tile_max = data_buf_size / low_padded;
    if (mid_tile_max < 1) mid_tile_max = 1;
    if (mid_tile_max > aux_buf_size) mid_tile_max = aux_buf_size;
    if (mid_tile_max > mid_size) mid_tile_max = mid_size;

    int mid_blocks = ceil_div(mid_size, mid_tile_max);
    long total_work = (long)high_size * mid_blocks;
    long work_start = total_work * block_index / block_dim;
    long work_end = total_work * (block_index + 1) / block_dim;
    if (work_start >= work_end) return;

    int prev_mk = -1;
    LocalTensor<T> scale_local, bias_local;

    for (long w = work_start; w < work_end; w++) {
        int mk = (int)(w / high_size);
        int hi = (int)(w - (long)mk * high_size);
        int k = mk * mid_tile_max;
        int mid_tile = min(mid_size - k, mid_tile_max);
        long base_off = ((long)hi * mid_size + k) * (long)low_size;

        if (mk != prev_mk) {
            if (prev_mk >= 0) {
                scale_q.FreeTensor(scale_local);
                if (tiling.has_bias) bias_q.FreeTensor(bias_local);
            }
            prev_mk = mk;

            LocalTensor<T> sc = scale_q.AllocTensor<T>();
            DataCopyExtParams cp_scale;
            cp_scale.blockLen = (uint32_t)(mid_tile * sizeof(T));
            DataCopyPad(sc, scale_gm[k], cp_scale, {});
            scale_q.EnQue(sc);
            scale_local = scale_q.DeQue<T>();

            if (tiling.has_bias) {
                LocalTensor<T> bi = bias_q.AllocTensor<T>();
                DataCopyExtParams cp_bias;
                cp_bias.blockLen = (uint32_t)(mid_tile * sizeof(T));
                DataCopyPad(bi, bias_gm[k], cp_bias, {});
                bias_q.EnQue(bi);
                bias_local = bias_q.DeQue<T>();
            }
        }

        LocalTensor<T> in = input_q.AllocTensor<T>();
        if (low_aligned) {
            DataCopyExtParams cp;
            cp.blockLen = (uint32_t)(mid_tile * low_bytes);
            DataCopyPad(in, input_gm[base_off], cp, {});
        } else {
            for (int m = 0; m < mid_tile; m++) {
                DataCopyExtParams cp;
                cp.blockLen = (uint32_t)low_bytes;
                DataCopyPad(in[m * low_padded],
                            input_gm[base_off + (long)m * low_size],
                            cp, {});
            }
        }
        input_q.EnQue(in);

        in = input_q.DeQue<T>();
        LocalTensor<T> res = out_q.AllocTensor<T>();
        for (int m = 0; m < mid_tile; m++) {
            int off = m * low_padded;
            Muls(res[off], in[off], scale_local.GetValue(m), low_size);
            if (tiling.has_bias)
                Adds(res[off], res[off], bias_local.GetValue(m), low_size);
        }
        input_q.FreeTensor(in);
        out_q.EnQue(res);

        res = out_q.DeQue<T>();
        if (low_aligned) {
            DataCopyExtParams cp;
            cp.blockLen = (uint32_t)(mid_tile * low_bytes);
            DataCopyPad(out_gm[base_off], res, cp);
        } else {
            for (int m = 0; m < mid_tile; m++) {
                DataCopyExtParams cp;
                cp.blockLen = (uint32_t)low_bytes;
                DataCopyPad(out_gm[base_off + (long)m * low_size],
                            res[m * low_padded], cp);
            }
        }
        out_q.FreeTensor(res);
    }

    if (prev_mk >= 0) {
        scale_q.FreeTensor(scale_local);
        if (tiling.has_bias) bias_q.FreeTensor(bias_local);
    }
}

template<typename T>
__aicore__ inline std::enable_if_t<std::is_same_v<T, bfloat16_t>>
scale_flat(GM_ADDR input, GM_ADDR scale, GM_ADDR bias,
           GM_ADDR out, ScaleTilingData &tiling)
{
    using U = float;
    int block_index = GetBlockIdx();
    int block_dim   = GetBlockNum();
    constexpr int DATA_BLOCK_SIZE = 32 / sizeof(T);

    int compute_blocks = ceil_div(tiling.size, DATA_BLOCK_SIZE);
    int compute_start  = (int)((long)compute_blocks * block_index
                               / block_dim * DATA_BLOCK_SIZE);
    int compute_end    = (int)min(
        (long)compute_blocks * (block_index + 1)
        / block_dim * DATA_BLOCK_SIZE,
        (long)tiling.size);

    GlobalTensor<T> input_gm, scale_gm, bias_gm, out_gm;
    input_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
    scale_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scale));
    out_gm  .SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
    if (tiling.has_bias)
        bias_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias));

    TPipe pipe;
    TQue<TPosition::VECIN,  1> input_q, scale_q, bias_q;
    TQue<TPosition::VECOUT, 1> out_q;

    int BUF_BYTES     = tiling.has_bias ? (24 << 10) : (32 << 10);
    int MAX_TILE_SIZE = BUF_BYTES / (int)sizeof(U);

    pipe.InitBuffer(input_q, 2, BUF_BYTES);
    pipe.InitBuffer(scale_q, 2, BUF_BYTES);
    pipe.InitBuffer(out_q,   2, BUF_BYTES);
    if (tiling.has_bias)
        pipe.InitBuffer(bias_q, 2, BUF_BYTES);

    uint16_t atomic_type, atomic_op;
    GetStoreAtomicConfig(atomic_type, atomic_op);

    int loop_start, loop_end, loop_step;
    if (atomic_type) {
        loop_start = compute_start
                     + ceil_round(compute_end - compute_start,
                                  MAX_TILE_SIZE)
                     - MAX_TILE_SIZE;
        loop_end   = compute_start - MAX_TILE_SIZE;
        loop_step  = -MAX_TILE_SIZE;
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_NONE,
                             AtomicOp::ATOMIC_SUM>();
    } else {
        loop_start = compute_start;
        loop_end   = compute_start
                     + ceil_round(compute_end - compute_start,
                                  MAX_TILE_SIZE);
        loop_step  = MAX_TILE_SIZE;
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_S32,
                             AtomicOp::ATOMIC_SUM>();
    }

    for (int i = loop_start; i != loop_end; i += loop_step) {
        int len = min(compute_end - i, MAX_TILE_SIZE);
        if (len <= 0) break;

        {
            DataCopyExtParams cp;
            cp.blockLen = (uint32_t)(len * sizeof(T));

            LocalTensor<U> in = input_q.AllocTensor<U>();
            DataCopyPad(in.ReinterpretCast<T>(), input_gm[i], cp, {});
            input_q.EnQue(in);

            LocalTensor<U> sc = scale_q.AllocTensor<U>();
            DataCopyPad(sc.ReinterpretCast<T>(), scale_gm[i], cp, {});
            scale_q.EnQue(sc);

            if (tiling.has_bias) {
                LocalTensor<U> bi = bias_q.AllocTensor<U>();
                DataCopyPad(bi.ReinterpretCast<T>(), bias_gm[i], cp, {});
                bias_q.EnQue(bi);
            }
        }
        {
            LocalTensor<U> in  = input_q.DeQue<U>();
            LocalTensor<U> sc  = scale_q.DeQue<U>();
            LocalTensor<U> res = out_q.AllocTensor<U>();
            Cast(res, in.ReinterpretCast<T>(),
                 RoundMode::CAST_NONE, len);
            Cast(in,  sc.ReinterpretCast<T>(),
                 RoundMode::CAST_NONE, len);
            Mul(res, res, in, len);
            if (tiling.has_bias) {
                LocalTensor<U> bi = bias_q.DeQue<U>();
                Cast(in, bi.ReinterpretCast<T>(),
                     RoundMode::CAST_NONE, len);
                Add(res, res, in, len);
                bias_q.FreeTensor(bi);
            }
            Cast(res.ReinterpretCast<T>(), res,
                 RoundMode::CAST_RINT, len);
            input_q.FreeTensor(in);
            scale_q.FreeTensor(sc);
            out_q.EnQue(res);
        }
        {
            LocalTensor<T> res = out_q.DeQue<T>();
            DataCopyExtParams cp;
            cp.blockLen = (uint32_t)(len * sizeof(T));
            DataCopyPad(out_gm[i], res, cp);
            out_q.FreeTensor(res);
        }
    }
}

// ═══════════════════════════════════════════════════════
//  do_broadcast_scale_bias：同时广播 scale 和 bias（stype==1）
//  ★ bias 始终写 workspace，不与 scale 共用 out 做 ping-pong
// ═══════════════════════════════════════════════════════
template<typename T>
__aicore__ inline void do_broadcast_scale_bias(
    GM_ADDR &scale, GM_ADDR &bias,
    GM_ADDR out, GM_ADDR workspace,
    ScaleTilingData &tiling)
{
    bool scale_need_bcast = (tiling.broadcast_offset[1] >= 0);
    bool bias_need_bcast  = (tiling.has_bias &&
                             tiling.broadcast_offset[2] >= 0);
    if (!scale_need_bcast && !bias_need_bcast) return;

    int block_index = GetBlockIdx();
    int block_dim   = GetBlockNum();
    constexpr int DATA_BLOCK_SIZE = 512 / sizeof(T);
    constexpr int MAX_TILE_SIZE   = (48 << 10) / sizeof(T);

    TPipe pipe;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> sc_que, bi_que;
    pipe.InitBuffer(sc_que, 2, MAX_TILE_SIZE * sizeof(T));
    if (bias_need_bcast)
        pipe.InitBuffer(bi_que, 2, MAX_TILE_SIZE * sizeof(T));

    int *ref_src_n = scale_need_bcast ? tiling.scale_n : tiling.bias_n;

    int broadcast_count = 0;
    for (int i = 0; i < 5; i++)
        if (ref_src_n[i] != tiling.out_n[i])
            broadcast_count++;

    GlobalTensor<T> sc_out_gm, sc_src_gm;
    GlobalTensor<T> bi_out_gm, bi_src_gm;

    // ★ scale: ping-pong,  bias: 始终写 workspace
    bool sc_out_to_out = (broadcast_count % 2 == 0);

    if (scale_need_bcast) {
        GM_ADDR sc_ws = workspace + tiling.broadcast_offset[1];
        sc_out_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(
            sc_out_to_out ? out : sc_ws));
        sc_src_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scale));
    }
    if (bias_need_bcast) {
        GM_ADDR bi_ws = workspace + tiling.broadcast_offset[2];
        bi_out_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bi_ws));
        bi_src_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias));
    }

    int high_size = ref_src_n[0] * ref_src_n[1] * ref_src_n[2]
                  * ref_src_n[3] * ref_src_n[4];
    int low_size  = 1;
    int rem_count = broadcast_count;

    for (int i = 0; i < 5; i++) {
        high_size /= ref_src_n[i];

        if (ref_src_n[i] != tiling.out_n[i]) {
            int mid_size = tiling.out_n[i];
            int high_from, high_to, low_from, low_to;

            if (low_size > MAX_TILE_SIZE) {
                high_from = 0;
                high_to   = high_size;
                int low_blocks = ceil_div(low_size, DATA_BLOCK_SIZE);
                low_from = low_blocks * block_index / block_dim
                           * DATA_BLOCK_SIZE;
                low_to   = min(
                    low_blocks * (block_index + 1) / block_dim
                    * DATA_BLOCK_SIZE, low_size);
            } else {
                high_from = high_size * block_index / block_dim;
                high_to   = high_size * (block_index + 1) / block_dim;
                low_from  = 0;
                low_to    = low_size;
            }

            for (int j = high_from; j < high_to; j++) {
                for (int k = low_from; k < low_to; k += MAX_TILE_SIZE) {
                    int len = min(low_to - k, MAX_TILE_SIZE);
                    DataCopyExtParams cp;
                    cp.blockLen = len * sizeof(T);

                    if (scale_need_bcast) {
                        LocalTensor<T> buf = sc_que.AllocTensor<T>();
                        DataCopyPad(buf,
                            sc_src_gm[low_size * j + k], cp, {});
                        sc_que.EnQue(buf);
                        LocalTensor<T> out_buf = sc_que.DeQue<T>();
                        for (int l = 0; l < mid_size; l++)
                            DataCopyPad(
                                sc_out_gm[(j * mid_size + l)
                                          * low_size + k],
                                out_buf, cp);
                        sc_que.FreeTensor(out_buf);
                    }
                    if (bias_need_bcast) {
                        LocalTensor<T> buf = bi_que.AllocTensor<T>();
                        DataCopyPad(buf,
                            bi_src_gm[low_size * j + k], cp, {});
                        bi_que.EnQue(buf);
                        LocalTensor<T> out_buf = bi_que.DeQue<T>();
                        for (int l = 0; l < mid_size; l++)
                            DataCopyPad(
                                bi_out_gm[(j * mid_size + l)
                                          * low_size + k],
                                out_buf, cp);
                        bi_que.FreeTensor(out_buf);
                    }
                }
            }

            --rem_count;
            if (scale_need_bcast) {
                GM_ADDR sc_ws = workspace + tiling.broadcast_offset[1];
                if (rem_count % 2 == 0) {
                    sc_out_gm.SetGlobalBuffer(
                        reinterpret_cast<__gm__ T*>(out));
                    sc_src_gm.SetGlobalBuffer(
                        reinterpret_cast<__gm__ T*>(sc_ws));
                } else {
                    sc_out_gm.SetGlobalBuffer(
                        reinterpret_cast<__gm__ T*>(sc_ws));
                    sc_src_gm.SetGlobalBuffer(
                        reinterpret_cast<__gm__ T*>(out));
                }
                sc_out_to_out = (rem_count % 2 == 0);
            }
            if (bias_need_bcast) {
                GM_ADDR bi_ws = workspace + tiling.broadcast_offset[2];
                bi_out_gm.SetGlobalBuffer(
                    reinterpret_cast<__gm__ T*>(bi_ws));
                bi_src_gm.SetGlobalBuffer(
                    reinterpret_cast<__gm__ T*>(bi_ws));
            }

            CrossCoreSetFlag<0x1C, PIPE_MTE3>(0);
            CrossCoreWaitFlag(0);
        }

        low_size *= tiling.out_n[i];
    }

    if (scale_need_bcast)
        scale = workspace + tiling.broadcast_offset[1];
    if (bias_need_bcast)
        bias  = workspace + tiling.broadcast_offset[2];

    pipe.Destroy();
}

// ═══════════════════════════════════════════════════════
//  scale_broadcast_1dim：float / half 版本（stype==2）
//  ★ 使用 DataCopyPad 1D 搬运，不要求 low_size 32B 对齐
// ═══════════════════════════════════════════════════════
__aicore__ inline void scale_broadcast_1dim_half_striped(
    GM_ADDR input, GM_ADDR scale, GM_ADDR bias,
    GM_ADDR out, ScaleTilingData &tiling)
{
    using T = half;
    int block_index = GetBlockIdx();
    int block_dim   = GetBlockNum();

    constexpr int LOW_STRIPE = 256;
    constexpr int MID_TILE_MAX = 64;
    constexpr int TILE_ELEMS = LOW_STRIPE * MID_TILE_MAX;
    constexpr int TILE_BYTES = TILE_ELEMS * (int)sizeof(T);
    constexpr int AUX_BYTES = MID_TILE_MAX * (int)sizeof(T);

    GlobalTensor<T> input_gm, scale_gm, bias_gm, out_gm;
    input_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
    scale_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scale));
    out_gm  .SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
    if (tiling.has_bias)
        bias_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias));

    int high_size = tiling.high_size;
    int mid_size  = tiling.size;
    int low_size  = tiling.low_size;

    int mid_blocks = ceil_div(mid_size, MID_TILE_MAX);
    int low_blocks = ceil_div(low_size, LOW_STRIPE);
    long high_low = (long)high_size * low_blocks;
    long total_work = (long)mid_blocks * high_low;
    long work_start = total_work * block_index / block_dim;
    long work_end = total_work * (block_index + 1) / block_dim;
    if (work_start >= work_end) return;

    TPipe pipe;
    TQue<TPosition::VECIN,   1> input_q, scale_q, bias_q;
    TQue<TPosition::VECOUT,  1> out_q;
    TQue<TPosition::VECCALC, 1> scale_bc_q, bias_bc_q;

    pipe.InitBuffer(input_q,    1, TILE_BYTES);
    pipe.InitBuffer(out_q,      1, TILE_BYTES);
    pipe.InitBuffer(scale_q,    1, AUX_BYTES);
    pipe.InitBuffer(scale_bc_q, 1, TILE_BYTES);
    if (tiling.has_bias) {
        pipe.InitBuffer(bias_q,    1, AUX_BYTES);
        pipe.InitBuffer(bias_bc_q, 1, TILE_BYTES);
    }

    int prev_mk = -1;
    int cur_mid_tile = 0;
    LocalTensor<T> scale_bc, bias_bc;

    for (long w = work_start; w < work_end; w++) {
        int mk = (int)(w / high_low);
        long rem = w - (long)mk * high_low;
        int hi = (int)(rem / low_blocks);
        int lb = (int)(rem - (long)hi * low_blocks);

        int m_start = mk * MID_TILE_MAX;
        int mid_tile = min(mid_size - m_start, MID_TILE_MAX);
        int low_start = lb * LOW_STRIPE;
        int valid_low = min(low_size - low_start, LOW_STRIPE);
        int count = mid_tile * LOW_STRIPE;

        if (mk != prev_mk) {
            if (prev_mk >= 0) {
                scale_bc_q.FreeTensor(scale_bc);
                if (tiling.has_bias) bias_bc_q.FreeTensor(bias_bc);
            }
            prev_mk = mk;
            cur_mid_tile = mid_tile;

            LocalTensor<T> sc = scale_q.AllocTensor<T>();
            DataCopyExtParams cp_scale;
            cp_scale.blockLen = (uint32_t)(mid_tile * sizeof(T));
            DataCopyPad(sc, scale_gm[m_start], cp_scale, {});
            scale_q.EnQue(sc);
            sc = scale_q.DeQue<T>();

            scale_bc = scale_bc_q.AllocTensor<T>();
            uint32_t dst[2] = {(uint32_t)cur_mid_tile,
                               (uint32_t)LOW_STRIPE};
            uint32_t src[2] = {(uint32_t)cur_mid_tile, 1};
            Broadcast<T, 2, 1>(scale_bc, sc, dst, src);
            scale_q.FreeTensor(sc);

            if (tiling.has_bias) {
                LocalTensor<T> bi = bias_q.AllocTensor<T>();
                DataCopyExtParams cp_bias;
                cp_bias.blockLen = (uint32_t)(mid_tile * sizeof(T));
                DataCopyPad(bi, bias_gm[m_start], cp_bias, {});
                bias_q.EnQue(bi);
                bi = bias_q.DeQue<T>();

                bias_bc = bias_bc_q.AllocTensor<T>();
                Broadcast<T, 2, 1>(bias_bc, bi, dst, src);
                bias_q.FreeTensor(bi);
            }
        }

        long base = ((long)hi * mid_size + m_start) * (long)low_size
                  + low_start;

        LocalTensor<T> in = input_q.AllocTensor<T>();
        for (int m = 0; m < mid_tile; m++) {
            DataCopyExtParams cp_in;
            cp_in.blockLen = (uint32_t)(valid_low * sizeof(T));
            DataCopyPad(in[m * LOW_STRIPE],
                        input_gm[base + (long)m * low_size],
                        cp_in, {});
        }
        input_q.EnQue(in);

        in = input_q.DeQue<T>();
        LocalTensor<T> res = out_q.AllocTensor<T>();
        Mul(res, in, scale_bc, count);
        if (tiling.has_bias)
            Add(res, res, bias_bc, count);
        input_q.FreeTensor(in);
        out_q.EnQue(res);

        res = out_q.DeQue<T>();
        for (int m = 0; m < mid_tile; m++) {
            DataCopyExtParams cp_out;
            cp_out.blockLen = (uint32_t)(valid_low * sizeof(T));
            DataCopyPad(out_gm[base + (long)m * low_size],
                        res[m * LOW_STRIPE], cp_out);
        }
        out_q.FreeTensor(res);
    }

    if (prev_mk >= 0) {
        scale_bc_q.FreeTensor(scale_bc);
        if (tiling.has_bias) bias_bc_q.FreeTensor(bias_bc);
    }
}

__aicore__ inline void scale_broadcast_1dim_half_fused(
    GM_ADDR input, GM_ADDR scale, GM_ADDR bias,
    GM_ADDR out, ScaleTilingData &tiling)
{
    using T = half;
    int block_index = GetBlockIdx();
    int block_dim   = GetBlockNum();

    constexpr int DATA_BYTES = 36 << 10;
    constexpr int AUX_BYTES  = 8 << 10;
    constexpr int DATA_ELEMS = DATA_BYTES / (int)sizeof(T);

    GlobalTensor<T> input_gm, scale_gm, bias_gm, out_gm;
    input_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
    scale_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scale));
    bias_gm .SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias));
    out_gm  .SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));

    int high_size = tiling.high_size;
    int mid_size  = tiling.size;
    int low_size  = tiling.low_size;
    int low_bytes = low_size * (int)sizeof(T);
    bool low_aligned = (low_bytes % 32 == 0);
    int row_stride = low_aligned ? low_size :
        (int)ceil_round(low_bytes, 32) / (int)sizeof(T);

    int mid_tile_max = DATA_ELEMS / row_stride;
    if (mid_tile_max < 1) mid_tile_max = 1;
    if (mid_tile_max > mid_size) mid_tile_max = mid_size;

    int mid_blocks = ceil_div(mid_size, mid_tile_max);
    if ((long)high_size * mid_blocks < block_dim) {
        int target_mid_blocks = ceil_div(block_dim, high_size);
        if (target_mid_blocks > mid_blocks) {
            int smaller_mid_tile = ceil_div(mid_size, target_mid_blocks);
            if (smaller_mid_tile < 1) smaller_mid_tile = 1;
            if (smaller_mid_tile < mid_tile_max) {
                mid_tile_max = smaller_mid_tile;
                mid_blocks = ceil_div(mid_size, mid_tile_max);
            }
        }
    }
    long total_work = (long)high_size * mid_blocks;
    long work_start = total_work * block_index / block_dim;
    long work_end = total_work * (block_index + 1) / block_dim;
    if (work_start >= work_end) return;

    TPipe pipe;
    TQue<TPosition::VECIN,   1> input_q, scale_q, bias_q;
    TQue<TPosition::VECOUT,  1> out_q;
    TQue<TPosition::VECCALC, 1> scale_bc_q, bias_bc_q;

    pipe.InitBuffer(input_q,    1, DATA_BYTES);
    pipe.InitBuffer(out_q,      1, DATA_BYTES);
    pipe.InitBuffer(scale_bc_q, 1, DATA_BYTES);
    pipe.InitBuffer(bias_bc_q,  1, DATA_BYTES);
    pipe.InitBuffer(scale_q,    1, AUX_BYTES);
    pipe.InitBuffer(bias_q,     1, AUX_BYTES);

    int prev_mk = -1;
    LocalTensor<T> scale_bc, bias_bc;

    for (long w = work_start; w < work_end; w++) {
        int mk = (int)(w / high_size);
        int hi = (int)(w - (long)mk * high_size);
        int m_start = mk * mid_tile_max;
        int mid_tile = min(mid_size - m_start, mid_tile_max);
        int count = mid_tile * row_stride;
        long base = ((long)hi * mid_size + m_start) * (long)low_size;

        if (mk != prev_mk) {
            if (prev_mk >= 0) {
                scale_bc_q.FreeTensor(scale_bc);
                bias_bc_q.FreeTensor(bias_bc);
            }
            prev_mk = mk;

            LocalTensor<T> sc = scale_q.AllocTensor<T>();
            DataCopyExtParams cp_scale;
            cp_scale.blockLen = (uint32_t)(mid_tile * sizeof(T));
            DataCopyPad(sc, scale_gm[m_start], cp_scale, {});
            scale_q.EnQue(sc);
            sc = scale_q.DeQue<T>();

            scale_bc = scale_bc_q.AllocTensor<T>();
            uint32_t dst[2] = {(uint32_t)mid_tile, (uint32_t)row_stride};
            uint32_t src[2] = {(uint32_t)mid_tile, 1};
            Broadcast<T, 2, 1>(scale_bc, sc, dst, src);
            scale_q.FreeTensor(sc);

            LocalTensor<T> bi = bias_q.AllocTensor<T>();
            DataCopyExtParams cp_bias;
            cp_bias.blockLen = (uint32_t)(mid_tile * sizeof(T));
            DataCopyPad(bi, bias_gm[m_start], cp_bias, {});
            bias_q.EnQue(bi);
            bi = bias_q.DeQue<T>();

            bias_bc = bias_bc_q.AllocTensor<T>();
            Broadcast<T, 2, 1>(bias_bc, bi, dst, src);
            bias_q.FreeTensor(bi);
        }

        LocalTensor<T> in = input_q.AllocTensor<T>();
        if (low_aligned) {
            DataCopyExtParams cp;
            cp.blockLen = (uint32_t)(mid_tile * low_bytes);
            DataCopyPad(in, input_gm[base], cp, {});
        } else {
            for (int m = 0; m < mid_tile; m++) {
                DataCopyExtParams cp;
                cp.blockLen = (uint32_t)low_bytes;
                DataCopyPad(in[m * row_stride],
                            input_gm[base + (long)m * low_size],
                            cp, {});
            }
        }
        input_q.EnQue(in);

        in = input_q.DeQue<T>();
        LocalTensor<T> res = out_q.AllocTensor<T>();
        Adds(res, in, (T)0, count);
        FusedMulAdd(res, scale_bc, bias_bc, count);
        input_q.FreeTensor(in);
        out_q.EnQue(res);

        res = out_q.DeQue<T>();
        if (low_aligned) {
            DataCopyExtParams cp;
            cp.blockLen = (uint32_t)(mid_tile * low_bytes);
            DataCopyPad(out_gm[base], res, cp);
        } else {
            for (int m = 0; m < mid_tile; m++) {
                DataCopyExtParams cp;
                cp.blockLen = (uint32_t)low_bytes;
                DataCopyPad(out_gm[base + (long)m * low_size],
                            res[m * row_stride], cp);
            }
        }
        out_q.FreeTensor(res);
    }

    if (prev_mk >= 0) {
        scale_bc_q.FreeTensor(scale_bc);
        bias_bc_q.FreeTensor(bias_bc);
    }
}

__aicore__ inline void scale_broadcast_1dim_half_contiguous(
    GM_ADDR input, GM_ADDR scale, GM_ADDR bias,
    GM_ADDR out, ScaleTilingData &tiling)
{
    using T = half;
    int block_index = GetBlockIdx();
    int block_dim   = GetBlockNum();

    GlobalTensor<T> input_gm, scale_gm, bias_gm, out_gm;
    input_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
    scale_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scale));
    bias_gm .SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias));
    out_gm  .SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));

    int high_size = tiling.high_size;
    int mid_size  = tiling.size;
    int low_size  = tiling.low_size;
    int low_bytes = low_size * (int)sizeof(T);

    constexpr int BUF_BYTES = 24 << 10;
    constexpr int MAX_TILE_SIZE = BUF_BYTES / (int)sizeof(T);

    int mid_tile_max = MAX_TILE_SIZE / low_size;
    if (mid_tile_max < 1) mid_tile_max = 1;
    if (mid_tile_max > mid_size) mid_tile_max = mid_size;

    int mid_blocks = ceil_div(mid_size, mid_tile_max);
    long total_work = (long)high_size * mid_blocks;
    long work_start = total_work * block_index / block_dim;
    long work_end = total_work * (block_index + 1) / block_dim;
    if (work_start >= work_end) return;

    TPipe pipe;
    TQue<TPosition::VECIN,  1> input_q, scale_q, bias_q;
    TQue<TPosition::VECOUT, 1> out_q;

    pipe.InitBuffer(input_q, 2, BUF_BYTES);
    pipe.InitBuffer(scale_q, 2, BUF_BYTES);
    pipe.InitBuffer(bias_q,  2, BUF_BYTES);
    pipe.InitBuffer(out_q,   2, BUF_BYTES);

    int prev_mk = -1;
    LocalTensor<T> scale_local, bias_local;

    for (long w = work_start; w < work_end; w++) {
        int mk = (int)(w / high_size);
        int hi = (int)(w - (long)mk * high_size);
        int k = mk * mid_tile_max;
        int mid_tile = min(mid_size - k, mid_tile_max);
        long base = ((long)hi * mid_size + k) * (long)low_size;

        if (mk != prev_mk) {
            if (prev_mk >= 0) {
                scale_q.FreeTensor(scale_local);
                bias_q.FreeTensor(bias_local);
            }
            prev_mk = mk;

            LocalTensor<T> sc = scale_q.AllocTensor<T>();
            DataCopyExtParams cp_scale;
            cp_scale.blockLen = (uint32_t)(mid_tile * sizeof(T));
            DataCopyPad(sc, scale_gm[k], cp_scale, {});
            scale_q.EnQue(sc);
            scale_local = scale_q.DeQue<T>();

            LocalTensor<T> bi = bias_q.AllocTensor<T>();
            DataCopyExtParams cp_bias;
            cp_bias.blockLen = (uint32_t)(mid_tile * sizeof(T));
            DataCopyPad(bi, bias_gm[k], cp_bias, {});
            bias_q.EnQue(bi);
            bias_local = bias_q.DeQue<T>();
        }

        int len = mid_tile * low_size;
        DataCopyExtParams cp;
        cp.blockLen = (uint32_t)(len * sizeof(T));

        LocalTensor<T> in = input_q.AllocTensor<T>();
        DataCopyPad(in, input_gm[base], cp, {});
        input_q.EnQue(in);

        in = input_q.DeQue<T>();
        LocalTensor<T> res = out_q.AllocTensor<T>();
        for (int m = 0; m < mid_tile; m++) {
            int off = m * low_size;
            Muls(res[off], in[off], scale_local.GetValue(m), low_size);
            Adds(res[off], res[off], bias_local.GetValue(m), low_size);
        }
        input_q.FreeTensor(in);
        out_q.EnQue(res);

        res = out_q.DeQue<T>();
        DataCopyPad(out_gm[base], res, cp);
        out_q.FreeTensor(res);
    }

    if (prev_mk >= 0) {
        scale_q.FreeTensor(scale_local);
        bias_q.FreeTensor(bias_local);
    }
}

__aicore__ inline void scale_broadcast_1dim_half_packed_nobias(
    GM_ADDR input, GM_ADDR scale, GM_ADDR out, ScaleTilingData &tiling)
{
    using T = half;
    int block_index = GetBlockIdx();
    int block_dim   = GetBlockNum();

    constexpr int DATA_BYTES = 56 << 10;
    constexpr int AUX_BYTES  = 8 << 10;
    constexpr int DATA_ELEMS = DATA_BYTES / (int)sizeof(T);

    GlobalTensor<T> input_gm, scale_gm, out_gm;
    input_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
    scale_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scale));
    out_gm  .SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));

    int high_size = tiling.high_size;
    int mid_size  = tiling.size;
    int low_size  = tiling.low_size;

    int mid_tile_max = DATA_ELEMS / low_size;
    if (mid_tile_max < 1) mid_tile_max = 1;
    if (mid_tile_max > mid_size) mid_tile_max = mid_size;

    int mid_blocks = ceil_div(mid_size, mid_tile_max);
    long total_work = (long)high_size * mid_blocks;
    long work_start = total_work * block_index / block_dim;
    long work_end = total_work * (block_index + 1) / block_dim;
    if (work_start >= work_end) return;

    TPipe pipe;
    TQue<TPosition::VECIN,   1> input_q, scale_q;
    TQue<TPosition::VECOUT,  1> out_q;
    TQue<TPosition::VECCALC, 1> scale_bc_q;

    pipe.InitBuffer(input_q,    1, DATA_BYTES);
    pipe.InitBuffer(out_q,      1, DATA_BYTES);
    pipe.InitBuffer(scale_bc_q, 1, DATA_BYTES);
    pipe.InitBuffer(scale_q,    1, AUX_BYTES);

    int prev_mk = -1;
    int prev_mid_tile = 0;
    LocalTensor<T> scale_bc;

    for (long w = work_start; w < work_end; w++) {
        int mk = (int)(w / high_size);
        int hi = (int)(w - (long)mk * high_size);
        int m_start = mk * mid_tile_max;
        int mid_tile = min(mid_size - m_start, mid_tile_max);
        int len = mid_tile * low_size;
        long base = ((long)hi * mid_size + m_start) * (long)low_size;

        if (mk != prev_mk) {
            if (prev_mk >= 0)
                scale_bc_q.FreeTensor(scale_bc);
            prev_mk = mk;
            prev_mid_tile = mid_tile;

            LocalTensor<T> sc = scale_q.AllocTensor<T>();
            DataCopyExtParams cp_scale;
            cp_scale.blockLen = (uint32_t)(mid_tile * sizeof(T));
            DataCopyPad(sc, scale_gm[m_start], cp_scale, {});
            scale_q.EnQue(sc);
            sc = scale_q.DeQue<T>();

            scale_bc = scale_bc_q.AllocTensor<T>();
            uint32_t dst[2] = {(uint32_t)prev_mid_tile,
                               (uint32_t)low_size};
            uint32_t src[2] = {(uint32_t)prev_mid_tile, 1};
            Broadcast<T, 2, 1>(scale_bc, sc, dst, src);
            scale_q.FreeTensor(sc);
        }

        DataCopyExtParams cp;
        cp.blockLen = (uint32_t)(len * sizeof(T));

        LocalTensor<T> in = input_q.AllocTensor<T>();
        DataCopyPad(in, input_gm[base], cp, {});
        input_q.EnQue(in);

        in = input_q.DeQue<T>();
        LocalTensor<T> res = out_q.AllocTensor<T>();
        Mul(res, in, scale_bc, len);
        input_q.FreeTensor(in);
        out_q.EnQue(res);

        res = out_q.DeQue<T>();
        DataCopyPad(out_gm[base], res, cp);
        out_q.FreeTensor(res);
    }

    if (prev_mk >= 0)
        scale_bc_q.FreeTensor(scale_bc);
}

__aicore__ inline void scale_broadcast_1dim_half_packed_bias(
    GM_ADDR input, GM_ADDR scale, GM_ADDR bias,
    GM_ADDR out, ScaleTilingData &tiling,TPipe * pipeIn)
{
    using T = half;
    int block_index = GetBlockIdx();
    int block_dim   = GetBlockNum();

    constexpr int DATA_BYTES = 38 << 10;
    constexpr int AUX_BYTES  = 8 << 10;
    constexpr int DATA_ELEMS = DATA_BYTES / (int)sizeof(T);
    constexpr int AUX_ELEMS  = AUX_BYTES / (int)sizeof(T);

    GlobalTensor<T> input_gm, scale_gm, bias_gm, out_gm;
    input_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
    scale_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scale));
    bias_gm .SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias));
    out_gm  .SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));

    int high_size = tiling.high_size;
    int mid_size  = tiling.size;
    int low_size  = tiling.low_size;

    int mid_tile_max = min(DATA_ELEMS / low_size, AUX_ELEMS);
    mid_tile_max = (mid_tile_max / 16) * 16;
    if (mid_tile_max < 16) return;
    if (mid_tile_max > mid_size) mid_tile_max = mid_size;

    int mid_blocks = ceil_div(mid_size, mid_tile_max);
    long total_work = (long)high_size * mid_blocks;
    long work_start = total_work * block_index / block_dim;
    long work_end = total_work * (block_index + 1) / block_dim;
    if (work_start >= work_end) return;

    TPipe * pipe;
    TQue<TPosition::VECIN,   1> input_q, scale_q, bias_q;
    TQue<TPosition::VECOUT,  1> out_q;
    TQue<TPosition::VECCALC, 1> scale_bc_q, bias_bc_q;

    pipeIn->InitBuffer(input_q,    1, DATA_BYTES);
    pipeIn->InitBuffer(out_q,      1, DATA_BYTES);
    pipeIn->InitBuffer(scale_bc_q, 1, DATA_BYTES);
    pipeIn->InitBuffer(bias_bc_q,  1, DATA_BYTES);
    pipeIn->InitBuffer(scale_q,    1, AUX_BYTES);
    pipeIn->InitBuffer(bias_q,     1, AUX_BYTES);

    int prev_mk = -1;
    int prev_mid_tile = 0;
    LocalTensor<T> scale_bc, bias_bc;

    for (long w = work_start; w < work_end; w++) {
        int mk = (int)(w / high_size);
        int hi = (int)(w - (long)mk * high_size);
        int m_start = mk * mid_tile_max;
        int mid_tile = min(mid_size - m_start, mid_tile_max);
        int len = mid_tile * low_size;
        long base = ((long)hi * mid_size + m_start) * (long)low_size;

        if (mk != prev_mk) {
            if (prev_mk >= 0) {
                scale_bc_q.FreeTensor(scale_bc);
                bias_bc_q.FreeTensor(bias_bc);
            }
            prev_mk = mk;
            prev_mid_tile = mid_tile;
            int mid_tile_align = (int)ceil_round(mid_tile, 16);

            LocalTensor<T> sc = scale_q.AllocTensor<T>();
            DataCopyExtParams cp_scale{
                1, (uint32_t)(mid_tile * sizeof(T)), 0, 0, 0};
            DataCopyPad(sc, scale_gm[m_start], cp_scale, {});
            scale_q.EnQue(sc);
            sc = scale_q.DeQue<T>();

            LocalTensor<T> bi = bias_q.AllocTensor<T>();
            DataCopyExtParams cp_bias{
                1, (uint32_t)(mid_tile * sizeof(T)), 0, 0, 0};
            DataCopyPad(bi, bias_gm[m_start], cp_bias, {});
            bias_q.EnQue(bi);
            bi = bias_q.DeQue<T>();

            scale_bc = scale_bc_q.AllocTensor<T>();
            bias_bc = bias_bc_q.AllocTensor<T>();
            uint32_t dst[2] = {(uint32_t)mid_tile_align,
                               (uint32_t)low_size};
            uint32_t src[2] = {(uint32_t)mid_tile_align, 1};
            Broadcast<T, 2, 1>(scale_bc, sc, dst, src);
            Broadcast<T, 2, 1>(bias_bc, bi, dst, src);
            scale_q.FreeTensor(sc);
            bias_q.FreeTensor(bi);
        }

        DataCopyExtParams cp;
        cp.blockLen = (uint32_t)(len * sizeof(T));

        LocalTensor<T> in = input_q.AllocTensor<T>();
        DataCopyPad(in, input_gm[base], cp, {});
        input_q.EnQue(in);

        in = input_q.DeQue<T>();
        LocalTensor<T> res = out_q.AllocTensor<T>();
        Mul(res, in, scale_bc, len);
        Add(res, res, bias_bc, len);
        input_q.FreeTensor(in);
        out_q.EnQue(res);

        res = out_q.DeQue<T>();
        DataCopyPad(out_gm[base], res, cp);
        out_q.FreeTensor(res);
    }

    if (prev_mk >= 0) {
        scale_bc_q.FreeTensor(scale_bc);
        bias_bc_q.FreeTensor(bias_bc);
    }
}

template<typename T>
__aicore__ inline std::enable_if_t<is_one_of_v<T, float, half>>
scale_broadcast_1dim(GM_ADDR input, GM_ADDR scale, GM_ADDR bias,
                     GM_ADDR out, ScaleTilingData &tiling)
{
    int block_index = GetBlockIdx();
    int block_dim   = GetBlockNum();

    GlobalTensor<T> input_gm, scale_gm, bias_gm, out_gm;
    input_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
    scale_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scale));
    out_gm  .SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
    if (tiling.has_bias)
        bias_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias));

    int mid_size = tiling.size;
    int low_size = tiling.low_size;
    int low_bytes = low_size * (int)sizeof(T);
    bool low_aligned = (low_bytes % 32 == 0);
    int low_padded = (int)ceil_round(low_bytes, 32) / (int)sizeof(T);

    int BUF_BYTES     = tiling.has_bias ? (24 << 10) : (32 << 10);
    int MAX_TILE_SIZE = BUF_BYTES / (int)sizeof(T);
    int mid_tile_max  = MAX_TILE_SIZE / low_padded;
    if (mid_tile_max < 1)  mid_tile_max = 1;
    if (mid_tile_max > mid_size) mid_tile_max = mid_size;

    int high_size   = (int)tiling.high_size;
    int mid_blocks  = ceil_div(mid_size, mid_tile_max);

    // ── 2D 划分：总任务 = high_size × mid_blocks，均分到所有核 ──
    long total_work = (long)high_size * mid_blocks;
    long work_start = total_work * block_index / block_dim;
    long work_end   = total_work * (block_index + 1) / block_dim;
    if (work_start >= work_end) return;

    TPipe pipe;
    TQue<TPosition::VECIN,  1> input_q, scale_q, bias_q;
    TQue<TPosition::VECOUT, 1> out_q;

    pipe.InitBuffer(input_q, 2, BUF_BYTES);
    pipe.InitBuffer(scale_q, 2, BUF_BYTES);
    pipe.InitBuffer(out_q,   2, BUF_BYTES);
    if (tiling.has_bias)
        pipe.InitBuffer(bias_q, 2, BUF_BYTES);

    if (low_padded > MAX_TILE_SIZE) {
        constexpr int DATA_BLOCK_SIZE = 32 / sizeof(T);
        int low_tile = (MAX_TILE_SIZE / DATA_BLOCK_SIZE) * DATA_BLOCK_SIZE;
        if (low_tile < DATA_BLOCK_SIZE) low_tile = MAX_TILE_SIZE;
        int low_blocks = ceil_div(low_size, low_tile);
        long high_low = (long)high_size * low_blocks;
        long total_work = (long)mid_size * high_low;
        long work_start = total_work * block_index / block_dim;
        long work_end = total_work * (block_index + 1) / block_dim;
        if (work_start >= work_end) return;

        int prev_m = -1;
        LocalTensor<T> scale_local, bias_local;

        for (long w = work_start; w < work_end; w++) {
            int m = (int)(w / high_low);
            long rem = w - (long)m * high_low;
            int hi = (int)(rem / low_blocks);
            int lb = (int)(rem - (long)hi * low_blocks);
            int low_off = lb * low_tile;
            int len = min(low_size - low_off, low_tile);
            long base = ((long)hi * mid_size + m) * (long)low_size
                      + low_off;

            if (m != prev_m) {
                if (prev_m >= 0) {
                    scale_q.FreeTensor(scale_local);
                    if (tiling.has_bias) bias_q.FreeTensor(bias_local);
                }
                prev_m = m;

                LocalTensor<T> sc = scale_q.AllocTensor<T>();
                DataCopyExtParams cp_scale;
                cp_scale.blockLen = (uint32_t)sizeof(T);
                DataCopyPad(sc, scale_gm[m], cp_scale, {});
                scale_q.EnQue(sc);
                scale_local = scale_q.DeQue<T>();

                if (tiling.has_bias) {
                    LocalTensor<T> bi = bias_q.AllocTensor<T>();
                    DataCopyExtParams cp_bias;
                    cp_bias.blockLen = (uint32_t)sizeof(T);
                    DataCopyPad(bi, bias_gm[m], cp_bias, {});
                    bias_q.EnQue(bi);
                    bias_local = bias_q.DeQue<T>();
                }
            }

            DataCopyExtParams cp;
            cp.blockLen = (uint32_t)(len * sizeof(T));

            LocalTensor<T> in = input_q.AllocTensor<T>();
            DataCopyPad(in, input_gm[base], cp, {});
            input_q.EnQue(in);

            in = input_q.DeQue<T>();
            LocalTensor<T> res = out_q.AllocTensor<T>();
            Muls(res, in, scale_local.GetValue(0), len);
            if (tiling.has_bias)
                Adds(res, res, bias_local.GetValue(0), len);
            input_q.FreeTensor(in);
            out_q.EnQue(res);

            res = out_q.DeQue<T>();
            DataCopyPad(out_gm[base], res, cp);
            out_q.FreeTensor(res);
        }

        if (prev_m >= 0) {
            scale_q.FreeTensor(scale_local);
            if (tiling.has_bias) bias_q.FreeTensor(bias_local);
        }
        return;
    }

    int prev_mk = -1;
    LocalTensor<T> scale_local, bias_local;

    for (long w = work_start; w < work_end; w++) {
        int mk = (int)(w / high_size);      // mid block 索引
        int hi = (int)(w % high_size);      // high 索引

        int k  = mk * mid_tile_max;
        int mid_tile = min(mid_size - k, mid_tile_max);
        long base_off = (long)hi * (long)mid_size * low_size
                      + (long)k * low_size;
        DataCopyExtParams cp;
        cp.blockLen = (uint32_t)(mid_tile * low_bytes);

        // ── mid block 变了 → 加载新 scale（跨 high 复用）──
        if (mk != prev_mk) {
            if (prev_mk >= 0) {
                scale_q.FreeTensor(scale_local);
                if (tiling.has_bias) bias_q.FreeTensor(bias_local);
            }
            prev_mk = mk;

            {
                LocalTensor<T> sc = scale_q.AllocTensor<T>();
                DataCopyExtParams cp_scale;
                cp_scale.blockLen = (uint32_t)(mid_tile * sizeof(T));
                DataCopyPad(sc, scale_gm[k], cp_scale, {});
                scale_q.EnQue(sc);
                scale_local = scale_q.DeQue<T>();
            }
            if (tiling.has_bias) {
                LocalTensor<T> bi = bias_q.AllocTensor<T>();
                DataCopyExtParams cp_bias;
                cp_bias.blockLen = (uint32_t)(mid_tile * sizeof(T));
                DataCopyPad(bi, bias_gm[k], cp_bias, {});
                bias_q.EnQue(bi);
                bias_local = bias_q.DeQue<T>();
            }
        }

        // ── load input ──
        {
            LocalTensor<T> in = input_q.AllocTensor<T>();
            if (low_aligned) {
                DataCopyPad(in, input_gm[base_off], cp, {});
            } else {
                DataCopyExtParams row_cp{
                    (uint16_t)mid_tile, (uint32_t)low_bytes, 0, 0, 0};
                DataCopyPad(in, input_gm[base_off], row_cp, {});
            }
            input_q.EnQue(in);
        }

        // ── compute ──
        {
            LocalTensor<T> in  = input_q.DeQue<T>();
            LocalTensor<T> res = out_q.AllocTensor<T>();

            int row_stride = low_padded;
            if (tiling.has_bias) {
                for (int m = 0; m < mid_tile; m++) {
                    int off = m * row_stride;
                    Muls(res[off], in[off], scale_local.GetValue(m), low_size);
                    Adds(res[off], res[off], bias_local.GetValue(m), low_size);
                }
            } else {
                for (int m = 0; m < mid_tile; m++) {
                    int off = m * row_stride;
                    Muls(res[off], in[off], scale_local.GetValue(m), low_size);
                }
            }
            input_q.FreeTensor(in);
            out_q.EnQue(res);
        }

        // ── store output ──
        {
            LocalTensor<T> res = out_q.DeQue<T>();
            if (low_aligned) {
                DataCopyPad(out_gm[base_off], res, cp);
            } else {
                DataCopyExtParams row_cp{
                    (uint16_t)mid_tile, (uint32_t)low_bytes, 0, 0, 0};
                DataCopyPad(out_gm[base_off], res, row_cp);
            }
            out_q.FreeTensor(res);
        }
    }

    // if (prev_mk >= 0) {
    // 最后释放
    if (prev_mk >= 0) {
        scale_q.FreeTensor(scale_local);
        if (tiling.has_bias) bias_q.FreeTensor(bias_local);
    }
}

// ═══════════════════════════════════════════════════════
//  scale_broadcast_1dim：bfloat16 版本（stype==2）
//  ★ 使用 DataCopyPad 1D 搬运，不要求 low_size 32B 对齐
// ═══════════════════════════════════════════════════════
__aicore__ inline void scale_broadcast_1dim_bf16_packed(
    GM_ADDR input, GM_ADDR scale, GM_ADDR bias,
    GM_ADDR out, ScaleTilingData &tiling)
{
    using T = bfloat16_t;
    using U = float;
    int block_index = GetBlockIdx();
    int block_dim   = GetBlockNum();

    constexpr int F32_BYTES = 28 << 10;
    constexpr int T_BYTES   = F32_BYTES / 2;
    constexpr int AUX_BYTES = 8 << 10;
    constexpr int F32_ELEMS = F32_BYTES / (int)sizeof(U);
    constexpr int AUX_ELEMS = AUX_BYTES / (int)sizeof(U);

    GlobalTensor<T> input_gm, scale_gm, bias_gm, out_gm;
    input_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
    scale_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scale));
    out_gm  .SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
    if (tiling.has_bias)
        bias_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias));

    int high_size = tiling.high_size;
    int mid_size  = tiling.size;
    int low_size  = tiling.low_size;

    int mid_tile_max = min(F32_ELEMS / low_size, AUX_ELEMS);
    mid_tile_max = (mid_tile_max / 8) * 8;
    if (mid_tile_max < 8) return;
    if (mid_tile_max > mid_size) mid_tile_max = mid_size;

    int mid_blocks = ceil_div(mid_size, mid_tile_max);
    long total_work = (long)high_size * mid_blocks;
    long work_start = total_work * block_index / block_dim;
    long work_end = total_work * (block_index + 1) / block_dim;
    if (work_start >= work_end) return;

    TPipe pipe;
    TQue<TPosition::VECIN,   1> raw_in_q, raw_scale_q, raw_bias_q;
    TQue<TPosition::VECIN,   1> in_q, scale_f32_q, bias_f32_q;
    TQue<TPosition::VECOUT,  1> out_q, raw_out_q;
    TQue<TPosition::VECCALC, 1> scale_bc_q, bias_bc_q;

    pipe.InitBuffer(raw_in_q,    1, T_BYTES);
    pipe.InitBuffer(raw_out_q,   1, T_BYTES);
    pipe.InitBuffer(in_q,        1, F32_BYTES);
    pipe.InitBuffer(out_q,       1, F32_BYTES);
    pipe.InitBuffer(scale_bc_q,  1, F32_BYTES);
    pipe.InitBuffer(raw_scale_q, 1, AUX_BYTES);
    pipe.InitBuffer(scale_f32_q, 1, AUX_BYTES);
    if (tiling.has_bias) {
        pipe.InitBuffer(bias_bc_q,  1, F32_BYTES);
        pipe.InitBuffer(raw_bias_q, 1, AUX_BYTES);
        pipe.InitBuffer(bias_f32_q, 1, AUX_BYTES);
    }

    int prev_mk = -1;
    LocalTensor<U> scale_bc, bias_bc;

    for (long w = work_start; w < work_end; w++) {
        int mk = (int)(w / high_size);
        int hi = (int)(w - (long)mk * high_size);
        int m_start = mk * mid_tile_max;
        int mid_tile = min(mid_size - m_start, mid_tile_max);
        int mid_tile_align = (int)ceil_round(mid_tile, 8);
        int len = mid_tile * low_size;
        long base = ((long)hi * mid_size + m_start) * (long)low_size;

        if (mk != prev_mk) {
            if (prev_mk >= 0) {
                scale_bc_q.FreeTensor(scale_bc);
                if (tiling.has_bias) bias_bc_q.FreeTensor(bias_bc);
            }
            prev_mk = mk;

            LocalTensor<T> sc_raw = raw_scale_q.AllocTensor<T>();
            DataCopyExtParams cp_scale{
                1, (uint32_t)(mid_tile * sizeof(T)), 0, 0, 0};
            DataCopyPad(sc_raw, scale_gm[m_start], cp_scale, {});
            raw_scale_q.EnQue(sc_raw);
            sc_raw = raw_scale_q.DeQue<T>();

            LocalTensor<U> sc = scale_f32_q.AllocTensor<U>();
            Cast(sc, sc_raw, RoundMode::CAST_NONE, mid_tile_align);
            raw_scale_q.FreeTensor(sc_raw);
            scale_f32_q.EnQue(sc);
            sc = scale_f32_q.DeQue<U>();

            scale_bc = scale_bc_q.AllocTensor<U>();
            uint32_t dst[2] = {(uint32_t)mid_tile_align,
                               (uint32_t)low_size};
            uint32_t src[2] = {(uint32_t)mid_tile_align, 1};
            Broadcast<U, 2, 1>(scale_bc, sc, dst, src);
            scale_f32_q.FreeTensor(sc);

            if (tiling.has_bias) {
                LocalTensor<T> bi_raw = raw_bias_q.AllocTensor<T>();
                DataCopyExtParams cp_bias{
                    1, (uint32_t)(mid_tile * sizeof(T)), 0, 0, 0};
                DataCopyPad(bi_raw, bias_gm[m_start], cp_bias, {});
                raw_bias_q.EnQue(bi_raw);
                bi_raw = raw_bias_q.DeQue<T>();

                LocalTensor<U> bi = bias_f32_q.AllocTensor<U>();
                Cast(bi, bi_raw, RoundMode::CAST_NONE, mid_tile_align);
                raw_bias_q.FreeTensor(bi_raw);
                bias_f32_q.EnQue(bi);
                bi = bias_f32_q.DeQue<U>();

                bias_bc = bias_bc_q.AllocTensor<U>();
                Broadcast<U, 2, 1>(bias_bc, bi, dst, src);
                bias_f32_q.FreeTensor(bi);
            }
        }

        DataCopyExtParams cp_in{
            1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        LocalTensor<T> raw_in = raw_in_q.AllocTensor<T>();
        DataCopyPad(raw_in, input_gm[base], cp_in, {});
        raw_in_q.EnQue(raw_in);
        raw_in = raw_in_q.DeQue<T>();

        LocalTensor<U> in_f32 = in_q.AllocTensor<U>();
        Cast(in_f32, raw_in, RoundMode::CAST_NONE, len);
        raw_in_q.FreeTensor(raw_in);
        in_q.EnQue(in_f32);

        in_f32 = in_q.DeQue<U>();
        LocalTensor<U> res = out_q.AllocTensor<U>();
        Mul(res, in_f32, scale_bc, len);
        if (tiling.has_bias)
            Add(res, res, bias_bc, len);
        in_q.FreeTensor(in_f32);
        out_q.EnQue(res);

        res = out_q.DeQue<U>();
        LocalTensor<T> raw_out = raw_out_q.AllocTensor<T>();
        Cast(raw_out, res, RoundMode::CAST_RINT, len);
        out_q.FreeTensor(res);
        raw_out_q.EnQue(raw_out);

        raw_out = raw_out_q.DeQue<T>();
        DataCopyPad(out_gm[base], raw_out, cp_in);
        raw_out_q.FreeTensor(raw_out);
    }

    if (prev_mk >= 0) {
        scale_bc_q.FreeTensor(scale_bc);
        if (tiling.has_bias) bias_bc_q.FreeTensor(bias_bc);
    }
}

template<typename T>
__aicore__ inline std::enable_if_t<std::is_same_v<T, bfloat16_t>>
scale_broadcast_1dim(GM_ADDR input, GM_ADDR scale, GM_ADDR bias,
                     GM_ADDR out, ScaleTilingData &tiling)
{
    using U = float;
    int block_index = GetBlockIdx();
    int block_dim   = GetBlockNum();

    GlobalTensor<T> input_gm, scale_gm, bias_gm, out_gm;
    input_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
    scale_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scale));
    out_gm  .SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
    if (tiling.has_bias)
        bias_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias));

    int mid_size = tiling.size;
    int low_size = tiling.low_size;
    // ★ 注意: 计算在 f32 下进行，对齐也按 f32
    bool low_aligned = (low_size * (int)sizeof(U) % 32 == 0);
    int  low_padded  = low_aligned ? low_size :
                       ((int)ceil_round(low_size * (int)sizeof(U), 32)
                        / (int)sizeof(U));

    long high_start = (long)tiling.high_size * block_index / block_dim;
    long high_end   = (long)tiling.high_size * (block_index + 1) / block_dim;

    constexpr int BUF_BYTES     = 16 << 10;
    constexpr int MAX_TILE_SIZE = BUF_BYTES / (int)sizeof(U);

    int mid_tile_max = MAX_TILE_SIZE / low_padded;
    if (mid_tile_max < 1)  mid_tile_max = 1;
    if (mid_tile_max > mid_size) mid_tile_max = mid_size;

    TPipe pipe;
    TQue<TPosition::VECIN,  1> input_q, tmp_q;
    TQue<TPosition::VECIN,  1> scale_q;
    TQue<TPosition::VECOUT, 1> out_q;
    TQue<TPosition::VECIN,  1> bias_q;

    pipe.InitBuffer(input_q, 2, BUF_BYTES);
    pipe.InitBuffer(tmp_q,   2, BUF_BYTES);
    pipe.InitBuffer(scale_q, 2, BUF_BYTES);
    pipe.InitBuffer(out_q,   2, BUF_BYTES);
    if (tiling.has_bias)
        pipe.InitBuffer(bias_q, 2, BUF_BYTES);

    for (long i = high_start; i < high_end; i++) {
        long base_off = i * (long)mid_size * low_size;

        for (int k = 0; k < mid_size; k += mid_tile_max) {
            int mid_tile = min(mid_size - k, mid_tile_max);
            int tile_len = mid_tile * low_size;

            {
                LocalTensor<T> tmp = tmp_q.AllocTensor<T>();
                DataCopyExtParams cp;
                cp.blockLen = (uint32_t)(mid_tile * sizeof(T));
                DataCopyPad(tmp, scale_gm[k], cp, {});
                tmp_q.EnQue(tmp);

                LocalTensor<T> tmp_sc = tmp_q.DeQue<T>();
                LocalTensor<U> sc     = scale_q.AllocTensor<U>();
                Cast(sc, tmp_sc, RoundMode::CAST_NONE, mid_tile);
                tmp_q.FreeTensor(tmp_sc);
                scale_q.EnQue(sc);
            }
            if (tiling.has_bias) {
                LocalTensor<T> tmp = tmp_q.AllocTensor<T>();
                DataCopyExtParams cp;
                cp.blockLen = (uint32_t)(mid_tile * sizeof(T));
                DataCopyPad(tmp, bias_gm[k], cp, {});
                tmp_q.EnQue(tmp);

                LocalTensor<T> tmp_bi = tmp_q.DeQue<T>();
                LocalTensor<U> bi     = bias_q.AllocTensor<U>();
                Cast(bi, tmp_bi, RoundMode::CAST_NONE, mid_tile);
                tmp_q.FreeTensor(tmp_bi);
                bias_q.EnQue(bi);
            }

            LocalTensor<U> scale_f32 = scale_q.DeQue<U>();
            LocalTensor<U> bias_f32;
            if (tiling.has_bias)
                bias_f32 = bias_q.DeQue<U>();

            {
                LocalTensor<T> tmp = tmp_q.AllocTensor<T>();
                if (low_aligned) {
                    DataCopyExtParams cp;
                    cp.blockLen = (uint32_t)(tile_len * sizeof(T));
                    DataCopyPad(tmp, input_gm[base_off + (long)k * low_size], cp, {});
                } else {
                    for (int m = 0; m < mid_tile; m++) {
                        DataCopyExtParams cp;
                        cp.blockLen = (uint32_t)(low_size * sizeof(T));
                        DataCopyPad(tmp[m * low_size],
                            input_gm[base_off + (long)(k + m) * low_size], cp, {});
                    }
                }
                tmp_q.EnQue(tmp);
            }
            {
                LocalTensor<T> tmp_in = tmp_q.DeQue<T>();
                LocalTensor<U> in_f32 = input_q.AllocTensor<U>();
                if (low_aligned) {
                    Cast(in_f32, tmp_in, RoundMode::CAST_NONE, tile_len);
                } else {
                    for (int m = 0; m < mid_tile; m++) {
                        Cast(in_f32[m * low_padded],
                             tmp_in[m * low_size],
                             RoundMode::CAST_NONE, low_size);
                    }
                }
                tmp_q.FreeTensor(tmp_in);

                LocalTensor<U> in  = input_q.DeQue<U>();
                LocalTensor<U> res = out_q.AllocTensor<U>();

                for (int m = 0; m < mid_tile; m++) {
                    Muls(res[m * low_padded], in[m * low_padded],
                         scale_f32.GetValue(m), low_size);
                    if (tiling.has_bias)
                        Adds(res[m * low_padded], res[m * low_padded],
                             bias_f32.GetValue(m), low_size);
                }
                input_q.FreeTensor(in);
                out_q.EnQue(res);
            }
            {
                LocalTensor<U> res = out_q.DeQue<U>();
                LocalTensor<T> tmp = tmp_q.AllocTensor<T>();
                if (low_aligned) {
                    Cast(tmp, res, RoundMode::CAST_RINT, tile_len);
                } else {
                    for (int m = 0; m < mid_tile; m++) {
                        Cast(tmp[m * low_size],
                             res[m * low_padded],
                             RoundMode::CAST_RINT, low_size);
                    }
                }
                out_q.FreeTensor(res);

                if (low_aligned) {
                    DataCopyExtParams cp;
                    cp.blockLen = (uint32_t)(tile_len * sizeof(T));
                    DataCopyPad(out_gm[base_off + (long)k * low_size], tmp, cp);
                } else {
                    for (int m = 0; m < mid_tile; m++) {
                        DataCopyExtParams cp;
                        cp.blockLen = (uint32_t)(low_size * sizeof(T));
                        DataCopyPad(out_gm[base_off + (long)(k + m) * low_size],
                                    tmp[m * low_size], cp);
                    }
                }
                tmp_q.FreeTensor(tmp);
            }

            scale_q.FreeTensor(scale_f32);
            if (tiling.has_bias)
                bias_q.FreeTensor(bias_f32);
        }
    }
}

// ═══════════════════════════════════════════════════════
//  kernel 入口
// ═══════════════════════════════════════════════════════
template<typename T>
__aicore__ inline void diag_zero_output(GM_ADDR out, ScaleTilingData &tiling)
{
    int block_index = GetBlockIdx();
    int block_dim   = GetBlockNum();
    long total = (long)tiling.high_size * (long)tiling.size
               * (long)tiling.low_size;
    if (total <= 0) return;

    GlobalTensor<T> out_gm;
    out_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));

    constexpr int DATA_BLOCK_SIZE = 32 / sizeof(T);
    int compute_blocks = ceil_div((int)total, DATA_BLOCK_SIZE);
    int compute_start = (int)((long)compute_blocks * block_index
                              / block_dim * DATA_BLOCK_SIZE);
    int compute_end = (int)min((long)compute_blocks * (block_index + 1)
                               / block_dim * DATA_BLOCK_SIZE, total);

    TPipe pipe;
    TQue<TPosition::VECOUT, 1> out_q;
    constexpr int BUF_BYTES = 16 << 10;
    constexpr int MAX_TILE_SIZE = BUF_BYTES / (int)sizeof(T);
    pipe.InitBuffer(out_q, 2, BUF_BYTES);

    for (int i = compute_start; i < compute_end; i += MAX_TILE_SIZE) {
        int len = min(compute_end - i, MAX_TILE_SIZE);
        if (len <= 0) break;

        LocalTensor<T> z = out_q.AllocTensor<T>();
        Duplicate(z, (T)0, len);
        out_q.EnQue(z);

        z = out_q.DeQue<T>();
        DataCopyExtParams cp;
        cp.blockLen = (uint32_t)(len * sizeof(T));
        DataCopyPad(out_gm[i], z, cp);
        out_q.FreeTensor(z);
    }
}

extern "C" __global__ __aicore__ void scale(
    GM_ADDR input, GM_ADDR scale, GM_ADDR bias,
    GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    // KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
   

   

    if  (TILING_KEY_IS(0)){

 
        GET_TILING_DATA(tiling_data, tiling);
        scale_flat_float_fast(input, scale, bias, out, tiling_data);
           }
    else if (TILING_KEY_IS(1)){
         GET_TILING_DATA(tiling_data, tiling);
         scale_flat<half>      (input, scale, bias, out, tiling_data);


    }
       
    else if (TILING_KEY_IS(27)){
        GET_TILING_DATA(tiling_data, tiling);
         scale_flat<bfloat16_t>(input, scale, bias, out, tiling_data);


    }
       
    else if (TILING_KEY_IS(13)){

        GET_TILING_DATA_WITH_STRUCT(ScaleTilingDataFloat, tiling_data, tiling);

        TilingParam paramList;
        TPipe pipe;
        paramList.total_length  = tiling_data.total_length;
        paramList.start_length  = tiling_data.start_length;
        paramList.end_length    = tiling_data.end_length;
        paramList.weight_length = tiling_data.weight_length;
        paramList.ALIGN_NUM     = tiling_data.ALIGN_NUM;
        paramList.tiling_size   = tiling_data.tiling_size;
        paramList.block_size    = tiling_data.block_size;
        paramList.core_size     = tiling_data.core_size;
        paramList.core_remain   = tiling_data.core_remain;
    // paramList.mode          = tiling_data.mode;
        //paramList.dim           = tiling_data.dim;
        paramList.has_bias      = tiling_data.has_bias;

        Scale<DTYPE_INPUT, DTYPE_SCALE, DTYPE_BIAS, DTYPE_OUTPUT> op;
        op.Init(input, scale, bias, out, paramList, &pipe);
        op.Process();




    }

    else if (TILING_KEY_IS(14)){

        GET_TILING_DATA_WITH_STRUCT(ScaleTilingDataFloatBroadCast, tiling_data, tiling);

        TilingParam paramList;
        TPipe pipe;
        paramList.total_length  = tiling_data.total_length;
        paramList.start_length  = tiling_data.start_length;
        paramList.end_length    = tiling_data.end_length;
        paramList.weight_length = tiling_data.weight_length;
        paramList.ALIGN_NUM     = tiling_data.ALIGN_NUM;
        paramList.tiling_size   = tiling_data.tiling_size;
        paramList.block_size    = tiling_data.block_size;
        paramList.core_size     = tiling_data.core_size;
        paramList.core_remain   = tiling_data.core_remain;
    // paramList.mode          = tiling_data.mode;
        paramList.dim           = tiling_data.dim;
        paramList.has_bias      = tiling_data.has_bias;

        for (int i = 0; i < 20; ++i) {
        paramList.shape[i]   = tiling_data.shape[i];
        paramList.reduce1[i] = tiling_data.reduce1[i];
        paramList.reduce2[i] = tiling_data.reduce2[i];
        paramList.reduce3[i] = tiling_data.reduce3[i];
    }

        Scale_Broadcast<DTYPE_INPUT, DTYPE_SCALE, DTYPE_BIAS, DTYPE_OUTPUT> op;
        op.Init(input, scale, bias, out, paramList, &pipe);
        op.Process();




    }

    else if (TILING_KEY_IS(100)) {
         KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
         GET_TILING_DATA(tiling_data, tiling);
        KERNEL_TASK_TYPE(100, KERNEL_TYPE_MIX_AIV_1_0);
        do_broadcast_scale_bias<float>(
            scale, bias, out, workspace, tiling_data);
        scale_flat_float_fast(input, scale, bias, out, tiling_data);
    }
    else if (TILING_KEY_IS(101)) {
         KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
         GET_TILING_DATA(tiling_data, tiling);
        KERNEL_TASK_TYPE(101, KERNEL_TYPE_MIX_AIV_1_0);
        do_broadcast_scale_bias<half>(
            scale, bias, out, workspace, tiling_data);
        scale_flat<half>(input, scale, bias, out, tiling_data);
    }
    else if (TILING_KEY_IS(127)) {
         KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
         GET_TILING_DATA(tiling_data, tiling);
        KERNEL_TASK_TYPE(127, KERNEL_TYPE_MIX_AIV_1_0);
        do_broadcast_scale_bias<bfloat16_t>(
            scale, bias, out, workspace, tiling_data);
        scale_flat<bfloat16_t>(input, scale, bias, out, tiling_data);
    }

    else if (TILING_KEY_IS(200)){

        GET_TILING_DATA(tiling_data, tiling);

        scale_broadcast_1dim<float>     (input, scale, bias, out, tiling_data);


    }

    else if (TILING_KEY_IS(201)) {
         GET_TILING_DATA(tiling_data, tiling);
          TPipe pipe;
        if (tiling_data.has_bias &&
            (tiling_data.low_size * (int)sizeof(half) % 32 != 0) &&
            tiling_data.low_size <= 1280)
            scale_broadcast_1dim_half_packed_bias(
                input, scale, bias, out, tiling_data ,&pipe);
        else
            scale_broadcast_1dim<half>(input, scale, bias, out, tiling_data);
    }
    else if (TILING_KEY_IS(227)) {
         GET_TILING_DATA(tiling_data, tiling);
        if (tiling_data.low_size <= 512)
            scale_broadcast_1dim_bf16_packed(
                input, scale, bias, out, tiling_data);
        else
            scale_broadcast_1dim<bfloat16_t>(input, scale, bias, out, tiling_data);
    }
}
