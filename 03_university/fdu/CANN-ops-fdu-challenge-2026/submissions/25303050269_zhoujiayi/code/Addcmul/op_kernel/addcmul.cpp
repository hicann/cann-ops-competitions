#include "kernel_operator.h"

#include "addcmul_tiling.h"
#include "tiling_key_addcmul.h"

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
struct AddcmulUseVector {
    static constexpr bool value = false;
};

template <>
struct AddcmulUseVector<half> {
    static constexpr bool value = true;
};

template <>
struct AddcmulUseVector<float> {
    static constexpr bool value = true;
};

template <>
struct AddcmulUseVector<int32_t> {
    static constexpr bool value = true;
};

template <typename T>
struct AddcmulUseLocalScalar {
    static constexpr bool value = false;
};

template <>
struct AddcmulUseLocalScalar<int8_t> {
    static constexpr bool value = true;
};

template <typename T>
class KernelAddcmul {
public:
    __aicore__ inline KernelAddcmul() {}

    __aicore__ inline void Init(GM_ADDR inputData, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
                                const AddcmulTilingData &tiling)
    {
        inputGm.SetGlobalBuffer((__gm__ T *)inputData, tiling.inputLength);
        x1Gm.SetGlobalBuffer((__gm__ T *)x1, tiling.x1Length);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2, tiling.x2Length);
        valueGm.SetGlobalBuffer((__gm__ T *)value, 1);
        yGm.SetGlobalBuffer((__gm__ T *)y, tiling.totalLength);

        totalLength = tiling.totalLength;
        blockLength = tiling.blockLength;
        tileLength = tiling.tileLength;
        dimNum = tiling.dimNum;
        fastMode = tiling.fastMode;
        lastDim = tiling.lastDim;
        if (fastMode == 99) {
            for (uint32_t i = 0; i < ADDCMUL_MAX_DIMS; ++i) {
                outShape[i] = tiling.outShape[i];
                outStride[i] = tiling.outStride[i];
                inputStride[i] = tiling.inputStride[i];
                x1Stride[i] = tiling.x1Stride[i];
                x2Stride[i] = tiling.x2Stride[i];
            }
        }

        if (tileLength > 0 && fastMode != 99 && totalLength >= 32 / sizeof(T)) {
            if constexpr (AddcmulUseLocalScalar<T>::value) {
                pipe.InitBuffer(inputQue, BUFFER_NUM, tileLength * sizeof(T));
                pipe.InitBuffer(x1Que, BUFFER_NUM, tileLength * sizeof(T));
                pipe.InitBuffer(x2Que, BUFFER_NUM, tileLength * sizeof(T));
            } else {
                const bool simpleMode = fastMode == 1 || fastMode == 2 || fastMode == 3 || fastMode == 10 ||
                                        fastMode == 11 || fastMode == 12;
                const bool genericMode = !simpleMode;
                if (genericMode || fastMode == 1 || fastMode == 2 || fastMode == 10) {
                    pipe.InitBuffer(inputQue, BUFFER_NUM, tileLength * sizeof(T));
                }
                if (genericMode || fastMode == 1 || fastMode == 3 || fastMode == 11) {
                    pipe.InitBuffer(x1Que, BUFFER_NUM, tileLength * sizeof(T));
                }
                if (genericMode || fastMode == 2 || fastMode == 3 || fastMode == 12) {
                    pipe.InitBuffer(x2Que, BUFFER_NUM, tileLength * sizeof(T));
                }
            }
            pipe.InitBuffer(yQue, BUFFER_NUM, tileLength * sizeof(T));
            pipe.InitBuffer(tmpBuf, tileLength * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        if (totalLength == 0) {
            return;
        }

        startOffset = AscendC::GetBlockIdx() * blockLength;
        uint32_t endOffset = startOffset + blockLength;
        if (endOffset > totalLength) {
            endOffset = totalLength;
        }
        if (startOffset >= endOffset) {
            return;
        }

        scale = valueGm.GetValue(0);
        inputScalar = inputGm.GetValue(0);
        x1Scalar = x1Gm.GetValue(0);
        x2Scalar = x2Gm.GetValue(0);
        uint32_t done = 0;
        const uint32_t coreLength = endOffset - startOffset;
        if constexpr (AddcmulUseVector<T>::value) {
            if (IsSimpleVectorMode()) {
                ProcessSimpleVector(coreLength);
                return;
            }
            const uint32_t alignNum = 32 / sizeof(T);
            uint32_t vectorLength = 0;
            if ((fastMode < 4 || (fastMode >= 10 && fastMode <= 12)) && tileLength > 0) {
                vectorLength = (coreLength / alignNum) * alignNum;
            }

            while (done < vectorLength) {
                currentLength = tileLength;
                if (currentLength > vectorLength - done) {
                    currentLength = vectorLength - done;
                }
                CopyIn(done);
                Compute();
                CopyOut(done);
                done += currentLength;
            }

            if (((fastMode >= 4 && fastMode <= 9) || (fastMode >= 13 && fastMode <= 21)) && tileLength > 0) {
                while (done < coreLength) {
                    const uint32_t absOffset = startOffset + done;
                    const uint32_t col = absOffset % lastDim;
                    if (col % alignNum != 0) {
                        ScalarCompute(absOffset);
                        ++done;
                        continue;
                    }
                    uint32_t rowRemain = lastDim - col;
                    uint32_t remain = coreLength - done;
                    currentLength = tileLength;
                    if (currentLength > rowRemain) {
                        currentLength = rowRemain;
                    }
                    if (currentLength > remain) {
                        currentLength = remain;
                    }
                    currentLength = (currentLength / alignNum) * alignNum;
                    if (currentLength == 0) {
                        ScalarCompute(absOffset);
                        ++done;
                        continue;
                    }
                    CopyIn(done);
                    Compute();
                    CopyOut(done);
                    done += currentLength;
                }
            }
        }
        if constexpr (AddcmulUseLocalScalar<T>::value) {
            const uint32_t alignNum = 32 / sizeof(T);
            uint32_t vectorLength = 0;
            if ((fastMode < 4 || (fastMode >= 10 && fastMode <= 12)) && tileLength > 0) {
                vectorLength = (coreLength / alignNum) * alignNum;
            }

            while (done < vectorLength) {
                currentLength = tileLength;
                if (currentLength > vectorLength - done) {
                    currentLength = vectorLength - done;
                }
                CopyIn(done);
                ComputeLocalScalar();
                CopyOut(done);
                done += currentLength;
            }

            if (((fastMode >= 4 && fastMode <= 9) || (fastMode >= 13 && fastMode <= 21)) && tileLength > 0) {
                while (done < coreLength) {
                    const uint32_t absOffset = startOffset + done;
                    const uint32_t col = absOffset % lastDim;
                    if (col % alignNum != 0) {
                        ScalarCompute(absOffset);
                        ++done;
                        continue;
                    }
                    uint32_t rowRemain = lastDim - col;
                    uint32_t remain = coreLength - done;
                    currentLength = tileLength;
                    if (currentLength > rowRemain) {
                        currentLength = rowRemain;
                    }
                    if (currentLength > remain) {
                        currentLength = remain;
                    }
                    currentLength = (currentLength / alignNum) * alignNum;
                    if (currentLength == 0) {
                        ScalarCompute(absOffset);
                        ++done;
                        continue;
                    }
                    CopyIn(done);
                    ComputeLocalScalar();
                    CopyOut(done);
                    done += currentLength;
                }
            }
        }

        if (done == 0 && fastMode == 99) {
            ScalarComputeRange(startOffset, coreLength);
            return;
        }

        for (uint32_t i = done; i < coreLength; ++i) {
            ScalarCompute(startOffset + i);
        }
    }

private:
    __aicore__ inline bool IsSimpleVectorMode() const
    {
        return fastMode == 1 || fastMode == 2 || fastMode == 3 || fastMode == 10 || fastMode == 11 ||
               fastMode == 12;
    }

    __aicore__ inline void ProcessSimpleVector(uint32_t coreLength)
    {
        const uint32_t alignNum = 32 / sizeof(T);
        const uint32_t vectorLength = (coreLength / alignNum) * alignNum;
        uint32_t done = 0;
        while (done < vectorLength) {
            currentLength = tileLength;
            if (currentLength > vectorLength - done) {
                currentLength = vectorLength - done;
            }
            CopyInSimple(done);
            ComputeSimple();
            CopyOut(done);
            done += currentLength;
        }
        for (uint32_t i = done; i < coreLength; ++i) {
            ScalarCompute(startOffset + i);
        }
    }

    __aicore__ inline void CopyInSimple(uint32_t progress)
    {
        if (fastMode == 1 || fastMode == 2 || fastMode == 10) {
            AscendC::LocalTensor<T> inputLocal = inputQue.AllocTensor<T>();
            AscendC::DataCopy(inputLocal, inputGm[startOffset + progress], currentLength);
            inputQue.EnQue(inputLocal);
        }
        if (fastMode == 1 || fastMode == 3 || fastMode == 11) {
            AscendC::LocalTensor<T> x1Local = x1Que.AllocTensor<T>();
            AscendC::DataCopy(x1Local, x1Gm[startOffset + progress], currentLength);
            x1Que.EnQue(x1Local);
        }
        if (fastMode == 2 || fastMode == 3 || fastMode == 12) {
            AscendC::LocalTensor<T> x2Local = x2Que.AllocTensor<T>();
            AscendC::DataCopy(x2Local, x2Gm[startOffset + progress], currentLength);
            x2Que.EnQue(x2Local);
        }
    }

    __aicore__ inline void ComputeSimple()
    {
        AscendC::LocalTensor<T> yLocal = yQue.AllocTensor<T>();
        AscendC::LocalTensor<T> tmpLocal = tmpBuf.Get<T>();
        if (fastMode == 1) {
            AscendC::LocalTensor<T> inputLocal = inputQue.DeQue<T>();
            AscendC::LocalTensor<T> x1Local = x1Que.DeQue<T>();
            AscendC::Muls(tmpLocal, x1Local, x2Scalar, currentLength);
            AscendC::Muls(tmpLocal, tmpLocal, scale, currentLength);
            AscendC::Add(yLocal, inputLocal, tmpLocal, currentLength);
            inputQue.FreeTensor(inputLocal);
            x1Que.FreeTensor(x1Local);
        } else if (fastMode == 2) {
            AscendC::LocalTensor<T> inputLocal = inputQue.DeQue<T>();
            AscendC::LocalTensor<T> x2Local = x2Que.DeQue<T>();
            AscendC::Muls(tmpLocal, x2Local, x1Scalar, currentLength);
            AscendC::Muls(tmpLocal, tmpLocal, scale, currentLength);
            AscendC::Add(yLocal, inputLocal, tmpLocal, currentLength);
            inputQue.FreeTensor(inputLocal);
            x2Que.FreeTensor(x2Local);
        } else if (fastMode == 3) {
            AscendC::LocalTensor<T> x1Local = x1Que.DeQue<T>();
            AscendC::LocalTensor<T> x2Local = x2Que.DeQue<T>();
            AscendC::Mul(tmpLocal, x1Local, x2Local, currentLength);
            AscendC::Muls(tmpLocal, tmpLocal, scale, currentLength);
            AscendC::Adds(yLocal, tmpLocal, inputScalar, currentLength);
            x1Que.FreeTensor(x1Local);
            x2Que.FreeTensor(x2Local);
        } else if (fastMode == 10) {
            AscendC::LocalTensor<T> inputLocal = inputQue.DeQue<T>();
            const T factor = ComputeScalar(static_cast<T>(0), x1Scalar, x2Scalar, scale);
            AscendC::Adds(yLocal, inputLocal, factor, currentLength);
            inputQue.FreeTensor(inputLocal);
        } else if (fastMode == 11) {
            AscendC::LocalTensor<T> x1Local = x1Que.DeQue<T>();
            AscendC::Muls(tmpLocal, x1Local, x2Scalar, currentLength);
            AscendC::Muls(tmpLocal, tmpLocal, scale, currentLength);
            AscendC::Adds(yLocal, tmpLocal, inputScalar, currentLength);
            x1Que.FreeTensor(x1Local);
        } else {
            AscendC::LocalTensor<T> x2Local = x2Que.DeQue<T>();
            AscendC::Muls(tmpLocal, x2Local, x1Scalar, currentLength);
            AscendC::Muls(tmpLocal, tmpLocal, scale, currentLength);
            AscendC::Adds(yLocal, tmpLocal, inputScalar, currentLength);
            x2Que.FreeTensor(x2Local);
        }
        yQue.EnQue<T>(yLocal);
    }

    __aicore__ inline void CopyIn(uint32_t progress)
    {
        AscendC::LocalTensor<T> inputLocal = inputQue.AllocTensor<T>();
        AscendC::LocalTensor<T> x1Local = x1Que.AllocTensor<T>();
        AscendC::LocalTensor<T> x2Local = x2Que.AllocTensor<T>();
        if (fastMode == 3) {
        } else if (fastMode == 9) {
        } else if (fastMode == 11 || fastMode == 12) {
        } else if (fastMode == 15 || fastMode == 16) {
        } else if (fastMode == 6 || fastMode == 17 || fastMode == 18 || fastMode == 20 || fastMode == 21) {
            AscendC::DataCopy(inputLocal, inputGm[(startOffset + progress) % lastDim], currentLength);
        } else {
            AscendC::DataCopy(inputLocal, inputGm[startOffset + progress], currentLength);
        }
        if (fastMode == 2) {
        } else if (fastMode == 8) {
        } else if (fastMode == 10 || fastMode == 12) {
        } else if (fastMode == 14 || fastMode == 18) {
        } else if (fastMode == 5 || fastMode == 13 || fastMode == 16 || fastMode == 19 || fastMode == 21) {
            AscendC::DataCopy(x1Local, x1Gm[(startOffset + progress) % lastDim], currentLength);
        } else {
            AscendC::DataCopy(x1Local, x1Gm[startOffset + progress], currentLength);
        }
        if (fastMode == 1) {
        } else if (fastMode == 7) {
        } else if (fastMode == 10 || fastMode == 11) {
        } else if (fastMode == 13 || fastMode == 17) {
        } else if (fastMode == 4 || fastMode == 14 || fastMode == 15 || fastMode == 19 || fastMode == 20) {
            AscendC::DataCopy(x2Local, x2Gm[(startOffset + progress) % lastDim], currentLength);
        } else {
            AscendC::DataCopy(x2Local, x2Gm[startOffset + progress], currentLength);
        }
        if (fastMode == 7) {
            x2Scalar = x2Gm.GetValue((startOffset + progress) / lastDim);
        } else if (fastMode == 8) {
            x1Scalar = x1Gm.GetValue((startOffset + progress) / lastDim);
        } else if (fastMode == 9) {
            inputScalar = inputGm.GetValue((startOffset + progress) / lastDim);
        }
        inputQue.EnQue(inputLocal);
        x1Que.EnQue(x1Local);
        x2Que.EnQue(x2Local);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> inputLocal = inputQue.DeQue<T>();
        AscendC::LocalTensor<T> x1Local = x1Que.DeQue<T>();
        AscendC::LocalTensor<T> x2Local = x2Que.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = yQue.AllocTensor<T>();
        AscendC::LocalTensor<T> tmpLocal = tmpBuf.Get<T>();

        if (fastMode == 1 || fastMode == 7 || fastMode == 13 || fastMode == 17) {
            AscendC::Muls(tmpLocal, x1Local, x2Scalar, currentLength);
            AscendC::Muls(tmpLocal, tmpLocal, scale, currentLength);
            AscendC::Add(yLocal, inputLocal, tmpLocal, currentLength);
        } else if (fastMode == 2 || fastMode == 8 || fastMode == 14 || fastMode == 18) {
            AscendC::Muls(tmpLocal, x2Local, x1Scalar, currentLength);
            AscendC::Muls(tmpLocal, tmpLocal, scale, currentLength);
            AscendC::Add(yLocal, inputLocal, tmpLocal, currentLength);
        } else if (fastMode == 3 || fastMode == 9 || fastMode == 15 || fastMode == 16) {
            AscendC::Mul(tmpLocal, x1Local, x2Local, currentLength);
            AscendC::Muls(tmpLocal, tmpLocal, scale, currentLength);
            AscendC::Adds(yLocal, tmpLocal, inputScalar, currentLength);
        } else if (fastMode == 10) {
            const T factor = ComputeScalar(static_cast<T>(0), x1Scalar, x2Scalar, scale);
            AscendC::Adds(yLocal, inputLocal, factor, currentLength);
        } else if (fastMode == 11) {
            AscendC::Muls(tmpLocal, x1Local, x2Scalar, currentLength);
            AscendC::Muls(tmpLocal, tmpLocal, scale, currentLength);
            AscendC::Adds(yLocal, tmpLocal, inputScalar, currentLength);
        } else if (fastMode == 12) {
            AscendC::Muls(tmpLocal, x2Local, x1Scalar, currentLength);
            AscendC::Muls(tmpLocal, tmpLocal, scale, currentLength);
            AscendC::Adds(yLocal, tmpLocal, inputScalar, currentLength);
        } else {
            AscendC::Mul(tmpLocal, x1Local, x2Local, currentLength);
            AscendC::Muls(tmpLocal, tmpLocal, scale, currentLength);
            AscendC::Add(yLocal, inputLocal, tmpLocal, currentLength);
        }

        yQue.EnQue<T>(yLocal);
        inputQue.FreeTensor(inputLocal);
        x1Que.FreeTensor(x1Local);
        x2Que.FreeTensor(x2Local);
    }

    __aicore__ inline void ComputeLocalScalar()
    {
        AscendC::LocalTensor<T> inputLocal = inputQue.DeQue<T>();
        AscendC::LocalTensor<T> x1Local = x1Que.DeQue<T>();
        AscendC::LocalTensor<T> x2Local = x2Que.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = yQue.AllocTensor<T>();

        uint32_t i = 0;
        for (; i + 7 < currentLength; i += 8) {
            ComputeLocalScalarAt(yLocal, inputLocal, x1Local, x2Local, i);
            ComputeLocalScalarAt(yLocal, inputLocal, x1Local, x2Local, i + 1);
            ComputeLocalScalarAt(yLocal, inputLocal, x1Local, x2Local, i + 2);
            ComputeLocalScalarAt(yLocal, inputLocal, x1Local, x2Local, i + 3);
            ComputeLocalScalarAt(yLocal, inputLocal, x1Local, x2Local, i + 4);
            ComputeLocalScalarAt(yLocal, inputLocal, x1Local, x2Local, i + 5);
            ComputeLocalScalarAt(yLocal, inputLocal, x1Local, x2Local, i + 6);
            ComputeLocalScalarAt(yLocal, inputLocal, x1Local, x2Local, i + 7);
        }
        for (; i < currentLength; ++i) {
            ComputeLocalScalarAt(yLocal, inputLocal, x1Local, x2Local, i);
        }

        yQue.EnQue<T>(yLocal);
        inputQue.FreeTensor(inputLocal);
        x1Que.FreeTensor(x1Local);
        x2Que.FreeTensor(x2Local);
    }

    __aicore__ inline void ComputeLocalScalarAt(AscendC::LocalTensor<T> yLocal,
                                                AscendC::LocalTensor<T> inputLocal,
                                                AscendC::LocalTensor<T> x1Local,
                                                AscendC::LocalTensor<T> x2Local, uint32_t index)
    {
        T inputValue = (fastMode == 3 || fastMode == 9 || fastMode == 11 || fastMode == 12 ||
                        fastMode == 15 || fastMode == 16)
                           ? inputScalar
                           : inputLocal.GetValue(index);
        T x1Value = (fastMode == 2 || fastMode == 8 || fastMode == 10 || fastMode == 12 ||
                     fastMode == 14 || fastMode == 18)
                        ? x1Scalar
                        : x1Local.GetValue(index);
        T x2Value = (fastMode == 1 || fastMode == 7 || fastMode == 10 || fastMode == 11 ||
                     fastMode == 13 || fastMode == 17)
                        ? x2Scalar
                        : x2Local.GetValue(index);
        yLocal.SetValue(index, ComputeScalar(inputValue, x1Value, x2Value, scale));
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<T> yLocal = yQue.DeQue<T>();
        AscendC::DataCopy(yGm[startOffset + progress], yLocal, currentLength);
        yQue.FreeTensor(yLocal);
    }

    __aicore__ inline void ScalarCompute(uint32_t outOffset)
    {
        if (fastMode < 99) {
            ScalarComputeFast(outOffset);
            return;
        }
        const uint32_t inputOffset = BroadcastOffset(outOffset, inputStride);
        const uint32_t x1Offset = BroadcastOffset(outOffset, x1Stride);
        const uint32_t x2Offset = BroadcastOffset(outOffset, x2Stride);
        T result = ComputeScalar(inputGm.GetValue(inputOffset), x1Gm.GetValue(x1Offset), x2Gm.GetValue(x2Offset), scale);
        yGm.SetValue(outOffset, result);
    }

    __aicore__ inline void ScalarComputeFast(uint32_t outOffset)
    {
        uint32_t inputOffset = outOffset;
        uint32_t x1Offset = outOffset;
        uint32_t x2Offset = outOffset;
        if (fastMode == 1) {
            x2Offset = 0;
        } else if (fastMode == 2) {
            x1Offset = 0;
        } else if (fastMode == 3) {
            inputOffset = 0;
        } else if (fastMode == 4) {
            x2Offset = outOffset % lastDim;
        } else if (fastMode == 5) {
            x1Offset = outOffset % lastDim;
        } else if (fastMode == 6) {
            inputOffset = outOffset % lastDim;
        } else if (fastMode == 7) {
            x2Offset = outOffset / lastDim;
        } else if (fastMode == 8) {
            x1Offset = outOffset / lastDim;
        } else if (fastMode == 9) {
            inputOffset = outOffset / lastDim;
        } else if (fastMode == 10) {
            x1Offset = 0;
            x2Offset = 0;
        } else if (fastMode == 11) {
            inputOffset = 0;
            x2Offset = 0;
        } else if (fastMode == 12) {
            inputOffset = 0;
            x1Offset = 0;
        } else if (fastMode == 13) {
            x1Offset = outOffset % lastDim;
            x2Offset = 0;
        } else if (fastMode == 14) {
            x1Offset = 0;
            x2Offset = outOffset % lastDim;
        } else if (fastMode == 15) {
            inputOffset = 0;
            x2Offset = outOffset % lastDim;
        } else if (fastMode == 16) {
            inputOffset = 0;
            x1Offset = outOffset % lastDim;
        } else if (fastMode == 17) {
            inputOffset = outOffset % lastDim;
            x2Offset = 0;
        } else if (fastMode == 18) {
            inputOffset = outOffset % lastDim;
            x1Offset = 0;
        } else if (fastMode == 19) {
            x1Offset = outOffset % lastDim;
            x2Offset = outOffset % lastDim;
        } else if (fastMode == 20) {
            inputOffset = outOffset % lastDim;
            x2Offset = outOffset % lastDim;
        } else if (fastMode == 21) {
            inputOffset = outOffset % lastDim;
            x1Offset = outOffset % lastDim;
        }
        T result = ComputeScalar(inputGm.GetValue(inputOffset), x1Gm.GetValue(x1Offset), x2Gm.GetValue(x2Offset), scale);
        yGm.SetValue(outOffset, result);
    }

    __aicore__ inline void ScalarComputeRange(uint32_t outOffset, uint32_t length)
    {
        uint32_t coord[ADDCMUL_MAX_DIMS];
        uint32_t inputOffset = 0;
        uint32_t x1Offset = 0;
        uint32_t x2Offset = 0;
        uint32_t tmp = outOffset;
        for (uint32_t i = 0; i < dimNum; ++i) {
            coord[i] = tmp / outStride[i];
            tmp -= coord[i] * outStride[i];
            inputOffset += coord[i] * inputStride[i];
            x1Offset += coord[i] * x1Stride[i];
            x2Offset += coord[i] * x2Stride[i];
        }

        for (uint32_t i = 0; i < length; ++i) {
            T result = ComputeScalar(inputGm.GetValue(inputOffset), x1Gm.GetValue(x1Offset), x2Gm.GetValue(x2Offset), scale);
            yGm.SetValue(outOffset + i, result);
            if (i + 1 == length) {
                break;
            }
            for (int32_t dim = static_cast<int32_t>(dimNum) - 1; dim >= 0; --dim) {
                ++coord[dim];
                inputOffset += inputStride[dim];
                x1Offset += x1Stride[dim];
                x2Offset += x2Stride[dim];
                if (coord[dim] < outShape[dim]) {
                    break;
                }
                coord[dim] = 0;
                inputOffset -= outShape[dim] * inputStride[dim];
                x1Offset -= outShape[dim] * x1Stride[dim];
                x2Offset -= outShape[dim] * x2Stride[dim];
            }
        }
    }

    __aicore__ inline uint32_t BroadcastOffset(uint32_t outOffset, const uint32_t *stride) const
    {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dimNum; ++i) {
            const uint32_t coord = outOffset / outStride[i];
            outOffset -= coord * outStride[i];
            offset += coord * stride[i];
        }
        return offset;
    }

    __aicore__ inline float ComputeScalar(float input, float x1, float x2, float scalar) const
    {
        return input + x1 * x2 * scalar;
    }

    __aicore__ inline half ComputeScalar(half input, half x1, half x2, half scalar) const
    {
        float result = static_cast<float>(input) + static_cast<float>(x1) * static_cast<float>(x2) * static_cast<float>(scalar);
        return static_cast<half>(result);
    }

    __aicore__ inline int32_t ComputeScalar(int32_t input, int32_t x1, int32_t x2, int32_t scalar) const
    {
        int64_t result = static_cast<int64_t>(input) + static_cast<int64_t>(x1) * static_cast<int64_t>(x2) * static_cast<int64_t>(scalar);
        return static_cast<int32_t>(result);
    }

    __aicore__ inline int8_t ComputeScalar(int8_t input, int8_t x1, int8_t x2, int8_t scalar) const
    {
        int32_t result = static_cast<int32_t>(input) + static_cast<int32_t>(x1) * static_cast<int32_t>(x2) * static_cast<int32_t>(scalar);
        return static_cast<int8_t>(result);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inputQue;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> x1Que;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> x2Que;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> yQue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<T> inputGm;
    AscendC::GlobalTensor<T> x1Gm;
    AscendC::GlobalTensor<T> x2Gm;
    AscendC::GlobalTensor<T> valueGm;
    AscendC::GlobalTensor<T> yGm;
    T scale;
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t currentLength;
    uint32_t startOffset;
    uint32_t dimNum;
    uint32_t fastMode;
    uint32_t lastDim;
    T inputScalar;
    T x1Scalar;
    T x2Scalar;
    uint32_t outShape[ADDCMUL_MAX_DIMS];
    uint32_t outStride[ADDCMUL_MAX_DIMS];
    uint32_t inputStride[ADDCMUL_MAX_DIMS];
    uint32_t x1Stride[ADDCMUL_MAX_DIMS];
    uint32_t x2Stride[ADDCMUL_MAX_DIMS];
};

template <typename T>
__global__ __aicore__ void addcmul(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
                                   GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddcmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddcmulTilingData, tilingData, tiling);
    KernelAddcmul<T> op;
    op.Init(input_data, x1, x2, value, y, tilingData);
    op.Process();
}
