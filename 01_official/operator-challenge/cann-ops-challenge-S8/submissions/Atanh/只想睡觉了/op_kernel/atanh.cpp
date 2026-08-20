#include "kernel_operator.h"
#include "kernel.h"
using namespace AscendC;

template <typename T>
class KernelAtanh {
public:
    __aicore__ inline KernelAtanh() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, int32_t baseSize) {
        this->coreSize = baseSize;
        int32_t srcBeginIndex = this->coreSize * (GetBlockIdx());
        inputGm.SetGlobalBuffer((__gm__ T *)input + srcBeginIndex, this->coreSize);
        outGm.SetGlobalBuffer((__gm__ T *)output + srcBeginIndex, this->coreSize);
        constexpr uint32_t max_spaceSize = (MAX_UB_SIZE_KB / (1 + 1 + GetTmpLocalMultiple<T>()) / BufferNum) * 1024;
        uint32_t spaceSize = min((uint32_t)(coreSize * sizeof(T)), max_spaceSize);
        this->n_elements_per_iter = spaceSize / sizeof(T);
        this->smallLoopTimes = (coreSize + n_elements_per_iter - 1) / n_elements_per_iter;
        this->tailSize = coreSize % n_elements_per_iter;
        if (tailSize == 0) {
            tailSize = n_elements_per_iter;
        }
    }
    template <bool Reverse>
    __aicore__ inline void Process() {
        // BufferNum == 2
        LocalTensor<T> xLocal[BufferNum] = {
            LocalTensor<T>(TPosition::VECIN, 0, n_elements_per_iter),
            LocalTensor<T>(TPosition::VECIN, n_elements_per_iter * 2 * sizeof(T) + 256, n_elements_per_iter),
        };
        LocalTensor<T> yLocal[BufferNum] = {
            LocalTensor<T>(TPosition::VECOUT, n_elements_per_iter * sizeof(T) + 256, n_elements_per_iter),
            LocalTensor<T>(TPosition::VECOUT, n_elements_per_iter * 3 * sizeof(T) + 512, n_elements_per_iter),
        };

        // 注意GetTmpLocalMultiple<T>()只可能是0或2
        LocalTensor<T> tmpyLocal[BufferNum] = {
            LocalTensor<T>(TPosition::VECCALC, 0, n_elements_per_iter * GetTmpLocalMultiple<T>()),
            LocalTensor<T>(TPosition::VECCALC, n_elements_per_iter * GetTmpLocalMultiple<T>() * sizeof(T) + 256, n_elements_per_iter * GetTmpLocalMultiple<T>()),
        };
        LocalTensor<T> tmpxLocal[BufferNum] = {
            LocalTensor<T>(TPosition::VECCALC, n_elements_per_iter * 4 * sizeof(T) + 512 + 256, n_elements_per_iter * GetTmpLocalMultiple<T>()),
            LocalTensor<T>(TPosition::VECCALC, n_elements_per_iter * 4 * sizeof(T) + 512 + 256 + n_elements_per_iter * GetTmpLocalMultiple<T>() * sizeof(T),
                           n_elements_per_iter * GetTmpLocalMultiple<T>()),
        };
        if constexpr (Reverse) {
            for (int32_t i = smallLoopTimes - 1; i >= -1; i -= BufferNum) {
                for (int32_t j = 0; j < BufferNum; ++j) {
                    int32_t iterIndex = i + j;
                    if (iterIndex >= smallLoopTimes || iterIndex < 0) {
                        continue;
                    }
                    uint32_t iterSize = (iterIndex == smallLoopTimes - 1) ? tailSize : n_elements_per_iter;
                    bool set_backward = (iterIndex >= BufferNum);
                    bool wait_backward = (iterIndex + BufferNum < smallLoopTimes);
                    if (wait_backward) {// 需要等到前序iter的计算完成，才能进行下一轮的MTE2
                        WaitFlag<AscendC::HardEvent::V_MTE2>(j);
                    }
                    DataCopy(xLocal[j], inputGm[iterIndex * n_elements_per_iter], iterSize);
                    SetFlag<HardEvent::MTE2_V>(j);
                    WaitFlag<HardEvent::MTE2_V>(j);
                    if (wait_backward) {// 需要等到前序iter的MTE3完成，才能进行下一轮的计算
                        WaitFlag<AscendC::HardEvent::MTE3_V>(j);
                    }

                    Compute(yLocal[j], xLocal[j], tmpyLocal[j], tmpxLocal[j], iterSize, set_backward, j);

                    SetFlag<HardEvent::V_MTE3>(j);
                    WaitFlag<HardEvent::V_MTE3>(j);

                    DataCopy(outGm[iterIndex * n_elements_per_iter], yLocal[j], iterSize);
                    if (set_backward) {// 通知下一轮iter的计算可以开始了
                        SetFlag<AscendC::HardEvent::MTE3_V>(j);
                    }
                }
            }
        } else {
            for (int32_t i = 0; i < smallLoopTimes; i += BufferNum) {
                for (int32_t j = 0; j < BufferNum; ++j) {
                    int32_t iterIndex = i + j;
                    if (iterIndex >= smallLoopTimes) {
                        break;
                    }
                    uint32_t iterSize = (iterIndex == smallLoopTimes - 1) ? tailSize : n_elements_per_iter;
                    bool set_backward = (iterIndex + BufferNum < smallLoopTimes);
                    bool wait_backward = (iterIndex >= BufferNum);
                    if (wait_backward) {// 需要等到前序iter的计算完成，才能进行下一轮的MTE2
                        WaitFlag<AscendC::HardEvent::V_MTE2>(j);
                    }
                    DataCopy(xLocal[j], inputGm[iterIndex * n_elements_per_iter], iterSize);
                    SetFlag<HardEvent::MTE2_V>(j);
                    WaitFlag<HardEvent::MTE2_V>(j);
                    if (wait_backward) {// 需要等到前序iter的MTE3完成，才能进行下一轮的计算
                        WaitFlag<AscendC::HardEvent::MTE3_V>(j);
                    }

                    Compute(yLocal[j], xLocal[j], tmpyLocal[j], tmpxLocal[j], iterSize, set_backward, j);

                    SetFlag<HardEvent::V_MTE3>(j);
                    WaitFlag<HardEvent::V_MTE3>(j);

                    DataCopy(outGm[iterIndex * n_elements_per_iter], yLocal[j], iterSize);
                    if (set_backward) {// 通知下一轮iter的计算可以开始了
                        SetFlag<AscendC::HardEvent::MTE3_V>(j);
                    }
                }
            }
        }
    }

private:
    GlobalTensor<T> inputGm;
    GlobalTensor<T> outGm;
    int32_t coreSize;
    int32_t smallLoopTimes;
    int32_t n_elements_per_iter;
    int32_t tailSize;

private:
    __aicore__ inline void Compute(const LocalTensor<T> &yLocal, const LocalTensor<T> &xLocal, const LocalTensor<T> &tmpyLocal, const LocalTensor<T> &tmpxLocal, uint32_t iterSize, bool set_backward,
                                   int32_t lockIdx) {
        if constexpr (is_one_of_v<T, float, half>) {
            ComputeKernel_reuseY(yLocal, xLocal, iterSize, set_backward, lockIdx);
        } else if constexpr (is_one_of_v<T, int32_t>) {// 等位宽
            auto xLocal_fp32 = xLocal.template ReinterpretCast<float>();
            auto yLocal_fp32 = yLocal.template ReinterpretCast<float>();
            Cast(xLocal_fp32, xLocal, RoundMode::CAST_RINT, iterSize);
            ComputeKernel_reuseY(yLocal_fp32, xLocal_fp32, iterSize, set_backward, lockIdx);
            Cast(yLocal, yLocal_fp32, RoundMode::CAST_RINT, iterSize);
        } else if constexpr (is_one_of_v<T, bfloat16_t, int16_t>) {// tmpyLocal复用了yLocal+xLocal的空间
            auto xLocal_fp32 = tmpxLocal.template ReinterpretCast<float>();
            auto yLocal_fp32 = tmpyLocal.template ReinterpretCast<float>();
            Cast(xLocal_fp32, xLocal, RoundMode::CAST_NONE, iterSize);
            ComputeKernel_reuseX(yLocal_fp32, xLocal_fp32, iterSize, set_backward, lockIdx);
            Cast(yLocal, xLocal_fp32, RoundMode::CAST_RINT, iterSize);
        } else if constexpr (is_one_of_v<T, int8_t, uint8_t>) {// tmpyLocal复用了yLocal+xLocal的空间
            auto xLocal_fp16 = tmpxLocal.template ReinterpretCast<half>();
            auto yLocal_fp16 = tmpyLocal.template ReinterpretCast<half>();
            Cast(xLocal_fp16, xLocal, RoundMode::CAST_NONE, iterSize);
            ComputeKernel_reuseX(yLocal_fp16, xLocal_fp16, iterSize, set_backward, lockIdx);
            Cast(yLocal, xLocal_fp16, RoundMode::CAST_RINT, iterSize);
        }
    }
    template <typename U>
    __aicore__ inline void ComputeKernel_reuseY(const LocalTensor<U> &yLocal, const LocalTensor<U> &xLocal, uint32_t iterSize, bool set_backward, int32_t lockIdx) {
        // 1+x
        Adds(yLocal, xLocal, static_cast<U>(1), iterSize);
        // 1-x
        Muls(xLocal, xLocal, static_cast<U>(-1), iterSize);
        Adds(xLocal, xLocal, static_cast<U>(1), iterSize);
        // (1+x)/(1-x)
        Div(yLocal, yLocal, xLocal, iterSize);
        if (set_backward) {// 通知下一轮iter的MTE2可以开始了
            SetFlag<AscendC::HardEvent::V_MTE2>(lockIdx);
        }
        // ln((1+x)/(1-x))
        Ln(yLocal, yLocal, iterSize);
        // 0.5*ln((1+x)/(1-x))
        Muls(yLocal, yLocal, static_cast<U>(0.5), iterSize);
    }

    template <typename U>
    __aicore__ inline void ComputeKernel_reuseX(const LocalTensor<U> &yLocal, const LocalTensor<U> &xLocal, uint32_t iterSize, bool set_backward, int32_t lockIdx) {
        // 1+x
        Adds(yLocal, xLocal, static_cast<U>(1), iterSize);
        // 1-x
        Muls(xLocal, xLocal, static_cast<U>(-1), iterSize);
        Adds(xLocal, xLocal, static_cast<U>(1), iterSize);
        // (1+x)/(1-x)
        Div(xLocal, yLocal, xLocal, iterSize);
        if (set_backward) { // 通知下一轮iter的MTE2可以开始了
            SetFlag<AscendC::HardEvent::V_MTE2>(lockIdx);
        }
        // ln((1+x)/(1-x))
        Ln(xLocal, xLocal, iterSize);
        // 0.5*ln((1+x)/(1-x))
        Muls(xLocal, xLocal, static_cast<U>(0.5), iterSize);
    }
};

extern "C" __global__ __aicore__ void atanh(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAtanh<DTYPE_INPUT> kernel;
    kernel.Init(input, output, tiling_data.baseSize);

    if (TILING_KEY_IS(1)) {
        kernel.Process<false>();
    } else if (TILING_KEY_IS(2)) {
        if (needReverse(workspace)) {
            kernel.Process<true>();
        } else {
            kernel.Process<false>();
        }
    }
}