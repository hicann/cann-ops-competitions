#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <complex>
#include <random>
#include <type_traits>
#include "acl/acl.h"
#include "aclnnop/aclnn_cumsum.h"

#define CHECK_RET(cond, return_expr) \
  do { if (!(cond)) { return_expr; } } while (0)
#define LOG_PRINT(message, ...) \
  do { printf(message, ##__VA_ARGS__); } while (0)

// ============================================================================
// 工具函数：类型转换、Shape计算与精度容差
// ============================================================================
int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 1;
    int64_t size = 1;
    for (auto i : shape) size *= i; 
    return size;
}

uint16_t fp32_to_fp16(float val) {
    uint32_t f; std::memcpy(&f, &val, 4);
    int s = (f >> 16) & 0x8000;
    int e = ((f >> 23) & 0xff) - 127 + 15;
    int m = f & 0x007fffff;
    if (e <= 0) return s;
    if (e >= 31) return s | 0x7c00;
    return s | (e << 10) | (m >> 13);
}

float fp16_to_fp32(uint16_t h) {
    int s = (h >> 15) & 1;
    int e = (h >> 10) & 0x1f;
    int f = h & 0x03ff;
    if (e == 0) return 0.0f;
    if (e == 31) return s ? -INFINITY : INFINITY;
    float val = (1.0f + f * std::pow(2.0f, -10.0f)) * std::pow(2.0f, e - 15.0f);
    return s ? -val : val;
}

uint16_t fp32_to_bf16(float f) {
    uint32_t res; std::memcpy(&res, &f, sizeof(f));
    return res >> 16;
}

float bf16_to_fp32(uint16_t b) {
    uint32_t res = b << 16;
    float f; std::memcpy(&f, &res, sizeof(f));
    return f;
}

bool AlmostEqual(double expected, double actual, double atol, double rtol) {
    if (std::isnan(expected) && std::isnan(actual)) return true;
    if (std::isinf(expected) && std::isinf(actual)) return (expected > 0) == (actual > 0);
    return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

// ============================================================================
// CPU 高精度参考实现 (标量)
// ============================================================================
template<typename T>
std::vector<double> CpuCumsum(const std::vector<T>& input, const std::vector<int64_t>& shape, 
                              int64_t dim, bool exclusive, bool reverse, aclDataType dtIn) {
    int64_t n = GetShapeSize(shape);
    std::vector<double> output(n, 0.0);
    if (n == 0) return output;
    
    int64_t ndim = shape.empty() ? 1 : (int64_t)shape.size();
    std::vector<int64_t> eff_shape = shape.empty() ? std::vector<int64_t>{1} : shape;
    int64_t actual_dim = dim < 0 ? dim + ndim : dim;
    
    int64_t outer_size = 1, inner_size = 1;
    for (int64_t i = 0; i < actual_dim; ++i) outer_size *= eff_shape[i];
    int64_t dim_size = eff_shape[actual_dim];
    for (int64_t i = actual_dim + 1; i < ndim; ++i) inner_size *= eff_shape[i];
    
    for (int64_t o = 0; o < outer_size; ++o) {
        for (int64_t i = 0; i < inner_size; ++i) {
            double sum = 0.0;
            int64_t start = reverse ? dim_size - 1 : 0;
            int64_t end = reverse ? -1 : dim_size;
            int64_t step = reverse ? -1 : 1;

            for (int64_t d = start; d != end; d += step) {
                int64_t idx = o * dim_size * inner_size + d * inner_size + i;
                double val = 0.0;
                
                if (dtIn == ACL_FLOAT16) val = fp16_to_fp32(static_cast<uint16_t>(input[idx]));
                else if (dtIn == ACL_BF16) val = bf16_to_fp32(static_cast<uint16_t>(input[idx]));
                else val = static_cast<double>(input[idx]);

                if (exclusive) {
                    output[idx] = sum;
                    sum += val;
                } else {
                    sum += val;
                    output[idx] = sum;
                }
            }
        }
    }
    return output;
}

// ============================================================================
// CPU 高精度参考实现 (复数)
// ============================================================================
template<typename T>
std::vector<std::complex<double>> CpuCumsumComplex(const std::vector<std::complex<T>>& input,
                                                   const std::vector<int64_t>& shape,
                                                   int64_t dim, bool exclusive, bool reverse) {
    int64_t n = GetShapeSize(shape);
    std::vector<std::complex<double>> output(n, {0.0, 0.0});
    if (n == 0) return output;
    
    int64_t ndim = shape.empty() ? 1 : (int64_t)shape.size();
    std::vector<int64_t> eff_shape = shape.empty() ? std::vector<int64_t>{1} : shape;
    int64_t actual_dim = dim < 0 ? dim + ndim : dim;
    
    int64_t outer_size = 1, inner_size = 1;
    for (int64_t i = 0; i < actual_dim; ++i) outer_size *= eff_shape[i];
    int64_t dim_size = eff_shape[actual_dim];
    for (int64_t i = actual_dim + 1; i < ndim; ++i) inner_size *= eff_shape[i];
    
    for (int64_t o = 0; o < outer_size; ++o) {
        for (int64_t i = 0; i < inner_size; ++i) {
            std::complex<double> sum(0.0, 0.0);
            int64_t start = reverse ? dim_size - 1 : 0;
            int64_t end = reverse ? -1 : dim_size;
            int64_t step = reverse ? -1 : 1;

            for (int64_t d = start; d != end; d += step) {
                int64_t idx = o * dim_size * inner_size + d * inner_size + i;
                std::complex<double> val(static_cast<double>(input[idx].real()),
                                         static_cast<double>(input[idx].imag()));
                if (exclusive) {
                    output[idx] = sum;
                    sum += val;
                } else {
                    sum += val;
                    output[idx] = sum;
                }
            }
        }
    }
    return output;
}

// ============================================================================
// ACL 基础设施层 (带安全内存分配校验)
// ============================================================================
int Init(int32_t deviceId, aclrtStream* stream) {
    CHECK_RET(aclInit(nullptr) == ACL_SUCCESS, return 1);
    CHECK_RET(aclrtSetDevice(deviceId) == ACL_SUCCESS, return 1);
    CHECK_RET(aclrtCreateStream(stream) == ACL_SUCCESS, return 1);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                    void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    if (size > 0) {
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return -1);
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, return -1);
    } else {
        *deviceAddr = nullptr; 
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; i--) strides[i] = shape[i+1] * strides[i+1];
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, 
                              strides.empty() ? nullptr : strides.data(), 0,
                              ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

aclTensor* CreateDummyAclTensor(const std::vector<int64_t>& shape, aclDataType dataType) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; i--) strides[i] = shape[i+1] * strides[i+1];
    return aclCreateTensor(shape.data(), shape.size(), dataType,
                           strides.empty() ? nullptr : strides.data(), 0,
                           ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
}

// ============================================================================
// 验证函数 (标量)
// ============================================================================
template <typename TOut>
int VerifyOutput(const char* name, void* outDev, std::vector<TOut>& outHost, int64_t n, 
                 const std::vector<double>& expected, double atol, double rtol, aclDataType dtOut) {
    if (n > 0) {
        auto ret = aclrtMemcpy(outHost.data(), n * sizeof(TOut), outDev, n * sizeof(TOut), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) return 1;
    }
    int failed = 0;
    for (int64_t i = 0; i < n; i++) {
        double actual = 0;
        if constexpr (std::is_same<TOut, uint16_t>::value) {
            actual = (dtOut == ACL_BF16) ? bf16_to_fp32(outHost[i]) : fp16_to_fp32(outHost[i]);
        } else if constexpr (std::is_arithmetic<TOut>::value) {
            actual = static_cast<double>(outHost[i]);
        }
        if (!AlmostEqual(expected[i], actual, atol, rtol)) {
            failed++;
            if (failed <= 5) {
                LOG_PRINT("NoOneLog  mismatch[%lld]: expected %.15e, actual %.15e\n", i, expected[i], actual);
            }
        }
    }
    LOG_PRINT(failed == 0 ? "NoOneLog[PASS] %s\n" : "NoOneLog[FAIL] %s: %d mismatches\n", name, failed);
    return failed > 0 ? 1 : 0;
}

// ============================================================================
// 验证函数 (复数)
// ============================================================================
template <typename TOut>
int VerifyOutputComplex(const char* name, void* outDev, std::vector<std::complex<TOut>>& outHost, int64_t n,
                        const std::vector<std::complex<double>>& expected, double atol, double rtol) {
    if (n > 0) {
        auto ret = aclrtMemcpy(outHost.data(), n * sizeof(std::complex<TOut>), outDev, n * sizeof(std::complex<TOut>), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) return 1;
    }
    int failed = 0;
    for (int64_t i = 0; i < n; i++) {
        double act_r = static_cast<double>(outHost[i].real());
        double act_i = static_cast<double>(outHost[i].imag());
        if (!AlmostEqual(expected[i].real(), act_r, atol, rtol) ||
            !AlmostEqual(expected[i].imag(), act_i, atol, rtol)) {
            failed++;
            if (failed <= 5) {
                LOG_PRINT("NoOneLog  mismatch[%lld]: expected (%.15e,%.15e), actual (%.15e,%.15e)\n",
                          i, expected[i].real(), expected[i].imag(), act_r, act_i);
            }
        }
    }
    LOG_PRINT(failed == 0 ? "NoOneLog[PASS] %s\n" : "NoOneLog[FAIL] %s: %d mismatches\n", name, failed);
    return failed > 0 ? 1 : 0;
}

// ============================================================================
// 测试执行器 (标量)
// ============================================================================
template <typename TIn, typename TOut>
int RunCumsumV1Test(const char* name, aclDataType dtIn, aclDataType dtOut,
                    const std::vector<TIn>& self, int64_t dim, const std::vector<int64_t>& shape, 
                    aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *selfDev=nullptr, *outDev=nullptr; aclTensor *selfT=nullptr, *outT=nullptr;
    if (CreateAclTensor(self, shape, &selfDev, dtIn, &selfT) != 0) return 1;
    std::vector<TOut> outHost(n > 0 ? n : 1, TOut(0));
    if (CreateAclTensor(outHost, shape, &outDev, dtOut, &outT) != 0) return 1;

    std::vector<double> expected = CpuCumsum(self, shape, dim, false, false, dtIn);
    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnCumsumGetWorkspaceSize(selfT, dim, dtOut, outT, &wsSize, &executor);
    
    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) {
            if (aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) fail_cnt = 1;
        }
        if (fail_cnt == 0) {
            auto execRet = aclnnCumsum(wsAddr, wsSize, executor, stream);
            aclrtSynchronizeStream(stream);
            if (execRet != ACL_SUCCESS) {
                LOG_PRINT("NoOneLog[FAIL] %s Execution Error %d\n", name, execRet);
                fail_cnt = 1;
            } else {
                fail_cnt = VerifyOutput(name, outDev, outHost, n, expected, atol, rtol, dtOut);
            }
        }
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("NoOneLog[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }
    if(selfT) aclDestroyTensor(selfT); if(outT) aclDestroyTensor(outT);
    if(selfDev) aclrtFree(selfDev); if(outDev) aclrtFree(outDev);
    return fail_cnt;
}

template <typename TIn, typename TOut>
int RunCumsumV2Test(const char* name, aclDataType dtIn, aclDataType dtOut,
                    const std::vector<TIn>& self, int64_t dim, bool exclusive, bool reverse,
                    const std::vector<int64_t>& shape, aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *selfDev=nullptr, *outDev=nullptr; aclTensor *selfT=nullptr, *outT=nullptr;
    if (CreateAclTensor(self, shape, &selfDev, dtIn, &selfT) != 0) return 1;
    std::vector<TOut> outHost(n > 0 ? n : 1, TOut(0));
    if (CreateAclTensor(outHost, shape, &outDev, dtOut, &outT) != 0) return 1;

    std::vector<double> expected = CpuCumsum(self, shape, dim, exclusive, reverse, dtIn);
    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnCumsumV2GetWorkspaceSize(selfT, dim, exclusive, reverse, outT, &wsSize, &executor);
    
    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) {
            if (aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) fail_cnt = 1;
        }
        if (fail_cnt == 0) {
            auto execRet = aclnnCumsumV2(wsAddr, wsSize, executor, stream);
            aclrtSynchronizeStream(stream);
            if (execRet != ACL_SUCCESS) {
                LOG_PRINT("NoOneLog[FAIL] %s Execution Error %d\n", name, execRet);
                fail_cnt = 1;
            } else {
                fail_cnt = VerifyOutput(name, outDev, outHost, n, expected, atol, rtol, dtOut);
            }
        }
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("NoOneLog[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }
    if(selfT) aclDestroyTensor(selfT); if(outT) aclDestroyTensor(outT);
    if(selfDev) aclrtFree(selfDev); if(outDev) aclrtFree(outDev);
    return fail_cnt;
}

// ============================================================================
// 测试执行器 (复数)
// ============================================================================
template <typename TIn, typename TOut>
int RunCumsumV1ComplexTest(const char* name, aclDataType dtIn, aclDataType dtOut,
                           const std::vector<std::complex<TIn>>& self, int64_t dim,
                           const std::vector<int64_t>& shape, aclrtStream stream,
                           double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *selfDev=nullptr, *outDev=nullptr; aclTensor *selfT=nullptr, *outT=nullptr;
    if (CreateAclTensor(self, shape, &selfDev, dtIn, &selfT) != 0) return 1;
    std::vector<std::complex<TOut>> outHost(n > 0 ? n : 1, {0, 0});
    if (CreateAclTensor(outHost, shape, &outDev, dtOut, &outT) != 0) return 1;

    auto expected = CpuCumsumComplex(self, shape, dim, false, false);
    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnCumsumGetWorkspaceSize(selfT, dim, dtOut, outT, &wsSize, &executor);
    
    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) {
            if (aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) fail_cnt = 1;
        }
        if (fail_cnt == 0) {
            auto execRet = aclnnCumsum(wsAddr, wsSize, executor, stream);
            aclrtSynchronizeStream(stream);
            if (execRet != ACL_SUCCESS) {
                LOG_PRINT("NoOneLog[FAIL] %s Execution Error %d\n", name, execRet);
                fail_cnt = 1;
            } else {
                fail_cnt = VerifyOutputComplex(name, outDev, outHost, n, expected, atol, rtol);
            }
        }
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("NoOneLog[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }
    if(selfT) aclDestroyTensor(selfT); if(outT) aclDestroyTensor(outT);
    if(selfDev) aclrtFree(selfDev); if(outDev) aclrtFree(outDev);
    return fail_cnt;
}

// ============================================================================
// MAIN 执行器
// ============================================================================
int main() {
    int32_t deviceId = 0; aclrtStream stream;
    if (Init(deviceId, &stream) != 0) return 1;
    int totalFail = 0;

    LOG_PRINT("\n>>> 组1：参数校验与边界 <<<\n");
    uint64_t ws = 0; aclOpExecutor* exec = nullptr;
    aclTensor *d_fp32 = CreateDummyAclTensor({2,3}, ACL_FLOAT);
    aclTensor *d_fp16 = CreateDummyAclTensor({2,3}, ACL_FLOAT16);
    aclTensor *d_shape_bad = CreateDummyAclTensor({2,4}, ACL_FLOAT);
    aclTensor *d_9d = CreateDummyAclTensor({1,1,1,1,1,1,1,1,1}, ACL_FLOAT);
    aclTensor *d_massive = CreateDummyAclTensor({50000001}, ACL_FLOAT);

    totalFail += (aclnnCumsumGetWorkspaceSize(nullptr, 0, ACL_FLOAT, d_fp32, &ws, &exec) == ACL_SUCCESS);
    totalFail += (aclnnCumsumGetWorkspaceSize(d_fp32, 0, ACL_FLOAT, nullptr, &ws, &exec) == ACL_SUCCESS);
    totalFail += (aclnnCumsumV2GetWorkspaceSize(nullptr, 0, false, false, d_fp32, &ws, &exec) == ACL_SUCCESS);
    totalFail += (aclnnCumsumV2GetWorkspaceSize(d_fp32, 0, false, false, nullptr, &ws, &exec) == ACL_SUCCESS);
    totalFail += (aclnnCumsumGetWorkspaceSize(d_fp32, 0, ACL_INT32, d_fp32, &ws, &exec) == ACL_SUCCESS);
    totalFail += (aclnnCumsumV2GetWorkspaceSize(d_fp32, 0, false, false, d_fp16, &ws, &exec) == ACL_SUCCESS);
    totalFail += (aclnnCumsumGetWorkspaceSize(d_fp32, 0, ACL_FLOAT, d_shape_bad, &ws, &exec) == ACL_SUCCESS);
    totalFail += (aclnnCumsumV2GetWorkspaceSize(d_fp32, 0, false, false, d_shape_bad, &ws, &exec) == ACL_SUCCESS);
    totalFail += (aclnnCumsumGetWorkspaceSize(d_9d, 0, ACL_FLOAT, d_9d, &ws, &exec) == ACL_SUCCESS);
    totalFail += (aclnnCumsumV2GetWorkspaceSize(d_9d, 0, false, false, d_9d, &ws, &exec) == ACL_SUCCESS);
    totalFail += (aclnnCumsumGetWorkspaceSize(d_fp32, 2, ACL_FLOAT, d_fp32, &ws, &exec) == ACL_SUCCESS);
    totalFail += (aclnnCumsumV2GetWorkspaceSize(d_fp32, 2, false, false, d_fp32, &ws, &exec) == ACL_SUCCESS);
    totalFail += (aclnnCumsumGetWorkspaceSize(d_fp32, -3, ACL_FLOAT, d_fp32, &ws, &exec) == ACL_SUCCESS);
    totalFail += (aclnnCumsumV2GetWorkspaceSize(d_fp32, -3, false, false, d_fp32, &ws, &exec) == ACL_SUCCESS);
    // dim=0 应成功
    totalFail += (aclnnCumsumGetWorkspaceSize(d_fp32, 0, ACL_FLOAT, d_fp32, &ws, &exec) != ACL_SUCCESS);
    // dim > INT32_MAX 应失败
    totalFail += (aclnnCumsumGetWorkspaceSize(d_fp32, 2147483648LL, ACL_FLOAT, d_fp32, &ws, &exec) == ACL_SUCCESS);

    totalFail += RunCumsumV1Test<float, float>("Empty_0x5", ACL_FLOAT, ACL_FLOAT, {}, 0, {0, 5}, stream);
    totalFail += RunCumsumV1Test<float, float>("Scalar_0D", ACL_FLOAT, ACL_FLOAT, {3.14f}, 0, {}, stream);
    LOG_PRINT("[INFO] Group 1 done.\n");

    LOG_PRINT("\n>>> 组2：基本功能与形状组合 <<<\n");
    std::vector<float> f32_10 = {1,2,3,4,5,6,7,8,9,10};
    totalFail += RunCumsumV2Test<float, float>("Basic_1D_dim0", ACL_FLOAT, ACL_FLOAT, f32_10, 0, false, false, {10}, stream);
    std::vector<uint16_t> fp16_3d(24, fp32_to_fp16(1.0f));
    totalFail += RunCumsumV1Test<uint16_t, uint16_t>("Basic_3D_dim1", ACL_FLOAT16, ACL_FLOAT16, fp16_3d, 1, {2,3,4}, stream, 1e-3, 1e-3);
    std::vector<int32_t> i32_2d(25, 1);
    totalFail += RunCumsumV1Test<int32_t, int32_t>("Basic_2D_dim-1", ACL_INT32, ACL_INT32, i32_2d, -1, {5,5}, stream);
    std::vector<float> f32_1x100(100, 1.5f);
    totalFail += RunCumsumV2Test<float, float>("Basic_1x100_dim1", ACL_FLOAT, ACL_FLOAT, f32_1x100, 1, false, false, {1,100}, stream);
    std::vector<double> f64_2d(128*256, 1.1);
    totalFail += RunCumsumV1Test<double, double>("Basic_DOUBLE", ACL_DOUBLE, ACL_DOUBLE, f64_2d, 0, {128,256}, stream);

    LOG_PRINT("\n>>> 组3：数据类型全覆盖 + 类型提升 <<<\n");
    std::vector<int8_t> i8_1d(10, 2);
    std::vector<int16_t> i16_1d(10, 2);
    std::vector<int64_t> i64_1d(10, 2);
    std::vector<uint8_t> ui8_1d(10, 2);
    std::vector<uint16_t> bf16_1d(10);
    for(int i=0;i<10;i++) bf16_1d[i]=fp32_to_bf16(1.0f);

    totalFail += RunCumsumV1Test<int8_t, int8_t>("DType_INT8", ACL_INT8, ACL_INT8, i8_1d, 0, {10}, stream);
    totalFail += RunCumsumV1Test<int16_t, int16_t>("DType_INT16", ACL_INT16, ACL_INT16, i16_1d, 0, {10}, stream);
    totalFail += RunCumsumV1Test<int64_t, int64_t>("DType_INT64", ACL_INT64, ACL_INT64, i64_1d, 0, {10}, stream);
    totalFail += RunCumsumV1Test<uint8_t, uint8_t>("DType_UINT8", ACL_UINT8, ACL_UINT8, ui8_1d, 0, {10}, stream);
    totalFail += RunCumsumV1Test<uint16_t, uint16_t>("DType_BF16", ACL_BF16, ACL_BF16, bf16_1d, 0, {10}, stream, 1e-2, 1e-2);

    std::vector<std::complex<float>> cf_1d(4, {1.0f, 0.5f});
    totalFail += RunCumsumV1ComplexTest<float, float>("DType_COMPLEX64", ACL_COMPLEX64, ACL_COMPLEX64, cf_1d, 0, {4}, stream, 1e-6, 1e-6);
    std::vector<std::complex<double>> cd_1d(4, {1.0, 0.5});
    totalFail += RunCumsumV1ComplexTest<double, double>("DType_COMPLEX128", ACL_COMPLEX128, ACL_COMPLEX128, cd_1d, 0, {4}, stream, 1e-12, 1e-12);

    std::vector<int8_t> i8_overflow(100, 50);
    totalFail += RunCumsumV1Test<int8_t, int32_t>("Cast_INT8_to_INT32", ACL_INT8, ACL_INT32, i8_overflow, 0, {100}, stream, 0.0, 0.0);
    std::vector<uint16_t> fp16_long(20000, fp32_to_fp16(10.0f));
    totalFail += RunCumsumV1Test<uint16_t, float>("Cast_FP16_to_FP32", ACL_FLOAT16, ACL_FLOAT, fp16_long, 0, {20000}, stream, 1e-4, 1e-4);

    LOG_PRINT("\n>>> 组4：Cube路径真实执行与回退 <<<\n");
    aclTensor *d_cube1 = CreateDummyAclTensor({12800,512}, ACL_FLOAT16);
    aclTensor *d_cube2 = CreateDummyAclTensor({100,128,512}, ACL_BF16);
    aclTensor *d_cube_bad_dim = CreateDummyAclTensor({12800,512}, ACL_FLOAT);
    aclTensor *d_cube_bad_ch = CreateDummyAclTensor({12800,256}, ACL_FLOAT16);
    aclTensor *d_cube_low_batch = CreateDummyAclTensor({10000,512}, ACL_FLOAT);
    totalFail += (aclnnCumsumGetWorkspaceSize(d_cube1, 1, ACL_FLOAT16, d_cube1, &ws, &exec) != ACL_SUCCESS);
    totalFail += (aclnnCumsumGetWorkspaceSize(d_cube2, 2, ACL_BF16, d_cube2, &ws, &exec) != ACL_SUCCESS);
    totalFail += (aclnnCumsumGetWorkspaceSize(d_cube_bad_dim, 0, ACL_FLOAT, d_cube_bad_dim, &ws, &exec) != ACL_SUCCESS);
    totalFail += (aclnnCumsumGetWorkspaceSize(d_cube_bad_ch, 1, ACL_FLOAT16, d_cube_bad_ch, &ws, &exec) != ACL_SUCCESS);
    totalFail += (aclnnCumsumGetWorkspaceSize(d_cube_low_batch, 1, ACL_FLOAT, d_cube_low_batch, &ws, &exec) != ACL_SUCCESS);
    totalFail += (aclnnCumsumGetWorkspaceSize(d_massive, 0, ACL_FLOAT, d_massive, &ws, &exec) != ACL_SUCCESS);

    std::vector<uint16_t> cube_real(12800*512, fp32_to_fp16(1.0f));
    totalFail += RunCumsumV1Test<uint16_t, uint16_t>("Cube_Real_FP16", ACL_FLOAT16, ACL_FLOAT16, cube_real, 1, {12800,512}, stream, 1e-1, 1e-1);

    LOG_PRINT("\n>>> 组5：Tiling策略覆盖 <<<\n");
    std::vector<float> tile_f1(2 * 200000 * 16, 1.0f);
    totalFail += RunCumsumV1Test<float, float>("Tiling_Float_BigR", ACL_FLOAT, ACL_FLOAT, tile_f1, 1, {2,200000,16}, stream);
    std::vector<uint16_t> tile_f2(64 * 20000 * 2, fp32_to_fp16(1.0f));
    totalFail += RunCumsumV1Test<uint16_t, uint16_t>("Tiling_FP16_SmallN", ACL_FLOAT16, ACL_FLOAT16, tile_f2, 1, {64,20000,2}, stream, 1e-3, 1e-3);
    std::vector<uint16_t> tile_f3(2 * 150000 * 1, fp32_to_bf16(1.0f));
    totalFail += RunCumsumV1Test<uint16_t, uint16_t>("Tiling_BF16_TwoWay", ACL_BF16, ACL_BF16, tile_f3, 1, {2,150000,1}, stream, 1e-2, 1e-2);
    std::vector<int32_t> tile_i1(500 * 10 * 100, 1);
    totalFail += RunCumsumV1Test<int32_t, int32_t>("Tiling_Int_RA", ACL_INT32, ACL_INT32, tile_i1, 1, {500,10,100}, stream);
    std::vector<int64_t> tile_i2(10 * 100 * 500, 1);
    totalFail += RunCumsumV1Test<int64_t, int64_t>("Tiling_Int_LastDim", ACL_INT64, ACL_INT64, tile_i2, 2, {10,100,500}, stream);
    std::vector<int8_t> tile_i3(2000 * 50 * 1, 2);
    totalFail += RunCumsumV1Test<int8_t, int8_t>("Tiling_Int_RBlock", ACL_INT8, ACL_INT8, tile_i3, 1, {2000,50,1}, stream);

    LOG_PRINT("\n>>> 组6：Exclusive/Reverse全组合 (V2) <<<\n");
    std::vector<float> f32_4 = {1,2,3,4};
    totalFail += RunCumsumV2Test<float, float>("V2_00", ACL_FLOAT, ACL_FLOAT, f32_4, 0, false, false, {4}, stream);
    totalFail += RunCumsumV2Test<float, float>("V2_10", ACL_FLOAT, ACL_FLOAT, f32_4, 0, true, false, {4}, stream);
    totalFail += RunCumsumV2Test<float, float>("V2_01", ACL_FLOAT, ACL_FLOAT, f32_4, 0, false, true, {4}, stream);
    totalFail += RunCumsumV2Test<float, float>("V2_11", ACL_FLOAT, ACL_FLOAT, f32_4, 0, true, true, {4}, stream);
    std::vector<int32_t> iv = {1,2,3,4,5,6};
    totalFail += RunCumsumV2Test<int32_t, int32_t>("V2_Int_2D_Rev", ACL_INT32, ACL_INT32, iv, 1, false, true, {2,3}, stream);

    LOG_PRINT("\n>>> 组7：精度极限与NaN传播 <<<\n");
    std::vector<float> prec_sub(10000, 1e-40f);
    totalFail += RunCumsumV1Test<float, float>("Prec_Subnormal", ACL_FLOAT, ACL_FLOAT, prec_sub, 0, {10000}, stream, 1e-5, 1e-5);
    std::vector<float> prec_cancel = {1e8f, -1e8f, 1e-8f, 1e-8f, 1e8f, -1e8f};
    totalFail += RunCumsumV1Test<float, float>("Prec_Cancellation", ACL_FLOAT, ACL_FLOAT, prec_cancel, 0, {6}, stream, 1e-6, 1e-6);
    std::vector<float> prec_near1 = {1.0000001f, -1.0000001f, 1.0000001f, -1.0000001f};
    totalFail += RunCumsumV1Test<float, float>("Prec_Near1", ACL_FLOAT, ACL_FLOAT, prec_near1, 0, {4}, stream, 1e-6, 1e-6);
    std::vector<float> prec_dec(100, 0.1f);
    totalFail += RunCumsumV1Test<float, float>("Prec_Decimal", ACL_FLOAT, ACL_FLOAT, prec_dec, 0, {100}, stream, 1e-5, 1e-5);
    std::vector<float> prec_ovf = {1e20f, 2e20f, 3e20f};
    totalFail += RunCumsumV1Test<float, float>("Prec_Overflow", ACL_FLOAT, ACL_FLOAT, prec_ovf, 0, {3}, stream, 1e-5, 1e-5);
    std::vector<uint16_t> prec_f16(4096, fp32_to_fp16(0.1f));
    totalFail += RunCumsumV1Test<uint16_t, uint16_t>("Prec_FP16_0.1", ACL_FLOAT16, ACL_FLOAT16, prec_f16, 0, {4096}, stream, 1e-3, 1e-3);
    std::vector<uint16_t> prec_bf16(4096, fp32_to_bf16(0.1f));
    totalFail += RunCumsumV1Test<uint16_t, uint16_t>("Prec_BF16_0.1", ACL_BF16, ACL_BF16, prec_bf16, 0, {4096}, stream, 1e-2, 1e-2);
    std::vector<int32_t> prec_int(1000, 3);
    totalFail += RunCumsumV1Test<int32_t, int32_t>("Prec_Int32_Exact", ACL_INT32, ACL_INT32, prec_int, 0, {1000}, stream, 0.0, 0.0);
    std::vector<float> prec_nan = {1.0f, 2.0f, NAN, 4.0f, 5.0f};
    totalFail += RunCumsumV1Test<float, float>("Prec_NaN_Propagation", ACL_FLOAT, ACL_FLOAT, prec_nan, 0, {5}, stream, 1e-6, 1e-6);

    LOG_PRINT("\n>>> 组8：Tiling 策略分支覆盖（极值 M/R/N） <<<\n");
    { std::vector<float> data(64 * 2 * 16, 1.0f); totalFail += RunCumsumV1Test<float, float>("Tiling_NGreaterClRFullLoad_M64_R2_N16", ACL_FLOAT, ACL_FLOAT, data, 1, {64, 2, 16}, stream); }
    { std::vector<float> data(2 * 2 * 16, 1.0f); totalFail += RunCumsumV1Test<float, float>("Tiling_NGreaterClRFullLoad_M2_R2_N16", ACL_FLOAT, ACL_FLOAT, data, 1, {2, 2, 16}, stream); }
    { std::vector<float> data(64 * 200000 * 16, 1.0f); totalFail += RunCumsumV1Test<float, float>("Tiling_NGreaterClRNotFullLoad_M64_R200000_N16", ACL_FLOAT, ACL_FLOAT, data, 1, {64, 200000, 16}, stream); }
    { std::vector<float> data(64 * 1000 * 2, 1.0f); totalFail += RunCumsumV1Test<float, float>("Tiling_RNGreaterClRNotFullLoadNotBorrowR_M64_R1000_N2", ACL_FLOAT, ACL_FLOAT, data, 1, {64, 1000, 2}, stream); }
    { std::vector<uint16_t> data(64 * 32768 * 1, fp32_to_bf16(1.0f)); totalFail += RunCumsumV1Test<uint16_t, uint16_t>("Tiling_TwoWay_BF16_M64_R32768_N1", ACL_BF16, ACL_BF16, data, 1, {64, 32768, 1}, stream, 1e-2, 1e-2); }
    { std::vector<int32_t> data(5000 * 10 * 10, 1); totalFail += RunCumsumV1Test<int32_t, int32_t>("Tiling_Int_AdjustTensor4TDLA", ACL_INT32, ACL_INT32, data, 1, {5000, 10, 10}, stream); }
    { std::vector<int32_t> data(2 * 100 * 1000, 1); totalFail += RunCumsumV1Test<int32_t, int32_t>("Tiling_Int_RA_Weight", ACL_INT32, ACL_INT32, data, 0, {2, 100, 1000}, stream); }
    { std::vector<int32_t> data(1000, 1); totalFail += RunCumsumV1Test<int32_t, int32_t>("Tiling_Int_RBlockAxis", ACL_INT32, ACL_INT32, data, 1, {1, 1000, 1}, stream); }

    LOG_PRINT("\n>>> 组9：错误注入与 AICPU 路径覆盖 <<<\n");
    { std::vector<float> empty_data; totalFail += RunCumsumV1Test<float, float>("Empty_Tensor_Cumsum", ACL_FLOAT, ACL_FLOAT, empty_data, 0, {0, 5}, stream); }
    { std::vector<double> d_data(128 * 256, 1.0); totalFail += RunCumsumV1Test<double, double>("AICPU_DOUBLE_Cumsum", ACL_DOUBLE, ACL_DOUBLE, d_data, 0, {128, 256}, stream, 1e-12, 1e-12); }
    { std::vector<float> large_data(4 * 1000000, 1.0f); totalFail += RunCumsumV1Test<float, float>("Tiling_Large_R", ACL_FLOAT, ACL_FLOAT, large_data, 1, {4, 1000000}, stream); }
    { std::vector<float> wide_data(2 * 10000 * 100, 1.0f); totalFail += RunCumsumV1Test<float, float>("Tiling_Wide_N", ACL_FLOAT, ACL_FLOAT, wide_data, 1, {2, 10000, 100}, stream); }
    { std::vector<std::complex<float>> c_data = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}}; totalFail += RunCumsumV1ComplexTest<float, float>("Complex64_Cumsum", ACL_COMPLEX64, ACL_COMPLEX64, c_data, 0, {3}, stream, 1e-6, 1e-6); }

    LOG_PRINT("\n>>> 组10：更细粒度的 Tiling 对抗 <<<\n");
    { std::vector<uint16_t> data(2 * 200000 * 16, fp32_to_fp16(1.0f)); totalFail += RunCumsumV1Test<uint16_t, uint16_t>("Tiling_BorrowR_FP16", ACL_FLOAT16, ACL_FLOAT16, data, 1, {2, 200000, 16}, stream, 1e-3, 1e-3); }
    { std::vector<float> data(1000 * 2 * 1, 1.0f); totalFail += RunCumsumV1Test<float, float>("Tiling_BorrowM", ACL_FLOAT, ACL_FLOAT, data, 1, {1000, 2, 1}, stream); }

    // ========================================================================
    // 新增补漏项（向100%覆盖率发起的最终冲锋）
    // ========================================================================
    LOG_PRINT("\n>>> 组11：Float 极致 Tiling 分支猎杀 <<<\n");
    // 触发 RNLesserCl: N < 8 且 R*N < 8. 例如 R=3, N=2 => R*N=6 < 8
    {
        std::vector<float> data(2 * 3 * 2, 1.0f);
        totalFail += RunCumsumV1Test<float, float>("Tiling_Float_RNLesserCl",
            ACL_FLOAT, ACL_FLOAT, data, 1, {2, 3, 2}, stream);
    }
    // 触发 M_Borrow 极致情况: M 极大, R=1, N=1
    {
        std::vector<float> data(100000 * 1 * 1, 1.0f);
        totalFail += RunCumsumV1Test<float, float>("Tiling_MBorrow_Ext",
            ACL_FLOAT, ACL_FLOAT, data, 1, {100000, 1, 1}, stream);
    }
    // 触发 N_Borrow 极致情况: N 极大, M=1, R=1
    {
        std::vector<float> data(1 * 1 * 100000, 1.0f);
        totalFail += RunCumsumV1Test<float, float>("Tiling_NBorrow_Ext",
            ACL_FLOAT, ACL_FLOAT, data, 1, {1, 1, 100000}, stream);
    }

    LOG_PRINT("\n>>> 组12：Int 极致 Tiling 切分探索 <<<\n");
    // 触发 AdjustTensor4TDLA (TD leftA分支): 左轴极长，右轴极短
    {
        std::vector<int32_t> data(1000 * 10 * 2, 1);
        totalFail += RunCumsumV1Test<int32_t, int32_t>("Tiling_Int_TDLA_Strict",
            ACL_INT32, ACL_INT32, data, 1, {1000, 10, 2}, stream);
    }
    // 触发 AdjustTensor4TDRA (TD rightA分支): 右轴极长，中轴极短
    {
        std::vector<int32_t> data(2 * 5 * 10000, 1);
        totalFail += RunCumsumV1Test<int32_t, int32_t>("Tiling_Int_TDRA_Strict",
            ACL_INT32, ACL_INT32, data, 1, {2, 5, 10000}, stream);
    }
    // 触发 AdjustTensor4TDR (TD R分支): 右轴中等(不够进LA或RA)，中轴极长
    {
        std::vector<int32_t> data(2 * 10000 * 8, 1);
        totalFail += RunCumsumV1Test<int32_t, int32_t>("Tiling_Int_TDR_Strict",
            ACL_INT32, ACL_INT32, data, 1, {2, 10000, 8}, stream);
    }
    // 触发 AdjustTensor4TDNone (TD None全长分支): 所有轴均等且较大
    {
        std::vector<int32_t> data(100 * 100 * 100, 1);
        totalFail += RunCumsumV1Test<int32_t, int32_t>("Tiling_Int_TDNone_Strict",
            ACL_INT32, ACL_INT32, data, 1, {100, 100, 100}, stream);
    }
    // 最小整型张量
    {
        std::vector<int32_t> data(1 * 1 * 1, 1);
        totalFail += RunCumsumV1Test<int32_t, int32_t>("Tiling_Int_Min_Shape",
            ACL_INT32, ACL_INT32, data, 0, {1, 1, 1}, stream);
    }
    
    // ================================================================
    // 组13：终极覆盖率补漏 (gcov 驱动精准打击)
    // ================================================================
    LOG_PRINT("\n>>> 组13：终极覆盖率补漏 (API Fallback 与 Sklansky 极限树) <<<\n");

    // 13.1 API 补漏：V2 API 触发 AICPU 路径 (DOUBLE 类型)
    // 覆盖 cumsum.cpp line 104
    {
        std::vector<double> v2_cpu(100, 1.0);
        totalFail += RunCumsumV2Test<double, double>("V2_AICPU_DOUBLE",
            ACL_DOUBLE, ACL_DOUBLE, v2_cpu, 0, false, false, {100}, stream, 1e-12, 1e-12);
    }

    // 13.2 API 补漏：V2 完全空 Tensor
    // 覆盖 aclnn_cumsum.cpp 空算子拦截分支
    {
        std::vector<float> v2_empty;
        totalFail += RunCumsumV2Test<float, float>("V2_Empty_Tensor_Strict",
            ACL_FLOAT, ACL_FLOAT, v2_empty, 0, false, false, {0}, stream);
    }

    // 13.3 Float Tiling: NGreaterClRFullLoad & SS_ONEWAY
    // 覆盖: N极大(>cl), R极小(全载), M大(切核)
    // Shape: M=64, R=2, N=1024 -> lenM=64, lenR=2, lenN=1024
    {
        std::vector<float> data(64 * 2 * 1024, 1.0f);
        totalFail += RunCumsumV1Test<float, float>("Float_NGreater_RFull_SS1Way",
            ACL_FLOAT, ACL_FLOAT, data, 1, {64, 2, 1024}, stream);
    }

    // 13.4 Float Tiling: 借轴 N 分支 (borrowNCount_)
    // 覆盖: M极小(<coreNum)，N极大，R不能全载
    // Shape: M=2, R=100000, N=512 -> lenM=2, lenR=100000, lenN=512
    {
        std::vector<float> data(2 * 100000 * 512, 1.0f);
        totalFail += RunCumsumV1Test<float, float>("Float_BorrowN_Ext",
            ACL_FLOAT, ACL_FLOAT, data, 1, {2, 100000, 512}, stream);
    }

    // 13.5 Float Tiling: 不借轴 (M大)，N适中 (走SS_ONEWAY)，R大
    // 覆盖: RNGreaterClRNotFullLoadNotBorrowR + TILING_KEY_UB_SS_ONEWAY
    // Shape: M=64, R=50000, N=256 -> lenM=64, lenR=50000, lenN=256
    {
        std::vector<float> data(64 * 50000 * 256, 1.0f);
        totalFail += RunCumsumV1Test<float, float>("Float_NoBorrowR_SS1Way",
            ACL_FLOAT, ACL_FLOAT, data, 1, {64, 50000, 256}, stream);
    }

    // 13.6 Float Tiling: 不借轴 (M大)，N极小 (走SS_TWOWAY)，R大
    // 覆盖: RNGreaterClRNotFullLoadNotBorrowR + TILING_KEY_UB_SS_TWOWAY
    // Shape: M=64, R=50000, N=2 -> lenM=64, lenR=50000, lenN=2
    {
        std::vector<uint16_t> data(64 * 50000 * 2, fp32_to_fp16(1.0f));
        totalFail += RunCumsumV1Test<uint16_t, uint16_t>("Float_NoBorrowR_SS2Way_FP16",
            ACL_FLOAT16, ACL_FLOAT16, data, 1, {64, 50000, 2}, stream, 1e-2, 1e-2);
    }

    // 13.7 Float Tiling: 借轴 R (M极小)，走 ONEWAY
    // 覆盖: TILING_KEY_CORE_SS_ONEWAY (borrowRCount_ > 1)
    // Shape: M=1, R=200000, N=256 -> lenM=1, lenR=200000, lenN=256
    {
        std::vector<float> data(1 * 200000 * 256, 1.0f);
        totalFail += RunCumsumV1Test<float, float>("Float_BorrowR_SS1Way",
            ACL_FLOAT, ACL_FLOAT, data, 1, {1, 200000, 256}, stream);
    }

    // 13.8 Float Tiling: 借轴 R (M极小)，走 TWOWAY
    // 覆盖: 双向 Sklansky Tree 在极大 R 下的缓存分块
    // Shape: M=1, R=2000000, N=1
    {
        std::vector<float> data(1 * 2000000 * 1, 1.0f);
        totalFail += RunCumsumV1Test<float, float>("Float_BorrowR_SS2Way",
            ACL_FLOAT, ACL_FLOAT, data, 1, {1, 2000000, 1}, stream);
    }

    // 13.9 Int Tiling: INT64 触发 vlSize /= 2 以及 AdjustLARLpUnit
    // 覆盖: cumsum_tiling_ascendc_int_arch35.cpp line 49, line 173-185
    // Shape: L大，R小，RA适中 -> [10000, 10, 200]
    {
        std::vector<int64_t> data(10000 * 10 * 200, 1);
        totalFail += RunCumsumV1Test<int64_t, int64_t>("Int_Tiling_INT64_AdjustLA",
            ACL_INT64, ACL_INT64, data, 1, {10000, 10, 200}, stream);
    }

    // 13.10 Int Tiling: INT8 触发 AdjustTensor4TDR 的极值
    // 覆盖: cumsum_tiling_ascendc_int_arch35.cpp line 116 (tensorSize_ = std::min...)
    // Shape: L极小，R极大，RA极小 -> [1, 200000, 2]
    {
        std::vector<int8_t> data(1 * 200000 * 2, 2);
        totalFail += RunCumsumV1Test<int8_t, int8_t>("Int_Tiling_INT8_AdjustTDR",
            ACL_INT8, ACL_INT8, data, 1, {1, 200000, 2}, stream);
    }

    LOG_PRINT("\n=== Total failed cases: %d ===\n", totalFail);
    
    aclDestroyTensor(d_fp32); aclDestroyTensor(d_fp16); aclDestroyTensor(d_shape_bad);
    aclDestroyTensor(d_9d); aclDestroyTensor(d_massive);
    aclDestroyTensor(d_cube1); aclDestroyTensor(d_cube2);
    aclDestroyTensor(d_cube_bad_dim); aclDestroyTensor(d_cube_bad_ch); aclDestroyTensor(d_cube_low_batch);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return totalFail;
}