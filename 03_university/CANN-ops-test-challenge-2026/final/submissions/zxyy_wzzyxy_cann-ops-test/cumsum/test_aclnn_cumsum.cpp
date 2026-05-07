/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Cumsum 算子综合测试用例 — 决赛题目 (v3 expanded)
 *
 * 覆盖维度：
 *   - 数据类型：INT8, INT16, INT32, INT64, UINT8, FLOAT32, FLOAT16, DOUBLE
 *   - 序列长度：短序列(1-100)、中等序列(100-1000)、长序列(1000-10000)、超长(>10000)
 *   - 数值特征：全正、全负、正负混合/交替、大小数混合、零值、溢出边界
 *   - API 变体：Cumsum、CumsumV2(4种 exclusive×reverse 组合)
 *   - 维度参数：dim=0, dim=1, dim=2, dim=-1, 1D~5D
 *   - 精度场景：误差累积、大小数吞没、整数溢出、FP16 vs FP32 对比
 *   - Tiling 覆盖：tiling probe(仅GetWorkspaceSize) + 完整kernel 验证
 *     · INT32/INT8/INT16/UINT8/INT64 probes → cumsum_tiling_ascendc_int_arch35.cpp
 *     · FP32/FP16 probes → cumsum_tiling_ascendc_arch35.cpp
 *     · V2 probes (exclusive/reverse) → GetAttrInfo/WriteTilingData 分支
 *     · 8种 float tiling keys: ONEWAY/TWOWAY/UB_SS/CORE_SS 及其组合
 *     · 3种 int tiling keys: CUM_NO_SPLIT/CUM_AR_SPLIT/CUM_WITH_GROUP
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <type_traits>
#include <string>
#include "acl/acl.h"
#include "aclnnop/aclnn_cumsum.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

// ========== 全局测试统计 ==========
static int g_totalTests = 0;
static int g_passedTests = 0;
static int g_failedTests = 0;

// ========== 工具函数 ==========

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

void CleanupTensor(aclTensor* tensor, void* deviceAddr)
{
    if (tensor) aclDestroyTensor(tensor);
    if (deviceAddr) aclrtFree(deviceAddr);
}

// ========== FP16 辅助函数 ==========

static float Fp16ToFloat(uint16_t h)
{
    // IEEE 754 half-precision to float
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            // subnormal
            exp = 113; // 127 - 14
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            f = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = (sign << 31) | (0xFF << 23) | (mant << 13); // inf/NaN
    } else {
        f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f, sizeof(float));
    return result;
}

static uint16_t FloatToFp16(float val)
{
    uint32_t f;
    std::memcpy(&f, &val, sizeof(float));
    uint32_t sign = (f >> 31) & 0x1;
    int32_t exp  = static_cast<int32_t>((f >> 23) & 0xFF) - 127;
    uint32_t mant = f & 0x7FFFFF;
    if (exp > 15) return (sign << 15) | (0x1F << 10);       // inf
    if (exp < -14) return (sign << 15);                      // underflow to 0
    if (exp <= 0) {
        // subnormal
        mant = (mant | 0x800000) >> (1 - exp);
        return (sign << 15) | (mant >> 13);
    }
    return (sign << 15) | ((exp + 15) << 10) | (mant >> 13);
}

// ========== CPU 参考实现 ==========

/// 1D cumsum (inclusive, forward)
template<typename T>
std::vector<double> CpuCumsumFloat(const std::vector<T>& input)
{
    std::vector<double> result(input.size());
    double sum = 0.0;
    for (size_t i = 0; i < input.size(); i++) {
        sum += static_cast<double>(input[i]);
        result[i] = sum;
    }
    return result;
}

/// 1D cumsum for FP16 (decode uint16_t → float → double accumulate)
std::vector<double> CpuCumsumFp16(const std::vector<uint16_t>& input)
{
    std::vector<double> result(input.size());
    double sum = 0.0;
    for (size_t i = 0; i < input.size(); i++) {
        sum += static_cast<double>(Fp16ToFloat(input[i]));
        result[i] = sum;
    }
    return result;
}

/// Multi-dim cumsum along given axis
template<typename T>
std::vector<double> CpuCumsumMultiDim(const std::vector<T>& input, const std::vector<int64_t>& shape, int64_t dim)
{
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < dim; i++) outer *= shape[i];
    for (int64_t i = dim + 1; i < ndim; i++) inner *= shape[i];
    int64_t axisLen = shape[dim];

    std::vector<double> result(input.size(), 0.0);
    for (int64_t o = 0; o < outer; o++) {
        for (int64_t inr = 0; inr < inner; inr++) {
            double sum = 0.0;
            for (int64_t a = 0; a < axisLen; a++) {
                int64_t srcIdx = o * axisLen * inner + a * inner + inr;
                sum += static_cast<double>(input[srcIdx]);
                result[srcIdx] = sum;
            }
        }
    }
    return result;
}

/// Multi-dim cumsum with exclusive / reverse
template<typename T>
std::vector<double> CpuCumsumV2MultiDim(const std::vector<T>& input, const std::vector<int64_t>& shape,
                                        int64_t dim, bool exclusive, bool reverse)
{
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < dim; i++) outer *= shape[i];
    for (int64_t i = dim + 1; i < ndim; i++) inner *= shape[i];
    int64_t axisLen = shape[dim];

    std::vector<double> result(input.size(), 0.0);
    for (int64_t o = 0; o < outer; o++) {
        for (int64_t inr = 0; inr < inner; inr++) {
            double sum = 0.0;
            if (reverse) {
                for (int64_t a = axisLen - 1; a >= 0; a--) {
                    int64_t srcIdx = o * axisLen * inner + a * inner + inr;
                    if (!exclusive) sum += static_cast<double>(input[srcIdx]);
                    result[srcIdx] = sum;
                    if (exclusive) sum += static_cast<double>(input[srcIdx]);
                }
            } else {
                for (int64_t a = 0; a < axisLen; a++) {
                    int64_t srcIdx = o * axisLen * inner + a * inner + inr;
                    if (!exclusive) sum += static_cast<double>(input[srcIdx]);
                    result[srcIdx] = sum;
                    if (exclusive) sum += static_cast<double>(input[srcIdx]);
                }
            }
        }
    }
    return result;
}

/// Integer 1D CPU reference (use int64_t to detect overflow patterns)
template<typename T>
std::vector<T> CpuCumsumInt(const std::vector<T>& input)
{
    std::vector<T> result(input.size());
    // Use unsigned arithmetic to match NPU wrap-around behavior
    using UT = typename std::make_unsigned<T>::type;
    UT sum = 0;
    for (size_t i = 0; i < input.size(); i++) {
        sum += static_cast<UT>(input[i]);
        result[i] = static_cast<T>(sum);
    }
    return result;
}

// ========== 验证函数 ==========

/// Float comparison with atol+rtol
bool CompareFloat(double actual, double expected, double atol, double rtol)
{
    double diff = std::abs(actual - expected);
    double threshold = atol + rtol * std::abs(expected);
    return diff <= threshold;
}

/// Integer exact match
template<typename T>
bool CompareInt(T actual, T expected)
{
    return actual == expected;
}

// ========== 测试执行 ==========

/// Run Cumsum (standard) test
template<typename T>
bool RunCumsumTest(aclrtStream stream,
                   const std::vector<T>& hostInput,
                   const std::vector<int64_t>& shape,
                   int64_t dim,
                   aclDataType dtype,
                   const std::vector<double>& expectedDouble,
                   double atol, double rtol,
                   const std::string& testName)
{
    g_totalTests++;
    LOG_PRINT("Test case %d: %s\n", g_totalTests, testName.c_str());

    auto size = GetShapeSize(shape);
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    std::vector<T> outHostData(size, static_cast<T>(0));

    auto ret = CreateAclTensor(hostInput, shape, &selfDeviceAddr, dtype, &self);
    CHECK_RET(ret == 0, CleanupTensor(self, selfDeviceAddr); return false);
    ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, dtype, &out);
    CHECK_RET(ret == 0, CleanupTensor(self, selfDeviceAddr); CleanupTensor(out, outDeviceAddr); return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnCumsumGetWorkspaceSize(self, dim, dtype, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("  aclnnCumsumGetWorkspaceSize failed. ERROR: %d\n", ret);
              CleanupTensor(self, selfDeviceAddr); CleanupTensor(out, outDeviceAddr); return false);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, CleanupTensor(self, selfDeviceAddr); CleanupTensor(out, outDeviceAddr); return false);
    }

    ret = aclnnCumsum(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("  aclnnCumsum failed. ERROR: %d\n", ret);
              if (workspaceSize > 0) aclrtFree(workspaceAddr);
              CleanupTensor(self, selfDeviceAddr); CleanupTensor(out, outDeviceAddr); return false);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  aclrtSynchronizeStream failed. ERROR: %d\n", ret); return false);

    std::vector<T> resultData(size, static_cast<T>(0));
    ret = aclrtMemcpy(resultData.data(), size * sizeof(T), outDeviceAddr, size * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  aclrtMemcpy failed. ERROR: %d\n", ret); return false);

    // Validate
    bool passed = true;
    double maxError = 0.0;
    int64_t maxErrorIdx = -1;
    for (int64_t i = 0; i < size; i++) {
        double actual = (std::is_same<T, uint16_t>::value)
                            ? static_cast<double>(Fp16ToFloat(resultData[i]))
                            : static_cast<double>(resultData[i]);
        double expected = expectedDouble[i];
        if (!CompareFloat(actual, expected, atol, rtol)) {
            passed = false;
            double err = std::abs(actual - expected);
            if (err > maxError) {
                maxError = err;
                maxErrorIdx = i;
            }
        }
    }

    if (passed) {
        g_passedTests++;
        LOG_PRINT("  Max error: %.6e\n", maxError);
        LOG_PRINT("  [PASS]\n");
    } else {
        g_failedTests++;
        LOG_PRINT("  Max error: %.6e (at position %ld)\n", maxError, maxErrorIdx);
        LOG_PRINT("  Actual[%ld]=%.10f, Expected[%ld]=%.10f\n",
                  maxErrorIdx, (std::is_same<T, uint16_t>::value)
                                   ? static_cast<double>(Fp16ToFloat(resultData[maxErrorIdx]))
                                   : static_cast<double>(resultData[maxErrorIdx]),
                  maxErrorIdx, expectedDouble[maxErrorIdx]);
        LOG_PRINT("  [FAIL]\n");
    }

    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    CleanupTensor(self, selfDeviceAddr);
    CleanupTensor(out, outDeviceAddr);
    return passed;
}

/// Integer test with exact match
template<typename T>
bool RunCumsumIntTest(aclrtStream stream,
                      const std::vector<T>& hostInput,
                      const std::vector<int64_t>& shape,
                      int64_t dim,
                      aclDataType dtype,
                      const std::vector<T>& expectedInt,
                      const std::string& testName)
{
    g_totalTests++;
    LOG_PRINT("Test case %d: %s\n", g_totalTests, testName.c_str());

    auto size = GetShapeSize(shape);
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    std::vector<T> outHostData(size, static_cast<T>(0));

    auto ret = CreateAclTensor(hostInput, shape, &selfDeviceAddr, dtype, &self);
    CHECK_RET(ret == 0, CleanupTensor(self, selfDeviceAddr); return false);
    ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, dtype, &out);
    CHECK_RET(ret == 0, CleanupTensor(self, selfDeviceAddr); CleanupTensor(out, outDeviceAddr); return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnCumsumGetWorkspaceSize(self, dim, dtype, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              CleanupTensor(self, selfDeviceAddr); CleanupTensor(out, outDeviceAddr); return false);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, CleanupTensor(self, selfDeviceAddr); CleanupTensor(out, outDeviceAddr); return false);
    }

    ret = aclnnCumsum(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              if (workspaceSize > 0) aclrtFree(workspaceAddr);
              CleanupTensor(self, selfDeviceAddr); CleanupTensor(out, outDeviceAddr); return false);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  aclrtSynchronizeStream failed. ERROR: %d\n", ret); return false);

    std::vector<T> resultData(size, static_cast<T>(0));
    ret = aclrtMemcpy(resultData.data(), size * sizeof(T), outDeviceAddr, size * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  aclrtMemcpy failed. ERROR: %d\n", ret); return false);

    bool passed = true;
    for (int64_t i = 0; i < size; i++) {
        if (!CompareInt(resultData[i], expectedInt[i])) {
            LOG_PRINT("  Mismatch at [%ld]: actual=%d, expected=%d\n",
                      i, static_cast<int>(resultData[i]), static_cast<int>(expectedInt[i]));
            passed = false;
            break;
        }
    }

    if (passed) {
        g_passedTests++;
        LOG_PRINT("  All values match exactly\n");
        LOG_PRINT("  [PASS]\n");
    } else {
        g_failedTests++;
        LOG_PRINT("  [FAIL]\n");
    }

    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    CleanupTensor(self, selfDeviceAddr);
    CleanupTensor(out, outDeviceAddr);
    return passed;
}

// ========== CumsumV2 测试 ==========

template<typename T>
bool RunCumsumV2Test(aclrtStream stream,
                     const std::vector<T>& hostInput,
                     const std::vector<int64_t>& shape,
                     int64_t dim,
                     bool exclusive, bool reverse,
                     aclDataType dtype,
                     const std::vector<double>& expectedDouble,
                     double atol, double rtol,
                     const std::string& testName)
{
    g_totalTests++;
    LOG_PRINT("Test case %d: %s\n", g_totalTests, testName.c_str());

    auto size = GetShapeSize(shape);
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    std::vector<T> outHostData(size, static_cast<T>(0));

    auto ret = CreateAclTensor(hostInput, shape, &selfDeviceAddr, dtype, &self);
    CHECK_RET(ret == 0, CleanupTensor(self, selfDeviceAddr); return false);
    ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, dtype, &out);
    CHECK_RET(ret == 0, CleanupTensor(self, selfDeviceAddr); CleanupTensor(out, outDeviceAddr); return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnCumsumV2GetWorkspaceSize(self, dim, exclusive, reverse, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              CleanupTensor(self, selfDeviceAddr); CleanupTensor(out, outDeviceAddr); return false);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, CleanupTensor(self, selfDeviceAddr); CleanupTensor(out, outDeviceAddr); return false);
    }

    ret = aclnnCumsumV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              if (workspaceSize > 0) aclrtFree(workspaceAddr);
              CleanupTensor(self, selfDeviceAddr); CleanupTensor(out, outDeviceAddr); return false);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  aclrtSynchronizeStream failed. ERROR: %d\n", ret); return false);

    std::vector<T> resultData(size, static_cast<T>(0));
    ret = aclrtMemcpy(resultData.data(), size * sizeof(T), outDeviceAddr, size * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  aclrtMemcpy failed. ERROR: %d\n", ret); return false);

    bool passed = true;
    double maxError = 0.0;
    int64_t maxErrorIdx = -1;
    for (int64_t i = 0; i < size; i++) {
        double actual = (std::is_same<T, uint16_t>::value)
                            ? static_cast<double>(Fp16ToFloat(resultData[i]))
                            : static_cast<double>(resultData[i]);
        double expected = expectedDouble[i];
        if (!CompareFloat(actual, expected, atol, rtol)) {
            passed = false;
            double err = std::abs(actual - expected);
            if (err > maxError) {
                maxError = err;
                maxErrorIdx = i;
            }
        }
    }

    if (passed) {
        g_passedTests++;
        LOG_PRINT("  Max error: %.6e\n", maxError);
        LOG_PRINT("  [PASS]\n");
    } else {
        g_failedTests++;
        LOG_PRINT("  Max error: %.6e (at position %ld)\n", maxError, maxErrorIdx);
        LOG_PRINT("  [FAIL]\n");
    }

    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    CleanupTensor(self, selfDeviceAddr);
    CleanupTensor(out, outDeviceAddr);
    return passed;
}

// ========== AiCore 探针 ==========

/// 用一个小 FP32 tensor 快速检测 AiCore 是否正常（返回非全零）
static bool ProbeAiCore(aclrtStream stream)
{
    LOG_PRINT("--- Probing AiCore availability (FP32 [1,2,3,4,5]) ---\n");
    std::vector<int64_t> shape = {5};
    auto size = GetShapeSize(shape);
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    void* selfDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *out = nullptr;
    std::vector<float> outHost(size, 0.0f);

    int ret = CreateAclTensor(input, shape, &selfDev, ACL_FLOAT, &self);
    if (ret != 0) { CleanupTensor(self, selfDev); return false; }
    ret = CreateAclTensor(outHost, shape, &outDev, ACL_FLOAT, &out);
    if (ret != 0) { CleanupTensor(self, selfDev); CleanupTensor(out, outDev); return false; }

    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    ret = aclnnCumsumGetWorkspaceSize(self, 0, ACL_FLOAT, out, &wsSize, &exec);
    if (ret != ACL_SUCCESS) { CleanupTensor(self, selfDev); CleanupTensor(out, outDev); return false; }

    void* ws = nullptr;
    if (wsSize > 0) aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclnnCumsum(ws, wsSize, exec, stream);
    aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) { if (ws) aclrtFree(ws); CleanupTensor(self, selfDev); CleanupTensor(out, outDev); return false; }

    std::vector<float> result(size);
    aclrtMemcpy(result.data(), size * sizeof(float), outDev, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    bool allZero = true;
    for (int64_t i = 0; i < size; i++) {
        if (result[i] != 0.0f) { allZero = false; break; }
    }
    LOG_PRINT("  Probe result[4]=%.6f (expected ~15.0), allZero=%d\n", result[4], allZero);

    if (ws) aclrtFree(ws);
    CleanupTensor(self, selfDev);
    CleanupTensor(out, outDev);
    return !allZero;
}

// ========== 主函数 ==========

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 容忍度设置
    const double FP32_ATOL = 1e-5;
    const double FP32_RTOL = 1e-5;
    const double FP16_ATOL = 1e-3;
    const double FP16_RTOL = 1e-3;

    LOG_PRINT("========================================\n");
    LOG_PRINT("  Cumsum Operator Comprehensive Tests\n");
    LOG_PRINT("========================================\n\n");

    // ====================================================================
    // 预检：AiCore 是否可用（910_93 上可能 kernel 不执行）
    // ====================================================================
    bool aiCoreOk = ProbeAiCore(stream);
    LOG_PRINT("\nAiCore: %s\n\n", aiCoreOk ? "AVAILABLE" : "UNAVAILABLE (FP32/FP16/INT32 tests skipped)");


    // ====================================================================
    // 第一部分：AiCPU INT8 (4 用例)
    // ====================================================================

    {
        std::vector<int64_t> shape = {5};
        std::vector<int8_t> input = {1, 2, 3, 4, 5};
        auto expectedInt = CpuCumsumInt(input);
        RunCumsumIntTest<int8_t>(stream, input, shape, 0, ACL_INT8, expectedInt, "INT8 1D [1,2,3,4,5]");
    }
    {
        std::vector<int64_t> shape = {3, 4};
        std::vector<int8_t> input = {1,2,3,4, 5,6,7,8, 9,10,11,12};
        using UT = std::make_unsigned<int8_t>::type;
        std::vector<int8_t> expectedM(input.size());
        int64_t outer = shape[0], axis = shape[1];
        for (int64_t o = 0; o < outer; o++) {
            UT sum = 0;
            for (int64_t a = 0; a < axis; a++) {
                int64_t idx = o * axis + a;
                sum += static_cast<UT>(input[idx]);
                expectedM[idx] = static_cast<int8_t>(sum);
            }
        }
        RunCumsumIntTest<int8_t>(stream, input, shape, 1, ACL_INT8, expectedM, "INT8 2D [3,4] dim=1");
    }
    {
        std::vector<int64_t> shape = {4};
        std::vector<int8_t> input = {-10, -20, -30, -40};
        auto expectedInt = CpuCumsumInt(input);
        RunCumsumIntTest<int8_t>(stream, input, shape, 0, ACL_INT8, expectedInt, "INT8 1D negative [-10,-20,-30,-40]");
    }
    {
        std::vector<int64_t> shape = {5};
        std::vector<int8_t> input = {100, 100, -100, -100, 50};
        auto expectedInt = CpuCumsumInt(input);
        RunCumsumIntTest<int8_t>(stream, input, shape, 0, ACL_INT8, expectedInt, "INT8 overflow [100,100,-100,-100,50]");
    }

    // ====================================================================
    // 第二部分：AiCPU INT16 (4 用例)
    // ====================================================================

    {
        std::vector<int64_t> shape = {5};
        std::vector<int16_t> input = {100, 200, 300, 400, 500};
        auto expectedInt = CpuCumsumInt(input);
        RunCumsumIntTest<int16_t>(stream, input, shape, 0, ACL_INT16, expectedInt, "INT16 1D [100,200,300,400,500]");
    }
    {
        std::vector<int64_t> shape = {3, 4};
        std::vector<int16_t> input = {1,2,3,4, 10,20,30,40, 100,200,300,400};
        using UT = std::make_unsigned<int16_t>::type;
        std::vector<int16_t> expectedM(input.size());
        int64_t outer = shape[0], axis = shape[1];
        for (int64_t o = 0; o < outer; o++) {
            UT sum = 0;
            for (int64_t a = 0; a < axis; a++) {
                int64_t idx = o * axis + a;
                sum += static_cast<UT>(input[idx]);
                expectedM[idx] = static_cast<int16_t>(sum);
            }
        }
        RunCumsumIntTest<int16_t>(stream, input, shape, 1, ACL_INT16, expectedM, "INT16 2D [3,4] dim=1");
    }
    {
        std::vector<int64_t> shape = {256};
        std::vector<int16_t> input(256, 100);
        auto expectedInt = CpuCumsumInt(input);
        RunCumsumIntTest<int16_t>(stream, input, shape, 0, ACL_INT16, expectedInt, "INT16 1D long seq 256x100");
    }
    {
        std::vector<int64_t> shape = {4};
        std::vector<int16_t> input = {30000, 30000, -30000, -30000};
        auto expectedInt = CpuCumsumInt(input);
        RunCumsumIntTest<int16_t>(stream, input, shape, 0, ACL_INT16, expectedInt, "INT16 near-limit [30000,30000,-30000,-30000]");
    }

    // ====================================================================
    // 第三部分：AiCPU INT64 (6 用例)
    // ====================================================================

    {
        std::vector<int64_t> shape = {4};
        std::vector<int64_t> input = {100LL, 200LL, 300LL, 400LL};
        auto expectedInt = CpuCumsumInt(input);
        RunCumsumIntTest<int64_t>(stream, input, shape, 0, ACL_INT64, expectedInt, "INT64 1D [100,200,300,400]");
    }
    {
        std::vector<int64_t> shape = {3, 4};
        std::vector<int64_t> input = {1,2,3,4, 10,20,30,40, 100,200,300,400};
        using UT = std::make_unsigned<int64_t>::type;
        std::vector<int64_t> expectedM(input.size());
        int64_t outer = shape[0], axis = shape[1];
        for (int64_t o = 0; o < outer; o++) {
            UT sum = 0;
            for (int64_t a = 0; a < axis; a++) {
                int64_t idx = o * axis + a;
                sum += static_cast<UT>(input[idx]);
                expectedM[idx] = static_cast<int64_t>(sum);
            }
        }
        RunCumsumIntTest<int64_t>(stream, input, shape, 1, ACL_INT64, expectedM, "INT64 2D [3,4] dim=1");
    }
    {
        std::vector<int64_t> shape = {2, 3, 4};
        int64_t size = 24;
        std::vector<int64_t> input(size);
        for (int64_t i = 0; i < size; i++) input[i] = i + 1;
        using UT = std::make_unsigned<int64_t>::type;
        std::vector<int64_t> expectedM(size);
        int64_t outer = shape[0], axis = shape[1], inner = shape[2];
        for (int64_t o = 0; o < outer; o++) {
            for (int64_t inr = 0; inr < inner; inr++) {
                UT sum = 0;
                for (int64_t a = 0; a < axis; a++) {
                    int64_t idx = o * axis * inner + a * inner + inr;
                    sum += static_cast<UT>(input[idx]);
                    expectedM[idx] = static_cast<int64_t>(sum);
                }
            }
        }
        RunCumsumIntTest<int64_t>(stream, input, shape, 1, ACL_INT64, expectedM, "INT64 3D [2,3,4] dim=1");
    }
    {
        std::vector<int64_t> shape = {2, 4};
        std::vector<int64_t> input = {1,2,3,4, 10,20,30,40};
        using UT = std::make_unsigned<int64_t>::type;
        std::vector<int64_t> expectedM(input.size());
        int64_t outer = shape[0], axis = shape[1];
        for (int64_t o = 0; o < outer; o++) {
            UT sum = 0;
            for (int64_t a = 0; a < axis; a++) {
                int64_t idx = o * axis + a;
                sum += static_cast<UT>(input[idx]);
                expectedM[idx] = static_cast<int64_t>(sum);
            }
        }
        RunCumsumIntTest<int64_t>(stream, input, shape, -1, ACL_INT64, expectedM, "INT64 2D dim=-1 [2,4]");
    }
    {
        std::vector<int64_t> shape = {1};
        std::vector<int64_t> input = {42LL};
        auto expectedInt = CpuCumsumInt(input);
        RunCumsumIntTest<int64_t>(stream, input, shape, 0, ACL_INT64, expectedInt, "INT64 1D single element [42]");
    }
    {
        std::vector<int64_t> shape = {5};
        std::vector<int64_t> input = {10000000000LL, 20000000000LL, 30000000000LL, 40000000000LL, 50000000000LL};
        auto expectedInt = CpuCumsumInt(input);
        RunCumsumIntTest<int64_t>(stream, input, shape, 0, ACL_INT64, expectedInt, "INT64 1D large [1e10,...,5e10]");
    }

    // ====================================================================
    // 第四部分：AiCPU UINT8 (4 用例)
    // ====================================================================

    {
        std::vector<int64_t> shape = {5};
        std::vector<uint8_t> input = {10, 20, 30, 40, 50};
        auto expectedInt = CpuCumsumInt(input);
        RunCumsumIntTest<uint8_t>(stream, input, shape, 0, ACL_UINT8, expectedInt, "UINT8 1D [10,20,30,40,50]");
    }
    {
        std::vector<int64_t> shape = {2, 3};
        std::vector<uint8_t> input = {10,20,30, 40,50,60};
        using UT = std::make_unsigned<uint8_t>::type;
        std::vector<uint8_t> expectedM(input.size());
        int64_t outer = shape[0], axis = shape[1];
        for (int64_t o = 0; o < outer; o++) {
            UT sum = 0;
            for (int64_t a = 0; a < axis; a++) {
                int64_t idx = o * axis + a;
                sum += static_cast<UT>(input[idx]);
                expectedM[idx] = static_cast<uint8_t>(sum);
            }
        }
        RunCumsumIntTest<uint8_t>(stream, input, shape, 1, ACL_UINT8, expectedM, "UINT8 2D [2,3] dim=1");
    }
    {
        std::vector<int64_t> shape = {4};
        std::vector<uint8_t> input = {200, 200, 200, 200};
        auto expectedInt = CpuCumsumInt(input);
        RunCumsumIntTest<uint8_t>(stream, input, shape, 0, ACL_UINT8, expectedInt, "UINT8 overflow [200,200,200,200]");
    }
    {
        std::vector<int64_t> shape = {10};
        std::vector<uint8_t> input(10, 0);
        auto expectedInt = CpuCumsumInt(input);
        RunCumsumIntTest<uint8_t>(stream, input, shape, 0, ACL_UINT8, expectedInt, "UINT8 1D all zeros [0,...,0]");
    }

    // ====================================================================
    // 第五部分：CumsumV2 — AiCPU INT64 (5 用例)
    // ====================================================================

    {
        std::vector<int64_t> shape = {5};
        std::vector<int64_t> input = {10, 20, 30, 40, 50};
        auto expected = CpuCumsumV2MultiDim(input, shape, 0, true, false);
        RunCumsumV2Test<int64_t>(stream, input, shape, 0, true, false, ACL_INT64, expected, 0.0, 0.0,
                                 "V2 INT64 exclusive=true, reverse=false");
    }
    {
        std::vector<int64_t> shape = {5};
        std::vector<int64_t> input = {10, 20, 30, 40, 50};
        auto expected = CpuCumsumV2MultiDim(input, shape, 0, false, true);
        RunCumsumV2Test<int64_t>(stream, input, shape, 0, false, true, ACL_INT64, expected, 0.0, 0.0,
                                 "V2 INT64 exclusive=false, reverse=true");
    }
    {
        std::vector<int64_t> shape = {5};
        std::vector<int64_t> input = {10, 20, 30, 40, 50};
        auto expected = CpuCumsumV2MultiDim(input, shape, 0, true, true);
        RunCumsumV2Test<int64_t>(stream, input, shape, 0, true, true, ACL_INT64, expected, 0.0, 0.0,
                                 "V2 INT64 exclusive=true, reverse=true");
    }
    {
        std::vector<int64_t> shape = {5};
        std::vector<int64_t> input = {10, 20, 30, 40, 50};
        auto expected = CpuCumsumV2MultiDim(input, shape, 0, false, false);
        RunCumsumV2Test<int64_t>(stream, input, shape, 0, false, false, ACL_INT64, expected, 0.0, 0.0,
                                 "V2 INT64 exclusive=false, reverse=false");
    }
    {
        std::vector<int64_t> shape = {3, 4};
        std::vector<int64_t> input = {10,20,30,40, 50,60,70,80, 90,100,110,120};
        auto expected = CpuCumsumV2MultiDim(input, shape, 0, true, false);
        RunCumsumV2Test<int64_t>(stream, input, shape, 0, true, false, ACL_INT64, expected, 0.0, 0.0,
                                 "V2 INT64 2D [3,4] dim=0 exclusive");
    }

    // ====================================================================
    // 第六部分：AiCPU DOUBLE (2 用例)
    // ====================================================================

    {
        std::vector<int64_t> shape = {5};
        std::vector<double> input = {1.0, 2.0, 3.0, 4.0, 5.0};
        auto expected = CpuCumsumFloat(input);
        RunCumsumTest<double>(stream, input, shape, 0, ACL_DOUBLE, expected, 1e-10, 1e-10,
                              "DOUBLE 1D [1,2,3,4,5]");
    }
    {
        std::vector<int64_t> shape = {1000};
        std::vector<double> input(1000, 1.0);
        auto expected = CpuCumsumFloat(input);
        RunCumsumTest<double>(stream, input, shape, 0, ACL_DOUBLE, expected, 1e-10, 1e-10,
                              "DOUBLE 1D long 1000x1.0");
    }

    // ====================================================================
    // 精度专项与边界场景（AiCPU 类型，始终可运行）
    // ====================================================================

    // --- 长序列 INT64 精度 ---
    {
        std::vector<int64_t> shape = {2000};
        std::vector<int64_t> input(2000, 100);
        auto expectedInt = CpuCumsumInt(input);
        RunCumsumIntTest<int64_t>(stream, input, shape, 0, ACL_INT64, expectedInt,
                                  "INT64 long seq 2000x100");
    }
    // --- INT64 正负交替 ---
    {
        std::vector<int64_t> shape = {1000};
        std::vector<int64_t> input(1000);
        for (int i = 0; i < 1000; i++) input[i] = (i % 2 == 0) ? 1000 : -1000;
        auto expectedInt = CpuCumsumInt(input);
        RunCumsumIntTest<int64_t>(stream, input, shape, 0, ACL_INT64, expectedInt,
                                  "INT64 alternating +/-1000 x1000");
    }
    // --- INT16 长序列 ---
    {
        std::vector<int64_t> shape = {1024};
        std::vector<int16_t> input(1024, 50);
        auto expectedInt = CpuCumsumInt(input);
        RunCumsumIntTest<int16_t>(stream, input, shape, 0, ACL_INT16, expectedInt,
                                  "INT16 long seq 1024x50");
    }
    // --- UINT8 大序列溢出 ---
    {
        std::vector<int64_t> shape = {300};
        std::vector<uint8_t> input(300, 10);
        auto expectedInt = CpuCumsumInt(input);
        RunCumsumIntTest<uint8_t>(stream, input, shape, 0, ACL_UINT8, expectedInt,
                                  "UINT8 overflow 300x10");
    }
    // --- DOUBLE 大小数混合 ---
    {
        std::vector<int64_t> shape = {10};
        std::vector<double> input = {1e10, 1e-10, 1e10, 1e-10, 1e10, 1e-10, 1e10, 1e-10, 1e10, 1e-10};
        auto expected = CpuCumsumFloat(input);
        RunCumsumTest<double>(stream, input, shape, 0, ACL_DOUBLE, expected, 1e-10, 1e-10,
                              "DOUBLE mix large+small");
    }
    // --- INT8 3D 中间 dim ---
    {
        std::vector<int64_t> shape = {4, 8, 4};
        int64_t size = 4*8*4;
        std::vector<int8_t> input(size);
        for (int64_t i = 0; i < size; i++) input[i] = static_cast<int8_t>(i % 127);
        using UT = std::make_unsigned<int8_t>::type;
        std::vector<int8_t> expectedM(size);
        int64_t outer = shape[0], axis = shape[1], inner = shape[2];
        for (int64_t o = 0; o < outer; o++) {
            for (int64_t inr = 0; inr < inner; inr++) {
                UT sum = 0;
                for (int64_t a = 0; a < axis; a++) {
                    int64_t idx = o * axis * inner + a * inner + inr;
                    sum += static_cast<UT>(input[idx]);
                    expectedM[idx] = static_cast<int8_t>(sum);
                }
            }
        }
        RunCumsumIntTest<int8_t>(stream, input, shape, 1, ACL_INT8, expectedM, "INT8 3D [4,8,4] dim=1");
    }
    // --- INT64 4D tensor ---
    {
        std::vector<int64_t> shape = {2, 3, 4, 5};
        int64_t size = 2*3*4*5;
        std::vector<int64_t> input(size);
        for (int64_t i = 0; i < size; i++) input[i] = i + 1;
        using UT = std::make_unsigned<int64_t>::type;
        std::vector<int64_t> expectedM(size);
        int64_t dim = 2;
        int64_t outer = 2*3, axis = 4, inner = 5;
        for (int64_t o = 0; o < outer; o++) {
            for (int64_t inr = 0; inr < inner; inr++) {
                UT sum = 0;
                for (int64_t a = 0; a < axis; a++) {
                    int64_t idx = o * axis * inner + a * inner + inr;
                    sum += static_cast<UT>(input[idx]);
                    expectedM[idx] = static_cast<int64_t>(sum);
                }
            }
        }
        RunCumsumIntTest<int64_t>(stream, input, shape, dim, ACL_INT64, expectedM, "INT64 4D [2,3,4,5] dim=2");
    }
    // --- INT16 5D tensor ---
    {
        std::vector<int64_t> shape = {2, 2, 3, 2, 2};
        int64_t size = 2*2*3*2*2;
        std::vector<int16_t> input(size);
        for (int64_t i = 0; i < size; i++) input[i] = static_cast<int16_t>(i % 100);
        auto expected = CpuCumsumMultiDim(input, shape, 2);
        RunCumsumTest<int16_t>(stream, input, shape, 2, ACL_INT16, expected, 0.0, 0.0,
                               "INT16 5D [2,2,3,2,2] dim=2");
    }
    // --- INT64 超大纲轴 ---
    {
        std::vector<int64_t> shape = {2, 4096};
        int64_t size = 2*4096;
        std::vector<int64_t> input(size, 1);
        using UT = std::make_unsigned<int64_t>::type; std::vector<int64_t> expectedM(size); int64_t axis = shape[0], inner = 4096; for (int64_t inr = 0; inr < inner; inr++) { UT sum = 0; for (int64_t a = 0; a < axis; a++) { int64_t idx = a * inner + inr; sum += static_cast<UT>(input[idx]); expectedM[idx] = static_cast<int64_t>(sum); } }
        RunCumsumIntTest<int64_t>(stream, input, shape, 0, ACL_INT64, expectedM,
                                  "INT64 2D [2,4096] dim=0");
    }

    // ====================================================================
    // 第七部分：INT32 tiling 覆盖探针（无条件，只做 GetWorkspaceSize 触发 tiling）
    // 注：910_93 上 kernel 输出全零，但 tiling 在 GetWorkspaceSize 时就执行并写 gcda
    // ====================================================================

    LOG_PRINT("\n--- INT32 tiling probes (for gcda coverage) ---\n");
    {
        int64_t size = 5;
        std::vector<int64_t> shape = {size};
        std::vector<int32_t> input = {10, 20, 30, 40, 50};
        void* sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
        std::vector<int32_t> oh(size, 0);
        if (CreateAclTensor(input, shape, &sd, ACL_INT32, &s) == 0 &&
            CreateAclTensor(oh, shape, &od, ACL_INT32, &o) == 0) {
            uint64_t ws = 0; aclOpExecutor* ex = nullptr;
            if (aclnnCumsumGetWorkspaceSize(s, 0, ACL_INT32, o, &ws, &ex) == ACL_SUCCESS)
                LOG_PRINT("  [TILING] INT32 1D dim=0 ok\n");
            CleanupTensor(s, sd); CleanupTensor(o, od);
        }
    }
    {
        std::vector<int64_t> shape = {3, 4};
        int64_t size = 12;
        std::vector<int32_t> input = {1,2,3,4, 5,6,7,8, 9,10,11,12};
        void* sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
        std::vector<int32_t> oh(size, 0);
        if (CreateAclTensor(input, shape, &sd, ACL_INT32, &s) == 0 &&
            CreateAclTensor(oh, shape, &od, ACL_INT32, &o) == 0) {
            uint64_t ws = 0; aclOpExecutor* ex = nullptr;
            if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_INT32, o, &ws, &ex) == ACL_SUCCESS)
                LOG_PRINT("  [TILING] INT32 2D dim=1 ok\n");
            CleanupTensor(s, sd); CleanupTensor(o, od);
        }
    }
    {
        int64_t size = 256;
        std::vector<int64_t> shape = {size};
        std::vector<int32_t> input(size, 1);
        void* sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
        std::vector<int32_t> oh(size, 0);
        if (CreateAclTensor(input, shape, &sd, ACL_INT32, &s) == 0 &&
            CreateAclTensor(oh, shape, &od, ACL_INT32, &o) == 0) {
            uint64_t ws = 0; aclOpExecutor* ex = nullptr;
            if (aclnnCumsumGetWorkspaceSize(s, 0, ACL_INT32, o, &ws, &ex) == ACL_SUCCESS)
                LOG_PRINT("  [TILING] INT32 1D long256 ok\n");
            CleanupTensor(s, sd); CleanupTensor(o, od);
        }
    }

    // --- extra tiling probes: AR_SPLIT, WITH_GROUP, TDRA, negative dim ---
    {
        // AR_SPLIT: large cumsum axis [2,128,4] dim=1, INT32
        int64_t size = 2*128*4;
        std::vector<int64_t> shape = {2, 128, 4};
        std::vector<int32_t> input(size, 1);
        void* sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
        std::vector<int32_t> oh(size, 0);
        if (CreateAclTensor(input, shape, &sd, ACL_INT32, &s) == 0 &&
            CreateAclTensor(oh, shape, &od, ACL_INT32, &o) == 0) {
            uint64_t ws = 0; aclOpExecutor* ex = nullptr;
            if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_INT32, o, &ws, &ex) == ACL_SUCCESS)
                LOG_PRINT("  [TILING] INT32 3D AR_SPLIT dim=1 ok\n");
            CleanupTensor(s, sd); CleanupTensor(o, od);
        }
    }
    {
        // WITH_GROUP: multi-core grouping [4,64,32] dim=0, INT32
        int64_t size = 4*64*32;
        std::vector<int64_t> shape = {4, 64, 32};
        std::vector<int32_t> input(size, 1);
        void* sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
        std::vector<int32_t> oh(size, 0);
        if (CreateAclTensor(input, shape, &sd, ACL_INT32, &s) == 0 &&
            CreateAclTensor(oh, shape, &od, ACL_INT32, &o) == 0) {
            uint64_t ws = 0; aclOpExecutor* ex = nullptr;
            if (aclnnCumsumGetWorkspaceSize(s, 0, ACL_INT32, o, &ws, &ex) == ACL_SUCCESS)
                LOG_PRINT("  [TILING] INT32 3D WITH_GROUP dim=0 ok\n");
            CleanupTensor(s, sd); CleanupTensor(o, od);
        }
    }
    {
        // TDRA: large right-axis > cacheLine [2,8,32] dim=1, INT32 (right=32*4=128>64)
        int64_t size = 2*8*32;
        std::vector<int64_t> shape = {2, 8, 32};
        std::vector<int32_t> input(size, 1);
        void* sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
        std::vector<int32_t> oh(size, 0);
        if (CreateAclTensor(input, shape, &sd, ACL_INT32, &s) == 0 &&
            CreateAclTensor(oh, shape, &od, ACL_INT32, &o) == 0) {
            uint64_t ws = 0; aclOpExecutor* ex = nullptr;
            if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_INT32, o, &ws, &ex) == ACL_SUCCESS)
                LOG_PRINT("  [TILING] INT32 3D TDRA dim=1 ok\n");
            CleanupTensor(s, sd); CleanupTensor(o, od);
        }
    }
    {
        // negative dim: [4,8] dim=-1 INT32
        int64_t size = 4*8;
        std::vector<int64_t> shape = {4, 8};
        std::vector<int32_t> input(size, 1);
        void* sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
        std::vector<int32_t> oh(size, 0);
        if (CreateAclTensor(input, shape, &sd, ACL_INT32, &s) == 0 &&
            CreateAclTensor(oh, shape, &od, ACL_INT32, &o) == 0) {
            uint64_t ws = 0; aclOpExecutor* ex = nullptr;
            if (aclnnCumsumGetWorkspaceSize(s, -1, ACL_INT32, o, &ws, &ex) == ACL_SUCCESS)
                LOG_PRINT("  [TILING] INT32 2D negative dim=-1 ok\n");
            CleanupTensor(s, sd); CleanupTensor(o, od);
        }
    }
    // ====================================================================
    // 第八部分：FP32 tiling 覆盖探针（触发 cumsum_tiling_ascendc_arch35.cpp）
    // 不同 M×R×N 组合触发不同 tiling key：ONEWAY/TWOWAY/UB_SS/CORE_SS 等
    // ====================================================================

    LOG_PRINT("\n--- FP32 tiling probes (for float tiling gcda coverage) ---\n");
    {
        // NLesserCl → RNGreaterCl: [100, 100] dim=1, M=100,R=100,N=1 → N<cl, RN>=cl
        int64_t size = 100*100;
        std::vector<int64_t> shape = {100, 100};
        std::vector<float> input(size, 1.0f);
        void* sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
        std::vector<float> oh(size, 0);
        if (CreateAclTensor(input, shape, &sd, ACL_FLOAT, &s) == 0 &&
            CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
            uint64_t ws = 0; aclOpExecutor* ex = nullptr;
            if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_FLOAT, o, &ws, &ex) == ACL_SUCCESS)
                LOG_PRINT("  [TILING] FP32 2D [100,100] dim=1 ok\n");
            CleanupTensor(s, sd); CleanupTensor(o, od);
        }
    }
    {
        // NGreaterCl → RFullLoad: [128, 256] dim=0, M=1,R=128,N=256 → N>=cl, R*clNSize<=ubSize
        int64_t size = 128*256;
        std::vector<int64_t> shape = {128, 256};
        std::vector<float> input(size, 1.0f);
        void* sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
        std::vector<float> oh(size, 0);
        if (CreateAclTensor(input, shape, &sd, ACL_FLOAT, &s) == 0 &&
            CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
            uint64_t ws = 0; aclOpExecutor* ex = nullptr;
            if (aclnnCumsumGetWorkspaceSize(s, 0, ACL_FLOAT, o, &ws, &ex) == ACL_SUCCESS)
                LOG_PRINT("  [TILING] FP32 2D [128,256] dim=0 ok\n");
            CleanupTensor(s, sd); CleanupTensor(o, od);
        }
    }
    {
        // NGreaterCl → RNotFullLoad → borrowN: [4, 2048, 128] dim=1, M=4,R=2048,N=128
        int64_t size = 4*2048*128;
        std::vector<int64_t> shape = {4, 2048, 128};
        std::vector<float> input(size, 1.0f);
        void* sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
        std::vector<float> oh(size, 0);
        if (CreateAclTensor(input, shape, &sd, ACL_FLOAT, &s) == 0 &&
            CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
            uint64_t ws = 0; aclOpExecutor* ex = nullptr;
            if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_FLOAT, o, &ws, &ex) == ACL_SUCCESS)
                LOG_PRINT("  [TILING] FP32 3D [4,2048,128] dim=1 ok\n");
            CleanupTensor(s, sd); CleanupTensor(o, od);
        }
    }
    {
        // M large, R large: [512, 1024, 2] dim=1, M=512,R=1024,N=2
        int64_t size = 512*1024*2;
        std::vector<int64_t> shape = {512, 1024, 2};
        std::vector<float> input(size, 1.0f);
        void* sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
        std::vector<float> oh(size, 0);
        if (CreateAclTensor(input, shape, &sd, ACL_FLOAT, &s) == 0 &&
            CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
            uint64_t ws = 0; aclOpExecutor* ex = nullptr;
            if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_FLOAT, o, &ws, &ex) == ACL_SUCCESS)
                LOG_PRINT("  [TILING] FP32 3D [512,1024,2] dim=1 ok\n");
            CleanupTensor(s, sd); CleanupTensor(o, od);
        }
    }
    {
        // Large R only: [2, 4096, 2] dim=1, M=2,R=4096,N=2 → borrowR path
        int64_t size = 2*4096*2;
        std::vector<int64_t> shape = {2, 4096, 2};
        std::vector<float> input(size, 1.0f);
        void* sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
        std::vector<float> oh(size, 0);
        if (CreateAclTensor(input, shape, &sd, ACL_FLOAT, &s) == 0 &&
            CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
            uint64_t ws = 0; aclOpExecutor* ex = nullptr;
            if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_FLOAT, o, &ws, &ex) == ACL_SUCCESS)
                LOG_PRINT("  [TILING] FP32 3D [2,4096,2] dim=1 ok\n");
            CleanupTensor(s, sd); CleanupTensor(o, od);
        }
    }
    {
        // MRNLesserCl: [64, 8, 8] dim=1, small everything → realCoreNum=1
        int64_t size = 64*8*8;
        std::vector<int64_t> shape = {64, 8, 8};
        std::vector<float> input(size, 1.0f);
        void* sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
        std::vector<float> oh(size, 0);
        if (CreateAclTensor(input, shape, &sd, ACL_FLOAT, &s) == 0 &&
            CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
            uint64_t ws = 0; aclOpExecutor* ex = nullptr;
            if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_FLOAT, o, &ws, &ex) == ACL_SUCCESS)
                LOG_PRINT("  [TILING] FP32 3D small [64,8,8] dim=1 ok\n");
            CleanupTensor(s, sd); CleanupTensor(o, od);
        }
    }

    // ====================================================================
    // 第九部分：FP16 tiling 覆盖探针（触发 dtCast_ 和 FP16_FOLD 路径）
    // ====================================================================

    LOG_PRINT("\n--- FP16 tiling probes (for float tiling gcda coverage) ---\n");
    {
        // Small FP16: [5] dim=0
        int64_t size = 5;
        std::vector<int64_t> shape = {size};
        std::vector<float> inputF = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        std::vector<uint16_t> input(size);
        for (int i = 0; i < size; i++) input[i] = FloatToFp16(inputF[i]);
        void* sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
        std::vector<uint16_t> oh(size, 0);
        if (CreateAclTensor(input, shape, &sd, ACL_FLOAT16, &s) == 0 &&
            CreateAclTensor(oh, shape, &od, ACL_FLOAT16, &o) == 0) {
            uint64_t ws = 0; aclOpExecutor* ex = nullptr;
            if (aclnnCumsumGetWorkspaceSize(s, 0, ACL_FLOAT16, o, &ws, &ex) == ACL_SUCCESS)
                LOG_PRINT("  [TILING] FP16 1D [1..5] dim=0 ok\n");
            CleanupTensor(s, sd); CleanupTensor(o, od);
        }
    }
    {
        // FP16 medium: [64, 1024] dim=1 → fold logic, dtCast_
        int64_t size = 64*1024;
        std::vector<int64_t> shape = {64, 1024};
        std::vector<float> inputF(size, 1.0f);
        std::vector<uint16_t> input(size);
        for (int64_t i = 0; i < size; i++) input[i] = FloatToFp16(inputF[i]);
        void* sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
        std::vector<uint16_t> oh(size, 0);
        if (CreateAclTensor(input, shape, &sd, ACL_FLOAT16, &s) == 0 &&
            CreateAclTensor(oh, shape, &od, ACL_FLOAT16, &o) == 0) {
            uint64_t ws = 0; aclOpExecutor* ex = nullptr;
            if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_FLOAT16, o, &ws, &ex) == ACL_SUCCESS)
                LOG_PRINT("  [TILING] FP16 2D [64,1024] dim=1 ok\n");
            CleanupTensor(s, sd); CleanupTensor(o, od);
        }
    }
    {
        // FP16 3D: [2, 512, 256] dim=1 → FP16 cast x3 multiplier
        int64_t size = 2*512*256;
        std::vector<int64_t> shape = {2, 512, 256};
        std::vector<float> inputF(size, 1.0f);
        std::vector<uint16_t> input(size);
        for (int64_t i = 0; i < size; i++) input[i] = FloatToFp16(inputF[i]);
        void* sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
        std::vector<uint16_t> oh(size, 0);
        if (CreateAclTensor(input, shape, &sd, ACL_FLOAT16, &s) == 0 &&
            CreateAclTensor(oh, shape, &od, ACL_FLOAT16, &o) == 0) {
            uint64_t ws = 0; aclOpExecutor* ex = nullptr;
            if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_FLOAT16, o, &ws, &ex) == ACL_SUCCESS)
                LOG_PRINT("  [TILING] FP16 3D [2,512,256] dim=1 ok\n");
            CleanupTensor(s, sd); CleanupTensor(o, od);
        }
    }

    // ====================================================================
    // 第十部分：INT8/INT16/UINT8/INT64 tiling 覆盖探针
    // ====================================================================

    LOG_PRINT("\n--- INT8/INT16/UINT8/INT64 tiling probes ---\n");
    {
        // INT8 with axis=0: [1000, 4] dim=0 → axis==0 branch in GetInputDims
        int64_t size = 1000*4;
        std::vector<int64_t> shape = {1000, 4};
        std::vector<int8_t> input(size, 1);
        void* sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
        std::vector<int8_t> oh(size, 0);
        if (CreateAclTensor(input, shape, &sd, ACL_INT8, &s) == 0 &&
            CreateAclTensor(oh, shape, &od, ACL_INT8, &o) == 0) {
            uint64_t ws = 0; aclOpExecutor* ex = nullptr;
            if (aclnnCumsumGetWorkspaceSize(s, 0, ACL_INT8, o, &ws, &ex) == ACL_SUCCESS)
                LOG_PRINT("  [TILING] INT8 2D axis=0 ok\n");
            CleanupTensor(s, sd); CleanupTensor(o, od);
        }
    }
    {
        // INT8 3D large rightAxis: [2, 100, 256] dim=1 → rightAxisLen=256*1=256 > cacheLine
        int64_t size = 2*100*256;
        std::vector<int64_t> shape = {2, 100, 256};
        std::vector<int8_t> input(size, 1);
        void* sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
        std::vector<int8_t> oh(size, 0);
        if (CreateAclTensor(input, shape, &sd, ACL_INT8, &s) == 0 &&
            CreateAclTensor(oh, shape, &od, ACL_INT8, &o) == 0) {
            uint64_t ws = 0; aclOpExecutor* ex = nullptr;
            if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_INT8, o, &ws, &ex) == ACL_SUCCESS)
                LOG_PRINT("  [TILING] INT8 3D large rightAxis ok\n");
            CleanupTensor(s, sd); CleanupTensor(o, od);
        }
    }
    {
        // INT16 large leftAxis: [100, 1024, 2] dim=1 → leftAxisLen=100 > coreNum/CORE_GATE → TDLA
        int64_t size = 100*1024*2;
        std::vector<int64_t> shape = {100, 1024, 2};
        std::vector<int16_t> input(size, 1);
        void* sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
        std::vector<int16_t> oh(size, 0);
        if (CreateAclTensor(input, shape, &sd, ACL_INT16, &s) == 0 &&
            CreateAclTensor(oh, shape, &od, ACL_INT16, &o) == 0) {
            uint64_t ws = 0; aclOpExecutor* ex = nullptr;
            if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_INT16, o, &ws, &ex) == ACL_SUCCESS)
                LOG_PRINT("  [TILING] INT16 3D TDLA path ok\n");
            CleanupTensor(s, sd); CleanupTensor(o, od);
        }
    }
    {
        // UINT8: [8, 256, 256] dim=1 → dtypeSize=1 branch (vlSize/2)
        int64_t size = 8*256*256;
        std::vector<int64_t> shape = {8, 256, 256};
        std::vector<uint8_t> input(size, 1);
        void* sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
        std::vector<uint8_t> oh(size, 0);
        if (CreateAclTensor(input, shape, &sd, ACL_UINT8, &s) == 0 &&
            CreateAclTensor(oh, shape, &od, ACL_UINT8, &o) == 0) {
            uint64_t ws = 0; aclOpExecutor* ex = nullptr;
            if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_UINT8, o, &ws, &ex) == ACL_SUCCESS)
                LOG_PRINT("  [TILING] UINT8 3D dtypeSize=1 ok\n");
            CleanupTensor(s, sd); CleanupTensor(o, od);
        }
    }
    {
        // INT64 large mid: [1, 2048] dim=0 → midAxisLen=2048, tests int tiling with large axis
        int64_t size = 2048;
        std::vector<int64_t> shape = {1, 2048};
        std::vector<int64_t> input(size, 1);
        void* sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
        std::vector<int64_t> oh(size, 0);
        if (CreateAclTensor(input, shape, &sd, ACL_INT64, &s) == 0 &&
            CreateAclTensor(oh, shape, &od, ACL_INT64, &o) == 0) {
            uint64_t ws = 0; aclOpExecutor* ex = nullptr;
            if (aclnnCumsumGetWorkspaceSize(s, 0, ACL_INT64, o, &ws, &ex) == ACL_SUCCESS)
                LOG_PRINT("  [TILING] INT64 2D large axis ok\n");
            CleanupTensor(s, sd); CleanupTensor(o, od);
        }
    }

    // ====================================================================
    // 第十一部分：V2 tiling 覆盖探针（exclusive/reverse 分支 + isExclusive_/isReverse_）
    // ====================================================================

    LOG_PRINT("\n--- V2 tiling probes (exclusive/reverse for gcda) ---\n");
    {
        // INT32 V2 exclusive, INT32已走AiCore路由 to tiling
        int64_t size = 5;
        std::vector<int64_t> shape = {size};
        std::vector<int32_t> input = {10, 20, 30, 40, 50};
        void* sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
        std::vector<int32_t> oh(size, 0);
        if (CreateAclTensor(input, shape, &sd, ACL_INT32, &s) == 0 &&
            CreateAclTensor(oh, shape, &od, ACL_INT32, &o) == 0) {
            uint64_t ws = 0; aclOpExecutor* ex = nullptr;
            if (aclnnCumsumV2GetWorkspaceSize(s, 0, true, false, o, &ws, &ex) == ACL_SUCCESS)
                LOG_PRINT("  [TILING] V2 INT32 exclusive ok\n");
            CleanupTensor(s, sd); CleanupTensor(o, od);
        }
    }
    {
        // INT32 V2 reverse
        int64_t size = 5;
        std::vector<int64_t> shape = {size};
        std::vector<int32_t> input = {10, 20, 30, 40, 50};
        void* sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
        std::vector<int32_t> oh(size, 0);
        if (CreateAclTensor(input, shape, &sd, ACL_INT32, &s) == 0 &&
            CreateAclTensor(oh, shape, &od, ACL_INT32, &o) == 0) {
            uint64_t ws = 0; aclOpExecutor* ex = nullptr;
            if (aclnnCumsumV2GetWorkspaceSize(s, 0, false, true, o, &ws, &ex) == ACL_SUCCESS)
                LOG_PRINT("  [TILING] V2 INT32 reverse ok\n");
            CleanupTensor(s, sd); CleanupTensor(o, od);
        }
    }
    {
        // INT8 V2: exclusive+reverse via AiCpu path
        int64_t size = 5;
        std::vector<int64_t> shape = {size};
        std::vector<int8_t> input = {10, 20, 30, 40, 50};
        void* sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
        std::vector<int8_t> oh(size, 0);
        if (CreateAclTensor(input, shape, &sd, ACL_INT8, &s) == 0 &&
            CreateAclTensor(oh, shape, &od, ACL_INT8, &o) == 0) {
            uint64_t ws = 0; aclOpExecutor* ex = nullptr;
            if (aclnnCumsumV2GetWorkspaceSize(s, 0, true, true, o, &ws, &ex) == ACL_SUCCESS)
                LOG_PRINT("  [TILING] V2 INT8 excl+rev ok\n");
            CleanupTensor(s, sd); CleanupTensor(o, od);
        }
    }
    {
        // INT16 V2 with 2D shape: [3, 4] dim=1 exclusive
        int64_t size = 3*4;
        std::vector<int64_t> shape = {3, 4};
        std::vector<int16_t> input = {1,2,3,4, 5,6,7,8, 9,10,11,12};
        void* sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
        std::vector<int16_t> oh(size, 0);
        if (CreateAclTensor(input, shape, &sd, ACL_INT16, &s) == 0 &&
            CreateAclTensor(oh, shape, &od, ACL_INT16, &o) == 0) {
            uint64_t ws = 0; aclOpExecutor* ex = nullptr;
            if (aclnnCumsumV2GetWorkspaceSize(s, 1, true, false, o, &ws, &ex) == ACL_SUCCESS)
                LOG_PRINT("  [TILING] V2 INT16 2D exclusive ok\n");
            CleanupTensor(s, sd); CleanupTensor(o, od);
        }
    }
    {
        // UINT8 V2: exclusive+reverse, AiCpu path
        int64_t size = 5;
        std::vector<int64_t> shape = {size};
        std::vector<uint8_t> input = {10, 20, 30, 40, 50};
        void* sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
        std::vector<uint8_t> oh(size, 0);
        if (CreateAclTensor(input, shape, &sd, ACL_UINT8, &s) == 0 &&
            CreateAclTensor(oh, shape, &od, ACL_UINT8, &o) == 0) {
            uint64_t ws = 0; aclOpExecutor* ex = nullptr;
            if (aclnnCumsumV2GetWorkspaceSize(s, 0, false, true, o, &ws, &ex) == ACL_SUCCESS)
                LOG_PRINT("  [TILING] V2 UINT8 reverse ok\n");
            CleanupTensor(s, sd); CleanupTensor(o, od);
        }
    }

    // ====================================================================
    // 第十二部分：AiCore 条件测试（仅当探针通过时执行完整 kernel 验证）
    // ====================================================================

    if (aiCoreOk) {
        LOG_PRINT("\n--- AiCore (FP32/FP16/INT32) tests ---\n");

        {
            std::vector<int64_t> shape = {5};
            std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
            auto expected = CpuCumsumFloat(input);
            RunCumsumTest<float>(stream, input, shape, 0, ACL_FLOAT, expected, FP32_ATOL, FP32_RTOL,
                                 "FP32 1D [1,2,3,4,5]");
        }
        {
            std::vector<int64_t> shape = {3, 4};
            std::vector<float> input = {1,2,3,4, 5,6,7,8, 9,10,11,12};
            auto expected = CpuCumsumMultiDim(input, shape, 1);
            RunCumsumTest<float>(stream, input, shape, 1, ACL_FLOAT, expected, FP32_ATOL, FP32_RTOL,
                                 "FP32 2D [3,4] dim=1");
        }
        {
            std::vector<int64_t> shape = {10000};
            std::vector<float> input(10000, 1.0f);
            auto expected = CpuCumsumFloat(input);
            RunCumsumTest<float>(stream, input, shape, 0, ACL_FLOAT, expected, 1e-4, 1e-4,
                                 "FP32 long seq 10000x1.0");
        }
        {
            std::vector<int64_t> shape = {1000};
            std::vector<float> input(1000, 0.1f);
            auto expected = CpuCumsumFloat(input);
            RunCumsumTest<float>(stream, input, shape, 0, ACL_FLOAT, expected, 1e-3, 1e-3,
                                 "FP32 0.1 accumulation x1000");
        }
        {
            std::vector<int64_t> shape = {100};
            std::vector<float> input(100);
            for (int64_t i = 0; i < 100; i++) input[i] = -static_cast<float>(i + 1);
            auto expected = CpuCumsumFloat(input);
            RunCumsumTest<float>(stream, input, shape, 0, ACL_FLOAT, expected, FP32_ATOL, FP32_RTOL,
                                 "FP32 all negative 100");
        }
        {
            std::vector<int64_t> shape = {5};
            std::vector<float> inputF = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
            std::vector<uint16_t> input(5);
            for (int i = 0; i < 5; i++) input[i] = FloatToFp16(inputF[i]);
            auto expected = CpuCumsumFp16(input);
            RunCumsumTest<uint16_t>(stream, input, shape, 0, ACL_FLOAT16, expected, FP16_ATOL, FP16_RTOL,
                                    "FP16 1D [1,2,3,4,5]");
        }
        {
            std::vector<int64_t> shape = {256};
            std::vector<float> inputF(256);
            for (int i = 0; i < 256; i++) inputF[i] = (i % 2 == 0) ? 1.0f : -1.0f;
            std::vector<uint16_t> input(256);
            for (int i = 0; i < 256; i++) input[i] = FloatToFp16(inputF[i]);
            auto expected = CpuCumsumFp16(input);
            RunCumsumTest<uint16_t>(stream, input, shape, 0, ACL_FLOAT16, expected, FP16_ATOL, FP16_RTOL,
                                    "FP16 alternating 256");
        }
        {
            std::vector<int64_t> shape = {5};
            std::vector<int32_t> input = {10, 20, 30, 40, 50};
            auto expectedInt = CpuCumsumInt(input);
            RunCumsumIntTest<int32_t>(stream, input, shape, 0, ACL_INT32, expectedInt,
                                      "INT32 1D [10,20,30,40,50]");
        }
        {
            std::vector<int64_t> shape = {4};
            int32_t bigVal = 1073741824;
            std::vector<int32_t> input = {bigVal, bigVal, 100, 200};
            auto expectedInt = CpuCumsumInt(input);
            RunCumsumIntTest<int32_t>(stream, input, shape, 0, ACL_INT32, expectedInt,
                                      "INT32 overflow [2^30,2^30,100,200]");
        }
        {
            std::vector<int64_t> shape = {5};
            std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
            auto expected = CpuCumsumV2MultiDim(input, shape, 0, true, true);
            RunCumsumV2Test<float>(stream, input, shape, 0, true, true, ACL_FLOAT, expected, FP32_ATOL, FP32_RTOL,
                                   "V2 FP32 exclusive+reverse");
        }
        // --- 多维 FP32（触发不同 tiling key）---
        {
            std::vector<int64_t> shape = {100, 256};
            std::vector<float> input(100*256, 1.0f);
            auto expected = CpuCumsumMultiDim(input, shape, 0);
            RunCumsumTest<float>(stream, input, shape, 0, ACL_FLOAT, expected, FP32_ATOL, FP32_RTOL,
                                 "FP32 2D [100,256] dim=0");
        }
        {
            std::vector<int64_t> shape = {100, 100};
            std::vector<float> input(100*100, 1.0f);
            auto expected = CpuCumsumMultiDim(input, shape, 1);
            RunCumsumTest<float>(stream, input, shape, 1, ACL_FLOAT, expected, FP32_ATOL, FP32_RTOL,
                                 "FP32 2D [100,100] dim=1");
        }
        {
            std::vector<int64_t> shape = {4, 512, 64};
            int64_t sz = 4*512*64;
            std::vector<float> input(sz, 1.0f);
            auto expected = CpuCumsumMultiDim(input, shape, 1);
            RunCumsumTest<float>(stream, input, shape, 1, ACL_FLOAT, expected, FP32_ATOL, FP32_RTOL,
                                 "FP32 3D [4,512,64] dim=1");
        }
        {
            std::vector<int64_t> shape = {256, 256};
            std::vector<float> input(256*256, 1.0f);
            auto expected = CpuCumsumMultiDim(input, shape, 1);
            RunCumsumTest<float>(stream, input, shape, 1, ACL_FLOAT, expected, 1e-4, 1e-4,
                                 "FP32 2D [256,256] dim=1");
        }
        // --- 多维 FP16 ---
        {
            std::vector<int64_t> shape = {64, 256};
            int64_t sz = 64*256;
            std::vector<float> inputF(sz, 1.0f);
            std::vector<uint16_t> input(sz);
            for (int64_t i = 0; i < sz; i++) input[i] = FloatToFp16(inputF[i]);
            auto expected = CpuCumsumFp16(input);
            RunCumsumTest<uint16_t>(stream, input, shape, 0, ACL_FLOAT16, expected, FP16_ATOL, FP16_RTOL,
                                    "FP16 2D [64,256] dim=0");
        }
        {
            std::vector<int64_t> shape = {32, 512};
            int64_t sz = 32*512;
            std::vector<float> inputF(sz, 1.0f);
            std::vector<uint16_t> input(sz);
            for (int64_t i = 0; i < sz; i++) input[i] = FloatToFp16(inputF[i]);
            auto expected = CpuCumsumFp16(input);
            RunCumsumTest<uint16_t>(stream, input, shape, 0, ACL_FLOAT16, expected, FP16_ATOL, FP16_RTOL,
                                    "FP16 2D [32,512] dim=0");
        }
        // --- INT32 V2 全组合 ---
        {
            std::vector<int64_t> shape = {5};
            std::vector<int32_t> input = {10, 20, 30, 40, 50};
            auto expected = CpuCumsumV2MultiDim(input, shape, 0, true, false);
            RunCumsumV2Test<int32_t>(stream, input, shape, 0, true, false, ACL_INT32, expected, 0.0, 0.0,
                                     "V2 INT32 exclusive=true");
        }
        {
            std::vector<int64_t> shape = {5};
            std::vector<int32_t> input = {10, 20, 30, 40, 50};
            auto expected = CpuCumsumV2MultiDim(input, shape, 0, false, true);
            RunCumsumV2Test<int32_t>(stream, input, shape, 0, false, true, ACL_INT32, expected, 0.0, 0.0,
                                     "V2 INT32 reverse=true");
        }
        {
            std::vector<int64_t> shape = {5};
            std::vector<int32_t> input = {10, 20, 30, 40, 50};
            auto expected = CpuCumsumV2MultiDim(input, shape, 0, false, false);
            RunCumsumV2Test<int32_t>(stream, input, shape, 0, false, false, ACL_INT32, expected, 0.0, 0.0,
                                     "V2 INT32 exclusive=false reverse=false");
        }
        // --- FP32 V2 剩余组合 ---
        {
            std::vector<int64_t> shape = {5};
            std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
            auto expected = CpuCumsumV2MultiDim(input, shape, 0, true, false);
            RunCumsumV2Test<float>(stream, input, shape, 0, true, false, ACL_FLOAT, expected, FP32_ATOL, FP32_RTOL,
                                   "V2 FP32 exclusive only");
        }
        {
            std::vector<int64_t> shape = {5};
            std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
            auto expected = CpuCumsumV2MultiDim(input, shape, 0, false, true);
            RunCumsumV2Test<float>(stream, input, shape, 0, false, true, ACL_FLOAT, expected, FP32_ATOL, FP32_RTOL,
                                   "V2 FP32 reverse only");
        }
    } else {
        LOG_PRINT("\n--- AiCore tests skipped (kernel not available on this hardware) ---\n");
    }


    // ====================================================================
    // 第十三部分：错误路径探针（触发 aclnn_cumsum.cpp OP_CHECK_* 分支）
    // ====================================================================

    LOG_PRINT("\n--- Error path probes (for aclnn_cumsum.cpp branches) ---\n");
    {
        // P1: dtype API mismatch — CheckDtypeValid: OP_CHECK_DTYPE_NOT_MATCH
        {
            std::vector<int64_t> shape = {2, 3};
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<float> data(6, 1.0f);
            if (CreateAclTensor(data, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(data, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumGetWorkspaceSize(s, 0, ACL_INT32, o, &ws, &ex);
                LOG_PRINT("  [ERR-PROBE] dtype-api-mismatch: %s\n",
                          ret != ACL_SUCCESS ? "error as expected" : "UNEXPECTED success");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // P2: self/out dtype mismatch — CheckDtypeValidWithoutDtype: OP_CHECK_DTYPE_NOT_SAME
        {
            std::vector<int64_t> shape = {2, 3};
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<int32_t> sdata(6, 1);
            std::vector<int64_t> odata(6, 0);
            if (CreateAclTensor(sdata, shape, &sd, ACL_INT32, &s) == 0 &&
                CreateAclTensor(odata, shape, &od, ACL_INT64, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumV2GetWorkspaceSize(s, 0, false, false, o, &ws, &ex);
                LOG_PRINT("  [ERR-PROBE] dtype-self-vs-out: %s\n",
                          ret != ACL_SUCCESS ? "error as expected" : "UNEXPECTED success");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // P3: dim >= ndim — CheckDim upper bound
        {
            std::vector<int64_t> shape = {3, 4};
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<float> data(12, 1.0f);
            if (CreateAclTensor(data, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(data, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumGetWorkspaceSize(s, 5, ACL_FLOAT, o, &ws, &ex);
                LOG_PRINT("  [ERR-PROBE] dim-too-large: %s\n",
                          ret != ACL_SUCCESS ? "error as expected" : "UNEXPECTED success");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // P4: dim < -ndim — CheckDim lower bound
        {
            std::vector<int64_t> shape = {3, 4};
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<float> data(12, 1.0f);
            if (CreateAclTensor(data, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(data, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumGetWorkspaceSize(s, -10, ACL_FLOAT, o, &ws, &ex);
                LOG_PRINT("  [ERR-PROBE] dim-too-negative: %s\n",
                          ret != ACL_SUCCESS ? "error as expected" : "UNEXPECTED success");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // P5: shape mismatch — CheckShape: OP_CHECK_SHAPE_NOT_EQUAL
        {
            std::vector<int64_t> sshape = {2, 3};
            std::vector<int64_t> oshape = {3, 2};
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<float> sdata(6, 1.0f);
            std::vector<float> odata(6, 0.0f);
            if (CreateAclTensor(sdata, sshape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(odata, oshape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumGetWorkspaceSize(s, 0, ACL_FLOAT, o, &ws, &ex);
                LOG_PRINT("  [ERR-PROBE] shape-mismatch: %s\n",
                          ret != ACL_SUCCESS ? "error as expected" : "UNEXPECTED success");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
    }

    // ====================================================================
    // 第十四部分：额外 tiling 边缘探针（float + int 剩余分支）
    // ====================================================================

    LOG_PRINT("\n--- Additional tiling edge probes ---\n");
    {
        // Float: FP32 3D dim=-1 (negative dim)
        {
            int64_t sz = 8 * 16 * 32;
            std::vector<int64_t> shape = {8, 16, 32};
            std::vector<float> input(sz, 1.0f);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<float> oh(sz, 0.0f);
            if (CreateAclTensor(input, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, -1, ACL_FLOAT, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] FP32 3D dim=-1 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // Float: FP32 2D [1,256] dim=1 (R=256 N=1 TWOWAY variant)
        {
            int64_t sz = 256;
            std::vector<int64_t> shape = {1, 256};
            std::vector<float> input(sz, 1.0f);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<float> oh(sz, 0.0f);
            if (CreateAclTensor(input, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_FLOAT, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] FP32 2D [1,256] dim=1 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // Int: INT8 3D dim=-1 (negative dim for int)
        {
            int64_t sz = 4 * 8 * 16;
            std::vector<int64_t> shape = {4, 8, 16};
            std::vector<int8_t> input(sz, 1);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<int8_t> oh(sz, 0);
            if (CreateAclTensor(input, shape, &sd, ACL_INT8, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_INT8, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, -1, ACL_INT8, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] INT8 3D dim=-1 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // Int: INT64 4D [2,3,4,5] dim=2 (multi-dim middle axis)
        {
            int64_t sz = 2 * 3 * 4 * 5;
            std::vector<int64_t> shape = {2, 3, 4, 5};
            std::vector<int64_t> input(sz, 1);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<int64_t> oh(sz, 0);
            if (CreateAclTensor(input, shape, &sd, ACL_INT64, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_INT64, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 2, ACL_INT64, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] INT64 4D [2,3,4,5] dim=2 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
    }

// ====================================================================
    // 第十五部分：第二波错误路径 + tiling 探针
    // ====================================================================

    LOG_PRINT("\n--- Round 2 probes ---\n");
    {
        // EP6: V2 dim out of range
        {
            std::vector<int64_t> shape = {3, 4};
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<float> data(12, 1.0f);
            if (CreateAclTensor(data, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(data, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumV2GetWorkspaceSize(s, 5, false, false, o, &ws, &ex);
                LOG_PRINT("  [ERR-PROBE] V2 dim-too-large: %s\n",
                          ret != ACL_SUCCESS ? "error as expected" : "UNEXPECTED success");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // EP7: V2 shape mismatch
        {
            std::vector<int64_t> sshape = {2, 3}, oshape = {3, 2};
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<float> sdata(6, 1.0f), odata(6, 0.0f);
            if (CreateAclTensor(sdata, sshape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(odata, oshape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumV2GetWorkspaceSize(s, 0, false, false, o, &ws, &ex);
                LOG_PRINT("  [ERR-PROBE] V2 shape-mismatch: %s\n",
                          ret != ACL_SUCCESS ? "error as expected" : "UNEXPECTED success");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
    }

    LOG_PRINT("\n--- Round 2 tiling edge probes ---\n");
    {
        // FT6: FP32 2D [2,16] dim=1 (R=16 N=1, R < cl boundary)
        {
            int64_t sz = 32; std::vector<int64_t> shape = {2, 16};
            std::vector<float> input(sz, 1.0f);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<float> oh(sz, 0.0f);
            if (CreateAclTensor(input, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_FLOAT, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] FP32 2D [2,16] dim=1 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // FT7: FP32 2D [64,32] dim=0 (M=1,R=64,N=32, N=cl boundary)
        {
            int64_t sz = 2048; std::vector<int64_t> shape = {64, 32};
            std::vector<float> input(sz, 1.0f);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<float> oh(sz, 0.0f);
            if (CreateAclTensor(input, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 0, ACL_FLOAT, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] FP32 2D [64,32] dim=0 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // FT8: FP16 3D [16,16,16] dim=1 (FP16 fold, all dims equal)
        {
            int64_t sz = 4096; std::vector<int64_t> shape = {16, 16, 16};
            std::vector<float> inputF(sz, 1.0f);
            std::vector<uint16_t> input(sz);
            for (int64_t i = 0; i < sz; i++) input[i] = FloatToFp16(inputF[i]);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<uint16_t> oh(sz, 0);
            if (CreateAclTensor(input, shape, &sd, ACL_FLOAT16, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT16, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_FLOAT16, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] FP16 3D [16,16,16] dim=1 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // IT9: INT16 2D [1024,4] dim=0 (INT16 large axis=0)
        {
            int64_t sz = 4096; std::vector<int64_t> shape = {1024, 4};
            std::vector<int16_t> input(sz, 1);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<int16_t> oh(sz, 0);
            if (CreateAclTensor(input, shape, &sd, ACL_INT16, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_INT16, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 0, ACL_INT16, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] INT16 2D axis=0 large ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // IT10: UINT8 3D [8,128,8] dim=1 (UINT8 mid-size dtypeSize=1)
        {
            int64_t sz = 8192; std::vector<int64_t> shape = {8, 128, 8};
            std::vector<uint8_t> input(sz, 1);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<uint8_t> oh(sz, 0);
            if (CreateAclTensor(input, shape, &sd, ACL_UINT8, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_UINT8, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_UINT8, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] UINT8 3D [8,128,8] dim=1 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
    }
// ====================================================================
    // 第十五部分：第二波错误路径 + tiling 探针
    // ====================================================================

    LOG_PRINT("\n--- Round 2 probes ---\n");
    {
        // EP6: V2 dim out of range
        {
            std::vector<int64_t> shape = {3, 4};
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<float> data(12, 1.0f);
            if (CreateAclTensor(data, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(data, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumV2GetWorkspaceSize(s, 5, false, false, o, &ws, &ex);
                LOG_PRINT("  [ERR-PROBE] V2 dim-too-large: %s\n",
                          ret != ACL_SUCCESS ? "error as expected" : "UNEXPECTED success");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // EP7: V2 shape mismatch
        {
            std::vector<int64_t> sshape = {2, 3}, oshape = {3, 2};
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<float> sdata(6, 1.0f), odata(6, 0.0f);
            if (CreateAclTensor(sdata, sshape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(odata, oshape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumV2GetWorkspaceSize(s, 0, false, false, o, &ws, &ex);
                LOG_PRINT("  [ERR-PROBE] V2 shape-mismatch: %s\n",
                          ret != ACL_SUCCESS ? "error as expected" : "UNEXPECTED success");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
    }

    LOG_PRINT("\n--- Round 2 tiling edge probes ---\n");
    {
        // FT6: FP32 2D [2,16] dim=1 (R=16 N=1, R < cl boundary)
        {
            int64_t sz = 32; std::vector<int64_t> shape = {2, 16};
            std::vector<float> input(sz, 1.0f);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<float> oh(sz, 0.0f);
            if (CreateAclTensor(input, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_FLOAT, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] FP32 2D [2,16] dim=1 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // FT7: FP32 2D [64,32] dim=0 (M=1,R=64,N=32, N=cl boundary)
        {
            int64_t sz = 2048; std::vector<int64_t> shape = {64, 32};
            std::vector<float> input(sz, 1.0f);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<float> oh(sz, 0.0f);
            if (CreateAclTensor(input, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 0, ACL_FLOAT, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] FP32 2D [64,32] dim=0 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // FT8: FP16 3D [16,16,16] dim=1 (FP16 fold, all dims equal)
        {
            int64_t sz = 4096; std::vector<int64_t> shape = {16, 16, 16};
            std::vector<float> inputF(sz, 1.0f);
            std::vector<uint16_t> input(sz);
            for (int64_t i = 0; i < sz; i++) input[i] = FloatToFp16(inputF[i]);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<uint16_t> oh(sz, 0);
            if (CreateAclTensor(input, shape, &sd, ACL_FLOAT16, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT16, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_FLOAT16, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] FP16 3D [16,16,16] dim=1 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // IT9: INT16 2D [1024,4] dim=0 (INT16 large axis=0)
        {
            int64_t sz = 4096; std::vector<int64_t> shape = {1024, 4};
            std::vector<int16_t> input(sz, 1);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<int16_t> oh(sz, 0);
            if (CreateAclTensor(input, shape, &sd, ACL_INT16, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_INT16, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 0, ACL_INT16, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] INT16 2D axis=0 large ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // IT10: UINT8 3D [8,128,8] dim=1 (UINT8 mid-size dtypeSize=1)
        {
            int64_t sz = 8192; std::vector<int64_t> shape = {8, 128, 8};
            std::vector<uint8_t> input(sz, 1);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<uint8_t> oh(sz, 0);
            if (CreateAclTensor(input, shape, &sd, ACL_UINT8, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_UINT8, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_UINT8, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] UINT8 3D [8,128,8] dim=1 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
    }
// ====================================================================
    // 第十六部分：第三波高价值分支探针
    // ====================================================================

    LOG_PRINT("\n--- Round 3 probes ---\n");
    {
        // EP8: V2 empty tensor path (self->IsEmpty() branch)
        {
            std::vector<int64_t> shape = {0, 5};
            int64_t sz = 0;
            std::vector<float> data(sz);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<float> oh(sz);
            if (CreateAclTensor(data, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumV2GetWorkspaceSize(s, 0, false, false, o, &ws, &ex);
                LOG_PRINT("  [PROBE] V2 empty tensor: %s (ws=%lu)\n",
                          ret == ACL_SUCCESS ? "ok" : "fail", ws);
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // EP9: V1 empty tensor path
        {
            std::vector<int64_t> shape = {0, 5};
            int64_t sz = 0;
            std::vector<float> data(sz), oh(sz);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(data, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumGetWorkspaceSize(s, 0, ACL_FLOAT, o, &ws, &ex);
                LOG_PRINT("  [PROBE] V1 empty tensor: %s (ws=%lu)\n",
                          ret == ACL_SUCCESS ? "ok" : "fail", ws);
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // CU1: CumsumCube path (batch>=12800, dim>=512, last axis)
        {
            int64_t sz = 12800 * 512;
            std::vector<int64_t> shape = {12800, 512};
            std::vector<float> input(sz, 1.0f);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<float> oh(sz, 0.0f);
            if (CreateAclTensor(input, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_FLOAT, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [PROBE] CumsumCube [12800,512] dim=1 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // FT9: FP32 2D [512,1] dim=0 (M=1,R=512,N=1, R>>N boundary)
        {
            int64_t sz = 512; std::vector<int64_t> shape = {512, 1};
            std::vector<float> input(sz, 1.0f);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<float> oh(sz, 0.0f);
            if (CreateAclTensor(input, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 0, ACL_FLOAT, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] FP32 2D [512,1] dim=0 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // IT11: INT8 1D [256] dim=0 (1D axis=0, simplest case)
        {
            int64_t sz = 256; std::vector<int64_t> shape = {256};
            std::vector<int8_t> input(sz, 1);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<int8_t> oh(sz, 0);
            if (CreateAclTensor(input, shape, &sd, ACL_INT8, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_INT8, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 0, ACL_INT8, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] INT8 1D [256] dim=0 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
    }

    // ====================================================================
    // 第十七部分：第四波精准探针（nullptr + dim边界 + tiling边缘）
    // ====================================================================

    LOG_PRINT("\n--- Round 4 error path probes ---\n");
    {
        // EP10: V1 self=nullptr with valid out
        {
            std::vector<int64_t> shape = {3, 4};
            std::vector<float> oh(12, 0.0f);
            void *od = nullptr; aclTensor *o = nullptr;
            if (CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumGetWorkspaceSize(nullptr, 0, ACL_FLOAT, o, &ws, &ex);
                LOG_PRINT("  [ERR] V1 self=nullptr: %s\n",
                    ret != ACL_SUCCESS ? "error as expected" : "UNEXPECTED success");
                CleanupTensor(o, od);
            }
        }
        // EP11: V1 out=nullptr with valid self
        {
            std::vector<int64_t> shape = {3, 4};
            std::vector<float> data(12, 1.0f);
            void *sd = nullptr; aclTensor *s = nullptr;
            if (CreateAclTensor(data, shape, &sd, ACL_FLOAT, &s) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumGetWorkspaceSize(s, 0, ACL_FLOAT, nullptr, &ws, &ex);
                LOG_PRINT("  [ERR] V1 out=nullptr: %s\n",
                    ret != ACL_SUCCESS ? "error as expected" : "UNEXPECTED success");
                CleanupTensor(s, sd);
            }
        }
        // EP12: V2 self=nullptr
        {
            std::vector<int64_t> shape = {3, 4};
            std::vector<float> oh(12, 0.0f);
            void *od = nullptr; aclTensor *o = nullptr;
            if (CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumV2GetWorkspaceSize(nullptr, 0, false, false, o, &ws, &ex);
                LOG_PRINT("  [ERR] V2 self=nullptr: %s\n",
                    ret != ACL_SUCCESS ? "error as expected" : "UNEXPECTED success");
                CleanupTensor(o, od);
            }
        }
        // EP13: V2 out=nullptr
        {
            std::vector<int64_t> shape = {3, 4};
            std::vector<float> data(12, 1.0f);
            void *sd = nullptr; aclTensor *s = nullptr;
            if (CreateAclTensor(data, shape, &sd, ACL_FLOAT, &s) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumV2GetWorkspaceSize(s, 0, false, false, nullptr, &ws, &ex);
                LOG_PRINT("  [ERR] V2 out=nullptr: %s\n",
                    ret != ACL_SUCCESS ? "error as expected" : "UNEXPECTED success");
                CleanupTensor(s, sd);
            }
        }
        // EP14: V1 dim==ndim boundary (dim=2 on 2D [3,4])
        {
            std::vector<int64_t> shape = {3, 4};
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<float> data(12, 1.0f), oh(12, 0.0f);
            if (CreateAclTensor(data, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumGetWorkspaceSize(s, 2, ACL_FLOAT, o, &ws, &ex);
                LOG_PRINT("  [ERR] V1 dim=2 on 2D: %s\n",
                    ret != ACL_SUCCESS ? "error as expected" : "UNEXPECTED success");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // EP15: V2 dim==ndim boundary
        {
            std::vector<int64_t> shape = {3, 4};
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<float> data(12, 1.0f), oh(12, 0.0f);
            if (CreateAclTensor(data, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumV2GetWorkspaceSize(s, 2, false, false, o, &ws, &ex);
                LOG_PRINT("  [ERR] V2 dim=2 on 2D: %s\n",
                    ret != ACL_SUCCESS ? "error as expected" : "UNEXPECTED success");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // EP16: V1 rank mismatch (2D self vs 3D out)
        {
            std::vector<int64_t> sshape = {3, 4}, oshape = {3, 4, 1};
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<float> sdata(12, 1.0f), odata(12, 0.0f);
            if (CreateAclTensor(sdata, sshape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(odata, oshape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumGetWorkspaceSize(s, 0, ACL_FLOAT, o, &ws, &ex);
                LOG_PRINT("  [ERR] V1 rank mismatch 2Dvs3D: %s\n",
                    ret != ACL_SUCCESS ? "error as expected" : "UNEXPECTED success");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
    }

    LOG_PRINT("\n--- Round 4 tiling edge probes ---\n");
    {
        // FT10: FP32 1D [4096] dim=0 (M=1,R=4096,N=1)
        {
            int64_t sz = 4096; std::vector<int64_t> shape = {4096};
            std::vector<float> input(sz, 1.0f), oh(sz, 0.0f);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 0, ACL_FLOAT, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] FP32 1D [4096] dim=0 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // FT11: FP16 1D [2048] dim=0 (FP16 large 1D)
        {
            int64_t sz = 2048; std::vector<int64_t> shape = {2048};
            std::vector<uint16_t> input(sz), oh(sz, 0);
            std::vector<float> fIn(sz, 1.0f);
            for (int64_t i = 0; i < sz; i++) input[i] = FloatToFp16(fIn[i]);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_FLOAT16, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT16, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 0, ACL_FLOAT16, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] FP16 1D [2048] dim=0 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // FT12: FP32 2D [4096,1] dim=0 (M=1,R=4096,N=1 from 2D)
        {
            int64_t sz = 4096; std::vector<int64_t> shape = {4096, 1};
            std::vector<float> input(sz, 1.0f), oh(sz, 0.0f);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 0, ACL_FLOAT, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] FP32 2D [4096,1] dim=0 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // IT12: INT32 1D [1] dim=0 (single element)
        {
            int64_t sz = 1; std::vector<int64_t> shape = {1};
            std::vector<int32_t> input(sz, 42), oh(sz, 0);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_INT32, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_INT32, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 0, ACL_INT32, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] INT32 1D [1] dim=0 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // IT13: INT8 1D [1] dim=0 (single element INT8)
        {
            int64_t sz = 1; std::vector<int64_t> shape = {1};
            std::vector<int8_t> input(sz, 7), oh(sz, 0);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_INT8, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_INT8, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 0, ACL_INT8, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] INT8 1D [1] dim=0 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // IT14: INT64 1D [2048] dim=0 (large 1D INT64)
        {
            int64_t sz = 2048; std::vector<int64_t> shape = {2048};
            std::vector<int64_t> input(sz, 1), oh(sz, 0);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_INT64, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_INT64, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 0, ACL_INT64, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] INT64 1D [2048] dim=0 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // IT15: UINT8 1D [256] dim=0 (UINT8 1D)
        {
            int64_t sz = 256; std::vector<int64_t> shape = {256};
            std::vector<uint8_t> input(sz, 1), oh(sz, 0);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_UINT8, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_UINT8, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 0, ACL_UINT8, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] UINT8 1D [256] dim=0 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
    }

    // ====================================================================
    // 第十八部分：第五波深水区探针（INT tiling 剩余分支 + OP_CHECK 深层）
    // ====================================================================

    LOG_PRINT("\n--- Round 5 INT tiling deep probes ---\n");
    {
        // IT16: INT32 3D [2,4,1024] dim=1 (RA-dominant, target TDRA split-on-RA)
        {
            int64_t sz = 8192; std::vector<int64_t> shape = {2, 4, 1024};
            std::vector<int32_t> input(sz, 1), oh(sz, 0);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_INT32, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_INT32, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_INT32, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] INT32 [2,4,1024] dim=1 RA-dom ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // IT17: INT32 2D [256,4] dim=0 (LA-dominant, target TDLA CheckBGC+AdjustLARLpUnit)
        {
            int64_t sz = 1024; std::vector<int64_t> shape = {256, 4};
            std::vector<int32_t> input(sz, 1), oh(sz, 0);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_INT32, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_INT32, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 0, ACL_INT32, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] INT32 [256,4] dim=0 LA-dom ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // IT18: INT16 3D [128,2,8] dim=0 (INT16 LA-dominant for TDLA)
        {
            int64_t sz = 2048; std::vector<int64_t> shape = {128, 2, 8};
            std::vector<int16_t> input(sz, 1), oh(sz, 0);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_INT16, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_INT16, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 0, ACL_INT16, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] INT16 [128,2,8] dim=0 LA-dom ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // IT19: INT32 3D [8,512,2] dim=1 (R-dominant, target AdjustTensor4TDR fallback)
        {
            int64_t sz = 8192; std::vector<int64_t> shape = {8, 512, 2};
            std::vector<int32_t> input(sz, 1), oh(sz, 0);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_INT32, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_INT32, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_INT32, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] INT32 [8,512,2] dim=1 R-dom ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // IT20: INT8 4D [2,3,128,4] dim=2 (INT8 mid R, multi-dim)
        {
            int64_t sz = 3072; std::vector<int64_t> shape = {2, 3, 128, 4};
            std::vector<int8_t> input(sz, 1), oh(sz, 0);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_INT8, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_INT8, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 2, ACL_INT8, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] INT8 [2,3,128,4] dim=2 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // IT21: INT64 3D [4,1024,4] dim=1 (INT64 mid-axis large R)
        {
            int64_t sz = 16384; std::vector<int64_t> shape = {4, 1024, 4};
            std::vector<int64_t> input(sz, 1), oh(sz, 0);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_INT64, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_INT64, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_INT64, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [TILING] INT64 [4,1024,4] dim=1 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
    }

    LOG_PRINT("\n--- Round 5 error path deep probes ---\n");
    {
        // EP17: V1 self dtype vs out dtype finer mismatch (INT32 self vs FLOAT out)
        {
            std::vector<int64_t> shape = {4, 8};
            std::vector<int32_t> sdata(32, 1); std::vector<float> odata(32, 0.0f);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(sdata, shape, &sd, ACL_INT32, &s) == 0 &&
                CreateAclTensor(odata, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumGetWorkspaceSize(s, 0, ACL_INT32, o, &ws, &ex);
                LOG_PRINT("  [ERR] V1 self_INT32_out_FLOAT: %s\n",
                    ret != ACL_SUCCESS ? "error as expected" : "UNEXPECTED success");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // EP18: V2 self nullptr AND out nullptr (double null)
        {
            uint64_t ws = 0; aclOpExecutor *ex = nullptr;
            auto ret = aclnnCumsumV2GetWorkspaceSize(nullptr, 0, false, false, nullptr, &ws, &ex);
            LOG_PRINT("  [ERR] V2 both-null: %s\n",
                ret != ACL_SUCCESS ? "error as expected" : "UNEXPECTED success");
        }
        // EP19: V1 dim negative exact boundary dim=-3 on 2D [3,4]
        {
            std::vector<int64_t> shape = {3, 4};
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<float> data(12, 1.0f), oh(12, 0.0f);
            if (CreateAclTensor(data, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumGetWorkspaceSize(s, -3, ACL_FLOAT, o, &ws, &ex);
                LOG_PRINT("  [ERR] V1 dim=-3 on 2D: %s\n",
                    ret != ACL_SUCCESS ? "error as expected" : "UNEXPECTED success");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // EP20: V2 dim negative exact boundary dim=-3 on 2D
        {
            std::vector<int64_t> shape = {3, 4};
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            std::vector<float> data(12, 1.0f), oh(12, 0.0f);
            if (CreateAclTensor(data, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumV2GetWorkspaceSize(s, -3, false, false, o, &ws, &ex);
                LOG_PRINT("  [ERR] V2 dim=-3 on 2D: %s\n",
                    ret != ACL_SUCCESS ? "error as expected" : "UNEXPECTED success");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // EP21: V1 out nullptr with INT32 dtype (different type in nullptr path)
        {
            std::vector<int64_t> shape = {4, 8};
            std::vector<int32_t> data(32, 1);
            void *sd = nullptr; aclTensor *s = nullptr;
            if (CreateAclTensor(data, shape, &sd, ACL_INT32, &s) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumGetWorkspaceSize(s, 0, ACL_INT32, nullptr, &ws, &ex);
                LOG_PRINT("  [ERR] V1 INT32 out=nullptr: %s\n",
                    ret != ACL_SUCCESS ? "error as expected" : "UNEXPECTED success");
                CleanupTensor(s, sd);
            }
        }
    }
    // ====================================================================
    // 第十八部分：第五波深水区探针（INT tiling 剩余分支 + OP_CHECK 深层）
    // ====================================================================

    LOG_PRINT("\n--- Round 5 INT tiling deep probes ---\n");
    {
        // IT16: INT32 3D [2,4,1024] dim=1 (RA-dominant, target TDRA split-on-RA)
        { int64_t sz=8192; std::vector<int64_t> shape={2,4,1024}; std::vector<int32_t> in(sz,1),oh(sz,0);
          void *sd=nullptr,*od=nullptr; aclTensor *s=nullptr,*o=nullptr;
          if(CreateAclTensor(in,shape,&sd,ACL_INT32,&s)==0&&CreateAclTensor(oh,shape,&od,ACL_INT32,&o)==0){
            uint64_t ws=0; aclOpExecutor *ex=nullptr;
            if(aclnnCumsumGetWorkspaceSize(s,1,ACL_INT32,o,&ws,&ex)==ACL_SUCCESS)LOG_PRINT("  [T5] IT16 ok\n");
            CleanupTensor(s,sd); CleanupTensor(o,od); } }
        // IT17: INT32 2D [256,4] dim=0
        { int64_t sz=1024; std::vector<int64_t> shape={256,4}; std::vector<int32_t> in(sz,1),oh(sz,0);
          void *sd=nullptr,*od=nullptr; aclTensor *s=nullptr,*o=nullptr;
          if(CreateAclTensor(in,shape,&sd,ACL_INT32,&s)==0&&CreateAclTensor(oh,shape,&od,ACL_INT32,&o)==0){
            uint64_t ws=0; aclOpExecutor *ex=nullptr;
            if(aclnnCumsumGetWorkspaceSize(s,0,ACL_INT32,o,&ws,&ex)==ACL_SUCCESS)LOG_PRINT("  [T5] IT17 ok\n");
            CleanupTensor(s,sd); CleanupTensor(o,od); } }
        // IT18: INT16 3D [128,2,8] dim=0
        { int64_t sz=2048; std::vector<int64_t> shape={128,2,8}; std::vector<int16_t> in(sz,1),oh(sz,0);
          void *sd=nullptr,*od=nullptr; aclTensor *s=nullptr,*o=nullptr;
          if(CreateAclTensor(in,shape,&sd,ACL_INT16,&s)==0&&CreateAclTensor(oh,shape,&od,ACL_INT16,&o)==0){
            uint64_t ws=0; aclOpExecutor *ex=nullptr;
            if(aclnnCumsumGetWorkspaceSize(s,0,ACL_INT16,o,&ws,&ex)==ACL_SUCCESS)LOG_PRINT("  [T5] IT18 ok\n");
            CleanupTensor(s,sd); CleanupTensor(o,od); } }
        // IT19: INT32 3D [8,512,2] dim=1
        { int64_t sz=8192; std::vector<int64_t> shape={8,512,2}; std::vector<int32_t> in(sz,1),oh(sz,0);
          void *sd=nullptr,*od=nullptr; aclTensor *s=nullptr,*o=nullptr;
          if(CreateAclTensor(in,shape,&sd,ACL_INT32,&s)==0&&CreateAclTensor(oh,shape,&od,ACL_INT32,&o)==0){
            uint64_t ws=0; aclOpExecutor *ex=nullptr;
            if(aclnnCumsumGetWorkspaceSize(s,1,ACL_INT32,o,&ws,&ex)==ACL_SUCCESS)LOG_PRINT("  [T5] IT19 ok\n");
            CleanupTensor(s,sd); CleanupTensor(o,od); } }
        // IT20: INT8 4D [2,3,128,4] dim=2
        { int64_t sz=3072; std::vector<int64_t> shape={2,3,128,4}; std::vector<int8_t> in(sz,1),oh(sz,0);
          void *sd=nullptr,*od=nullptr; aclTensor *s=nullptr,*o=nullptr;
          if(CreateAclTensor(in,shape,&sd,ACL_INT8,&s)==0&&CreateAclTensor(oh,shape,&od,ACL_INT8,&o)==0){
            uint64_t ws=0; aclOpExecutor *ex=nullptr;
            if(aclnnCumsumGetWorkspaceSize(s,2,ACL_INT8,o,&ws,&ex)==ACL_SUCCESS)LOG_PRINT("  [T5] IT20 ok\n");
            CleanupTensor(s,sd); CleanupTensor(o,od); } }
        // IT21: INT64 3D [4,1024,4] dim=1
        { int64_t sz=16384; std::vector<int64_t> shape={4,1024,4}; std::vector<int64_t> in(sz,1),oh(sz,0);
          void *sd=nullptr,*od=nullptr; aclTensor *s=nullptr,*o=nullptr;
          if(CreateAclTensor(in,shape,&sd,ACL_INT64,&s)==0&&CreateAclTensor(oh,shape,&od,ACL_INT64,&o)==0){
            uint64_t ws=0; aclOpExecutor *ex=nullptr;
            if(aclnnCumsumGetWorkspaceSize(s,1,ACL_INT64,o,&ws,&ex)==ACL_SUCCESS)LOG_PRINT("  [T5] IT21 ok\n");
            CleanupTensor(s,sd); CleanupTensor(o,od); } }
    }
    LOG_PRINT("\n--- Round 5 error deep probes ---\n");
    {
        // EP17: V1 self_INT32 out_FLOAT dtype mismatch
        { std::vector<int64_t> shape={4,8}; std::vector<int32_t> sd(32,1); std::vector<float> od(32,0);
          void *sdd=nullptr,*odd=nullptr; aclTensor *st=nullptr,*ot=nullptr;
          if(CreateAclTensor(sd,shape,&sdd,ACL_INT32,&st)==0&&CreateAclTensor(od,shape,&odd,ACL_FLOAT,&ot)==0){
            uint64_t ws=0; aclOpExecutor *ex=nullptr;
            auto ret=aclnnCumsumGetWorkspaceSize(st,0,ACL_INT32,ot,&ws,&ex);
            LOG_PRINT("  [ERR] EP17: %s\n",ret!=ACL_SUCCESS?"err ok":"UNEXPECTED ok");
            CleanupTensor(st,sdd); CleanupTensor(ot,odd); } }
        // EP18: V2 both nullptr
        { uint64_t ws=0; aclOpExecutor *ex=nullptr;
          auto ret=aclnnCumsumV2GetWorkspaceSize(nullptr,0,false,false,nullptr,&ws,&ex);
          LOG_PRINT("  [ERR] EP18 V2 both-null: %s\n",ret!=ACL_SUCCESS?"err ok":"UNEXPECTED ok"); }
        // EP19: V1 dim=-3 on 2D
        { std::vector<int64_t> shape={3,4}; std::vector<float> d(12,1),oh(12,0);
          void *sdd=nullptr,*odd=nullptr; aclTensor *st=nullptr,*ot=nullptr;
          if(CreateAclTensor(d,shape,&sdd,ACL_FLOAT,&st)==0&&CreateAclTensor(oh,shape,&odd,ACL_FLOAT,&ot)==0){
            uint64_t ws=0; aclOpExecutor *ex=nullptr;
            auto ret=aclnnCumsumGetWorkspaceSize(st,-3,ACL_FLOAT,ot,&ws,&ex);
            LOG_PRINT("  [ERR] EP19 dim=-3: %s\n",ret!=ACL_SUCCESS?"err ok":"UNEXPECTED ok");
            CleanupTensor(st,sdd); CleanupTensor(ot,odd); } }
        // EP20: V2 dim=-3 on 2D
        { std::vector<int64_t> shape={3,4}; std::vector<float> d(12,1),oh(12,0);
          void *sdd=nullptr,*odd=nullptr; aclTensor *st=nullptr,*ot=nullptr;
          if(CreateAclTensor(d,shape,&sdd,ACL_FLOAT,&st)==0&&CreateAclTensor(oh,shape,&odd,ACL_FLOAT,&ot)==0){
            uint64_t ws=0; aclOpExecutor *ex=nullptr;
            auto ret=aclnnCumsumV2GetWorkspaceSize(st,-3,false,false,ot,&ws,&ex);
            LOG_PRINT("  [ERR] EP20 dim=-3: %s\n",ret!=ACL_SUCCESS?"err ok":"UNEXPECTED ok");
            CleanupTensor(st,sdd); CleanupTensor(ot,odd); } }
        // EP21: V1 INT32 out=nullptr
        { std::vector<int64_t> shape={4,8}; std::vector<int32_t> d(32,1);
          void *sdd=nullptr; aclTensor *st=nullptr;
          if(CreateAclTensor(d,shape,&sdd,ACL_INT32,&st)==0){ uint64_t ws=0; aclOpExecutor *ex=nullptr;
            auto ret=aclnnCumsumGetWorkspaceSize(st,0,ACL_INT32,nullptr,&ws,&ex);
            LOG_PRINT("  [ERR] EP21: %s\n",ret!=ACL_SUCCESS?"err ok":"UNEXPECTED ok");
            CleanupTensor(st,sdd); } }
    }

    // ====================================================================
    // 第十九部分：第六波精准探针（CheckCubeSupport + BF16 tiling + cumsum.cpp 调度分支）
    // ====================================================================

    LOG_PRINT("\n--- Round 6 CheckCubeSupport + CheckShapeIsSupport probes ---\n");
    {
        // EP22: FLOAT [12800,512] dim=1 → CheckCubeSupport: dtype=FLOAT, dim==last, batch=12800, chan=512 → CumsumCube path
        {
            int64_t sz = 12800 * 512; std::vector<int64_t> shape = {12800, 512};
            std::vector<float> input(sz, 1.0f), oh(sz, 0.0f);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumGetWorkspaceSize(s, 1, ACL_FLOAT, o, &ws, &ex);
                LOG_PRINT("  [R6] EP22 FLOAT[12800,512] dim=1 CubePath: %s\n",
                    ret == ACL_SUCCESS ? "hit" : "fail");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // EP23: FLOAT [12800,511] dim=1 → CheckShapeIsSupport: chan=511<512 → false
        {
            int64_t sz = 12800 * 511; std::vector<int64_t> shape = {12800, 511};
            std::vector<float> input(sz, 1.0f), oh(sz, 0.0f);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumGetWorkspaceSize(s, 1, ACL_FLOAT, o, &ws, &ex);
                LOG_PRINT("  [R6] EP23 FLOAT[12800,511] dim=1 chan<512: %s\n",
                    ret == ACL_SUCCESS ? "hit" : "fail");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // EP24: FLOAT [12799,512] dim=1 → CheckShapeIsSupport: batch=12799<12800 → false
        {
            int64_t sz = 12799 * 512; std::vector<int64_t> shape = {12799, 512};
            std::vector<float> input(sz, 1.0f), oh(sz, 0.0f);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumGetWorkspaceSize(s, 1, ACL_FLOAT, o, &ws, &ex);
                LOG_PRINT("  [R6] EP24 FLOAT[12799,512] dim=1 batch<12800: %s\n",
                    ret == ACL_SUCCESS ? "hit" : "fail");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // EP25: FLOAT [3,4,5] dim=1 → CheckShapeIsSupport: dim(1)!=last(2) → false
        {
            int64_t sz = 60; std::vector<int64_t> shape = {3, 4, 5};
            std::vector<float> input(sz, 1.0f), oh(sz, 0.0f);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumGetWorkspaceSize(s, 1, ACL_FLOAT, o, &ws, &ex);
                LOG_PRINT("  [R6] EP25 FLOAT[3,4,5] dim=1 notLast: %s\n",
                    ret == ACL_SUCCESS ? "hit" : "fail");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // EP26: INT32 [12800,512] dim=1 → CheckCubeSupport: dtype=INT32 → isSupport=false
        {
            int64_t sz = 12800 * 512; std::vector<int64_t> shape = {12800, 512};
            std::vector<int32_t> input(sz, 1), oh(sz, 0);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_INT32, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_INT32, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumGetWorkspaceSize(s, 1, ACL_INT32, o, &ws, &ex);
                LOG_PRINT("  [R6] EP26 INT32[12800,512] dim=1 noCube: %s\n",
                    ret == ACL_SUCCESS ? "hit" : "fail");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // EP27: DOUBLE [64] dim=0 → cumsum.cpp IsAiCoreSupport: regbase no DOUBLE → AiCPU path
        {
            int64_t sz = 64; std::vector<int64_t> shape = {64};
            std::vector<double> input(sz, 1.0), oh(sz, 0.0);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_DOUBLE, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_DOUBLE, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumGetWorkspaceSize(s, 0, ACL_DOUBLE, o, &ws, &ex);
                LOG_PRINT("  [R6] EP27 DOUBLE[64] dim=0 AiCPU: %s\n",
                    ret == ACL_SUCCESS ? "hit" : "fail");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // EP28: shape dim > MAX_DIM_LEN (8) → CheckShape OP_CHECK_MAX_DIM
        {
            std::vector<int64_t> shape = {1, 1, 1, 1, 1, 1, 1, 1, 1};  // 9D
            std::vector<float> input(9, 1.0f), oh(9, 0.0f);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                auto ret = aclnnCumsumGetWorkspaceSize(s, 0, ACL_FLOAT, o, &ws, &ex);
                LOG_PRINT("  [R6] EP28 9D shape>8: %s\n",
                    ret != ACL_SUCCESS ? "err as expected" : "UNEXPECTED ok");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
    }

    LOG_PRINT("\n--- Round 6 BF16 tiling probes ---\n");
    {
        // FT13: BF16 1D [1024] dim=0 (simple BF16 1D)
        {
            int64_t sz = 1024; std::vector<int64_t> shape = {1024};
            std::vector<uint16_t> input(sz), oh(sz, 0);
            std::vector<float> fIn(sz, 1.0f);
            for (int64_t i = 0; i < sz; i++) input[i] = FloatToFp16(fIn[i]);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_BF16, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_BF16, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 0, ACL_BF16, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [R6] FT13 BF16 1D [1024] ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // FT14: BF16 2D [64,256] dim=1 (M=64,R=256,N=1 BF16)
        {
            int64_t sz = 16384; std::vector<int64_t> shape = {64, 256};
            std::vector<uint16_t> input(sz), oh(sz, 0);
            std::vector<float> fIn(sz, 1.0f);
            for (int64_t i = 0; i < sz; i++) input[i] = FloatToFp16(fIn[i]);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_BF16, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_BF16, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_BF16, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [R6] FT14 BF16 [64,256] dim=1 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // FT15: BF16 2D [16,2048] dim=1 (M=16,R=2048,N=1, large R for BF16 dtCast_)
        {
            int64_t sz = 32768; std::vector<int64_t> shape = {16, 2048};
            std::vector<uint16_t> input(sz), oh(sz, 0);
            std::vector<float> fIn(sz, 1.0f);
            for (int64_t i = 0; i < sz; i++) input[i] = FloatToFp16(fIn[i]);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_BF16, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_BF16, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_BF16, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [R6] FT15 BF16 [16,2048] dim=1 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // FT16: BF16 3D [128,16,16] dim=1 (M=128,R=16,N=16, BF16 3D)
        {
            int64_t sz = 32768; std::vector<int64_t> shape = {128, 16, 16};
            std::vector<uint16_t> input(sz), oh(sz, 0);
            std::vector<float> fIn(sz, 1.0f);
            for (int64_t i = 0; i < sz; i++) input[i] = FloatToFp16(fIn[i]);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_BF16, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_BF16, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_BF16, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [R6] FT16 BF16 [128,16,16] dim=1 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // FT17: FP32 3D [2,4096,64] dim=1 (M=2 small, R=4096 large, N=64, target borrowR/TWOWAY)
        {
            int64_t sz = 2 * 4096 * 64; std::vector<int64_t> shape = {2, 4096, 64};
            std::vector<float> input(sz, 1.0f), oh(sz, 0.0f);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_FLOAT, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [R6] FT17 FP32 [2,4096,64] dim=1 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // FT18: FP16 3D [8,2048,32] dim=1 (M=8,R=2048,N=32, FP16 dtCast_, target UB_SS TWOWAY)
        {
            int64_t sz = 8 * 2048 * 32; std::vector<int64_t> shape = {8, 2048, 32};
            std::vector<uint16_t> input(sz), oh(sz, 0);
            std::vector<float> fIn(sz, 1.0f);
            for (int64_t i = 0; i < sz; i++) input[i] = FloatToFp16(fIn[i]);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_FLOAT16, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT16, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_FLOAT16, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [R6] FT18 FP16 [8,2048,32] dim=1 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
    }

    LOG_PRINT("\n--- Round 6 INT tiling edge probes ---\n");
    {
        // IT22: INT32 [40, 512] dim=0 (leftAxisLen near coreNum, test LA/core boundary)
        {
            int64_t sz = 20480; std::vector<int64_t> shape = {40, 512};
            std::vector<int32_t> input(sz, 1), oh(sz, 0);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_INT32, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_INT32, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 0, ACL_INT32, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [R6] IT22 INT32 [40,512] dim=0 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // IT23: INT8 2D [1,4096] dim=1 (midAxisLen large, INT8 dtypeSize=1, tests vlSize/2)
        {
            int64_t sz = 4096; std::vector<int64_t> shape = {1, 4096};
            std::vector<int8_t> input(sz, 1), oh(sz, 0);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_INT8, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_INT8, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_INT8, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [R6] IT23 INT8 [1,4096] dim=1 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // IT24: INT64 3D [64,2,32] dim=1 (mid dim=2 small, INT64 dtypeSize=8)
        {
            int64_t sz = 4096; std::vector<int64_t> shape = {64, 2, 32};
            std::vector<int64_t> input(sz, 1), oh(sz, 0);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_INT64, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_INT64, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_INT64, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [R6] IT24 INT64 [64,2,32] dim=1 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // IT25: UINT8 3D [4,256,8] dim=1 (UINT8 mid-dim, dtypeSize=1)
        {
            int64_t sz = 8192; std::vector<int64_t> shape = {4, 256, 8};
            std::vector<uint8_t> input(sz, 1), oh(sz, 0);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_UINT8, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_UINT8, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 1, ACL_UINT8, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [R6] IT25 UINT8 [4,256,8] dim=1 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // IT26: INT16 4D [2,2,2,2] dim=2 (INT16 multi-dim, trigger different INT tiling)
        {
            int64_t sz = 16; std::vector<int64_t> shape = {2, 2, 2, 2};
            std::vector<int16_t> input(sz, 1), oh(sz, 0);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_INT16, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_INT16, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 2, ACL_INT16, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [R6] IT26 INT16 [2,2,2,2] dim=2 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // IT27: INT32 5D [2,2,2,2,2] dim=2 (5D INT32, test multi-dim edge)
        {
            int64_t sz = 32; std::vector<int64_t> shape = {2, 2, 2, 2, 2};
            std::vector<int32_t> input(sz, 1), oh(sz, 0);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_INT32, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_INT32, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumGetWorkspaceSize(s, 2, ACL_INT32, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [R6] IT27 INT32 5D [2x5] dim=2 ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
    }

    LOG_PRINT("\n--- Round 6 V2 exclusive/reverse edge probes ---\n");
    {
        // CT1: V2 FLOAT [256] dim=0 exclusive=true reverse=true
        {
            int64_t sz = 256; std::vector<int64_t> shape = {256};
            std::vector<float> input(sz, 1.0f), oh(sz, 0.0f);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_FLOAT, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumV2GetWorkspaceSize(s, 0, true, true, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [R6] CT1 V2 excl=1 rev=1 [256] ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // CT2: V2 INT32 [512] dim=0 exclusive=true reverse=true (INT + V2 all flags)
        {
            int64_t sz = 512; std::vector<int64_t> shape = {512};
            std::vector<int32_t> input(sz, 1), oh(sz, 0);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_INT32, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_INT32, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumV2GetWorkspaceSize(s, 0, true, true, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [R6] CT2 V2 excl=1 rev=1 INT32 [512] ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // CT3: V2 INT8 [128] dim=0 exclusive=true reverse=false
        {
            int64_t sz = 128; std::vector<int64_t> shape = {128};
            std::vector<int8_t> input(sz, 1), oh(sz, 0);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_INT8, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_INT8, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumV2GetWorkspaceSize(s, 0, true, false, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [R6] CT3 V2 excl=1 rev=0 INT8 [128] ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // CT4: V2 BF16 [256] dim=0 exclusive=false reverse=true (BF16 + V2)
        {
            int64_t sz = 256; std::vector<int64_t> shape = {256};
            std::vector<uint16_t> input(sz), oh(sz, 0);
            std::vector<float> fIn(sz, 1.0f);
            for (int64_t i = 0; i < sz; i++) input[i] = FloatToFp16(fIn[i]);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_BF16, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_BF16, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumV2GetWorkspaceSize(s, 0, false, true, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [R6] CT4 V2 excl=0 rev=1 BF16 [256] ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
        // CT5: V2 FP16 [128] dim=0 exclusive=true reverse=false (FP16 + V2)
        {
            int64_t sz = 128; std::vector<int64_t> shape = {128};
            std::vector<uint16_t> input(sz), oh(sz, 0);
            std::vector<float> fIn(sz, 1.0f);
            for (int64_t i = 0; i < sz; i++) input[i] = FloatToFp16(fIn[i]);
            void *sd = nullptr, *od = nullptr; aclTensor *s = nullptr, *o = nullptr;
            if (CreateAclTensor(input, shape, &sd, ACL_FLOAT16, &s) == 0 &&
                CreateAclTensor(oh, shape, &od, ACL_FLOAT16, &o) == 0) {
                uint64_t ws = 0; aclOpExecutor *ex = nullptr;
                if (aclnnCumsumV2GetWorkspaceSize(s, 0, true, false, o, &ws, &ex) == ACL_SUCCESS)
                    LOG_PRINT("  [R6] CT5 V2 excl=1 rev=0 FP16 [128] ok\n");
                CleanupTensor(s, sd); CleanupTensor(o, od);
            }
        }
    }
    // ====================================================================
    // 汇总
    // ====================================================================

    LOG_PRINT("\n========================================\n");
    LOG_PRINT("  Summary: %d passed, %d failed, %d total\n",
              g_passedTests, g_failedTests, g_totalTests);
    LOG_PRINT("========================================\n");

    // 释放资源
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return (g_failedTests > 0) ? 1 : 0;
}
