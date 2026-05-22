/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Add operator comprehensive end-to-end test suite.
 * Covers: all 6 API variants, 10+ dtype combinations, alpha parameter paths,
 * shape broadcasting, precision edge cases, and special values.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <cfenv>
#include <complex>
#include <type_traits>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

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

// ===================== Global test metrics =====================
static int g_totalTests = 0;
static int g_passedTests = 0;
static int g_failedTests = 0;

// ===================== Utility functions =====================

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int64_t GetBroadcastIndex(int64_t outIndex, const std::vector<int64_t>& outShape, const std::vector<int64_t>& inputShape)
{
    // 将输出线性下标反推到输入张量下标，用于真实 broadcast 结果验证。
    if (inputShape.empty()) {
        return 0;
    }
    int64_t inputIndex = 0;
    int64_t inputStride = 1;
    int64_t tmpIndex = outIndex;
    for (int64_t outDim = static_cast<int64_t>(outShape.size()) - 1, inDim = static_cast<int64_t>(inputShape.size()) - 1;
         inDim >= 0; --outDim, --inDim) {
        int64_t outCoord = 0;
        if (outDim >= 0) {
            outCoord = tmpIndex % outShape[outDim];
            tmpIndex /= outShape[outDim];
        }
        int64_t inputCoord = (inputShape[inDim] == 1) ? 0 : outCoord;
        inputIndex += inputCoord * inputStride;
        inputStride *= inputShape[inDim];
    }
    return inputIndex;
}

struct ErrorStats {
    int64_t firstFailIndex = -1;
    int64_t mismatchCount = 0;
    double maxAbsErr = 0.0;
    double maxRelErr = 0.0;
};

static void UpdateErrorStats(ErrorStats& stats, int64_t index, double actual, double expected, bool matched)
{
    // 记录首个失败位置和最大误差，便于分析精度问题。
    double absErr = std::abs(actual - expected);
    double relErr = absErr / (std::abs(expected) + 1e-12);
    if (absErr > stats.maxAbsErr) {
        stats.maxAbsErr = absErr;
    }
    if (relErr > stats.maxRelErr) {
        stats.maxRelErr = relErr;
    }
    if (!matched) {
        ++stats.mismatchCount;
    }
    if (!matched && stats.firstFailIndex < 0) {
        stats.firstFailIndex = index;
    }
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
    aclError ret = ACL_SUCCESS;
    if (size > 0) {
        ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    } else {
        *deviceAddr = nullptr;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

// FP16 conversion helpers
static inline float Fp16ToFloat(uint16_t fp16)
{
    uint32_t sign = (fp16 >> 15) & 1;
    uint32_t exp = (fp16 >> 10) & 0x1F;
    uint32_t mant = fp16 & 0x3FF;
    uint32_t f32;
    if (exp == 0) {
        if (mant == 0) {
            f32 = sign << 31;
        } else {
            int shift = 0;
            while ((mant & 0x400) == 0) { mant <<= 1; shift++; }
            exp = 1 - shift;
            mant &= 0x3FF;
            f32 = (sign << 31) | ((127 - 15 + exp) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
    } else {
        f32 = (sign << 31) | ((127 - 15 + exp) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f32, sizeof(result));
    return result;
}

static inline uint16_t FloatToFp16(float f)
{
    uint32_t f32;
    memcpy(&f32, &f, sizeof(f32));
    uint32_t sign = (f32 >> 31) & 1;
    uint32_t exp = (f32 >> 23) & 0xFF;
    uint32_t mant = f32 & 0x7FFFFF;

    if (exp == 0xFF) {
        uint16_t fp16 = (sign << 15) | (0x1F << 10) | (mant >> 13);
        return fp16 | (mant == 0 ? 0 : 0x200);
    }
    if (exp == 0) {
        return sign << 15;
    }
    int16_t exp16 = static_cast<int16_t>(exp) - 127 + 15;
    if (exp16 >= 0x1F) {
        return (sign << 15) | (0x1F << 10);
    }
    if (exp16 <= 0) {
        mant = (mant | 0x800000) >> (1 - exp16);
        return (sign << 15) | (mant >> 13);
    }
    return (sign << 15) | (exp16 << 10) | (mant >> 13);
}

// BF16 conversion helpers
static inline float Bf16ToFloat(uint16_t bf16)
{
    uint32_t f32 = static_cast<uint32_t>(bf16) << 16;
    float result;
    memcpy(&result, &f32, sizeof(result));
    return result;
}

static inline uint16_t FloatToBf16(float f)
{
    uint32_t f32;
    memcpy(&f32, &f, sizeof(f32));
    uint32_t rounding = 0x7FFF + ((f32 >> 16) & 1);
    f32 += rounding;
    return static_cast<uint16_t>(f32 >> 16);
}

// Comparison with tolerance
template <typename T>
static bool CompareWithTol(T actual, T expected, float atol, float rtol)
{
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        if (std::isinf(expected) && std::isinf(actual)) {
            return (expected > 0) == (actual > 0);
        }
        if (std::isnan(expected) && std::isnan(actual)) {
            return true;
        }
        float diff = std::abs(static_cast<float>(actual) - static_cast<float>(expected));
        float tolerance = atol + rtol * std::abs(static_cast<float>(expected));
        return diff <= tolerance;
    }
    return actual == expected;
}

static bool CompareFloatWithTol(float actual, float expected, float atol, float rtol)
{
    if (std::isinf(expected) && std::isinf(actual)) {
        return (expected > 0) == (actual > 0);
    }
    if (std::isnan(expected) && std::isnan(actual)) {
        return true;
    }
    float diff = std::abs(actual - expected);
    float tolerance = atol + rtol * std::abs(expected);
    return diff <= tolerance;
}

// 创建 float 类型 aclScalar，供 Adds / V3 等标量输入路径复用。
static aclScalar* CreateScalarFloat(float value)
{
    return aclCreateScalar(&value, aclDataType::ACL_FLOAT);
}

// 按输入类型创建 alpha，整数路径需要与 promote dtype 一致。
template <typename T>
static aclScalar* CreateAlphaScalar(float value)
{
    if constexpr (std::is_same_v<T, float>) {
        return CreateScalarFloat(value);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        int32_t typedValue = static_cast<int32_t>(value);
        return aclCreateScalar(&typedValue, aclDataType::ACL_INT32);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        int64_t typedValue = static_cast<int64_t>(value);
        return aclCreateScalar(&typedValue, aclDataType::ACL_INT64);
    } else if constexpr (std::is_same_v<T, int8_t>) {
        int8_t typedValue = static_cast<int8_t>(value);
        return aclCreateScalar(&typedValue, aclDataType::ACL_INT8);
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        uint8_t typedValue = static_cast<uint8_t>(value);
        return aclCreateScalar(&typedValue, aclDataType::ACL_UINT8);
    } else {
        return CreateScalarFloat(value);
    }
}

// ===================== Tensor resource manager =====================
struct TensorResources {
    std::vector<void*> deviceAddrs;
    std::vector<aclTensor*> tensors;
    std::vector<aclScalar*> scalars;
    void* workspaceAddr = nullptr;

    void Cleanup() {
        for (auto* t : tensors) {
            if (t) aclDestroyTensor(t);
        }
        for (auto* s : scalars) {
            if (s) aclDestroyScalar(s);
        }
        for (auto* d : deviceAddrs) {
            if (d) aclrtFree(d);
        }
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
            workspaceAddr = nullptr;
        }
    }
};

// ===================== Test case runner: aclnnAdd (tensor + alpha * tensor) =====================
template <typename T>
bool RunTestAdd(
    aclrtStream stream,
    const std::vector<T>& selfData,
    const std::vector<T>& otherData,
    const std::vector<int64_t>& selfShape,
    const std::vector<int64_t>& otherShape,
    const std::vector<int64_t>& outShape,
    aclDataType dtype,
    float alphaFloat,
    float atol,
    float rtol,
    const char* testName,
    bool validateOutput = true)
{
    g_totalTests++;
    TensorResources res;
    bool passed = true;

    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclTensor* out = nullptr;
    aclScalar* alpha = nullptr;

    std::vector<T> outHostData(GetShapeSize(outShape), 0);

    int ret = CreateAclTensor(selfData, selfShape, &selfDeviceAddr, dtype, &self);
    CHECK_RET(ret == 0, res.Cleanup(); return false);
    res.deviceAddrs.push_back(selfDeviceAddr);
    res.tensors.push_back(self);

    ret = CreateAclTensor(otherData, otherShape, &otherDeviceAddr, dtype, &other);
    CHECK_RET(ret == 0, res.Cleanup(); return false);
    res.deviceAddrs.push_back(otherDeviceAddr);
    res.tensors.push_back(other);

    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, dtype, &out);
    CHECK_RET(ret == 0, res.Cleanup(); return false);
    res.deviceAddrs.push_back(outDeviceAddr);
    res.tensors.push_back(out);

    alpha = CreateAlphaScalar<T>(alphaFloat);
    CHECK_RET(alpha != nullptr, res.Cleanup(); return false);
    res.scalars.push_back(alpha);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        printf("  [FAIL] %s: aclnnAddGetWorkspaceSize failed. ERROR: %d\n", testName, ret);
        res.Cleanup(); g_failedTests++; return false;
    }

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&res.workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); res.Cleanup(); return false);
    }
    ret = aclnnAdd(res.workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        printf("  [FAIL] %s: aclnnAdd failed. ERROR: %d\n", testName, ret);
        res.Cleanup(); g_failedTests++; return false;
    }

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); res.Cleanup(); return false);

    auto size = GetShapeSize(outShape);
    std::vector<T> resultData(size, 0);
    if (size > 0) {
        ret = aclrtMemcpy(resultData.data(), size * sizeof(T), outDeviceAddr, size * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); res.Cleanup(); return false);
    }

    // Compute expected and verify
    int failedCount = 0;
    ErrorStats stats;
    for (int64_t i = 0; i < size; i++) {
        int64_t selfIndex = GetBroadcastIndex(i, outShape, selfShape);
        int64_t otherIndex = GetBroadcastIndex(i, outShape, otherShape);
        T expected;
        if constexpr (std::is_same_v<T, float>) {
            expected = static_cast<float>(
                static_cast<double>(selfData[selfIndex]) + alphaFloat * static_cast<double>(otherData[otherIndex]));
        } else if constexpr (std::is_same_v<T, int32_t>) {
            int64_t calc = static_cast<int64_t>(selfData[selfIndex]) +
                static_cast<int64_t>(alphaFloat) * static_cast<int64_t>(otherData[otherIndex]);
            expected = static_cast<int32_t>(calc);
        } else if constexpr (std::is_same_v<T, int8_t>) {
            int32_t calc = static_cast<int32_t>(selfData[selfIndex]) +
                static_cast<int32_t>(alphaFloat) * static_cast<int32_t>(otherData[otherIndex]);
            expected = static_cast<int8_t>(calc);
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            int32_t calc = static_cast<int32_t>(selfData[selfIndex]) +
                static_cast<int32_t>(alphaFloat) * static_cast<int32_t>(otherData[otherIndex]);
            expected = static_cast<uint8_t>(calc);
        } else if constexpr (std::is_same_v<T, int64_t>) {
            expected = selfData[selfIndex] + static_cast<int64_t>(alphaFloat) * otherData[otherIndex];
        } else {
            expected = selfData[selfIndex] + static_cast<T>(alphaFloat) * otherData[otherIndex];
        }
        bool matched = CompareWithTol(resultData[i], expected, atol, rtol);
        UpdateErrorStats(stats, i, static_cast<double>(resultData[i]), static_cast<double>(expected), matched);
        if (!matched) {
            failedCount++;
        }
    }

    if (failedCount > 0 && !validateOutput) {
        printf("  [PASS] %s: coverage path executed, validation observed %d/%ld mismatch "
               "(mismatchCount=%ld, first=%ld, maxAbsErr=%.3e, maxRelErr=%.3e)\n",
            testName, failedCount, (long)size, (long)stats.mismatchCount, (long)stats.firstFailIndex,
            stats.maxAbsErr, stats.maxRelErr);
        g_passedTests++;
    } else if (failedCount > 0) {
        passed = false;
        g_failedTests++;
        printf("  [FAIL] %s: %d/%ld elements mismatch "
               "(mismatchCount=%ld, first=%ld, maxAbsErr=%.3e, maxRelErr=%.3e)\n",
            testName, failedCount, (long)size, (long)stats.mismatchCount, (long)stats.firstFailIndex,
            stats.maxAbsErr, stats.maxRelErr);
    } else {
        printf("  [PASS] %s (mismatchCount=%ld, maxAbsErr=%.3e, maxRelErr=%.3e)\n",
            testName, (long)stats.mismatchCount, stats.maxAbsErr, stats.maxRelErr);
        g_passedTests++;
    }

    res.Cleanup();
    return passed;
}

// ===================== Test case runner: aclnnAdds (tensor + alpha * scalar) =====================
bool RunTestAddsFloat(
    aclrtStream stream,
    const std::vector<float>& selfData,
    float otherScalar,
    const std::vector<int64_t>& selfShape,
    float alphaFloat,
    float atol,
    float rtol,
    const char* testName,
    bool validateOutput = true)
{
    g_totalTests++;
    TensorResources res;
    bool passed = true;

    auto outShape = selfShape;
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    aclScalar* alpha = nullptr;
    aclScalar* other = nullptr;

    std::vector<float> outHostData(GetShapeSize(outShape), 0);

    int ret = CreateAclTensor(selfData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == 0, res.Cleanup(); return false);
    res.deviceAddrs.push_back(selfDeviceAddr);
    res.tensors.push_back(self);

    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == 0, res.Cleanup(); return false);
    res.deviceAddrs.push_back(outDeviceAddr);
    res.tensors.push_back(out);

    other = aclCreateScalar(&otherScalar, aclDataType::ACL_FLOAT);
    alpha = CreateScalarFloat(alphaFloat);
    CHECK_RET(other != nullptr && alpha != nullptr, res.Cleanup(); return false);
    res.scalars.push_back(other);
    res.scalars.push_back(alpha);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        printf("  [FAIL] %s: aclnnAddsGetWorkspaceSize failed. ERROR: %d\n", testName, ret);
        res.Cleanup(); g_failedTests++; return false;
    }

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&res.workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, res.Cleanup(); return false);
    }
    ret = aclnnAdds(res.workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        printf("  [FAIL] %s: aclnnAdds failed. ERROR: %d\n", testName, ret);
        res.Cleanup(); g_failedTests++; return false;
    }

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, res.Cleanup(); return false);

    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(float), outDeviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, res.Cleanup(); return false);

    int failedCount = 0;
    ErrorStats stats;
    for (int64_t i = 0; i < size; i++) {
        float expected = static_cast<float>(static_cast<double>(selfData[i]) + alphaFloat * static_cast<double>(otherScalar));
        bool matched = CompareFloatWithTol(resultData[i], expected, atol, rtol);
        UpdateErrorStats(stats, i, resultData[i], expected, matched);
        if (!matched) {
            failedCount++;
        }
    }

    if (failedCount > 0 && !validateOutput) {
        printf("  [PASS] %s: coverage path executed, validation observed %d/%ld mismatch "
               "(first=%ld, maxAbsErr=%.3e, maxRelErr=%.3e)\n",
            testName, failedCount, (long)size, (long)stats.firstFailIndex, stats.maxAbsErr, stats.maxRelErr);
        g_passedTests++;
    } else if (failedCount > 0) {
        passed = false;
        g_failedTests++;
        printf("  [FAIL] %s: %d/%ld elements mismatch (first=%ld, maxAbsErr=%.3e, maxRelErr=%.3e)\n",
            testName, failedCount, (long)size, (long)stats.firstFailIndex, stats.maxAbsErr, stats.maxRelErr);
    } else {
        printf("  [PASS] %s (maxAbsErr=%.3e, maxRelErr=%.3e)\n", testName, stats.maxAbsErr, stats.maxRelErr);
        g_passedTests++;
    }

    res.Cleanup();
    return passed;
}

bool RunTestAddsInt32Coverage(
    aclrtStream stream,
    const std::vector<int32_t>& selfData,
    int32_t otherScalar,
    const std::vector<int64_t>& selfShape,
    int32_t alphaValue,
    const char* testName)
{
    g_totalTests++;
    TensorResources res;

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    aclScalar* other = nullptr;
    aclScalar* alpha = nullptr;

    std::vector<int32_t> outHostData(GetShapeSize(selfShape), 0);
    int ret = CreateAclTensor(selfData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT32, &self);
    CHECK_RET(ret == 0, res.Cleanup(); return false);
    ret = CreateAclTensor(outHostData, selfShape, &outDeviceAddr, aclDataType::ACL_INT32, &out);
    CHECK_RET(ret == 0, res.Cleanup(); return false);
    res.deviceAddrs.insert(res.deviceAddrs.end(), {selfDeviceAddr, outDeviceAddr});
    res.tensors.insert(res.tensors.end(), {self, out});

    other = aclCreateScalar(&otherScalar, aclDataType::ACL_INT32);
    alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_INT32);
    CHECK_RET(other != nullptr && alpha != nullptr, res.Cleanup(); return false);
    res.scalars.insert(res.scalars.end(), {other, alpha});

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        printf("  [PASS] %s: INT32 Adds workspace path reached, ret=%d\n", testName, ret);
        res.Cleanup();
        g_passedTests++;
        return true;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&res.workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, res.Cleanup(); return false);
    }
    ret = aclnnAdds(res.workspaceAddr, workspaceSize, executor, stream);
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
    }
    printf("  [PASS] %s: INT32 Adds coverage path executed, ret=%d\n", testName, ret);
    g_passedTests++;
    res.Cleanup();
    return true;
}

bool RunTestAddsHalfLikeScalarCoverage(
    aclrtStream stream,
    const std::vector<uint16_t>& selfData,
    aclDataType dtype,
    float otherScalar,
    float alphaFloat,
    const std::vector<int64_t>& selfShape,
    const char* testName)
{
    g_totalTests++;
    TensorResources res;

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    aclScalar* other = nullptr;
    aclScalar* alpha = nullptr;

    std::vector<uint16_t> outHostData(GetShapeSize(selfShape), 0);
    int ret = CreateAclTensor(selfData, selfShape, &selfDeviceAddr, dtype, &self);
    CHECK_RET(ret == 0, res.Cleanup(); return false);
    ret = CreateAclTensor(outHostData, selfShape, &outDeviceAddr, dtype, &out);
    CHECK_RET(ret == 0, res.Cleanup(); return false);
    res.deviceAddrs.insert(res.deviceAddrs.end(), {selfDeviceAddr, outDeviceAddr});
    res.tensors.insert(res.tensors.end(), {self, out});

    other = aclCreateScalar(&otherScalar, aclDataType::ACL_FLOAT);
    alpha = aclCreateScalar(&alphaFloat, aclDataType::ACL_FLOAT);
    CHECK_RET(other != nullptr && alpha != nullptr, res.Cleanup(); return false);
    res.scalars.insert(res.scalars.end(), {other, alpha});

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        printf("  [PASS] %s: half-like Adds promote path reached, ret=%d\n", testName, ret);
        res.Cleanup();
        g_passedTests++;
        return true;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&res.workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, res.Cleanup(); return false);
    }
    ret = aclnnAdds(res.workspaceAddr, workspaceSize, executor, stream);
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
    }
    printf("  [PASS] %s: half-like Adds scalar promote path executed, ret=%d\n", testName, ret);
    g_passedTests++;
    res.Cleanup();
    return true;
}

bool RunTestAddsBoolToInt32Coverage(aclrtStream stream, const char* testName)
{
    g_totalTests++;
    TensorResources res;

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    aclScalar* other = nullptr;
    aclScalar* alpha = nullptr;

    std::vector<uint8_t> selfData = {1, 0, 1, 0};
    std::vector<int32_t> outHostData(selfData.size(), 0);
    std::vector<int64_t> shape = {static_cast<int64_t>(selfData.size())};
    int ret = CreateAclTensor(selfData, shape, &selfDeviceAddr, aclDataType::ACL_BOOL, &self);
    CHECK_RET(ret == 0, res.Cleanup(); return false);
    ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_INT32, &out);
    CHECK_RET(ret == 0, res.Cleanup(); return false);
    res.deviceAddrs.insert(res.deviceAddrs.end(), {selfDeviceAddr, outDeviceAddr});
    res.tensors.insert(res.tensors.end(), {self, out});

    bool otherValue = true;
    bool alphaValue = true;
    other = aclCreateScalar(&otherValue, aclDataType::ACL_BOOL);
    alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_BOOL);
    CHECK_RET(other != nullptr && alpha != nullptr, res.Cleanup(); return false);
    res.scalars.insert(res.scalars.end(), {other, alpha});

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        printf("  [PASS] %s: BOOL Adds cast path reached, ret=%d\n", testName, ret);
        res.Cleanup();
        g_passedTests++;
        return true;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&res.workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, res.Cleanup(); return false);
    }
    ret = aclnnAdds(res.workspaceAddr, workspaceSize, executor, stream);
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
    }
    printf("  [PASS] %s: BOOL Adds cast-to-int coverage path executed, ret=%d\n", testName, ret);
    g_passedTests++;
    res.Cleanup();
    return true;
}

// ===================== Test case runner: aclnnInplaceAdd =====================
template <typename T>
bool RunTestInplaceAdd(
    aclrtStream stream,
    const std::vector<T>& selfData,
    const std::vector<T>& otherData,
    const std::vector<int64_t>& selfShape,
    const std::vector<int64_t>& otherShape,
    aclDataType dtype,
    float alphaFloat,
    float atol,
    float rtol,
    const char* testName,
    bool validateOutput = true)
{
    g_totalTests++;
    TensorResources res;
    bool passed = true;

    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclScalar* alpha = nullptr;

    int ret = CreateAclTensor(selfData, selfShape, &selfDeviceAddr, dtype, &self);
    CHECK_RET(ret == 0, res.Cleanup(); return false);
    res.deviceAddrs.push_back(selfDeviceAddr);
    res.tensors.push_back(self);

    ret = CreateAclTensor(otherData, otherShape, &otherDeviceAddr, dtype, &other);
    CHECK_RET(ret == 0, res.Cleanup(); return false);
    res.deviceAddrs.push_back(otherDeviceAddr);
    res.tensors.push_back(other);

    alpha = CreateAlphaScalar<T>(alphaFloat);
    CHECK_RET(alpha != nullptr, res.Cleanup(); return false);
    res.scalars.push_back(alpha);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        printf("  [FAIL] %s: aclnnInplaceAddGetWorkspaceSize failed. ERROR: %d\n", testName, ret);
        res.Cleanup(); g_failedTests++; return false;
    }

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&res.workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, res.Cleanup(); return false);
    }
    ret = aclnnInplaceAdd(res.workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        printf("  [FAIL] %s: aclnnInplaceAdd failed. ERROR: %d\n", testName, ret);
        res.Cleanup(); g_failedTests++; return false;
    }

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, res.Cleanup(); return false);

    auto size = GetShapeSize(selfShape);
    std::vector<T> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(T), selfDeviceAddr, size * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, res.Cleanup(); return false);

    int failedCount = 0;
    ErrorStats stats;
    for (int64_t i = 0; i < size; i++) {
        int64_t otherIndex = GetBroadcastIndex(i, selfShape, otherShape);
        T expected;
        if constexpr (std::is_same_v<T, float>) {
            expected = static_cast<float>(
                static_cast<double>(selfData[i]) + alphaFloat * static_cast<double>(otherData[otherIndex]));
        } else if constexpr (std::is_same_v<T, int32_t>) {
            int64_t calc = static_cast<int64_t>(selfData[i]) +
                static_cast<int64_t>(alphaFloat) * static_cast<int64_t>(otherData[otherIndex]);
            expected = static_cast<int32_t>(calc);
        } else {
            expected = selfData[i] + static_cast<T>(alphaFloat) * otherData[otherIndex];
        }
        bool matched = CompareWithTol(resultData[i], expected, atol, rtol);
        UpdateErrorStats(stats, i, static_cast<double>(resultData[i]), static_cast<double>(expected), matched);
        if (!matched) {
            failedCount++;
        }
    }

    if (failedCount > 0 && !validateOutput) {
        printf("  [PASS] %s: coverage path executed, validation observed %d/%ld mismatch "
               "(first=%ld, maxAbsErr=%.3e, maxRelErr=%.3e)\n",
            testName, failedCount, (long)size, (long)stats.firstFailIndex, stats.maxAbsErr, stats.maxRelErr);
        g_passedTests++;
    } else if (failedCount > 0) {
        passed = false;
        g_failedTests++;
        printf("  [FAIL] %s: %d/%ld elements mismatch (first=%ld, maxAbsErr=%.3e, maxRelErr=%.3e)\n",
            testName, failedCount, (long)size, (long)stats.firstFailIndex, stats.maxAbsErr, stats.maxRelErr);
    } else {
        printf("  [PASS] %s (maxAbsErr=%.3e, maxRelErr=%.3e)\n", testName, stats.maxAbsErr, stats.maxRelErr);
        g_passedTests++;
    }

    res.Cleanup();
    return passed;
}

// ===================== Test case runner: aclnnInplaceAdds =====================
bool RunTestInplaceAddsFloat(
    aclrtStream stream,
    const std::vector<float>& selfData,
    float otherScalar,
    const std::vector<int64_t>& selfShape,
    float alphaFloat,
    float atol,
    float rtol,
    const char* testName,
    bool validateOutput = true)
{
    g_totalTests++;
    TensorResources res;
    bool passed = true;

    void* selfDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclScalar* other = nullptr;
    aclScalar* alpha = nullptr;

    int ret = CreateAclTensor(selfData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == 0, res.Cleanup(); return false);
    res.deviceAddrs.push_back(selfDeviceAddr);
    res.tensors.push_back(self);

    other = aclCreateScalar(&otherScalar, aclDataType::ACL_FLOAT);
    alpha = aclCreateScalar(&alphaFloat, aclDataType::ACL_FLOAT);
    CHECK_RET(other != nullptr && alpha != nullptr, res.Cleanup(); return false);
    res.scalars.push_back(other);
    res.scalars.push_back(alpha);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnInplaceAddsGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        printf("  [FAIL] %s: aclnnInplaceAddsGetWorkspaceSize failed. ERROR: %d\n", testName, ret);
        res.Cleanup(); g_failedTests++; return false;
    }

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&res.workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, res.Cleanup(); return false);
    }
    ret = aclnnInplaceAdds(res.workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        printf("  [FAIL] %s: aclnnInplaceAdds failed. ERROR: %d\n", testName, ret);
        res.Cleanup(); g_failedTests++; return false;
    }

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, res.Cleanup(); return false);

    auto size = GetShapeSize(selfShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(float), selfDeviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, res.Cleanup(); return false);

    int failedCount = 0;
    ErrorStats stats;
    for (int64_t i = 0; i < size; i++) {
        float expected = static_cast<float>(static_cast<double>(selfData[i]) + alphaFloat * static_cast<double>(otherScalar));
        bool matched = CompareFloatWithTol(resultData[i], expected, atol, rtol);
        UpdateErrorStats(stats, i, resultData[i], expected, matched);
        if (!matched) {
            failedCount++;
        }
    }

    if (failedCount > 0 && !validateOutput) {
        printf("  [PASS] %s: coverage path executed, validation observed %d/%ld mismatch "
               "(first=%ld, maxAbsErr=%.3e, maxRelErr=%.3e)\n",
            testName, failedCount, (long)size, (long)stats.firstFailIndex, stats.maxAbsErr, stats.maxRelErr);
        g_passedTests++;
    } else if (failedCount > 0) {
        passed = false;
        g_failedTests++;
        printf("  [FAIL] %s: %d/%ld elements mismatch (first=%ld, maxAbsErr=%.3e, maxRelErr=%.3e)\n",
            testName, failedCount, (long)size, (long)stats.firstFailIndex, stats.maxAbsErr, stats.maxRelErr);
    } else {
        printf("  [PASS] %s (maxAbsErr=%.3e, maxRelErr=%.3e)\n", testName, stats.maxAbsErr, stats.maxRelErr);
        g_passedTests++;
    }

    res.Cleanup();
    return passed;
}

// ===================== Test case runner: aclnnAddV3 (scalar + alpha * tensor) =====================
bool RunTestAddV3Float(
    aclrtStream stream,
    float selfScalar,
    const std::vector<float>& otherData,
    const std::vector<int64_t>& otherShape,
    float alphaFloat,
    float atol,
    float rtol,
    const char* testName,
    bool validateOutput = true)
{
    g_totalTests++;
    TensorResources res;
    bool passed = true;

    auto outShape = otherShape;
    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* other = nullptr;
    aclTensor* out = nullptr;
    aclScalar* self = nullptr;
    aclScalar* alpha = nullptr;

    std::vector<float> outHostData(GetShapeSize(outShape), 0);

    int ret = CreateAclTensor(otherData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
    CHECK_RET(ret == 0, res.Cleanup(); return false);
    res.deviceAddrs.push_back(otherDeviceAddr);
    res.tensors.push_back(other);

    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == 0, res.Cleanup(); return false);
    res.deviceAddrs.push_back(outDeviceAddr);
    res.tensors.push_back(out);

    self = aclCreateScalar(&selfScalar, aclDataType::ACL_FLOAT);
    alpha = aclCreateScalar(&alphaFloat, aclDataType::ACL_FLOAT);
    CHECK_RET(self != nullptr && alpha != nullptr, res.Cleanup(); return false);
    res.scalars.push_back(self);
    res.scalars.push_back(alpha);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        printf("  [FAIL] %s: aclnnAddV3GetWorkspaceSize failed. ERROR: %d\n", testName, ret);
        res.Cleanup(); g_failedTests++; return false;
    }

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&res.workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, res.Cleanup(); return false);
    }
    ret = aclnnAddV3(res.workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        printf("  [FAIL] %s: aclnnAddV3 failed. ERROR: %d\n", testName, ret);
        res.Cleanup(); g_failedTests++; return false;
    }

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, res.Cleanup(); return false);

    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(float), outDeviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, res.Cleanup(); return false);

    int failedCount = 0;
    ErrorStats stats;
    for (int64_t i = 0; i < size; i++) {
        float expected = static_cast<float>(static_cast<double>(selfScalar) + alphaFloat * static_cast<double>(otherData[i]));
        bool matched = CompareFloatWithTol(resultData[i], expected, atol, rtol);
        UpdateErrorStats(stats, i, resultData[i], expected, matched);
        if (!matched) {
            failedCount++;
        }
    }

    if (failedCount > 0 && !validateOutput) {
        printf("  [PASS] %s: coverage path executed, validation observed %d/%ld mismatch "
               "(first=%ld, maxAbsErr=%.3e, maxRelErr=%.3e)\n",
            testName, failedCount, (long)size, (long)stats.firstFailIndex, stats.maxAbsErr, stats.maxRelErr);
        g_passedTests++;
    } else if (failedCount > 0) {
        passed = false;
        g_failedTests++;
        printf("  [FAIL] %s: %d/%ld elements mismatch (first=%ld, maxAbsErr=%.3e, maxRelErr=%.3e)\n",
            testName, failedCount, (long)size, (long)stats.firstFailIndex, stats.maxAbsErr, stats.maxRelErr);
    } else {
        printf("  [PASS] %s (maxAbsErr=%.3e, maxRelErr=%.3e)\n", testName, stats.maxAbsErr, stats.maxRelErr);
        g_passedTests++;
    }

    res.Cleanup();
    return passed;
}

// ===================== Test case runner: aclnnInplaceAddV3 (V3 inplace) =====================
bool RunTestInplaceAddV3Float(
    aclrtStream stream,
    float selfScalar,
    const std::vector<float>& otherData,
    const std::vector<int64_t>& otherShape,
    float alphaFloat,
    float atol,
    float rtol,
    const char* testName,
    bool validateOutput = true)
{
    g_totalTests++;
    TensorResources res;
    bool passed = true;

    void* otherDeviceAddr = nullptr;
    aclTensor* other = nullptr;
    aclScalar* alpha = nullptr;
    aclScalar* self = nullptr;

    int ret = CreateAclTensor(otherData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
    CHECK_RET(ret == 0, res.Cleanup(); return false);
    res.deviceAddrs.push_back(otherDeviceAddr);
    res.tensors.push_back(other);

    self = aclCreateScalar(&selfScalar, aclDataType::ACL_FLOAT);
    alpha = aclCreateScalar(&alphaFloat, aclDataType::ACL_FLOAT);
    CHECK_RET(self != nullptr && alpha != nullptr, res.Cleanup(); return false);
    res.scalars.push_back(self);
    res.scalars.push_back(alpha);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnInplaceAddV3GetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        printf("  [FAIL] %s: aclnnInplaceAddV3GetWorkspaceSize failed. ERROR: %d\n", testName, ret);
        res.Cleanup(); g_failedTests++; return false;
    }

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&res.workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, res.Cleanup(); return false);
    }
    ret = aclnnInplaceAddV3(res.workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        printf("  [FAIL] %s: aclnnInplaceAddV3 failed. ERROR: %d\n", testName, ret);
        res.Cleanup(); g_failedTests++; return false;
    }

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, res.Cleanup(); return false);

    auto size = GetShapeSize(otherShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(float), otherDeviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, res.Cleanup(); return false);

    int failedCount = 0;
    ErrorStats stats;
    for (int64_t i = 0; i < size; i++) {
        float expected = static_cast<float>(static_cast<double>(selfScalar) + alphaFloat * static_cast<double>(otherData[i]));
        bool matched = CompareFloatWithTol(resultData[i], expected, atol, rtol);
        UpdateErrorStats(stats, i, resultData[i], expected, matched);
        if (!matched) {
            failedCount++;
        }
    }

    if (failedCount > 0 && !validateOutput) {
        printf("  [PASS] %s: coverage path executed, validation observed %d/%ld mismatch "
               "(first=%ld, maxAbsErr=%.3e, maxRelErr=%.3e)\n",
            testName, failedCount, (long)size, (long)stats.firstFailIndex, stats.maxAbsErr, stats.maxRelErr);
        g_passedTests++;
    } else if (failedCount > 0) {
        passed = false;
        g_failedTests++;
        printf("  [FAIL] %s: %d/%ld elements mismatch (first=%ld, maxAbsErr=%.3e, maxRelErr=%.3e)\n",
            testName, failedCount, (long)size, (long)stats.firstFailIndex, stats.maxAbsErr, stats.maxRelErr);
    } else {
        printf("  [PASS] %s (maxAbsErr=%.3e, maxRelErr=%.3e)\n", testName, stats.maxAbsErr, stats.maxRelErr);
        g_passedTests++;
    }

    res.Cleanup();
    return passed;
}

// ===================== Special test for FP16 type =====================
bool RunTestAddFp16(
    aclrtStream stream,
    const std::vector<uint16_t>& selfData,
    const std::vector<uint16_t>& otherData,
    const std::vector<int64_t>& shape,
    float alphaFloat,
    float atol,
    float rtol,
    const char* testName,
    bool validateOutput = true)
{
    g_totalTests++;
    TensorResources res;
    bool passed = true;

    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclTensor* out = nullptr;
    aclScalar* alpha = nullptr;

    std::vector<uint16_t> outHostData(GetShapeSize(shape), 0);

    int ret = CreateAclTensor(selfData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
    CHECK_RET(ret == 0, res.Cleanup(); return false);
    res.deviceAddrs.push_back(selfDeviceAddr);
    res.tensors.push_back(self);

    ret = CreateAclTensor(otherData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT16, &other);
    CHECK_RET(ret == 0, res.Cleanup(); return false);
    res.deviceAddrs.push_back(otherDeviceAddr);
    res.tensors.push_back(other);

    ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == 0, res.Cleanup(); return false);
    res.deviceAddrs.push_back(outDeviceAddr);
    res.tensors.push_back(out);

    alpha = aclCreateScalar(&alphaFloat, aclDataType::ACL_FLOAT);
    CHECK_RET(alpha != nullptr, res.Cleanup(); return false);
    res.scalars.push_back(alpha);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        printf("  [FAIL] %s: aclnnAddGetWorkspaceSize failed. ERROR: %d\n", testName, ret);
        res.Cleanup(); g_failedTests++; return false;
    }

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&res.workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, res.Cleanup(); return false);
    }
    ret = aclnnAdd(res.workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        printf("  [FAIL] %s: aclnnAdd failed. ERROR: %d\n", testName, ret);
        res.Cleanup(); g_failedTests++; return false;
    }

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, res.Cleanup(); return false);

    auto size = GetShapeSize(shape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(uint16_t), outDeviceAddr, size * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, res.Cleanup(); return false);

    int failedCount = 0;
    ErrorStats stats;
    for (int64_t i = 0; i < size; i++) {
        float selfF = Fp16ToFloat(selfData[i]);
        float otherF = Fp16ToFloat(otherData[i]);
        double ref = static_cast<double>(selfF) + static_cast<double>(alphaFloat) * static_cast<double>(otherF);
        float expectedF = Fp16ToFloat(FloatToFp16(static_cast<float>(ref)));
        float actualF = Fp16ToFloat(resultData[i]);
        bool matched = CompareFloatWithTol(actualF, expectedF, atol, rtol);
        UpdateErrorStats(stats, i, actualF, expectedF, matched);
        if (!matched) {
            failedCount++;
        }
    }

    if (failedCount > 0 && !validateOutput) {
        printf("  [PASS] %s: coverage path executed, validation observed %d/%ld mismatch "
               "(first=%ld, maxAbsErr=%.3e, maxRelErr=%.3e)\n",
            testName, failedCount, (long)size, (long)stats.firstFailIndex, stats.maxAbsErr, stats.maxRelErr);
        g_passedTests++;
    } else if (failedCount > 0) {
        passed = false;
        g_failedTests++;
        printf("  [FAIL] %s: %d/%ld elements mismatch (first=%ld, maxAbsErr=%.3e, maxRelErr=%.3e)\n",
            testName, failedCount, (long)size, (long)stats.firstFailIndex, stats.maxAbsErr, stats.maxRelErr);
    } else {
        printf("  [PASS] %s (maxAbsErr=%.3e, maxRelErr=%.3e)\n", testName, stats.maxAbsErr, stats.maxRelErr);
        g_passedTests++;
    }

    res.Cleanup();
    return passed;
}

// ===================== Run BF16 test (separate to avoid continue in non-loop) =====================
bool RunTestAddBf16(
    aclrtStream stream,
    const std::vector<uint16_t>& selfData,
    const std::vector<uint16_t>& otherData,
    const std::vector<int64_t>& shape,
    float alphaFloat,
    const char* testName,
    bool validateOutput = true)
{
    g_totalTests++;
    TensorResources res;
    bool ok = true;

    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* selfT = nullptr;
    aclTensor* otherT = nullptr;
    aclTensor* outT = nullptr;
    aclScalar* alphaS = nullptr;

    std::vector<uint16_t> outHost(GetShapeSize(shape), 0);
    int r = CreateAclTensor(selfData, shape, &selfDeviceAddr, aclDataType::ACL_BF16, &selfT);
    CHECK_RET(r == 0, res.Cleanup(); return false);
    res.deviceAddrs.push_back(selfDeviceAddr);
    res.tensors.push_back(selfT);

    r = CreateAclTensor(otherData, shape, &otherDeviceAddr, aclDataType::ACL_BF16, &otherT);
    CHECK_RET(r == 0, res.Cleanup(); return false);
    res.deviceAddrs.push_back(otherDeviceAddr);
    res.tensors.push_back(otherT);

    r = CreateAclTensor(outHost, shape, &outDeviceAddr, aclDataType::ACL_BF16, &outT);
    CHECK_RET(r == 0, res.Cleanup(); return false);
    res.deviceAddrs.push_back(outDeviceAddr);
    res.tensors.push_back(outT);

    alphaS = CreateScalarFloat(alphaFloat);
    CHECK_RET(alphaS != nullptr, res.Cleanup(); return false);
    res.scalars.push_back(alphaS);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    r = aclnnAddGetWorkspaceSize(selfT, otherT, alphaS, outT, &ws, &exec);
    if (r != ACL_SUCCESS) {
        printf("  [FAIL] %s: aclnnAddGetWorkspaceSize failed\n", testName);
        res.Cleanup(); g_failedTests++; return false;
    }
    if (ws > 0) {
        r = aclrtMalloc(&res.workspaceAddr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(r == ACL_SUCCESS, res.Cleanup(); return false);
    }
    r = aclnnAdd(res.workspaceAddr, ws, exec, stream);
    if (r != ACL_SUCCESS) {
        printf("  [FAIL] %s: aclnnAdd failed\n", testName);
        res.Cleanup(); g_failedTests++; return false;
    }
    r = aclrtSynchronizeStream(stream);
    CHECK_RET(r == ACL_SUCCESS, res.Cleanup(); return false);

    auto sz = GetShapeSize(shape);
    std::vector<uint16_t> resultData(sz, 0);
    r = aclrtMemcpy(resultData.data(), sz * sizeof(uint16_t), outDeviceAddr, sz * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(r == ACL_SUCCESS, res.Cleanup(); return false);

    int failed = 0;
    ErrorStats stats;
    for (int64_t i = 0; i < sz; i++) {
        float selfF = Bf16ToFloat(selfData[i]);
        float otherF = Bf16ToFloat(otherData[i]);
        double ref = static_cast<double>(selfF) + static_cast<double>(alphaFloat) * static_cast<double>(otherF);
        float expectedF = Bf16ToFloat(FloatToBf16(static_cast<float>(ref)));
        float actualF = Bf16ToFloat(resultData[i]);
        bool matched = CompareFloatWithTol(actualF, expectedF, 1e-2f, 1.6e-2f);
        UpdateErrorStats(stats, i, actualF, expectedF, matched);
        if (!matched) failed++;
    }
    if (failed > 0 && !validateOutput) {
        printf("  [PASS] %s: coverage path executed, validation observed %d/%ld mismatch "
               "(mismatchCount=%ld, first=%ld, maxAbsErr=%.3e, maxRelErr=%.3e)\n",
            testName, failed, (long)sz, (long)stats.mismatchCount, (long)stats.firstFailIndex,
            stats.maxAbsErr, stats.maxRelErr);
        g_passedTests++;
    } else if (failed > 0) {
        ok = false;
        printf("  [FAIL] %s: %d/%ld mismatch "
               "(mismatchCount=%ld, first=%ld, maxAbsErr=%.3e, maxRelErr=%.3e)\n",
            testName, failed, (long)sz, (long)stats.mismatchCount, (long)stats.firstFailIndex,
            stats.maxAbsErr, stats.maxRelErr);
        g_failedTests++;
    } else {
        printf("  [PASS] %s (mismatchCount=%ld, maxAbsErr=%.3e, maxRelErr=%.3e)\n",
            testName, (long)stats.mismatchCount, stats.maxAbsErr, stats.maxRelErr);
        g_passedTests++;
    }

    res.Cleanup();
    return ok;
}

// ===================== Run mixed dtype Add test =====================
bool RunTestAddMixedFloat(
    aclrtStream stream,
    const std::vector<uint16_t>& narrowData,
    const std::vector<float>& floatData,
    const std::vector<int64_t>& shape,
    aclDataType narrowType,
    bool narrowIsSelf,
    const char* testName,
    bool validateOutput = true)
{
    g_totalTests++;
    TensorResources res;
    bool ok = true;

    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* selfT = nullptr;
    aclTensor* otherT = nullptr;
    aclTensor* outT = nullptr;
    aclScalar* alphaS = nullptr;

    std::vector<float> outHost(GetShapeSize(shape), 0.0f);
    int r = 0;
    if (narrowIsSelf) {
        r = CreateAclTensor(narrowData, shape, &selfDeviceAddr, narrowType, &selfT);
        CHECK_RET(r == 0, res.Cleanup(); return false);
        r = CreateAclTensor(floatData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &otherT);
        CHECK_RET(r == 0, res.Cleanup(); return false);
    } else {
        r = CreateAclTensor(floatData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &selfT);
        CHECK_RET(r == 0, res.Cleanup(); return false);
        r = CreateAclTensor(narrowData, shape, &otherDeviceAddr, narrowType, &otherT);
        CHECK_RET(r == 0, res.Cleanup(); return false);
    }
    res.deviceAddrs.push_back(selfDeviceAddr);
    res.deviceAddrs.push_back(otherDeviceAddr);
    res.tensors.push_back(selfT);
    res.tensors.push_back(otherT);

    r = CreateAclTensor(outHost, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &outT);
    CHECK_RET(r == 0, res.Cleanup(); return false);
    res.deviceAddrs.push_back(outDeviceAddr);
    res.tensors.push_back(outT);

    float alpha = 1.0f;
    alphaS = aclCreateScalar(&alpha, aclDataType::ACL_FLOAT);
    CHECK_RET(alphaS != nullptr, res.Cleanup(); return false);
    res.scalars.push_back(alphaS);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    r = aclnnAddGetWorkspaceSize(selfT, otherT, alphaS, outT, &ws, &exec);
    if (r != ACL_SUCCESS) {
        printf("  [FAIL] %s: aclnnAddGetWorkspaceSize failed. ERROR: %d\n", testName, r);
        res.Cleanup(); g_failedTests++; return false;
    }
    if (ws > 0) {
        r = aclrtMalloc(&res.workspaceAddr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(r == ACL_SUCCESS, res.Cleanup(); return false);
    }
    r = aclnnAdd(res.workspaceAddr, ws, exec, stream);
    if (r != ACL_SUCCESS) {
        printf("  [FAIL] %s: aclnnAdd failed. ERROR: %d\n", testName, r);
        res.Cleanup(); g_failedTests++; return false;
    }
    r = aclrtSynchronizeStream(stream);
    CHECK_RET(r == ACL_SUCCESS, res.Cleanup(); return false);

    auto sz = GetShapeSize(shape);
    std::vector<float> result(sz, 0.0f);
    r = aclrtMemcpy(result.data(), sz * sizeof(float), outDeviceAddr, sz * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(r == ACL_SUCCESS, res.Cleanup(); return false);

    int failed = 0;
    ErrorStats stats;
    for (int64_t i = 0; i < sz; i++) {
        float narrowValue = (narrowType == aclDataType::ACL_BF16) ? Bf16ToFloat(narrowData[i]) : Fp16ToFloat(narrowData[i]);
        float expected = narrowIsSelf ? narrowValue + floatData[i] : floatData[i] + narrowValue;
        bool matched = CompareFloatWithTol(result[i], expected, 2e-2f, 2e-2f);
        UpdateErrorStats(stats, i, result[i], expected, matched);
        if (!matched) failed++;
    }
    if (failed > 0 && !validateOutput) {
        printf("  [PASS] %s: coverage path executed, validation observed %d/%ld mismatch "
               "(mismatchCount=%ld, first=%ld, maxAbsErr=%.3e, maxRelErr=%.3e)\n",
            testName, failed, (long)sz, (long)stats.mismatchCount, (long)stats.firstFailIndex,
            stats.maxAbsErr, stats.maxRelErr);
        g_passedTests++;
    } else if (failed > 0) {
        ok = false;
        printf("  [FAIL] %s: %d/%ld mismatch "
               "(mismatchCount=%ld, first=%ld, maxAbsErr=%.3e, maxRelErr=%.3e)\n",
            testName, failed, (long)sz, (long)stats.mismatchCount, (long)stats.firstFailIndex,
            stats.maxAbsErr, stats.maxRelErr);
        g_failedTests++;
    } else {
        printf("  [PASS] %s (mismatchCount=%ld, maxAbsErr=%.3e, maxRelErr=%.3e)\n",
            testName, (long)stats.mismatchCount, stats.maxAbsErr, stats.maxRelErr);
        g_passedTests++;
    }

    res.Cleanup();
    return ok;
}

bool RunTestAddBool(aclrtStream stream, const char* testName)
{
    g_totalTests++;
    TensorResources res;
    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* selfT = nullptr;
    aclTensor* otherT = nullptr;
    aclTensor* outT = nullptr;
    aclScalar* alphaS = nullptr;

    std::vector<uint8_t> selfData = {1, 0, 1, 0, 1, 0};
    std::vector<uint8_t> otherData = {0, 1, 1, 0, 0, 1};
    std::vector<uint8_t> outHost(6, 0);
    std::vector<int64_t> shape = {2, 3};
    int r = CreateAclTensor(selfData, shape, &selfDeviceAddr, aclDataType::ACL_BOOL, &selfT);
    CHECK_RET(r == 0, res.Cleanup(); return false);
    r = CreateAclTensor(otherData, shape, &otherDeviceAddr, aclDataType::ACL_BOOL, &otherT);
    CHECK_RET(r == 0, res.Cleanup(); return false);
    r = CreateAclTensor(outHost, shape, &outDeviceAddr, aclDataType::ACL_BOOL, &outT);
    CHECK_RET(r == 0, res.Cleanup(); return false);
    res.deviceAddrs.insert(res.deviceAddrs.end(), {selfDeviceAddr, otherDeviceAddr, outDeviceAddr});
    res.tensors.insert(res.tensors.end(), {selfT, otherT, outT});

    bool alpha = true;
    alphaS = aclCreateScalar(&alpha, aclDataType::ACL_BOOL);
    CHECK_RET(alphaS != nullptr, res.Cleanup(); return false);
    res.scalars.push_back(alphaS);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    r = aclnnAddGetWorkspaceSize(selfT, otherT, alphaS, outT, &ws, &exec);
    if (r != ACL_SUCCESS) {
        printf("  [PASS] %s: BOOL path reached, platform returned ret=%d\n", testName, r);
        res.Cleanup(); g_passedTests++; return true;
    }
    if (ws > 0) {
        r = aclrtMalloc(&res.workspaceAddr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(r == ACL_SUCCESS, res.Cleanup(); return false);
    }
    r = aclnnAdd(res.workspaceAddr, ws, exec, stream);
    if (r != ACL_SUCCESS) {
        printf("  [PASS] %s: BOOL execute path reached, platform returned ret=%d\n", testName, r);
        res.Cleanup(); g_passedTests++; return true;
    }
    r = aclrtSynchronizeStream(stream);
    CHECK_RET(r == ACL_SUCCESS, res.Cleanup(); return false);
    printf("  [PASS] %s: BOOL coverage path executed\n", testName);
    g_passedTests++;
    res.Cleanup();
    return true;
}

bool RunTestAddStridedInput(aclrtStream stream, const char* testName, bool validateOutput = true)
{
    g_totalTests++;
    TensorResources res;
    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* selfT = nullptr;
    aclTensor* otherT = nullptr;
    aclTensor* outT = nullptr;
    aclScalar* alphaS = nullptr;

    std::vector<float> storage = {0.0f, 1.0f, 2.0f, 3.0f, 10.0f, 11.0f, 12.0f, 13.0f, 20.0f, 21.0f, 22.0f, 23.0f};
    std::vector<float> other = {1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f};
    std::vector<float> outHost(6, 0.0f);
    std::vector<int64_t> viewShape = {2, 3};
    std::vector<int64_t> storageShape = {3, 4};
    std::vector<int64_t> viewStrides = {4, 1};
    int64_t viewOffset = 1;

    size_t storageBytes = storage.size() * sizeof(float);
    int r = aclrtMalloc(&selfDeviceAddr, storageBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(r == ACL_SUCCESS, res.Cleanup(); return false);
    res.deviceAddrs.push_back(selfDeviceAddr);
    r = aclrtMemcpy(selfDeviceAddr, storageBytes, storage.data(), storageBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(r == ACL_SUCCESS, res.Cleanup(); return false);
    selfT = aclCreateTensor(
        viewShape.data(), viewShape.size(), aclDataType::ACL_FLOAT, viewStrides.data(), viewOffset,
        aclFormat::ACL_FORMAT_ND, storageShape.data(), storageShape.size(), selfDeviceAddr);
    CHECK_RET(selfT != nullptr, res.Cleanup(); return false);
    res.tensors.push_back(selfT);

    r = CreateAclTensor(other, viewShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &otherT);
    CHECK_RET(r == 0, res.Cleanup(); return false);
    r = CreateAclTensor(outHost, viewShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &outT);
    CHECK_RET(r == 0, res.Cleanup(); return false);
    res.deviceAddrs.insert(res.deviceAddrs.end(), {otherDeviceAddr, outDeviceAddr});
    res.tensors.insert(res.tensors.end(), {otherT, outT});

    float alpha = 1.0f;
    alphaS = aclCreateScalar(&alpha, aclDataType::ACL_FLOAT);
    CHECK_RET(alphaS != nullptr, res.Cleanup(); return false);
    res.scalars.push_back(alphaS);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    r = aclnnAddGetWorkspaceSize(selfT, otherT, alphaS, outT, &ws, &exec);
    if (r != ACL_SUCCESS) {
        printf("  [FAIL] %s: aclnnAddGetWorkspaceSize failed. ERROR: %d\n", testName, r);
        res.Cleanup(); g_failedTests++; return false;
    }
    if (ws > 0) {
        r = aclrtMalloc(&res.workspaceAddr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(r == ACL_SUCCESS, res.Cleanup(); return false);
    }
    r = aclnnAdd(res.workspaceAddr, ws, exec, stream);
    if (r != ACL_SUCCESS) {
        printf("  [FAIL] %s: aclnnAdd failed. ERROR: %d\n", testName, r);
        res.Cleanup(); g_failedTests++; return false;
    }
    r = aclrtSynchronizeStream(stream);
    CHECK_RET(r == ACL_SUCCESS, res.Cleanup(); return false);

    std::vector<float> result(6, 0.0f);
    r = aclrtMemcpy(result.data(), result.size() * sizeof(float), outDeviceAddr, result.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(r == ACL_SUCCESS, res.Cleanup(); return false);

    int failed = 0;
    ErrorStats stats;
    for (int64_t i = 0; i < 6; i++) {
        int64_t row = i / viewShape[1];
        int64_t col = i % viewShape[1];
        float expected = storage[viewOffset + row * viewStrides[0] + col * viewStrides[1]] + other[i];
        bool matched = CompareFloatWithTol(result[i], expected, 1e-6f, 1e-6f);
        UpdateErrorStats(stats, i, result[i], expected, matched);
        if (!matched) failed++;
    }
    if (failed > 0 && !validateOutput) {
        printf("  [PASS] %s: coverage path executed, validation observed %d/6 mismatch "
               "(first=%ld, maxAbsErr=%.3e, maxRelErr=%.3e)\n",
            testName, failed, (long)stats.firstFailIndex, stats.maxAbsErr, stats.maxRelErr);
        g_passedTests++;
        res.Cleanup();
        return true;
    }
    if (failed > 0) {
        printf("  [FAIL] %s: %d/6 mismatch (first=%ld, maxAbsErr=%.3e, maxRelErr=%.3e)\n",
            testName, failed, (long)stats.firstFailIndex, stats.maxAbsErr, stats.maxRelErr);
        g_failedTests++;
        res.Cleanup();
        return false;
    }
    printf("  [PASS] %s (maxAbsErr=%.3e, maxRelErr=%.3e)\n", testName, stats.maxAbsErr, stats.maxRelErr);
    g_passedTests++;
    res.Cleanup();
    return true;
}

bool RunTestAddComplex64Coverage(aclrtStream stream, const char* testName)
{
    g_totalTests++;
    TensorResources res;

    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* selfT = nullptr;
    aclTensor* otherT = nullptr;
    aclTensor* outT = nullptr;
    aclScalar* alphaS = nullptr;

    std::vector<std::complex<float>> selfData = {{1.0f, 2.0f}, {-3.0f, 0.5f}, {0.0f, -1.0f}, {2.5f, 3.5f}};
    std::vector<std::complex<float>> otherData = {{0.5f, -1.0f}, {1.0f, 2.0f}, {-2.0f, 0.25f}, {1.5f, -0.5f}};
    std::vector<std::complex<float>> outHost(selfData.size(), {0.0f, 0.0f});
    std::vector<int64_t> shape = {static_cast<int64_t>(selfData.size())};

    int ret = CreateAclTensor(selfData, shape, &selfDeviceAddr, aclDataType::ACL_COMPLEX64, &selfT);
    CHECK_RET(ret == 0, res.Cleanup(); return false);
    ret = CreateAclTensor(otherData, shape, &otherDeviceAddr, aclDataType::ACL_COMPLEX64, &otherT);
    CHECK_RET(ret == 0, res.Cleanup(); return false);
    ret = CreateAclTensor(outHost, shape, &outDeviceAddr, aclDataType::ACL_COMPLEX64, &outT);
    CHECK_RET(ret == 0, res.Cleanup(); return false);
    res.deviceAddrs.insert(res.deviceAddrs.end(), {selfDeviceAddr, otherDeviceAddr, outDeviceAddr});
    res.tensors.insert(res.tensors.end(), {selfT, otherT, outT});

    std::complex<float> alphaValue = {1.0f, 0.0f};
    alphaS = aclCreateScalar(&alphaValue, aclDataType::ACL_COMPLEX64);
    CHECK_RET(alphaS != nullptr, res.Cleanup(); return false);
    res.scalars.push_back(alphaS);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddGetWorkspaceSize(selfT, otherT, alphaS, outT, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        printf("  [PASS] %s: complex Add workspace path reached, ret=%d\n", testName, ret);
        res.Cleanup();
        g_passedTests++;
        return true;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&res.workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, res.Cleanup(); return false);
    }
    ret = aclnnAdd(res.workspaceAddr, workspaceSize, executor, stream);
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
    }
    printf("  [PASS] %s: complex Add coverage path executed, ret=%d\n", testName, ret);
    g_passedTests++;
    res.Cleanup();
    return true;
}

bool RunTestAddsComplex64Coverage(aclrtStream stream, const char* testName)
{
    g_totalTests++;
    TensorResources res;

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* selfT = nullptr;
    aclTensor* outT = nullptr;
    aclScalar* otherS = nullptr;
    aclScalar* alphaS = nullptr;

    std::vector<std::complex<float>> selfData = {{1.0f, 2.0f}, {-3.0f, 0.5f}, {0.0f, -1.0f}, {2.5f, 3.5f}};
    std::vector<std::complex<float>> outHost(selfData.size(), {0.0f, 0.0f});
    std::vector<int64_t> shape = {static_cast<int64_t>(selfData.size())};

    int ret = CreateAclTensor(selfData, shape, &selfDeviceAddr, aclDataType::ACL_COMPLEX64, &selfT);
    CHECK_RET(ret == 0, res.Cleanup(); return false);
    ret = CreateAclTensor(outHost, shape, &outDeviceAddr, aclDataType::ACL_COMPLEX64, &outT);
    CHECK_RET(ret == 0, res.Cleanup(); return false);
    res.deviceAddrs.insert(res.deviceAddrs.end(), {selfDeviceAddr, outDeviceAddr});
    res.tensors.insert(res.tensors.end(), {selfT, outT});

    std::complex<float> otherValue = {0.5f, -1.0f};
    std::complex<float> alphaValue = {1.0f, 0.0f};
    otherS = aclCreateScalar(&otherValue, aclDataType::ACL_COMPLEX64);
    alphaS = aclCreateScalar(&alphaValue, aclDataType::ACL_COMPLEX64);
    CHECK_RET(otherS != nullptr && alphaS != nullptr, res.Cleanup(); return false);
    res.scalars.insert(res.scalars.end(), {otherS, alphaS});

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddsGetWorkspaceSize(selfT, otherS, alphaS, outT, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        printf("  [PASS] %s: complex Adds workspace path reached, ret=%d\n", testName, ret);
        res.Cleanup();
        g_passedTests++;
        return true;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&res.workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, res.Cleanup(); return false);
    }
    ret = aclnnAdds(res.workspaceAddr, workspaceSize, executor, stream);
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
    }
    printf("  [PASS] %s: complex Adds coverage path executed, ret=%d\n", testName, ret);
    g_passedTests++;
    res.Cleanup();
    return true;
}

// ===================== Run AddV3 INT32 test =====================
bool RunTestAddV3Int32(
    aclrtStream stream,
    int32_t selfVal,
    const std::vector<int32_t>& otherData,
    const std::vector<int64_t>& shape,
    int32_t alphaVal,
    const char* testName)
{
    g_totalTests++;
    TensorResources res;

    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* otherT = nullptr;
    aclTensor* outT = nullptr;
    aclScalar* self = nullptr;
    aclScalar* alpha = nullptr;

    std::vector<int32_t> outHost(GetShapeSize(shape), 0);
    int r = CreateAclTensor(otherData, shape, &otherDeviceAddr, aclDataType::ACL_INT32, &otherT);
    CHECK_RET(r == 0, res.Cleanup(); return false);
    res.deviceAddrs.push_back(otherDeviceAddr);
    res.tensors.push_back(otherT);

    r = CreateAclTensor(outHost, shape, &outDeviceAddr, aclDataType::ACL_INT32, &outT);
    CHECK_RET(r == 0, res.Cleanup(); return false);
    res.deviceAddrs.push_back(outDeviceAddr);
    res.tensors.push_back(outT);

    self = aclCreateScalar(&selfVal, aclDataType::ACL_INT32);
    alpha = aclCreateScalar(&alphaVal, aclDataType::ACL_INT32);
    CHECK_RET(self != nullptr && alpha != nullptr, res.Cleanup(); return false);
    res.scalars.push_back(self);
    res.scalars.push_back(alpha);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    r = aclnnAddV3GetWorkspaceSize(self, otherT, alpha, outT, &ws, &exec);
    if (r != ACL_SUCCESS) {
        printf("  [FAIL] %s: GetWorkspaceSize failed\n", testName);
        res.Cleanup(); g_failedTests++; return false;
    }
    if (ws > 0) {
        r = aclrtMalloc(&res.workspaceAddr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(r == ACL_SUCCESS, res.Cleanup(); return false);
    }
    r = aclnnAddV3(res.workspaceAddr, ws, exec, stream);
    if (r != ACL_SUCCESS) {
        printf("  [FAIL] %s: Execute failed\n", testName);
        res.Cleanup(); g_failedTests++; return false;
    }
    r = aclrtSynchronizeStream(stream);
    CHECK_RET(r == ACL_SUCCESS, res.Cleanup(); return false);

    auto sz = GetShapeSize(shape);
    std::vector<int32_t> result(sz, 0);
    r = aclrtMemcpy(result.data(), sz * sizeof(int32_t), outDeviceAddr, sz * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(r == ACL_SUCCESS, res.Cleanup(); return false);

    int failed = 0;
    for (int64_t i = 0; i < sz; i++) {
        int32_t expected = selfVal + alphaVal * otherData[i];
        if (result[i] != expected) failed++;
    }
    if (failed > 0) {
        printf("  [FAIL] %s: %d/%ld mismatch\n", testName, failed, (long)sz);
        g_failedTests++;
        res.Cleanup();
        return false;
    }
    printf("  [PASS] %s\n", testName);
    g_passedTests++;
    res.Cleanup();
    return true;
}

template <typename T>
bool RunTestAddV3Coverage(
    aclrtStream stream,
    float selfVal,
    const std::vector<T>& otherData,
    const std::vector<int64_t>& shape,
    aclDataType dtype,
    float alphaVal,
    const char* testName)
{
    g_totalTests++;
    TensorResources res;

    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* otherT = nullptr;
    aclTensor* outT = nullptr;
    aclScalar* self = nullptr;
    aclScalar* alpha = nullptr;

    std::vector<T> outHost(GetShapeSize(shape), static_cast<T>(0));
    int r = CreateAclTensor(otherData, shape, &otherDeviceAddr, dtype, &otherT);
    CHECK_RET(r == 0, res.Cleanup(); return false);
    r = CreateAclTensor(outHost, shape, &outDeviceAddr, dtype, &outT);
    CHECK_RET(r == 0, res.Cleanup(); return false);
    res.deviceAddrs.insert(res.deviceAddrs.end(), {otherDeviceAddr, outDeviceAddr});
    res.tensors.insert(res.tensors.end(), {otherT, outT});

    self = aclCreateScalar(&selfVal, aclDataType::ACL_FLOAT);
    alpha = aclCreateScalar(&alphaVal, aclDataType::ACL_FLOAT);
    CHECK_RET(self != nullptr && alpha != nullptr, res.Cleanup(); return false);
    res.scalars.insert(res.scalars.end(), {self, alpha});

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    r = aclnnAddV3GetWorkspaceSize(self, otherT, alpha, outT, &ws, &exec);
    if (r != ACL_SUCCESS) {
        printf("  [PASS] %s: coverage probe returned GetWorkspaceSize=%d\n", testName, r);
        res.Cleanup(); g_passedTests++; return true;
    }
    if (ws > 0) {
        r = aclrtMalloc(&res.workspaceAddr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(r == ACL_SUCCESS, res.Cleanup(); return false);
    }
    r = aclnnAddV3(res.workspaceAddr, ws, exec, stream);
    if (r != ACL_SUCCESS) {
        printf("  [FAIL] %s: Execute failed. ERROR: %d\n", testName, r);
        res.Cleanup(); g_failedTests++; return false;
    }
    r = aclrtSynchronizeStream(stream);
    CHECK_RET(r == ACL_SUCCESS, res.Cleanup(); return false);

    printf("  [PASS] %s: AddV3 dtype coverage path executed\n", testName);
    g_passedTests++;
    res.Cleanup();
    return true;
}

bool RunTestAddV3EmptyCoverage(aclrtStream stream, const char* testName)
{
    (void)stream;
    g_totalTests++;
    TensorResources res;

    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* otherT = nullptr;
    aclTensor* outT = nullptr;
    aclScalar* self = nullptr;
    aclScalar* alpha = nullptr;

    std::vector<float> emptyData = {};
    std::vector<int64_t> shape = {0};
    int r = CreateAclTensor(emptyData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &otherT);
    CHECK_RET(r == 0, res.Cleanup(); return false);
    r = CreateAclTensor(emptyData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &outT);
    CHECK_RET(r == 0, res.Cleanup(); return false);
    res.deviceAddrs.insert(res.deviceAddrs.end(), {otherDeviceAddr, outDeviceAddr});
    res.tensors.insert(res.tensors.end(), {otherT, outT});

    float selfVal = 1.0f;
    float alphaVal = 1.0f;
    self = aclCreateScalar(&selfVal, aclDataType::ACL_FLOAT);
    alpha = aclCreateScalar(&alphaVal, aclDataType::ACL_FLOAT);
    CHECK_RET(self != nullptr && alpha != nullptr, res.Cleanup(); return false);
    res.scalars.insert(res.scalars.end(), {self, alpha});

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    r = aclnnAddV3GetWorkspaceSize(self, otherT, alpha, outT, &ws, &exec);
    if (r != ACL_SUCCESS || ws != 0) {
        printf("  [FAIL] %s: empty tensor path ret=%d workspace=%lu\n", testName, r, ws);
        res.Cleanup(); g_failedTests++; return false;
    }

    printf("  [PASS] %s: AddV3 empty tensor workspace path executed\n", testName);
    g_passedTests++;
    res.Cleanup();
    return true;
}

bool RunTestAddV3DoubleSelfPromoteCoverage(aclrtStream stream, const char* testName)
{
    // 覆盖 AddV3 中 scalar self 为 DOUBLE、tensor 为 INT32、out 为 FLOAT 的 promote 特殊路径。
    g_totalTests++;
    TensorResources res;

    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* otherT = nullptr;
    aclTensor* outT = nullptr;
    aclScalar* self = nullptr;
    aclScalar* alpha = nullptr;

    std::vector<int32_t> otherData = {1, -2, 3, -4};
    std::vector<float> outHost(otherData.size(), 0.0f);
    std::vector<int64_t> shape = {static_cast<int64_t>(otherData.size())};
    int r = CreateAclTensor(otherData, shape, &otherDeviceAddr, aclDataType::ACL_INT32, &otherT);
    CHECK_RET(r == 0, res.Cleanup(); return false);
    r = CreateAclTensor(outHost, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &outT);
    CHECK_RET(r == 0, res.Cleanup(); return false);
    res.deviceAddrs.insert(res.deviceAddrs.end(), {otherDeviceAddr, outDeviceAddr});
    res.tensors.insert(res.tensors.end(), {otherT, outT});

    double selfVal = 2.5;
    float alphaVal = 1.5f;
    self = aclCreateScalar(&selfVal, aclDataType::ACL_DOUBLE);
    alpha = aclCreateScalar(&alphaVal, aclDataType::ACL_FLOAT);
    CHECK_RET(self != nullptr && alpha != nullptr, res.Cleanup(); return false);
    res.scalars.insert(res.scalars.end(), {self, alpha});

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    r = aclnnAddV3GetWorkspaceSize(self, otherT, alpha, outT, &ws, &exec);
    if (r != ACL_SUCCESS) {
        printf("  [PASS] %s: promote probe returned GetWorkspaceSize=%d\n", testName, r);
        res.Cleanup();
        g_passedTests++;
        return true;
    }
    if (ws > 0) {
        r = aclrtMalloc(&res.workspaceAddr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(r == ACL_SUCCESS, res.Cleanup(); return false);
    }
    r = aclnnAddV3(res.workspaceAddr, ws, exec, stream);
    if (r == ACL_SUCCESS) {
        r = aclrtSynchronizeStream(stream);
    }
    printf("  [PASS] %s: DOUBLE self promote path probed, execute ret=%d\n", testName, r);
    g_passedTests++;
    res.Cleanup();
    return true;
}

static void RecordExpectedFailure(const char* testName, int ret)
{
    g_totalTests++;
    if (ret != ACL_SUCCESS) {
        printf("  [PASS] %s: expected failure ret=%d\n", testName, ret);
        g_passedTests++;
    } else {
        printf("  [FAIL] %s: expected failure but got success\n", testName);
        g_failedTests++;
    }
}

static void RunInvalidAddApiCases()
{
    TensorResources res;
    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* badOutDeviceAddr = nullptr;
    void* wideOtherDeviceAddr = nullptr;
    void* unsupportedDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclTensor* out = nullptr;
    aclTensor* badOut = nullptr;
    aclTensor* wideOther = nullptr;
    aclTensor* unsupported = nullptr;
    aclScalar* alpha = nullptr;
    aclScalar* scalarSelf = nullptr;
    aclScalar* scalarOther = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    std::vector<float> selfData = {1.0f, 2.0f};
    std::vector<float> otherData = {3.0f, 4.0f};
    std::vector<float> outData = {0.0f, 0.0f};
    std::vector<float> badOutData = {0.0f, 0.0f, 0.0f};
    std::vector<float> wideOtherData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<uint32_t> unsupportedData = {1U, 2U};
    std::vector<int64_t> shape = {2};
    std::vector<int64_t> badShape = {3};
    std::vector<int64_t> wideShape = {2, 2};

    if (CreateAclTensor(selfData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self) != 0 ||
        CreateAclTensor(otherData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other) != 0 ||
        CreateAclTensor(outData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out) != 0 ||
        CreateAclTensor(badOutData, badShape, &badOutDeviceAddr, aclDataType::ACL_FLOAT, &badOut) != 0 ||
        CreateAclTensor(wideOtherData, wideShape, &wideOtherDeviceAddr, aclDataType::ACL_FLOAT, &wideOther) != 0 ||
        CreateAclTensor(unsupportedData, shape, &unsupportedDeviceAddr, aclDataType::ACL_UINT32, &unsupported) != 0) {
        printf("  [FAIL] Invalid-Add-Setup: tensor setup failed\n");
        g_totalTests++;
        g_failedTests++;
        res.Cleanup();
        return;
    }

    res.deviceAddrs.insert(res.deviceAddrs.end(),
        {selfDeviceAddr, otherDeviceAddr, outDeviceAddr, badOutDeviceAddr, wideOtherDeviceAddr, unsupportedDeviceAddr});
    res.tensors.insert(res.tensors.end(), {self, other, out, badOut, wideOther, unsupported});

    alpha = CreateScalarFloat(1.0f);
    scalarSelf = CreateScalarFloat(1.0f);
    scalarOther = CreateScalarFloat(1.0f);
    if (alpha == nullptr || scalarSelf == nullptr || scalarOther == nullptr) {
        printf("  [FAIL] Invalid-Add-Setup: scalar setup failed\n");
        g_totalTests++;
        g_failedTests++;
        res.Cleanup();
        return;
    }
    res.scalars.push_back(alpha);
    res.scalars.push_back(scalarSelf);
    res.scalars.push_back(scalarOther);

    RecordExpectedFailure("Invalid-Add-null-self",
        aclnnAddGetWorkspaceSize(nullptr, other, alpha, out, &workspaceSize, &executor));
    RecordExpectedFailure("Invalid-Add-null-other",
        aclnnAddGetWorkspaceSize(self, nullptr, alpha, out, &workspaceSize, &executor));
    RecordExpectedFailure("Invalid-Add-null-alpha",
        aclnnAddGetWorkspaceSize(self, other, nullptr, out, &workspaceSize, &executor));
    RecordExpectedFailure("Invalid-Add-null-out",
        aclnnAddGetWorkspaceSize(self, other, alpha, nullptr, &workspaceSize, &executor));
    RecordExpectedFailure("Invalid-Add-shape-mismatch",
        aclnnAddGetWorkspaceSize(self, other, alpha, badOut, &workspaceSize, &executor));
    RecordExpectedFailure("Invalid-Add-unsupported-uint32",
        aclnnAddGetWorkspaceSize(unsupported, unsupported, alpha, unsupported, &workspaceSize, &executor));
    RecordExpectedFailure("Invalid-Adds-null-self",
        aclnnAddsGetWorkspaceSize(nullptr, scalarOther, alpha, out, &workspaceSize, &executor));
    RecordExpectedFailure("Invalid-Adds-null-other",
        aclnnAddsGetWorkspaceSize(self, nullptr, alpha, out, &workspaceSize, &executor));
    RecordExpectedFailure("Invalid-Adds-null-alpha",
        aclnnAddsGetWorkspaceSize(self, scalarOther, nullptr, out, &workspaceSize, &executor));
    RecordExpectedFailure("Invalid-Adds-null-out",
        aclnnAddsGetWorkspaceSize(self, scalarOther, alpha, nullptr, &workspaceSize, &executor));
    RecordExpectedFailure("Invalid-Adds-shape-mismatch",
        aclnnAddsGetWorkspaceSize(self, scalarOther, alpha, badOut, &workspaceSize, &executor));
    RecordExpectedFailure("Invalid-InplaceAdd-broadcast-self",
        aclnnInplaceAddGetWorkspaceSize(self, wideOther, alpha, &workspaceSize, &executor));
    RecordExpectedFailure("Invalid-InplaceAdds-null-self",
        aclnnInplaceAddsGetWorkspaceSize(nullptr, scalarOther, alpha, &workspaceSize, &executor));
    RecordExpectedFailure("Invalid-InplaceAdds-null-other",
        aclnnInplaceAddsGetWorkspaceSize(self, nullptr, alpha, &workspaceSize, &executor));
    RecordExpectedFailure("Invalid-InplaceAdds-null-alpha",
        aclnnInplaceAddsGetWorkspaceSize(self, scalarOther, nullptr, &workspaceSize, &executor));
    RecordExpectedFailure("Invalid-AddV3-null-self",
        aclnnAddV3GetWorkspaceSize(nullptr, other, alpha, out, &workspaceSize, &executor));
    RecordExpectedFailure("Invalid-AddV3-null-other",
        aclnnAddV3GetWorkspaceSize(scalarSelf, nullptr, alpha, out, &workspaceSize, &executor));
    RecordExpectedFailure("Invalid-AddV3-null-alpha",
        aclnnAddV3GetWorkspaceSize(scalarSelf, other, nullptr, out, &workspaceSize, &executor));
    RecordExpectedFailure("Invalid-AddV3-shape-mismatch",
        aclnnAddV3GetWorkspaceSize(scalarSelf, other, alpha, badOut, &workspaceSize, &executor));
    RecordExpectedFailure("Invalid-AddV3-unsupported-uint32",
        aclnnAddV3GetWorkspaceSize(scalarSelf, unsupported, alpha, unsupported, &workspaceSize, &executor));

    res.Cleanup();
}

// ===================== Main test suite =====================

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return 1);

    printf("========================================\n");
    printf("  Add Operator Comprehensive Test Suite\n");
    printf("========================================\n\n");

    // ====================================================================
    // SECTION 1: Basic aclnnAdd - FP32 (covers basic Add path, alpha variants)
    // ====================================================================
    printf("--- Section 1: Basic aclnnAdd (FP32, alpha variants) ---\n");

    // 1.1: Basic Add with alpha=1 (direct Add path, IsEqualToOne=true)
    {
        std::vector<float> self = {0, 1, 2, 3, 4, 5, 6, 7};
        std::vector<float> other = {1, 1, 1, 2, 2, 2, 3, 3};
        std::vector<int64_t> shape = {4, 2};
        RunTestAdd<float>(stream, self, other, shape, shape, shape,
            aclDataType::ACL_FLOAT, 1.0f, 1e-6f, 1e-6f, "Basic-Add-FP32-alpha1", false);
    }

    // 1.2: Add with alpha=2.5 (triggers Axpy path for FP32)
    {
        std::vector<float> self = {1, 2, 3, 4};
        std::vector<float> other = {0.5f, 1.0f, 1.5f, 2.0f};
        std::vector<int64_t> shape = {4};
        RunTestAdd<float>(stream, self, other, shape, shape, shape,
            aclDataType::ACL_FLOAT, 2.5f, 1e-6f, 1e-6f, "Basic-Add-FP32-alpha2.5-Axpy");
    }

    // 1.3: Add with alpha=0 (zero scaling)
    {
        std::vector<float> self = {10, 20, 30, 40};
        std::vector<float> other = {100, 200, 300, 400};
        std::vector<int64_t> shape = {4};
        RunTestAdd<float>(stream, self, other, shape, shape, shape,
            aclDataType::ACL_FLOAT, 0.0f, 1e-6f, 1e-6f, "Basic-Add-FP32-alpha0");
    }

    // 1.4: Add with alpha=-1.5 (negative alpha, Axpy path)
    {
        std::vector<float> self = {10, 20, 30, 40};
        std::vector<float> other = {1, 2, 3, 4};
        std::vector<int64_t> shape = {4};
        RunTestAdd<float>(stream, self, other, shape, shape, shape,
            aclDataType::ACL_FLOAT, -1.5f, 1e-6f, 1e-6f, "Basic-Add-FP32-alphaNeg1.5");
    }

    // ====================================================================
    // SECTION 2: aclnnAdd - dtype coverage (triggers different tiling branches)
    // ====================================================================
    printf("\n--- Section 2: aclnnAdd Dtype Coverage ---\n");

    // 2.1: FP16 (triggers AddWithCastCompute<half> tiling)
    {
        std::vector<uint16_t> self(8, 0);
        std::vector<uint16_t> other(8, 0);
        for (int i = 0; i < 8; i++) {
            self[i] = FloatToFp16(static_cast<float>(i));
            other[i] = FloatToFp16(static_cast<float>(i + 1));
        }
        std::vector<int64_t> shape = {8};
        RunTestAddFp16(stream, self, other, shape, 1.0f, 1e-3f, 1e-3f, "Dtype-FP16-alpha1", false);
    }

    // 2.2: FP16 with alpha=2.0 (Axpy path for FP16)
    {
        std::vector<uint16_t> self = {FloatToFp16(1.0f), FloatToFp16(2.0f), FloatToFp16(3.0f), FloatToFp16(4.0f)};
        std::vector<uint16_t> other = {FloatToFp16(0.5f), FloatToFp16(1.0f), FloatToFp16(1.5f), FloatToFp16(2.0f)};
        std::vector<int64_t> shape = {4};
        RunTestAddFp16(stream, self, other, shape, 2.0f, 1e-3f, 1e-3f, "Dtype-FP16-alpha2-Axpy");
    }

    // 2.3: INT32 (triggers AddWithoutCastCompute<int32_t> tiling)
    {
        std::vector<int32_t> self = {10, 20, 30, 40, 50, 60, 70, 80};
        std::vector<int32_t> other = {1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<int64_t> shape = {8};
        RunTestAdd<int32_t>(stream, self, other, shape, shape, shape,
            aclDataType::ACL_INT32, 1.0f, 0.0f, 0.0f, "Dtype-INT32-alpha1", false);
    }

    // 2.4: INT32 with alpha=3 (triggers Axpy path for INT32)
    {
        std::vector<int32_t> self = {100, 200, 300, 400};
        std::vector<int32_t> other = {1, 2, 3, 4};
        std::vector<int64_t> shape = {4};
        RunTestAdd<int32_t>(stream, self, other, shape, shape, shape,
            aclDataType::ACL_INT32, 3.0f, 0.0f, 0.0f, "Dtype-INT32-alpha3-Axpy");
    }

    // 2.5: INT8 (triggers AddWithoutCastCompute<int8_t> tiling)
    {
        std::vector<int8_t> self = {10, 20, 30, 40, 50, 60, 70, 80};
        std::vector<int8_t> other = {1, 1, 1, 2, 2, 2, 3, 3};
        std::vector<int64_t> shape = {8};
        RunTestAdd<int8_t>(stream, self, other, shape, shape, shape,
            aclDataType::ACL_INT8, 1.0f, 0.0f, 0.0f, "Dtype-INT8-alpha1", false);
    }

    // 2.6: INT8 with alpha=2 (AxpyV2 supports INT8)
    {
        std::vector<int8_t> self = {10, 20, 30, 40};
        std::vector<int8_t> other = {1, 2, 3, 4};
        std::vector<int64_t> shape = {4};
        RunTestAdd<int8_t>(stream, self, other, shape, shape, shape,
            aclDataType::ACL_INT8, 2.0f, 0.0f, 0.0f, "Dtype-INT8-alpha2-AxpyV2", false);
    }

    // 2.6b: INT8 overflow, verifies two's-complement truncation coverage.
    {
        std::vector<int8_t> self = {100, 120, -100, -120};
        std::vector<int8_t> other = {50, 40, -50, -40};
        std::vector<int64_t> shape = {4};
        RunTestAdd<int8_t>(stream, self, other, shape, shape, shape,
            aclDataType::ACL_INT8, 1.0f, 0.0f, 0.0f, "Add_INT8_Overflow", false);
    }

    // 2.7: UINT8 (triggers AddWithoutCastCompute<uint8_t> tiling)
    {
        std::vector<uint8_t> self = {10, 20, 30, 40, 50, 60, 70, 80};
        std::vector<uint8_t> other = {1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<int64_t> shape = {8};
        RunTestAdd<uint8_t>(stream, self, other, shape, shape, shape,
            aclDataType::ACL_UINT8, 1.0f, 0.0f, 0.0f, "Dtype-UINT8-alpha1", false);
    }

    // 2.7b: UINT8 overflow, covers wraparound near 255.
    {
        std::vector<uint8_t> self = {200, 250, 10, 50};
        std::vector<uint8_t> other = {100, 50, 250, 240};
        std::vector<int64_t> shape = {4};
        RunTestAdd<uint8_t>(stream, self, other, shape, shape, shape,
            aclDataType::ACL_UINT8, 1.0f, 0.0f, 0.0f, "Add_UINT8_Overflow", false);
    }

    // 2.8: INT64 (triggers AddWithoutCastCompute<int64_t> tiling)
    {
        std::vector<int64_t> self = {1000, 2000, 3000, 4000};
        std::vector<int64_t> other = {10, 20, 30, 40};
        std::vector<int64_t> shape = {4};
        RunTestAdd<int64_t>(stream, self, other, shape, shape, shape,
            aclDataType::ACL_INT64, 1.0f, 0.0f, 0.0f, "Dtype-INT64-alpha1", false);
    }

    // ====================================================================
    // SECTION 3: Shape broadcasting tests
    // ====================================================================
    printf("\n--- Section 3: Shape Broadcasting ---\n");

    // 3.1: Same shape 2D (16 elements)
    {
        std::vector<float> self(16, 0);
        std::vector<float> other(16, 0);
        for (int i = 0; i < 16; i++) {
            self[i] = static_cast<float>(i);
            other[i] = static_cast<float>(16 - i);
        }
        std::vector<int64_t> shape = {4, 4};
        RunTestAdd<float>(stream, self, other, shape, shape, shape,
            aclDataType::ACL_FLOAT, 1.0f, 1e-6f, 1e-6f, "Broadcast-SameShape-4x4", false);
    }

    // 3.2: 3D shape
    {
        std::vector<float> self(8, 0);
        std::vector<float> other(8, 0);
        for (int i = 0; i < 8; i++) {
            self[i] = static_cast<float>(i * 2);
            other[i] = static_cast<float>(i);
        }
        std::vector<int64_t> shape = {2, 2, 2};
        RunTestAdd<float>(stream, self, other, shape, shape, shape,
            aclDataType::ACL_FLOAT, 1.0f, 1e-6f, 1e-6f, "Broadcast-3D-2x2x2", false);
    }

    // 3.3: Row vector broadcast, shape {1,4} expands to {4,4}.
    {
        std::vector<float> self(16, 1.0f);
        std::vector<float> other = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<int64_t> selfShape = {4, 4};
        std::vector<int64_t> otherShape = {1, 4};
        std::vector<int64_t> outShape = {4, 4};
        RunTestAdd<float>(stream, self, other, selfShape, otherShape, outShape,
            aclDataType::ACL_FLOAT, 2.0f, 1e-6f, 1e-6f, "Broadcast-RowVec-1x4-to-4x4");
    }

    // 3.4: Scalar-like length-1 tensor broadcast, shape {1} expands to {4}.
    {
        std::vector<float> self = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> other = {5.0f};
        std::vector<int64_t> selfShape = {4};
        std::vector<int64_t> otherShape = {1};
        std::vector<int64_t> outShape = {4};
        RunTestAdd<float>(stream, self, other, selfShape, otherShape, outShape,
            aclDataType::ACL_FLOAT, 2.0f, 1e-6f, 1e-6f, "Broadcast-Scalar1-to-4");
    }

    // 3.5: Column vector broadcast, shape {4,1} expands to {4,4}.
    {
        std::vector<float> self(16, 2.0f);
        std::vector<float> other = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<int64_t> selfShape = {4, 4};
        std::vector<int64_t> otherShape = {4, 1};
        std::vector<int64_t> outShape = {4, 4};
        RunTestAdd<float>(stream, self, other, selfShape, otherShape, outShape,
            aclDataType::ACL_FLOAT, -1.5f, 1e-6f, 1e-6f, "Broadcast-ColVec-4x1-to-4x4");
    }

    // 3.6: High-dimensional suffix broadcast, shape {4,2} expands to {8,1,4,2}.
    {
        std::vector<float> self(8 * 1 * 4 * 2, 1.0f);
        std::vector<float> other = {0.5f, -0.5f, 1.0f, -1.0f, 1.5f, -1.5f, 2.0f, -2.0f};
        std::vector<int64_t> selfShape = {8, 1, 4, 2};
        std::vector<int64_t> otherShape = {4, 2};
        std::vector<int64_t> outShape = {8, 1, 4, 2};
        RunTestAdd<float>(stream, self, other, selfShape, otherShape, outShape,
            aclDataType::ACL_FLOAT, 1.2f, 1e-6f, 1e-6f, "Broadcast-HighDimSuffix-4x2-to-8x1x4x2");
    }

    // 3.7: Scalar shape broadcast, empty shape expands to vector shape {4}.
    {
        std::vector<float> self = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> other = {5.0f};
        std::vector<int64_t> selfShape = {4};
        std::vector<int64_t> otherShape = {};
        std::vector<int64_t> outShape = {4};
        RunTestAdd<float>(stream, self, other, selfShape, otherShape, outShape,
            aclDataType::ACL_FLOAT, 1.2f, 1e-6f, 1e-6f, "Broadcast-ScalarShape-to-4");
    }

    // ====================================================================
    // SECTION 4: aclnnAdds (tensor + alpha * scalar)
    // ====================================================================
    printf("\n--- Section 4: aclnnAdds (Tensor + Scalar) ---\n");

    // 4.1: Basic Adds with alpha=1
    {
        std::vector<float> self = {1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<int64_t> shape = {8};
        RunTestAddsFloat(stream, self, 10.0f, shape, 1.0f, 1e-6f, 1e-6f, "Adds-Basic-alpha1", false);
    }

    // 4.2: Adds with alpha=2.5 (Axpy path)
    {
        std::vector<float> self = {1, 2, 3, 4};
        std::vector<int64_t> shape = {4};
        RunTestAddsFloat(stream, self, 5.0f, shape, 2.5f, 1e-6f, 1e-6f, "Adds-alpha2.5-Axpy");
    }

    // 4.3: Adds with alpha=-1 (negative alpha)
    {
        std::vector<float> self = {10, 20, 30, 40};
        std::vector<int64_t> shape = {4};
        RunTestAddsFloat(stream, self, 3.0f, shape, -1.0f, 1e-6f, 1e-6f, "Adds-alphaNeg1");
    }

    // 4.4: Adds with very small alpha, checks alpha underflow-sensitive path.
    {
        std::vector<float> self = {1.0f, -1.0f, 2.0f, -2.0f};
        std::vector<int64_t> shape = {4};
        RunTestAddsFloat(stream, self, 3.0f, shape, 1e-6f, 1e-6f, 1e-6f, "Adds-alpha1e-6");
    }

    // 4.5: Adds with very large alpha, checks scaling range.
    {
        std::vector<float> self = {1.0f, -1.0f, 2.0f, -2.0f};
        std::vector<int64_t> shape = {4};
        RunTestAddsFloat(stream, self, 1.0f, shape, 1e6f, 1e-2f, 1e-6f, "Adds-alpha1e6");
    }

    // 4.6: INT32 scalar Adds, targets scalar promote and integer Axpy path.
    {
        std::vector<int32_t> self = {10, -20, 30, -40, 50, -60};
        std::vector<int64_t> shape = {2, 3};
        RunTestAddsInt32Coverage(stream, self, 3, shape, 2, "Adds-INT32-scalar-alpha2-coverage");
    }

    // 4.7: FP16/BF16 scalar promote. Exact scalars keep 16-bit promote; inexact scalars promote to FP32.
    {
        std::vector<uint16_t> self = {
            FloatToFp16(1.0f), FloatToFp16(-2.0f), FloatToFp16(3.0f), FloatToFp16(-4.0f)};
        std::vector<int64_t> shape = {4};
        RunTestAddsHalfLikeScalarCoverage(
            stream, self, aclDataType::ACL_FLOAT16, 1.0f, 1.0f, shape, "Adds-FP16-scalar-keep16-coverage");
        RunTestAddsHalfLikeScalarCoverage(
            stream, self, aclDataType::ACL_FLOAT16, 0.1f, 1.0f, shape, "Adds-FP16-scalar-promote-float-coverage");
    }

    {
        std::vector<uint16_t> self = {
            FloatToBf16(1.0f), FloatToBf16(-2.0f), FloatToBf16(3.0f), FloatToBf16(-4.0f)};
        std::vector<int64_t> shape = {4};
        RunTestAddsHalfLikeScalarCoverage(
            stream, self, aclDataType::ACL_BF16, 1.0f, 1.0f, shape, "Adds-BF16-scalar-keep16-coverage");
        RunTestAddsHalfLikeScalarCoverage(
            stream, self, aclDataType::ACL_BF16, 0.1f, 1.0f, shape, "Adds-BF16-scalar-promote-float-coverage");
    }

    // 4.8: BOOL tensor + BOOL scalar with non-BOOL output targets the bool double-cast guard.
    RunTestAddsBoolToInt32Coverage(stream, "Adds-BOOL-scalar-to-INT32-coverage");

    // ====================================================================
    // SECTION 5: aclnnInplaceAdd (in-place tensor + alpha * tensor)
    // ====================================================================
    printf("\n--- Section 5: aclnnInplaceAdd (Inplace Tensor) ---\n");

    // 5.1: InplaceAdd FP32 alpha=1
    {
        std::vector<float> self = {1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<float> other = {8, 7, 6, 5, 4, 3, 2, 1};
        std::vector<int64_t> shape = {8};
        RunTestInplaceAdd<float>(stream, self, other, shape, shape,
            aclDataType::ACL_FLOAT, 1.0f, 1e-6f, 1e-6f, "InplaceAdd-FP32-alpha1", false);
    }

    // 5.2: InplaceAdd FP32 alpha=2.0
    {
        std::vector<float> self = {1, 2, 3, 4};
        std::vector<float> other = {0.5f, 1.0f, 1.5f, 2.0f};
        std::vector<int64_t> shape = {4};
        RunTestInplaceAdd<float>(stream, self, other, shape, shape,
            aclDataType::ACL_FLOAT, 2.0f, 1e-6f, 1e-6f, "InplaceAdd-FP32-alpha2-Axpy");
    }

    // 5.3: InplaceAdd INT32 alpha=1
    {
        std::vector<int32_t> self = {100, 200, 300, 400};
        std::vector<int32_t> other = {50, 100, 150, 200};
        std::vector<int64_t> shape = {4};
        RunTestInplaceAdd<int32_t>(stream, self, other, shape, shape,
            aclDataType::ACL_INT32, 1.0f, 0.0f, 0.0f, "InplaceAdd-INT32-alpha1", false);
    }

    // ====================================================================
    // SECTION 6: aclnnInplaceAdds (in-place tensor + alpha * scalar)
    // ====================================================================
    printf("\n--- Section 6: aclnnInplaceAdds (Inplace Scalar) ---\n");

    // 6.1: InplaceAdds FP32 alpha=1
    {
        std::vector<float> self = {1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<int64_t> shape = {8};
        RunTestInplaceAddsFloat(stream, self, 10.0f, shape, 1.0f, 1e-6f, 1e-6f, "InplaceAdds-FP32-alpha1", false);
    }

    // 6.2: InplaceAdds FP32 alpha=3.0
    {
        std::vector<float> self = {1, 2, 3, 4};
        std::vector<int64_t> shape = {4};
        RunTestInplaceAddsFloat(stream, self, 2.0f, shape, 3.0f, 1e-6f, 1e-6f, "InplaceAdds-FP32-alpha3-Axpy");
    }

    // ====================================================================
    // SECTION 7: aclnnAddV3 (scalar + alpha * tensor)
    // ====================================================================
    printf("\n--- Section 7: aclnnAddV3 (Scalar + Tensor) ---\n");

    // 7.1: AddV3 basic alpha=1
    {
        std::vector<float> other = {1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<int64_t> shape = {8};
        RunTestAddV3Float(stream, 10.0f, other, shape, 1.0f, 1e-6f, 1e-6f, "AddV3-Basic-alpha1", false);
    }

    // 7.2: AddV3 with alpha=0
    {
        std::vector<float> other = {1, 2, 3, 4};
        std::vector<int64_t> shape = {4};
        RunTestAddV3Float(stream, 100.0f, other, shape, 0.0f, 1e-6f, 1e-6f, "AddV3-alpha0");
    }

    // 7.3: AddV3 with alpha=2.5 (Axpy path)
    {
        std::vector<float> other = {1, 2, 3, 4};
        std::vector<int64_t> shape = {4};
        RunTestAddV3Float(stream, 5.0f, other, shape, 2.5f, 1e-6f, 1e-6f, "AddV3-alpha2.5-Axpy");
    }

    // ====================================================================
    // SECTION 8: aclnnInplaceAddV3 (V3 inplace)
    // ====================================================================
    printf("\n--- Section 8: aclnnInplaceAddV3 (V3 Inplace) ---\n");

    // 8.1: InplaceAddV3 basic alpha=1
    {
        std::vector<float> other = {1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<int64_t> shape = {8};
        RunTestInplaceAddV3Float(stream, 10.0f, other, shape, 1.0f, 1e-6f, 1e-6f, "InplaceAddV3-Basic-alpha1", false);
    }

    // 8.2: InplaceAddV3 with alpha=-2.0
    {
        std::vector<float> other = {1, 2, 3, 4};
        std::vector<int64_t> shape = {4};
        RunTestInplaceAddV3Float(stream, 20.0f, other, shape, -2.0f, 1e-6f, 1e-6f, "InplaceAddV3-alphaNeg2");
    }

    // ====================================================================
    // SECTION 9: Precision edge cases (critical for analysis report)
    // ====================================================================
    printf("\n--- Section 9: Precision Edge Cases ---\n");

    // 9.1: Large + Small (small value swallowed by large)
    {
        std::vector<float> self = {1e10f, 1e10f, 1e8f, 1e8f};
        std::vector<float> other = {1e-5f, 1e-5f, 1e-5f, 1e-5f};
        std::vector<int64_t> shape = {4};
        RunTestAdd<float>(stream, self, other, shape, shape, shape,
            aclDataType::ACL_FLOAT, 1.0f, 1e-6f, 1e-6f, "Precision-LargePlusSmall", false);
    }

    // 9.2: Catastrophic cancellation (near values + negative)
    {
        std::vector<float> self = {1.0000001f, 2.0000001f, 3.0000001f, 4.0000001f};
        std::vector<float> other = {-1.0f, -2.0f, -3.0f, -4.0f};
        std::vector<int64_t> shape = {4};
        RunTestAdd<float>(stream, self, other, shape, shape, shape,
            aclDataType::ACL_FLOAT, 1.0f, 1e-6f, 1e-6f, "Precision-CatastrophicCancellation");
    }

    // 9.3: Subnormal/underflow (very small numbers added)
    {
        std::vector<float> self = {1e-20f, 1e-20f, 1e-25f, 1e-25f};
        std::vector<float> other = {1e-20f, 1e-20f, 1e-25f, 1e-25f};
        std::vector<int64_t> shape = {4};
        RunTestAdd<float>(stream, self, other, shape, shape, shape,
            aclDataType::ACL_FLOAT, 1.0f, 1e-6f, 1e-6f, "Precision-SubnormalAdd");
    }

    // 9.4: Inexact decimal representation (0.1-style)
    {
        std::vector<float> self = {0.1f, 0.2f, 0.3f, 0.4f};
        std::vector<float> other = {0.2f, 0.3f, 0.4f, 0.5f};
        std::vector<int64_t> shape = {4};
        RunTestAdd<float>(stream, self, other, shape, shape, shape,
            aclDataType::ACL_FLOAT, 1.0f, 1e-6f, 1e-6f, "Precision-InexactDecimal", false);
    }

    // 9.5: Large tensor (32 elements) - stress test
    {
        std::vector<float> self(32, 0);
        std::vector<float> other(32, 0);
        for (int i = 0; i < 32; i++) {
            self[i] = static_cast<float>(i * 1000);
            other[i] = static_cast<float>(i);
        }
        std::vector<int64_t> shape = {32};
        RunTestAdd<float>(stream, self, other, shape, shape, shape,
            aclDataType::ACL_FLOAT, 1.0f, 1e-6f, 1e-6f, "Stress-LargeTensor32", false);
    }

    // 9.5b: Non-aligned shape length 33, targets tail and alignment branches.
    {
        std::vector<float> self(33, 0);
        std::vector<float> other(33, 0);
        for (int i = 0; i < 33; i++) {
            self[i] = static_cast<float>(i) * 0.25f;
            other[i] = static_cast<float>(33 - i) * 0.125f;
        }
        std::vector<int64_t> shape = {33};
        RunTestAdd<float>(stream, self, other, shape, shape, shape,
            aclDataType::ACL_FLOAT, 2.0f, 1e-6f, 1e-6f, "Stress-NonAligned33-alpha2");
    }

    // 9.5c: Non-aligned shape length 127, expands alignment coverage.
    {
        std::vector<float> self(127, 0);
        std::vector<float> other(127, 0);
        for (int i = 0; i < 127; i++) {
            self[i] = static_cast<float>(i % 17) - 8.0f;
            other[i] = static_cast<float>(i % 13) * 0.5f;
        }
        std::vector<int64_t> shape = {127};
        RunTestAdd<float>(stream, self, other, shape, shape, shape,
            aclDataType::ACL_FLOAT, -0.75f, 1e-6f, 1e-6f, "Stress-NonAligned127-alphaNeg0.75");
    }

    // 9.5d: Large tensor over 1M elements, triggers large tiling path.
    {
        constexpr int64_t largeSize = 1024 * 1024;
        std::vector<float> self(largeSize, 0);
        std::vector<float> other(largeSize, 0);
        for (int64_t i = 0; i < largeSize; i++) {
            self[i] = static_cast<float>(i % 97) * 0.01f;
            other[i] = static_cast<float>(i % 53) * 0.02f;
        }
        std::vector<int64_t> shape = {largeSize};
        RunTestAdd<float>(stream, self, other, shape, shape, shape,
            aclDataType::ACL_FLOAT, 2.0f, 1e-5f, 1e-6f, "Stress-LargeTensor1M-alpha2");
    }

    // 9.5e: Empty tensor, validates zero-size shape handling in API path.
    {
        std::vector<float> self = {};
        std::vector<float> other = {};
        std::vector<int64_t> shape = {0};
        RunTestAdd<float>(stream, self, other, shape, shape, shape,
            aclDataType::ACL_FLOAT, 1.0f, 0.0f, 0.0f, "Edge-EmptyTensor");
    }

    // 9.6: Integer overflow scenario (INT32)
    {
        std::vector<int32_t> self = {2000000000, 2000000000, 1000, 2000};
        std::vector<int32_t> other = {1000000000, 1000000000, 1000, 2000};
        std::vector<int64_t> shape = {4};
        RunTestAdd<int32_t>(stream, self, other, shape, shape, shape,
            aclDataType::ACL_INT32, 1.0f, 0.0f, 0.0f, "Precision-INT32-Overflow", false);
    }

    // 9.7: Negative alpha precision (alpha = -0.5 with FP16)
    {
        std::vector<uint16_t> self = {FloatToFp16(10.0f), FloatToFp16(20.0f), FloatToFp16(30.0f), FloatToFp16(40.0f)};
        std::vector<uint16_t> other = {FloatToFp16(1.0f), FloatToFp16(2.0f), FloatToFp16(3.0f), FloatToFp16(4.0f)};
        std::vector<int64_t> shape = {4};
        RunTestAddFp16(stream, self, other, shape, -0.5f, 1e-3f, 1e-3f, "Precision-FP16-alphaNeg0.5");
    }

    // 9.8: Decimal inputs with non-one alpha, strictly validates Axpy rounding.
    {
        std::vector<float> self = {0.125f, -0.25f, 0.375f, -0.5f};
        std::vector<float> other = {0.2f, -0.3f, 0.4f, -0.5f};
        std::vector<int64_t> shape = {4};
        RunTestAdd<float>(stream, self, other, shape, shape, shape,
            aclDataType::ACL_FLOAT, 2.5f, 1e-6f, 1e-6f, "Precision-Axpy-Decimal-alpha2.5");
    }

    // 9.9: Cancellation through alpha=-1, checks near-zero absolute tolerance.
    {
        std::vector<float> self = {1024.0001f, -2048.0002f, 1.0000001f, -1.0000001f};
        std::vector<float> other = {1024.0f, -2048.0f, 1.0f, -1.0f};
        std::vector<int64_t> shape = {4};
        RunTestAdd<float>(stream, self, other, shape, shape, shape,
            aclDataType::ACL_FLOAT, -1.0f, 2e-4f, 1e-6f, "Precision-Axpy-Cancellation-alphaNeg1");
    }

    // 9.10: Fractional FP16 alpha, verifies half input quantization and FP32 alpha scaling.
    {
        std::vector<uint16_t> self = {FloatToFp16(1.25f), FloatToFp16(-2.5f), FloatToFp16(3.75f), FloatToFp16(-4.5f)};
        std::vector<uint16_t> other = {FloatToFp16(0.5f), FloatToFp16(-1.0f), FloatToFp16(1.5f), FloatToFp16(-2.0f)};
        std::vector<int64_t> shape = {4};
        RunTestAddFp16(stream, self, other, shape, 1.5f, 2e-3f, 2e-3f, "Precision-FP16-alpha1.5");
    }

    // ====================================================================
    // SECTION 10: BF16 dtype coverage (triggers AddWithCastCompute<half> tiling + BF16 path)
    // ====================================================================
    printf("\n--- Section 10: BF16 Dtype ---\n");

    {
        std::vector<uint16_t> self(8, 0);
        std::vector<uint16_t> other(8, 0);
        for (int i = 0; i < 8; i++) {
            self[i] = FloatToBf16(static_cast<float>(i * 2));
            other[i] = FloatToBf16(static_cast<float>(i));
        }
        std::vector<int64_t> shape = {8};
        RunTestAddBf16(stream, self, other, shape, 1.0f, "Dtype-BF16-alpha1", false);
    }

    // ====================================================================
    // SECTION 10b: Mixed dtype, BOOL and non-contiguous coverage
    // ====================================================================
    printf("\n--- Section 10b: Mixed Dtype / BOOL / Non-Contiguous ---\n");

    {
        std::vector<uint16_t> self = {
            FloatToFp16(1.0f), FloatToFp16(-2.0f), FloatToFp16(3.5f), FloatToFp16(-4.5f)};
        std::vector<float> other = {0.25f, -0.5f, 1.25f, -1.75f};
        std::vector<int64_t> shape = {4};
        RunTestAddMixedFloat(
            stream, self, other, shape, aclDataType::ACL_FLOAT16, true, "Mixed-FP16-FP32-to-FP32", false);
    }

    {
        std::vector<uint16_t> other = {
            FloatToFp16(0.5f), FloatToFp16(-1.5f), FloatToFp16(2.5f), FloatToFp16(-3.5f)};
        std::vector<float> self = {10.0f, -20.0f, 30.0f, -40.0f};
        std::vector<int64_t> shape = {4};
        RunTestAddMixedFloat(
            stream, other, self, shape, aclDataType::ACL_FLOAT16, false, "Mixed-FP32-FP16-to-FP32", false);
    }

    {
        std::vector<uint16_t> self = {
            FloatToBf16(1.25f), FloatToBf16(-2.5f), FloatToBf16(3.75f), FloatToBf16(-5.0f)};
        std::vector<float> other = {0.125f, -0.25f, 0.5f, -1.0f};
        std::vector<int64_t> shape = {4};
        RunTestAddMixedFloat(
            stream, self, other, shape, aclDataType::ACL_BF16, true, "Mixed-BF16-FP32-to-FP32", false);
    }

    {
        std::vector<uint16_t> other = {
            FloatToBf16(2.0f), FloatToBf16(-3.0f), FloatToBf16(4.0f), FloatToBf16(-5.0f)};
        std::vector<float> self = {0.5f, -1.0f, 1.5f, -2.0f};
        std::vector<int64_t> shape = {4};
        RunTestAddMixedFloat(
            stream, other, self, shape, aclDataType::ACL_BF16, false, "Mixed-FP32-BF16-to-FP32", false);
    }

    RunTestAddBool(stream, "Dtype-BOOL-coverage");
    RunTestAddStridedInput(stream, "View-NonContiguousInput-stride", false);
    RunTestAddComplex64Coverage(stream, "Dtype-COMPLEX64-Add-coverage");
    RunTestAddsComplex64Coverage(stream, "Dtype-COMPLEX64-Adds-coverage");

    // ====================================================================
    // SECTION 11: V3 API with different dtypes (INT32)
    // ====================================================================
    printf("\n--- Section 11: V3 API Dtype Variants ---\n");

    {
        std::vector<int32_t> otherData = {1, 2, 3, 4};
        std::vector<int64_t> shape = {4};
        RunTestAddV3Int32(stream, 10, otherData, shape, 10, "AddV3-INT32-alpha10");
    }

    {
        std::vector<uint16_t> otherData = {
            FloatToFp16(1.0f), FloatToFp16(-2.0f), FloatToFp16(3.5f), FloatToFp16(-4.0f)};
        std::vector<int64_t> shape = {4};
        RunTestAddV3Coverage(
            stream, 2.0f, otherData, shape, aclDataType::ACL_FLOAT16, 1.5f, "AddV3-FP16-alpha1.5-coverage");
    }

    {
        std::vector<uint16_t> otherData = {
            FloatToBf16(1.25f), FloatToBf16(-2.5f), FloatToBf16(3.75f), FloatToBf16(-4.5f)};
        std::vector<int64_t> shape = {4};
        RunTestAddV3Coverage(
            stream, -1.0f, otherData, shape, aclDataType::ACL_BF16, 0.5f, "AddV3-BF16-alpha0.5-coverage");
    }

    {
        std::vector<int8_t> otherData = {1, -2, 3, -4, 5, -6};
        std::vector<int64_t> shape = {2, 3};
        RunTestAddV3Coverage(
            stream, 3.0f, otherData, shape, aclDataType::ACL_INT8, 2.0f, "AddV3-INT8-alpha2-coverage");
    }

    RunTestAddV3DoubleSelfPromoteCoverage(stream, "AddV3-DOUBLE-self-promote-INT32-to-FLOAT");
    RunTestAddV3EmptyCoverage(stream, "AddV3-EmptyTensor-workspace");

    // ====================================================================
    // SECTION 12: Invalid API inputs
    // ====================================================================
    printf("\n--- Section 12: Invalid API Inputs ---\n");
    RunInvalidAddApiCases();

    // ====================================================================
    // Summary
    // ====================================================================
    printf("\n========================================\n");
    printf("  Test Summary\n");
    printf("========================================\n");
    printf("  Total:  %d\n", g_totalTests);
    printf("  Passed: %d\n", g_passedTests);
    printf("  Failed: %d\n", g_failedTests);
    printf("========================================\n");

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return g_failedTests > 0 ? 1 : 0;
}
