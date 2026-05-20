/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Extended test cases for Add operator
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <limits>
#include <cfenv>
#include <iomanip>
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

// Helper to create unsigned int tensor
template <>
int CreateAclTensor<uint8_t>(
    const std::vector<uint8_t>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(uint8_t);
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

// Helper to create int8 tensor
template <>
int CreateAclTensor<int8_t>(
    const std::vector<int8_t>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(int8_t);
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

// ==================== Test Framework ====================
struct TestResult {
    bool passed;
    std::string name;
    std::string errorMsg;
};

class TestRunner {
private:
    std::vector<TestResult> results;
    int32_t deviceId;
    aclrtStream stream;

public:
    TestRunner(int32_t devId) : deviceId(devId) {}

    bool Init() {
        auto ret = ::Init(deviceId, &stream);
        if (ret != 0) {
            LOG_PRINT("Failed to init ACL\n");
            return false;
        }
        return true;
    }

    aclrtStream GetStream() { return stream; }

    void AddResult(const std::string& name, bool passed, const std::string& errorMsg = "") {
        results.push_back({passed, name, errorMsg});
        if (passed) {
            LOG_PRINT("  [PASS] %s\n", name.c_str());
        } else {
            LOG_PRINT("  [FAIL] %s - %s\n", name.c_str(), errorMsg.c_str());
        }
    }

    void PrintSummary() {
        int passed = 0, failed = 0;
        for (const auto& r : results) {
            if (r.passed) passed++;
            else failed++;
        }
        LOG_PRINT("\n========================================\n");
        LOG_PRINT("Summary: %d passed, %d failed\n", passed, failed);
        LOG_PRINT("========================================\n");
    }

    int GetFailedCount() {
        int count = 0;
        for (const auto& r : results) {
            if (!r.passed) count++;
        }
        return count;
    }
};

// ==================== Utility Functions ====================
template <typename T>
bool CompareFloatResults(
    const std::vector<T>& actual, const std::vector<double>& expected,
    double atol, double rtol, std::string& errorMsg)
{
    for (size_t i = 0; i < actual.size(); i++) {
        double act = static_cast<double>(actual[i]);
        double exp = expected[i];
        double diff = std::abs(act - exp);
        double threshold = atol + rtol * std::abs(exp);
        
        // Handle NaN and Inf
        if (std::isnan(exp) && std::isnan(act)) continue;
        if (std::isinf(exp) && std::isinf(act) && std::signbit(exp) == std::signbit(act)) continue;
        
        if (diff > threshold) {
            char buf[256];
            snprintf(buf, sizeof(buf), "Mismatch at [%zu]: actual=%f, expected=%f, diff=%e, threshold=%e",
                     i, act, exp, diff, threshold);
            errorMsg = buf;
            return false;
        }
    }
    return true;
}

template <typename T>
bool CompareIntResults(
    const std::vector<T>& actual, const std::vector<T>& expected, std::string& errorMsg)
{
    for (size_t i = 0; i < actual.size(); i++) {
        if (actual[i] != expected[i]) {
            char buf[256];
            snprintf(buf, sizeof(buf), "Mismatch at [%zu]: actual=%d, expected=%d",
                     i, static_cast<int>(actual[i]), static_cast<int>(expected[i]));
            errorMsg = buf;
            return false;
        }
    }
    return true;
}

template <typename T>
void PrintArray(const std::vector<T>& arr, const char* label) {
    LOG_PRINT("  %s: [", label);
    for (size_t i = 0; i < arr.size(); i++) {
        if (i > 0) LOG_PRINT(", ");
        if constexpr (std::is_integral_v<T> && sizeof(T) == 1) {
            LOG_PRINT("%d", static_cast<int>(arr[i]));
        } else {
            LOG_PRINT("%f", static_cast<double>(arr[i]));
        }
    }
    LOG_PRINT("]\n");
}

// ==================== Precision Analysis Functions ====================
void AnalyzePrecision_LargeSmall() {
    LOG_PRINT("\n========== Precision Analysis: Large + Small ==========\n");
    LOG_PRINT("Scenario: Adding small numbers to very large numbers\n");
    LOG_PRINT("Input: [1e10, 1e10] + [1e-5, 1e-5]\n");
    LOG_PRINT("Expected exact: [10000000000.00001, 10000000000.00001]\n");
    LOG_PRINT("Float32 result: [10000000000.0, 10000000000.0]\n");
    LOG_PRINT("Loss: 1e-5 is completely swallowed by 1e10\n\n");
    LOG_PRINT("Principle: Float32 has only ~7 significant decimal digits.\n");
    LOG_PRINT("When computing 1e10 + 1e-5, the smaller value falls below\n");
    LOG_PRINT("the representable precision at that magnitude.\n");
    LOG_PRINT("The spacing between adjacent Float32 values at 1e10 is:\n");
    LOG_PRINT("  epsilon(1e10) = 1e10 * 2^-23 ≈ 1192\n");
    LOG_PRINT("Since 1e-5 << 1192, it cannot affect the result.\n");
}

void AnalyzePrecision_CatastrophicCancellation() {
    LOG_PRINT("\n========== Precision Analysis: Catastrophic Cancellation ==========\n");
    LOG_PRINT("Scenario: Subtracting nearly equal values\n");
    LOG_PRINT("Input: [1.0000001, 2.0000001] + [-1.0, -2.0]\n");
    LOG_PRINT("Expected exact: [0.0000001, 0.0000001] = [1e-7, 1e-7]\n");
    LOG_PRINT("Float32 result will lose significant digits.\n\n");
    LOG_PRINT("Principle: When two close floating-point numbers are subtracted,\n");
    LOG_PRINT("the leading significant digits cancel out, leaving only the\n");
    LOG_PRINT("least significant digits which may already contain rounding error.\n");
    LOG_PRINT("Relative error can be greatly amplified.\n");
}

void AnalyzePrecision_Float16Limits() {
    LOG_PRINT("\n========== Precision Analysis: Float16 Limits ==========\n");
    LOG_PRINT("Float16 has only 11 bits of mantissa (~3.3 decimal digits).\n");
    LOG_PRINT("Max representable: 65504, Min positive normal: 6.1e-5\n");
    LOG_PRINT("Adding values with ratio > 2048 will lose the smaller one.\n");
}

void AnalyzePrecision_BFloat16Limits() {
    LOG_PRINT("\n========== Precision Analysis: BFloat16 Limits ==========\n");
    LOG_PRINT("BFloat16 has 8 bits of mantissa (~2.4 decimal digits) but\n");
    LOG_PRINT("same exponent range as Float32 (8 bits).\n");
    LOG_PRINT("Max representable: ~3.4e38, but precision is very low.\n");
}

void AnalyzePrecision_IntegerOverflow() {
    LOG_PRINT("\n========== Precision Analysis: Integer Overflow ==========\n");
    LOG_PRINT("INT8 range: [-128, 127]. Adding 100 + 50 = 150 overflows.\n");
    LOG_PRINT("INT32 range: [-2147483648, 2147483647].\n");
    LOG_PRINT("Behavior depends on implementation: saturation, wrap-around, or error.\n");
}

void AnalyzePrecision_AlphaPrecision() {
    LOG_PRINT("\n========== Precision Analysis: Alpha Parameter ==========\n");
    LOG_PRINT("Alpha multiplication introduces additional rounding error.\n");
    LOG_PRINT("Example: x2=1.234567, alpha=0.1\n");
    LOG_PRINT("  alpha*x2 introduces relative error up to 0.5 ULP of the product.\n");
    LOG_PRINT("  This error then propagates to the final addition.\n");
    LOG_PRINT("When alpha != 1 and Axpy path is not used, two rounding steps occur:\n");
    LOG_PRINT("  1. Mul: alpha * x2 (with rounding)\n");
    LOG_PRINT("  2. Add: x1 + result (with rounding)\n");
    LOG_PRINT("This can lead to larger cumulative error than fused Axpy.\n");
}

// ==================== Test Cases ====================

// Test 1: Basic Add with Float32
bool Test_BasicAdd_Float32(TestRunner& runner) {
    std::vector<int64_t> shape = {4, 2};
    std::vector<float> selfData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> otherData = {1, 1, 1, 2, 2, 2, 3, 3};
    float alphaValue = 1.0f;
    
    // Expected: self + alpha * other
    std::vector<double> expected(8);
    for (int i = 0; i < 8; i++) {
        expected[i] = static_cast<double>(selfData[i]) + alphaValue * static_cast<double>(otherData[i]);
    }

    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = nullptr;
    std::vector<float> outData(8, 0);

    int ret = CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    if (ret != 0) { runner.AddResult("BasicAdd_Float32", false, "Create self tensor failed"); return false; }
    ret = CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    if (ret != 0) { runner.AddResult("BasicAdd_Float32", false, "Create other tensor failed"); return false; }
    ret = CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    if (ret != 0) { runner.AddResult("BasicAdd_Float32", false, "Create out tensor failed"); return false; }
    
    alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    if (alpha == nullptr) { runner.AddResult("BasicAdd_Float32", false, "Create alpha failed"); return false; }

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &executor);
    if (ret != ACL_SUCCESS) { runner.AddResult("BasicAdd_Float32", false, "GetWorkspaceSize failed"); return false; }

    void* wsAddr = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) { runner.AddResult("BasicAdd_Float32", false, "Malloc workspace failed"); return false; }
    }

    ret = aclnnAdd(wsAddr, wsSize, executor, runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("BasicAdd_Float32", false, "Add execution failed"); return false; }

    ret = aclrtSynchronizeStream(runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("BasicAdd_Float32", false, "Sync failed"); return false; }

    ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(float), outDev, outData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) { runner.AddResult("BasicAdd_Float32", false, "Memcpy failed"); return false; }

    std::string errorMsg;
    bool passed = CompareFloatResults(outData, expected, 1e-6, 1e-6, errorMsg);
    runner.AddResult("BasicAdd_Float32", passed, errorMsg);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (wsSize > 0) aclrtFree(wsAddr);
    return passed;
}

// Test 2: Add with alpha != 1 (triggers Axpy or Mul+Add path)
bool Test_AddWithAlpha(TestRunner& runner) {
    std::vector<int64_t> shape = {3};
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f};
    std::vector<float> otherData = {0.5f, 1.0f, 2.0f};
    float alphaValue = 2.5f;  // alpha != 1
    
    std::vector<double> expected(3);
    for (int i = 0; i < 3; i++) {
        expected[i] = static_cast<double>(selfData[i]) + alphaValue * static_cast<double>(otherData[i]);
    }

    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = nullptr;
    std::vector<float> outData(3, 0);

    int ret = CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    if (ret != 0) { runner.AddResult("AddWithAlpha", false, "Create self tensor failed"); return false; }
    ret = CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    if (ret != 0) { runner.AddResult("AddWithAlpha", false, "Create other tensor failed"); return false; }
    ret = CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    if (ret != 0) { runner.AddResult("AddWithAlpha", false, "Create out tensor failed"); return false; }
    
    alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    if (alpha == nullptr) { runner.AddResult("AddWithAlpha", false, "Create alpha failed"); return false; }

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &executor);
    if (ret != ACL_SUCCESS) { runner.AddResult("AddWithAlpha", false, "GetWorkspaceSize failed"); return false; }

    void* wsAddr = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) { runner.AddResult("AddWithAlpha", false, "Malloc workspace failed"); return false; }
    }

    ret = aclnnAdd(wsAddr, wsSize, executor, runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("AddWithAlpha", false, "Add execution failed"); return false; }

    ret = aclrtSynchronizeStream(runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("AddWithAlpha", false, "Sync failed"); return false; }

    ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(float), outDev, outData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) { runner.AddResult("AddWithAlpha", false, "Memcpy failed"); return false; }

    std::string errorMsg;
    bool passed = CompareFloatResults(outData, expected, 1e-6, 1e-6, errorMsg);
    runner.AddResult("AddWithAlpha", passed, errorMsg);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (wsSize > 0) aclrtFree(wsAddr);
    return passed;
}

// Test 3: Add with alpha=0
bool Test_AddWithAlphaZero(TestRunner& runner) {
    std::vector<int64_t> shape = {2, 3};
    std::vector<float> selfData = {1, 2, 3, 4, 5, 6};
    std::vector<float> otherData = {10, 20, 30, 40, 50, 60};
    float alphaValue = 0.0f;
    
    std::vector<double> expected(6);
    for (int i = 0; i < 6; i++) {
        expected[i] = static_cast<double>(selfData[i]);  // alpha=0 so just self
    }

    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = nullptr;
    std::vector<float> outData(6, 0);

    int ret = CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    if (ret != 0) { runner.AddResult("AddWithAlphaZero", false, "Create self tensor failed"); return false; }
    ret = CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    if (ret != 0) { runner.AddResult("AddWithAlphaZero", false, "Create other tensor failed"); return false; }
    ret = CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    if (ret != 0) { runner.AddResult("AddWithAlphaZero", false, "Create out tensor failed"); return false; }
    
    alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    if (alpha == nullptr) { runner.AddResult("AddWithAlphaZero", false, "Create alpha failed"); return false; }

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &executor);
    if (ret != ACL_SUCCESS) { runner.AddResult("AddWithAlphaZero", false, "GetWorkspaceSize failed"); return false; }

    void* wsAddr = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) { runner.AddResult("AddWithAlphaZero", false, "Malloc workspace failed"); return false; }
    }

    ret = aclnnAdd(wsAddr, wsSize, executor, runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("AddWithAlphaZero", false, "Add execution failed"); return false; }

    ret = aclrtSynchronizeStream(runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("AddWithAlphaZero", false, "Sync failed"); return false; }

    ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(float), outDev, outData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) { runner.AddResult("AddWithAlphaZero", false, "Memcpy failed"); return false; }

    std::string errorMsg;
    bool passed = CompareFloatResults(outData, expected, 1e-6, 1e-6, errorMsg);
    runner.AddResult("AddWithAlphaZero", passed, errorMsg);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (wsSize > 0) aclrtFree(wsAddr);
    return passed;
}

// Test 4: Add with negative alpha
bool Test_AddWithNegativeAlpha(TestRunner& runner) {
    std::vector<int64_t> shape = {2, 3};
    std::vector<float> selfData = {5, 5, 5, 5, 5, 5};
    std::vector<float> otherData = {1, 2, 3, 4, 5, 6};
    float alphaValue = -1.0f;
    
    std::vector<double> expected(6);
    for (int i = 0; i < 6; i++) {
        expected[i] = static_cast<double>(selfData[i]) + alphaValue * static_cast<double>(otherData[i]);
    }

    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = nullptr;
    std::vector<float> outData(6, 0);

    int ret = CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    if (ret != 0) { runner.AddResult("AddWithNegativeAlpha", false, "Create self tensor failed"); return false; }
    ret = CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    if (ret != 0) { runner.AddResult("AddWithNegativeAlpha", false, "Create other tensor failed"); return false; }
    ret = CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    if (ret != 0) { runner.AddResult("AddWithNegativeAlpha", false, "Create out tensor failed"); return false; }
    
    alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    if (alpha == nullptr) { runner.AddResult("AddWithNegativeAlpha", false, "Create alpha failed"); return false; }

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &executor);
    if (ret != ACL_SUCCESS) { runner.AddResult("AddWithNegativeAlpha", false, "GetWorkspaceSize failed"); return false; }

    void* wsAddr = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) { runner.AddResult("AddWithNegativeAlpha", false, "Malloc workspace failed"); return false; }
    }

    ret = aclnnAdd(wsAddr, wsSize, executor, runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("AddWithNegativeAlpha", false, "Add execution failed"); return false; }

    ret = aclrtSynchronizeStream(runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("AddWithNegativeAlpha", false, "Sync failed"); return false; }

    ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(float), outDev, outData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) { runner.AddResult("AddWithNegativeAlpha", false, "Memcpy failed"); return false; }

    std::string errorMsg;
    bool passed = CompareFloatResults(outData, expected, 1e-6, 1e-6, errorMsg);
    runner.AddResult("AddWithNegativeAlpha", passed, errorMsg);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (wsSize > 0) aclrtFree(wsAddr);
    return passed;
}

// Test 5: Broadcast addition
bool Test_BroadcastAdd(TestRunner& runner) {
    std::vector<int64_t> selfShape = {2, 3};
    std::vector<int64_t> otherShape = {3};
    std::vector<int64_t> outShape = {2, 3};
    std::vector<float> selfData = {1, 2, 3, 4, 5, 6};
    std::vector<float> otherData = {10, 20, 30};
    float alphaValue = 1.0f;
    
    std::vector<double> expected(6);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            expected[i * 3 + j] = static_cast<double>(selfData[i * 3 + j]) + alphaValue * static_cast<double>(otherData[j]);
        }
    }

    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = nullptr;
    std::vector<float> outData(6, 0);

    int ret = CreateAclTensor(selfData, selfShape, &selfDev, ACL_FLOAT, &self);
    if (ret != 0) { runner.AddResult("BroadcastAdd", false, "Create self tensor failed"); return false; }
    ret = CreateAclTensor(otherData, otherShape, &otherDev, ACL_FLOAT, &other);
    if (ret != 0) { runner.AddResult("BroadcastAdd", false, "Create other tensor failed"); return false; }
    ret = CreateAclTensor(outData, outShape, &outDev, ACL_FLOAT, &out);
    if (ret != 0) { runner.AddResult("BroadcastAdd", false, "Create out tensor failed"); return false; }
    
    alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    if (alpha == nullptr) { runner.AddResult("BroadcastAdd", false, "Create alpha failed"); return false; }

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &executor);
    if (ret != ACL_SUCCESS) { runner.AddResult("BroadcastAdd", false, "GetWorkspaceSize failed"); return false; }

    void* wsAddr = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) { runner.AddResult("BroadcastAdd", false, "Malloc workspace failed"); return false; }
    }

    ret = aclnnAdd(wsAddr, wsSize, executor, runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("BroadcastAdd", false, "Add execution failed"); return false; }

    ret = aclrtSynchronizeStream(runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("BroadcastAdd", false, "Sync failed"); return false; }

    ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(float), outDev, outData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) { runner.AddResult("BroadcastAdd", false, "Memcpy failed"); return false; }

    std::string errorMsg;
    bool passed = CompareFloatResults(outData, expected, 1e-6, 1e-6, errorMsg);
    runner.AddResult("BroadcastAdd", passed, errorMsg);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (wsSize > 0) aclrtFree(wsAddr);
    return passed;
}

// Test 6: Adds (tensor + scalar)
bool Test_Adds(TestRunner& runner) {
    std::vector<int64_t> shape = {2, 3};
    std::vector<float> selfData = {1, 2, 3, 4, 5, 6};
    float otherValue = 10.0f;
    float alphaValue = 1.0f;
    
    std::vector<double> expected(6);
    for (int i = 0; i < 6; i++) {
        expected[i] = static_cast<double>(selfData[i]) + alphaValue * static_cast<double>(otherValue);
    }

    void *selfDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *out = nullptr;
    aclScalar *other = nullptr, *alpha = nullptr;
    std::vector<float> outData(6, 0);

    int ret = CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    if (ret != 0) { runner.AddResult("Adds", false, "Create self tensor failed"); return false; }
    ret = CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    if (ret != 0) { runner.AddResult("Adds", false, "Create out tensor failed"); return false; }
    
    other = aclCreateScalar(&otherValue, ACL_FLOAT);
    alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    if (other == nullptr || alpha == nullptr) { runner.AddResult("Adds", false, "Create scalars failed"); return false; }

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &wsSize, &executor);
    if (ret != ACL_SUCCESS) { runner.AddResult("Adds", false, "GetWorkspaceSize failed"); return false; }

    void* wsAddr = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) { runner.AddResult("Adds", false, "Malloc workspace failed"); return false; }
    }

    ret = aclnnAdds(wsAddr, wsSize, executor, runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("Adds", false, "Adds execution failed"); return false; }

    ret = aclrtSynchronizeStream(runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("Adds", false, "Sync failed"); return false; }

    ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(float), outDev, outData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) { runner.AddResult("Adds", false, "Memcpy failed"); return false; }

    std::string errorMsg;
    bool passed = CompareFloatResults(outData, expected, 1e-6, 1e-6, errorMsg);
    runner.AddResult("Adds", passed, errorMsg);

    aclDestroyTensor(self); aclDestroyTensor(out);
    aclDestroyScalar(other); aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(outDev);
    if (wsSize > 0) aclrtFree(wsAddr);
    return passed;
}

// Test 7: InplaceAdd
bool Test_InplaceAdd(TestRunner& runner) {
    std::vector<int64_t> shape = {2, 3};
    std::vector<float> selfData = {1, 2, 3, 4, 5, 6};
    std::vector<float> otherData = {0.5, 1, 1.5, 2, 2.5, 3};
    float alphaValue = 1.0f;
    
    std::vector<double> expected(6);
    for (int i = 0; i < 6; i++) {
        expected[i] = static_cast<double>(selfData[i]) + alphaValue * static_cast<double>(otherData[i]);
    }

    void *selfDev = nullptr, *otherDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr;
    aclScalar* alpha = nullptr;

    int ret = CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    if (ret != 0) { runner.AddResult("InplaceAdd", false, "Create self tensor failed"); return false; }
    ret = CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    if (ret != 0) { runner.AddResult("InplaceAdd", false, "Create other tensor failed"); return false; }
    
    alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    if (alpha == nullptr) { runner.AddResult("InplaceAdd", false, "Create alpha failed"); return false; }

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &wsSize, &executor);
    if (ret != ACL_SUCCESS) { runner.AddResult("InplaceAdd", false, "GetWorkspaceSize failed"); return false; }

    void* wsAddr = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) { runner.AddResult("InplaceAdd", false, "Malloc workspace failed"); return false; }
    }

    ret = aclnnInplaceAdd(wsAddr, wsSize, executor, runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("InplaceAdd", false, "InplaceAdd execution failed"); return false; }

    ret = aclrtSynchronizeStream(runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("InplaceAdd", false, "Sync failed"); return false; }

    std::vector<float> resultData(6, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), selfDev, resultData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) { runner.AddResult("InplaceAdd", false, "Memcpy failed"); return false; }

    std::string errorMsg;
    bool passed = CompareFloatResults(resultData, expected, 1e-6, 1e-6, errorMsg);
    runner.AddResult("InplaceAdd", passed, errorMsg);

    aclDestroyTensor(self); aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev);
    if (wsSize > 0) aclrtFree(wsAddr);
    return passed;
}

// Test 8: InplaceAdds
bool Test_InplaceAdds(TestRunner& runner) {
    std::vector<int64_t> shape = {2, 3};
    std::vector<float> selfData = {1, 2, 3, 4, 5, 6};
    float otherValue = 0.5f;
    float alphaValue = 2.0f;
    
    std::vector<double> expected(6);
    for (int i = 0; i < 6; i++) {
        expected[i] = static_cast<double>(selfData[i]) + alphaValue * static_cast<double>(otherValue);
    }

    void *selfDev = nullptr;
    aclTensor *self = nullptr;
    aclScalar *other = nullptr, *alpha = nullptr;

    int ret = CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    if (ret != 0) { runner.AddResult("InplaceAdds", false, "Create self tensor failed"); return false; }
    
    other = aclCreateScalar(&otherValue, ACL_FLOAT);
    alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    if (other == nullptr || alpha == nullptr) { runner.AddResult("InplaceAdds", false, "Create scalars failed"); return false; }

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnInplaceAddsGetWorkspaceSize(self, other, alpha, &wsSize, &executor);
    if (ret != ACL_SUCCESS) { runner.AddResult("InplaceAdds", false, "GetWorkspaceSize failed"); return false; }

    void* wsAddr = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) { runner.AddResult("InplaceAdds", false, "Malloc workspace failed"); return false; }
    }

    ret = aclnnInplaceAdds(wsAddr, wsSize, executor, runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("InplaceAdds", false, "InplaceAdds execution failed"); return false; }

    ret = aclrtSynchronizeStream(runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("InplaceAdds", false, "Sync failed"); return false; }

    std::vector<float> resultData(6, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), selfDev, resultData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) { runner.AddResult("InplaceAdds", false, "Memcpy failed"); return false; }

    std::string errorMsg;
    bool passed = CompareFloatResults(resultData, expected, 1e-6, 1e-6, errorMsg);
    runner.AddResult("InplaceAdds", passed, errorMsg);

    aclDestroyTensor(self);
    aclDestroyScalar(other); aclDestroyScalar(alpha);
    aclrtFree(selfDev);
    if (wsSize > 0) aclrtFree(wsAddr);
    return passed;
}

// Test 9: AddV3 (scalar + alpha * tensor) - covers aclnn_add_v3.cpp
bool Test_AddV3_Basic(TestRunner& runner) {
    std::vector<int64_t> shape = {3};
    std::vector<float> otherData = {1, 2, 3};
    float selfValue = 10.0f;
    float alphaValue = 1.0f;
    
    std::vector<double> expected(3);
    for (int i = 0; i < 3; i++) {
        expected[i] = static_cast<double>(selfValue) + alphaValue * static_cast<double>(otherData[i]);
    }

    void *otherDev = nullptr, *outDev = nullptr;
    aclTensor *other = nullptr, *out = nullptr;
    aclScalar *self = nullptr, *alpha = nullptr;
    std::vector<float> outData(3, 0);

    int ret = CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    if (ret != 0) { runner.AddResult("AddV3_Basic", false, "Create other tensor failed"); return false; }
    ret = CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    if (ret != 0) { runner.AddResult("AddV3_Basic", false, "Create out tensor failed"); return false; }
    
    self = aclCreateScalar(&selfValue, ACL_FLOAT);
    alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    if (self == nullptr || alpha == nullptr) { runner.AddResult("AddV3_Basic", false, "Create scalars failed"); return false; }

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &wsSize, &executor);
    if (ret != ACL_SUCCESS) { runner.AddResult("AddV3_Basic", false, "GetWorkspaceSize failed"); return false; }

    void* wsAddr = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) { runner.AddResult("AddV3_Basic", false, "Malloc workspace failed"); return false; }
    }

    ret = aclnnAddV3(wsAddr, wsSize, executor, runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("AddV3_Basic", false, "AddV3 execution failed"); return false; }

    ret = aclrtSynchronizeStream(runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("AddV3_Basic", false, "Sync failed"); return false; }

    ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(float), outDev, outData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) { runner.AddResult("AddV3_Basic", false, "Memcpy failed"); return false; }

    std::string errorMsg;
    bool passed = CompareFloatResults(outData, expected, 1e-6, 1e-6, errorMsg);
    runner.AddResult("AddV3_Basic", passed, errorMsg);

    aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(self); aclDestroyScalar(alpha);
    aclrtFree(otherDev); aclrtFree(outDev);
    if (wsSize > 0) aclrtFree(wsAddr);
    return passed;
}

// Test 10: AddV3 with alpha != 1 (covers Axpy path in V3)
bool Test_AddV3_WithAlpha(TestRunner& runner) {
    std::vector<int64_t> shape = {3};
    std::vector<float> otherData = {1, 2, 3};
    float selfValue = 5.0f;
    float alphaValue = 2.0f;
    
    std::vector<double> expected(3);
    for (int i = 0; i < 3; i++) {
        expected[i] = static_cast<double>(selfValue) + alphaValue * static_cast<double>(otherData[i]);
    }

    void *otherDev = nullptr, *outDev = nullptr;
    aclTensor *other = nullptr, *out = nullptr;
    aclScalar *self = nullptr, *alpha = nullptr;
    std::vector<float> outData(3, 0);

    int ret = CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    if (ret != 0) { runner.AddResult("AddV3_WithAlpha", false, "Create other tensor failed"); return false; }
    ret = CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    if (ret != 0) { runner.AddResult("AddV3_WithAlpha", false, "Create out tensor failed"); return false; }
    
    self = aclCreateScalar(&selfValue, ACL_FLOAT);
    alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    if (self == nullptr || alpha == nullptr) { runner.AddResult("AddV3_WithAlpha", false, "Create scalars failed"); return false; }

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &wsSize, &executor);
    if (ret != ACL_SUCCESS) { runner.AddResult("AddV3_WithAlpha", false, "GetWorkspaceSize failed"); return false; }

    void* wsAddr = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) { runner.AddResult("AddV3_WithAlpha", false, "Malloc workspace failed"); return false; }
    }

    ret = aclnnAddV3(wsAddr, wsSize, executor, runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("AddV3_WithAlpha", false, "AddV3 execution failed"); return false; }

    ret = aclrtSynchronizeStream(runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("AddV3_WithAlpha", false, "Sync failed"); return false; }

    ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(float), outDev, outData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) { runner.AddResult("AddV3_WithAlpha", false, "Memcpy failed"); return false; }

    std::string errorMsg;
    bool passed = CompareFloatResults(outData, expected, 1e-6, 1e-6, errorMsg);
    runner.AddResult("AddV3_WithAlpha", passed, errorMsg);

    aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(self); aclDestroyScalar(alpha);
    aclrtFree(otherDev); aclrtFree(outDev);
    if (wsSize > 0) aclrtFree(wsAddr);
    return passed;
}

// Test 11: InplaceAddV3
bool Test_InplaceAddV3(TestRunner& runner) {
    std::vector<int64_t> shape = {3};
    std::vector<float> otherData = {1, 2, 3};
    float selfValue = 10.0f;
    float alphaValue = 1.0f;
    
    std::vector<double> expected(3);
    for (int i = 0; i < 3; i++) {
        expected[i] = static_cast<double>(selfValue) + alphaValue * static_cast<double>(otherData[i]);
    }

    void *otherDev = nullptr;
    aclTensor *other = nullptr;
    aclScalar *selfRef = nullptr, *alpha = nullptr;

    int ret = CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    if (ret != 0) { runner.AddResult("InplaceAddV3", false, "Create other tensor failed"); return false; }
    
    selfRef = aclCreateScalar(&selfValue, ACL_FLOAT);
    alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    if (selfRef == nullptr || alpha == nullptr) { runner.AddResult("InplaceAddV3", false, "Create scalars failed"); return false; }

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnInplaceAddV3GetWorkspaceSize(selfRef, other, alpha, &wsSize, &executor);
    if (ret != ACL_SUCCESS) { runner.AddResult("InplaceAddV3", false, "GetWorkspaceSize failed"); return false; }

    void* wsAddr = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) { runner.AddResult("InplaceAddV3", false, "Malloc workspace failed"); return false; }
    }

    ret = aclnnInplaceAddV3(wsAddr, wsSize, executor, runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("InplaceAddV3", false, "InplaceAddV3 execution failed"); return false; }

    ret = aclrtSynchronizeStream(runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("InplaceAddV3", false, "Sync failed"); return false; }

    std::vector<float> resultData(3, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), otherDev, resultData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) { runner.AddResult("InplaceAddV3", false, "Memcpy failed"); return false; }

    std::string errorMsg;
    bool passed = CompareFloatResults(resultData, expected, 1e-6, 1e-6, errorMsg);
    runner.AddResult("InplaceAddV3", passed, errorMsg);

    aclDestroyTensor(other);
    aclDestroyScalar(selfRef); aclDestroyScalar(alpha);
    aclrtFree(otherDev);
    if (wsSize > 0) aclrtFree(wsAddr);
    return passed;
}

// Test 12: Float16 type (covers more tiling branches)
bool Test_Add_Float16(TestRunner& runner) {
    std::vector<int64_t> shape = {2, 3};
    // Using float as intermediate, will be cast to half precision
    std::vector<float> selfDataFloat = {1, 2, 3, 4, 5, 6};
    std::vector<float> otherDataFloat = {0.5, 1, 1.5, 2, 2.5, 3};
    float alphaValue = 1.0f;
    
    // Convert to half precision (using uint16_t as storage)
    std::vector<uint16_t> selfData(6), otherData(6);
    for (int i = 0; i < 6; i++) {
        // Simple float to half conversion using truncation (for test purposes)
        float f1 = selfDataFloat[i];
        float f2 = otherDataFloat[i];
        // Store as uint16_t representing FP16
        uint32_t u1 = *(uint32_t*)&f1;
        uint16_t h1 = (uint16_t)((u1 >> 16) & 0xFFFF);
        selfData[i] = h1;
        
        uint32_t u2 = *(uint32_t*)&f2;
        uint16_t h2 = (uint16_t)((u2 >> 16) & 0xFFFF);
        otherData[i] = h2;
    }
    
    std::vector<double> expected(6);
    for (int i = 0; i < 6; i++) {
        expected[i] = static_cast<double>(selfDataFloat[i]) + alphaValue * static_cast<double>(otherDataFloat[i]);
    }

    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = nullptr;
    std::vector<uint16_t> outData(6, 0);

    int ret = CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT16, &self);
    if (ret != 0) { runner.AddResult("Add_Float16", false, "Create self tensor failed"); return false; }
    ret = CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT16, &other);
    if (ret != 0) { runner.AddResult("Add_Float16", false, "Create other tensor failed"); return false; }
    ret = CreateAclTensor(outData, shape, &outDev, ACL_FLOAT16, &out);
    if (ret != 0) { runner.AddResult("Add_Float16", false, "Create out tensor failed"); return false; }
    
    alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    if (alpha == nullptr) { runner.AddResult("Add_Float16", false, "Create alpha failed"); return false; }

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &executor);
    // Don't fail immediately - some platforms may not support FP16
    if (ret != ACL_SUCCESS) {
        runner.AddResult("Add_Float16", false, "GetWorkspaceSize failed (platform may not support FP16)");
        aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
        aclDestroyScalar(alpha);
        aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
        return false;
    }

    void* wsAddr = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) { runner.AddResult("Add_Float16", false, "Malloc workspace failed"); return false; }
    }

    ret = aclnnAdd(wsAddr, wsSize, executor, runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("Add_Float16", false, "Add execution failed"); return false; }

    ret = aclrtSynchronizeStream(runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("Add_Float16", false, "Sync failed"); return false; }

    ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(uint16_t), outDev, outData.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) { runner.AddResult("Add_Float16", false, "Memcpy failed"); return false; }

    // Convert FP16 results back to float for comparison
    std::vector<float> outFloat(6);
    for (int i = 0; i < 6; i++) {
        // Simple FP16 to float conversion
        uint16_t h = outData[i];
        uint32_t u = ((uint32_t)h) << 16;
        outFloat[i] = *(float*)&u;
    }

    std::string errorMsg;
    bool passed = CompareFloatResults(outFloat, expected, 1e-3, 1e-3, errorMsg);
    runner.AddResult("Add_Float16", passed, errorMsg);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (wsSize > 0) aclrtFree(wsAddr);
    return passed;
}

// Test 13: INT32 type (exact comparison)
bool Test_Add_Int32(TestRunner& runner) {
    std::vector<int64_t> shape = {3};
    std::vector<int32_t> selfData = {1, 2, 3};
    std::vector<int32_t> otherData = {4, 5, 6};
    int32_t alphaValue = 1;
    
    std::vector<int32_t> expected(3);
    for (int i = 0; i < 3; i++) {
        expected[i] = selfData[i] + alphaValue * otherData[i];
    }

    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = nullptr;
    std::vector<int32_t> outData(3, 0);

    int ret = CreateAclTensor(selfData, shape, &selfDev, ACL_INT32, &self);
    if (ret != 0) { runner.AddResult("Add_Int32", false, "Create self tensor failed"); return false; }
    ret = CreateAclTensor(otherData, shape, &otherDev, ACL_INT32, &other);
    if (ret != 0) { runner.AddResult("Add_Int32", false, "Create other tensor failed"); return false; }
    ret = CreateAclTensor(outData, shape, &outDev, ACL_INT32, &out);
    if (ret != 0) { runner.AddResult("Add_Int32", false, "Create out tensor failed"); return false; }
    
    alpha = aclCreateScalar(&alphaValue, ACL_INT32);
    if (alpha == nullptr) { runner.AddResult("Add_Int32", false, "Create alpha failed"); return false; }

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &executor);
    if (ret != ACL_SUCCESS) { runner.AddResult("Add_Int32", false, "GetWorkspaceSize failed"); return false; }

    void* wsAddr = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) { runner.AddResult("Add_Int32", false, "Malloc workspace failed"); return false; }
    }

    ret = aclnnAdd(wsAddr, wsSize, executor, runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("Add_Int32", false, "Add execution failed"); return false; }

    ret = aclrtSynchronizeStream(runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("Add_Int32", false, "Sync failed"); return false; }

    ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(int32_t), outDev, outData.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) { runner.AddResult("Add_Int32", false, "Memcpy failed"); return false; }

    std::string errorMsg;
    bool passed = CompareIntResults(outData, expected, errorMsg);
    runner.AddResult("Add_Int32", passed, errorMsg);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (wsSize > 0) aclrtFree(wsAddr);
    return passed;
}

// Test 14: Large tensor (triggers tiling)
bool Test_LargeTensor(TestRunner& runner) {
    std::vector<int64_t> shape = {16, 64};
    int64_t totalSize = GetShapeSize(shape);
    std::vector<float> selfData(totalSize, 1.0f);
    std::vector<float> otherData(totalSize, 2.0f);
    float alphaValue = 1.0f;
    
    std::vector<double> expected(totalSize, 3.0);

    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = nullptr;
    std::vector<float> outData(totalSize, 0);

    int ret = CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    if (ret != 0) { runner.AddResult("LargeTensor", false, "Create self tensor failed"); return false; }
    ret = CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    if (ret != 0) { runner.AddResult("LargeTensor", false, "Create other tensor failed"); return false; }
    ret = CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    if (ret != 0) { runner.AddResult("LargeTensor", false, "Create out tensor failed"); return false; }
    
    alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    if (alpha == nullptr) { runner.AddResult("LargeTensor", false, "Create alpha failed"); return false; }

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &executor);
    if (ret != ACL_SUCCESS) { runner.AddResult("LargeTensor", false, "GetWorkspaceSize failed"); return false; }

    void* wsAddr = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) { runner.AddResult("LargeTensor", false, "Malloc workspace failed"); return false; }
    }

    ret = aclnnAdd(wsAddr, wsSize, executor, runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("LargeTensor", false, "Add execution failed"); return false; }

    ret = aclrtSynchronizeStream(runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("LargeTensor", false, "Sync failed"); return false; }

    ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(float), outDev, outData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) { runner.AddResult("LargeTensor", false, "Memcpy failed"); return false; }

    std::string errorMsg;
    bool passed = CompareFloatResults(outData, expected, 1e-6, 1e-6, errorMsg);
    runner.AddResult("LargeTensor", passed, errorMsg);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (wsSize > 0) aclrtFree(wsAddr);
    return passed;
}

// Test 15: NaN handling
bool Test_NaN(TestRunner& runner) {
    std::vector<int64_t> shape = {2};
    std::vector<float> selfData = {std::nanf(""), 1.0f};
    std::vector<float> otherData = {1.0f, std::nanf("")};
    float alphaValue = 1.0f;
    
    std::vector<double> expected(2);
    expected[0] = std::nan("");
    expected[1] = std::nan("");

    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = nullptr;
    std::vector<float> outData(2, 0);

    int ret = CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    if (ret != 0) { runner.AddResult("NaN", false, "Create self tensor failed"); return false; }
    ret = CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    if (ret != 0) { runner.AddResult("NaN", false, "Create other tensor failed"); return false; }
    ret = CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    if (ret != 0) { runner.AddResult("NaN", false, "Create out tensor failed"); return false; }
    
    alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    if (alpha == nullptr) { runner.AddResult("NaN", false, "Create alpha failed"); return false; }

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &executor);
    if (ret != ACL_SUCCESS) { runner.AddResult("NaN", false, "GetWorkspaceSize failed"); return false; }

    void* wsAddr = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) { runner.AddResult("NaN", false, "Malloc workspace failed"); return false; }
    }

    ret = aclnnAdd(wsAddr, wsSize, executor, runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("NaN", false, "Add execution failed"); return false; }

    ret = aclrtSynchronizeStream(runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("NaN", false, "Sync failed"); return false; }

    ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(float), outDev, outData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) { runner.AddResult("NaN", false, "Memcpy failed"); return false; }

    std::string errorMsg;
    bool passed = CompareFloatResults(outData, expected, 1e-6, 1e-6, errorMsg);
    runner.AddResult("NaN", passed, errorMsg);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (wsSize > 0) aclrtFree(wsAddr);
    return passed;
}

// Test 16: Inf handling
bool Test_Inf(TestRunner& runner) {
    std::vector<int64_t> shape = {2};
    std::vector<float> selfData = {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
    std::vector<float> otherData = {1.0f, 1.0f};
    float alphaValue = 1.0f;
    
    std::vector<double> expected(2);
    expected[0] = std::numeric_limits<double>::infinity();
    expected[1] = -std::numeric_limits<double>::infinity();

    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = nullptr;
    std::vector<float> outData(2, 0);

    int ret = CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    if (ret != 0) { runner.AddResult("Inf", false, "Create self tensor failed"); return false; }
    ret = CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    if (ret != 0) { runner.AddResult("Inf", false, "Create other tensor failed"); return false; }
    ret = CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    if (ret != 0) { runner.AddResult("Inf", false, "Create out tensor failed"); return false; }
    
    alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    if (alpha == nullptr) { runner.AddResult("Inf", false, "Create alpha failed"); return false; }

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &executor);
    if (ret != ACL_SUCCESS) { runner.AddResult("Inf", false, "GetWorkspaceSize failed"); return false; }

    void* wsAddr = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) { runner.AddResult("Inf", false, "Malloc workspace failed"); return false; }
    }

    ret = aclnnAdd(wsAddr, wsSize, executor, runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("Inf", false, "Add execution failed"); return false; }

    ret = aclrtSynchronizeStream(runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("Inf", false, "Sync failed"); return false; }

    ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(float), outDev, outData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) { runner.AddResult("Inf", false, "Memcpy failed"); return false; }

    std::string errorMsg;
    bool passed = CompareFloatResults(outData, expected, 1e-6, 1e-6, errorMsg);
    runner.AddResult("Inf", passed, errorMsg);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (wsSize > 0) aclrtFree(wsAddr);
    return passed;
}

// Test 17: Precision test - large + small
bool Test_Precision_LargeSmall(TestRunner& runner) {
    std::vector<int64_t> shape = {2};
    std::vector<float> selfData = {1e10f, 1e10f};
    std::vector<float> otherData = {1e-5f, 1e-5f};
    float alphaValue = 1.0f;
    
    std::vector<double> expected(2);
    expected[0] = 1e10 + 1e-5;
    expected[1] = 1e10 + 1e-5;

    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = nullptr;
    std::vector<float> outData(2, 0);

    int ret = CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    if (ret != 0) { runner.AddResult("Precision_LargeSmall", false, "Create self tensor failed"); return false; }
    ret = CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    if (ret != 0) { runner.AddResult("Precision_LargeSmall", false, "Create other tensor failed"); return false; }
    ret = CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    if (ret != 0) { runner.AddResult("Precision_LargeSmall", false, "Create out tensor failed"); return false; }
    
    alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    if (alpha == nullptr) { runner.AddResult("Precision_LargeSmall", false, "Create alpha failed"); return false; }

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &executor);
    if (ret != ACL_SUCCESS) { runner.AddResult("Precision_LargeSmall", false, "GetWorkspaceSize failed"); return false; }

    void* wsAddr = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) { runner.AddResult("Precision_LargeSmall", false, "Malloc workspace failed"); return false; }
    }

    ret = aclnnAdd(wsAddr, wsSize, executor, runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("Precision_LargeSmall", false, "Add execution failed"); return false; }

    ret = aclrtSynchronizeStream(runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("Precision_LargeSmall", false, "Sync failed"); return false; }

    ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(float), outDev, outData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) { runner.AddResult("Precision_LargeSmall", false, "Memcpy failed"); return false; }

    // For precision test, we expect precision loss - use larger tolerance
    std::string errorMsg;
    bool passed = CompareFloatResults(outData, expected, 1e-5, 1e-5, errorMsg);
    if (!passed) {
        // Expected failure due to precision limitation - document it
        LOG_PRINT("  Expected precision loss: small values below epsilon of large magnitude\n");
        LOG_PRINT("  Float32 at 1e10: epsilon = 1e10 * 2^-23 = %.2f\n", 1e10f * std::pow(2.0f, -23));
        LOG_PRINT("  Actual: [");
        for (size_t i = 0; i < outData.size(); i++) {
            if (i > 0) LOG_PRINT(", ");
            LOG_PRINT("%.10f", outData[i]);
        }
        LOG_PRINT("]\n");
    }
    runner.AddResult("Precision_LargeSmall", passed, errorMsg);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (wsSize > 0) aclrtFree(wsAddr);
    return passed;
}

// Test 18: Precision test - catastrophic cancellation
bool Test_Precision_CatastrophicCancellation(TestRunner& runner) {
    std::vector<int64_t> shape = {2};
    std::vector<float> selfData = {1.0000001f, 2.0000001f};
    std::vector<float> otherData = {-1.0f, -2.0f};
    float alphaValue = 1.0f;
    
    std::vector<double> expected(2);
    expected[0] = 1.0000001 + (-1.0);
    expected[1] = 2.0000001 + (-2.0);

    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = nullptr;
    std::vector<float> outData(2, 0);

    int ret = CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    if (ret != 0) { runner.AddResult("Precision_CatastrophicCancellation", false, "Create self tensor failed"); return false; }
    ret = CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    if (ret != 0) { runner.AddResult("Precision_CatastrophicCancellation", false, "Create other tensor failed"); return false; }
    ret = CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    if (ret != 0) { runner.AddResult("Precision_CatastrophicCancellation", false, "Create out tensor failed"); return false; }
    
    alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    if (alpha == nullptr) { runner.AddResult("Precision_CatastrophicCancellation", false, "Create alpha failed"); return false; }

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &executor);
    if (ret != ACL_SUCCESS) { runner.AddResult("Precision_CatastrophicCancellation", false, "GetWorkspaceSize failed"); return false; }

    void* wsAddr = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) { runner.AddResult("Precision_CatastrophicCancellation", false, "Malloc workspace failed"); return false; }
    }

    ret = aclnnAdd(wsAddr, wsSize, executor, runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("Precision_CatastrophicCancellation", false, "Add execution failed"); return false; }

    ret = aclrtSynchronizeStream(runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("Precision_CatastrophicCancellation", false, "Sync failed"); return false; }

    ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(float), outDev, outData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) { runner.AddResult("Precision_CatastrophicCancellation", false, "Memcpy failed"); return false; }

    // Check with high precision expectation
    std::string errorMsg;
    bool passed = CompareFloatResults(outData, expected, 1e-7, 1e-7, errorMsg);
    if (!passed) {
        LOG_PRINT("  Catastrophic cancellation occurred:\n");
        LOG_PRINT("  Expected: [%.10f, %.10f]\n", expected[0], expected[1]);
        LOG_PRINT("  Actual:   [%.10f, %.10f]\n", outData[0], outData[1]);
        LOG_PRINT("  Returned %.10f - %.10f = %.10f\n", selfData[0], -otherData[0], outData[0]);
    }
    runner.AddResult("Precision_CatastrophicCancellation", passed, errorMsg);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (wsSize > 0) aclrtFree(wsAddr);
    return passed;
}

// Test 19: Mixed dtype - FP16 self + FP32 other
bool Test_MixedDtype_FP16_FP32(TestRunner& runner) {
    std::vector<int64_t> shape = {2};
    // FP16 data
    std::vector<uint16_t> selfData = {0x3C00, 0x4000};  // 1.0, 2.0 in FP16
    // FP32 data
    std::vector<float> otherData = {3.0f, 4.0f};
    float alphaValue = 1.0f;
    
    std::vector<double> expected(2);
    expected[0] = 1.0 + 3.0;
    expected[1] = 2.0 + 4.0;

    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = nullptr;
    std::vector<float> outData(2, 0);  // Output is FP32 for mixed dtype

    int ret = CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT16, &self);
    if (ret != 0) { runner.AddResult("MixedDtype_FP16_FP32", false, "Create self tensor failed"); return false; }
    ret = CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    if (ret != 0) { runner.AddResult("MixedDtype_FP16_FP32", false, "Create other tensor failed"); return false; }
    ret = CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    if (ret != 0) { runner.AddResult("MixedDtype_FP16_FP32", false, "Create out tensor failed"); return false; }
    
    alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    if (alpha == nullptr) { runner.AddResult("MixedDtype_FP16_FP32", false, "Create alpha failed"); return false; }

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &executor);
    if (ret != ACL_SUCCESS) {
        runner.AddResult("MixedDtype_FP16_FP32", false, "GetWorkspaceSize failed (platform may not support mixed FP16-FP32)");
        aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
        aclDestroyScalar(alpha);
        aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
        return false;
    }

    void* wsAddr = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) { runner.AddResult("MixedDtype_FP16_FP32", false, "Malloc workspace failed"); return false; }
    }

    ret = aclnnAdd(wsAddr, wsSize, executor, runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("MixedDtype_FP16_FP32", false, "Add execution failed"); return false; }

    ret = aclrtSynchronizeStream(runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("MixedDtype_FP16_FP32", false, "Sync failed"); return false; }

    ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(float), outDev, outData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) { runner.AddResult("MixedDtype_FP16_FP32", false, "Memcpy failed"); return false; }

    std::string errorMsg;
    bool passed = CompareFloatResults(outData, expected, 1e-6, 1e-6, errorMsg);
    runner.AddResult("MixedDtype_FP16_FP32", passed, errorMsg);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (wsSize > 0) aclrtFree(wsAddr);
    return passed;
}

// Test 20: INT8 type
bool Test_Add_Int8(TestRunner& runner) {
    std::vector<int64_t> shape = {3};
    std::vector<int8_t> selfData = {10, 20, 30};
    std::vector<int8_t> otherData = {5, 10, 15};
    int8_t alphaValue = 1;
    
    std::vector<int8_t> expected(3);
    for (int i = 0; i < 3; i++) {
        expected[i] = static_cast<int8_t>(selfData[i] + alphaValue * otherData[i]);
    }

    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = nullptr;
    std::vector<int8_t> outData(3, 0);

    int ret = CreateAclTensor(selfData, shape, &selfDev, ACL_INT8, &self);
    if (ret != 0) { runner.AddResult("Add_Int8", false, "Create self tensor failed"); return false; }
    ret = CreateAclTensor(otherData, shape, &otherDev, ACL_INT8, &other);
    if (ret != 0) { runner.AddResult("Add_Int8", false, "Create other tensor failed"); return false; }
    ret = CreateAclTensor(outData, shape, &outDev, ACL_INT8, &out);
    if (ret != 0) { runner.AddResult("Add_Int8", false, "Create out tensor failed"); return false; }
    
    alpha = aclCreateScalar(&alphaValue, ACL_INT8);
    if (alpha == nullptr) { runner.AddResult("Add_Int8", false, "Create alpha failed"); return false; }

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &executor);
    if (ret != ACL_SUCCESS) { runner.AddResult("Add_Int8", false, "GetWorkspaceSize failed"); return false; }

    void* wsAddr = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) { runner.AddResult("Add_Int8", false, "Malloc workspace failed"); return false; }
    }

    ret = aclnnAdd(wsAddr, wsSize, executor, runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("Add_Int8", false, "Add execution failed"); return false; }

    ret = aclrtSynchronizeStream(runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("Add_Int8", false, "Sync failed"); return false; }

    ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(int8_t), outDev, outData.size() * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) { runner.AddResult("Add_Int8", false, "Memcpy failed"); return false; }

    std::string errorMsg;
    bool passed = CompareIntResults(outData, expected, errorMsg);
    runner.AddResult("Add_Int8", passed, errorMsg);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (wsSize > 0) aclrtFree(wsAddr);
    return passed;
}

// Test 21: Scalar tensor addition (1-element tensor)
bool Test_ScalarTensor(TestRunner& runner) {
    std::vector<int64_t> selfShape = {1};
    std::vector<int64_t> otherShape = {1};
    std::vector<int64_t> outShape = {1};
    std::vector<float> selfData = {5.0f};
    std::vector<float> otherData = {3.0f};
    float alphaValue = 1.0f;
    
    std::vector<double> expected(1, 8.0);

    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = nullptr;
    std::vector<float> outData(1, 0);

    int ret = CreateAclTensor(selfData, selfShape, &selfDev, ACL_FLOAT, &self);
    if (ret != 0) { runner.AddResult("ScalarTensor", false, "Create self tensor failed"); return false; }
    ret = CreateAclTensor(otherData, otherShape, &otherDev, ACL_FLOAT, &other);
    if (ret != 0) { runner.AddResult("ScalarTensor", false, "Create other tensor failed"); return false; }
    ret = CreateAclTensor(outData, outShape, &outDev, ACL_FLOAT, &out);
    if (ret != 0) { runner.AddResult("ScalarTensor", false, "Create out tensor failed"); return false; }
    
    alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    if (alpha == nullptr) { runner.AddResult("ScalarTensor", false, "Create alpha failed"); return false; }

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &executor);
    if (ret != ACL_SUCCESS) { runner.AddResult("ScalarTensor", false, "GetWorkspaceSize failed"); return false; }

    void* wsAddr = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) { runner.AddResult("ScalarTensor", false, "Malloc workspace failed"); return false; }
    }

    ret = aclnnAdd(wsAddr, wsSize, executor, runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("ScalarTensor", false, "Add execution failed"); return false; }

    ret = aclrtSynchronizeStream(runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("ScalarTensor", false, "Sync failed"); return false; }

    ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(float), outDev, outData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) { runner.AddResult("ScalarTensor", false, "Memcpy failed"); return false; }

    std::string errorMsg;
    bool passed = CompareFloatResults(outData, expected, 1e-6, 1e-6, errorMsg);
    runner.AddResult("ScalarTensor", passed, errorMsg);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (wsSize > 0) aclrtFree(wsAddr);
    return passed;
}

// Test 22: Adds with alpha != 1 (covers Axpy/AxpyV2/Mul paths for Adds)
bool Test_Adds_WithAlpha2(TestRunner& runner) {
    std::vector<int64_t> shape = {3};
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f};
    float otherValue = 4.0f;
    float alphaValue = 2.5f;
    
    std::vector<double> expected(3);
    for (int i = 0; i < 3; i++) {
        expected[i] = static_cast<double>(selfData[i]) + alphaValue * static_cast<double>(otherValue);
    }

    void *selfDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *out = nullptr;
    aclScalar *other = nullptr, *alpha = nullptr;
    std::vector<float> outData(3, 0);

    int ret = CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    if (ret != 0) { runner.AddResult("Adds_WithAlpha2", false, "Create self tensor failed"); return false; }
    ret = CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    if (ret != 0) { runner.AddResult("Adds_WithAlpha2", false, "Create out tensor failed"); return false; }
    
    other = aclCreateScalar(&otherValue, ACL_FLOAT);
    alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    if (other == nullptr || alpha == nullptr) { runner.AddResult("Adds_WithAlpha2", false, "Create scalars failed"); return false; }

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &wsSize, &executor);
    if (ret != ACL_SUCCESS) { runner.AddResult("Adds_WithAlpha2", false, "GetWorkspaceSize failed"); return false; }

    void* wsAddr = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) { runner.AddResult("Adds_WithAlpha2", false, "Malloc workspace failed"); return false; }
    }

    ret = aclnnAdds(wsAddr, wsSize, executor, runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("Adds_WithAlpha2", false, "Adds execution failed"); return false; }

    ret = aclrtSynchronizeStream(runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("Adds_WithAlpha2", false, "Sync failed"); return false; }

    ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(float), outDev, outData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) { runner.AddResult("Adds_WithAlpha2", false, "Memcpy failed"); return false; }

    std::string errorMsg;
    bool passed = CompareFloatResults(outData, expected, 1e-6, 1e-6, errorMsg);
    runner.AddResult("Adds_WithAlpha2", passed, errorMsg);

    aclDestroyTensor(self); aclDestroyTensor(out);
    aclDestroyScalar(other); aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(outDev);
    if (wsSize > 0) aclrtFree(wsAddr);
    return passed;
}

// Test 23: UINT8 type
bool Test_Add_Uint8(TestRunner& runner) {
    std::vector<int64_t> shape = {3};
    std::vector<uint8_t> selfData = {10, 20, 30};
    std::vector<uint8_t> otherData = {5, 10, 15};
    uint8_t alphaValue = 1;
    
    std::vector<uint8_t> expected(3);
    for (int i = 0; i < 3; i++) {
        expected[i] = static_cast<uint8_t>(selfData[i] + alphaValue * otherData[i]);
    }

    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = nullptr;
    std::vector<uint8_t> outData(3, 0);

    int ret = CreateAclTensor(selfData, shape, &selfDev, ACL_UINT8, &self);
    if (ret != 0) { runner.AddResult("Add_Uint8", false, "Create self tensor failed"); return false; }
    ret = CreateAclTensor(otherData, shape, &otherDev, ACL_UINT8, &other);
    if (ret != 0) { runner.AddResult("Add_Uint8", false, "Create other tensor failed"); return false; }
    ret = CreateAclTensor(outData, shape, &outDev, ACL_UINT8, &out);
    if (ret != 0) { runner.AddResult("Add_Uint8", false, "Create out tensor failed"); return false; }
    
    alpha = aclCreateScalar(&alphaValue, ACL_UINT8);
    if (alpha == nullptr) { runner.AddResult("Add_Uint8", false, "Create alpha failed"); return false; }

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &executor);
    if (ret != ACL_SUCCESS) { runner.AddResult("Add_Uint8", false, "GetWorkspaceSize failed"); return false; }

    void* wsAddr = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) { runner.AddResult("Add_Uint8", false, "Malloc workspace failed"); return false; }
    }

    ret = aclnnAdd(wsAddr, wsSize, executor, runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("Add_Uint8", false, "Add execution failed"); return false; }

    ret = aclrtSynchronizeStream(runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("Add_Uint8", false, "Sync failed"); return false; }

    ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(uint8_t), outDev, outData.size() * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) { runner.AddResult("Add_Uint8", false, "Memcpy failed"); return false; }

    std::string errorMsg;
    bool passed = CompareIntResults(outData, expected, errorMsg);
    runner.AddResult("Add_Uint8", passed, errorMsg);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (wsSize > 0) aclrtFree(wsAddr);
    return passed;
}

// Test 24: INT64 type
bool Test_Add_Int64(TestRunner& runner) {
    std::vector<int64_t> shape = {2};
    std::vector<int64_t> selfData = {100000, 200000};
    std::vector<int64_t> otherData = {50000, 100000};
    int64_t alphaValue = 1;
    
    std::vector<int64_t> expected(2);
    for (int i = 0; i < 2; i++) {
        expected[i] = selfData[i] + alphaValue * otherData[i];
    }

    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = nullptr;
    std::vector<int64_t> outData(2, 0);

    int ret = CreateAclTensor(selfData, shape, &selfDev, ACL_INT64, &self);
    if (ret != 0) { runner.AddResult("Add_Int64", false, "Create self tensor failed"); return false; }
    ret = CreateAclTensor(otherData, shape, &otherDev, ACL_INT64, &other);
    if (ret != 0) { runner.AddResult("Add_Int64", false, "Create other tensor failed"); return false; }
    ret = CreateAclTensor(outData, shape, &outDev, ACL_INT64, &out);
    if (ret != 0) { runner.AddResult("Add_Int64", false, "Create out tensor failed"); return false; }
    
    alpha = aclCreateScalar(&alphaValue, ACL_INT64);
    if (alpha == nullptr) { runner.AddResult("Add_Int64", false, "Create alpha failed"); return false; }

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &executor);
    if (ret != ACL_SUCCESS) { runner.AddResult("Add_Int64", false, "GetWorkspaceSize failed"); return false; }

    void* wsAddr = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) { runner.AddResult("Add_Int64", false, "Malloc workspace failed"); return false; }
    }

    ret = aclnnAdd(wsAddr, wsSize, executor, runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("Add_Int64", false, "Add execution failed"); return false; }

    ret = aclrtSynchronizeStream(runner.GetStream());
    if (ret != ACL_SUCCESS) { runner.AddResult("Add_Int64", false, "Sync failed"); return false; }

    ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(int64_t), outDev, outData.size() * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) { runner.AddResult("Add_Int64", false, "Memcpy failed"); return false; }

    std::string errorMsg;
    bool passed = (outData[0] == expected[0] && outData[1] == expected[1]);
    if (!passed) {
        errorMsg = "INT64 mismatch";
    }
    runner.AddResult("Add_Int64", passed, errorMsg);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (wsSize > 0) aclrtFree(wsAddr);
    return passed;
}

// ==================== Main ====================
int main()
{
    int32_t deviceId = 0;
    TestRunner runner(deviceId);
    
    LOG_PRINT("========================================\n");
    LOG_PRINT("  Add Operator Test Suite\n");
    LOG_PRINT("========================================\n\n");

    if (!runner.Init()) {
        LOG_PRINT("Failed to initialize ACL\n");
        return 1;
    }

    // Run precision analysis first
    AnalyzePrecision_LargeSmall();
    AnalyzePrecision_CatastrophicCancellation();
    AnalyzePrecision_Float16Limits();
    AnalyzePrecision_BFloat16Limits();
    AnalyzePrecision_IntegerOverflow();
    AnalyzePrecision_AlphaPrecision();

    LOG_PRINT("\n========================================\n");
    LOG_PRINT("  Running Test Cases\n");
    LOG_PRINT("========================================\n\n");

    // Basic API tests
    Test_BasicAdd_Float32(runner);
    Test_AddWithAlpha(runner);
    Test_AddWithAlphaZero(runner);
    Test_AddWithNegativeAlpha(runner);
    Test_BroadcastAdd(runner);
    Test_ScalarTensor(runner);

    // Adds variants
    Test_Adds(runner);
    Test_Adds_WithAlpha2(runner);

    // Inplace variants
    Test_InplaceAdd(runner);
    Test_InplaceAdds(runner);

    // V3 API variants (covers aclnn_add_v3.cpp)
    Test_AddV3_Basic(runner);
    Test_AddV3_WithAlpha(runner);
    Test_InplaceAddV3(runner);

    // Different dtypes (covers tiling branches)
    Test_Add_Float16(runner);
    Test_Add_Int32(runner);
    Test_Add_Int8(runner);
    Test_Add_Uint8(runner);
    Test_Add_Int64(runner);
    Test_MixedDtype_FP16_FP32(runner);

    // Large tensor (triggers tiling)
    Test_LargeTensor(runner);

    // Special values
    Test_NaN(runner);
    Test_Inf(runner);

    // Precision tests
    Test_Precision_LargeSmall(runner);
    Test_Precision_CatastrophicCancellation(runner);

    // Print summary
    runner.PrintSummary();

    // Cleanup
    aclrtDestroyStream(runner.GetStream());
    aclrtResetDevice(deviceId);
    aclFinalize();

    return runner.GetFailedCount() > 0 ? 1 : 0;
}